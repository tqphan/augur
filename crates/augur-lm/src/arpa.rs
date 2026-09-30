//! Reads ARPA back-off language model files.

use std::collections::HashMap;
use std::io::BufRead;

use crate::error::{Error, Result};
use crate::{LOG_ZERO, MAX_ORDER};

/// A back-off n-gram model read from an ARPA file, held in memory until
/// it is written in the binary format with [`crate::write_model`].
#[derive(Debug, Clone, Default)]
pub struct Arpa {
    /// Words in the order of the unigram section.
    pub(crate) words: Vec<String>,
    /// N-grams per order, index 0 is unigrams.
    pub(crate) orders: Vec<ArpaOrder>,
}

/// The n-grams of one order, in file order.
#[derive(Debug, Clone, Default)]
pub(crate) struct ArpaOrder {
    /// Word ids (indices into [`Arpa::words`]), `n` per n-gram.
    pub(crate) ids: Vec<u32>,
    /// Log10 probabilities.
    pub(crate) probs: Vec<f32>,
    /// Log10 back-off weights (0 when absent).
    pub(crate) backoffs: Vec<f32>,
}

impl ArpaOrder {
    pub(crate) fn len(&self) -> usize {
        self.probs.len()
    }
}

impl Arpa {
    /// Parses an ARPA file.
    ///
    /// Log probabilities of `-inf` or below [`LOG_ZERO`] are stored as
    /// `LOG_ZERO`. Back-off weights on the highest order are ignored.
    ///
    /// # Errors
    ///
    /// On I/O errors, malformed lines, counts that do not match the
    /// `\data\` header, duplicate n-grams in the unigram section, or words
    /// in higher orders that have no unigram.
    pub fn read(reader: impl BufRead) -> Result<Self> {
        Parser::new(reader).parse()
    }

    /// The highest n-gram order.
    pub fn order(&self) -> usize {
        self.orders.len()
    }

    /// Number of n-grams of each order, from unigrams.
    pub fn counts(&self) -> Vec<u64> {
        self.orders.iter().map(|o| o.len() as u64).collect()
    }

    /// Number of distinct words (unigrams).
    pub fn vocab_len(&self) -> usize {
        self.words.len()
    }
}

struct Parser<R> {
    reader: R,
    line: String,
    line_no: u64,
}

impl<R: BufRead> Parser<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            line: String::new(),
            line_no: 0,
        }
    }

    fn error(&self, message: impl Into<String>) -> Error {
        Error::Arpa {
            line: self.line_no,
            message: message.into(),
        }
    }

    /// Reads the next line into `self.line` without its line ending.
    /// Returns false at end of input.
    fn next_line(&mut self) -> Result<bool> {
        self.line.clear();
        if self.reader.read_line(&mut self.line)? == 0 {
            return Ok(false);
        }
        self.line_no += 1;
        let len = self.line.trim_end_matches(['\n', '\r']).len();
        self.line.truncate(len);
        Ok(true)
    }

    /// Reads lines until a non-blank one; returns false at end of input.
    fn next_nonblank(&mut self) -> Result<bool> {
        while self.next_line()? {
            if !self.line.trim().is_empty() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn parse(mut self) -> Result<Arpa> {
        // Anything before `\data\` is free-form commentary.
        loop {
            if !self.next_line()? {
                return Err(self.error("missing \\data\\ header"));
            }
            if self.line.trim() == "\\data\\" {
                break;
            }
        }

        let mut declared: Vec<usize> = Vec::new();
        loop {
            if !self.next_nonblank()? {
                return Err(self.error("unexpected end of file in \\data\\ header"));
            }
            let line = self.line.trim();
            if line.starts_with('\\') {
                break;
            }
            let (n, count): (usize, usize) = line
                .strip_prefix("ngram ")
                .and_then(|rest| rest.split_once('='))
                .and_then(|(n, count)| Some((n.trim().parse().ok()?, count.trim().parse().ok()?)))
                .ok_or_else(|| self.error(format!("expected `ngram N=COUNT`, found {line:?}")))?;
            if n != declared.len() + 1 {
                return Err(self.error(format!("expected ngram {}=…", declared.len() + 1)));
            }
            declared.push(count);
        }
        if declared.is_empty() {
            return Err(self.error("the \\data\\ header declares no n-grams"));
        }
        if declared.len() > MAX_ORDER {
            return Err(self.error(format!(
                "order {} exceeds the maximum of {MAX_ORDER}",
                declared.len()
            )));
        }
        let order = declared.len();

        let mut arpa = Arpa::default();
        let mut index: HashMap<String, u32> = HashMap::new();
        for (i, &count) in declared.iter().enumerate() {
            let n = i + 1;
            if self.line.trim() != format!("\\{n}-grams:") {
                return Err(self.error(format!("expected \\{n}-grams:, found {:?}", self.line)));
            }
            let section = self.section(n, n == order, count, &mut arpa.words, &mut index)?;
            arpa.orders.push(section);
        }
        if self.line.trim() != "\\end\\" {
            return Err(self.error(format!("expected \\end\\, found {:?}", self.line)));
        }
        Ok(arpa)
    }

    /// Reads the entries of order `n` up to the next `\` line, which is
    /// left in `self.line`.
    fn section(
        &mut self,
        n: usize,
        highest: bool,
        count: usize,
        words: &mut Vec<String>,
        index: &mut HashMap<String, u32>,
    ) -> Result<ArpaOrder> {
        let mut out = ArpaOrder {
            ids: Vec::with_capacity(count.saturating_mul(n)),
            probs: Vec::with_capacity(count),
            backoffs: Vec::with_capacity(count),
        };
        loop {
            if !self.next_nonblank()? {
                return Err(self.error("unexpected end of file (missing \\end\\)"));
            }
            if self.line.trim_start().starts_with('\\') {
                break;
            }
            let mut fields = self.line.split_whitespace();
            let Some(prob) = fields.next().and_then(parse_log) else {
                return Err(self.error("expected a log probability"));
            };
            for _ in 0..n {
                let Some(word) = fields.next() else {
                    return Err(self.error(format!("expected {n} words")));
                };
                let id = if n == 1 {
                    if index.contains_key(word) {
                        return Err(self.error(format!("duplicate unigram {word:?}")));
                    }
                    let id =
                        u32::try_from(words.len()).map_err(|_| self.error("too many unigrams"))?;
                    index.insert(word.to_owned(), id);
                    words.push(word.to_owned());
                    id
                } else {
                    *index
                        .get(word)
                        .ok_or_else(|| self.error(format!("word {word:?} has no unigram")))?
                };
                out.ids.push(id);
            }
            let backoff = match fields.next() {
                None => 0.0,
                Some(text) => parse_log(text)
                    .ok_or_else(|| self.error(format!("invalid back-off weight {text:?}")))?,
            };
            if fields.next().is_some() {
                return Err(self.error(format!("too many fields for a {n}-gram")));
            }
            out.probs.push(prob);
            out.backoffs.push(if highest { 0.0 } else { backoff });
        }
        if out.len() != count {
            return Err(self.error(format!(
                "\\{n}-grams: has {} entries but the header declares {count}",
                out.len()
            )));
        }
        Ok(out)
    }
}

/// Parses a log10 value, mapping `-inf` and anything below [`LOG_ZERO`] to
/// `LOG_ZERO`.
fn parse_log(text: &str) -> Option<f32> {
    let value: f32 = text.parse().ok()?;
    if value == f32::NEG_INFINITY || value < LOG_ZERO {
        Some(LOG_ZERO)
    } else if value.is_finite() {
        Some(value)
    } else {
        None
    }
}
