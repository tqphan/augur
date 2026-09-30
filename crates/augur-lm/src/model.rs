//! Reads models in the binary format.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use crate::error::{Error, Result, invalid};
use crate::format::{
    CODEBOOK_LEN, Cursor, HEADER_LEN, MAGIC, SECTION_ENTRY_LEN, TAG_CASING, TAG_INFO, TAG_META,
    TAG_QUANT, TAG_VOCAB, VERSION, ngram_tag, record_len, u32_at,
};
use crate::quantize::Codebook;
use crate::{LOG_ZERO, MAX_ORDER, NO_WORD, WordId};

/// A back-off n-gram language model loaded into memory.
///
/// Words are identified by [`WordId`]s, which follow the byte order of the
/// words: all words starting with a given prefix have consecutive ids (see
/// [`Model::prefix_range`]). All probabilities are log10 values; see
/// [`LOG_ZERO`].
///
/// The whole file is validated when it is loaded, so lookups never fail.
pub struct Model {
    data: Box<[u8]>,
    order: usize,
    vocab_len: usize,
    bos: Option<WordId>,
    eos: Option<WordId>,
    unk: Option<WordId>,
    counts: Vec<u64>,
    meta: Vec<(String, String)>,
    /// Byte position of the vocabulary offsets and of the words.
    vocab: (usize, usize),
    /// Probability and back-off codebooks per order.
    books: Vec<(Codebook, Codebook)>,
    /// Byte position of the records of each order.
    ngrams: Vec<usize>,
    casing: Casing,
}

/// Byte positions of the casing table.
#[derive(Debug, Clone, Copy, Default)]
struct Casing {
    len: usize,
    ids: usize,
    offsets: usize,
    forms: usize,
}

/// The stored values of one n-gram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NgramEntry {
    /// Log10 probability of the last word given the others.
    pub log_prob: f32,
    /// Log10 back-off weight when this n-gram is used as a context (0 for
    /// the highest order).
    pub log_backoff: f32,
}

impl Model {
    /// Reads and validates a model file.
    ///
    /// # Errors
    ///
    /// On I/O errors, or if the file is not a valid model (see
    /// [`Model::from_bytes`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Validates and wraps a model held in memory, e.g. one embedded with
    /// `include_bytes!`.
    ///
    /// # Errors
    ///
    /// If the data is not a model ([`Error::BadMagic`]), has an unsupported
    /// version, fails a checksum, or is structurally invalid.
    pub fn from_bytes(data: impl Into<Box<[u8]>>) -> Result<Self> {
        let data = data.into();
        let sections = read_sections(&data)?;
        let find = |tag: [u8; 4]| {
            sections
                .iter()
                .find(|s| s.tag == tag)
                .map(|s| s.range.clone())
        };
        let require = |tag: [u8; 4]| {
            find(tag).ok_or_else(|| invalid(format!("missing section {:?}", tag_name(tag))))
        };

        let meta = match find(TAG_META) {
            Some(range) => parse_meta(&data[range])?,
            None => Vec::new(),
        };

        let mut info = Cursor::new(&data[require(TAG_INFO)?], "info section");
        let order = info.len()?;
        if !(1..=MAX_ORDER).contains(&order) {
            return Err(invalid(format!("unsupported order {order}")));
        }
        let vocab_len = info.len()?;
        let special = |id: u32| -> Result<Option<WordId>> {
            match id {
                NO_WORD => Ok(None),
                id if (id as usize) < vocab_len => Ok(Some(id)),
                _ => Err(invalid("special word id out of range")),
            }
        };
        let bos = special(info.u32()?)?;
        let eos = special(info.u32()?)?;
        let unk = special(info.u32()?)?;
        let counts = (0..order).map(|_| info.u64()).collect::<Result<Vec<_>>>()?;
        info.finish()?;
        if counts[0] != vocab_len as u64 {
            return Err(invalid("unigram count differs from the vocabulary size"));
        }
        let counts_usize = counts
            .iter()
            .map(|&c| usize::try_from(c).map_err(|_| invalid("n-gram count too large")))
            .collect::<Result<Vec<_>>>()?;

        let vocab_range = require(TAG_VOCAB)?;
        validate_vocab(&data[vocab_range.clone()], vocab_len)?;
        let vocab = (vocab_range.start, vocab_range.start + (vocab_len + 1) * 4);

        let mut quant = Cursor::new(&data[require(TAG_QUANT)?], "quantization section");
        let mut books = Vec::with_capacity(order);
        for _ in 0..order {
            let prob = read_codebook(&mut quant)?;
            let backoff = read_codebook(&mut quant)?;
            books.push((prob, backoff));
        }
        quant.finish()?;

        let mut ngrams = Vec::with_capacity(order);
        for n in 1..=order {
            let tag = ngram_tag(n);
            let range = require(tag)?;
            let expected = counts_usize[n - 1]
                .checked_add(usize::from(n < order || n == 1))
                .and_then(|rows| rows.checked_mul(record_len(n, order)));
            if expected != Some(range.len()) {
                return Err(invalid(format!(
                    "section {:?} has the wrong size",
                    tag_name(tag)
                )));
            }
            ngrams.push(range.start);
        }

        let casing = match find(TAG_CASING) {
            Some(range) => validate_casing(&data[range.clone()], range.start, vocab_len)?,
            None => Casing::default(),
        };

        let model = Self {
            data,
            order,
            vocab_len,
            bos,
            eos,
            unk,
            counts,
            meta,
            vocab,
            books,
            ngrams,
            casing,
        };
        model.validate_ngrams(&counts_usize)?;
        Ok(model)
    }

    /// The highest n-gram order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of n-grams of each order, from unigrams.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Provenance and settings, as `key`, `value` pairs in file order.
    pub fn meta(&self) -> &[(String, String)] {
        &self.meta
    }

    /// The metadata value of `key`.
    pub fn meta_value(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Number of words, including markers such as `<s>`.
    pub fn vocab_len(&self) -> usize {
        self.vocab_len
    }

    /// The sentence start marker `<s>`, if the model has one.
    pub fn bos(&self) -> Option<WordId> {
        self.bos
    }

    /// The sentence end marker `</s>`, if the model has one.
    pub fn eos(&self) -> Option<WordId> {
        self.eos
    }

    /// The unknown word `<unk>`, if the model has one.
    pub fn unk(&self) -> Option<WordId> {
        self.unk
    }

    /// The word with the given id.
    ///
    /// # Panics
    ///
    /// If `id` is not below [`Model::vocab_len`].
    pub fn word(&self, id: WordId) -> &str {
        validated_str(self.word_bytes(id as usize))
    }

    /// The id of `word`, if it is in the vocabulary.
    pub fn id(&self, word: &str) -> Option<WordId> {
        let i = self.partition(0..self.vocab_len, |w| w < word.as_bytes());
        (i < self.vocab_len && self.word_bytes(i) == word.as_bytes()).then(|| to_id(i))
    }

    /// The ids of all words starting with `prefix` (all words if it is
    /// empty).
    pub fn prefix_range(&self, prefix: &str) -> Range<WordId> {
        let prefix = prefix.as_bytes();
        let start = self.partition(0..self.vocab_len, |w| w < prefix);
        let end = self.partition(start..self.vocab_len, |w| w.starts_with(prefix));
        to_id(start)..to_id(end)
    }

    /// The usual display form of a word written with capitals (e.g.
    /// `London` for `london`), if the model has one.
    pub fn casing(&self, id: WordId) -> Option<&str> {
        let c = self.casing;
        let (mut lo, mut hi) = (0, c.len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match u32_at(&self.data, c.ids + mid * 4).cmp(&id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let start = u32_at(&self.data, c.offsets + mid * 4) as usize;
                    let end = u32_at(&self.data, c.offsets + mid * 4 + 4) as usize;
                    return Some(validated_str(&self.data[c.forms + start..c.forms + end]));
                }
            }
        }
        None
    }

    /// The stored entry of an n-gram (oldest word first), without backing
    /// off.
    pub fn ngram(&self, words: &[WordId]) -> Option<NgramEntry> {
        let i = self.find(words)?;
        let n = words.len();
        Some(NgramEntry {
            log_prob: self.prob(n, i),
            log_backoff: self.backoff(n, i),
        })
    }

    /// Log10 probability of `word` following `context` (oldest word
    /// first), backing off to shorter contexts as needed. Only the last
    /// `order - 1` words of `context` are used. Ids that are not in the
    /// model make the result [`LOG_ZERO`] (for `word`) or end the usable
    /// context (for `context`); map unknown words to [`Model::unk`] first.
    pub fn log_prob(&self, context: &[WordId], word: WordId) -> f32 {
        if word as usize >= self.vocab_len {
            return LOG_ZERO;
        }
        let context = &context[context.len().saturating_sub(self.order - 1)..];
        let mut backoff = 0.0;
        for start in 0..context.len() {
            let history = &context[start..];
            if let Some(h) = self.find(history) {
                let n = history.len();
                if let Some(i) = self.find_child(n, h, word) {
                    return (backoff + self.prob(n + 1, i)).max(LOG_ZERO);
                }
                backoff += self.backoff(n, h);
            }
        }
        (backoff + self.prob(1, word as usize)).max(LOG_ZERO)
    }

    /// The words stored as following `context` exactly (oldest word
    /// first), restricted to ids in `words`, with the stored log10
    /// probability of each n-gram. An empty context yields unigrams. The
    /// iterator is empty if the context is not in the model or is as long
    /// as the model order.
    pub fn continuations(&self, context: &[WordId], words: Range<WordId>) -> Continuations<'_> {
        let n = context.len() + 1;
        let range = if context.is_empty() {
            Some(0..self.vocab_len)
        } else if n > self.order {
            None
        } else {
            self.find(context).map(|i| self.children(n - 1, i))
        };
        let range = range.map_or(0..0, |r| {
            let lo = self.partition_records(n, r.clone(), |id| id < words.start);
            let hi = self.partition_records(n, lo..r.end, |id| id < words.end);
            lo..hi
        });
        Continuations {
            model: self,
            n,
            range,
        }
    }

    // ---- records ----

    fn record(&self, n: usize, i: usize) -> usize {
        self.ngrams[n - 1] + i * record_len(n, self.order)
    }

    fn record_id(&self, n: usize, i: usize) -> WordId {
        if n == 1 {
            to_id(i)
        } else {
            u32_at(&self.data, self.record(n, i))
        }
    }

    fn prob(&self, n: usize, i: usize) -> f32 {
        let at = self.record(n, i) + if n == 1 { 0 } else { 4 };
        self.books[n - 1].0.decode(self.data[at])
    }

    fn backoff(&self, n: usize, i: usize) -> f32 {
        if n == self.order {
            return 0.0;
        }
        let at = self.record(n, i) + if n == 1 { 1 } else { 5 };
        self.books[n - 1].1.decode(self.data[at])
    }

    fn child(&self, n: usize, i: usize) -> usize {
        let at = self.record(n, i) + if n == 1 { 2 } else { 6 };
        u32_at(&self.data, at) as usize
    }

    /// Record range of order `n + 1` extending record `i` of order `n`.
    fn children(&self, n: usize, i: usize) -> Range<usize> {
        self.child(n, i)..self.child(n, i + 1)
    }

    /// Record index of an n-gram of order `words.len()`.
    fn find(&self, words: &[WordId]) -> Option<usize> {
        let (&first, rest) = words.split_first()?;
        if words.len() > self.order || first as usize >= self.vocab_len {
            return None;
        }
        let mut i = first as usize;
        for (n, &word) in (1..).zip(rest) {
            i = self.find_child(n, i, word)?;
        }
        Some(i)
    }

    /// Record index in order `n + 1` of `word` extending record `i` of
    /// order `n`.
    fn find_child(&self, n: usize, i: usize, word: WordId) -> Option<usize> {
        if n >= self.order {
            return None;
        }
        let range = self.children(n, i);
        let j = self.partition_records(n + 1, range.clone(), |id| id < word);
        (j < range.end && self.record_id(n + 1, j) == word).then_some(j)
    }

    /// First record in `range` of order `n` whose word id fails `pred`
    /// (ids ascend within the range).
    fn partition_records(
        &self,
        n: usize,
        range: Range<usize>,
        pred: impl Fn(WordId) -> bool,
    ) -> usize {
        let (mut lo, mut hi) = (range.start, range.end);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.record_id(n, mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // ---- vocabulary ----

    fn word_bytes(&self, i: usize) -> &[u8] {
        let (offsets, words) = self.vocab;
        let start = u32_at(&self.data, offsets + i * 4) as usize;
        let end = u32_at(&self.data, offsets + i * 4 + 4) as usize;
        &self.data[words + start..words + end]
    }

    /// First word index in `range` whose bytes fail `pred`.
    fn partition(&self, range: Range<usize>, pred: impl Fn(&[u8]) -> bool) -> usize {
        let (mut lo, mut hi) = (range.start, range.end);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.word_bytes(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    // ---- validation ----

    /// Checks child pointers and word ids, so that lookups cannot go out
    /// of bounds or miss entries.
    fn validate_ngrams(&self, counts: &[usize]) -> Result<()> {
        for n in 1..self.order {
            let parents = counts[n - 1];
            if self.child(n, 0) != 0 || self.child(n, parents) != counts[n] {
                return Err(invalid(format!(
                    "order {n} child pointers do not span order {}",
                    n + 1
                )));
            }
            let mut previous = 0;
            for i in 0..parents {
                let range = self.children(n, i);
                if range.start < previous || range.start > range.end {
                    return Err(invalid(format!(
                        "order {n} child pointers are not ascending"
                    )));
                }
                previous = range.start;
                let mut last: Option<WordId> = None;
                for j in range {
                    let id = self.record_id(n + 1, j);
                    if id as usize >= self.vocab_len || last.is_some_and(|last| id <= last) {
                        return Err(invalid(format!(
                            "order {} word ids are invalid or unsorted",
                            n + 1
                        )));
                    }
                    last = Some(id);
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("order", &self.order)
            .field("vocab_len", &self.vocab_len)
            .field("counts", &self.counts)
            .field("bytes", &self.data.len())
            .finish_non_exhaustive()
    }
}

/// Iterator over `(word, log10 probability)` returned by
/// [`Model::continuations`], in ascending word id order.
#[derive(Debug, Clone)]
pub struct Continuations<'a> {
    model: &'a Model,
    n: usize,
    range: Range<usize>,
}

impl Iterator for Continuations<'_> {
    type Item = (WordId, f32);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.range.next()?;
        Some((self.model.record_id(self.n, i), self.model.prob(self.n, i)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl ExactSizeIterator for Continuations<'_> {}

fn to_id(i: usize) -> WordId {
    u32::try_from(i).expect("vocabulary fits u32")
}

/// Text that was checked to be UTF-8 when the model was loaded.
fn validated_str(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("validated when loaded")
}

fn tag_name(tag: [u8; 4]) -> String {
    String::from_utf8_lossy(&tag).into_owned()
}

struct Section {
    tag: [u8; 4],
    range: Range<usize>,
}

/// Checks the header and all section checksums.
fn read_sections(data: &[u8]) -> Result<Vec<Section>> {
    if data.len() < HEADER_LEN || data[..MAGIC.len()] != MAGIC {
        return Err(Error::BadMagic);
    }
    let mut header = Cursor::new(data, "header");
    header.bytes(MAGIC.len())?;
    let version = header.u32()?;
    if version != VERSION {
        return Err(Error::UnsupportedVersion {
            found: version,
            supported: VERSION,
        });
    }
    let count = header.len()?;
    let table_len = count
        .checked_mul(SECTION_ENTRY_LEN)
        .ok_or_else(|| invalid("header is truncated"))?;
    let mut table = Cursor::new(header.bytes(table_len)?, "header");
    let header_len = header.pos();
    if header.u32()? != crc32fast::hash(&data[..header_len]) {
        return Err(Error::Checksum("header".into()));
    }

    let mut sections: Vec<Section> = Vec::with_capacity(count);
    for _ in 0..count {
        let tag: [u8; 4] = table.bytes(4)?.try_into().expect("4 bytes");
        let crc = table.u32()?;
        let offset = usize::try_from(table.u64()?).ok();
        let len = usize::try_from(table.u64()?).ok();
        let range = offset
            .zip(len)
            .and_then(|(offset, len)| Some(offset..offset.checked_add(len)?))
            .filter(|r| r.start >= header.pos() && r.end <= data.len())
            .ok_or_else(|| invalid(format!("section {:?} is out of bounds", tag_name(tag))))?;
        if sections.iter().any(|s| s.tag == tag) {
            return Err(invalid(format!("duplicate section {:?}", tag_name(tag))));
        }
        if crc32fast::hash(&data[range.clone()]) != crc {
            return Err(Error::Checksum(format!("section {:?}", tag_name(tag))));
        }
        sections.push(Section { tag, range });
    }
    Ok(sections)
}

fn parse_meta(data: &[u8]) -> Result<Vec<(String, String)>> {
    let mut cursor = Cursor::new(data, "meta section");
    let count = cursor.len()?;
    let mut text = || -> Result<String> {
        let len = cursor.len()?;
        String::from_utf8(cursor.bytes(len)?.to_vec()).map_err(|_| invalid("metadata is not UTF-8"))
    };
    let mut meta = Vec::new();
    for _ in 0..count {
        meta.push((text()?, text()?));
    }
    cursor.finish()?;
    Ok(meta)
}

fn read_codebook(cursor: &mut Cursor<'_>) -> Result<Codebook> {
    let mut centers = [0.0; CODEBOOK_LEN];
    for center in &mut centers {
        *center = cursor.f32()?;
    }
    Ok(Codebook::from_centers(&centers))
}

/// Reads `len + 1` ascending offsets ending at `blob_len`.
fn read_offsets(cursor: &mut Cursor<'_>, len: usize) -> Result<Vec<usize>> {
    let offsets = (0..=len)
        .map(|_| cursor.len())
        .collect::<Result<Vec<_>>>()?;
    if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(invalid("string offsets are not ascending"));
    }
    Ok(offsets)
}

fn validate_vocab(data: &[u8], vocab_len: usize) -> Result<()> {
    let mut cursor = Cursor::new(data, "vocabulary section");
    let offsets = read_offsets(&mut cursor, vocab_len)?;
    let words = cursor.bytes(offsets[vocab_len])?;
    cursor.finish()?;
    let mut previous: Option<&[u8]> = None;
    for w in offsets.windows(2) {
        let word = &words[w[0]..w[1]];
        if std::str::from_utf8(word).is_err() {
            return Err(invalid("vocabulary word is not UTF-8"));
        }
        if previous.is_some_and(|p| p >= word) {
            return Err(invalid("vocabulary is not sorted"));
        }
        previous = Some(word);
    }
    Ok(())
}

fn validate_casing(data: &[u8], start: usize, vocab_len: usize) -> Result<Casing> {
    let mut cursor = Cursor::new(data, "casing section");
    let len = cursor.len()?;
    let ids = start + cursor.pos();
    let mut previous: Option<u32> = None;
    for _ in 0..len {
        let id = cursor.u32()?;
        if id as usize >= vocab_len || previous.is_some_and(|p| p >= id) {
            return Err(invalid("casing word ids are invalid or unsorted"));
        }
        previous = Some(id);
    }
    let offsets = start + cursor.pos();
    let ends = read_offsets(&mut cursor, len)?;
    let forms = start + cursor.pos();
    let blob = cursor.bytes(ends[len])?;
    cursor.finish()?;
    if ends
        .windows(2)
        .any(|w| std::str::from_utf8(&blob[w[0]..w[1]]).is_err())
    {
        return Err(invalid("casing form is not UTF-8"));
    }
    Ok(Casing {
        len,
        ids,
        offsets,
        forms,
    })
}
