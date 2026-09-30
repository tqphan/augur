//! Writes models in the binary format.

use std::collections::HashMap;
use std::io::Write;

use crate::arpa::Arpa;
use crate::error::{Error, Result, invalid};
use crate::format::{
    HEADER_LEN, MAGIC, PutLe, SECTION_ENTRY_LEN, TAG_CASING, TAG_INFO, TAG_META, TAG_QUANT,
    TAG_VOCAB, VERSION, ngram_tag, record_len,
};
use crate::quantize::Codebook;
use crate::{NO_WORD, WordId};

/// Summary of a model written by [`write_model`].
#[derive(Debug, Clone, PartialEq)]
pub struct WriteStats {
    /// Number of n-grams of each order, from unigrams.
    pub counts: Vec<u64>,
    /// Mean absolute quantization error of log10 probabilities, per order.
    pub prob_error: Vec<f32>,
    /// Mean absolute quantization error of log10 back-off weights, per
    /// order (0 for the highest order, which has none).
    pub backoff_error: Vec<f32>,
    /// Casing entries stored.
    pub casing_entries: usize,
    /// Casing entries dropped because their word is not in the vocabulary
    /// (or is listed twice).
    pub casing_skipped: usize,
    /// Size of the written model in bytes.
    pub bytes: u64,
}

/// Writes `arpa` in the binary format, together with provenance metadata
/// (`key`, `value` pairs, kept in order) and a casing table (`word`,
/// display form) for words usually written with capitals.
///
/// # Errors
///
/// If the model is too large for the format, contains duplicate n-grams,
/// or contains an n-gram whose prefix (all words but the last) is not
/// itself in the model; or on write errors.
pub fn write_model(
    arpa: &Arpa,
    meta: &[(String, String)],
    casing: &[(String, String)],
    out: &mut impl Write,
) -> Result<WriteStats> {
    let order = arpa.order();
    if order == 0 {
        return Err(invalid("the model has no n-grams"));
    }
    let too_large = || invalid("the model is too large for the format");
    let vocab_len = u32::try_from(arpa.words.len()).map_err(|_| too_large())?;
    if vocab_len == NO_WORD {
        return Err(too_large());
    }

    // Ids follow the byte order of the words, so that the words sharing a
    // prefix have consecutive ids.
    let mut by_bytes: Vec<u32> = (0..vocab_len).collect();
    by_bytes.sort_unstable_by(|&a, &b| {
        arpa.words[a as usize]
            .as_bytes()
            .cmp(arpa.words[b as usize].as_bytes())
    });
    let mut new_id = vec![0u32; by_bytes.len()];
    for (id, &old) in (0..).zip(&by_bytes) {
        new_id[old as usize] = id;
    }
    let words: Vec<&str> = by_bytes
        .iter()
        .map(|&old| arpa.words[old as usize].as_str())
        .collect();

    // Sort every order by (remapped) word ids and quantize it.
    let mut orders: Vec<SortedOrder> = Vec::with_capacity(order);
    let mut prob_error = Vec::with_capacity(order);
    let mut backoff_error = Vec::with_capacity(order);
    for (i, source) in arpa.orders.iter().enumerate() {
        let n = i + 1;
        let prob_book = Codebook::build(&source.probs);
        let backoff_book = Codebook::build(if n < order { &source.backoffs } else { &[] });
        prob_error.push(mean_error(&prob_book, &source.probs));
        backoff_error.push(if n < order {
            mean_error(&backoff_book, &source.backoffs)
        } else {
            0.0
        });
        let sorted = SortedOrder::new(n, source, &new_id, &words, &prob_book, &backoff_book)?;
        u32::try_from(sorted.len()).map_err(|_| too_large())?;
        orders.push(sorted);
    }

    let mut sections: Vec<([u8; 4], Vec<u8>)> = vec![
        (TAG_META, meta_section(meta)?),
        (TAG_INFO, info_section(&orders, vocab_len, &words)),
        (TAG_VOCAB, vocab_section(&words)?),
        (TAG_QUANT, quant_section(&orders)),
    ];
    for n in 1..=order {
        let children = orders
            .get(n)
            .map(|next| child_pointers(&orders[n - 1], next, &words));
        let children = children.transpose()?;
        sections.push((
            ngram_tag(n),
            ngram_section(&orders[n - 1], children.as_deref(), n, order),
        ));
    }
    let (casing_data, casing_entries) = casing_section(casing, &words)?;
    sections.push((TAG_CASING, casing_data));

    let bytes = write_sections(&sections, out)?;
    Ok(WriteStats {
        counts: orders.iter().map(|o| o.len() as u64).collect(),
        prob_error,
        backoff_error,
        casing_entries,
        casing_skipped: casing.len() - casing_entries,
        bytes,
    })
}

/// One order, sorted by word ids, with quantized values.
struct SortedOrder {
    n: usize,
    /// `n` ids per n-gram.
    ids: Vec<WordId>,
    probs: Vec<u8>,
    backoffs: Vec<u8>,
    prob_book: Codebook,
    backoff_book: Codebook,
}

impl SortedOrder {
    fn new(
        n: usize,
        source: &crate::arpa::ArpaOrder,
        new_id: &[u32],
        words: &[&str],
        prob_book: &Codebook,
        backoff_book: &Codebook,
    ) -> Result<Self> {
        let remapped: Vec<WordId> = source.ids.iter().map(|&id| new_id[id as usize]).collect();
        let mut perm: Vec<usize> = (0..source.len()).collect();
        perm.sort_unstable_by(|&a, &b| {
            remapped[a * n..(a + 1) * n].cmp(&remapped[b * n..(b + 1) * n])
        });

        let mut ids = Vec::with_capacity(remapped.len());
        let mut probs = Vec::with_capacity(perm.len());
        let mut backoffs = Vec::with_capacity(perm.len());
        for (k, &i) in perm.iter().enumerate() {
            let gram = &remapped[i * n..(i + 1) * n];
            if k > 0 && ids[(k - 1) * n..] == *gram {
                return Err(Error::Arpa {
                    line: 0,
                    message: format!("duplicate n-gram {:?}", render(gram, words)),
                });
            }
            ids.extend_from_slice(gram);
            probs.push(prob_book.encode(source.probs[i]));
            backoffs.push(backoff_book.encode(source.backoffs[i]));
        }
        Ok(Self {
            n,
            ids,
            probs,
            backoffs,
            prob_book: prob_book.clone(),
            backoff_book: backoff_book.clone(),
        })
    }

    fn len(&self) -> usize {
        self.probs.len()
    }

    fn gram(&self, i: usize) -> &[WordId] {
        &self.ids[i * self.n..(i + 1) * self.n]
    }
}

fn render(gram: &[WordId], words: &[&str]) -> String {
    gram.iter()
        .map(|&id| words[id as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn mean_error(book: &Codebook, values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values
        .iter()
        .map(|&v| f64::from((book.decode(book.encode(v)) - v).abs()))
        .sum();
    (sum / values.len() as f64) as f32
}

/// For each n-gram of `parents` (plus a sentinel), the index of its first
/// extension in `children`.
fn child_pointers(
    parents: &SortedOrder,
    children: &SortedOrder,
    words: &[&str],
) -> Result<Vec<u32>> {
    let orphan = |k: usize| Error::Arpa {
        line: 0,
        message: format!(
            "n-gram {:?} has no entry for its prefix; every n-gram's first {} words must be an n-gram too",
            render(children.gram(k), words),
            parents.n
        ),
    };
    let mut pointers = Vec::with_capacity(parents.len() + 1);
    let mut k = 0;
    for j in 0..parents.len() {
        let parent = parents.gram(j);
        if k < children.len() && children.gram(k)[..parents.n] < *parent {
            return Err(orphan(k));
        }
        pointers.push(u32::try_from(k).expect("order size checked"));
        while k < children.len() && children.gram(k)[..parents.n] == *parent {
            k += 1;
        }
    }
    if k < children.len() {
        return Err(orphan(k));
    }
    pointers.push(u32::try_from(k).expect("order size checked"));
    Ok(pointers)
}

fn put_len(out: &mut Vec<u8>, len: usize) -> Result<()> {
    out.put_u32(u32::try_from(len).map_err(|_| invalid("section too large for the format"))?);
    Ok(())
}

fn meta_section(meta: &[(String, String)]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    put_len(&mut out, meta.len())?;
    for (key, value) in meta {
        for text in [key, value] {
            put_len(&mut out, text.len())?;
            out.extend_from_slice(text.as_bytes());
        }
    }
    Ok(out)
}

fn info_section(orders: &[SortedOrder], vocab_len: u32, words: &[&str]) -> Vec<u8> {
    let find = |word: &str| {
        words
            .binary_search_by(|w| w.as_bytes().cmp(word.as_bytes()))
            .map_or(NO_WORD, |id| {
                u32::try_from(id).expect("vocabulary fits u32")
            })
    };
    let mut out = Vec::new();
    out.put_u32(u32::try_from(orders.len()).expect("order checked"));
    out.put_u32(vocab_len);
    out.put_u32(find("<s>"));
    out.put_u32(find("</s>"));
    out.put_u32(find("<unk>"));
    for order in orders {
        out.put_u64(order.len() as u64);
    }
    out
}

fn vocab_section(words: &[&str]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut offset = 0;
    put_len(&mut out, offset)?;
    for word in words {
        offset += word.len();
        put_len(&mut out, offset)?;
    }
    for word in words {
        out.extend_from_slice(word.as_bytes());
    }
    Ok(out)
}

fn quant_section(orders: &[SortedOrder]) -> Vec<u8> {
    let mut out = Vec::new();
    for order in orders {
        for book in [&order.prob_book, &order.backoff_book] {
            for &center in book.centers() {
                out.put_f32(center);
            }
        }
    }
    out
}

fn ngram_section(
    sorted: &SortedOrder,
    children: Option<&[u32]>,
    n: usize,
    order: usize,
) -> Vec<u8> {
    let rows = sorted.len() + usize::from(n < order || n == 1);
    let mut out = Vec::with_capacity(rows * record_len(n, order));
    for i in 0..rows {
        let real = i < sorted.len();
        let child = children.map_or(0, |c| c[i]);
        let (prob, backoff) = if real {
            (sorted.probs[i], sorted.backoffs[i])
        } else {
            (0, 0)
        };
        if n == 1 {
            out.push(prob);
            out.push(backoff);
            out.put_u32(child);
        } else {
            out.put_u32(if real { sorted.gram(i)[n - 1] } else { 0 });
            out.push(prob);
            if n < order {
                out.push(backoff);
                out.put_u32(child);
            }
        }
    }
    out
}

/// Returns the section and the number of entries stored.
fn casing_section(casing: &[(String, String)], words: &[&str]) -> Result<(Vec<u8>, usize)> {
    let index: HashMap<&str, u32> = words.iter().copied().zip(0..).collect();
    let mut entries: Vec<(u32, &str)> = casing
        .iter()
        .filter_map(|(word, form)| Some((*index.get(word.as_str())?, form.as_str())))
        .collect();
    entries.sort_by_key(|&(id, _)| id);
    entries.dedup_by_key(|&mut (id, _)| id);

    let mut out = Vec::new();
    put_len(&mut out, entries.len())?;
    for &(id, _) in &entries {
        out.put_u32(id);
    }
    let mut offset = 0;
    put_len(&mut out, offset)?;
    for (_, form) in &entries {
        offset += form.len();
        put_len(&mut out, offset)?;
    }
    for (_, form) in &entries {
        out.extend_from_slice(form.as_bytes());
    }
    Ok((out, entries.len()))
}

/// Writes the header and sections; returns the total size.
fn write_sections(sections: &[([u8; 4], Vec<u8>)], out: &mut impl Write) -> Result<u64> {
    let mut header = Vec::with_capacity(HEADER_LEN + sections.len() * SECTION_ENTRY_LEN + 4);
    header.extend_from_slice(&MAGIC);
    header.put_u32(VERSION);
    header.put_u32(u32::try_from(sections.len()).expect("few sections"));
    let mut offset = (HEADER_LEN + sections.len() * SECTION_ENTRY_LEN + 4) as u64;
    for (tag, data) in sections {
        header.extend_from_slice(tag);
        header.put_u32(crc32fast::hash(data));
        header.put_u64(offset);
        header.put_u64(data.len() as u64);
        offset += data.len() as u64;
    }
    header.put_u32(crc32fast::hash(&header));

    out.write_all(&header)?;
    for (_, data) in sections {
        out.write_all(data)?;
    }
    out.flush()?;
    Ok(offset)
}
