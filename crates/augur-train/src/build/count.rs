//! Vocabulary selection and n-gram counting.
//!
//! Counting follows the approach of KenLM's `lmplz`:
//!
//! 1. Each sentence is padded as `<s> w1 … wk </s>`. At every position the
//!    longest n-gram ending there (at most `order` words) is emitted. These
//!    are full-order n-grams, or shorter ones that begin with `<s>`.
//! 2. They are sorted by *suffix* (last word first) and deduplicated. In
//!    that order all n-grams sharing a suffix are adjacent, so a single pass
//!    yields every lower-order n-gram with its raw count and its Kneser-Ney
//!    adjusted count (the number of distinct words seen before it).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use augur_text::{SENTENCE_END, SENTENCE_START, UNKNOWN};

use crate::Result;
use crate::sort::{Sorted, Sorter, TempDir, add_u64_at, read_id, read_u64};

/// Marks unused slots of short n-grams in suffix-ordered keys.
const PAD: u32 = u32::MAX;

/// The model vocabulary; word ids follow the lexicographic order of words.
#[derive(Debug)]
pub(super) struct Vocab {
    words: Vec<String>,
    ids: HashMap<String, u32>,
    /// Id of `<s>`.
    pub(super) bos: u32,
    /// Id of `</s>`.
    pub(super) eos: u32,
    /// Id of `<unk>`.
    pub(super) unk: u32,
}

impl Vocab {
    /// Chooses the `size` most frequent words of the corpus (ties broken
    /// alphabetically) plus `<s>`, `</s>` and `<unk>`.
    fn from_counts(counts: HashMap<String, u64>, size: usize) -> Self {
        let mut by_count: Vec<(String, u64)> = counts.into_iter().collect();
        by_count.sort_unstable_by(|(wa, ca), (wb, cb)| cb.cmp(ca).then_with(|| wa.cmp(wb)));
        by_count.truncate(size);
        let mut words: Vec<String> = by_count.into_iter().map(|(word, _)| word).collect();
        words.extend([SENTENCE_START, SENTENCE_END, UNKNOWN].map(str::to_owned));
        words.sort_unstable();
        let ids: HashMap<String, u32> = words
            .iter()
            .enumerate()
            .map(|(id, word)| {
                (
                    word.clone(),
                    u32::try_from(id).expect("vocabulary fits u32"),
                )
            })
            .collect();
        Self {
            bos: ids[SENTENCE_START],
            eos: ids[SENTENCE_END],
            unk: ids[UNKNOWN],
            words,
            ids,
        }
    }

    /// Number of entries, including `<s>`, `</s>` and `<unk>`.
    pub(super) fn len(&self) -> usize {
        self.words.len()
    }

    /// The word with the given id.
    pub(super) fn word(&self, id: u32) -> &str {
        &self.words[id as usize]
    }

    fn id(&self, word: &str) -> u32 {
        self.ids.get(word).copied().unwrap_or(self.unk)
    }
}

/// Corpus statistics gathered while counting.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CorpusStats {
    /// Sentences read.
    pub(super) sentences: u64,
    /// Word tokens read (excluding `<s>` and `</s>`).
    pub(super) tokens: u64,
    /// Distinct words in the corpus.
    pub(super) types: u64,
    /// Word tokens mapped to `<unk>`.
    pub(super) unknown_tokens: u64,
}

/// One n-gram with its counts, as produced by [`Counts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Gram {
    /// Word ids, first word first.
    pub(super) ids: Vec<u32>,
    /// Number of occurrences.
    pub(super) raw: u64,
    /// Kneser-Ney adjusted count: the raw count for full-order n-grams and
    /// n-grams starting with `<s>`, otherwise the number of distinct words
    /// seen before the n-gram.
    pub(super) adjusted: u64,
}

/// All n-grams of each order, sorted by word ids.
pub(super) struct Counts {
    orders: Vec<Sorted>,
    /// `count_of_counts[n][k]`: number of n-grams of order `n + 1` whose
    /// adjusted count is `k` (for `k` in 1..=4).
    pub(super) count_of_counts: Vec<[u64; 5]>,
}

impl Counts {
    /// Next n-gram of `order` (1-based), in ascending id order.
    ///
    /// # Errors
    ///
    /// If temporary files cannot be read.
    pub(super) fn next(&mut self, order: usize) -> Result<Option<Gram>> {
        let Some(record) = self.orders[order - 1].next_record()? else {
            return Ok(None);
        };
        let ids = (0..order).map(|i| read_id(&record[4 * i..])).collect();
        Ok(Some(Gram {
            ids,
            raw: read_u64(&record[4 * order..]),
            adjusted: read_u64(&record[4 * order + 8..]),
        }))
    }
}

/// Reads the corpus once to choose the vocabulary.
///
/// # Errors
///
/// If the corpus cannot be read.
pub(super) fn vocabulary(corpus: &Path, size: usize) -> Result<(Vocab, CorpusStats)> {
    let mut counts: HashMap<String, u64> = HashMap::new();
    let mut stats = CorpusStats::default();
    for line in BufReader::with_capacity(1 << 20, File::open(corpus)?).lines() {
        let line = line?;
        let mut words = line.split(' ').filter(|w| !w.is_empty()).peekable();
        if words.peek().is_none() {
            continue;
        }
        stats.sentences += 1;
        for word in words {
            stats.tokens += 1;
            if let Some(count) = counts.get_mut(word) {
                *count += 1;
            } else {
                counts.insert(word.to_owned(), 1);
            }
        }
    }
    stats.types = counts.len() as u64;
    let kept: u64 = {
        let mut sorted: Vec<u64> = counts.values().copied().collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        sorted.iter().take(size).sum()
    };
    stats.unknown_tokens = stats.tokens - kept;
    Ok((Vocab::from_counts(counts, size), stats))
}

/// Sorts the longest n-gram ending at each corpus position by suffix, with
/// duplicates counted.
fn maximal_ngrams(
    corpus: &Path,
    vocab: &Vocab,
    order: usize,
    memory: usize,
    tmp: &TempDir,
) -> Result<Sorted> {
    let key_len = 4 * order;
    let mut maximal = Sorter::new(
        tmp,
        "maximal",
        key_len + 8,
        key_len,
        memory,
        Some(combine_count(order)),
    );
    let mut ids: Vec<u32> = Vec::new();
    let mut record = Vec::with_capacity(key_len + 8);
    for line in BufReader::with_capacity(1 << 20, File::open(corpus)?).lines() {
        let line = line?;
        ids.clear();
        ids.push(vocab.bos);
        ids.extend(
            line.split(' ')
                .filter(|w| !w.is_empty())
                .map(|w| vocab.id(w)),
        );
        if ids.len() == 1 {
            continue;
        }
        ids.push(vocab.eos);
        for end in 1..ids.len() {
            let start = (end + 1).saturating_sub(order);
            record.clear();
            for &id in ids[start..=end].iter().rev() {
                record.extend_from_slice(&id.to_be_bytes());
            }
            record.resize(key_len, 0xFF); // PAD
            record.extend_from_slice(&1u64.to_le_bytes());
            maximal.push(&record)?;
        }
    }
    maximal.finish()
}

/// Reads the corpus a second time and counts n-grams of every order up to
/// `order`, using about `memory` bytes of RAM.
///
/// # Errors
///
/// If the corpus or temporary files cannot be read or written.
pub(super) fn count(
    corpus: &Path,
    vocab: &Vocab,
    order: usize,
    memory: usize,
    tmp: &TempDir,
) -> Result<Counts> {
    let key_len = 4 * order;
    let mut maximal = maximal_ngrams(corpus, vocab, order, memory, tmp)?;

    let per_order_memory = memory / order;
    let mut sorters: Vec<Sorter<'_>> = (1..=order)
        .map(|n| Sorter::new(tmp, "counts", 4 * n + 16, 4 * n, per_order_memory, None))
        .collect();
    let mut count_of_counts = vec![[0u64; 5]; order];
    let mut levels: Vec<Level> = (0..=order).map(|_| Level::default()).collect();
    let mut emit = |n: usize, level: &Level| -> Result<()> {
        let adjusted = if level.full {
            level.raw
        } else {
            level.adjusted
        };
        if let Some(slot) = count_of_counts[n - 1].get_mut(usize::try_from(adjusted).unwrap_or(0)) {
            *slot += 1;
        }
        let mut record = Vec::with_capacity(4 * n + 16);
        for &id in level.key.iter().rev() {
            record.extend_from_slice(&id.to_be_bytes());
        }
        record.extend_from_slice(&level.raw.to_le_bytes());
        record.extend_from_slice(&adjusted.to_le_bytes());
        sorters[n - 1].push(&record)
    };

    let mut reversed = vec![0u32; order];
    while let Some(record) = maximal.next_record()? {
        for (i, id) in reversed.iter_mut().enumerate() {
            *id = read_id(&record[4 * i..]);
        }
        let len = reversed.iter().take_while(|&&id| id != PAD).count();
        let count = read_u64(&record[key_len..]);
        // First order whose suffix differs from the previous n-gram's.
        let changed = (1..=len)
            .find(|&n| !levels[n].active || levels[n].key[..] != reversed[..n])
            .expect("maximal n-grams are distinct");
        for n in (changed..=order).rev() {
            if levels[n].active {
                emit(n, &levels[n])?;
                levels[n].active = false;
            }
        }
        for (n, level) in levels.iter_mut().enumerate().take(len + 1).skip(changed) {
            *level = Level {
                active: true,
                key: reversed[..n].to_vec(),
                ..Level::default()
            };
        }
        for (n, level) in levels.iter_mut().enumerate().take(len + 1).skip(1) {
            level.raw += count;
            if n < len && n + 1 >= changed {
                level.adjusted += 1;
            }
        }
        levels[len].full = true;
    }
    for n in (1..=order).rev() {
        if levels[n].active {
            emit(n, &levels[n])?;
        }
    }

    let orders = sorters
        .into_iter()
        .map(Sorter::finish)
        .collect::<Result<Vec<_>>>()?;
    Ok(Counts {
        orders,
        count_of_counts,
    })
}

/// State for the suffix of one order during the suffix-ordered pass.
#[derive(Debug, Default)]
struct Level {
    active: bool,
    /// The suffix, last word first.
    key: Vec<u32>,
    raw: u64,
    adjusted: u64,
    /// Whether this n-gram was itself a maximal n-gram.
    full: bool,
}

fn combine_count(order: usize) -> fn(&mut [u8], &[u8]) {
    match order {
        1 => add_u64_at::<4>,
        2 => add_u64_at::<8>,
        3 => add_u64_at::<12>,
        4 => add_u64_at::<16>,
        5 => add_u64_at::<20>,
        6 => add_u64_at::<24>,
        _ => panic!("order must be 1..=6"),
    }
}

/// Highest supported n-gram order.
pub const MAX_ORDER: usize = 6;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn corpus(text: &str) -> (TempDir, std::path::PathBuf) {
        let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
        let path = tmp.file("corpus");
        File::create(&path)
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
        (tmp, path)
    }

    fn all(counts: &mut Counts, vocab: &Vocab, order: usize) -> Vec<(String, u64, u64)> {
        let mut out = Vec::new();
        while let Some(gram) = counts.next(order).unwrap() {
            let words: Vec<&str> = gram.ids.iter().map(|&id| vocab.word(id)).collect();
            out.push((words.join(" "), gram.raw, gram.adjusted));
        }
        out
    }

    #[test]
    fn vocabulary_keeps_most_frequent_words() {
        let (_tmp, path) = corpus("a b a\nc a b\n\n");
        let (vocab, stats) = vocabulary(&path, 2).unwrap();
        assert_eq!(vocab.words, ["</s>", "<s>", "<unk>", "a", "b"]);
        assert_eq!(vocab.id("c"), vocab.unk);
        assert_eq!(stats.sentences, 2);
        assert_eq!(stats.tokens, 6);
        assert_eq!(stats.types, 3);
        assert_eq!(stats.unknown_tokens, 1);
    }

    #[test]
    fn counts_raw_and_adjusted() {
        let (tmp, path) = corpus("a b\nb b\n");
        let (vocab, _) = vocabulary(&path, 10).unwrap();
        for memory in [1 << 20, 0] {
            let mut counts = count(&path, &vocab, 3, memory, &tmp).unwrap();
            // Unigrams: adjusted = distinct left contexts.
            assert_eq!(
                all(&mut counts, &vocab, 1),
                [
                    ("</s>".into(), 2, 1),
                    ("a".into(), 1, 1),
                    ("b".into(), 3, 3)
                ]
            );
            // Bigrams starting with <s> keep raw counts.
            assert_eq!(
                all(&mut counts, &vocab, 2),
                [
                    ("<s> a".into(), 1, 1),
                    ("<s> b".into(), 1, 1),
                    ("a b".into(), 1, 1),
                    ("b </s>".into(), 2, 2),
                    ("b b".into(), 1, 1),
                ]
            );
            assert_eq!(
                all(&mut counts, &vocab, 3),
                [
                    ("<s> a b".into(), 1, 1),
                    ("<s> b b".into(), 1, 1),
                    ("a b </s>".into(), 1, 1),
                    ("b b </s>".into(), 1, 1),
                ]
            );
            assert_eq!(counts.count_of_counts[1], [0, 4, 1, 0, 0]);
        }
    }
}
