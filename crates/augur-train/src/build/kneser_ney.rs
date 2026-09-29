//! Interpolated modified Kneser-Ney smoothing (Chen & Goodman, 1998), as
//! estimated by KenLM's `lmplz`.
//!
//! For a context `h` and word `w` with adjusted count `a(hw)`:
//!
//! ```text
//! p(w | h) = (a(hw) - D(a(hw))) / a(h·)  +  γ(h) · p(w | h')
//! γ(h)     = (D1·N1(h·) + D2·N2(h·) + D3+·N3+(h·)) / a(h·)
//! ```
//!
//! where `h'` drops the first word of `h`, and unigrams interpolate with the
//! uniform distribution. N-grams removed by `--min-count` give their whole
//! adjusted count to `γ(h)`, which keeps every distribution normalized.
//!
//! Orders are processed lowest first. The n-grams of order `n` are sorted by
//! their suffix so they can be joined with the finished probabilities of
//! order `n - 1`, which are in the same order as that order's back-off
//! weights and ARPA lines.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use super::arpa::ArpaWriter;
use super::count::{Counts, Gram, Vocab};
use super::for_each_context;
use crate::Result;
use crate::sort::{Sorted, Sorter, TempDir, read_f64, read_id};

/// Discounts for n-grams with adjusted count 1, 2 and 3 or more.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Discounts(pub(super) [f64; 3]);

impl Discounts {
    /// Used when the counts of counts do not give valid estimates, as with
    /// KenLM's `--discount_fallback`.
    pub(super) const FALLBACK: Self = Self([0.5, 1.0, 1.5]);

    /// Estimates discounts from the number of n-grams with adjusted counts
    /// 1 to 4. Returns `None` if the estimate is invalid.
    #[allow(clippy::cast_precision_loss)]
    pub(super) fn estimate(count_of_counts: &[u64; 5]) -> Option<Self> {
        let [_, t1, t2, t3, t4] = count_of_counts.map(|t| t as f64);
        if t1 == 0.0 || t2 == 0.0 || t3 == 0.0 || t4 == 0.0 {
            return None;
        }
        let y = t1 / (t1 + 2.0 * t2);
        let d = [
            1.0 - 2.0 * y * t2 / t1,
            2.0 - 3.0 * y * t3 / t2,
            3.0 - 4.0 * y * t4 / t3,
        ];
        let valid = d.iter().zip(1..).all(|(&d, k)| d > 0.0 && d < f64::from(k));
        valid.then_some(Self(d))
    }

    fn of(&self, adjusted: u64) -> f64 {
        match adjusted {
            0 => 0.0,
            1 => self.0[0],
            2 => self.0[1],
            _ => self.0[2],
        }
    }
}

/// The finished probabilities of the previous order, in id order.
enum Lower {
    Unigrams { probs: Vec<f64>, next: usize },
    Grams { order: usize, sorted: Sorted },
}

impl Lower {
    fn next(&mut self) -> Result<Option<(Vec<u32>, f64)>> {
        match self {
            Self::Unigrams { probs, next } => {
                let Some(&prob) = probs.get(*next) else {
                    return Ok(None);
                };
                let id = u32::try_from(*next).expect("vocabulary fits u32");
                *next += 1;
                Ok(Some((vec![id], prob)))
            }
            Self::Grams { order, sorted } => {
                let order = *order;
                Ok(sorted.next_record()?.map(|record| {
                    let ids = (0..order).map(|i| read_id(&record[4 * i..])).collect();
                    (ids, read_f64(&record[4 * order..]))
                }))
            }
        }
    }
}

/// Writes interpolated modified Kneser-Ney probabilities for all orders.
///
/// # Errors
///
/// If temporary files cannot be read or written.
pub(super) fn write(
    counts: &mut Counts,
    vocab: &Vocab,
    min_count: &[u64],
    discounts: &[Discounts],
    memory: usize,
    tmp: &TempDir,
    arpa: &mut ArpaWriter,
) -> Result<()> {
    let mut lower = Lower::Unigrams {
        probs: unigrams(counts, vocab, &discounts[0])?,
        next: 0,
    };
    for n in 2..=min_count.len() {
        let order = Order {
            n,
            discounts: &discounts[n - 1],
            min_count: min_count[n - 1],
            memory: memory / 3,
            tmp,
        };
        let (pending, backoffs) = order.discount(counts)?;
        lower = Lower::Grams {
            order: n,
            sorted: order.interpolate(lower, pending, &backoffs, vocab, arpa)?,
        };
    }
    while let Some((ids, prob)) = lower.next()? {
        arpa.entry(vocab, &ids, prob, None)?;
    }
    Ok(())
}

/// Unigram probabilities by word id, interpolated with the uniform
/// distribution over every word except `<s>` (which gets 0).
#[allow(clippy::cast_precision_loss)]
fn unigrams(counts: &mut Counts, vocab: &Vocab, d: &Discounts) -> Result<Vec<f64>> {
    let mut adjusted = vec![0u64; vocab.len()];
    while let Some(gram) = counts.next(1)? {
        adjusted[gram.ids[0] as usize] = gram.adjusted;
    }
    let total = adjusted.iter().sum::<u64>() as f64;
    let gamma = adjusted.iter().map(|&a| d.of(a)).sum::<f64>() / total;
    let uniform = gamma / (vocab.len() - 1) as f64;
    Ok(adjusted
        .iter()
        .enumerate()
        .map(|(id, &a)| {
            if id == vocab.bos as usize {
                0.0
            } else {
                (a as f64 - d.of(a)) / total + uniform
            }
        })
        .collect())
}

/// Settings for estimating one order above unigrams.
struct Order<'a> {
    n: usize,
    discounts: &'a Discounts,
    min_count: u64,
    memory: usize,
    tmp: &'a TempDir,
}

impl Order<'_> {
    /// Computes discounted probabilities, sorted by suffix, and writes the
    /// back-off weight of each context to a file, in context order.
    #[allow(clippy::cast_precision_loss)]
    fn discount(&self, counts: &mut Counts) -> Result<(Sorted, PathBuf)> {
        let n = self.n;
        let d = self.discounts;
        let mut pending = Sorter::new(self.tmp, "pending", 4 * n + 16, 4 * n, self.memory, None);
        let backoff_path = self.tmp.file("backoff");
        let mut backoffs = BufWriter::with_capacity(1 << 20, File::create(&backoff_path)?);
        let mut record = Vec::with_capacity(4 * n + 16);
        for_each_context(counts, n, |group: &[Gram]| {
            let kept = |g: &Gram| g.raw >= self.min_count;
            if !group.iter().any(kept) {
                return Ok(());
            }
            let total = group.iter().map(|g| g.adjusted).sum::<u64>() as f64;
            let freed: f64 = group
                .iter()
                .map(|g| {
                    if kept(g) {
                        d.of(g.adjusted)
                    } else {
                        g.adjusted as f64
                    }
                })
                .sum();
            let gamma = freed / total;
            for &id in &group[0].ids[..n - 1] {
                backoffs.write_all(&id.to_be_bytes())?;
            }
            backoffs.write_all(&gamma.to_le_bytes())?;

            for gram in group.iter().filter(|g| kept(g)) {
                let discounted = (gram.adjusted as f64 - d.of(gram.adjusted)) / total;
                record.clear();
                for &id in gram.ids[1..].iter().chain(&gram.ids[..1]) {
                    record.extend_from_slice(&id.to_be_bytes());
                }
                record.extend_from_slice(&discounted.to_le_bytes());
                record.extend_from_slice(&gamma.to_le_bytes());
                pending.push(&record)?;
            }
            Ok(())
        })?;
        backoffs.flush()?;
        Ok((pending.finish()?, backoff_path))
    }

    /// Walks the lower order in id order: writes its ARPA lines with their
    /// back-off weights, and interpolates this order's probabilities.
    /// Returns this order's probabilities in id order.
    fn interpolate(
        &self,
        mut lower: Lower,
        mut pending: Sorted,
        backoff_path: &Path,
        vocab: &Vocab,
        arpa: &mut ArpaWriter,
    ) -> Result<Sorted> {
        let n = self.n;
        let context_len = 4 * (n - 1);
        let mut pending_record = Vec::new();
        let mut has_pending = advance(&mut pending, &mut pending_record)?;
        let mut backoffs = BufReader::with_capacity(1 << 20, File::open(backoff_path)?);
        let mut backoff_record = vec![0u8; context_len + 8];
        let mut has_backoff = read_record(&mut backoffs, &mut backoff_record)?;
        let mut finished = Sorter::new(self.tmp, "final", 4 * n + 8, 4 * n, self.memory, None);
        let mut key = Vec::with_capacity(context_len);
        let mut record = Vec::with_capacity(4 * n + 8);

        while let Some((ids, prob)) = lower.next()? {
            key.clear();
            for &id in &ids {
                key.extend_from_slice(&id.to_be_bytes());
            }
            let mut backoff = None;
            if has_backoff && backoff_record[..context_len] == key[..] {
                backoff = Some(read_f64(&backoff_record[context_len..]));
                has_backoff = read_record(&mut backoffs, &mut backoff_record)?;
            }
            arpa.entry(vocab, &ids, prob, backoff)?;

            while has_pending && pending_record[..context_len] == key[..] {
                let discounted = read_f64(&pending_record[4 * n..]);
                let gamma = read_f64(&pending_record[4 * n + 8..]);
                record.clear();
                record.extend_from_slice(&pending_record[context_len..4 * n]);
                record.extend_from_slice(&key);
                record.extend_from_slice(&(discounted + gamma * prob).to_le_bytes());
                finished.push(&record)?;
                has_pending = advance(&mut pending, &mut pending_record)?;
            }
        }
        if has_pending || has_backoff {
            return Err(
                format!("internal error: order-{n} n-gram without lower-order entry").into(),
            );
        }
        finished.finish()
    }
}

fn advance(sorted: &mut Sorted, buf: &mut Vec<u8>) -> Result<bool> {
    buf.clear();
    match sorted.next_record()? {
        Some(record) => {
            buf.extend_from_slice(record);
            Ok(true)
        }
        None => Ok(false),
    }
}

fn read_record(reader: &mut impl Read, buf: &mut [u8]) -> Result<bool> {
    match reader.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_discounts() {
        let d = Discounts::estimate(&[0, 100, 40, 20, 10]).unwrap();
        let y = 100.0 / 180.0;
        assert!((d.0[0] - (1.0 - 2.0 * y * 0.4)).abs() < 1e-12);
        assert!((d.0[1] - (2.0 - 3.0 * y * 0.5)).abs() < 1e-12);
        assert!((d.0[2] - (3.0 - 4.0 * y * 0.5)).abs() < 1e-12);
        assert_eq!(Discounts::estimate(&[0, 5, 0, 1, 1]), None);
    }
}
