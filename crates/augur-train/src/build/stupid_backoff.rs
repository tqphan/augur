//! Stupid Backoff (Brants et al., 2007): relative frequencies with a
//! constant back-off factor. Scores are not normalized probabilities.

use super::arpa::ArpaWriter;
use super::count::{Counts, Gram, Vocab};
use super::for_each_context;
use crate::Result;

/// Writes Stupid Backoff scores for all orders.
///
/// # Errors
///
/// If temporary files cannot be read or written.
pub(super) fn write(
    counts: &mut Counts,
    vocab: &Vocab,
    min_count: &[u64],
    alpha: f64,
    arpa: &mut ArpaWriter,
) -> Result<()> {
    let order = min_count.len();
    let backoff = (order > 1).then_some(alpha);

    let mut unigrams: Vec<u64> = vec![0; vocab.len()];
    while let Some(gram) = counts.next(1)? {
        unigrams[gram.ids[0] as usize] = gram.raw;
    }
    let total: u64 = unigrams.iter().sum();
    for (id, &raw) in unigrams.iter().enumerate() {
        let id = u32::try_from(id).expect("vocabulary fits u32");
        let prob = if id == vocab.bos {
            0.0
        } else {
            ratio(raw, total)
        };
        arpa.entry(vocab, &[id], prob, backoff)?;
    }

    for n in 2..=order {
        let backoff = (n < order).then_some(alpha);
        for_each_context(counts, n, |group: &[Gram]| {
            let total: u64 = group.iter().map(|g| g.raw).sum();
            for gram in group.iter().filter(|g| g.raw >= min_count[n - 1]) {
                arpa.entry(vocab, &gram.ids, ratio(gram.raw, total), backoff)?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)] // counts far below 2^52
fn ratio(count: u64, total: u64) -> f64 {
    count as f64 / total as f64
}
