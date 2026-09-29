//! `augur-train build`: estimates an n-gram model from a corpus and writes
//! it in ARPA format.

mod arpa;
mod count;
mod kneser_ney;
mod stupid_backoff;

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::Result;
use crate::meta::Meta;
use crate::sort::TempDir;

pub use count::MAX_ORDER;
use count::{Counts, Gram};
use kneser_ney::Discounts;

/// Smoothing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Smoothing {
    /// Interpolated modified Kneser-Ney.
    Mkn,
    /// Stupid Backoff (unnormalized scores).
    StupidBackoff,
}

/// Settings for [`run`].
#[derive(Debug, Clone)]
pub struct BuildOptions {
    /// Corpus with one tokenized sentence per line.
    pub corpus: PathBuf,
    /// Output ARPA file.
    pub output: PathBuf,
    /// Smoothing method.
    pub smoothing: Smoothing,
    /// Minimum raw count per order (index 0 is unigrams and must be 1).
    pub min_count: Vec<u64>,
    /// Number of words kept in the vocabulary, excluding markers.
    pub vocab_size: usize,
    /// Back-off factor for Stupid Backoff.
    pub alpha: f64,
    /// Approximate memory budget in bytes.
    pub memory: usize,
    /// Directory for temporary files.
    pub tmp_dir: PathBuf,
}

/// Builds the model.
///
/// # Errors
///
/// On I/O errors or an empty corpus.
pub fn run(options: &BuildOptions) -> Result<()> {
    let order = options.min_count.len();
    let tmp = TempDir::new(&options.tmp_dir)?;

    eprintln!("Reading vocabulary from {}", options.corpus.display());
    let (vocab, stats) = count::vocabulary(&options.corpus, options.vocab_size)?;
    if stats.sentences == 0 {
        return Err("the corpus has no sentences".into());
    }
    eprintln!(
        "{} sentences, {} tokens, {} distinct words; vocabulary {} ({} tokens become <unk>)",
        stats.sentences,
        stats.tokens,
        stats.types,
        vocab.len(),
        stats.unknown_tokens
    );

    eprintln!("Counting {order}-grams");
    let mut counts = count::count(&options.corpus, &vocab, order, options.memory, &tmp)?;

    eprintln!("Estimating probabilities");
    let mut arpa = arpa::ArpaWriter::new(order, &tmp)?;
    let mut discounts = Vec::new();
    match options.smoothing {
        Smoothing::Mkn => {
            for (n, count_of_counts) in counts.count_of_counts.iter().enumerate() {
                discounts.push(Discounts::estimate(count_of_counts).unwrap_or_else(|| {
                    eprintln!(
                        "warning: cannot estimate order-{} discounts from counts of counts {:?}; using {:?}",
                        n + 1,
                        &count_of_counts[1..],
                        Discounts::FALLBACK.0
                    );
                    Discounts::FALLBACK
                }));
            }
            kneser_ney::write(
                &mut counts,
                &vocab,
                &options.min_count,
                &discounts,
                options.memory,
                &tmp,
                &mut arpa,
            )?;
        }
        Smoothing::StupidBackoff => {
            stupid_backoff::write(
                &mut counts,
                &vocab,
                &options.min_count,
                options.alpha,
                &mut arpa,
            )?;
        }
    }
    let ngrams = arpa.finish(&options.output)?;
    eprintln!("Wrote {} ({ngrams:?} n-grams)", options.output.display());

    let casing_source = options.corpus.with_extension("casing");
    let casing_output = options.output.with_extension("casing");
    let casing_entries = if casing_source.exists() {
        Some(filter_casing(&casing_source, &casing_output, &vocab)?)
    } else {
        None
    };

    let mut meta = Meta::read_if_exists(&options.corpus.with_extension("meta"))?;
    meta.set("model.format", "arpa");
    meta.set("model.order", order);
    meta.set(
        "model.smoothing",
        match options.smoothing {
            Smoothing::Mkn => "modified-kneser-ney",
            Smoothing::StupidBackoff => "stupid-backoff",
        },
    );
    if options.smoothing == Smoothing::StupidBackoff {
        meta.set("model.alpha", options.alpha);
        meta.set("model.normalized", "false");
    }
    for (n, d) in discounts.iter().enumerate() {
        meta.set(
            &format!("model.discounts.{}", n + 1),
            format!("{:.6} {:.6} {:.6}", d.0[0], d.0[1], d.0[2]),
        );
    }
    meta.set("model.vocab_size", vocab.len());
    meta.set("model.min_count", join(&options.min_count));
    meta.set("model.ngrams", join(&ngrams));
    meta.set("corpus.sentences", stats.sentences);
    meta.set("corpus.tokens", stats.tokens);
    meta.set("corpus.types", stats.types);
    meta.set("corpus.unknown_tokens", stats.unknown_tokens);
    if let Some(entries) = casing_entries {
        meta.set("model.casing_entries", entries);
    }
    meta.write(&options.output.with_extension("meta"))?;
    Ok(())
}

/// Calls `f` with each group of n-grams of order `n` that share a context
/// (all words but the last).
fn for_each_context(
    counts: &mut Counts,
    n: usize,
    mut f: impl FnMut(&[Gram]) -> Result<()>,
) -> Result<()> {
    let mut group: Vec<Gram> = Vec::new();
    while let Some(gram) = counts.next(n)? {
        if group
            .first()
            .is_some_and(|first| first.ids[..n - 1] != gram.ids[..n - 1])
        {
            f(&group)?;
            group.clear();
        }
        group.push(gram);
    }
    if !group.is_empty() {
        f(&group)?;
    }
    Ok(())
}

/// Copies the casing entries of in-vocabulary words.
fn filter_casing(source: &Path, output: &Path, vocab: &count::Vocab) -> Result<u64> {
    let words: HashSet<&str> = (0..vocab.len())
        .map(|id| vocab.word(u32::try_from(id).expect("vocabulary fits u32")))
        .collect();
    let mut out = BufWriter::new(File::create(output)?);
    let mut entries = 0;
    for line in BufReader::new(File::open(source)?).lines() {
        let line = line?;
        if line
            .split_once('\t')
            .is_some_and(|(word, _)| words.contains(word))
        {
            writeln!(out, "{line}")?;
            entries += 1;
        }
    }
    out.flush()?;
    Ok(entries)
}

fn join<T: ToString>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests;
