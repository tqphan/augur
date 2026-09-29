//! Builds augur language models from text corpora.
//!
//! The pipeline has two steps, each a subcommand of the `augur-train`
//! binary:
//!
//! 1. [`extract`]: MediaWiki XML dump → tokenized corpus (one sentence per
//!    line), a casing table and a provenance file.
//! 2. [`build`]: corpus → n-gram model in ARPA format, smoothed with
//!    modified Kneser-Ney or Stupid Backoff, counted with bounded memory.

pub mod build;
pub mod dump;
pub mod extract;
pub mod meta;
pub mod sort;
pub mod wikitext;

/// Error type of the training tools.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Result type of the training tools.
pub type Result<T, E = Error> = std::result::Result<T, E>;
