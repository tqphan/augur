//! N-gram language model format and reader for augur.
//!
//! Models are back-off n-gram models (as in ARPA files) stored in a
//! compact binary file that is read into memory in one piece:
//!
//! - the vocabulary is a sorted string table, so word ids follow the byte
//!   order of the words and a typed prefix maps to a contiguous id range;
//! - n-grams form a sorted trie, one array of fixed-width records per
//!   order, so all continuations of a context can be listed;
//! - probabilities and back-off weights are quantized to 8 bits with a
//!   256-entry codebook per order;
//! - provenance metadata and a casing table travel in the same file;
//! - every section carries a CRC-32, checked when the model is loaded.
//!
//! [`Arpa::read`] imports an ARPA file and [`write_model`] writes the binary
//! format; [`Model`] reads it.

mod arpa;
mod error;
mod format;
mod model;
mod quantize;
mod write;

pub use arpa::Arpa;
pub use error::{Error, Result};
pub use model::{Continuations, Model, NgramEntry};
pub use write::{WriteStats, write_model};

/// Identifies a word within one [`Model`].
pub type WordId = u32;

/// Log10 probability used for impossible events, such as predicting `<s>`
/// (the ARPA convention).
pub const LOG_ZERO: f32 = -99.0;

/// Highest supported n-gram order.
pub const MAX_ORDER: usize = 6;

/// Stored in place of a missing special word id.
const NO_WORD: WordId = u32::MAX;

#[cfg(test)]
mod tests;
