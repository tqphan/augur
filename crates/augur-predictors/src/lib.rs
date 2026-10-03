//! Predictor implementations for augur.
//!
//! - [`NgramPredictor`]: next-word prediction and word completion from a
//!   static back-off n-gram model ([`augur_lm::Model`]).
//! - [`RecencyPredictor`]: recently typed words, regardless of context.

mod ngram;
mod recency;

pub use ngram::NgramPredictor;
pub use recency::{RecencyConfig, RecencyPredictor};
