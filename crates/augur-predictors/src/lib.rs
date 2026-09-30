//! Predictor implementations for augur.
//!
//! - [`NgramPredictor`]: next-word prediction and word completion from a
//!   static back-off n-gram model ([`augur_lm::Model`]).

mod ngram;

pub use ngram::NgramPredictor;
