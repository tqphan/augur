//! Core traits and types shared by all augur crates.
//!
//! The prediction pipeline is:
//!
//! 1. A [`LanguagePack`]'s [`Tokenizer`] turns the raw [`Context`] around the
//!    cursor into a [`TokenizedContext`].
//! 2. Each [`Predictor`] proposes [`Candidate`]s for that context.
//! 3. A [`Combiner`] merges the per-predictor candidates into ranked
//!    [`Suggestion`]s.

mod combiner;
mod context;
mod error;
mod language;
mod predictor;
mod suggestion;

pub use combiner::{Combiner, PredictorOutput};
pub use context::{Context, Token, TokenSpan, TokenizedContext};
pub use error::{Error, Result};
pub use language::{LanguageId, LanguagePack, Tokenizer};
pub use predictor::{Candidate, Predictor};
pub use suggestion::{SourceMask, Suggestion};
