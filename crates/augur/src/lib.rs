//! augur: word prediction.
//!
//! Build an [`Engine`] from a [`LanguagePack`], one or more [`Predictor`]s and
//! a [`Combiner`], then call [`Engine::predict`] with the text around the
//! cursor.

mod engine;

pub use augur_core::*;
pub use engine::{Engine, EngineBuilder};
