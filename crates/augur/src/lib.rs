//! augur: word prediction.
//!
//! The quickest start is [`Engine::from_model`], which sets up the built-in
//! language pack, an n-gram predictor and the default combiner for a model
//! file. For other setups, build an [`Engine`] from a [`LanguagePack`], one
//! or more [`Predictor`]s and a [`Combiner`]. Then call [`Engine::predict`]
//! with the text around the cursor.

mod combiner;
mod engine;

pub use augur_core::*;
pub use augur_lang_en::EnglishPack;
pub use augur_lm::Model;
pub use augur_predictors::NgramPredictor;
pub use combiner::LinearCombiner;
pub use engine::{Engine, EngineBuilder};
