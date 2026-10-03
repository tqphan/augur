//! augur: word prediction.
//!
//! The quickest start is [`Engine::from_model`], or
//! [`Engine::with_options`] to save what the engine learns from the user;
//! both set up the built-in language pack, a user model, an n-gram
//! predictor, a recency predictor and the default combiner for a model
//! file. For other setups, build an [`Engine`] from a [`LanguagePack`], one
//! or more [`Predictor`]s and a [`Combiner`]. Then call [`Engine::predict`]
//! with the text around the cursor, and [`Engine::learn`] with the text the
//! user commits.

mod combiner;
mod engine;
mod options;

pub use augur_core::*;
pub use augur_lang_en::EnglishPack;
pub use augur_lm::Model;
pub use augur_predictors::{NgramPredictor, RecencyConfig, RecencyPredictor};
pub use augur_user::{UserConfig, UserPredictor};
pub use combiner::LinearCombiner;
pub use engine::{Engine, EngineBuilder};
pub use options::EngineOptions;
