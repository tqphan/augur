use std::path::{Path, PathBuf};

use augur_core::{Error, Result};
use augur_predictors::RecencyConfig;
use augur_user::UserConfig;

/// Configuration for [`Engine::with_options`](crate::Engine::with_options).
///
/// The engine combines three predictors: the static n-gram model, the
/// user model (learned n-grams, kept in memory and saved to
/// [`EngineOptions::user_store`] if set) and recently used words.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineOptions {
    pub(crate) user_store: Option<PathBuf>,
    pub(crate) weights: [f32; 3],
    pub(crate) learning: bool,
    pub(crate) user: UserConfig,
    pub(crate) recency: RecencyConfig,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            user_store: None,
            weights: [0.6, 0.3, 0.1],
            learning: true,
            user: UserConfig::default(),
            recency: RecencyConfig::default(),
        }
    }
}

impl EngineOptions {
    /// The defaults: no user store, weights 0.6 (n-gram), 0.3 (user) and
    /// 0.1 (recency), learning on.
    pub fn new() -> Self {
        Self::default()
    }

    /// Saves the user model in the file at `path`, created if needed.
    #[must_use]
    pub fn user_store(mut self, path: impl AsRef<Path>) -> Self {
        self.user_store = Some(path.as_ref().to_path_buf());
        self
    }

    /// Keeps the user model in memory only (the default).
    #[must_use]
    pub fn without_user_store(mut self) -> Self {
        self.user_store = None;
        self
    }

    /// Sets the combiner weights of the n-gram, user and recency
    /// predictors. They are relative: only their ratios matter.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] if a weight is negative or not finite, or all are
    /// zero.
    pub fn weights(mut self, ngram: f32, user: f32, recency: f32) -> Result<Self> {
        let weights = [ngram, user, recency];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || weights.iter().all(|&w| w == 0.0) {
            return Err(Error::Config(format!(
                "weights must be finite, non-negative and not all zero: {weights:?}"
            )));
        }
        self.weights = weights;
        Ok(self)
    }

    /// Whether the engine starts with learning on (the default).
    #[must_use]
    pub fn learning(mut self, on: bool) -> Self {
        self.learning = on;
        self
    }

    /// Sets the user model's limits.
    #[must_use]
    pub fn user_config(mut self, config: UserConfig) -> Self {
        self.user = config;
        self
    }

    /// Sets the recency predictor's configuration.
    #[must_use]
    pub fn recency_config(mut self, config: RecencyConfig) -> Self {
        self.recency = config;
        self
    }
}
