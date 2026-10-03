/// Errors produced by augur.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A language tag is not a syntactically valid BCP-47 tag.
    #[error("invalid language tag: {0:?}")]
    InvalidLanguageTag(String),

    /// The engine or a component was configured inconsistently.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// No built-in language pack exists for the language.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// A model file is corrupted, has an unsupported format or does not fit
    /// the configuration.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// A user store is corrupted, in use by another engine, from an
    /// incompatible version or for another language.
    #[error("user store: {0}")]
    UserStore(String),

    /// An I/O error, e.g. while opening a model or user store.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias for results with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
