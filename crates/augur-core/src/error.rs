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

    /// An I/O error, e.g. while opening a model or user store.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenience alias for results with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
