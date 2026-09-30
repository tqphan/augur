/// Errors produced while reading, importing or writing models.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The data does not start with the augur model magic bytes.
    #[error("not an augur language model")]
    BadMagic,

    /// The model was written in a format version this reader does not
    /// support.
    #[error("unsupported model format version {found} (this reader supports {supported})")]
    UnsupportedVersion {
        /// Version found in the file.
        found: u32,
        /// Version this reader supports.
        supported: u32,
    },

    /// A checksum did not match; the file is corrupted or truncated.
    #[error("checksum mismatch in {0} (the model file is corrupted or truncated)")]
    Checksum(String),

    /// The model is structurally invalid.
    #[error("invalid model: {0}")]
    Invalid(String),

    /// An ARPA file could not be parsed or is inconsistent.
    #[error("ARPA line {line}: {message}")]
    Arpa {
        /// 1-based line number (0 if the problem is not tied to a line).
        line: u64,
        /// What is wrong.
        message: String,
    },
}

/// Convenience alias for results with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
