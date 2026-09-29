use std::borrow::Cow;
use std::fmt;

use crate::{Context, Error, Result, Token, TokenizedContext};

/// A BCP-47 language tag such as `en` or `de-CH`.
///
/// Only the syntax is checked (ASCII alphanumeric subtags of 1–8 characters
/// separated by `-`); the tag is not validated against the IANA registry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguageId(String);

impl LanguageId {
    /// Parses a language tag.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLanguageTag`] if `tag` is not syntactically
    /// valid.
    pub fn new(tag: &str) -> Result<Self> {
        let valid = !tag.is_empty()
            && tag.split('-').all(|subtag| {
                (1..=8).contains(&subtag.len()) && subtag.bytes().all(|b| b.is_ascii_alphanumeric())
            });
        if valid {
            Ok(Self(tag.to_owned()))
        } else {
            Err(Error::InvalidLanguageTag(tag.to_owned()))
        }
    }

    /// The tag as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LanguageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Splits text into normalized tokens for a language.
pub trait Tokenizer: Send + Sync {
    /// Tokenizes the text around the cursor for prediction.
    fn tokenize_context(&self, context: Context<'_>) -> TokenizedContext;

    /// Tokenizes committed text for learning, inserting
    /// [`Token::SentenceStart`] at sentence boundaries.
    fn tokenize(&self, text: &str) -> Vec<Token>;
}

/// Everything language-specific that the engine needs.
pub trait LanguagePack: Send + Sync {
    /// The language this pack handles.
    fn id(&self) -> &LanguageId;

    /// The tokenizer for this language.
    fn tokenizer(&self) -> &dyn Tokenizer;

    /// Normalizes a word for lookup (case folding, apostrophe unification,
    /// Unicode normalization).
    fn normalize<'a>(&self, word: &'a str) -> Cow<'a, str>;

    /// Restores display casing for a suggested word in the given context,
    /// e.g. capitalizing at the start of a sentence.
    fn restore_case(&self, word: &str, _context: &TokenizedContext) -> String {
        word.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_tags() {
        for tag in ["en", "de-CH", "zh-Hant-TW", "sr-Latn"] {
            assert_eq!(LanguageId::new(tag).unwrap().as_str(), tag);
        }
    }

    #[test]
    fn rejects_invalid_tags() {
        for tag in ["", "-", "en-", "en_US", "toolongsubtag", "é"] {
            assert!(LanguageId::new(tag).is_err(), "{tag:?} should be rejected");
        }
    }
}
