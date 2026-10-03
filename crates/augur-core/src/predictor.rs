use std::borrow::Cow;

use crate::{Result, Token, TokenSpan, TokenizedContext};

/// A word proposed by a single [`Predictor`].
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The normalized word.
    pub word: String,
    /// The predictor's probability for this word, in `0.0..=1.0`.
    pub prob: f32,
}

/// Committed text to learn from, as handed to [`Predictor::learn`].
#[derive(Debug, Clone, Copy)]
pub struct LearnInput<'a> {
    /// Tokens of the sentence in progress that were learned by earlier
    /// calls, oldest first (starting with [`Token::SentenceStart`] if the
    /// sentence start is known). They are history for the new tokens only
    /// and must not be counted again. Empty when the new tokens start a
    /// sentence.
    pub context: &'a [Token],
    /// The new tokens, with byte ranges into [`LearnInput::text`].
    pub tokens: &'a [TokenSpan],
    /// The text the token spans refer to.
    pub text: &'a str,
}

impl LearnInput<'_> {
    /// The text that token `index` of [`LearnInput::tokens`] was read
    /// from, as typed (empty for a sentence start).
    ///
    /// # Panics
    ///
    /// If `index` is out of range.
    pub fn surface(&self, index: usize) -> &str {
        &self.text[self.tokens[index].span.clone()]
    }
}

/// A source of word predictions, e.g. an n-gram model or a recency cache.
pub trait Predictor: Send + Sync {
    /// A short, stable name used in configuration and diagnostics.
    fn name(&self) -> &str;

    /// Appends up to `max` candidates for `context` to `out`, most likely
    /// first. Candidates must start with `context.prefix`.
    fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>);

    /// Learns from committed text. The default does nothing, for static
    /// predictors.
    fn learn(&self, _input: &LearnInput<'_>) {}

    /// Forgets everything learned about a normalized word. The default does
    /// nothing.
    ///
    /// # Errors
    ///
    /// If persisted data cannot be updated.
    fn forget(&self, _word: &str) -> Result<()> {
        Ok(())
    }

    /// Forgets everything learned. The default does nothing.
    ///
    /// # Errors
    ///
    /// If persisted data cannot be updated.
    fn clear(&self) -> Result<()> {
        Ok(())
    }

    /// Writes learned data that is not yet persisted. The default does
    /// nothing.
    ///
    /// # Errors
    ///
    /// If the data cannot be written.
    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// The usual display form of a normalized word, if this predictor knows
    /// one (e.g. `London` for `london`). The engine asks its predictors in
    /// order and uses the first answer; the language pack then applies
    /// context-dependent casing such as sentence-initial capitals. The
    /// default knows none.
    fn display_form(&self, _word: &str) -> Option<Cow<'_, str>> {
        None
    }
}

/// Whether a normalized word is a marker such as `<s>`, `<unk>` or `<num>`
/// (written `<…>`), which predictors never suggest.
pub fn is_marker(word: &str) -> bool {
    word.len() > 2 && word.starts_with('<') && word.ends_with('>')
}
