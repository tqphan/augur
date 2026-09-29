use crate::{Token, TokenizedContext};

/// A word proposed by a single [`Predictor`].
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The normalized word.
    pub word: String,
    /// The predictor's probability for this word, in `0.0..=1.0`.
    pub prob: f32,
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
    fn learn(&self, _tokens: &[Token]) {}
}
