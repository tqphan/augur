use crate::{Candidate, Suggestion};

/// The candidates produced by one predictor for one request.
#[derive(Debug, Clone, Copy)]
pub struct PredictorOutput<'a> {
    /// Index of the predictor in the engine; used for [`crate::SourceMask`].
    pub index: usize,
    /// The predictor's candidates, most likely first.
    pub candidates: &'a [Candidate],
}

/// Merges the outputs of all predictors into a single ranked list.
pub trait Combiner: Send + Sync {
    /// Returns at most `max` suggestions, most likely first, with no
    /// duplicate words.
    fn combine(&self, outputs: &[PredictorOutput<'_>], max: usize) -> Vec<Suggestion>;
}
