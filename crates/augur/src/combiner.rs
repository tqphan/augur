use std::collections::HashMap;

use augur_core::{Combiner, PredictorOutput, SourceMask, Suggestion};

/// Linear interpolation: a word's probability is the weighted sum of the
/// probabilities the predictors gave it, `Σ wᵢ · pᵢ`, with the weights
/// normalized to sum to 1 over the predictors that proposed any word for
/// the request (so that, e.g., a user model that has learned nothing yet
/// does not dilute the others). A predictor that did not propose a word
/// contributes 0 for it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearCombiner {
    weights: Vec<f32>,
}

impl LinearCombiner {
    /// Equal weights for all predictors.
    pub fn new() -> Self {
        Self::default()
    }

    /// Weights by predictor index; predictors beyond the list get weight
    /// 1. Negative and non-finite weights count as 0.
    pub fn with_weights(weights: impl Into<Vec<f32>>) -> Self {
        Self {
            weights: weights.into(),
        }
    }

    fn weight(&self, index: usize) -> f32 {
        let weight = self.weights.get(index).copied().unwrap_or(1.0);
        if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            0.0
        }
    }
}

impl Combiner for LinearCombiner {
    fn combine(&self, outputs: &[PredictorOutput<'_>], max: usize) -> Vec<Suggestion> {
        let total: f32 = outputs
            .iter()
            .filter(|o| !o.candidates.is_empty())
            .map(|o| self.weight(o.index))
            .sum();
        if total <= 0.0 {
            return Vec::new();
        }
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut suggestions: Vec<Suggestion> = Vec::new();
        for output in outputs {
            let weight = self.weight(output.index) / total;
            if weight <= 0.0 {
                continue;
            }
            for candidate in output.candidates {
                let i = *index.entry(&candidate.word).or_insert_with(|| {
                    suggestions.push(Suggestion {
                        text: candidate.word.clone(),
                        prob: 0.0,
                        sources: SourceMask::default(),
                    });
                    suggestions.len() - 1
                });
                suggestions[i].prob += weight * candidate.prob;
                suggestions[i].sources.insert(output.index);
            }
        }
        // Stable: ties keep the order in which words were first proposed.
        suggestions.sort_by(|a, b| b.prob.total_cmp(&a.prob));
        suggestions.truncate(max);
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use augur_core::Candidate;

    use super::*;

    fn candidates(words: &[(&str, f32)]) -> Vec<Candidate> {
        words
            .iter()
            .map(|&(word, prob)| Candidate {
                word: word.to_owned(),
                prob,
            })
            .collect()
    }

    fn combine(
        combiner: &LinearCombiner,
        lists: &[Vec<Candidate>],
        max: usize,
    ) -> Vec<(String, f32)> {
        let outputs: Vec<PredictorOutput<'_>> = lists
            .iter()
            .enumerate()
            .map(|(index, candidates)| PredictorOutput { index, candidates })
            .collect();
        combiner
            .combine(&outputs, max)
            .into_iter()
            .map(|s| (s.text, s.prob))
            .collect()
    }

    fn assert_combined(actual: &[(String, f32)], expected: &[(&str, f32)]) {
        let words: Vec<&str> = actual.iter().map(|(w, _)| w.as_str()).collect();
        let expected_words: Vec<&str> = expected.iter().map(|&(w, _)| w).collect();
        assert_eq!(words, expected_words);
        for ((_, p), (_, q)) in actual.iter().zip(expected) {
            assert!((p - q).abs() < 1e-6, "{actual:?}");
        }
    }

    #[test]
    fn interpolates_and_merges() {
        let lists = [
            candidates(&[("a", 0.6), ("b", 0.4)]),
            candidates(&[("b", 0.8), ("c", 0.2)]),
        ];
        let out = combine(&LinearCombiner::new(), &lists, 10);
        assert_combined(&out, &[("b", 0.6), ("a", 0.3), ("c", 0.1)]);

        let out = combine(&LinearCombiner::with_weights([3.0, 1.0]), &lists, 2);
        assert_combined(&out, &[("b", 0.5), ("a", 0.45)]);
    }

    #[test]
    fn records_sources() {
        let lists = [
            candidates(&[("a", 1.0)]),
            candidates(&[("a", 1.0), ("b", 0.5)]),
        ];
        let outputs: Vec<PredictorOutput<'_>> = lists
            .iter()
            .enumerate()
            .map(|(index, candidates)| PredictorOutput { index, candidates })
            .collect();
        let out = LinearCombiner::new().combine(&outputs, 10);
        assert!(out[0].sources.contains(0) && out[0].sources.contains(1));
        assert!(!out[1].sources.contains(0) && out[1].sources.contains(1));
    }

    #[test]
    fn ignores_invalid_weights() {
        let lists = [candidates(&[("a", 1.0)]), candidates(&[("b", 1.0)])];
        let out = combine(&LinearCombiner::with_weights([f32::NAN, -1.0]), &lists, 10);
        assert!(out.is_empty());
        let out = combine(&LinearCombiner::with_weights([0.0]), &lists, 10);
        assert_combined(&out, &[("b", 1.0)]);
    }

    #[test]
    fn renormalizes_over_predictors_with_candidates() {
        let lists = [candidates(&[("a", 1.0)]), Vec::new(), candidates(&[])];
        let out = combine(&LinearCombiner::with_weights([0.6, 0.3, 0.1]), &lists, 10);
        assert_combined(&out, &[("a", 1.0)]);
    }
}
