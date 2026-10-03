//! Prediction of recently used words.

use std::collections::{HashMap, VecDeque};
use std::sync::{PoisonError, RwLock};

use augur_core::{Candidate, LearnInput, Predictor, Result, Token, TokenizedContext, is_marker};

/// Configuration of a [`RecencyPredictor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecencyConfig {
    /// How many of the most recently learned words are remembered.
    pub capacity: usize,
    /// After how many newer words an occurrence counts half as much.
    pub half_life: f32,
}

impl Default for RecencyConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            half_life: 100.0,
        }
    }
}

/// Suggests words the user typed recently, regardless of context.
///
/// Each remembered occurrence of a word scores `0.5^(age / half_life)`,
/// where `age` is the number of words learned after it; a word's
/// probability is its total score divided by the total of all words that
/// match the typed prefix. Nothing is persisted.
#[derive(Debug)]
pub struct RecencyPredictor {
    config: RecencyConfig,
    /// Learned words, oldest first.
    words: RwLock<VecDeque<Box<str>>>,
}

impl RecencyPredictor {
    /// Creates an empty predictor.
    pub fn new(config: RecencyConfig) -> Self {
        Self {
            config,
            words: RwLock::new(VecDeque::with_capacity(config.capacity)),
        }
    }

    /// The configuration.
    pub fn config(&self) -> &RecencyConfig {
        &self.config
    }
}

impl Default for RecencyPredictor {
    fn default() -> Self {
        Self::new(RecencyConfig::default())
    }
}

impl Predictor for RecencyPredictor {
    fn name(&self) -> &'static str {
        "recency"
    }

    fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>) {
        if max == 0 {
            return;
        }
        let words = self.words.read().unwrap_or_else(PoisonError::into_inner);
        let decay = if self.config.half_life > 0.0 {
            0.5f64.powf(1.0 / f64::from(self.config.half_life))
        } else {
            0.0
        };
        let next = context.next_word.as_deref();

        // Newest first; `order` keeps the most recent word first among ties.
        let mut scores: HashMap<&str, (f64, usize)> = HashMap::new();
        let mut weight = 1.0;
        let mut total = 0.0;
        for word in words.iter().rev() {
            if word.starts_with(context.prefix.as_str()) && Some(&**word) != next {
                let order = scores.len();
                scores.entry(word).or_insert((0.0, order)).0 += weight;
                total += weight;
            }
            weight *= decay;
        }
        if total <= 0.0 {
            return;
        }
        let mut ranked: Vec<(&str, (f64, usize))> = scores.into_iter().collect();
        ranked.sort_unstable_by(|a, b| b.1.0.total_cmp(&a.1.0).then(a.1.1.cmp(&b.1.1)));
        out.extend(
            ranked
                .into_iter()
                .take(max)
                .map(|(word, (score, _))| Candidate {
                    word: word.to_owned(),
                    #[allow(clippy::cast_possible_truncation)]
                    prob: (score / total) as f32,
                }),
        );
    }

    fn learn(&self, input: &LearnInput<'_>) {
        if self.config.capacity == 0 {
            return;
        }
        let mut words = self.words.write().unwrap_or_else(PoisonError::into_inner);
        for span in input.tokens {
            if let Token::Word(word) = &span.token
                && !is_marker(word)
            {
                if words.len() == self.config.capacity {
                    words.pop_front();
                }
                words.push_back(word.as_str().into());
            }
        }
    }

    fn forget(&self, word: &str) -> Result<()> {
        self.words
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|w| &**w != word);
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        self.words
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use augur_core::TokenSpan;

    use super::*;

    fn learn(predictor: &RecencyPredictor, text: &str) {
        let tokens: Vec<TokenSpan> = text
            .split_whitespace()
            .map(|w| {
                let start = w.as_ptr() as usize - text.as_ptr() as usize;
                TokenSpan {
                    token: Token::Word(w.to_owned()),
                    span: start..start + w.len(),
                }
            })
            .collect();
        predictor.learn(&LearnInput {
            context: &[],
            tokens: &tokens,
            text,
        });
    }

    fn predict(predictor: &RecencyPredictor, prefix: &str, next: Option<&str>) -> Vec<Candidate> {
        let context = TokenizedContext {
            history: Vec::new(),
            prefix: prefix.to_owned(),
            next_word: next.map(str::to_owned),
        };
        let mut out = Vec::new();
        predictor.predict(&context, 10, &mut out);
        out
    }

    fn words(out: &[Candidate]) -> Vec<&str> {
        out.iter().map(|c| c.word.as_str()).collect()
    }

    #[test]
    fn ranks_by_decayed_frequency() {
        let predictor = RecencyPredictor::new(RecencyConfig {
            capacity: 10,
            half_life: 1.0,
        });
        learn(&predictor, "apple banana apple cherry <num>");
        let out = predict(&predictor, "", None);
        // The marker is not remembered. Ages: cherry 0, apple 1 and 3,
        // banana 2; scores 1, 0.5 + 0.125, 0.25.
        assert_eq!(words(&out), ["cherry", "apple", "banana"]);
        let total: f32 = out.iter().map(|c| c.prob).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!((out[0].prob - 1.0 / 1.875).abs() < 1e-6);
        assert!((out[1].prob - 0.625 / 1.875).abs() < 1e-6);

        assert_eq!(words(&predict(&predictor, "a", None)), ["apple"]);
        assert_eq!(
            words(&predict(&predictor, "", Some("cherry"))),
            ["apple", "banana"]
        );
    }

    #[test]
    fn keeps_only_capacity_words() {
        let predictor = RecencyPredictor::new(RecencyConfig {
            capacity: 2,
            half_life: 100.0,
        });
        learn(&predictor, "one two three");
        assert_eq!(words(&predict(&predictor, "", None)), ["three", "two"]);
    }

    #[test]
    fn forgets_and_clears() {
        let predictor = RecencyPredictor::default();
        learn(&predictor, "secret word secret");
        predictor.forget("secret").unwrap();
        assert_eq!(words(&predict(&predictor, "", None)), ["word"]);
        predictor.clear().unwrap();
        assert!(predict(&predictor, "", None).is_empty());
    }
}
