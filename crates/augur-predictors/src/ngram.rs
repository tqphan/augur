//! Prediction from a static back-off n-gram model.

use std::borrow::Cow;
use std::ops::Range;
use std::path::Path;

use augur_core::{Candidate, Predictor, Token, TokenizedContext};
use augur_lm::{Model, NgramEntry, WordId};

/// Prefix ranges up to this many words are scanned in full for unigram
/// candidates; larger ones walk the unigrams in order of probability.
const SCAN_LIMIT: usize = 4096;

/// Predicts the next word, or completes the current one, from a back-off
/// n-gram model.
///
/// Candidates are ranked exactly by the model's back-off probability
/// `P(word | history)`. Their reported probability is conditioned on the
/// typed prefix: `P(word | history, word starts with prefix)`, i.e.
/// renormalized over the words that match the prefix, so that it is
/// comparable with other predictors'.
///
/// Markers such as `<s>`, `</s>`, `<unk>` and `<num>` (any vocabulary
/// entry written `<…>`) are never suggested, nor is the word right after
/// the cursor. The model's casing table provides
/// [`Predictor::display_form`].
pub struct NgramPredictor {
    model: Model,
    /// Marker word ids, ascending.
    markers: Vec<WordId>,
    /// Non-marker word ids, most probable unigram first.
    by_prob: Vec<WordId>,
    /// `cumulative[i]` is the total unigram probability of the non-marker
    /// words with ids below `i`.
    cumulative: Vec<f64>,
}

impl NgramPredictor {
    /// Wraps a loaded model.
    pub fn new(model: Model) -> Self {
        let markers: Vec<WordId> = model
            .prefix_range("<")
            .filter(|&id| model.word(id).ends_with('>'))
            .collect();
        let is_marker = |id: WordId| markers.binary_search(&id).is_ok();

        let unigrams: Vec<(WordId, f32)> = model
            .continuations(&[], 0..to_id(model.vocab_len()))
            .collect();
        let mut cumulative = Vec::with_capacity(unigrams.len() + 1);
        let mut total = 0.0;
        cumulative.push(total);
        for &(id, log_prob) in &unigrams {
            if !is_marker(id) {
                total += exp10(log_prob);
            }
            cumulative.push(total);
        }
        let mut by_prob: Vec<(WordId, f32)> = unigrams
            .into_iter()
            .filter(|&(id, _)| !is_marker(id))
            .collect();
        by_prob.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        Self {
            by_prob: by_prob.into_iter().map(|(id, _)| id).collect(),
            model,
            markers,
            cumulative,
        }
    }

    /// Loads a model file and wraps it.
    ///
    /// # Errors
    ///
    /// If the model cannot be read or is invalid.
    pub fn open(path: impl AsRef<Path>) -> augur_lm::Result<Self> {
        Ok(Self::new(Model::open(path)?))
    }

    /// The underlying model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    fn is_marker(&self, id: WordId) -> bool {
        self.markers.binary_search(&id).is_ok()
    }

    /// The model ids of the last `order - 1` tokens of the history, within
    /// the current sentence (starting with `<s>` if the sentence start is
    /// reached). Unknown words map to `<unk>`; without `<unk>` in the model
    /// the context stops at them.
    fn context_ids(&self, history: &[Token]) -> Vec<WordId> {
        let mut ids = Vec::with_capacity(self.model.order() - 1);
        for token in history.iter().rev() {
            if ids.len() + 1 >= self.model.order() {
                break;
            }
            match token {
                Token::SentenceStart => {
                    ids.extend(self.model.bos());
                    break;
                }
                Token::Word(word) => match self.model.id(word).or(self.model.unk()) {
                    Some(id) => ids.push(id),
                    None => break,
                },
            }
        }
        ids.reverse();
        ids
    }

    /// Log10 probability of `word` (in the prefix range) after the context
    /// whose present suffixes are `levels` (shortest first).
    fn log_prob_in(&self, levels: &[Level], word: WordId) -> f32 {
        let mut backoff = 0.0;
        for level in levels.iter().rev() {
            if let Some(log_prob) = level.get(word) {
                return backoff + log_prob;
            }
            backoff += level.backoff;
        }
        backoff + self.unigram(word)
    }

    fn unigram(&self, word: WordId) -> f32 {
        self.model
            .ngram(&[word])
            .map_or(augur_lm::LOG_ZERO, |e| e.log_prob)
    }

    /// Total probability of the non-marker words in `range` after the
    /// context, computed through the back-off recursion
    /// `Z(h) = Σ_{w∈C(h)} p(h w) + bo(h) · (Z(h') − Σ_{w∈C(h)} P(w | h'))`
    /// where `C(h)` are the stored continuations of `h` in the range and
    /// `h'` is the next shorter context present in the model.
    fn total(&self, levels: &[Level], range: &Range<WordId>) -> f64 {
        let mut total = self.cumulative[range.end as usize] - self.cumulative[range.start as usize];
        for (i, level) in levels.iter().enumerate() {
            let (mut stored, mut backed_off) = (0.0, 0.0);
            for &(id, log_prob) in &level.words {
                if !self.is_marker(id) {
                    stored += exp10(log_prob);
                    backed_off += exp10(self.log_prob_in(&levels[..i], id));
                }
            }
            total = stored + exp10(level.backoff) * (total - backed_off);
        }
        total
    }
}

/// A suffix of the context that is present in the model.
struct Level {
    /// Log10 back-off weight of the context.
    backoff: f32,
    /// Stored continuations in the prefix range, ascending by id.
    words: Vec<(WordId, f32)>,
}

impl Level {
    fn get(&self, word: WordId) -> Option<f32> {
        self.words
            .binary_search_by_key(&word, |&(id, _)| id)
            .ok()
            .map(|i| self.words[i].1)
    }
}

impl Predictor for NgramPredictor {
    fn name(&self) -> &'static str {
        "ngram"
    }

    fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>) {
        if max == 0 {
            return;
        }
        let model = &self.model;
        let range = model.prefix_range(&context.prefix);
        if range.is_empty() {
            return;
        }
        let ids = self.context_ids(&context.history);
        let next = context.next_word.as_deref().and_then(|w| model.id(w));
        let skip = |id: WordId| self.is_marker(id) || Some(id) == next;

        // Context suffixes present in the model, shortest first, with their
        // continuations in the prefix range.
        let levels: Vec<Level> = (1..=ids.len())
            .filter_map(|len| {
                let history = &ids[ids.len() - len..];
                let entry: NgramEntry = model.ngram(history)?;
                Some(Level {
                    backoff: entry.log_backoff,
                    words: model.continuations(history, range.clone()).collect(),
                })
            })
            .collect();
        let stored_longer =
            |i: usize, id: WordId| levels[i + 1..].iter().any(|l| l.get(id).is_some());

        // Every word's probability is its entry in the longest context that
        // stores it, plus the back-off weights of the longer contexts; words
        // stored in no context back off to their unigram.
        let mut scored: Vec<(WordId, f32)> = Vec::new();
        let mut backoff = 0.0;
        for (i, level) in levels.iter().enumerate().rev() {
            for &(id, log_prob) in &level.words {
                if !skip(id) && !stored_longer(i, id) {
                    scored.push((id, backoff + log_prob));
                }
            }
            backoff += level.backoff;
        }
        let in_any_level = |id: WordId| levels.iter().any(|l| l.get(id).is_some());
        if range.len() <= SCAN_LIMIT {
            for (id, log_prob) in model.continuations(&[], range.clone()) {
                if !skip(id) && !in_any_level(id) {
                    scored.push((id, backoff + log_prob));
                }
            }
        } else {
            // The best unigram-only words are the most probable unigrams.
            let mut found = 0;
            for &id in &self.by_prob {
                if found == max {
                    break;
                }
                if range.contains(&id) && !skip(id) && !in_any_level(id) {
                    scored.push((id, backoff + self.unigram(id)));
                    found += 1;
                }
            }
        }

        let order = |a: &(WordId, f32), b: &(WordId, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if scored.len() > max {
            scored.select_nth_unstable_by(max, order);
            scored.truncate(max);
        }
        scored.sort_unstable_by(order);

        let mut total = self.total(&levels, &range);
        if let Some(next) = next.filter(|&id| range.contains(&id) && !self.is_marker(id)) {
            total -= exp10(self.log_prob_in(&levels, next));
        }
        // Quantization can make the recursion slightly inconsistent.
        let shown: f64 = scored.iter().map(|&(_, log_prob)| exp10(log_prob)).sum();
        let total = total.max(shown);

        out.extend(scored.into_iter().map(|(id, log_prob)| Candidate {
            word: model.word(id).to_owned(),
            #[allow(clippy::cast_possible_truncation)]
            prob: (exp10(log_prob) / total) as f32,
        }));
    }

    fn display_form(&self, word: &str) -> Option<Cow<'_, str>> {
        let id = self.model.id(word)?;
        self.model.casing(id).map(Cow::Borrowed)
    }
}

impl std::fmt::Debug for NgramPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramPredictor")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

fn exp10(log_prob: f32) -> f64 {
    10f64.powf(f64::from(log_prob))
}

fn to_id(i: usize) -> WordId {
    WordId::try_from(i).expect("vocabulary fits u32")
}

#[cfg(test)]
mod tests;
