//! In-memory n-gram counts of the user's text.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::mem;
use std::ops::Bound;

use augur_core::{Candidate, LearnInput, Token, TokenizedContext, is_marker};

pub(crate) type WordId = u32;

/// The sentence-start marker, counted like a word so that it can be
/// history.
pub(crate) const BOS: &str = "<s>";

/// The absolute discount subtracted from every stored count.
const DISCOUNT: f64 = 0.75;

/// A word and what is known about it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WordRecord {
    pub(crate) count: u32,
    /// The most recent capitalized form the user typed in mid-sentence.
    pub(crate) form: Option<Box<str>>,
    /// Mid-sentence uses with a capital minus uses in lowercase; `form` is
    /// the display form while this is positive.
    pub(crate) votes: i32,
}

#[derive(Debug, Default)]
struct Word {
    /// Empty once the word is forgotten (ids are not reused until the next
    /// compaction).
    text: Box<str>,
    record: WordRecord,
}

/// Statistics of a history (one or two words) over its continuations.
#[derive(Debug, Default)]
struct History {
    total: u32,
    followers: Vec<WordId>,
}

/// What changed since the last save, by id.
#[derive(Debug, Default)]
pub(crate) struct Dirty {
    /// Everything must be rewritten (after a halving or clear).
    rewrite: bool,
    forgotten: Vec<Box<str>>,
    words: HashSet<WordId>,
    bigrams: HashSet<[WordId; 2]>,
    trigrams: HashSet<[WordId; 3]>,
}

/// Changes to persist, by word text.
#[derive(Debug, Default)]
pub(crate) struct Changes {
    /// Remove all stored data first.
    pub(crate) rewrite: bool,
    /// Remove these words and every n-gram that contains them.
    pub(crate) forgotten: Vec<Box<str>>,
    /// `None` removes the word.
    pub(crate) words: Vec<(Box<str>, Option<WordRecord>)>,
    /// A count of 0 removes the n-gram.
    pub(crate) bigrams: Vec<([Box<str>; 2], u32)>,
    pub(crate) trigrams: Vec<([Box<str>; 3], u32)>,
    pub(crate) since_decay: u64,
}

impl Changes {
    pub(crate) fn is_empty(&self) -> bool {
        !self.rewrite
            && self.forgotten.is_empty()
            && self.words.is_empty()
            && self.bigrams.is_empty()
            && self.trigrams.is_empty()
    }
}

/// Interpolated absolute-discounting trigram model over the user's text.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    words: Vec<Word>,
    ids: BTreeMap<Box<str>, WordId>,
    bigrams: HashMap<[WordId; 2], u32>,
    trigrams: HashMap<[WordId; 3], u32>,
    histories1: HashMap<WordId, History>,
    histories2: HashMap<[WordId; 2], History>,
    /// Total count of all words but `<s>`.
    total: u64,
    /// Whether changes are recorded for saving.
    track: bool,
    dirty: Dirty,
    /// Words learned since the last halving.
    pub(crate) since_decay: u64,
    /// Words learned since the last save.
    pub(crate) since_save: u64,
}

impl Counts {
    /// Empty counts; `track` records changes for saving.
    pub(crate) fn new(track: bool) -> Self {
        Self {
            track,
            ..Self::default()
        }
    }

    /// Counts loaded from storage. N-grams with unknown words are dropped.
    pub(crate) fn from_records(
        words: impl IntoIterator<Item = (String, WordRecord)>,
        bigrams: impl IntoIterator<Item = ([String; 2], u32)>,
        trigrams: impl IntoIterator<Item = ([String; 3], u32)>,
        since_decay: u64,
    ) -> Self {
        let mut counts = Self::new(true);
        for (text, record) in words {
            if record.count > 0 && !text.is_empty() {
                let id = counts.intern(&text);
                counts.words[id as usize].record = record;
            }
        }
        for (key, count) in bigrams {
            if let [Some(a), Some(b)] = key.each_ref().map(|w| counts.id(w))
                && count > 0
            {
                counts.bigrams.insert([a, b], count);
            }
        }
        for (key, count) in trigrams {
            if let [Some(a), Some(b), Some(c)] = key.each_ref().map(|w| counts.id(w))
                && count > 0
            {
                counts.trigrams.insert([a, b, c], count);
            }
        }
        counts.since_decay = since_decay;
        counts.rebuild_stats();
        counts
    }

    /// Number of stored words and n-grams.
    pub(crate) fn entries(&self) -> usize {
        self.ids.len() + self.bigrams.len() + self.trigrams.len()
    }

    fn id(&self, word: &str) -> Option<WordId> {
        self.ids.get(word).copied()
    }

    fn text(&self, id: WordId) -> &str {
        &self.words[id as usize].text
    }

    fn count(&self, id: WordId) -> u32 {
        self.words[id as usize].record.count
    }

    fn intern(&mut self, word: &str) -> WordId {
        if let Some(id) = self.id(word) {
            return id;
        }
        let id = WordId::try_from(self.words.len()).expect("fewer than 2^32 words");
        self.words.push(Word {
            text: word.into(),
            record: WordRecord::default(),
        });
        self.ids.insert(word.into(), id);
        id
    }

    /// Counts the new tokens of `input`; returns how many words it held.
    pub(crate) fn learn(&mut self, input: &LearnInput<'_>) -> u64 {
        // History: the ids of up to two preceding tokens in the sentence.
        let mut history: Vec<WordId> = Vec::with_capacity(3);
        for token in input.context {
            let id = match token {
                Token::SentenceStart => {
                    history.clear();
                    self.id(BOS)
                }
                Token::Word(word) => self.id(word),
            };
            match id {
                Some(id) => push_history(&mut history, id),
                None => history.clear(),
            }
        }

        let mut sentence_start = matches!(input.context.last(), Some(Token::SentenceStart));
        let mut learned = 0;
        for (i, span) in input.tokens.iter().enumerate() {
            let id = match &span.token {
                Token::SentenceStart => {
                    history.clear();
                    self.intern(BOS)
                }
                Token::Word(word) => {
                    learned += 1;
                    let id = self.intern(word);
                    if !sentence_start && !is_marker(word) && !is_shouted(input, i) {
                        self.vote_casing(id, input.surface(i));
                    }
                    id
                }
            };
            let record = &mut self.words[id as usize].record;
            record.count = record.count.saturating_add(1);
            if span.token != Token::SentenceStart {
                self.total += 1;
            }
            self.mark_word(id);
            if let &[.., last] = history.as_slice() {
                self.add_bigram([last, id]);
            }
            if let &[first, second] = history.as_slice() {
                self.add_trigram([first, second, id]);
            }
            push_history(&mut history, id);
            sentence_start = span.token == Token::SentenceStart;
        }
        learned
    }

    fn vote_casing(&mut self, id: WordId, surface: &str) {
        let record = &mut self.words[id as usize].record;
        if surface.chars().any(char::is_uppercase) {
            if record.form.as_deref() != Some(surface) {
                record.form = Some(surface.into());
            }
            record.votes = record.votes.saturating_add(1);
        } else {
            record.votes = record.votes.saturating_sub(1);
        }
    }

    fn add_bigram(&mut self, key: [WordId; 2]) {
        let count = self.bigrams.entry(key).or_insert(0);
        if *count == 0 {
            self.histories1
                .entry(key[0])
                .or_default()
                .followers
                .push(key[1]);
        }
        *count = count.saturating_add(1);
        let history = self.histories1.entry(key[0]).or_default();
        history.total = history.total.saturating_add(1);
        if self.track {
            self.dirty.bigrams.insert(key);
        }
    }

    fn add_trigram(&mut self, key: [WordId; 3]) {
        let count = self.trigrams.entry(key).or_insert(0);
        if *count == 0 {
            let history = [key[0], key[1]];
            self.histories2
                .entry(history)
                .or_default()
                .followers
                .push(key[2]);
        }
        *count = count.saturating_add(1);
        let history = self.histories2.entry([key[0], key[1]]).or_default();
        history.total = history.total.saturating_add(1);
        if self.track {
            self.dirty.trigrams.insert(key);
        }
    }

    fn mark_word(&mut self, id: WordId) {
        if self.track {
            self.dirty.words.insert(id);
        }
    }

    /// Halves every count and drops what falls to zero.
    pub(crate) fn halve(&mut self) {
        let mut remap: Vec<Option<WordId>> = vec![None; self.words.len()];
        let mut words = Vec::with_capacity(self.ids.len());
        for (old, word) in mem::take(&mut self.words).into_iter().enumerate() {
            let count = word.record.count / 2;
            if count == 0 || word.text.is_empty() {
                continue;
            }
            remap[old] = Some(WordId::try_from(words.len()).expect("fewer than 2^32 words"));
            words.push(Word {
                text: word.text,
                record: WordRecord {
                    count,
                    form: word.record.form,
                    votes: word.record.votes / 2,
                },
            });
        }
        let map = |id: WordId| remap[id as usize];
        self.bigrams = mem::take(&mut self.bigrams)
            .into_iter()
            .filter_map(|([a, b], count)| Some(([map(a)?, map(b)?], count / 2)))
            .filter(|&(_, count)| count > 0)
            .collect();
        self.trigrams = mem::take(&mut self.trigrams)
            .into_iter()
            .filter_map(|([a, b, c], count)| Some(([map(a)?, map(b)?, map(c)?], count / 2)))
            .filter(|&(_, count)| count > 0)
            .collect();
        self.ids = words
            .iter()
            .enumerate()
            .map(|(id, word)| {
                (
                    word.text.clone(),
                    WordId::try_from(id).expect("checked above"),
                )
            })
            .collect();
        self.words = words;
        self.rebuild_stats();
        self.mark_rewrite();
    }

    /// Forgets a word and every n-gram containing it. Returns whether the
    /// word was known.
    pub(crate) fn forget(&mut self, word: &str) -> bool {
        if word == BOS {
            return false;
        }
        let Some(id) = self.ids.remove(word) else {
            return false;
        };
        self.words[id as usize] = Word::default();
        self.bigrams.retain(|key, _| !key.contains(&id));
        self.trigrams.retain(|key, _| !key.contains(&id));
        self.rebuild_stats();
        if self.track {
            self.dirty.words.remove(&id);
            self.dirty.bigrams.retain(|key| !key.contains(&id));
            self.dirty.trigrams.retain(|key| !key.contains(&id));
            self.dirty.forgotten.push(word.into());
        }
        true
    }

    /// Forgets everything.
    pub(crate) fn clear(&mut self) {
        *self = Self::new(self.track);
        self.mark_rewrite();
    }

    fn mark_rewrite(&mut self) {
        if self.track {
            self.dirty = Dirty {
                rewrite: true,
                ..Dirty::default()
            };
        }
    }

    /// Recomputes the history statistics and total from the counts.
    fn rebuild_stats(&mut self) {
        self.histories1.clear();
        for (&[a, b], &count) in &self.bigrams {
            let history = self.histories1.entry(a).or_default();
            history.total = history.total.saturating_add(count);
            history.followers.push(b);
        }
        self.histories2.clear();
        for (&[a, b, c], &count) in &self.trigrams {
            let history = self.histories2.entry([a, b]).or_default();
            history.total = history.total.saturating_add(count);
            history.followers.push(c);
        }
        self.total = self
            .words
            .iter()
            .filter(|w| !w.text.is_empty() && &*w.text != BOS)
            .map(|w| u64::from(w.record.count))
            .sum();
    }

    /// Takes the changes to save. Hand the returned [`Dirty`] back to
    /// [`Counts::restore`] if saving fails.
    pub(crate) fn take_changes(&mut self) -> (Dirty, Changes) {
        let dirty = mem::take(&mut self.dirty);
        let text = |id: WordId| self.words[id as usize].text.clone();
        let mut changes = Changes {
            since_decay: self.since_decay,
            ..Changes::default()
        };
        if dirty.rewrite {
            changes.rewrite = true;
            changes.words = self
                .words
                .iter()
                .filter(|w| !w.text.is_empty())
                .map(|w| (w.text.clone(), Some(w.record.clone())))
                .collect();
            changes.bigrams = self
                .bigrams
                .iter()
                .map(|(key, &count)| (key.map(text), count))
                .collect();
            changes.trigrams = self
                .trigrams
                .iter()
                .map(|(key, &count)| (key.map(text), count))
                .collect();
        } else {
            changes.forgotten.clone_from(&dirty.forgotten);
            changes.words = dirty
                .words
                .iter()
                .map(|&id| &self.words[id as usize])
                .filter(|w| !w.text.is_empty())
                .map(|w| {
                    let record = (w.record.count > 0).then(|| w.record.clone());
                    (w.text.clone(), record)
                })
                .collect();
            changes.bigrams = dirty
                .bigrams
                .iter()
                .map(|key| (key.map(text), self.bigrams.get(key).copied().unwrap_or(0)))
                .collect();
            changes.trigrams = dirty
                .trigrams
                .iter()
                .map(|key| (key.map(text), self.trigrams.get(key).copied().unwrap_or(0)))
                .collect();
        }
        (dirty, changes)
    }

    /// Puts back changes whose saving failed.
    pub(crate) fn restore(&mut self, dirty: Dirty) {
        if dirty.rewrite {
            self.mark_rewrite();
        } else if !self.dirty.rewrite {
            // Words forgotten since then are already gone from the counts;
            // saving their ids writes nothing for them.
            let live = |id: &WordId| !self.words[*id as usize].text.is_empty();
            let mut forgotten = dirty.forgotten;
            forgotten.append(&mut self.dirty.forgotten);
            self.dirty.forgotten = forgotten;
            let words: Vec<WordId> = dirty.words.into_iter().filter(live).collect();
            let bigrams: Vec<[WordId; 2]> = dirty
                .bigrams
                .into_iter()
                .filter(|key| key.iter().all(live))
                .collect();
            let trigrams: Vec<[WordId; 3]> = dirty
                .trigrams
                .into_iter()
                .filter(|key| key.iter().all(live))
                .collect();
            self.dirty.words.extend(words);
            self.dirty.bigrams.extend(bigrams);
            self.dirty.trigrams.extend(trigrams);
        }
    }

    /// The user's display form of a word, if they mostly capitalize it.
    pub(crate) fn display_form(&self, word: &str) -> Option<String> {
        let record = &self.words[self.id(word)? as usize].record;
        if record.votes > 0 {
            record.form.as_deref().map(str::to_owned)
        } else {
            None
        }
    }

    /// The ids of up to the last two tokens of the history within the
    /// current sentence; stops at unknown words.
    fn history_ids(&self, history: &[Token]) -> Vec<WordId> {
        let mut ids = Vec::with_capacity(2);
        for token in history.iter().rev() {
            if ids.len() == 2 {
                break;
            }
            match token {
                Token::SentenceStart => {
                    ids.extend(self.id(BOS));
                    break;
                }
                Token::Word(word) => match self.id(word) {
                    Some(id) => ids.push(id),
                    None => break,
                },
            }
        }
        ids.reverse();
        ids
    }

    /// Appends up to `max` candidates, ranked by `P(word | history)` and
    /// with probabilities conditioned on the typed prefix.
    pub(crate) fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>) {
        if max == 0 || self.total == 0 {
            return;
        }
        let prefix = context.prefix.as_str();
        let next = context.next_word.as_deref().and_then(|w| self.id(w));
        let skip = |id: WordId| Some(id) == next || is_marker(self.text(id));
        let candidate = |id: WordId| self.text(id).starts_with(prefix) && !skip(id);

        // Present histories, shortest first.
        let ids = self.history_ids(&context.history);
        let mut levels: Vec<Level<'_>> = Vec::with_capacity(2);
        if let Some(history) = ids.last().and_then(|v| self.histories1.get(v)) {
            levels.push(Level {
                key: [ids[ids.len() - 1], 0],
                len: 1,
                history,
            });
            if let &[u, v] = ids.as_slice()
                && let Some(history) = self.histories2.get(&[u, v])
            {
                levels.push(Level {
                    key: [u, v],
                    len: 2,
                    history,
                });
            }
        }
        let ngram_count = |level: &Level<'_>, id: WordId| -> u32 {
            let [u, v] = level.key;
            if level.len == 1 {
                self.bigrams.get(&[u, id]).copied().unwrap_or(0)
            } else {
                self.trigrams.get(&[u, v, id]).copied().unwrap_or(0)
            }
        };
        #[allow(clippy::cast_precision_loss)]
        let total = self.total as f64;
        let prob = |id: WordId| -> f64 {
            let mut p = f64::from(self.count(id)) / total;
            for level in &levels {
                let c = f64::from(ngram_count(level, id));
                p = ((c - DISCOUNT).max(0.0) + DISCOUNT * level.distinct() * p) / level.total();
            }
            p
        };

        let followers: HashSet<WordId> = levels
            .iter()
            .flat_map(|l| l.history.followers.iter().copied())
            .filter(|&id| candidate(id))
            .collect();

        // Unigram mass of the prefix range, and the most frequent words in
        // it that follow no history (their order is that of their unigram
        // counts).
        let mut mass = 0.0;
        let mut others: Vec<(WordId, u32)> = Vec::new();
        let range = (Bound::Included(prefix), Bound::Unbounded);
        for (text, &id) in self.ids.range::<str, _>(range) {
            if !text.starts_with(prefix) {
                break;
            }
            if skip(id) {
                continue;
            }
            mass += f64::from(self.count(id));
            if !followers.contains(&id) {
                others.push((id, self.count(id)));
            }
        }
        if mass == 0.0 {
            return;
        }
        let by_count = |a: &(WordId, u32), b: &(WordId, u32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        if others.len() > max {
            others.select_nth_unstable_by(max, by_count);
            others.truncate(max);
        }

        // Probability mass of the prefix range under the full model.
        let mut z = mass / total;
        for level in &levels {
            let discounted: f64 = level
                .history
                .followers
                .iter()
                .filter(|&&id| candidate(id))
                .map(|&id| (f64::from(ngram_count(level, id)) - DISCOUNT).max(0.0))
                .sum();
            z = (discounted + DISCOUNT * level.distinct() * z) / level.total();
        }

        let mut scored: Vec<(WordId, f64)> = followers
            .into_iter()
            .chain(others.into_iter().map(|(id, _)| id))
            .map(|id| (id, prob(id)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(max);
        out.extend(scored.into_iter().map(|(id, p)| Candidate {
            word: self.text(id).to_owned(),
            #[allow(clippy::cast_possible_truncation)]
            prob: (p / z).min(1.0) as f32,
        }));
    }
}

/// A present history of one or two words.
struct Level<'a> {
    key: [WordId; 2],
    len: usize,
    history: &'a History,
}

impl Level<'_> {
    fn total(&self) -> f64 {
        f64::from(self.history.total)
    }

    #[allow(clippy::cast_precision_loss)]
    fn distinct(&self) -> f64 {
        self.history.followers.len() as f64
    }
}

fn push_history(history: &mut Vec<WordId>, id: WordId) {
    history.push(id);
    if history.len() > 2 {
        history.remove(0);
    }
}

/// Whether token `index` is written in capitals next to another word in
/// capitals, i.e. probably typed with caps lock rather than an acronym.
fn is_shouted(input: &LearnInput<'_>, index: usize) -> bool {
    let caps = |i: usize| {
        input
            .tokens
            .get(i)
            .is_some_and(|t| matches!(t.token, Token::Word(_)) && all_caps(input.surface(i)))
    };
    caps(index) && ((index > 0 && caps(index - 1)) || caps(index + 1))
}

fn all_caps(text: &str) -> bool {
    text.chars().filter(|c| c.is_alphabetic()).count() > 1 && !text.chars().any(char::is_lowercase)
}

#[cfg(test)]
pub(crate) mod tests;
