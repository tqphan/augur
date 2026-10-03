//! User model store and learning for augur.
//!
//! [`UserPredictor`] learns word trigrams and capitalization from the text
//! the user commits, predicts with interpolated absolute discounting and
//! optionally persists what it learned in a [redb] database.
//!
//! [redb]: https://www.redb.org

mod counts;
mod store;

use std::borrow::Cow;
use std::path::Path;
use std::sync::{Mutex, PoisonError, RwLock};

use augur_core::{Candidate, LearnInput, Predictor, Result, TokenizedContext};

use crate::counts::Counts;
use crate::store::Store;

/// Limits of a [`UserPredictor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserConfig {
    /// The most words and n-grams kept. When learning exceeds it, all
    /// counts are halved (see [`UserConfig::decay_interval`]) until it
    /// fits.
    pub max_entries: usize,
    /// Every this many learned words, all counts are halved and those
    /// that fall to zero are dropped, so that old habits fade. 0 never
    /// halves on this schedule.
    pub decay_interval: u64,
    /// With a store, changes are saved after this many learned words (and
    /// on [`Predictor::flush`] and drop). 0 saves only then.
    pub autosave_interval: u64,
}

impl Default for UserConfig {
    fn default() -> Self {
        Self {
            max_entries: 300_000,
            decay_interval: 200_000,
            autosave_interval: 256,
        }
    }
}

/// Predicts from the user's own text.
///
/// Counts words, bigrams and trigrams of the committed text (within
/// sentences) and ranks words by interpolated absolute discounting; the
/// reported probabilities are conditioned on the typed prefix. Unlike a
/// static model it suggests any word the user has typed. Markers such as
/// `<num>` are counted as history but never suggested.
///
/// It also learns how the user capitalizes words in mid-sentence and
/// reports that as the [display form](Predictor::display_form) once the
/// capitalized uses outnumber the lowercase ones; words in a run of capitals
/// (caps lock) do not count.
///
/// The counts are kept in memory; [`UserPredictor::open`] loads them from
/// and saves them to a store file.
#[derive(Debug)]
pub struct UserPredictor {
    config: UserConfig,
    counts: RwLock<Counts>,
    /// Held while saving, so that saves are applied in order.
    store: Option<Mutex<Store>>,
}

impl UserPredictor {
    /// A predictor that learns in memory only.
    pub fn in_memory(config: UserConfig) -> Self {
        Self {
            config,
            counts: RwLock::new(Counts::new(false)),
            store: None,
        }
    }

    /// Opens the store at `path`, creating it if needed, and loads what was
    /// learned. A store belongs to one language (e.g. `"en"`) and can be
    /// open in one predictor at a time.
    ///
    /// # Errors
    ///
    /// [`Error::Io`](augur_core::Error::Io) if the file cannot be accessed;
    /// [`Error::UserStore`](augur_core::Error::UserStore) if it is not a
    /// valid store, is for another language or is already open.
    pub fn open(path: impl AsRef<Path>, language: &str, config: UserConfig) -> Result<Self> {
        let (store, counts) = Store::open(path.as_ref(), language)?;
        Ok(Self {
            config,
            counts: RwLock::new(counts),
            store: Some(Mutex::new(store)),
        })
    }

    /// The configuration.
    pub fn config(&self) -> &UserConfig {
        &self.config
    }

    /// Number of words and n-grams currently stored.
    pub fn entries(&self) -> usize {
        self.read().entries()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, Counts> {
        self.counts.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Counts> {
        self.counts.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Predictor for UserPredictor {
    fn name(&self) -> &'static str {
        "user"
    }

    fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>) {
        self.read().predict(context, max, out);
    }

    fn learn(&self, input: &LearnInput<'_>) {
        let save = {
            let mut counts = self.write();
            let learned = counts.learn(input);
            counts.since_decay += learned;
            counts.since_save += learned;
            let interval = self.config.decay_interval;
            if interval > 0 && counts.since_decay >= interval {
                counts.halve();
                counts.since_decay = 0;
            }
            while counts.entries() > self.config.max_entries {
                counts.halve();
            }
            let autosave = self.config.autosave_interval;
            self.store.is_some() && autosave > 0 && counts.since_save >= autosave
        };
        if save {
            // A failed autosave keeps the changes for the next attempt;
            // `flush` reports errors.
            let _ = self.flush();
        }
    }

    fn forget(&self, word: &str) -> Result<()> {
        if self.write().forget(word) {
            self.flush()
        } else {
            Ok(())
        }
    }

    fn clear(&self) -> Result<()> {
        self.write().clear();
        self.flush()
    }

    fn flush(&self) -> Result<()> {
        let Some(store) = &self.store else {
            return Ok(());
        };
        let store = store.lock().unwrap_or_else(PoisonError::into_inner);
        let (dirty, changes) = {
            let mut counts = self.write();
            counts.since_save = 0;
            counts.take_changes()
        };
        if changes.is_empty() {
            return Ok(());
        }
        store
            .save(&changes)
            .inspect_err(|_| self.write().restore(dirty))
    }

    fn display_form(&self, word: &str) -> Option<Cow<'_, str>> {
        self.read().display_form(word).map(Cow::Owned)
    }
}

impl Drop for UserPredictor {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests;
