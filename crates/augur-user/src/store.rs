//! Persistence of user counts in a redb database.

use std::path::Path;

use augur_core::{Error, Result};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};

use crate::counts::{Changes, Counts, WordRecord};

/// Format version of the stored tables.
const SCHEMA: &str = "1";

/// `schema`, `language` and `since_decay`.
const META: TableDefinition<'_, &str, &str> = TableDefinition::new("meta");
/// Word → (count, casing votes, capitalized form or "").
const WORDS: TableDefinition<'_, &str, (u32, i32, &str)> = TableDefinition::new("words");
const BIGRAMS: TableDefinition<'_, (&str, &str), u32> = TableDefinition::new("bigrams");
const TRIGRAMS: TableDefinition<'_, (&str, &str, &str), u32> = TableDefinition::new("trigrams");

/// An open user store.
pub(crate) struct Store {
    db: Database,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").finish_non_exhaustive()
    }
}

impl Store {
    /// Opens or creates the store at `path` for `language` and loads its
    /// counts.
    pub(crate) fn open(path: &Path, language: &str) -> Result<(Self, Counts)> {
        let db = Database::create(path).map_err(store_error)?;
        let txn = db.begin_write().map_err(store_error)?;
        {
            let mut meta = txn.open_table(META).map_err(store_error)?;
            let schema = meta
                .get("schema")
                .map_err(store_error)?
                .map(|v| v.value().to_owned());
            match schema.as_deref() {
                None => {
                    meta.insert("schema", SCHEMA).map_err(store_error)?;
                    meta.insert("language", language).map_err(store_error)?;
                }
                Some(SCHEMA) => {
                    let stored = meta.get("language").map_err(store_error)?;
                    let stored = stored
                        .as_ref()
                        .map(redb::AccessGuard::value)
                        .unwrap_or_default();
                    if !stored.eq_ignore_ascii_case(language) {
                        return Err(Error::UserStore(format!(
                            "the store is for language {stored:?}, not {language:?}"
                        )));
                    }
                }
                Some(other) => {
                    return Err(Error::UserStore(format!(
                        "unsupported store version {other:?} (expected {SCHEMA:?})"
                    )));
                }
            }
            txn.open_table(WORDS).map_err(store_error)?;
            txn.open_table(BIGRAMS).map_err(store_error)?;
            txn.open_table(TRIGRAMS).map_err(store_error)?;
        }
        txn.commit().map_err(store_error)?;

        let txn = db.begin_read().map_err(store_error)?;
        let since_decay = txn
            .open_table(META)
            .map_err(store_error)?
            .get("since_decay")
            .map_err(store_error)?
            .and_then(|v| v.value().parse().ok())
            .unwrap_or(0);
        let mut words = Vec::new();
        for entry in txn
            .open_table(WORDS)
            .map_err(store_error)?
            .iter()
            .map_err(store_error)?
        {
            let (key, value) = entry.map_err(store_error)?;
            let (count, votes, form) = value.value();
            words.push((
                key.value().to_owned(),
                WordRecord {
                    count,
                    form: (!form.is_empty()).then(|| form.into()),
                    votes,
                },
            ));
        }
        let mut bigrams = Vec::new();
        for entry in txn
            .open_table(BIGRAMS)
            .map_err(store_error)?
            .iter()
            .map_err(store_error)?
        {
            let (key, value) = entry.map_err(store_error)?;
            let (a, b) = key.value();
            bigrams.push(([a.to_owned(), b.to_owned()], value.value()));
        }
        let mut trigrams = Vec::new();
        for entry in txn
            .open_table(TRIGRAMS)
            .map_err(store_error)?
            .iter()
            .map_err(store_error)?
        {
            let (key, value) = entry.map_err(store_error)?;
            let (a, b, c) = key.value();
            trigrams.push(([a.to_owned(), b.to_owned(), c.to_owned()], value.value()));
        }
        let counts = Counts::from_records(words, bigrams, trigrams, since_decay);
        Ok((Self { db }, counts))
    }

    /// Applies `changes` in one transaction.
    pub(crate) fn save(&self, changes: &Changes) -> Result<()> {
        let txn = self.db.begin_write().map_err(store_error)?;
        if changes.rewrite {
            txn.delete_table(WORDS).map_err(store_error)?;
            txn.delete_table(BIGRAMS).map_err(store_error)?;
            txn.delete_table(TRIGRAMS).map_err(store_error)?;
        }
        {
            let mut words = txn.open_table(WORDS).map_err(store_error)?;
            let mut bigrams = txn.open_table(BIGRAMS).map_err(store_error)?;
            let mut trigrams = txn.open_table(TRIGRAMS).map_err(store_error)?;
            for word in &changes.forgotten {
                let word = &**word;
                words.remove(word).map_err(store_error)?;
                bigrams
                    .retain(|(a, b), _| a != word && b != word)
                    .map_err(store_error)?;
                trigrams
                    .retain(|(a, b, c), _| a != word && b != word && c != word)
                    .map_err(store_error)?;
            }
            for (word, record) in &changes.words {
                match record {
                    Some(r) => {
                        let form = r.form.as_deref().unwrap_or_default();
                        words
                            .insert(&**word, (r.count, r.votes, form))
                            .map_err(store_error)?;
                    }
                    None => {
                        words.remove(&**word).map_err(store_error)?;
                    }
                }
            }
            for ([a, b], count) in &changes.bigrams {
                let key = (&**a, &**b);
                if *count == 0 {
                    bigrams.remove(key).map_err(store_error)?;
                } else {
                    bigrams.insert(key, count).map_err(store_error)?;
                }
            }
            for ([a, b, c], count) in &changes.trigrams {
                let key = (&**a, &**b, &**c);
                if *count == 0 {
                    trigrams.remove(key).map_err(store_error)?;
                } else {
                    trigrams.insert(key, count).map_err(store_error)?;
                }
            }
            let mut meta = txn.open_table(META).map_err(store_error)?;
            meta.insert("since_decay", changes.since_decay.to_string().as_str())
                .map_err(store_error)?;
        }
        txn.commit().map_err(store_error)
    }
}

fn store_error(error: impl Into<redb::Error>) -> Error {
    match error.into() {
        redb::Error::Io(e) => Error::Io(e),
        redb::Error::DatabaseAlreadyOpen => {
            Error::UserStore("the store is already open in another engine".into())
        }
        other => Error::UserStore(other.to_string()),
    }
}
