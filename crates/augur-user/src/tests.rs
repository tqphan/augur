use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use augur_core::{Error, Token};

use super::*;
use crate::counts::tests::spans;

/// A directory removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "augur-user-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn learn(predictor: &UserPredictor, text: &str) {
    predictor.learn(&LearnInput {
        context: &[],
        tokens: &spans(text),
        text,
    });
}

fn predict(predictor: &UserPredictor, history: &str, prefix: &str) -> Vec<String> {
    let context = TokenizedContext {
        history: spans(history).into_iter().map(|s| s.token).collect(),
        prefix: prefix.to_owned(),
        next_word: None,
    };
    let mut out = Vec::new();
    predictor.predict(&context, 10, &mut out);
    out.into_iter().map(|c| c.word).collect()
}

fn open(path: &Path, config: UserConfig) -> UserPredictor {
    UserPredictor::open(path, "en", config).unwrap()
}

#[test]
fn predicts_learned_words() {
    let predictor = UserPredictor::in_memory(UserConfig::default());
    assert!(predict(&predictor, "", "").is_empty());
    learn(&predictor, "| see you at Augurfest");
    assert_eq!(
        predict(&predictor, "| see you at", ""),
        ["augurfest", "see", "you", "at"]
    );
    assert_eq!(predict(&predictor, "| see", "y"), ["you"]);
    assert_eq!(
        predictor.display_form("augurfest").as_deref(),
        Some("Augurfest")
    );
    assert_eq!(predictor.name(), "user");
    predictor.flush().unwrap();
}

#[test]
fn persists_across_opens() {
    let dir = TempDir::new();
    let path = dir.file("user.redb");
    {
        let predictor = open(&path, UserConfig::default());
        learn(&predictor, "| ping Quoc today | ping Quoc now");
    }
    let predictor = open(&path, UserConfig::default());
    assert_eq!(predict(&predictor, "| ping", "")[0], "quoc");
    assert_eq!(predictor.display_form("quoc").as_deref(), Some("Quoc"));

    // Incremental saves.
    learn(&predictor, "| ping Quoc again");
    predictor.flush().unwrap();
    drop(predictor);
    let predictor = open(&path, UserConfig::default());
    let entries = predictor.entries();
    assert_eq!(predict(&predictor, "| ping quoc", "a"), ["again"]);

    // Halving rewrites everything.
    predictor.write().halve();
    predictor.flush().unwrap();
    let halved = predictor.entries();
    assert!(halved < entries);
    drop(predictor);
    assert_eq!(open(&path, UserConfig::default()).entries(), halved);
}

#[test]
fn autosaves() {
    let dir = TempDir::new();
    let path = dir.file("user.redb");
    let config = UserConfig {
        autosave_interval: 3,
        ..UserConfig::default()
    };
    let predictor = open(&path, config);
    learn(&predictor, "| one two three");
    // Nothing is left to save after the autosave; one more word is not
    // enough for the next.
    assert!(predictor.write().take_changes().1.is_empty());
    learn(&predictor, "| four");
    assert!(!predictor.write().take_changes().1.is_empty());
}

#[test]
fn forgets_and_clears_persistently() {
    let dir = TempDir::new();
    let path = dir.file("user.redb");
    {
        let predictor = open(&path, UserConfig::default());
        learn(&predictor, "| my password hunter2 | my password");
        predictor.flush().unwrap();
        predictor.forget("hunter2").unwrap();
        predictor.forget("unknown").unwrap();
    }
    {
        let predictor = open(&path, UserConfig::default());
        assert!(!predict(&predictor, "", "").contains(&"hunter2".to_owned()));
        assert!(predict(&predictor, "| my", "").contains(&"password".to_owned()));
        predictor.clear().unwrap();
        assert_eq!(predictor.entries(), 0);
    }
    assert_eq!(open(&path, UserConfig::default()).entries(), 0);
}

#[test]
fn decays_and_caps() {
    let predictor = UserPredictor::in_memory(UserConfig {
        decay_interval: 4,
        ..UserConfig::default()
    });
    learn(&predictor, "| a a b");
    assert_eq!(predictor.read().since_decay, 3);
    learn(&predictor, "| c");
    // Halved: a 2 → 1, <s> 2 → 1, b and c dropped.
    assert_eq!(predictor.read().since_decay, 0);
    assert_eq!(predict(&predictor, "", ""), ["a"]);

    let predictor = UserPredictor::in_memory(UserConfig {
        max_entries: 10,
        ..UserConfig::default()
    });
    learn(&predictor, "| the quick brown fox jumps over the lazy dog");
    assert!(predictor.entries() <= 10);
}

#[test]
fn rejects_other_language_and_second_open() {
    let dir = TempDir::new();
    let path = dir.file("user.redb");
    let predictor = open(&path, UserConfig::default());
    assert!(matches!(
        UserPredictor::open(&path, "en", UserConfig::default()),
        Err(Error::UserStore(_))
    ));
    drop(predictor);
    let err = UserPredictor::open(&path, "de", UserConfig::default()).unwrap_err();
    assert!(err.to_string().contains("\"en\""), "{err}");

    let garbage = dir.file("garbage.redb");
    std::fs::write(&garbage, b"not a database at all, just some bytes").unwrap();
    assert!(UserPredictor::open(&garbage, "en", UserConfig::default()).is_err());
}

#[test]
fn context_tokens_continue_ngrams() {
    let predictor = UserPredictor::in_memory(UserConfig::default());
    learn(&predictor, "| good");
    let text = "morning";
    predictor.learn(&LearnInput {
        context: &[Token::SentenceStart, Token::Word("good".into())],
        tokens: &spans(text),
        text,
    });
    assert_eq!(predict(&predictor, "| good", ""), ["morning", "good"]);
}
