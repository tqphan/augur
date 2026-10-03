use augur_core::TokenSpan;

use super::*;

/// Tokens of whitespace-separated text: `|` is a sentence start, other
/// words are lowercased.
pub(crate) fn spans(text: &str) -> Vec<TokenSpan> {
    text.split_whitespace()
        .map(|w| {
            let start = w.as_ptr() as usize - text.as_ptr() as usize;
            if w == "|" {
                TokenSpan {
                    token: Token::SentenceStart,
                    span: start..start,
                }
            } else {
                TokenSpan {
                    token: Token::Word(w.to_lowercase()),
                    span: start..start + w.len(),
                }
            }
        })
        .collect()
}

fn learn(counts: &mut Counts, text: &str) -> u64 {
    counts.learn(&LearnInput {
        context: &[],
        tokens: &spans(text),
        text,
    })
}

fn context(history: &str, prefix: &str) -> TokenizedContext {
    TokenizedContext {
        history: spans(history).into_iter().map(|s| s.token).collect(),
        prefix: prefix.to_owned(),
        next_word: None,
    }
}

fn predict(counts: &Counts, context: &TokenizedContext, max: usize) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    counts.predict(context, max, &mut out);
    out.into_iter().map(|c| (c.word, c.prob)).collect()
}

fn words(out: &[(String, f32)]) -> Vec<&str> {
    out.iter().map(|(w, _)| w.as_str()).collect()
}

const TEXT: &str = "| see you soon | see you tomorrow | see you soon | i see a cat";

#[test]
fn learns_and_predicts_trigrams() {
    let mut counts = Counts::new(false);
    assert_eq!(learn(&mut counts, TEXT), 13);
    // Distinct words, <s> included.
    assert_eq!(counts.ids.len(), 8);

    let out = predict(&counts, &context("| see you", ""), 3);
    assert_eq!(words(&out), ["soon", "tomorrow", "see"]);
    assert!(out.windows(2).all(|w| w[0].1 >= w[1].1));

    // Exact interpolated probabilities: total 13 words.
    let p1 = |c: f64| c / 13.0;
    // "you": bigram history total 3, followers soon(2) tomorrow(1).
    let p2 = |c: f64, lower: f64| ((c - 0.75).max(0.0) + 0.75 * 2.0 * lower) / 3.0;
    // "see you": same counts.
    let soon = p2(2.0, p2(2.0, p1(2.0)));
    let tomorrow = p2(1.0, p2(1.0, p1(1.0)));
    let see = p2(0.0, p2(0.0, p1(4.0)));
    // The prefix range is every word but <s>: unigram mass 1.
    let level = |z: f64| (1.5 + 0.75 * 2.0 * z) / 3.0;
    let z = level(level(1.0));
    for (word, p) in [("soon", soon), ("tomorrow", tomorrow), ("see", see)] {
        let (_, prob) = out.iter().find(|(w, _)| w == word).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let expected = (p / z) as f32;
        assert!(
            (prob - expected).abs() < 1e-6,
            "{word}: {prob} vs {expected}"
        );
    }

    // All candidates' probabilities sum to 1 over the prefix range.
    let all = predict(&counts, &context("| see you", ""), 100);
    let sum: f32 = all.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    let all = predict(&counts, &context("| see you", "s"), 100);
    assert_eq!(words(&all), ["soon", "see"]);
    let sum: f32 = all.iter().map(|(_, p)| p).sum();
    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
}

#[test]
fn skips_markers_and_next_word() {
    let mut counts = Counts::new(false);
    learn(&mut counts, "| buy <num> eggs | buy milk");
    let out = predict(&counts, &context("| buy", ""), 10);
    assert!(!words(&out).contains(&"<num>"), "{out:?}");
    assert!(!words(&out).contains(&"<s>"));
    let mut ctx = context("| buy", "");
    ctx.next_word = Some("milk".into());
    assert!(!words(&predict(&counts, &ctx, 10)).contains(&"milk"));
    assert!(predict(&counts, &context("", "zz"), 10).is_empty());
    assert!(predict(&Counts::new(false), &context("", ""), 10).is_empty());
}

#[test]
fn continues_from_context() {
    let mut counts = Counts::new(false);
    learn(&mut counts, "| see you");
    let text = "later";
    let context_tokens = [
        Token::SentenceStart,
        Token::Word("see".into()),
        Token::Word("you".into()),
    ];
    counts.learn(&LearnInput {
        context: &context_tokens,
        tokens: &spans(text),
        text,
    });
    assert_eq!(counts.trigrams.len(), 2);
    let you = counts.id("you").unwrap();
    assert_eq!(counts.bigrams[&[you, counts.id("later").unwrap()]], 1);
    // Context tokens are not counted again.
    assert_eq!(counts.count(you), 1);
}

#[test]
fn learns_casing_in_mid_sentence() {
    let mut counts = Counts::new(false);
    learn(
        &mut counts,
        "| Hello Quoc | i met iPhone users | THE NASA STAFF | ask NASA",
    );
    assert_eq!(counts.display_form("quoc").as_deref(), Some("Quoc"));
    assert_eq!(counts.display_form("iphone").as_deref(), Some("iPhone"));
    assert_eq!(counts.display_form("nasa").as_deref(), Some("NASA"));
    // Sentence-initial and lowercase words have no form; caps lock runs
    // do not vote.
    assert_eq!(counts.display_form("hello"), None);
    assert_eq!(counts.display_form("met"), None);
    assert_eq!(counts.display_form("staff"), None);
    learn(&mut counts, "| to quoc and quoc");
    assert_eq!(counts.display_form("quoc"), None);
}

#[test]
fn halves_and_drops() {
    let mut counts = Counts::new(true);
    learn(&mut counts, "| a b | a b | a c");
    let before = counts.entries();
    counts.halve();
    assert!(counts.entries() < before);
    assert_eq!(counts.id("c"), None);
    let a = counts.id("a").unwrap();
    let b = counts.id("b").unwrap();
    assert_eq!(counts.count(a), 1);
    assert_eq!(counts.bigrams[&[a, b]], 1);
    assert_eq!(counts.total, 2);
    let out = predict(&counts, &context("| a", ""), 10);
    assert_eq!(words(&out), ["b", "a"]);

    let (_, changes) = counts.take_changes();
    assert!(changes.rewrite);
    assert_eq!(changes.words.len(), 3);
}

#[test]
fn forgets_words_and_their_ngrams() {
    let mut counts = Counts::new(true);
    learn(&mut counts, "| my secret code | my secret");
    let _ = counts.take_changes();
    assert!(counts.forget("secret"));
    assert!(!counts.forget("secret"));
    assert!(!counts.forget(BOS));
    assert_eq!(counts.bigrams.len(), 1);
    assert!(counts.trigrams.is_empty());
    let out = predict(&counts, &context("| my", ""), 10);
    assert!(!words(&out).contains(&"secret"));

    let (dirty, changes) = counts.take_changes();
    assert_eq!(changes.forgotten, vec![Box::from("secret")]);
    counts.restore(dirty);
    let (_, changes) = counts.take_changes();
    assert_eq!(changes.forgotten, vec![Box::from("secret")]);
}

#[test]
fn tracks_incremental_changes() {
    let mut counts = Counts::new(true);
    learn(&mut counts, "| a b");
    let (dirty, changes) = counts.take_changes();
    assert_eq!(changes.words.len(), 3);
    assert_eq!(changes.bigrams.len(), 2);
    assert_eq!(changes.trigrams.len(), 1);
    assert!(counts.take_changes().1.is_empty());
    counts.restore(dirty);
    assert_eq!(counts.take_changes().1.words.len(), 3);

    let mut untracked = Counts::new(false);
    learn(&mut untracked, "| a b");
    assert!(untracked.take_changes().1.is_empty());
}

#[test]
fn round_trips_through_records() {
    let mut counts = Counts::new(true);
    learn(&mut counts, TEXT);
    counts.halve();
    let (_, changes) = counts.take_changes();
    let loaded = Counts::from_records(
        changes
            .words
            .into_iter()
            .map(|(w, r)| (w.into_string(), r.unwrap())),
        changes
            .bigrams
            .into_iter()
            .map(|(k, c)| (k.map(String::from), c)),
        changes
            .trigrams
            .into_iter()
            .map(|(k, c)| (k.map(String::from), c)),
        7,
    );
    assert_eq!(loaded.entries(), counts.entries());
    assert_eq!(loaded.total, counts.total);
    assert_eq!(loaded.since_decay, 7);
    let ctx = context("| see you", "");
    let (a, b) = (predict(&loaded, &ctx, 5), predict(&counts, &ctx, 5));
    assert_eq!(words(&a), words(&b));
    assert!(a.iter().zip(&b).all(|(x, y)| (x.1 - y.1).abs() < 1e-6));
}
