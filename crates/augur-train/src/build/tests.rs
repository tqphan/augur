use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;

use super::*;

/// A parsed ARPA model: n-gram → (log10 prob, log10 back-off).
struct Arpa {
    entries: HashMap<Vec<String>, (f64, Option<f64>)>,
    counts: Vec<usize>,
}

impl Arpa {
    fn parse(text: &str) -> Self {
        let mut entries = HashMap::new();
        let mut counts = Vec::new();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("\\data\\"));
        for line in lines {
            if let Some(count) = line.strip_prefix("ngram ") {
                counts.push(count.split_once('=').unwrap().1.parse().unwrap());
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 2 {
                continue;
            }
            let words = fields[1].split(' ').map(str::to_owned).collect();
            let backoff = fields.get(2).map(|b| b.parse().unwrap());
            entries.insert(words, (fields[0].parse().unwrap(), backoff));
        }
        Self { entries, counts }
    }

    fn get(&self, words: &[&str]) -> Option<(f64, Option<f64>)> {
        let key: Vec<String> = words.iter().map(|&w| w.to_owned()).collect();
        self.entries.get(&key).copied()
    }

    /// Back-off probability of `word` after `context`.
    fn prob(&self, context: &[&str], word: &str) -> f64 {
        let mut gram = context.to_vec();
        gram.push(word);
        if let Some((logp, _)) = self.get(&gram) {
            return 10f64.powf(logp);
        }
        let backoff = self
            .get(context)
            .and_then(|(_, b)| b)
            .map_or(1.0, |b| 10f64.powf(b));
        backoff * self.prob(&context[1..], word)
    }

    fn vocab(&self) -> Vec<&str> {
        self.entries
            .keys()
            .filter(|k| k.len() == 1 && k[0] != "<s>")
            .map(|k| k[0].as_str())
            .collect()
    }
}

struct Built {
    arpa: Arpa,
    arpa_text: String,
    meta: String,
    casing: Option<String>,
    _tmp: TempDir,
}

fn build(corpus: &str, smoothing: Smoothing, min_count: &[u64], memory: usize) -> Built {
    build_with(corpus, smoothing, min_count, memory, None)
}

fn build_with(
    corpus: &str,
    smoothing: Smoothing,
    min_count: &[u64],
    memory: usize,
    casing: Option<&str>,
) -> Built {
    let tmp = TempDir::new(&std::env::temp_dir()).unwrap();
    let corpus_path = tmp.file("corpus").with_extension("txt");
    fs::write(&corpus_path, corpus).unwrap();
    if let Some(casing) = casing {
        fs::write(corpus_path.with_extension("casing"), casing).unwrap();
    }
    let output = tmp.file("model").with_extension("arpa");
    run(&BuildOptions {
        corpus: corpus_path,
        output: output.clone(),
        smoothing,
        min_count: min_count.to_vec(),
        vocab_size: 1000,
        alpha: 0.4,
        memory,
        tmp_dir: std::env::temp_dir(),
    })
    .unwrap();
    let arpa_text = fs::read_to_string(&output).unwrap();
    Built {
        arpa: Arpa::parse(&arpa_text),
        arpa_text,
        meta: fs::read_to_string(output.with_extension("meta")).unwrap(),
        casing: fs::read_to_string(output.with_extension("casing")).ok(),
        _tmp: tmp,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn kneser_ney_matches_hand_computation() {
    // Counts are too small for discount estimation, so the fallback
    // discounts 0.5, 1.0, 1.5 apply. See the derivation in the comments.
    let model = build("a b\nb b\n", Smoothing::Mkn, &[1, 1], 1 << 20).arpa;
    assert_eq!(model.counts, [5, 5]);
    // Unigram adjusted counts: a=1, b=3, </s>=1 (total 5); γ = 2.5/5;
    // uniform share 0.5/4 over </s>, <unk>, a, b.
    assert_close(model.prob(&[], "a"), 0.5 / 5.0 + 0.125);
    assert_close(model.prob(&[], "b"), 1.5 / 5.0 + 0.125);
    assert_close(model.prob(&[], "<unk>"), 0.125);
    assert_close(model.get(&["<s>"]).unwrap().0, -99.0);
    // p(b | a) = (1 - 0.5)/1 + 0.5 · p(b)
    assert_close(model.prob(&["a"], "b"), 0.5 + 0.5 * 0.425);
    // Context b: b </s> = 2, b b = 1; γ = (1.0 + 0.5)/3
    assert_close(model.prob(&["b"], "</s>"), 1.0 / 3.0 + 0.5 * 0.225);
    assert_close(model.prob(&["b"], "b"), 0.5 / 3.0 + 0.5 * 0.425);
    assert_close(10f64.powf(model.get(&["b"]).unwrap().1.unwrap()), 0.5);
    // Unseen bigram backs off.
    assert_close(model.prob(&["a"], "a"), 0.5 * 0.225);
}

/// Deterministic pseudo-random corpus with a skewed word distribution.
fn synthetic_corpus(sentences: usize) -> String {
    let mut state: u64 = 42;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let mut corpus = String::new();
    for _ in 0..sentences {
        let len = 1 + next() % 8;
        for i in 0..len {
            // Squaring skews the distribution towards low word numbers.
            let r = next() % 40;
            let word = r * r / 40;
            if i > 0 {
                corpus.push(' ');
            }
            write!(corpus, "w{word}").unwrap();
        }
        corpus.push('\n');
    }
    corpus
}

fn assert_normalized(model: &Arpa) {
    let vocab = model.vocab();
    let order = model.counts.len();
    let mut checked = 0;
    for context in model.entries.keys().filter(|k| k.len() < order) {
        if context.last().is_some_and(|w| w == "</s>") {
            continue;
        }
        let context: Vec<&str> = context.iter().map(String::as_str).collect();
        let total: f64 = vocab.iter().map(|w| model.prob(&context, w)).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "context {context:?} sums to {total}"
        );
        checked += 1;
    }
    let total: f64 = vocab.iter().map(|w| model.prob(&[], w)).sum();
    assert!((total - 1.0).abs() < 1e-4, "unigrams sum to {total}");
    assert!(checked > 10);
}

#[test]
fn kneser_ney_distributions_sum_to_one() {
    let corpus = synthetic_corpus(3000);
    let model = build(&corpus, Smoothing::Mkn, &[1, 1, 1], 1 << 24).arpa;
    assert_normalized(&model);
}

#[test]
fn pruned_kneser_ney_distributions_sum_to_one() {
    let corpus = synthetic_corpus(3000);
    let full = build(&corpus, Smoothing::Mkn, &[1, 1, 1], 1 << 24).arpa;
    let pruned = build(&corpus, Smoothing::Mkn, &[1, 2, 3], 1 << 24).arpa;
    assert!(pruned.counts[1] < full.counts[1]);
    assert!(pruned.counts[2] < full.counts[2]);
    assert_normalized(&pruned);
}

#[test]
fn spilling_to_disk_gives_identical_models() {
    let corpus = synthetic_corpus(3000);
    for smoothing in [Smoothing::Mkn, Smoothing::StupidBackoff] {
        let in_memory = build(&corpus, smoothing, &[1, 2, 2], 1 << 24).arpa;
        let on_disk = build(&corpus, smoothing, &[1, 2, 2], 0).arpa;
        assert_eq!(in_memory.counts, on_disk.counts);
        assert_eq!(in_memory.entries, on_disk.entries);
    }
}

#[test]
fn stupid_backoff_scores() {
    let built = build("a b\nb b\n", Smoothing::StupidBackoff, &[1, 1], 1 << 20);
    let model = &built.arpa;
    // Unigrams: a=1, b=3, </s>=2 out of 6.
    assert_close(model.prob(&[], "b"), 0.5);
    assert_close(model.prob(&["<s>"], "a"), 0.5);
    assert_close(model.prob(&["b"], "</s>"), 2.0 / 3.0);
    assert_close(10f64.powf(model.get(&["a"]).unwrap().1.unwrap()), 0.4);
    assert_close(model.prob(&["a"], "a"), 0.4 / 6.0);
    assert!(built.meta.contains("model.normalized = false"));
}

#[test]
fn pruned_models_compile_to_the_binary_format() {
    let corpus = synthetic_corpus(3000);
    for smoothing in [Smoothing::Mkn, Smoothing::StupidBackoff] {
        let built = build(&corpus, smoothing, &[1, 2, 3], 1 << 24);
        let parsed = augur_lm::Arpa::read(built.arpa_text.as_bytes()).unwrap();
        let mut bytes = Vec::new();
        augur_lm::write_model(&parsed, &[], &[], &mut bytes).unwrap();
        let model = augur_lm::Model::from_bytes(bytes).unwrap();

        let reference = &built.arpa;
        let vocab = reference.vocab();
        let mut checked = 0;
        for context in reference.entries.keys().filter(|k| k.len() < 3).take(100) {
            let context: Vec<&str> = context.iter().map(String::as_str).collect();
            let ids: Vec<u32> = context.iter().map(|w| model.id(w).unwrap()).collect();
            for word in &vocab {
                // Impossible events stay at LOG_ZERO in the binary model.
                let expected = reference
                    .prob(&context, word)
                    .log10()
                    .max(f64::from(augur_lm::LOG_ZERO));
                let actual = f64::from(model.log_prob(&ids, model.id(word).unwrap()));
                assert!(
                    (expected - actual).abs() < 0.05,
                    "{smoothing:?} {context:?} {word}: {expected} vs {actual}"
                );
                checked += 1;
            }
        }
        assert!(checked > 1000);
    }
}

#[test]
fn writes_meta_and_filters_casing() {
    let built = build_with(
        "london is big\nparis is big\n",
        Smoothing::Mkn,
        &[1, 1],
        1 << 20,
        Some("london\tLondon\nrome\tRome\n"),
    );
    assert_eq!(built.casing.as_deref(), Some("london\tLondon\n"));
    for expected in [
        "model.format = arpa",
        "model.order = 2",
        "model.smoothing = modified-kneser-ney",
        "model.ngrams = 7,",
        "corpus.sentences = 2",
        "corpus.tokens = 6",
        "model.casing_entries = 1",
    ] {
        assert!(
            built.meta.contains(expected),
            "missing {expected:?} in\n{}",
            built.meta
        );
    }
}
