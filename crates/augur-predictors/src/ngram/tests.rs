use std::collections::BTreeMap;
use std::fmt::Write as _;

use augur_lm::{Arpa, write_model};

use super::*;

const SMALL: &str = "\
\\data\\
ngram 1=9
ngram 2=7
ngram 3=3

\\1-grams:
-1.0\t</s>
-99\t<s>\t-0.5
-2.0\t<unk>\t-0.1
-1.3\t<num>
-1.2\tthe\t-0.3
-1.5\tcat\t-0.2
-1.6\tcar\t-0.1
-1.7\tsat
-1.1\tlondon

\\2-grams:
-0.4\t<s> the\t-0.25
-0.6\tthe cat\t-0.15
-0.9\tthe car
-0.3\tcat sat\t-0.05
-0.8\tsat </s>
-1.1\t<s> cat
-0.5\t<unk> sat

\\3-grams:
-0.2\t<s> the cat
-0.1\tthe cat sat
-0.7\t<s> the car

\\end\\
";

fn predictor(arpa: &str, casing: &[(&str, &str)]) -> NgramPredictor {
    let arpa = Arpa::read(arpa.as_bytes()).unwrap();
    let casing: Vec<(String, String)> = casing
        .iter()
        .map(|&(w, f)| (w.to_owned(), f.to_owned()))
        .collect();
    let mut bytes = Vec::new();
    write_model(&arpa, &[], &casing, &mut bytes).unwrap();
    NgramPredictor::new(Model::from_bytes(bytes).unwrap())
}

fn context(history: &str, prefix: &str, next: Option<&str>) -> TokenizedContext {
    TokenizedContext {
        history: history
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(|w| {
                if w == "<s>" {
                    Token::SentenceStart
                } else {
                    Token::Word(w.to_owned())
                }
            })
            .collect(),
        prefix: prefix.to_owned(),
        next_word: next.map(str::to_owned),
    }
}

fn predict(p: &NgramPredictor, ctx: &TokenizedContext, max: usize) -> Vec<Candidate> {
    let mut out = Vec::new();
    p.predict(ctx, max, &mut out);
    out
}

fn words(candidates: &[Candidate]) -> Vec<&str> {
    candidates.iter().map(|c| c.word.as_str()).collect()
}

/// Ranks every word of the prefix range by `Model::log_prob` and
/// normalizes over the range.
fn brute_force(p: &NgramPredictor, ctx: &TokenizedContext, max: usize) -> Vec<(String, f64)> {
    let model = p.model();
    let ids = p.context_ids(&ctx.history);
    let next = ctx.next_word.as_deref().and_then(|w| model.id(w));
    let mut all: Vec<(WordId, f32)> = model
        .prefix_range(&ctx.prefix)
        .filter(|&id| !p.is_marker(id))
        .map(|id| (id, model.log_prob(&ids, id)))
        .collect();
    let total: f64 = all.iter().map(|&(_, lp)| exp10(lp)).sum::<f64>()
        - next
            .filter(|id| all.iter().any(|(w, _)| w == id))
            .map_or(0.0, |id| exp10(model.log_prob(&ids, id)));
    all.retain(|&(id, _)| Some(id) != next);
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(max);
    all.into_iter()
        .map(|(id, lp)| (model.word(id).to_owned(), exp10(lp) / total))
        .collect()
}

fn assert_matches_brute_force(p: &NgramPredictor, ctx: &TokenizedContext, max: usize) {
    let actual = predict(p, ctx, max);
    let expected = brute_force(p, ctx, max);
    assert_eq!(
        words(&actual),
        expected.iter().map(|(w, _)| w.as_str()).collect::<Vec<_>>(),
        "{ctx:?}"
    );
    for (candidate, (_, prob)) in actual.iter().zip(&expected) {
        assert!(
            (f64::from(candidate.prob) - prob).abs() < 1e-4,
            "{ctx:?}: {} {} vs {prob}",
            candidate.word,
            candidate.prob
        );
    }
}

#[test]
fn predicts_next_words() {
    let p = predictor(SMALL, &[]);
    let ctx = context("<s> the", "", None);
    let out = predict(&p, &ctx, 3);
    assert_eq!(words(&out), ["cat", "car", "london"]);
    assert_matches_brute_force(&p, &ctx, 3);
    assert_matches_brute_force(&p, &ctx, 100);
}

#[test]
fn completes_prefixes_with_conditional_probabilities() {
    let p = predictor(SMALL, &[]);
    let ctx = context("<s> the", "ca", None);
    let out = predict(&p, &ctx, 10);
    assert_eq!(words(&out), ["cat", "car"]);
    let sum: f32 = out.iter().map(|c| c.prob).sum();
    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    assert_matches_brute_force(&p, &ctx, 10);
    assert!(predict(&p, &context("the", "x", None), 5).is_empty());
}

#[test]
fn full_distribution_sums_to_one() {
    let p = predictor(SMALL, &[]);
    for history in ["", "<s>", "<s> the", "the cat", "sat", "zebra"] {
        let out = predict(&p, &context(history, "", None), 100);
        let sum: f32 = out.iter().map(|c| c.prob).sum();
        assert!((sum - 1.0).abs() < 1e-5, "{history:?}: {sum}");
    }
}

#[test]
fn never_suggests_markers_or_the_next_word() {
    let p = predictor(SMALL, &[]);
    let out = predict(&p, &context("cat sat", "", None), 100);
    assert!(words(&out).iter().all(|w| !w.starts_with('<')), "{out:?}");
    assert!(predict(&p, &context("the", "<", None), 5).is_empty());

    let ctx = context("<s> the", "", Some("cat"));
    let out = predict(&p, &ctx, 100);
    assert!(!words(&out).contains(&"cat"));
    let sum: f32 = out.iter().map(|c| c.prob).sum();
    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    assert_matches_brute_force(&p, &ctx, 100);
}

#[test]
fn uses_only_the_current_sentence_and_unknown_words() {
    let p = predictor(SMALL, &[]);
    // Words before the last sentence start are ignored.
    assert_eq!(
        p.context_ids(&context("cat sat <s> the", "", None).history)
            .len(),
        2
    );
    assert_eq!(
        predict(&p, &context("cat sat <s> the", "", None), 3),
        predict(&p, &context("<s> the", "", None), 3)
    );
    // An unknown word acts as <unk>: "<unk> sat" is stored.
    let out = predict(&p, &context("zebra", "", None), 1);
    assert_eq!(words(&out), ["sat"]);
}

#[test]
fn provides_display_forms() {
    let p = predictor(SMALL, &[("london", "London")]);
    assert_eq!(p.display_form("london").as_deref(), Some("London"));
    assert_eq!(p.display_form("cat"), None);
    assert_eq!(p.display_form("zebra"), None);
}

#[test]
fn zero_max_or_unigram_model() {
    let p = predictor(SMALL, &[]);
    assert!(predict(&p, &context("the", "", None), 0).is_empty());

    let unigrams = "\\data\\\nngram 1=3\n\n\\1-grams:\n-0.2 b\n-0.5 a\n-1 <unk>\n\n\\end\\\n";
    let p = predictor(unigrams, &[]);
    let out = predict(&p, &context("<s> x y", "", None), 5);
    assert_eq!(words(&out), ["b", "a"]);
}

/// Deterministic pseudo-random numbers (xorshift).
struct Rng(u64);

impl Rng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % bound as u64).unwrap()
    }

    #[allow(clippy::cast_precision_loss)]
    fn log(&mut self) -> f32 {
        -(self.below(1_000_000) as f32) / 150_000.0
    }
}

/// A random model with more words than `SCAN_LIMIT`, so that both ways of
/// finding unigram candidates are exercised.
fn random_model(rng: &mut Rng) -> (NgramPredictor, Vec<String>) {
    let mut words: Vec<String> = vec!["<s>".into(), "</s>".into(), "<unk>".into(), "<num>".into()];
    let mut known = std::collections::HashSet::new();
    while words.len() < 6000 {
        let len = 1 + rng.below(6);
        let word: String = (0..len)
            .map(|_| char::from(b'a' + u8::try_from(rng.below(6)).unwrap()))
            .collect();
        if known.insert(word.clone()) {
            words.push(word);
        }
    }
    let mut grams: Vec<BTreeMap<Vec<usize>, (f32, f32)>> = vec![BTreeMap::new(); 3];
    for w in 0..words.len() {
        let prob = if w == 0 {
            augur_lm::LOG_ZERO
        } else {
            rng.log() - 2.0
        };
        grams[0].insert(vec![w], (prob, rng.log() / 4.0));
    }
    for n in 1..3 {
        let parents: Vec<Vec<usize>> = grams[n - 1].keys().cloned().collect();
        for _ in 0..40_000 {
            let mut gram = parents[rng.below(parents.len())].clone();
            // Favour a few frequent contexts so they get many continuations.
            if rng.below(3) == 0 {
                gram = parents[rng.below(20)].clone();
            }
            gram.push(1 + rng.below(words.len() - 1));
            grams[n].insert(gram, (rng.log(), if n < 2 { rng.log() / 4.0 } else { 0.0 }));
        }
    }
    let mut text = String::from("\\data\\\n");
    for (n, g) in grams.iter().enumerate() {
        writeln!(text, "ngram {}={}", n + 1, g.len()).unwrap();
    }
    for (n, g) in grams.iter().enumerate() {
        writeln!(text, "\n\\{}-grams:", n + 1).unwrap();
        for (gram, (p, b)) in g {
            let joined: Vec<&str> = gram.iter().map(|&w| words[w].as_str()).collect();
            writeln!(text, "{p}\t{}\t{b}", joined.join(" ")).unwrap();
        }
    }
    text += "\n\\end\\\n";
    (predictor(&text, &[]), words)
}

#[test]
fn matches_brute_force_on_a_random_model() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (p, words) = random_model(&mut rng);
    let mut contexts: Vec<String> = vec![String::new(), "<s>".into()];
    // Frequent contexts (low indices were favoured) and random ones.
    for i in 0..30 {
        let a = if i % 2 == 0 {
            4 + i
        } else {
            4 + rng.below(words.len() - 4)
        };
        let b = 4 + rng.below(words.len() - 4);
        contexts.push(format!("{} {}", words[a], words[b]));
        contexts.push(format!("<s> {}", words[a]));
        contexts.push(words[a].clone());
    }
    for history in &contexts {
        for prefix in ["", "a", "bc", "ffff"] {
            let next = (rng.below(4) == 0).then(|| words[4 + rng.below(20)].clone());
            let ctx = context(history, prefix, next.as_deref());
            assert_matches_brute_force(&p, &ctx, 1 + rng.below(12));
        }
    }
}
