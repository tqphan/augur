// Values stored exactly (few distinct values) are compared exactly.
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::*;

const SMALL: &str = "\
Produced by a test.

\\data\\
ngram 1=7
ngram 2=6
ngram 3=3

\\1-grams:
-1.0\t</s>
-99\t<s>\t-0.5
-2.0\t<unk>
-1.2\tthe\t-0.3
-1.5\tcat\t-0.2
-1.6\tcar\t-0.1
-1.7\tsat

\\2-grams:
-0.4\t<s> the\t-0.25
-0.6\tthe cat\t-0.15
-0.9\tthe car
-0.3\tcat sat\t-0.05
-0.8\tsat </s>
-1.1\t<s> cat

\\3-grams:
-0.2\t<s> the cat
-0.1\tthe cat sat
-0.7\t<s> the car

\\end\\
";

fn compile(arpa: &str, meta: &[(&str, &str)], casing: &[(&str, &str)]) -> (Vec<u8>, WriteStats) {
    let arpa = Arpa::read(arpa.as_bytes()).unwrap();
    let own = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|&(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    };
    let mut bytes = Vec::new();
    let stats = write_model(&arpa, &own(meta), &own(casing), &mut bytes).unwrap();
    (bytes, stats)
}

fn small() -> Model {
    let (bytes, _) = compile(
        SMALL,
        &[("source", "test"), ("license", "CC0-1.0")],
        &[("cat", "Cat"), ("london", "London")],
    );
    Model::from_bytes(bytes).unwrap()
}

fn ids(model: &Model, words: &str) -> Vec<WordId> {
    words.split(' ').map(|w| model.id(w).unwrap()).collect()
}

#[test]
fn reads_header_data() {
    let model = small();
    assert_eq!(model.order(), 3);
    assert_eq!(model.counts(), [7, 6, 3]);
    assert_eq!(model.vocab_len(), 7);
    assert_eq!(model.meta_value("license"), Some("CC0-1.0"));
    assert_eq!(model.meta()[0], ("source".to_owned(), "test".to_owned()));
    assert_eq!(model.bos().map(|id| model.word(id)), Some("<s>"));
    assert_eq!(model.eos().map(|id| model.word(id)), Some("</s>"));
    assert_eq!(model.unk().map(|id| model.word(id)), Some("<unk>"));
}

#[test]
fn vocabulary_is_sorted_by_bytes() {
    let model = small();
    let words: Vec<&str> = (0..7).map(|id| model.word(id)).collect();
    assert_eq!(words, ["</s>", "<s>", "<unk>", "car", "cat", "sat", "the"]);
    assert_eq!(model.id("cat"), Some(4));
    assert_eq!(model.id("ca"), None);
    assert_eq!(model.id("zebra"), None);

    let range = model.prefix_range("ca");
    let found: Vec<&str> = range.map(|id| model.word(id)).collect();
    assert_eq!(found, ["car", "cat"]);
    assert_eq!(model.prefix_range(""), 0..7);
    assert!(model.prefix_range("x").is_empty());
    assert!(model.prefix_range("cats").is_empty());
}

#[test]
fn stores_ngrams_exactly_when_few_values() {
    let model = small();
    let entry = model.ngram(&ids(&model, "the cat")).unwrap();
    assert_eq!(entry.log_prob, -0.6);
    assert_eq!(entry.log_backoff, -0.15);
    assert_eq!(
        model.ngram(&ids(&model, "the cat sat")).unwrap().log_prob,
        -0.1
    );
    assert_eq!(model.ngram(&ids(&model, "<s>")).unwrap().log_prob, LOG_ZERO);
    assert!(model.ngram(&ids(&model, "cat the")).is_none());
    assert!(model.ngram(&[]).is_none());
}

#[test]
fn backs_off() {
    let model = small();
    let p = |context: &str, word: &str| {
        let context = if context.is_empty() {
            Vec::new()
        } else {
            ids(&model, context)
        };
        model.log_prob(&context, model.id(word).unwrap())
    };
    let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
    // Found at the highest order.
    assert!(close(p("<s> the", "cat"), -0.2));
    // Only the last two words of the context matter.
    assert!(close(p("sat <s> the", "cat"), -0.2));
    // "the cat car" is missing: bo(the cat) + p(car | cat) → bo(cat) + p(car).
    assert!(close(p("the cat", "car"), -0.15 + -0.2 + -1.6));
    // "<s> the sat" missing, "the sat" missing: bo(<s> the) + bo(the) + p(sat).
    assert!(close(p("<s> the", "sat"), -0.25 + -0.3 + -1.7));
    // Unknown context ids are skipped.
    assert!(close(p("", "the"), -1.2));
    assert!(close(
        model.log_prob(&[999, model.id("the").unwrap()], model.id("cat").unwrap()),
        -0.6
    ));
    // <s> is never predicted.
    assert_eq!(p("the", "<s>"), LOG_ZERO);
    assert_eq!(model.log_prob(&[], 999), LOG_ZERO);
}

#[test]
fn lists_continuations() {
    let model = small();
    let words = |context: &str, prefix: &str| -> Vec<(String, f32)> {
        let context = if context.is_empty() {
            Vec::new()
        } else {
            ids(&model, context)
        };
        model
            .continuations(&context, model.prefix_range(prefix))
            .map(|(id, p)| (model.word(id).to_owned(), p))
            .collect()
    };
    assert_eq!(
        words("the", ""),
        [("car".to_owned(), -0.9), ("cat".to_owned(), -0.6)]
    );
    assert_eq!(words("the", "cat"), [("cat".to_owned(), -0.6)]);
    assert_eq!(
        words("<s> the", ""),
        [("car".to_owned(), -0.7), ("cat".to_owned(), -0.2)]
    );
    assert_eq!(words("", "s").len(), 1);
    assert_eq!(model.continuations(&[], 0..7).len(), 7);
    assert!(words("car", "").is_empty());
    assert!(words("the cat sat", "").is_empty());
}

#[test]
fn stores_casing_of_known_words() {
    let (bytes, stats) = compile(SMALL, &[], &[("cat", "Cat"), ("london", "London")]);
    assert_eq!((stats.casing_entries, stats.casing_skipped), (1, 1));
    let model = Model::from_bytes(bytes).unwrap();
    assert_eq!(model.casing(model.id("cat").unwrap()), Some("Cat"));
    assert_eq!(model.casing(model.id("the").unwrap()), None);
}

#[test]
fn reports_stats() {
    let (bytes, stats) = compile(SMALL, &[], &[]);
    assert_eq!(stats.counts, [7, 6, 3]);
    assert_eq!(stats.bytes, bytes.len() as u64);
    assert!(stats.prob_error.iter().all(|&e| e == 0.0));
}

#[test]
fn opens_files() {
    let (bytes, _) = compile(SMALL, &[], &[]);
    let path = std::env::temp_dir().join(format!("augur-lm-{}.bin", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let model = Model::open(&path);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(model.unwrap().counts(), [7, 6, 3]);
}

#[test]
fn unigram_model() {
    let arpa = "\\data\\\nngram 1=2\n\n\\1-grams:\n-0.3 a\n-0.2 b\n\n\\end\\\n";
    let (bytes, _) = compile(arpa, &[], &[]);
    let model = Model::from_bytes(bytes).unwrap();
    assert_eq!(model.order(), 1);
    assert_eq!(model.log_prob(&[0], 1), -0.2);
    assert_eq!(model.continuations(&[0], 0..2).len(), 0);
    assert_eq!(model.bos(), None);
}

#[test]
fn rejects_damaged_files() {
    let (bytes, _) = compile(SMALL, &[], &[]);

    assert!(matches!(
        Model::from_bytes(b"hello".to_vec()),
        Err(Error::BadMagic)
    ));

    let mut version = bytes.clone();
    version[8] = 9;
    assert!(matches!(
        Model::from_bytes(version),
        Err(Error::UnsupportedVersion { found: 9, .. })
    ));

    // Flipping any byte after the magic and version is caught.
    for i in 12..bytes.len() {
        let mut damaged = bytes.clone();
        damaged[i] ^= 0x40;
        assert!(Model::from_bytes(damaged).is_err(), "byte {i}");
    }
    for len in [bytes.len() - 1, bytes.len() / 2, 20] {
        assert!(Model::from_bytes(&bytes[..len]).is_err(), "length {len}");
    }
}

#[test]
fn rejects_bad_arpa() {
    let error = |text: &str| Arpa::read(text.as_bytes()).unwrap_err().to_string();
    let wrap = |sections: &str| format!("\\data\\\nngram 1=2\nngram 2=1\n\n{sections}\\end\\\n");

    assert!(error("no header").contains("missing \\data\\"));
    assert!(
        error(&wrap("\\1-grams:\n-1 a\n-1 b\n\n\\2-grams:\n-1 a c\n\n"))
            .contains("\"c\" has no unigram")
    );
    assert!(error(&wrap("\\1-grams:\n-1 a\n\n\\2-grams:\n-1 a a\n\n")).contains("declares 2"));
    assert!(
        error(&wrap("\\1-grams:\n-1 a\n-1 a\n\n\\2-grams:\n-1 a a\n\n"))
            .contains("duplicate unigram")
    );
    assert!(
        error(&wrap("\\1-grams:\n-1 a\n-1 b\n\n\\2-grams:\n-1 a\n\n")).contains("expected 2 words")
    );
    assert!(
        error(&wrap("\\1-grams:\n-1 a\n-1 b\n\n\\2-grams:\nx a b\n\n")).contains("log probability")
    );
    assert!(
        error(&wrap(
            "\\1-grams:\n-1 a\n-1 b\n\n\\2-grams:\n-1 a b -1 x\n\n"
        ))
        .contains("too many fields")
    );

    let orphan = "\\data\\\nngram 1=2\nngram 2=1\nngram 3=1\n\n\\1-grams:\n-1 a\n-1 b\n\n\
                  \\2-grams:\n-1 a b\n\n\\3-grams:\n-1 b a b\n\n\\end\\\n";
    let arpa = Arpa::read(orphan.as_bytes()).unwrap();
    let error = write_model(&arpa, &[], &[], &mut Vec::new()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("\"b a b\" has no entry for its prefix"),
        "{error}"
    );

    let duplicate = "\\data\\\nngram 1=2\nngram 2=2\n\n\\1-grams:\n-1 a\n-1 b\n\n\
                     \\2-grams:\n-1 a b\n-2 a b\n\n\\end\\\n";
    let arpa = Arpa::read(duplicate.as_bytes()).unwrap();
    let error = write_model(&arpa, &[], &[], &mut Vec::new()).unwrap_err();
    assert!(
        error.to_string().contains("duplicate n-gram \"a b\""),
        "{error}"
    );
}

#[test]
fn accepts_arpa_variants() {
    // Space separators, CRLF, -inf, backoffs on the highest order.
    let text = "\\data\\\r\nngram 1=2\r\nngram 2=1\r\n\r\n\\1-grams:\r\n-inf <s> -0.5\r\n-0.1 a 0\r\n\r\n\
                \\2-grams:\r\n-0.2 <s> a -0.3\r\n\r\n\\end\\\r\n";
    let (bytes, _) = compile(text, &[], &[]);
    let model = Model::from_bytes(bytes).unwrap();
    let (s, a) = (model.id("<s>").unwrap(), model.id("a").unwrap());
    assert_eq!(model.ngram(&[s]).unwrap().log_prob, LOG_ZERO);
    assert_eq!(model.ngram(&[s, a]).unwrap().log_backoff, 0.0);
    assert_eq!(model.log_prob(&[s], a), -0.2);
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

/// A random, consistent back-off model with enough distinct values to
/// need real quantization, checked against a direct implementation of
/// back-off over the ARPA entries.
#[test]
fn matches_reference_back_off_on_a_large_model() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let words: Vec<String> = (0..500).map(|i| format!("w{i}")).collect();
    let mut grams: Vec<BTreeMap<Vec<usize>, (f32, f32)>> = vec![BTreeMap::new(); 3];
    for w in 0..words.len() {
        grams[0].insert(vec![w], (rng.log(), rng.log() / 4.0));
    }
    for n in 1..3 {
        let parents: Vec<Vec<usize>> = grams[n - 1].keys().cloned().collect();
        for _ in 0..20_000 {
            let mut gram = parents[rng.below(parents.len())].clone();
            gram.push(rng.below(words.len()));
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

    let (bytes, stats) = compile(&text, &[], &[]);
    assert!(
        stats.prob_error.iter().all(|&e| e < 0.01),
        "{:?}",
        stats.prob_error
    );
    let model = Model::from_bytes(bytes).unwrap();

    let reference = |context: &[usize], word: usize| -> f32 {
        let mut backoff = 0.0;
        for start in 0..=context.len() {
            let mut gram = context[start..].to_vec();
            gram.push(word);
            if let Some(&(p, _)) = grams[gram.len() - 1].get(&gram) {
                return backoff + p;
            }
            let history = &context[start..];
            if let Some(&(_, b)) = history
                .len()
                .checked_sub(1)
                .and_then(|i| grams[i].get(history))
            {
                backoff += b;
            }
        }
        unreachable!("every word has a unigram")
    };
    let id = |w: usize| model.id(&words[w]).unwrap();

    let mut worst: f32 = 0.0;
    for _ in 0..20_000 {
        let context: Vec<usize> = (0..2).map(|_| rng.below(words.len())).collect();
        // Half of the time, use a context that exists.
        let context = if rng.below(2) == 0 {
            grams[1].keys().nth(rng.below(100)).unwrap().clone()
        } else {
            context
        };
        let word = rng.below(words.len());
        let expected = reference(&context, word);
        let actual = model.log_prob(
            &context.iter().map(|&w| id(w)).collect::<Vec<_>>(),
            id(word),
        );
        worst = worst.max((expected - actual).abs());
    }
    assert!(worst < 0.1, "worst difference {worst}");

    // Continuations of every bigram context agree with the ARPA entries.
    for gram in grams[1].keys().take(200) {
        let context: Vec<WordId> = gram.iter().map(|&w| id(w)).collect();
        let expected = grams[2].keys().filter(|g| g[..2] == gram[..]).count();
        assert_eq!(model.continuations(&context, 0..500).len(), expected);
    }
}
