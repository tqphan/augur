//! `augur eval`: measures an engine by simulating a user who types a text.
//!
//! Each line of the text is a separate document. For every word, the
//! simulated user types one character at a time and looks at the top
//! suggestions before each character, starting with none typed. As soon as
//! the word is among them, the user selects it for one keystroke. Words
//! match ignoring case. Only the characters of words count as keystrokes;
//! spaces and punctuation are typed either way.

use std::ops::Range;
use std::time::Instant;

use augur::{Context, Engine, LanguagePack, Token};
use serde::Serialize;

/// Settings of a simulation.
#[derive(Debug, Clone)]
pub(crate) struct EvalOptions {
    /// Suggestions shown at a time.
    pub suggestions: usize,
    /// Learn each sentence after it has been typed.
    pub learn: bool,
    /// Stop after this many words.
    pub max_words: Option<u64>,
}

/// Results of a simulation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Report {
    /// Suggestions shown at a time.
    pub suggestions: usize,
    /// Whether the engine learned while typing.
    pub learn: bool,
    /// Words typed.
    pub words: u64,
    /// Keystrokes to type the words without prediction.
    pub keystrokes_without: u64,
    /// Keystrokes to type the words with prediction.
    pub keystrokes_with: u64,
    /// Share of keystrokes saved, `1 - with / without`.
    pub keystroke_savings: f64,
    /// Share of words suggested first before typing any character.
    pub next_word_top1: f64,
    /// Share of words suggested before typing any character.
    pub next_word_top_n: f64,
    /// Calls to `Engine::predict`.
    pub predictions: u64,
    /// Mean time per prediction, in microseconds.
    pub mean_us: f64,
    /// Median time per prediction, in microseconds.
    pub p50_us: f64,
    /// 99th percentile time per prediction, in microseconds.
    pub p99_us: f64,
}

/// Simulates typing `text` with `engine`.
pub(crate) fn evaluate(engine: &Engine, text: &str, options: &EvalOptions) -> Report {
    let mut sim = Simulation {
        engine,
        pack: engine.language_pack(),
        options,
        words: 0,
        without: 0,
        with: 0,
        top1: 0,
        top_n: 0,
        times_us: Vec::new(),
    };
    for line in text.lines() {
        if sim.done() {
            break;
        }
        sim.line(line);
    }
    sim.report()
}

struct Simulation<'a> {
    engine: &'a Engine,
    pack: &'a dyn LanguagePack,
    options: &'a EvalOptions,
    words: u64,
    without: u64,
    with: u64,
    top1: u64,
    top_n: u64,
    times_us: Vec<f64>,
}

impl Simulation<'_> {
    fn done(&self) -> bool {
        self.options.max_words.is_some_and(|max| self.words >= max)
    }

    fn line(&mut self, line: &str) {
        let mut sentence = 0;
        for span in self.pack.tokenizer().tokenize_spans(line) {
            if self.done() {
                break;
            }
            match span.token {
                Token::SentenceStart => {
                    self.learn(&line[sentence..span.span.start]);
                    sentence = span.span.start;
                }
                Token::Word(_) => self.word(line, span.span),
            }
        }
        self.learn(&line[sentence..]);
        self.engine.reset_learning_context();
    }

    fn learn(&self, sentence: &str) {
        if self.options.learn && !sentence.trim().is_empty() {
            self.engine.learn(sentence);
        }
    }

    /// Types the word at `span` of `line`.
    fn word(&mut self, line: &str, span: Range<usize>) {
        let word = &line[span.clone()];
        let target = self.pack.normalize(word);
        let len = word.chars().count() as u64;
        let mut cost = len;
        for (typed, (offset, _)) in (0..).zip(word.char_indices()) {
            let past = &line[..span.start + offset];
            let start = Instant::now();
            let suggestions = self
                .engine
                .predict(Context::past(past), self.options.suggestions);
            self.times_us.push(start.elapsed().as_secs_f64() * 1e6);
            let rank = suggestions
                .iter()
                .position(|s| self.pack.normalize(&s.text) == target);
            if typed == 0 {
                self.top1 += u64::from(rank == Some(0));
                self.top_n += u64::from(rank.is_some());
            }
            if rank.is_some() {
                cost = typed + 1;
                break;
            }
        }
        self.words += 1;
        self.without += len;
        self.with += cost;
    }

    #[allow(clippy::cast_precision_loss)]
    fn report(mut self) -> Report {
        self.times_us.sort_by(f64::total_cmp);
        let predictions = self.times_us.len();
        let mean_us = if predictions == 0 {
            0.0
        } else {
            self.times_us.iter().sum::<f64>() / predictions as f64
        };
        Report {
            suggestions: self.options.suggestions,
            learn: self.options.learn,
            words: self.words,
            keystrokes_without: self.without,
            keystrokes_with: self.with,
            keystroke_savings: 1.0 - ratio(self.with, self.without).unwrap_or(1.0),
            next_word_top1: ratio(self.top1, self.words).unwrap_or(0.0),
            next_word_top_n: ratio(self.top_n, self.words).unwrap_or(0.0),
            predictions: predictions as u64,
            mean_us,
            p50_us: percentile(&self.times_us, 0.50),
            p99_us: percentile(&self.times_us, 0.99),
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn ratio(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| part as f64 / whole as f64)
}

/// The value at quantile `q` of `sorted`, or 0 if it is empty.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::tiny_model;

    fn options(suggestions: usize) -> EvalOptions {
        EvalOptions {
            suggestions,
            learn: false,
            max_words: None,
        }
    }

    #[test]
    fn counts_keystrokes() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        // "London" is the top suggestion after "I live in ", so it costs 1
        // keystroke instead of 6.
        let alone = evaluate(&engine, "I live in", &options(1));
        let report = evaluate(&engine, "I live in London", &options(1));
        assert_eq!(report.words, 4);
        assert_eq!(report.keystrokes_without, alone.keystrokes_without + 6);
        assert_eq!(report.keystrokes_with, alone.keystrokes_with + 1);
        assert!(report.keystroke_savings > 0.0);
        assert!(report.next_word_top1 >= 0.25);
        assert!(report.predictions >= 4);

        // A word the model does not know is typed in full.
        let report = evaluate(&engine, "Zzyzx", &options(5));
        assert_eq!((report.keystrokes_without, report.keystrokes_with), (5, 5));
        assert!(report.keystroke_savings.abs() < f64::EPSILON);
        assert_eq!(report.predictions, 5);
    }

    #[test]
    fn matches_ignoring_case() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        let lower = evaluate(&engine, "i live in london", &options(1));
        let upper = evaluate(&engine, "I LIVE IN LONDON", &options(1));
        assert_eq!(lower.keystrokes_with, upper.keystrokes_with);
    }

    #[test]
    fn learns_when_asked() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        let text = "Zanzibar is nice. Zanzibar is nice.\nZanzibar is nice.";
        let fixed = evaluate(&engine, text, &options(3));
        let learning = evaluate(
            &engine,
            text,
            &EvalOptions {
                learn: true,
                ..options(3)
            },
        );
        assert_eq!(fixed.words, 9);
        assert!(learning.keystrokes_with < fixed.keystrokes_with);
        assert!(learning.learn);
    }

    #[test]
    fn stops_after_max_words() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        let report = evaluate(
            &engine,
            "the cat sat\non the mat",
            &EvalOptions {
                max_words: Some(4),
                ..options(5)
            },
        );
        assert_eq!(report.words, 4);
    }

    #[test]
    fn empty_text_reports_zeros() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        let report = evaluate(&engine, "", &options(5));
        assert_eq!(report.words, 0);
        assert!(report.keystroke_savings.abs() < f64::EPSILON);
        assert!(report.mean_us.abs() < f64::EPSILON);
    }

    #[test]
    fn computes_percentiles() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.5) - 3.0).abs() < f64::EPSILON);
        assert!((percentile(&sorted, 0.99) - 5.0).abs() < f64::EPSILON);
        assert!(percentile(&[], 0.5).abs() < f64::EPSILON);
    }
}
