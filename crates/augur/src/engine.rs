use std::borrow::Cow;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use augur_core::{
    Candidate, Combiner, Context, Error, LanguageId, LanguagePack, LearnInput, Predictor,
    PredictorOutput, Result, SourceMask, Suggestion, Token, is_marker,
};
use augur_lang_en::EnglishPack;
use augur_lm::Model;
use augur_predictors::{NgramPredictor, RecencyPredictor};
use augur_user::UserPredictor;

use crate::{EngineOptions, LinearCombiner};

/// How much of the most recently learned text is kept as history for the
/// next [`Engine::learn`] call.
const LEARN_TAIL_BYTES: usize = 256;

/// A configured prediction engine.
///
/// `Engine` is `Send + Sync`; share it between threads with an `Arc`.
pub struct Engine {
    language: Box<dyn LanguagePack>,
    predictors: Vec<Box<dyn Predictor>>,
    combiner: Box<dyn Combiner>,
    learning: AtomicBool,
    /// The end of the text learned so far, continued by the next `learn`.
    learn_tail: Mutex<String>,
}

impl Engine {
    /// Starts building an engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Creates an engine for a language from an n-gram model file with the
    /// default [`EngineOptions`]: the user model learns in memory only.
    ///
    /// # Errors
    ///
    /// As [`Engine::with_options`].
    pub fn from_model(language: &str, model: impl AsRef<Path>) -> Result<Self> {
        Self::with_options(language, model, &EngineOptions::default())
    }

    /// Creates an engine for a language from an n-gram model file, with the
    /// built-in language pack for `language` (a BCP-47 tag; only English,
    /// `en` and its regional variants, is built in so far), a
    /// [`UserPredictor`], an [`NgramPredictor`] and a [`RecencyPredictor`]
    /// (in this order) and a [`LinearCombiner`] with the weights of
    /// `options`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidLanguageTag`] or [`Error::UnsupportedLanguage`] for
    /// the language; [`Error::Io`] if the model or user store cannot be
    /// read; [`Error::InvalidModel`] if the model is invalid or its metadata
    /// names a different language; [`Error::UserStore`] if the user store
    /// is invalid, for another language or already open.
    pub fn with_options(
        language: &str,
        model: impl AsRef<Path>,
        options: &EngineOptions,
    ) -> Result<Self> {
        let id = LanguageId::new(language)?;
        let primary = id
            .as_str()
            .split('-')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let builder = match primary.as_str() {
            "en" => Engine::builder().language(EnglishPack::new()),
            _ => return Err(Error::UnsupportedLanguage(id.to_string())),
        };
        let model = Model::open(model).map_err(model_error)?;
        if let Some(trained) = model.meta_value("language")
            && !trained.eq_ignore_ascii_case(&primary)
        {
            return Err(Error::InvalidModel(format!(
                "the model is for language {trained:?}, not {:?}",
                id.as_str()
            )));
        }
        let user = match &options.user_store {
            Some(path) => UserPredictor::open(path, &primary, options.user)?,
            None => UserPredictor::in_memory(options.user),
        };
        let [ngram_weight, user_weight, recency_weight] = options.weights;
        let engine = builder
            .predictor(user)
            .predictor(NgramPredictor::new(model))
            .predictor(RecencyPredictor::new(options.recency))
            .combiner(LinearCombiner::with_weights([
                user_weight,
                ngram_weight,
                recency_weight,
            ]))
            .build()?;
        engine.set_learning(options.learning);
        Ok(engine)
    }

    /// The language this engine predicts for.
    pub fn language(&self) -> &LanguageId {
        self.language.id()
    }

    /// The language pack, e.g. to tokenize text the same way the engine
    /// does.
    pub fn language_pack(&self) -> &dyn LanguagePack {
        &*self.language
    }

    /// The names of the predictors, in the order of the bits of
    /// [`Suggestion::sources`].
    pub fn predictor_names(&self) -> Vec<&str> {
        self.predictors.iter().map(|p| p.name()).collect()
    }

    /// Returns up to `max` suggestions for the text around the cursor, most
    /// likely first.
    pub fn predict(&self, context: Context<'_>, max: usize) -> Vec<Suggestion> {
        if max == 0 {
            return Vec::new();
        }
        let tokenized = self.language.tokenizer().tokenize_context(context);

        let candidates: Vec<Vec<Candidate>> = self
            .predictors
            .iter()
            .map(|predictor| {
                let mut out = Vec::with_capacity(max);
                predictor.predict(&tokenized, max, &mut out);
                out
            })
            .collect();
        let outputs: Vec<PredictorOutput<'_>> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidates)| PredictorOutput { index, candidates })
            .collect();

        let mut suggestions = self.combiner.combine(&outputs, max);
        suggestions.truncate(max);
        for suggestion in &mut suggestions {
            let text = {
                let display = self
                    .predictors
                    .iter()
                    .find_map(|p| p.display_form(&suggestion.text))
                    .unwrap_or(Cow::Borrowed(&suggestion.text));
                self.language.restore_case(&display, &tokenized)
            };
            suggestion.text = text;
        }
        suggestions
    }

    /// Feeds committed text to every predictor so it can learn from it.
    ///
    /// Text may be passed in pieces of any size, down to single words: each
    /// call continues the sentence where the previous one stopped, unless
    /// that sentence ended or [`Engine::reset_learning_context`] was
    /// called. A space is assumed between the pieces when the new text
    /// starts with a letter or digit. Does nothing while learning is off.
    pub fn learn(&self, text: &str) {
        if !self.is_learning() {
            return;
        }
        let mut tail = self.lock_tail();
        let joined = join(&tail, text);
        let boundary = joined.len() - text.len();
        let spans = self.language.tokenizer().tokenize_spans(&joined);
        let first_new = spans
            .iter()
            .position(|s| s.span.start >= boundary)
            .unwrap_or(spans.len());
        let (old, new) = spans.split_at(first_new);

        // The learned part of the sentence that the new text continues.
        let context: Vec<Token> = match new.first() {
            Some(span) if span.token != Token::SentenceStart => {
                let start = old
                    .iter()
                    .rposition(|s| s.token == Token::SentenceStart)
                    .unwrap_or(0);
                old[start..].iter().map(|s| s.token.clone()).collect()
            }
            _ => Vec::new(),
        };
        if !new.is_empty() {
            let input = LearnInput {
                context: &context,
                tokens: new,
                text: &joined,
            };
            for predictor in &self.predictors {
                predictor.learn(&input);
            }
        }
        tail_of(&joined).clone_into(&mut tail);
    }

    /// Makes the next [`Engine::learn`] call start a new text, e.g. when
    /// the user moves to another input field.
    pub fn reset_learning_context(&self) {
        self.lock_tail().clear();
    }

    /// Turns learning on or off, e.g. off while the user types into a
    /// password field. Turning it off also resets the learning context.
    pub fn set_learning(&self, on: bool) {
        self.learning.store(on, Ordering::Relaxed);
        if !on {
            self.reset_learning_context();
        }
    }

    /// Whether [`Engine::learn`] learns.
    pub fn is_learning(&self) -> bool {
        self.learning.load(Ordering::Relaxed)
    }

    /// Forgets the words of `text` (normally a single word) in every
    /// predictor, including saved data, and resets the learning context.
    ///
    /// # Errors
    ///
    /// The first error of a predictor that could not save the change; the
    /// remaining predictors still forget.
    pub fn forget(&self, text: &str) -> Result<()> {
        self.reset_learning_context();
        let mut result = Ok(());
        for token in self.language.tokenizer().tokenize(text) {
            if let Token::Word(word) = token
                && !is_marker(&word)
            {
                for predictor in &self.predictors {
                    result = result.and(predictor.forget(&word));
                }
            }
        }
        result
    }

    /// Forgets everything learned, including saved data, and resets the
    /// learning context.
    ///
    /// # Errors
    ///
    /// The first error of a predictor that could not save the change; the
    /// remaining predictors still clear.
    pub fn clear(&self) -> Result<()> {
        self.reset_learning_context();
        self.predictors
            .iter()
            .map(|p| p.clear())
            .fold(Ok(()), Result::and)
    }

    /// Saves learned data that is not saved yet. The engine also saves
    /// periodically while learning, and when it is dropped (ignoring
    /// errors then).
    ///
    /// # Errors
    ///
    /// The first error of a predictor that could not save; the remaining
    /// predictors still save.
    pub fn flush(&self) -> Result<()> {
        self.predictors
            .iter()
            .map(|p| p.flush())
            .fold(Ok(()), Result::and)
    }

    fn lock_tail(&self) -> std::sync::MutexGuard<'_, String> {
        self.learn_tail
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// `tail` followed by `text`, with a space between them if `text` starts a
/// new word right after a previous one.
fn join(tail: &str, text: &str) -> String {
    let separate = !tail.is_empty()
        && !tail.ends_with(char::is_whitespace)
        && text.starts_with(char::is_alphanumeric);
    let mut joined = String::with_capacity(tail.len() + 1 + text.len());
    joined.push_str(tail);
    if separate {
        joined.push(' ');
    }
    joined.push_str(text);
    joined
}

/// The last [`LEARN_TAIL_BYTES`] or so of `text`, starting at a word.
fn tail_of(text: &str) -> &str {
    if text.len() <= LEARN_TAIL_BYTES {
        return text;
    }
    let mut cut = text.len() - LEARN_TAIL_BYTES;
    while !text.is_char_boundary(cut) {
        cut += 1;
    }
    let rest = &text[cut..];
    match rest.find(char::is_whitespace) {
        Some(space) => rest[space..].trim_start(),
        None => "",
    }
}

fn model_error(error: augur_lm::Error) -> Error {
    match error {
        augur_lm::Error::Io(e) => Error::Io(e),
        other => Error::InvalidModel(other.to_string()),
    }
}

/// Builder for [`Engine`].
#[derive(Default)]
pub struct EngineBuilder {
    language: Option<Box<dyn LanguagePack>>,
    predictors: Vec<Box<dyn Predictor>>,
    combiner: Option<Box<dyn Combiner>>,
}

impl EngineBuilder {
    /// Sets the language pack (required).
    #[must_use]
    pub fn language(mut self, language: impl LanguagePack + 'static) -> Self {
        self.language = Some(Box::new(language));
        self
    }

    /// Adds a predictor (at least one is required).
    #[must_use]
    pub fn predictor(mut self, predictor: impl Predictor + 'static) -> Self {
        self.predictors.push(Box::new(predictor));
        self
    }

    /// Sets the combiner (required).
    #[must_use]
    pub fn combiner(mut self, combiner: impl Combiner + 'static) -> Self {
        self.combiner = Some(Box::new(combiner));
        self
    }

    /// Builds the engine.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the language pack or combiner is missing,
    /// or if there are no predictors or more than [`SourceMask::CAPACITY`].
    pub fn build(self) -> Result<Engine> {
        let language = self
            .language
            .ok_or_else(|| Error::Config("no language pack set".into()))?;
        let combiner = self
            .combiner
            .ok_or_else(|| Error::Config("no combiner set".into()))?;
        if self.predictors.is_empty() {
            return Err(Error::Config("no predictors added".into()));
        }
        if self.predictors.len() > SourceMask::CAPACITY {
            return Err(Error::Config(format!(
                "at most {} predictors are supported",
                SourceMask::CAPACITY
            )));
        }
        Ok(Engine {
            language,
            predictors: self.predictors,
            combiner,
            learning: AtomicBool::new(true),
            learn_tail: Mutex::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::{Arc, Mutex};

    use augur_core::{Token, TokenSpan, TokenizedContext, Tokenizer};

    use super::*;

    /// Whitespace tokenizer that lowercases words.
    struct TestLanguage(LanguageId);

    impl Tokenizer for TestLanguage {
        fn tokenize_context(&self, context: Context<'_>) -> TokenizedContext {
            let mut words: Vec<&str> = context.past.split_whitespace().collect();
            let prefix = if context.past.ends_with(char::is_whitespace) {
                ""
            } else {
                words.pop().unwrap_or("")
            };
            TokenizedContext {
                history: words
                    .into_iter()
                    .map(|w| Token::Word(w.to_lowercase()))
                    .collect(),
                prefix: prefix.to_lowercase(),
                next_word: context.future.split_whitespace().next().map(str::to_owned),
            }
        }

        fn tokenize_spans(&self, text: &str) -> Vec<TokenSpan> {
            text.split_whitespace()
                .map(|w| {
                    let start = w.as_ptr() as usize - text.as_ptr() as usize;
                    TokenSpan {
                        token: Token::Word(w.to_lowercase()),
                        span: start..start + w.len(),
                    }
                })
                .collect()
        }
    }

    impl LanguagePack for TestLanguage {
        fn id(&self) -> &LanguageId {
            &self.0
        }
        fn tokenizer(&self) -> &dyn Tokenizer {
            self
        }
        fn normalize<'a>(&self, word: &'a str) -> Cow<'a, str> {
            Cow::Owned(word.to_lowercase())
        }
        fn restore_case(&self, word: &str, context: &TokenizedContext) -> String {
            if context.history.is_empty() {
                let mut chars = word.chars();
                chars
                    .next()
                    .map(|c| c.to_uppercase().chain(chars).collect())
                    .unwrap_or_default()
            } else {
                word.to_owned()
            }
        }
    }

    /// Predicts fixed words that match the prefix and records learned tokens.
    struct FixedPredictor {
        words: Vec<(&'static str, f32)>,
        learned: Arc<Mutex<Vec<Token>>>,
    }

    impl FixedPredictor {
        fn new(words: Vec<(&'static str, f32)>) -> Self {
            Self {
                words,
                learned: Arc::default(),
            }
        }
    }

    impl Predictor for FixedPredictor {
        fn name(&self) -> &'static str {
            "fixed"
        }
        fn predict(&self, context: &TokenizedContext, max: usize, out: &mut Vec<Candidate>) {
            out.extend(
                self.words
                    .iter()
                    .filter(|(w, _)| w.starts_with(&context.prefix))
                    .take(max)
                    .map(|&(word, prob)| Candidate {
                        word: word.to_owned(),
                        prob,
                    }),
            );
        }
        fn learn(&self, input: &LearnInput<'_>) {
            let mut learned = self.learned.lock().unwrap();
            learned.extend(input.tokens.iter().map(|s| s.token.clone()));
        }
    }

    /// Concatenates outputs in predictor order.
    struct ConcatCombiner;

    impl Combiner for ConcatCombiner {
        fn combine(&self, outputs: &[PredictorOutput<'_>], _max: usize) -> Vec<Suggestion> {
            outputs
                .iter()
                .flat_map(|output| {
                    output.candidates.iter().map(|c| {
                        let mut sources = SourceMask::default();
                        sources.insert(output.index);
                        Suggestion {
                            text: c.word.clone(),
                            prob: c.prob,
                            sources,
                        }
                    })
                })
                .collect()
        }
    }

    fn engine() -> Engine {
        Engine::builder()
            .language(TestLanguage(LanguageId::new("en").unwrap()))
            .predictor(FixedPredictor::new(vec![("hello", 0.6), ("help", 0.3)]))
            .predictor(FixedPredictor::new(vec![("world", 0.9)]))
            .combiner(ConcatCombiner)
            .build()
            .unwrap()
    }

    #[test]
    fn engine_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    }

    #[test]
    fn predicts_through_pipeline_and_restores_case() {
        let engine = engine();
        let suggestions = engine.predict(Context::past("hel"), 5);
        let texts: Vec<&str> = suggestions.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["Hello", "Help"]);
        assert!(suggestions[0].sources.contains(0));

        let suggestions = engine.predict(Context::past("say "), 5);
        let texts: Vec<&str> = suggestions.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["hello", "help", "world"]);
        assert!(suggestions[2].sources.contains(1));
    }

    #[test]
    fn truncates_to_max() {
        assert_eq!(engine().predict(Context::past("say "), 2).len(), 2);
        assert!(engine().predict(Context::past("say "), 0).is_empty());
    }

    #[test]
    fn learn_reaches_predictors() {
        let predictor = FixedPredictor::new(vec![]);
        let learned = Arc::clone(&predictor.learned);
        let engine = Engine::builder()
            .language(TestLanguage(LanguageId::new("en").unwrap()))
            .predictor(predictor)
            .combiner(ConcatCombiner)
            .build()
            .unwrap();
        engine.learn("Good Morning");
        assert_eq!(
            *learned.lock().unwrap(),
            [Token::Word("good".into()), Token::Word("morning".into())]
        );

        engine.set_learning(false);
        assert!(!engine.is_learning());
        engine.learn("secret");
        engine.set_learning(true);
        engine.learn("again");
        assert_eq!(learned.lock().unwrap().len(), 3);
    }

    #[test]
    fn joins_learned_pieces() {
        assert_eq!(join("", "hello"), "hello");
        assert_eq!(join("hello", "world"), "hello world");
        assert_eq!(join("hello", ", world"), "hello, world");
        assert_eq!(join("hello ", "world"), "hello world");
        assert_eq!(join("Hi.", "There"), "Hi. There");

        let long = "word ".repeat(100);
        let tail = tail_of(&long);
        assert!(tail.len() <= LEARN_TAIL_BYTES && tail.starts_with("word"));
        assert_eq!(tail_of("short text"), "short text");
        assert_eq!(tail_of(&"x".repeat(300)), "");
    }

    /// Words of the user model's (predictor 0) suggestions.
    fn from_user(suggestions: &[Suggestion]) -> Vec<&str> {
        suggestions
            .iter()
            .filter(|s| s.sources.contains(0))
            .map(|s| s.text.as_str())
            .collect()
    }

    #[test]
    fn learns_from_the_user() {
        let engine = Engine::from_model("en", tiny_model()).unwrap();
        assert!(from_user(&engine.predict(Context::past("I like "), 10)).is_empty());

        // Learned word by word, across calls, with the user's casing.
        for word in ["I", "like", "Zanzibar", "a", "lot."] {
            engine.learn(word);
        }
        let out = engine.predict(Context::past("I like "), 10);
        assert_eq!(texts(&out)[0], "Zanzibar");
        assert_eq!(from_user(&out)[0], "Zanzibar");
        let out = engine.predict(Context::past("I like z"), 10);
        assert_eq!(texts(&out), ["Zanzibar"]);
        assert!((out[0].prob - 1.0).abs() < 1e-6);
        // Recency (predictor 2) suggests it anywhere.
        let out = engine.predict(Context::past("Then z"), 10);
        assert!(out[0].sources.contains(2));

        // The sentence ended: the next piece starts a new one.
        engine.learn("Zanzibar");
        let out = engine.predict(Context::past("Hello. "), 10);
        assert!(from_user(&out).contains(&"Zanzibar"));

        engine.forget("ZANZIBAR").unwrap();
        let out = engine.predict(Context::past("I like z"), 10);
        assert!(out.is_empty(), "{out:?}");

        engine.learn("I like tea.");
        engine.clear().unwrap();
        assert!(from_user(&engine.predict(Context::past("I like "), 10)).is_empty());
        engine.flush().unwrap();
    }

    #[test]
    fn saves_the_user_model() {
        let dir = std::env::temp_dir().join(format!("augur-engine-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let store = dir.join("user.redb");
        let _ = std::fs::remove_file(&store);
        let options = EngineOptions::new().user_store(&store);
        {
            let engine = Engine::with_options("en", tiny_model(), &options).unwrap();
            engine.learn("I like Zanzibar.");
            // The store is locked while the engine is open.
            assert!(matches!(
                Engine::with_options("en", tiny_model(), &options),
                Err(Error::UserStore(_))
            ));
        }
        let engine = Engine::with_options("en-US", tiny_model(), &options).unwrap();
        let out = engine.predict(Context::past("I like "), 3);
        assert_eq!(texts(&out)[0], "Zanzibar");
        drop(engine);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn options_select_weights_and_learning() {
        assert!(EngineOptions::new().weights(1.0, -1.0, 0.0).is_err());
        assert!(EngineOptions::new().weights(0.0, 0.0, 0.0).is_err());
        assert!(EngineOptions::new().weights(f32::NAN, 1.0, 0.0).is_err());

        // Without the n-gram model, only learned words are suggested.
        let options = EngineOptions::new()
            .weights(0.0, 1.0, 0.0)
            .unwrap()
            .learning(false);
        let engine = Engine::with_options("en", tiny_model(), &options).unwrap();
        assert!(!engine.is_learning());
        engine.learn("I like Zanzibar.");
        assert!(engine.predict(Context::past("I like "), 3).is_empty());
        engine.set_learning(true);
        engine.learn("I like Zanzibar.");
        let out = engine.predict(Context::past("I like "), 3);
        assert_eq!(texts(&out), ["Zanzibar", "I", "like"]);
        assert!(out.iter().all(|s| s.sources == SourceMask(1)));
    }

    fn tiny_model() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny.bin")
    }

    fn texts(suggestions: &[Suggestion]) -> Vec<&str> {
        suggestions.iter().map(|s| s.text.as_str()).collect()
    }

    #[test]
    fn predicts_from_a_model() {
        let engine = Engine::from_model("en-GB", tiny_model()).unwrap();
        assert_eq!(engine.language().as_str(), "en");
        assert_eq!(engine.language_pack().id().as_str(), "en");
        assert_eq!(engine.predictor_names(), ["user", "ngram", "recency"]);

        // Casing from the model's table, then sentence-start capitals.
        let out = engine.predict(Context::past("I live in "), 2);
        assert_eq!(texts(&out), ["London", "Paris"]);
        let out = engine.predict(Context::past("I live in l"), 3);
        assert_eq!(texts(&out), ["London", "live", "love"]);
        let out = engine.predict(Context::past("I live in Lon"), 3);
        assert_eq!(texts(&out), ["London"]);
        assert!((out[0].prob - 1.0).abs() < 1e-6);
        let out = engine.predict(Context::past("We went there. "), 1);
        assert_eq!(texts(&out), ["The"]);
        let out = engine.predict(Context::past("the cat "), 3);
        assert!(texts(&out).contains(&"sat"), "{out:?}");
        assert!(out.windows(2).all(|w| w[0].prob >= w[1].prob));
        assert!(out.iter().all(|s| s.sources.contains(1)));

        // Probabilities are conditioned on the prefix.
        let out = engine.predict(Context::past("the "), 100);
        let sum: f32 = out.iter().map(|s| s.prob).sum();
        assert!((sum - 1.0).abs() < 1e-4, "{sum}");

        // The word after the cursor is not suggested again.
        let out = engine.predict(Context::new("I live in ", " London"), 1);
        assert_eq!(texts(&out), ["Paris"]);
    }

    #[test]
    fn from_model_reports_errors() {
        assert!(matches!(
            Engine::from_model("de", tiny_model()),
            Err(Error::UnsupportedLanguage(tag)) if tag == "de"
        ));
        assert!(matches!(
            Engine::from_model("en_US", tiny_model()),
            Err(Error::InvalidLanguageTag(_))
        ));
        assert!(matches!(
            Engine::from_model("en", tiny_model().with_extension("missing")),
            Err(Error::Io(_))
        ));
        let arpa = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus.txt");
        assert!(matches!(
            Engine::from_model("en", arpa),
            Err(Error::InvalidModel(_))
        ));
    }

    #[test]
    fn build_rejects_incomplete_configuration() {
        let err = Engine::builder()
            .combiner(ConcatCombiner)
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, Error::Config(_)));

        let err = Engine::builder()
            .language(TestLanguage(LanguageId::new("en").unwrap()))
            .combiner(ConcatCombiner)
            .build()
            .err()
            .unwrap();
        assert!(matches!(err, Error::Config(_)));
    }
}
