use std::borrow::Cow;
use std::path::Path;

use augur_core::{
    Candidate, Combiner, Context, Error, LanguageId, LanguagePack, Predictor, PredictorOutput,
    Result, SourceMask, Suggestion,
};
use augur_lang_en::EnglishPack;
use augur_lm::Model;
use augur_predictors::NgramPredictor;

use crate::LinearCombiner;

/// A configured prediction engine.
///
/// `Engine` is `Send + Sync`; share it between threads with an `Arc`.
pub struct Engine {
    language: Box<dyn LanguagePack>,
    predictors: Vec<Box<dyn Predictor>>,
    combiner: Box<dyn Combiner>,
}

impl Engine {
    /// Starts building an engine.
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    /// Creates an engine for a language from an n-gram model file, with the
    /// built-in language pack for `language` (a BCP-47 tag; only English,
    /// `en` and its regional variants, is built in so far), an
    /// [`NgramPredictor`] and a [`LinearCombiner`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidLanguageTag`] or [`Error::UnsupportedLanguage`] for
    /// the language; [`Error::Io`] if the model cannot be read;
    /// [`Error::InvalidModel`] if it is invalid or its metadata names a
    /// different language.
    pub fn from_model(language: &str, model: impl AsRef<Path>) -> Result<Self> {
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
        builder
            .predictor(NgramPredictor::new(model))
            .combiner(LinearCombiner::new())
            .build()
    }

    /// The language this engine predicts for.
    pub fn language(&self) -> &LanguageId {
        self.language.id()
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
    pub fn learn(&self, text: &str) {
        let tokens = self.language.tokenizer().tokenize(text);
        for predictor in &self.predictors {
            predictor.learn(&tokens);
        }
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
        fn learn(&self, tokens: &[Token]) {
            self.learned.lock().unwrap().extend_from_slice(tokens);
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
        assert!(out.iter().all(|s| s.sources.contains(0)));

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
