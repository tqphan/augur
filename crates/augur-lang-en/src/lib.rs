//! English language pack for augur.

use std::borrow::Cow;

use augur_core::{LanguageId, LanguagePack, Token, TokenizedContext, Tokenizer};
use augur_text::{SegmentRules, TextTokenizer};

/// Abbreviations that do not end a sentence when followed by a period,
/// normalized and without the final period.
#[rustfmt::skip]
const ABBREVIATIONS: &[&str] = &[
    // Titles
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "gen", "col", "capt", "lt", "sgt",
    "gov", "sen", "rep", "pres", "hon",
    // Latin and common
    "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "ca", "c", "vol", "fig", "p", "pp", "ed",
    "eds", "est", "dept", "inc", "ltd", "co", "corp", "mt", "ft",
    // Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    // Places and dotted acronyms
    "u.s", "u.k", "u.s.a", "u.n", "d.c", "a.d", "b.c", "a.m", "p.m", "ph.d",
];

/// Contractions of the pronoun "I", which is always capitalized.
const PRONOUN_I: &[&str] = &["i", "i'm", "i'll", "i've", "i'd"];

/// The English language pack.
#[derive(Debug, Clone)]
pub struct EnglishPack {
    id: LanguageId,
    tokenizer: TextTokenizer,
}

impl EnglishPack {
    /// Creates the English pack.
    ///
    /// # Panics
    ///
    /// Never; the language tag is a valid constant.
    pub fn new() -> Self {
        Self {
            id: LanguageId::new("en").expect("valid language tag"),
            tokenizer: TextTokenizer::new(
                SegmentRules::new()
                    .abbreviations(ABBREVIATIONS.iter().copied())
                    .initials(true),
            ),
        }
    }
}

impl Default for EnglishPack {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguagePack for EnglishPack {
    fn id(&self) -> &LanguageId {
        &self.id
    }

    fn tokenizer(&self) -> &dyn Tokenizer {
        &self.tokenizer
    }

    fn normalize<'a>(&self, word: &'a str) -> Cow<'a, str> {
        augur_text::normalize(word)
    }

    fn restore_case(&self, word: &str, context: &TokenizedContext) -> String {
        if PRONOUN_I.contains(&word) {
            return capitalize(word);
        }
        match context.history.last() {
            None | Some(Token::SentenceStart) => capitalize(word),
            Some(Token::Word(_)) => word.to_owned(),
        }
    }
}

fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use augur_core::Context;

    use super::*;

    fn words(tokens: &[Token]) -> Vec<&str> {
        tokens
            .iter()
            .map(|t| match t {
                Token::SentenceStart => "<s>",
                Token::Word(w) => w.as_str(),
            })
            .collect()
    }

    #[test]
    fn tokenizes_english() {
        let pack = EnglishPack::new();
        let tokens = pack
            .tokenizer()
            .tokenize("Dr. J. Smith\u{2019}s well-known U.S. firm, e.g. Acme, opened in 1999. It\u{2019}s big!");
        assert_eq!(
            words(&tokens),
            [
                "<s>", "dr", "j", "smith's", "well", "known", "u.s", "firm", "e.g", "acme",
                "opened", "in", "<num>", "<s>", "it's", "big"
            ]
        );
    }

    #[test]
    fn restores_case() {
        let pack = EnglishPack::new();
        let tokenizer = pack.tokenizer();
        let start = tokenizer.tokenize_context(Context::past("Done. "));
        let middle = tokenizer.tokenize_context(Context::past("and "));
        assert_eq!(pack.restore_case("the", &start), "The");
        assert_eq!(pack.restore_case("the", &middle), "the");
        assert_eq!(pack.restore_case("i'm", &middle), "I'm");
    }
}
