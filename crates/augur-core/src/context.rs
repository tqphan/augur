use std::ops::Range;

/// Raw text surrounding the cursor, as supplied by the host application.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Context<'a> {
    /// Text before the cursor.
    pub past: &'a str,
    /// Text after the cursor.
    pub future: &'a str,
}

impl<'a> Context<'a> {
    /// Creates a context with text before and after the cursor.
    pub fn new(past: &'a str, future: &'a str) -> Self {
        Self { past, future }
    }

    /// Creates a context with only text before the cursor.
    pub fn past(past: &'a str) -> Self {
        Self { past, future: "" }
    }
}

/// A normalized token as seen by predictors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    /// Start of a sentence (`<s>` in n-gram terminology).
    SentenceStart,
    /// A normalized word.
    Word(String),
}

/// A [`Token`] together with the byte range of the input text it was read
/// from. A [`Token::SentenceStart`] has an empty range at the start of its
/// sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    /// The normalized token.
    pub token: Token,
    /// Byte range of the token in the tokenized text.
    pub span: Range<usize>,
}

/// A [`Context`] after tokenization and normalization.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenizedContext {
    /// Completed tokens before the cursor, oldest first.
    pub history: Vec<Token>,
    /// The normalized, partially typed word at the cursor (empty if the
    /// cursor follows a word boundary).
    pub prefix: String,
    /// The normalized word immediately after the cursor, if any. Used to
    /// avoid suggesting a word that is already there.
    pub next_word: Option<String>,
}
