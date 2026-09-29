use augur_core::{Context, Token, TokenSpan, TokenizedContext, Tokenizer};

use crate::{NUMBER, SegmentRules, Segmenter, is_number, normalize, words};

/// How much text before the cursor [`TextTokenizer::tokenize_context`]
/// looks at, in bytes.
const MAX_CONTEXT_BYTES: usize = 1024;

/// A [`Tokenizer`] built on Unicode segmentation, configured per language
/// with [`SegmentRules`].
///
/// Words are [normalized](normalize); numbers become [`NUMBER`]; each
/// sentence that contains a word starts with [`Token::SentenceStart`].
#[derive(Debug, Clone, Default)]
pub struct TextTokenizer {
    segmenter: Segmenter,
}

impl TextTokenizer {
    /// Creates a tokenizer with the given sentence rules.
    pub fn new(rules: SegmentRules) -> Self {
        Self {
            segmenter: Segmenter::new(rules),
        }
    }

    /// The underlying segmenter.
    pub fn segmenter(&self) -> &Segmenter {
        &self.segmenter
    }
}

/// The token for a raw word segment: [`NUMBER`] for numbers, otherwise the
/// normalized word.
pub fn word_token(raw: &str) -> String {
    if is_number(raw) {
        NUMBER.to_owned()
    } else {
        normalize(raw).into_owned()
    }
}

impl Tokenizer for TextTokenizer {
    fn tokenize_spans(&self, text: &str) -> Vec<TokenSpan> {
        let mut tokens = Vec::new();
        for sentence in self.segmenter.sentences(text) {
            let mut first = true;
            for (offset, word) in words(&text[sentence.clone()]) {
                let start = sentence.start + offset;
                if first {
                    tokens.push(TokenSpan {
                        token: Token::SentenceStart,
                        span: start..start,
                    });
                    first = false;
                }
                tokens.push(TokenSpan {
                    token: Token::Word(word_token(word)),
                    span: start..start + word.len(),
                });
            }
        }
        tokens
    }

    fn tokenize_context(&self, context: Context<'_>) -> TokenizedContext {
        let mut cut = context.past.len().saturating_sub(MAX_CONTEXT_BYTES);
        while !context.past.is_char_boundary(cut) {
            cut += 1;
        }
        let past = &context.past[cut..];
        let truncated = cut > 0;

        let mut spans = self.tokenize_spans(past);

        // A word touching the cursor (possibly followed by an apostrophe, as
        // in "don'") is still being typed.
        let mut prefix = String::new();
        if let Some(last) = spans.last()
            && matches!(last.token, Token::Word(_))
            && matches!(&past[last.span.end..], "" | "'" | "\u{2019}")
        {
            prefix = normalize(&past[last.span.start..]).into_owned();
            spans.pop();
        }

        // The text was cut mid-sentence: its first sentence start is not
        // real, and its first word may be partial.
        if truncated {
            if spans
                .first()
                .is_some_and(|s| s.token == Token::SentenceStart)
            {
                spans.remove(0);
            }
            if spans
                .first()
                .is_some_and(|s| matches!(s.token, Token::Word(_)))
            {
                spans.remove(0);
            }
        }

        let mut history: Vec<Token> = spans.into_iter().map(|s| s.token).collect();
        let at_start = history.is_empty() && !truncated;
        let after_sentence = prefix.is_empty()
            && history.last().is_some_and(|t| *t != Token::SentenceStart)
            && self.segmenter.ends_sentence(past);
        if at_start || after_sentence {
            history.push(Token::SentenceStart);
        }

        let future = context.future;
        let next_word = words(future)
            .next()
            .filter(|&(start, _)| future[..start].trim().is_empty())
            .map(|(_, word)| normalize(word).into_owned());

        TokenizedContext {
            history,
            prefix,
            next_word,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> TextTokenizer {
        TextTokenizer::new(SegmentRules::new().abbreviations(["mr"]))
    }

    fn word(w: &str) -> Token {
        Token::Word(w.to_owned())
    }

    const S: Token = Token::SentenceStart;

    #[test]
    fn tokenizes_sentences_numbers_and_spans() {
        let text = "In 1984 it rained. Don\u{2019}t go!";
        let spans = tokenizer().tokenize_spans(text);
        let tokens: Vec<Token> = spans.iter().map(|s| s.token.clone()).collect();
        assert_eq!(
            tokens,
            [
                S,
                word("in"),
                word(NUMBER),
                word("it"),
                word("rained"),
                S,
                word("don't"),
                word("go")
            ]
        );
        assert_eq!(&text[spans[2].span.clone()], "1984");
        assert_eq!(&text[spans[6].span.clone()], "Don\u{2019}t");
        assert_eq!(spans[5].span, 19..19);
    }

    #[test]
    fn skips_sentences_without_words() {
        assert_eq!(tokenizer().tokenize("* * *\n\nOk."), [S, word("ok")]);
        assert!(tokenizer().tokenize("").is_empty());
    }

    #[test]
    fn context_prefix_and_history() {
        let t = tokenizer();
        let ctx = t.tokenize_context(Context::past("I am Goi"));
        assert_eq!(ctx.history, [S, word("i"), word("am")]);
        assert_eq!(ctx.prefix, "goi");

        let ctx = t.tokenize_context(Context::past("I am "));
        assert_eq!(ctx.history, [S, word("i"), word("am")]);
        assert_eq!(ctx.prefix, "");

        let ctx = t.tokenize_context(Context::past("I don\u{2019}"));
        assert_eq!(ctx.history, [S, word("i")]);
        assert_eq!(ctx.prefix, "don'");
    }

    #[test]
    fn context_sentence_starts() {
        let t = tokenizer();
        assert_eq!(t.tokenize_context(Context::past("")).history, [S]);
        assert_eq!(t.tokenize_context(Context::past("Hel")).history, [S]);
        assert_eq!(
            t.tokenize_context(Context::past("Hello. ")).history,
            [S, word("hello"), S]
        );
        assert_eq!(
            t.tokenize_context(Context::past("Hello. Wo")).history,
            [S, word("hello"), S]
        );
        assert_eq!(
            t.tokenize_context(Context::past("Ask Mr. ")).history,
            [S, word("ask"), word("mr")]
        );
    }

    #[test]
    fn context_truncates_long_text() {
        let past = format!("{}word tail ", "x".repeat(2000));
        let ctx = tokenizer().tokenize_context(Context::past(&past));
        assert_eq!(ctx.history, [word("tail")]);
    }

    #[test]
    fn context_next_word() {
        let t = tokenizer();
        let ctx = t.tokenize_context(Context::new("see ", " World again"));
        assert_eq!(ctx.next_word.as_deref(), Some("world"));
        let ctx = t.tokenize_context(Context::new("see ", ", world"));
        assert_eq!(ctx.next_word, None);
    }
}
