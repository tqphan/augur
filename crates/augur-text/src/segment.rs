use std::collections::HashSet;
use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

use crate::normalize;

/// Language-specific adjustments to Unicode (UAX #29) sentence boundaries.
#[derive(Debug, Clone, Default)]
pub struct SegmentRules {
    abbreviations: HashSet<String>,
    initials: bool,
}

impl SegmentRules {
    /// Rules that follow UAX #29 unchanged.
    pub fn new() -> Self {
        Self::default()
    }

    /// Words that, followed by a period, do not end a sentence (e.g. `mr`,
    /// `e.g`). Give them normalized and without the final period.
    #[must_use]
    pub fn abbreviations<I, S>(mut self, abbreviations: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.abbreviations
            .extend(abbreviations.into_iter().map(Into::into));
        self
    }

    /// Whether a single uppercase letter followed by a period is an initial
    /// (as in `J. Smith`) rather than the end of a sentence.
    #[must_use]
    pub fn initials(mut self, initials: bool) -> Self {
        self.initials = initials;
        self
    }
}

/// Splits text into sentences and words.
#[derive(Debug, Clone, Default)]
pub struct Segmenter {
    rules: SegmentRules,
}

impl Segmenter {
    /// Creates a segmenter with the given rules.
    pub fn new(rules: SegmentRules) -> Self {
        Self { rules }
    }

    /// Byte ranges of the sentences in `text`, in order. Together they
    /// cover the whole text.
    pub fn sentences(&self, text: &str) -> Vec<Range<usize>> {
        let mut sentences: Vec<Range<usize>> = Vec::new();
        for (start, sentence) in text.split_sentence_bound_indices() {
            let end = start + sentence.len();
            match sentences.last_mut() {
                Some(previous) if self.continues_after(&text[previous.clone()]) => {
                    previous.end = end;
                }
                _ => sentences.push(start..end),
            }
        }
        sentences
    }

    /// Whether a sentence starting right after `text` would be a new
    /// sentence, i.e. `text` ends with a complete sentence.
    pub fn ends_sentence(&self, text: &str) -> bool {
        if text.trim().is_empty() {
            return true;
        }
        // Probe with a capitalized word: UAX #29 then breaks after a
        // sentence terminator, and the abbreviation rules can undo it.
        let probe = format!("{text} A");
        self.sentences(&probe)
            .last()
            .is_some_and(|last| last.start >= text.len())
    }

    /// Whether a UAX #29 sentence break after `sentence` should be ignored
    /// because the final period belongs to an abbreviation or initial.
    fn continues_after(&self, sentence: &str) -> bool {
        let trimmed = sentence.trim_end();
        let Some(before) = trimmed.strip_suffix('.') else {
            return false;
        };
        if sentence[trimmed.len()..].contains(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']) {
            return false;
        }
        let Some((start, last)) = words(before).last() else {
            return false;
        };
        if start + last.len() != before.len() {
            return false;
        }
        let mut chars = last.chars();
        let initial = matches!((chars.next(), chars.next()), (Some(c), None) if c.is_uppercase());
        (self.rules.initials && initial)
            || self.rules.abbreviations.contains(normalize(last).as_ref())
    }
}

/// Word segments of `text` with their byte offsets: UAX #29 word segments
/// that contain at least one letter or digit.
pub fn words(text: &str) -> impl DoubleEndedIterator<Item = (usize, &str)> {
    text.split_word_bound_indices()
        .filter(|(_, segment)| segment.chars().any(char::is_alphanumeric))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn english() -> Segmenter {
        Segmenter::new(
            SegmentRules::new()
                .abbreviations(["mr", "u.s", "e.g"])
                .initials(true),
        )
    }

    fn split<'a>(segmenter: &Segmenter, text: &'a str) -> Vec<&'a str> {
        segmenter
            .sentences(text)
            .into_iter()
            .map(|range| text[range].trim())
            .collect()
    }

    #[test]
    fn words_keep_contractions_and_split_hyphens() {
        let found: Vec<&str> = words("Don't stop, well-known U.S. 3.5 caf\u{e9}!")
            .map(|(_, w)| w)
            .collect();
        assert_eq!(
            found,
            ["Don't", "stop", "well", "known", "U.S", "3.5", "caf\u{e9}"]
        );
    }

    #[test]
    fn word_offsets_point_into_text() {
        let text = "a  bc";
        let found: Vec<(usize, &str)> = words(text).collect();
        assert_eq!(found, [(0, "a"), (3, "bc")]);
    }

    #[test]
    fn splits_sentences_but_not_after_abbreviations() {
        let text = "Mr. Smith went to the U.S. Army. He left! Did J. Doe stay? Yes.";
        assert_eq!(
            split(&english(), text),
            [
                "Mr. Smith went to the U.S. Army.",
                "He left!",
                "Did J. Doe stay?",
                "Yes."
            ]
        );
        assert_eq!(split(&Segmenter::default(), "Mr. Smith").len(), 2);
    }

    #[test]
    fn newline_always_ends_sentence() {
        assert_eq!(
            split(&english(), "Title\nBody text"),
            ["Title", "Body text"]
        );
        assert_eq!(split(&english(), "see Mr.\nNext"), ["see Mr.", "Next"]);
    }

    #[test]
    fn detects_complete_sentences() {
        let segmenter = english();
        assert!(segmenter.ends_sentence(""));
        assert!(segmenter.ends_sentence("It rained."));
        assert!(segmenter.ends_sentence("It rained. "));
        assert!(segmenter.ends_sentence("Really?\n"));
        assert!(!segmenter.ends_sentence("It rained"));
        assert!(!segmenter.ends_sentence("I met Mr."));
        assert!(!segmenter.ends_sentence("It rained, "));
    }
}
