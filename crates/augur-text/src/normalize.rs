use std::borrow::Cow;

use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

/// Characters treated as an apostrophe and mapped to `'` (U+0027).
const APOSTROPHES: [char; 4] = ['\u{2019}', '\u{2018}', '\u{02BC}', '\u{FF07}'];

/// Canonicalizes a word without changing its case: NFC normalization and
/// apostrophe unification.
pub fn canonicalize(word: &str) -> Cow<'_, str> {
    let word = if is_nfc_quick(word.chars()) == IsNormalized::Yes {
        Cow::Borrowed(word)
    } else {
        Cow::Owned(word.nfc().collect())
    };
    if word.contains(APOSTROPHES) {
        Cow::Owned(word.replace(APOSTROPHES, "'"))
    } else {
        word
    }
}

/// Lowercases a word, borrowing it if it has no uppercase characters.
pub fn fold_case(word: &str) -> Cow<'_, str> {
    if word.chars().any(char::is_uppercase) {
        Cow::Owned(word.to_lowercase())
    } else {
        Cow::Borrowed(word)
    }
}

/// Normalizes a word for lookup: [`canonicalize`] followed by
/// [`fold_case`].
pub fn normalize(word: &str) -> Cow<'_, str> {
    match canonicalize(word) {
        Cow::Borrowed(word) => fold_case(word),
        Cow::Owned(word) => Cow::Owned(fold_case(&word).into_owned()),
    }
}

/// Whether a word segment is a number (starts with a numeric character),
/// e.g. `1984`, `3.5`, `1,000` or `21st`.
pub fn is_number(word: &str) -> bool {
    word.chars().next().is_some_and(char::is_numeric)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_case_apostrophes_and_composition() {
        assert_eq!(normalize("Don\u{2019}t"), "don't");
        assert_eq!(normalize("Cafe\u{301}"), "caf\u{e9}");
        assert_eq!(normalize("\u{c9}COLE"), "\u{e9}cole");
        assert!(matches!(normalize("plain"), Cow::Borrowed(_)));
    }

    #[test]
    fn canonicalize_keeps_case() {
        assert_eq!(canonicalize("O\u{2019}Brien"), "O'Brien");
        assert!(matches!(canonicalize("London"), Cow::Borrowed(_)));
    }

    #[test]
    fn detects_numbers() {
        for word in ["1984", "3.5", "1,000", "21st", "\u{0663}"] {
            assert!(is_number(word), "{word}");
        }
        for word in ["mp3", "a", ""] {
            assert!(!is_number(word), "{word}");
        }
    }
}
