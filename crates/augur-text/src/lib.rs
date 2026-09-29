//! Unicode-aware tokenization and normalization for augur.
//!
//! Word and sentence boundaries follow Unicode UAX #29, adjusted per
//! language with [`SegmentRules`]. Words are normalized to NFC, with
//! apostrophes unified and case folded; numbers are replaced by the
//! [`NUMBER`] class token.
//!
//! The same [`TextTokenizer`] is used when training models and when
//! predicting, so both see identical tokens.

mod normalize;
mod segment;
mod tokenizer;

pub use normalize::{canonicalize, fold_case, is_number, normalize};
pub use segment::{SegmentRules, Segmenter, words};
pub use tokenizer::{TextTokenizer, word_token};

/// Sentence start marker in n-gram models.
pub const SENTENCE_START: &str = "<s>";
/// Sentence end marker in n-gram models.
pub const SENTENCE_END: &str = "</s>";
/// Out-of-vocabulary word marker in n-gram models.
pub const UNKNOWN: &str = "<unk>";
/// Class token that replaces every number. It never appears as a
/// suggestion.
pub const NUMBER: &str = "<num>";
