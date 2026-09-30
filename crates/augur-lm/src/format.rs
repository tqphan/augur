//! Layout constants and little-endian helpers for the binary model format.
//!
//! All integers are little-endian. A file is a header followed by sections:
//!
//! ```text
//! header   magic "AUGURLM\0", u32 version, u32 section count S,
//!          S × { [u8; 4] tag, u32 crc32, u64 offset, u64 length },
//!          u32 crc32 of all preceding header bytes
//! "meta"   u32 count, count × { u32 len, key, u32 len, value }   (optional)
//! "info"   u32 order, u32 vocabulary size V, u32 <s> id, u32 </s> id,
//!          u32 <unk> id (NO_WORD if absent), order × u64 n-gram count
//! "voca"   u32 offsets[V + 1], UTF-8 words sorted by bytes (id = index)
//! "quan"   order × { f32 prob[256], f32 backoff[256] } (log10 values)
//! "ngNN"   records of order NN (01, 02, …), sorted by word ids:
//!            order 1:        { u8 prob, u8 backoff, u32 child } × (V + 1)
//!            middle orders:  { u32 id, u8 prob, u8 backoff, u32 child } × (count + 1)
//!            highest order:  { u32 id, u8 prob } × count
//!          `child` is the first record of the next order that extends this
//!          n-gram; the extensions end at the next record's `child`, so each
//!          order but the highest has a sentinel record at the end. Order 1
//!          is indexed directly by word id.
//! "case"   u32 count, u32 word ids[count] (ascending), u32 offsets[count + 1],
//!          UTF-8 display forms                                     (optional)
//! ```
//!
//! Unknown sections are ignored, so later versions can add optional data
//! without breaking older readers.

use crate::error::{Result, invalid};

/// Magic bytes at the start of every model file.
pub(crate) const MAGIC: [u8; 8] = *b"AUGURLM\0";

/// Format version written and read by this crate.
pub(crate) const VERSION: u32 = 1;

/// Size of the fixed header before the section table.
pub(crate) const HEADER_LEN: usize = 16;

/// Size of one section table entry.
pub(crate) const SECTION_ENTRY_LEN: usize = 24;

pub(crate) const TAG_META: [u8; 4] = *b"meta";
pub(crate) const TAG_INFO: [u8; 4] = *b"info";
pub(crate) const TAG_VOCAB: [u8; 4] = *b"voca";
pub(crate) const TAG_QUANT: [u8; 4] = *b"quan";
pub(crate) const TAG_CASING: [u8; 4] = *b"case";

/// Tag of the section holding n-grams of order `n` (1-based).
pub(crate) fn ngram_tag(n: usize) -> [u8; 4] {
    let digit = |d: usize| b'0' + u8::try_from(d).expect("single digit");
    [b'n', b'g', digit(n / 10 % 10), digit(n % 10)]
}

/// Entries in each quantization codebook.
pub(crate) const CODEBOOK_LEN: usize = 256;

/// Record size of unigrams: prob, backoff, child.
pub(crate) const UNIGRAM_RECORD: usize = 6;
/// Record size of orders between the first and the highest: id, prob,
/// backoff, child.
pub(crate) const MIDDLE_RECORD: usize = 10;
/// Record size of the highest order: id, prob.
pub(crate) const HIGHEST_RECORD: usize = 5;

/// Record size of order `n` (1-based) in a model of the given order.
pub(crate) fn record_len(n: usize, order: usize) -> usize {
    if n == 1 {
        UNIGRAM_RECORD
    } else if n == order {
        HIGHEST_RECORD
    } else {
        MIDDLE_RECORD
    }
}

/// Reads a little-endian `u32` at `pos`. The caller guarantees bounds.
pub(crate) fn u32_at(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(data[pos..pos + 4].try_into().expect("4 bytes"))
}

/// Sequential little-endian reader over a byte slice.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Cursor<'a> {
    /// A cursor over `data`; `what` names it in error messages.
    pub(crate) fn new(data: &'a [u8], what: &'static str) -> Self {
        Self { data, pos: 0, what }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| invalid(format!("{} is truncated", self.what)))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?.try_into().expect("4 bytes"),
        ))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(8)?.try_into().expect("8 bytes"),
        ))
    }

    pub(crate) fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(
            self.bytes(4)?.try_into().expect("4 bytes"),
        ))
    }

    /// A `u32` length or count, as `usize`.
    pub(crate) fn len(&mut self) -> Result<usize> {
        usize::try_from(self.u32()?).map_err(|_| invalid("length does not fit in memory"))
    }

    /// Fails unless all bytes were consumed.
    pub(crate) fn finish(&self) -> Result<()> {
        if self.pos == self.data.len() {
            Ok(())
        } else {
            Err(invalid(format!("{} has trailing bytes", self.what)))
        }
    }
}

/// Appends little-endian values to a section buffer.
pub(crate) trait PutLe {
    fn put_u32(&mut self, value: u32);
    fn put_u64(&mut self, value: u64);
    fn put_f32(&mut self, value: f32);
}

impl PutLe for Vec<u8> {
    fn put_u32(&mut self, value: u32) {
        self.extend_from_slice(&value.to_le_bytes());
    }
    fn put_u64(&mut self, value: u64) {
        self.extend_from_slice(&value.to_le_bytes());
    }
    fn put_f32(&mut self, value: f32) {
        self.extend_from_slice(&value.to_le_bytes());
    }
}
