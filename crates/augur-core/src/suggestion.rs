/// Which predictors contributed to a [`Suggestion`], one bit per predictor
/// index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SourceMask(pub u64);

impl SourceMask {
    /// The maximum number of predictors an engine can hold.
    pub const CAPACITY: usize = u64::BITS as usize;

    /// Marks predictor `index` as a contributor.
    ///
    /// # Panics
    ///
    /// Panics if `index >= SourceMask::CAPACITY`.
    pub fn insert(&mut self, index: usize) {
        assert!(
            index < Self::CAPACITY,
            "predictor index {index} out of range"
        );
        self.0 |= 1 << index;
    }

    /// Whether predictor `index` contributed.
    pub fn contains(self, index: usize) -> bool {
        index < Self::CAPACITY && self.0 & (1 << index) != 0
    }
}

/// A ranked prediction returned to the caller.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// The word as it should be displayed or inserted.
    pub text: String,
    /// Combined probability, in `0.0..=1.0`.
    pub prob: f32,
    /// Predictors that proposed this word.
    pub sources: SourceMask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_mask_tracks_indices() {
        let mut mask = SourceMask::default();
        mask.insert(0);
        mask.insert(63);
        assert!(mask.contains(0));
        assert!(mask.contains(63));
        assert!(!mask.contains(1));
        assert!(!mask.contains(64));
    }
}
