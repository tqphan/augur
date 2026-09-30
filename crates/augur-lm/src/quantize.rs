//! 8-bit quantization of log values with per-order codebooks.
//!
//! A codebook holds 256 ascending centers. If a set of values has at most
//! 256 distinct values they are stored exactly; otherwise the sorted values
//! are split into bins of equal population and each bin is represented by
//! its mean. [`LOG_ZERO`] always gets an exact code of its own, so
//! impossible events stay impossible.

use crate::LOG_ZERO;
use crate::format::CODEBOOK_LEN;

/// Maps log values to 8-bit codes and back.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Codebook {
    centers: [f32; CODEBOOK_LEN],
}

impl Codebook {
    /// A codebook with the given centers (ascending).
    pub(crate) fn from_centers(centers: &[f32; CODEBOOK_LEN]) -> Self {
        Self { centers: *centers }
    }

    pub(crate) fn centers(&self) -> &[f32; CODEBOOK_LEN] {
        &self.centers
    }

    /// Builds a codebook for `values`.
    pub(crate) fn build(values: &[f32]) -> Self {
        let mut sorted: Vec<f32> = values.iter().copied().filter(|&v| v > LOG_ZERO).collect();
        sorted.sort_unstable_by(f32::total_cmp);
        let has_zero = sorted.len() < values.len();
        let slots = CODEBOOK_LEN - usize::from(has_zero);

        let mut centers: Vec<f32> = Vec::with_capacity(CODEBOOK_LEN);
        if has_zero {
            centers.push(LOG_ZERO);
        }
        let mut distinct = sorted.clone();
        distinct.dedup();
        if distinct.len() <= slots {
            centers.extend(distinct);
        } else {
            let mut means: Vec<f64> = (0..slots)
                .map(|bin| {
                    mean(&sorted[bin * sorted.len() / slots..(bin + 1) * sorted.len() / slots])
                })
                .collect();
            refine(&sorted, &mut means);
            #[allow(clippy::cast_possible_truncation)]
            centers.extend(means.iter().map(|&m| m as f32));
            centers.dedup();
        }
        let last = centers.last().copied().unwrap_or(0.0);
        centers.resize(CODEBOOK_LEN, last);
        Self {
            centers: centers.try_into().expect("codebook length"),
        }
    }

    /// The code of the center nearest to `value`.
    pub(crate) fn encode(&self, value: f32) -> u8 {
        // First center >= value; then pick the closer of it and its
        // predecessor.
        let above = self.centers.partition_point(|&c| c < value);
        let code = if above == 0 {
            0
        } else if above == CODEBOOK_LEN
            || value - self.centers[above - 1] <= self.centers[above] - value
        {
            // Among equal (padding) centers this yields the first.
            self.centers
                .partition_point(|&c| c < self.centers[above - 1])
        } else {
            above
        };
        u8::try_from(code).expect("code fits u8")
    }

    /// The value represented by `code`.
    pub(crate) fn decode(&self, code: u8) -> f32 {
        self.centers[usize::from(code)]
    }
}

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f32]) -> f64 {
    values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64
}

/// Lloyd iterations (1-D k-means) on ascending `sorted` values, starting
/// from ascending `centers`. Equal-population bins alone are coarse where
/// values are sparse, e.g. for rare events; this moves centers to reduce
/// the squared error. Centers stay ascending.
fn refine(sorted: &[f32], centers: &mut [f64]) {
    const ITERATIONS: usize = 20;
    for _ in 0..ITERATIONS {
        let boundaries: Vec<f64> = centers
            .windows(2)
            .map(|w| f64::midpoint(w[0], w[1]))
            .collect();
        let mut moved: f64 = 0.0;
        let mut start = 0;
        for (j, center) in centers.iter_mut().enumerate() {
            let end = match boundaries.get(j) {
                Some(&boundary) => {
                    start + sorted[start..].partition_point(|&v| f64::from(v) <= boundary)
                }
                None => sorted.len(),
            };
            if end > start {
                let updated = mean(&sorted[start..end]);
                moved = moved.max((updated - *center).abs());
                *center = updated;
            }
            start = end;
        }
        if moved < 1e-6 {
            break;
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn few_values_are_exact() {
        let values = [-1.5, -0.25, 0.0, -0.25, LOG_ZERO];
        let book = Codebook::build(&values);
        for v in values {
            assert_eq!(book.decode(book.encode(v)), v);
        }
    }

    #[test]
    fn many_values_are_close() {
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..100_000)
            .map(|i| -7.0 * (i as f32) / 100_000.0)
            .collect();
        let book = Codebook::build(&values);
        assert!(book.centers().windows(2).all(|w| w[0] <= w[1]));
        let worst = values
            .iter()
            .map(|&v| (book.decode(book.encode(v)) - v).abs())
            .fold(0.0, f32::max);
        assert!(worst < 0.03, "worst error {worst}");
    }

    #[test]
    fn log_zero_keeps_its_own_code() {
        #[allow(clippy::cast_precision_loss)]
        let mut values: Vec<f32> = (0..10_000).map(|i| -(i as f32) / 1000.0).collect();
        values.push(LOG_ZERO);
        let book = Codebook::build(&values);
        assert_eq!(book.decode(book.encode(LOG_ZERO)), LOG_ZERO);
        assert!(book.decode(book.encode(-9.99)) > -10.1);
    }
}
