//! Deterministic reductions used by the moment-based models.

use ndarray::ArrayView1;

#[derive(Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    error: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let total = self.sum + value;
        self.error += if self.sum.abs() >= value.abs() {
            (self.sum - total) + value
        } else {
            (value - total) + self.sum
        };
        self.sum = total;
    }
}

pub(crate) fn population_mean(values: ArrayView1<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() <= 8 {
        let mut total = CompensatedSum::default();
        for &value in values {
            total.add(value);
        }
        let sum = total.sum + total.error;
        return if sum.is_finite() {
            sum / values.len() as f64
        } else {
            scaled_mean(values)
        };
    }
    compensated_mean(values)
}

// chunks_exact keeps this implementation usable before slice::as_chunks
// became stable, while exposing a fixed-width loop to the optimizer.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
#[inline(never)]
fn compensated_mean(values: ArrayView1<f64>) -> f64 {
    // Fixed independent lanes expose instruction-level parallelism without
    // letting the thread count change the reduction order.
    let mut sums = [0.0_f64; 8];
    let mut errors = [0.0_f64; 8];
    if let Some(slice) = values.as_slice() {
        let (head, tail) = slice.split_at(slice.len() / 8 * 8);
        #[cfg(target_arch = "x86_64")]
        let used_avx = if std::is_x86_feature_detected!("avx") {
            // SAFETY: runtime detection includes OS support for AVX state.
            unsafe { accumulate_avx(head, &mut sums, &mut errors) };
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "x86_64"))]
        let used_avx = false;
        if !used_avx {
            for chunk in head.chunks_exact(8) {
                add_lanes(&mut sums, &mut errors, chunk.try_into().unwrap());
            }
        }
        if !tail.is_empty() {
            let mut padded = [0.0; 8];
            padded[..tail.len()].copy_from_slice(tail);
            add_lanes(&mut sums, &mut errors, &padded);
        }
    } else {
        for (i, &value) in values.iter().enumerate() {
            let k = i % 8;
            let total = sums[k] + value;
            let recovered = total - sums[k];
            errors[k] += (sums[k] - (total - recovered)) + (value - recovered);
            sums[k] = total;
        }
    }
    let mut total = CompensatedSum::default();
    for k in 0..8 {
        total.add(sums[k]);
        total.add(errors[k]);
    }
    let sum = total.sum + total.error;
    if sum.is_finite() {
        return sum / values.len() as f64;
    }
    scaled_mean(values)
}

#[inline]
fn add_lanes(sums: &mut [f64; 8], errors: &mut [f64; 8], values: &[f64; 8]) {
    // Knuth TwoSum keeps the residual without a magnitude-dependent branch.
    // Preserve its operation order; reassociation destroys compensation.
    // See Ogita, Rump and Oishi (2005), doi:10.1137/030601818.
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        use std::arch::x86_64::{_mm_add_pd, _mm_loadu_pd, _mm_storeu_pd, _mm_sub_pd};
        // SAFETY: SSE2 is enabled for this target. Every two-element load/store is
        // within one of the eight-element arrays; unaligned access is allowed.
        unsafe {
            for k in (0..8).step_by(2) {
                let a = _mm_loadu_pd(sums.as_ptr().add(k));
                let b = _mm_loadu_pd(values.as_ptr().add(k));
                let total = _mm_add_pd(a, b);
                let recovered = _mm_sub_pd(total, a);
                let error = _mm_add_pd(
                    _mm_sub_pd(a, _mm_sub_pd(total, recovered)),
                    _mm_sub_pd(b, recovered),
                );
                let previous = _mm_loadu_pd(errors.as_ptr().add(k));
                _mm_storeu_pd(errors.as_mut_ptr().add(k), _mm_add_pd(previous, error));
                _mm_storeu_pd(sums.as_mut_ptr().add(k), total);
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    for k in 0..8 {
        let total = sums[k] + values[k];
        let recovered = total - sums[k];
        errors[k] += (sums[k] - (total - recovered)) + (values[k] - recovered);
        sums[k] = total;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
unsafe fn accumulate_avx(values: &[f64], sums: &mut [f64; 8], errors: &mut [f64; 8]) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_storeu_pd, _mm256_sub_pd,
    };
    // SAFETY: the caller checked AVX support. Full eight-element chunks permit both
    // four-element loads; array outputs permit both stores. All are unaligned.
    unsafe {
        let mut low = _mm256_setzero_pd();
        let mut high = _mm256_setzero_pd();
        let mut error_low = _mm256_setzero_pd();
        let mut error_high = _mm256_setzero_pd();
        for chunk in values.chunks_exact(8) {
            let a = _mm256_loadu_pd(chunk.as_ptr());
            let b = _mm256_loadu_pd(chunk.as_ptr().add(4));
            let total_low = _mm256_add_pd(low, a);
            let total_high = _mm256_add_pd(high, b);
            let recovered_low = _mm256_sub_pd(total_low, low);
            let recovered_high = _mm256_sub_pd(total_high, high);
            let residual_low = _mm256_add_pd(
                _mm256_sub_pd(low, _mm256_sub_pd(total_low, recovered_low)),
                _mm256_sub_pd(a, recovered_low),
            );
            let residual_high = _mm256_add_pd(
                _mm256_sub_pd(high, _mm256_sub_pd(total_high, recovered_high)),
                _mm256_sub_pd(b, recovered_high),
            );
            error_low = _mm256_add_pd(error_low, residual_low);
            error_high = _mm256_add_pd(error_high, residual_high);
            low = total_low;
            high = total_high;
        }
        _mm256_storeu_pd(sums.as_mut_ptr(), low);
        _mm256_storeu_pd(sums.as_mut_ptr().add(4), high);
        _mm256_storeu_pd(errors.as_mut_ptr(), error_low);
        _mm256_storeu_pd(errors.as_mut_ptr().add(4), error_high);
    }
}

#[cold]
fn scaled_mean(values: ArrayView1<f64>) -> f64 {
    // A finite mean can exist even when an intermediate sum overflows.
    // This slower path normalizes the whole population before accumulating.
    let mut scale = 0.0_f64;
    for &value in values {
        if !value.is_finite() {
            return f64::NAN;
        }
        scale = scale.max(value.abs());
    }
    if scale == 0.0 {
        return 0.0;
    }
    let mut total = CompensatedSum::default();
    for &value in values {
        total.add(value / scale);
    }
    let normalized = (total.sum + total.error) / values.len() as f64;
    // The exact normalized mean lies in [-1, 1]; protect the final multiply
    // from roundoff at f64::MAX.
    normalized.clamp(-1.0, 1.0) * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_lanes_match_scalar_compensation() {
        let values: Vec<f64> = (0..257 * 8)
            .map(|i| match i % 7 {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                3 => f64::MIN_POSITIVE,
                4 => -f64::MIN_POSITIVE,
                5 => f64::from_bits(1),
                _ => -(i as f64),
            })
            .collect();
        let mut expected = [CompensatedSum::default(); 8];
        for (i, &value) in values.iter().enumerate() {
            expected[i % 8].add(value);
        }
        let mut sums = [0.0; 8];
        let mut errors = [0.0; 8];
        for chunk in values.chunks(8) {
            add_lanes(&mut sums, &mut errors, chunk.try_into().unwrap());
        }
        for k in 0..8 {
            assert_eq!(sums[k], expected[k].sum);
            assert_eq!(errors[k], expected[k].error);
        }
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: same feature guard as the production dispatch.
            unsafe { accumulate_avx(&values, &mut sums, &mut errors) };
            for k in 0..8 {
                assert_eq!(sums[k], expected[k].sum);
                assert_eq!(errors[k], expected[k].error);
            }
        }
    }
}
