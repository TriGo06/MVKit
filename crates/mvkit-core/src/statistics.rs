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
    if let Some(slice) = values.as_slice() {
        // Bound the ordinary accumulation depth to 16 inputs per lane.
        // Compensate between blocks and revisit cancellation-heavy data with
        // compensation at every addition. Both passes have a fixed order.
        let mut total = CompensatedSum::default();
        let mut magnitude = 0.0;
        for block in slice.chunks(128) {
            let mut sums = [0.0; 8];
            let mut magnitudes = [0.0; 8];
            for chunk in block.chunks(8) {
                for (k, &value) in chunk.iter().enumerate() {
                    sums[k] += value;
                    magnitudes[k] += value.abs();
                }
            }
            total.add(sums.iter().sum());
            magnitude += magnitudes.iter().sum::<f64>();
        }
        let sum = total.sum + total.error;
        // A condition estimate above 8 amplifies the block rounding bound
        // (roughly 24 eps * sum(abs(x))) beyond about 200 eps relative error.
        // This is a float64 reduction, not an exact or correctly rounded sum.
        if sum.is_finite() && magnitude.is_finite() && magnitude <= 8.0 * sum.abs() {
            return sum / values.len() as f64;
        }
    }
    compensated_mean(values)
}

fn compensated_mean(values: ArrayView1<f64>) -> f64 {
    // Fixed independent lanes expose instruction-level parallelism without
    // letting the thread count change the reduction order.
    let mut lanes = [CompensatedSum::default(); 8];
    if let Some(slice) = values.as_slice() {
        for chunk in slice.chunks(8) {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                lane.add(value);
            }
        }
    } else {
        for (i, &value) in values.iter().enumerate() {
            lanes[i % 8].add(value);
        }
    }
    let mut total = CompensatedSum::default();
    for lane in lanes {
        total.add(lane.sum);
        total.add(lane.error);
    }
    let sum = total.sum + total.error;
    if sum.is_finite() {
        return sum / values.len() as f64;
    }
    scaled_mean(values)
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
