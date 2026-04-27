//! Built-in mean-field models.

use crate::traits::MeanFieldSDE;
use ndarray::{s, ArrayView2, ArrayViewMut2, Axis};
use rayon::prelude::*;

/// Cucker-Smale flocking model in `spatial_dim` dimensions.
///
/// Per particle, the state is the concatenation `(x, v)` where both `x` and
/// `v` lie in `R^{spatial_dim}`. The total state dimension is therefore
/// `2 * spatial_dim`.
///
/// Dynamics:
/// ```text
/// dx_i = v_i dt
/// dv_i = (1/N) sum_j K(|x_j - x_i|) (v_j - v_i) dt + sigma dW_i
/// ```
/// with kernel `K(r) = (1 + r^2)^{-beta}`.
///
/// Conservation: the empirical mean velocity is conserved by the
/// deterministic part of the dynamics (sum of pairwise interactions is
/// antisymmetric in i, j). This is the classical result of Cucker and Smale
/// (2007) and a useful test for any implementation.
pub struct CuckerSmale {
    pub spatial_dim: usize,
    pub beta: f64,
    sigma: Vec<f64>,
}

impl CuckerSmale {
    /// Create a Cucker-Smale model with diffusion `sigma_v` applied only to
    /// the velocity coordinates (positions are not noised).
    pub fn new(spatial_dim: usize, beta: f64, sigma_v: f64) -> Self {
        assert!(spatial_dim >= 1, "spatial_dim must be >= 1");
        let mut sigma = vec![0.0; 2 * spatial_dim];
        for s in sigma.iter_mut().skip(spatial_dim) {
            *s = sigma_v;
        }
        Self {
            spatial_dim,
            beta,
            sigma,
        }
    }
}

impl MeanFieldSDE for CuckerSmale {
    fn dim(&self) -> usize {
        2 * self.spatial_dim
    }

    fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    fn drift(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        let d = self.spatial_dim;
        let beta = self.beta;
        let n_inv = 1.0 / n as f64;

        // Parallel over particles. O(N^2) per step; fine for N up to a few
        // thousand. Kernel-based fast methods (FFT, FMM) come in v0.2.
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut out_row)| {
                let xi = state.slice(s![i, 0..d]);
                let vi = state.slice(s![i, d..2 * d]);

                // Position drift = velocity.
                for k in 0..d {
                    out_row[k] = vi[k];
                }

                // Velocity drift = mean-field interaction.
                let mut accum = vec![0.0_f64; d];
                for j in 0..n {
                    let xj = state.slice(s![j, 0..d]);
                    let vj = state.slice(s![j, d..2 * d]);
                    let mut r2 = 0.0_f64;
                    for k in 0..d {
                        let dx = xj[k] - xi[k];
                        r2 += dx * dx;
                    }
                    let kernel = (1.0 + r2).powf(-beta);
                    for k in 0..d {
                        accum[k] += kernel * (vj[k] - vi[k]);
                    }
                }
                for k in 0..d {
                    out_row[d + k] = n_inv * accum[k];
                }
            });
    }
}
