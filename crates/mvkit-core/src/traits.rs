//! Abstraction for mean-field SDEs.

use ndarray::{ArrayView2, ArrayViewMut2};

/// A mean-field SDE with diagonal diffusion of the form
///
/// ```text
/// dX_i^k = b^k(X_i, mu_N) dt + sigma^k(X_i, mu_N) dW_i^k
/// ```
///
/// where `mu_N = (1/N) sum_j delta_{X_j}` is the empirical measure of the
/// particle system and `dW_i^k` are independent Brownian increments. The
/// diffusion is per-coordinate (diagonal); full matrix-valued diffusion is
/// a future extension.
///
/// Drift, diffusion, and the diagonal diffusion Jacobian are all written
/// row-by-row into a caller-allocated `(N, dim())` output buffer. Models
/// are free to parallelize internally; the integrator does not assume a
/// specific schedule.
pub trait MeanFieldSDE: Sync {
    /// State dimension per particle.
    fn dim(&self) -> usize;

    /// Compute the drift `b(X_i, mu_N)` for every particle.
    ///
    /// `state` and `out` both have shape `(N, dim())`.
    fn drift(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>);

    /// Compute the diffusion coefficient `sigma(X_i, mu_N)` for every
    /// particle. `out[i, k]` is the coefficient on `dW_i^k`. `state` and
    /// `out` both have shape `(N, dim())`. For constant-diffusion models
    /// this simply fills `out` with the per-coordinate constants.
    fn diffusion(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>);

    /// Compute the diagonal of the diffusion Jacobian:
    /// `out[i, k] = d sigma^k / d X_i^k`. Needed by Milstein-type
    /// schemes. Default returns zeros, which means Milstein collapses to
    /// Euler-Maruyama on the implementing model. Models with non-trivial
    /// state-dependent diffusion should override this.
    fn diffusion_derivative(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let _ = state;
        out.fill(0.0);
    }
}
