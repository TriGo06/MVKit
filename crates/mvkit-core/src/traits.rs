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

    /// Optional per-coordinate, state-independent diffusion coefficients.
    ///
    /// When present, the slice must have length `dim()` and agree with
    /// `diffusion` for every state. All diffusion derivatives must be zero.
    /// Integrators may then fill the diffusion buffer once per simulation
    /// and use Euler for a requested Milstein step after checking its opt-in.
    /// Defaults to `None`, preserving the general path for custom models.
    fn constant_diffusion(&self) -> Option<&[f64]> {
        None
    }

    /// Opt in to the coordinatewise Milstein implementation.
    ///
    /// For distinct particle-coordinate pairs a and b, the model must
    /// satisfy sigma_b * partial_b sigma_a = 0. Dependence on other noisy
    /// coordinates or on the empirical measure can violate this condition,
    /// even when the diffusion matrix is diagonal. The self-derivative
    /// returned by `diffusion_derivative`, or the coefficient supplied by
    /// `milstein_coefficient`, must also be correct. Constant diffusion and
    /// coefficients depending only on their own coordinate satisfy the
    /// structural condition. Defaults to false for custom models.
    fn supports_milstein(&self) -> bool {
        false
    }

    /// Compute the diagonal of the diffusion Jacobian:
    /// `out[i, k] = d sigma^k / d X_i^k`. Used by Milstein-type
    /// schemes. Default returns zeros, suitable for constant diffusion.
    /// State-dependent models must provide the actual derivative before
    /// opting in via `supports_milstein`.
    fn diffusion_derivative(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let _ = state;
        out.fill(0.0);
    }

    /// Optionally fill `out` with `0.5 * dt * sigma * partial_self(sigma)`
    /// and return true. This coefficient multiplies `(Z^2 - 1)` in Milstein.
    ///
    /// Models with a singular derivative but a regular product can evaluate
    /// the coefficient directly. Define the discrete boundary extension
    /// explicitly. The default returns false without touching `out`, so the
    /// integrator uses `diffusion_derivative` and its existing operation order.
    /// This does not relax the cross-noise condition in `supports_milstein`.
    fn milstein_coefficient(
        &self,
        _state: ArrayView2<f64>,
        _dt: f64,
        _out: ArrayViewMut2<f64>,
    ) -> bool {
        false
    }
}
