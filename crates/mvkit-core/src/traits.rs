//! Abstraction for mean-field SDEs.

use ndarray::{ArrayView2, ArrayViewMut2};

/// A mean-field SDE of the form
///
/// ```text
/// dX_i = b(X_i, mu_N) dt + sigma_k dW_i^k
/// ```
///
/// where `mu_N = (1/N) sum_j delta_{X_j}` is the empirical measure of the
/// particle system.
///
/// For now, the diffusion is restricted to a per-coordinate constant vector
/// `sigma`, which covers a large class of useful models (Cucker-Smale,
/// Kuramoto, McKean-Vlasov linear-quadratic, ...). State-dependent and
/// matrix-valued diffusion will come in a later version.
pub trait MeanFieldSDE: Sync {
    /// State dimension per particle.
    fn dim(&self) -> usize;

    /// Per-coordinate constant diffusion coefficient. Length must equal `dim()`.
    fn sigma(&self) -> &[f64];

    /// Compute the drift `b(X_i, mu_N)` for every particle.
    ///
    /// `state` and `out` both have shape `(N, dim())`.
    fn drift(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>);
}
