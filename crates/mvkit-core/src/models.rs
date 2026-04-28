//! Built-in mean-field models.

use crate::traits::MeanFieldSDE;
use ndarray::{s, Array1, ArrayView2, ArrayViewMut2, Axis};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Linear-quadratic McKean-Vlasov model on the real line.
///
/// Each particle has scalar state and dynamics
/// ```text
/// dX_i = (a X_i + b * mean(X)) dt + sigma dW_i
/// ```
/// where `mean(X) = (1/N) sum_j X_j` is the empirical mean of the population.
///
/// If the initial law is Gaussian `X_0 ~ N(m_0, v_0)`, the marginal law stays
/// Gaussian for all `t`, with mean and variance solving the closed-form ODEs
/// ```text
/// dm/dt = (a + b) m,            m(t) = m_0 exp((a + b) t)
/// dv/dt = 2 a v + sigma^2,      v(t) = v_0 exp(2 a t)
///                                       + sigma^2 (exp(2 a t) - 1) / (2 a)
/// ```
/// (with the obvious `v(t) = v_0 + sigma^2 t` limit when `a = 0`).
///
/// The closed-form moments make this model a quantitative benchmark for any
/// mean-field integrator: empirical moments at time `T` can be compared
/// directly against `m(T)` and `v(T)`, which gives a clean handle on the weak
/// order of convergence (Talay and Tubaro, 1990).
pub struct LinearQuadratic {
    pub a: f64,
    pub b: f64,
    sigma: Vec<f64>,
}

impl LinearQuadratic {
    /// Create a linear-quadratic McKean-Vlasov model with parameters
    /// `a`, `b` and constant scalar diffusion `sigma`.
    pub fn new(a: f64, b: f64, sigma: f64) -> Self {
        Self {
            a,
            b,
            sigma: vec![sigma],
        }
    }
}

impl MeanFieldSDE for LinearQuadratic {
    fn dim(&self) -> usize {
        1
    }

    fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    fn drift(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        // Empirical mean. Sequential add over a single column is already
        // vectorized by the compiler and is cheap compared to the parallel
        // update below.
        let mean = state.column(0).sum() / n as f64;
        let a = self.a;
        let b_mean = self.b * mean;

        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut out_row)| {
                out_row[0] = a * state[[i, 0]] + b_mean;
            });
    }
}

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

/// Kuramoto coupled phase oscillators on the real line.
///
/// Each particle has a scalar phase `theta_i` (left unwrapped; the user can
/// reduce mod `2 pi` for visualization). The dynamics are
/// ```text
/// dtheta_i = omega_i dt + (K / N) sum_j sin(theta_j - theta_i) dt
///                       + sigma dW_i
/// ```
/// where `omega_i` are heterogeneous natural frequencies and `K` is the
/// coupling strength. The synchronization of the population is captured by
/// the order parameter
/// ```text
/// r(t) e^{i psi(t)} = (1/N) sum_j exp(i theta_j(t))
/// ```
/// with `r in [0, 1]`. For Gaussian `omega_i ~ N(0, omega_std^2)` the
/// classical critical coupling is `K_c = 2 omega_std sqrt(2 / pi)`: below
/// `K_c` the population stays incoherent (`r -> 0`), above it a fraction of
/// oscillators lock and `r` stabilizes between 0 and 1.
///
/// The drift is implemented in O(N) per time step using the trig identity
/// `sum_j sin(theta_j - theta_i) = S cos(theta_i) - C sin(theta_i)` with
/// `C = sum_j cos(theta_j)` and `S = sum_j sin(theta_j)`. A naive O(N^2)
/// nested-loop form is intentionally avoided; the reduction is sequential
/// (deterministic, single pass) and only the per-particle application is
/// parallelized.
///
/// Reference: Kuramoto, Y. (1975). *Self-entrainment of a population of
/// coupled non-linear oscillators*. International Symposium on Mathematical
/// Problems in Theoretical Physics.
pub struct Kuramoto {
    pub coupling_k: f64,
    pub omegas: Array1<f64>,
    sigma: Vec<f64>,
}

impl Kuramoto {
    /// Create a Kuramoto model with the given coupling, frequency vector
    /// and constant scalar diffusion. `omegas.len()` must equal the number
    /// of particles passed to `drift` (checked at simulation time).
    pub fn new(coupling_k: f64, omegas: Array1<f64>, sigma: f64) -> Self {
        Self {
            coupling_k,
            omegas,
            sigma: vec![sigma],
        }
    }

    /// Convenience constructor: `n_particles` natural frequencies sampled
    /// i.i.d. from a centered Gaussian with standard deviation `omega_std`,
    /// using a deterministic Xoshiro256++ seeded by `seed`.
    pub fn with_gaussian_omegas(
        coupling_k: f64,
        n_particles: usize,
        omega_std: f64,
        sigma: f64,
        seed: u64,
    ) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let dist = Normal::new(0.0, omega_std).expect("omega_std must be finite and >= 0");
        let omegas = Array1::from_iter((0..n_particles).map(|_| dist.sample(&mut rng)));
        Self::new(coupling_k, omegas, sigma)
    }
}

impl MeanFieldSDE for Kuramoto {
    fn dim(&self) -> usize {
        1
    }

    fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    fn drift(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        assert_eq!(
            n,
            self.omegas.len(),
            "Kuramoto drift: state has {} rows but omegas has {} entries",
            n,
            self.omegas.len()
        );

        // Pass 1: reduction over the single phase column. Sequential and
        // single-threaded so the floating-point summation order is fixed,
        // which the integrator's reproducibility contract relies on.
        let mut sum_cos = 0.0_f64;
        let mut sum_sin = 0.0_f64;
        for i in 0..n {
            let theta = state[[i, 0]];
            sum_cos += theta.cos();
            sum_sin += theta.sin();
        }

        let k_over_n = self.coupling_k / n as f64;
        let omegas = &self.omegas;

        // Pass 2: per-particle application, parallel over rows. Each row
        // depends only on the row's own theta and the two reduction
        // scalars, so there is no cross-row data dependency.
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut out_row)| {
                let theta = state[[i, 0]];
                let interaction = sum_sin * theta.cos() - sum_cos * theta.sin();
                out_row[0] = omegas[i] + k_over_n * interaction;
            });
    }
}
