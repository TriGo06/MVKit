//! Built-in mean-field models.

use crate::parallel::map_coordinates;
use crate::traits::MeanFieldSDE;
use ndarray::{Array1, ArrayView2, ArrayViewMut2, Axis};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Internal helper: write a per-coordinate constant diffusion vector into
/// every row of an `(N, dim)` output buffer. Used by all the
/// constant-diffusion models below to implement `MeanFieldSDE::diffusion`.
fn fill_constant_diffusion(values: &[f64], mut out: ArrayViewMut2<f64>) {
    debug_assert_eq!(values.len(), out.ncols());
    for (k, &v) in values.iter().enumerate() {
        out.column_mut(k).fill(v);
    }
}

/// Linear-quadratic McKean-Vlasov model on the real line.
///
/// Each particle has scalar state and dynamics
/// ```text
/// dX_i = (a X_i + b * mean(X)) dt + sigma dW_i
/// ```
/// where `mean(X) = (1/N) sum_j X_j` is the empirical mean of the population.
///
/// If the initial law is Gaussian `X_0 ~ N(m_0, v_0)`, the marginal law stays
/// Gaussian for all `t`. The mean and McKean-Vlasov limit variance solve
/// ```text
/// dm/dt = (a + b) m,            m(t) = m_0 exp((a + b) t)
/// dv/dt = 2 a v + sigma^2,      v(t) = v_0 exp(2 a t)
///                                       + sigma^2 (exp(2 a t) - 1) / (2 a)
/// ```
/// (with the obvious `v(t) = v_0 + sigma^2 t` limit when `a = 0`).
///
/// For finite N and i.i.d. initial states, let v_c be the variance formula
/// with a replaced by c. The marginal variance is (1-1/N) v_a + v_(a+b)/N,
/// while the expected empirical population variance is (1-1/N) v_a.
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
    fn constant_diffusion(&self) -> Option<&[f64]> {
        Some(&self.sigma)
    }

    fn supports_milstein(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        1
    }

    fn drift(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        // Keep the reduction order independent of the thread count.
        let mean = state.column(0).sum() / n as f64;
        let a = self.a;
        let b_mean = self.b * mean;

        map_coordinates(state, out, |x| a * x + b_mean);
    }

    fn diffusion(&self, _state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        fill_constant_diffusion(&self.sigma, out);
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
    fn constant_diffusion(&self) -> Option<&[f64]> {
        Some(&self.sigma)
    }

    fn supports_milstein(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        2 * self.spatial_dim
    }

    fn diffusion(&self, _state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        fill_constant_diffusion(&self.sigma, out);
    }

    fn drift(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        let d = self.spatial_dim;
        let beta = self.beta;
        let n_inv = 1.0 / n as f64;

        if let (Some(x), Some(output)) = (state.as_slice(), out.as_slice_mut()) {
            match d {
                1 => return cucker_smale_contiguous::<1>(x, output, beta),
                2 => return cucker_smale_contiguous::<2>(x, output, beta),
                3 => return cucker_smale_contiguous::<3>(x, output, beta),
                _ => {}
            }
        }

        // Arbitrary dimensions and strided views use the same exact sum.
        // Accumulate in the output row, without a temporary allocation.
        let row_drift = |(i, mut row): (usize, ndarray::ArrayViewMut1<f64>)| {
            let xi = state.row(i);
            for k in 0..d {
                row[k] = xi[d + k];
                row[d + k] = 0.0;
            }
            for xj in state.rows() {
                let mut r2 = 0.0;
                for k in 0..d {
                    let dx = xj[k] - xi[k];
                    r2 += dx * dx;
                }
                let kernel = (1.0 + r2).powf(-beta);
                for k in 0..d {
                    row[d + k] += kernel * (xj[d + k] - xi[d + k]);
                }
            }
            for k in 0..d {
                row[d + k] *= n_inv;
            }
        };
        if n >= 32 && rayon::current_num_threads() > 1 {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .with_min_len(8)
                .enumerate()
                .for_each(row_drift);
        } else {
            out.axis_iter_mut(Axis(0)).enumerate().for_each(row_drift);
        }
    }
}

// Specialize common spatial dimensions, keeping j in its original order.
// No cutoff, symmetry reduction, approximate power, or reassociation is used.
fn cucker_smale_contiguous<const D: usize>(state: &[f64], out: &mut [f64], beta: f64) {
    let width = 2 * D;
    let n = state.len() / width;
    let n_inv = 1.0 / n as f64;
    let row_drift = |(i, row): (usize, &mut [f64])| {
        let xi = &state[i * width..(i + 1) * width];
        let mut accum = [0.0; D];
        for xj in state.chunks_exact(width) {
            let mut r2 = 0.0;
            for k in 0..D {
                let dx = xj[k] - xi[k];
                r2 += dx * dx;
            }
            let kernel = (1.0 + r2).powf(-beta);
            for k in 0..D {
                accum[k] += kernel * (xj[D + k] - xi[D + k]);
            }
        }
        row[..D].copy_from_slice(&xi[D..]);
        for k in 0..D {
            row[D + k] = n_inv * accum[k];
        }
    };
    if n >= 32 && rayon::current_num_threads() > 1 {
        out.par_chunks_exact_mut(width)
            .with_min_len(8)
            .enumerate()
            .for_each(row_drift);
    } else {
        out.chunks_exact_mut(width).enumerate().for_each(row_drift);
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
    fn constant_diffusion(&self) -> Option<&[f64]> {
        Some(&self.sigma)
    }

    fn supports_milstein(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        1
    }

    fn diffusion(&self, _state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        fill_constant_diffusion(&self.sigma, out);
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

/// Mean-field Cox-Ingersoll-Ross (CIR) process on the half-line.
///
/// Each particle has scalar state and dynamics
/// ```text
/// dX_i = kappa (theta - X_i) dt + b (mean(X) - X_i) dt
///        + sigma sqrt(max(X_i, 0)) dW_i
/// ```
/// where `mean(X) = (1/N) sum_j X_j`. The diffusion is square-root in the
/// state, which makes Milstein non-trivial: `(d/dx)(sigma sqrt(x)) =
/// 0.5 sigma / sqrt(x)`. The truncation `max(X_i, 0)` keeps the diffusion
/// real if the discretization underflows below zero. With the Feller
/// condition `2 kappa theta >= sigma^2`, the continuous-time process stays
/// strictly positive and the truncation is rarely needed.
///
/// In the limit `b -> 0`, each particle is an independent classical CIR
/// process and the marginal mean satisfies the closed form
/// `E[X_t] = theta + (X_0 - theta) exp(-kappa t)`. We use this as a
/// quantitative benchmark in the Milstein vs Euler test suite.
///
/// References: Cox, J. C., Ingersoll, J. E., and Ross, S. A. (1985). *A
/// theory of the term structure of interest rates*. Econometrica 53, 385-407,
/// for the original CIR process. McKean-Vlasov extensions are standard
/// (see Carmona and Delarue, 2018).
pub struct MeanFieldCIR {
    pub kappa: f64,
    pub theta: f64,
    pub b: f64,
    pub sigma: f64,
}

impl MeanFieldCIR {
    pub fn new(kappa: f64, theta: f64, b: f64, sigma: f64) -> Self {
        Self {
            kappa,
            theta,
            b,
            sigma,
        }
    }
}

impl MeanFieldSDE for MeanFieldCIR {
    fn supports_milstein(&self) -> bool {
        true
    }

    fn dim(&self) -> usize {
        1
    }

    fn drift(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        let n = state.nrows();
        let mean = state.column(0).sum() / n as f64;
        let kappa = self.kappa;
        let theta = self.theta;
        let b = self.b;

        map_coordinates(state, out, |x| kappa * (theta - x) + b * (mean - x));
    }

    fn diffusion(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        let sigma = self.sigma;
        // Truncate at zero if a discrete step crossed the boundary.
        map_coordinates(state, out, |x| sigma * x.max(0.0).sqrt());
    }

    fn diffusion_derivative(&self, state: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        // d/dx (sigma sqrt(x)) = 0.5 * sigma / sqrt(x), which diverges at
        // x = 0. We floor x at a small positive eps so the derivative
        // stays finite even when a particle is exactly at, or just below,
        // zero. In the Milstein update the correction term is
        // 0.5 * sigma * sigma_deriv * dt * (Z^2 - 1) = 0.25 * sigma^2 * dt * (Z^2 - 1)
        // for x > 0, which matches the standard CIR Milstein scheme.
        // When x <= 0, the diffusion itself is zero (truncated), so the
        // entire correction term vanishes regardless of the floored
        // derivative.
        const EPS: f64 = 1e-12;
        let sigma = self.sigma;
        map_coordinates(state, out, |x| 0.5 * sigma / x.max(EPS).sqrt());
    }
}
