//! Numerical integrators for mean-field SDEs.

use crate::parallel::{use_parallel, MIN_PARALLEL_LEN};
use crate::traits::MeanFieldSDE;
use ndarray::{Array2, Array3, Axis, Zip};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

/// Euler-Maruyama integrator for a mean-field SDE.
///
/// # Arguments
/// * `model`        - the mean-field SDE to integrate
/// * `x0`           - initial state, shape `(N, dim)`
/// * `t_final`      - final time `T`
/// * `n_steps`      - number of time steps; `dt = T / n_steps`
/// * `record_every` - record state every `k` steps. If 0, only the initial and
///   final states are recorded
/// * `seed`         - master RNG seed (fully deterministic given this seed,
///   used only when `increments` is `None`)
/// * `increments`   - optional precomputed standard-normal increments of
///   shape `(n_steps, N, dim)`. When `Some`, the integrator reads noise
///   from this array instead of sampling internally; the values are the
///   raw `Z_n` (not yet scaled by `sqrt(dt)`). Used by strong-error tests
///   that drive coarse and fine simulations from the same Brownian path.
///
/// # Returns
/// A 3D array of shape `(n_recorded, N, dim)` containing recorded states.
pub fn euler_maruyama<M: MeanFieldSDE>(
    model: &M,
    x0: &Array2<f64>,
    t_final: f64,
    n_steps: usize,
    record_every: usize,
    seed: u64,
    increments: Option<&Array3<f64>>,
) -> Array3<f64> {
    assert_eq!(x0.ncols(), model.dim(), "state dim mismatch");
    assert!(n_steps > 0, "n_steps must be > 0");
    assert!(t_final > 0.0, "t_final must be > 0");

    let n = x0.nrows();
    let d = model.dim();
    let dt = t_final / n_steps as f64;
    let sqrt_dt = dt.sqrt();

    if let Some(inc) = increments {
        assert_eq!(
            inc.shape(),
            [n_steps, n, d],
            "increments shape mismatch: expected (n_steps, N, dim)"
        );
    }

    // Original division-and-remainder form, kept verbatim so this crate
    // builds on Rust toolchains older than 1.87 (the stabilization of
    // is_multiple_of). Two recent clippy lints would otherwise rewrite
    // this into a higher-MSRV form: manual_is_multiple_of (1.94+) and
    // manual_checked_ops (1.95+, fires on the if-zero/else-divide
    // pattern below). We allow both; unknown_lints keeps the 1.95-only
    // name from breaking older clippy versions.
    #[allow(
        unknown_lints,
        clippy::manual_is_multiple_of,
        clippy::manual_checked_ops
    )]
    let n_recorded = if record_every == 0 {
        2
    } else {
        let regular = n_steps / record_every;
        let needs_final = n_steps % record_every != 0;
        1 + regular + usize::from(needs_final)
    };

    let mut history = Array3::<f64>::zeros((n_recorded, n, d));
    history.index_axis_mut(Axis(0), 0).assign(x0);

    // Normalize once so coordinate updates traverse contiguous memory.
    let mut state = x0.as_standard_layout().into_owned();
    let mut drift_buf = Array2::<f64>::zeros((n, d));
    // Supplied increments are borrowed one step at a time, without a copy.
    let mut noise_buf = increments.is_none().then(|| Array2::<f64>::zeros((n, d)));
    let mut sigma_buf = Array2::<f64>::zeros((n, d));
    let constant_diffusion = if let Some(values) = model.constant_diffusion() {
        assert_eq!(values.len(), d, "constant diffusion dimension mismatch");
        for (k, &value) in values.iter().enumerate() {
            sigma_buf.column_mut(k).fill(value);
        }
        true
    } else {
        false
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let parallel_update = use_parallel(state.len());

    let mut record_idx = 1usize;

    for step in 1..=n_steps {
        let noise = match increments {
            Some(inc) => inc.index_axis(Axis(0), step - 1),
            None => {
                let buffer = noise_buf.as_mut().unwrap();
                // Fixed sampling order preserves results across thread counts.
                for v in buffer.iter_mut() {
                    *v = StandardNormal.sample(&mut rng);
                }
                buffer.view()
            }
        };

        model.drift(state.view(), drift_buf.view_mut());
        if !constant_diffusion {
            model.diffusion(state.view(), sigma_buf.view_mut());
        }

        let update = |s: &mut f64, &b: &f64, &z: &f64, &sigma: &f64| {
            *s += b * dt + sigma * sqrt_dt * z;
        };
        let zip = Zip::from(&mut state)
            .and(&drift_buf)
            .and(noise)
            .and(&sigma_buf);
        if parallel_update {
            zip.into_par_iter()
                .with_min_len(MIN_PARALLEL_LEN)
                .for_each(|(s, b, z, sigma)| update(s, b, z, sigma));
        } else {
            zip.for_each(update);
        }

        let should_record = if record_every == 0 {
            step == n_steps
        } else {
            (step % record_every == 0) || step == n_steps
        };
        if should_record {
            debug_assert!(record_idx < n_recorded);
            history.index_axis_mut(Axis(0), record_idx).assign(&state);
            record_idx += 1;
        }
    }

    history
}

/// Milstein integrator for diagonal mean-field SDEs.
///
/// For diffusion satisfying [`MeanFieldSDE::supports_milstein`], the
/// coordinatewise Milstein update reads
/// ```text
/// X_{n+1}^k = X_n^k + b^k dt + sigma^k sqrt(dt) Z^k
///                  + 0.5 sigma^k (sigma^k)' dt (Z^k^2 - 1)
/// ```
/// where `(sigma^k)' = d sigma^k / d X_i^k` is the diagonal of the
/// diffusion Jacobian. Strong order 1 requires the usual coefficient
/// regularity and moment assumptions as well as the structural condition;
/// diagonal diffusion alone is insufficient. Cross-coordinate iterated
/// stochastic integrals are not implemented. When diffusion is constant (the
/// default `diffusion_derivative` returns zero), the correction vanishes
/// and the scheme reduces to Euler-Maruyama exactly.
///
/// Determinism contract: same `seed`, same `n_steps`, same model state
/// gives bit-exact identical noise samples to `euler_maruyama` thanks to
/// the shared sequential noise-sampling strategy. On any model whose
/// `diffusion_derivative` returns zero, `milstein` and `euler_maruyama`
/// produce bit-exact identical trajectories.
///
/// Same arguments and return shape as [`euler_maruyama`], including the
/// optional precomputed `increments` array. Panics if the model has not
/// explicitly opted in through `supports_milstein`.
pub fn milstein<M: MeanFieldSDE>(
    model: &M,
    x0: &Array2<f64>,
    t_final: f64,
    n_steps: usize,
    record_every: usize,
    seed: u64,
    increments: Option<&Array3<f64>>,
) -> Array3<f64> {
    assert!(
        model.supports_milstein(),
        "coordinatewise Milstein requires supports_milstein() and vanishing cross-noise derivatives"
    );
    if model.constant_diffusion().is_some() {
        return euler_maruyama(model, x0, t_final, n_steps, record_every, seed, increments);
    }
    assert_eq!(x0.ncols(), model.dim(), "state dim mismatch");
    assert!(n_steps > 0, "n_steps must be > 0");
    assert!(t_final > 0.0, "t_final must be > 0");

    let n = x0.nrows();
    let d = model.dim();
    let dt = t_final / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let half_dt = 0.5 * dt;

    if let Some(inc) = increments {
        assert_eq!(
            inc.shape(),
            [n_steps, n, d],
            "increments shape mismatch: expected (n_steps, N, dim)"
        );
    }

    #[allow(
        unknown_lints,
        clippy::manual_is_multiple_of,
        clippy::manual_checked_ops
    )]
    let n_recorded = if record_every == 0 {
        2
    } else {
        let regular = n_steps / record_every;
        let needs_final = n_steps % record_every != 0;
        1 + regular + usize::from(needs_final)
    };

    let mut history = Array3::<f64>::zeros((n_recorded, n, d));
    history.index_axis_mut(Axis(0), 0).assign(x0);

    // Normalize once so coordinate updates traverse contiguous memory.
    let mut state = x0.as_standard_layout().into_owned();
    let mut drift_buf = Array2::<f64>::zeros((n, d));
    // Supplied increments are borrowed one step at a time, without a copy.
    let mut noise_buf = increments.is_none().then(|| Array2::<f64>::zeros((n, d)));
    let mut sigma_buf = Array2::<f64>::zeros((n, d));
    let mut sigma_deriv_buf = Array2::<f64>::zeros((n, d));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let parallel_update = use_parallel(state.len());

    let mut record_idx = 1usize;

    for step in 1..=n_steps {
        let noise = match increments {
            Some(inc) => inc.index_axis(Axis(0), step - 1),
            None => {
                let buffer = noise_buf.as_mut().unwrap();
                for v in buffer.iter_mut() {
                    *v = StandardNormal.sample(&mut rng);
                }
                buffer.view()
            }
        };

        model.drift(state.view(), drift_buf.view_mut());
        model.diffusion(state.view(), sigma_buf.view_mut());
        model.diffusion_derivative(state.view(), sigma_deriv_buf.view_mut());

        let update = |s: &mut f64, &b: &f64, &z: &f64, &sigma: &f64, &deriv: &f64| {
            let correction = half_dt * sigma * deriv * (z * z - 1.0);
            *s += b * dt + sigma * sqrt_dt * z + correction;
        };
        let zip = Zip::from(&mut state)
            .and(&drift_buf)
            .and(noise)
            .and(&sigma_buf)
            .and(&sigma_deriv_buf);
        if parallel_update {
            zip.into_par_iter()
                .with_min_len(MIN_PARALLEL_LEN)
                .for_each(|(s, b, z, sigma, deriv)| update(s, b, z, sigma, deriv));
        } else {
            zip.for_each(update);
        }

        let should_record = if record_every == 0 {
            step == n_steps
        } else {
            (step % record_every == 0) || step == n_steps
        };
        if should_record {
            debug_assert!(record_idx < n_recorded);
            history.index_axis_mut(Axis(0), record_idx).assign(&state);
            record_idx += 1;
        }
    }

    history
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CuckerSmale, Kuramoto, LinearQuadratic};
    use approx::assert_abs_diff_eq;

    struct CrossDiffusion;

    impl MeanFieldSDE for CrossDiffusion {
        fn dim(&self) -> usize {
            2
        }

        fn drift(&self, _: ndarray::ArrayView2<f64>, mut out: ndarray::ArrayViewMut2<f64>) {
            out.fill(0.0);
        }

        fn diffusion(&self, state: ndarray::ArrayView2<f64>, mut out: ndarray::ArrayViewMut2<f64>) {
            out.column_mut(0).fill(1.0);
            out.column_mut(1).assign(&state.column(0));
        }
    }

    #[test]
    #[should_panic(expected = "vanishing cross-noise derivatives")]
    fn milstein_rejects_cross_coordinate_diffusion() {
        // dX_1 = dW_1, dX_2 = X_1 dW_2 needs a cross iterated integral,
        // although its diagonal diffusion derivatives are both zero.
        milstein(&CrossDiffusion, &Array2::zeros((2, 2)), 1.0, 10, 0, 0, None);
    }

    #[test]
    fn constant_diffusion_milstein_matches_euler() {
        let model = LinearQuadratic::new(-0.5, 0.1, 0.3);
        let x0 = Array2::zeros((4, 1));
        let euler = euler_maruyama(&model, &x0, 1.0, 20, 1, 7, None);
        let mil = milstein(&model, &x0, 1.0, 20, 1, 7, None);
        assert_eq!(euler, mil);
    }

    #[test]
    fn kuramoto_free_rotation_uncoupled_noiseless() {
        // K = 0, sigma = 0, deterministic ICs and explicit omegas. Each
        // phase obeys the trivial ODE dtheta_i/dt = omega_i, exact under
        // Euler since the drift is constant in time. After T = 1.0 the
        // discretized solution should match theta_i(0) + omega_i * T to
        // floating-point accuracy.
        let n = 100usize;
        let omegas = ndarray::Array1::from_iter((0..n).map(|i| 0.1 * (i as f64)));
        let model = Kuramoto::new(0.0, omegas.clone(), 0.0);

        let mut x0 = ndarray::Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x0[[i, 0]] = 0.05 * (i as f64);
        }
        let t_final = 1.0;
        let hist = euler_maruyama(&model, &x0, t_final, 1000, 0, 99, None);
        let final_state = hist.index_axis(Axis(0), hist.shape()[0] - 1);
        for i in 0..n {
            let expected = x0[[i, 0]] + omegas[i] * t_final;
            assert_abs_diff_eq!(final_state[[i, 0]], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn lq_no_noise_mean_follows_exponential() {
        // With sigma = 0 the mean ODE dm/dt = (a + b) m is exact, so the
        // empirical mean of the discretized particle system should match
        // m_0 exp((a + b) T) up to the Euler-Maruyama bias on the mean ODE,
        // which is O(dt).
        let n = 500;
        let m0 = 1.0;
        let x0 = ndarray::Array2::<f64>::from_elem((n, 1), m0);
        let a = -0.7;
        let b = 1.4;
        let t_final = 1.0;
        let model = LinearQuadratic::new(a, b, 0.0);
        let hist = euler_maruyama(&model, &x0, t_final, 5000, 0, 7, None);
        let final_state = hist.index_axis(Axis(0), hist.shape()[0] - 1);
        let final_mean: f64 = final_state.column(0).sum() / n as f64;
        let expected = m0 * ((a + b) * t_final).exp();
        assert_abs_diff_eq!(final_mean, expected, epsilon = 5e-4);
    }

    #[test]
    fn no_noise_conserves_mean_velocity() {
        let n = 50;
        let d = 2;
        let mut x0 = Array2::<f64>::zeros((n, 2 * d));
        for i in 0..n {
            x0[[i, 0]] = (i as f64) * 0.1;
            x0[[i, 1]] = ((i % 5) as f64) * 0.2;
            x0[[i, 2]] = ((i as f64) * 0.07).sin();
            x0[[i, 3]] = ((i as f64) * 0.13).cos();
        }
        let init_mean_vx: f64 = (0..n).map(|i| x0[[i, 2]]).sum::<f64>() / n as f64;
        let init_mean_vy: f64 = (0..n).map(|i| x0[[i, 3]]).sum::<f64>() / n as f64;

        let model = CuckerSmale::new(d, 0.3, 0.0);
        let hist = euler_maruyama(&model, &x0, 5.0, 500, 0, 42, None);
        let final_state = hist.index_axis(Axis(0), hist.shape()[0] - 1);
        let final_mean_vx: f64 = (0..n).map(|i| final_state[[i, 2]]).sum::<f64>() / n as f64;
        let final_mean_vy: f64 = (0..n).map(|i| final_state[[i, 3]]).sum::<f64>() / n as f64;

        assert_abs_diff_eq!(init_mean_vx, final_mean_vx, epsilon = 1e-10);
        assert_abs_diff_eq!(init_mean_vy, final_mean_vy, epsilon = 1e-10);
    }
}
