//! Numerical integrators for mean-field SDEs.

use crate::traits::MeanFieldSDE;
use ndarray::{Array2, Array3, Axis, Zip};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Euler-Maruyama integrator for a mean-field SDE.
///
/// # Arguments
/// * `model`        - the mean-field SDE to integrate
/// * `x0`           - initial state, shape `(N, dim)`
/// * `t_final`      - final time `T`
/// * `n_steps`      - number of time steps; `dt = T / n_steps`
/// * `record_every` - record state every `k` steps. If 0, only the initial and
///   final states are recorded
/// * `seed`         - master RNG seed (fully deterministic given this seed)
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
) -> Array3<f64> {
    assert_eq!(x0.ncols(), model.dim(), "state dim mismatch");
    assert!(n_steps > 0, "n_steps must be > 0");
    assert!(t_final > 0.0, "t_final must be > 0");

    let n = x0.nrows();
    let d = model.dim();
    let dt = t_final / n_steps as f64;
    let sqrt_dt = dt.sqrt();

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

    let mut state: Array2<f64> = x0.clone();
    let mut drift_buf = Array2::<f64>::zeros((n, d));
    let mut noise_buf = Array2::<f64>::zeros((n, d));
    let mut sigma_buf = Array2::<f64>::zeros((n, d));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut record_idx = 1usize;

    for step in 1..=n_steps {
        // Sample noise sequentially. For typical N this is cheap (~10^7
        // samples/s with Ziggurat) and gives strict reproducibility regardless
        // of the parallel scheduler.
        for v in noise_buf.iter_mut() {
            *v = StandardNormal.sample(&mut rng);
        }

        // Drift and diffusion are filled by the model. They may be parallel
        // internally; the integrator only relies on row independence.
        model.drift(state.view(), drift_buf.view_mut());
        model.diffusion(state.view(), sigma_buf.view_mut());

        // x <- x + b dt + sigma sqrt(dt) Z, parallel update over rows.
        Zip::from(state.rows_mut())
            .and(drift_buf.rows())
            .and(noise_buf.rows())
            .and(sigma_buf.rows())
            .par_for_each(|mut s_row, b_row, z_row, sigma_row| {
                for k in 0..d {
                    s_row[k] += b_row[k] * dt + sigma_row[k] * sqrt_dt * z_row[k];
                }
            });

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
        let hist = euler_maruyama(&model, &x0, t_final, 1000, 0, 99);
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
        let hist = euler_maruyama(&model, &x0, t_final, 5000, 0, 7);
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
        let hist = euler_maruyama(&model, &x0, 5.0, 500, 0, 42);
        let final_state = hist.index_axis(Axis(0), hist.shape()[0] - 1);
        let final_mean_vx: f64 = (0..n).map(|i| final_state[[i, 2]]).sum::<f64>() / n as f64;
        let final_mean_vy: f64 = (0..n).map(|i| final_state[[i, 3]]).sum::<f64>() / n as f64;

        assert_abs_diff_eq!(init_mean_vx, final_mean_vx, epsilon = 1e-10);
        assert_abs_diff_eq!(init_mean_vy, final_mean_vy, epsilon = 1e-10);
    }
}
