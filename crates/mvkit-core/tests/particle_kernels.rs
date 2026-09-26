//! Numerical contracts exercised by the optimized coordinate and pair kernels.

use mvkit_core::{
    models::{CuckerSmale, Kuramoto, LinearQuadratic},
    schemes::{euler_maruyama, milstein},
    MeanFieldSDE,
};
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayViewMut2, Axis, ShapeBuilder};
use rayon::ThreadPoolBuilder;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 3e-13 * (1.0 + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn linear_particles_match_discrete_closed_form_with_recording() {
    // The empirical mean and centered particles have different eigenvalues.
    // Propagate each eigenmode analytically, including every noise impulse.
    let (n, steps) = (11, 9);
    let (a, b, sigma, dt): (f64, f64, f64, f64) = (-0.7, 0.4, 0.3, 0.125);
    let model = LinearQuadratic::new(a, b, sigma);
    let x0 = Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.7).cos());
    let z = Array3::from_shape_fn((steps, n, 1), |(t, i, _)| ((t * n + i) as f64).sin());
    let m0 = x0.sum() / n as f64;
    let r = 1.0 + a * dt;
    let q = 1.0 + (a + b) * dt;
    for record_every in [0, 1, 4, 12] {
        let history = euler_maruyama(
            &model,
            &x0,
            steps as f64 * dt,
            steps,
            record_every,
            17,
            Some(&z),
        );
        let recorded: Vec<usize> = (0..=steps)
            .filter(|&t| t == 0 || t == steps || (record_every > 0 && t % record_every == 0))
            .collect();
        assert_eq!(history.len_of(Axis(0)), recorded.len());
        for (frame, &t) in recorded.iter().enumerate() {
            for i in 0..n {
                let mut expected = r.powi(t as i32) * (x0[[i, 0]] - m0) + q.powi(t as i32) * m0;
                for k in 0..t {
                    let z_mean = z.index_axis(Axis(0), k).sum() / n as f64;
                    let power = (t - k - 1) as i32;
                    expected += sigma
                        * dt.sqrt()
                        * (r.powi(power) * (z[[k, i, 0]] - z_mean) + q.powi(power) * z_mean);
                }
                close(history[[frame, i, 0]], expected);
            }
        }
    }
}

struct Geometric;

impl MeanFieldSDE for Geometric {
    fn dim(&self) -> usize {
        4
    }
    fn drift(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        out.zip_mut_with(&state, |b, &x| *b = -0.5 * x);
    }
    fn diffusion(&self, state: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        out.zip_mut_with(&state, |sigma, &x| *sigma = 0.25 * x);
    }
    fn supports_milstein(&self) -> bool {
        true
    }
    fn diffusion_derivative(&self, _: ArrayView2<f64>, mut out: ArrayViewMut2<f64>) {
        out.fill(0.25);
    }
}

#[test]
fn state_dependent_schemes_preserve_strides_and_thread_invariance() {
    // The larger case crosses the parallel coordinate-update threshold.
    for n in [7, 32_770] {
        let steps = 5;
        let x0 = Array2::from_shape_fn((n, 4), |(i, k)| 1.0 + ((i + k) as f64).sin() * 0.1);
        let x_fortran = Array2::from_shape_fn((n, 4).f(), |idx| x0[idx]);
        let z = Array3::from_shape_fn((steps, n, 4), |(t, i, k)| ((t + i + k) as f64).cos());
        // Non-unit coordinate and time strides, with a different memory order.
        let z_strided =
            Array3::from_shape_fn((2 * steps, n, 8).f(), |(t, i, k)| z[[t / 2, i, k / 2]])
                .slice_move(s![..;2, .., ..;2]);
        assert!(!z_strided.is_standard_layout());
        for use_milstein in [false, true] {
            let solve = |x, noise| {
                if use_milstein {
                    milstein(&Geometric, x, 0.25, steps, 2, 123, noise)
                } else {
                    euler_maruyama(&Geometric, x, 0.25, steps, 2, 123, noise)
                }
            };
            for noise in [None, Some(&z)] {
                let one = ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(|| solve(&x0, noise));
                let four = ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build()
                    .unwrap()
                    .install(|| solve(&x_fortran, noise.map(|_| &z_strided)));
                assert_eq!(one, four);
                if noise.is_some() {
                    // Independent scalar product formula for the discrete scheme.
                    let dt: f64 = 0.25 / steps as f64;
                    for i in 0..n {
                        for k in 0..4 {
                            let mut expected = x0[[i, k]];
                            for t in 0..steps {
                                let z = z[[t, i, k]];
                                let correction = if use_milstein {
                                    0.5 * 0.25_f64.powi(2) * dt * (z * z - 1.0)
                                } else {
                                    0.0
                                };
                                expected *= 1.0 - 0.5 * dt + 0.25 * dt.sqrt() * z + correction;
                            }
                            close(one[[3, i, k]], expected);
                        }
                    }
                }
            }
        }
    }
}

// Use the default capability to exercise uncached diffusion and the full
// Milstein correction for the same mathematical model.
struct Uncached<M>(M);
impl<M: MeanFieldSDE> MeanFieldSDE for Uncached<M> {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn drift(&self, x: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        self.0.drift(x, out);
    }
    fn diffusion(&self, x: ArrayView2<f64>, out: ArrayViewMut2<f64>) {
        self.0.diffusion(x, out);
    }
    fn supports_milstein(&self) -> bool {
        self.0.supports_milstein()
    }
}

fn compare_constant_paths<M: MeanFieldSDE>(model: M, n: usize) {
    let d = model.dim();
    let x0 = Array2::from_shape_fn((n, d), |(i, k)| ((i + k) as f64).sin());
    let z = Array3::from_shape_fn((7, n, d), |(t, i, k)| ((t + i + k) as f64).cos());
    let uncached = Uncached(model);
    for noise in [None, Some(&z)] {
        let expected = euler_maruyama(&uncached, &x0, 0.1, 7, 3, 91, noise);
        assert_eq!(
            expected,
            euler_maruyama(&uncached.0, &x0, 0.1, 7, 3, 91, noise)
        );
        assert_eq!(expected, milstein(&uncached, &x0, 0.1, 7, 3, 91, noise));
        assert_eq!(expected, milstein(&uncached.0, &x0, 0.1, 7, 3, 91, noise));
    }
}

#[test]
fn constant_diffusion_optimization_matches_general_schemes() {
    compare_constant_paths(LinearQuadratic::new(-0.5, 0.2, 0.3), 39);
    compare_constant_paths(CuckerSmale::new(3, 0.4, 0.3), 39);
    compare_constant_paths(Kuramoto::new(0.3, Array1::linspace(-0.2, 0.2, 39), 0.3), 39);
}

#[test]
fn pairwise_kernels_match_reference_for_dimensions_and_strided_views() {
    for d in [1, 2, 3, 5] {
        for n in [7, 40] {
            let x = Array2::from_shape_fn((n, 2 * d), |(i, k)| ((i * 7 + k) as f64 * 0.3).sin());
            for beta in [0.0, 0.4, 1.5] {
                let model = CuckerSmale::new(d, beta, 0.0);
                // Independent symmetric pair accumulation. Every unordered pair
                // contributes opposite accelerations, preserving mean velocity.
                let mut expected = Array2::<f64>::zeros((n, 2 * d));
                for i in 0..n {
                    for k in 0..d {
                        expected[[i, k]] = x[[i, d + k]];
                    }
                    for j in i + 1..n {
                        let distance: f64 = (0..d).map(|k| (x[[j, k]] - x[[i, k]]).powi(2)).sum();
                        let weight = (1.0 + distance).powf(-beta) / n as f64;
                        for k in d..2 * d {
                            let force = weight * (x[[j, k]] - x[[i, k]]);
                            expected[[i, k]] += force;
                            expected[[j, k]] -= force;
                        }
                    }
                }
                let mut contiguous = Array2::zeros((n, 2 * d));
                ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(|| model.drift(x.view(), contiguous.view_mut()));
                let mut parallel = Array2::zeros((n, 2 * d));
                ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build()
                    .unwrap()
                    .install(|| model.drift(x.view(), parallel.view_mut()));
                assert_eq!(contiguous, parallel);
                let x_fortran = Array2::from_shape_fn((n, 2 * d).f(), |idx| x[idx]);
                let mut storage = Array2::from_elem((n, 4 * d), f64::NAN);
                let mut strided = storage.slice_mut(s![.., ..;2]);
                ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build()
                    .unwrap()
                    .install(|| model.drift(x_fortran.view(), strided.view_mut()));
                assert_eq!(contiguous, strided);
                for (&actual, &reference) in contiguous.iter().zip(expected.iter()) {
                    close(actual, reference);
                }
                for k in d..2 * d {
                    close(contiguous.column(k).sum(), 0.0);
                }
                assert!(storage.slice(s![.., 1..;2]).iter().all(|x| x.is_nan()));
            }
        }
    }
}
