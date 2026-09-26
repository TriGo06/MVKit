//! Regression oracles for stream composition and extreme finite states.

use mvkit_core::{
    models::{Kuramoto, LinearQuadratic, MeanFieldCIR},
    schemes::{euler_maruyama, milstein},
    MeanFieldSDE,
};
use ndarray::{array, Array2, Array3};
use rayon::ThreadPoolBuilder;

#[test]
fn kuramoto_parameter_and_brownian_streams_are_separate() {
    let n = 32_768;
    for seed in [0, 7, u64::MAX] {
        let model = Kuramoto::with_gaussian_omegas(0.0, n, 1.0, 1.0, seed);
        let repeat = Kuramoto::with_gaussian_omegas(0.0, n, 1.0, 1.0, seed);
        assert_eq!(model.omegas, repeat.omegas);
        let history = euler_maruyama(&model, &Array2::zeros((n, 1)), 1.0, 1, 0, seed, None);
        let theta = history.index_axis(ndarray::Axis(0), 1);
        let mean = theta.sum() / n as f64;
        let variance = theta.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let cross_moment = model
            .omegas
            .iter()
            .enumerate()
            .map(|(i, omega)| omega * (theta[[i, 0]] - omega))
            .sum::<f64>()
            / n as f64;
        // Var(omega + Z) = 2; tolerance exceeds eight standard errors.
        assert!(
            (variance - 2.0).abs() < 0.125,
            "seed {seed}: variance {variance}"
        );
        assert!(
            cross_moment.abs() < 0.08,
            "seed {seed}: cross moment {cross_moment}"
        );
    }
}

#[test]
fn cir_milstein_matches_the_one_step_formula_at_small_positive_states() {
    let model = MeanFieldCIR::new(1.0, 1.0, 0.0, 1.0);
    for x in [1e-20, 1e-16, 1e-12, 1e-8] {
        let dt = x;
        for z in [0.0, 0.5, 2.0] {
            let history = milstein(
                &model,
                &array![[x]],
                dt,
                1,
                0,
                7,
                Some(&Array3::from_elem((1, 1, 1), z)),
            );
            let expected =
                x + (1.0 - x) * dt + x.sqrt() * dt.sqrt() * z + 0.25 * dt * (z * z - 1.0);
            assert!((history[[1, 0, 0]] / expected - 1.0).abs() < 2e-15);
        }
    }
}

#[test]
fn cir_milstein_boundary_extension_has_zero_noise_correction() {
    let model = MeanFieldCIR::new(1.0, 1.0, 0.0, 1.0);
    for x in [0.0, -1e-8] {
        let history = milstein(
            &model,
            &array![[x]],
            0.01,
            1,
            0,
            7,
            Some(&Array3::zeros((1, 1, 1))),
        );
        assert_eq!(history[[1, 0, 0]], x + (1.0 - x) * 0.01);
    }
}

#[test]
fn cir_direct_coefficient_preserves_thread_and_stride_invariance() {
    // Cross the coordinate-parallelism threshold and exercise both RNG paths.
    let n = 131_073;
    let model = MeanFieldCIR::new(1.0, 1.0, 0.25, 0.2);
    let x = Array2::from_shape_fn((n, 1), |(i, _)| match i % 3 {
        0 => 0.0,
        1 => 1e-20,
        _ => 1.0,
    });
    let z = Array3::from_shape_fn((3, n, 1), |(t, i, _)| ((t + i) as f64).cos());
    let strided = Array3::from_shape_fn((6, n, 2), |(t, i, k)| z[[t / 2, i, k / 2]])
        .slice_move(ndarray::s![..;2, .., ..;2]);
    let one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let four = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for noise in [None, Some(&z)] {
        let expected = one.install(|| milstein(&model, &x, 1e-4, 3, 0, 7, noise));
        let actual = four.install(|| milstein(&model, &x, 1e-4, 3, 0, 7, noise.map(|_| &strided)));
        assert!(actual.iter().all(|x| x.is_finite()));
        assert_eq!(actual, expected);
    }
}

#[test]
fn finite_population_means_do_not_overflow() {
    let x = array![[1e308], [1e308]];
    let mut drift = Array2::zeros(x.raw_dim());
    LinearQuadratic::new(0.0, 1.0, 0.0).drift(x.view(), drift.view_mut());
    assert_eq!(drift, x);
    MeanFieldCIR::new(1.0, 1.0, 1.0, 1.0).drift(x.view(), drift.view_mut());
    assert!(drift.iter().all(|&v| v == -1e308));
    let result = euler_maruyama(
        &LinearQuadratic::new(0.0, 0.0, 0.0),
        &x,
        1.0,
        1,
        0,
        7,
        Some(&Array3::zeros((1, 2, 1))),
    );
    assert_eq!(result.index_axis(ndarray::Axis(0), 1), x);
}

#[test]
fn population_mean_retains_small_terms_under_cancellation() {
    for values in [[1e16, 1.0, -1e16], [1e16, -1e16, 1.0], [1.0, 1e16, -1e16]] {
        let x = Array2::from_shape_vec((3, 1), values.to_vec()).unwrap();
        let mut drift = Array2::zeros(x.raw_dim());
        LinearQuadratic::new(0.0, 1.0, 0.0).drift(x.view(), drift.view_mut());
        assert!(drift.iter().all(|&v| v == 1.0 / 3.0));
    }
}

#[test]
fn mean_handles_extreme_magnitudes_and_strided_columns() {
    let model = LinearQuadratic::new(0.0, 1.0, 0.0);
    for value in [f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
        let x = Array2::from_elem((19, 2), value);
        let mut drift = Array2::zeros((19, 1));
        model.drift(x.slice(ndarray::s![.., ..;2]), drift.view_mut());
        assert!(drift.iter().all(|&actual| actual == value));
    }
    let x = Array2::from_shape_fn((19, 1), |(i, _)| match i {
        0 => 1e16,
        16 => -1e16,
        _ => 1.0,
    });
    let mut drift = Array2::zeros(x.raw_dim());
    model.drift(x.view(), drift.view_mut());
    assert!(drift.iter().all(|&actual| actual == 17.0 / 19.0));
}

#[test]
#[should_panic(expected = "b must be non-negative and finite")]
fn cir_rejects_outward_boundary_coupling() {
    MeanFieldCIR::new(1.0, 1.0, -10.0, 1.0);
}
