//! Criterion benchmarks for the Euler-Maruyama integrator on the built-in
//! mean-field models.
//!
//! Run with `cargo bench --bench integrators` from the workspace root. The
//! `Throughput::Elements(N * n_steps)` setting makes Criterion report
//! particle-step throughput directly, which is the natural unit to track
//! for SDE simulators.
//!
//! Two suites:
//! - `euler_maruyama_lq`: linear-quadratic, scalar drift. Cheap drift,
//!   so this exposes the integrator overhead (noise sampling, parallel
//!   row update, recording).
//! - `euler_maruyama_cucker_smale`: O(N^2) drift, dominates at moderate N.
//!   The (2000, 200) point uses fewer steps so the bench wall time stays
//!   reasonable.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mvkit_core::models::{CuckerSmale, LinearQuadratic};
use mvkit_core::schemes::euler_maruyama;
use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256PlusPlus;

fn standard_normal_array(n_rows: usize, n_cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x0 = Array2::<f64>::zeros((n_rows, n_cols));
    for v in x0.iter_mut() {
        *v = StandardNormal.sample(&mut rng);
    }
    x0
}

fn bench_linear_quadratic(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler_maruyama_lq");
    let model = LinearQuadratic::new(-0.5, 1.0, 0.5);
    let n_steps: usize = 1000;
    let t_final = 1.0;
    for &n in &[1_000usize, 10_000, 100_000] {
        let x0 = standard_normal_array(n, 1, 0);
        group.throughput(Throughput::Elements((n as u64) * (n_steps as u64)));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_with_large_drop(|| {
                euler_maruyama(&model, &x0, t_final, n_steps, 0, 0xDEADBEEF, None)
            });
        });
    }
    group.finish();
}

fn bench_cucker_smale(c: &mut Criterion) {
    let mut group = c.benchmark_group("euler_maruyama_cucker_smale");
    let spatial_dim = 2usize;
    let model = CuckerSmale::new(spatial_dim, 0.4, 0.05);
    let t_final = 1.0;
    for &(n, n_steps) in &[(100usize, 1000usize), (500, 1000), (2000, 200)] {
        let x0 = standard_normal_array(n, 2 * spatial_dim, 0);
        group.throughput(Throughput::Elements((n as u64) * (n_steps as u64)));
        let id = format!("N={}_steps={}", n, n_steps);
        group.bench_with_input(
            BenchmarkId::from_parameter(id),
            &(n, n_steps),
            |b, &(_, ns)| {
                b.iter_with_large_drop(|| {
                    euler_maruyama(&model, &x0, t_final, ns, 0, 0xDEADBEEF, None)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_linear_quadratic, bench_cucker_smale);
criterion_main!(benches);
