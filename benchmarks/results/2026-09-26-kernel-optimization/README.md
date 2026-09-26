# Particle kernel optimization results

This study compares the merged benchmark baseline with the first engine optimization.
The equations, Euler grid, precomputed noise, benchmark models and release compiler flags are the same.

- Baseline: `7daf9f4aef62ab45783ad2edba7a2650bc5093e1`.
- Optimized source: `3b363e5ac3496ae2aed39e8e7fb58e7f2224440b`.
- AMD EPYC 9V74 host, Linux x86_64, Rust 1.98.1, Python 3.12.14, NumPy 2.5.3.
- Worker affinity restricted to one or four logical CPUs; container quota eight CPUs.
- Seeds 7, 19 and 41; seven warm calls per configuration. Imports, input preparation and serialization are outside the solver timer.
- Each reported time is the median of the three per-seed medians. Speedup is baseline median divided by optimized median.

The baseline campaigns ran first, followed by the optimized campaigns. Each campaign rotated backend order reproducibly. These are observations on one shared host, without confidence intervals; raw repetitions and seed ranges are retained below.

## Rust core

| Case | N | d | Steps | CPU budget | Before ms | After ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| moments | 4096 | 1 | 128 | 1 | 4.505 | 0.332 | 13.59x |
| moments | 4096 | 1 | 128 | 4 | 11.691 | 0.342 | 34.15x |
| moments | 65536 | 1 | 128 | 1 | 45.675 | 11.195 | 4.08x |
| moments | 65536 | 1 | 128 | 4 | 62.855 | 9.958 | 6.31x |
| moments | 262144 | 1 | 32 | 1 | 47.400 | 10.477 | 4.52x |
| moments | 262144 | 1 | 32 | 4 | 32.458 | 8.986 | 3.61x |
| multiplicative | 2048 | 4 | 128 | 1 | 6.469 | 4.044 | 1.60x |
| multiplicative | 2048 | 4 | 128 | 4 | 15.574 | 8.621 | 1.81x |
| pairwise | 128 | 2 | 128 | 1 | 115.999 | 29.877 | 3.88x |
| pairwise | 128 | 2 | 128 | 4 | 43.862 | 17.436 | 2.52x |

For pairwise interactions, d is the spatial dimension and each particle has 2d state coordinates.

## Python interface

| Case | N | d | Steps | CPU budget | Before ms | After ms | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| moments | 4096 | 1 | 128 | 1 | 4.915 | 0.479 | 10.26x |
| moments | 4096 | 1 | 128 | 4 | 19.822 | 0.437 | 45.39x |
| moments | 65536 | 1 | 128 | 1 | 73.001 | 38.855 | 1.88x |
| moments | 65536 | 1 | 128 | 4 | 96.042 | 36.782 | 2.61x |
| moments | 262144 | 1 | 32 | 1 | 76.354 | 39.850 | 1.92x |
| moments | 262144 | 1 | 32 | 4 | 59.716 | 38.818 | 1.54x |
| pairwise | 128 | 2 | 128 | 1 | 119.565 | 30.628 | 3.90x |
| pairwise | 128 | 2 | 128 | 4 | 43.030 | 15.441 | 2.79x |

## Changes and interpretation

- Coordinate updates traverse elements directly instead of constructing a row view for each particle. Cheap loops run sequentially below 131,072 coordinates; larger loops use Rayon tasks with at least 65,536 coordinates.
- Supplied Rust increments are borrowed by time slice instead of copied to another buffer every step. Internally sampled normals retain their sequential sampling order.
- Models can explicitly declare state-independent diffusion through the new defaulted `constant_diffusion` trait method. Those coefficients are filled once; eligible Milstein calls reuse Euler after checking `supports_milstein`.
- Exact Cucker-Smale interactions use fixed-size accumulators for spatial dimensions 1, 2 and 3, and allocation-free output-row accumulation for other dimensions or strided arrays. Parallel interaction tasks start at 32 particles with at least eight rows per task.
- The interaction sum, power function and floating-point evaluation order are preserved. The benchmark-local multiplicative model was left unchanged, so its smaller improvement measures the benefit available through the generic integrator.

These measurements evaluate the combined change and do not attribute a percentage of the gain to an individual optimization. The scheduling cutoffs are conservative choices for these measured workloads, not hardware-independent optima.

The old scheduling penalty is removed from the small scalar model. On the larger 262,144-particle case, four allowed CPUs reduce the optimized median from 10.477 ms to 8.986 ms, but individual seed medians vary substantially. The unchanged custom multiplicative drift still launches Rayon for a cheap loop and remains slower with four CPUs than one.

At 65,536 particles and 128 steps, supplied scalar increments occupy 64 MiB. Python still takes an owned copy before releasing the GIL, and its median remains substantially above the native core. Reusable Rust-owned inputs/workspaces and compiled custom-model kernels are the next measured targets.

## Numerical and regression checks

- All 288 supported benchmark configurations passed independent NumPy agreement checks; the largest absolute discrepancy was 4.441e-16.
- There are 48 explicit unsupported entries across both revisions: the custom multiplicative model has no Python wrapper. No configuration failed.
- All 48 matched configuration groups improved their warm median; the smallest observed ratio was 1.31x. This includes the small-problem sweep.
- The compatibility check compares 56 model/seed/scheme/noise combinations against the single-thread baseline: baseline with four threads, optimized with one, and optimized with four. All 168 comparisons were bitwise identical for every recorded state.
- Compatibility covers built-in Euler and Milstein, supplied and generated noise, recording a non-divisible final step, and Cucker-Smale dimensions 1, 2, 3 and 5. It is a same-machine check, not a cross-platform bitwise guarantee.
- 247 Python tests passed with two optional skips. Nine Rust tests passed in release mode, including discrete closed-form solutions, conservation of mean velocity, strided input/output layouts and parallel-update boundaries. Formatting and Clippy passed.

Timing uses supplied increments only. The generated-noise check verifies correctness and reproducibility, not RNG throughput. This study compares MVKit revisions; competitor timings, higher-order methods, GPU performance and time to a statistical accuracy are outside its scope.

## Raw measurements and reproduction

- [Baseline rows](baseline.jsonl), [optimized rows](optimized.jsonl), [run metadata and binary hashes](metadata.json), [compatibility checks](compatibility.json).
- JSONL files concatenate the pilot, large, small and threshold campaigns in that order, without changing the original rows. Every original run configuration and environment is retained under `runs` in metadata.
- Build each revision in a separate checkout/environment using the dependency locks and release commands in the [benchmark protocol](../../README.md). Run these four campaigns separately for each revision:

```bash
python benchmarks/run_particle_benchmarks.py --profile pilot \
  --backends mvkit-core mvkit-python --threads 1 4 --seeds 7 19 41 \
  --steps 64 128 --repeats 7 --output benchmarks/local-results/kernels-pilot
python benchmarks/run_particle_benchmarks.py --profile pilot --cases moments \
  --particles 65536 --backends mvkit-core mvkit-python --threads 1 4 \
  --seeds 7 19 41 --steps 128 --repeats 7 --output benchmarks/local-results/kernels-large
python benchmarks/run_particle_benchmarks.py --profile smoke \
  --backends mvkit-core mvkit-python --threads 1 4 --seeds 7 19 41 \
  --steps 8 16 --repeats 7 --output benchmarks/local-results/kernels-small
python benchmarks/run_particle_benchmarks.py --profile pilot --cases moments \
  --particles 262144 --backends mvkit-core mvkit-python --threads 1 4 \
  --seeds 7 19 41 --steps 32 --repeats 7 --output benchmarks/local-results/kernels-threshold
python benchmarks/check_engine_compatibility.py \
  --baseline-python /path/to/baseline/.venv/bin/python \
  --candidate-python /path/to/optimized/.venv/bin/python \
  --output benchmarks/local-results/compatibility.json
```

## All configuration groups

Ranges span the three per-seed warm medians; they are not confidence intervals. Times are milliseconds.

| Case | N | d | Steps | CPU budget | Backend | Before ms (range) | After ms (range) | Speedup |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| moments | 32 | 1 | 8 | 1 | mvkit-core | 0.104688 (0.102295 to 0.106441) | 0.000801 (0.000762 to 0.000811) | 130.70x |
| moments | 32 | 1 | 8 | 1 | mvkit-python | 0.108094 (0.106861 to 0.108634) | 0.002995 (0.002995 to 0.002995) | 36.09x |
| moments | 32 | 1 | 8 | 4 | mvkit-core | 0.307635 (0.305782 to 0.582791) | 0.000761 (0.000751 to 0.000761) | 404.25x |
| moments | 32 | 1 | 8 | 4 | mvkit-python | 0.321776 (0.300894 to 0.347655) | 0.003085 (0.003014 to 0.003455) | 104.30x |
| moments | 32 | 1 | 16 | 1 | mvkit-core | 0.207614 (0.205690 to 0.221565) | 0.001212 (0.001182 to 0.001332) | 171.30x |
| moments | 32 | 1 | 16 | 1 | mvkit-python | 0.210939 (0.208806 to 0.213282) | 0.003746 (0.003515 to 0.003835) | 56.31x |
| moments | 32 | 1 | 16 | 4 | mvkit-core | 0.597143 (0.571214 to 0.661530) | 0.001212 (0.001172 to 0.001242) | 492.69x |
| moments | 32 | 1 | 16 | 4 | mvkit-python | 0.591534 (0.445324 to 0.628019) | 0.003865 (0.003826 to 0.003976) | 153.05x |
| moments | 4096 | 1 | 64 | 1 | mvkit-core | 2.189970 (2.122379 to 2.363723) | 0.161363 (0.160592 to 0.162766) | 13.57x |
| moments | 4096 | 1 | 64 | 1 | mvkit-python | 2.539027 (2.329121 to 2.603295) | 0.239762 (0.217168 to 0.247975) | 10.59x |
| moments | 4096 | 1 | 64 | 4 | mvkit-core | 5.684882 (5.653736 to 6.006158) | 0.163847 (0.163167 to 0.223898) | 34.70x |
| moments | 4096 | 1 | 64 | 4 | mvkit-python | 8.501905 (6.199089 to 11.053657) | 0.293694 (0.233643 to 0.301636) | 28.95x |
| moments | 4096 | 1 | 128 | 1 | mvkit-core | 4.504929 (4.410022 to 4.823080) | 0.331591 (0.323429 to 0.361967) | 13.59x |
| moments | 4096 | 1 | 128 | 1 | mvkit-python | 4.914726 (4.724020 to 5.192188) | 0.479114 (0.473565 to 0.545304) | 10.26x |
| moments | 4096 | 1 | 128 | 4 | mvkit-core | 11.690663 (11.204505 to 13.282592) | 0.342347 (0.337730 to 0.384170) | 34.15x |
| moments | 4096 | 1 | 128 | 4 | mvkit-python | 19.821921 (12.284828 to 21.980930) | 0.436669 (0.434667 to 0.448858) | 45.39x |
| moments | 65536 | 1 | 128 | 1 | mvkit-core | 45.675093 (45.498396 to 46.505218) | 11.195275 (10.678033 to 11.313704) | 4.08x |
| moments | 65536 | 1 | 128 | 1 | mvkit-python | 73.000910 (72.573042 to 73.408410) | 38.854856 (37.679440 to 39.813504) | 1.88x |
| moments | 65536 | 1 | 128 | 4 | mvkit-core | 62.855194 (59.905810 to 94.750369) | 9.957795 (9.934100 to 10.658023) | 6.31x |
| moments | 65536 | 1 | 128 | 4 | mvkit-python | 96.042240 (78.067233 to 139.768897) | 36.782405 (36.669845 to 37.918791) | 2.61x |
| moments | 262144 | 1 | 32 | 1 | mvkit-core | 47.399653 (46.177112 to 47.871161) | 10.476790 (10.156696 to 10.975462) | 4.52x |
| moments | 262144 | 1 | 32 | 1 | mvkit-python | 76.353504 (75.567180 to 77.537053) | 39.850400 (39.743619 to 41.013047) | 1.92x |
| moments | 262144 | 1 | 32 | 4 | mvkit-core | 32.458486 (25.818363 to 65.780154) | 8.985686 (8.172268 to 19.816376) | 3.61x |
| moments | 262144 | 1 | 32 | 4 | mvkit-python | 59.715915 (58.698410 to 84.380805) | 38.818420 (38.403464 to 55.995043) | 1.54x |
| multiplicative | 24 | 4 | 8 | 1 | mvkit-core | 0.106080 (0.105579 to 0.108524) | 0.052569 (0.052399 to 0.053240) | 2.02x |
| multiplicative | 24 | 4 | 8 | 4 | mvkit-core | 0.795467 (0.354225 to 1.038468) | 0.248065 (0.236056 to 0.478773) | 3.21x |
| multiplicative | 24 | 4 | 16 | 1 | mvkit-core | 0.211740 (0.211199 to 0.212701) | 0.105019 (0.104918 to 0.108484) | 2.02x |
| multiplicative | 24 | 4 | 16 | 4 | mvkit-core | 1.489981 (0.871497 to 1.543672) | 0.428127 (0.410280 to 0.632626) | 3.48x |
| multiplicative | 2048 | 4 | 64 | 1 | mvkit-core | 3.175129 (3.047826 to 3.515192) | 2.024639 (2.017788 to 2.037168) | 1.57x |
| multiplicative | 2048 | 4 | 64 | 4 | mvkit-core | 7.039870 (6.805186 to 13.734928) | 5.378878 (4.044962 to 7.582748) | 1.31x |
| multiplicative | 2048 | 4 | 128 | 1 | mvkit-core | 6.469298 (6.251128 to 6.830924) | 4.043680 (3.874094 to 4.071983) | 1.60x |
| multiplicative | 2048 | 4 | 128 | 4 | mvkit-core | 15.574088 (13.984796 to 23.880473) | 8.620655 (8.333281 to 13.156439) | 1.81x |
| pairwise | 12 | 2 | 8 | 1 | mvkit-core | 0.167413 (0.166281 to 0.168755) | 0.016134 (0.016114 to 0.016165) | 10.38x |
| pairwise | 12 | 2 | 8 | 1 | mvkit-python | 0.173292 (0.172341 to 0.175154) | 0.019290 (0.019039 to 0.020811) | 8.98x |
| pairwise | 12 | 2 | 8 | 4 | mvkit-core | 0.527708 (0.390881 to 0.581119) | 0.016225 (0.016155 to 0.016324) | 32.52x |
| pairwise | 12 | 2 | 8 | 4 | mvkit-python | 0.364967 (0.269817 to 0.368036) | 0.019249 (0.018789 to 0.019399) | 18.96x |
| pairwise | 12 | 2 | 16 | 1 | mvkit-core | 0.334115 (0.332913 to 0.341586) | 0.032018 (0.031898 to 0.032099) | 10.44x |
| pairwise | 12 | 2 | 16 | 1 | mvkit-python | 0.348546 (0.347074 to 0.354986) | 0.035414 (0.035223 to 0.035593) | 9.84x |
| pairwise | 12 | 2 | 16 | 4 | mvkit-core | 1.007171 (0.640858 to 1.305844) | 0.031988 (0.031889 to 0.032239) | 31.49x |
| pairwise | 12 | 2 | 16 | 4 | mvkit-python | 0.704063 (0.637733 to 0.728231) | 0.035544 (0.035384 to 0.035573) | 19.81x |
| pairwise | 128 | 2 | 64 | 1 | mvkit-core | 55.467184 (55.137547 to 56.561857) | 14.831128 (14.435567 to 14.833218) | 3.74x |
| pairwise | 128 | 2 | 64 | 1 | mvkit-python | 56.006508 (55.978785 to 56.390369) | 14.738175 (14.621508 to 14.867142) | 3.80x |
| pairwise | 128 | 2 | 64 | 4 | mvkit-core | 23.617150 (22.802738 to 23.642770) | 8.107990 (7.856158 to 8.706573) | 2.91x |
| pairwise | 128 | 2 | 64 | 4 | mvkit-python | 25.721552 (22.599987 to 26.436137) | 8.027597 (7.937211 to 8.768358) | 3.20x |
| pairwise | 128 | 2 | 128 | 1 | mvkit-core | 115.998569 (114.151058 to 118.044684) | 29.877180 (29.525717 to 30.357611) | 3.88x |
| pairwise | 128 | 2 | 128 | 1 | mvkit-python | 119.565227 (116.608873 to 125.237610) | 30.628120 (29.818652 to 30.695911) | 3.90x |
| pairwise | 128 | 2 | 128 | 4 | mvkit-core | 43.861705 (42.978543 to 45.412067) | 17.436056 (16.493101 to 18.128131) | 2.52x |
| pairwise | 128 | 2 | 128 | 4 | mvkit-python | 43.030275 (40.750482 to 48.636432) | 15.440911 (15.409088 to 16.837982) | 2.79x |
