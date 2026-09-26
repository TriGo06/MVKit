# Performance roadmap

MVKit aims to become a reference implementation for stochastic interacting
particle systems. Progress is measured by time and memory at a specified
numerical accuracy, on reproducible problems and hardware configurations.
The library remains general purpose across applications of these systems.

## 1. Establish a competitive baseline

The [particle benchmark](../benchmarks/README.md) compares the Rust core,
the available Python wrappers, NumPy, a compiled Numba implementation,
Diffrax, and SciML on the same equations and Brownian increments.

This first stage compares fixed-grid Euler implementations. A later
work-precision study must allow each library to select suitable higher-order
or adaptive methods. It must also include native random generation and
statistical estimation costs. Fixed-grid timing alone cannot establish
overall superiority.

Acceptance criteria:

- Shared-noise numerical agreement before publishing a timing.
- Raw measurements, dependency versions, source fingerprint and machine details.
- Multiple seeds, grid resolutions and CPU budgets; separate first-call and warm costs.
- Coupled reference refinement, with reference uncertainty made visible.
- Explicit reporting of unsupported models, failures and missing measurements.

## 2. Optimize measured costs

Use profiles to identify which costs dominate each problem size. Candidates
include temporary allocation, memory traffic, redundant diffusion work,
parallel scheduling and random generation. Preserve numerical convergence
and define any changes to reproducibility before changing algorithms.

Accept optimizations against representative cases, including small problems
where scheduling overhead matters. Track regressions as well as improvements.
Keep CPU and GPU measurements separate and account for data transfers.

## 3. Expose interaction structure and custom models

Design a model interface that distinguishes moment interactions, all-pairs
kernels and local interactions. Provide a compiled execution path for user
models. Benchmark-only implementations of a custom model do not constitute
a supported Python model API.

Approximate interaction algorithms, such as random batching or grid-based
convolution, need an explicit approximation contract and error measurements.
The choice of approximation belongs in the user-visible numerical method.

## 4. Optimize the complete simulation task

Develop streaming observables, reproducible parallel random streams and
methods selected for the requested observable and accuracy. Investigate
particle count, discretization error and sampling uncertainty together.
Validate proposed error estimators on problems with analytical or independently
verified references before using them to select simulation parameters.

GPU execution and additional numerical schemes are evaluated against these
criteria. Each published performance claim must name its problem class,
hardware, precision, numerical method and scope of measurement.
