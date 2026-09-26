# Competitive particle benchmark

This suite establishes a reproducible CPU baseline for MVKit's particle
simulation engine. It compares implementations of **the same fixed-grid
Euler-Maruyama method**, using identical precomputed normal increments and
float64 states. It is the first stage of the
[performance roadmap](../docs/performance-roadmap.md).

The [initial CPU study](results/2026-09-26-findings.md) includes the raw
measurements, observed performance gaps and next profiling priorities.

## Problems

All cases use T=1, a=-0.5, b=0.3, sigma=0.25 and, for Cucker-Smale, beta=0.4.
Every particle-coordinate pair has its own independent Brownian increment.

| Case | Equation | Pilot size | Initial distribution |
|---|---|---|---|
| moments | dX_i = (a X_i + b mean(X)) dt + sigma dW_i | N=4096, d=1 | Standard normal |
| pairwise | dx_i = v_i dt; dv_i = N^-1 sum_j (1+abs(x_j-x_i)^2)^(-beta)(v_j-v_i) dt + sigma dW_i | N=128, spatial d=2 | Independent standard normal positions and velocities |
| multiplicative | dX_i^k = (a X_i^k + b mean_j(X_j^k)) dt + sigma X_i^k dW_i^k | N=2048, d=4 | exp(0.2 Z), Z standard normal |

The multiplicative case is a benchmark-local implementation of the public
Rust `MeanFieldSDE` trait. MVKit's Python wrappers do not yet expose this
custom model. The `mvkit-python` backend records it as unsupported.
The pairwise case computes every interaction exactly; no cutoff or random
batch approximation is used. Its state width is twice its spatial dimension.

## Install and build

Use a separate Python 3.12 environment for the pinned benchmark dependencies.
These packages are not runtime dependencies of MVKit.

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r benchmarks/requirements.lock
# Optionally copy benchmarks/locks/Cargo.lock to Cargo.lock first.
maturin develop --release
cargo build --release -p mvkit-core --example particle_benchmark
```

The Rust executable is built in release mode with the workspace's LTO
configuration. The dependency snapshot used for the initial study is
`benchmarks/locks/Cargo.lock`; copy it to the workspace `Cargo.lock` before
building to reproduce those versions. Record the Rust compiler version and
any custom `RUSTFLAGS` when comparing subsequent builds. The default build
does not request machine-specific target features.

For SciML, install Julia, then prepare its separate project:

```bash
julia --project=benchmarks/sciml -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

The checked-in Julia manifest records the tested environment. Precompile
dependencies before measuring; the first solver call still includes
specialization for the benchmark problem. Julia package installation and
package precompilation are not solver performance measurements.

## Run

Quick validation of the Rust engine and independent NumPy implementation:

```bash
python benchmarks/run_particle_benchmarks.py --profile smoke \
  --backends mvkit-core numpy --seeds 7 --steps 8 --repeats 2 \
  --output benchmarks/local-results/smoke
```

CPU pilot with all implementations:

```bash
python benchmarks/run_particle_benchmarks.py --profile pilot \
  --backends mvkit-core mvkit-python numpy numba diffrax sciml \
  --threads 1 4 --seeds 7 19 41 --steps 64 128 --repeats 7 \
  --output benchmarks/local-results/pilot
python benchmarks/summarize_particle_benchmarks.py benchmarks/local-results/pilot
```

Use `--julia PATH` if Julia is not on PATH. `--cases` selects a subset.
Use `--particles N` to study larger or smaller systems; it overrides the
particle count for every selected case. The exact pairwise case costs
O(N^2) per step, so select cases deliberately for large-N studies.
Use a new output directory for each run. A failure is retained in the raw
results and makes the command exit unsuccessfully. Optional dependencies
must be installed for the selected backends; missing packages are failures,
not silent exclusions. Avoid running other CPU-heavy work during a study.

On Linux, each worker inherits an affinity mask limited to the requested
logical CPU budget. Rayon, Numba and Julia thread counts are configured;
BLAS is restricted to one thread. JAX is restricted to CPU execution and
float64, and shares the same affinity budget. Elsewhere, affinity may be
unavailable and is explicitly recorded. A CPU budget is not a claim that
every backend uses exactly that many threads.

## Measurement contract

- A fresh process is launched for every case, resolution, seed, CPU budget
  and backend. Backends run sequentially, with reproducibly shuffled order.
- Input generation, file I/O, imports and input preparation are excluded
  from the solver timer. First-call timing includes solver compilation and
  lazy initialization. Warm timings start after that first call.
- Every solve returns the initial and final particle states. Intermediate
  trajectories and differentiation are not requested.
- The Rust core timer includes its normal input clone, workspace allocation
  and output allocation. The Python wrapper timer additionally includes its
  existing conversion/copy costs. The two interfaces have separate labels.
- Diffrax receives device-resident inputs before timing; transfer time is
  recorded separately. Each timed call is synchronized with
  `jax.block_until_ready`. The diffusion is represented as a diagonal linear
  operator, not a dense N*d by N*d matrix.
- SciML uses `EM(false)`: its default split Euler variant would evaluate
  multiplicative diffusion at a different state. It uses `NoiseGrid` on the
  prescribed time grid and saves only the first and last state.
- The NumPy reference uses blocks for all-pairs interactions to bound
  temporary storage. Numba uses compiled parallel loops without `fastmath`.
- Raw warm samples, first-call time, total process wall time, input transfer
  time and backend versions are saved. Total process wall time includes
  imports, input loading, all repetitions and output serialization.
- Each worker reports its process peak RSS when supported (Linux `/proc`
  high-water mark for Rust, `getrusage` for Python, `Sys.maxrss` for Julia).
  Parent-side sampling every 5 ms is also recorded when available, and is
  the fallback for the report. Sampled peaks may miss short allocations.
  Both include runtime, inputs and compilation; neither is solver-only
  working memory. Unavailable values are null, not zero.

## Numerical validation and error

The finest grid has `max(steps) * reference_factor` steps; the default factor
is four. All coarse increments are obtained by summing groups of fine
standard normals and dividing by the square root of the group size.
This preserves the Brownian path at shared times.

Each backend must agree with a separate NumPy Euler implementation on its
own grid to `atol=rtol=3e-11`. Nonfinite outputs and shape mismatches fail.
Timing results from failed validation must not be used to claim a speedup.

`reference_rmse` is the root mean square endpoint difference over particles
and coordinates relative to the coupled fine Euler trajectory. It is an
empirical finite-particle discretization comparison, not a certified error
bound, propagation-of-chaos error, or confidence interval. Interacting
particles must not be treated as independent replicates.

The comparison also records the difference between the reference and a
reference with half as many steps. This makes reference resolution visible;
it does not certify that the reference is exact. Seeds give independent
system replicates. The reported mean discrepancy is a paired sample mean,
not an exact weak bias estimate. Tests additionally verify an exact
deterministic mean-field solution, conservation of the Cucker-Smale mean
velocity, and convergence to exact independent geometric Brownian endpoints.

## Reading results

Each output directory contains:

- `metadata.json`: protocol, settings, package versions, CPU information,
  source revision and source fingerprint.
- `results.jsonl`: one raw row per attempted configuration, including
  unsupported and failed cases.
- `summary.md`: tables generated by the summarizer, with all raw measurements
  retained in the JSONL file.

Compare the same case, particle count, dimension, output contract, CPU
budget and error level. First-call latency and warm throughput answer
different questions. Small cases are useful for finding overhead but do
not establish large-scale performance. Container CPU quotas and host
contention can affect measurements.

This initial suite excludes native RNG generation, GPU execution, adaptive
and higher-order solvers, and statistically calibrated time-to-accuracy
comparisons. A later work-precision study must include those capabilities
before drawing conclusions about the best library for a complete task.

## Adapter references

- [Diffrax terms and structured diffusion](https://docs.kidger.site/diffrax/api/terms/)
- [Diffrax solver and output controls](https://docs.kidger.site/diffrax/api/diffeqsolve/)
- [SciML stochastic methods](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/)
- [DiffEqNoiseProcess noise grids](https://docs.sciml.ai/DiffEqNoiseProcess/stable/abstract_noise_processes/)

The adapters are validated against the installed implementations, including
SciML's split-step default, rather than relying on solver names alone.
