# mvkit

McKean-Vlasov particle simulation with a Rust core and a Python API.

[![CI](https://github.com/TriGo06/MVKit/actions/workflows/ci.yml/badge.svg)](https://github.com/TriGo06/MVKit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

`mvkit` simulates interacting stochastic particles whose dynamics depend on
the population distribution. It also provides numerical Mean Field Game
(MFG) solvers in Python, for studying equilibria between individual controls
and population dynamics.

**Status: early alpha, v0.1.** The current development priority is particle
simulation performance, reproducibility and custom model support. See the
[performance roadmap](docs/performance-roadmap.md) and [changelog](CHANGELOG.md).

[Installation](#installation) · [Quick start](#quick-start) ·
[Particle models](#particle-models) · [MFG solvers](#mean-field-games) ·
[Documentation](#documentation) · [Tests](#tests-and-contributing)

## Installation

Install from this repository with:

- **Python 3.9+** with `pip` and `venv`, and **Git**.
- **Rust stable and Cargo**, installed with [rustup](https://rust-lang.org/tools/install/).
- A native build toolchain:
  - **Windows:** Visual Studio Build Tools with **Desktop development with C++**
    (MSVC and a Windows SDK), using the Rust MSVC toolchain.
  - **macOS:** Xcode Command Line Tools.
  - **Linux:** a C/C++ compiler and linker (for example, `build-essential`
    on Debian/Ubuntu).

Open a new terminal after installing the tools. Check that `python --version`
(or `python3 --version` on macOS/Linux), `cargo --version` and `git --version`
work before continuing.

### Windows (PowerShell)

```powershell
git clone https://github.com/TriGo06/MVKit.git
cd MVKit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "maturin>=1.7,<2.0"
maturin develop --release
```

If PowerShell blocks activation, use the virtual environment's interpreter
directly to build and install the package:

```powershell
.\.venv\Scripts\python.exe -m pip install .
```

After that, use `.\.venv\Scripts\python.exe` to run examples in that environment.

### macOS / Linux

```bash
git clone https://github.com/TriGo06/MVKit.git
cd MVKit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "maturin>=1.7,<2.0"
maturin develop --release
```

Maturin builds the Rust extension and installs the package with its NumPy and
SciPy runtime dependencies. The activated environment is required by
`maturin develop`; see the [Maturin tutorial](https://www.maturin.rs/tutorial).
Use `--release` for simulations and performance measurements.

## Quick start

Simulate a flock in two spatial dimensions:

```python
import numpy as np
from mvkit import simulate_cucker_smale

rng = np.random.default_rng(0)
n, d = 500, 2
x0 = np.empty((n, 2 * d))
x0[:, :d] = rng.normal(scale=2.0, size=(n, d))   # initial positions
x0[:, d:] = rng.normal(scale=1.5, size=(n, d))   # initial velocities

history = simulate_cucker_smale(
    x0,
    t_final=10.0,
    n_steps=2000,
    spatial_dim=d,
    beta=0.4,
    sigma=0.05,
    record_every=20,
    seed=42,
)
print(history.shape)  # (101, 500, 4)
```

The state stores positions first, then velocities. Here each saved frame has
500 particles and four coordinates. `record_every=20` saves every twentieth
step; the initial state is included. Set `record_every=0` to keep only the
initial and final states.

For plots, install the optional development dependencies with
`maturin develop --release --extras dev`, then run
`python examples/cucker_smale_demo.py` in the activated environment.

## Particle models

All four functions are available from `mvkit` and accept
`scheme="euler"` (default) or `scheme="milstein"`, `seed`,
`record_every` and optional shared normal `increments`.

| Model | Function | Initial state shape |
|---|---|---|
| Cucker-Smale flocking | `simulate_cucker_smale` | `(N, 2 * spatial_dim)`, with `spatial_dim >= 1` |
| Linear-quadratic mean-field dynamics | `simulate_linear_quadratic` | `(N, 1)` |
| Kuramoto phase oscillators | `simulate_kuramoto` | `(N, 1)` |
| Mean-field Cox-Ingersoll-Ross (CIR) | `simulate_mean_field_cir` | `(N, 1)`, finite non-negative values |

The Rust core provides Rayon-parallel particle updates and a `MeanFieldSDE`
trait for custom Rust models. Python currently exposes the built-in models.
Kuramoto uses an `O(N)` drift evaluation; Cucker-Smale evaluates all pairs
in `O(N²)` per step.

For constant diffusion, Euler and Milstein produce identical trajectories
with the same inputs and noise. CIR exercises the state-dependent Milstein
correction; neither scheme guarantees non-negative discrete CIR states.

`mvkit.brownian` generates and coarsens shared normal increments for pathwise
comparisons. `mvkit.poc` measures 1D Wasserstein distances and fits empirical
particle-count convergence rates. See the
[numerical guide](docs/numerical-guide.md) for assumptions, limitations and
examples, and the [reproducibility contract](docs/reproducibility.md) for
the scope of seeded results.

## Mean Field Games

Choose a solver from `mvkit.mfg`:

| Problem | Function | Numerical approach |
|---|---|---|
| Scalar linear-quadratic (LQ) | `solve_lq_mfg` | Particle response iteration with scalar Riccati and variance ODEs |
| Vector LQ | `solve_lq_mfg_vector` | Particle response iteration with matrix Riccati and covariance ODEs |
| General costs, 1D state | `solve_mfg` | Coupled HJB / Fokker-Planck grid |
| General costs, 2D state | `solve_mfg_2d` | Coupled HJB / Fokker-Planck grid |

All four support `method="picard"` and `method="fictitious_play"`.
LQ reference trajectories are computed by numerical ODE integration;
closed-form scalar cases serve as regression checks.

Both grid solvers support periodic and Neumann boundaries. The 1D solver
accepts custom convex Hamiltonians satisfying the documented normalization
conditions; the 2D solver uses a quadratic Hamiltonian and supports
`sigma=(sigma_x, sigma_y)` for anisotropic diffusion.
Standalone HJB and Fokker-Planck solvers are also available in 1D and 2D.

Check `converged` and `fixed_point_residual` before interpreting a solution.
See [MFG methods and example](docs/numerical-guide.md#mean-field-games)
for convergence criteria, time-step constraints and the Hamiltonian contract.

## Documentation

- [Numerical guide](docs/numerical-guide.md): model equations, Wasserstein
  distances, strong-error experiments, MFG methods and references.
- [Random streams and reproducibility](docs/reproducibility.md).
- [Competitive particle benchmark](benchmarks/README.md).
- [Performance roadmap](docs/performance-roadmap.md).
- [Runnable examples](examples): flocking, synchronization, convergence rates
  and scalar/grid MFG studies.

## Tests and contributing

From the repository root, with the virtual environment activated:

```bash
maturin develop --release --extras dev
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
python -m pytest -v tests/
```

The `dev` extra installs pytest, matplotlib and tqdm for tests and demos.
Some comparison tests require optional benchmark backends; their setup is
documented in the [benchmark guide](benchmarks/README.md).

[CI](https://github.com/TriGo06/MVKit/actions/workflows/ci.yml) runs Rust checks
on Linux, macOS and Windows, and Python tests on 3.9 and 3.12 across those
platforms. It also checks installation of the built wheel in a fresh environment.

## Benchmarks

The [benchmark guide](benchmarks/README.md) defines the problems, shared
noise, dependency snapshots and measurement protocol. Published studies:

- [CPU baseline, 26 September 2026](benchmarks/results/2026-09-26-findings.md).
- [Kernel optimization study, 26 September 2026](benchmarks/results/2026-09-26-kernel-optimization/README.md).

Results apply to the recorded hardware, revision, CPU budget and numerical
method. Use the supplied scripts to measure your own workload.

For the Rust Criterion microbenchmarks:

```bash
cargo bench --bench integrators
```

HTML reports are written to `target/criterion/`. The
[manual benchmark workflow](.github/workflows/bench.yml) uploads those reports.
A separate [particle benchmark workflow](.github/workflows/particle-benchmarks.yml)
checks shared-noise agreement on relevant pull requests; its timings are not
a performance guarantee.

## Project layout

```text
crates/mvkit-core/   Rust models, traits and integrators
crates/mvkit-py/     Python bindings
python/mvkit/       Python API, Brownian helpers and Wasserstein utilities
python/mvkit/mfg/   Scalar/vector LQ and 1D/2D grid MFG solvers
tests/             Python tests
examples/          Runnable demonstrations
docs/              Numerical guide, reproducibility and roadmap
benchmarks/        Comparative benchmarks and published results
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE),
at your option.
