# mvkit

Fast McKean-Vlasov particle simulation. Rust core, Python API.

[![CI](https://github.com/yourusername/mvkit/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mvkit/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

`mvkit` simulates systems of interacting particles whose dynamics depend on the empirical distribution of the population, i.e. mean-field SDEs of McKean-Vlasov type:

$$
\mathrm{d}X^{i,N}_t = b\!\left(X^{i,N}_t, \tfrac{1}{N}\sum_{j=1}^N \delta_{X^{j,N}_t}\right)\mathrm{d}t + \sigma\,\mathrm{d}W^i_t.
$$

When $N \to \infty$, each particle converges in law to the McKean-Vlasov SDE thanks to the propagation of chaos. `mvkit` lets you simulate such systems efficiently from Python while keeping the heavy lifting in Rust.

## Status

Early alpha (v0.1). The current scope:

- Generic mean-field SDE trait (`MeanFieldSDE`) with state-dependent diagonal diffusion
- Euler-Maruyama and Milstein integrators with Rayon-parallel particle updates. Selected via `scheme="euler"` (default) or `scheme="milstein"` on every `simulate_*` entry point. On constant-diffusion models, Milstein reduces to Euler exactly because the diffusion derivative is zero; the two schemes produce bit-exact identical trajectories there.
- Built-in **Cucker-Smale** flocking model (any spatial dimension)
- Built-in **linear-quadratic** McKean-Vlasov model with closed-form Gaussian moments, used as a quantitative weak-order benchmark for the integrator
- Built-in **Kuramoto** model of coupled phase oscillators with O(N) drift via the order-parameter trick
- Built-in **mean-field Cox-Ingersoll-Ross** model with square-root diffusion: the first model with non-trivial state-dependent diffusion, used to exercise the Milstein correction term
- Reproducible seeded RNG (Xoshiro256++)
- PyO3 bindings, abi3 wheels for Python 3.9+

Roadmap (non-binding) for v0.2 and beyond: tamed schemes, kernel-based interactions via FFT, propagation-of-chaos rate estimation tools, strong-error tests via shared Brownian increments, MFG fixed-point iterations.

## Install (from source)

```bash
git clone https://github.com/yourusername/mvkit
cd mvkit
pip install maturin
maturin develop --release
```

Once a release is published:

```bash
pip install mvkit
```

## Quick start

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

See `examples/cucker_smale_demo.py` for a runnable visualization.

## Math summary: linear-quadratic McKean-Vlasov

Each particle has scalar state with dynamics

$$
\mathrm{d}X_i = (a\, X_i + b\, \bar X)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

where $\bar X = (1/N)\sum_j X_j$. If the initial law is Gaussian, the marginal law stays Gaussian for all $t$, with mean $m(t) = m_0 \exp((a+b) t)$ and variance $v(t) = v_0 \exp(2at) + \sigma^2 (\exp(2at) - 1)/(2a)$. These closed-form moments make the model a clean benchmark for the weak order of any integrator; the test suite fits the log-log slope of $|E[\bar X^h_T] - m(T)|$ and the analogous variance error against $\mathrm{d}t$, and asserts a slope of 1 (Talay and Tubaro, 1990).

## Math summary: Cucker-Smale

Per particle the state is $(x_i, v_i) \in \mathbb{R}^{2d}$. The dynamics:

$$
\mathrm{d}x_i = v_i\,\mathrm{d}t,\qquad
\mathrm{d}v_i = \frac{1}{N}\sum_{j=1}^N K(|x_j - x_i|)(v_j - v_i)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

with kernel $K(r) = (1 + r^2)^{-\beta}$. The deterministic part conserves the mean velocity $\bar v = \frac{1}{N}\sum_i v_i$, used as a sanity check in the test suite. For $\beta < 1/2$, velocities concentrate around $\bar v$ unconditionally (Cucker and Smale, 2007).

## Math summary: Kuramoto

Each particle has a scalar phase $\theta_i \in \mathbb{R}$ (left unwrapped during integration; the user can reduce mod $2\pi$ for visualization). The dynamics:

$$
\mathrm{d}\theta_i = \omega_i\,\mathrm{d}t + \frac{K}{N}\sum_{j=1}^N \sin(\theta_j - \theta_i)\,\mathrm{d}t + \sigma\,\mathrm{d}W_i,
$$

where $\omega_i$ are heterogeneous natural frequencies and $K$ is the coupling strength. Synchronization is captured by the Kuramoto order parameter

$$
r(t)\, e^{i\psi(t)} = \frac{1}{N}\sum_{j=1}^N e^{i\theta_j(t)},
$$

with $r \in [0, 1]$. For Gaussian $\omega_i \sim N(0, \sigma_\omega^2)$ the classical critical coupling is $K_c = 2\sigma_\omega \sqrt{2/\pi}$. Below $K_c$ the population stays incoherent ($r \to 0$ as $N \to \infty$); above $K_c$ a fraction of oscillators lock and $r$ stabilizes between 0 and 1, with the mean-field prediction $r_\infty \approx \sqrt{1 - K_c / K}$ for $K$ slightly above $K_c$.

The drift is implemented in $O(N)$ per time step via the trig identity $\sum_j \sin(\theta_j - \theta_i) = S\cos(\theta_i) - C\sin(\theta_i)$ with $C = \sum_j \cos(\theta_j)$ and $S = \sum_j \sin(\theta_j)$: a sequential reduction over the phase column gives $C, S$, then a parallel-over-rows application sets each particle's drift in constant time. A naive nested-loop form is intentionally avoided; on the test cases it would be ~1000x slower.

Reference: Kuramoto, Y. (1975). *Self-entrainment of a population of coupled non-linear oscillators*. International Symposium on Mathematical Problems in Theoretical Physics.

See `examples/kuramoto_demo.py` for a runnable visualization showing phase trajectories and $r(t)$ on either side of $K_c$.

## Math summary: mean-field CIR

Each particle has scalar state $X_i \ge 0$ with dynamics

$$
\mathrm{d}X_i = \kappa(\theta - X_i)\,\mathrm{d}t + b\,(\bar X - X_i)\,\mathrm{d}t + \sigma\sqrt{\max(X_i, 0)}\,\mathrm{d}W_i,
$$

where $\bar X = (1/N)\sum_j X_j$. The first drift term is the standard CIR mean-reversion towards $\theta$ at rate $\kappa$; the second is the McKean-Vlasov interaction. The diffusion is square-root in the state, which is what makes Milstein non-trivial on this model: the diagonal Jacobian $(\sigma\sqrt{x})' = 0.5 \sigma / \sqrt{x}$ is non-zero, so the Milstein correction term $\tfrac{1}{2}\sigma\sigma'\,\mathrm{d}t (Z^2 - 1)$ fires and modifies trajectories pathwise.

The truncation $\max(X_i, 0)$ keeps the diffusion real if a discretization step underflows below zero. The Feller condition $2\kappa\theta \ge \sigma^2$ guarantees that the continuous-time process stays strictly positive.

In the limit $b \to 0$, each particle is an independent classical CIR process and the marginal mean satisfies the closed form $\mathbb{E}[X_t] = \theta + (X_0 - \theta) e^{-\kappa t}$, used in the test suite as a smoke check of both schemes.

A note on Milstein's improvement. Milstein has strong order 1 (vs Euler's strong order 1/2), but on weak error of smooth functionals of $X_T$ both schemes are order 1; the constants of the leading $O(\mathrm{d}t)$ terms can go either way depending on the functional, and on CIR the Milstein-only contribution to $E[X_{n+1}^2 \mid X_n]$ is $+\tfrac{1}{8}\sigma^4\,\mathrm{d}t^2$, i.e. very slightly worse on the second moment by an $O(\mathrm{d}t^2)$ amount. The strong-order-1 improvement only shows up on pathwise error or non-smooth functionals (barrier hits, trajectory maxima). A clean strong-error test on CIR requires sharing Brownian increments between coarse-dt and fine-dt runs, which we have deferred.

References: Cox, J. C., Ingersoll, J. E., and Ross, S. A. (1985). *A theory of the term structure of interest rates*. Econometrica 53, 385-407. McKean-Vlasov extensions are standard, see Carmona and Delarue (2018).

## Propagation of chaos

For a McKean-Vlasov SDE with i.i.d. initial conditions, Sznitman's classical result (1991) says the empirical measure $\mu_N$ of the $N$-particle system converges in distribution to the McKean-Vlasov limit law $\mu$ as $N \to \infty$. Fournier and Guillin (2015) sharpened this to a quantitative rate: in dimension 1, for laws with finite $(4 + \varepsilon)$-th moment, $\mathbb{E}[W_2(\mu_N, \mu)] = O(N^{-1/2})$, where $W_2$ is the 2-Wasserstein distance.

`mvkit.poc` provides a small utility that takes any 1D mean-field simulator wrapped as `(n_particles, seed) -> 1D array of terminal-time states`, sweeps `N`, computes the exact 1D Wasserstein-2 distance to a reference inverse CDF on each run, and fits the log-log slope of the median against $\log N$. The slope should land near $-1/2$:

```python
import numpy as np
from scipy.stats import norm

from mvkit import simulate_linear_quadratic
from mvkit.poc import estimate_propagation_of_chaos_rate

a, b, sigma, T = -0.5, 1.0, 0.5, 1.0
m_0, v_0 = 0.0, 1.0
m_T = m_0 * np.exp((a + b) * T)
v_T = v_0 * np.exp(2 * a * T) + sigma**2 * (np.exp(2 * a * T) - 1) / (2 * a)

def simulator(n, seed):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(m_0, np.sqrt(v_0), size=(n, 1))
    h = simulate_linear_quadratic(x0, T, 1000, a=a, b=b, sigma=sigma, seed=seed)
    return h[-1, :, 0]

result = estimate_propagation_of_chaos_rate(
    simulator=simulator,
    reference_inv_cdf=norm(loc=m_T, scale=np.sqrt(v_T)).ppf,
    n_values=[100, 300, 1000, 3000, 10000],
    n_seeds=16,
)
print(result.fitted_slope)  # should be ~ -0.5
```

See `examples/poc_rate_lq.py` for a runnable two-panel figure showing the log-log fit and the empirical-vs-analytical CDF overlay at the largest $N$.

Scope. Currently 1D only (LinearQuadratic, Kuramoto via the order parameter, MeanFieldCIR). Cucker-Smale state is 4D, which requires sliced or projected Wasserstein and is on the v0.2 roadmap.

## Benchmarks

Criterion benchmarks track Euler-Maruyama throughput in particle-steps per second. Run them from the workspace root:

```bash
cargo bench --bench integrators
```

Each suite reports throughput via `Throughput::Elements(N * n_steps)` so Criterion prints the unit directly. HTML reports land under `target/criterion/` and are gitignored alongside the rest of `target/`. Indicative numbers on an Apple-silicon laptop, single process:

- Linear-quadratic (cheap drift): ~14 M particle-steps/s at N=1000, ~96 M at N=10000, ~316 M at N=100000. The sub-linear region at small N is dominated by sequential noise sampling and parallel-launch overhead; once N is large enough to amortize that, the integrator scales near-linearly with the rayon pool.
- Cucker-Smale (O(N^2) pairwise drift): ~940 K at N=100, ~460 K at N=500, ~130 K at N=2000. The drift cost dominates once N grows; an FFT-convolution path for translation-invariant kernels is on the roadmap.

The benchmarks are not part of CI by default; they are too noisy on shared GitHub runners. A manual workflow at `.github/workflows/bench.yml` (triggered via the Actions tab) runs them on `ubuntu-latest` and uploads the HTML report as an artifact.

## Architecture

```
mvkit/
├── crates/
│   ├── mvkit-core/   # pure Rust: traits, integrators, models
│   └── mvkit-py/     # PyO3 bindings, no business logic
├── python/mvkit/     # Python package, wraps _core
├── tests/            # pytest test suite
└── examples/         # runnable demos
```

The split between `mvkit-core` and `mvkit-py` keeps the FFI surface thin and lets the core crate be reused from pure Rust. The Python wrapper layer adds input validation and ergonomic defaults without touching the Rust ABI.

## Contributing

PRs welcome. Before submitting, please run:

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
maturin develop
pytest
```

## References

- Cucker, F. and Smale, S. (2007). *Emergent behavior in flocks*. IEEE Trans. Automatic Control.
- Sznitman, A.-S. (1991). *Topics in propagation of chaos*. Ecole d'Eté de Probabilités de Saint-Flour XIX.
- Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications I & II*. Springer.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
