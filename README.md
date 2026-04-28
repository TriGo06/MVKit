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

- Generic mean-field SDE trait (`MeanFieldSDE`)
- Euler-Maruyama integrator with Rayon-parallel particle updates
- Built-in **Cucker-Smale** flocking model (any spatial dimension)
- Built-in **linear-quadratic** McKean-Vlasov model with closed-form Gaussian moments, used as a quantitative weak-order benchmark for the integrator
- Reproducible seeded RNG (Xoshiro256++)
- PyO3 bindings, abi3 wheels for Python 3.9+

Roadmap (non-binding) for v0.2 and beyond: Milstein and tamed schemes, kernel-based interactions via FFT, Kuramoto, propagation-of-chaos rate estimation tools, MFG fixed-point iterations.

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
