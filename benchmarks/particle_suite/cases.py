"""Shared equations, inputs, and an independently vectorized reference."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    name: str
    n: int
    dim: int
    steps: int
    t_final: float = 1.0
    a: float = -0.5
    b: float = 0.3
    sigma: float = 0.25
    beta: float = 0.4

    def __post_init__(self):
        if self.name not in ("moments", "pairwise", "multiplicative"):
            raise ValueError(f"Unknown case: {self.name}")
        if min(self.n, self.dim, self.steps) < 1:
            raise ValueError("n, dim and steps must be positive")
        if self.name == "moments" and self.dim != 1:
            raise ValueError("The built-in linear-quadratic benchmark is scalar")
        if not np.isfinite([self.t_final, self.a, self.b, self.sigma, self.beta]).all():
            raise ValueError("Parameters must be finite")
        if self.t_final <= 0 or self.sigma < 0 or self.beta < 0:
            raise ValueError("Invalid horizon, diffusion or kernel exponent")

    @property
    def width(self):
        return 2 * self.dim if self.name == "pairwise" else self.dim

    def to_dict(self):
        return asdict(self)


def inputs(case, seed):
    """Generate raw standard normals; every backend applies sqrt(dt) itself."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(case.n, case.width))
    if case.name == "multiplicative":
        x0 = np.exp(0.2 * x0)
    z = rng.standard_normal((case.steps, case.n, case.width))
    return x0, z


def coarsen(z, factor):
    if factor < 1 or z.shape[0] % factor:
        raise ValueError("Coarsening factor must divide the number of increments")
    return z.reshape(z.shape[0] // factor, factor, *z.shape[1:]).sum(axis=1) / np.sqrt(factor)


def drift(case, x):
    if case.name != "pairwise":
        return case.a * x + case.b * x.mean(axis=0)
    # Blocks bound temporary storage without truncating the all-pairs sum.
    d = case.dim
    out = np.empty_like(x)
    out[:, :d] = x[:, d:]
    for start in range(0, case.n, 64):
        stop = min(start + 64, case.n)
        delta = x[None, :, :d] - x[start:stop, None, :d]
        weights = (1 + np.sum(delta * delta, axis=-1)) ** (-case.beta)
        dv = x[None, :, d:] - x[start:stop, None, d:]
        out[start:stop, d:] = np.sum(weights[..., None] * dv, axis=1) / case.n
    return out


def diffusion(case, x):
    if case.name == "multiplicative":
        return case.sigma * x
    if case.name == "pairwise":
        return np.r_[np.zeros(case.dim), np.full(case.dim, case.sigma)]
    return case.sigma


def numpy_solve(case, x0, z):
    dt = case.t_final / case.steps
    x = x0.copy()
    for noise in z:
        x += drift(case, x) * dt + diffusion(case, x) * np.sqrt(dt) * noise
    return np.stack((x0, x))


def independent_geometric_exact(case, x0, z):
    """Exact pathwise endpoint for the multiplicative case with b=0 only."""
    if case.name != "multiplicative" or case.b != 0:
        raise ValueError("Exact formula requires independent geometric diffusions")
    w = np.sqrt(case.t_final / case.steps) * z.sum(axis=0)
    return x0 * np.exp((case.a - 0.5 * case.sigma**2) * case.t_final + case.sigma * w)
