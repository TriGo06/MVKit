"""mvkit: fast McKean-Vlasov particle simulation.

The numerical core is implemented in Rust and exposed via PyO3. This
top-level module provides an ergonomic Python API on top of the raw
bindings in :mod:`mvkit._core`.
"""

from __future__ import annotations

import numpy as np

from . import _core

__version__ = _core.__version__
__all__ = ["simulate_cucker_smale", "simulate_linear_quadratic"]


def simulate_cucker_smale(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    spatial_dim: int = 2,
    beta: float = 0.5,
    sigma: float = 0.1,
    record_every: int = 0,
    seed: int = 42,
) -> np.ndarray:
    r"""Simulate the Cucker-Smale flocking model.

    Each particle has state :math:`(x, v) \in \mathbb{R}^{2d}` with
    :math:`d` = ``spatial_dim``. The dynamics are

    .. math::
        \mathrm{d}x_i &= v_i \, \mathrm{d}t \\
        \mathrm{d}v_i &= \frac{1}{N}\sum_j K(\lvert x_j - x_i \rvert)
                        (v_j - v_i)\, \mathrm{d}t + \sigma\, \mathrm{d}W_i

    with kernel :math:`K(r) = (1 + r^2)^{-\beta}`. The empirical mean
    velocity is conserved by the deterministic part, a property useful as
    a sanity check on the integrator.

    Parameters
    ----------
    x0 : ndarray, shape (N, 2 * spatial_dim)
        Initial state. Columns 0..d are positions, columns d..2d are
        velocities.
    t_final : float
        Final integration time.
    n_steps : int
        Number of Euler-Maruyama steps. ``dt = t_final / n_steps``.
    spatial_dim : int, default 2
        Spatial dimension :math:`d`.
    beta : float, default 0.5
        Kernel exponent. For :math:`\beta < 1/2` the model converges to
        consensus unconditionally (Cucker-Smale 2007).
    sigma : float, default 0.1
        Diffusion applied only to the velocity coordinates.
    record_every : int, default 0
        Record state every ``k`` steps. If 0, only the initial and final
        states are stored.
    seed : int, default 42
        RNG seed. Identical seeds give bit-exact identical trajectories.

    Returns
    -------
    history : ndarray, shape (n_recorded, N, 2 * spatial_dim)
        Recorded trajectories.
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D, got shape {x0.shape}")
    if x0.shape[1] != 2 * spatial_dim:
        raise ValueError(
            f"x0 has {x0.shape[1]} columns, expected 2*spatial_dim = "
            f"{2 * spatial_dim}"
        )
    return _core.simulate_cucker_smale(
        x0,
        float(t_final),
        int(n_steps),
        int(spatial_dim),
        float(beta),
        float(sigma),
        int(record_every),
        int(seed),
    )


def simulate_linear_quadratic(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    a: float,
    b: float,
    sigma: float,
    record_every: int = 0,
    seed: int = 42,
) -> np.ndarray:
    r"""Simulate the linear-quadratic McKean-Vlasov model.

    Each particle has scalar state with dynamics

    .. math::
        \mathrm{d}X_i = (a\, X_i + b\, \bar X)\, \mathrm{d}t
                       + \sigma\, \mathrm{d}W_i

    where :math:`\bar X = (1/N) \sum_j X_j` is the empirical mean of the
    population. If the initial law is Gaussian :math:`X_0 \sim N(m_0, v_0)`,
    the marginal law stays Gaussian for all :math:`t`, with mean and variance
    given by

    .. math::
        m(t) &= m_0 \exp((a + b) t) \\
        v(t) &= v_0 \exp(2 a t)
                + \sigma^2 \frac{\exp(2 a t) - 1}{2 a}

    (with the obvious :math:`v(t) = v_0 + \sigma^2 t` limit when
    :math:`a = 0`). The closed-form moments make this a useful benchmark for
    any mean-field integrator.

    Parameters
    ----------
    x0 : ndarray, shape (N, 1)
        Initial state. One scalar per particle, stored as a column vector.
    t_final : float
        Final integration time.
    n_steps : int
        Number of Euler-Maruyama steps. ``dt = t_final / n_steps``.
    a : float
        Self-coupling coefficient.
    b : float
        Mean-field coupling coefficient.
    sigma : float
        Constant scalar diffusion coefficient. Must be non-negative.
    record_every : int, default 0
        Record state every ``k`` steps. If 0, only the initial and final
        states are stored.
    seed : int, default 42
        RNG seed. Identical seeds give bit-exact identical trajectories.

    Returns
    -------
    history : ndarray, shape (n_recorded, N, 1)
        Recorded trajectories.

    References
    ----------
    Talay, D. and Tubaro, L. (1990). Expansion of the global error for
    numerical schemes solving stochastic differential equations.
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D, got shape {x0.shape}")
    if x0.shape[1] != 1:
        raise ValueError(
            f"x0 has {x0.shape[1]} columns, expected 1 (scalar state)"
        )
    return _core.simulate_linear_quadratic(
        x0,
        float(t_final),
        int(n_steps),
        float(a),
        float(b),
        float(sigma),
        int(record_every),
        int(seed),
    )
