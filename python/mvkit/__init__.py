"""mvkit: fast McKean-Vlasov particle simulation.

The numerical core is implemented in Rust and exposed via PyO3. This
top-level module provides an ergonomic Python API on top of the raw
bindings in :mod:`mvkit._core`.
"""

from __future__ import annotations

import numpy as np

from . import _core, poc

__version__ = _core.__version__
__all__ = [
    "poc",
    "simulate_cucker_smale",
    "simulate_kuramoto",
    "simulate_linear_quadratic",
    "simulate_mean_field_cir",
]


def simulate_cucker_smale(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    spatial_dim: int = 2,
    beta: float = 0.5,
    sigma: float = 0.1,
    record_every: int = 0,
    seed: int = 42,
    scheme: str = "euler",
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
    scheme : str, default "euler"
        Integrator to use: ``"euler"`` (Euler-Maruyama) or ``"milstein"``
        (Milstein). On constant-diffusion models like Cucker-Smale,
        Milstein reduces to Euler exactly because the diffusion derivative
        is zero; both schemes produce bit-exact identical trajectories.

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
        str(scheme),
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
    scheme: str = "euler",
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
        str(scheme),
    )


def simulate_kuramoto(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    coupling_k: float,
    omegas: np.ndarray,
    sigma: float,
    record_every: int = 0,
    seed: int = 42,
    scheme: str = "euler",
) -> np.ndarray:
    r"""Simulate the Kuramoto model of coupled phase oscillators.

    Each particle has a scalar phase :math:`\theta_i \in \mathbb{R}` (left
    unwrapped during integration; the user can reduce mod :math:`2\pi` for
    visualization). The dynamics are

    .. math::
        \mathrm{d}\theta_i = \omega_i\, \mathrm{d}t
            + \frac{K}{N}\sum_{j=1}^N \sin(\theta_j - \theta_i)\, \mathrm{d}t
            + \sigma\, \mathrm{d}W_i

    where :math:`\omega_i` are heterogeneous natural frequencies and
    :math:`K` is the coupling strength. Synchronization is captured by the
    Kuramoto order parameter

    .. math::
        r(t)\, e^{i \psi(t)} = \frac{1}{N}\sum_{j=1}^N e^{i \theta_j(t)},

    with :math:`r \in [0, 1]`. ``r(t)`` is **not** returned by this function;
    the user computes it from the trajectory, e.g.
    ``np.abs(np.exp(1j * history[..., 0]).mean(axis=1))``.

    Phase transition. For Gaussian :math:`\omega_i \sim N(0, \sigma_\omega^2)`
    with density :math:`g(\omega) = (\sigma_\omega \sqrt{2\pi})^{-1}
    \exp(-\omega^2 / (2 \sigma_\omega^2))`, the classical Kuramoto critical
    coupling is

    .. math::
        K_c = \frac{2}{\pi g(0)} = 2\, \sigma_\omega \sqrt{2 / \pi}.

    Below :math:`K_c` the population stays incoherent (:math:`r(\infty) \to 0`
    as :math:`N \to \infty`); above :math:`K_c` a fraction of oscillators lock
    and :math:`r(\infty) > 0`.

    The drift is implemented in :math:`O(N)` per step via the order-parameter
    trick (see the README), not :math:`O(N^2)`.

    Parameters
    ----------
    x0 : ndarray, shape (N, 1)
        Initial phases. One scalar per particle, stored as a column vector.
    t_final : float
        Final integration time.
    n_steps : int
        Number of Euler-Maruyama steps. ``dt = t_final / n_steps``.
    coupling_k : float
        Coupling strength :math:`K`.
    omegas : ndarray, shape (N,)
        Natural frequency per particle. Must have length equal to
        ``x0.shape[0]``.
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
        Recorded phase trajectories.

    References
    ----------
    Kuramoto, Y. (1975). Self-entrainment of a population of coupled
    non-linear oscillators. International Symposium on Mathematical Problems
    in Theoretical Physics.
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    omegas = np.ascontiguousarray(omegas, dtype=np.float64)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D, got shape {x0.shape}")
    if x0.shape[1] != 1:
        raise ValueError(
            f"x0 has {x0.shape[1]} columns, expected 1 (scalar phase)"
        )
    if omegas.ndim != 1:
        raise ValueError(f"omegas must be 1D, got shape {omegas.shape}")
    if omegas.shape[0] != x0.shape[0]:
        raise ValueError(
            f"omegas has length {omegas.shape[0]} but x0 has "
            f"{x0.shape[0]} rows; they must match"
        )
    return _core.simulate_kuramoto(
        x0,
        float(t_final),
        int(n_steps),
        float(coupling_k),
        omegas,
        float(sigma),
        int(record_every),
        int(seed),
        str(scheme),
    )


def simulate_mean_field_cir(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    kappa: float,
    theta: float,
    b: float,
    sigma: float,
    record_every: int = 0,
    seed: int = 42,
    scheme: str = "euler",
) -> np.ndarray:
    r"""Simulate the McKean-Vlasov Cox-Ingersoll-Ross model.

    Each particle has scalar state :math:`X_i \ge 0` with dynamics

    .. math::
        \mathrm{d}X_i = \kappa(\theta - X_i)\,\mathrm{d}t
            + b\,(\bar X - X_i)\,\mathrm{d}t
            + \sigma\, \sqrt{\max(X_i, 0)}\, \mathrm{d}W_i,

    where :math:`\bar X = (1/N)\sum_j X_j` is the empirical mean. The first
    drift term is the standard CIR mean-reversion towards :math:`\theta` at
    rate :math:`\kappa`; the second is the McKean-Vlasov interaction. The
    diffusion is square-root in the state, which makes Milstein non-trivial.

    The truncation :math:`\max(X_i, 0)` keeps the diffusion real if a
    discretization step underflows below zero. The Feller condition
    :math:`2\kappa\theta \ge \sigma^2` guarantees that the continuous-time
    process stays strictly positive, in which case the truncation is rarely
    activated.

    In the limit :math:`b = 0`, the interaction term vanishes and each
    particle is an independent classical CIR process. The marginal mean
    then satisfies the closed-form ODE solution

    .. math::
        \mathbb{E}[X_t] = \theta + (X_0 - \theta) \exp(-\kappa t),

    used as a quantitative benchmark for Milstein vs Euler in the test
    suite.

    Parameters
    ----------
    x0 : ndarray, shape (N, 1)
        Initial states. Should be non-negative.
    t_final : float
        Final integration time.
    n_steps : int
        Number of integration steps. ``dt = t_final / n_steps``.
    kappa : float
        Mean-reversion rate. Must be positive.
    theta : float
        Long-run mean. Must be positive.
    b : float
        Mean-field interaction coefficient. Set to 0 to recover independent
        classical CIR.
    sigma : float
        Diffusion strength on the square-root noise term. Must be positive.
    record_every : int, default 0
        Record state every ``k`` steps. If 0, only the initial and final
        states are stored.
    seed : int, default 42
        RNG seed. Identical seeds give bit-exact identical trajectories.
    scheme : str, default "euler"
        Integrator: ``"euler"`` for Euler-Maruyama or ``"milstein"`` for
        Milstein. Milstein has a smaller bias on this model thanks to its
        strong-order-1 correction term :math:`0.25 \sigma^2 dt (Z^2 - 1)`,
        which is non-trivial here because the diffusion derivative
        :math:`d/dx (\sigma \sqrt{x}) = 0.5 \sigma / \sqrt{x}` is non-zero.

    Returns
    -------
    history : ndarray, shape (n_recorded, N, 1)
        Recorded trajectories.

    References
    ----------
    Cox, J. C., Ingersoll, J. E., and Ross, S. A. (1985). A theory of the
    term structure of interest rates. Econometrica 53, 385-407.

    Carmona, R. and Delarue, F. (2018). Probabilistic Theory of Mean Field
    Games with Applications I & II. Springer. (For the McKean-Vlasov
    extension.)
    """
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    if x0.ndim != 2:
        raise ValueError(f"x0 must be 2D, got shape {x0.shape}")
    if x0.shape[1] != 1:
        raise ValueError(
            f"x0 has {x0.shape[1]} columns, expected 1 (scalar state)"
        )
    return _core.simulate_mean_field_cir(
        x0,
        float(t_final),
        int(n_steps),
        float(kappa),
        float(theta),
        float(b),
        float(sigma),
        int(record_every),
        int(seed),
        str(scheme),
    )
