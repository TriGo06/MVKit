"""Linear-quadratic Mean Field Game solver in scalar dimension.

The agent has scalar state :math:`X_t` evolving as

.. math::
    \\mathrm{d}X_t = \\alpha_t\\,\\mathrm{d}t + \\sigma\\,\\mathrm{d}W_t,
    \\quad X_0 \\sim \\mathcal{N}(m_0, v_0),

and minimizes, against a fixed mean trajectory :math:`m_t`,

.. math::
    J(\\alpha; m) = \\mathbb{E}\\Big[
        \\int_0^T \\big(\\tfrac{1}{2}\\alpha_t^2
            + \\tfrac{1}{2} q (X_t - m_t)^2\\big)\\,\\mathrm{d}t
        + \\tfrac{1}{2} q_T (X_T - m_T)^2 \\Big].

The HJB ansatz :math:`u(t,x) = \\tfrac{1}{2} P(t) x^2 + Q(t) x + R(t)` reduces
the problem to a Riccati ODE for :math:`P(t)` (independent of :math:`m`) and a
linear ODE for :math:`Q(t)` driven by :math:`m`. The optimal control is
:math:`\\alpha^*(t, x) = -P(t) x - Q(t)`, so simulation under the equilibrium
control is a controlled SDE with time-varying linear drift. In the symmetric
LQ case the equilibrium mean is constant, :math:`m_t = m_0`, and the variance
:math:`V(t)` solves the linear scalar ODE
:math:`\\dot V = -2 P V + \\sigma^2`.

Two iterative schemes share a common best-response operator BR(m): given an
input mean trajectory m, simulate N particles under the optimal control
against m and return the empirical mean. Picard sets m^(k+1) = BR(m^(k));
Fictitious Play sets m^(k+1) = BR(bar_m^(k)) where bar_m^(k) is the
historical average of m^(0), ..., m^(k). On LQ both converge to the same
equilibrium; Picard is geometric where it contracts, while Fictitious Play
trades the contraction requirement for an O(1/k) rate (Cardaliaguet and
Hadikhanloo, 2017).

References
----------
Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field Games
with Applications I*, Section 3.5 (LQ MFG).

Cardaliaguet, P. and Hadikhanloo, S. (2017). *Learning in mean field games:
the fictitious play*. ESAIM: Control, Optimisation and Calculus of Variations
23, 569-591.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ._riccati import solve_riccati


@dataclass
class LQMFGSolution:
    """Output of :func:`solve_lq_mfg`.

    Attributes
    ----------
    t_grid : ndarray, shape (G,)
        Increasing time grid from 0 to T, where ``G = n_grid + 1``.
    P : ndarray, shape (G,)
        Riccati solution P(t) at the grid points.
    m : ndarray, shape (G,)
        Equilibrium mean trajectory, equal to the empirical particle mean
        of the converged Picard iterate. In the symmetric LQ case the
        analytical equilibrium is constant, ``m(t) = mu_0_mean``.
    V : ndarray, shape (G,)
        Variance trajectory solving the linear ODE ``dot V = -2 P V +
        sigma^2`` with ``V(0) = mu_0_var``. Computed from the Riccati
        solution, independently of the particle simulation.
    x_trajectory : ndarray, shape (G, N)
        Final particle trajectory under the converged control. Provided
        so the user can compare empirical statistics to the analytical
        :attr:`V` and inspect sample paths in demos.
    m_iterates : list of ndarray
        Picard iterates, ``m_iterates[0]`` being the initial guess and
        ``m_iterates[k]`` the trajectory after the k-th update.
    n_iterations : int
        Number of Picard updates performed (length of
        ``m_iterates`` minus one).
    converged : bool
        True if ``max_t |m^(k+1)(t) - m^(k)(t)| < tol`` was met before
        ``n_iterations_max`` updates.
    """

    t_grid: np.ndarray
    P: np.ndarray
    m: np.ndarray
    V: np.ndarray
    x_trajectory: np.ndarray
    m_iterates: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def lq_mfg_analytical_variance(
    P: np.ndarray, t_grid: np.ndarray, sigma: float, mu_0_var: float
) -> np.ndarray:
    r"""Solve the variance ODE :math:`\dot V = -2 P V + \sigma^2`.

    Linear scalar ODE with time-varying coefficient :math:`P(t)`. Closed
    form:

    .. math::
        V(t) = V_0 \exp\!\Big(-2 \int_0^t P(s)\,\mathrm{d}s\Big)
            + \sigma^2 \int_0^t \exp\!\Big(-2 \int_s^t P(u)\,\mathrm{d}u\Big)\,\mathrm{d}s.

    We integrate the ODE numerically via ``scipy.solve_ivp`` with linear
    interpolation of P from the input grid. Strictly tighter than evaluating
    the closed form by trapezoidal cumulative integration, and stable for the
    grid sizes we use.

    Parameters
    ----------
    P : ndarray, shape (G,)
        Riccati solution.
    t_grid : ndarray, shape (G,)
        Increasing time grid; same one returned by :func:`solve_riccati`.
    sigma : float
        Diffusion coefficient. Must be non-negative.
    mu_0_var : float
        Initial variance V(0). Must be non-negative.

    Returns
    -------
    V : ndarray, shape (G,)
        Variance at the grid points.
    """
    if mu_0_var < 0.0:
        raise ValueError(f"mu_0_var must be non-negative, got {mu_0_var}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    P_arr = np.asarray(P, dtype=np.float64)
    t_arr = np.asarray(t_grid, dtype=np.float64)
    if P_arr.shape != t_arr.shape:
        raise ValueError(
            "P and t_grid must have the same shape; got "
            f"{P_arr.shape} vs {t_arr.shape}"
        )
    if t_arr.ndim != 1 or t_arr.size < 2:
        raise ValueError("t_grid must be 1D with at least 2 entries")

    sigma2 = float(sigma) * float(sigma)

    def rhs(t: float, V: np.ndarray) -> np.ndarray:
        P_t = float(np.interp(t, t_arr, P_arr))
        return np.array([-2.0 * P_t * V[0] + sigma2], dtype=np.float64)

    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_arr[0]), float(t_arr[-1])),
        y0=np.array([float(mu_0_var)], dtype=np.float64),
        method="Radau",
        t_eval=t_arr,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Variance ODE solver failed: {sol.message}")
    return np.asarray(sol.y[0], dtype=np.float64)


def _simulate_under_control(
    x0: np.ndarray,
    t_grid: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Forward Euler-Maruyama for ``dX = (-P(t) X - Q(t)) dt + sigma dW``.

    Pure-Python loop. The Rust core only accepts constant coefficients
    ``(a, b)``; this controlled SDE has time-varying drift via the Riccati
    P(t) and the Picard-current Q(t). For the grid sizes we use
    (``n_grid <= ~1000``) and particle counts (``N <= ~50000``) this is
    fast enough that the FFI is unnecessary.

    TODO: a Rust ``simulate_lq_time_varying`` that takes precomputed (P, Q)
    arrays would be cleaner once we move to non-LQ MFG and do many Picard
    or fictitious-play iterations. Tracked as a v0.2 item.

    Returns the trajectory of shape ``(G, N)``.
    """
    g = t_grid.size
    n = x0.size
    x = x0.astype(np.float64, copy=True)
    traj = np.empty((g, n), dtype=np.float64)
    traj[0] = x
    for step in range(g - 1):
        dt = float(t_grid[step + 1] - t_grid[step])
        drift = -P[step] * x - Q[step]
        z = rng.standard_normal(n)
        x = x + drift * dt + sigma * np.sqrt(dt) * z
        traj[step + 1] = x
    return traj


_VALID_METHODS = ("picard", "fictitious_play")
_SIM_SEED_MASK = 0xA5A5A5A5


def _draw_x0(
    mu_0_mean: float, mu_0_var: float, n_particles: int, seed: int
) -> np.ndarray:
    """Sample the initial particle states ``x0 ~ N(mu_0_mean, mu_0_var)``."""
    rng = np.random.default_rng(int(seed))
    return rng.normal(
        loc=float(mu_0_mean),
        scale=np.sqrt(float(mu_0_var)),
        size=int(n_particles),
    )


def _simulate_with_seed(
    m_input: np.ndarray,
    P: np.ndarray,
    t_grid: np.ndarray,
    mu_0_mean: float,
    mu_0_var: float,
    sigma: float,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    """Simulate the controlled SDE under input mean ``m_input`` from a
    reproducible ``seed``. Returns the trajectory of shape ``(G, N)``.

    Same ``seed`` plus same ``m_input`` give the same trajectory, so this
    can be invoked once per iteration to produce the BR mean and once
    again at the end of the solve to recover the trajectory consistent
    with the converged mean, with no extra randomness.
    """
    x0 = _draw_x0(mu_0_mean, mu_0_var, n_particles, seed)
    Q = -P * m_input
    sim_rng = np.random.default_rng(int(seed) ^ _SIM_SEED_MASK)
    return _simulate_under_control(
        x0=x0,
        t_grid=t_grid,
        P=P,
        Q=Q,
        sigma=float(sigma),
        rng=sim_rng,
    )


def _lq_best_response(
    m_input: np.ndarray,
    P: np.ndarray,
    t_grid: np.ndarray,
    mu_0_mean: float,
    mu_0_var: float,
    sigma: float,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    """Best-response operator BR(m_input) for the LQ-MFG.

    Given an input mean trajectory ``m_input(t)``, simulate N particles
    under the optimal control ``alpha*(t, x) = -P(t) x + P(t) m_input(t)``
    starting from ``x0 ~ N(mu_0_mean, mu_0_var)``, and return the
    empirical mean of the simulated trajectory at each grid time.

    The ``seed`` deterministically fixes both ``x0`` and the Brownian
    increments. With fixed noise, BR is an affine function of ``m_input``,
    which is the property both Picard and Fictitious Play exploit.

    Parameters
    ----------
    m_input : ndarray, shape (G,)
        Input mean trajectory.
    P : ndarray, shape (G,)
        Riccati solution.
    t_grid : ndarray, shape (G,)
        Time grid.
    mu_0_mean, mu_0_var : float
        Initial Gaussian distribution.
    sigma : float
        Diffusion coefficient.
    n_particles : int
        Particle count.
    seed : int
        RNG seed.

    Returns
    -------
    br : ndarray, shape (G,)
        BR(m_input) at the grid points.
    """
    traj = _simulate_with_seed(
        m_input,
        P,
        t_grid,
        mu_0_mean,
        mu_0_var,
        sigma,
        n_particles,
        seed,
    )
    return traj.mean(axis=1)


def _validate_common_inputs(
    q: float,
    q_T: float,
    sigma: float,
    T: float,
    mu_0_var: float,
    n_particles: int,
    n_grid: int,
    n_iterations_max: int,
    tol: float,
) -> None:
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if q < 0.0:
        raise ValueError(f"q must be non-negative, got {q}")
    if q_T < 0.0:
        raise ValueError(f"q_T must be non-negative, got {q_T}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if mu_0_var < 0.0:
        raise ValueError(f"mu_0_var must be non-negative, got {mu_0_var}")
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if n_iterations_max < 1:
        raise ValueError(
            f"n_iterations_max must be >= 1, got {n_iterations_max}"
        )
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")


def _initial_mean(
    m_initial: Optional[np.ndarray], g: int, mu_0_mean: float
) -> np.ndarray:
    if m_initial is None:
        return np.full(g, float(mu_0_mean), dtype=np.float64)
    arr = np.asarray(m_initial, dtype=np.float64).copy()
    if arr.shape != (g,):
        raise ValueError(
            f"m_initial must have shape ({g},), got {arr.shape}"
        )
    return arr


def solve_lq_mfg(
    q: float,
    q_T: float,
    sigma: float,
    T: float,
    mu_0_mean: float,
    mu_0_var: float,
    n_particles: int = 5000,
    n_grid: int = 500,
    n_iterations_max: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    m_initial: Optional[np.ndarray] = None,
    method: str = "picard",
    damping_burn_in: int = 0,
) -> LQMFGSolution:
    r"""Solve the scalar LQ Mean Field Game iteratively.

    Algorithm (independent of method):

    1. Solve the Riccati ODE for :math:`P(t)` once (independent of m).
    2. Sample :math:`x_0^{1\dots N} \sim \mathcal{N}(m_0, v_0)` and pick an
       initial mean guess :math:`m^{(0)}(t)` (constant equal to ``mu_0_mean``
       by default).
    3. Repeat: form an input mean :math:`m^\mathrm{in}_k`, set
       :math:`m^{(k+1)} = \mathrm{BR}(m^\mathrm{in}_k)`. Stop when
       :math:`\max_t |m^{(k+1)} - m^{(k)}| < \mathrm{tol}`.

    Parameter ``method`` selects the rule for :math:`m^\mathrm{in}_k`:

    - ``"picard"`` (default): :math:`m^\mathrm{in}_k = m^{(k)}`.
    - ``"fictitious_play"``: :math:`m^\mathrm{in}_k = \bar m^{(k)}`, where
      :math:`\bar m^{(k)} = \tfrac{1}{k - b + 1} \sum_{j = b}^{k} m^{(j)}`
      and ``b = damping_burn_in``. For ``k < b`` the iteration falls back
      to a Picard step (input is :math:`m^{(k)}`), then the historical
      average accumulates from iteration ``b`` onward.

    The Brownian increments are reseeded identically at the start of each
    iteration. With fixed noise the BR map becomes affine in :math:`m`,
    which suppresses spurious MC fluctuations between iterations and lets
    Picard's fixed-point error decay geometrically.

    Parameters
    ----------
    q, q_T : float
        Running- and terminal-cost weights; both non-negative.
    sigma : float
        Diffusion coefficient; non-negative.
    T : float
        Horizon; strictly positive.
    mu_0_mean, mu_0_var : float
        Mean and variance of the Gaussian initial law :math:`\mu_0`.
    n_particles : int, default 5000
        Particle count for the inner forward simulation.
    n_grid : int, default 500
        Time discretization. The returned grids have length ``n_grid + 1``.
    n_iterations_max : int, default 20
        Cap on iteration updates. Fictitious Play typically needs more
        iterations than Picard; bump this if you select that method.
    tol : float, default 1e-4
        Sup-norm tolerance on consecutive mean iterates.
    seed : int, default 42
        Master RNG seed. Identical seeds plus identical inputs give
        identical outputs.
    m_initial : ndarray, optional
        Initial guess for the mean trajectory, shape ``(n_grid + 1,)``.
        Defaults to a constant trajectory equal to ``mu_0_mean``. Useful
        for stress-testing the contraction property of the iteration.
    method : {"picard", "fictitious_play"}, default "picard"
        Selects the iteration scheme. Picard converges fast where the BR
        map is contracting (LQ-MFG with moderate ``int_0^T P``); Fictitious
        Play trades the contraction requirement for slower O(1/k)
        convergence and better robustness on non-monotone perturbations
        (Cardaliaguet and Hadikhanloo, 2017).
    damping_burn_in : int, default 0
        Only used when ``method="fictitious_play"``. The historical
        average starts accumulating from iteration ``damping_burn_in``;
        earlier iterations behave like Picard. Default 0 is standard
        Fictitious Play.

    Returns
    -------
    LQMFGSolution

    See Also
    --------
    mvkit.mfg.solve_mfg : The generic grid-based solver. Accepts an
        :class:`mvkit.mfg.MFGProblem` with arbitrary running and
        terminal cost callables, and is the right tool for non-LQ
        problems. ``solve_lq_mfg`` is faster and free of spatial
        discretization error on the value-function structure (it
        carries the analytical Riccati ODE for ``P(t)``); use it
        whenever the cost is exactly LQ.
    """
    _validate_common_inputs(
        q, q_T, sigma, T, mu_0_var, n_particles, n_grid, n_iterations_max, tol
    )
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown method '{method}'; expected one of "
            f"{list(_VALID_METHODS)}"
        )
    if damping_burn_in < 0:
        raise ValueError(
            f"damping_burn_in must be non-negative, got {damping_burn_in}"
        )

    t_grid, P = solve_riccati(q, q_T, T, n_grid=n_grid)
    g = t_grid.size

    m_curr = _initial_mean(m_initial, g, mu_0_mean)
    m_iterates: list = [m_curr.copy()]
    converged = False
    m_input_last = m_curr.copy()

    sum_post_burnin: Optional[np.ndarray] = None
    count_post_burnin = 0

    for k in range(n_iterations_max):
        if method == "picard":
            m_input = m_curr
        else:  # fictitious_play
            if k < damping_burn_in:
                m_input = m_curr
            else:
                if sum_post_burnin is None:
                    sum_post_burnin = m_curr.copy()
                    count_post_burnin = 1
                m_input = sum_post_burnin / count_post_burnin

        m_input_last = m_input.copy() if m_input is not m_curr else m_curr.copy()

        m_next = _lq_best_response(
            m_input=m_input,
            P=P,
            t_grid=t_grid,
            mu_0_mean=mu_0_mean,
            mu_0_var=mu_0_var,
            sigma=sigma,
            n_particles=n_particles,
            seed=seed,
        )
        diff = float(np.max(np.abs(m_next - m_curr)))
        m_iterates.append(m_next.copy())

        if method == "fictitious_play" and k >= damping_burn_in:
            assert sum_post_burnin is not None
            sum_post_burnin += m_next
            count_post_burnin += 1

        m_curr = m_next
        if diff < tol:
            converged = True
            break

    # Reproduce the trajectory consistent with the converged mean by
    # re-running the simulation under the last BR input. Same seed, same
    # output as the final loop iteration.
    traj = _simulate_with_seed(
        m_input=m_input_last,
        P=P,
        t_grid=t_grid,
        mu_0_mean=mu_0_mean,
        mu_0_var=mu_0_var,
        sigma=sigma,
        n_particles=n_particles,
        seed=seed,
    )

    n_iterations = len(m_iterates) - 1
    V = lq_mfg_analytical_variance(P, t_grid, float(sigma), float(mu_0_var))

    return LQMFGSolution(
        t_grid=t_grid,
        P=P,
        m=m_curr,
        V=V,
        x_trajectory=traj,
        m_iterates=m_iterates,
        n_iterations=n_iterations,
        converged=converged,
    )


def solve_lq_mfg_fictitious_play(
    q: float,
    q_T: float,
    sigma: float,
    T: float,
    mu_0_mean: float,
    mu_0_var: float,
    n_particles: int = 5000,
    n_grid: int = 500,
    n_iterations_max: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
    m_initial: Optional[np.ndarray] = None,
    damping_burn_in: int = 0,
) -> LQMFGSolution:
    r"""Solve the scalar LQ Mean Field Game by Fictitious Play.

    Convenience wrapper around :func:`solve_lq_mfg` with
    ``method="fictitious_play"``. The default ``n_iterations_max`` is
    raised to 50 to reflect the slower O(1/k) convergence rate compared
    to Picard's geometric.

    The update rule is :math:`m^{(k+1)} = \mathrm{BR}(\bar m^{(k)})`
    where :math:`\bar m^{(k)}` is the historical average of past iterates.
    Cardaliaguet and Hadikhanloo (2017) show that under MFG monotonicity,
    Fictitious Play converges to the equilibrium without requiring the BR
    map to be a strict contraction, the price being the slower rate.

    Parameters
    ----------
    damping_burn_in : int, default 0
        Number of leading Picard steps before Fictitious Play takes over.
        The historical average accumulates from iteration
        ``damping_burn_in`` onward. Useful when the initial guess is far
        from the equilibrium and the first few iterates would corrupt
        the average.

    See :func:`solve_lq_mfg` for the remaining parameters.

    Returns
    -------
    LQMFGSolution
    """
    return solve_lq_mfg(
        q=q,
        q_T=q_T,
        sigma=sigma,
        T=T,
        mu_0_mean=mu_0_mean,
        mu_0_var=mu_0_var,
        n_particles=n_particles,
        n_grid=n_grid,
        n_iterations_max=n_iterations_max,
        tol=tol,
        seed=seed,
        m_initial=m_initial,
        method="fictitious_play",
        damping_burn_in=damping_burn_in,
    )
