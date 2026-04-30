"""Grid-based solver for non-LQ Mean Field Games in 1D scalar state.

Couples the Phase 1 backward HJB solver (:func:`mvkit.mfg.solve_hjb`)
and the Phase 2 forward Fokker-Planck solver
(:func:`mvkit.mfg.solve_fokker_planck`) via an outer Picard or
Fictitious Play iteration.

A user-supplied :class:`MFGProblem` specifies the running cost
:math:`F(t, x, m_t)`, terminal cost :math:`g(x, m_T)`, initial density
:math:`m_0(x)`, diffusion :math:`\\sigma`, horizon :math:`T`, periodic
spatial domain and the spatial discretization :math:`n_x`. The solver
discretizes time at :math:`n_t + 1` grid points and iterates:

- HJB step: solve the backward HJB given the current flow input
  (the latest iterate for Picard, or its historical average for
  Fictitious Play) to obtain the value function and its discrete
  gradient :math:`\\partial_x u` (the optimal control).
- FP step: solve the forward Fokker-Planck with that control as drift
  to obtain the next density iterate.
- Convergence test: stop when
  :math:`\\max_{t,x}|m^{(k+1)} - m^{(k)}| < \\mathrm{tol}`.

The Lasry-Lions monotonicity condition makes the coupled system a
contraction in a suitable metric; on monotone problems Picard
converges geometrically. Fictitious Play (Cardaliaguet and Hadikhanloo,
2017) trades the contraction requirement for an :math:`O(1/k)` rate
under monotonicity alone, and is the safety net when Picard fails to
contract.

Validation against the LQ-MFG closed form (the symmetric scalar case
solved analytically by ``mvkit.mfg.solve_lq_mfg`` via the Riccati ODE)
is the centerpiece of the test suite.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.

Cardaliaguet, P. and Hadikhanloo, S. (2017). *Learning in mean field
games: the fictitious play*. ESAIM: Control, Optimisation and Calculus
of Variations 23, 569-591.

Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field
Games with Applications I*, Chapter 4. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .._progress import progress_iter
from .fokker_planck import solve_fokker_planck
from .hjb import solve_hjb


_VALID_METHODS = ("picard", "fictitious_play")


@dataclass
class MFGProblem:
    """A 1D scalar Mean Field Game on a periodic domain.

    Attributes
    ----------
    sigma : float
        Diffusion coefficient. Must be strictly positive.
    T : float
        Time horizon. Must be strictly positive.
    domain : tuple of (float, float)
        Half-open spatial domain ``[a, b)`` interpreted with period
        ``L = b - a``. The grid does not include ``b``.
    n_x : int
        Number of spatial grid points. Must be at least 4.
    initial_density : callable
        Signature ``(x_grid: ndarray) -> ndarray of shape (n_x,)`` giving
        the initial density :math:`m_0(x_i)`. The returned values are
        renormalized on the grid so that ``sum * dx == 1``.
    running_cost : callable
        Signature ``(t: float, x_grid: ndarray, m_grid: ndarray) ->
        ndarray of shape (n_x,)`` giving :math:`F(t, x, m_t)` at the
        grid points. The third argument is the current flow at time
        ``t`` evaluated on the spatial grid.
    terminal_cost : callable
        Signature ``(x_grid: ndarray, m_grid: ndarray) -> ndarray of
        shape (n_x,)`` giving :math:`g(x, m_T)` at the grid points. The
        second argument is the terminal-time density on the grid.
    hamiltonian : str, default "quadratic"
        Hamiltonian shape. Currently only ``"quadratic"``
        (:math:`H(p) = p^2/2 - F`) is supported.
    boundary : str, default "periodic"
        Boundary condition. Currently only ``"periodic"`` is supported.
    """

    sigma: float
    T: float
    domain: Tuple[float, float]
    n_x: int
    initial_density: Callable[[np.ndarray], np.ndarray]
    running_cost: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    terminal_cost: Callable[[np.ndarray, np.ndarray], np.ndarray]
    hamiltonian: str = "quadratic"
    boundary: str = "periodic"


@dataclass
class MFGGridSolution:
    """Output of :func:`solve_mfg`.

    Attributes
    ----------
    t_grid : ndarray, shape (n_t + 1,)
        Increasing time grid from 0 to T.
    x_grid : ndarray, shape (n_x,)
        Periodic spatial grid (does not include the right endpoint).
    u : ndarray, shape (n_t + 1, n_x)
        Value function. ``u[-1]`` is the terminal cost evaluated on the
        grid against the converged ``m[-1]``; ``u[0]`` is the value at
        :math:`t = 0`.
    m : ndarray, shape (n_t + 1, n_x)
        Density. ``m[0]`` is the initial density (renormalized on the
        grid); ``m[-1]`` is the terminal density.
    optimal_drift : ndarray, shape (n_t + 1, n_x)
        Optimal control :math:`\\alpha^*(t, x) = -\\partial_x u`.
    m_iterates : list of ndarray
        Outer-iteration density iterates. ``m_iterates[0]`` is the
        starting flow guess; ``m_iterates[k]`` is the flow after the
        :math:`k`-th outer update.
    n_iterations : int
        Number of outer updates performed.
    converged : bool
        True if the sup-norm tolerance was met before the iteration cap.
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    u: np.ndarray
    m: np.ndarray
    optimal_drift: np.ndarray
    m_iterates: List[np.ndarray] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def _validate_problem(problem: MFGProblem) -> None:
    if not (np.isfinite(problem.sigma) and problem.sigma > 0.0):
        raise ValueError(f"sigma must be positive and finite, got {problem.sigma}")
    if not (np.isfinite(problem.T) and problem.T > 0.0):
        raise ValueError(f"T must be positive and finite, got {problem.T}")
    if problem.n_x < 4:
        raise ValueError(f"n_x must be >= 4, got {problem.n_x}")
    a, b = problem.domain
    if not (np.isfinite(a) and np.isfinite(b) and b > a):
        raise ValueError(
            f"domain must be (a, b) with a < b finite, got {problem.domain}"
        )
    if problem.boundary != "periodic":
        raise ValueError(
            f"boundary={problem.boundary!r} not supported in v0.1; "
            f"expected 'periodic'"
        )
    if problem.hamiltonian != "quadratic":
        raise ValueError(
            f"hamiltonian={problem.hamiltonian!r} not supported in v0.1; "
            f"expected 'quadratic'"
        )


def _build_grid(problem: MFGProblem) -> Tuple[np.ndarray, float, float]:
    """Return ``(x_grid, dx, L)`` for the periodic domain."""
    a, b = problem.domain
    L = float(b - a)
    n_x = int(problem.n_x)
    x_grid = np.linspace(a, b, n_x, endpoint=False)
    dx = L / n_x
    return x_grid, dx, L


def _initial_density_on_grid(
    problem: MFGProblem, x_grid: np.ndarray, dx: float
) -> np.ndarray:
    """Sample the user's initial density on the grid and renormalize so
    that ``sum * dx == 1``. The sampled density must be non-negative."""
    m0 = np.asarray(problem.initial_density(x_grid), dtype=np.float64)
    if m0.shape != (problem.n_x,):
        raise ValueError(
            f"initial_density must return shape ({problem.n_x},), got {m0.shape}"
        )
    if (m0 < 0.0).any():
        raise ValueError("initial_density must be non-negative")
    mass = float(np.sum(m0) * dx)
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError(
            f"initial_density integrates to {mass}, must be positive and finite"
        )
    return m0 / mass


def _make_running_cost_for_hjb(
    problem: MFGProblem, m_input: np.ndarray, t_grid: np.ndarray
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Wrap the user's three-argument running cost into the two-argument
    form expected by :func:`solve_hjb`, by indexing the current flow
    ``m_input`` at the appropriate time slice.

    The HJB solver evaluates the running cost at ``t = t_grid[1], ...,
    t_grid[n_t]`` (the "start of each backward step" convention). For
    each such ``t`` we look up the matching slice of ``m_input``.
    """
    n_t = t_grid.size - 1
    T = float(t_grid[-1])

    def rc(t: float, x: np.ndarray) -> np.ndarray:
        # Map t to index n with t_grid[n] = n * T / n_t.
        idx = int(np.round(t * n_t / T))
        idx = max(0, min(idx, n_t))
        return problem.running_cost(t, x, m_input[idx])

    return rc


def _make_drift_for_fp(
    optimal_drift: np.ndarray, t_grid: np.ndarray
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Index the HJB-output drift array at the FP solver's current time."""
    n_t = t_grid.size - 1
    T = float(t_grid[-1])

    def drift(t: float, x: np.ndarray) -> np.ndarray:
        idx = int(np.round(t * n_t / T))
        idx = max(0, min(idx, n_t))
        return optimal_drift[idx]

    return drift


def solve_mfg(
    problem: MFGProblem,
    n_t: int = 100,
    method: str = "picard",
    n_iterations_max: int = 50,
    tol: float = 1e-4,
    damping_burn_in: int = 0,
    m_initial_iterate: Optional[np.ndarray] = None,
    progress: bool = False,
) -> MFGGridSolution:
    r"""Solve a 1D Mean Field Game on a periodic grid by outer iteration.

    Discretize the user-supplied problem on its spatial grid (size
    ``problem.n_x``) and on a uniform time grid of size ``n_t + 1``,
    then iterate Picard or Fictitious Play between the HJB and FP
    solvers until consecutive density iterates differ by less than
    ``tol`` in sup norm.

    The fixed-point structure:

    - Given a flow ``m`` of shape ``(n_t + 1, n_x)``, the HJB step
      computes the value function ``u`` and the optimal drift
      :math:`\alpha^* = -\partial_x u`.
    - Given that drift, the FP step propagates the initial density
      forward, producing the next ``m`` iterate.
    - Picard uses the latest ``m`` directly; Fictitious Play uses the
      historical average of past iterates (after an optional burn-in
      of leading Picard steps).

    Parameters
    ----------
    problem : MFGProblem
        The MFG problem definition.
    n_t : int, default 100
        Number of time steps for both HJB and FP discretization.
    method : {"picard", "fictitious_play"}, default "picard"
        Outer iteration scheme.
    n_iterations_max : int, default 50
        Cap on outer-iteration updates.
    tol : float, default 1e-4
        Sup-norm tolerance on consecutive density iterates.
    damping_burn_in : int, default 0
        Only used when ``method="fictitious_play"``. Run this many
        leading Picard steps before starting to accumulate the
        historical average.
    m_initial_iterate : ndarray of shape (n_t + 1, n_x), optional
        Custom initial flow guess for the outer iteration. Defaults to
        the constant-in-time extension of the initial density,
        ``m_initial_iterate[k] = m_0`` for all ``k``.
    progress : bool, default False
        If True and ``tqdm`` is installed, wrap the outer iteration in
        a progress bar; if ``tqdm`` is missing the call still runs and
        prints one informational line. Default ``False`` is silent and
        incurs no overhead.

    Returns
    -------
    MFGGridSolution

    See Also
    --------
    mvkit.mfg.solve_lq_mfg : The closed-form scalar LQ-MFG solver,
        which carries the analytical Riccati ODE for ``P(t)`` and
        therefore has no spatial discretization error on the value
        function. Faster and more accurate when the running and
        terminal costs are exactly LQ; ``solve_mfg`` is the right
        choice as soon as the costs are non-LQ.
    """
    _validate_problem(problem)
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown method '{method}'; expected one of "
            f"{list(_VALID_METHODS)}"
        )
    if n_iterations_max < 1:
        raise ValueError(
            f"n_iterations_max must be >= 1, got {n_iterations_max}"
        )
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if damping_burn_in < 0:
        raise ValueError(
            f"damping_burn_in must be non-negative, got {damping_burn_in}"
        )

    x_grid, dx, _L = _build_grid(problem)
    t_grid = np.linspace(0.0, float(problem.T), int(n_t) + 1)
    m0 = _initial_density_on_grid(problem, x_grid, dx)

    if m_initial_iterate is None:
        m_curr = np.tile(m0, (n_t + 1, 1))
    else:
        arr = np.ascontiguousarray(m_initial_iterate, dtype=np.float64)
        if arr.shape != (n_t + 1, problem.n_x):
            raise ValueError(
                f"m_initial_iterate must have shape ({n_t + 1}, {problem.n_x}), "
                f"got {arr.shape}"
            )
        m_curr = arr.copy()

    m_iterates: List[np.ndarray] = [m_curr.copy()]
    converged = False

    sum_post_burnin: Optional[np.ndarray] = None
    count_post_burnin = 0

    last_u = None
    last_drift = None

    iterator = progress_iter(
        range(n_iterations_max),
        total=n_iterations_max,
        description=f"solve_mfg ({method})",
        enabled=progress,
    )
    for k in iterator:
        # Pick the input flow for this iteration's HJB.
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

        # Backward HJB given the input flow.
        terminal_arr = np.asarray(
            problem.terminal_cost(x_grid, m_input[-1]), dtype=np.float64
        )
        if terminal_arr.shape != (problem.n_x,):
            raise ValueError(
                f"terminal_cost must return shape ({problem.n_x},), "
                f"got {terminal_arr.shape}"
            )
        rc = _make_running_cost_for_hjb(problem, m_input, t_grid)
        try:
            hjb_sol = solve_hjb(
                sigma=problem.sigma,
                T=problem.T,
                x_grid=x_grid,
                n_t=int(n_t),
                terminal=terminal_arr,
                running_cost=rc,
                boundary=problem.boundary,
                hamiltonian=problem.hamiltonian,
            )
        except ValueError as exc:
            raise ValueError(
                f"HJB step failed during outer iteration k={k}: {exc}"
            ) from exc
        last_u = hjb_sol.u
        last_drift = hjb_sol.optimal_drift

        # Forward FP under the resulting optimal drift.
        drift_callable = _make_drift_for_fp(last_drift, t_grid)
        try:
            fp_sol = solve_fokker_planck(
                sigma=problem.sigma,
                T=problem.T,
                x_grid=x_grid,
                n_t=int(n_t),
                initial=m0,
                drift=drift_callable,
                boundary=problem.boundary,
            )
        except ValueError as exc:
            raise ValueError(
                f"Fokker-Planck step failed during outer iteration k={k}: {exc}"
            ) from exc
        m_next = fp_sol.m

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

    n_iterations = len(m_iterates) - 1
    assert last_u is not None and last_drift is not None

    return MFGGridSolution(
        t_grid=t_grid,
        x_grid=x_grid,
        u=last_u,
        m=m_curr,
        optimal_drift=last_drift,
        m_iterates=m_iterates,
        n_iterations=n_iterations,
        converged=converged,
    )
