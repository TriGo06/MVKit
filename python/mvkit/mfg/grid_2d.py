"""Grid-based solver for non-LQ Mean Field Games in 2D scalar state.

Couples the 2D backward HJB solver (:func:`mvkit.mfg.solve_hjb_2d`)
and the 2D forward Fokker-Planck solver
(:func:`mvkit.mfg.solve_fokker_planck_2d`) via Picard or Fictitious
Play outer iteration.

A user-supplied :class:`MFGProblem2D` specifies the running cost
:math:`F(t, x, y, m_t)`, terminal cost :math:`g(x, y, m_T)`, initial
density :math:`m_0(x, y)`, isotropic diffusion :math:`\\sigma`,
horizon :math:`T`, rectangular spatial domain, and the per-axis
spatial discretization. The solver discretizes time at :math:`n_t + 1`
grid points and iterates HJB / FP until consecutive density iterates
differ by less than ``tol`` in sup norm.

Same dispatch as :func:`mvkit.mfg.solve_mfg` (the 1D solver):
``method="picard"`` (default, geometric where contracting) or
``method="fictitious_play"`` (slower :math:`O(1/k)` rate but contraction-free).

Validated quantitatively against the closed-form **vector LQ-MFG**
from :func:`mvkit.mfg.solve_lq_mfg_vector` at d=2: configuring an
``MFGProblem2D`` whose costs reproduce the 2D LQ recovers the
closed-form Lyapunov covariance trajectory at first order.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .._progress import progress_iter
from .fokker_planck_2d import solve_fokker_planck_2d
from .hjb_2d import _normalize_sigma_2d, solve_hjb_2d


_VALID_METHODS = ("picard", "fictitious_play")


@dataclass
class MFGProblem2D:
    """A 2D scalar Mean Field Game on a rectangular periodic or
    Neumann domain.

    Attributes
    ----------
    sigma : float or (float, float)
        Diffusion coefficient. Scalar is interpreted as isotropic
        (same diffusion in both axes). A 2-tuple
        ``(sigma_x, sigma_y)`` enables anisotropic diffusion (different
        magnitudes per axis); both entries must be strictly positive.
    T : float
        Time horizon.
    domain : tuple of ((float, float), (float, float))
        Spatial domain ``((a_x, b_x), (a_y, b_y))``.
    n_x : tuple of (int, int)
        ``(n_x, n_y)`` per-axis grid sizes.
    initial_density : callable
        Signature ``(X, Y) -> ndarray of shape (n_x, n_y)``.
    running_cost : callable
        Signature ``(t, X, Y, m) -> ndarray of shape (n_x, n_y)`` giving
        :math:`F(t, x, y, m_t)`.
    terminal_cost : callable
        Signature ``(X, Y, m_T) -> ndarray of shape (n_x, n_y)``.
    boundary : {"periodic", "neumann"}, default "periodic"
    hamiltonian : str, default "quadratic"
        Only ``"quadratic"`` (i.e. :math:`H(p) = |p|^2/2 - F`) is
        supported in v0.1.
    """

    sigma: Union[float, Tuple[float, float]]
    T: float
    domain: Tuple[Tuple[float, float], Tuple[float, float]]
    n_x: Tuple[int, int]
    initial_density: Callable[[np.ndarray, np.ndarray], np.ndarray]
    running_cost: Callable[
        [float, np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    terminal_cost: Callable[
        [np.ndarray, np.ndarray, np.ndarray], np.ndarray
    ]
    boundary: str = "periodic"
    hamiltonian: str = "quadratic"


@dataclass
class MFGGrid2DSolution:
    """Output of :func:`solve_mfg_2d`.

    Attributes
    ----------
    t_grid : ndarray, shape (G,)
    x_grid : ndarray, shape (n_x,)
    y_grid : ndarray, shape (n_y,)
    u : ndarray, shape (G, n_x, n_y)
        Value function from the final HJB solve.
    m : ndarray, shape (G, n_x, n_y)
        Density (the converged Picard / FP fixed point).
    optimal_drift : ndarray, shape (G, n_x, n_y, 2)
        Optimal control :math:`\\alpha^* = -\\nabla u`.
    m_iterates : list of ndarray
        Outer-iteration density iterates.
    n_iterations : int
    converged : bool
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    u: np.ndarray
    m: np.ndarray
    optimal_drift: np.ndarray
    m_iterates: List[np.ndarray] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def _validate_problem(problem: MFGProblem2D) -> None:
    # Sigma can be scalar or (sigma_x, sigma_y); shared validator.
    _normalize_sigma_2d(problem.sigma)
    if not (np.isfinite(problem.T) and problem.T > 0.0):
        raise ValueError(f"T must be positive and finite, got {problem.T}")
    if (
        not isinstance(problem.domain, tuple)
        or len(problem.domain) != 2
        or any(len(p) != 2 for p in problem.domain)
    ):
        raise ValueError(
            "domain must be ((a_x, b_x), (a_y, b_y)), got "
            f"{problem.domain}"
        )
    for axis, (a, b) in enumerate(problem.domain):
        if not (np.isfinite(a) and np.isfinite(b) and b > a):
            raise ValueError(
                f"axis {axis}: domain (a, b)=({a}, {b}) must satisfy a < b"
            )
    if (
        not isinstance(problem.n_x, tuple)
        or len(problem.n_x) != 2
        or any(n < 4 for n in problem.n_x)
    ):
        raise ValueError(
            f"n_x must be (n_x, n_y) with both >= 4, got {problem.n_x}"
        )
    if problem.boundary not in ("periodic", "neumann"):
        raise ValueError(
            f"boundary={problem.boundary!r} not supported; "
            "expected 'periodic' or 'neumann'"
        )
    if problem.hamiltonian != "quadratic":
        raise ValueError(
            f"hamiltonian={problem.hamiltonian!r} not supported; "
            "expected 'quadratic'"
        )


def _build_2d_grids(
    problem: MFGProblem2D,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    (a_x, b_x), (a_y, b_y) = problem.domain
    n_x, n_y = problem.n_x
    if problem.boundary == "periodic":
        x = np.linspace(a_x, b_x, n_x, endpoint=False)
        y = np.linspace(a_y, b_y, n_y, endpoint=False)
    else:  # neumann: cell-centered
        dx_v = (b_x - a_x) / n_x
        dy_v = (b_y - a_y) / n_y
        x = np.linspace(a_x + dx_v / 2.0, b_x - dx_v / 2.0, n_x)
        y = np.linspace(a_y + dy_v / 2.0, b_y - dy_v / 2.0, n_y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, dx, dy, X, Y


def _initial_density_on_grid(
    problem: MFGProblem2D, X: np.ndarray, Y: np.ndarray, dx: float, dy: float,
) -> np.ndarray:
    m0 = np.asarray(problem.initial_density(X, Y), dtype=np.float64)
    if m0.shape != X.shape:
        raise ValueError(
            f"initial_density must return shape {X.shape}, got {m0.shape}"
        )
    if (m0 < 0.0).any():
        raise ValueError("initial_density must be non-negative")
    mass = float(np.sum(m0) * dx * dy)
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError(
            f"initial_density integrates to {mass}, must be positive and finite"
        )
    return m0 / mass


def _make_running_cost_for_hjb_2d(
    problem: MFGProblem2D, m_input: np.ndarray, t_grid: np.ndarray,
):
    n_t = t_grid.size - 1
    T = float(t_grid[-1])

    def rc(t: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        idx = int(np.round(t * n_t / T))
        idx = max(0, min(idx, n_t))
        return problem.running_cost(t, X, Y, m_input[idx])

    return rc


def _make_drift_for_fp_2d(
    optimal_drift: np.ndarray, t_grid: np.ndarray,
):
    n_t = t_grid.size - 1
    T = float(t_grid[-1])

    def drift(t: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        idx = int(np.round(t * n_t / T))
        idx = max(0, min(idx, n_t))
        return optimal_drift[idx]

    return drift


def solve_mfg_2d(
    problem: MFGProblem2D,
    n_t: int = 100,
    method: str = "picard",
    n_iterations_max: int = 50,
    tol: float = 1e-4,
    damping_burn_in: int = 0,
    m_initial_iterate: Optional[np.ndarray] = None,
    progress: bool = False,
) -> MFGGrid2DSolution:
    r"""Solve a 2D Mean Field Game on a rectangular grid by outer iteration.

    Mirrors :func:`mvkit.mfg.solve_mfg` (the 1D solver) for 2D state.
    Same ``method=`` and ``damping_burn_in=`` knobs.

    Parameters
    ----------
    problem : MFGProblem2D
    n_t : int, default 100
    method : {"picard", "fictitious_play"}, default "picard"
    n_iterations_max : int, default 50
    tol : float, default 1e-4
    damping_burn_in : int, default 0
    m_initial_iterate : ndarray of shape (n_t + 1, n_x, n_y), optional
    progress : bool, default False

    Returns
    -------
    MFGGrid2DSolution

    See Also
    --------
    mvkit.mfg.solve_mfg : The 1D analogue.
    mvkit.mfg.solve_lq_mfg_vector : Closed-form vector LQ-MFG, used as
        the gold-standard validation for this solver in d=2.
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

    x, y, dx, dy, X, Y = _build_2d_grids(problem)
    n_x, n_y = problem.n_x
    t_grid = np.linspace(0.0, float(problem.T), int(n_t) + 1)
    m0 = _initial_density_on_grid(problem, X, Y, dx, dy)

    if m_initial_iterate is None:
        m_curr = np.broadcast_to(m0, (n_t + 1, n_x, n_y)).copy()
    else:
        arr = np.ascontiguousarray(m_initial_iterate, dtype=np.float64)
        if arr.shape != (n_t + 1, n_x, n_y):
            raise ValueError(
                f"m_initial_iterate must have shape "
                f"({n_t + 1}, {n_x}, {n_y}), got {arr.shape}"
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
        description=f"solve_mfg_2d ({method})",
        enabled=progress,
    )
    for k in iterator:
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

        terminal_arr = np.asarray(
            problem.terminal_cost(X, Y, m_input[-1]), dtype=np.float64,
        )
        if terminal_arr.shape != (n_x, n_y):
            raise ValueError(
                f"terminal_cost must return shape ({n_x}, {n_y}), got "
                f"{terminal_arr.shape}"
            )
        rc = _make_running_cost_for_hjb_2d(problem, m_input, t_grid)
        try:
            hjb_sol = solve_hjb_2d(
                sigma=problem.sigma,
                T=problem.T,
                x_grid=x,
                y_grid=y,
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

        drift_callable = _make_drift_for_fp_2d(last_drift, t_grid)
        try:
            fp_sol = solve_fokker_planck_2d(
                sigma=problem.sigma,
                T=problem.T,
                x_grid=x,
                y_grid=y,
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

    return MFGGrid2DSolution(
        t_grid=t_grid,
        x_grid=x,
        y_grid=y,
        u=last_u,
        m=m_curr,
        optimal_drift=last_drift,
        m_iterates=m_iterates,
        n_iterations=n_iterations,
        converged=converged,
    )
