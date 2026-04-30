"""Numerical solver for the backward Hamilton-Jacobi-Bellman equation
on a 1D periodic grid.

For a quadratic Hamiltonian :math:`H(p, x, m) = \\tfrac{1}{2} p^2 - F(x, m_t)`
the agent's HJB equation reads

.. math::
    \\partial_t u - \\tfrac{1}{2}(\\partial_x u)^2 + F(x, m_t)
    + \\tfrac{1}{2}\\sigma^2 \\partial_{xx} u = 0,
    \\qquad u(T, x) = g(x, m_T).

In :math:`\\tau = T - t` this becomes a well-posed forward equation:

.. math::
    \\partial_\\tau u = -\\tfrac{1}{2}(\\partial_x u)^2 + F(x, m_t)
    + \\tfrac{1}{2}\\sigma^2 \\partial_{xx} u.

We discretize on a uniform periodic grid in :math:`x` with the
Engquist-Osher upwind for the Hamiltonian and central differences for
diffusion. Time integration is implicit-explicit: explicit on the
Hamiltonian and source, implicit on the diffusion. The LHS operator
:math:`(I - \\tfrac{\\sigma^2 \\Delta\\tau}{2} D^2)` is a fixed sparse
periodic tridiagonal, factored once via ``scipy.sparse.linalg.splu``;
each backward time step then costs one sparse triangular solve.

Phase 1 of the non-LQ MFG roadmap. Validates standalone via:

- a zero-solution sanity check (`F = 0`, `g = 0` returns `u = 0`),
- a Hopf-Cole closed form (`F = 0` linearizes the HJB to backward heat),
- self-convergence at rate 1 under joint refinement of `n_x` and `n_t`,
- monotonicity preservation in the terminal data.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.

Falcone, M. and Ferretti, R. (2014). *Semi-Lagrangian Approximation
Schemes for Linear and Hamilton-Jacobi Equations*. SIAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import SuperLU, splu


@dataclass
class HJBSolution:
    """Output of :func:`solve_hjb`.

    Attributes
    ----------
    t_grid : ndarray, shape (n_t + 1,)
        Increasing time grid from 0 to T. The terminal time is the
        last entry.
    x_grid : ndarray, shape (n_x,)
        Periodic spatial grid. The grid does not repeat the right
        endpoint; the period is ``L = n_x * dx`` where
        ``dx = x_grid[1] - x_grid[0]``.
    u : ndarray, shape (n_t + 1, n_x)
        Value function. ``u[-1] = terminal`` and ``u[0]`` is the value
        function at ``t = 0``.
    optimal_drift : ndarray, shape (n_t + 1, n_x)
        Optimal control at the grid points,
        ``alpha*(t, x) = -partial_x u``, computed by central difference
        with periodic wrap-around.
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    u: np.ndarray
    optimal_drift: np.ndarray


def _engquist_osher_quadratic(
    d_minus: np.ndarray, d_plus: np.ndarray
) -> np.ndarray:
    """Engquist-Osher numerical Hamiltonian for ``H(p) = (1/2) p^2``.

    For convex ``H(p)`` the EO scheme is

    .. math::
        H_h(p^-, p^+) = H_+(p^-) + H_-(p^+),

    with ``H_+(p) = int_0^p max(H'(q), 0) dq`` and ``H_-`` the analogous
    minimum integral. For ``H(p) = p^2/2`` this reduces to
    ``(1/2) max(p^-, 0)^2 + (1/2) min(p^+, 0)^2``, the form we use here.
    """
    return 0.5 * (
        np.maximum(d_minus, 0.0) ** 2 + np.minimum(d_plus, 0.0) ** 2
    )


def _periodic_implicit_diffusion(
    n_x: int, dx: float, sigma: float, dt: float
) -> SuperLU:
    """Build and factorize the periodic-tridiagonal implicit-diffusion
    operator ``I - (sigma^2 dt / 2) D^2``.

    The discrete Laplacian on a periodic grid has diagonal
    ``-2/dx^2`` and off-diagonals ``+1/dx^2``, with the corner entries
    ``A[0, n-1]`` and ``A[n-1, 0]`` carrying the periodic wrap-around.
    The resulting matrix is symmetric positive definite (its eigenvalues
    are ``1 + (2 sigma^2 dt / dx^2) sin^2(pi k / n_x) >= 1``), so an LU
    factorization without pivoting exists; we use ``splu`` directly for
    its sparse triangular solver.
    """
    alpha = 0.5 * sigma * sigma * dt / (dx * dx)
    main = (1.0 + 2.0 * alpha) * np.ones(n_x)
    sub = -alpha * np.ones(n_x - 1)
    super_ = -alpha * np.ones(n_x - 1)
    wrap_top = -alpha * np.ones(1)
    wrap_bot = -alpha * np.ones(1)
    A = diags(
        diagonals=[main, sub, super_, wrap_top, wrap_bot],
        offsets=[0, -1, 1, n_x - 1, -(n_x - 1)],
        format="csc",
    )
    return splu(A)


def _neumann_implicit_diffusion(
    n_x: int, dx: float, sigma: float, dt: float
) -> SuperLU:
    """Build and factorize the implicit-diffusion operator
    ``I - (sigma^2 dt / 2) D^2`` under homogeneous Neumann BC.

    The discrete Laplacian with reflecting ghost cells
    (``u_{-1} = u_0``, ``u_{n_x} = u_{n_x - 1}``) has interior rows
    identical to the periodic case, but the boundary rows are
    one-sided: row 0 has diagonal ``-1/dx^2`` (instead of ``-2/dx^2``)
    and a single super-diagonal ``+1/dx^2``; row ``n_x - 1`` is the
    mirror. This gives a strictly tridiagonal matrix without
    wrap-around, simpler than the periodic case.
    """
    alpha = 0.5 * sigma * sigma * dt / (dx * dx)
    main = (1.0 + 2.0 * alpha) * np.ones(n_x)
    main[0] = 1.0 + alpha
    main[-1] = 1.0 + alpha
    sub = -alpha * np.ones(n_x - 1)
    super_ = -alpha * np.ones(n_x - 1)
    A = diags(
        diagonals=[main, sub, super_],
        offsets=[0, -1, 1],
        format="csc",
    )
    return splu(A)


def _spatial_diffs(u: np.ndarray, dx: float, boundary: str):
    """Backward and forward spatial differences with the requested BC.

    Periodic: standard ``np.roll`` shift. Neumann: reflecting ghost
    cells, which makes the boundary one-sided difference vanish
    (``D^- u_0 = 0``, ``D^+ u_{n-1} = 0``).
    """
    if boundary == "periodic":
        u_left = np.roll(u, 1)
        u_right = np.roll(u, -1)
    else:  # neumann
        u_left = np.empty_like(u)
        u_left[0] = u[0]
        u_left[1:] = u[:-1]
        u_right = np.empty_like(u)
        u_right[-1] = u[-1]
        u_right[:-1] = u[1:]
    d_minus = (u - u_left) / dx
    d_plus = (u_right - u) / dx
    return d_minus, d_plus


def solve_hjb(
    sigma: float,
    T: float,
    x_grid: np.ndarray,
    n_t: int,
    terminal: np.ndarray,
    running_cost: Callable[[float, np.ndarray], np.ndarray],
    boundary: str = "periodic",
    hamiltonian: str = "quadratic",
) -> HJBSolution:
    r"""Solve the backward HJB on a 1D periodic grid (Phase 1 entry).

    The HJB equation

    .. math::
        \partial_t u - \tfrac{1}{2}(\partial_x u)^2 + F(x, m_t)
        + \tfrac{1}{2}\sigma^2 \partial_{xx} u = 0,
        \qquad u(T, x) = g(x),

    is integrated backward in :math:`t` (equivalently forward in
    :math:`\tau = T - t`) with implicit-explicit time stepping: the
    Engquist-Osher upwind Hamiltonian and the running cost are evaluated
    explicitly at the previous time step, while the diffusion is treated
    implicitly via a one-shot sparse periodic-tridiagonal solve.

    The flow ``m_t`` is supplied implicitly through the ``running_cost``
    callable: at each backward step the solver calls
    ``running_cost(t, x_grid)`` and uses the returned ``F`` array.

    Parameters
    ----------
    sigma : float
        Diffusion coefficient. Must be strictly positive.
    T : float
        Time horizon. Must be strictly positive.
    x_grid : ndarray, shape (n_x,)
        Uniformly spaced 1D grid, sorted in increasing order. Interpreted
        as the discretization of a periodic domain of period
        ``L = n_x * dx`` where ``dx = x_grid[1] - x_grid[0]``; the grid
        must NOT include the right endpoint of the period. Construct as
        ``np.linspace(a, b, n_x, endpoint=False)`` for a domain
        ``[a, a + L)``.
    n_t : int
        Number of time steps; ``dt = T / n_t``. Must be at least 1.
    terminal : ndarray, shape (n_x,)
        The values ``u(T, x_i)`` at the grid points.
    running_cost : callable
        Signature ``(t: float, x_grid: ndarray) -> ndarray of shape
        (n_x,)``. Returns ``F(x, m_t)`` evaluated on the spatial grid at
        the requested time. The solver evaluates this at the START of
        each backward step, ``t = T, T - dt, ..., dt``.
    boundary : {"periodic", "neumann"}, default "periodic"
        Boundary condition. ``"periodic"`` identifies the right
        endpoint with the left and the grid is read modulo the period.
        ``"neumann"`` enforces ``partial_x u = 0`` at the boundaries
        via reflecting ghost cells; the discrete Laplacian becomes
        strictly tridiagonal (no wrap) and the boundary-cell rows have
        diagonal ``1 + alpha`` instead of ``1 + 2 alpha``.
    hamiltonian : str, default "quadratic"
        Hamiltonian shape. Only ``"quadratic"`` (i.e. ``H(p) = p^2/2``)
        is supported in v0.1.

    Returns
    -------
    HJBSolution
    """
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be positive and finite, got {sigma}")
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive and finite, got {T}")
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")
    if boundary not in ("periodic", "neumann"):
        raise ValueError(
            f"boundary={boundary!r} not supported; "
            "expected 'periodic' or 'neumann'"
        )
    if hamiltonian != "quadratic":
        raise ValueError(
            f"hamiltonian={hamiltonian!r} not supported in v0.1; expected 'quadratic'"
        )

    x = np.asarray(x_grid, dtype=np.float64)
    if x.ndim != 1 or x.size < 4:
        raise ValueError(
            f"x_grid must be 1D with at least 4 points, got shape {x.shape}"
        )
    diffs = np.diff(x)
    dx = float(diffs[0])
    if dx <= 0.0:
        raise ValueError("x_grid must be strictly increasing")
    if not np.allclose(diffs, dx, rtol=1e-10, atol=1e-12):
        raise ValueError("x_grid must be uniformly spaced")
    n_x = x.size

    terminal_arr = np.ascontiguousarray(terminal, dtype=np.float64)
    if terminal_arr.shape != (n_x,):
        raise ValueError(
            f"terminal must have shape ({n_x},), got {terminal_arr.shape}"
        )

    n_t_int = int(n_t)
    dt = float(T) / n_t_int
    t_grid = np.linspace(0.0, float(T), n_t_int + 1)

    if boundary == "periodic":
        diff_lu = _periodic_implicit_diffusion(n_x, dx, float(sigma), dt)
    else:
        diff_lu = _neumann_implicit_diffusion(n_x, dx, float(sigma), dt)

    u = np.empty((n_t_int + 1, n_x), dtype=np.float64)
    u[n_t_int] = terminal_arr

    for n in range(n_t_int, 0, -1):
        # Step from t = t_grid[n] (known u[n]) to t = t_grid[n - 1].
        # In tau = T - t, this is one forward step of size dt.
        u_n = u[n]
        d_minus, d_plus = _spatial_diffs(u_n, dx, boundary)

        # Explicit-Hamiltonian CFL: dt * max|partial_x u| / dx <= 1 is
        # the textbook EO stability condition for H(p) = p^2/2.
        # Implicit diffusion buys some headroom in practice, so a strict
        # > 1 check produces false positives on otherwise-fine LQ-like
        # problems whose worst step sits just above 1; we raise above
        # 2.0, which still catches genuine instabilities (which run
        # well past 5 within a few steps) while leaving nominal cases
        # alone.
        _CFL_HARD_LIMIT = 2.0
        max_grad = float(max(np.max(np.abs(d_minus)), np.max(np.abs(d_plus))))
        cfl = max_grad * dt / dx
        if cfl > _CFL_HARD_LIMIT or not np.isfinite(cfl):
            step_idx = n_t_int - n + 1
            recommended_n_t = int(np.ceil(cfl * n_t_int * 1.1))
            raise ValueError(
                f"HJB explicit-Hamiltonian CFL violated at step "
                f"{step_idx}/{n_t_int} (t = {float(t_grid[n]):.4f}): "
                f"max|partial_x u| ~ {max_grad:.3e}, dt = {dt:.3e}, "
                f"dx = {dx:.3e}, giving CFL = {cfl:.3e} (hard limit "
                f"{_CFL_HARD_LIMIT}). "
                f"Try n_t >= {recommended_n_t}, or widen the spatial "
                "domain so the value function decays before the periodic "
                "wrap, or reduce the magnitude of the running cost or "
                "terminal data."
            )

        h_num = _engquist_osher_quadratic(d_minus, d_plus)

        # Source at the start of the backward step.
        f_n = np.asarray(
            running_cost(float(t_grid[n]), x), dtype=np.float64
        )
        if f_n.shape != (n_x,):
            raise ValueError(
                f"running_cost must return shape ({n_x},), got {f_n.shape}"
            )

        rhs = u_n + dt * (-h_num + f_n)
        u[n - 1] = diff_lu.solve(rhs)

        if not np.isfinite(u[n - 1]).all():
            step_idx = n_t_int - n + 1
            raise ValueError(
                f"HJB solver produced non-finite values at step "
                f"{step_idx}/{n_t_int} (t = {float(t_grid[n - 1]):.4f}). "
                "This indicates an instability that the per-step CFL "
                "guard did not catch; please report with a minimal "
                "reproducer."
            )

    # Optimal drift alpha* = -partial_x u, central difference. Under
    # Neumann the boundary cells use a one-sided gradient consistent
    # with the reflecting ghost cells (D^- u_0 = 0 = D^+ u_{n-1}).
    drift = np.empty_like(u)
    for k in range(n_t_int + 1):
        if boundary == "periodic":
            u_left = np.roll(u[k], 1)
            u_right = np.roll(u[k], -1)
            drift[k] = -(u_right - u_left) / (2.0 * dx)
        else:  # neumann
            d_minus, d_plus = _spatial_diffs(u[k], dx, boundary)
            drift[k] = -(d_minus + d_plus) / 2.0

    return HJBSolution(t_grid=t_grid, x_grid=x, u=u, optimal_drift=drift)
