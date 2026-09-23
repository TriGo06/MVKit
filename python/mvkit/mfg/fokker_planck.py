"""Numerical solver for the forward Fokker-Planck (continuity) equation
on a 1D periodic grid.

For a given drift :math:`\\alpha(t, x)` the FP equation reads

.. math::
    \\partial_t m + \\partial_x(\\alpha(t, x)\\, m)
    - \\tfrac{1}{2}\\sigma^2 \\partial_{xx} m = 0,
    \\qquad m(0, x) = m_0(x).

Equivalently, with total flux :math:`J = \\alpha m - \\tfrac{\\sigma^2}{2}
\\partial_x m`, the equation is :math:`\\partial_t m + \\partial_x J = 0`,
which the conservative discretization turns into exact mass preservation
on a periodic grid.

Discretization:

- **Convection** (explicit, upwind in conservative form): half-grid drift
  :math:`\\alpha_{i+1/2} = \\tfrac{1}{2}(\\alpha_i + \\alpha_{i+1})`,
  upwind density :math:`m_{i+1/2} = m_i` if :math:`\\alpha_{i+1/2} > 0`
  else :math:`m_{i+1}`. The flux divergence
  :math:`(J^c_{i+1/2} - J^c_{i-1/2})/\\Delta x` telescopes to zero when
  summed over the periodic grid, so the scheme conserves mass exactly.
- **Diffusion** (implicit, central): the same sparse periodic-tridiagonal
  operator :math:`(I - \\tfrac{\\sigma^2 \\Delta t}{2} D^2)` as in
  :mod:`mvkit.mfg.hjb`, factored once at setup. The discrete Laplacian
  has total-mass nullspace, so the implicit update preserves mass too.
- **CFL on convection**: :math:`\\Delta t \\le \\Delta x / \\max|\\alpha|`.
  Documented but not enforced; the user controls ``n_t``.

Phase 2 of the non-LQ MFG roadmap. Phase 3 will couple this with the HJB
solver into a full ``solve_mfg`` outer loop.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.

LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*.
Cambridge University Press. (Conservative upwind for the convection
flux.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import SuperLU, splu


@dataclass
class FPSolution:
    """Output of :func:`solve_fokker_planck`.

    Attributes
    ----------
    t_grid : ndarray, shape (n_t + 1,)
        Increasing time grid from 0 to T.
    x_grid : ndarray, shape (n_x,)
        Periodic spatial grid; the period is ``L = n_x * dx``.
    m : ndarray, shape (n_t + 1, n_x)
        Density at the grid points. ``m[0]`` is the initial condition
        sampled on the grid; ``m[-1]`` is the density at the terminal
        time.
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    m: np.ndarray


def _periodic_implicit_diffusion(
    n_x: int, dx: float, sigma: float, dt: float
) -> SuperLU:
    """Factor the periodic-tridiagonal operator
    ``I - (sigma^2 dt / 2) D^2``. Same construction as in the HJB
    solver; duplicated here to keep the modules independent.
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
    """Factor the Neumann tridiagonal operator
    ``I - (sigma^2 dt / 2) D^2`` with reflecting ghost cells at the
    endpoints. Boundary rows have diagonal ``1 + alpha`` (instead of
    ``1 + 2 alpha``); the matrix is strictly tridiagonal without
    wrap-around.
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


def _convective_flux_divergence(
    alpha: np.ndarray, m: np.ndarray, dx: float, boundary: str
) -> np.ndarray:
    r"""Conservative-form upwind flux divergence
    :math:`(J^c_{i+1/2} - J^c_{i-1/2}) / \Delta x`.

    Computes the interior half-grid fluxes :math:`J^c_{i+1/2}` for
    :math:`i = 0, \dots, n_x - 2` (length ``n_x - 1``) via the upwind
    rule, then forms the divergence according to the boundary condition:

    - ``"periodic"``: pad with the circular wrap so the boundary cell
      sees the flux at the periodic seam.
    - ``"neumann"``: pad with zeros at both ends, enforcing the no-flux
      condition :math:`J^c_{-1/2} = J^c_{n_x - 1/2} = 0`.

    Either way, the telescoping sum vanishes when summed over the
    spatial grid (periodic sum because of the wrap; Neumann sum because
    the boundary fluxes are zero), so the discrete scheme conserves
    total mass exactly.
    """
    n_x = m.size
    # Half-grid drift and upwinded density for the n_x - 1 interior
    # half-grid points i + 1/2, i = 0, ..., n_x - 2.
    alpha_half = 0.5 * (alpha[:-1] + alpha[1:])
    flux_interior = np.where(
        alpha_half > 0.0, alpha_half * m[:-1], alpha_half * m[1:]
    )
    # Pad to length n_x + 1: index k carries J^c_{k - 1/2}.
    flux_padded = np.empty(n_x + 1, dtype=np.float64)
    flux_padded[1:n_x] = flux_interior
    if boundary == "periodic":
        # Wrap: J^c_{-1/2} = J^c_{n_x - 1/2} = alpha_{n_x - 1/2} m_upwind.
        wrap_alpha = 0.5 * (alpha[-1] + alpha[0])
        wrap_flux = wrap_alpha * (m[-1] if wrap_alpha > 0.0 else m[0])
        flux_padded[0] = wrap_flux
        flux_padded[-1] = wrap_flux
    else:  # neumann
        flux_padded[0] = 0.0
        flux_padded[-1] = 0.0
    return (flux_padded[1:] - flux_padded[:-1]) / dx


def solve_fokker_planck(
    sigma: float,
    T: float,
    x_grid: np.ndarray,
    n_t: int,
    initial: np.ndarray,
    drift: Callable[[float, np.ndarray], np.ndarray],
    boundary: str = "periodic",
) -> FPSolution:
    r"""Solve the forward Fokker-Planck equation on a 1D periodic grid.

    For a user-supplied drift :math:`\alpha(t, x)`, integrate

    .. math::
        \partial_t m + \partial_x(\alpha m)
        - \tfrac{1}{2}\sigma^2 \partial_{xx} m = 0,
        \qquad m(0, x) = m_0(x),

    forward in time from :math:`m(0)` to :math:`m(T)` using
    implicit-explicit time stepping (explicit conservative upwind for
    convection, implicit central for diffusion). Mass is conserved
    exactly modulo float arithmetic.

    Parameters
    ----------
    sigma : float
        Diffusion coefficient. Must be strictly positive.
    T : float
        Time horizon. Must be strictly positive.
    x_grid : ndarray, shape (n_x,)
        Uniformly spaced grid, sorted in increasing order. Periodic with
        period ``L = n_x * dx``; must NOT include the right endpoint
        (use ``np.linspace(a, b, n_x, endpoint=False)``).
    n_t : int
        Number of time steps. ``dt = T / n_t``. The CFL condition
        ``dt <= dx / max|alpha|`` is required for stability of the
        explicit upwind convection; the user is responsible for picking
        a suitable ``n_t``.
    initial : ndarray, shape (n_x,)
        Initial density ``m_0(x_i)``. Should be non-negative; total
        mass ``sum(initial) * dx`` is preserved through the simulation.
    drift : callable
        Signature ``(t: float, x_grid: ndarray) -> ndarray of shape
        (n_x,)``. Evaluated at the START of each forward step,
        ``t = 0, dt, ..., T - dt``.
    boundary : {"periodic", "neumann"}, default "periodic"
        Boundary condition. ``"periodic"`` identifies the right
        endpoint with the left and the grid is read modulo the period.
        ``"neumann"`` enforces no-flux at both endpoints
        (:math:`J(t, a) = J(t, b) = 0`); the conservative flux scheme
        zeros the boundary half-grid fluxes, which preserves total mass
        exactly under the discrete divergence.

    Returns
    -------
    FPSolution
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

    initial_arr = np.ascontiguousarray(initial, dtype=np.float64)
    if initial_arr.shape != (n_x,):
        raise ValueError(
            f"initial must have shape ({n_x},), got {initial_arr.shape}"
        )

    if not np.isfinite(initial_arr).all() or (initial_arr < 0).any():
        raise ValueError("initial must be finite and non-negative")

    n_t_int = int(n_t)
    dt = float(T) / n_t_int
    t_grid = np.linspace(0.0, float(T), n_t_int + 1)

    if boundary == "periodic":
        diff_lu = _periodic_implicit_diffusion(n_x, dx, float(sigma), dt)
    else:
        diff_lu = _neumann_implicit_diffusion(n_x, dx, float(sigma), dt)

    m = np.empty((n_t_int + 1, n_x), dtype=np.float64)
    m[0] = initial_arr

    for n in range(n_t_int):
        # Forward step from t_grid[n] (known m[n]) to t_grid[n + 1].
        m_n = m[n]
        alpha_n = np.asarray(
            drift(float(t_grid[n]), x), dtype=np.float64
        )
        if alpha_n.shape != (n_x,):
            raise ValueError(
                f"drift must return shape ({n_x},), got {alpha_n.shape}"
            )

        # A monotone explicit upwind step preserves non-negative mass
        # when its transport CFL is at most one. Diffusion cannot
        # justify a larger bound uniformly, particularly as sigma -> 0.
        _CFL_HARD_LIMIT = 1.0
        max_alpha = float(np.max(np.abs(alpha_n)))
        cfl = max_alpha * dt / dx
        if cfl > _CFL_HARD_LIMIT or not np.isfinite(cfl):
            recommended_n_t = (
                int(np.ceil(cfl * n_t_int * 1.1)) if np.isfinite(cfl)
                else 2 * n_t_int
            )
            raise ValueError(
                f"Fokker-Planck explicit-convection CFL violated at "
                f"step {n + 1}/{n_t_int} (t = {float(t_grid[n]):.4f}): "
                f"max|drift| = {max_alpha:.3e}, dt = {dt:.3e}, "
                f"dx = {dx:.3e}, giving CFL = {cfl:.3e} (hard limit "
                f"{_CFL_HARD_LIMIT}). Try n_t >= {recommended_n_t}."
            )

        flux_div = _convective_flux_divergence(alpha_n, m_n, dx, boundary)
        rhs = m_n - dt * flux_div
        m[n + 1] = diff_lu.solve(rhs)

        if not np.isfinite(m[n + 1]).all():
            raise ValueError(
                f"Fokker-Planck solver produced non-finite values at "
                f"step {n + 1}/{n_t_int} (t = {float(t_grid[n + 1]):.4f}). "
                "This indicates an instability that the per-step CFL "
                "guard did not catch; please report with a minimal "
                "reproducer."
            )

    return FPSolution(t_grid=t_grid, x_grid=x, m=m)
