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


def _convective_flux_divergence(
    alpha: np.ndarray, m: np.ndarray, dx: float
) -> np.ndarray:
    """Conservative-form upwind flux divergence
    :math:`(J^c_{i+1/2} - J^c_{i-1/2}) / \\Delta x` for periodic BC.

    The half-grid drift is the central average of adjacent grid drifts;
    the half-grid density is upwinded based on the sign of the half-grid
    drift. The resulting telescoping sum vanishes under periodic
    summation, which is what gives exact mass conservation.
    """
    alpha_right = np.roll(alpha, -1)
    alpha_half = 0.5 * (alpha + alpha_right)
    m_right = np.roll(m, -1)
    flux_right = np.where(alpha_half > 0.0, alpha_half * m, alpha_half * m_right)
    flux_left = np.roll(flux_right, 1)
    return (flux_right - flux_left) / dx


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
    boundary : str, default "periodic"
        Boundary condition. Only ``"periodic"`` is supported in v0.1.

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
    if boundary != "periodic":
        raise ValueError(
            f"boundary={boundary!r} not supported in v0.1; expected 'periodic'"
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

    n_t_int = int(n_t)
    dt = float(T) / n_t_int
    t_grid = np.linspace(0.0, float(T), n_t_int + 1)

    diff_lu = _periodic_implicit_diffusion(n_x, dx, float(sigma), dt)

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

        flux_div = _convective_flux_divergence(alpha_n, m_n, dx)
        rhs = m_n - dt * flux_div
        m[n + 1] = diff_lu.solve(rhs)

    return FPSolution(t_grid=t_grid, x_grid=x, m=m)
