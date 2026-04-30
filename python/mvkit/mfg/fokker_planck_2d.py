"""Forward Fokker-Planck (continuity) solver on a 2D rectangular grid.

For a given drift :math:`\\alpha(t, x, y) = (\\alpha_x, \\alpha_y)` and
isotropic scalar diffusion :math:`\\sigma`, the FP equation reads

.. math::
    \\partial_t m + \\nabla \\cdot (\\alpha\\, m)
    - \\tfrac{1}{2}\\sigma^2 \\Delta m = 0,
    \\qquad m(0, x, y) = m_0(x, y).

Discretization:

- **Convection** (explicit, conservative upwind per axis): the total
  flux :math:`J = \\alpha m` is split into x- and y-components, each
  upwinded by the sign of the corresponding half-grid drift. Boundary
  fluxes are wrapped (periodic) or zero (Neumann).
- **Diffusion** (implicit): same 2D Kronecker-sum Laplacian as the
  HJB solver, factored once at setup. The discrete Laplacian
  preserves total mass on a periodic grid (corner wraps make the
  row sums zero) and on a Neumann grid (the modified boundary rows
  also have zero row sums).
- **CFL on convection**: :math:`\\Delta t (\\max|\\alpha_x| / \\Delta x
  + \\max|\\alpha_y| / \\Delta y) \\le 1`. Hard limit 2.0 (mirrors 1D).

Mass is conserved exactly modulo float arithmetic in both BCs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
from scipy.sparse.linalg import SuperLU

from .hjb_2d import _build_implicit_diffusion_2d, _normalize_sigma_2d


_CFL_HARD_LIMIT = 2.0


@dataclass
class FP2DSolution:
    """Output of :func:`solve_fokker_planck_2d`.

    Attributes
    ----------
    t_grid : ndarray, shape (n_t + 1,)
    x_grid : ndarray, shape (n_x,)
    y_grid : ndarray, shape (n_y,)
    m : ndarray, shape (n_t + 1, n_x, n_y)
        Density on the spatio-temporal grid.
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    m: np.ndarray


def _convective_flux_divergence_2d(
    alpha_x: np.ndarray, alpha_y: np.ndarray,
    m: np.ndarray, dx: float, dy: float, boundary: str,
) -> np.ndarray:
    """Conservative-form upwind flux divergence on a 2D grid.

    Computes the flux divergence
    ``(J^x_{i+1/2,j} - J^x_{i-1/2,j}) / dx + (J^y_{i,j+1/2} - J^y_{i,j-1/2}) / dy``
    where each half-grid flux is upwinded by the sign of the half-grid
    drift along the respective axis. Boundary fluxes are wrapped
    (periodic) or set to zero (Neumann), so the telescoping sum
    vanishes when summed over the grid: total mass is preserved.
    """
    n_x, n_y = m.shape
    # x-axis flux at i + 1/2 (length n_x - 1 along axis 0).
    alpha_x_half = 0.5 * (alpha_x[:-1, :] + alpha_x[1:, :])
    flux_x_int = np.where(
        alpha_x_half > 0.0,
        alpha_x_half * m[:-1, :],
        alpha_x_half * m[1:, :],
    )
    # Pad to length n_x + 1 along axis 0.
    flux_x_padded = np.empty((n_x + 1, n_y), dtype=np.float64)
    flux_x_padded[1:n_x, :] = flux_x_int
    if boundary == "periodic":
        wrap_alpha_x = 0.5 * (alpha_x[-1, :] + alpha_x[0, :])
        wrap_flux_x = np.where(
            wrap_alpha_x > 0.0,
            wrap_alpha_x * m[-1, :],
            wrap_alpha_x * m[0, :],
        )
        flux_x_padded[0, :] = wrap_flux_x
        flux_x_padded[-1, :] = wrap_flux_x
    else:  # neumann
        flux_x_padded[0, :] = 0.0
        flux_x_padded[-1, :] = 0.0
    div_x = (flux_x_padded[1:, :] - flux_x_padded[:-1, :]) / dx

    # y-axis flux at j + 1/2 (length n_y - 1 along axis 1).
    alpha_y_half = 0.5 * (alpha_y[:, :-1] + alpha_y[:, 1:])
    flux_y_int = np.where(
        alpha_y_half > 0.0,
        alpha_y_half * m[:, :-1],
        alpha_y_half * m[:, 1:],
    )
    flux_y_padded = np.empty((n_x, n_y + 1), dtype=np.float64)
    flux_y_padded[:, 1:n_y] = flux_y_int
    if boundary == "periodic":
        wrap_alpha_y = 0.5 * (alpha_y[:, -1] + alpha_y[:, 0])
        wrap_flux_y = np.where(
            wrap_alpha_y > 0.0,
            wrap_alpha_y * m[:, -1],
            wrap_alpha_y * m[:, 0],
        )
        flux_y_padded[:, 0] = wrap_flux_y
        flux_y_padded[:, -1] = wrap_flux_y
    else:  # neumann
        flux_y_padded[:, 0] = 0.0
        flux_y_padded[:, -1] = 0.0
    div_y = (flux_y_padded[:, 1:] - flux_y_padded[:, :-1]) / dy

    return div_x + div_y


def solve_fokker_planck_2d(
    sigma: Union[float, Tuple[float, float]],
    T: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_t: int,
    initial: np.ndarray,
    drift: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    boundary: str = "periodic",
) -> FP2DSolution:
    r"""Solve the forward Fokker-Planck equation on a 2D rectangular grid.

    Parameters
    ----------
    sigma : float or (float, float)
        Diffusion coefficient. Scalar interpreted as isotropic
        (``sigma_x = sigma_y = sigma``); a 2-tuple
        ``(sigma_x, sigma_y)`` enables anisotropic diffusion. Both
        entries must be strictly positive.
    T : float
        Time horizon.
    x_grid, y_grid : ndarray
        Uniformly spaced 1D grids along axes 0 and 1.
    n_t : int
        Number of time steps; ``dt = T / n_t``.
    initial : ndarray, shape (n_x, n_y)
        Initial density. Must be non-negative.
    drift : callable
        Signature ``(t: float, X: (n_x, n_y), Y: (n_x, n_y))
        -> ndarray of shape (n_x, n_y, 2)`` giving the two drift
        components ``(alpha_x, alpha_y)``. The arguments ``X, Y`` are
        the ``ij``-indexed meshgrid arrays.
    boundary : {"periodic", "neumann"}, default "periodic"
        Boundary condition. Periodic wraps fluxes around the box.
        Neumann enforces zero half-grid fluxes at the four boundaries
        (no-flux, conserves total mass).

    Returns
    -------
    FP2DSolution
    """
    sigma_x, sigma_y = _normalize_sigma_2d(sigma)
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
    y = np.asarray(y_grid, dtype=np.float64)
    if x.ndim != 1 or x.size < 4 or y.ndim != 1 or y.size < 4:
        raise ValueError(
            f"x_grid and y_grid must be 1D with >= 4 points, "
            f"got {x.shape}, {y.shape}"
        )
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("grids must be strictly increasing")
    if not np.allclose(np.diff(x), dx, rtol=1e-10, atol=1e-12):
        raise ValueError("x_grid must be uniformly spaced")
    if not np.allclose(np.diff(y), dy, rtol=1e-10, atol=1e-12):
        raise ValueError("y_grid must be uniformly spaced")
    n_x = x.size
    n_y = y.size

    initial_arr = np.ascontiguousarray(initial, dtype=np.float64)
    if initial_arr.shape != (n_x, n_y):
        raise ValueError(
            f"initial must have shape ({n_x}, {n_y}), got {initial_arr.shape}"
        )

    n_t_int = int(n_t)
    dt = float(T) / n_t_int
    t_grid = np.linspace(0.0, float(T), n_t_int + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    diff_lu: SuperLU = _build_implicit_diffusion_2d(
        n_x, n_y, dx, dy, (sigma_x, sigma_y), dt, boundary,
    )

    m = np.empty((n_t_int + 1, n_x, n_y), dtype=np.float64)
    m[0] = initial_arr

    for n in range(n_t_int):
        m_n = m[n]
        alpha_pair = np.asarray(
            drift(float(t_grid[n]), X, Y), dtype=np.float64,
        )
        if alpha_pair.shape != (n_x, n_y, 2):
            raise ValueError(
                f"drift must return shape ({n_x}, {n_y}, 2), "
                f"got {alpha_pair.shape}"
            )
        alpha_x = alpha_pair[..., 0]
        alpha_y = alpha_pair[..., 1]

        max_alpha_x = float(np.max(np.abs(alpha_x)))
        max_alpha_y = float(np.max(np.abs(alpha_y)))
        cfl = max_alpha_x * dt / dx + max_alpha_y * dt / dy
        if cfl > _CFL_HARD_LIMIT or not np.isfinite(cfl):
            recommended_n_t = int(np.ceil(cfl * n_t_int * 1.1))
            raise ValueError(
                f"FP 2D explicit-convection CFL violated at step "
                f"{n + 1}/{n_t_int} (t = {float(t_grid[n]):.4f}): "
                f"max|alpha_x| = {max_alpha_x:.3e}, "
                f"max|alpha_y| = {max_alpha_y:.3e}, "
                f"dt = {dt:.3e}, dx = {dx:.3e}, dy = {dy:.3e}, "
                f"giving CFL = {cfl:.3e} (hard limit {_CFL_HARD_LIMIT}). "
                f"Try n_t >= {recommended_n_t}."
            )

        flux_div = _convective_flux_divergence_2d(
            alpha_x, alpha_y, m_n, dx, dy, boundary,
        )
        rhs = m_n - dt * flux_div
        m_new = diff_lu.solve(rhs.ravel())
        m[n + 1] = m_new.reshape(n_x, n_y)

        if not np.isfinite(m[n + 1]).all():
            raise ValueError(
                f"FP 2D solver produced non-finite values at step "
                f"{n + 1}/{n_t_int}; the per-step CFL guard did not "
                "catch this."
            )

    return FP2DSolution(t_grid=t_grid, x_grid=x, y_grid=y, m=m)
