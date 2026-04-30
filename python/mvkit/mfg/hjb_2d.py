"""Backward Hamilton-Jacobi-Bellman solver on a 2D rectangular grid.

For a quadratic Hamiltonian :math:`H(\\nabla u, x, m_t) = \\tfrac{1}{2}
|\\nabla u|^2 - F(x, m_t)` with isotropic scalar diffusion :math:`\\sigma`,
the 2D HJB reads

.. math::
    \\partial_t u - \\tfrac{1}{2}|\\nabla u|^2 + F(x, m_t)
    + \\tfrac{1}{2}\\sigma^2 \\Delta u = 0,
    \\qquad u(T, x) = g(x, m_T).

Discretization on a rectangular grid:

- **Spatial**: per-axis Engquist-Osher upwind summed across axes
  :math:`H_h = \\tfrac{1}{2}\\sum_{a \\in \\{x, y\\}}
  [\\max(D^-_a u, 0)^2 + \\min(D^+_a u, 0)^2]`,
  central differences for the Laplacian.
- **Time**: implicit-explicit (IMEX). The implicit-diffusion operator
  is the 2D periodic-or-Neumann Laplacian assembled via a Kronecker
  sum :math:`A_x \\otimes I_{n_y} + I_{n_x} \\otimes A_y`, then
  ``I - (sigma^2 dt / 2) D^2`` is factored once at setup with sparse
  LU (``scipy.sparse.linalg.splu``).

Per backward step the cost is one sparse triangular solve of size
:math:`n_x n_y`, which is fast for :math:`n_x, n_y \\lesssim 200`.

CFL guard: the explicit-Hamiltonian step requires
:math:`\\Delta t (\\max|\\partial_x u| / \\Delta x +
\\max|\\partial_y u| / \\Delta y) \\le 1`. We raise above ``2.0`` (the
same hard limit as the 1D solver, which leaves headroom for the
implicit-diffusion smoothing).

Both periodic (the grid wraps in both axes) and Neumann
(:math:`\\partial_n u = 0` at all four boundaries via reflecting ghost
cells) are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Union

import numpy as np
from scipy.sparse import csc_matrix, eye as sparse_eye, kron as sparse_kron
from scipy.sparse.linalg import SuperLU, splu


_CFL_HARD_LIMIT = 2.0


def _normalize_sigma_2d(
    sigma: Union[float, Tuple[float, float]],
) -> Tuple[float, float]:
    """Coerce ``sigma`` to a ``(sigma_x, sigma_y)`` tuple.

    A scalar broadcasts to both axes (isotropic, the v0.1 behavior).
    A 2-tuple is taken verbatim. Both entries must be strictly
    positive and finite.
    """
    if np.isscalar(sigma):
        s = float(sigma)
        sigma_xy = (s, s)
    else:
        try:
            sx, sy = sigma  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sigma must be a positive float or a (sigma_x, sigma_y) "
                f"tuple, got {sigma!r}"
            ) from exc
        sigma_xy = (float(sx), float(sy))
    for name, val in zip(("sigma_x", "sigma_y"), sigma_xy):
        if not (np.isfinite(val) and val > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {val}")
    return sigma_xy


@dataclass
class HJB2DSolution:
    """Output of :func:`solve_hjb_2d`.

    Attributes
    ----------
    t_grid : ndarray, shape (n_t + 1,)
        Increasing time grid from 0 to T.
    x_grid : ndarray, shape (n_x,)
        Spatial grid along axis 0.
    y_grid : ndarray, shape (n_y,)
        Spatial grid along axis 1.
    u : ndarray, shape (n_t + 1, n_x, n_y)
        Value function on the spatio-temporal grid.
    optimal_drift : ndarray, shape (n_t + 1, n_x, n_y, 2)
        Optimal control :math:`\\alpha^* = -\\nabla u`, with components
        ``[..., 0]`` (along x) and ``[..., 1]`` (along y).
    """

    t_grid: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    u: np.ndarray
    optimal_drift: np.ndarray


def _periodic_1d_laplacian(n: int, dx: float) -> csc_matrix:
    """Sparse 1D periodic discrete Laplacian, shape (n, n)."""
    inv_dx2 = 1.0 / (dx * dx)
    rows = []
    cols = []
    data = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(-2.0 * inv_dx2)
        rows.append(i)
        cols.append((i - 1) % n)
        data.append(inv_dx2)
        rows.append(i)
        cols.append((i + 1) % n)
        data.append(inv_dx2)
    return csc_matrix((data, (rows, cols)), shape=(n, n))


def _neumann_1d_laplacian(n: int, dx: float) -> csc_matrix:
    """Sparse 1D Neumann discrete Laplacian (reflecting ghosts), shape (n, n).

    Boundary rows have diagonal ``-1/dx^2`` (instead of ``-2/dx^2``)
    and a single off-diagonal ``+1/dx^2``.
    """
    inv_dx2 = 1.0 / (dx * dx)
    rows = []
    cols = []
    data = []
    for i in range(n):
        if i == 0:
            rows.append(i)
            cols.append(i)
            data.append(-inv_dx2)
            rows.append(i)
            cols.append(1)
            data.append(inv_dx2)
        elif i == n - 1:
            rows.append(i)
            cols.append(i)
            data.append(-inv_dx2)
            rows.append(i)
            cols.append(n - 2)
            data.append(inv_dx2)
        else:
            rows.append(i)
            cols.append(i)
            data.append(-2.0 * inv_dx2)
            rows.append(i)
            cols.append(i - 1)
            data.append(inv_dx2)
            rows.append(i)
            cols.append(i + 1)
            data.append(inv_dx2)
    return csc_matrix((data, (rows, cols)), shape=(n, n))


def _build_implicit_diffusion_2d(
    n_x: int, n_y: int, dx: float, dy: float,
    sigma: Union[float, Tuple[float, float]],
    dt: float, boundary: str,
) -> SuperLU:
    """Factor ``I - (dt / 2) (sigma_x^2 partial_{xx} + sigma_y^2 partial_{yy})``
    for the 2D anisotropic-diffusion operator under periodic or Neumann
    BC. Returned as a ``scipy.sparse.linalg.SuperLU`` object.

    ``sigma`` may be a scalar (isotropic, equivalent to a tuple
    ``(s, s)``) or a 2-tuple ``(sigma_x, sigma_y)``.
    """
    sigma_x, sigma_y = _normalize_sigma_2d(sigma)
    if boundary == "periodic":
        Ax = _periodic_1d_laplacian(n_x, dx)
        Ay = _periodic_1d_laplacian(n_y, dy)
    else:
        Ax = _neumann_1d_laplacian(n_x, dx)
        Ay = _neumann_1d_laplacian(n_y, dy)
    Ix = sparse_eye(n_x, format="csc")
    Iy = sparse_eye(n_y, format="csc")
    L = (
        (sigma_x * sigma_x) * sparse_kron(Ax, Iy, format="csc")
        + (sigma_y * sigma_y) * sparse_kron(Ix, Ay, format="csc")
    )
    n_total = n_x * n_y
    M = sparse_eye(n_total, format="csc") - 0.5 * dt * L
    return splu(M.tocsc())


def _spatial_diffs_2d(
    u: np.ndarray, dx: float, dy: float, boundary: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis backward and forward differences with the requested BC."""
    if boundary == "periodic":
        u_xm = np.roll(u, 1, axis=0)
        u_xp = np.roll(u, -1, axis=0)
        u_ym = np.roll(u, 1, axis=1)
        u_yp = np.roll(u, -1, axis=1)
    else:  # neumann
        u_xm = np.empty_like(u)
        u_xm[0, :] = u[0, :]
        u_xm[1:, :] = u[:-1, :]
        u_xp = np.empty_like(u)
        u_xp[-1, :] = u[-1, :]
        u_xp[:-1, :] = u[1:, :]
        u_ym = np.empty_like(u)
        u_ym[:, 0] = u[:, 0]
        u_ym[:, 1:] = u[:, :-1]
        u_yp = np.empty_like(u)
        u_yp[:, -1] = u[:, -1]
        u_yp[:, :-1] = u[:, 1:]
    Dx_minus = (u - u_xm) / dx
    Dx_plus = (u_xp - u) / dx
    Dy_minus = (u - u_ym) / dy
    Dy_plus = (u_yp - u) / dy
    return Dx_minus, Dx_plus, Dy_minus, Dy_plus


def _engquist_osher_quadratic_2d(
    Dx_minus: np.ndarray, Dx_plus: np.ndarray,
    Dy_minus: np.ndarray, Dy_plus: np.ndarray,
) -> np.ndarray:
    """EO numerical Hamiltonian for ``H(p) = (1/2) |p|^2`` summed over axes."""
    return 0.5 * (
        np.maximum(Dx_minus, 0.0) ** 2
        + np.minimum(Dx_plus, 0.0) ** 2
        + np.maximum(Dy_minus, 0.0) ** 2
        + np.minimum(Dy_plus, 0.0) ** 2
    )


def solve_hjb_2d(
    sigma: Union[float, Tuple[float, float]],
    T: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    n_t: int,
    terminal: np.ndarray,
    running_cost: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    boundary: str = "periodic",
    hamiltonian: str = "quadratic",
) -> HJB2DSolution:
    r"""Solve the backward HJB on a 2D rectangular grid.

    Parameters
    ----------
    sigma : float or (float, float)
        Diffusion coefficient. A scalar is interpreted as isotropic
        (``sigma_x = sigma_y = sigma``). A 2-tuple
        ``(sigma_x, sigma_y)`` allows different diffusion magnitudes
        per axis. Both entries must be strictly positive.
    T : float
        Time horizon.
    x_grid : ndarray, shape (n_x,)
        Uniformly spaced spatial grid along axis 0; ``dx`` is inferred
        as the constant spacing.
    y_grid : ndarray, shape (n_y,)
        Uniformly spaced spatial grid along axis 1.
    n_t : int
        Number of time steps; ``dt = T / n_t``.
    terminal : ndarray, shape (n_x, n_y)
        Terminal data ``u(T, x_i, y_j)``.
    running_cost : callable
        Signature ``(t: float, X: (n_x, n_y), Y: (n_x, n_y)) ->
        ndarray of shape (n_x, n_y)``. The arguments ``X, Y`` are the
        ``ij``-indexed meshgrid arrays.
    boundary : {"periodic", "neumann"}, default "periodic"
    hamiltonian : str, default "quadratic"
        Only ``"quadratic"`` (i.e. :math:`H(p) = |p|^2/2 - F`) is
        supported in v0.1.

    Returns
    -------
    HJB2DSolution
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
    if hamiltonian != "quadratic":
        raise ValueError(
            f"hamiltonian={hamiltonian!r} not supported; expected 'quadratic'"
        )

    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    if x.ndim != 1 or x.size < 4:
        raise ValueError(
            f"x_grid must be 1D with >= 4 points, got {x.shape}"
        )
    if y.ndim != 1 or y.size < 4:
        raise ValueError(
            f"y_grid must be 1D with >= 4 points, got {y.shape}"
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

    terminal_arr = np.ascontiguousarray(terminal, dtype=np.float64)
    if terminal_arr.shape != (n_x, n_y):
        raise ValueError(
            f"terminal must have shape ({n_x}, {n_y}), got "
            f"{terminal_arr.shape}"
        )

    n_t_int = int(n_t)
    dt = float(T) / n_t_int
    t_grid = np.linspace(0.0, float(T), n_t_int + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    diff_lu = _build_implicit_diffusion_2d(
        n_x, n_y, dx, dy, (sigma_x, sigma_y), dt, boundary,
    )

    u = np.empty((n_t_int + 1, n_x, n_y), dtype=np.float64)
    u[n_t_int] = terminal_arr

    for n in range(n_t_int, 0, -1):
        u_n = u[n]
        Dx_minus, Dx_plus, Dy_minus, Dy_plus = _spatial_diffs_2d(
            u_n, dx, dy, boundary,
        )

        # CFL: dt * (max|d_x u| / dx + max|d_y u| / dy) <= hard limit.
        max_grad_x = float(max(
            np.max(np.abs(Dx_minus)), np.max(np.abs(Dx_plus)),
        ))
        max_grad_y = float(max(
            np.max(np.abs(Dy_minus)), np.max(np.abs(Dy_plus)),
        ))
        cfl = max_grad_x * dt / dx + max_grad_y * dt / dy
        if cfl > _CFL_HARD_LIMIT or not np.isfinite(cfl):
            step_idx = n_t_int - n + 1
            recommended_n_t = int(np.ceil(cfl * n_t_int * 1.1))
            raise ValueError(
                f"HJB 2D explicit-Hamiltonian CFL violated at step "
                f"{step_idx}/{n_t_int} (t = {float(t_grid[n]):.4f}): "
                f"max|d_x u| ~ {max_grad_x:.3e}, max|d_y u| ~ "
                f"{max_grad_y:.3e}, dt = {dt:.3e}, dx = {dx:.3e}, "
                f"dy = {dy:.3e}, giving CFL = {cfl:.3e} (hard limit "
                f"{_CFL_HARD_LIMIT}). Try n_t >= {recommended_n_t}."
            )

        h_num = _engquist_osher_quadratic_2d(
            Dx_minus, Dx_plus, Dy_minus, Dy_plus,
        )
        f_n = np.asarray(
            running_cost(float(t_grid[n]), X, Y), dtype=np.float64,
        )
        if f_n.shape != (n_x, n_y):
            raise ValueError(
                f"running_cost must return shape ({n_x}, {n_y}), "
                f"got {f_n.shape}"
            )
        rhs = u_n + dt * (-h_num + f_n)
        u_new = diff_lu.solve(rhs.ravel())
        u[n - 1] = u_new.reshape(n_x, n_y)

        if not np.isfinite(u[n - 1]).all():
            step_idx = n_t_int - n + 1
            raise ValueError(
                f"HJB 2D solver produced non-finite values at step "
                f"{step_idx}/{n_t_int}; the per-step CFL guard did not "
                "catch this. Please report with a minimal reproducer."
            )

    # Optimal drift alpha* = -grad u, central difference per axis,
    # with periodic / Neumann boundary handling.
    drift = np.empty((n_t_int + 1, n_x, n_y, 2), dtype=np.float64)
    for k in range(n_t_int + 1):
        Dx_minus, Dx_plus, Dy_minus, Dy_plus = _spatial_diffs_2d(
            u[k], dx, dy, boundary,
        )
        drift[k, ..., 0] = -(Dx_minus + Dx_plus) / 2.0
        drift[k, ..., 1] = -(Dy_minus + Dy_plus) / 2.0

    return HJB2DSolution(
        t_grid=t_grid, x_grid=x, y_grid=y, u=u, optimal_drift=drift,
    )
