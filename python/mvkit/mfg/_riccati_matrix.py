"""Matrix Riccati and matrix Lyapunov ODE solvers for the vector
linear-quadratic Mean Field Game.

For a vector LQ-MFG with state :math:`X_t \\in \\mathbb{R}^d`, control-cost
matrix :math:`R \\in \\mathbb{R}^{d \\times d}` (positive definite, often
``I``), state-cost matrix :math:`Q \\in \\mathbb{R}^{d \\times d}`
(symmetric positive semi-definite), terminal weight :math:`Q_T`, and
diffusion matrix :math:`\\Sigma`, the value-function quadratic
coefficient :math:`P(t)` solves the symmetric matrix Riccati ODE

.. math::
    \\dot P = P R^{-1} P - Q, \\qquad P(T) = Q_T,

integrated backward in :math:`t`. In the symmetric LQ-MFG case the
equilibrium mean is constant and the population covariance
:math:`V(t) = \\mathbb{E}[(X_t - m_0)(X_t - m_0)^\\top]` solves the
matrix Lyapunov ODE

.. math::
    \\dot V = -R^{-1} P V - V P R^{-1} + \\Sigma \\Sigma^\\top,
    \\qquad V(0) = V_0,

forward in :math:`t`. Both are integrated by ``scipy.integrate.solve_ivp``
with the Radau method by flattening / reshaping the matrix state.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _check_square_psd(
    M: np.ndarray, name: str, *, positive_definite: bool = False
) -> None:
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(M, M.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric (within tolerance)")
    eigenvalues = np.linalg.eigvalsh(0.5 * (M + M.T))
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if positive_definite:
        if eigenvalues[0] <= 0.0:
            raise ValueError(f"{name} must be positive definite")
    elif eigenvalues[0] < -tolerance:
        raise ValueError(f"{name} must be positive semi-definite")


def solve_matrix_riccati(
    Q: np.ndarray,
    Q_T: np.ndarray,
    R: np.ndarray,
    T: float,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Solve the matrix Riccati ODE

    .. math::
        \dot P = P R^{-1} P - Q, \quad P(T) = Q_T,

    backward in :math:`t` on :math:`[0, T]`.

    Parameters
    ----------
    Q : ndarray, shape (d, d)
        Symmetric state-cost matrix; should be positive semi-definite.
    Q_T : ndarray, shape (d, d)
        Symmetric terminal-cost matrix.
    R : ndarray, shape (d, d)
        Symmetric positive-definite control-cost matrix.
    T : float
        Time horizon.
    n_grid : int, default 200
        Number of intervals; the returned grids have length ``n_grid + 1``.

    Returns
    -------
    t_grid : ndarray, shape (n_grid + 1,)
        Increasing time grid from 0 to T.
    P : ndarray, shape (n_grid + 1, d, d)
        Riccati solution at the grid points.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    Q_arr = np.asarray(Q, dtype=np.float64)
    Q_T_arr = np.asarray(Q_T, dtype=np.float64)
    R_arr = np.asarray(R, dtype=np.float64)
    _check_square_psd(Q_arr, "Q")
    _check_square_psd(Q_T_arr, "Q_T")
    _check_square_psd(R_arr, "R", positive_definite=True)
    d = Q_arr.shape[0]
    if Q_T_arr.shape != (d, d) or R_arr.shape != (d, d):
        raise ValueError(
            f"Q, Q_T, R must all be ({d}, {d}); got Q_T={Q_T_arr.shape}, "
            f"R={R_arr.shape}"
        )

    R_inv = np.linalg.inv(R_arr)
    t_grid = np.linspace(0.0, float(T), int(n_grid) + 1)
    eval_descending = t_grid[::-1]

    def rhs(t: float, P_flat: np.ndarray) -> np.ndarray:
        P = P_flat.reshape(d, d)
        dP = P @ R_inv @ P - Q_arr
        return dP.ravel()

    sol = solve_ivp(
        fun=rhs,
        t_span=(float(T), 0.0),
        y0=Q_T_arr.ravel(),
        method="Radau",
        t_eval=eval_descending,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"matrix Riccati solver failed: {sol.message}")
    P_descending = sol.y.T.reshape(-1, d, d)
    P = P_descending[::-1]
    # Symmetrize to suppress numerical drift away from symmetry.
    P = 0.5 * (P + np.swapaxes(P, 1, 2))
    return t_grid, P


def lq_mfg_analytical_covariance(
    P: np.ndarray,
    t_grid: np.ndarray,
    Sigma: np.ndarray,
    R: np.ndarray,
    V_0: np.ndarray,
) -> np.ndarray:
    r"""Solve the matrix Lyapunov ODE for the equilibrium covariance.

    .. math::
        \dot V = -R^{-1} P V - V P R^{-1} + \Sigma \Sigma^\top,
        \quad V(0) = V_0,

    where ``P(t)`` is the precomputed Riccati solution. ``R^{-1} P`` is
    the closed-loop drift matrix ``A(t)``; the equation reads
    :math:`\dot V = -A V - V A^\top + \Sigma \Sigma^\top` in standard
    Lyapunov form.

    Parameters
    ----------
    P : ndarray, shape (G, d, d)
        Riccati solution from :func:`solve_matrix_riccati`.
    t_grid : ndarray, shape (G,)
        Time grid (must match the one P was sampled on).
    Sigma : ndarray, shape (d, d)
        Diffusion matrix.
    R : ndarray, shape (d, d)
        Control-cost matrix (used to form ``R^{-1} P``).
    V_0 : ndarray, shape (d, d)
        Initial covariance, symmetric positive semi-definite.

    Returns
    -------
    V : ndarray, shape (G, d, d)
        Covariance trajectory at the grid points.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    t_arr = np.asarray(t_grid, dtype=np.float64)
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)
    R_arr = np.asarray(R, dtype=np.float64)
    V0_arr = np.asarray(V_0, dtype=np.float64)
    if P_arr.ndim != 3 or P_arr.shape[1] != P_arr.shape[2]:
        raise ValueError(
            f"P must have shape (G, d, d), got {P_arr.shape}"
        )
    G, d, _ = P_arr.shape
    if t_arr.shape != (G,):
        raise ValueError(
            f"t_grid must have length {G} matching P, got {t_arr.shape}"
        )
    for name, M in (("Sigma", Sigma_arr), ("R", R_arr), ("V_0", V0_arr)):
        if M.shape != (d, d):
            raise ValueError(
                f"{name} must have shape ({d}, {d}), got {M.shape}"
            )
    _check_square_psd(R_arr, "R", positive_definite=True)
    _check_square_psd(V0_arr, "V_0")

    R_inv = np.linalg.inv(R_arr)
    SS_T = Sigma_arr @ Sigma_arr.T

    # Build a flat-index interpolation of P over the grid for the RHS.
    # Linear interpolation is enough; Radau handles smooth integrands.
    def P_at(t: float) -> np.ndarray:
        return np.stack([
            np.interp(t, t_arr, P_arr[:, i, j])
            for i in range(d) for j in range(d)
        ]).reshape(d, d)

    def rhs(t: float, V_flat: np.ndarray) -> np.ndarray:
        V = V_flat.reshape(d, d)
        Pt = P_at(t)
        A = R_inv @ Pt
        dV = -A @ V - V @ A.T + SS_T
        return dV.ravel()

    sol = solve_ivp(
        fun=rhs,
        t_span=(float(t_arr[0]), float(t_arr[-1])),
        y0=V0_arr.ravel(),
        method="Radau",
        t_eval=t_arr,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"matrix Lyapunov solver failed: {sol.message}")
    V = sol.y.T.reshape(-1, d, d)
    V = 0.5 * (V + np.swapaxes(V, 1, 2))
    return V
