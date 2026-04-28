"""Numerical Riccati ODE solver for the scalar LQ Mean Field Game.

Solves the backward scalar ODE

.. math::
    \\dot P(t) = P(t)^2 - q, \\quad P(T) = q_T,

on :math:`[0, T]`. Used by :mod:`mvkit.mfg.linear_quadratic` once per call,
independently of the equilibrium mean trajectory. ``scipy.integrate.solve_ivp``
with the Radau method handles the mild stiffness near the terminal time when
``q_T`` is close to ``\\sqrt{q}`` (where :math:`\\tanh \\to 1`).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def solve_riccati(
    q: float,
    q_T: float,
    T: float,
    n_grid: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Solve the scalar LQ-MFG Riccati ODE.

    .. math::
        \dot P(t) = P(t)^2 - q, \quad P(T) = q_T,

    integrated backward in :math:`t` on :math:`[0, T]`.

    Parameters
    ----------
    q : float
        Running-cost weight on :math:`(X_t - m_t)^2`. Must be non-negative.
    q_T : float
        Terminal-cost weight on :math:`(X_T - m_T)^2`. Must be non-negative.
    T : float
        Horizon. Must be strictly positive.
    n_grid : int, default 500
        Number of integration intervals. The returned grids have length
        ``n_grid + 1``.

    Returns
    -------
    t_grid : ndarray, shape (n_grid + 1,)
        Increasing time grid from 0 to T.
    P : ndarray, shape (n_grid + 1,)
        Riccati solution P(t) at the grid points.

    Notes
    -----
    For ``q > 0`` and ``q_T < sqrt(q)`` the closed form is

    .. math::
        P(t) = \omega \tanh\big(\omega (T - t) + \mathrm{atanh}(q_T / \omega)\big),
        \qquad \omega = \sqrt{q}.

    Used as a regression check in the test suite.
    """
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if q < 0.0:
        raise ValueError(f"q must be non-negative, got {q}")
    if q_T < 0.0:
        raise ValueError(f"q_T must be non-negative, got {q_T}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    t_grid = np.linspace(0.0, float(T), int(n_grid) + 1)
    eval_descending = t_grid[::-1]

    sol = solve_ivp(
        fun=lambda t, P: P * P - q,
        t_span=(float(T), 0.0),
        y0=np.array([float(q_T)], dtype=np.float64),
        method="Radau",
        t_eval=eval_descending,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Riccati ODE solver failed: {sol.message}")
    P = np.asarray(sol.y[0, ::-1], dtype=np.float64)
    return t_grid, P
