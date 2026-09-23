"""Affine HJB coefficients and mean responses for linear-quadratic control."""

from __future__ import annotations

import numpy as np


def affine_response_operator(P, Q, Q_T, R_inv, t_grid):
    """Build the response to a prescribed mean flow.

    The linear coefficient s in u = x.T P x / 2 + s.T x + r solves
    s' = P R^{-1} s + Q m, s(T) = -Q_T m(T). Integrate it backward
    with the second-order trapezoidal rule. The induced deterministic
    state shift is advanced with the same Euler steps as the particles.

    Coefficients have shapes (G, d, d), (d, d), and (G, d). The return
    value maps a mean flow to (s, state_shift). Factoring the backward
    steps once avoids repeating matrix solves during outer iterations.
    """
    dt = np.diff(t_grid)
    d = Q.shape[0]
    identity = np.eye(d)
    backward = P @ R_inv
    left = identity + .5 * dt[:, None, None] * backward[:-1]
    transition = np.linalg.solve(
        left, identity - .5 * dt[:, None, None] * backward[1:]
    )
    forcing = np.linalg.solve(left, .5 * dt[:, None, None] * Q)
    forward = R_inv @ P[:-1]

    def response(m):
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (len(t_grid), d) or not np.isfinite(m).all():
            raise ValueError(f"mean flow must be finite with shape {(len(t_grid), d)}")
        s = np.empty_like(m)
        s[-1] = -Q_T @ m[-1]
        for k in range(len(dt) - 1, -1, -1):
            s[k] = transition[k] @ s[k + 1] - forcing[k] @ (m[k] + m[k + 1])
        shift = np.zeros_like(m)
        for k, h in enumerate(dt):
            shift[k + 1] = shift[k] - h * (forward[k] @ shift[k] + R_inv @ s[k])
        return s, shift

    return response
