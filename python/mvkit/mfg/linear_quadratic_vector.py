"""Vector linear-quadratic Mean Field Game in :math:`\\mathbb{R}^d`.

State :math:`X_t \\in \\mathbb{R}^d` evolves as

.. math::
    \\mathrm{d}X_t = \\alpha_t \\, \\mathrm{d}t + \\Sigma \\, \\mathrm{d}W_t,
    \\qquad X_0 \\sim \\mathcal{N}(\\mu_0, V_0),

where :math:`\\alpha_t \\in \\mathbb{R}^d` is the control,
:math:`\\Sigma \\in \\mathbb{R}^{d \\times d}` shapes the noise, and
:math:`W_t` is a :math:`d`-dimensional standard Brownian motion.

The cost functional, against a fixed mean trajectory
:math:`m_t \\in \\mathbb{R}^d`, is

.. math::
    J(\\alpha; m) = \\mathbb{E}\\Big[
      \\int_0^T \\big(\\tfrac{1}{2} \\alpha_t^\\top R \\alpha_t
        + \\tfrac{1}{2} (X_t - m_t)^\\top Q (X_t - m_t)\\big)\\,\\mathrm{d}t
      + \\tfrac{1}{2}(X_T - m_T)^\\top Q_T (X_T - m_T) \\Big].

The HJB ansatz ``u(t,x) = x.T P(t) x / 2 + s(t).T x + r(t)``
reduces the problem to a matrix Riccati equation and the backward equation
``s' = P R^{-1} s + Q m``, with ``s(T) = -Q_T m(T)``. The control is
``alpha*(t,x) = -R^{-1}(P(t) x + s(t))``. Only for a constant input mean
does this reduce to ``-R^{-1} P(t)(x-m)``.
In the symmetric LQ-MFG case the equilibrium mean is constant
(:math:`m_t = \\mu_0`) and the variance matrix
:math:`V(t) = \\mathbb{E}[(X_t - \\mu_0)(X_t - \\mu_0)^\\top]` solves the
matrix Lyapunov ODE
:math:`\\dot V = -R^{-1} P V - V P R^{-1} + \\Sigma \\Sigma^\\top`.

Both Picard and Fictitious Play outer iterations are supported, with
the same dispatch as :func:`solve_lq_mfg`. The simulation runs in pure
numpy (vectorized over particles); for reasonable problem sizes
(d <= ~10, N <= ~10000, n_grid <= ~500) this is fast enough that no
Rust offload is needed.

References
----------
Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field
Games with Applications I*, Chapter 3 (vector LQ-MFG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .._progress import progress_iter
from ._random import StreamRole, rng_for_role
from ._lq_control import affine_response_operator
from ._riccati_matrix import (
    _check_square_psd,
    lq_mfg_analytical_covariance,
    solve_matrix_riccati,
)

_VALID_METHODS = ("picard", "fictitious_play")


@dataclass
class LQMFGVectorSolution:
    """Output of :func:`solve_lq_mfg_vector`.

    Attributes
    ----------
    t_grid : ndarray, shape (G,)
        Increasing time grid from 0 to T.
    P : ndarray, shape (G, d, d)
        Matrix Riccati solution.
    m : ndarray, shape (G, d)
        Equilibrium mean trajectory (the converged Picard fixed point).
    V : ndarray, shape (G, d, d)
        Variance-matrix trajectory (analytical, from the Lyapunov ODE).
    x_trajectory : ndarray, shape (G, N, d)
        Particle trajectory for the last best response; inspect ``converged``.
    m_iterates : list of ndarray
        Outer-iteration means; ``m_iterates[0]`` is the starting guess.
    n_iterations : int
        Number of outer updates performed.
    converged : bool
        True if ``fixed_point_residual < tol``.
    fixed_point_residual : float
        Sup norm of ``BR(m) - m`` at the returned mean and fixed noise.
    """

    t_grid: np.ndarray
    P: np.ndarray
    m: np.ndarray
    V: np.ndarray
    x_trajectory: np.ndarray
    m_iterates: List[np.ndarray] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    fixed_point_residual: float = float("inf")


def _validate_inputs(
    Q: np.ndarray, Q_T: np.ndarray, Sigma: np.ndarray, R: np.ndarray,
    T: float, mu_0_mean: np.ndarray, mu_0_var: np.ndarray,
    n_particles: int, n_grid: int,
    n_iterations_max: int, tol: float,
) -> int:
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if n_iterations_max < 1:
        raise ValueError(
            f"n_iterations_max must be >= 1, got {n_iterations_max}"
        )
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    Q_arr = np.asarray(Q, dtype=np.float64)
    if Q_arr.ndim != 2 or Q_arr.shape[0] != Q_arr.shape[1]:
        raise ValueError(f"Q must be (d, d), got shape {Q_arr.shape}")
    d = Q_arr.shape[0]
    for name, M in (("Q_T", Q_T), ("Sigma", Sigma), ("R", R)):
        arr = np.asarray(M, dtype=np.float64)
        if arr.shape != (d, d):
            raise ValueError(
                f"{name} must have shape ({d}, {d}), got {arr.shape}"
            )
    mu_arr = np.asarray(mu_0_mean, dtype=np.float64)
    if mu_arr.shape != (d,):
        raise ValueError(
            f"mu_0_mean must have shape ({d},), got {mu_arr.shape}"
        )
    V0_arr = np.asarray(mu_0_var, dtype=np.float64)
    if V0_arr.shape != (d, d):
        raise ValueError(
            f"mu_0_var must have shape ({d}, {d}), got {V0_arr.shape}"
        )
    return d


def _draw_x0(
    mu_0_mean: np.ndarray, V_0: np.ndarray, n_particles: int, seed: int
) -> np.ndarray:
    """Sample ``x0 ~ N(mu_0_mean, V_0)`` for the particle ensemble.

    Returns an array of shape ``(N, d)``.
    """
    rng = rng_for_role(seed, StreamRole.VECTOR_INITIAL)
    d = mu_0_mean.shape[0]
    _check_square_psd(V_0, "mu_0_var")
    values, vectors = np.linalg.eigh(V_0)
    L = vectors * np.sqrt(np.maximum(values, 0.0))
    z = rng.standard_normal(size=(int(n_particles), d))
    return mu_0_mean + z @ L.T


def _simulate_under_vector_control(
    x0: np.ndarray,
    t_grid: np.ndarray,
    P: np.ndarray,
    linear_term: np.ndarray,
    Sigma: np.ndarray,
    R_inv: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Forward Euler-Maruyama for ``dX = -R^{-1}(P X + s) dt + Sigma dW``.

    ``P`` has shape ``(G, d, d)``, ``linear_term`` has shape ``(G, d)``,
    ``Sigma`` has shape ``(d, d)``. Returns the trajectory of shape
    ``(G, N, d)``.
    """
    G = t_grid.size
    N, d = x0.shape
    rng = rng_for_role(seed, StreamRole.VECTOR_DYNAMICS)
    x = x0.astype(np.float64, copy=True)
    traj = np.empty((G, N, d), dtype=np.float64)
    traj[0] = x
    for step in range(G - 1):
        dt = float(t_grid[step + 1] - t_grid[step])
        A = R_inv @ P[step]  # (d, d)
        drift = -x @ A.T - linear_term[step] @ R_inv.T
        z = rng.standard_normal(size=(N, d))
        noise = z @ Sigma.T
        x = x + drift * dt + noise * np.sqrt(dt)
        traj[step + 1] = x
    return traj


def _initial_mean_iterate(
    m_initial: Optional[np.ndarray], G: int, d: int, mu_0_mean: np.ndarray,
) -> np.ndarray:
    if m_initial is None:
        return np.broadcast_to(mu_0_mean, (G, d)).copy()
    arr = np.asarray(m_initial, dtype=np.float64).copy()
    if arr.shape != (G, d):
        raise ValueError(
            f"m_initial must have shape ({G}, {d}), got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("m_initial must be finite")
    return arr


def solve_lq_mfg_vector(
    Q: np.ndarray,
    Q_T: np.ndarray,
    Sigma: np.ndarray,
    T: float,
    mu_0_mean: np.ndarray,
    mu_0_var: np.ndarray,
    R: Optional[np.ndarray] = None,
    n_particles: int = 5000,
    n_grid: int = 500,
    n_iterations_max: int = 20,
    tol: float = 1e-4,
    seed: int = 42,
    m_initial: Optional[np.ndarray] = None,
    method: str = "picard",
    damping_burn_in: int = 0,
    progress: bool = False,
) -> LQMFGVectorSolution:
    r"""Solve the vector linear-quadratic Mean Field Game in
    :math:`\mathbb{R}^d` iteratively.

    Algorithm:

    1. Solve the matrix Riccati ODE for ``P(t)`` once.
    2. Sample ``x0 ~ N(mu_0_mean, mu_0_var)`` and pick an initial
       mean guess ``m^(0)(t)`` (constant equal to ``mu_0_mean`` by
       default).
    3. Repeat: form an input mean ``m^in_k`` (latest iterate for
       Picard, historical average for Fictitious Play), simulate the
       controlled SDE forward, set ``m^(k+1) = empirical mean of
       trajectory``. Stop when
       ``max_t || BR(m^(k+1)) - m^(k+1) ||_inf < tol``.

    See :func:`mvkit.mfg.solve_lq_mfg` for the scalar version (``d = 1``);
    the two solvers agree on the d=1 LQ-MFG within MC tolerance.

    Parameters
    ----------
    Q, Q_T : ndarray, shape (d, d)
        Symmetric state-cost and terminal-cost matrices. Both must be
        positive semi-definite.
    Sigma : ndarray, shape (d, d)
        Noise scale matrix. Total diffusion is ``Sigma @ Sigma.T``.
    T : float
        Time horizon.
    mu_0_mean : ndarray, shape (d,)
        Mean of the initial Gaussian distribution.
    mu_0_var : ndarray, shape (d, d)
        Covariance of the initial Gaussian distribution; symmetric
        positive semi-definite.
    R : ndarray, shape (d, d), optional
        Symmetric positive-definite control-cost matrix. Defaults to
        identity.
    n_particles : int, default 5000
        Particle count for the inner forward simulation.
    n_grid : int, default 500
        Time discretization (grids of length ``n_grid + 1``).
    n_iterations_max : int, default 20
        Cap on outer-iteration updates.
    tol : float, default 1e-4
        Sup-norm tolerance on ``BR(m) - m`` at the returned mean.
    seed : int, default 42
        Master RNG seed, with separate PCG64/SeedSequence roles for initial
        states and dynamics. Identical inputs and seed are repeatable in the
        same environment. Noise is reused across outer iterations.
    m_initial : ndarray of shape (n_grid + 1, d), optional
        Custom initial mean guess for the outer iteration.
    method : {"picard", "fictitious_play"}, default "picard"
    damping_burn_in : int, default 0
        Only used with Fictitious Play.
    progress : bool, default False
        If True and ``tqdm`` is available, wrap the outer iteration
        in a progress bar.

    Returns
    -------
    LQMFGVectorSolution

    See Also
    --------
    mvkit.mfg.solve_lq_mfg : The scalar (d=1) version with simpler API.
    mvkit.mfg.solve_mfg : The generic grid-based solver for non-LQ
        problems in 1D.
    """
    if R is None:
        Q_shape = np.asarray(Q).shape
        d_guess = Q_shape[0] if len(Q_shape) == 2 else 1
        R = np.eye(d_guess)
    if method not in _VALID_METHODS:
        raise ValueError(
            f"unknown method '{method}'; expected one of "
            f"{list(_VALID_METHODS)}"
        )
    if damping_burn_in < 0:
        raise ValueError(
            f"damping_burn_in must be non-negative, got {damping_burn_in}"
        )
    d = _validate_inputs(
        Q, Q_T, Sigma, R, T, mu_0_mean, mu_0_var,
        n_particles, n_grid, n_iterations_max, tol,
    )

    Q_arr = np.asarray(Q, dtype=np.float64)
    Q_T_arr = np.asarray(Q_T, dtype=np.float64)
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)
    R_arr = np.asarray(R, dtype=np.float64)
    mu_arr = np.asarray(mu_0_mean, dtype=np.float64)
    V0_arr = np.asarray(mu_0_var, dtype=np.float64)

    for name, matrix in (("Q", Q_arr), ("Q_T", Q_T_arr), ("mu_0_var", V0_arr)):
        _check_square_psd(matrix, name)
    _check_square_psd(R_arr, "R", positive_definite=True)
    if not np.isfinite(Sigma_arr).all() or not np.isfinite(mu_arr).all():
        raise ValueError("Sigma and mu_0_mean must be finite")
    R_inv = np.linalg.inv(R_arr)

    t_grid, P = solve_matrix_riccati(Q_arr, Q_T_arr, R_arr, T, n_grid=n_grid)
    G = t_grid.size

    m_curr = _initial_mean_iterate(m_initial, G, d, mu_arr)
    m_iterates: List[np.ndarray] = [m_curr.copy()]
    converged = False
    operator = affine_response_operator(P, Q_arr, Q_T_arr, R_inv, t_grid)
    base = _simulate_under_vector_control(
        _draw_x0(mu_arr, V0_arr, int(n_particles), int(seed)), t_grid,
        P, np.zeros((G, d)), Sigma_arr, R_inv, int(seed),
    )
    base_mean = base.mean(axis=1)

    def best_response(m):
        _, shift = operator(m)
        return base_mean + shift, shift

    residual = float("inf")
    last_shift = np.zeros((G, d))

    sum_post_burnin: Optional[np.ndarray] = None
    count_post_burnin = 0

    iterator = progress_iter(
        range(n_iterations_max),
        total=n_iterations_max,
        description=f"solve_lq_mfg_vector ({method})",
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

        m_next, last_shift = best_response(m_input)
        check, _ = best_response(m_next)
        residual = float(np.max(np.abs(check - m_next)))
        m_iterates.append(m_next.copy())

        if method == "fictitious_play" and k >= damping_burn_in:
            assert sum_post_burnin is not None
            sum_post_burnin += m_next
            count_post_burnin += 1

        m_curr = m_next
        if residual < tol:
            converged = True
            break

    traj = base + last_shift[:, None, :]
    m_curr = traj.mean(axis=1)
    check, _ = best_response(m_curr)
    residual = float(np.max(np.abs(check - m_curr)))
    converged = residual < tol

    n_iterations = len(m_iterates) - 1
    V = lq_mfg_analytical_covariance(P, t_grid, Sigma_arr, R_arr, V0_arr)

    return LQMFGVectorSolution(
        t_grid=t_grid,
        P=P,
        m=m_curr,
        V=V,
        x_trajectory=traj,
        m_iterates=m_iterates,
        n_iterations=n_iterations,
        converged=converged,
        fixed_point_residual=residual,
    )
