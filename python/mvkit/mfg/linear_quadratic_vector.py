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

The HJB ansatz :math:`u(t, x) = \\tfrac{1}{2}(x - m)^\\top P(t) (x - m)
+ Q^\\mathrm{lin}(t)^\\top(x - m) + R^\\mathrm{const}(t)` reduces the
problem to a matrix Riccati ODE for :math:`P(t)` (independent of
:math:`m`) and a linear ODE for the linear part. The optimal control is
:math:`\\alpha^*(t, x) = -R^{-1} P(t)(x - m_t)`, so the closed-loop SDE
is :math:`\\mathrm{d}X = -R^{-1} P (X - m)\\,\\mathrm{d}t + \\Sigma\\,\\mathrm{d}W`.
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
from ._riccati_matrix import (
    lq_mfg_analytical_covariance,
    solve_matrix_riccati,
)

_VALID_METHODS = ("picard", "fictitious_play")
_SIM_SEED_MASK = 0x5A5A5A5A


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
        Particle trajectory under the converged control.
    m_iterates : list of ndarray
        Outer-iteration means; ``m_iterates[0]`` is the starting guess.
    n_iterations : int
        Number of outer updates performed.
    converged : bool
        True if the sup-norm tolerance was met before
        ``n_iterations_max``.
    """

    t_grid: np.ndarray
    P: np.ndarray
    m: np.ndarray
    V: np.ndarray
    x_trajectory: np.ndarray
    m_iterates: List[np.ndarray] = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False


def _validate_inputs(
    Q: np.ndarray, Q_T: np.ndarray, Sigma: np.ndarray, R: np.ndarray,
    T: float, mu_0_mean: np.ndarray, mu_0_var: np.ndarray,
    n_particles: int, n_grid: int,
    n_iterations_max: int, tol: float,
) -> int:
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    if n_particles < 2:
        raise ValueError(f"n_particles must be >= 2, got {n_particles}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if n_iterations_max < 1:
        raise ValueError(
            f"n_iterations_max must be >= 1, got {n_iterations_max}"
        )
    if tol <= 0.0:
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
    rng = np.random.default_rng(int(seed))
    d = mu_0_mean.shape[0]
    L = np.linalg.cholesky(V_0) if d > 0 else np.zeros((0, 0))
    z = rng.standard_normal(size=(int(n_particles), d))
    return mu_0_mean + z @ L.T


def _simulate_under_vector_control(
    x0: np.ndarray,
    t_grid: np.ndarray,
    P: np.ndarray,
    m_input: np.ndarray,
    Sigma: np.ndarray,
    R_inv: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Forward Euler-Maruyama for ``dX = -R^{-1} P (X - m_input) dt
    + Sigma dW`` on the time grid.

    ``P`` has shape ``(G, d, d)``, ``m_input`` has shape ``(G, d)``,
    ``Sigma`` has shape ``(d, d)``. Returns the trajectory of shape
    ``(G, N, d)``.
    """
    G = t_grid.size
    N, d = x0.shape
    rng = np.random.default_rng(int(seed) ^ _SIM_SEED_MASK)
    x = x0.astype(np.float64, copy=True)
    traj = np.empty((G, N, d), dtype=np.float64)
    traj[0] = x
    for step in range(G - 1):
        dt = float(t_grid[step + 1] - t_grid[step])
        A = R_inv @ P[step]  # (d, d)
        drift = -(x - m_input[step]) @ A.T
        z = rng.standard_normal(size=(N, d))
        noise = z @ Sigma.T
        x = x + drift * dt + noise * np.sqrt(dt)
        traj[step + 1] = x
    return traj


def _lq_vector_best_response(
    m_input: np.ndarray,
    P: np.ndarray,
    t_grid: np.ndarray,
    mu_0_mean: np.ndarray,
    mu_0_var: np.ndarray,
    Sigma: np.ndarray,
    R_inv: np.ndarray,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    """BR(m_input) for the vector LQ-MFG. Returns the empirical mean
    trajectory of the simulated particles, shape ``(G, d)``.
    """
    x0 = _draw_x0(mu_0_mean, mu_0_var, n_particles, seed)
    traj = _simulate_under_vector_control(
        x0=x0, t_grid=t_grid, P=P, m_input=m_input,
        Sigma=Sigma, R_inv=R_inv, seed=seed,
    )
    return traj.mean(axis=1)


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
       ``max_t || m^(k+1) - m^(k) ||_inf < tol``.

    See :func:`mvkit.mfg.solve_lq_mfg` for the scalar version (``d = 1``);
    the two solvers agree on the d=1 LQ-MFG within MC tolerance.

    Parameters
    ----------
    Q, Q_T : ndarray, shape (d, d)
        Symmetric state-cost and terminal-cost matrices. Both should be
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
        Sup-norm tolerance on consecutive mean iterates.
    seed : int, default 42
        Master RNG seed; identical seed plus identical inputs give
        identical outputs.
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
        d_guess = np.asarray(Q).shape[0] if hasattr(Q, "shape") else 1
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

    R_inv = np.linalg.inv(R_arr)

    t_grid, P = solve_matrix_riccati(Q_arr, Q_T_arr, R_arr, T, n_grid=n_grid)
    G = t_grid.size

    m_curr = _initial_mean_iterate(m_initial, G, d, mu_arr)
    m_iterates: List[np.ndarray] = [m_curr.copy()]
    converged = False
    m_input_last = m_curr.copy()

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

        m_input_last = m_input.copy() if m_input is not m_curr else m_curr.copy()

        m_next = _lq_vector_best_response(
            m_input=m_input,
            P=P,
            t_grid=t_grid,
            mu_0_mean=mu_arr,
            mu_0_var=V0_arr,
            Sigma=Sigma_arr,
            R_inv=R_inv,
            n_particles=int(n_particles),
            seed=int(seed),
        )
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

    # Reproduce the trajectory consistent with the converged mean by
    # re-running the simulation under the last BR input.
    x0 = _draw_x0(mu_arr, V0_arr, int(n_particles), int(seed))
    traj = _simulate_under_vector_control(
        x0=x0, t_grid=t_grid, P=P, m_input=m_input_last,
        Sigma=Sigma_arr, R_inv=R_inv, seed=int(seed),
    )

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
    )
