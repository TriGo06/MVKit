"""Tests for the vector linear-quadratic Mean Field Game solver.

Two tiers:

1. **Reduction to scalar at d=1**: the matrix Riccati and matrix
   Lyapunov ODEs evaluated on 1x1 matrices must agree exactly with the
   scalar versions (``_riccati.solve_riccati`` and
   ``linear_quadratic.lq_mfg_analytical_variance``). Same property at
   the solver level: ``solve_lq_mfg_vector`` at ``d=1`` produces an
   analytical variance trajectory identical to ``solve_lq_mfg``'s
   (modulo the simulation, which has its own RNG and so differs
   stochastically).

2. **d >= 2 quantitative**: with isotropic and anisotropic
   parameter sets, the converged particle covariance must match the
   closed-form Lyapunov solution within Monte Carlo tolerance, the
   equilibrium mean must stay at the initial mean (symmetric
   LQ-MFG), and Picard must converge in a handful of iterations.

References
----------
Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field
Games with Applications I*, Chapter 3 (vector LQ-MFG).
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import (
    LQMFGVectorSolution,
    solve_lq_mfg,
    solve_lq_mfg_vector,
)
from mvkit.mfg._riccati import solve_riccati
from mvkit.mfg._riccati_matrix import (
    lq_mfg_analytical_covariance,
    solve_matrix_riccati,
)
from mvkit.mfg.linear_quadratic import lq_mfg_analytical_variance


# ---------- d = 1 reduction ----------


def test_matrix_riccati_matches_scalar_at_d_equal_1():
    """For ``d = 1`` the matrix Riccati ODE on 1x1 matrices reduces to
    the scalar Riccati ODE; the two solvers must agree to round-off."""
    q, q_T, T = 1.0, 0.5, 1.0
    t_s, P_s = solve_riccati(q, q_T, T, n_grid=200)
    t_v, P_v = solve_matrix_riccati(
        np.array([[q]]), np.array([[q_T]]), np.array([[1.0]]), T, n_grid=200,
    )
    np.testing.assert_array_equal(t_s, t_v)
    np.testing.assert_array_equal(P_s, P_v[:, 0, 0])


def test_matrix_lyapunov_matches_scalar_variance_at_d_equal_1():
    """The matrix Lyapunov ODE on 1x1 matrices reduces to the scalar
    variance ODE ``dot V = -2 P V + sigma^2`` from the LQ-MFG closed
    form."""
    q, q_T, T = 1.0, 0.5, 1.0
    sigma_scalar = 0.5
    V0_scalar = 1.0

    t_s, P_s = solve_riccati(q, q_T, T, n_grid=200)
    V_scalar = lq_mfg_analytical_variance(P_s, t_s, sigma_scalar, V0_scalar)

    t_v, P_v = solve_matrix_riccati(
        np.array([[q]]), np.array([[q_T]]), np.array([[1.0]]), T, n_grid=200,
    )
    V_matrix = lq_mfg_analytical_covariance(
        P_v, t_v, np.array([[sigma_scalar]]),
        np.array([[1.0]]), np.array([[V0_scalar]]),
    )
    np.testing.assert_array_equal(V_scalar, V_matrix[:, 0, 0])


def test_solve_lq_mfg_vector_matches_scalar_solver_at_d_equal_1():
    """At d=1 the vector solver must reproduce the scalar
    ``solve_lq_mfg``'s analytical V trajectory exactly (different RNG
    sequences mean the simulated trajectories differ pathwise, but the
    Lyapunov closed form is identical) and the equilibrium statistics
    agree within MC tolerance."""
    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_mean, mu_0_var = 0.5, 1.0
    N = 5000

    sol_scalar = solve_lq_mfg(
        q=q, q_T=q_T, sigma=sigma, T=T,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var,
        n_particles=N, n_grid=200, tol=1e-5, seed=0,
    )
    sol_vector = solve_lq_mfg_vector(
        Q=np.array([[q]]), Q_T=np.array([[q_T]]),
        Sigma=np.array([[sigma]]), T=T,
        mu_0_mean=np.array([mu_0_mean]),
        mu_0_var=np.array([[mu_0_var]]),
        n_particles=N, n_grid=200, tol=1e-5, seed=0,
    )
    # Analytical V trajectories must match exactly (same closed form).
    np.testing.assert_array_equal(sol_scalar.V, sol_vector.V[:, 0, 0])

    # Equilibrium means agree within MC tolerance (~ 1/sqrt(N) ~ 0.014).
    diff = float(np.max(np.abs(sol_scalar.m - sol_vector.m[:, 0])))
    assert diff < 5.0 / np.sqrt(N), (
        f"d=1 mean trajectories differ by {diff:.3e}, above 5/sqrt(N)"
    )


# ---------- d >= 2 quantitative ----------


def test_d_equal_2_isotropic_mean_conservation_and_covariance():
    """In a symmetric, isotropic d=2 LQ-MFG, the equilibrium mean stays
    at ``mu_0_mean`` to MC tolerance and the empirical covariance at
    the terminal time matches the analytical Lyapunov solution.

    Empirical covariance at N=10000 in our environment: sup error
    against the analytical V(T) ~ 1.3e-2; the bound is generous."""
    d = 2
    Q = np.eye(d)
    Q_T = 0.5 * np.eye(d)
    Sigma = 0.5 * np.eye(d)
    mu_0_mean = np.zeros(d)
    mu_0_var = np.eye(d)
    N = 10000
    sol = solve_lq_mfg_vector(
        Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var,
        n_particles=N, n_grid=200, tol=1e-5, seed=0,
    )
    assert isinstance(sol, LQMFGVectorSolution)
    assert sol.converged
    # Mean stays at zero within MC tolerance.
    assert np.max(np.abs(sol.m)) < 5.0 / np.sqrt(N)

    # Empirical covariance vs analytical at T.
    emp_cov_T = np.cov(sol.x_trajectory[-1].T)
    sup_err = float(np.max(np.abs(emp_cov_T - sol.V[-1])))
    assert sup_err < 0.05, (
        f"empirical V(T) sup error {sup_err:.3e} above 0.05"
    )


def test_d_equal_2_anisotropic_recovers_correlations():
    """With non-diagonal Sigma the equilibrium covariance has non-zero
    off-diagonals; the solver should recover them within MC tolerance.

    Empirical cov sup error at N=10000 in our environment: ~ 1.4e-3."""
    d = 2
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    Q_T = np.array([[1.0, 0.2], [0.2, 0.5]])
    Sigma = np.array([[0.3, 0.0], [0.1, 0.4]])
    mu_0_mean = np.zeros(d)
    mu_0_var = 0.5 * np.eye(d)
    N = 10000
    sol = solve_lq_mfg_vector(
        Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var,
        n_particles=N, n_grid=200, tol=1e-5, seed=1,
    )
    assert sol.converged
    # The off-diagonal of the analytical V(T) is genuinely non-zero
    # (Sigma has a (1, 0) entry of 0.1, so the noise correlates the
    # two coordinates over time).
    assert abs(sol.V[-1, 0, 1]) > 1e-2, (
        "test setup error: pick a Sigma that produces correlated state"
    )
    emp_cov_T = np.cov(sol.x_trajectory[-1].T)
    sup_err = float(np.max(np.abs(emp_cov_T - sol.V[-1])))
    assert sup_err < 0.02


def test_d_equal_2_picard_converges_in_few_iterations():
    """The vector LQ-MFG Picard map is a contraction in the same
    sense as the scalar version; the symmetric problem converges to
    its constant-mean fixed point in a handful of steps."""
    d = 2
    Q = np.eye(d)
    Q_T = 0.5 * np.eye(d)
    Sigma = 0.5 * np.eye(d)
    sol = solve_lq_mfg_vector(
        Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
        mu_0_mean=np.zeros(d), mu_0_var=np.eye(d),
        n_particles=10000, n_grid=200, tol=1e-5, seed=0,
    )
    assert sol.converged
    assert sol.n_iterations <= 10


def test_d_equal_2_fictitious_play_agrees_with_picard():
    """On the (monotone) LQ-MFG, Fictitious Play and Picard converge
    to the same fixed point of the discrete system."""
    d = 2
    Q = np.eye(d)
    Q_T = 0.5 * np.eye(d)
    Sigma = 0.4 * np.eye(d)
    kwargs = dict(
        Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
        mu_0_mean=np.zeros(d), mu_0_var=np.eye(d),
        n_particles=5000, n_grid=200, tol=1e-5, seed=0,
    )
    sol_p = solve_lq_mfg_vector(**kwargs)
    sol_fp = solve_lq_mfg_vector(
        **kwargs, method="fictitious_play", n_iterations_max=80,
    )
    assert sol_p.converged and sol_fp.converged
    # Picard contracts geometrically and stops in ~7 steps; FP has
    # the slower O(1/k) rate and stops at ~40. The two converged means
    # therefore disagree by ~ 5e-4 in our environment, which is well
    # below the MC noise floor (5/sqrt(N) ~ 0.07) but above what a
    # strict byte-equal check would allow.
    assert np.max(np.abs(sol_p.m - sol_fp.m)) < 1e-3


# ---------- input validation ----------


def test_invalid_inputs():
    Q = np.eye(2)
    Q_T = np.eye(2)
    Sigma = np.eye(2)
    mu = np.zeros(2)
    V0 = np.eye(2)

    # Wrong shape.
    with pytest.raises(ValueError, match="Q_T"):
        solve_lq_mfg_vector(
            Q=Q, Q_T=np.eye(3), Sigma=Sigma, T=1.0,
            mu_0_mean=mu, mu_0_var=V0,
        )
    with pytest.raises(ValueError, match="mu_0_mean"):
        solve_lq_mfg_vector(
            Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
            mu_0_mean=np.zeros(3), mu_0_var=V0,
        )
    # Bad method.
    with pytest.raises(ValueError, match="newton"):
        solve_lq_mfg_vector(
            Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
            mu_0_mean=mu, mu_0_var=V0,
            method="newton",
        )
    # Bad T.
    with pytest.raises(ValueError, match="T"):
        solve_lq_mfg_vector(
            Q=Q, Q_T=Q_T, Sigma=Sigma, T=-1.0,
            mu_0_mean=mu, mu_0_var=V0,
        )


def test_solver_is_deterministic():
    Q = np.eye(2)
    Q_T = 0.5 * np.eye(2)
    Sigma = 0.5 * np.eye(2)
    kwargs = dict(
        Q=Q, Q_T=Q_T, Sigma=Sigma, T=1.0,
        mu_0_mean=np.zeros(2), mu_0_var=np.eye(2),
        n_particles=2000, n_grid=100, tol=1e-3, seed=7,
    )
    s1 = solve_lq_mfg_vector(**kwargs)
    s2 = solve_lq_mfg_vector(**kwargs)
    np.testing.assert_array_equal(s1.m, s2.m)
    np.testing.assert_array_equal(s1.x_trajectory, s2.x_trajectory)
    assert s1.n_iterations == s2.n_iterations
