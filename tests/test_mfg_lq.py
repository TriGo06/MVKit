"""Tests for the scalar linear-quadratic Mean Field Game solver.

The LQ-MFG admits closed forms at every step (Riccati, equilibrium mean
trajectory, variance trajectory), making it the right model to validate
the API on. We exercise:

- the Riccati ODE numerics against the closed-form ``tanh`` solution and
  the boundary condition,
- mean conservation: in the symmetric LQ case the equilibrium mean is
  constant equal to ``mu_0_mean``,
- the analytical variance vs the empirical particle variance at terminal
  time,
- Picard convergence speed at the standard initial guess (closed form
  predicts 1 to 2 iterations modulo MC noise),
- contraction property of the Picard map under a deliberately bad initial
  guess,
- shape and basic sanity (positivity, V(0)) on a non-trivial parameter
  set.

Reference: Carmona and Delarue (2018), Volume I, Section 3.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mvkit.mfg import LQMFGSolution, lq_mfg_analytical_variance, solve_lq_mfg
from mvkit.mfg._riccati import solve_riccati


def test_riccati_matches_closed_form_tanh():
    """For ``q=1, q_T=0, T=1`` the closed form is ``P(t) = tanh(T - t)``."""
    q, q_T, T = 1.0, 0.0, 1.0
    t_grid, P = solve_riccati(q, q_T, T, n_grid=500)
    P_exact = np.tanh(T - t_grid)
    assert np.max(np.abs(P - P_exact)) < 1e-6


def test_riccati_general_closed_form():
    """For ``q > 0`` and ``q_T < sqrt(q)`` the Riccati has the tanh form
    ``P(t) = omega tanh(omega (T-t) + atanh(q_T / omega))``.

    Verifies the numerical solver on the parameter set used by most other
    tests below.
    """
    q, q_T, T = 1.0, 0.5, 1.0
    t_grid, P = solve_riccati(q, q_T, T, n_grid=500)
    omega = math.sqrt(q)
    P_exact = omega * np.tanh(omega * (T - t_grid) + math.atanh(q_T / omega))
    assert np.max(np.abs(P - P_exact)) < 1e-6


def test_riccati_boundary_value():
    q, q_T, T = 2.0, 1.0, 2.0
    _, P = solve_riccati(q, q_T, T, n_grid=500)
    assert abs(P[-1] - q_T) < 1e-8


def test_mean_conservation():
    """In the symmetric LQ case the equilibrium mean is ``m(t) = m_0``.

    The empirical mean of the converged particle ensemble should match
    ``mu_0_mean`` within Monte Carlo tolerance ``O(1 / sqrt(N))``.
    """
    n_part = 5000
    sol = solve_lq_mfg(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=1.5,
        mu_0_var=1.0,
        n_particles=n_part,
        n_grid=200,
        tol=1e-4,
        seed=0,
    )
    assert sol.converged
    mc_bound = 5.0 / np.sqrt(n_part)
    assert np.max(np.abs(sol.m - 1.5)) < mc_bound


def test_variance_matches_analytical():
    """The empirical variance of the terminal ensemble must match the
    analytical variance from the linear ODE within MC tolerance.

    The chosen parameters give ``V(T) ~ 0.36``, well above the variance
    estimator's MC noise floor at N=10000 (``~ V * sqrt(2 / N) ~ 5e-3``).
    """
    n_part = 10000
    sol = solve_lq_mfg(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=0.0,
        mu_0_var=1.0,
        n_particles=n_part,
        n_grid=200,
        tol=1e-4,
        seed=1,
    )
    assert sol.converged
    V_T_analytical = sol.V[-1]
    assert V_T_analytical > 0.05, "pre-flight: V(T) too small for a robust MC test"
    V_T_empirical = float(sol.x_trajectory[-1].var(ddof=0))
    rel_err = abs(V_T_empirical - V_T_analytical) / V_T_analytical
    assert rel_err < 0.05


def test_picard_converges_fast():
    """Closed-form analysis predicts 1-2 noiseless iterations. With MC
    noise the Picard map converges geometrically, with a contraction
    factor near ``1 - exp(-int P)``. At N=20000 the noise floor is small
    enough that ``tol = 1e-4`` is reached in at most 5 updates across the
    seeds we tested.
    """
    sol = solve_lq_mfg(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=1.5,
        mu_0_var=1.0,
        n_particles=20000,
        n_grid=200,
        tol=1e-4,
        seed=0,
    )
    assert sol.converged
    assert sol.n_iterations <= 5


def test_picard_robust_to_bad_initialization():
    """Force a far-from-truth initial guess ``m^(0) = 5`` while the true
    equilibrium is ``m_t = 0``. The Picard map is a contraction with
    factor below 1, so the iteration should still converge to within MC
    tolerance of zero.
    """
    n_part = 50000
    n_grid = 200
    sol = solve_lq_mfg(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=0.0,
        mu_0_var=1.0,
        n_particles=n_part,
        n_grid=n_grid,
        tol=1e-4,
        seed=2,
        m_initial=np.full(n_grid + 1, 5.0),
        n_iterations_max=30,
    )
    assert sol.converged
    assert np.max(np.abs(sol.m)) < 1e-2


def test_non_trivial_parameters_basic_sanity():
    """Run with parameters distinct from the rest of the suite. Check
    output shapes, positivity of V(t), V(0) equal to the prescribed
    initial variance both analytically and empirically.
    """
    q, q_T, sigma, T = 2.0, 1.0, 0.3, 2.0
    mu_0_var = 0.5
    n_grid = 400
    sol = solve_lq_mfg(
        q=q,
        q_T=q_T,
        sigma=sigma,
        T=T,
        mu_0_mean=0.0,
        mu_0_var=mu_0_var,
        n_particles=5000,
        n_grid=n_grid,
        tol=1e-4,
        seed=3,
    )
    assert isinstance(sol, LQMFGSolution)
    assert sol.t_grid.shape == (n_grid + 1,)
    assert sol.P.shape == (n_grid + 1,)
    assert sol.m.shape == (n_grid + 1,)
    assert sol.V.shape == (n_grid + 1,)
    assert sol.x_trajectory.shape == (n_grid + 1, 5000)
    assert sol.t_grid[0] == 0.0
    assert sol.t_grid[-1] == pytest.approx(T)
    assert np.all(sol.V > 0.0)
    assert sol.V[0] == pytest.approx(mu_0_var, rel=1e-8)
    emp_V0 = float(sol.x_trajectory[0].var(ddof=0))
    assert abs(emp_V0 - mu_0_var) < 5.0 * mu_0_var * np.sqrt(2.0 / 5000)


def test_lq_mfg_analytical_variance_standalone():
    """``lq_mfg_analytical_variance`` solves the linear ODE
    ``dot V = -2 P V + sigma^2``. Check it agrees with the closed-form
    integral via cumulative trapezoidal rule.
    """
    from scipy.integrate import cumulative_trapezoid

    t_grid, P = solve_riccati(q=1.0, q_T=0.5, T=1.0, n_grid=500)
    sigma, V0 = 0.5, 1.0

    V = lq_mfg_analytical_variance(P, t_grid, sigma=sigma, mu_0_var=V0)

    G_int = cumulative_trapezoid(P, t_grid, initial=0.0)
    expG = np.exp(2.0 * G_int)
    inner = cumulative_trapezoid(expG, t_grid, initial=0.0)
    V_closed = V0 * np.exp(-2.0 * G_int) + sigma * sigma * np.exp(-2.0 * G_int) * inner

    assert np.max(np.abs(V - V_closed)) < 1e-5
    assert V[0] == pytest.approx(V0, rel=1e-10)


def test_solver_is_deterministic():
    """Same seed plus same inputs gives identical outputs."""
    kwargs = dict(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=0.0,
        mu_0_var=1.0,
        n_particles=2000,
        n_grid=100,
        tol=1e-3,
        seed=7,
    )
    sol1 = solve_lq_mfg(**kwargs)
    sol2 = solve_lq_mfg(**kwargs)
    np.testing.assert_array_equal(sol1.m, sol2.m)
    np.testing.assert_array_equal(sol1.x_trajectory, sol2.x_trajectory)
    assert sol1.n_iterations == sol2.n_iterations


def test_invalid_inputs():
    with pytest.raises(ValueError):
        solve_lq_mfg(q=-1.0, q_T=0.0, sigma=0.5, T=1.0, mu_0_mean=0.0, mu_0_var=1.0)
    with pytest.raises(ValueError):
        solve_lq_mfg(q=1.0, q_T=-0.1, sigma=0.5, T=1.0, mu_0_mean=0.0, mu_0_var=1.0)
    with pytest.raises(ValueError):
        solve_lq_mfg(q=1.0, q_T=0.0, sigma=-0.5, T=1.0, mu_0_mean=0.0, mu_0_var=1.0)
    with pytest.raises(ValueError):
        solve_lq_mfg(q=1.0, q_T=0.0, sigma=0.5, T=0.0, mu_0_mean=0.0, mu_0_var=1.0)
    with pytest.raises(ValueError):
        solve_lq_mfg(q=1.0, q_T=0.0, sigma=0.5, T=1.0, mu_0_mean=0.0, mu_0_var=-1.0)
    with pytest.raises(ValueError):
        solve_lq_mfg(
            q=1.0, q_T=0.0, sigma=0.5, T=1.0, mu_0_mean=0.0, mu_0_var=1.0,
            m_initial=np.zeros(7),  # wrong shape
        )
