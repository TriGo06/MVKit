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
  is checked through the fixed-point residual),
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

from mvkit.mfg import (
    LQMFGSolution,
    lq_mfg_analytical_variance,
    solve_lq_mfg,
    solve_lq_mfg_fictitious_play,
)
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
    """On this parameter set, Picard reaches a small fixed-point residual."""
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
    assert sol.n_iterations <= 10
    assert sol.fixed_point_residual < 1e-4


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


# ---------- Fictitious Play ----------


def _standard_fp_kwargs(n_particles: int, n_grid: int = 200, seed: int = 0):
    return dict(
        q=2.0,
        q_T=1.0,
        sigma=0.3,
        T=2.0,
        mu_0_mean=0.0,
        mu_0_var=0.5,
        n_particles=n_particles,
        n_grid=n_grid,
        tol=1e-4,
        seed=seed,
    )


def test_fp_and_picard_converge_to_same_equilibrium():
    """Both methods solve the same fixed-point problem; on LQ they must
    agree on the equilibrium mean within MC tolerance.

    The variance trajectory is computed analytically from P, so it is
    bit-exact identical between the two methods modulo float arithmetic
    in the variance ODE solver, which is why we compare with a small
    relative tolerance rather than ``assert_array_equal``.
    """
    kwargs = _standard_fp_kwargs(n_particles=10000, seed=1)
    sol_p = solve_lq_mfg(**kwargs, n_iterations_max=100)
    sol_fp = solve_lq_mfg(
        **kwargs, n_iterations_max=200, method="fictitious_play", damping_burn_in=10
    )
    assert sol_p.converged and sol_fp.converged
    assert np.max(np.abs(sol_p.m - sol_fp.m)) < 1e-2
    rel_v = np.max(np.abs(sol_p.V - sol_fp.V) / sol_p.V)
    assert rel_v < 0.05


def test_fp_reaches_a_small_fixed_point_residual():
    """Pure averaging converges on a weakly coupled deterministic problem."""
    sol = solve_lq_mfg_fictitious_play(
        q=0.1, q_T=0.1, sigma=0.0, T=0.5,
        mu_0_mean=0.0, mu_0_var=0.0, n_particles=2, n_grid=40,
        m_initial=np.ones(41), n_iterations_max=1200, tol=1e-4,
    )
    assert sol.converged
    assert sol.fixed_point_residual < 1e-4


def test_fp_robust_to_bad_initialization():
    """Standard Fictitious Play averages the initial bad guess into the
    historical mean, which slows convergence dramatically when ``m^(0)``
    is far from the equilibrium. Setting ``damping_burn_in=20`` runs twenty
    Picard steps first to drag ``m^(k)`` close to equilibrium, then the
    historical average accumulates from a sensible starting point.
    """
    n_grid = 200
    sol = solve_lq_mfg_fictitious_play(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=0.0,
        mu_0_var=1.0,
        n_particles=50000,
        n_grid=n_grid,
        tol=1e-4,
        seed=2,
        m_initial=np.full(n_grid + 1, 5.0),
        damping_burn_in=20,
        n_iterations_max=50,
    )
    assert sol.converged
    assert np.max(np.abs(sol.m)) < 1e-2


def test_fp_historical_average_is_non_trivial():
    """At iteration 5 the historical average bar_m^(5) and the latest
    iterate m^(5) must differ non-trivially. If they were equal, the
    Picard fallback path would be active and Fictitious Play would not
    actually be running.
    """
    sol = solve_lq_mfg_fictitious_play(
        **_standard_fp_kwargs(n_particles=10000, seed=0),
        n_iterations_max=50,
    )
    iterates = sol.m_iterates
    assert len(iterates) >= 6, "need at least 6 iterates to inspect k=5"
    bar_m_5 = np.mean(np.stack(iterates[:6]), axis=0)
    m_5 = iterates[5]
    assert np.max(np.abs(bar_m_5 - m_5)) > 1e-4


def test_default_method_is_picard():
    """Backward compatibility: omitting ``method=`` must reproduce the
    Picard solver byte-for-byte.
    """
    kwargs = dict(
        q=1.0,
        q_T=0.5,
        sigma=0.5,
        T=1.0,
        mu_0_mean=1.0,
        mu_0_var=0.5,
        n_particles=2000,
        n_grid=100,
        n_iterations_max=10,
        tol=1e-3,
        seed=7,
    )
    sol_default = solve_lq_mfg(**kwargs)
    sol_picard = solve_lq_mfg(**kwargs, method="picard")
    np.testing.assert_array_equal(sol_default.m, sol_picard.m)
    np.testing.assert_array_equal(
        sol_default.x_trajectory, sol_picard.x_trajectory
    )
    assert sol_default.n_iterations == sol_picard.n_iterations
    assert sol_default.converged == sol_picard.converged


def test_unknown_method_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        solve_lq_mfg(
            q=1.0, q_T=0.0, sigma=0.5, T=1.0,
            mu_0_mean=0.0, mu_0_var=1.0,
            method="newton",
        )
    msg = str(exc_info.value)
    assert "newton" in msg
    assert "picard" in msg
    assert "fictitious_play" in msg
