"""Tests for the 2D grid-based non-LQ MFG solver
(:func:`mvkit.mfg.solve_mfg_2d`).

The centerpiece is the **2D LQ comparison**: configure an
:class:`MFGProblem2D` whose running and terminal costs reproduce a
2D isotropic LQ-MFG, and verify that the empirical covariance at
the terminal time matches the closed-form Lyapunov ODE solution
from :func:`mvkit.mfg.solve_lq_mfg_vector` (via
``_riccati_matrix.lq_mfg_analytical_covariance``). This is the
killer test that ties the multi-dim grid pipeline to the closed-form
vector LQ-MFG we already validated.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import (
    MFGGrid2DSolution,
    MFGProblem2D,
    solve_mfg_2d,
)
from mvkit.mfg._riccati_matrix import (
    lq_mfg_analytical_covariance,
    solve_matrix_riccati,
)


# ---------- Plumbing ----------


def _trivial_problem(n_x: int = 16, boundary: str = "periodic") -> MFGProblem2D:
    a, b = -1.0, 1.0
    return MFGProblem2D(
        sigma=0.5, T=1.0,
        domain=((a, b), (a, b)),
        n_x=(n_x, n_x),
        initial_density=lambda X, Y: np.ones_like(X) / 4.0,
        running_cost=lambda t, X, Y, m: np.zeros_like(X),
        terminal_cost=lambda X, Y, m: np.zeros_like(X),
        boundary=boundary,
    )


def test_solve_mfg_2d_returns_solution_dataclass():
    problem = _trivial_problem(n_x=16)
    sol = solve_mfg_2d(problem, n_t=10, method="picard", tol=1e-3)
    assert isinstance(sol, MFGGrid2DSolution)
    assert sol.t_grid.shape == (11,)
    assert sol.x_grid.shape == (16,)
    assert sol.y_grid.shape == (16,)
    assert sol.u.shape == (11, 16, 16)
    assert sol.m.shape == (11, 16, 16)
    assert sol.optimal_drift.shape == (11, 16, 16, 2)


def test_solver_is_deterministic():
    problem = _trivial_problem(n_x=16, boundary="neumann")
    s1 = solve_mfg_2d(problem, n_t=10, method="picard", tol=1e-3)
    s2 = solve_mfg_2d(problem, n_t=10, method="picard", tol=1e-3)
    np.testing.assert_array_equal(s1.m, s2.m)
    np.testing.assert_array_equal(s1.u, s2.u)


def test_unknown_method_raises():
    problem = _trivial_problem(n_x=8)
    with pytest.raises(ValueError, match="newton"):
        solve_mfg_2d(problem, n_t=10, method="newton")


def test_invalid_problem_raises():
    base = _trivial_problem(n_x=8)
    bad = MFGProblem2D(
        sigma=0.0, T=base.T, domain=base.domain, n_x=base.n_x,
        initial_density=base.initial_density,
        running_cost=base.running_cost,
        terminal_cost=base.terminal_cost,
    )
    with pytest.raises(ValueError, match="sigma"):
        solve_mfg_2d(bad, n_t=10)


# ---------- LQ comparison: the centerpiece ----------


def test_solve_mfg_2d_recovers_isotropic_lq_covariance():
    """A symmetric isotropic 2D LQ-MFG (Q = q I, Q_T = q_T I,
    Sigma = sigma I) on a Neumann square recovers the closed-form
    Lyapunov covariance trajectory from the vector LQ-MFG up to
    first-order discretization error.

    Empirical sup error vs the analytical V(T) at n_x = n_y = 64,
    n_t = 50: ~ 5.3e-2 in our environment. Convergence slope ~ 0.9.
    """
    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_mean = np.array([0.0, 0.0])
    mu_0_var = np.eye(2)
    Q_mat = q * np.eye(2)
    Q_T_mat = q_T * np.eye(2)
    Sigma_mat = sigma * np.eye(2)
    R_mat = np.eye(2)

    t_riccati, P_anal = solve_matrix_riccati(Q_mat, Q_T_mat, R_mat, T, n_grid=200)
    V_anal = lq_mfg_analytical_covariance(
        P_anal, t_riccati, Sigma_mat, R_mat, mu_0_var,
    )

    a, b = -5.0, 5.0
    n_xy = 64
    n_t = 50
    dx = (b - a) / n_xy
    x_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
    y_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
    X_g, Y_g = np.meshgrid(x_v, y_v, indexing="ij")

    def initial_density(X, Y):
        return np.exp(-0.5 * (X ** 2 + Y ** 2)) / (2.0 * np.pi)

    def F(t, X, Y, m):
        mx = float(np.sum(X_g * m) * dx * dx)
        my = float(np.sum(Y_g * m) * dx * dx)
        return 0.5 * q * ((X - mx) ** 2 + (Y - my) ** 2)

    def g(X, Y, m):
        mx = float(np.sum(X_g * m) * dx * dx)
        my = float(np.sum(Y_g * m) * dx * dx)
        return 0.5 * q_T * ((X - mx) ** 2 + (Y - my) ** 2)

    problem = MFGProblem2D(
        sigma=sigma, T=T, domain=((a, b), (a, b)), n_x=(n_xy, n_xy),
        initial_density=initial_density,
        running_cost=F, terminal_cost=g,
        boundary="neumann",
    )
    sol = solve_mfg_2d(
        problem, n_t=n_t, method="picard", tol=1e-5,
        n_iterations_max=20,
    )
    assert sol.converged

    # Mean conservation: mu_0_mean = 0 stays at 0 (no MC noise on the grid).
    means_x = np.array([
        float(np.sum(X_g * sol.m[k]) * dx * dx) for k in range(n_t + 1)
    ])
    means_y = np.array([
        float(np.sum(Y_g * sol.m[k]) * dx * dx) for k in range(n_t + 1)
    ])
    assert np.max(np.abs(means_x)) < 1e-6
    assert np.max(np.abs(means_y)) < 1e-6

    # Empirical V(T) vs analytical (diagonal entries; off-diagonal is
    # essentially zero by isotropy).
    V_T_xx = float(np.sum(X_g ** 2 * sol.m[-1]) * dx * dx)
    V_T_yy = float(np.sum(Y_g ** 2 * sol.m[-1]) * dx * dx)
    V_T_anal_xx = float(V_anal[-1, 0, 0])
    V_T_anal_yy = float(V_anal[-1, 1, 1])
    err_xx = abs(V_T_xx - V_T_anal_xx)
    err_yy = abs(V_T_yy - V_T_anal_yy)
    # n_xy = 64 gives ~ 5.3e-2 absolute error; bound 0.08 is generous.
    assert err_xx < 0.08
    assert err_yy < 0.08

    # Mass conservation under Neumann (no flux at boundaries).
    mass = sol.m.sum(axis=(1, 2)) * dx * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_solve_mfg_2d_first_order_convergence_on_lq():
    """Doubling the grid resolution should halve the V error.
    Empirical slope ~ 0.9 in our environment."""
    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_var = np.eye(2)
    Q_mat = q * np.eye(2)
    Q_T_mat = q_T * np.eye(2)
    Sigma_mat = sigma * np.eye(2)
    R_mat = np.eye(2)
    t_riccati, P_anal = solve_matrix_riccati(Q_mat, Q_T_mat, R_mat, T, n_grid=200)
    V_anal = lq_mfg_analytical_covariance(
        P_anal, t_riccati, Sigma_mat, R_mat, mu_0_var,
    )
    V_T_anal = float(V_anal[-1, 0, 0])

    a, b = -5.0, 5.0

    errs, dxs = [], []
    for n_xy, n_t in [(48, 25), (64, 50), (96, 75), (128, 100)]:
        dx = (b - a) / n_xy
        x_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
        y_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
        X_g, Y_g = np.meshgrid(x_v, y_v, indexing="ij")

        def initial_density(X, Y):
            return np.exp(-0.5 * (X ** 2 + Y ** 2)) / (2.0 * np.pi)

        def F(t, X, Y, m, X_g=X_g, Y_g=Y_g, dx=dx):
            mx = float(np.sum(X_g * m) * dx * dx)
            my = float(np.sum(Y_g * m) * dx * dx)
            return 0.5 * q * ((X - mx) ** 2 + (Y - my) ** 2)

        def g(X, Y, m, X_g=X_g, Y_g=Y_g, dx=dx):
            mx = float(np.sum(X_g * m) * dx * dx)
            my = float(np.sum(Y_g * m) * dx * dx)
            return 0.5 * q_T * ((X - mx) ** 2 + (Y - my) ** 2)

        problem = MFGProblem2D(
            sigma=sigma, T=T, domain=((a, b), (a, b)), n_x=(n_xy, n_xy),
            initial_density=initial_density,
            running_cost=F, terminal_cost=g,
            boundary="neumann",
        )
        sol = solve_mfg_2d(
            problem, n_t=n_t, method="picard", tol=1e-5,
            n_iterations_max=20,
        )
        V_T_xx = float(np.sum(X_g ** 2 * sol.m[-1]) * dx * dx)
        errs.append(abs(V_T_xx - V_T_anal))
        dxs.append(dx)

    slope = float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])
    assert slope > 0.7, f"2D LQ convergence slope {slope:.3f} below 0.7"


# ---------- Non-LQ smoke ----------


def test_solve_mfg_2d_recovers_anisotropic_lq_covariance():
    """Anisotropic 2D LQ with ``sigma_x != sigma_y`` recovers the
    closed-form Lyapunov covariance from ``solve_lq_mfg_vector`` with
    ``Sigma = diag(sigma_x, sigma_y)``. The test pins down the
    capability that motivated the anisotropic-sigma extension: the
    equilibrium covariance is no longer a multiple of the identity,
    and the diagonal asymmetry is recovered correctly.
    """
    q, q_T, T = 1.0, 0.5, 1.0
    sigma_x, sigma_y = 0.5, 0.3  # asymmetric per axis
    mu_0_var = np.eye(2)
    Q_mat = q * np.eye(2)
    Q_T_mat = q_T * np.eye(2)
    Sigma_mat = np.diag([sigma_x, sigma_y])
    R_mat = np.eye(2)
    t_riccati, P_anal = solve_matrix_riccati(Q_mat, Q_T_mat, R_mat, T, n_grid=200)
    V_anal = lq_mfg_analytical_covariance(
        P_anal, t_riccati, Sigma_mat, R_mat, mu_0_var,
    )

    a, b = -5.0, 5.0
    n_xy = 64
    n_t = 50
    dx = (b - a) / n_xy
    x_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
    y_v = np.linspace(a + dx / 2.0, b - dx / 2.0, n_xy)
    X_g, Y_g = np.meshgrid(x_v, y_v, indexing="ij")

    def initial_density(X, Y):
        return np.exp(-0.5 * (X ** 2 + Y ** 2)) / (2.0 * np.pi)

    def F(t, X, Y, m):
        mx = float(np.sum(X_g * m) * dx * dx)
        my = float(np.sum(Y_g * m) * dx * dx)
        return 0.5 * q * ((X - mx) ** 2 + (Y - my) ** 2)

    def g(X, Y, m):
        mx = float(np.sum(X_g * m) * dx * dx)
        my = float(np.sum(Y_g * m) * dx * dx)
        return 0.5 * q_T * ((X - mx) ** 2 + (Y - my) ** 2)

    problem = MFGProblem2D(
        sigma=(sigma_x, sigma_y), T=T, domain=((a, b), (a, b)),
        n_x=(n_xy, n_xy),
        initial_density=initial_density,
        running_cost=F, terminal_cost=g,
        boundary="neumann",
    )
    sol = solve_mfg_2d(
        problem, n_t=n_t, method="picard", tol=1e-5,
        n_iterations_max=20,
    )
    assert sol.converged

    V_T_xx = float(np.sum(X_g ** 2 * sol.m[-1]) * dx * dx)
    V_T_yy = float(np.sum(Y_g ** 2 * sol.m[-1]) * dx * dx)
    V_T_anal_xx = float(V_anal[-1, 0, 0])
    V_T_anal_yy = float(V_anal[-1, 1, 1])
    # Anisotropic V(T): the two diagonals are genuinely different.
    assert V_T_anal_xx > V_T_anal_yy + 0.05, (
        "test setup error: pick sigma_x > sigma_y so V_xx > V_yy"
    )
    # First-order error at n_xy = 64 is ~ 5e-2 absolute on each diagonal
    # in our environment; the bound is generous.
    assert abs(V_T_xx - V_T_anal_xx) < 0.08
    assert abs(V_T_yy - V_T_anal_yy) < 0.08
    # Asymmetry direction must be preserved (V_xx > V_yy empirically).
    assert V_T_xx > V_T_yy

    # Mass conservation under Neumann + anisotropic diffusion.
    mass = sol.m.sum(axis=(1, 2)) * dx * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_anisotropic_sigma_passes_through_to_inner_solvers():
    """The MFG dispatcher must accept tuple sigma and forward it
    correctly to both the HJB and FP inner solvers. Sanity: the
    output is finite and Picard converges."""
    a, b = -np.pi, np.pi
    n_xy = 32

    def initial_density(X, Y):
        return (1.0 + 0.4 * np.cos(X) * np.cos(Y)) / ((b - a) ** 2)

    def F(t, X, Y, m):
        return 0.5 * (X ** 2 + Y ** 2)

    def g(X, Y, m):
        return np.zeros_like(X)

    problem = MFGProblem2D(
        sigma=(0.5, 0.3),
        T=1.0,
        domain=((a, b), (a, b)),
        n_x=(n_xy, n_xy),
        initial_density=initial_density,
        running_cost=F,
        terminal_cost=g,
        boundary="periodic",
    )
    sol = solve_mfg_2d(
        problem, n_t=40, method="picard", tol=1e-3, n_iterations_max=20,
    )
    assert sol.converged
    assert np.isfinite(sol.m).all()
    assert np.isfinite(sol.u).all()


def test_non_lq_2d_runs_to_convergence():
    """A non-LQ congestion problem in 2D: F = (1/2)((x - x_t)^2 +
    (y - y_t)^2) + lambda * m. Picard converges, mass is conserved,
    density stays non-negative."""
    a, b = -np.pi, np.pi
    n_xy = 32
    L = b - a
    sigma = 0.4
    T = 1.0
    x_target, y_target = 0.0, 0.0
    lam = 0.3

    def initial_density(X, Y):
        return (1.0 + 0.4 * np.cos(X) * np.cos(Y)) / (L * L)

    def F(t, X, Y, m):
        return 0.5 * ((X - x_target) ** 2 + (Y - y_target) ** 2) + lam * m

    def g(X, Y, m):
        return np.zeros_like(X)

    problem = MFGProblem2D(
        sigma=sigma, T=T, domain=((a, b), (a, b)), n_x=(n_xy, n_xy),
        initial_density=initial_density,
        running_cost=F, terminal_cost=g,
        boundary="periodic",
    )
    sol = solve_mfg_2d(
        problem, n_t=50, method="picard", tol=1e-4,
        n_iterations_max=50,
    )
    assert sol.converged
    dx = L / n_xy
    mass = sol.m.sum(axis=(1, 2)) * dx * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10
    assert sol.m.min() > -1e-10
