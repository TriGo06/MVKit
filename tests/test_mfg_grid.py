"""Tests for the grid-based non-LQ MFG solver (:func:`mvkit.mfg.solve_mfg`).

Two tiers:

1. **Plumbing**: the dataclasses behave, the dispatcher routes to the
   right inner loop, both methods return solutions of the right shape,
   determinism holds, an unknown method raises a helpful error, and
   input validation triggers on out-of-domain arguments.

2. **LQ comparison (the centerpiece)**: configure an :class:`MFGProblem`
   whose running and terminal costs reproduce the symmetric
   linear-quadratic MFG, run the grid solver on a wide periodic
   domain, and verify the marginal density's mean stays at the initial
   mean while its variance matches the closed-form Riccati ODE. Plus
   first-order convergence under joint refinement of ``n_x`` and
   ``n_t``, and Picard / Fictitious Play converge to the same
   equilibrium on the (monotone) LQ problem.

Plus a non-LQ smoke test: the solver runs to convergence on a
congestion problem with ``F(x, m) = (1/2) q (x - x_target)^2 + lambda *
m`` and produces a non-negative density of unit mass, demonstrating
the LQ machinery is no longer required.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.

Cardaliaguet, P. and Hadikhanloo, S. (2017). *Learning in mean field
games: the fictitious play*. ESAIM: Control, Optimisation and Calculus
of Variations 23, 569-591.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import MFGGridSolution, MFGProblem, solve_mfg
from mvkit.mfg._riccati import solve_riccati
from mvkit.mfg.linear_quadratic import lq_mfg_analytical_variance


# ---------- Shared LQ-MFG problem builders ----------


def _x_mean_factory(x_grid: np.ndarray, dx: float):
    def x_mean(m: np.ndarray) -> float:
        return float(np.sum(x_grid * m) * dx)

    return x_mean


def _build_lq_problem(
    q: float, q_T: float, sigma: float, T: float,
    mu_0_mean: float, mu_0_var: float,
    a: float = -6.0, b: float = 6.0, n_x: int = 256,
) -> MFGProblem:
    L = b - a
    dx = L / n_x
    x_full = np.linspace(a, b, n_x, endpoint=False)
    x_mean = _x_mean_factory(x_full, dx)

    def initial_density(x: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (x - mu_0_mean) ** 2 / mu_0_var) / np.sqrt(
            2.0 * np.pi * mu_0_var
        )

    def running_cost(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return 0.5 * q * (x - x_mean(m)) ** 2

    def terminal_cost(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return 0.5 * q_T * (x - x_mean(m)) ** 2

    return MFGProblem(
        sigma=sigma, T=T, domain=(a, b), n_x=n_x,
        initial_density=initial_density,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
    )


# ---------- Plumbing ----------


def test_solve_mfg_returns_solution_dataclass():
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=64,
    )
    sol = solve_mfg(problem, n_t=20, method="picard", tol=1e-4)
    assert isinstance(sol, MFGGridSolution)
    assert sol.t_grid.shape == (21,)
    assert sol.x_grid.shape == (64,)
    assert sol.u.shape == (21, 64)
    assert sol.m.shape == (21, 64)
    assert sol.optimal_drift.shape == (21, 64)
    assert sol.t_grid[0] == 0.0
    assert sol.t_grid[-1] == pytest.approx(1.0)
    assert len(sol.m_iterates) == sol.n_iterations + 1


def test_solver_is_deterministic():
    """Same problem and same parameters must produce bit-identical
    output. The grid-based solver has no MC, so determinism is exact."""
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=0.5, n_x=64,
    )
    s1 = solve_mfg(problem, n_t=40, method="picard", tol=1e-5)
    s2 = solve_mfg(problem, n_t=40, method="picard", tol=1e-5)
    np.testing.assert_array_equal(s1.m, s2.m)
    np.testing.assert_array_equal(s1.u, s2.u)
    assert s1.n_iterations == s2.n_iterations


def test_unknown_method_raises():
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=32,
    )
    with pytest.raises(ValueError) as exc_info:
        solve_mfg(problem, n_t=10, method="newton")
    msg = str(exc_info.value)
    assert "newton" in msg
    assert "picard" in msg
    assert "fictitious_play" in msg


def test_invalid_problem_raises():
    base = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=32,
    )
    # sigma must be positive
    bad = MFGProblem(
        sigma=0.0, T=base.T, domain=base.domain, n_x=base.n_x,
        initial_density=base.initial_density,
        running_cost=base.running_cost,
        terminal_cost=base.terminal_cost,
    )
    with pytest.raises(ValueError, match="sigma"):
        solve_mfg(bad, n_t=10)
    # n_x must be >= 4
    bad2 = MFGProblem(
        sigma=base.sigma, T=base.T, domain=base.domain, n_x=2,
        initial_density=base.initial_density,
        running_cost=base.running_cost,
        terminal_cost=base.terminal_cost,
    )
    with pytest.raises(ValueError, match="n_x"):
        solve_mfg(bad2, n_t=10)


def test_invalid_n_t_raises():
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=32,
    )
    with pytest.raises(ValueError, match="n_t"):
        solve_mfg(problem, n_t=0)


def test_initial_density_must_be_nonnegative():
    a, b = -3.0, 3.0
    n_x = 32
    bad_problem = MFGProblem(
        sigma=0.5, T=1.0, domain=(a, b), n_x=n_x,
        initial_density=lambda x: np.where(x > 0.0, 1.0, -1.0),
        running_cost=lambda t, x, m: np.zeros_like(x),
        terminal_cost=lambda x, m: np.zeros_like(x),
    )
    with pytest.raises(ValueError, match="non-negative"):
        solve_mfg(bad_problem, n_t=10)


# ---------- LQ comparison: the centerpiece ----------


def test_solve_mfg_recovers_lq_mean_and_variance():
    """The LQ-MFG with symmetric quadratic costs has a closed-form
    equilibrium: ``m_t`` is constant equal to ``mu_0_mean`` and the
    variance solves the linear ODE driven by the Riccati P. The grid
    solver should reproduce both within first-order discretization
    error on the chosen grid.

    Empirical numbers at ``n_x=256, n_t=100``: mean error ~ 1e-9 (just
    round-off), variance relative error ~ 4 percent, density sup error
    ~ 6e-3. The bounds below leave comfortable headroom.
    """
    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_mean, mu_0_var = 0.0, 1.0

    problem = _build_lq_problem(
        q=q, q_T=q_T, sigma=sigma, T=T,
        mu_0_mean=mu_0_mean, mu_0_var=mu_0_var, n_x=256,
    )
    sol = solve_mfg(problem, n_t=100, method="picard", tol=1e-6)
    assert sol.converged

    a, b = problem.domain
    dx = (b - a) / problem.n_x

    grid_means = np.array([
        np.sum(sol.x_grid * sol.m[k]) * dx for k in range(sol.t_grid.size)
    ])
    grid_vars = np.array([
        np.sum((sol.x_grid - grid_means[k]) ** 2 * sol.m[k]) * dx
        for k in range(sol.t_grid.size)
    ])

    # Analytical variance trajectory.
    t_riccati, P_anal = solve_riccati(q, q_T, T, n_grid=400)
    V_anal = lq_mfg_analytical_variance(P_anal, t_riccati, sigma, mu_0_var)
    V_on_grid = np.interp(sol.t_grid, t_riccati, V_anal)

    # Equilibrium mean stays at mu_0_mean to round-off (no MC noise).
    assert np.max(np.abs(grid_means - mu_0_mean)) < 1e-6

    # Variance trajectory matches the Riccati ODE within 6 percent
    # relative at this resolution.
    rel_err = np.max(np.abs(grid_vars - V_on_grid) / V_on_grid)
    assert rel_err < 0.06, f"V relative error {rel_err:.3e} above 0.06"

    # Marginal density vs analytical Gaussian at the terminal time.
    V_T = V_on_grid[-1]
    gaussian_T = np.exp(
        -0.5 * (sol.x_grid - mu_0_mean) ** 2 / V_T
    ) / np.sqrt(2.0 * np.pi * V_T)
    dens_err = float(np.max(np.abs(sol.m[-1] - gaussian_T)))
    assert dens_err < 1e-2, (
        f"density sup error vs Gaussian {dens_err:.3e} above 1e-2"
    )

    # Mass conservation throughout.
    mass = sol.m.sum(axis=1) * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_solve_mfg_first_order_convergence_on_lq():
    """Halving ``dx`` (and ``dt``) should halve the variance error
    against the closed-form LQ Riccati. The fitted log-log slope should
    sit close to 1, the rate predicted for the underlying first-order
    HJB and FP discretizations."""
    q, q_T, sigma, T = 1.0, 0.5, 0.5, 1.0
    mu_0_mean, mu_0_var = 0.0, 1.0

    t_riccati, P_anal = solve_riccati(q, q_T, T, n_grid=400)
    V_anal = lq_mfg_analytical_variance(P_anal, t_riccati, sigma, mu_0_var)

    errs = []
    dxs = []
    for n_x, n_t in [(128, 50), (256, 100), (512, 200), (1024, 400)]:
        problem = _build_lq_problem(
            q=q, q_T=q_T, sigma=sigma, T=T,
            mu_0_mean=mu_0_mean, mu_0_var=mu_0_var, n_x=n_x,
        )
        sol = solve_mfg(problem, n_t=n_t, method="picard", tol=1e-6)
        a, b = problem.domain
        dx = (b - a) / problem.n_x
        means = np.array([
            np.sum(sol.x_grid * sol.m[k]) * dx for k in range(n_t + 1)
        ])
        vars_ = np.array([
            np.sum((sol.x_grid - means[k]) ** 2 * sol.m[k]) * dx
            for k in range(n_t + 1)
        ])
        V_on_grid = np.interp(sol.t_grid, t_riccati, V_anal)
        errs.append(float(np.max(np.abs(vars_ - V_on_grid))))
        dxs.append(dx)

    slope = float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])
    assert slope > 0.7, f"LQ convergence slope {slope:.3f} below 0.7"


def test_picard_and_fp_agree_on_lq():
    """On the (monotone) LQ-MFG, Picard and Fictitious Play converge to
    the same fixed point of the discrete system. The two solutions
    should agree to within solver tolerance, much tighter than the
    discretization error against the closed form.

    Uses the symmetric ``mu_0_mean = 0`` setup so that the LQ analytical
    value function ``u = (1/2) P(t) x^2 + R(t)`` is symmetric around
    the origin and the wrap-around at the periodic boundary creates only
    a smooth, small mismatch (the closed form is not periodic but is
    symmetric, so the jump is dominated by diffusion smoothing). With
    ``mu_0_mean != 0`` the analytical ``u`` has a steep linear part that
    creates a large gradient at the wrap, which is a periodic-BC
    limitation of v0.1; revisit when Neumann BC ships.
    """
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=128,
    )
    sol_picard = solve_mfg(problem, n_t=100, method="picard", tol=1e-6)
    sol_fp = solve_mfg(
        problem, n_t=100, method="fictitious_play",
        tol=1e-6, n_iterations_max=80,
    )
    assert sol_picard.converged and sol_fp.converged
    assert np.max(np.abs(sol_picard.m - sol_fp.m)) < 1e-5


def test_picard_converges_in_few_iterations_on_lq():
    """Symmetric LQ-MFG is a contracting Picard map and the equilibrium
    flow ``m_t = m_0`` is the constant-in-time extension of the
    initial. The default initial iterate is exactly that, so Picard
    should converge in at most a handful of steps."""
    problem = _build_lq_problem(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0, n_x=128,
    )
    sol = solve_mfg(problem, n_t=100, method="picard", tol=1e-6)
    assert sol.converged
    assert sol.n_iterations <= 5


# ---------- Non-LQ smoke ----------


def test_non_lq_congestion_problem_runs_to_convergence():
    """The solver works on a problem where the LQ machinery does not
    apply: a congestion MFG with running cost
    ``F(x, m) = (1/2) q (x - x_target)^2 + lambda * m``. The linear-in-
    m congestion term is the standard Lasry-Lions monotone case. We
    only check that the solver converges, the density stays
    non-negative, and total mass is preserved.
    """
    a, b = -np.pi, np.pi
    n_x = 128
    L = b - a
    x_target = 0.0
    q_quad = 1.0
    lam = 0.3
    sigma = 0.4
    T = 1.0

    def initial_density(x):
        return (1.0 + 0.4 * np.cos(x)) / L

    def running_cost(t, x, m):
        return 0.5 * q_quad * (x - x_target) ** 2 + lam * m

    def terminal_cost(x, m):
        return np.zeros_like(x)

    problem = MFGProblem(
        sigma=sigma, T=T, domain=(a, b), n_x=n_x,
        initial_density=initial_density,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
    )
    sol = solve_mfg(
        problem, n_t=80, method="picard", tol=1e-5, n_iterations_max=50,
    )
    assert sol.converged, (
        f"non-LQ Picard did not converge in 50 iterations; "
        f"final diff = "
        f"{float(np.max(np.abs(sol.m_iterates[-1] - sol.m_iterates[-2]))):.3e}"
    )
    dx = L / n_x
    mass = sol.m.sum(axis=1) * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10
    assert sol.m.min() > -1e-10
