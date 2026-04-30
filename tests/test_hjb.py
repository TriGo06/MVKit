"""Tests for the standalone HJB grid solver.

Validates :func:`mvkit.mfg.solve_hjb` on four anchored cases:

1. **Zero data**: ``F = 0``, ``g = 0`` must give ``u = 0``.
2. **Spatially constant data**: ``F = c`` constant, ``g = 0`` gives
   ``u(t, x) = (T - t) * c`` (no Hamiltonian or diffusion contribution
   when ``u`` has no spatial variation).
3. **Hopf-Cole closed form**: with ``F = 0`` and ``u = -sigma^2 log v``,
   the HJB linearizes to a backward heat equation in ``v``. For
   ``v_T(x) = 1 + amplitude * cos(2 pi k x / L)`` the closed-form
   evolution is one Fourier mode, against which the numerical solution
   is compared at sup norm. Convergence rate is checked over a sequence
   of refinements.
4. **Monotonicity**: the HJB operator is monotone in the data, so
   ``g_1 <= g_2`` and ``F_1 <= F_2`` pointwise must give ``u_1 <= u_2``
   pointwise. This is a structural property the discretization must
   preserve.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games: numerical
methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import HJBSolution, solve_hjb


def _periodic_grid(n_x: int, length: float = 2.0 * np.pi) -> np.ndarray:
    return np.linspace(0.0, length, n_x, endpoint=False)


# ---------- Sanity ----------


def test_solve_hjb_returns_solution_dataclass():
    n_x = 64
    x = _periodic_grid(n_x)
    sol = solve_hjb(
        sigma=0.5, T=1.0, x_grid=x, n_t=10,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: np.zeros_like(xx),
    )
    assert isinstance(sol, HJBSolution)
    assert sol.t_grid.shape == (11,)
    assert sol.x_grid.shape == (n_x,)
    assert sol.u.shape == (11, n_x)
    assert sol.optimal_drift.shape == (11, n_x)
    assert sol.t_grid[0] == 0.0
    assert sol.t_grid[-1] == pytest.approx(1.0)


def test_zero_data_gives_zero_solution():
    n_x = 64
    x = _periodic_grid(n_x)
    sol = solve_hjb(
        sigma=0.5, T=1.0, x_grid=x, n_t=50,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: np.zeros_like(xx),
    )
    assert np.max(np.abs(sol.u)) < 1e-12
    assert np.max(np.abs(sol.optimal_drift)) < 1e-12


def test_spatially_constant_running_cost_gives_linear_in_time():
    """If F is spatially constant and g == 0, the solution is
    ``u(t, x) = (T - t) * F`` (no H or diffusion contribution). Tests
    the time integration in isolation.
    """
    n_x = 64
    x = _periodic_grid(n_x)
    T = 1.0
    c = 0.7
    sol = solve_hjb(
        sigma=0.5, T=T, x_grid=x, n_t=40,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: np.full_like(xx, c),
    )
    # Each row should be spatially constant.
    spatial_var = sol.u.std(axis=1).max()
    assert spatial_var < 1e-12
    # The spatial mean should follow (T - t) * c exactly.
    expected = (T - sol.t_grid) * c
    np.testing.assert_allclose(sol.u.mean(axis=1), expected, atol=1e-12)


# ---------- Hopf-Cole closed form ----------


def _hopf_cole_solution(sigma: float, T: float, amplitude: float = 0.1):
    r"""Closed form for the test problem ``F == 0`` with terminal
    ``u(T, x) = -sigma^2 log v_T(x)`` where
    ``v_T(x) = 1 + amplitude * cos(x)``.

    Hopf-Cole transforms the quadratic-Hamiltonian HJB into a backward
    heat equation in ``v``. With domain ``[0, 2 pi)`` the only non-zero
    mode is ``k = 1``, so

    .. math::
        v(t, x) = 1 + \mathrm{amplitude}\,
                  \mathrm{e}^{-(T - t) \sigma^2 / 2}\, \cos(x).
    """

    def v_at(t: float, x: np.ndarray) -> np.ndarray:
        decay = np.exp(-(T - t) * 0.5 * sigma * sigma)
        return 1.0 + amplitude * decay * np.cos(x)

    def u_at(t: float, x: np.ndarray) -> np.ndarray:
        return -sigma * sigma * np.log(v_at(t, x))

    return u_at


def test_hopf_cole_closed_form_matches_at_sup_norm():
    """Quantitative: at n_x = n_t = 64 the error against the closed form
    should be well below 1e-4 in sup norm. Empirically ~ 8e-6 in our
    environment; the bound is 10x looser."""
    sigma = 0.5
    T = 1.0
    amplitude = 0.1
    u_at = _hopf_cole_solution(sigma=sigma, T=T, amplitude=amplitude)
    n_x = 64
    x = _periodic_grid(n_x)
    sol = solve_hjb(
        sigma=sigma, T=T, x_grid=x, n_t=64,
        terminal=u_at(T, x),
        running_cost=lambda t, xx: np.zeros_like(xx),
    )
    err = float(np.max(np.abs(sol.u[0] - u_at(0.0, x))))
    assert err < 1e-4, f"Hopf-Cole sup error {err:.3e} above 1e-4"


def test_hopf_cole_first_order_convergence_under_refinement():
    """Halving ``dx`` (and ``dt``) should roughly halve the error. The
    fitted log-log slope of error vs ``dx`` should sit close to 1, the
    rate predicted for the IMEX / Engquist-Osher discretization."""
    sigma = 0.5
    T = 1.0
    amplitude = 0.1
    u_at = _hopf_cole_solution(sigma=sigma, T=T, amplitude=amplitude)

    errs, dxs = [], []
    for n_x in [32, 64, 128, 256]:
        n_t = n_x
        x = _periodic_grid(n_x)
        sol = solve_hjb(
            sigma=sigma, T=T, x_grid=x, n_t=n_t,
            terminal=u_at(T, x),
            running_cost=lambda t, xx: np.zeros_like(xx),
        )
        errs.append(float(np.max(np.abs(sol.u[0] - u_at(0.0, x)))))
        dxs.append(2.0 * np.pi / n_x)

    slope, _ = np.polyfit(np.log(dxs), np.log(errs), 1)
    assert slope > 0.6, f"Hopf-Cole slope {slope:.3f} below 0.6"


# ---------- Monotonicity ----------


def test_monotonicity_in_terminal_data():
    n_x = 64
    x = _periodic_grid(n_x)
    F = lambda t, xx: np.zeros_like(xx)
    g1 = np.zeros(n_x)
    g2 = 0.5 + 0.1 * np.cos(x)  # g2 > g1 pointwise (min g2 = 0.4)
    s1 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g1, running_cost=F)
    s2 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g2, running_cost=F)
    diff = s2.u - s1.u
    assert diff.min() > -1e-10, (
        f"Monotonicity violated in terminal: min(u2 - u1) = {diff.min():.3e}"
    )


def test_monotonicity_in_running_cost():
    n_x = 64
    x = _periodic_grid(n_x)
    g = np.zeros(n_x)
    F1 = lambda t, xx: np.zeros_like(xx)
    F2 = lambda t, xx: 0.3 + 0.1 * np.cos(xx)  # F2 > F1 pointwise (min F2 = 0.2)
    s1 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g, running_cost=F1)
    s2 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g, running_cost=F2)
    diff = s2.u - s1.u
    assert diff.min() > -1e-10, (
        f"Monotonicity violated in F: min(u_F2 - u_F1) = {diff.min():.3e}"
    )


# ---------- Self-convergence (no closed form) ----------


def test_self_convergence_under_grid_refinement():
    """For a smooth periodic problem with no closed form, the
    fine-vs-coarse difference at common grid points should decrease at
    rate close to 1 under joint refinement."""
    L = 2.0 * np.pi
    sigma = 0.4
    T = 1.0
    F = lambda t, xx: 0.5 * (1.0 - np.cos(xx))
    g = lambda xx: 0.2 * np.cos(2.0 * xx)

    prev_u = None
    diffs = []
    n_x_grid = [32, 64, 128, 256, 512]
    for n_x in n_x_grid:
        x = np.linspace(0.0, L, n_x, endpoint=False)
        sol = solve_hjb(
            sigma=sigma, T=T, x_grid=x, n_t=n_x,
            terminal=g(x), running_cost=F,
        )
        if prev_u is not None:
            d = float(np.max(np.abs(sol.u[0][::2] - prev_u)))
            diffs.append(d)
        prev_u = sol.u[0]

    # Slope from coarse-grid spacings to the diffs.
    dxs = [L / n for n in n_x_grid[:-1]]
    slope = float(np.polyfit(np.log(dxs), np.log(diffs), 1)[0])
    assert slope > 0.7, f"Self-convergence slope {slope:.3f} below 0.7"


# ---------- Input validation ----------


def test_invalid_sigma_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError):
        solve_hjb(sigma=0.0, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros_like(xx))


def test_invalid_T_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError):
        solve_hjb(sigma=0.5, T=-1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros_like(xx))


def test_invalid_n_t_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError):
        solve_hjb(sigma=0.5, T=1.0, x_grid=x, n_t=0,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros_like(xx))


def test_non_uniform_grid_raises():
    bad = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        solve_hjb(sigma=0.5, T=1.0, x_grid=bad, n_t=10,
                  terminal=np.zeros(5),
                  running_cost=lambda t, xx: np.zeros_like(xx))


def test_terminal_shape_mismatch_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="terminal"):
        solve_hjb(sigma=0.5, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(10),
                  running_cost=lambda t, xx: np.zeros_like(xx))


def test_unsupported_boundary_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="boundary"):
        solve_hjb(sigma=0.5, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros_like(xx),
                  boundary="neumann")


def test_unsupported_hamiltonian_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="hamiltonian"):
        solve_hjb(sigma=0.5, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros_like(xx),
                  hamiltonian="general_convex")


def test_running_cost_wrong_shape_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="running_cost"):
        solve_hjb(sigma=0.5, T=1.0, x_grid=x, n_t=4,
                  terminal=np.zeros(16),
                  running_cost=lambda t, xx: np.zeros(32))
