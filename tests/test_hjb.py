"""Tests for the standalone HJB grid solver.

Validates :func:`mvkit.mfg.solve_hjb` on five anchored cases:

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
5. **Non-quadratic Hamiltonians**: the generic Engquist-Osher path
   reproduces the quadratic case at ``power_hamiltonian(2)``, stays
   first-order under refinement for the cubic ``H(p) = |p|^3/3``, and
   rejects Hamiltonians that are not convex with a minimum at ``p = 0``.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games: numerical
methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import Hamiltonian, HJBSolution, power_hamiltonian, solve_hjb


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
                  boundary="dirichlet")


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


# ---------- Neumann boundary ----------


def _cell_centered_grid(a: float, b: float, n_x: int) -> np.ndarray:
    """Cell-centered grid on ``[a, b]``: x_i = a + (i + 0.5) dx."""
    dx = (b - a) / n_x
    return np.linspace(a + dx / 2.0, b - dx / 2.0, n_x)


def test_neumann_zero_data_gives_zero_solution():
    n_x = 64
    x = _cell_centered_grid(0.0, 2.0, n_x)
    sol = solve_hjb(
        sigma=0.4, T=1.0, x_grid=x, n_t=40,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: np.zeros_like(xx),
        boundary="neumann",
    )
    assert np.max(np.abs(sol.u)) < 1e-12
    assert np.max(np.abs(sol.optimal_drift)) < 1e-12


def test_neumann_hopf_cole_matches_closed_form():
    """On a Neumann domain ``[0, L]`` the Laplacian's eigenfunctions are
    ``cos(k pi x / L)``. With Hopf-Cole and a single cosine mode the
    backward-heat closed form is

    .. math::
        v(t, x) = 1 + a \\cos(\\pi x / L)
                  \\exp(-(\\sigma^2/2)(\\pi/L)^2 (T - t)),

    and ``u = -sigma^2 log v``. Empirically the sup error is ~ 2e-6 at
    n_x = n_t = 64; the bound is generously looser."""
    L = 2.0
    sigma = 0.4
    T = 1.0
    amplitude = 0.1
    n_x = 64

    def u_at(t, x):
        decay = np.exp(-(T - t) * 0.5 * sigma * sigma * (np.pi / L) ** 2)
        return -sigma * sigma * np.log(
            1.0 + amplitude * decay * np.cos(np.pi * x / L)
        )

    x = _cell_centered_grid(0.0, L, n_x)
    sol = solve_hjb(
        sigma=sigma, T=T, x_grid=x, n_t=n_x,
        terminal=u_at(T, x),
        running_cost=lambda t, xx: np.zeros_like(xx),
        boundary="neumann",
    )
    err = float(np.max(np.abs(sol.u[0] - u_at(0.0, x))))
    assert err < 1e-4, f"Neumann Hopf-Cole sup error {err:.3e} above 1e-4"


def test_neumann_first_order_convergence():
    """Joint refinement of ``n_x`` and ``n_t`` halves the sup error."""
    L = 2.0
    sigma = 0.4
    T = 1.0
    amplitude = 0.1

    def u_at(t, x):
        decay = np.exp(-(T - t) * 0.5 * sigma * sigma * (np.pi / L) ** 2)
        return -sigma * sigma * np.log(
            1.0 + amplitude * decay * np.cos(np.pi * x / L)
        )

    errs, dxs = [], []
    for n_x in [32, 64, 128, 256]:
        x = _cell_centered_grid(0.0, L, n_x)
        sol = solve_hjb(
            sigma=sigma, T=T, x_grid=x, n_t=n_x,
            terminal=u_at(T, x),
            running_cost=lambda t, xx: np.zeros_like(xx),
            boundary="neumann",
        )
        errs.append(float(np.max(np.abs(sol.u[0] - u_at(0.0, x)))))
        dxs.append(L / n_x)

    slope, _ = np.polyfit(np.log(dxs), np.log(errs), 1)
    assert slope > 0.6, f"Neumann Hopf-Cole slope {slope:.3f} below 0.6"


def test_neumann_handles_asymmetric_data_periodic_raises():
    """On asymmetric quadratic data the periodic BC creates a
    discontinuous wrap-around that the EO upwind cannot stabilize: the
    CFL guard rightly raises. The Neumann solver, by contrast, sees a
    smooth ``partial_x u = 0`` boundary and integrates without
    incident. This test pins down the qualitative win of the new BC.
    """
    n_x = 64
    x = _cell_centered_grid(-2.0, 2.0, n_x)
    terminal = 0.5 * (x - 0.5) ** 2  # asymmetric in x

    sol_neumann = solve_hjb(
        sigma=0.4, T=1.0, x_grid=x, n_t=n_x,
        terminal=terminal,
        running_cost=lambda t, xx: np.zeros_like(xx),
        boundary="neumann",
    )
    assert np.isfinite(sol_neumann.u).all()

    with pytest.raises(ValueError, match="CFL"):
        solve_hjb(
            sigma=0.4, T=1.0, x_grid=x, n_t=n_x,
            terminal=terminal,
            running_cost=lambda t, xx: np.zeros_like(xx),
            boundary="periodic",
        )


# ---------- Non-quadratic Hamiltonians ----------


def test_power_q2_matches_quadratic_string():
    """``power_hamiltonian(2)`` is exactly ``H(p) = p^2/2``, so the
    generic Engquist-Osher path must reproduce the built-in
    ``"quadratic"`` case to round-off in both ``u`` and the drift."""
    n_x = 64
    x = _periodic_grid(n_x)
    common = dict(
        sigma=0.4, T=1.0, x_grid=x, n_t=80,
        terminal=0.2 * np.cos(2.0 * x),
        running_cost=lambda t, xx: 0.5 * (1.0 - np.cos(xx)),
    )
    s_str = solve_hjb(**common, hamiltonian="quadratic")
    s_pow = solve_hjb(**common, hamiltonian=power_hamiltonian(2.0))
    np.testing.assert_allclose(s_pow.u, s_str.u, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        s_pow.optimal_drift, s_str.optimal_drift, rtol=0.0, atol=1e-12
    )


def test_nonquadratic_constant_data_linear_in_time():
    """With ``F`` spatially constant and ``g = 0`` the solution stays
    spatially constant, so ``partial_x u = 0`` and ``H(0) = 0``
    contributes nothing: ``u = (T - t) F`` for any Hamiltonian. Confirms
    the generic path respects ``H(0) = 0``."""
    n_x = 64
    x = _periodic_grid(n_x)
    T = 1.0
    c = 0.7
    sol = solve_hjb(
        sigma=0.5, T=T, x_grid=x, n_t=40,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: np.full_like(xx, c),
        hamiltonian=power_hamiltonian(3.0),
    )
    assert sol.u.std(axis=1).max() < 1e-12
    np.testing.assert_allclose(
        sol.u.mean(axis=1), (T - sol.t_grid) * c, atol=1e-12
    )
    assert np.max(np.abs(sol.optimal_drift)) < 1e-12


def test_nonquadratic_self_convergence_under_refinement():
    """For a smooth periodic problem with the cubic Hamiltonian
    ``H(p) = |p|^3/3`` (no closed form), the fine-vs-coarse difference
    decreases at a rate close to 1 under joint refinement, the rate of
    the Engquist-Osher upwind scheme."""
    L = 2.0 * np.pi
    sigma = 0.4
    T = 1.0
    ham = power_hamiltonian(3.0)
    F = lambda t, xx: 0.5 * (1.0 - np.cos(xx))
    g = lambda xx: 0.2 * np.cos(2.0 * xx)

    prev_u = None
    diffs = []
    n_x_grid = [32, 64, 128, 256, 512]
    for n_x in n_x_grid:
        x = np.linspace(0.0, L, n_x, endpoint=False)
        sol = solve_hjb(
            sigma=sigma, T=T, x_grid=x, n_t=n_x,
            terminal=g(x), running_cost=F, hamiltonian=ham,
        )
        if prev_u is not None:
            diffs.append(float(np.max(np.abs(sol.u[0][::2] - prev_u))))
        prev_u = sol.u[0]

    dxs = [L / n for n in n_x_grid[:-1]]
    slope = float(np.polyfit(np.log(dxs), np.log(diffs), 1)[0])
    assert slope > 0.7, (
        f"non-quadratic self-convergence slope {slope:.3f} below 0.7"
    )


def test_nonquadratic_monotonicity_in_terminal_data():
    """The HJB operator is monotone in the data for any convex
    Hamiltonian: ``g1 <= g2`` pointwise must give ``u1 <= u2``."""
    n_x = 64
    x = _periodic_grid(n_x)
    ham = power_hamiltonian(3.0)
    F = lambda t, xx: np.zeros_like(xx)
    g1 = np.zeros(n_x)
    g2 = 0.5 + 0.1 * np.cos(x)
    s1 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g1,
                   running_cost=F, hamiltonian=ham)
    s2 = solve_hjb(sigma=0.3, T=1.0, x_grid=x, n_t=40, terminal=g2,
                   running_cost=F, hamiltonian=ham)
    diff = s2.u - s1.u
    assert diff.min() > -1e-10, (
        f"monotonicity violated: min(u2 - u1) = {diff.min():.3e}"
    )


def test_nonquadratic_value_function_differs_from_quadratic():
    """A genuinely non-quadratic Hamiltonian must change the value
    function: ``power_hamiltonian(4)`` and the quadratic case, on the
    same problem with order-one gradients, differ well above
    discretization noise."""
    n_x = 128
    x = _periodic_grid(n_x)
    common = dict(
        sigma=0.3, T=1.0, x_grid=x, n_t=160,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: 1.0 - np.cos(xx),
    )
    s_quad = solve_hjb(**common, hamiltonian="quadratic")
    s_pow4 = solve_hjb(**common, hamiltonian=power_hamiltonian(4.0))
    sup_diff = float(np.max(np.abs(s_pow4.u - s_quad.u)))
    assert sup_diff > 0.02, (
        f"power(4) and quadratic differ by only {sup_diff:.3e}; the "
        "Hamiltonian is not taking effect"
    )


def test_custom_callable_hamiltonian_runs_and_differs():
    """A user-supplied convex Hamiltonian (here ``H(p) = cosh(p) - 1``,
    ``H'(p) = sinh(p)``) is accepted and yields a finite solution
    distinct from the quadratic case."""
    n_x = 96
    x = _periodic_grid(n_x)
    ham = Hamiltonian(
        H=lambda p: np.cosh(p) - 1.0,
        dH=lambda p: np.sinh(p),
        name="cosh",
    )
    common = dict(
        sigma=0.4, T=1.0, x_grid=x, n_t=120,
        terminal=np.zeros(n_x),
        running_cost=lambda t, xx: 0.5 * (1.0 - np.cos(xx)),
    )
    sol = solve_hjb(**common, hamiltonian=ham)
    assert isinstance(sol, HJBSolution)
    assert np.isfinite(sol.u).all()
    s_quad = solve_hjb(**common, hamiltonian="quadratic")
    assert float(np.max(np.abs(sol.u - s_quad.u))) > 1e-4


def test_nonquadratic_neumann_runs():
    """The non-quadratic path composes with the Neumann boundary: a
    power Hamiltonian on a reflecting domain integrates cleanly."""
    n_x = 64
    x = _cell_centered_grid(0.0, 2.0, n_x)
    sol = solve_hjb(
        sigma=0.4, T=1.0, x_grid=x, n_t=64,
        terminal=0.1 * np.cos(np.pi * x / 2.0),
        running_cost=lambda t, xx: np.zeros_like(xx),
        boundary="neumann",
        hamiltonian=power_hamiltonian(3.0),
    )
    assert np.isfinite(sol.u).all()
    assert np.isfinite(sol.optimal_drift).all()


def test_nonconvex_hamiltonian_rejected():
    """A concave 'Hamiltonian' (``H = -p^2/2``) violates the
    Engquist-Osher assumptions and must be rejected on entry."""
    n_x = 32
    x = _periodic_grid(n_x)
    bad = Hamiltonian(H=lambda p: -0.5 * p**2, dH=lambda p: -p, name="concave")
    with pytest.raises(ValueError):
        solve_hjb(sigma=0.4, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(n_x),
                  running_cost=lambda t, xx: np.zeros_like(xx),
                  hamiltonian=bad)


def test_hamiltonian_nonzero_at_origin_rejected():
    """``H(0)`` must be 0; a Hamiltonian shifted off the origin is
    rejected."""
    n_x = 32
    x = _periodic_grid(n_x)
    bad = Hamiltonian(H=lambda p: 0.5 * p**2 + 1.0, dH=lambda p: p, name="shifted")
    with pytest.raises(ValueError, match="H\\(0\\)"):
        solve_hjb(sigma=0.4, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(n_x),
                  running_cost=lambda t, xx: np.zeros_like(xx),
                  hamiltonian=bad)


def test_invalid_hamiltonian_type_rejected():
    """A ``hamiltonian`` that is neither the string ``"quadratic"`` nor
    a :class:`Hamiltonian` instance is rejected with a clear error."""
    n_x = 32
    x = _periodic_grid(n_x)
    with pytest.raises(ValueError, match="hamiltonian"):
        solve_hjb(sigma=0.4, T=1.0, x_grid=x, n_t=10,
                  terminal=np.zeros(n_x),
                  running_cost=lambda t, xx: np.zeros_like(xx),
                  hamiltonian=42)


def test_power_hamiltonian_rejects_q_not_above_one():
    """``power_hamiltonian`` needs ``q > 1`` for ``H`` to be convex and
    continuously differentiable."""
    for bad_q in [1.0, 0.5, -2.0]:
        with pytest.raises(ValueError, match="q > 1"):
            power_hamiltonian(bad_q)
