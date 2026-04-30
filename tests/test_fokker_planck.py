"""Tests for the standalone forward Fokker-Planck grid solver.

Validates :func:`mvkit.mfg.solve_fokker_planck` on a hierarchy of cases:

1. **Trivial steady state**: zero drift plus uniform initial density
   stays uniform.
2. **Pure-diffusion Fourier mode**: with ``alpha = 0`` the lone non-zero
   Fourier mode of a cosine initial decays as
   :math:`\\exp(-(\\sigma^2/2) k^2 t)` (the heat semigroup). The
   numerical solution is compared to this closed form at sup norm.
3. **Constant-drift Fourier mode**: with ``alpha = c`` constant the
   cosine mode translates and decays:
   :math:`m(t, x) = 1/L + (a/L) \\exp(-(\\sigma^2/2) k^2 t)
   \\cos(k(x - c t))`. This is the joint convection-diffusion test.
4. **Mass conservation**: the conservative-form upwind plus periodic
   implicit-Laplacian discretization preserves total mass exactly,
   modulo float arithmetic. Verified across multiple time-varying
   drifts.
5. **Convergence rate**: under joint refinement of ``n_x`` and ``n_t``
   the sup error against the closed form should decrease at first
   order.
6. **Positivity preservation**: with non-negative initial density and
   the CFL on convection respected, the discrete scheme keeps
   :math:`m \\ge 0` modulo round-off.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import FPSolution, solve_fokker_planck


def _periodic_grid(n_x: int, length: float = 2.0 * np.pi) -> np.ndarray:
    return np.linspace(0.0, length, n_x, endpoint=False)


def _zero_drift(t: float, x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


# ---------- Sanity ----------


def test_solve_fp_returns_solution_dataclass():
    n_x, T = 64, 1.0
    x = _periodic_grid(n_x)
    L = 2.0 * np.pi
    sol = solve_fokker_planck(
        sigma=0.4, T=T, x_grid=x, n_t=10,
        initial=np.full(n_x, 1.0 / L),
        drift=_zero_drift,
    )
    assert isinstance(sol, FPSolution)
    assert sol.t_grid.shape == (11,)
    assert sol.x_grid.shape == (n_x,)
    assert sol.m.shape == (11, n_x)
    assert sol.t_grid[0] == 0.0
    assert sol.t_grid[-1] == pytest.approx(T)
    np.testing.assert_array_equal(sol.m[0], np.full(n_x, 1.0 / L))


def test_uniform_initial_zero_drift_stays_uniform():
    n_x = 64
    L = 2.0 * np.pi
    x = _periodic_grid(n_x, L)
    sol = solve_fokker_planck(
        sigma=0.5, T=1.0, x_grid=x, n_t=50,
        initial=np.full(n_x, 1.0 / L),
        drift=_zero_drift,
    )
    # Spatial variance at every recorded time should be machine zero.
    assert sol.m.std(axis=1).max() < 1e-12
    assert np.max(np.abs(sol.m - 1.0 / L)) < 1e-12


# ---------- Closed-form Fourier mode ----------


def _fourier_solution(
    sigma: float, c: float, amplitude: float, T: float, length: float = 2.0 * np.pi
):
    """Return the closed form for the test problem
    ``m_0(x) = (1 + amplitude * cos(2 pi x / L)) / L`` under constant
    drift ``c`` (``c = 0`` is pure diffusion).
    """

    k_freq = 2.0 * np.pi / length

    def m_at(t: float, x: np.ndarray) -> np.ndarray:
        decay = np.exp(-0.5 * sigma * sigma * k_freq * k_freq * t)
        return (1.0 + amplitude * decay * np.cos(k_freq * (x - c * t))) / length

    return m_at


def test_pure_diffusion_matches_fourier_closed_form():
    """``alpha = 0``, cosine initial: matches the heat-semigroup solution
    in sup norm. Empirically ~ 9e-6 at n_x = n_t = 64; the bound is 10x
    looser."""
    L = 2.0 * np.pi
    sigma, T = 0.4, 1.0
    amplitude = 0.5
    n_x, n_t = 64, 50
    x = _periodic_grid(n_x, L)
    m_at = _fourier_solution(sigma=sigma, c=0.0, amplitude=amplitude, T=T, length=L)
    sol = solve_fokker_planck(
        sigma=sigma, T=T, x_grid=x, n_t=n_t,
        initial=m_at(0.0, x),
        drift=_zero_drift,
    )
    err = float(np.max(np.abs(sol.m[-1] - m_at(T, x))))
    assert err < 1e-4, f"pure-diffusion sup error {err:.3e} above 1e-4"


def test_constant_drift_matches_fourier_closed_form():
    """``alpha = c`` constant: the cosine mode translates (with phase
    velocity ``c``) while decaying. Tests convection and diffusion
    together. CFL: ``dt <= dx / |c|``; we pick ``n_t`` accordingly."""
    L = 2.0 * np.pi
    sigma, T = 0.4, 1.0
    c = 1.0
    amplitude = 0.5
    n_x = 128
    n_t = 8 * n_x  # comfortably below the CFL dt = dx / c
    x = _periodic_grid(n_x, L)
    m_at = _fourier_solution(sigma=sigma, c=c, amplitude=amplitude, T=T, length=L)
    sol = solve_fokker_planck(
        sigma=sigma, T=T, x_grid=x, n_t=n_t,
        initial=m_at(0.0, x),
        drift=lambda t, xx: np.full_like(xx, c),
    )
    err = float(np.max(np.abs(sol.m[-1] - m_at(T, x))))
    assert err < 5e-3, f"convection+diffusion sup error {err:.3e} above 5e-3"


def test_first_order_convergence_under_refinement():
    """Joint refinement of ``n_x`` and ``n_t`` halves the sup error.
    Theoretical rate for the upwind / IMEX scheme is 1; the fitted
    log-log slope should comfortably exceed 0.7."""
    L = 2.0 * np.pi
    sigma, T = 0.4, 1.0
    c = 1.0
    amplitude = 0.5
    m_at = _fourier_solution(sigma=sigma, c=c, amplitude=amplitude, T=T, length=L)

    errs, dxs = [], []
    for n_x in [32, 64, 128, 256]:
        x = _periodic_grid(n_x, L)
        n_t = 8 * n_x
        sol = solve_fokker_planck(
            sigma=sigma, T=T, x_grid=x, n_t=n_t,
            initial=m_at(0.0, x),
            drift=lambda t, xx: np.full_like(xx, c),
        )
        errs.append(float(np.max(np.abs(sol.m[-1] - m_at(T, x)))))
        dxs.append(L / n_x)

    slope, _ = np.polyfit(np.log(dxs), np.log(errs), 1)
    assert slope > 0.7, f"convergence slope {slope:.3f} below 0.7"


# ---------- Mass conservation ----------


def _total_mass(m: np.ndarray, dx: float) -> np.ndarray:
    return m.sum(axis=1) * dx


def test_mass_conserved_pure_diffusion():
    L = 2.0 * np.pi
    n_x = 128
    x = _periodic_grid(n_x, L)
    m0 = (1.0 + 0.6 * np.cos(x)) / L
    sol = solve_fokker_planck(
        sigma=0.5, T=2.0, x_grid=x, n_t=200,
        initial=m0, drift=_zero_drift,
    )
    mass = _total_mass(sol.m, L / n_x)
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_mass_conserved_under_time_varying_drift():
    L = 2.0 * np.pi
    n_x = 128
    x = _periodic_grid(n_x, L)
    m0 = (1.0 + 0.6 * np.cos(x)) / L

    def drift(t, xx):
        return -np.sin(xx) + 0.5 * np.cos(2.0 * xx + t)

    sol = solve_fokker_planck(
        sigma=0.3, T=2.0, x_grid=x, n_t=2000,
        initial=m0, drift=drift,
    )
    mass = _total_mass(sol.m, L / n_x)
    # 7e-14 typical accumulated round-off across 2000 implicit solves.
    assert np.max(np.abs(mass - 1.0)) < 1e-10


# ---------- Positivity ----------


def test_positivity_preserved_under_cfl():
    """With CFL respected, the upwind / implicit diffusion scheme keeps
    densities non-negative. Use a smooth strictly-positive initial and
    a non-trivial drift; check ``min(m) > -roundoff`` over the whole
    spacetime grid."""
    L = 2.0 * np.pi
    n_x = 128
    x = _periodic_grid(n_x, L)
    m0 = (1.0 + 0.6 * np.cos(x)) / L
    sol = solve_fokker_planck(
        sigma=0.3, T=2.0, x_grid=x, n_t=2000,
        initial=m0,
        drift=lambda t, xx: -np.sin(xx) + 0.3 * np.cos(2.0 * xx),
    )
    assert sol.m.min() > -1e-12


# ---------- Input validation ----------


def test_invalid_sigma_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="sigma"):
        solve_fokker_planck(sigma=0.0, T=1.0, x_grid=x, n_t=10,
                            initial=np.zeros(16), drift=_zero_drift)


def test_invalid_T_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="T"):
        solve_fokker_planck(sigma=0.5, T=-1.0, x_grid=x, n_t=10,
                            initial=np.zeros(16), drift=_zero_drift)


def test_invalid_n_t_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="n_t"):
        solve_fokker_planck(sigma=0.5, T=1.0, x_grid=x, n_t=0,
                            initial=np.zeros(16), drift=_zero_drift)


def test_non_uniform_grid_raises():
    bad = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        solve_fokker_planck(sigma=0.5, T=1.0, x_grid=bad, n_t=10,
                            initial=np.zeros(5), drift=_zero_drift)


def test_initial_shape_mismatch_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="initial"):
        solve_fokker_planck(sigma=0.5, T=1.0, x_grid=x, n_t=10,
                            initial=np.zeros(10), drift=_zero_drift)


def test_unsupported_boundary_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="boundary"):
        solve_fokker_planck(sigma=0.5, T=1.0, x_grid=x, n_t=10,
                            initial=np.zeros(16), drift=_zero_drift,
                            boundary="dirichlet")


def test_drift_wrong_shape_raises():
    x = _periodic_grid(16)
    with pytest.raises(ValueError, match="drift"):
        solve_fokker_planck(sigma=0.5, T=1.0, x_grid=x, n_t=4,
                            initial=np.zeros(16),
                            drift=lambda t, xx: np.zeros(32))


# ---------- Neumann boundary ----------


def _cell_centered_grid(a: float, b: float, n_x: int) -> np.ndarray:
    dx = (b - a) / n_x
    return np.linspace(a + dx / 2.0, b - dx / 2.0, n_x)


def test_neumann_pure_diffusion_matches_cosine_eigenmode():
    """On a Neumann domain ``[0, L]`` the Laplacian's eigenfunctions
    are ``cos(k pi x / L)``. With ``alpha = 0`` and a single cosine
    mode initial, the closed-form decay is

    .. math::
        m(t, x) = (1 + a \\cos(\\pi x / L)
                       \\exp(-(\\sigma^2/2)(\\pi/L)^2 t)) / L.

    Empirically the sup error is ~ 7e-5 at ``n_x = n_t = 64``."""
    L = 2.0
    sigma = 0.4
    T = 1.0

    def m_at(t, x):
        decay = np.exp(-0.5 * sigma * sigma * (np.pi / L) ** 2 * t)
        return (1.0 + 0.5 * decay * np.cos(np.pi * x / L)) / L

    n_x = 64
    x = _cell_centered_grid(0.0, L, n_x)
    sol = solve_fokker_planck(
        sigma=sigma, T=T, x_grid=x, n_t=n_x,
        initial=m_at(0.0, x),
        drift=_zero_drift,
        boundary="neumann",
    )
    err = float(np.max(np.abs(sol.m[-1] - m_at(T, x))))
    assert err < 1e-3, f"Neumann pure-diffusion sup error {err:.3e} above 1e-3"


def test_neumann_first_order_convergence_under_refinement():
    """Joint refinement of ``n_x`` and ``n_t`` should halve the sup
    error. Empirical slope ~ 1.1 in our environment."""
    L = 2.0
    sigma = 0.4
    T = 1.0

    def m_at(t, x):
        decay = np.exp(-0.5 * sigma * sigma * (np.pi / L) ** 2 * t)
        return (1.0 + 0.5 * decay * np.cos(np.pi * x / L)) / L

    errs, dxs = [], []
    for n_x in [32, 64, 128, 256]:
        x = _cell_centered_grid(0.0, L, n_x)
        sol = solve_fokker_planck(
            sigma=sigma, T=T, x_grid=x, n_t=n_x,
            initial=m_at(0.0, x), drift=_zero_drift,
            boundary="neumann",
        )
        errs.append(float(np.max(np.abs(sol.m[-1] - m_at(T, x)))))
        dxs.append(L / n_x)

    slope, _ = np.polyfit(np.log(dxs), np.log(errs), 1)
    assert slope > 0.7, f"Neumann FP slope {slope:.3f} below 0.7"


def test_neumann_mass_conserved_under_variable_drift():
    """No-flux BC plus conservative-form discretization must preserve
    total mass exactly, even under a non-zero, non-trivial drift."""
    L = 2.0 * np.pi
    n_x = 128
    x = _cell_centered_grid(0.0, L, n_x)
    m0 = (1.0 + 0.5 * np.cos(np.pi * x / L)) / L
    dx = L / n_x

    def drift(t, xx):
        return 0.3 * np.sin(np.pi * xx / L) + 0.2 * np.cos(2 * np.pi * xx / L + t)

    sol = solve_fokker_planck(
        sigma=0.3, T=2.0, x_grid=x, n_t=400,
        initial=m0, drift=drift, boundary="neumann",
    )
    mass = sol.m.sum(axis=1) * dx
    assert np.max(np.abs(mass - 1.0)) < 1e-10
    assert sol.m.min() > -1e-10
