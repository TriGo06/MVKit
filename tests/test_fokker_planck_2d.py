"""Tests for the 2D Fokker-Planck grid solver
(:func:`mvkit.mfg.solve_fokker_planck_2d`).
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import FP2DSolution, solve_fokker_planck_2d


def _periodic_grid(n: int, length: float = 2.0 * np.pi) -> np.ndarray:
    return np.linspace(0.0, length, n, endpoint=False)


def _cell_centered_grid(a: float, b: float, n: int) -> np.ndarray:
    dx = (b - a) / n
    return np.linspace(a + dx / 2.0, b - dx / 2.0, n)


def _zero_drift(t, X, Y):
    return np.zeros((X.shape[0], X.shape[1], 2), dtype=np.float64)


# ---------- Plumbing ----------


def test_solve_fp_2d_returns_solution_dataclass():
    n = 16
    L = 2.0 * np.pi
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    sol = solve_fokker_planck_2d(
        sigma=0.4, T=1.0, x_grid=x, y_grid=y, n_t=10,
        initial=np.full((n, n), 1.0 / (L * L)),
        drift=_zero_drift,
    )
    assert isinstance(sol, FP2DSolution)
    assert sol.t_grid.shape == (11,)
    assert sol.m.shape == (11, n, n)


def test_uniform_initial_zero_drift_stays_uniform():
    n = 32
    L = 2.0 * np.pi
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    m0 = np.full((n, n), 1.0 / (L * L))
    sol = solve_fokker_planck_2d(
        sigma=0.5, T=1.0, x_grid=x, y_grid=y, n_t=20,
        initial=m0, drift=_zero_drift,
    )
    assert np.max(np.abs(sol.m - 1.0 / (L * L))) < 1e-12


# ---------- Pure-diffusion closed form ----------


def _fourier_periodic_solution(sigma: float, T: float, length: float, amplitude: float = 0.5):
    """``m(t, x, y) = (1 + a cos(x) cos(y) exp(-(sigma^2/2) * 2 * t)) / L^2``
    on a 2pi-periodic square (eigenvalue -2 of d_xx + d_yy on the
    tensor-product cos mode)."""

    def m_at(t, X, Y):
        decay = np.exp(-0.5 * sigma * sigma * 2.0 * t)
        return (1.0 + amplitude * decay * np.cos(X) * np.cos(Y)) / (length * length)

    return m_at


def test_pure_diffusion_periodic_matches_eigenmode():
    sigma, T = 0.4, 1.0
    L = 2.0 * np.pi
    n = 64
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m_at = _fourier_periodic_solution(sigma, T, L)
    sol = solve_fokker_planck_2d(
        sigma=sigma, T=T, x_grid=x, y_grid=y, n_t=n,
        initial=m_at(0.0, X, Y), drift=_zero_drift,
        boundary="periodic",
    )
    err = float(np.max(np.abs(sol.m[-1] - m_at(T, X, Y))))
    assert err < 1e-3


def test_pure_diffusion_periodic_first_order_convergence():
    sigma, T = 0.4, 1.0
    L = 2.0 * np.pi
    m_at = _fourier_periodic_solution(sigma, T, L)
    errs, dxs = [], []
    for n in [16, 32, 64, 128]:
        x = _periodic_grid(n, L)
        y = _periodic_grid(n, L)
        X, Y = np.meshgrid(x, y, indexing="ij")
        sol = solve_fokker_planck_2d(
            sigma=sigma, T=T, x_grid=x, y_grid=y, n_t=n,
            initial=m_at(0.0, X, Y), drift=_zero_drift,
        )
        errs.append(float(np.max(np.abs(sol.m[-1] - m_at(T, X, Y)))))
        dxs.append(L / n)
    slope, _ = np.polyfit(np.log(dxs), np.log(errs), 1)
    assert slope > 0.7


# ---------- Mass conservation ----------


def _total_mass_2d(m: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return m.sum(axis=(1, 2)) * dx * dy


def test_mass_conserved_periodic_under_variable_drift():
    L = 2.0 * np.pi
    n = 64
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m0 = (1.0 + 0.4 * np.cos(X) * np.cos(Y)) / (L * L)

    def drift(t, X, Y):
        return np.stack([
            0.3 * np.sin(X) + 0.2 * np.cos(Y),
            -0.2 * np.cos(X) + 0.4 * np.sin(Y + t),
        ], axis=-1)

    sol = solve_fokker_planck_2d(
        sigma=0.3, T=2.0, x_grid=x, y_grid=y, n_t=200,
        initial=m0, drift=drift, boundary="periodic",
    )
    mass = _total_mass_2d(sol.m, L / n, L / n)
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_mass_conserved_neumann_under_variable_drift():
    L = 2.0
    n = 64
    x = _cell_centered_grid(0.0, L, n)
    y = _cell_centered_grid(0.0, L, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m0 = (1.0 + 0.4 * np.cos(np.pi * X / L) * np.cos(np.pi * Y / L)) / (L * L)

    def drift_neu(t, X, Y):
        return np.stack([
            0.3 * np.sin(np.pi * X / L),
            0.2 * np.cos(np.pi * Y / L),
        ], axis=-1)

    sol = solve_fokker_planck_2d(
        sigma=0.3, T=2.0, x_grid=x, y_grid=y, n_t=400,
        initial=m0, drift=drift_neu, boundary="neumann",
    )
    mass = _total_mass_2d(sol.m, L / n, L / n)
    assert np.max(np.abs(mass - 1.0)) < 1e-10


# ---------- CFL diagnostic ----------


def test_cfl_diagnostic_raises_on_extreme_drift():
    n = 16
    L = 2.0 * np.pi
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    m0 = (1.0 + 0.5 * np.cos(np.linspace(0, L, n, endpoint=False)[:, None]
                              * np.ones(n)[None, :])) / (L * L)
    with pytest.raises(ValueError, match="CFL"):
        solve_fokker_planck_2d(
            sigma=0.1, T=1.0, x_grid=x, y_grid=y, n_t=4,
            initial=m0,
            drift=lambda t, X, Y: np.stack(
                [np.full_like(X, 50.0), np.full_like(Y, 50.0)], axis=-1,
            ),
        )


# ---------- Input validation ----------


def test_invalid_sigma_raises():
    n = 8
    x = _periodic_grid(n)
    y = _periodic_grid(n)
    with pytest.raises(ValueError, match="sigma"):
        solve_fokker_planck_2d(
            sigma=0.0, T=1.0, x_grid=x, y_grid=y, n_t=10,
            initial=np.zeros((n, n)), drift=_zero_drift,
        )


def test_initial_shape_mismatch_raises():
    n = 8
    x = _periodic_grid(n)
    y = _periodic_grid(n)
    with pytest.raises(ValueError, match="initial"):
        solve_fokker_planck_2d(
            sigma=0.5, T=1.0, x_grid=x, y_grid=y, n_t=10,
            initial=np.zeros((n, n + 1)), drift=_zero_drift,
        )


def test_drift_shape_mismatch_raises():
    n = 8
    x = _periodic_grid(n)
    y = _periodic_grid(n)
    with pytest.raises(ValueError, match="drift"):
        solve_fokker_planck_2d(
            sigma=0.5, T=1.0, x_grid=x, y_grid=y, n_t=4,
            initial=np.zeros((n, n)),
            drift=lambda t, X, Y: np.zeros((n, n)),  # missing axis-2 size 2
        )


def test_unsupported_boundary_raises():
    n = 8
    x = _periodic_grid(n)
    y = _periodic_grid(n)
    with pytest.raises(ValueError, match="boundary"):
        solve_fokker_planck_2d(
            sigma=0.5, T=1.0, x_grid=x, y_grid=y, n_t=10,
            initial=np.zeros((n, n)), drift=_zero_drift,
            boundary="dirichlet",
        )


# ---------- Anisotropic sigma ----------


def test_scalar_vs_tuple_sigma_byte_identical():
    """Backward compat: ``sigma=s`` and ``sigma=(s, s)`` must produce
    byte-identical density trajectories."""
    n = 24
    L = 2.0 * np.pi
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m0 = (1.0 + 0.4 * np.cos(X) * np.cos(Y)) / (L * L)
    s1 = solve_fokker_planck_2d(
        sigma=0.4, T=1.0, x_grid=x, y_grid=y, n_t=20,
        initial=m0, drift=_zero_drift,
    )
    s2 = solve_fokker_planck_2d(
        sigma=(0.4, 0.4), T=1.0, x_grid=x, y_grid=y, n_t=20,
        initial=m0, drift=_zero_drift,
    )
    np.testing.assert_array_equal(s1.m, s2.m)


def test_anisotropic_pure_diffusion_eigenmode():
    """The cosine-product eigenmode of the weighted Laplacian decays
    with rate ``(sigma_x^2 k_x^2 + sigma_y^2 k_y^2) / 2``. With
    ``sigma_x != sigma_y`` and ``k_x = k_y = 1`` on a 2pi-periodic
    square the decay is ``(sigma_x^2 + sigma_y^2) / 2``."""
    sigma_x, sigma_y = 0.5, 0.3
    T = 1.0
    n = 64
    L = 2.0 * np.pi
    x = _periodic_grid(n, L)
    y = _periodic_grid(n, L)
    X, Y = np.meshgrid(x, y, indexing="ij")
    amplitude = 0.5

    def m_at(t, X, Y):
        decay = np.exp(-0.5 * (sigma_x ** 2 + sigma_y ** 2) * t)
        return (1.0 + amplitude * decay * np.cos(X) * np.cos(Y)) / (L * L)

    sol = solve_fokker_planck_2d(
        sigma=(sigma_x, sigma_y), T=T, x_grid=x, y_grid=y, n_t=n,
        initial=m_at(0.0, X, Y), drift=_zero_drift,
    )
    err = float(np.max(np.abs(sol.m[-1] - m_at(T, X, Y))))
    # Empirical at n=64 is ~ 4e-6; bound 1e-3 is generous.
    assert err < 1e-3
    # Mass conservation under anisotropic diffusion.
    mass = sol.m.sum(axis=(1, 2)) * (L / n) * (L / n)
    assert np.max(np.abs(mass - 1.0)) < 1e-10


def test_invalid_anisotropic_sigma_raises():
    n = 8
    x = _periodic_grid(n)
    y = _periodic_grid(n)
    with pytest.raises(ValueError, match="sigma"):
        solve_fokker_planck_2d(
            sigma=(0.5, 0.0), T=1.0, x_grid=x, y_grid=y, n_t=10,
            initial=np.zeros((n, n)), drift=_zero_drift,
        )
