"""Test suite for mvkit's Kuramoto simulator.

The Kuramoto critical coupling for centered Gaussian frequencies of standard
deviation sigma_omega is K_c = 2 sigma_omega sqrt(2 / pi). Below K_c the
population stays incoherent; above K_c, a fraction of oscillators lock and
the order parameter r stabilizes at a non-zero value.

The drift is O(N) per step via the order-parameter trick. If any of the
order-parameter tests below take more than a few seconds wall time, that
is a strong signal of an O(N^2) regression in the drift implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit import simulate_kuramoto


def order_parameter(phases: np.ndarray) -> float | np.ndarray:
    """Magnitude of the Kuramoto order parameter from a phase array.

    Accepts either a 1D array (single time) or 2D (time x particle)."""
    z = np.exp(1j * phases)
    return np.abs(z.mean(axis=-1))


def test_output_shape_full_history():
    rng = np.random.default_rng(0)
    n = 50
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = rng.normal(0, 0.5, size=n)
    history = simulate_kuramoto(
        x0,
        t_final=2.0,
        n_steps=100,
        coupling_k=1.0,
        omegas=omegas,
        sigma=0.05,
        record_every=10,
        seed=1,
    )
    assert history.shape == (11, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_output_shape_terminal_only():
    rng = np.random.default_rng(0)
    n = 30
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = rng.normal(0, 0.5, size=n)
    history = simulate_kuramoto(
        x0, t_final=1.0, n_steps=50, coupling_k=0.5, omegas=omegas, sigma=0.0
    )
    assert history.shape == (2, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_determinism():
    rng = np.random.default_rng(0)
    n = 40
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = rng.normal(0, 0.5, size=n)
    h1 = simulate_kuramoto(
        x0, 1.0, 100, coupling_k=1.0, omegas=omegas, sigma=0.1, seed=123
    )
    h2 = simulate_kuramoto(
        x0, 1.0, 100, coupling_k=1.0, omegas=omegas, sigma=0.1, seed=123
    )
    np.testing.assert_array_equal(h1, h2)


def test_different_seeds_produce_different_trajectories():
    rng = np.random.default_rng(0)
    n = 40
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = rng.normal(0, 0.5, size=n)
    h1 = simulate_kuramoto(
        x0, 1.0, 100, coupling_k=1.0, omegas=omegas, sigma=0.1, seed=1
    )
    h2 = simulate_kuramoto(
        x0, 1.0, 100, coupling_k=1.0, omegas=omegas, sigma=0.1, seed=2
    )
    assert not np.allclose(h1[-1], h2[-1])


def test_omegas_length_mismatch_raises():
    x0 = np.zeros((10, 1))
    omegas = np.zeros(7)
    with pytest.raises(ValueError, match="omegas"):
        simulate_kuramoto(
            x0, 1.0, 10, coupling_k=1.0, omegas=omegas, sigma=0.0
        )


def test_x0_wrong_columns_raises():
    x0 = np.zeros((10, 2))
    omegas = np.zeros(10)
    with pytest.raises(ValueError, match="columns"):
        simulate_kuramoto(
            x0, 1.0, 10, coupling_k=1.0, omegas=omegas, sigma=0.0
        )


def test_invalid_n_steps_raises():
    x0 = np.zeros((10, 1))
    omegas = np.zeros(10)
    with pytest.raises(ValueError):
        simulate_kuramoto(
            x0, 1.0, 0, coupling_k=1.0, omegas=omegas, sigma=0.0
        )


def test_negative_sigma_raises():
    x0 = np.zeros((10, 1))
    omegas = np.zeros(10)
    with pytest.raises(ValueError):
        simulate_kuramoto(
            x0, 1.0, 10, coupling_k=1.0, omegas=omegas, sigma=-0.1
        )


def test_free_rotation_uncoupled_noiseless():
    """K = 0, sigma = 0: each phase evolves as theta_i(0) + omega_i * t,
    exact under Euler since the drift is constant in time."""
    n = 50
    omegas = np.linspace(-1.0, 1.0, n)
    x0 = (np.arange(n, dtype=np.float64) * 0.07).reshape(n, 1)
    t_final = 3.0
    n_steps = 1000
    history = simulate_kuramoto(
        x0,
        t_final,
        n_steps,
        coupling_k=0.0,
        omegas=omegas,
        sigma=0.0,
        seed=0,
    )
    expected = x0[:, 0] + omegas * t_final
    np.testing.assert_allclose(history[-1, :, 0], expected, atol=1e-10)


def test_identical_frequencies_lock_to_common_mean():
    """K > 0, sigma = 0, all omegas = 0: phases converge exponentially fast
    to the centroid (rate ~ K). After K * T = 25 the spread is well below
    floating-point noise and certainly below 1e-3."""
    n = 100
    rng = np.random.default_rng(7)
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = np.zeros(n)
    history = simulate_kuramoto(
        x0,
        t_final=5.0,
        n_steps=2000,
        coupling_k=5.0,
        omegas=omegas,
        sigma=0.0,
        seed=0,
    )
    final_phases = history[-1, :, 0]
    mean_phase = np.angle(np.exp(1j * final_phases).mean())
    deviations = np.angle(np.exp(1j * (final_phases - mean_phase)))
    assert np.max(np.abs(deviations)) < 1e-3, (
        f"max angular deviation = {np.max(np.abs(deviations)):.2e}"
    )


def test_super_critical_synchronization():
    """K well above K_c: the order parameter saturates near
    r_inf = sqrt(1 - K_c / K).

    Gaussian omegas of std 0.5 give K_c = 2 * 0.5 * sqrt(2/pi) ~ 0.798. With
    K = 5.0 we have K / K_c ~ 6.27 and r_inf ~ 0.92, so r(T) > 0.7 is a
    very loose threshold; failures here would point at a real bug, not
    statistical noise.
    """
    n = 2000
    omega_std = 0.5
    rng = np.random.default_rng(42)
    omegas = rng.normal(0, omega_std, size=n)
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    history = simulate_kuramoto(
        x0,
        t_final=30.0,
        n_steps=6000,
        coupling_k=5.0,
        omegas=omegas,
        sigma=0.05,
        seed=11,
    )
    r_T = order_parameter(history[-1, :, 0])
    assert r_T > 0.7, f"r(T) = {r_T:.3f}, expected > 0.7"


def test_sub_critical_incoherence():
    """K well below K_c: the order parameter stays at the finite-N
    fluctuation level ~ 1/sqrt(N) ~ 0.022 for N = 2000.

    We assert two things: r(T) is small, and r(t) does not drift up over
    the second half of the trajectory (a slow climb would be a bug).
    """
    n = 2000
    omega_std = 0.5
    rng = np.random.default_rng(42)
    omegas = rng.normal(0, omega_std, size=n)
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    n_steps = 6000
    record_every = 60  # 100 + 1 frames
    history = simulate_kuramoto(
        x0,
        t_final=30.0,
        n_steps=n_steps,
        coupling_k=0.2,
        omegas=omegas,
        sigma=0.05,
        record_every=record_every,
        seed=11,
    )
    r_traj = order_parameter(history[..., 0])
    r_T = r_traj[-1]
    assert r_T < 0.25, f"r(T) = {r_T:.3f}, expected < 0.25"
    half = len(r_traj) // 2
    r_max_late = float(np.max(r_traj[half:]))
    assert r_max_late < 0.25, (
        f"max r(t) for t > T/2 = {r_max_late:.3f}, expected < 0.25"
    )
