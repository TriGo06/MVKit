"""Test suite for mvkit's Cucker-Smale simulator."""

import numpy as np
import pytest

from mvkit import simulate_cucker_smale


def test_output_shape_full_history():
    rng = np.random.default_rng(0)
    n, d = 50, 2
    x0 = rng.normal(size=(n, 2 * d))
    history = simulate_cucker_smale(
        x0,
        t_final=1.0,
        n_steps=100,
        spatial_dim=d,
        sigma=0.0,
        record_every=10,
        seed=1,
    )
    assert history.shape == (11, n, 2 * d)
    np.testing.assert_array_equal(history[0], x0)


def test_output_shape_terminal_only():
    rng = np.random.default_rng(0)
    n, d = 30, 2
    x0 = rng.normal(size=(n, 2 * d))
    history = simulate_cucker_smale(
        x0, t_final=1.0, n_steps=50, spatial_dim=d, sigma=0.0, seed=1
    )
    assert history.shape == (2, n, 2 * d)
    np.testing.assert_array_equal(history[0], x0)


def test_mean_velocity_conserved_without_noise():
    """Cucker-Smale fundamental conservation law (sigma = 0)."""
    rng = np.random.default_rng(42)
    n, d = 100, 2
    x0 = np.empty((n, 2 * d))
    x0[:, :d] = rng.normal(scale=0.5, size=(n, d))
    x0[:, d:] = rng.normal(scale=1.0, size=(n, d))

    initial_mean_v = x0[:, d:].mean(axis=0)

    history = simulate_cucker_smale(
        x0,
        t_final=10.0,
        n_steps=2000,
        spatial_dim=d,
        beta=0.3,
        sigma=0.0,
        seed=7,
    )
    final_state = history[-1]
    final_mean_v = final_state[:, d:].mean(axis=0)

    np.testing.assert_allclose(final_mean_v, initial_mean_v, atol=1e-8)


def test_velocity_concentration_for_small_beta():
    """For beta < 1/2, velocities concentrate around the common mean."""
    rng = np.random.default_rng(123)
    n, d = 80, 2
    x0 = np.empty((n, 2 * d))
    x0[:, :d] = rng.normal(scale=0.5, size=(n, d))
    x0[:, d:] = rng.normal(scale=1.0, size=(n, d))

    initial_var = x0[:, d:].var(axis=0).sum()

    history = simulate_cucker_smale(
        x0,
        t_final=15.0,
        n_steps=3000,
        spatial_dim=d,
        beta=0.3,
        sigma=0.0,
        seed=11,
    )
    final_var = history[-1, :, d:].var(axis=0).sum()
    assert final_var < 0.1 * initial_var, (
        f"velocities did not concentrate: {initial_var=:.4f}, {final_var=:.4f}"
    )


def test_determinism():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(20, 4))
    h1 = simulate_cucker_smale(x0, 1.0, 100, spatial_dim=2, sigma=0.5, seed=123)
    h2 = simulate_cucker_smale(x0, 1.0, 100, spatial_dim=2, sigma=0.5, seed=123)
    np.testing.assert_array_equal(h1, h2)


def test_different_seeds_produce_different_trajectories():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(20, 4))
    h1 = simulate_cucker_smale(x0, 1.0, 100, spatial_dim=2, sigma=0.5, seed=1)
    h2 = simulate_cucker_smale(x0, 1.0, 100, spatial_dim=2, sigma=0.5, seed=2)
    assert not np.allclose(h1[-1], h2[-1])


def test_dimension_mismatch_raises():
    x0 = np.zeros((10, 5))  # not 2 * spatial_dim
    with pytest.raises(ValueError, match="columns"):
        simulate_cucker_smale(x0, 1.0, 10, spatial_dim=2)


def test_invalid_n_steps_raises():
    x0 = np.zeros((10, 4))
    with pytest.raises(ValueError):
        simulate_cucker_smale(x0, 1.0, 0, spatial_dim=2)


def test_handles_1d_spatial():
    rng = np.random.default_rng(7)
    n = 40
    x0 = rng.normal(size=(n, 2))  # 1D positions and velocities
    history = simulate_cucker_smale(
        x0, t_final=2.0, n_steps=200, spatial_dim=1, sigma=0.0, seed=0
    )
    assert history.shape == (2, n, 2)
    initial_mean_v = x0[:, 1].mean()
    final_mean_v = history[-1, :, 1].mean()
    np.testing.assert_allclose(final_mean_v, initial_mean_v, atol=1e-8)
