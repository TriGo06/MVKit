"""Test suite for mvkit's linear-quadratic McKean-Vlasov simulator.

The closed-form Gaussian moments of the LQ model make it a quantitative
benchmark for any mean-field integrator. By Talay and Tubaro (1990,
*Expansion of the global error for numerical schemes solving stochastic
differential equations*), Euler-Maruyama applied to an SDE with smooth drift
and additive noise has weak error of order 1 on smooth functionals of the
terminal state. We check this on the empirical mean and the empirical
variance.

A genuine *strong* error test (single-particle trajectory error against an
analytical reference using the same Brownian path) is intentionally deferred:
it requires exposing Brownian increments through the integrator, which is a
larger change. See roadmap item 2 onward.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit import simulate_linear_quadratic


def _v_T(a: float, sigma: float, t: float) -> float:
    """Closed-form variance of the LQ marginal at time t, with v_0 = 0."""
    if a == 0.0:
        return sigma * sigma * t
    return sigma * sigma * (np.exp(2.0 * a * t) - 1.0) / (2.0 * a)


def test_output_shape_full_history():
    rng = np.random.default_rng(0)
    n = 50
    x0 = rng.normal(size=(n, 1))
    history = simulate_linear_quadratic(
        x0,
        t_final=1.0,
        n_steps=100,
        a=-0.5,
        b=1.0,
        sigma=0.0,
        record_every=10,
        seed=1,
    )
    assert history.shape == (11, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_output_shape_terminal_only():
    rng = np.random.default_rng(0)
    n = 30
    x0 = rng.normal(size=(n, 1))
    history = simulate_linear_quadratic(
        x0, t_final=1.0, n_steps=50, a=0.5, b=-0.3, sigma=0.0, seed=1
    )
    assert history.shape == (2, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_determinism():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(40, 1))
    h1 = simulate_linear_quadratic(
        x0, 1.0, 100, a=-0.5, b=1.0, sigma=0.5, seed=123
    )
    h2 = simulate_linear_quadratic(
        x0, 1.0, 100, a=-0.5, b=1.0, sigma=0.5, seed=123
    )
    np.testing.assert_array_equal(h1, h2)


def test_different_seeds_produce_different_trajectories():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(40, 1))
    h1 = simulate_linear_quadratic(
        x0, 1.0, 100, a=-0.5, b=1.0, sigma=0.5, seed=1
    )
    h2 = simulate_linear_quadratic(
        x0, 1.0, 100, a=-0.5, b=1.0, sigma=0.5, seed=2
    )
    assert not np.allclose(h1[-1], h2[-1])


def test_dimension_mismatch_raises():
    x0 = np.zeros((10, 2))
    with pytest.raises(ValueError, match="columns"):
        simulate_linear_quadratic(x0, 1.0, 10, a=0.0, b=0.0, sigma=0.5)


def test_invalid_n_steps_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError):
        simulate_linear_quadratic(x0, 1.0, 0, a=0.0, b=0.0, sigma=0.5)


def test_negative_sigma_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError):
        simulate_linear_quadratic(x0, 1.0, 10, a=0.0, b=0.0, sigma=-0.1)


def test_mean_no_noise_matches_ode():
    """With sigma = 0 the empirical mean follows m_0 exp((a + b) t) exactly,
    up to the Euler bias on a 1D linear ODE which is O(dt)."""
    n = 200
    m0 = 1.5
    x0 = np.full((n, 1), m0)
    a, b = -0.7, 1.4
    t_final = 1.5
    history = simulate_linear_quadratic(
        x0, t_final, n_steps=5000, a=a, b=b, sigma=0.0, seed=0
    )
    final_mean = history[-1, :, 0].mean()
    expected = m0 * np.exp((a + b) * t_final)
    np.testing.assert_allclose(final_mean, expected, atol=1e-3)


def test_smoke_mean_at_finite_n_steps():
    """At n_steps = 2000, the empirical mean is within 1% of m(T).

    Complements the parametric weak-order test: catches a broken
    implementation that happens to have a clean convergence slope.
    """
    a, b, sigma = -2.0, 2.5, 0.5
    t_final = 8.0
    m_0 = 1.0
    n = 50000
    x0 = np.full((n, 1), m_0)
    history = simulate_linear_quadratic(
        x0, t_final, n_steps=2000, a=a, b=b, sigma=sigma, seed=42
    )
    empirical_mean = history[-1, :, 0].mean()
    m_T = m_0 * np.exp((a + b) * t_final)
    rel_err = abs(empirical_mean - m_T) / m_T
    assert rel_err < 0.01, f"relative mean error = {rel_err:.4f}"


def test_smoke_variance_at_finite_n_steps():
    """At n_steps = 2000, the empirical variance is within 5% of v(T)."""
    a, b, sigma = -2.0, 2.5, 0.5
    t_final = 8.0
    m_0 = 1.0
    n = 50000
    x0 = np.full((n, 1), m_0)
    history = simulate_linear_quadratic(
        x0, t_final, n_steps=2000, a=a, b=b, sigma=sigma, seed=42
    )
    empirical_var = history[-1, :, 0].var(ddof=1)
    v_T = _v_T(a, sigma, t_final)
    rel_err = abs(empirical_var - v_T) / v_T
    assert rel_err < 0.05, f"relative variance error = {rel_err:.4f}"


@pytest.fixture(scope="module")
def lq_convergence_data():
    """Run the LQ Euler-Maruyama sweep used by the weak-order tests.

    Parameters are calibrated so that, at the smallest dt of the range, the
    Euler bias on both the empirical mean and the empirical variance sits
    well above the Monte Carlo noise floor. Cached at module scope so the
    mean and variance slope tests share a single sweep.
    """
    a, b, sigma = -2.0, 2.5, 0.5
    t_final = 8.0
    m_0 = 1.0
    n_particles = 50_000
    n_seeds = 16
    n_steps_list = [50, 100, 200, 400, 800]

    rng = np.random.default_rng(20260428)
    seeds = rng.integers(1, 2**31 - 1, size=(len(n_steps_list), n_seeds))

    means = np.empty((len(n_steps_list), n_seeds))
    variances = np.empty((len(n_steps_list), n_seeds))
    for k, n_steps in enumerate(n_steps_list):
        x0 = np.full((n_particles, 1), m_0)
        for s in range(n_seeds):
            history = simulate_linear_quadratic(
                x0,
                t_final,
                n_steps,
                a=a,
                b=b,
                sigma=sigma,
                seed=int(seeds[k, s]),
            )
            terminal = history[-1, :, 0]
            means[k, s] = terminal.mean()
            variances[k, s] = terminal.var(ddof=1)

    m_T = m_0 * np.exp((a + b) * t_final)
    v_T = _v_T(a, sigma, t_final)
    return {
        "n_steps": np.array(n_steps_list),
        "dt": t_final / np.array(n_steps_list, dtype=float),
        "means": means,
        "variances": variances,
        "m_T": m_T,
        "v_T": v_T,
        "n_particles": n_particles,
        "n_seeds": n_seeds,
    }


def test_weak_order_mean(lq_convergence_data):
    """Slope of log |E[empirical_mean] - m(T)| vs log dt should be 1.

    Talay-Tubaro (1990) weak order 1 of Euler-Maruyama with smooth drift
    and additive noise.
    """
    data = lq_convergence_data
    bias_est = data["means"].mean(axis=1) - data["m_T"]

    mc_noise = np.sqrt(data["v_T"] / (data["n_particles"] * data["n_seeds"]))
    assert abs(bias_est[-1]) > 3.0 * mc_noise, (
        "MC noise too close to bias at smallest dt; slope fit unreliable. "
        f"|bias|={abs(bias_est[-1]):.3e}, MC={mc_noise:.3e}"
    )

    log_dt = np.log(data["dt"])
    log_err = np.log(np.abs(bias_est))
    slope, _ = np.polyfit(log_dt, log_err, 1)
    assert 0.85 <= slope <= 1.15, f"weak-order slope on mean = {slope:.4f}"


def test_weak_order_variance(lq_convergence_data):
    """Slope of log |E[empirical_var] - v(T)| vs log dt should be 1."""
    data = lq_convergence_data
    bias_est = data["variances"].mean(axis=1) - data["v_T"]

    mc_noise = data["v_T"] * np.sqrt(
        2.0 / (data["n_particles"] * data["n_seeds"])
    )
    assert abs(bias_est[-1]) > 3.0 * mc_noise, (
        "MC noise too close to bias at smallest dt; slope fit unreliable. "
        f"|bias|={abs(bias_est[-1]):.3e}, MC={mc_noise:.3e}"
    )

    log_dt = np.log(data["dt"])
    log_err = np.log(np.abs(bias_est))
    slope, _ = np.polyfit(log_dt, log_err, 1)
    assert 0.85 <= slope <= 1.15, f"weak-order slope on variance = {slope:.4f}"
