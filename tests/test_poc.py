"""Tests for the propagation-of-chaos rate-estimation utility.

The Fournier-Guillin (2015) theorem gives ``E[W_2(emp_N, mu)] = O(N^{-1/2})``
in dimension 1 for laws with finite ``(4 + epsilon)``-th moment. The
slope-fit tests below check that the empirical convergence on Gaussian
samples and on the linear-quadratic McKean-Vlasov model (whose marginal
law at terminal time is Gaussian, so the assumptions are clearly met)
falls in ``[-0.6, -0.4]``. These bounds are tight; the theorem is solid,
so a failure should be investigated as a bug, not relaxed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from mvkit import simulate_linear_quadratic
from mvkit.poc import (
    estimate_propagation_of_chaos_rate,
    wasserstein2_between_samples,
    wasserstein2_to_reference,
)


def test_w2_identical_samples_is_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    assert wasserstein2_between_samples(x, x.copy()) == 0.0


def test_w2_location_shift():
    """W2(N(0, 1), N(2, 1)) = 2 exactly. At n = 10000 the empirical
    estimator is well within 0.05 of the true value."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(10_000)
    y = rng.standard_normal(10_000) + 2.0
    w2 = wasserstein2_between_samples(x, y)
    assert abs(w2 - 2.0) < 0.05, f"W2 = {w2:.4f}, expected close to 2.0"


def test_w2_scale_shift():
    """W2(N(0, 1), N(0, 4)) = |sigma_1 - sigma_2| = 1 exactly. Tolerance
    0.05 at n = 10000."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(10_000)
    y = rng.standard_normal(10_000) * 2.0
    w2 = wasserstein2_between_samples(x, y)
    assert abs(w2 - 1.0) < 0.05, f"W2 = {w2:.4f}, expected close to 1.0"


def test_w2_to_reference_inv_cdf_matches_between_samples():
    """Sanity: drawing from N(0,1) and using norm.ppf as the reference
    inverse CDF should give a small W2 of order 1/sqrt(n)."""
    rng = np.random.default_rng(3)
    n = 10_000
    samples = rng.standard_normal(n)
    w2 = wasserstein2_to_reference(samples, norm.ppf)
    assert w2 < 0.05, f"W2 to true N(0, 1) at n={n} = {w2:.4f}"


def test_rate_on_gaussian_reference():
    """Gaussian samples vs analytical N(0, 1) reference. Fournier-Guillin
    predicts slope = -0.5; we assert [-0.6, -0.4]."""
    n_values = [100, 300, 1_000, 3_000, 10_000]

    def simulator(n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed * 100 + 7).standard_normal(n)

    result = estimate_propagation_of_chaos_rate(
        simulator=simulator,
        reference_inv_cdf=norm.ppf,
        n_values=n_values,
        n_seeds=16,
    )
    assert -0.6 <= result.fitted_slope <= -0.4, (
        f"Gaussian convergence slope = {result.fitted_slope:.4f}, "
        f"expected in [-0.6, -0.4]. w2_median = {result.w2_median}"
    )


def test_rate_on_linear_quadratic():
    """End-to-end on LQ. Initial condition x0 ~ N(m_0, v_0) per particle
    sampled from a per-seed deterministic numpy RNG; terminal law is
    Gaussian with closed-form mean and variance. Reference inverse CDF is
    the corresponding norm.ppf. Expected slope -0.5; assert
    [-0.6, -0.4]."""
    a, b, sigma = -0.5, 1.0, 0.5
    t_final = 1.0
    n_steps = 1000
    m_0, v_0 = 0.0, 1.0

    m_T = m_0 * np.exp((a + b) * t_final)
    v_T = v_0 * np.exp(2 * a * t_final) + sigma**2 * (
        np.exp(2 * a * t_final) - 1
    ) / (2 * a)
    inv_cdf = norm(loc=m_T, scale=np.sqrt(v_T)).ppf

    def simulator(n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(m_0, np.sqrt(v_0), size=(n, 1))
        history = simulate_linear_quadratic(
            x0, t_final, n_steps, a=a, b=b, sigma=sigma, seed=seed
        )
        return history[-1, :, 0]

    n_values = [100, 300, 1_000, 3_000, 10_000]
    result = estimate_propagation_of_chaos_rate(
        simulator=simulator,
        reference_inv_cdf=inv_cdf,
        n_values=n_values,
        n_seeds=16,
    )
    assert -0.6 <= result.fitted_slope <= -0.4, (
        f"LQ convergence slope = {result.fitted_slope:.4f}, "
        f"expected in [-0.6, -0.4]. w2_median = {result.w2_median}"
    )


def test_estimate_rejects_singleton_n_values():
    with pytest.raises(ValueError, match="at least 2"):
        estimate_propagation_of_chaos_rate(
            simulator=lambda n, s: np.zeros(n),
            reference_inv_cdf=lambda q: np.zeros_like(q),
            n_values=[100],
            n_seeds=4,
        )
