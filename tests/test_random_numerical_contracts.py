"""Independent regression cases for stream roles and numerical boundaries."""

import numpy as np
import pytest

from mvkit import simulate_mean_field_cir
from mvkit.mfg import solve_lq_mfg, solve_lq_mfg_vector
from mvkit.poc import (
    estimate_propagation_of_chaos_rate,
    wasserstein2_between_samples,
    wasserstein2_to_reference,
)


def _uncontrolled_trajectory(seed, vector):
    common = dict(T=1.0, n_grid=2, n_particles=4096, n_iterations_max=1, seed=seed)
    if vector:
        return solve_lq_mfg_vector(
            Q=np.zeros((1, 1)), Q_T=np.zeros((1, 1)), Sigma=np.ones((1, 1)),
            mu_0_mean=np.zeros(1), mu_0_var=np.eye(1), **common,
        ).x_trajectory[:, :, 0]
    return solve_lq_mfg(
        q=0.0, q_T=0.0, sigma=1.0, mu_0_mean=0.0, mu_0_var=1.0, **common,
    ).x_trajectory


@pytest.mark.parametrize("vector,mask", [(False, 0xA5A5A5A5), (True, 0x5A5A5A5A)])
def test_initial_and_dynamic_streams_do_not_swap_between_replicates(vector, mask):
    first = _uncontrolled_trajectory(0, vector)
    other = _uncontrolled_trajectory(mask, vector)
    np.testing.assert_array_equal(first, _uncontrolled_trajectory(0, vector))
    first_noise = (first[1] - first[0]) / np.sqrt(0.5)
    other_noise = (other[1] - other[0]) / np.sqrt(0.5)
    # The old XOR construction gives correlation one across these roles.
    for left, right in [(first_noise, other[0]), (other_noise, first[0]),
                        (first_noise, first[0])]:
        assert abs(np.corrcoef(left, right)[0, 1]) < 0.1


@pytest.mark.parametrize("bad_b", [-10.0, -1e-30, np.nan, np.inf])
def test_cir_rejects_invalid_interaction(bad_b):
    with pytest.raises(ValueError, match="b must"):
        simulate_mean_field_cir(np.array([[0.0], [1.0]]), 0.001, 1,
                                kappa=1, theta=1, b=bad_b, sigma=1)


@pytest.mark.parametrize("bad_x", [-1.0, np.nan, np.inf])
def test_cir_rejects_invalid_initial_state(bad_x):
    with pytest.raises(ValueError, match="x0"):
        simulate_mean_field_cir(np.array([[bad_x]]), 0.001, 1,
                                kappa=1, theta=1, b=0, sigma=1)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_reference_w2_resolves_declared_rare_atom(n):
    # E[Y**2] = 100**2 * 0.0001 = 1 for the only coupling to delta_0.
    distance = wasserstein2_to_reference(
        np.zeros(n), lambda u: np.where(u < 0.9999, 0.0, 100.0),
        reference_breakpoints=[0.9999],
    )
    assert distance == pytest.approx(1.0, rel=2e-11)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
@pytest.mark.parametrize("copies", [1, 3])
def test_empirical_w2_preserves_extreme_representable_scale(scale, copies):
    result = wasserstein2_between_samples([0.0], [scale] * copies)
    assert result / scale == pytest.approx(1.0, rel=1e-14)


def test_w2_rejects_multivariate_samples():
    with pytest.raises(ValueError, match="1D"):
        wasserstein2_between_samples([[0.0, 1.0]], [[1.0, 0.0]])


def test_w2_handles_overflowing_difference_when_distance_is_representable():
    result = wasserstein2_between_samples([-1e308, 0.0, 0.0, 0.0], [1e308] * 4)
    assert result / 1e308 == pytest.approx(np.sqrt(1.75), rel=1e-14)


def test_w2_rejects_unrepresentable_distance():
    with pytest.raises(ValueError, match="float64 range"):
        wasserstein2_between_samples([-1e308], [1e308])


def test_rate_estimation_passes_quantile_breakpoints_through():
    result = estimate_propagation_of_chaos_rate(
        lambda n, seed: np.zeros(n),
        lambda u: np.where(u < 0.9999, 0.0, 100.0),
        [1, 3, 10], n_seeds=1, reference_breakpoints=[0.9999],
    )
    np.testing.assert_allclose(result.w2_median, 1.0, rtol=2e-11)
    assert abs(result.fitted_slope) < 1e-10


@pytest.mark.parametrize("breaks", [[np.nan], [np.inf], [-0.1], [0.0], [1.0], [1.1], [[0.5]]])
def test_reference_w2_rejects_invalid_breakpoints(breaks):
    with pytest.raises(ValueError, match="reference_breakpoints"):
        wasserstein2_to_reference([0.0], lambda u: u, reference_breakpoints=breaks)
