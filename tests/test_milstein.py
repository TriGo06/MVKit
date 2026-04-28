"""Test suite for the Milstein scheme and the McKean-Vlasov CIR model.

Limits of weak-error testing for Milstein within the current API
================================================================

The Milstein scheme has strong order 1 (vs Euler's strong order 1/2), but
on weak error of smooth functionals of `X_T` it is also order 1, like
Euler. The constants of those leading O(dt) terms can go in either
direction depending on the functional and the model: there is no
theoretical reason to expect Milstein's constant to be smaller than
Euler's, and on CIR a direct calculation shows the Milstein-only
contribution to `E[X_{n+1}^2 | X_n]` is `+0.125 sigma^4 dt^2`, i.e. the
second-moment bias goes the wrong way (slightly worse for Milstein) by
an O(dt^2) amount that is in any case below the MC noise floor.

The strong-order-1 improvement is therefore *not* observable via weak
error of `f(X_T)` for smooth `f`. It only shows up on:

  * pathwise (strong) error `E[sup_t |X_t^h - X_t^exact|]`, which
    requires sharing Brownian increments between a coarse-dt run and a
    fine-dt reference, and
  * non-smooth functionals of the trajectory (barrier hits, trajectory
    maxima, etc.), which also rely on accurate pathwise behavior.

Both require an integrator hook that exposes the Brownian increments,
which we have explicitly deferred to a future PR. Until that lands, the
tests below restrict themselves to what *is* verifiable with the current
public API:

  1. On models with constant diffusion (LQ, Cucker-Smale, Kuramoto) the
     default `diffusion_derivative` returns zero, so Milstein's
     correction term vanishes and the two schemes produce bit-exact
     identical trajectories. We assert this byte-identity to validate
     that the trait extension is a true superset of the previous Euler.
  2. On CIR (state-dependent diffusion) the correction fires: the two
     schemes produce measurably non-identical trajectories at the same
     seed, with a per-particle difference whose magnitude matches the
     accumulated Milstein correction.
  3. Both schemes converge to the same limit as `dt -> 0`, so the
     pairwise distance between Euler and Milstein trajectories at the
     same seed and same n_steps decreases monotonically with n_steps.
  4. At fine dt, both schemes match the closed-form mean of CIR within
     MC noise. Catches gross bugs in either scheme.

A genuine strong-order test on CIR using a shared Brownian path is
deferred to the PR that adds the increment hook.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit import (
    simulate_cucker_smale,
    simulate_kuramoto,
    simulate_linear_quadratic,
    simulate_mean_field_cir,
)


# -- Equivalence on constant-diffusion models -------------------------------


def test_euler_milstein_byte_identical_on_linear_quadratic():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(60, 1))
    h_e = simulate_linear_quadratic(
        x0, 1.0, 200, a=-0.5, b=1.0, sigma=0.5, seed=7, scheme="euler"
    )
    h_m = simulate_linear_quadratic(
        x0, 1.0, 200, a=-0.5, b=1.0, sigma=0.5, seed=7, scheme="milstein"
    )
    np.testing.assert_array_equal(h_e, h_m)


def test_euler_milstein_byte_identical_on_cucker_smale():
    rng = np.random.default_rng(0)
    n, d = 40, 2
    x0 = rng.normal(size=(n, 2 * d))
    h_e = simulate_cucker_smale(
        x0,
        1.0,
        100,
        spatial_dim=d,
        beta=0.4,
        sigma=0.1,
        seed=11,
        scheme="euler",
    )
    h_m = simulate_cucker_smale(
        x0,
        1.0,
        100,
        spatial_dim=d,
        beta=0.4,
        sigma=0.1,
        seed=11,
        scheme="milstein",
    )
    np.testing.assert_array_equal(h_e, h_m)


def test_euler_milstein_byte_identical_on_kuramoto():
    rng = np.random.default_rng(0)
    n = 50
    x0 = rng.uniform(0, 2 * np.pi, size=(n, 1))
    omegas = rng.normal(0, 0.5, size=n)
    h_e = simulate_kuramoto(
        x0,
        1.0,
        200,
        coupling_k=1.0,
        omegas=omegas,
        sigma=0.05,
        seed=3,
        scheme="euler",
    )
    h_m = simulate_kuramoto(
        x0,
        1.0,
        200,
        coupling_k=1.0,
        omegas=omegas,
        sigma=0.05,
        seed=3,
        scheme="milstein",
    )
    np.testing.assert_array_equal(h_e, h_m)


# -- CIR shape, determinism, validation, positivity -------------------------


def test_cir_output_shape_full_history():
    n = 50
    x0 = np.full((n, 1), 0.5)
    history = simulate_mean_field_cir(
        x0,
        t_final=1.0,
        n_steps=100,
        kappa=1.0,
        theta=1.0,
        b=0.0,
        sigma=0.6,
        record_every=10,
        seed=1,
    )
    assert history.shape == (11, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_cir_output_shape_terminal_only():
    n = 30
    x0 = np.full((n, 1), 0.5)
    history = simulate_mean_field_cir(
        x0,
        t_final=1.0,
        n_steps=50,
        kappa=1.0,
        theta=1.0,
        b=0.0,
        sigma=0.6,
        seed=1,
    )
    assert history.shape == (2, n, 1)
    np.testing.assert_array_equal(history[0], x0)


def test_cir_determinism():
    n = 30
    x0 = np.full((n, 1), 0.5)
    h1 = simulate_mean_field_cir(
        x0, 1.0, 100, kappa=1.0, theta=1.0, b=0.5, sigma=0.6, seed=42
    )
    h2 = simulate_mean_field_cir(
        x0, 1.0, 100, kappa=1.0, theta=1.0, b=0.5, sigma=0.6, seed=42
    )
    np.testing.assert_array_equal(h1, h2)


def test_cir_different_seeds_differ():
    n = 30
    x0 = np.full((n, 1), 0.5)
    h1 = simulate_mean_field_cir(
        x0, 1.0, 100, kappa=1.0, theta=1.0, b=0.5, sigma=0.6, seed=1
    )
    h2 = simulate_mean_field_cir(
        x0, 1.0, 100, kappa=1.0, theta=1.0, b=0.5, sigma=0.6, seed=2
    )
    assert not np.allclose(h1[-1], h2[-1])


def test_cir_x0_wrong_columns_raises():
    x0 = np.zeros((10, 2))
    with pytest.raises(ValueError, match="columns"):
        simulate_mean_field_cir(
            x0, 1.0, 10, kappa=1.0, theta=1.0, b=0.0, sigma=0.6
        )


def test_cir_invalid_n_steps_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError):
        simulate_mean_field_cir(
            x0, 1.0, 0, kappa=1.0, theta=1.0, b=0.0, sigma=0.6
        )


def test_cir_invalid_kappa_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError):
        simulate_mean_field_cir(
            x0, 1.0, 10, kappa=0.0, theta=1.0, b=0.0, sigma=0.6
        )


def test_cir_invalid_sigma_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError):
        simulate_mean_field_cir(
            x0, 1.0, 10, kappa=1.0, theta=1.0, b=0.0, sigma=-0.1
        )


def test_unknown_scheme_raises():
    x0 = np.zeros((10, 1))
    with pytest.raises(ValueError, match="scheme"):
        simulate_linear_quadratic(
            x0, 1.0, 10, a=0.0, b=0.0, sigma=0.5, scheme="heun"
        )


def test_cir_positivity_with_truncation():
    """Feller condition 2*kappa*theta = 2 >= sigma^2 = 0.36 holds, so the
    truncation should rarely be active. Either way, the positivity guarantee
    of the integrator (with truncation) keeps min(X) effectively at zero."""
    n = 5_000
    x0 = np.full((n, 1), 0.5)
    history = simulate_mean_field_cir(
        x0,
        t_final=2.0,
        n_steps=2_000,
        kappa=1.0,
        theta=1.0,
        b=0.0,
        sigma=0.6,
        seed=0,
        scheme="milstein",
    )
    assert history[-1, :, 0].min() > -1e-10


# -- Milstein on CIR: correctness within the API limits --------------------


def test_schemes_differ_on_cir_at_same_seed():
    """Validates that the Milstein correction term fires on a model with
    state-dependent diffusion. With the same seed, same n_steps, same x0,
    Euler and Milstein on CIR must produce non-identical terminal states.
    The accumulated correction has magnitude
    `0.25 sigma^2 sqrt(2 T dt)` per particle in expectation, which for
    our parameters at n_steps=100 is ~ 1e-2; we assert a loose lower
    bound to catch the case where the correction is silently zeroed."""
    n = 5_000
    x0 = np.full((n, 1), 0.5)
    h_e = simulate_mean_field_cir(
        x0,
        1.0,
        100,
        kappa=1.0,
        theta=1.0,
        b=0.0,
        sigma=0.6,
        seed=42,
        scheme="euler",
    )
    h_m = simulate_mean_field_cir(
        x0,
        1.0,
        100,
        kappa=1.0,
        theta=1.0,
        b=0.0,
        sigma=0.6,
        seed=42,
        scheme="milstein",
    )
    diff = np.abs(h_m[-1, :, 0] - h_e[-1, :, 0])
    assert not np.allclose(h_m, h_e), (
        "Milstein and Euler produced identical trajectories on CIR; "
        "diffusion_derivative may be silently returning zero."
    )
    assert diff.mean() > 1e-3, (
        f"Mean |X_milstein - X_euler| = {diff.mean():.3e}, expected > 1e-3"
    )


def test_path_distance_decreases_with_n_steps():
    """Both Euler and Milstein converge to the same exact solution as
    `dt -> 0`, so the per-particle pathwise distance between the two
    schemes (at the same seed and same n_steps) must decrease as
    n_steps grows. Each (Euler, Milstein) pair shares a noise stream
    because they use the same seed and the same number of integration
    steps, so the comparison is well-defined; the noise streams differ
    *across* n_steps values, but the trend is robust at N = 2000.

    This is a strong-error proxy that doesn't require the deferred
    shared-Brownian-increment hook."""
    n = 2_000
    x0 = np.full((n, 1), 0.5)
    n_steps_list = [50, 100, 200, 400, 800]
    distances = []
    for n_steps in n_steps_list:
        h_e = simulate_mean_field_cir(
            x0,
            1.0,
            n_steps,
            kappa=1.0,
            theta=1.0,
            b=0.0,
            sigma=0.6,
            seed=7,
            scheme="euler",
        )
        h_m = simulate_mean_field_cir(
            x0,
            1.0,
            n_steps,
            kappa=1.0,
            theta=1.0,
            b=0.0,
            sigma=0.6,
            seed=7,
            scheme="milstein",
        )
        rms = np.sqrt(np.mean((h_m[-1, :, 0] - h_e[-1, :, 0]) ** 2))
        distances.append(rms)
    distances = np.array(distances)
    diffs = np.diff(distances)
    assert np.all(diffs < 0), (
        f"pairwise distance not monotonically decreasing in n_steps: "
        f"distances = {distances}, diffs = {diffs}"
    )


def test_cir_first_moment_smoke_both_schemes():
    """At fine dt, both schemes' empirical mean matches the closed-form
    `theta + (X_0 - theta) exp(-kappa T)` within MC noise. Catches gross
    bugs in either scheme."""
    kappa, theta, b, sigma = 1.0, 1.0, 0.0, 0.6
    x_0 = 0.5
    t_final = 1.0
    n_steps = 2_000
    n = 50_000
    x0 = np.full((n, 1), x_0)
    m_T = theta + (x_0 - theta) * np.exp(-kappa * t_final)

    for scheme in ("euler", "milstein"):
        history = simulate_mean_field_cir(
            x0,
            t_final,
            n_steps,
            kappa=kappa,
            theta=theta,
            b=b,
            sigma=sigma,
            seed=42,
            scheme=scheme,
        )
        empirical_mean = history[-1, :, 0].mean()
        rel_err = abs(empirical_mean - m_T) / m_T
        assert rel_err < 0.01, (
            f"{scheme}: |empirical - m_T| / m_T = {rel_err:.4f}, expected < 0.01"
        )
