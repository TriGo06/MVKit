"""Tests for the Brownian-increment hook on the integrators.

Two layers:

1. The :mod:`mvkit.brownian` helpers (``generate_increments``, ``coarsen``)
   are tested for shape, determinism, variance preservation, and the
   bridge-relation correctness (partial sums on the coarse grid agree
   with the fine path at the coarse times).
2. The integrator ``increments=`` parameter is tested for backward
   compatibility (default unchanged), seed-irrelevance (when increments
   are supplied, ``seed`` does not affect output), shape validation,
   and most importantly the **strong-order rates** measured via
   coarse-vs-fine comparison on a shared Brownian path:

   - Euler-Maruyama on multiplicative-noise CIR: slope around 0.5.
   - Milstein on CIR: slope around 1.0, and noticeably smaller than
     Euler at the smallest grids.
   - Euler-Maruyama on additive-noise LQ: slope around 1.0 (the
     :math:`\\sigma\\sqrt{\\Delta t} Z` term is exact, so the
     strong-order-1/2 lower bound for multiplicative noise does not
     apply).

References
----------
Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
Differential Equations*. Springer.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit import simulate_linear_quadratic, simulate_mean_field_cir
from mvkit.brownian import coarsen, generate_increments


# ---------- Brownian helpers ----------


def test_generate_increments_shape_and_dtype():
    Z = generate_increments(n_steps=10, n_particles=7, dim=3, seed=0)
    assert Z.shape == (10, 7, 3)
    assert Z.dtype == np.float64


def test_generate_increments_is_deterministic_given_seed():
    Z1 = generate_increments(n_steps=50, n_particles=20, dim=2, seed=123)
    Z2 = generate_increments(n_steps=50, n_particles=20, dim=2, seed=123)
    np.testing.assert_array_equal(Z1, Z2)


def test_generate_increments_accepts_explicit_generator():
    rng = np.random.default_rng(7)
    Z1 = generate_increments(n_steps=10, n_particles=5, dim=1, seed=rng)
    rng2 = np.random.default_rng(7)
    Z2 = generate_increments(n_steps=10, n_particles=5, dim=1, seed=rng2)
    np.testing.assert_array_equal(Z1, Z2)


def test_generate_increments_invalid_args():
    with pytest.raises(ValueError):
        generate_increments(n_steps=0, n_particles=5)
    with pytest.raises(ValueError):
        generate_increments(n_steps=10, n_particles=0)
    with pytest.raises(ValueError):
        generate_increments(n_steps=10, n_particles=5, dim=0)


def test_coarsen_shape_and_variance():
    Z_fine = generate_increments(n_steps=1024, n_particles=2000, dim=1, seed=0)
    Z_coarse = coarsen(Z_fine, factor=2)
    assert Z_coarse.shape == (512, 2000, 1)
    # Variance is preserved (large-sample MC estimate).
    assert abs(Z_coarse.var() - 1.0) < 0.05


def test_coarsen_factor_one_returns_copy():
    Z = generate_increments(n_steps=8, n_particles=4, dim=1, seed=0)
    Zc = coarsen(Z, factor=1)
    np.testing.assert_array_equal(Z, Zc)
    assert Zc is not Z, "coarsen(factor=1) should return a copy"


def test_coarsen_invalid_args():
    Z = generate_increments(n_steps=10, n_particles=2, seed=0)
    with pytest.raises(ValueError):
        coarsen(Z, factor=3)  # 3 does not divide 10
    with pytest.raises(ValueError):
        coarsen(Z, factor=0)
    with pytest.raises(ValueError):
        coarsen(np.zeros((10, 2)), factor=2)  # not 3D


def test_coarsen_preserves_brownian_path_at_coarse_times():
    """The bridge relation guarantees that the discretized Brownian path
    ``W(t_k) = sqrt(dt) * cumsum(Z)[k]`` evaluated on the coarse grid
    matches what the fine grid gives at the same times. This is the
    property that makes pathwise (strong) error well-defined.
    """
    n_fine, factor = 64, 4
    n_coarse = n_fine // factor
    T = 1.0
    dt_fine = T / n_fine
    dt_coarse = T / n_coarse

    Z_fine = generate_increments(n_steps=n_fine, n_particles=8, dim=1, seed=42)
    Z_coarse = coarsen(Z_fine, factor=factor)

    cum_fine = np.sqrt(dt_fine) * np.cumsum(Z_fine, axis=0)
    cum_coarse = np.sqrt(dt_coarse) * np.cumsum(Z_coarse, axis=0)

    # Sample fine cumsum at coarse times t_k = k * factor * dt_fine.
    coarse_indices = np.arange(1, n_coarse + 1) * factor - 1
    sampled_fine = cum_fine[coarse_indices]
    np.testing.assert_allclose(sampled_fine, cum_coarse, atol=1e-12, rtol=0.0)


# ---------- Integrator hook: backward compat, determinism, validation ----------


def _lq_default_kwargs():
    return dict(
        t_final=1.0,
        n_steps=200,
        a=-0.5,
        b=1.0,
        sigma=0.5,
        record_every=20,
        seed=42,
    )


def test_default_path_unchanged_when_increments_none():
    """The seed-driven internal sampling must be byte-identical to the
    pre-hook behavior (which is what every existing test relies on)."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 1))
    h_no_kw = simulate_linear_quadratic(x0, **_lq_default_kwargs())
    h_kw = simulate_linear_quadratic(x0, increments=None, **_lq_default_kwargs())
    np.testing.assert_array_equal(h_no_kw, h_kw)


def test_increments_make_seed_irrelevant():
    """When the user supplies their own ``increments``, the integrator
    must use them in full. ``seed`` is then irrelevant."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 1))
    Z = generate_increments(n_steps=200, n_particles=50, dim=1, seed=7)

    kwargs = _lq_default_kwargs()
    kwargs["seed"] = 1
    h1 = simulate_linear_quadratic(x0, increments=Z, **kwargs)
    kwargs["seed"] = 1234567
    h2 = simulate_linear_quadratic(x0, increments=Z, **kwargs)
    np.testing.assert_array_equal(h1, h2)


def test_same_increments_same_trajectory_milstein():
    """Same property on the Milstein integrator (state-dependent diffusion
    path)."""
    rng = np.random.default_rng(0)
    x0 = rng.uniform(0.01, 0.10, size=(50, 1))
    Z = generate_increments(n_steps=200, n_particles=50, dim=1, seed=11)
    kwargs = dict(t_final=1.0, n_steps=200, kappa=1.0, theta=0.04,
                  b=0.0, sigma=0.2, scheme="milstein")
    h1 = simulate_mean_field_cir(x0, seed=1, increments=Z, **kwargs)
    h2 = simulate_mean_field_cir(x0, seed=999, increments=Z, **kwargs)
    np.testing.assert_array_equal(h1, h2)


def test_increments_shape_validation_lq():
    x0 = np.zeros((5, 1))
    bad_shape = np.zeros((10, 5, 2))  # last dim must be 1 for LQ
    with pytest.raises(ValueError, match="increments"):
        simulate_linear_quadratic(
            x0, t_final=1.0, n_steps=10, a=-0.5, b=1.0, sigma=0.5,
            increments=bad_shape,
        )
    wrong_n_steps = np.zeros((9, 5, 1))
    with pytest.raises(ValueError, match="increments"):
        simulate_linear_quadratic(
            x0, t_final=1.0, n_steps=10, a=-0.5, b=1.0, sigma=0.5,
            increments=wrong_n_steps,
        )


def test_increments_shape_validation_cucker_smale():
    from mvkit import simulate_cucker_smale

    rng = np.random.default_rng(0)
    n, d = 4, 2
    x0 = rng.normal(size=(n, 2 * d))
    # Last dim must be 2 * spatial_dim = 4
    bad = np.zeros((10, n, 1))
    with pytest.raises(ValueError, match="increments"):
        simulate_cucker_smale(
            x0, t_final=1.0, n_steps=10, spatial_dim=d, increments=bad,
        )


# ---------- Strong-order convergence ----------


def _strong_error_slope(simulate_fn, x0, t_final, n_steps_grid,
                        n_fine_ref, scheme, model_kwargs, seed):
    """Compute the strong-error slope by coarsening a fine Brownian path
    onto each coarse grid and comparing terminal-time states.

    Returns (errors, dts, slope).
    """
    n_particles = x0.shape[0]
    dim = x0.shape[1]
    Z_finest = generate_increments(
        n_steps=n_fine_ref, n_particles=n_particles, dim=dim, seed=seed
    )
    ref_hist = simulate_fn(
        x0, t_final=t_final, n_steps=n_fine_ref,
        increments=Z_finest, scheme=scheme, **model_kwargs,
    )
    X_ref = ref_hist[-1, :, 0]

    errs, dts = [], []
    for n_steps in n_steps_grid:
        factor = n_fine_ref // n_steps
        Z_n = coarsen(Z_finest, factor=factor)
        h = simulate_fn(
            x0, t_final=t_final, n_steps=n_steps,
            increments=Z_n, scheme=scheme, **model_kwargs,
        )
        err = np.sqrt(np.mean((h[-1, :, 0] - X_ref) ** 2))
        errs.append(err)
        dts.append(t_final / n_steps)

    slope, _ = np.polyfit(np.log(dts), np.log(errs), 1)
    return np.array(errs), np.array(dts), float(slope)


def test_strong_error_euler_on_cir_slope_one_half():
    """Multiplicative (state-dependent) noise: Euler-Maruyama strong
    order 1/2. The fitted log-log slope should sit in [0.3, 0.7]."""
    N, T = 2000, 1.0
    x0 = np.full((N, 1), 0.04)
    errs, _, slope = _strong_error_slope(
        simulate_fn=simulate_mean_field_cir,
        x0=x0,
        t_final=T,
        n_steps_grid=[32, 64, 128, 256, 512, 1024],
        n_fine_ref=4096,
        scheme="euler",
        model_kwargs=dict(kappa=1.0, theta=0.04, b=0.0, sigma=0.2),
        seed=0,
    )
    assert 0.3 < slope < 0.7, f"Euler/CIR slope {slope:.3f} outside [0.3, 0.7]"


def test_strong_error_milstein_on_cir_slope_one():
    """Milstein on multiplicative-noise CIR: strong order 1. Slope in
    [0.7, 1.3]; also assert it is markedly smaller than Euler at the
    smallest dt to confirm the scheme actually dominates the noise."""
    N, T = 2000, 1.0
    x0 = np.full((N, 1), 0.04)
    errs_m, _, slope_m = _strong_error_slope(
        simulate_fn=simulate_mean_field_cir,
        x0=x0,
        t_final=T,
        n_steps_grid=[32, 64, 128, 256, 512, 1024],
        n_fine_ref=4096,
        scheme="milstein",
        model_kwargs=dict(kappa=1.0, theta=0.04, b=0.0, sigma=0.2),
        seed=0,
    )
    assert 0.7 < slope_m < 1.3, (
        f"Milstein/CIR slope {slope_m:.3f} outside [0.7, 1.3]"
    )
    errs_e, _, _ = _strong_error_slope(
        simulate_fn=simulate_mean_field_cir,
        x0=x0,
        t_final=T,
        n_steps_grid=[32, 64, 128, 256, 512, 1024],
        n_fine_ref=4096,
        scheme="euler",
        model_kwargs=dict(kappa=1.0, theta=0.04, b=0.0, sigma=0.2),
        seed=0,
    )
    # At the finest dt of the test (n_steps=1024), Milstein error should be
    # at least 5x smaller than Euler. Empirically ~25x in our environment.
    assert errs_m[-1] < 0.2 * errs_e[-1], (
        f"Milstein {errs_m[-1]:.6f} not enough below Euler {errs_e[-1]:.6f}"
    )


def test_strong_error_euler_on_lq_slope_one_for_additive_noise():
    """Linear-quadratic has constant (additive) diffusion, so the
    Euler-Maruyama noise term ``sigma sqrt(dt) Z`` is exact and the
    strong-order-1/2 lower bound for multiplicative noise does not apply.
    The drift discretization error then dominates and the slope is 1."""
    N, T = 2000, 1.0
    x0 = np.zeros((N, 1))
    _, _, slope = _strong_error_slope(
        simulate_fn=simulate_linear_quadratic,
        x0=x0,
        t_final=T,
        n_steps_grid=[64, 128, 256, 512, 1024, 2048],
        n_fine_ref=4096,
        scheme="euler",
        model_kwargs=dict(a=-0.5, b=1.0, sigma=0.5),
        seed=0,
    )
    assert 0.7 < slope < 1.3, f"LQ Euler slope {slope:.3f} outside [0.7, 1.3]"
