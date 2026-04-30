"""Brownian-increment utilities for strong-error tests and shared paths.

The :mod:`mvkit` integrators accept an optional ``increments`` array of
shape ``(n_steps, N, dim)`` containing standard normals :math:`Z_n`. When
supplied, the integrator uses them in place of internal sampling, with
the same scaling ``sigma * sqrt(dt) * Z_n`` as the seed-driven path.

This module provides two helpers:

- :func:`generate_increments` for drawing a fresh array with a fixed RNG.
- :func:`coarsen` for aggregating a fine increment array onto a coarser
  time grid via the variance-preserving bridge relation, so that two
  simulations at different ``n_steps`` are driven by the same Brownian
  path. This is the standard construction for measuring strong order:

.. math::
    Z^{\\text{coarse}}_k
    = \\frac{1}{\\sqrt{f}} \\sum_{j=0}^{f-1} Z^{\\text{fine}}_{f k + j}.

References
----------
Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic
Differential Equations*. Springer. Chapter 9 for strong-order theory.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def generate_increments(
    n_steps: int,
    n_particles: int,
    dim: int = 1,
    seed: Union[int, np.random.Generator] = 42,
) -> np.ndarray:
    """Sample standard-normal increments of shape ``(n_steps, N, dim)``.

    Parameters
    ----------
    n_steps : int
        Number of time steps. Must be positive.
    n_particles : int
        Number of particles. Must be positive.
    dim : int, default 1
        Noise dimension per particle. Equals the state dimension on every
        built-in mvkit model: 1 for ``LinearQuadratic``, ``Kuramoto``, and
        ``MeanFieldCIR``; ``2 * spatial_dim`` for ``CuckerSmale``.
    seed : int or numpy.random.Generator, default 42
        RNG seed (or a pre-built ``Generator``). Identical seeds give
        bit-exact identical arrays.

    Returns
    -------
    Z : ndarray, shape (n_steps, n_particles, dim)
        Standard-normal samples, ready to be passed as ``increments=`` to
        any ``simulate_*`` function.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    rng = (
        seed
        if isinstance(seed, np.random.Generator)
        else np.random.default_rng(int(seed))
    )
    return rng.standard_normal(
        size=(int(n_steps), int(n_particles), int(dim))
    ).astype(np.float64, copy=False)


def coarsen(
    fine_increments: np.ndarray,
    factor: int = 2,
) -> np.ndarray:
    r"""Aggregate fine increments onto a coarse grid via the bridge relation.

    Given fine standard-normal increments ``Z_fine`` driving ``n_fine``
    Euler steps of size ``dt_fine``, the coarse increments driving
    ``n_coarse = n_fine // factor`` steps of size ``dt_coarse = factor *
    dt_fine`` on the **same Brownian path** are

    .. math::
        Z^{\text{coarse}}_k
        = \frac{1}{\sqrt{f}} \sum_{j=0}^{f-1} Z^{\text{fine}}_{f k + j}.

    Both samples produce the same :math:`\sigma\, W(t_k)` at every coarse
    grid point, which is what makes pathwise (strong) error estimation
    well-defined. The factor of ``1/sqrt(factor)`` rescales the sum back
    to a unit-variance standard normal, matching mvkit's convention where
    the integrator multiplies ``Z`` by ``sqrt(dt)``.

    Parameters
    ----------
    fine_increments : ndarray, shape (n_fine, N, dim)
        Fine-grid standard normals (e.g. from :func:`generate_increments`).
    factor : int, default 2
        Coarsening factor. Must divide ``n_fine``.

    Returns
    -------
    Z_coarse : ndarray, shape (n_fine // factor, N, dim)
        Aggregated standard normals on the coarse grid.

    Notes
    -----
    The ``factor=2`` case is the standard "halving" used for log-log fits
    of the strong-error rate.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    arr = np.asarray(fine_increments, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"fine_increments must be 3D (n_fine, N, dim), got shape {arr.shape}"
        )
    n_fine = arr.shape[0]
    if n_fine % factor != 0:
        raise ValueError(
            f"factor={factor} does not divide n_fine={n_fine}"
        )
    if factor == 1:
        return arr.copy()
    n_coarse = n_fine // factor
    n, d = arr.shape[1], arr.shape[2]
    reshaped = arr.reshape(n_coarse, factor, n, d)
    return reshaped.sum(axis=1) / np.sqrt(float(factor))
