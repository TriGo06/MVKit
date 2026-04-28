"""Propagation-of-chaos rate estimation.

A 1D utility that takes a mean-field simulator, runs it at a sequence of
particle counts, computes the Wasserstein-2 distance from each empirical
distribution to a reference law, and fits the convergence rate. Validates
numerically against the Fournier and Guillin (2015) result, which gives
``E[W_2(emp, mu)] = O(N^{-1/2})`` in dimension 1 for laws with finite
``(4 + epsilon)``-th moment.

Scope: 1D models only (LinearQuadratic, Kuramoto via the order parameter,
MeanFieldCIR). The exact 1D Wasserstein-2 distance via sorted samples is
fast enough in pure numpy that we do not need a C dependency. Higher-d
support via sliced Wasserstein is a v0.2 item.

References
----------
- Sznitman, A.-S. (1991). Topics in propagation of chaos. Ecole d'Ete de
  Probabilites de Saint-Flour XIX. (Original propagation-of-chaos result.)
- Fournier, N. and Guillin, A. (2015). On the rate of convergence in
  Wasserstein distance of the empirical measure. Probability Theory and
  Related Fields 162, 707-738. (The N^{-1/2} rate in 1D.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def wasserstein2_to_reference(
    samples: np.ndarray,
    reference_inv_cdf: Callable[[np.ndarray], np.ndarray],
) -> float:
    r"""Exact 1D Wasserstein-2 distance from an empirical distribution to a
    continuous reference law specified by its inverse CDF.

    Uses the mid-quantile evaluation
    :math:`W_2^2 \approx (1/n) \sum_i (x_{(i)} - F^{-1}((i + 0.5) / n))^2`
    where :math:`x_{(i)}` is the i-th order statistic.

    Parameters
    ----------
    samples : ndarray, shape (n,)
        Empirical sample.
    reference_inv_cdf : callable
        Function mapping an array of quantiles in (0, 1) to the
        corresponding values of the reference inverse CDF. For example
        ``scipy.stats.norm(loc=mu, scale=sigma).ppf``.
    """
    arr = np.asarray(samples, dtype=np.float64).ravel()
    n = arr.size
    if n == 0:
        raise ValueError("samples must be non-empty")
    sorted_x = np.sort(arr)
    quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n
    ref = np.asarray(reference_inv_cdf(quantiles), dtype=np.float64)
    if ref.shape != sorted_x.shape:
        raise ValueError(
            "reference_inv_cdf must return an array of the same length "
            f"as samples; got {ref.shape} for n = {n}"
        )
    return float(np.sqrt(np.mean((sorted_x - ref) ** 2)))


def _empirical_inv_cdf(samples_sorted: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear interpolation of the empirical inverse CDF at quantiles ``q``.

    Anchored at the mid-quantile points ``(i + 0.5) / n`` of the sorted
    sample, with values clipped to the sample range outside that grid.
    """
    n = samples_sorted.size
    sample_grid = (np.arange(n, dtype=np.float64) + 0.5) / n
    return np.interp(q, sample_grid, samples_sorted)


def wasserstein2_between_samples(x: np.ndarray, y: np.ndarray) -> float:
    """Exact 1D Wasserstein-2 distance between two empirical distributions.

    Equal-size case: sort both samples and average the squared difference of
    order statistics. Unequal-size case: evaluate both empirical inverse
    CDFs at the common mid-quantile grid of size ``max(n, m)`` via linear
    interpolation, then average squared differences.
    """
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    n, m = xa.size, ya.size
    if n == 0 or m == 0:
        raise ValueError("samples must be non-empty")
    x_sorted = np.sort(xa)
    y_sorted = np.sort(ya)
    if n == m:
        return float(np.sqrt(np.mean((x_sorted - y_sorted) ** 2)))
    common_n = max(n, m)
    grid = (np.arange(common_n, dtype=np.float64) + 0.5) / common_n
    x_q = _empirical_inv_cdf(x_sorted, grid)
    y_q = _empirical_inv_cdf(y_sorted, grid)
    return float(np.sqrt(np.mean((x_q - y_q) ** 2)))


@dataclass
class PropagationOfChaosResult:
    """Output of :func:`estimate_propagation_of_chaos_rate`.

    Attributes
    ----------
    n_values : ndarray, shape (K,)
        The particle counts swept.
    w2_median : ndarray, shape (K,)
        Median over seeds of the empirical-to-reference W2 at each N.
    w2_q25, w2_q75 : ndarray, shape (K,)
        25th and 75th percentiles over seeds (interquartile range).
    fitted_slope : float
        Least-squares slope of ``log(w2_median)`` against ``log(n_values)``.
        Expected to be close to ``-0.5`` in dimension 1 under the
        Fournier-Guillin moment condition.
    fitted_intercept : float
        Intercept of the same fit, with ``w2 ~ exp(intercept) * N**slope``.
    """

    n_values: np.ndarray
    w2_median: np.ndarray
    w2_q25: np.ndarray
    w2_q75: np.ndarray
    fitted_slope: float
    fitted_intercept: float


def estimate_propagation_of_chaos_rate(
    simulator: Callable[[int, int], np.ndarray],
    reference_inv_cdf: Callable[[np.ndarray], np.ndarray],
    n_values: Sequence[int],
    n_seeds: int = 10,
) -> PropagationOfChaosResult:
    """Sweep particle count, compute W2 to a reference law per seed, and
    fit the log-log convergence rate.

    Parameters
    ----------
    simulator : callable
        Signature ``(n_particles: int, seed: int) -> ndarray``. Must
        return a 1D array of terminal-time particle states. Wrapping any
        of the ``mvkit.simulate_*`` functions into this shape is the
        caller's responsibility.
    reference_inv_cdf : callable
        Inverse CDF of the McKean-Vlasov limit law at the same time. Must
        accept an array of quantiles in (0, 1).
    n_values : sequence of int
        Particle counts to sweep. Should span at least one order of
        magnitude for the slope fit to be meaningful.
    n_seeds : int, default 10
        Number of independent seeds at each ``N``. The W2 estimator is
        noisy at fixed ``N``; averaging the median across seeds (rather
        than the mean) makes the fit robust to occasional outlier sims.

    Returns
    -------
    result : PropagationOfChaosResult
    """
    n_array = np.asarray(list(n_values), dtype=np.int64)
    if n_array.ndim != 1 or n_array.size < 2:
        raise ValueError("n_values must be 1D with at least 2 entries")
    if (n_array <= 0).any():
        raise ValueError("n_values must be strictly positive")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    k = n_array.size
    w2 = np.empty((k, n_seeds), dtype=np.float64)
    for i, n in enumerate(n_array):
        for s in range(n_seeds):
            samples = np.asarray(simulator(int(n), s)).ravel()
            if samples.size != n:
                raise ValueError(
                    f"simulator returned {samples.size} samples for "
                    f"n_particles = {n}"
                )
            w2[i, s] = wasserstein2_to_reference(samples, reference_inv_cdf)

    median = np.median(w2, axis=1)
    q25 = np.quantile(w2, 0.25, axis=1)
    q75 = np.quantile(w2, 0.75, axis=1)

    log_n = np.log(n_array.astype(np.float64))
    log_w = np.log(median)
    slope, intercept = np.polyfit(log_n, log_w, 1)

    return PropagationOfChaosResult(
        n_values=n_array,
        w2_median=median,
        w2_q25=q25,
        w2_q75=q75,
        fitted_slope=float(slope),
        fitted_intercept=float(intercept),
    )
