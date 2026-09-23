"""Propagation-of-chaos rate estimation.

A 1D utility that takes a mean-field simulator, runs it at a sequence of
particle counts, computes the Wasserstein-2 distance from each empirical
distribution to a reference law, and fits an empirical convergence rate.
For i.i.d. samples in dimension one with a finite (4 + epsilon)-th moment,
Fournier and Guillin (2015) bound E[W2**2] by O(N**(-1/2)); Jensen's
inequality then gives E[W2] = O(N**(-1/4)). Faster rates need more structure.
Interacting particles additionally require a model-specific coupling bound.

Scope: real-valued 1D samples. Empirical-to-empirical W2 is integrated
exactly over the quantile steps; distance to an inverse-CDF callback uses
adaptive numerical quadrature. These are line distances, not circular
Wasserstein distances for phases.

References
----------
- Sznitman, A.-S. (1991). Topics in propagation of chaos. Ecole d'Ete de
  Probabilites de Saint-Flour XIX. (Original propagation-of-chaos result.)
- Fournier, N. and Guillin, A. (2015). On the rate of convergence in
  Wasserstein distance of the empirical measure. Probability Theory and
  Related Fields 162, 707-738. (Bounds for powers of Wasserstein distance.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad


def _sorted_samples(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("samples must be non-empty and finite")
    return np.sort(arr)


def wasserstein2_to_reference(
    samples: np.ndarray,
    reference_inv_cdf: Callable[[np.ndarray], np.ndarray],
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-8,
) -> float:
    r"""Numerically integrate the 1D Wasserstein-2 quantile formula.

    The empirical quantile is constant on each interval of width 1/n.
    Integrate the squared difference over these full intervals, including
    the reference's tails, instead of replacing it by midpoint samples.
    The reference must have a finite second moment.

    ``epsabs`` and ``epsrel`` control QUADPACK's estimated absolute and
    relative error on W2 squared, not on W2. Failure to reach the requested
    quadrature accuracy raises ``ValueError``. This is numerical quadrature,
    not a symbolic exact integral for an arbitrary inverse CDF.

    ``reference_inv_cdf`` must accept and return equally shaped arrays of
    interior probabilities. For example, pass ``scipy.stats.norm.ppf``.
    """
    if not (np.isfinite(epsabs) and epsabs > 0
            and np.isfinite(epsrel) and epsrel > 0):
        raise ValueError("epsabs and epsrel must be positive and finite")
    sorted_x = _sorted_samples(samples)
    n = sorted_x.size
    indices = np.arange(n, dtype=np.float64)

    def integrand(v):
        # u = (i + v) / n on the i-th quantile interval. Summation
        # before quadrature keeps all inverse-CDF calls vectorized.
        probabilities = (indices + v) / n
        probabilities = np.clip(
            probabilities, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        )
        ref = np.asarray(reference_inv_cdf(probabilities), dtype=np.float64)
        if ref.shape != sorted_x.shape or not np.isfinite(ref).all():
            raise ValueError("reference_inv_cdf must return finite values with the input shape")
        value = float(np.mean((sorted_x - ref) ** 2))
        if not np.isfinite(value):
            raise ValueError("reference has a non-finite squared transport cost")
        return value

    result = quad(integrand, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel,
                  limit=200, full_output=1)
    if len(result) != 3:
        raise ValueError(f"reference quantile integration failed: {result[3]}")
    return float(np.sqrt(max(0.0, result[0])))


def wasserstein2_between_samples(x: np.ndarray, y: np.ndarray) -> float:
    """Exact W2 between equally weighted empirical distributions in 1D.

    For unequal counts, integrate the piecewise-constant empirical
    quantiles over their combined breakpoints. Replicating every atom
    preserves the measure and therefore preserves this distance.
    """
    x_sorted = _sorted_samples(x)
    y_sorted = _sorted_samples(y)
    n, m = x_sorted.size, y_sorted.size
    if n == m:
        return float(np.sqrt(np.mean((x_sorted - y_sorted) ** 2)))
    x_breaks = np.arange(n + 1, dtype=np.float64) / n
    y_breaks = np.arange(m + 1, dtype=np.float64) / m
    breaks = np.union1d(x_breaks, y_breaks)
    midpoints = (breaks[:-1] + breaks[1:]) / 2.0
    x_idx = np.searchsorted(x_breaks[1:], midpoints, side="left")
    y_idx = np.searchsorted(y_breaks[1:], midpoints, side="left")
    cost = np.sum(np.diff(breaks) * (x_sorted[x_idx] - y_sorted[y_idx]) ** 2)
    return float(np.sqrt(cost))


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
        This is an observed slope, not a universal theoretical rate.
        Sampling, interaction, and time discretization can all affect it.
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
    progress: bool = False,
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
    progress : bool, default False
        If True and ``tqdm`` is installed, wrap the
        ``len(n_values) * n_seeds`` simulations in a progress bar; if
        ``tqdm`` is missing the call still runs and prints one
        informational line. Default ``False`` is silent and incurs
        no overhead.

    Returns
    -------
    result : PropagationOfChaosResult
    """
    from ._progress import progress_iter

    n_array = np.asarray(list(n_values), dtype=np.int64)
    if n_array.ndim != 1 or n_array.size < 2:
        raise ValueError("n_values must be 1D with at least 2 entries")
    if (n_array <= 0).any():
        raise ValueError("n_values must be strictly positive")
    if n_seeds < 1:
        raise ValueError("n_seeds must be >= 1")

    k = n_array.size
    w2 = np.empty((k, n_seeds), dtype=np.float64)
    pairs = [(i, n, s) for i, n in enumerate(n_array) for s in range(n_seeds)]
    iterator = progress_iter(
        pairs,
        total=len(pairs),
        description=f"poc rate sweep (N x seeds = {k} x {n_seeds})",
        enabled=progress,
    )
    for i, n, s in iterator:
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

    if np.unique(n_array).size < 2:
        raise ValueError("n_values must include at least two distinct counts")
    if not np.isfinite(median).all() or (median <= 0).any():
        raise ValueError("a log-log rate requires positive finite median distances")
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
