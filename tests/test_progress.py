"""Smoke tests for the optional progress-bar plumbing.

The long-running solvers expose a ``progress: bool = False`` keyword
that wraps the outer iteration in a :mod:`tqdm.auto.tqdm` when
``tqdm`` is installed and prints one informational line otherwise.
``progress=False`` (the default) is fully silent.

These tests verify:

- ``progress_iter`` is a no-op iterator when ``enabled=False``;
- ``progress_iter`` returns the original iterable plus prints a
  fallback note when ``enabled=True`` and ``tqdm`` is unavailable
  (we simulate this by patching the import path);
- each progress-aware solver runs to completion with ``progress=True``
  and produces output identical to ``progress=False`` (the bar is
  cosmetic, not load-bearing).
"""

from __future__ import annotations

import builtins
import io
import sys
from contextlib import redirect_stderr

import numpy as np
import pytest

from mvkit._progress import progress_iter


def test_progress_iter_disabled_returns_original():
    src = [1, 2, 3]
    out = progress_iter(src, total=3, description="test", enabled=False)
    assert out is src


def test_progress_iter_enabled_wraps_when_tqdm_available():
    pytest.importorskip("tqdm", reason="tqdm not installed")
    out = progress_iter(range(5), total=5, description="test", enabled=True)
    assert list(out) == [0, 1, 2, 3, 4]


def test_progress_iter_falls_back_when_tqdm_missing(monkeypatch):
    """If ``tqdm`` cannot be imported, ``progress_iter(..., enabled=True)``
    must still run (returning the unwrapped iterable) and print a
    one-line note to stderr."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("simulated missing tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Drop any cached tqdm module too so the fake_import is consulted.
    monkeypatch.delitem(sys.modules, "tqdm.auto", raising=False)
    monkeypatch.delitem(sys.modules, "tqdm", raising=False)

    buf = io.StringIO()
    src = [10, 20, 30]
    with redirect_stderr(buf):
        out = list(progress_iter(src, total=3, description="x", enabled=True))
    assert out == src
    assert "tqdm" in buf.getvalue()


# ---------- Smoke tests on the actual solvers ----------


def test_solve_lq_mfg_progress_does_not_crash():
    from mvkit.mfg import solve_lq_mfg

    sol = solve_lq_mfg(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0,
        n_particles=200, n_grid=50, tol=1e-3, seed=0,
        progress=True,
    )
    assert sol.converged or sol.n_iterations > 0


def test_solve_lq_mfg_with_and_without_progress_match():
    """The progress bar is purely cosmetic; outputs must be identical."""
    from mvkit.mfg import solve_lq_mfg

    kwargs = dict(
        q=1.0, q_T=0.5, sigma=0.5, T=1.0,
        mu_0_mean=0.0, mu_0_var=1.0,
        n_particles=200, n_grid=50, tol=1e-3, seed=0,
    )
    a = solve_lq_mfg(**kwargs, progress=False)
    b = solve_lq_mfg(**kwargs, progress=True)
    np.testing.assert_array_equal(a.m, b.m)
    np.testing.assert_array_equal(a.x_trajectory, b.x_trajectory)
    assert a.n_iterations == b.n_iterations


def test_solve_mfg_progress_does_not_crash():
    from mvkit.mfg import MFGProblem, solve_mfg

    a, b = -3.0, 3.0
    n_x = 32
    L = b - a
    dx = L / n_x
    x_full = np.linspace(a, b, n_x, endpoint=False)

    def initial_density(x):
        return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)

    def x_mean(m):
        return float(np.sum(x_full * m) * dx)

    def running_cost(t, x, m):
        return 0.5 * (x - x_mean(m)) ** 2

    def terminal_cost(x, m):
        return 0.5 * 0.5 * (x - x_mean(m)) ** 2

    problem = MFGProblem(
        sigma=0.5, T=1.0, domain=(a, b), n_x=n_x,
        initial_density=initial_density,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
    )
    sol = solve_mfg(problem, n_t=40, method="picard", tol=1e-3, progress=True)
    assert sol.converged


def test_estimate_poc_rate_progress_does_not_crash():
    from scipy.stats import norm

    from mvkit import simulate_linear_quadratic
    from mvkit.poc import estimate_propagation_of_chaos_rate

    a, b, sigma, T = -0.5, 1.0, 0.5, 1.0
    m0, v0 = 0.0, 1.0
    m_T = m0 * np.exp((a + b) * T)
    v_T = v0 * np.exp(2 * a * T) + sigma ** 2 * (
        np.exp(2 * a * T) - 1
    ) / (2 * a)

    def simulator(n, seed):
        rng = np.random.default_rng(seed)
        x0 = rng.normal(m0, np.sqrt(v0), size=(n, 1))
        h = simulate_linear_quadratic(x0, T, 200, a=a, b=b, sigma=sigma, seed=seed)
        return h[-1, :, 0]

    result = estimate_propagation_of_chaos_rate(
        simulator=simulator,
        reference_inv_cdf=norm(loc=m_T, scale=np.sqrt(v_T)).ppf,
        n_values=[100, 300, 1000],
        n_seeds=2,
        progress=True,
    )
    assert result.fitted_slope < 0  # negative slope on a converging sweep
