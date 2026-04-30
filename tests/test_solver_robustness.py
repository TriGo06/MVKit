"""Robustness tests for the grid-based MFG solvers.

The HJB and FP discretizations are first-order schemes with conditional
stability: each has a CFL-style requirement on the time step. Without
guards, a CFL violation propagates as silent NaN values through the
solver output. These tests confirm that the solvers detect non-finite
values after each time step and raise a descriptive ``ValueError``
instead, including the diagnostic information needed to fix the call
(observed CFL ratio, suggested ``n_t``).

The tests also confirm that ``solve_mfg`` enriches the inner-solver
error with outer-iteration context, so a user debugging a slow
non-convergent problem sees the iteration index where the failure
occurred.
"""

from __future__ import annotations

import numpy as np
import pytest

from mvkit.mfg import (
    MFGProblem,
    solve_fokker_planck,
    solve_hjb,
    solve_mfg,
)


def _periodic_grid(n_x: int, length: float = 2.0 * np.pi) -> np.ndarray:
    return np.linspace(0.0, length, n_x, endpoint=False)


# ---------- Fokker-Planck CFL diagnostic ----------


def test_fp_raises_on_cfl_violation_with_diagnostic():
    """Pick parameters that violate ``dt <= dx / max|drift|`` by a wide
    margin and confirm the FP solver raises with a message that names
    the CFL ratio and a recommended ``n_t``."""
    n_x = 64
    L = 2.0 * np.pi
    x = _periodic_grid(n_x, L)
    c = 50.0  # CFL ratio at n_t=4: 50 * 0.25 / 0.098 ~ 128
    n_t = 4
    with pytest.raises(ValueError, match="CFL") as exc_info:
        solve_fokker_planck(
            sigma=0.1, T=1.0, x_grid=x, n_t=n_t,
            initial=np.full(n_x, 1.0 / L),
            drift=lambda t, xx: np.full_like(xx, c),
        )
    msg = str(exc_info.value)
    assert "max|drift|" in msg
    assert "Try n_t" in msg
    assert "CFL = 1.27" in msg or "CFL = 1.28" in msg or "CFL = 1.2" in msg


def test_fp_reasonable_drift_with_adequate_n_t_succeeds():
    """The same drift solved with adequate ``n_t`` should run cleanly,
    confirming the CFL guard does not raise false positives."""
    n_x = 64
    L = 2.0 * np.pi
    x = _periodic_grid(n_x, L)
    c = 50.0
    # CFL ratio = c * dt / dx = c * (T/n_t) * (n_x/L). Need n_t > c * T * n_x / L.
    # = 50 * 1 * 64 / (2 pi) ~ 510. Use 1024 for margin.
    sol = solve_fokker_planck(
        sigma=0.1, T=1.0, x_grid=x, n_t=1024,
        initial=np.full(n_x, 1.0 / L),
        drift=lambda t, xx: np.full_like(xx, c),
    )
    assert np.isfinite(sol.m).all()


# ---------- HJB instability diagnostic ----------


def test_hjb_raises_on_periodic_wrap_blowup():
    """Asymmetric quadratic terminal data on a periodic grid creates a
    large gradient at the wrap-around. With too few time steps the
    explicit Hamiltonian becomes unstable; the solver should raise
    with a helpful message about CFL."""
    n_x = 32
    x = np.linspace(-6.0, 6.0, n_x, endpoint=False)
    # Quadratic terminal centered off-axis: g(x) = 0.5 * 5 * (x - 3)^2.
    # The wrap-around creates a large gradient at the boundary which
    # the EO upwind amplifies.
    terminal = 0.5 * 5.0 * (x - 3.0) ** 2
    with pytest.raises(ValueError, match="CFL"):
        solve_hjb(
            sigma=0.1, T=2.0, x_grid=x, n_t=3,
            terminal=terminal,
            running_cost=lambda t, xx: np.zeros_like(xx),
        )


# ---------- solve_mfg propagates inner-solver errors with context ----------


def test_solve_mfg_wraps_inner_cfl_error_with_outer_context():
    """If an inner solver fails inside ``solve_mfg``, the user must see
    both the inner CFL diagnostic AND the outer iteration index."""
    n_x = 32
    a, b = -np.pi, np.pi

    def initial_density(x):
        return (1.0 + 0.4 * np.cos(x)) / (b - a)

    # A running cost steep enough that the optimal drift, evaluated at
    # the first HJB step, is large; combined with a tiny n_t this
    # violates an inner CFL on the first outer iteration.
    def running_cost(t, x, m):
        return 50.0 * x ** 2

    def terminal_cost(x, m):
        return np.zeros_like(x)

    problem = MFGProblem(
        sigma=0.1, T=1.0, domain=(a, b), n_x=n_x,
        initial_density=initial_density,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
    )
    with pytest.raises(ValueError) as exc_info:
        solve_mfg(problem, n_t=3, method="picard", n_iterations_max=5)
    msg = str(exc_info.value)
    assert "outer iteration" in msg
    # The chained inner exception is preserved so the user can see it.
    assert exc_info.value.__cause__ is not None
    assert "CFL" in str(exc_info.value.__cause__)
