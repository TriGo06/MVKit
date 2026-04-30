"""Numerical Mean Field Games sub-module.

Currently ships the scalar linear-quadratic case, where the Riccati ODE,
the equilibrium mean trajectory, and the variance trajectory all admit
closed forms. The :func:`solve_lq_mfg` entry point uses a generic Picard
iteration that will extend to non-LQ MFG once a full HJB solver is in
place; the LQ case validates the API on a problem with quantitative
ground truth.

References
----------
Lasry, J.-M. and Lions, P.-L. (2007). *Mean field games*. Japanese Journal
of Mathematics 2, 229-260.

Carmona, R. and Delarue, F. (2018). *Probabilistic Theory of Mean Field
Games with Applications I & II*. Springer.
"""

from __future__ import annotations

from .hjb import HJBSolution, solve_hjb
from .linear_quadratic import (
    LQMFGSolution,
    lq_mfg_analytical_variance,
    solve_lq_mfg,
    solve_lq_mfg_fictitious_play,
)

__all__ = [
    "HJBSolution",
    "LQMFGSolution",
    "lq_mfg_analytical_variance",
    "solve_hjb",
    "solve_lq_mfg",
    "solve_lq_mfg_fictitious_play",
]
