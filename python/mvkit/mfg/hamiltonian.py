"""Convex Hamiltonians for the backward HJB solver.

The 1D HJB solver :func:`mvkit.mfg.solve_hjb` integrates

.. math::
    \\partial_t u + \\tfrac{1}{2}\\sigma^2 \\partial_{xx} u
    - H(\\partial_x u) + F(x, m_t) = 0,
    \\qquad u(T, x) = g(x),

for a convex Hamiltonian :math:`H`, with optimal feedback control
:math:`\\alpha^*(t, x) = -H'(\\partial_x u)`. The quadratic case
:math:`H(p) = p^2/2` recovers the familiar
:math:`\\alpha^* = -\\partial_x u`.

A :class:`Hamiltonian` bundles :math:`H` and :math:`H'` as vectorized
callables. The Engquist-Osher numerical Hamiltonian used by the solver
assumes :math:`H` is convex with its minimum at :math:`p = 0` and
:math:`H(0) = 0`, in which case it reduces to the explicit form
:math:`H_h(p^-, p^+) = H(\\max(p^-, 0)) + H(\\min(p^+, 0))`. Every
Hamiltonian dual to a control cost :math:`L(\\alpha) \\ge 0` minimized at
:math:`\\alpha = 0` meets these assumptions, since
:math:`H(p) = \\sup_\\alpha[-\\alpha p - L(\\alpha)]` is then convex with
:math:`H(0) = H'(0) = 0`.

References
----------
Achdou, Y. and Capuzzo-Dolcetta, I. (2010). *Mean field games:
numerical methods*. SIAM Journal on Numerical Analysis 48, 1136-1162.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Hamiltonian:
    """A convex Hamiltonian :math:`H(p)` for :func:`mvkit.mfg.solve_hjb`.

    Attributes
    ----------
    H : callable
        Vectorized Hamiltonian ``H(p) -> ndarray``, shape-preserving.
        Must be convex with a global minimum at ``p = 0`` and
        ``H(0) = 0``.
    dH : callable
        Vectorized derivative ``H'(p) -> ndarray``, shape-preserving.
        Sets the optimal control ``alpha* = -dH(partial_x u)`` and the
        characteristic speed of the solver's CFL guard. Must satisfy
        ``dH(0) = 0``.
    name : str, default "custom"
        Short label, used in error messages and ``repr``.

    Notes
    -----
    The convexity and minimum-at-zero assumptions are what collapse the
    Engquist-Osher numerical Hamiltonian to the explicit
    ``H(max(p-, 0)) + H(min(p+, 0))``. They are spot-checked, on a
    sample of gradient values, when the Hamiltonian reaches
    :func:`solve_hjb`; the check catches sign errors and obvious
    non-convexity but is not a proof.
    """

    H: Callable[[np.ndarray], np.ndarray]
    dH: Callable[[np.ndarray], np.ndarray]
    name: str = "custom"

    def __repr__(self) -> str:
        return f"Hamiltonian(name={self.name!r})"


def quadratic_hamiltonian() -> Hamiltonian:
    """Return the quadratic Hamiltonian :math:`H(p) = p^2 / 2`.

    This is the default for :func:`solve_hjb`, and the only case with a
    Hopf-Cole linearization. Its optimal control is
    :math:`\\alpha^* = -\\partial_x u`.
    """
    return Hamiltonian(
        H=lambda p: 0.5 * np.asarray(p, dtype=np.float64) ** 2,
        dH=lambda p: np.array(p, dtype=np.float64),
        name="quadratic",
    )


def power_hamiltonian(q: float) -> Hamiltonian:
    """Return the power Hamiltonian :math:`H(p) = |p|^q / q`, ``q > 1``.

    This is the Legendre dual of the power control cost
    :math:`L(\\alpha) = |\\alpha|^{q'}/q'` with :math:`1/q + 1/q' = 1`,
    the standard non-quadratic family in the MFG literature. ``q = 2``
    reproduces :func:`quadratic_hamiltonian`. The derivative is
    :math:`H'(p) = \\mathrm{sign}(p)\\,|p|^{q-1}`, written in the
    ``sign * |p|**(q-1)`` form so it stays finite at ``p = 0`` for every
    ``q > 1``.

    Parameters
    ----------
    q : float
        Exponent, strictly greater than 1, so that ``H`` is convex and
        continuously differentiable with ``H(0) = H'(0) = 0``.
    """
    if not (np.isfinite(q) and q > 1.0):
        raise ValueError(f"power Hamiltonian needs q > 1, got {q}")
    q = float(q)

    def H(p: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(p, dtype=np.float64)) ** q / q

    def dH(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return np.sign(p) * np.abs(p) ** (q - 1.0)

    return Hamiltonian(H=H, dH=dH, name=f"power(q={q:g})")


def _validate_hamiltonian(ham: Hamiltonian) -> None:
    """Spot-check the Engquist-Osher assumptions on ``ham``.

    Verifies that ``H`` and ``dH`` are vectorized, shape-preserving and
    finite, that ``H(0) = H'(0) = 0``, that ``H`` is non-negative, and
    that ``H'`` is non-decreasing (``H`` convex). Sampled on a fixed
    gradient range; this catches sign errors and obvious non-convexity
    but is not a proof.
    """
    if not callable(ham.H) or not callable(ham.dH):
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H and dH must both be callable"
        )

    probe = np.linspace(-5.0, 5.0, 41)
    mid = probe.size // 2  # probe[mid] == 0.0 exactly
    try:
        h_vals = np.asarray(ham.H(probe), dtype=np.float64)
        dh_vals = np.asarray(ham.dH(probe), dtype=np.float64)
    except Exception as exc:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H or dH raised on a probe array "
            f"({exc})"
        ) from exc

    if h_vals.shape != probe.shape or dh_vals.shape != probe.shape:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H and dH must be vectorized and "
            f"shape-preserving (got H -> {h_vals.shape}, dH -> "
            f"{dh_vals.shape} for input {probe.shape})"
        )
    if not (np.isfinite(h_vals).all() and np.isfinite(dh_vals).all()):
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H or dH returned non-finite "
            "values on the probe range [-5, 5]"
        )
    if abs(float(h_vals[mid])) > 1e-9:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H(0) must be 0, got "
            f"{float(h_vals[mid]):.3e}"
        )
    if abs(float(dh_vals[mid])) > 1e-9:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H'(0) must be 0 (the minimum of "
            f"H sits at p=0), got {float(dh_vals[mid]):.3e}"
        )
    if float(h_vals.min()) < -1e-9:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H must be non-negative with its "
            f"minimum at p=0, but H reaches {float(h_vals.min()):.3e}"
        )
    if float(np.diff(dh_vals).min()) < -1e-9:
        raise ValueError(
            f"Hamiltonian {ham.name!r}: H' must be non-decreasing (H "
            "convex); it is not, on the probe range [-5, 5]"
        )


def _as_hamiltonian(spec: str | Hamiltonian) -> Hamiltonian:
    """Normalize a ``hamiltonian`` argument to a validated :class:`Hamiltonian`.

    Accepts the string ``"quadratic"`` (the back-compatible default) or a
    :class:`Hamiltonian` instance; anything else raises ``ValueError``.
    """
    if isinstance(spec, Hamiltonian):
        _validate_hamiltonian(spec)
        return spec
    if isinstance(spec, str):
        if spec == "quadratic":
            return quadratic_hamiltonian()
        raise ValueError(
            f"hamiltonian={spec!r} is not a known name; pass 'quadratic' "
            "or a Hamiltonian instance (see mvkit.mfg.power_hamiltonian)"
        )
    raise ValueError(
        "hamiltonian must be the string 'quadratic' or a Hamiltonian "
        f"instance, got {type(spec).__name__}"
    )
