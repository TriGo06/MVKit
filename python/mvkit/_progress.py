"""Optional progress-bar support for long-running solvers.

When the user passes ``progress=True`` to a solver, the outer iteration
is wrapped in a :mod:`tqdm.auto.tqdm` progress bar if the dependency
is installed; otherwise the solver runs silently and prints a single
informational line. ``progress=False`` (the default) is fully silent
and incurs no overhead beyond a no-op wrapper.

This module is a thin indirection so the solvers themselves do not
need to handle the import-or-fall-back logic, and so testing the
fallback path is trivial.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


def _maybe_tqdm():
    """Return the ``tqdm.auto.tqdm`` callable if available, else None."""
    try:
        from tqdm.auto import tqdm  # type: ignore[import-untyped]
    except ImportError:
        return None
    return tqdm


def progress_iter(
    iterable: Iterable[T],
    total: Optional[int] = None,
    description: str = "",
    enabled: bool = False,
) -> Iterable[T]:
    """Wrap ``iterable`` in a progress bar when ``enabled`` is True.

    Parameters
    ----------
    iterable : iterable
        The iterable to wrap.
    total : int, optional
        Length hint for the progress bar (if the iterable does not
        define ``__len__``).
    description : str, optional
        Short label shown next to the progress bar.
    enabled : bool, default False
        If False, the original iterable is returned unchanged. If True
        and ``tqdm`` is importable, a ``tqdm.auto.tqdm`` is returned;
        otherwise a one-line note is printed and the original iterable
        is returned.

    Returns
    -------
    iterable
        Either the input iterable, or a tqdm-wrapped version of it.
    """
    if not enabled:
        return iterable
    tqdm = _maybe_tqdm()
    if tqdm is None:
        msg = (
            f"[mvkit] progress=True requested but tqdm is not installed; "
            f"running silently ({description})."
        )
        print(msg, file=sys.stderr)
        return iterable
    return tqdm(iterable, total=total, desc=description, leave=False)
