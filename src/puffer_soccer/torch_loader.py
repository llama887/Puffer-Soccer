"""Load `torch` with a narrow workaround for broken package metadata reads.

PyTorch 2.10 scans Python entry points while importing device backends. In some
container and network-filesystem setups, a single unreadable
`*.dist-info/entry_points.txt` file can make that scan fail with `OSError`
before this project executes any real code. That failure is outside our code,
but it blocks both training and tests.

This module keeps the workaround in one place so the rest of the codebase can
continue to import and use `torch` normally. The retry path is intentionally
small: it only activates for the specific unreadable-`entry_points.txt` case,
and it only changes how `importlib.metadata.distributions()` reports entry
points during the retry import.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import sys
from typing import Any


class _DistributionEntryPointGuard:
    """Proxy one distribution and hide unreadable entry-point metadata.

    `importlib.metadata.entry_points()` only needs each distribution object's
    `entry_points` property during the scan that PyTorch triggers at import
    time. Some filesystems can raise `OSError` when that property tries to read
    `entry_points.txt`. Returning an empty tuple for that one distribution lets
    the overall scan continue, which matches the effective meaning of "no entry
    points were readable here" and is safer than failing the whole process.
    """

    def __init__(self, distribution: Any) -> None:
        self._distribution = distribution

    @property
    def entry_points(self) -> tuple[()]:
        """Return the wrapped distribution's entry points or an empty set.

        The fallback is deliberately narrow. We only suppress `OSError` because
        that is the failure mode caused by unreadable metadata files on some
        mounts. All other exceptions still surface so real packaging bugs are
        not hidden.
        """

        try:
            return self._distribution.entry_points
        except OSError:
            return ()

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped distribution."""

        return getattr(self._distribution, name)


def _clear_partial_torch_modules() -> None:
    """Remove partially imported `torch` modules before a retry.

    When the first import fails partway through module initialization, Python
    can leave `torch` and submodules in `sys.modules`. Retrying against that
    half-initialized state can produce confusing follow-on errors, so we clear
    only the `torch` namespace before trying again.
    """

    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            sys.modules.pop(module_name, None)


def _is_unreadable_entry_points_error(error: OSError) -> bool:
    """Return `True` only for the metadata-read failure we intend to handle.

    The retry workaround should not hide arbitrary filesystem or import errors.
    We therefore require the exception text to mention `entry_points.txt`,
    which is the concrete failure seen when `importlib.metadata` cannot read one
    package's entry-point file during PyTorch startup.
    """

    return "entry_points.txt" in str(error)


def import_torch():
    """Import and return `torch`, retrying once for broken metadata scans.

    The normal path is a plain `importlib.import_module("torch")`. If that
    raises an `OSError` for an unreadable `entry_points.txt`, we temporarily
    wrap `importlib.metadata.distributions()` so unreadable distributions report
    no entry points instead of aborting the scan, then retry the import. The
    patch is removed immediately after the retry, regardless of outcome.
    """

    try:
        return importlib.import_module("torch")
    except OSError as error:
        if not _is_unreadable_entry_points_error(error):
            raise

    _clear_partial_torch_modules()
    original_distributions = importlib_metadata.distributions

    def guarded_distributions(**kwargs: Any):
        """Yield distributions whose `entry_points` reads are OSError-safe."""

        for distribution in original_distributions(**kwargs):
            yield _DistributionEntryPointGuard(distribution)

    importlib_metadata.distributions = guarded_distributions
    try:
        return importlib.import_module("torch")
    finally:
        importlib_metadata.distributions = original_distributions
