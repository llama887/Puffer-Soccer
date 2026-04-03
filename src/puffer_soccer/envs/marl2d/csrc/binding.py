"""Fallback module for environments where the native MARL2D extension is not compiled.

This package normally imports `binding` from a compiled C extension
(`binding.cpython-*.so`). That extension gives the high-performance simulation code
required for training and evaluation. Some development machines, CI jobs, or remote
notebooks do not have a C compiler available, which means editable installation can
still be useful for non-environment tasks (linting, docs, config checks, or pure-Python
tests) as long as import-time behavior remains explicit.

This file intentionally provides a runtime-only fallback. Importing the package succeeds,
but the first attempted attribute access raises a clear error explaining how to install
tooling needed to build the extension.
"""


def __getattr__(name: str):
    """Raise an actionable error when code tries to use the missing native binding.

    We fail lazily instead of at package install time so contributors without a compiler
    can still run workflows that do not touch the native MARL2D environment. The error is
    raised on attribute access because environment code always calls functions on the
    `binding` module. This produces one consistent failure mode with direct remediation
    steps and avoids partial behavior that could hide performance regressions.
    """

    raise ImportError(
        "puffer_soccer.envs.marl2d.csrc.binding native extension is unavailable. "
        "Install a C compiler (for example `gcc`) and reinstall with `uv run pip install -e .` "
        "to build the MARL2D native backend."
    )
