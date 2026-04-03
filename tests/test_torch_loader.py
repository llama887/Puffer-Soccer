"""Regression tests for the defensive `torch` import helper."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from types import SimpleNamespace
from unittest.mock import patch

from puffer_soccer import torch_loader
from puffer_soccer.torch_loader import import_torch


def test_import_torch_retries_when_entry_points_file_is_unreadable():
    """Verify the helper retries once with guarded distribution metadata.

    The first import simulates the filesystem-specific failure reported in batch
    jobs, where reading one distribution's `entry_points.txt` raises
    `OSError`. The retry should temporarily patch `importlib.metadata` so a
    distribution with unreadable entry-point metadata is treated as contributing
    no entry points, which is enough for the second import to succeed.
    """

    original_distributions = importlib_metadata.distributions
    fake_torch = SimpleNamespace(__name__="torch")
    attempts = {"count": 0}

    class BrokenDistribution:  # pylint: disable=too-few-public-methods
        """Minimal distribution stub whose entry points cannot be read."""

        @property
        def entry_points(self):
            """Raise the same `OSError` the real broken metadata file raises."""

            raise OSError("Unknown error 521: '/tmp/pkg.dist-info/entry_points.txt'")

    def fake_import_module(name: str):
        """Simulate a first failed import followed by a guarded retry import."""

        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("Unknown error 521: '/tmp/pkg.dist-info/entry_points.txt'")
        assert name == "torch"
        recovered = list(importlib_metadata.distributions())
        assert len(recovered) == 1
        assert recovered[0].entry_points == ()
        return fake_torch

    with patch.object(
        importlib_metadata, "distributions", return_value=iter([BrokenDistribution()])
    ), patch.object(
        torch_loader.importlib, "import_module", side_effect=fake_import_module
    ), patch.object(torch_loader, "_clear_partial_torch_modules"):
        loaded_torch = import_torch()

    assert loaded_torch is fake_torch
    assert attempts["count"] == 2
    assert importlib_metadata.distributions is original_distributions


def test_import_torch_does_not_retry_for_other_os_errors():
    """Verify unrelated import `OSError`s still surface without patching."""

    with patch.object(
        torch_loader.importlib,
        "import_module",
        side_effect=OSError("permission denied"),
    ) as mocked_import:
        try:
            import_torch()
        except OSError as error:
            assert str(error) == "permission denied"
        else:
            raise AssertionError("expected import_torch() to re-raise the OSError")

    mocked_import.assert_called_once_with("torch")
