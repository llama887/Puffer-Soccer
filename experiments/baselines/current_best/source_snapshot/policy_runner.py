"""Load and run frozen policy bundles without depending on the live repo policy class.

This module is intentionally small because a copy of it is written into every exported
policy bundle under ``source_snapshot/policy_runner.py``. The bundle's primary executable
artifact is a TorchScript module, but keeping this human-readable helper beside it makes the
saved baseline easier to inspect and revive later when the in-repo policy architecture has
changed.

The helpers here focus on only two jobs:
- load one exported TorchScript policy module onto the requested device
- normalize the module's forward output into the ``(logits, values)`` shape expected by
  evaluation code in this repo
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from puffer_soccer.torch_loader import import_torch

if TYPE_CHECKING:
    import torch

torch = import_torch()


def load_policy_module(path: str | Path, device: str = "cpu") -> "torch.jit.ScriptModule":
    """Load one frozen policy module and switch it into eval mode.

    Exported bundles are meant for deterministic evaluation, not for further optimization.
    The loader therefore always maps tensors onto the requested device and immediately calls
    ``eval()`` so downstream code does not need to remember that step each time it revives a
    saved baseline.
    """

    module = torch.jit.load(str(Path(path)), map_location=device)
    module.eval()
    return module


def forward_policy_module(
    policy_module: "torch.nn.Module",
    observations: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Run one eval forward pass and normalize the result shape.

    The exported TorchScript module is traced from the training policy's ordinary forward
    method, so calling the module directly returns the same ``(logits, values)`` tuple that
    the live policy does. Keeping this normalization here means evaluators can treat frozen
    modules and live Torch modules the same way.
    """

    output = policy_module(observations)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError(
            "Expected exported policy module to return a (logits, values) tuple"
        )
    logits, values = output
    if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError("Exported policy module returned non-tensor outputs")
    return logits, values
