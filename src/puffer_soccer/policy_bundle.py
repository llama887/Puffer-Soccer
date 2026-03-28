"""Export, load, and manage self-contained policy bundles for long-lived evaluation.

The project keeps one canonical "current best" baseline that later training runs should
evaluate against. A raw state dict alone is not enough for that job because the live repo
may change its policy architecture over time. This module therefore exports a compact,
durable bundle that includes both a TorchScript policy module for future evaluation and the
raw checkpoint state for debugging and backward compatibility.

The bundle layout is intentionally simple and filesystem-friendly:
- ``policy_module.pt`` stores the eval-ready TorchScript module
- ``checkpoint_state.pt`` stores the raw CPU state dict
- ``manifest.json`` records enough metadata to understand where the bundle came from
- ``source_snapshot/policy_runner.py`` keeps a small frozen loader helper beside the model
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

from puffer_soccer import policy_bundle_runner
from puffer_soccer.policy_bundle_runner import load_policy_module
from puffer_soccer.torch_loader import import_torch

if TYPE_CHECKING:
    import torch

torch = import_torch()

BUNDLE_SCHEMA_VERSION = 1
POLICY_MODULE_FILENAME = "policy_module.pt"
CHECKPOINT_STATE_FILENAME = "checkpoint_state.pt"
MANIFEST_FILENAME = "manifest.json"
SOURCE_SNAPSHOT_DIRNAME = "source_snapshot"
POLICY_RUNNER_SNAPSHOT_FILENAME = "policy_runner.py"


def _json_default(value: object) -> object:
    """Convert bundle metadata values into JSON-safe data.

    Bundle manifests primarily contain simple scalars and dictionaries, but paths are also
    useful because they help future debugging and bootstrap scripts reconnect the bundle to
    the original training run. Converting paths here keeps every manifest write path
    consistent and avoids duplicating tiny serialization helpers elsewhere.
    """

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def current_timestamp() -> str:
    """Return a compact UTC timestamp string for manifest metadata."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def current_git_commit(cwd: Path | None = None) -> str | None:
    """Return the current git commit when the bundle is exported from a git checkout.

    The saved baseline should be traceable back to the exact code revision that produced it.
    Git metadata is not mandatory, though, because some export paths may run from unpacked
    archives or detached artifact directories. Returning ``None`` in those cases keeps export
    best-effort rather than brittle.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=None if cwd is None else str(cwd),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    """Return the standard file layout for one exported bundle directory.

    Centralizing the path layout makes bundle reads, writes, and tests share the same
    contract. That matters because the bundle is meant to survive code refactors, so its
    on-disk structure should stay explicit and stable.
    """

    return {
        "bundle_dir": bundle_dir,
        "policy_module_path": bundle_dir / POLICY_MODULE_FILENAME,
        "checkpoint_state_path": bundle_dir / CHECKPOINT_STATE_FILENAME,
        "manifest_path": bundle_dir / MANIFEST_FILENAME,
        "source_snapshot_dir": bundle_dir / SOURCE_SNAPSHOT_DIRNAME,
        "policy_runner_snapshot_path": bundle_dir
        / SOURCE_SNAPSHOT_DIRNAME
        / POLICY_RUNNER_SNAPSHOT_FILENAME,
    }


def write_json_record(path: Path, payload: dict[str, object]) -> None:
    """Persist one JSON record atomically.

    Bundle manifests and pointer files are consulted by future runs, so a partial write would
    be worse than no write at all. Using a sibling temporary file keeps updates atomic on the
    local filesystem and matches the metadata policy already used elsewhere in this repo.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def read_bundle_manifest(bundle_dir: Path) -> dict[str, object]:
    """Read and validate one bundle manifest from disk.

    The manifest is the stable source of truth for a bundle. Loading it through one helper
    keeps validation consistent and gives callers a clear error when the directory exists but
    the export is incomplete.
    """

    manifest_path = bundle_paths(bundle_dir)["manifest_path"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {manifest_path}")
    return payload


def _replace_directory_atomically(source_dir: Path, target_dir: Path) -> None:
    """Replace one bundle directory with another fully written directory tree.

    Bundle export first writes into a temporary directory and only swaps it into place after
    every file is ready. That keeps the canonical ``current_best`` directory from ever being
    observed in a half-written state during a long training run.
    """

    backup_dir = target_dir.with_name(f"{target_dir.name}.bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if target_dir.exists():
        os.replace(target_dir, backup_dir)
    os.replace(source_dir, target_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def _copy_policy_runner_snapshot(destination: Path) -> None:
    """Copy the frozen loader helper into the exported bundle.

    The human-readable snapshot is intentionally a plain file copy from the shared source
    module. That keeps the snapshot small and auditable while ensuring the exported bundle
    carries the same loader logic that the rest of the codebase used at export time.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(policy_bundle_runner.__file__), destination)


def _cpu_state_dict(
    state_dict: dict[str, "torch.Tensor"],
) -> dict[str, "torch.Tensor"]:
    """Clone a state dict onto CPU tensors for durable export.

    Bundle artifacts should be portable across machines and should not keep references into a
    live training model. Cloning every tensor onto CPU memory makes the saved checkpoint
    stable and independent of the original device placement.
    """

    return {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in state_dict.items()
    }


def export_policy_bundle(
    *,
    policy: "torch.nn.Module",
    checkpoint_state: dict[str, "torch.Tensor"],
    bundle_dir: Path,
    example_observation: "torch.Tensor",
    metadata: dict[str, object],
) -> dict[str, object]:
    """Export one self-contained evaluation bundle for a policy snapshot.

    The export intentionally writes both the raw state dict and a TorchScript module. The
    state dict is helpful for debugging and for old tooling that still expects checkpoints,
    while the TorchScript module is the forward-compatible artifact that lets future code
    evaluate the saved baseline even after the live ``Policy`` class has changed.
    """

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f"{bundle_dir.name}.tmp-", dir=str(bundle_dir.parent))
    )
    tmp_paths = bundle_paths(tmp_dir)

    cpu_state = _cpu_state_dict(checkpoint_state)
    policy_copy = copy.deepcopy(policy).to("cpu")
    was_training = policy_copy.training
    policy_copy.eval()
    example_cpu = example_observation.detach().cpu()
    traced_policy = torch.jit.trace(policy_copy, example_cpu)
    traced_policy.save(str(tmp_paths["policy_module_path"]))
    torch.save(cpu_state, tmp_paths["checkpoint_state_path"])
    _copy_policy_runner_snapshot(tmp_paths["policy_runner_snapshot_path"])

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": current_timestamp(),
        "model_format": "torchscript",
        "policy_module_path": POLICY_MODULE_FILENAME,
        "checkpoint_state_path": CHECKPOINT_STATE_FILENAME,
        "source_snapshot_path": str(
            Path(SOURCE_SNAPSHOT_DIRNAME) / POLICY_RUNNER_SNAPSHOT_FILENAME
        ),
        **metadata,
    }
    write_json_record(tmp_paths["manifest_path"], manifest)
    if was_training:
        policy_copy.train()

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    _replace_directory_atomically(tmp_dir, bundle_dir)
    final_paths = bundle_paths(bundle_dir)
    return {
        "bundle_dir": str(final_paths["bundle_dir"]),
        "bundle_manifest_path": str(final_paths["manifest_path"]),
        "bundle_policy_module_path": str(final_paths["policy_module_path"]),
        "bundle_checkpoint_state_path": str(final_paths["checkpoint_state_path"]),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def load_policy_module_from_bundle(
    bundle_dir: Path,
    *,
    device: str,
) -> tuple["torch.jit.ScriptModule", dict[str, object]]:
    """Load the frozen policy module and manifest from one bundle directory.

    Callers usually need both the executable module and the manifest metadata that explains
    what run and checkpoint produced it. Returning them together keeps the loader convenient
    while still making the bundle contents explicit to the caller.
    """

    manifest = read_bundle_manifest(bundle_dir)
    module = load_policy_module(bundle_paths(bundle_dir)["policy_module_path"], device=device)
    return module, manifest


def bundle_dir_from_record(record: dict[str, object] | None) -> Path | None:
    """Return the bundle directory encoded in a best-checkpoint record when present.

    Newer best-checkpoint records carry explicit bundle metadata, while older ones only know
    about raw checkpoint paths. This helper gives callers one small compatibility boundary:
    either a bundle directory is available, or the caller should fall back to the legacy
    checkpoint-loading path.
    """

    if record is None:
        return None
    bundle_dir = record.get("bundle_dir")
    if not isinstance(bundle_dir, str) or not bundle_dir:
        return None
    path = Path(bundle_dir)
    if not path.exists():
        return None
    return path
