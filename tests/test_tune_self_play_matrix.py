import importlib.util
from pathlib import Path
import sys


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_MODULE = _load_module(
    REPO_ROOT / "scripts" / "tune_self_play_matrix.py",
    "tune_self_play_matrix",
)


def test_matrix_variants_lists_all_six_rl_comparison_combinations():
    variants = MATRIX_MODULE.matrix_variants()

    assert len(variants) == 6
    assert variants[0]["label"] == "self_play__kl_off"
    assert variants[-1]["label"] == "marlodonna__kl_on"


def test_build_child_command_includes_variant_flags_and_output_folder(tmp_path):
    command = MATRIX_MODULE.build_child_command(
        variant={
            "label": "league__kl_on",
            "rl_alg": "league",
            "kl_regularization_mode": "on",
        },
        output_dir=tmp_path,
        forwarded_args=["--device", "cpu", "--max-runs", "2"],
    )

    assert "--rl-alg" in command
    assert command[command.index("--rl-alg") + 1] == "league"
    assert command[command.index("--kl-regularization-mode") + 1] == "on"
    assert command[command.index("--output-dir") + 1] == str(
        tmp_path / "league__kl_on"
    )
    assert command[-4:] == ["--device", "cpu", "--max-runs", "2"]
