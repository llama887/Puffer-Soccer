import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODULE = _load_module(REPO_ROOT / "scripts" / "train_pufferl.py", "train_pufferl")
TUNE_MODULE = _load_module(
    REPO_ROOT / "scripts" / "tune_best_checkpoint_rl.py",
    "tune_best_checkpoint_rl",
)


def test_choose_valid_minibatch_size_projects_to_nearest_legal_value():
    assert TRAIN_MODULE.choose_valid_minibatch_size(20_480, 64, 5_000) == 5_120
    assert TRAIN_MODULE.choose_valid_minibatch_size(20_480, 64, 400) == 320


def test_resolve_requested_train_sizes_respects_override_constraints():
    batch_size, minibatch_size = TRAIN_MODULE.resolve_requested_train_sizes(
        total_agents=320,
        horizon=128,
        requested_batch_size=50_000,
        requested_minibatch_size=7_000,
    )

    assert batch_size == 50_048
    assert minibatch_size == 2_944
    assert batch_size % 128 == 0
    assert batch_size % minibatch_size == 0


def test_resolve_total_timesteps_rounds_fixed_budget_down_to_full_batches():
    assert TRAIN_MODULE.resolve_total_timesteps(30_000_000, 1000, 81_920) == 29_982_720


def test_resolve_rollout_hyperparameters_maps_latent_space_to_actual_sizes():
    rollout = TUNE_MODULE.resolve_rollout_hyperparameters(
        {
            "rollout": {
                "horizon": 128,
                "batch_multiple": 2,
                "minibatch_divisor": 8,
            }
        },
        total_agents=320,
    )

    assert rollout == {
        "bptt_horizon": 128,
        "train_batch_size": 81_920,
        "minibatch_size": 10_240,
    }


def test_build_trial_command_freezes_runtime_and_best_checkpoint_target(tmp_path):
    args = SimpleNamespace(
        players_per_team=5,
        device="cpu",
        total_timesteps=30_000_000,
        final_eval_games=128,
        best_checkpoint_config_path="experiments/best_checkpoint.json",
        total_agents=320,
    )
    command = TUNE_MODULE.build_trial_command(
        args=args,
        vec_config=TRAIN_MODULE.VecEnvConfig(
            backend="native",
            shard_num_envs=32,
            num_shards=1,
        ),
        suggestion={
            "rollout": {
                "horizon": 64,
                "batch_multiple": 1,
                "minibatch_divisor": 4,
            },
            "train": {
                "learning_rate": 3e-4,
                "update_epochs": 2,
                "gamma": 0.995,
                "gae_lambda": 0.9,
                "clip_coef": 0.2,
                "vf_coef": 2.0,
                "vf_clip_coef": 0.2,
                "max_grad_norm": 1.5,
                "ent_coef": 1e-4,
                "prio_alpha": 0.8,
                "prio_beta0": 0.2,
            },
            "regularization": {
                "past_kl_coef": 0.1,
                "uniform_kl_base_coef": 0.05,
                "uniform_kl_power": 0.3,
            },
        },
        seed=123,
        summary_path=tmp_path / "summary.json",
    )

    assert "--fixed-best-checkpoint" in command
    assert "--no-wandb" in command
    assert "--no-export-videos" in command
    assert "--vec-backend" in command
    assert command[command.index("--num-envs") + 1] == "32"
