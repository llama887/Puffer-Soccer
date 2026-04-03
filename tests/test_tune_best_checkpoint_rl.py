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
        rl_alg=TRAIN_MODULE.RL_ALG_LEAGUE,
        kl_regularization_mode=TRAIN_MODULE.KL_REGULARIZATION_OFF,
        total_timesteps=30_000_000,
        final_eval_games=128,
        best_checkpoint_config_path="experiments/best_checkpoint.json",
        cached_warm_start_path="experiments/cached_warm_start_policy.pt",
        reuse_cached_warm_start=True,
        save_cached_warm_start=True,
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
    assert command[command.index("--rl-alg") + 1] == "league"
    assert command[command.index("--kl-regularization-mode") + 1] == "off"
    assert command[command.index("--cached-warm-start-path") + 1] == (
        "experiments/cached_warm_start_policy.pt"
    )
    assert "--reuse-cached-warm-start" in command
    assert "--save-cached-warm-start" in command


def test_resolve_runtime_vec_config_reuses_cached_runtime_file(tmp_path):
    runtime_path = tmp_path / "runtime_config.json"
    TRAIN_MODULE.write_json_record(
        runtime_path,
        {
            "vec_config": {
                "backend": "multiprocessing",
                "shard_num_envs": 10,
                "num_shards": 24,
                "num_workers": 24,
                "batch_size": 10,
                "zero_copy": True,
                "overwork": False,
            }
        },
    )
    args = SimpleNamespace(
        runtime_config_path=str(runtime_path),
        reuse_runtime_config=True,
        vec_backend="auto",
        players_per_team=5,
        num_envs=32,
        vec_num_shards=None,
        vec_batch_size=None,
        autotune_seconds=1.5,
        autotune_max_num_envs=None,
        autotune_max_num_shards=None,
    )

    vec_config, autotune_result = TUNE_MODULE.resolve_runtime_vec_config(args)

    assert vec_config.backend == "multiprocessing"
    assert vec_config.shard_num_envs == 10
    assert vec_config.num_shards == 24
    assert vec_config.batch_size == 10
    assert autotune_result is None
