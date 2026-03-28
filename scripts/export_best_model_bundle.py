# pylint: disable=duplicate-code
"""Export the current best checkpoint into a self-contained local policy bundle.

This utility exists for two related workflows:
- bootstrap the already-known best checkpoint into the new bundle format
- export an explicit checkpoint path into the canonical `experiments/baselines/current_best`
  location for later evaluation

The script rebuilds the current soccer policy architecture only once at export time. After
that, the saved TorchScript module inside the bundle becomes the forward-compatible artifact
that future evaluation should prefer.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv


def _load_train_module():
    script_path = Path(__file__).resolve().with_name("train_pufferl.py")
    spec = importlib.util.spec_from_file_location("train_pufferl_export", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_pufferl = _load_train_module()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI used to export one best-model bundle.

    The export path needs just enough information to rebuild the current policy class around a
    raw checkpoint: how many players per team the environment has, where the source checkpoint
    comes from, and whether the local best-pointer record should be updated after the bundle is
    written.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--best-checkpoint-config-path",
        type=str,
        default="experiments/best_checkpoint.json",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="bootstrap")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--global-step", type=int, default=0)
    parser.add_argument("--event", type=str, default="manual_export")
    return parser


def resolve_export_source(
    *,
    args,
    best_config_path: Path,
    best_record: dict[str, object] | None,
) -> tuple[Path, dict[str, object] | None, dict[str, object], str, int, int, str]:
    """Resolve the checkpoint and metadata that should seed a bundle export.

    The exporter supports two modes: use an explicit checkpoint path, or bootstrap from the
    existing local best-pointer record. Turning that branching into one helper keeps the main
    flow compact and makes the source selection policy easier to test and reason about.
    """

    checkpoint_path: Path
    state_dict: dict[str, object] | None = None
    run_id = args.run_id
    epoch = int(args.epoch)
    global_step = int(args.global_step)
    event = args.event
    previous_best = None if best_record is None else dict(best_record)

    if args.checkpoint_path is not None:
        checkpoint_path = train_pufferl.resolve_checkpoint_file(Path(args.checkpoint_path))
        return checkpoint_path, previous_best, {}, run_id, epoch, global_step, event

    if best_record is None:
        raise RuntimeError(
            "No best checkpoint record exists and no explicit --checkpoint-path was provided"
        )
    loaded_state_dict, resolved_checkpoint = train_pufferl.load_best_checkpoint_state(
        best_record,
        logger=None,
        cache_dir=best_config_path.parent / "wandb_artifacts",
    )
    state_dict = dict(loaded_state_dict)
    checkpoint_path = resolved_checkpoint
    run_id = str(best_record.get("run_id") or run_id)
    record_epoch = best_record.get("epoch")
    if isinstance(record_epoch, int):
        epoch = record_epoch
    record_global_step = best_record.get("global_step")
    if isinstance(record_global_step, int):
        global_step = record_global_step
    event = str(best_record.get("event") or event)
    return checkpoint_path, previous_best, state_dict, run_id, epoch, global_step, event


def main() -> None:
    """Export a self-contained bundle and update the local best-pointer record.

    When `--checkpoint-path` is omitted, the exporter treats the existing
    `best_checkpoint.json` file as the source of truth and updates that same pointer record
    with the newly created bundle paths. This keeps bootstrapping the current best baseline
    simple: one command converts the old raw-checkpoint-only pointer into the new durable
    bundle-aware format.
    """

    args = build_parser().parse_args()
    device = train_pufferl.resolve_device(args.device)
    best_config_path = Path(args.best_checkpoint_config_path)
    best_record = train_pufferl.read_json_record(best_config_path)
    (
        checkpoint_path,
        previous_best,
        state_dict,
        run_id,
        epoch,
        global_step,
        event,
    ) = resolve_export_source(
        args=args,
        best_config_path=best_config_path,
        best_record=best_record,
    )

    env = make_soccer_vecenv(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=0,
        opponents_enabled=True,
        vec=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
    )
    try:
        policy = train_pufferl.Policy(env).to(device)
        if args.checkpoint_path is not None:
            state_dict = train_pufferl.load_checkpoint_state_dict(checkpoint_path)
        if not state_dict:
            raise RuntimeError("failed to resolve a state dict for bundle export")
        policy.load_state_dict(state_dict, strict=True)
        bundle_metadata = train_pufferl.build_bundle_export_metadata(
            args=argparse.Namespace(
                past_iterate_eval_games=0,
                past_iterate_eval_game_length=400,
                final_best_eval_games=0,
                best_checkpoint_promotion_confidence=0.95,
                best_checkpoint_promotion_min_batches=4,
                best_checkpoint_promotion_max_batches=64,
            ),
            run_id=run_id,
            epoch=epoch,
            global_step=global_step,
            event=event,
            checkpoint_path=checkpoint_path,
            previous_best=previous_best,
            train_config=None,
            objective_metrics=None,
            promotion_metrics=None,
        )
        record = train_pufferl.register_best_checkpoint(
            logger=None,
            checkpoint_path=checkpoint_path,
            best_config_path=best_config_path,
            best_history_path=best_config_path.parent / "best_checkpoint_history.jsonl",
            previous_best=previous_best,
            vec_config=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
            run_id=run_id,
            epoch=epoch,
            global_step=global_step,
            event=event,
            promotion_metrics=None,
            policy=policy,
            observation_shape=tuple(env.single_observation_space.shape),
            bundle_metadata=bundle_metadata,
        )
    finally:
        env.close()

    print(f"Exported best-model bundle to {record['bundle_dir']}")
    print(f"Bundle manifest: {record['bundle_manifest_path']}")


if __name__ == "__main__":
    main()
