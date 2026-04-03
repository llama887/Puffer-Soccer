"""Register an already-finished checkpoint as the canonical current best baseline.

This utility exists for the common research workflow where a long `sbatch` run finishes, the
result looks strong, and we want to promote that final checkpoint into the best-baseline files
without rerunning training. The script intentionally reuses the exact same helper functions as
the training entrypoint so manual backfills and automatic end-of-training promotions produce the
same pointer files, W&B artifact metadata, and self-contained policy bundles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pufferlib import pufferl

import train_pufferl


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI used to register one finished checkpoint as the active best model.

    The required inputs are intentionally small: the checkpoint path plus the run metadata that
    should appear in the best-baseline record. Optional arguments control whether W&B is used for
    artifact upload and let the caller keep the saved metadata aligned with the environment shape
    and vector-layout context that produced the checkpoint.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Promote an existing checkpoint into experiments/best_checkpoint.json and the "
            "current_best bundle without rerunning training."
        )
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--event", type=str, default="manual_backfill")
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--best-checkpoint-config-path",
        type=Path,
        default=Path("experiments/best_checkpoint.json"),
    )
    parser.add_argument(
        "--best-checkpoint-history-path",
        type=Path,
        default=Path("experiments/best_checkpoint_history.jsonl"),
    )
    parser.add_argument("--vec-backend", type=str, default="native")
    parser.add_argument("--vec-shard-num-envs", type=int, default=8)
    parser.add_argument("--vec-num-shards", type=int, default=1)
    parser.add_argument("--vec-batch-size", type=int, default=None)
    parser.add_argument("--vec-num-workers", type=int, default=None)
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb-project", type=str, default="robot-soccer")
    parser.add_argument("--wandb-group", type=str, default="manual-best-checkpoint")
    parser.add_argument("--wandb-tag", type=str, default=None)
    return parser


def build_bundle_metadata(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    previous_best: dict[str, object] | None,
) -> dict[str, object]:
    """Build the bundle manifest payload for one manual best-baseline registration.

    Automatic promotions can describe themselves from the live training arguments, while this
    script only knows about the checkpoint being backfilled. Writing a focused manifest here keeps
    manual promotions self-describing and makes it clear that the record came from a post-hoc
    registration step rather than from the end of an in-process training run.
    """

    return {
        "run_id": args.run_id,
        "epoch": int(args.epoch),
        "global_step": int(args.global_step),
        "event": str(args.event),
        "git_commit": train_pufferl.policy_bundle.current_git_commit(Path.cwd()),
        "original_checkpoint_path": str(checkpoint_path),
        "previous_best_artifact_ref": None
        if previous_best is None
        else previous_best.get("artifact_ref"),
        "manual_registration": True,
        "manual_registration_args": {
            "players_per_team": int(args.players_per_team),
            "device": str(args.device),
            "vec_backend": str(args.vec_backend),
            "vec_shard_num_envs": int(args.vec_shard_num_envs),
            "vec_num_shards": int(args.vec_num_shards),
            "vec_batch_size": None
            if args.vec_batch_size is None
            else int(args.vec_batch_size),
            "vec_num_workers": None
            if args.vec_num_workers is None
            else int(args.vec_num_workers),
        },
    }


def register_existing_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    """Promote one existing checkpoint into the active best-baseline records and bundles.

    The helper loads the checkpoint into the current policy architecture, exports the durable
    evaluation bundle, updates the pointer and history files, and optionally uploads the W&B
    model artifact. Returning the final record makes the function easy to reuse in tests and
    gives the CLI a single JSON payload to print on success.
    """

    train_pufferl.load_env_file(".env")
    checkpoint_path = train_pufferl.resolve_checkpoint_file(args.checkpoint_path)
    previous_best = train_pufferl.read_json_record(args.best_checkpoint_config_path)
    device = train_pufferl.resolve_device(str(args.device))
    logger = None
    if args.wandb:
        logger = pufferl.WandbLogger(
            {
                "wandb_project": args.wandb_project,
                "wandb_group": args.wandb_group,
                "tag": args.wandb_tag,
            }
        )

    vec_config = train_pufferl.VecEnvConfig(
        backend=str(args.vec_backend),
        shard_num_envs=int(args.vec_shard_num_envs),
        num_shards=int(args.vec_num_shards),
        num_workers=args.vec_num_workers,
        batch_size=args.vec_batch_size,
    )
    vecenv = train_pufferl.make_soccer_vecenv(
        players_per_team=int(args.players_per_team),
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=0,
        opponents_enabled=True,
        vec=vec_config,
    )
    policy = train_pufferl.Policy(vecenv).to(device)
    checkpoint_state = train_pufferl.load_checkpoint_state_dict(checkpoint_path)
    policy.load_state_dict(checkpoint_state, strict=True)
    bundle_metadata = build_bundle_metadata(
        args=args,
        checkpoint_path=checkpoint_path,
        previous_best=previous_best,
    )
    checkpoint_path_for_close = str(checkpoint_path)
    try:
        record = train_pufferl.register_best_checkpoint(
            logger=logger,
            checkpoint_path=checkpoint_path,
            best_config_path=args.best_checkpoint_config_path,
            best_history_path=args.best_checkpoint_history_path,
            previous_best=previous_best,
            vec_config=vec_config,
            run_id=str(args.run_id),
            epoch=int(args.epoch),
            global_step=int(args.global_step),
            event=str(args.event),
            promotion_metrics=None,
            policy=policy,
            observation_shape=tuple(vecenv.single_observation_space.shape),
            bundle_metadata=bundle_metadata,
        )
    finally:
        vecenv.close()
        if logger is not None:
            logger.close(checkpoint_path_for_close)
    return record


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, register the checkpoint, and print the resulting record as JSON."""

    args = build_parser().parse_args(argv)
    record = register_existing_checkpoint(args)
    print(json.dumps(record, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
