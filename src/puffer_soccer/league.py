"""League training helpers shared by experimental RL training modes.

This module keeps the non-environment-specific parts of the league logic out of the main
training script. The project now wants to compare ordinary self-play against MARLadona-style
league training, and that comparison is much easier to reason about when the basic concepts
have names of their own instead of being encoded as ad hoc dictionaries scattered across the
trainer.

The helpers here intentionally stay small and generic:

- ``LeagueConfig`` describes the policy-pool and promotion defaults for one RL algorithm mode.
- ``LeagueEntry`` stores one immutable opponent snapshot together with stable ids and labels.
- ``LeagueManager`` owns the capped league pool, deterministic uniform sampling, promotion
  bookkeeping, and simple summary payloads.

The actual rollout collection still lives in the trainer because it needs deep knowledge of
the active vector environment and policy class. Keeping that boundary explicit makes it easier
to test the league bookkeeping separately while still letting the training loop adapt it to
our soccer environment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class LeagueConfig:
    """Describe the policy-pool behavior for one experimental RL algorithm mode.

    The project wants two related but distinct league-style algorithms:

    - ``league``: the minimal MARLadona-inspired league core
    - ``marlodonna``: the closest practical paper-faithful reproduction we can run here

    Both modes share the same underlying concepts, but they differ in defaults such as league
    capacity and whether standardized evaluation is expected. Bundling those settings into one
    dataclass keeps the later trainer code decision-complete and avoids magic constants leaking
    into multiple call sites.
    """

    rl_alg: str
    max_size: int
    promotion_win_rate_threshold: float
    standardized_eval_ratio: float
    standardized_eval_enabled: bool
    side_balance: bool = True
    opponent_sampling: str = "uniform"
    same_policy_per_team: bool = True


@dataclass(frozen=True)
class LeagueEntry:
    """Store one immutable opponent snapshot together with stable human-readable metadata.

    Training and evaluation need more than raw tensors. We also want stable ids for logs,
    deterministic ordering for oldest/newest comparisons, and a label that explains where the
    snapshot came from when reading JSON summaries later. This dataclass keeps those pieces
    together so the trainer does not have to reconstruct them repeatedly from parallel arrays.
    """

    entry_id: int
    label: str
    source_epoch: int
    state_dict: dict[str, Any]


@dataclass(frozen=True)
class LeaguePromotionResult:
    """Summarize one promotion check against the current league pool.

    The trainer uses this structure to log a promotion event consistently across stdout, W&B,
    and the run summary. Keeping the result explicit helps the later code answer three separate
    questions clearly: how strong was the current policy against the pool, what threshold was
    required, and did that check actually add a new snapshot to the league.
    """

    promoted: bool
    aggregate_win_rate: float
    aggregate_score_diff: float
    threshold: float
    opponents_evaluated: int
    promoted_entry_id: int | None


class LeagueManager:
    """Own a capped pool of opponent snapshots and deterministic sampling state.

    The main training loop needs a single source of truth for league membership. This manager
    stores immutable snapshots, trims the pool when it grows past the configured capacity, and
    exposes deterministic uniform sampling so rollout wrappers can assign one frozen opponent
    per environment. Promotion decisions are intentionally left simple here: the trainer passes
    in the aggregated evaluation numbers, and the manager decides whether that is enough to add
    a new snapshot.
    """

    def __init__(self, config: LeagueConfig, *, seed: int) -> None:
        """Create an empty league manager with deterministic random sampling.

        A dedicated ``random.Random`` instance is used instead of the module-level RNG so that
        sampling stays reproducible even when unrelated code elsewhere in training also draws
        random numbers.
        """

        self.config = config
        self._rng = random.Random(seed)
        self._entries: list[LeagueEntry] = []
        self._next_entry_id = 0

    @property
    def entries(self) -> tuple[LeagueEntry, ...]:
        """Return the current league entries in stable oldest-to-newest order."""

        return tuple(self._entries)

    def size(self) -> int:
        """Return the number of currently retained opponent snapshots."""

        return len(self._entries)

    def oldest(self) -> LeagueEntry | None:
        """Return the oldest retained league entry when one exists."""

        return None if not self._entries else self._entries[0]

    def newest(self) -> LeagueEntry | None:
        """Return the newest retained league entry when one exists."""

        return None if not self._entries else self._entries[-1]

    def bootstrap(
        self,
        initial_state_dict: Mapping[str, Any],
        *,
        label: str,
        source_epoch: int,
    ) -> LeagueEntry:
        """Seed the league with one initial opponent snapshot when the pool is empty.

        League training cannot start without at least one frozen opponent to play against.
        Our environment does not provide MARLadona's separate bot and random-controller setup,
        so the closest practical analogue is to begin with a frozen copy of the current policy.
        This helper makes that bootstrap explicit and idempotent.
        """

        if self._entries:
            return self._entries[0]
        return self.append_snapshot(
            initial_state_dict,
            label=label,
            source_epoch=source_epoch,
        )

    def append_snapshot(
        self,
        state_dict: Mapping[str, Any],
        *,
        label: str,
        source_epoch: int,
    ) -> LeagueEntry:
        """Add one immutable snapshot to the pool and enforce the configured capacity.

        The state dict is copied into a plain ``dict`` so later training updates cannot mutate
        the stored snapshot through shared references. When the pool exceeds ``max_size``, the
        oldest snapshot is evicted first. That FIFO policy matches the simplest interpretation
        of a capped replay buffer and is easy to explain when reading training logs.
        """

        entry = LeagueEntry(
            entry_id=self._next_entry_id,
            label=str(label),
            source_epoch=int(source_epoch),
            state_dict=dict(state_dict),
        )
        self._next_entry_id += 1
        self._entries.append(entry)
        if len(self._entries) > self.config.max_size:
            self._entries.pop(0)
        return entry

    def sample_entry_ids(self, count: int) -> list[int]:
        """Sample opponent ids uniformly with replacement from the active league pool.

        The rollout wrapper uses one opponent id per environment. Sampling with replacement is
        the intended behavior here because many environments may concurrently reuse the same
        frozen opponent, and uniform sampling keeps the training distribution simple and fully
        specified.
        """

        if count < 0:
            raise ValueError("count must be non-negative")
        if not self._entries:
            raise ValueError("cannot sample from an empty league")
        entry_ids = [entry.entry_id for entry in self._entries]
        return [self._rng.choice(entry_ids) for _ in range(count)]

    def resolve_entry(self, entry_id: int) -> LeagueEntry:
        """Return the retained entry with the requested stable id.

        Rollout code stores only integer ids in its per-environment assignment arrays. This
        lookup helper converts those ids back into the full entry payload while keeping error
        handling local to the league subsystem.
        """

        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(f"unknown league entry id: {entry_id}")

    # pylint: disable=too-many-arguments
    def maybe_promote(
        self,
        *,
        aggregate_win_rate: float,
        aggregate_score_diff: float,
        snapshot_state_dict: Mapping[str, Any],
        source_epoch: int,
        label: str,
    ) -> LeaguePromotionResult:
        """Promote the current policy into the pool when the configured threshold is met.

        The paper-first MARLadona comparison requested by this project uses a win-rate gate
        rather than the public repo's score-difference gate. The score difference is still
        carried through the result because it is informative for dashboards and summaries, but
        the actual decision is based only on ``aggregate_win_rate`` versus the configured
        threshold.
        """

        promoted_entry_id: int | None = None
        promoted = aggregate_win_rate >= self.config.promotion_win_rate_threshold
        if promoted:
            promoted_entry = self.append_snapshot(
                snapshot_state_dict,
                label=label,
                source_epoch=source_epoch,
            )
            promoted_entry_id = promoted_entry.entry_id
        return LeaguePromotionResult(
            promoted=promoted,
            aggregate_win_rate=float(aggregate_win_rate),
            aggregate_score_diff=float(aggregate_score_diff),
            threshold=float(self.config.promotion_win_rate_threshold),
            opponents_evaluated=len(self._entries),
            promoted_entry_id=promoted_entry_id,
        )

    def summary(self) -> dict[str, object]:
        """Return a compact JSON-ready description of the current league state.

        The end-of-run summary and periodic metadata both need a stable serialization of the
        active pool. Returning a deliberately compact payload here keeps those call sites short
        while still making the retained oldest/newest checkpoints discoverable later.
        """

        return {
            "rl_alg": self.config.rl_alg,
            "max_size": int(self.config.max_size),
            "promotion_win_rate_threshold": float(
                self.config.promotion_win_rate_threshold
            ),
            "opponent_sampling": self.config.opponent_sampling,
            "same_policy_per_team": bool(self.config.same_policy_per_team),
            "standardized_eval_enabled": bool(self.config.standardized_eval_enabled),
            "standardized_eval_ratio": float(self.config.standardized_eval_ratio),
            "side_balance": bool(self.config.side_balance),
            "size": len(self._entries),
            "entry_ids": [entry.entry_id for entry in self._entries],
            "entry_epochs": [entry.source_epoch for entry in self._entries],
        }


def league_assignment_histogram(entry_ids: Sequence[int]) -> dict[int, int]:
    """Summarize how often each retained opponent was assigned to environments.

    Uniform sampling is part of the intended league behavior, but we still want a human-readable
    count of what a particular rollout assignment looked like. Returning a plain ``dict`` keeps
    the result easy to log and serialize.
    """

    counts = Counter(int(entry_id) for entry_id in entry_ids)
    return dict(sorted(counts.items(), key=lambda item: item[0]))
