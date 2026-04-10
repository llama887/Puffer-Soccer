"""ELO rating calculations."""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that player A wins against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Return updated ratings for both players after a match.

    score_a is the actual score from A's perspective:
      1.0 = A wins, 0.0 = A loses, 0.5 = draw.
    """
    ea = expected_score(rating_a, rating_b)
    eb = 1.0 - ea
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - ea)
    new_b = rating_b + k * (score_b - eb)
    return new_a, new_b
