from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
    seat_asymmetry_ds,
)
import pytest


def games(scores: dict[tuple[int, int], float]) -> list[dict]:
    return [
        {"opening_index": opening, "challenger_player": seat, "score": score}
        for (opening, seat), score in scores.items()
    ]


def test_duplicate_current_can_have_nonzero_seat_asymmetry() -> None:
    scores = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 1.0, (1, 1): 0.0}
    result = paired_opening_candidate_effect(games(scores), games(scores))
    assert seat_asymmetry_ds(1.0, 0.0) == 1.0
    assert result["paired_candidate_effect"] == 0.0


def test_equal_improvement_does_not_change_seat_asymmetry() -> None:
    control = {(0, 0): 0.2, (0, 1): 0.6}
    candidate = {(0, 0): 0.4, (0, 1): 0.8}
    assert seat_asymmetry_ds(0.4, 0.8) == pytest.approx(seat_asymmetry_ds(0.2, 0.6))
    assert paired_opening_candidate_effect(games(candidate), games(control))[
        "paired_candidate_effect"
    ] == pytest.approx(0.2)


def test_worse_candidate_can_have_larger_seat_asymmetry() -> None:
    control = {(0, 0): 0.6, (0, 1): 0.6}
    candidate = {(0, 0): 0.7, (0, 1): 0.3}
    assert seat_asymmetry_ds(0.7, 0.3) > seat_asymmetry_ds(0.6, 0.6)
    assert (
        paired_opening_candidate_effect(games(candidate), games(control))[
            "paired_candidate_effect"
        ]
        < 0
    )


def test_duplicate_current_effect_is_exactly_zero() -> None:
    scores = {(0, 0): 0.5, (0, 1): 1.0, (1, 0): 0.0, (1, 1): 0.5}
    assert (
        paired_opening_candidate_effect(games(scores), games(scores))[
            "paired_candidate_effect"
        ]
        == 0.0
    )


def test_effect_is_invariant_to_seat_label_swaps() -> None:
    control = {(0, 0): 0.2, (0, 1): 0.6, (1, 0): 0.4, (1, 1): 0.8}
    candidate = {(0, 0): 0.4, (0, 1): 0.7, (1, 0): 0.5, (1, 1): 0.9}
    swapped_control = {
        (opening, 1 - seat): score for (opening, seat), score in control.items()
    }
    swapped_candidate = {
        (opening, 1 - seat): score for (opening, seat), score in candidate.items()
    }
    assert (
        paired_opening_candidate_effect(games(candidate), games(control))[
            "paired_candidate_effect"
        ]
        == paired_opening_candidate_effect(
            games(swapped_candidate), games(swapped_control)
        )["paired_candidate_effect"]
    )


def test_effect_is_invariant_to_record_order() -> None:
    control = {(0, 0): 0.2, (0, 1): 0.6, (1, 0): 0.4, (1, 1): 0.8}
    candidate = {(0, 0): 0.4, (0, 1): 0.7, (1, 0): 0.5, (1, 1): 0.9}
    assert (
        paired_opening_candidate_effect(games(candidate), games(control))[
            "paired_candidate_effect"
        ]
        == paired_opening_candidate_effect(
            list(reversed(games(candidate))), list(reversed(games(control)))
        )["paired_candidate_effect"]
    )


def test_effect_rejects_mismatched_opponent_identity() -> None:
    candidate = games({(0, 0): 0.5, (0, 1): 0.5})
    control = games({(0, 0): 0.5, (0, 1): 0.5})
    for record in candidate:
        record.update(
            opponent_weights_sha256="current", opponent_config_sha256="puct-a"
        )
    for record in control:
        record.update(
            opponent_weights_sha256="baseline", opponent_config_sha256="puct-a"
        )
    with pytest.raises(ValueError, match="identical opponent identity"):
        paired_opening_candidate_effect(candidate, control)


def test_effect_difference_uses_per_opening_matched_controls() -> None:
    left = paired_opening_candidate_effect(
        games({(0, 0): 0.8, (0, 1): 0.8}), games({(0, 0): 0.5, (0, 1): 0.5})
    )
    right = paired_opening_candidate_effect(
        games({(0, 0): 0.6, (0, 1): 0.6}), games({(0, 0): 0.5, (0, 1): 0.5})
    )
    assert paired_effect_difference(left, right)[
        "paired_candidate_effect"
    ] == pytest.approx(0.2)
