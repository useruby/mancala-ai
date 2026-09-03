"""Focused contract tests for the frozen PR #265 runner."""

import copy

import pytest
import torch

from ml.alphazero_lite import run_pr265_unique_data_scale as pr265
from ml.alphazero_lite import self_play


def row(game_index: int, *, completed: bool = True) -> dict[str, object]:
    state = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    return {
        "game_index": game_index,
        "game_completed": completed,
        "state": self_play.encode_state(state, input_encoding="kalah_v3"),
        "policy": [1 / 6] * 6,
        "value": 1.0,
        "winner": 0,
        "player": 0,
        "value_target_mode": "default",
        "policy_target_mode": "default",
        "policy_target_noise_mode": "noisy",
        "simulations": 384,
    }


def screened(effect: float = 0.03) -> dict[str, object]:
    return {
        "per_seed": {str(seed): {"effect": effect} for seed in pr265.SEEDS},
        "pooled": {
            "mean": effect,
            "hierarchical_ci95": [0.01, 0.05],
            "positive_seed_count": 5,
        },
    }


def test_synthetic_plan_and_pure_az_telemetry_are_matched_and_deterministic() -> None:
    rows = [row(game_index) for game_index in range(10) for _ in range(2)]
    eligible, completion = pr265.validate_pure_targets(rows, 68)
    first = pr265.build_plans(eligible, 68, "replay-sha")
    second = pr265.build_plans(eligible, 68, "replay-sha")

    assert completion["completed_game_ratio"] == 1.0
    assert first["plan_sha256"] == second["plan_sha256"]
    assert not (
        set(first["game_split"]["train_games"])
        & set(first["game_split"]["validation_games"])
    )
    assert set(first["small_subset_games"]) <= set(first["game_split"]["train_games"])
    assert (
        first["lanes"]["unique100_once"]["presentations"]
        == first["lanes"]["repeat20_matched"]["presentations"]
    )
    assert [len(batch) for batch in first["lanes"]["unique100_once"]["batches"]] == [
        len(batch) for batch in first["lanes"]["repeat20_matched"]["batches"]
    ]

    model = pr265.new_model(torch.device("cpu"))
    initial = pr265.state_copy(model)
    heldout = pr265.pure_az_telemetry(model, initial, eligible[:2], 0.0, 0, 0)
    assert "policy_ce_p1" not in heldout
    assert "policy_ce_own_target" in heldout


def test_validate_pure_targets_excludes_incomplete_games_and_rejects_invalid_rows() -> (
    None
):
    rows = [row(game_index, completed=game_index != 0) for game_index in range(10)]
    eligible, completion = pr265.validate_pure_targets(rows, 68)

    assert {entry["game_index"] for entry in eligible} == set(range(1, 10))
    assert completion["games_excluded_incomplete"] == 1
    invalid = copy.deepcopy(eligible)
    invalid[0]["policy"] = [0.3] * 5 + [0.0]
    with pytest.raises(RuntimeError, match="invalid policy target"):
        pr265.validate_pure_targets(invalid, 68)


@pytest.mark.parametrize(
    ("unique", "repeat", "paired", "expected"),
    [
        (
            screened(),
            screened(),
            screened(),
            "unique_scale_beats_incumbent_and_repetition",
        ),
        (screened(), screened(), screened(-0.01), "optimizer_exposure_is_sufficient"),
        (
            screened(),
            screened(-0.01),
            screened(-0.01),
            "replacement_without_unique_data_evidence",
        ),
        (
            {
                **screened(),
                "pooled": {
                    "mean": 0.01,
                    "hierarchical_ci95": [-0.01, 0.02],
                    "positive_seed_count": 2,
                },
            },
            screened(-0.01),
            screened(-0.01),
            "unique_scale_improves_but_not_replacement",
        ),
        (screened(-0.03), screened(-0.02), screened(-0.01), "scale_degrades_strength"),
        (
            screened(-0.01),
            screened(0.0),
            screened(-0.01),
            "scale_does_not_rescue_high_budget_strength",
        ),
    ],
)
def test_classify_covers_all_preregistered_branches(
    unique: dict[str, object],
    repeat: dict[str, object],
    paired: dict[str, object],
    expected: str,
) -> None:
    assert (
        pr265.classify({"unique100_once": unique, "repeat20_matched": repeat}, paired)
        == expected
    )
