from __future__ import annotations

import json
from pathlib import Path

from ml.alphazero_lite.run_fresh_uniform1200_confirmation import (
    LANES,
    EXPECTED_PARENT_SHA256,
    EXTENSION_SEEDS,
    FIXED_REPLAY_SHA256S,
    SEEDS,
    SIX_SEEDS,
    classify,
    classify_six_seed,
    extension_lane_config,
    lane_config,
    non_budget_identity,
    paired_transitions,
    repeated_changes,
    six_seed_aggregation,
    write_json,
)


def plan() -> dict:
    return {
        "lanes": {
            "control": {
                "simulations": 192,
                "opening_min_simulations": 384,
                "opening_min_simulations_plies": 8,
            },
            "uniform1200": {
                "simulations": 1200,
                "opening_min_simulations": None,
                "opening_min_simulations_plies": None,
            },
        },
        "regenerated_replay_sources": [
            {"path": "generic.jsonl", "weight": 4},
            {"path": "teacher.jsonl", "weight": 1},
            {"path": "disagreement.jsonl", "weight": 8},
            {"path": "stability.jsonl", "weight": 4},
        ],
    }


def base() -> dict:
    return {
        "steps": [
            {
                "name": "self_play",
                "command": [
                    "x",
                    "--seed",
                    "42",
                    "--seed-sweep",
                    "41,42,43",
                    "--simulations",
                    "192",
                    "--opening-min-simulations",
                    "384",
                    "--opening-min-simulations-plies",
                    "8",
                ],
            },
            {
                "name": "train",
                "command": ["x", "--seed", "42", "--epochs", "4", "--lr", "0.001"],
            },
        ]
    }


def exact_row() -> dict:
    return {
        "id": "capture_available-x",
        "canonical_state": "state",
        "exact_status": "exact_solved",
        "state": {"current_player": 0},
        "exact_action_values": {"0": 2, "1": 0, "2": -2},
    }


def test_three_matched_seeds_and_only_budget_difference() -> None:
    assert SEEDS == (47, 48, 49)
    left = lane_config(plan(), base(), 47, LANES[0], Path("/work"))
    right = lane_config(plan(), base(), 47, LANES[1], Path("/work"))
    assert non_budget_identity(left) == non_budget_identity(right)
    assert left["steps"][1] == right["steps"][1]
    assert "rowmatch" not in str(left).lower()


def test_outcome_transitions_and_repeated_definition() -> None:
    rows = []
    for seed in (47, 48):
        rows.extend(paired_transitions([exact_row()], {"state": 0}, {"state": 1}, seed))
    regressions = repeated_changes(rows, worse=True)
    assert regressions[0]["seeds_affected"] == [47, 48]
    assert regressions[0]["outcome_transition_by_seed"] == {
        "47": "win_to_draw",
        "48": "win_to_draw",
    }


def test_classification_requires_shadow_and_separates_production_gate() -> None:
    assert (
        classify(
            [0.1, 0.02, -0.01],
            2,
            [],
            aggregate_regression=False,
            production_misaligned=False,
            heterogeneous=False,
        )
        == "uniform1200_fresh_strength_confirmed"
    )
    assert (
        classify(
            [0.1, 0.02, -0.01],
            2,
            [],
            aggregate_regression=False,
            production_misaligned=True,
            heterogeneous=False,
        )
        == "uniform1200_current_production_gate_misaligned"
    )
    assert (
        classify(
            [0.1, 0.02, -0.01],
            1,
            [],
            aggregate_regression=False,
            production_misaligned=False,
            heterogeneous=False,
        )
        == "uniform1200_fresh_no_strength_gain"
    )


def extension_plan() -> dict:
    return json.loads(
        Path(
            "ml/alphazero_lite/configs/fresh_uniform1200_confirmation_extension.json"
        ).read_text()
    )


def six_seed_rows() -> list[dict]:
    return [
        {
            "seed": seed,
            "arena_score": score,
            "control": {"top1": 0.5, "mean_regret": 2.0, "blunder_rate": 0.4},
            "uniform1200": {
                "top1": 0.6,
                "mean_regret": 1.0,
                "blunder_rate": 0.3,
            },
        }
        for seed, score in zip(SIX_SEEDS, (0.5, 0.5, 1.0, 0.5, 0.75, 1.0))
    ]


def test_extension_freezes_artifacts_and_original_contract() -> None:
    plan_data = extension_plan()
    assert SEEDS == (47, 48, 49)
    assert EXTENSION_SEEDS == (50, 51, 52)
    assert plan_data["inherited_seeds"] == [47, 48, 49]
    assert plan_data["seeds"] == [50, 51, 52]
    assert plan_data["expected_parent_weights_sha256"] == EXPECTED_PARENT_SHA256
    assert {
        row["name"]: row["sha256"] for row in plan_data["fixed_replay_sources"]
    } == FIXED_REPLAY_SHA256S
    assert plan_data["promotion"]["performed"] is False
    assert "r61" not in json.dumps(plan_data["fixed_replay_sources"]).lower()
    assert plan_data["secondary_r61_panel"].endswith(
        "r61-expanded-exact-subcluster.json"
    )


def test_extension_only_changes_selfplay_budget_and_keeps_arena() -> None:
    plan_data = extension_plan()
    left = extension_lane_config(plan_data, base(), 50, "control", Path("/work"))
    right = extension_lane_config(plan_data, base(), 50, "uniform1200", Path("/work"))
    assert non_budget_identity(left) == non_budget_identity(right)
    assert left["steps"][1] == right["steps"][1]
    assert plan_data["arena"] == {
        "games": 120,
        "seed": 90417,
        "simulations": 384,
        "workers": 24,
        "seed_contract": "azlite_eval_seed_v1",
    }
    assert [
        left["steps"][0]["command"][
            left["steps"][0]["command"].index("--seed-sweep") + 1
        ],
        extension_lane_config(plan_data, base(), 51, "control", Path("/work"))["steps"][
            0
        ]["command"][
            extension_lane_config(plan_data, base(), 51, "control", Path("/work"))[
                "steps"
            ][0]["command"].index("--seed-sweep")
            + 1
        ],
    ] == ["49,50,51", "50,51,52"]


def test_six_seed_aggregation_is_seed_paired_and_strictly_positive() -> None:
    aggregate = six_seed_aggregation(six_seed_rows())
    assert aggregate["experimental_unit"] == "matched_training_seed_pair"
    assert aggregate["arena"]["number_positive"] == 3
    assert aggregate["arena"]["number_neutral"] == 3
    assert aggregate["arena"]["effects"][0] == 0.0
    assert aggregate["exact_paired_deltas"]["top1_accuracy"]["number_improved"] == 6
    assert (
        classify_six_seed(
            six_seed_rows(),
            exact_metrics_improved=6,
            repeated_win_to_loss=False,
            shadow_passes=6,
            production_misaligned=False,
        )
        == "uniform1200_six_seed_exact_gain_arena_uncertain"
    )


def test_six_seed_repeated_threshold_is_four_for_both_directions() -> None:
    rows = []
    for seed in (47, 48, 49, 50):
        rows.extend(paired_transitions([exact_row()], {"state": 0}, {"state": 1}, seed))
    assert len(repeated_changes(rows, worse=True, threshold=4)) == 1
    assert repeated_changes(rows, worse=True, threshold=5) == []
    improvements = []
    for seed in (47, 48, 49, 50):
        improvements.extend(
            paired_transitions([exact_row()], {"state": 1}, {"state": 0}, seed)
        )
    assert len(repeated_changes(improvements, worse=False, threshold=4)) == 1


def test_extension_serialization_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"b": [2, 1], "a": {"seed": 50}}
    write_json(first, payload)
    write_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
