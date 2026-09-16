from __future__ import annotations

from pathlib import Path

from ml.alphazero_lite.run_fresh_uniform1200_confirmation import (
    LANES,
    SEEDS,
    classify,
    lane_config,
    non_budget_identity,
    paired_transitions,
    repeated_changes,
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
        }
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
