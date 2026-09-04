from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite.correct_generation3_turn_completion_puct import (
    PR274_ARTIFACTS,
    historical_previous_classification,
)
from ml.alphazero_lite.gumbel_root_search import run_puct_root_search
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.arena import sha256_file
from ml.alphazero_lite.run_generation3_turn_completion_puct import (
    classify,
    comparison_metrics,
    compare_deterministic_result,
    sliced_metrics,
)
from ml.alphazero_lite.self_play import HeuristicEvaluator, standard_start_state
from ml.alphazero_lite.turn_completion_puct import run_turn_completion_puct


ROOT = Path(__file__).resolve().parents[2]


def metric(n_distinct_states: int, catastrophic_delta: float) -> dict:
    return {
        "n_distinct_states": n_distinct_states,
        "n_state_seed_pairs": n_distinct_states * 3,
        "catastrophic_miss_rate_delta": catastrophic_delta,
    }


def aggregate_metrics() -> dict:
    lane = {
        "mean_regret": 1.0,
        "best_reference_action_agreement": 1.0,
        "catastrophic_miss_rate": 0.0,
        "p95_runtime_seconds": 1.0,
    }
    return {
        "root_extra_turn": {
            "baseline": lane,
            "candidate": lane,
            "paired_hierarchical_bootstrap": {"upper_95": 1.0},
        },
        "full": {
            "baseline": lane,
            "candidate": lane,
            "paired_hierarchical_bootstrap": {"upper_95": 1.0},
        },
    }


def classification(
    slices: dict, *, invariants_ok: bool = True, budget_ok: bool = True
) -> str:
    return classify(
        aggregate_metrics(), slices, invariants_ok=invariants_ok, budget_ok=budget_ok
    )


def game(player: int = 0) -> KalahGame:
    return KalahGame.from_state(standard_start_state() | {"current_player": player})


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize("budget", [16, 48])
def test_disabled_search_is_byte_equivalent_to_exact_budget_puct(player, budget):
    baseline = run_puct_root_search(
        game(player), HeuristicEvaluator(), seed=275101, budget=budget
    )
    disabled = run_turn_completion_puct(
        game(player), HeuristicEvaluator(), seed=275101, budget=budget, enabled=False
    )
    assert disabled.selected_move == baseline.selected_move
    assert disabled.visits == baseline.visits
    assert disabled.q_values == baseline.q_values
    assert disabled.evaluator_calls == baseline.evaluator_calls == budget
    assert disabled.budget_padding_calls == baseline.budget_padding_calls


def test_non_extra_turn_leaf_stops_at_normal_evaluation():
    state = {
        "player_pits": [1, 1, 0, 0, 0, 0],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    result = run_turn_completion_puct(
        KalahGame.from_state(state), HeuristicEvaluator(), seed=275101, budget=2
    )
    assert result.extensions_started == 0
    assert result.evaluator_calls == 2


def test_extra_turn_continues_and_discards_intermediate_value():
    # Pit 5 lands in the store, then the remaining pit ends the turn.
    state = {
        "player_pits": [1, 0, 0, 0, 0, 1],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    result = run_turn_completion_puct(
        KalahGame.from_state(state), HeuristicEvaluator(), seed=275101, budget=4
    )
    assert result.extensions_started >= 1
    assert result.extensions_completed >= 1
    assert result.extra_turn_actions >= 1
    assert result.discarded_intermediate_values
    assert result.final_backed_up_values


def test_terminal_move_is_not_an_extra_turn_and_capture_metadata_is_canonical():
    terminal = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [0] * 6,
        "player_store": 23,
        "opponent_store": 24,
        "current_player": 0,
    }
    consequence = move_consequence_for_state(terminal, 0)
    assert consequence["game_over_after_move"]
    assert not consequence["gives_extra_turn"]
    capture = {
        "player_pits": [1, 0, 0, 0, 0, 0],
        "opponent_pits": [0, 0, 0, 0, 1, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    assert move_consequence_for_state(capture, 0)["produces_capture"]


def test_candidate_is_deterministic_legal_and_never_exceeds_budget():
    for player in (0, 1):
        first = run_turn_completion_puct(
            game(player), HeuristicEvaluator(), seed=275102, budget=32
        )
        second = run_turn_completion_puct(
            game(player), HeuristicEvaluator(), seed=275102, budget=32
        )
        assert first == second
        assert first.selected_move in game(player).possible_moves()
        assert first.evaluator_calls == 32
        assert first.incomplete_due_to_budget <= 1
        assert first.defensive_cap_events == first.repeated_state_events == 0


def test_qualifying_eight_state_slice_regresses_subsets():
    assert classification({"phase": {"late": metric(8, 0.020001)}}) == (
        "turn_completion_regresses_subsets"
    )


def test_repeated_correction_preserves_pr275_historical_classification():
    corrected = {
        "classification": "turn_completion_regresses_subsets",
        "correction": {
            "previous_classification": "turn_completion_regresses_subsets",
        },
    }
    assert (
        historical_previous_classification(corrected)
        == "turn_completion_no_search_gain"
    )


def test_three_seeds_count_as_one_distinct_state():
    rows = [
        {
            "state_hash": "one",
            "phase": "late",
            "player": 0,
            "legal_action_count": 3,
            "capture_available": False,
            "extra_turn_available": True,
            "baseline": {
                str(seed): {
                    "regret": 0.0,
                    "best_reference_action_agreement": True,
                    "catastrophic_miss": False,
                }
                for seed in (1, 2, 3)
            },
            "candidate": {
                str(seed): {
                    "regret": 0.0,
                    "best_reference_action_agreement": True,
                    "catastrophic_miss": False,
                }
                for seed in (1, 2, 3)
            },
        }
    ]
    metrics = comparison_metrics(rows)
    assert metrics["n_distinct_states"] == 1
    assert metrics["n_state_seed_pairs"] == 3


def test_seven_distinct_states_cannot_trigger_subset_rule():
    assert classification({"phase": {"late": metric(7, 1.0)}}) == (
        "turn_completion_no_search_gain"
    )


def test_subset_delta_equal_to_limit_does_not_trigger():
    assert classification({"phase": {"late": metric(8, 0.02)}}) == (
        "turn_completion_no_search_gain"
    )


def test_invariant_failure_has_priority_over_subset_regression():
    assert classification({"phase": {"late": metric(8, 1.0)}}, invariants_ok=False) == (
        "turn_completion_invariant_failure"
    )


def test_budget_failure_has_priority_over_subset_regression():
    assert classification({"phase": {"late": metric(8, 1.0)}}, budget_ok=False) == (
        "turn_completion_budget_contract_failed"
    )


def test_subset_regression_has_priority_over_aggregate_qualification():
    qualifying = aggregate_metrics()
    for name in ("root_extra_turn", "full"):
        qualifying[name]["candidate"] = {
            "mean_regret": 0.1,
            "best_reference_action_agreement": 1.0,
            "catastrophic_miss_rate": 0.0,
            "p95_runtime_seconds": 1.0,
        }
        qualifying[name]["paired_hierarchical_bootstrap"] = {"upper_95": -0.1}
    assert (
        classify(
            qualifying,
            {"phase": {"late": metric(8, 0.03)}},
            invariants_ok=True,
            budget_ok=True,
        )
        == "turn_completion_regresses_subsets"
    )


def test_deterministic_comparison_ignores_runtime_only():
    assert compare_deterministic_result(
        {"selected_move": 2, "runtime_seconds": 1.0},
        {"selected_move": 2, "runtime_seconds": 2.0},
    )["equal_excluding_runtime"]


def test_deterministic_comparison_normalizes_stored_json_mapping_keys():
    assert compare_deterministic_result({"visits": {"2": 4}}, {"visits": {2: 4}})[
        "equal_excluding_runtime"
    ]


def test_deterministic_comparison_rejects_behavior_or_telemetry_mismatch():
    comparison = compare_deterministic_result(
        {"selected_move": 2, "visits": {2: 4}, "runtime_seconds": 1.0},
        {"selected_move": 3, "visits": {2: 4}, "runtime_seconds": 1.0},
    )
    assert not comparison["equal_excluding_runtime"]
    assert "selected_move" in comparison["mismatches"]


def test_stored_root_extra_turn_metrics_trigger_regression_and_regret_is_nonnegative():
    payload = json.loads(
        (
            ROOT / "docs/data/alphazero-lite-generation3-turn-completion-puct-full.json"
        ).read_text()
    )
    slices = sliced_metrics(payload["results"])
    root_extra_turn = slices["aggregate_groups"]["root_extra_turn"]
    assert root_extra_turn["n_distinct_states"] == 64
    assert root_extra_turn["catastrophic_miss_rate_delta"] == 0.0625
    assert classification(slices) == "turn_completion_regresses_subsets"
    for player in ("0", "1"):
        for lane in ("baseline", "candidate"):
            assert all(
                result["regret"] >= 0
                for row in payload["results"]
                if str(row["player"]) == player
                for result in row[lane].values()
            )


def test_pr274_artifact_hashes_match_merge_commit_restoration():
    for relative_path, expected_hash in PR274_ARTIFACTS.items():
        assert sha256_file(ROOT / relative_path) == expected_hash
