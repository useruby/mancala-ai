from __future__ import annotations

import pytest

from ml.alphazero_lite.root_q_trust_region import parent_q_counterfactual


def _children() -> list[dict]:
    return [
        {"move": 0, "visit_count": 3, "q_value": 0.4, "u_component": 0.2},
        {"move": 1, "visit_count": 2, "q_value": 0.3, "u_component": 0.1},
        {"move": 2, "visit_count": 0, "q_value": 0.0, "u_component": 0.5},
    ]


def test_p1_self_reference_and_exact_q_equality_are_noops() -> None:
    result = parent_q_counterfactual(
        _children(),
        {0: {"visits": 3, "q_value": 0.4}, 1: {"visits": 2, "q_value": 0.3}},
        0,
    )
    assert result["cf_move"] == 0
    assert result["selection_flip"] == 0
    assert result["selection_regret"] == 0


def test_irrelevant_q_difference_has_zero_regret() -> None:
    result = parent_q_counterfactual(
        _children(),
        {0: {"visits": 3, "q_value": 0.4}, 1: {"visits": 2, "q_value": 0.35}},
        0,
    )
    assert result["selection_regret"] == 0


def test_parent_q_crossing_causal_boundary_flips_with_exact_regret() -> None:
    result = parent_q_counterfactual(
        _children(),
        {0: {"visits": 3, "q_value": 0.4}, 1: {"visits": 2, "q_value": 0.7}},
        0,
    )
    assert result["cf_move"] == 1
    assert result["selection_flip"] == 1
    assert result["selection_regret"] == pytest.approx(0.2)


def test_unvisited_edges_keep_candidate_q_and_p1_unvisited_edges_are_ineligible() -> (
    None
):
    result = parent_q_counterfactual(
        _children(),
        {
            0: {"visits": 0, "q_value": 9.0},
            1: {"visits": 2, "q_value": 0.3},
            2: {"visits": 4, "q_value": 9.0},
        },
        0,
    )
    rows = {row["move"]: row for row in result["rows"]}
    assert rows[0]["counterfactual_q"] == 0.4
    assert rows[2]["counterfactual_q"] == 0.0
    assert not rows[0]["synchronized"] and not rows[2]["synchronized"]
