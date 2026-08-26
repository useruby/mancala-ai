from __future__ import annotations

from ml.alphazero_lite.context_action_q_selection import corrected_selection
from ml.alphazero_lite.root_q_trust_region import parent_q_counterfactual


def test_zero_correction_reproduces_ordinary_winner() -> None:
    children = [
        {"move": 0, "visit_count": 3, "q_value": 0.2, "u_component": 0.1},
        {"move": 1, "visit_count": 2, "q_value": 0.1, "u_component": 0.3},
    ]

    assert corrected_selection(children, {})["move"] == 1


def test_exact_correction_reproduces_parent_q_counterfactual() -> None:
    children = [
        {"move": 0, "visit_count": 3, "q_value": 0.2, "u_component": 0.1},
        {"move": 1, "visit_count": 2, "q_value": 0.1, "u_component": 0.3},
        {"move": 2, "visit_count": 0, "q_value": 0.0, "u_component": 0.5},
    ]
    reference = {
        0: {"visits": 4, "q_value": 0.0},
        1: {"visits": 4, "q_value": 0.5},
        2: {"visits": 4, "q_value": 1.0},
    }
    correction = {0: -0.2, 1: 0.4, 2: 1.0}

    exact = parent_q_counterfactual(children, reference, actual_move=0)
    learned = corrected_selection(children, correction)

    assert learned["move"] == exact["cf_move"] == 1
    assert next(row for row in learned["rows"] if row["move"] == 2)["correction"] == 0.0


def test_ties_prefer_lowest_puct_action() -> None:
    children = [
        {"move": 4, "visit_count": 1, "q_value": 0.2, "u_component": 0.1},
        {"move": 2, "visit_count": 1, "q_value": 0.1, "u_component": 0.2},
    ]

    assert corrected_selection(children, {})["move"] == 2
