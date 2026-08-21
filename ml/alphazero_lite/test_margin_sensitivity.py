from __future__ import annotations

import numpy as np

from ml.alphazero_lite.margin_sensitivity import decision_sensitivity


def _decision() -> dict:
    return {
        "chosen_move": 0,
        "legal_moves": [0, 1],
        "parent_visit_count": 4,
        "children": [
            {"move": 0, "prior": 0.6, "visit_count": 1, "selection_score": 1.0},
            {"move": 1, "prior": 0.4, "visit_count": 1, "selection_score": 0.8},
        ],
    }


def test_frozen_puct_margin_algebra() -> None:
    result = decision_sensitivity(
        _decision(), np.asarray([0.4, 0.6, 0, 0, 0, 0]), c_puct=1.25
    )

    assert np.isclose(result["delta_u"][0], -0.25)
    assert np.isclose(result["delta_u"][1], 0.25)
    assert np.isclose(result["max_flip_excess"], 0.3)
    assert np.isclose(result["max_pressure_ratio"], 2.5)
    assert result["counterfactual_flip"]


def test_pressure_ratio_exceeds_one_iff_candidate_changes_frozen_winner() -> None:
    no_flip = decision_sensitivity(
        _decision(), np.asarray([0.53, 0.47, 0, 0, 0, 0]), c_puct=1.25
    )
    flip = decision_sensitivity(
        _decision(), np.asarray([0.4, 0.6, 0, 0, 0, 0]), c_puct=1.25
    )

    assert no_flip["max_pressure_ratio"] <= 1.0
    assert not no_flip["counterfactual_flip"]
    assert flip["max_pressure_ratio"] > 1.0
    assert flip["counterfactual_flip"]
