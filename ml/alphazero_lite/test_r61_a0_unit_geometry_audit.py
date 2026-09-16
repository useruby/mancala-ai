import torch
import pytest

from ml.alphazero_lite.run_r61_a0_unit_geometry_audit import (
    PREFIX_SIZES,
    causal_score,
    feature_groups,
    gating_classification,
    minimal_prefix,
    prefix_sizes,
    ranked_units,
    transfer_fraction,
)


def row(unit: int, score: float, score108: float, early: float) -> dict:
    return {
        "unit": unit,
        "eligible": True,
        "final": {"causal_score": score},
        "108": {"causal_score": score108},
        "82": {
            "forward": {"cluster_margin_delta": early},
            "reverse": {"cluster_margin_delta": -early},
        },
    }


def test_registered_causal_score_uses_control_allowance() -> None:
    assert causal_score(0.2, -0.1, 0.005) == 0.1
    assert causal_score(0.2, -0.1, 0.03) == pytest.approx(0.08)
    assert causal_score(-0.2, -0.1, 0) == 0


def test_ranking_and_prefixes_are_deterministic() -> None:
    ranked = ranked_units(
        [row(3, 0.1, 0.2, 0.1), row(2, 0.1, 0.2, 0.1), row(1, 0.2, 0.1, 0.1)]
    )
    assert [value["unit"] for value in ranked] == [1, 2, 3]
    assert PREFIX_SIZES == (1, 2, 4, 8, 16)
    assert prefix_sizes(3) == [1, 2]


def test_transfer_and_minimal_prefix_are_bidirectional() -> None:
    assert transfer_fraction(1.0, 1.6, 2.0) == pytest.approx(0.6)
    prefix = {
        "forward_transfer_fraction": 0.6,
        "reverse_transfer_fraction": 0.6,
        "forward_correct_direction_fraction": 0.6,
        "control_degradation_fraction": 0.19,
        "step108_directional": True,
    }
    assert minimal_prefix([prefix]) == prefix
    prefix["reverse_transfer_fraction"] = 0.59
    assert minimal_prefix([prefix]) is None


def test_gating_classes_and_encoding_feature_groups_are_explicit() -> None:
    assert gating_classification(-0.1, -0.2) == "both_inactive"
    assert gating_classification(-0.1, 0.2) == "relu_threshold_crossing"
    assert gating_classification(0.1, 0.2) == "both_active_magnitude_difference"
    groups = feature_groups()
    assert groups["player_pits"][0] == "player_pits_0"
    assert "player_extra_turn_available" in groups["v3_tactical"]


def test_single_coordinate_patch_can_leave_other_coordinates_exact() -> None:
    left, right = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[4.0, 5.0, 6.0]])
    patched = left.clone()
    patched[:, [1]] = right[:, [1]]
    assert torch.equal(patched[:, [0, 2]], left[:, [0, 2]])
    assert torch.equal(patched[:, [1]], right[:, [1]])
