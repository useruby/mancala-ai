from __future__ import annotations

from ml.alphazero_lite import (
    seed542_update432_strength_screen as experiment,
    verify_seed542_update432_strength_screen as verifier,
)


def test_challenger_composition_is_update432_policy_with_frozen_initializer_value() -> (
    None
):
    assert experiment.ASSIGNMENTS == {
        "challenger_policy": "D-final",
        "challenger_policy_trunk": "D-final own trunk",
        "challenger_value": "initializer",
        "challenger_value_trunk": "initializer own trunk",
        "opponent_policy": "E4",
        "opponent_policy_trunk": "E4 own trunk",
        "opponent_value": "initializer",
        "opponent_value_trunk": "initializer own trunk",
    }


def test_advancement_requires_complete_screen_and_strictly_greater_lower_bound() -> (
    None
):
    assert not verifier.advance_eligible(2047, 0.51)
    assert not verifier.advance_eligible(2048, 0.50)
    assert verifier.advance_eligible(2048, 0.5000001)
