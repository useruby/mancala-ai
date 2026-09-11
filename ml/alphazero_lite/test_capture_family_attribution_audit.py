from ml.alphazero_lite.run_capture_family_attribution_audit import (
    cross_seed_dominance,
    decision_critical_ids,
    mechanism,
    policy_quality,
    search_induced_regression,
    transition,
)


def test_decision_critical_extraction_is_deterministic():
    shadow = {
        "decision_flips": {
            str(seed): {
                "genuine_regressions": [
                    {
                        "bucket": "capture_available",
                        "id": "capture_available-002",
                        "uniform_minus_control_exact_regret": 12,
                    }
                ]
            }
            for seed in (44, 45, 46)
        }
    }
    assert decision_critical_ids(shadow) == ["capture_available-002"]


def test_target_quality_preserves_root_player_perspective():
    exact = {
        "state": {"current_player": 1},
        "exact_action_values": {"0": 4, "1": -2},
        "exact_optimal_actions": [1],
    }
    result = policy_quality([0.25, 0.75, 0, 0, 0, 0], exact)
    assert result["top_optimal"] and result["expected_exact_regret"] == 1.5


def test_search_transition_and_mechanism_precedence():
    assert transition(0, 2) == "raw_correct_to_search_wrong"
    assert search_induced_regression(0, 4, 2)
    assert (
        mechanism(
            search_induced=True,
            coverage_adequate=False,
            target_worse=True,
            raw_worse=True,
        )
        == "inference_search_degradation"
    )


def test_cross_seed_dominance_requires_two_seeds():
    assert (
        cross_seed_dominance(
            {
                44: ["coverage_deficit"],
                45: ["coverage_deficit"],
                46: ["mixed_or_unclear"],
            }
        )
        == "coverage_deficit"
    )
