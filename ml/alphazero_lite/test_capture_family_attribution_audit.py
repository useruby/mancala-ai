from ml.alphazero_lite.run_capture_family_attribution_audit import (
    cross_seed_dominance,
    capture_rows,
    comparative_search_regression,
    decision_critical_ids,
    mechanism,
    policy_quality,
    paired_target_deltas,
    PR290_LANES,
    search_induced_regression,
    teacher_student_inversions,
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


def test_teacher_student_inversion_requires_worse_raw_student():
    key = next(
        row["canonical_state"]
        for row in capture_rows()
        if row["id"] == "capture_available-002"
    )
    targets = [{"canonical_state": key, "expected_exact_regret": 0.0}]
    evaluations = [{"id": "capture_available-002", "raw": {"regret": 0.0}}]
    assert teacher_student_inversions(targets, targets, evaluations, evaluations) == {
        "shared_states": 1,
        "inversion_ids": [],
        "inversion_rate": 0.0,
    }


def test_pr290_cells_have_the_historical_lane_mapping():
    assert PR290_LANES == {
        "B": "control_like_exposure__unsharpened",
        "C": "uniform_exposure__unsharpened",
        "D": "control_like_exposure__sharpened",
    }


def test_paired_target_deltas_and_comparative_search_regression():
    control = [
        {
            "canonical_state": "x",
            "optimal_mass": 0.5,
            "expected_exact_regret": 2,
            "entropy": 1,
            "top_optimal": False,
        }
    ]
    uniform = [
        {
            "canonical_state": "x",
            "optimal_mass": 0.75,
            "expected_exact_regret": 1,
            "entropy": 0.5,
            "top_optimal": True,
        }
    ]
    assert paired_target_deltas(control, uniform)["expected_exact_regret_delta"] == -1
    assert comparative_search_regression(12, 0, 12, 12)
