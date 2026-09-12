from ml.alphazero_lite.run_root_q_visit_capture_audit import (
    STABILITY_WINDOW,
    classify,
    correction_times,
    leader,
    pairwise_q_order_accuracy,
    spearman_q_exact,
)


def test_root_leaders_are_deterministic_on_ties():
    moves = [
        {"move": 2, "q_value": 0.1, "visit_count": 3},
        {"move": 1, "q_value": 0.1, "visit_count": 3},
    ]
    assert leader(moves, "q_value") == 1
    assert leader(moves, "visit_count") == 1


def test_pairwise_q_order_excludes_exact_ties():
    exact = {"exact_action_values": {"0": 5, "1": 5, "2": 1}}
    moves = [
        {"move": 0, "q_value": 0.1},
        {"move": 1, "q_value": 0.2},
        {"move": 2, "q_value": 0.0},
    ]
    assert pairwise_q_order_accuracy(moves, exact) == 1.0
    assert spearman_q_exact(moves, exact) is not None


def test_stable_correction_requires_all_remaining_simulations():
    assert correction_times([False, True, False, True, True]) == (2, 4)


def test_hysteresis_classification_requires_64_final_simulations():
    metrics = {
        "visit_top_exact_final": False,
        "q_top_exact_final": True,
        "stable_q_correct_sim": 384 - STABILITY_WINDOW,
    }
    assert classify(metrics) == "visit_hysteresis_after_q_recovery"
    metrics["stable_q_correct_sim"] = 384 - STABILITY_WINDOW + 1
    assert classify(metrics) == "late_q_recovery_insufficient_time"
