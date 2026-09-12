from ml.alphazero_lite.forensic_exact_references import exact_regret
from ml.alphazero_lite.run_root_fpu_capture_diagnostic import (
    assert_baseline,
    first_visit_simulations,
    transition,
)


def test_first_visit_simulation_extraction_is_stable():
    history = [
        {"simulation": 1, "action": 2},
        {"simulation": 2, "action": 1},
        {"simulation": 3, "action": 2},
    ]
    assert first_visit_simulations(history, [0, 1, 2]) == {
        "0": None,
        "1": 2,
        "2": 1,
    }


def test_raw_to_search_transition_classification():
    assert transition(12, 0) == "raw_wrong_to_search_correct"
    assert transition(0, 12) == "raw_correct_to_search_wrong"


def test_exact_regret_and_pr298_baseline_fixture():
    exact = {
        "exact_status": "exact_solved",
        "state": {"current_player": 1},
        "exact_action_values": {"1": -4, "2": -16},
    }
    assert exact_regret(exact, 1) == 12
    expected = {
        "search": {
            "selected_action": 2,
            "regret": 0,
            "visits": [0, 1, 2, 0, 0, 0],
            "root_prior": [0, 0.5, 0.5, 0, 0, 0],
            "child_q": [
                {"move": 1, "q_value": 0.1},
                {"move": 2, "q_value": 0.2},
            ],
        }
    }
    actual = {
        "selected_action": 2,
        "regret": 0,
        "visits": [0, 1, 2, 0, 0, 0],
        "root_prior": [0, 0.5, 0.5, 0, 0, 0],
        "child_q": {"1": 0.1, "2": 0.2},
    }
    assert_baseline(actual, expected)


def test_runner_is_explicitly_read_only():
    source = __import__(
        "ml.alphazero_lite.run_root_fpu_capture_diagnostic", fromlist=["__doc__"]
    )
    assert source.__doc__.startswith("Read-only ROOT-only FPU")
