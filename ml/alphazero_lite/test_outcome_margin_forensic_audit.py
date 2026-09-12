import json

from ml.alphazero_lite.forensic_exact_references import (
    exact_outcome_utility,
    exact_regret,
    outcome_optimal_actions,
    outcome_regret,
    outcome_regression,
    outcome_utilities,
    same_outcome_margin_regression,
)
from ml.alphazero_lite.run_outcome_margin_forensic_audit import ROOT, audit_exact_row


def exact_row(current_player: int = 0) -> dict:
    return {
        "exact_status": "exact_solved",
        "state": {"current_player": current_player},
        "exact_action_values": {"0": 4, "1": 0, "2": -3},
        "exact_optimal_actions": [0] if current_player == 0 else [2],
        "exact_root_value": 1.0,
    }


def test_margin_to_outcome_and_root_perspective_conversion():
    assert exact_outcome_utility(margin=4, current_player=0) == 1
    assert exact_outcome_utility(margin=0, current_player=0) == 0
    assert exact_outcome_utility(margin=-4, current_player=0) == -1
    assert exact_outcome_utility(margin=4, current_player=1) == -1
    assert exact_outcome_utility(margin=-4, current_player=1) == 1
    assert set(outcome_utilities(exact_row()).values()) <= {-1, 0, 1}


def test_outcome_optimal_actions_and_regret_classes():
    row = exact_row()
    assert outcome_optimal_actions(row) == [0]
    assert outcome_regret(row, 0) == 0
    assert outcome_regret(row, 1) == 1
    assert outcome_regret(row, 2) == 2
    assert outcome_regression(row, 1)
    assert not same_outcome_margin_regression(row, 1)


def test_player_one_minimization_retains_same_outcome_actions():
    row = {
        "exact_status": "exact_solved",
        "state": {"current_player": 1},
        "exact_action_values": {"0": 2, "1": -4, "2": -16, "3": -10, "4": -6},
        "exact_optimal_actions": [2],
        "exact_root_value": 1.0,
    }
    assert outcome_optimal_actions(row) == [1, 2, 3, 4]
    assert outcome_regret(row, 1) == 0
    assert same_outcome_margin_regression(row, 1)
    assert exact_regret(row, 1) == 12.0


def test_capture_002_fixture_and_exact_root_value_consistency():
    rows = json.loads(
        (
            ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
        ).read_text()
    )["rows"]
    row = next(row for row in rows if row["id"] == "capture_available-002")
    assert outcome_optimal_actions(row) == [1, 2, 3, 4]
    audited = audit_exact_row(row)
    assert max(audited["outcome_utilities"].values()) == row["exact_root_value"]


def test_strict_margin_regret_remains_unchanged():
    row = exact_row()
    assert exact_regret(row, 2) == 7.0


def test_runner_is_read_only_and_reuses_historical_selected_moves():
    source = (
        ROOT / "ml/alphazero_lite/run_outcome_margin_forensic_audit.py"
    ).read_text()
    assert 'selected_metrics(exact, int(row["selected_move"]))' in source
    assert '"mcts_rerun": False' in source
    assert '"training": False' in source
    assert '"self_play": False' in source
    assert '"promotion": False' in source
    assert "from ml.alphazero_lite.self_play" not in source
    assert "production_gate" not in source.replace(
        '"production_gate_modified": False', ""
    )
