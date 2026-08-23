from __future__ import annotations

from ml.alphazero_lite.run_fresh_p1_adapter_candidate_q_confidence import (
    reconstruct_backup_stats,
    snapshot_features,
)


def _summary(*, p1: bool = False) -> dict:
    history = [
        {"simulation": 1, "action": 0, "root_value": 0.2},
        {"simulation": 2, "action": 1, "root_value": -0.4},
        {"simulation": 3, "action": 0, "root_value": 0.6},
        {"simulation": 4, "action": 2, "root_value": 0.1},
    ]
    if p1:
        history = [
            {"simulation": 1, "action": 0, "root_value": -0.1},
            {"simulation": 2, "action": 1, "root_value": 0.7},
            {"simulation": 3, "action": 0, "root_value": -0.3},
            {"simulation": 4, "action": 2, "root_value": 0.2},
        ]

    def moves(
        values: tuple[float, float, float], visits: tuple[int, int, int]
    ) -> list[dict]:
        return [
            {
                "move": 0,
                "visit_count": visits[0],
                "stored_q_value": values[0],
                "selection_score": 0.7,
                "u_component": 0.2,
                "prior": 0.5,
            },
            {
                "move": 1,
                "visit_count": visits[1],
                "stored_q_value": values[1],
                "selection_score": 0.6,
                "u_component": 0.15,
                "prior": 0.3,
            },
            {
                "move": 2,
                "visit_count": visits[2],
                "stored_q_value": values[2],
                "selection_score": 0.1,
                "u_component": 0.05,
                "prior": 0.2,
            },
        ]

    snapshot_values = (
        ((-0.1, 0.7, 0.9), (-0.2, 0.7, 1.0))
        if p1
        else ((0.2, -0.4, 0.0), (0.4, -0.4, 0.1))
    )
    return {
        "selected_move": 0,
        "root_backup_history": history,
        "root_snapshots": [
            {
                "simulation": 2,
                "selected_move": 0,
                "moves": moves(snapshot_values[0], (1, 1, 0)),
            },
            {
                "simulation": 4,
                "selected_move": 0,
                "moves": moves(snapshot_values[1], (2, 1, 1)),
            },
        ],
    }


def test_reconstruction_reports_exact_count_and_q_contract() -> None:
    result = reconstruct_backup_stats(_summary())

    assert result["checkpoints"][0]["counts_match"]
    assert result["checkpoints"][0]["q_match"]
    assert result["checkpoints"][1]["actions"][0]["value_sum"] == 0.8
    assert result["valid"] is False  # Fixture intentionally is not a 1200-search.


def test_primary_features_cover_all_actions_and_all_action_counterfactual() -> None:
    a16, p1 = _summary(), _summary(p1=True)
    rows = snapshot_features(a16, p1)
    final = rows[-1]
    by_move = {row["move"]: row for row in final["actions"]}

    assert [row["move"] for row in final["candidate_selection_pair"]] == [0, 1]
    assert final["counterfactual_eligible_actions"] == [0, 1, 2]
    assert final["p1_q_counterfactual_move"] == 2
    assert by_move[2]["p1_q_counterfactual_score"] == 1.05
    assert by_move[0]["q_gap_z"] is not None
    assert {
        "std",
        "mad",
        "recent_drift_8",
        "recent_drift_32",
        "q_change_from_previous_checkpoint",
        "last_visit_age",
    } <= by_move[0].keys()
    assert by_move[2]["future_q_order_1200"] == 2
    assert by_move[0]["policy_l1_vs_p1"] == 0.0
