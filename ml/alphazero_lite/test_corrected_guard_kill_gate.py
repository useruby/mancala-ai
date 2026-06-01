import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite import corrected_guard_kill_gate as module


def reference_row(row_id: str, reference_move: int) -> dict:
    return {
        "id": row_id,
        "canonical_state": f"canonical-{row_id}",
        "state": {
            "player_pits": [1, 1, 1, 1, 1, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        "reference_move": reference_move,
        "child_stats": [
            {"move": move, "visits": 1, "q_value": 0.0} for move in range(6)
        ],
    }


class CorrectedGuardKillGateTest(unittest.TestCase):
    def test_parse_budgets_rejects_schema_incompatible_values(self):
        with self.assertRaisesRegex(ValueError, "exactly 384,1200"):
            module.parse_budgets("384")

    def test_validate_reference_rows_uses_fallback_for_missing_row(self):
        with tempfile.TemporaryDirectory(prefix="azlite-guard-gate-") as tmp:
            tmp_path = Path(tmp)
            reference_path = tmp_path / "reference.json"
            fallback_path = tmp_path / "fallback.json"

            reference_rows = [
                reference_row("capture_available-002", 2),
                reference_row("capture_available-003", 2),
                reference_row("capture_available-007", 2),
                reference_row("capture_available-008", 1),
            ]
            fallback_rows = [reference_row("capture_available-006", 2)]

            reference_path.write_text(
                json.dumps({"rows": reference_rows}), encoding="utf-8"
            )
            fallback_path.write_text(
                json.dumps({"rows": fallback_rows}), encoding="utf-8"
            )

            validation, rows = module.validate_reference_rows(
                reference_path,
                fallback_reference_artifact=fallback_path,
            )

        self.assertEqual("ok", validation["status"])
        self.assertFalse(validation["missing_metadata"])
        self.assertIn("capture_available-006", rows)
        self.assertEqual(2, rows["capture_available-006"]["corrected_reference_move"])

    def test_run_corrected_guard_kill_gate_rejects_non_reference_selection(self):
        with tempfile.TemporaryDirectory(prefix="azlite-guard-gate-") as tmp:
            tmp_path = Path(tmp)
            reference_path = tmp_path / "reference.json"
            candidate_path = tmp_path / "candidate"
            candidate_path.mkdir()

            rows = [
                reference_row("capture_available-002", 2),
                reference_row("capture_available-003", 2),
                reference_row("capture_available-006", 2),
                reference_row("capture_available-007", 2),
                reference_row("capture_available-008", 1),
            ]
            for index, row in enumerate(rows):
                row["state"]["player_store"] = index
            reference_path.write_text(json.dumps({"rows": rows}), encoding="utf-8")

            class FakeEvaluator:
                def __init__(self, artifact_path: Path):
                    self.artifact_path = artifact_path

            reference_move_by_row_tag = {
                row["state"]["player_store"]: row["reference_move"] for row in rows
            }

            def fake_probe(*, state, simulations, **kwargs):
                del kwargs
                row_tag = state["player_store"]
                if row_tag == 0 and simulations == 384:
                    selected_move = 1
                elif row_tag == 1 and simulations == 384:
                    selected_move = 1
                else:
                    selected_move = reference_move_by_row_tag[row_tag]
                visits = [0.0] * 6
                visits[selected_move] = 60.0
                reference_move = reference_move_by_row_tag[row_tag]
                visits[reference_move] += (
                    40.0 if selected_move != reference_move else 0.0
                )
                return {
                    "selected_move": selected_move,
                    "visits": visits,
                    "child_stats": [
                        {"move": move, "q_value": 0.0} for move in range(6)
                    ],
                }

            with (
                mock.patch.object(module, "ArtifactEvaluator", FakeEvaluator),
                mock.patch.object(
                    module, "evaluate_artifact_position", side_effect=fake_probe
                ),
            ):
                payload = module.run_corrected_guard_kill_gate(
                    candidate_path=candidate_path,
                    reference_artifact=reference_path,
                    fallback_reference_artifact=None,
                )

        self.assertFalse(payload["pass"])
        self.assertEqual("reject_guard_regression", payload["decision"])
        failed_rows = [row for row in payload["rows"] if not row["pass"]]
        self.assertEqual(
            ["capture_available-002", "capture_available-003"],
            [row["row_id"] for row in failed_rows],
        )


if __name__ == "__main__":
    unittest.main()
