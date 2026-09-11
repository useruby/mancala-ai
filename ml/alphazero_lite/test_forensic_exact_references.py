import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.forensic_exact_references import (
    SCHEMA,
    exact_regret,
    validate_v2,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.run_forensic_suite import build_row


class ExactForensicReferenceTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "player_pits": [1, 1, 0, 0, 0, 0],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 10,
            "opponent_store": 10,
            "current_player": 0,
        }
        self.position = type(
            "Position",
            (),
            {
                "id": "capture_available-001",
                "state": self.state,
                "canonical_key": canonical_state_key(self.state),
                "side_to_move": 0,
                "legal_moves": (0, 1),
                "phase": "late",
                "bucket": "capture_available",
                "tags": ("capture_available",),
                "source": "seed",
            },
        )()
        self.exact = {
            "oracle_kind": "exact",
            "exact_status": "exact_solved",
            "state": self.state,
            "exact_root_value": 1.0,
            "exact_action_values": {"0": 4, "1": 4},
            "exact_optimal_actions": [0, 1],
        }

    def test_exact_optimal_set_accepts_ties_and_zero_regret(self):
        row = build_row(
            position=self.position,
            reference=self.exact,
            system={"selected_move": 1, "value": 1.0},
        )
        self.assertTrue(row["agrees_top1"])
        self.assertEqual(0.0, row["regret"])

    def test_exact_regret_respects_root_player_and_illegal_actions(self):
        self.exact["exact_action_values"] = {"0": 4, "1": 1}
        self.assertEqual(3.0, exact_regret(self.exact, 1))
        self.assertIsNone(exact_regret(self.exact, 5))
        self.exact["state"] = self.state | {"current_player": 1}
        self.assertEqual(3.0, exact_regret(self.exact, 0))

    def test_unresolved_row_has_no_scored_metrics(self):
        row = build_row(
            position=self.position,
            reference={
                "oracle_kind": "exact",
                "exact_status": "unresolved",
                "exact_root_value": None,
                "exact_action_values": None,
                "exact_optimal_actions": None,
            },
            system={"selected_move": 0, "value": 0.2},
        )
        self.assertIsNone(row["agrees_top1"])
        self.assertIsNone(row["regret"])
        self.assertIsNone(row["value_error"])

    def test_v2_validation_rejects_manufactured_unresolved_label(self):
        suite_row = {
            "id": self.position.id,
            "state": self.state,
            "side_to_move": 0,
            "legal_moves": [0, 1],
            "phase": "late",
            "bucket": "capture_available",
            "tags": ["capture_available"],
            "source": "seed",
        }
        artifact = {
            "schema": SCHEMA,
            "rows": [
                {
                    "id": self.position.id,
                    "canonical_state": self.position.canonical_key,
                    "state": self.state,
                    "legal_moves": [0, 1],
                    "oracle_kind": "exact",
                    "exact_status": "unresolved",
                    "exact_root_value": 1.0,
                    "exact_action_values": None,
                    "exact_optimal_actions": None,
                    "oracle_provenance": {},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.json"
            suite.write_text(json.dumps([suite_row]), encoding="utf-8")
            self.assertTrue(validate_v2(artifact, suite))


if __name__ == "__main__":
    unittest.main()
