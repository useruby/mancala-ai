import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class BuildTrackedForensicReferenceArtifactTest(unittest.TestCase):
    def suite_rows(self):
        return [
            {
                "id": "capture_available-002",
                "state": {
                    "player_pits": [1, 0, 7, 6, 6, 5],
                    "opponent_pits": [5, 4, 4, 4, 4, 0],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 1,
                },
                "side_to_move": 1,
                "legal_moves": [0, 2, 3, 4, 5],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "seed", "ply_4"],
                "source": "seed",
            },
            {
                "id": "capture_available-003",
                "state": {
                    "player_pits": [1, 6, 0, 6, 6, 5],
                    "opponent_pits": [5, 5, 4, 4, 4, 0],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 1,
                },
                "side_to_move": 1,
                "legal_moves": [0, 1, 3, 4, 5],
                "phase": "opening",
                "bucket": "capture_available",
                "tags": ["capture_available", "seed", "ply_4"],
                "source": "seed",
            },
        ]

    def test_builder_merges_audited_override_rows(self):
        from ml.alphazero_lite.build_tracked_forensic_reference_artifact import (
            build_tracked_reference_artifact,
        )
        from ml.alphazero_lite.forensic_suite import canonical_state_key

        suite_rows = self.suite_rows()
        state_002 = suite_rows[0]["state"]
        state_003 = suite_rows[1]["state"]
        canonical_002 = canonical_state_key(state_002)
        canonical_003 = canonical_state_key(state_003)

        def fake_build_reference_artifact(**kwargs):
            out_path = Path(kwargs["out_path"])
            payload = {
                "schema": "azlite_forensic_references_v1",
                "suite_path": str(kwargs["suite_path"]),
                "reference": {
                    "policy_simulations": 1200,
                    "value_simulations": 1800,
                    "sample_seeds": [2040],
                },
                "rows": [
                    {
                        "id": "capture_available-002",
                        "canonical_state": canonical_002,
                        "state": state_002,
                        "reference_move": 0,
                        "teacher_value": 0.1,
                        "reference_unstable": False,
                        "observed_reference_moves": [0],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 0, "teacher_value": 0.1}
                        ],
                    },
                    {
                        "id": "capture_available-003",
                        "canonical_state": canonical_003,
                        "state": state_003,
                        "reference_move": 1,
                        "teacher_value": 0.2,
                        "reference_unstable": False,
                        "observed_reference_moves": [1],
                        "seed_samples": [
                            {"seed": 2040, "reference_move": 1, "teacher_value": 0.2}
                        ],
                    },
                ],
            }
            out_path.write_text(json.dumps(payload), encoding="utf-8")
            return payload["rows"]

        override_payload = {
            "schema": "azlite_forensic_references_v1",
            "rows": [
                {
                    "id": "capture_available-002",
                    "canonical_state": canonical_002,
                    "state": state_002,
                    "reference_move": 2,
                    "teacher_value": 0.7048,
                    "reference_unstable": False,
                    "observed_reference_moves": [2],
                    "seed_samples": [
                        {"seed": 11, "reference_move": 2, "teacher_value": 0.7048}
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            suite_path = tmp_path / "suite.json"
            out_path = tmp_path / "references.json"
            override_path = tmp_path / "overrides.json"
            suite_path.write_text(json.dumps(suite_rows), encoding="utf-8")
            override_path.write_text(json.dumps(override_payload), encoding="utf-8")

            with mock.patch(
                "ml.alphazero_lite.build_tracked_forensic_reference_artifact.build_reference_artifact",
                side_effect=fake_build_reference_artifact,
            ):
                payload = build_tracked_reference_artifact(
                    suite_path=suite_path,
                    out_path=out_path,
                    override_artifact_path=override_path,
                    policy_simulations=1200,
                    value_simulations=1800,
                    seed=2040,
                    sample_seeds=None,
                )

        self.assertEqual(2, len(payload["rows"]))
        self.assertEqual("audited_override", payload["rows"][0]["reference_source"])
        self.assertEqual(2, payload["rows"][0]["reference_move"])
        self.assertEqual("generated", payload["rows"][1]["reference_source"])
        self.assertEqual(1, payload["rows"][1]["reference_move"])


if __name__ == "__main__":
    unittest.main()
