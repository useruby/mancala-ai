from __future__ import annotations

import unittest
from unittest import mock
from typing import Any


class BuildOpening017CorrectiveArtifactTest(unittest.TestCase):
    def _raw_state(self, *, player_store: int) -> dict[str, Any]:
        return {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": player_store,
            "opponent_store": 0,
            "current_player": 0,
        }

    def _encoded_state(self, raw_state: dict[str, Any]) -> list[float]:
        return [
            *[
                float(value) / 48.0
                for value in list(raw_state["player_pits"])
                + list(raw_state["opponent_pits"])
            ],
            float(raw_state["player_store"]) / 48.0,
            float(raw_state["opponent_store"]) / 48.0,
            float(raw_state["current_player"]),
        ]

    def _reference_rows(self, module: Any) -> dict[str, dict[str, Any]]:
        reference_moves = {
            "opening_plies_1_8-017": 2,
            "capture_available-002": 2,
            "capture_available-003": 3,
            "capture_available-007": 4,
            "capture_available-006": 5,
            "capture_available-008": 0,
        }
        reference_rows = {}
        for index, row_id in enumerate(module.TARGET_ROW_IDS):
            raw_state = self._raw_state(player_store=index)
            reference_rows[row_id] = {
                "id": row_id,
                "canonical_state": module.canonical_state_key(raw_state),
                "state": raw_state,
                "reference_move": reference_moves[row_id],
                "teacher_value": 0.25 + (0.05 * index),
                "reference_source": f"fixture-{row_id}",
                "child_stats": [
                    {"move": reference_moves[row_id], "visits": 90, "win_rate": 0.8},
                    {
                        "move": (reference_moves[row_id] + 1) % 6,
                        "visits": 10,
                        "win_rate": 0.4,
                    },
                ],
                "reference_artifact_path": "/tmp/reference.json",
                "reference_artifact_kind": "primary",
            }
        reference_rows["capture_available-006"]["reference_artifact_path"] = (
            "/tmp/fallback.json"
        )
        reference_rows["capture_available-006"]["reference_artifact_kind"] = "fallback"
        return reference_rows

    def test_validate_corrective_rows_ignores_missing_optional_controls(self) -> None:
        from ml.alphazero_lite import build_opening_017_corrective_artifact as module

        rows = []
        for row_id in module.REQUIRED_ROW_IDS:
            rows.append(
                {
                    "canonical_state": f"canon-{row_id}",
                    "legal_moves": [0, 1, 2, 3, 4, 5],
                    "policy": [0.025, 0.025, 0.875, 0.025, 0.025, 0.025],
                    "corrected_reference_move": 2,
                    "source_runs": [{"id": row_id}],
                    "train_only": True,
                    "exclude_from_validation": True,
                    "reference_artifact_kind": "primary",
                }
            )

        validation = module.validate_corrective_rows(rows)

        self.assertEqual("ok", validation["status"])
        self.assertTrue(validation["all_required_rows_present"])
        self.assertEqual(["ok"], validation["notes"])

    def test_build_failure_chain_rows_sets_expected_diagnoses(self) -> None:
        from ml.alphazero_lite import build_opening_017_corrective_artifact as module

        reference_rows = self._reference_rows(module)
        policies = {
            "opening_plies_1_8-017": [0.05, 0.1, 0.1, 0.1, 0.55, 0.1],
            "capture_available-002": [0.05, 0.6, 0.1, 0.1, 0.1, 0.05],
            "capture_available-003": [0.05, 0.1, 0.55, 0.1, 0.1, 0.1],
            "capture_available-007": [0.05, 0.55, 0.1, 0.1, 0.1, 0.1],
        }
        self_play_rows = []
        for row_id, policy in policies.items():
            self_play_rows.append(
                {
                    "state": self._encoded_state(reference_rows[row_id]["state"]),
                    "policy": policy,
                    "teacher_source": "pr36-reuse",
                    "move_index": 7,
                }
            )

        extracted_rows = module.build_failure_chain_rows(
            self_play_rows=self_play_rows,
            reference_rows=reference_rows,
        )
        extracted_by_id = {row["row_id"]: row for row in extracted_rows}

        self.assertEqual(
            "predecessor_target_drift",
            extracted_by_id["opening_plies_1_8-017"]["diagnosis"],
        )
        self.assertEqual(
            "descendant_shift_to_move_1",
            extracted_by_id["capture_available-002"]["diagnosis"],
        )
        self.assertEqual(
            "descendant_target_noise",
            extracted_by_id["capture_available-003"]["diagnosis"],
        )
        self.assertEqual(
            "descendant_shift_to_move_1",
            extracted_by_id["capture_available-007"]["diagnosis"],
        )
        self.assertEqual(
            "control_reference_only",
            extracted_by_id["capture_available-006"]["diagnosis"],
        )
        self.assertEqual(0, extracted_by_id["capture_available-006"]["self_play_count"])

    def test_build_corrective_rows_uses_corrective_policy_and_reference_source(
        self,
    ) -> None:
        from ml.alphazero_lite import build_opening_017_corrective_artifact as module

        reference_rows = self._reference_rows(module)
        extracted_rows = [
            {
                "row_id": row_id,
                "averaged_self_play_top_move": 1,
                "corrected_reference_mass": 0.2,
            }
            for row_id in module.TARGET_ROW_IDS
        ]
        reference_artifact = {
            "reference": {
                "policy_simulations": 1200,
                "value_simulations": 1800,
                "sample_seeds": [2040],
            }
        }
        fallback_reference_artifact = {
            "reference": {
                "policy_simulations": 384,
                "value_simulations": 768,
                "sample_seeds": [2041],
            }
        }

        with mock.patch.object(
            module,
            "encode_raw_state",
            side_effect=lambda *, raw_state, input_encoding: {
                "encoded": raw_state,
                "input_encoding": input_encoding,
            },
        ):
            rows = module.build_corrective_rows(
                extracted_rows=extracted_rows,
                reference_rows=reference_rows,
                reference_artifact=reference_artifact,
                fallback_reference_artifact=fallback_reference_artifact,
                input_encoding="kalah_v3",
                policy_target_mode="sharpened",
                value_target_mode="sharpened",
            )

        rows_by_id = {row["source_runs"][0]["id"]: row for row in rows}
        opening_row = rows_by_id["opening_plies_1_8-017"]
        fallback_row = rows_by_id["capture_available-006"]
        residual = module.NON_REFERENCE_POLICY_MASS / 5.0

        self.assertEqual(6, len(rows))
        self.assertAlmostEqual(module.CORRECTED_POLICY_MASS, opening_row["policy"][2])
        self.assertAlmostEqual(residual, opening_row["policy"][0])
        self.assertEqual(True, opening_row["train_only"])
        self.assertEqual(True, opening_row["exclude_from_validation"])
        self.assertEqual(2040, opening_row["teacher_seed"])
        self.assertEqual("primary", opening_row["reference_artifact_kind"])
        self.assertEqual(["/tmp/reference.json"], opening_row["source_artifacts"])
        self.assertEqual(2041, fallback_row["teacher_seed"])
        self.assertEqual("fallback", fallback_row["reference_artifact_kind"])
        self.assertEqual(["/tmp/fallback.json"], fallback_row["source_artifacts"])


if __name__ == "__main__":
    unittest.main()
