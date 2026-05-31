from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class RunRuleConditionedOpeningFullGuardedExperimentTest(unittest.TestCase):
    def test_load_opening_subfamily_filter_groups_row_ids(self) -> None:
        from ml.alphazero_lite import (
            run_rule_conditioned_opening_full_guarded_experiment as module,
        )

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-subfamily-filter-"
        ) as tmp:
            path = Path(tmp) / "opening_subfamilies.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": module.OPENING_SUBFAMILY_DIAGNOSTIC_SCHEMA,
                        "rows": [
                            {
                                "row_id": "opening_plies_1_8-004",
                                "subfamily": "opening_extra_turn_overbias",
                            },
                            {
                                "row_id": "opening_plies_1_8-007",
                                "subfamily": "opening_extra_turn_overbias",
                            },
                            {
                                "row_id": "opening_plies_1_8-014",
                                "subfamily": "opening_edge_move_5_preference",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = module.load_opening_subfamily_filter(path)

        self.assertEqual(str(path), payload["path"])
        self.assertEqual(
            {
                "opening_edge_move_5_preference": ["opening_plies_1_8-014"],
                "opening_extra_turn_overbias": [
                    "opening_plies_1_8-004",
                    "opening_plies_1_8-007",
                ],
            },
            payload["rows_by_subfamily"],
        )

    def test_load_opening_subfamily_filter_rejects_wrong_schema(self) -> None:
        from ml.alphazero_lite import (
            run_rule_conditioned_opening_full_guarded_experiment as module,
        )

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-subfamily-filter-"
        ) as tmp:
            path = Path(tmp) / "opening_subfamilies.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "wrong_schema",
                        "rows": [],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SystemExit):
                module.load_opening_subfamily_filter(path)

    def test_materialize_filtered_opening_artifact_writes_filtered_paths(self) -> None:
        from ml.alphazero_lite import (
            run_rule_conditioned_opening_full_guarded_experiment as module,
        )

        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-subfamily-materialize-"
        ) as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "artifact.jsonl"
            artifact_summary_path = tmp_path / "artifact_summary.json"
            diagnostic_path = tmp_path / "subfamilies.json"
            run_root = tmp_path / "run"

            artifact_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_runs": [{"id": "opening_plies_1_8-014"}],
                                "replay_role": "opening_capture_no_extra_turn_reference",
                                "teacher_selected_move": 4,
                            }
                        ),
                        json.dumps(
                            {
                                "source_runs": [{"id": "capture_available-002"}],
                                "replay_role": "rule_collision_no_extra_turn_reference_guard",
                                "teacher_selected_move": 2,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            artifact_summary_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_rule_conditioned_opening_family_full_guarded_artifact_summary_v1",
                        "row_count": 2,
                        "rule_collision_guard_row_ids": ["capture_available-002"],
                    }
                ),
                encoding="utf-8",
            )
            diagnostic_path.write_text(
                json.dumps(
                    {
                        "schema": module.OPENING_SUBFAMILY_DIAGNOSTIC_SCHEMA,
                        "rows": [
                            {
                                "row_id": "opening_plies_1_8-014",
                                "subfamily": "opening_edge_move_5_preference",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            filtered_artifact_path, filtered_summary_path = (
                module.materialize_filtered_opening_artifact(
                    root=Path(__file__).resolve().parents[2],
                    artifact_path=artifact_path,
                    artifact_summary_path=artifact_summary_path,
                    opening_subfamily_diagnostic_path=diagnostic_path,
                    opening_subfamily="opening_edge_move_5_preference",
                    run_root=run_root,
                    python=sys.executable,
                    dry_run=False,
                )
            )

            self.assertTrue(filtered_artifact_path.exists())
            self.assertTrue(filtered_summary_path.exists())
            summary = json.loads(filtered_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "opening_edge_move_5_preference",
                summary["filtered_opening_subfamily"],
            )


if __name__ == "__main__":
    unittest.main()
