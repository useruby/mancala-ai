from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_BIN = Path(sys.executable)


class FilterOpeningFamilyArtifactBySubfamilyTest(unittest.TestCase):
    def test_filter_rows_keeps_selected_opening_rows_and_all_guards(self) -> None:
        from ml.alphazero_lite import (
            filter_opening_family_artifact_by_subfamily as module,
        )

        rows, kept = module.filter_rows(
            artifact_rows=[
                {
                    "source_runs": [{"id": "opening_plies_1_8-004"}],
                    "replay_role": "opening_capture_no_extra_turn_reference",
                },
                {
                    "source_runs": [{"id": "opening_plies_1_8-014"}],
                    "replay_role": "opening_capture_no_extra_turn_reference",
                },
                {
                    "source_runs": [{"id": "capture_available-002"}],
                    "replay_role": "rule_collision_no_extra_turn_reference_guard",
                },
            ],
            selected_row_ids={"opening_plies_1_8-014"},
        )

        self.assertEqual(2, len(rows))
        self.assertEqual(["opening_plies_1_8-014"], kept)

    def test_cli_filters_artifact_and_updates_summary(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="azlite-filter-opening-subfamily-"
        ) as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "opening.jsonl"
            summary_path = tmp_path / "opening_summary.json"
            diagnostic_path = tmp_path / "subfamilies.json"
            out_path = tmp_path / "filtered.jsonl"
            out_summary_path = tmp_path / "filtered_summary.json"

            artifact_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "source_runs": [{"id": "opening_plies_1_8-004"}],
                                "replay_role": "opening_capture_no_extra_turn_reference",
                                "teacher_selected_move": 4,
                            }
                        ),
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
            summary_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_rule_conditioned_opening_family_full_guarded_artifact_summary_v1",
                        "row_count": 3,
                        "rule_collision_guard_row_ids": ["capture_available-002"],
                    }
                ),
                encoding="utf-8",
            )
            diagnostic_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_opening_plies_subfamily_diagnostic_v1",
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

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.filter_opening_family_artifact_by_subfamily",
                    "--artifact",
                    str(artifact_path),
                    "--artifact-summary",
                    str(summary_path),
                    "--opening-subfamily-diagnostic",
                    str(diagnostic_path),
                    "--subfamily",
                    "opening_edge_move_5_preference",
                    "--out",
                    str(out_path),
                    "--out-summary",
                    str(out_summary_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            filtered_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            filtered_summary = json.loads(out_summary_path.read_text(encoding="utf-8"))

            self.assertEqual(2, len(filtered_rows))
            self.assertEqual(
                ["opening_plies_1_8-014"],
                filtered_summary["filtered_opening_row_ids"],
            )
            self.assertEqual(
                "opening_edge_move_5_preference",
                filtered_summary["filtered_opening_subfamily"],
            )
            self.assertTrue(filtered_summary["filtered"])


if __name__ == "__main__":
    unittest.main()
