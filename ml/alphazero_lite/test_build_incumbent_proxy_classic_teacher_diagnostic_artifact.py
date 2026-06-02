from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class BuildIncumbentProxyClassicTeacherDiagnosticArtifactTest(unittest.TestCase):
    def _state(self, index: int) -> dict[str, object]:
        return {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": index,
            "opponent_store": 0,
            "current_player": index % 2,
        }

    def test_build_artifact_constructs_valid_classic_rows(self) -> None:
        from ml.alphazero_lite import (
            build_incumbent_proxy_classic_teacher_diagnostic_artifact as module,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            current_artifact = tmp_path / "current"
            current_artifact.mkdir()
            (current_artifact / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v3"}) + "\n",
                encoding="utf-8",
            )
            reference_path = tmp_path / "references.json"

            suite_rows = []
            source_rows = []
            reference_rows = []
            reference_moves = {
                "incumbent_proxy_disagreement-014": 4,
                "incumbent_proxy_disagreement-022": 3,
                "incumbent_proxy_disagreement-024": 2,
                "incumbent_proxy_disagreement-025": 2,
                "incumbent_proxy_disagreement-035": 3,
                "incumbent_proxy_disagreement-008": 2,
            }

            for index, row_id in enumerate(module.EXPECTED_ROW_IDS):
                state = self._state(index)
                reference_move = reference_moves[row_id]
                canonical = module.canonical_state_key(state)
                suite_rows.append(
                    SimpleNamespace(
                        id=row_id,
                        state=state,
                        legal_moves=[0, 1, 2, 3, 4, 5],
                    )
                )
                source_rows.append(
                    {
                        "row_id": row_id,
                        "preferred_teacher": "classic_mcts",
                        "preferred_move": reference_move,
                        "evidence_summary": f"evidence-{row_id}",
                    }
                )
                source_rows[-1]["recommended_role"] = module.TARGET_ROLE_BY_ID[row_id]
                reference_rows.append(
                    {
                        "id": row_id,
                        "canonical_state": canonical,
                        "state": state,
                        "reference_move": reference_move,
                        "teacher_value": 0.1 + (0.1 * index),
                        "reference_source": "generated",
                        "child_stats": [
                            {"move": reference_move, "visits": 80, "win_rate": 0.8},
                            {
                                "move": (reference_move + 1) % 6,
                                "visits": 20,
                                "win_rate": 0.4,
                            },
                        ],
                    }
                )

            reference_path.write_text(
                json.dumps(
                    {
                        "reference": {
                            "policy_simulations": 1200,
                            "value_simulations": 1800,
                        },
                        "rows": reference_rows,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            args = argparse.Namespace(
                reference_path=reference_path,
                suite_path=tmp_path / "suite.json",
                current_artifact=current_artifact,
                classic_rows_path=tmp_path / "classic_rows.jsonl",
                puct_rows_path=tmp_path / "puct_rows.jsonl",
                excluded_rows_path=tmp_path / "excluded_rows.jsonl",
                bucket_summary_path=tmp_path / "summary.json",
                report_path=tmp_path / "report.md",
                output_dir=tmp_path / "output",
                artifact_out=tmp_path / "output" / "artifact.jsonl",
                summary_out=tmp_path / "output" / "artifact_summary.json",
                target_only_out=tmp_path / "output" / "targets.jsonl",
                control_only_out=tmp_path / "output" / "controls.jsonl",
                input_encoding="kalah_v3",
                policy_target_mode="default",
                value_target_mode="default",
            )

            with (
                mock.patch.object(
                    module,
                    "load_source_classic_rows",
                    return_value=(source_rows, "mocked_source_rows.jsonl"),
                ),
                mock.patch.object(module, "load_bucket_rows", return_value=[]),
                mock.patch.object(module, "load_suite", return_value=suite_rows),
            ):
                built = module.build_artifact(args)

            summary = built["summary"]
            self.assertEqual("ok", summary["validation"]["status"])
            self.assertEqual(6, summary["validation"]["row_count"])
            self.assertEqual(5, summary["validation"]["target_candidate_count"])
            self.assertEqual(1, summary["validation"]["preservation_control_count"])
            self.assertEqual([], summary["excluded_rows"])

            rows_by_id = {
                row["source_runs"][0]["id"]: row for row in built["artifact_rows"]
            }
            control_row = rows_by_id["incumbent_proxy_disagreement-008"]
            self.assertEqual("preservation_control", control_row["role"])
            self.assertAlmostEqual(1.0, sum(control_row["policy"]))
            self.assertEqual(
                0.85, control_row["policy"][control_row["active_reference_move"]]
            )

            target_row = rows_by_id["incumbent_proxy_disagreement-014"]
            self.assertEqual("target_candidate", target_row["role"])
            self.assertEqual(
                "incumbent_proxy_classic_teacher_diagnostic", target_row["source"]
            )
            self.assertTrue(target_row["train_only"])
            self.assertTrue(target_row["exclude_from_validation"])


if __name__ == "__main__":
    unittest.main()
