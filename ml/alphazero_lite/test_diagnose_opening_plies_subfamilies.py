import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PYTHON_BIN = Path(sys.executable)


def forensic_report(rows: list[dict]) -> dict:
    return {
        "schema": "azlite_forensic_suite_v1",
        "systems": {
            "current": {
                "rows": rows,
            }
        },
    }


class DiagnoseOpeningPliesSubfamiliesTest(unittest.TestCase):
    def test_classify_opening_failure_subfamilies(self):
        from ml.alphazero_lite import diagnose_opening_plies_subfamilies as module

        extra_turn_overbias = {
            "id": "opening_plies_1_8-004",
            "bucket": "opening_plies_1_8",
            "state": {
                "player_pits": [4, 0, 5, 5, 5, 5],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 1,
            },
            "reference_move": 4,
            "selected_move": 2,
            "regret": 0.0059,
            "value_error": 0.5074,
        }
        missed_extra_turn = {
            "id": "opening_plies_1_8-009",
            "bucket": "opening_plies_1_8",
            "state": {
                "player_pits": [0, 5, 5, 5, 5, 4],
                "opponent_pits": [0, 5, 5, 5, 5, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            "reference_move": 1,
            "selected_move": 2,
            "regret": 0.1784,
            "value_error": 0.4701,
        }
        edge_move_five = {
            "id": "opening_plies_1_8-014",
            "bucket": "opening_plies_1_8",
            "state": {
                "player_pits": [0, 5, 5, 5, 5, 4],
                "opponent_pits": [4, 4, 0, 5, 5, 5],
                "player_store": 0,
                "opponent_store": 1,
                "current_player": 1,
            },
            "reference_move": 4,
            "selected_move": 5,
            "regret": 0.0451,
            "value_error": 0.5810,
        }
        other = {
            "id": "opening_plies_1_8-017",
            "bucket": "opening_plies_1_8",
            "state": {
                "player_pits": [1, 6, 6, 5, 5, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 0],
                "player_store": 0,
                "opponent_store": 1,
                "current_player": 0,
            },
            "reference_move": 2,
            "selected_move": 1,
            "regret": 0.2410,
            "value_error": 0.0991,
        }

        self.assertEqual(
            module.SUBFAMILY_EXTRA_TURN_OVERBIAS,
            module.classify_opening_failure(extra_turn_overbias)["subfamily"],
        )
        self.assertEqual(
            module.SUBFAMILY_MISSED_EXTRA_TURN,
            module.classify_opening_failure(missed_extra_turn)["subfamily"],
        )
        self.assertEqual(
            module.SUBFAMILY_EDGE_MOVE_5,
            module.classify_opening_failure(edge_move_five)["subfamily"],
        )
        self.assertEqual(
            module.SUBFAMILY_OTHER,
            module.classify_opening_failure(other)["subfamily"],
        )

    def test_build_report_counts_subfamilies(self):
        from ml.alphazero_lite import diagnose_opening_plies_subfamilies as module

        rows = [
            {
                "id": "opening_plies_1_8-004",
                "bucket": "opening_plies_1_8",
                "state": {
                    "player_pits": [4, 0, 5, 5, 5, 5],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 1,
                },
                "reference_move": 4,
                "selected_move": 2,
                "regret": 0.0059,
                "value_error": 0.5074,
            },
            {
                "id": "opening_plies_1_8-009",
                "bucket": "opening_plies_1_8",
                "state": {
                    "player_pits": [0, 5, 5, 5, 5, 4],
                    "opponent_pits": [0, 5, 5, 5, 5, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                "reference_move": 1,
                "selected_move": 2,
                "regret": 0.1784,
                "value_error": 0.4701,
            },
            {
                "id": "opening_plies_1_8-014",
                "bucket": "opening_plies_1_8",
                "state": {
                    "player_pits": [0, 5, 5, 5, 5, 4],
                    "opponent_pits": [4, 4, 0, 5, 5, 5],
                    "player_store": 0,
                    "opponent_store": 1,
                    "current_player": 1,
                },
                "reference_move": 4,
                "selected_move": 5,
                "regret": 0.0451,
                "value_error": 0.5810,
            },
            {
                "id": "opening_plies_1_8-060",
                "bucket": "opening_plies_1_8",
                "state": {
                    "player_pits": [5, 5, 4, 4, 0, 5],
                    "opponent_pits": [5, 5, 4, 4, 0, 5],
                    "player_store": 1,
                    "opponent_store": 1,
                    "current_player": 0,
                },
                "reference_move": 2,
                "selected_move": 2,
                "regret": 0.0,
                "value_error": 0.0,
            },
        ]

        report = module.build_report(
            forensic_report(rows),
            source_path="/tmp/example_forensics.json",
        )

        self.assertEqual(module.REPORT_SCHEMA, report["schema"])
        self.assertEqual(4, report["opening_row_count"])
        self.assertEqual(3, report["opening_failure_count"])
        self.assertEqual(
            [
                {
                    "subfamily": module.SUBFAMILY_EDGE_MOVE_5,
                    "count": 1,
                },
                {
                    "subfamily": module.SUBFAMILY_EXTRA_TURN_OVERBIAS,
                    "count": 1,
                },
                {
                    "subfamily": module.SUBFAMILY_MISSED_EXTRA_TURN,
                    "count": 1,
                },
            ],
            report["dominant_subfamilies"],
        )

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory(
            prefix="azlite-opening-plies-subfamily-diagnostic-"
        ) as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "forensics.json"
            out_path = tmp_path / "opening_subfamilies.json"
            in_path.write_text(
                json.dumps(
                    forensic_report(
                        [
                            {
                                "id": "opening_plies_1_8-004",
                                "bucket": "opening_plies_1_8",
                                "state": {
                                    "player_pits": [4, 0, 5, 5, 5, 5],
                                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                                    "player_store": 0,
                                    "opponent_store": 0,
                                    "current_player": 1,
                                },
                                "reference_move": 4,
                                "selected_move": 2,
                                "regret": 0.0059,
                                "value_error": 0.5074,
                            }
                        ]
                    )
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.diagnose_opening_plies_subfamilies",
                    "--forensics",
                    str(in_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                "azlite_opening_plies_subfamily_diagnostic_v1", payload["schema"]
            )
            self.assertEqual(1, payload["opening_failure_count"])
            self.assertEqual(
                "opening_extra_turn_overbias",
                payload["rows"][0]["subfamily"],
            )


if __name__ == "__main__":
    unittest.main()
