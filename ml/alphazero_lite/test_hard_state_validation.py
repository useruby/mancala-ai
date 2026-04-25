import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


class HardStateValidationTest(unittest.TestCase):
    def test_main_writes_stable_json_summary(self):
        from ml.alphazero_lite import hard_state_validation

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-validation-") as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "artifact"
            validation_path = tmp_path / "validation.json"
            out_path = tmp_path / "report.json"
            validation_path.write_text("[]", encoding="utf-8")

            args = Namespace(
                artifact_path=str(artifact_path),
                validation_path=str(validation_path),
                teacher_simulations=1200,
                artifact_simulations=384,
                c_puct=1.25,
                seed=42,
                out=str(out_path),
            )

            suite = [
                mock.Mock(id="pos-1", state={"marker": 1}, bucket="capture_available"),
                mock.Mock(id="pos-2", state={"marker": 2}, bucket="sparse_endgame"),
            ]
            references = [
                {"selected_move": 1, "child_stats": [{"move": 1, "win_rate": 0.9, "visits": 10}], "teacher_value": 0.4},
                {"selected_move": 2, "child_stats": [{"move": 2, "win_rate": 0.8, "visits": 10}], "teacher_value": -0.2},
            ]
            systems = [
                {"selected_move": 1, "value": 0.3},
                {"selected_move": 4, "value": -0.1},
            ]
            built_rows = [
                {
                    "id": "pos-1",
                    "bucket": "capture_available",
                    "agrees_top1": True,
                    "regret": 0.0,
                    "value_error": 0.1,
                },
                {
                    "id": "pos-2",
                    "bucket": "sparse_endgame",
                    "agrees_top1": False,
                    "regret": 0.2,
                    "value_error": 0.1,
                },
            ]
            summarized = {
                "overall": {
                    "positions": 2,
                    "top1_agreement": 0.5,
                    "average_regret": 0.1,
                    "value_calibration_mae": 0.1,
                },
                "buckets": {
                    "capture_available": {
                        "positions": 1,
                        "top1_agreement": 1.0,
                        "average_regret": 0.0,
                        "value_calibration_mae": 0.1,
                    },
                    "sparse_endgame": {
                        "positions": 1,
                        "top1_agreement": 0.0,
                        "average_regret": 0.2,
                        "value_calibration_mae": 0.1,
                    },
                },
                "rows": built_rows,
            }
            bucket_matrix = {
                "capture_available": {"positions": 1, "systems": {"artifact": summarized["buckets"]["capture_available"]}},
                "sparse_endgame": {"positions": 1, "systems": {"artifact": summarized["buckets"]["sparse_endgame"]}},
            }

            with mock.patch("ml.alphazero_lite.hard_state_validation.parse_args", return_value=args), mock.patch(
                "ml.alphazero_lite.hard_state_validation.load_suite",
                return_value=suite,
            ) as load_suite_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.build_eval_search_options",
                return_value={"search": "options"},
            ) as build_search_options_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.ArtifactEvaluator",
                return_value=mock.sentinel.evaluator,
            ) as artifact_evaluator_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.run_reference",
                side_effect=references,
            ) as run_reference_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.evaluate_artifact_position",
                side_effect=systems,
            ) as evaluate_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.build_row",
                side_effect=built_rows,
            ) as build_row_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.summarize_system",
                return_value=summarized,
            ) as summarize_system_mock, mock.patch(
                "ml.alphazero_lite.hard_state_validation.summarize_bucket_matrix",
                return_value=bucket_matrix,
            ) as summarize_bucket_matrix_mock:
                hard_state_validation.main()

            report = json.loads(out_path.read_text(encoding="utf-8"))

            self.assertEqual("azlite_hard_state_validation_v1", report["schema"])
            self.assertEqual(str(artifact_path), report["artifact_path"])
            self.assertEqual(str(validation_path), report["validation_path"])
            self.assertEqual(2, report["position_count"])
            self.assertEqual(0.5, report["policy_top1_agreement"])
            self.assertEqual(0.1, report["average_regret"])
            self.assertEqual(0.1, report["value_calibration_mae"])
            self.assertEqual(summarized["overall"], report["overall"])
            self.assertEqual(bucket_matrix, report["buckets"])

            load_suite_mock.assert_called_once_with(validation_path)
            build_search_options_mock.assert_called_once_with()
            artifact_evaluator_mock.assert_called_once_with(artifact_path)
            self.assertEqual(2, run_reference_mock.call_count)
            self.assertEqual(2, evaluate_mock.call_count)
            self.assertEqual(2, build_row_mock.call_count)
            summarize_system_mock.assert_called_once_with(built_rows)
            summarize_bucket_matrix_mock.assert_called_once_with({"artifact": built_rows})

    def test_main_fails_when_validation_path_is_missing(self):
        from ml.alphazero_lite import hard_state_validation

        with tempfile.TemporaryDirectory(prefix="azlite-hard-state-validation-missing-") as tmp:
            tmp_path = Path(tmp)
            args = Namespace(
                artifact_path=str(tmp_path / "artifact"),
                validation_path=str(tmp_path / "missing.json"),
                teacher_simulations=1200,
                artifact_simulations=384,
                c_puct=1.25,
                seed=42,
                out=str(tmp_path / "report.json"),
            )

            with mock.patch("ml.alphazero_lite.hard_state_validation.parse_args", return_value=args):
                with self.assertRaisesRegex(SystemExit, "validation path does not exist"):
                    hard_state_validation.main()


if __name__ == "__main__":
    unittest.main()
