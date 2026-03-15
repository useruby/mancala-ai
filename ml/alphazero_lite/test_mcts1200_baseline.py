import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MCTS1200BaselineScriptTest(unittest.TestCase):
    def test_parse_args_uses_search_defaults_without_flags(self):
        from ml.alphazero_lite.mcts1200_baseline import parse_args

        original_argv = list(sys.argv)
        try:
            sys.argv = [
                "mcts1200_baseline.py",
                "--challenger-path",
                "/tmp/challenger",
                "--out",
                "/tmp/report.json",
            ]
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual("visit_count", args.root_policy_mode)
        self.assertEqual(0.0, args.tactical_root_bias)

    def test_cli_writes_mcts1200_report_with_expected_schema(self):
        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "30",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "6",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "schema",
                    "games",
                    "az_base_simulations",
                    "mcts_simulations",
                    "az_wins",
                    "mcts_wins",
                    "draws",
                    "score",
                },
                set(report.keys()),
            )
            self.assertEqual("azlite_vs_mcts_v1", report["schema"])
            self.assertEqual(30, report["games"])
            self.assertEqual(640, report["az_base_simulations"])
            self.assertEqual(1200, report["mcts_simulations"])

    def test_partitioning_preserves_global_game_indexes(self):
        from ml.alphazero_lite.mcts1200_baseline import partition_counts, partition_starts

        self.assertEqual([5, 5, 5, 5, 5, 5], partition_counts(30, 6))
        self.assertEqual([0, 5, 10, 15, 20, 25], partition_starts([5, 5, 5, 5, 5, 5]))

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "mcts1200_report.json"

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/mcts1200_baseline.py",
                    "--challenger-path",
                    str(tmp_path / "challenger"),
                    "--games",
                    "6",
                    "--seed",
                    "42",
                    "--az-base-simulations",
                    "640",
                    "--mcts-simulations",
                    "1200",
                    "--workers",
                    "4",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_MCTS1200_BASELINE_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(6, report["games"])
            self.assertEqual(3, report["az_wins"])
            self.assertEqual(2, report["mcts_wins"])
            self.assertEqual(1, report["draws"])
            self.assertEqual(0.5833, report["score"])

    def test_module_no_longer_exposes_ruby_worker_script(self):
        from ml.alphazero_lite import mcts1200_baseline

        self.assertFalse(hasattr(mcts1200_baseline, "ruby_worker_script"))

    def test_run_worker_stays_in_python(self):
        from ml.alphazero_lite import mcts1200_baseline
        from ml.alphazero_lite.mcts1200_baseline import run_worker

        with tempfile.TemporaryDirectory(prefix="azlite-mcts1200-") as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"input_encoding": "kalah_v1", "architecture": {"model_type": "mlp_v1"}}),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w1": [[0.0] * 16 for _ in range(15)],
                        "b1": [0.0] * 16,
                        "w2": [[0.0] * 16 for _ in range(16)],
                        "b2": [0.0] * 16,
                        "w_policy": [[0.0] * 6 for _ in range(16)],
                        "b_policy": [0.0] * 6,
                        "w_value": [[0.0] for _ in range(16)],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(hasattr(mcts1200_baseline, "subprocess"))
            result = run_worker(
                challenger_path=str(artifact_dir),
                games=1,
                start_index=0,
                seed=42,
                az_base_simulations=64,
                mcts_simulations=64,
                search_options={
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.1,
                },
            )

        self.assertEqual(1, result["games"])


if __name__ == "__main__":
    unittest.main()
