import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import mine_failure_replay_dataset


class MineFailureReplayDatasetTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def _current_path(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        return str(repo_root / "storage/ai/alphazero_lite/current")

    def test_run_failure_mining_produces_rows(self):
        train_rows, holdout_rows, summary = (
            mine_failure_replay_dataset.run_failure_mining(
                current_path=self._current_path(),
                games=2,
                teacher_simulations=16,
                seed=42,
                max_positions_per_game=12,
                input_encoding="kalah_v3",
                policy_target_mode="default",
                value_target_mode="default",
                policy_temperature=1.0,
                train_split=0.8,
                disagreement_prob_threshold=0.15,
                top_visit_margin_threshold=0.6,
                tactical_mode=False,
            )
        )

        self.assertGreater(summary["total_positions_visited"], 0)
        self.assertGreaterEqual(len(train_rows), 0)
        self.assertGreaterEqual(len(holdout_rows), 0)
        self.assertIn("early", summary["rows_per_phase"])
        self.assertIn("mid", summary["rows_per_phase"])
        self.assertIn("late", summary["rows_per_phase"])

        all_rows = train_rows + holdout_rows
        if all_rows:
            row = all_rows[0]
            self.assertIn("state", row)
            self.assertIn("policy", row)
            self.assertIn("value", row)
            self.assertIn("player", row)
            self.assertIn("move_index", row)
            self.assertIn("policy_target_mode", row)
            self.assertIn("value_target_mode", row)
            self.assertIn("source_game_id", row)
            self.assertIn("teacher_top_move", row)
            self.assertIn("current_top_move", row)
            self.assertIn("filters_passed", row)
            self.assertEqual(len(row["state"]), 27)  # kalah_v3
            self.assertEqual(len(row["policy"]), 6)
            self.assertGreaterEqual(row["value"], -1.0)
            self.assertLessEqual(row["value"], 1.0)
            total_policy = sum(float(v) for v in row["policy"])
            self.assertAlmostEqual(total_policy, 1.0, delta=0.01)

    def test_phase_label(self):
        self.assertEqual(mine_failure_replay_dataset._phase_label(0), "early")
        self.assertEqual(mine_failure_replay_dataset._phase_label(8), "early")
        self.assertEqual(mine_failure_replay_dataset._phase_label(9), "mid")
        self.assertEqual(mine_failure_replay_dataset._phase_label(24), "mid")
        self.assertEqual(mine_failure_replay_dataset._phase_label(25), "late")

    def test_state_fingerprint_deterministic(self):
        state_a = [float(i) / 48.0 for i in range(27)]
        state_b = [float(i) / 48.0 for i in range(27)]
        fp_a = mine_failure_replay_dataset._state_fingerprint(state_a)
        fp_b = mine_failure_replay_dataset._state_fingerprint(state_b)
        self.assertEqual(fp_a, fp_b)

        state_c = [float(i + 1) / 48.0 for i in range(27)]
        fp_c = mine_failure_replay_dataset._state_fingerprint(state_c)
        self.assertNotEqual(fp_a, fp_c)

    def test_top_move(self):
        visits = [10.0, 5.0, 1.0, 0.0, 8.0, 3.0]
        legal = [0, 1, 2, 4]
        self.assertEqual(mine_failure_replay_dataset._top_move(visits, legal), 0)
        self.assertEqual(mine_failure_replay_dataset._top_move(visits, [0, 4]), 0)
        self.assertAlmostEqual(
            mine_failure_replay_dataset._top_visit_share(visits, legal),
            10.0 / (10.0 + 5.0 + 1.0 + 8.0),
            delta=0.01,
        )

    def test_cli_writes_output_and_summary(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-failure-mining-test-") as tmp:
            train_path = Path(tmp) / "failure_mined_train.jsonl"
            holdout_path = Path(tmp) / "failure_mined_holdout.jsonl"
            summary_path = Path(tmp) / "failure_mining_summary.json"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mine_failure_replay_dataset.py",
                    "--out-train",
                    str(train_path),
                    "--out-holdout",
                    str(holdout_path),
                    "--out-summary",
                    str(summary_path),
                    "--current",
                    self._current_path(),
                    "--games",
                    "2",
                    "--teacher-simulations",
                    "16",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "8",
                    "--input-encoding",
                    "kalah_v3",
                    "--tactical-mode",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertIn("failure_mining_train_rows=", result.stdout)

            self.assertTrue(train_path.exists())
            self.assertTrue(holdout_path.exists())
            self.assertTrue(summary_path.exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("train_sha256", summary)
            self.assertIn("holdout_sha256", summary)
            self.assertIn("mined_games", summary)
            self.assertIn("total_positions_visited", summary)

            train_rows = [
                json.loads(line)
                for line in train_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            holdout_rows = [
                json.loads(line)
                for line in holdout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            if train_rows:
                row = train_rows[0]
                self.assertIn("state", row)
                self.assertIn("policy", row)
                self.assertIn("value", row)
                self.assertIn("source_game_id", row)
                self.assertIn("filters_passed", row)

            if holdout_rows:
                row = holdout_rows[0]
                self.assertIn("state", row)
                self.assertIn("policy", row)
                self.assertIn("value", row)

    def test_output_rows_are_loadable_by_train_load_jsonl(self):
        from ml.alphazero_lite.train import load_jsonl

        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(
            prefix="azlite-failure-mining-load-test-"
        ) as tmp:
            train_path = Path(tmp) / "failure_mined_train.jsonl"
            holdout_path = Path(tmp) / "failure_mined_holdout.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/mine_failure_replay_dataset.py",
                    "--out-train",
                    str(train_path),
                    "--out-holdout",
                    str(holdout_path),
                    "--current",
                    self._current_path(),
                    "--games",
                    "2",
                    "--teacher-simulations",
                    "32",
                    "--seed",
                    "42",
                    "--max-positions-per-game",
                    "8",
                    "--input-encoding",
                    "kalah_v3",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)

            if train_path.exists() and train_path.stat().st_size > 0:
                x, p, v = load_jsonl(
                    train_path,
                    policy_target_mode="default",
                    value_target_mode="default",
                )
                self.assertGreater(x.shape[0], 0)
                self.assertEqual(x.shape[1], 27)
                self.assertEqual(p.shape[1], 6)
                self.assertEqual(v.shape[1], 1)

            if holdout_path.exists() and holdout_path.stat().st_size > 0:
                x, p, v = load_jsonl(
                    holdout_path,
                    policy_target_mode="default",
                    value_target_mode="default",
                )
                self.assertGreater(x.shape[0], 0)
                self.assertEqual(x.shape[1], 27)
                self.assertEqual(p.shape[1], 6)
                self.assertEqual(v.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
