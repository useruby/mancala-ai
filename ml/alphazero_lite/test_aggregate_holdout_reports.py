import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import aggregate_holdout_reports


PYTHON_BIN = Path(sys.executable)


def arena(wins, losses, draws, mean=100.0, p95=180.0):
    games = wins + losses + draws
    return {
        "schema": "arena_v1",
        "games_played": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "promotion_decision": {"passed": True},
        "notes": {
            "move_time_mean_ms": mean,
            "move_time_p95_ms": p95,
        },
    }


def mcts(wins, losses, draws):
    games = wins + losses + draws
    return {
        "schema": "azlite_vs_mcts_v1",
        "games": games,
        "az_wins": wins,
        "mcts_wins": losses,
        "draws": draws,
    }


class AggregateHoldoutReportsTest(unittest.TestCase):
    def test_aggregate_arena_sums_counts_and_weighted_latency(self):
        result = aggregate_holdout_reports.aggregate_arena(
            [
                arena(3, 1, 0, mean=90.0, p95=140.0),
                arena(2, 1, 1, mean=110.0, p95=200.0),
            ]
        )

        self.assertEqual("arena_v1", result["schema"])
        self.assertEqual(8, result["games_played"])
        self.assertEqual(5, result["wins"])
        self.assertEqual(2, result["losses"])
        self.assertEqual(1, result["draws"])
        self.assertTrue(result["promotion_decision"]["passed"])
        self.assertEqual(100.0, result["notes"]["move_time_mean_ms"])
        self.assertEqual(200.0, result["notes"]["move_time_p95_ms"])

    def test_aggregate_arena_marks_promotion_failed_below_threshold(self):
        result = aggregate_holdout_reports.aggregate_arena(
            [
                arena(2, 3, 1),
                arena(1, 3, 0),
            ]
        )

        self.assertFalse(result["promotion_decision"]["passed"])

    def test_aggregate_mcts_sums_counts(self):
        result = aggregate_holdout_reports.aggregate_mcts(
            [
                mcts(6, 3, 1),
                mcts(4, 2, 0),
            ]
        )

        self.assertEqual("azlite_vs_mcts_v1", result["schema"])
        self.assertEqual(16, result["games"])
        self.assertEqual(10, result["az_wins"])
        self.assertEqual(5, result["mcts_wins"])
        self.assertEqual(1, result["draws"])

    def test_cli_writes_three_outputs(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-aggregate-holdout-") as tmp:
            tmp_path = Path(tmp)
            arena_a = tmp_path / "arena_a.json"
            arena_b = tmp_path / "arena_b.json"
            candidate_a = tmp_path / "candidate_a.json"
            candidate_b = tmp_path / "candidate_b.json"
            current_a = tmp_path / "current_a.json"
            out_arena = tmp_path / "out" / "arena.json"
            out_candidate = tmp_path / "out" / "candidate_mcts.json"
            out_current = tmp_path / "out" / "current_mcts.json"

            arena_a.write_text(json.dumps(arena(4, 1, 0, mean=80.0, p95=120.0)), encoding="utf-8")
            arena_b.write_text(json.dumps(arena(3, 1, 1, mean=120.0, p95=220.0)), encoding="utf-8")
            candidate_a.write_text(json.dumps(mcts(5, 2, 1)), encoding="utf-8")
            candidate_b.write_text(json.dumps(mcts(4, 4, 0)), encoding="utf-8")
            current_a.write_text(json.dumps(mcts(2, 5, 1)), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.aggregate_holdout_reports",
                    "--arena-inputs",
                    f"{arena_a},{arena_b}",
                    "--candidate-mcts-inputs",
                    f"{candidate_a},{candidate_b}",
                    "--current-mcts-inputs",
                    str(current_a),
                    "--out-arena",
                    str(out_arena),
                    "--out-candidate-mcts",
                    str(out_candidate),
                    "--out-current-mcts",
                    str(out_current),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            arena_report = json.loads(out_arena.read_text(encoding="utf-8"))
            candidate_report = json.loads(out_candidate.read_text(encoding="utf-8"))
            current_report = json.loads(out_current.read_text(encoding="utf-8"))

            self.assertEqual(10, arena_report["games_played"])
            self.assertTrue(arena_report["promotion_decision"]["passed"])
            self.assertEqual(100.0, arena_report["notes"]["move_time_mean_ms"])
            self.assertEqual(220.0, arena_report["notes"]["move_time_p95_ms"])
            self.assertEqual(16, candidate_report["games"])
            self.assertEqual(8, current_report["games"])

    def test_cli_applies_min_arena_score_threshold(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-aggregate-holdout-") as tmp:
            tmp_path = Path(tmp)
            arena_path = tmp_path / "arena.json"
            candidate_path = tmp_path / "candidate.json"
            current_path = tmp_path / "current.json"
            out_arena = tmp_path / "out" / "arena.json"
            out_candidate = tmp_path / "out" / "candidate_mcts.json"
            out_current = tmp_path / "out" / "current_mcts.json"

            arena_path.write_text(json.dumps(arena(3, 2, 0)), encoding="utf-8")
            candidate_path.write_text(json.dumps(mcts(1, 0, 0)), encoding="utf-8")
            current_path.write_text(json.dumps(mcts(0, 1, 0)), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.aggregate_holdout_reports",
                    "--arena-inputs",
                    str(arena_path),
                    "--candidate-mcts-inputs",
                    str(candidate_path),
                    "--current-mcts-inputs",
                    str(current_path),
                    "--out-arena",
                    str(out_arena),
                    "--out-candidate-mcts",
                    str(out_candidate),
                    "--out-current-mcts",
                    str(out_current),
                    "--min-arena-score",
                    "0.7",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            arena_report = json.loads(out_arena.read_text(encoding="utf-8"))
            self.assertFalse(arena_report["promotion_decision"]["passed"])

    def test_aggregate_arena_rejects_zero_total_games_cleanly(self):
        with self.assertRaises(ValueError) as error:
            aggregate_holdout_reports.aggregate_arena([arena(0, 0, 0)])

        self.assertIn("zero", str(error.exception).lower())

    def test_cli_fails_cleanly_for_zero_game_arena_input(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-aggregate-holdout-") as tmp:
            tmp_path = Path(tmp)
            arena_path = tmp_path / "arena.json"
            candidate_path = tmp_path / "candidate.json"
            current_path = tmp_path / "current.json"
            out_arena = tmp_path / "out" / "arena.json"
            out_candidate = tmp_path / "out" / "candidate_mcts.json"
            out_current = tmp_path / "out" / "current_mcts.json"

            arena_path.write_text(json.dumps(arena(0, 0, 0)), encoding="utf-8")
            candidate_path.write_text(json.dumps(mcts(1, 0, 0)), encoding="utf-8")
            current_path.write_text(json.dumps(mcts(0, 1, 0)), encoding="utf-8")

            result = subprocess.run(
                [
                    str(PYTHON_BIN),
                    "-m",
                    "ml.alphazero_lite.aggregate_holdout_reports",
                    "--arena-inputs",
                    str(arena_path),
                    "--candidate-mcts-inputs",
                    str(candidate_path),
                    "--current-mcts-inputs",
                    str(current_path),
                    "--out-arena",
                    str(out_arena),
                    "--out-candidate-mcts",
                    str(out_candidate),
                    "--out-current-mcts",
                    str(out_current),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, result.returncode)
            self.assertIn("zero", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
