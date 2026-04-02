import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class LocalPromotionGateTest(unittest.TestCase):
    def run_gate(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            ["script/ai/local_promotion_gate", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_report(self, path: Path, *, games_played: int | None = None, games: int | None = None, wins: int, losses: int, draws: int, move_time_mean_ms=None, move_time_p95_ms=None, az_wins=None) -> None:
        report = {
            "schema": "arena_v1",
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "promotion_decision": {"passed": True},
        }
        if games_played is not None:
            report["games_played"] = games_played
        if games is not None:
            report["games"] = games
        if move_time_mean_ms is not None:
            report["notes"] = report.get("notes", {})
            report["notes"]["move_time_mean_ms"] = move_time_mean_ms
        if move_time_p95_ms is not None:
            report["notes"] = report.get("notes", {})
            report["notes"]["move_time_p95_ms"] = move_time_p95_ms
        if az_wins is not None:
            report["az_wins"] = az_wins
        path.write_text(json.dumps(report), encoding="utf-8")

    def test_rejects_candidate_that_does_not_beat_current_superhuman(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=20, losses=0, draws=40, move_time_mean_ms=120, move_time_p95_ms=180)
            self.write_report(tmp / "hard.json", games_played=120, wins=72, losses=0, draws=48, move_time_mean_ms=110, move_time_p95_ms=170)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=25, losses=5, draws=10, az_wins=25)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=6, draws=10, az_wins=24)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_score_below_threshold" for reason in report["failure_reasons"]))

    def test_rejects_latency_regression(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=80, losses=0, draws=40, move_time_mean_ms=260, move_time_p95_ms=310)
            self.write_report(tmp / "hard.json", games_played=120, wins=70, losses=10, draws=40, move_time_mean_ms=100, move_time_p95_ms=150)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=5, draws=5, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=10, draws=6, az_wins=24)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_move_time_mean_above_threshold" for reason in report["failure_reasons"]))
            self.assertTrue(any(reason["code"] == "arena_move_time_p95_above_threshold" for reason in report["failure_reasons"]))

    def test_accepts_candidate_that_beats_current_superhuman_and_hard(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28, move_time_mean_ms=120, move_time_p95_ms=160)
            self.write_report(tmp / "hard.json", games_played=120, wins=72, losses=0, draws=48, move_time_mean_ms=110, move_time_p95_ms=150)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--hard-path",
                str(tmp / "hard_model"),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-hard-report",
                str(tmp / "hard.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--out",
                str(out),
                "--max-arena-move-time-mean-ms",
                "200",
                "--max-arena-move-time-p95-ms",
                "250",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])

    def test_accepts_candidate_without_hard_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=92, losses=0, draws=28)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=30, losses=2, draws=8, az_wins=30)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=8, draws=8, az_wins=24)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--out",
                str(out),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertIsNone(report["hard_path"])
            self.assertIsNone(report["hard_score"])
            self.assertIsNone(report["hard_report_path"])
            self.assertTrue(all(e["id"] != "candidate_vs_hard_arena" for e in report["evaluations"]))

    def test_rejects_candidate_without_hard_path_when_arena_score_low(self):
        with tempfile.TemporaryDirectory(prefix="azlite-gate-") as tmp:
            tmp = Path(tmp)
            candidate = tmp / "candidate"
            candidate.mkdir()
            out = tmp / "report.json"
            self.write_report(tmp / "arena.json", games_played=120, wins=20, losses=0, draws=40)
            self.write_report(tmp / "cand_mcts.json", games=40, wins=25, losses=5, draws=10, az_wins=25)
            self.write_report(tmp / "cur_mcts.json", games=40, wins=24, losses=6, draws=10, az_wins=24)

            result = self.run_gate(
                "--candidate-path",
                str(candidate),
                "--stub-arena-report",
                str(tmp / "arena.json"),
                "--stub-candidate-mcts-report",
                str(tmp / "cand_mcts.json"),
                "--stub-current-mcts-report",
                str(tmp / "cur_mcts.json"),
                "--out",
                str(out),
            )

            self.assertNotEqual(0, result.returncode)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertTrue(any(reason["code"] == "arena_score_below_threshold" for reason in report["failure_reasons"]))
            self.assertFalse(any(reason["code"] == "candidate_not_stronger_than_hard" for reason in report["failure_reasons"]))


if __name__ == "__main__":
    unittest.main()
