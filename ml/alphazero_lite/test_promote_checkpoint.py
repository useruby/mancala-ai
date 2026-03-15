import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class PromoteCheckpointScriptTest(unittest.TestCase):
    def test_cli_rejects_invalid_arena_report(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "metadata.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (checkpoint_dir / "weights.json").write_text(json.dumps({}), encoding="utf-8")
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "promotion_decision": {"passed": False},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("games_played must be greater than 0", result.stderr)

    def test_cli_rejects_npz_only_checkpoint(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "metadata.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            (checkpoint_dir / "model.npz").write_bytes(b"fake")
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 3,
                        "draws": 1,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [".venv/bin/python", "ml/alphazero_lite/promote_checkpoint.py", str(checkpoint_dir)],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("weights.json", result.stderr)

    def test_cli_does_not_clear_target_when_required_source_file_is_missing(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            target_dir = Path(tmp) / "current"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "metadata.json").write_text("live-metadata", encoding="utf-8")
            (checkpoint_dir / "weights.json").write_text(json.dumps({}), encoding="utf-8")
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 10,
                        "wins": 6,
                        "losses": 3,
                        "draws": 1,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    ".venv/bin/python",
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--target",
                    str(target_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing required file", result.stderr)
            self.assertEqual("live-metadata", (target_dir / "metadata.json").read_text(encoding="utf-8"))
