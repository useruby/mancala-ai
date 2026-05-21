import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PromoteCheckpointScriptTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def test_cli_rejects_negative_max_losses(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 400,
                        "wins": 200,
                        "losses": 0,
                        "draws": 200,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--min-score",
                    "0.0",
                    "--require-lossless",
                    "--max-losses",
                    "-1",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("must be a non-negative integer", result.stderr)

    def test_cli_rejects_invalid_arena_report(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
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
                    self.executable_python(),
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
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
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
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                ],
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
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
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
                    self.executable_python(),
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
            self.assertEqual(
                "live-metadata",
                (target_dir / "metadata.json").read_text(encoding="utf-8"),
            )

    def test_cli_rejects_losses_when_lossless_is_required(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            target_dir = Path(tmp) / "current"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 400,
                        "wins": 399,
                        "losses": 1,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--target",
                    str(target_dir),
                    "--min-score",
                    "0.0",
                    "--require-lossless",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("lossless requirement failed", result.stderr)

    def test_cli_rejects_missing_gate_report_when_supplied(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
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
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--gate-report",
                    str(checkpoint_dir / "local_promotion_gate.json"),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Missing gate report", result.stderr)

    def test_cli_rejects_failed_gate_report(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
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
            (checkpoint_dir / "local_promotion_gate.json").write_text(
                json.dumps({"passed": False}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--gate-report",
                    str(checkpoint_dir / "local_promotion_gate.json"),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Gate report did not pass", result.stderr)

    def test_cli_accepts_relative_gate_report_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            target_dir = Path(tmp) / "current"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            (checkpoint_dir / "metadata.json").write_text(
                json.dumps({"schema_version": 1}), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
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

            gate_dir = repo_root / "tmp"
            gate_dir.mkdir(parents=True, exist_ok=True)
            gate_report_path = gate_dir / "local_promotion_gate_test.json"
            gate_report_path.write_text(json.dumps({"passed": True}), encoding="utf-8")

            try:
                relative_gate_path = gate_report_path.relative_to(repo_root).as_posix()
                result = subprocess.run(
                    [
                        self.executable_python(),
                        str(repo_root / "ml/alphazero_lite/promote_checkpoint.py"),
                        str(checkpoint_dir),
                        "--target",
                        str(target_dir),
                        "--min-score",
                        "0.0",
                        "--gate-report",
                        relative_gate_path,
                    ],
                    cwd=checkpoint_dir,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            finally:
                gate_report_path.unlink(missing_ok=True)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue((target_dir / "metadata.json").exists())

    def test_cli_promotes_when_lossless_requirement_is_met(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            target_dir = Path(tmp) / "current"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            metadata = {"schema_version": 1, "version": "candidate"}
            weights = {"w_input": [[0.1]]}
            arena_report = {
                "schema": "arena_v1",
                "games_played": 400,
                "wins": 200,
                "losses": 0,
                "draws": 200,
                "promotion_decision": {"passed": True},
            }
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps(weights), encoding="utf-8"
            )
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(arena_report), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--target",
                    str(target_dir),
                    "--min-score",
                    "0.0",
                    "--require-lossless",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(
                metadata,
                json.loads((target_dir / "metadata.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(
                weights,
                json.loads((target_dir / "weights.json").read_text(encoding="utf-8")),
            )
            self.assertEqual(
                arena_report,
                json.loads(
                    (target_dir / "arena_report.json").read_text(encoding="utf-8")
                ),
            )

    def test_cli_promotes_to_multiple_targets(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            checkpoint_dir = Path(tmp) / "checkpoint"
            target_a = Path(tmp) / "current"
            target_b = Path(tmp) / "current_b"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            metadata = {"schema_version": 1, "version": "multi-target-test"}
            weights = {"w_input": [[0.5]]}
            arena_report = {
                "schema": "arena_v1",
                "games_played": 100,
                "wins": 60,
                "losses": 40,
                "draws": 0,
                "promotion_decision": {"passed": True},
            }
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (checkpoint_dir / "weights.json").write_text(
                json.dumps(weights), encoding="utf-8"
            )
            (checkpoint_dir / "arena_report.json").write_text(
                json.dumps(arena_report), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--target",
                    str(target_a),
                    "--target",
                    str(target_b),
                    "--min-score",
                    "0.0",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            for target in (target_a, target_b):
                self.assertEqual(
                    metadata,
                    json.loads((target / "metadata.json").read_text(encoding="utf-8")),
                )
                self.assertEqual(
                    weights,
                    json.loads((target / "weights.json").read_text(encoding="utf-8")),
                )
                self.assertEqual(
                    arena_report,
                    json.loads(
                        (target / "arena_report.json").read_text(encoding="utf-8")
                    ),
                )
