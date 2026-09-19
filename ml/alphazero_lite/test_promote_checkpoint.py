import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PromoteCheckpointScriptTest(unittest.TestCase):
    def write_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        metadata: dict | None = None,
        weights: dict | None = None,
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata or {"schema_version": 1}), encoding="utf-8"
        )
        (checkpoint_dir / "weights.json").write_text(
            json.dumps(weights or {"w_input": [[0.1]]}), encoding="utf-8"
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

    def write_gate_report(
        self, path: Path, checkpoint_dir: Path, *, passed: bool = True
    ) -> None:
        identity = {
            f"{filename.replace('.', '_')}_sha256": hashlib.sha256(
                (checkpoint_dir / filename).read_bytes()
            ).hexdigest()
            for filename in ("weights.json", "metadata.json")
        }
        arena_path = checkpoint_dir / "arena_report.json"
        arena = json.loads(arena_path.read_text(encoding="utf-8"))
        path.write_text(
            json.dumps(
                {
                    "passed": passed,
                    "candidate_identity": identity,
                    "require_lossless": False,
                    "max_losses": 0,
                    "min_arena_score": 0.0,
                    "min_arena_games": arena["games_played"],
                    "arena_losses": arena["losses"],
                    "arena_report_path": str(arena_path),
                    "arena_evidence": {
                        "path": str(arena_path),
                        "sha256": hashlib.sha256(arena_path.read_bytes()).hexdigest(),
                    },
                }
            ),
            encoding="utf-8",
        )

    def run_promotion(
        self, checkpoint_dir: Path, target_dir: Path, gate_report: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self.executable_python(),
                "ml/alphazero_lite/promote_checkpoint.py",
                str(checkpoint_dir),
                "--target",
                str(target_dir),
                "--gate-report",
                str(gate_report),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

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
            target_dir = Path(tmp) / "current"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "sentinel").write_bytes(b"unchanged")

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
                    "--target",
                    str(target_dir),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("Gate report did not pass", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

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
            self.write_gate_report(gate_report_path, checkpoint_dir)

            try:
                relative_gate_path = gate_report_path.relative_to(repo_root).as_posix()
                result = subprocess.run(
                    [
                        self.executable_python(),
                        str(repo_root / "ml/alphazero_lite/promote_checkpoint.py"),
                        str(checkpoint_dir),
                        "--target",
                        str(target_dir),
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

    def test_cli_promotes_with_matching_gate_candidate_identity(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir = tmp_path / "checkpoint"
            target_dir = tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(
                (checkpoint_dir / "weights.json").read_bytes(),
                (target_dir / "weights.json").read_bytes(),
            )

    def test_cli_rejects_gate_for_different_candidate_without_changing_target(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            candidate_a, candidate_b = tmp_path / "a", tmp_path / "b"
            target_dir, gate_report = tmp_path / "current", tmp_path / "gate.json"
            self.write_checkpoint(candidate_a, weights={"w_input": [[0.1]]})
            self.write_checkpoint(candidate_b, weights={"w_input": [[0.2]]})
            self.write_gate_report(gate_report, candidate_a)
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(candidate_b, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_candidate_identity_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

    def test_cli_rejects_weights_mutated_after_gate_without_changing_target(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)
            (checkpoint_dir / "weights.json").write_text(
                '{"w_input":[[0.2]]}', encoding="utf-8"
            )
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_candidate_identity_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

    def test_cli_rejects_metadata_mutated_after_gate_without_changing_target(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)
            (checkpoint_dir / "metadata.json").write_text(
                '{"schema_version":2}', encoding="utf-8"
            )
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_candidate_identity_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

    def test_cli_rejects_passing_gate_missing_candidate_identity(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            gate_report.write_text(json.dumps({"passed": True}), encoding="utf-8")
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_candidate_identity_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

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

    def test_gate_promotes_non_lossless_evidence_without_local_arena_report(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report, arena_path = tmp_path / "gate.json", tmp_path / "arena.json"
            self.write_checkpoint(checkpoint_dir)
            arena_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 120,
                        "wins": 70,
                        "losses": 20,
                        "draws": 30,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )
            self.write_gate_report(gate_report, checkpoint_dir)
            gate = json.loads(gate_report.read_text(encoding="utf-8"))
            gate.update(
                {
                    "arena_report_path": str(arena_path),
                    "arena_evidence": {
                        "path": str(arena_path),
                        "sha256": hashlib.sha256(arena_path.read_bytes()).hexdigest(),
                    },
                    "arena_losses": 20,
                    "min_arena_games": 120,
                }
            )
            gate_report.write_text(json.dumps(gate), encoding="utf-8")
            (checkpoint_dir / "arena_report.json").unlink()

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertEqual(
                arena_path.read_bytes(), (target_dir / "arena_report.json").read_bytes()
            )

    def test_gate_rejects_inconsistent_lossless_policy_before_target_mutation(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            arena_path = checkpoint_dir / "arena_report.json"
            arena = json.loads(arena_path.read_text(encoding="utf-8"))
            arena.update({"wins": 399, "losses": 1, "draws": 0})
            arena_path.write_text(json.dumps(arena), encoding="utf-8")
            self.write_gate_report(gate_report, checkpoint_dir)
            gate = json.loads(gate_report.read_text(encoding="utf-8"))
            gate["require_lossless"] = True
            gate_report.write_text(json.dumps(gate), encoding="utf-8")
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_policy_inconsistent", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

    def test_gate_rejects_lossless_override_for_non_lossless_evidence(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/promote_checkpoint.py",
                    str(checkpoint_dir),
                    "--target",
                    str(target_dir),
                    "--gate-report",
                    str(gate_report),
                    "--require-lossless",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_policy_override_forbidden", result.stderr)
            self.assertFalse(target_dir.exists())

    def test_gate_rejects_arena_modified_after_gate_before_target_mutation(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report = tmp_path / "gate.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)
            (checkpoint_dir / "arena_report.json").write_text("{}", encoding="utf-8")
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_arena_evidence_hash_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())

    def test_gate_rejects_wrong_arena_reference_before_target_mutation(self):
        with tempfile.TemporaryDirectory(prefix="azlite-promote-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir, target_dir = tmp_path / "checkpoint", tmp_path / "current"
            gate_report, wrong_arena = tmp_path / "gate.json", tmp_path / "wrong.json"
            self.write_checkpoint(checkpoint_dir)
            self.write_gate_report(gate_report, checkpoint_dir)
            wrong_arena.write_bytes((checkpoint_dir / "arena_report.json").read_bytes())
            gate = json.loads(gate_report.read_text(encoding="utf-8"))
            gate["arena_evidence"] = {
                "path": str(wrong_arena),
                "sha256": hashlib.sha256(wrong_arena.read_bytes()).hexdigest(),
            }
            gate_report.write_text(json.dumps(gate), encoding="utf-8")
            target_dir.mkdir()
            (target_dir / "sentinel").write_bytes(b"unchanged")

            result = self.run_promotion(checkpoint_dir, target_dir, gate_report)

            self.assertNotEqual(0, result.returncode)
            self.assertIn("gate_arena_evidence_path_mismatch", result.stderr)
            self.assertEqual(b"unchanged", (target_dir / "sentinel").read_bytes())
