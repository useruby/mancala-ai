"""Unprivileged integration tests for the generic experiment lifecycle."""

from __future__ import annotations

import hashlib
import json
import time
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import experiment_lifecycle as lifecycle

REPO = Path(__file__).parents[2]


class ExperimentLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=REPO / ".tmp", prefix="experiment-lifecycle-"
        )
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self._git("init", "-b", "main")
        self._git("config", "user.name", "Lifecycle Test")
        self._git("config", "user.email", "lifecycle-test@example.invalid")
        (self.repo / "README.md").write_text("fixture repository\n")
        self.input_path = self.repo / "input.json"
        self.input_path.write_text('{"input":1}\n')
        self._git("add", "README.md", "input.json")
        self._git("commit", "-m", "initial fixture")
        self.base_commit = self._git("rev-parse", "HEAD")
        self.worktrees = self.root / "worktrees"
        self.archives = self.root / "archive-root"
        self.worktree = lifecycle.create_worktree(
            self.repo,
            "demo-001",
            "experiment/demo-001",
            self.base_commit,
            hypothesis="The deterministic fixture emits its registered result.",
            objective="Exercise process and artifact lifecycle only.",
            experiment_type="synthetic_lifecycle_test",
            authorization_reference="test-owner-approval:synthetic-only",
            random_seeds=[17],
            input_artifacts={
                str(self.input_path): hashlib.sha256(
                    self.input_path.read_bytes()
                ).hexdigest()
            },
            checkpoint_identity=None,
            acceptance_criteria=["result JSON has answer 42"],
            worktree_root=self.worktrees,
        )
        self._git_at(
            self.worktree,
            "add",
            ".experiment/registration.json",
            ".experiment/.gitignore",
            ".experiment/base-commit.json",
        )
        self._git_at(self.worktree, "commit", "-m", "register synthetic experiment")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    @staticmethod
    def _git_at(path: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _freeze(self, attempt_id: str, code: str, *, timeout: int = 15) -> Path:
        output_path = (
            self.worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / attempt_id
            / "result.json"
        )
        command = [sys.executable, "-c", code, str(output_path)]
        registered_inputs = json.loads(
            (self.worktree / ".experiment" / "registration.json").read_bytes()
        )["input_artifacts"]
        return lifecycle.freeze_attempt(
            self.worktree,
            attempt_id=attempt_id,
            command=command,
            configuration={"fixture": "synthetic"},
            runtime_versions={"python_executable": sys.executable},
            device={"cpu": "unprivileged-host", "gpu": None},
            worker_count=1,
            expected_runtime_seconds=1,
            wall_time_limit_seconds=timeout,
            minimum_free_disk_bytes=0,
            outputs=["result.json"],
            publication_outputs=["result.json"],
            input_artifacts=registered_inputs,
            random_seeds=[17],
        )

    @staticmethod
    def _execute(attempt_dir: Path, **kwargs):
        return lifecycle.execute_attempt(
            attempt_dir,
            expected_manifest_sha256=lifecycle.sha256_file(
                attempt_dir / "manifest.json"
            ),
            **kwargs,
        )

    def test_success_archive_publication_and_safe_cleanup(self) -> None:
        with self.assertRaisesRegex(ValueError, "registered_linked_worktree"):
            lifecycle._validate_experiment_worktree(self.repo)
        registered = lifecycle.inspect_worktrees(self.repo)
        self.assertTrue(
            any(record["experiment_id"] == "demo-001" for record in registered)
        )
        code = "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'answer':42},sort_keys=True)+'\\n')"
        attempt_dir = self._freeze("attempt-001", code)
        result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(lifecycle.verify_attempt(attempt_dir)["outputs_verified"], 1)
        reference = lifecycle.archive_attempt(attempt_dir, self.archives)
        archive_result = lifecycle.verify_archive(
            Path(reference["archive_path"]), Path(reference["archive_manifest_path"])
        )
        self.assertTrue(archive_result["verified"])
        publication_dir = self.worktree / "docs" / "data" / "demo-001"
        publication = lifecycle.prepare_publication(attempt_dir, publication_dir)
        self.assertGreaterEqual(len(publication["files"]), 6)
        self._git_at(self.worktree, "add", "docs/data/demo-001")
        self._git_at(
            self.worktree, "commit", "-m", "publish synthetic lifecycle result"
        )
        dirty_marker = self.worktree / "unpublished.txt"
        dirty_marker.write_text("do not remove\n")
        with self.assertRaisesRegex(ValueError, "unpublished_changes"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        dirty_marker.unlink()
        self._git(
            "merge", "--no-ff", "experiment/demo-001", "-m", "merge lifecycle fixture"
        )
        lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        self.assertFalse(self.worktree.exists())
        with self.assertRaisesRegex(FileExistsError, "already_registered"):
            lifecycle.create_worktree(
                self.repo,
                "demo-001",
                "experiment/demo-duplicate",
                self.base_commit,
                hypothesis="duplicate",
                objective="duplicate",
                experiment_type="test",
                authorization_reference="test-owner-approval",
                random_seeds=[17],
                input_artifacts={
                    str(self.input_path): hashlib.sha256(
                        self.input_path.read_bytes()
                    ).hexdigest()
                },
                checkpoint_identity=None,
                acceptance_criteria=["criterion"],
                worktree_root=self.worktrees,
            )

    def test_execution_requires_the_retained_freeze_digest_and_unique_attempt(
        self,
    ) -> None:
        attempt_dir = self._freeze(
            "attempt-freeze-pin",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ok')",
        )
        manifest_digest = lifecycle.sha256_file(attempt_dir / "manifest.json")
        with self.assertRaisesRegex(ValueError, "freeze_digest_mismatch"):
            lifecycle.execute_attempt(attempt_dir, expected_manifest_sha256="0" * 64)
        result = lifecycle.execute_attempt(
            attempt_dir, expected_manifest_sha256=manifest_digest
        )
        self.assertEqual(result["status"], "completed")
        with self.assertRaises(FileExistsError):
            self._freeze(
                "attempt-freeze-pin",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('duplicate')",
            )
        with self.assertRaises(FileExistsError):
            lifecycle.execute_attempt(
                attempt_dir, expected_manifest_sha256=manifest_digest
            )

    def test_cleanup_refuses_a_started_attempt_without_verified_archive(self) -> None:
        attempt_dir = self._freeze(
            "attempt-unarchived",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        self._execute(attempt_dir)
        self._git(
            "merge", "--no-ff", "experiment/demo-001", "-m", "merge unarchived fixture"
        )
        with self.assertRaisesRegex(
            ValueError, "required_attempt_artifacts_not_archived"
        ):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")

    def test_timeout_is_recorded_and_archived_without_retry(self) -> None:
        attempt_dir = self._freeze(
            "attempt-timeout", "import time; time.sleep(30)", timeout=1
        )
        result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "wall_time_limit_exceeded")
        self.assertEqual(lifecycle.verify_attempt(attempt_dir)["status"], "failed")
        lifecycle.archive_attempt(attempt_dir, self.archives)
        with self.assertRaises(FileExistsError):
            lifecycle.archive_attempt(attempt_dir, self.archives)

    def test_timeout_kills_descendant_process_group(self) -> None:
        marker = self.root / "child-survived.txt"
        code = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',"
            f"\"import time; time.sleep(1.5); open({str(marker)!r},'w').write('alive')\"]); "
            "time.sleep(30)"
        )
        attempt_dir = self._freeze("attempt-child-timeout", code, timeout=1)
        result = self._execute(attempt_dir)
        self.assertEqual(result["reason"], "wall_time_limit_exceeded")
        time.sleep(0.7)
        self.assertFalse(marker.exists())

    def test_interruption_cleans_process_group_and_records_interrupted(self) -> None:
        attempt_dir = self._freeze("attempt-interrupted", "pass")

        class FakeProcess:
            pid = 123456
            returncode = None

            def __init__(self, *_args, **_kwargs):
                pass

            def wait(self, timeout=None):
                raise KeyboardInterrupt

        def stop(process):
            process.returncode = -15

        with (
            patch.object(lifecycle, "_launch_command", FakeProcess),
            patch.object(lifecycle, "_stop_process_group", side_effect=stop),
        ):
            result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(lifecycle.verify_attempt(attempt_dir)["status"], "interrupted")

    def test_launch_and_disk_preflight_failures_have_terminal_records(self) -> None:
        launch_attempt = self._freeze("attempt-launch-failure", "pass")
        with patch.object(
            lifecycle, "_launch_command", side_effect=OSError("cannot launch")
        ):
            launch = self._execute(launch_attempt)
        self.assertEqual(launch["status"], "failed")
        self.assertIn("cannot launch", launch["reason"])
        self.assertEqual(lifecycle.verify_attempt(launch_attempt)["status"], "failed")

        disk_attempt = self._freeze("attempt-disk-preflight", "pass")
        with patch.object(
            lifecycle.shutil,
            "disk_usage",
            return_value=type("Usage", (), {"free": 0})(),
        ):
            disk = self._execute(disk_attempt, minimum_free_disk_bytes=1)
        self.assertEqual(disk["status"], "failed")
        self.assertIn("disk_space_preflight_failed", disk["reason"])
        self.assertEqual(lifecycle.verify_attempt(disk_attempt)["status"], "failed")

    def test_input_authentication_and_archive_tampering_fail_closed(self) -> None:
        self.input_path.write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "input_artifact_hash_mismatch"):
            self._freeze("attempt-bad-input", "pass")
        self.input_path.write_text('{"input":1}\n')
        attempt_dir = self._freeze(
            "attempt-archive-tamper",
            "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'answer':42})+'\\n')",
        )
        self._execute(attempt_dir)
        reference = lifecycle.archive_attempt(attempt_dir, self.archives)
        archive_path = Path(reference["archive_path"])
        archive_path.write_bytes(archive_path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "digest_or_size_mismatch"):
            lifecycle.verify_archive(
                archive_path, Path(reference["archive_manifest_path"])
            )


if __name__ == "__main__":
    unittest.main()
