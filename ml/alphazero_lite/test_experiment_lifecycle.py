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

from jsonschema import Draft202012Validator

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
        self.dataset_dir = self.repo / "dataset"
        self.dataset_dir.mkdir()
        self.dataset_member = self.dataset_dir / "shard.jsonl"
        self.dataset_member.write_text('{"sample":1}\n')
        self._git("add", "README.md", "input.json", "dataset/shard.jsonl")
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
            random_seeds=[17],
            input_artifacts={
                str(self.input_path): hashlib.sha256(
                    self.input_path.read_bytes()
                ).hexdigest(),
                str(self.dataset_dir): lifecycle.artifact_sha256(self.dataset_dir),
            },
            checkpoint_identity=None,
            acceptance_criteria=["result JSON has answer 42"],
            worktree_root=self.worktrees,
            registration_metadata={
                "approved_protocol_revision": self.base_commit,
                "protocol_documents": {"docs/protocol.md": "a" * 64},
                "approved_decisions": ["synthetic-only"],
            },
        )
        registration, registration_path = lifecycle.load_worktree_registration(
            self.worktree
        )
        self.assertEqual(registration["approved_protocol_revision"], self.base_commit)
        self.assertEqual(registration["approved_decisions"], ["synthetic-only"])
        self.assertIsNone(registration["authorization_reference"])
        self.assertEqual(
            registration_path, "docs/experiments/demo-001/registration.json"
        )
        self._git_at(
            self.worktree,
            "add",
            ".experiment/.gitignore",
            "docs/experiments/demo-001/registration.json",
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

    def _freeze(
        self,
        attempt_id: str,
        code: str,
        *,
        timeout: int = 15,
        output_path: str = "result.json",
    ) -> Path:
        output_relative = output_path
        output_path = (
            self.worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / attempt_id
            / output_path
        )
        command = [sys.executable, "-c", code, str(output_path)]
        registered_inputs = lifecycle.load_worktree_registration(self.worktree)[0][
            "input_artifacts"
        ]
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
            outputs=[output_relative],
            publication_outputs=[output_relative],
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

    def _publish_and_merge_attempt(self, attempt_id: str) -> Path:
        attempt_dir = self._freeze(
            attempt_id,
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        self._execute(attempt_dir)
        lifecycle.archive_attempt(attempt_dir, self.archives)
        lifecycle.prepare_publication(
            attempt_dir,
            self.worktree / "docs" / "data" / "demo-001" / attempt_id,
        )
        self._git_at(self.worktree, "add", "docs/data/demo-001")
        self._git_at(self.worktree, "commit", "-m", f"publish {attempt_id}")
        self._git(
            "merge", "--no-ff", "experiment/demo-001", "-m", f"merge {attempt_id}"
        )
        return attempt_dir

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
        archive_reference_path = attempt_dir / "archive-reference.json"
        archive_reference_bytes = archive_reference_path.read_bytes()
        result_path = attempt_dir / "result.json"
        result_bytes = result_path.read_bytes()
        result_path.write_bytes(result_bytes + b"tamper")
        with self.assertRaisesRegex(
            ValueError,
            "attempt_output_hash_mismatch|archive_reference_identity_mismatch",
        ):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        result_path.write_bytes(result_bytes)
        result_path.unlink()
        with self.assertRaisesRegex(
            ValueError,
            "attempt_output_hash_mismatch|archive_reference_identity_mismatch",
        ):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        result_path.write_bytes(result_bytes)
        extra_path = attempt_dir / "post-archive-extra.bin"
        extra_path.write_bytes(b"new artifact")
        with self.assertRaisesRegex(ValueError, "archive_reference_identity_mismatch"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        extra_path.unlink()
        substituted_reference = json.loads(archive_reference_bytes)
        substituted_reference["archive_sha256"] = "0" * 64
        archive_reference_path.write_text(json.dumps(substituted_reference))
        with self.assertRaisesRegex(ValueError, "archive_reference_identity_mismatch"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        archive_reference_path.write_bytes(archive_reference_bytes)
        archive_reference_path.unlink()
        archive_reference_path.symlink_to(publication_dir / "archive-reference.json")
        with self.assertRaisesRegex(ValueError, "archive_reference_symlink"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        archive_reference_path.unlink()
        archive_reference_path.write_bytes(archive_reference_bytes)
        lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")
        self.assertFalse(self.worktree.exists())

    def test_consecutive_experiment_registrations_are_permanent_and_distinct(self):
        registration_a = self.worktree / "docs/experiments/demo-001/registration.json"
        bytes_a = registration_a.read_bytes()
        self._git("merge", "--no-ff", "experiment/demo-001", "-m", "merge experiment A")
        main_after_a = self._git("rev-parse", "HEAD")
        main_registration_a = self.repo / "docs/experiments/demo-001/registration.json"
        self.assertEqual(main_registration_a.read_bytes(), bytes_a)

        worktree_b = lifecycle.create_worktree(
            self.repo,
            "demo-002",
            "experiment/demo-002",
            main_after_a,
            hypothesis="A second experiment registration remains independently stored.",
            objective="Exercise consecutive experiment registration identities.",
            experiment_type="synthetic_lifecycle_test",
            random_seeds=[23],
            input_artifacts={
                str(self.input_path): hashlib.sha256(
                    self.input_path.read_bytes()
                ).hexdigest(),
            },
            checkpoint_identity=None,
            acceptance_criteria=["Both permanent registrations remain unchanged"],
            worktree_root=self.worktrees,
            registration_metadata={
                "resource_budgets": {"wall_time_limit_seconds": 60},
                "stop_conditions": ["Stop if either registration is missing"],
            },
        )
        registration_a_in_b = worktree_b / "docs/experiments/demo-001/registration.json"
        registration_b = worktree_b / "docs/experiments/demo-002/registration.json"
        self.assertEqual(registration_a_in_b.read_bytes(), bytes_a)
        self.assertEqual(
            json.loads(registration_b.read_bytes())["experiment_id"], "demo-002"
        )
        worktree_metadata = json.loads(
            (worktree_b / ".experiment/registration.json").read_bytes()
        )
        self.assertEqual(worktree_metadata["experiment_id"], "demo-002")
        self.assertEqual(
            worktree_metadata["registration_path"],
            "docs/experiments/demo-002/registration.json",
        )
        self.assertEqual(self._git_at(worktree_b, "rev-parse", "HEAD"), main_after_a)
        self._git_at(
            worktree_b,
            "add",
            "docs/experiments/demo-002/registration.json",
        )
        self._git_at(worktree_b, "commit", "-m", "register experiment B")
        registration_b_on_branch = self._git_at(
            worktree_b,
            "show",
            "HEAD:docs/experiments/demo-002/registration.json",
        )
        self.assertEqual(
            json.loads(registration_b_on_branch)["experiment_id"], "demo-002"
        )
        self.assertEqual(main_registration_a.read_bytes(), bytes_a)
        self.assertEqual(registration_a_in_b.read_bytes(), bytes_a)
        self.assertEqual(registration_a.read_bytes(), bytes_a)
        worktree_records = {
            item["experiment_id"]: item
            for item in lifecycle.inspect_worktrees(self.repo)
        }
        identities = set(worktree_records)
        self.assertTrue({"demo-001", "demo-002"}.issubset(identities))
        self.assertEqual(
            worktree_records["demo-001"]["registration_path"],
            "docs/experiments/demo-001/registration.json",
        )
        self.assertEqual(
            worktree_records["demo-002"]["registration_path"],
            "docs/experiments/demo-002/registration.json",
        )

    def test_synthetic_freeze_and_legacy_authorization_reference_validate_schema(self):
        attempt_dir = self._freeze(
            "schema-attempt",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ok')",
        )
        schema = json.loads(
            (
                REPO / "docs/schemas/mancala-experiment-attempt-v1.schema.json"
            ).read_bytes()
        )
        validator = Draft202012Validator(schema)
        manifest = json.loads((attempt_dir / "manifest.json").read_bytes())
        validator.validate(manifest)
        self.assertIsNone(manifest["authorization_reference"])

        new_contract_manifest = dict(manifest)
        new_contract_manifest.pop("authorization_reference")
        validator.validate(new_contract_manifest)

        historical_manifest = dict(manifest)
        historical_manifest["authorization_reference"] = "historic-owner-approval"
        historical_manifest.pop("registration_path")
        historical_manifest.pop("registration_sha256")
        validator.validate(historical_manifest)
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

    def test_cleanup_enumerates_attempt_directories_missing_manifest(self) -> None:
        self._publish_and_merge_attempt("attempt-manifest-missing")
        invalid_attempt = (
            self.worktree
            / ".experiment"
            / "runtime"
            / "attempts"
            / "attempt-no-manifest"
        )
        invalid_attempt.mkdir()
        with self.assertRaisesRegex(ValueError, "attempt_manifest_missing"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")

    def test_cleanup_rejects_runtime_files_outside_attempt_directories(self) -> None:
        self._publish_and_merge_attempt("attempt-orphan-runtime")
        orphan = self.worktree / ".experiment" / "runtime" / "orphan.json"
        orphan.write_text("unaccounted\n")
        with self.assertRaisesRegex(ValueError, "unaccounted_runtime_artifact"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")

    def test_cleanup_rejects_attempt_directory_identity_mismatch(self) -> None:
        self._publish_and_merge_attempt("attempt-identity-source")
        attempts_root = self.worktree / ".experiment" / "runtime" / "attempts"
        invalid_attempt = attempts_root / "attempt-identity-substitution"
        invalid_attempt.mkdir()
        source_manifest = attempts_root / "attempt-identity-source" / "manifest.json"
        (invalid_attempt / "manifest.json").write_bytes(source_manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "attempt_directory_identity_invalid"):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")

    def test_cleanup_rejects_inconsistent_terminal_records(self) -> None:
        attempt_dir = self._publish_and_merge_attempt("attempt-terminal-conflict")
        (attempt_dir / "failed.json").write_text("{}\n")
        with self.assertRaisesRegex(
            ValueError, "attempt_lifecycle_records_inconsistent"
        ):
            lifecycle.safe_remove_worktree(self.repo, self.worktree, merged_into="main")

    def test_execution_rejects_dirty_source_after_freeze(self) -> None:
        attempt_dir = self._freeze(
            "attempt-source-dirty",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        (self.worktree / "README.md").write_text("modified after freeze\n")
        result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "failed")
        self.assertIn("execution_worktree_not_clean", result["reason"])
        self.assertFalse((attempt_dir / "result.json").exists())

    def test_execution_rejects_changed_head_after_freeze(self) -> None:
        attempt_dir = self._freeze(
            "attempt-head-changed",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        (self.worktree / "README.md").write_text("new source commit\n")
        self._git_at(self.worktree, "add", "README.md")
        self._git_at(self.worktree, "commit", "-m", "change source after freeze")
        result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "failed")
        self.assertIn("frozen_source_commit_or_tree_changed", result["reason"])
        self.assertFalse((attempt_dir / "result.json").exists())
        self.assertEqual(lifecycle.verify_attempt(attempt_dir)["status"], "failed")

    def test_execution_rejects_changed_input_after_freeze(self) -> None:
        attempt_dir = self._freeze(
            "attempt-input-changed",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        self.dataset_member.write_text('{"sample":2}\n')
        result = self._execute(attempt_dir)
        self.assertEqual(result["status"], "failed")
        self.assertIn("input_artifact_hash_mismatch", result["reason"])
        self.assertFalse((attempt_dir / "result.json").exists())

    def test_directory_inputs_hash_verify_and_reject_changed_members(self) -> None:
        attempt_dir = self._freeze(
            "attempt-directory-input",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        frozen_directory_hash = lifecycle.artifact_sha256(self.dataset_dir)
        self.assertEqual(
            json.loads((attempt_dir / "manifest.json").read_bytes())["input_artifacts"][
                str(self.dataset_dir)
            ],
            frozen_directory_hash,
        )
        self._execute(attempt_dir)
        lifecycle.verify_attempt(attempt_dir)
        self.dataset_member.write_text('{"sample":changed}\n')
        with self.assertRaisesRegex(ValueError, "input_artifact_hash_mismatch"):
            lifecycle.verify_attempt(attempt_dir)
        self.dataset_member.write_text('{"sample":1}\n')

        changed_before_launch = self._freeze(
            "attempt-directory-modified-before-run",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('result')",
        )
        self.dataset_member.write_text('{"sample":3}\n')
        result = self._execute(changed_before_launch)
        self.assertEqual(result["status"], "failed")
        self.assertIn("input_artifact_hash_mismatch", result["reason"])
        self.assertFalse((changed_before_launch / "result.json").exists())
        failed_archive = lifecycle.archive_attempt(changed_before_launch, self.archives)
        failed_archive_manifest = json.loads(
            Path(failed_archive["archive_manifest_path"]).read_bytes()
        )
        self.assertFalse(
            failed_archive_manifest["attempt_verification"]["input_artifacts_verified"]
        )
        self.assertEqual(
            failed_archive_manifest["attempt_verification"][
                "input_artifact_mismatches"
            ],
            [str(self.dataset_dir)],
        )
        self.dataset_member.write_text('{"sample":1}\n')

        directory_output = self._freeze(
            "attempt-directory-output",
            "import sys; from pathlib import Path; output=Path(sys.argv[1]); output.mkdir(); (output/'weights.bin').write_bytes(b'model')",
            output_path="model-artifact",
        )
        self.assertEqual(self._execute(directory_output)["status"], "completed")
        self.assertEqual(
            lifecycle.verify_attempt(directory_output)["outputs_verified"], 1
        )
        reference = lifecycle.archive_attempt(directory_output, self.archives)
        sidecar = json.loads(Path(reference["archive_manifest_path"]).read_bytes())
        self.assertIn(
            "model-artifact/weights.bin", {item["path"] for item in sidecar["files"]}
        )
        publication = lifecycle.prepare_publication(
            directory_output, self.worktree / "docs" / "data" / "directory-model-output"
        )
        self.assertIn(
            "model-artifact/weights.bin",
            {item["path"] for item in publication["files"]},
        )

    def test_historical_attempts_survive_source_and_publication_commits(self) -> None:
        attempt_a = self._freeze(
            "attempt-0001",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('A')",
        )
        result_a = self._execute(attempt_a)
        self.assertEqual(result_a["status"], "completed")
        commit_a = json.loads((attempt_a / "manifest.json").read_bytes())[
            "source_commit"
        ]

        (self.worktree / "README.md").write_text("implementation commit B\n")
        self._git_at(self.worktree, "add", "README.md")
        self._git_at(self.worktree, "commit", "-m", "update implementation to B")
        commit_b = self._git_at(self.worktree, "rev-parse", "HEAD")
        self.assertNotEqual(commit_a, commit_b)

        attempt_b = self._freeze(
            "attempt-0002",
            "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('B')",
        )
        result_b = self._execute(attempt_b)
        self.assertEqual(result_b["status"], "completed")
        self.assertEqual(
            json.loads((attempt_b / "manifest.json").read_bytes())["source_commit"],
            commit_b,
        )

        lifecycle.archive_attempt(attempt_a, self.archives)
        lifecycle.archive_attempt(attempt_b, self.archives)
        self.assertEqual(lifecycle.verify_attempt(attempt_a)["status"], "completed")
        self.assertEqual(lifecycle.verify_attempt(attempt_b)["status"], "completed")
        lifecycle.prepare_publication(
            attempt_a, self.worktree / "docs" / "data" / "demo-001" / "attempt-0001"
        )
        lifecycle.prepare_publication(
            attempt_b, self.worktree / "docs" / "data" / "demo-001" / "attempt-0002"
        )
        self._git_at(self.worktree, "add", "docs/data/demo-001")
        self._git_at(self.worktree, "commit", "-m", "publish both historical attempts")
        self.assertEqual(lifecycle.verify_attempt(attempt_a)["status"], "completed")
        self.assertEqual(lifecycle.verify_attempt(attempt_b)["status"], "completed")
        self._git(
            "merge", "--no-ff", "experiment/demo-001", "-m", "merge both attempts"
        )
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
