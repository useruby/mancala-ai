"""Synthetic checks for Seed 636 GitHub approval and lifecycle bindings."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed636_authorization as authorization


class Seed636AuthorizationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, dict, dict]:
        (root / "docs").mkdir()
        experiment_dir = root / ".experiment"
        experiment_dir.mkdir()
        document_names = sorted(authorization.REQUIRED_PROTOCOL_DOCUMENTS)
        documents = {}
        for name in document_names:
            document = root / name
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_text("approved protocol\n")
            documents[name] = authorization.sha256(document.read_bytes())
        input_path = root / "registered-input.json"
        input_path.write_text('{"input":true}\n')
        protocol_documents = documents
        registration = {
            "experiment_id": "seed636-prospective-population",
            "approved_protocol_revision": "a" * 40,
            "protocol_documents": protocol_documents,
            "input_artifacts": {
                str(input_path): authorization.sha256(input_path.read_bytes())
            },
            "approved_decisions": authorization.APPROVED_DECISIONS,
        }
        registration_path = experiment_dir / "registration.json"
        registration_raw = json.dumps(registration, sort_keys=True).encode()
        registration_path.write_bytes(registration_raw)
        body_record = {
            "marker": authorization.APPROVAL_MARKER,
            "experiment_id": registration["experiment_id"],
            "registration_sha256": authorization.sha256(registration_raw),
            "approved_protocol_revision": registration["approved_protocol_revision"],
            "protocol_documents": protocol_documents,
            "approved_decisions": authorization.APPROVED_DECISIONS,
            "execution_scope": authorization.EXECUTION_SCOPE,
        }
        body = (
            authorization.APPROVAL_MARKER
            + "\n"
            + json.dumps(body_record, sort_keys=True)
        )
        url = "https://github.com/useruby/mancala-ai/pull/636#issuecomment-1234"
        timestamp = "2026-11-24T12:00:00Z"
        comment = {
            "id": 1234,
            "html_url": url,
            "created_at": timestamp,
            "user": {"login": authorization.PROJECT_OWNER},
            "body": body,
        }
        record = {
            "schema": authorization.SCHEMA,
            "project_owner_identity": authorization.PROJECT_OWNER,
            "approval_url": url,
            "approval_timestamp": timestamp,
            "approved_protocol_revision": registration["approved_protocol_revision"],
            "protocol_documents": protocol_documents,
            "approved_decisions": authorization.APPROVED_DECISIONS,
            "execution_scope": authorization.EXECUTION_SCOPE,
            "experiment_id": registration["experiment_id"],
            "registration_sha256": authorization.sha256(registration_raw),
        }
        approval_path = root / "approval.json"
        approval_path.write_text(json.dumps(record))
        return registration_path, approval_path, record, comment

    def test_valid_github_owner_approval_is_bound_to_registration_and_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration, approval, _record, comment = self._fixture(root)
            loaded_registration, loaded_approval = authorization.validate_approval(
                registration, approval, comment_fetcher=lambda _comment_id: comment
            )
            self.assertEqual(
                loaded_registration["experiment_id"],
                "seed636-prospective-population",
            )
            self.assertEqual(
                loaded_approval["github_comment_evidence"]["author"], "useruby"
            )

    def test_missing_wrong_identity_scope_and_changed_documents_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration, approval_path, record, comment = self._fixture(root)
            with self.assertRaisesRegex(PermissionError, "approval_missing"):
                authorization.validate_approval(
                    registration,
                    root / ".experiment" / "missing.json",
                    comment_fetcher=lambda _comment_id: comment,
                )
            for field, value, message in (
                ("project_owner_identity", "not-owner", "identity_mismatch"),
                ("execution_scope", "all-experiments", "scope_mismatch"),
                ("approved_protocol_revision", "b" * 40, "protocol_revision_mismatch"),
            ):
                changed = dict(record, **{field: value})
                approval_path.write_text(json.dumps(changed))
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(PermissionError, message),
                ):
                    authorization.validate_approval(
                        registration,
                        approval_path,
                        comment_fetcher=lambda _comment_id: comment,
                    )
            approval_path.write_text(json.dumps(record))
            (root / "docs" / "experiment-execution-protocol.md").write_text("altered\n")
            with self.assertRaisesRegex(
                PermissionError, "protocol_document_hash_mismatch"
            ):
                authorization.validate_approval(
                    registration,
                    approval_path,
                    comment_fetcher=lambda _comment_id: comment,
                )

    def test_lifecycle_attempt_rejects_changed_source_and_input_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration_path, approval_path, record, comment = self._fixture(root)
            experiment_dir = root / ".experiment"
            (experiment_dir / "seed636-owner-approval.json").write_text(
                json.dumps(record)
            )
            attempt_dir = experiment_dir / "runtime" / "attempts" / "attempt-0001"
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "started.json").write_text("{}")
            registration = json.loads(registration_path.read_text())
            manifest = {
                "status": "frozen",
                "experiment_id": registration["experiment_id"],
                "attempt_id": "attempt-0001",
                "worktree_path": str(root),
                "source_commit": "b" * 40,
                "execution_source_identity": "c" * 40,
                "command": [
                    "python",
                    "-m",
                    "ml.alphazero_lite.seed624_prospective_benchmark",
                ],
                "configuration": {
                    "execution_scope": authorization.EXECUTION_SCOPE,
                    "registration_sha256": authorization.sha256(
                        registration_path.read_bytes()
                    ),
                },
                "input_artifacts": registration["input_artifacts"],
                "output_paths": [
                    ".experiment/runtime/attempts/attempt-0001/output/population"
                ],
                "worker_count": 1,
            }
            (attempt_dir / "manifest.json").write_text(json.dumps(manifest))
            with (
                patch(
                    "ml.alphazero_lite.experiment_lifecycle._validate_experiment_worktree",
                    return_value=registration,
                ),
                patch.object(
                    authorization,
                    "_git",
                    side_effect=[
                        ".experiment/seed636-owner-approval.json",
                        "b" * 40,
                        "c" * 40,
                        "",
                    ],
                ),
            ):
                loaded, _ = authorization.validate_attempt(
                    attempt_dir, comment_fetcher=lambda _id: comment
                )
            self.assertEqual(loaded["attempt_id"], "attempt-0001")

            input_file = Path(next(iter(registration["input_artifacts"])))
            input_file.write_text("changed input\n")
            with (
                patch(
                    "ml.alphazero_lite.experiment_lifecycle._validate_experiment_worktree",
                    return_value=registration,
                ),
                patch.object(
                    authorization,
                    "_git",
                    side_effect=[
                        ".experiment/seed636-owner-approval.json",
                        "b" * 40,
                        "c" * 40,
                        "",
                    ],
                ),
                self.assertRaisesRegex(
                    PermissionError, "lifecycle_input_identity_mismatch"
                ),
            ):
                authorization.validate_attempt(
                    attempt_dir, comment_fetcher=lambda _id: comment
                )

            input_file.write_text('{"input":true}\n')
            with (
                patch(
                    "ml.alphazero_lite.experiment_lifecycle._validate_experiment_worktree",
                    return_value=registration,
                ),
                patch.object(
                    authorization,
                    "_git",
                    side_effect=[
                        ".experiment/seed636-owner-approval.json",
                        "d" * 40,
                    ],
                ),
                self.assertRaisesRegex(
                    PermissionError, "lifecycle_source_commit_mismatch"
                ),
            ):
                authorization.validate_attempt(
                    attempt_dir, comment_fetcher=lambda _id: comment
                )

    def test_github_comment_author_timestamp_and_body_are_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration, approval, _record, comment = self._fixture(root)
            for field, value, message in (
                ("user", {"login": "not-owner"}, "comment_author_mismatch"),
                ("created_at", "2026-11-25T12:00:00Z", "timestamp_mismatch"),
                ("body", "approved", "comment_text_invalid"),
            ):
                altered = dict(comment, **{field: value})
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(PermissionError, message),
                ):
                    authorization.validate_approval(
                        registration, approval, comment_fetcher=lambda _id: altered
                    )


if __name__ == "__main__":
    unittest.main()
