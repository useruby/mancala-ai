"""Synthetic checks for Seed 636 merged-registration authorization."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed636_authorization as authorization
from ml.alphazero_lite import verify_seed624_prospective_benchmark as verifier


class Seed636RegistrationAuthorizationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, str, dict, dict]:
        subprocess.run(
            ["git", "-C", str(root), "init", "-b", "main"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "Seed636 test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "config",
                "user.email",
                "seed636-test@example.invalid",
            ],
            check=True,
        )

        docs = {}
        for name in sorted(authorization.REQUIRED_PROTOCOL_DOCUMENTS):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"pinned protocol: {name}\n")
            docs[name] = authorization.sha256(path.read_bytes())
        subprocess.run(
            ["git", "-C", str(root), "add", "docs"], check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(root), "commit", "-m", "base protocol"],
            check=True,
            capture_output=True,
        )
        base_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        registration = {
            "schema": "mancala-experiment-registration-v1",
            "experiment_id": authorization.EXPERIMENT_ID,
            "authorization_contract": authorization.SCHEMA,
            "execution_scope": authorization.EXECUTION_SCOPE,
            "random_seeds": [62320261121],
            "approved_decisions": authorization.APPROVED_DECISIONS,
            "protocol_documents": docs,
            "resource_budgets": {
                "wall_time_limit_seconds": 14400,
                "cpu_time_limit_seconds": 10800,
                "process_rss_limit_bytes": 17179869184,
                "output_and_temporary_storage_limit_bytes": 21474836480,
                "execution_concurrency": 1,
                "selection_optimizer_workers": 1,
            },
            "scientific_acceptance": {
                "openings": 256,
                "continuation_games": 2048,
                "selected_states": 512,
                "stone_bin_quotas": {
                    "17": 103,
                    "18": 103,
                    "19": 102,
                    "20": 102,
                    "21": 102,
                },
                "minimum_eligible_candidate_supply": {
                    "17": 206,
                    "18": 206,
                    "19": 204,
                    "20": 204,
                    "21": 204,
                },
                "minimum_distinct_opening_clusters": 128,
                "max_states_per_opening": 4,
                "max_states_per_continuation_game": 2,
                "exposure_classification": "known-exposure-screened; inherited P-E4 exposure unresolved",
                "deduplication": ["canonical_state", "production_float32_input"],
                "authenticated_known_exposure_screening": True,
                "independent_reproduction_required": True,
                "no_inference_mcts_exact_solving_tablebase_probes_training_or_promotion": True,
                "failure_policy": "preserve_evidence_and_stop_without_parameter_change_or_automatic_retry",
            },
        }
        registration_path = (
            root / "docs/experiments/seed636-prospective-population/registration.json"
        )
        registration_path.parent.mkdir(parents=True, exist_ok=True)
        registration_path.write_text(json.dumps(registration, sort_keys=True) + "\n")
        subprocess.run(
            ["git", "-C", str(root), "add", "docs/experiments"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(root), "commit", "-m", "register experiment"],
            check=True,
            capture_output=True,
        )
        merge_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        digest = authorization.sha256(registration_path.read_bytes())
        pull = {
            "number": 84,
            "state": "closed",
            "merged_at": "2026-11-25T12:30:00Z",
            "merged_by": {"login": "useruby"},
            "base": {
                "ref": "main",
                "sha": base_commit,
                "repo": {"full_name": "useruby/mancala-ai"},
            },
            "merge_commit_sha": merge_commit,
            "html_url": "https://github.com/useruby/mancala-ai/pull/84",
        }
        return (
            registration_path,
            merge_commit,
            pull,
            {
                "registration_sha256": digest,
                "registration_path": "docs/experiments/seed636-prospective-population/registration.json",
            },
        )

    def _authorize(self, registration: Path, commit: str, pull: dict) -> dict:
        return authorization.authorize_registration(
            registration,
            commit,
            associated_prs=[{"number": pull["number"]}],
            pr_fetcher=lambda _number: pull,
        )

    def test_owner_merged_pr_records_complete_authorization_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration, merge_commit, pull, expected = self._fixture(Path(temporary))
            receipt = self._authorize(registration, merge_commit, pull)
            self.assertEqual(receipt["schema"], authorization.SCHEMA)
            self.assertEqual(receipt["pr_number"], 84)
            self.assertEqual(receipt["merge_commit"], merge_commit)
            self.assertEqual(receipt["merged_by"], "useruby")
            self.assertEqual(receipt["merge_timestamp"], "2026-11-25T12:30:00Z")
            self.assertEqual(
                receipt["registration_path"], expected["registration_path"]
            )
            self.assertEqual(
                receipt["registration_sha256"], expected["registration_sha256"]
            )

    def test_non_owner_unmerged_or_wrong_base_pr_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration, merge_commit, pull, _expected = self._fixture(Path(temporary))
            for field, value in (
                ("merged_by", {"login": "another-user"}),
                ("merged_at", None),
                (
                    "base",
                    {"ref": "develop", "repo": {"full_name": "useruby/mancala-ai"}},
                ),
            ):
                modified = dict(pull, **{field: value})
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        PermissionError, "owner_merged_registration_pr_not_unique"
                    ),
                ):
                    self._authorize(registration, merge_commit, modified)

    def test_registration_and_protocol_must_match_merge_commit_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration, merge_commit, pull, _expected = self._fixture(Path(temporary))
            registration.write_text(registration.read_text() + " ")
            with self.assertRaisesRegex(
                PermissionError, "owner_merged_registration_pr_not_unique"
            ):
                self._authorize(registration, merge_commit, pull)

    def test_unrelated_later_pr_cannot_reauthorize_inherited_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration, registration_merge, _registration_pr, expected = (
                self._fixture(root)
            )
            unrelated = root / "unrelated.txt"
            unrelated.write_text("unrelated change\n")
            subprocess.run(
                ["git", "-C", str(root), "add", "unrelated.txt"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-m", "unrelated later PR"],
                check=True,
                capture_output=True,
            )
            later_merge = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            later_pull = {
                "number": 85,
                "state": "closed",
                "merged_at": "2026-11-25T13:30:00Z",
                "merged_by": {"login": "useruby"},
                "base": {
                    "ref": "main",
                    "sha": registration_merge,
                    "repo": {"full_name": "useruby/mancala-ai"},
                },
                "merge_commit_sha": later_merge,
                "html_url": "https://github.com/useruby/mancala-ai/pull/85",
            }
            with self.assertRaisesRegex(
                PermissionError, "owner_merged_registration_pr_not_unique"
            ):
                self._authorize(registration, later_merge, later_pull)

            registration_raw = registration.read_bytes()
            inherited_receipt = {
                "schema": authorization.SCHEMA,
                "repository": authorization.REPOSITORY,
                "experiment_id": authorization.EXPERIMENT_ID,
                "registration_path": expected["registration_path"],
                "execution_scope": authorization.EXECUTION_SCOPE,
                "approved_decisions": authorization.APPROVED_DECISIONS,
                "registration_sha256": expected["registration_sha256"],
                "pr_number": 85,
                "pr_url": later_pull["html_url"],
                "merge_commit": later_merge,
                "merged_by": "useruby",
                "merge_timestamp": later_pull["merged_at"],
            }
            lifecycle = {
                "registration_sha256": expected["registration_sha256"],
                "registration_path": expected["registration_path"],
                "registration_raw_utf8": registration_raw.decode(),
            }
            with (
                patch.object(verifier, "_fetch_pull_associations", return_value={85}),
                patch.object(verifier, "_fetch_github_pull", return_value=later_pull),
                patch.object(
                    verifier, "_resolve_git_commit", return_value=registration_merge
                ),
                patch.object(verifier, "_git_is_ancestor", return_value=True),
                patch.object(
                    verifier,
                    "_git_blob",
                    side_effect=lambda revision, path: (
                        subprocess.run(
                            ["git", "-C", str(root), "show", f"{revision}:{path}"],
                            check=True,
                            capture_output=True,
                        ).stdout
                    ),
                ),
                self.assertRaisesRegex(
                    ValueError, "registration_pr_inherited_registration_not_authorized"
                ),
            ):
                verifier.verify_registration_pr_authorization(
                    inherited_receipt, lifecycle, later_merge
                )

    def test_unapproved_scope_or_budget_cannot_be_authorized(self):
        with tempfile.TemporaryDirectory() as temporary:
            registration, merge_commit, pull, _expected = self._fixture(Path(temporary))
            value = json.loads(registration.read_text())
            value["execution_scope"] = "model-inference"
            registration.write_text(json.dumps(value))
            with self.assertRaisesRegex(
                PermissionError, "merged_registration_scope_or_protocol_invalid"
            ):
                self._authorize(registration, merge_commit, pull)

    def test_independent_verifier_rechecks_live_pr_metadata_and_git_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registration, merge_commit, pull, expected = self._fixture(root)
            receipt = self._authorize(registration, merge_commit, pull)
            registration_raw = registration.read_bytes()
            lifecycle = {
                "registration_sha256": expected["registration_sha256"],
                "registration_path": expected["registration_path"],
                "registration_raw_utf8": registration_raw.decode(),
            }
            with (
                patch.object(verifier, "_fetch_pull_associations", return_value={84}),
                patch.object(verifier, "_fetch_github_pull", return_value=pull),
                patch.object(
                    verifier, "_resolve_git_commit", return_value=pull["base"]["sha"]
                ),
                patch.object(verifier, "_git_is_ancestor", return_value=True),
                patch.object(
                    verifier,
                    "_git_blob",
                    side_effect=lambda revision, path: (
                        subprocess.run(
                            ["git", "-C", str(root), "show", f"{revision}:{path}"],
                            check=True,
                            capture_output=True,
                        ).stdout
                    ),
                ),
            ):
                self.assertEqual(
                    verifier.verify_registration_pr_authorization(
                        receipt, lifecycle, merge_commit
                    ),
                    "useruby",
                )

            tampered = dict(receipt, merge_timestamp="2026-11-26T00:00:00Z")
            with (
                patch.object(verifier, "_fetch_pull_associations", return_value={84}),
                patch.object(verifier, "_fetch_github_pull", return_value=pull),
                patch.object(
                    verifier, "_resolve_git_commit", return_value=pull["base"]["sha"]
                ),
                patch.object(verifier, "_git_is_ancestor", return_value=True),
                self.assertRaisesRegex(
                    ValueError, "registration_pr_live_metadata_mismatch"
                ),
            ):
                verifier.verify_registration_pr_authorization(
                    tampered, lifecycle, merge_commit
                )


if __name__ == "__main__":
    unittest.main()
