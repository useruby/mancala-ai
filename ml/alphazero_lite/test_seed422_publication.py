from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed422_recovery import reconcile_resume_progress
from ml.alphazero_lite.verify_seed422_adam_memory import DATA, verify
from ml.alphazero_lite.seed422_corrected_analysis import analyze


class Seed422PublicationTests(unittest.TestCase):
    def test_protected_training_source_and_execution_snapshot_are_unchanged(self):
        registration = json.loads((DATA / "registration.json").read_text())
        relative = "ml/alphazero_lite/train.py"
        protected = Path(relative)
        snapshot = DATA / registration["source_snapshots"] / relative

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        expected = registration["source_hashes"][relative]
        self.assertEqual(digest(protected), expected)
        self.assertEqual(digest(snapshot), expected)

    def test_export_binding_matches_fixed_e4_and_root16_contract(self):
        training = json.loads((DATA / "training-results.json").read_text())
        binding = json.loads((DATA / "runtime-binding.json").read_text())
        self.assertEqual(
            binding["training_sha256"],
            hashlib.sha256((DATA / "training-results.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(binding["runtime_contract"]["exact_root_solve_threshold"], 16)
        for lane in ("A", "B"):
            candidate = binding["candidates"][lane]
            e4 = training["lanes"][lane]["epochs"]["4"]
            self.assertEqual(candidate["checkpoint_sha256"], e4)
            self.assertEqual(candidate["artifact_files"]["model.npz"], e4)
            self.assertEqual(
                candidate["artifact_files"]["search_policy.json"],
                binding["runtime_policy_sha256"],
            )
            self.assertEqual(candidate["runtime_contract"], binding["runtime_contract"])

    def test_checkpoint_and_export_identity_bindings_are_immutable(self):
        training = json.loads((DATA / "training-results.json").read_text())
        binding = json.loads((DATA / "runtime-binding.json").read_text())
        amendment = json.loads(
            (DATA / "execution-accounting-amendment.json").read_text()
        )
        self.assertEqual(
            amendment["A_epoch_checkpoint_hashes"], training["lanes"]["A"]["epochs"]
        )
        for lane in ("A", "B"):
            epoch_hashes = training["lanes"][lane]["epochs"]
            self.assertEqual(set(epoch_hashes), {"1", "2", "3", "4"})
            self.assertTrue(all(len(value) == 64 for value in epoch_hashes.values()))
            candidate = binding["candidates"][lane]
            self.assertEqual(candidate["checkpoint_sha256"], epoch_hashes["4"])
            self.assertEqual(
                candidate["artifact_files"]["model.npz"], epoch_hashes["4"]
            )

    def test_resume_accounting_accepts_only_committed_or_single_pending_epoch(self):
        progress = {"completed_epoch": 2, "history": [{}, {}]}
        committed = {"completed_epoch": 3, "history": [{}, {}, {}]}
        self.assertEqual(reconcile_resume_progress(progress, committed), committed)
        with self.assertRaisesRegex(ValueError, "epoch_state_conflict"):
            reconcile_resume_progress(progress, {"completed_epoch": 4, "history": []})
        with self.assertRaisesRegex(ValueError, "history_count_conflict"):
            reconcile_resume_progress(progress, {"completed_epoch": 3, "history": []})

    def test_read_only_verifier_replays_full_publication(self):
        result = verify()
        self.assertTrue(result["verified"])
        self.assertFalse(result["torch_required"])
        self.assertFalse(result["runtime_artifacts_required"])
        self.assertEqual(result["game_count"], 2048)
        self.assertEqual(result["suite_count"], 512)

    def test_corrected_seat_accounting_uses_512_game_denominators(self):
        rows = [
            json.loads(line)
            for line in (DATA / "outcome-ledger.jsonl").read_text().splitlines()
        ]
        result = analyze(rows)
        self.assertEqual(
            result["lanes"]["A"]["seat_scores"], {"0": 0.5322265625, "1": 0.43359375}
        )
        self.assertEqual(
            result["lanes"]["B"]["seat_scores"], {"0": 0.533203125, "1": 0.4658203125}
        )
        self.assertEqual(
            result["decision"], "retain_baseline_close_fixed_beta1_intervention"
        )
        rows.pop()
        with self.assertRaisesRegex(ValueError, "expected_exactly_2048_games"):
            analyze(rows)

    def test_correction_verifier_and_publication(self):
        from ml.alphazero_lite.verify_seed422_correction import (
            verify as verify_correction,
        )

        result = verify_correction()
        self.assertTrue(result["verified"])
        self.assertTrue(result["primary_estimates_unchanged"])

    def test_portable_public_entry_point_on_relocated_evidence(self):
        repository = Path(__file__).resolve().parents[2]
        git_files = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--",
                "ml",
                "docs/data/seed414-root-budget-confirmation",
                "docs/data/seed416-policy-target-softening",
                "docs/data/seed418-native-root-handoff",
                "docs/data/seed420-artifact-evaluator-memoization",
                "docs/data/seed421-memoization-timing-correction",
                "docs/data/seed422-adam-first-moment",
            ],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout.split(b"\0")
        relative_files = [Path(path.decode()) for path in git_files if path]
        with tempfile.TemporaryDirectory(
            prefix="seed422-public-verifier-", dir=repository / ".tmp"
        ) as temporary:
            relocated = Path(temporary)
            for relative in relative_files:
                source = repository / relative
                target = relocated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            # The new entry points are part of this branch and may not yet be
            # present in git's index when the focused tests are run.
            for name in (
                "seed422_public_exclusions.py",
                "verify_seed422_publication.py",
            ):
                source = repository / "ml/alphazero_lite" / name
                target = relocated / "ml/alphazero_lite" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            receipt_source = (
                repository
                / "docs/data/seed422-adam-first-moment/supplemental-verification-receipt.json"
            )
            receipt_target = (
                relocated
                / "docs/data/seed422-adam-first-moment/supplemental-verification-receipt.json"
            )
            shutil.copy2(receipt_source, receipt_target)
            documentation = repository / "docs/seed422-publication-verification.md"
            documentation_target = (
                relocated / "docs/seed422-publication-verification.md"
            )
            documentation_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(documentation, documentation_target)

            all_paths = list(relocated.rglob("*"))
            self.assertFalse(any(path.is_symlink() for path in all_paths))
            self.assertFalse(any("model-artifact" in path.parts for path in all_paths))
            self.assertFalse(
                any(".tmp" in path.relative_to(relocated).parts for path in all_paths)
            )
            fixture_hashes = {
                path.relative_to(relocated): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in all_paths
                if path.is_file()
            }
            env = {**os.environ, "PYTHONPATH": str(relocated)}
            command = [
                sys.executable,
                str(relocated / "ml/alphazero_lite/verify_seed422_publication.py"),
                "--root",
                str(relocated),
            ]

            def invoke(expected_success: bool) -> subprocess.CompletedProcess[str]:
                result = subprocess.run(
                    command,
                    cwd=relocated,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if expected_success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertTrue(payload["verified"])
                    self.assertEqual(payload["excluded_identity_count"], 266575)
                    self.assertEqual(
                        payload["decision"],
                        "retain_baseline_close_fixed_beta1_intervention",
                    )
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            invoke(expected_success=True)
            for path in all_paths:
                if path.is_file():
                    self.assertEqual(
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        fixture_hashes[path.relative_to(relocated)],
                    )

            data = relocated / "docs/data/seed422-adam-first-moment"
            mutations = {
                "registered source": relocated / "ml/alphazero_lite/arena.py",
                "registered snapshot": data
                / "execution-source-snapshots/ml/alphazero_lite/arena.py",
                "amendment": data / "execution-accounting-amendment.json",
                "runtime binding and candidate identity": data / "runtime-binding.json",
                "outcome binding and report": data / "outcome-binding.json",
                "raw ledger": data / "outcome-ledger.jsonl",
                "training settings": data / "training-results.json",
            }
            for _label, path in mutations.items():
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    invoke(expected_success=False)
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
