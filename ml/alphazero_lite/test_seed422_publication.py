from __future__ import annotations

import hashlib
import json
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


if __name__ == "__main__":
    unittest.main()
