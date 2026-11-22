"""Integration and adversarial tests for the Seed 627 → Seed 624 handoff."""

from __future__ import annotations

import json
import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed624_prospective_benchmark as seed624
from ml.alphazero_lite import seed628_authenticated_screening as adapter
from ml.alphazero_lite import verify_seed628_authenticated_screening as independent
from ml.alphazero_lite.verify_seed627_membership_snapshots import ROOT as SEED627_ROOT


class Seed628HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = adapter.authenticate_snapshot_bundle()

    def candidates(self) -> list[dict]:
        exact = self.evidence["B512"]["rows"][0]
        heldout = self.evidence["protected_holdout"]["rows"][0]
        state = {
            "current_player": 0,
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "opponent_store": 0,
            "player_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 48,
        }
        cid, fid = seed624.identities(state)
        nonmatch = {
            "canonical_state": state,
            "canonical_identity": cid,
            "production_float32_identity": fid,
        }
        return [
            {
                "canonical_identity": row["canonical_identity"],
                "production_float32_identity": row["production_float32_identity"],
                "opening_identity": "synthetic-opening-a",
                "game_id": "synthetic-game-a",
            }
            for row in (exact, heldout, nonmatch)
        ]

    def test_authenticated_adapter_and_independent_classification_agree(self) -> None:
        runtime = self.evidence["authentication"]["historical_runtime"]
        self.assertEqual(runtime["numpy"], "2.5.2")
        self.assertEqual(runtime["packages"], adapter._locked_packages())
        self.assertEqual(
            self.evidence["authentication"]["identity_runtime"]["screening_numpy"],
            "2.5.3",
        )
        candidates = self.candidates()
        screened = seed624.screen_known_exposure(
            candidates,
            self.evidence,
            expected_snapshot_hashes=adapter.expected_snapshot_hashes(self.evidence),
            expected_provenance_hash=adapter.expected_provenance_hash(self.evidence),
        )
        report = independent.verify_screening(candidates, self.evidence, screened)
        self.assertEqual(
            report["source_provenance_status"], "authenticated_non_crosswalkable"
        )
        self.assertEqual(report["classifications"]["confirmed_known_exposure"], 2)
        self.assertEqual(
            report["classifications"][
                "known_exposure_nonmatch_inherited_PE4_unresolved"
            ],
            1,
        )
        self.assertFalse(report["inherited_PE4_clearance_claimed"])
        self.assertTrue(screened[0]["exposure"]["B512"])
        self.assertTrue(screened[1]["exposure"]["protected_holdout"])
        self.assertIsNone(screened[2]["source_provenance_match"])
        self.assertEqual(screened[2]["inherited_PE4_exposure"], "unresolved")

    def test_spoofed_runtime_metadata_is_not_accepted(self) -> None:
        fake = copy.deepcopy(self.evidence)
        fake["authentication"]["historical_runtime"]["numpy"] = "2.5.3"
        with self.assertRaisesRegex(
            ValueError, "production_evidence_does_not_match_pinned_seed627_chain"
        ):
            independent.require_pinned_evidence(fake, self.evidence)

    def test_incorrect_historical_environment_is_rejected_by_actual_runtime_probe(
        self,
    ) -> None:
        wrong_runtime = adapter._historical_runtime_probe(
            SEED627_ROOT / ".venv-azlite/bin/python"
        )
        manifest = json.loads(
            (
                adapter.SNAPSHOT_DIR / "source-identity-lineage-manifest.json"
            ).read_bytes()
        )
        with self.assertRaisesRegex(
            ValueError, "historical_runtime_packages_do_not_match_lock"
        ):
            adapter._validate_historical_runtime(wrong_runtime, manifest)

    def test_recomputed_forged_membership_rows_do_not_authenticate(self) -> None:
        forged = copy.deepcopy(self.evidence)
        source = forged["B512"]
        row = source["rows"][0]
        row["canonical_state"]["forged_membership_claim"] = True
        row["canonical_identity"], row["production_float32_identity"] = (
            seed624.identities(row["canonical_state"])
        )
        source["sha256"] = seed624.sha256(seed624.canonical_bytes(source["rows"]))
        source["file_sha256"] = seed624.sha256(
            seed624.canonical_bytes(source["rows"]) + b"\n"
        )
        with self.assertRaisesRegex(
            ValueError, "production_evidence_does_not_match_pinned_seed627_chain"
        ):
            independent.require_pinned_evidence(forged, self.evidence)

    def test_valid_looking_non_pinned_manifest_is_rejected(self) -> None:
        forged = copy.deepcopy(self.evidence)
        forged["authentication"]["manifest_sha256"] = "a" * 64
        forged["source_provenance"]["source_manifest_sha256"] = "a" * 64
        provenance = forged["source_provenance"]
        provenance["sha256"] = seed624.sha256(
            seed624.canonical_bytes(
                {
                    key: value
                    for key, value in provenance.items()
                    if key != "authenticated"
                }
            )
        )
        with self.assertRaisesRegex(
            ValueError, "production_evidence_does_not_match_pinned_seed627_chain"
        ):
            independent.require_pinned_evidence(forged, self.evidence)

    def test_caller_controlled_self_consistent_evidence_cannot_enter_production(
        self,
    ) -> None:
        forged_rows = [self.evidence["B512"]["rows"][0]]
        fake_hash = seed624.sha256(seed624.canonical_bytes(forged_rows))
        freeze = {
            "membership_snapshot_sha256": {name: fake_hash for name in adapter.NAMES},
            "source_provenance_sha256": "f" * 64,
        }
        with tempfile.TemporaryDirectory(dir=".tmp") as temporary:
            evidence_path = Path(temporary) / "caller-evidence.json"
            evidence_path.write_text(
                json.dumps({"rows": forged_rows, "sha256": fake_hash})
            )
            with self.assertRaisesRegex(
                PermissionError, "caller_controlled_membership_evidence_forbidden"
            ):
                seed624.authenticate_frozen_membership_evidence(
                    freeze, caller_evidence_path=str(evidence_path)
                )
            with patch.object(
                adapter, "authenticate_snapshot_bundle", return_value=self.evidence
            ):
                with self.assertRaisesRegex(
                    PermissionError,
                    "execution_freeze_seed627_snapshot_hash_mismatch",
                ):
                    seed624.authenticate_frozen_membership_evidence(freeze)

    def test_tampered_snapshot_bytes_and_manifest_lineage_fail_pinned_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as temporary:
            copied = Path(temporary) / "snapshots"
            shutil.copytree(adapter.SNAPSHOT_DIR, copied)
            path = copied / "B512.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                ValueError, "pinned_seed627_file_hash_mismatch:B512.json"
            ):
                adapter.authenticate_snapshot_bundle(copied)

        with tempfile.TemporaryDirectory(dir=".tmp") as temporary:
            copied = Path(temporary) / "snapshots"
            shutil.copytree(adapter.SNAPSHOT_DIR, copied)
            manifest_path = copied / "source-identity-lineage-manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["training_lineage"].pop("B512")
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                ValueError,
                "pinned_seed627_file_hash_mismatch:source-identity-lineage-manifest.json",
            ):
                independent.reconstruct_evidence(copied)

    def test_missing_provenance_hash_and_invalid_crosswalk_claim_fail_closed(
        self,
    ) -> None:
        candidates = self.candidates()
        provenance = self.evidence["source_provenance"]
        original = provenance["sha256"]
        provenance["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source_provenance_hash_mismatch"):
            seed624.screen_known_exposure(
                candidates,
                self.evidence,
                expected_snapshot_hashes=adapter.expected_snapshot_hashes(
                    self.evidence
                ),
                expected_provenance_hash=original,
            )
        provenance["sha256"] = original
        forged = dict(self.evidence)
        forged_provenance = dict(provenance, crosswalk_status="complete")
        forged_provenance["sha256"] = seed624.sha256(
            seed624.canonical_bytes(
                {
                    key: value
                    for key, value in forged_provenance.items()
                    if key not in {"authenticated", "sha256"}
                }
            )
        )
        forged["source_provenance"] = forged_provenance
        with self.assertRaisesRegex(ValueError, "crosswalk_claim_invalid"):
            independent.reconstruct_screening(candidates, forged)

    def test_inconsistent_classification_label_is_rejected(self) -> None:
        candidates = self.candidates()
        screened = seed624.screen_known_exposure(
            candidates,
            self.evidence,
            expected_snapshot_hashes=adapter.expected_snapshot_hashes(self.evidence),
            expected_provenance_hash=adapter.expected_provenance_hash(self.evidence),
        )
        reconstructed = independent.reconstruct_screening(candidates, self.evidence)
        screened[2]["classification"] = "confirmed_known_exposure"
        with self.assertRaisesRegex(
            ValueError, "independent_screening_reconstruction_mismatch"
        ):
            independent.require_screening_matches(reconstructed, screened)

    def test_missing_and_unauthenticated_provenance_are_distinct(self) -> None:
        candidate = self.candidates()[1:]
        missing = dict(self.evidence)
        missing.pop("source_provenance")
        with self.assertRaisesRegex(ValueError, "source_provenance_missing"):
            seed624.screen_known_exposure(
                candidate,
                missing,
                expected_snapshot_hashes=adapter.expected_snapshot_hashes(
                    self.evidence
                ),
                expected_provenance_hash=adapter.expected_provenance_hash(
                    self.evidence
                ),
            )
        unauthenticated = dict(self.evidence)
        unauthenticated["source_provenance"] = dict(
            self.evidence["source_provenance"], authenticated=False
        )
        with self.assertRaisesRegex(ValueError, "source_provenance_unauthenticated"):
            seed624.screen_known_exposure(
                candidate,
                unauthenticated,
                expected_snapshot_hashes=adapter.expected_snapshot_hashes(
                    self.evidence
                ),
                expected_provenance_hash=adapter.expected_provenance_hash(
                    self.evidence
                ),
            )


if __name__ == "__main__":
    unittest.main()
