"""Regression checks for the read-only #561 historical inventory."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import seed561_historical_exclusion_inventory as inventory


class HistoricalInventoryTest(unittest.TestCase):
    def test_published_index_check_accepts_gzip_variants_and_rejects_content_read_only(
        self,
    ):
        root = inventory.ROOT
        canonical_jsonl = b'{"identity":"a","source":"fixture","provenance":{}}\n'
        float32_jsonl = b'{"identity":"b","source":"fixture","provenance":{}}\n'
        artifacts = {
            "source-manifest.json": b"manifest\n",
            "reconciliation-report.json": b"report\n",
            "canonical-index.jsonl.gz": gzip.compress(canonical_jsonl, mtime=0),
            "kalah-v3-float32-index.jsonl.gz": gzip.compress(float32_jsonl, mtime=0),
        }
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as temporary:
            output = Path(temporary)
            for name, payload in artifacts.items():
                (output / name).write_bytes(payload)
            # Alternate gzip metadata/compression preserves the JSONL evidence.
            canonical_path = output / "canonical-index.jsonl.gz"
            canonical_path.write_bytes(
                gzip.compress(canonical_jsonl, compresslevel=1, mtime=123)
            )
            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in output.iterdir()
            }
            with patch.object(inventory, "build", return_value=(artifacts, {})):
                inventory.verify(root, output)
                self.assertEqual(
                    {
                        path: (path.read_bytes(), path.stat().st_mtime_ns)
                        for path in output.iterdir()
                    },
                    before,
                )
                changed_jsonl = (
                    b'{"identity":"changed","source":"fixture","provenance":{}}\n'
                )
                canonical_path.write_bytes(gzip.compress(changed_jsonl, mtime=0))
                tampered = (
                    canonical_path.read_bytes(),
                    canonical_path.stat().st_mtime_ns,
                )
                with self.assertRaisesRegex(
                    inventory.InventoryError,
                    "published_artifact_mismatch:canonical-index.jsonl.gz",
                ):
                    inventory.verify(root, output)
                self.assertEqual(
                    (canonical_path.read_bytes(), canonical_path.stat().st_mtime_ns),
                    tampered,
                )

    def test_regenerated_local_source_receipt_cannot_replace_the_pinned_registration(
        self,
    ):
        root = inventory.ROOT
        registration = json.loads((root / inventory.REGISTRATION_REL).read_text())
        source, _ = next(iter(registration["exclusion_source_sha256"].items()))
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as temporary:
            copy = Path(temporary)
            registration_path = copy / inventory.REGISTRATION_REL
            registration_path.parent.mkdir(parents=True)
            (copy / source).parent.mkdir(parents=True, exist_ok=True)
            source_bytes = (root / source).read_bytes()
            (copy / source).write_bytes(source_bytes + b"\nchanged\n")
            registration["exclusion_source_sha256"][source] = inventory._sha(
                copy / source
            )
            registration_path.write_text(json.dumps(registration))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml.alphazero_lite.seed561_historical_exclusion_inventory",
                    "--root",
                    str(copy),
                    "--out",
                    str(copy / inventory.OUT_REL),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("seed511_registration_hash_mismatch", completed.stderr)

    def test_versioned_inventory_verifies_from_a_physical_evidence_copy(self):
        root = inventory.ROOT
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as temporary:
            copy = Path(temporary) / "copy"
            copy.mkdir()
            # Copy only pinned evidence and pure-Python source. Runtime models,
            # weights, and tablebases are omitted.
            source_map = json.loads((root / inventory.REGISTRATION_REL).read_text())[
                "exclusion_source_sha256"
            ]
            required = set(source_map) | {
                str(inventory.REGISTRATION_REL),
                "docs/data/seed556-560-exclusion-inventory/identity-inventory.json",
                "docs/data/seed556-560-exclusion-inventory/source-manifest.json",
                "docs/data/seed556-560-exclusion-inventory/count-report.json",
                "docs/data/expanded-training-dev-exclusion-inventory/identity-inventory.jsonl",
                "docs/data/expanded-training-dev-exclusion-inventory/source-manifest.json",
                "docs/data/expanded-training-dev-exclusion-inventory/count-report.json",
                "docs/data/seed564-archived-screen-identity-audit/seed564-seven-source-manifest.json",
                "docs/data/seed564-archived-screen-identity-audit/seven-source-canonical-index.jsonl.gz",
                "docs/data/seed564-archived-screen-identity-audit/seven-source-kalah-v3-float32-index.jsonl.gz",
                "docs/data/seed564-archived-screen-identity-audit/seven-source-count-report.json",
                "docs/data/seed564-archived-screen-identity-audit/seed545-source-manifest.json",
                "docs/data/seed564-archived-screen-identity-audit/seed545-canonical-index.jsonl.gz",
                "docs/data/seed564-archived-screen-identity-audit/seed545-kalah-v3-float32-index.jsonl.gz",
                "docs/data/seed564-archived-screen-identity-audit/seed545-replay-report.json",
                "docs/data/seed564-archived-screen-identity-audit/runtime-cross-checks.json",
                "docs/data/seed556-exploratory-arena/screen-0005/verification-receipt.json",
                "docs/data/seed556-exploratory-arena/screen-0005/registration.json",
                "docs/data/seed556-exploratory-arena/screen-0005/execution-freeze.json",
                "docs/data/seed556-exploratory-arena/screen-0005/openings.jsonl",
                "docs/data/seed556-exploratory-arena/screen-0005/outcomes.jsonl",
                "docs/data/seed560-paired-first-action/run-0004/cohort-manifest.json",
                "docs/data/seed560-paired-first-action/run-0004/execution-freeze.json",
                "docs/data/seed560-paired-first-action/run-0004/continuations.jsonl",
                "docs/data/seed557-retrospective-diagnostic/cohort.json",
                "docs/data/seed558-value-only-swap/cohort.json",
                "docs/data/seed559-fixed-position-feasibility/run-0003/cohort-manifest.json",
            }
            seven_manifest_path = (
                root
                / "docs/data/seed564-archived-screen-identity-audit/seed564-seven-source-manifest.json"
            )
            seven_manifest = json.loads(seven_manifest_path.read_text())
            required.add(str(seven_manifest_path.relative_to(root)))
            for source in seven_manifest["sources"]:
                required.update(source["anchor_files"])
                required.update(source["failed_or_superseded_attempts"])
                required.update(
                    f"{source['archive']}/{relative}"
                    for relative in source["game_batches"]
                )
                required.add(source["receipt_path"])
                required.add(f"{source['archive']}/openings.jsonl")
                required.add(f"{source['archive']}/outcomes.jsonl")
                required.update(
                    f"{source['archive']}/batches/report-{index:04d}.json"
                    for index in range(1024)
                )
            required.update(
                f"docs/data/seed556-exploratory-arena/screen-0005/batches/games-{index:04d}.jsonl"
                for index in range(1024)
            )
            for relative in required:
                source = root / relative
                if not source.is_file():
                    continue
                destination = copy / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            shutil.copytree(root / "ml", copy / "ml", dirs_exist_ok=True)
            output = copy / inventory.OUT_REL
            for name in (
                "source-manifest.json",
                "reconciliation-report.json",
                "canonical-index.jsonl.gz",
                "kalah-v3-float32-index.jsonl.gz",
            ):
                source = root / inventory.OUT_REL / name
                if source.is_file():
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            evidence_paths = [copy / relative for relative in source_map]
            evidence_paths.extend(
                copy / relative
                for relative in required
                if relative not in source_map
                and (copy / relative).is_file()
                and relative.startswith("docs/data/")
            )
            before = {
                path: (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in evidence_paths
            }
            self.assertFalse((copy / "model-artifact").exists())
            self.assertFalse(any(copy.rglob("*.kvtb")))
            self.assertFalse(any(copy.rglob("*.npz")))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ml.alphazero_lite.seed561_historical_exclusion_inventory",
                    "--root",
                    str(copy),
                    "--out",
                    str(output),
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(
                result["subtotal_562_564"],
                {"canonical": 302402, "kalah_v3_float32": 302402},
            )
            after = {
                path: (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in evidence_paths
            }
            self.assertEqual(after, before)
