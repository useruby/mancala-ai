"""Adversarial tests for seed418's artifact-independent public verifier."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.verify_seed418_public_correction import verify

ROOT = Path(__file__).resolve().parents[2]
DATA_REL = Path("docs/data/seed418-native-root-handoff")


class Seed418PublicVerifierTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "docs/data").mkdir(parents=True)
        shutil.copytree(ROOT / DATA_REL, self.root / DATA_REL)
        (self.root / "docs/data/seed416-policy-target-softening").symlink_to(
            ROOT / "docs/data/seed416-policy-target-softening"
        )
        (self.root / "ml").symlink_to(ROOT / "ml")
        (self.root / "native").symlink_to(ROOT / "native")
        self.data = self.root / DATA_REL

    def tearDown(self):
        self.temp.cleanup()

    def read_rows(self, name):
        path = self.data / name
        return [json.loads(line) for line in path.read_text().splitlines()]

    def write_rows(self, name, rows):
        (self.data / name).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )

    def refresh_receipt_hash(self, receipt_field, filename):
        path = self.data / "correction-receipt.json"
        receipt = json.loads(path.read_text())
        digest = hashlib.sha256((self.data / filename).read_bytes()).hexdigest()
        if receipt_field == "published_records":
            receipt[receipt_field][filename] = digest
        else:
            receipt[receipt_field]["sha256"] = digest
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    def test_public_verify_needs_no_runtime_artifacts_and_is_read_only(self):
        for name in ("search-records.jsonl", "oracle-records.jsonl"):
            (self.data / name).chmod(0o444)
        # Runtime exports, native executable, and tablebase are intentionally absent.
        before = {
            path.name: path.read_bytes()
            for path in self.data.iterdir()
            if path.is_file()
        }
        result = verify(self.root)
        self.assertEqual(128, result["cases"])
        self.assertEqual("stop_seed418_branch", result["decision"])
        self.assertEqual(
            before,
            {
                path.name: path.read_bytes()
                for path in self.data.iterdir()
                if path.is_file()
            },
        )

    def test_verifier_imports_no_torch_or_arena(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import ml.alphazero_lite.verify_seed418_public_correction; "
                "assert 'torch' not in sys.modules; "
                "assert 'ml.alphazero_lite.arena' not in sys.modules",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_rejects_altered_checkpoint_registration_hash(self):
        rows = self.read_rows("execution-checkpoint-v2.jsonl")
        rows[0]["preregistration_sha256"] = "0" * 64
        self.write_rows("execution-checkpoint-v2.jsonl", rows)
        self.refresh_receipt_hash(
            "completed_checkpoint", "execution-checkpoint-v2.jsonl"
        )
        with self.assertRaisesRegex(
            ValueError, "checkpoint_registration_identity_invalid"
        ):
            verify(self.root)

    def test_rejects_substituted_search_oracle_pair(self):
        rows = self.read_rows("execution-checkpoint-v2.jsonl")
        rows[0]["search"], rows[1]["search"] = rows[1]["search"], rows[0]["search"]
        self.write_rows("execution-checkpoint-v2.jsonl", rows)
        self.refresh_receipt_hash(
            "completed_checkpoint", "execution-checkpoint-v2.jsonl"
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_published_pair_mismatch"):
            verify(self.root)

    def test_rejects_missing_or_duplicate_checkpoint_case(self):
        rows = self.read_rows("execution-checkpoint-v2.jsonl")
        rows.pop()
        self.write_rows("execution-checkpoint-v2.jsonl", rows)
        self.refresh_receipt_hash(
            "completed_checkpoint", "execution-checkpoint-v2.jsonl"
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_case_count_invalid"):
            verify(self.root)
        rows = [
            json.loads(line)
            for line in (ROOT / DATA_REL / "execution-checkpoint-v2.jsonl")
            .read_text()
            .splitlines()
        ]
        rows[1] = rows[0]
        self.write_rows("execution-checkpoint-v2.jsonl", rows)
        self.refresh_receipt_hash(
            "completed_checkpoint", "execution-checkpoint-v2.jsonl"
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_published_pair_mismatch"):
            verify(self.root)

    def test_rejects_wrong_seed_context(self):
        rows = self.read_rows("search-records.jsonl")
        rows[0]["seed_context"]["opening_index"] += 1
        self.write_rows("search-records.jsonl", rows)
        checkpoint = self.read_rows("execution-checkpoint-v2.jsonl")
        checkpoint[0]["search"] = rows[0]
        self.write_rows("execution-checkpoint-v2.jsonl", checkpoint)
        self.refresh_receipt_hash(
            "completed_checkpoint", "execution-checkpoint-v2.jsonl"
        )
        self.refresh_receipt_hash("published_records", "search-records.jsonl")
        with self.assertRaisesRegex(ValueError, "search_seed_context_invalid"):
            verify(self.root)

    def test_rejects_altered_evidence_and_missing_receipt(self):
        (self.data / "analysis.json").write_text("{}\n")
        with self.assertRaisesRegex(
            ValueError, "correction_receipt_identity_mismatch:analysis"
        ):
            verify(self.root)
        (self.data / "correction-receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "correction_receipt_missing"):
            verify(self.root)


if __name__ == "__main__":
    unittest.main()
