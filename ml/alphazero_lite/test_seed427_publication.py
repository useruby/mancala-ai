"""Portable regression tests for supplemental seed426 publication checks."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.verify_seed427_publication import verify

REPOSITORY = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed426-canonical-overlap")


class Seed427PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch_parent = REPOSITORY / ".tmp"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="seed427-publication-", dir=scratch_parent
        )
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)
        self.root = self.scratch / "relocated publication"
        self.root.mkdir()
        paths = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--",
                "ml",
                "docs/data/seed416-policy-target-softening",
                str(DATA),
            ],
            cwd=REPOSITORY,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        for item in paths.split(b"\0"):
            if not item:
                continue
            relative = Path(item.decode())
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, target)
        for relative in (
            Path("ml/alphazero_lite/verify_seed427_publication.py"),
            Path("ml/alphazero_lite/verify_seed426_overlap_audit.py"),
            Path("ml/alphazero_lite/seed426_overlap_analysis.py"),
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, target)

    def _entry(self) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [
                sys.executable,
                str(self.root / "ml/alphazero_lite/verify_seed427_publication.py"),
                "--root",
                str(self.root),
            ],
            cwd=self.scratch,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_actual_publication_verifies_from_unrelated_directory_read_only(
        self,
    ) -> None:
        before = {
            p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        result = self._entry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "verified")
        after = {
            p.relative_to(self.root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(after, before)

    def test_supplemental_bound_fields_reject_false_values(self) -> None:
        path = self.root / DATA / "results.json"
        original = json.loads(path.read_text())
        cases = (
            (
                "split_positions",
                {"train": 1, "validation": 1},
                "supplemental_split_positions_mismatch",
            ),
            (
                "input_hashes",
                {name: "0" * 64 for name in original["input_hashes"]},
                "supplemental_input_hashes_mismatch",
            ),
        )
        for field, value, reason in cases:
            with self.subTest(field=field):
                altered = dict(original)
                altered[field] = value
                path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, reason):
                    verify(self.root)
        altered = dict(original)
        altered["loader_parity"] = {
            **original["loader_parity"],
            "frozen_train_sha256": "0" * 64,
        }
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(
            ValueError, "supplemental_frozen_train_identity_mismatch"
        ):
            verify(self.root)
        path.write_text(json.dumps(original))

    def test_publication_rejects_mutated_inputs_split_mapping_accounting_and_sources(
        self,
    ) -> None:
        data = self.root / DATA
        registration = json.loads(
            (
                self.root
                / "docs/data/seed416-policy-target-softening/registration-v3.json"
            ).read_text()
        )
        # Mutate a valid source row rather than its container, preserving gzip
        # readability so the verifier reaches the expected identity check.
        replay_path = data / "sources/fresh.jsonl.gz"
        original = replay_path.read_bytes()
        replay_lines = gzip.decompress(original).splitlines()
        row = json.loads(replay_lines[0])
        row["value"] = float(row["value"]) + 0.001
        replay_lines[0] = json.dumps(row).encode()
        self._assert_rejected(
            replay_path,
            gzip.compress(b"\n".join(replay_lines) + b"\n", mtime=0),
            "replay_snapshot_hash_mismatch:fresh",
        )

        split_path = self.root / registration["training"]["source_row_split"]["path"]
        split_original = split_path.read_bytes()
        self._assert_rejected(
            split_path,
            gzip.compress(gzip.decompress(split_original) + b" ", mtime=0),
            "registered_split_hash_mismatch",
        )

        mapping_path = data / "row-accounting.jsonl.gz"
        mapping_lines = gzip.decompress(mapping_path.read_bytes()).splitlines()
        first = json.loads(mapping_lines[0])
        first["weighted_position"] += 1
        mapping_lines[0] = json.dumps(first).encode()
        self._assert_rejected(
            mapping_path,
            gzip.compress(b"\n".join(mapping_lines) + b"\n", mtime=0),
            "row_mapping_mismatch",
        )

        results_path = data / "results.json"
        results = json.loads(results_path.read_text())
        results["identity_census"] = {}
        self._assert_rejected(
            results_path, json.dumps(results).encode(), "published_accounting_mismatch"
        )

        source_path = data / "execution-source-snapshots/train.py"
        self._assert_rejected(
            source_path,
            source_path.read_bytes() + b"\n",
            "execution_source_snapshot_mismatch:execution-source-snapshots/train.py",
        )

    def _assert_rejected(self, path: Path, altered: bytes, reason: str) -> None:
        original = path.read_bytes()
        try:
            path.write_bytes(altered)
            with self.assertRaisesRegex(ValueError, reason):
                verify(self.root)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
