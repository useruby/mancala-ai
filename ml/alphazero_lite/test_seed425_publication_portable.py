"""Clean-checkout regression coverage for the portable seed422 publication."""

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

from ml.alphazero_lite.seed422_corrected_analysis import analyze
from ml.alphazero_lite.verify_seed422_publication import verify


REPOSITORY = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed422-adam-first-moment")
PUBLIC_PATHS = (
    "ml",
    "docs/data/seed414-root-budget-confirmation",
    "docs/data/seed416-policy-target-softening",
    "docs/data/seed418-native-root-handoff",
    "docs/data/seed420-artifact-evaluator-memoization",
    "docs/data/seed421-memoization-timing-correction",
    "docs/data/seed422-adam-first-moment",
    "docs/seed422-publication-verification.md",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Seed425PortablePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        # A fresh checkout does not have this ignored parent yet.
        scratch_parent = REPOSITORY / ".tmp"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="seed425-portable-", dir=scratch_parent
        )
        self.addCleanup(self._temporary.cleanup)
        self.scratch = Path(self._temporary.name)
        self.root = self.scratch / "relocated checkout"
        self.root.mkdir()
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", *PUBLIC_PATHS],
            cwd=REPOSITORY,
            check=True,
            stdout=subprocess.PIPE,
        )
        files = [Path(item.decode()) for item in result.stdout.split(b"\0") if item]
        for relative in files:
            source = REPOSITORY / relative
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        self._assert_portable_fixture()

    def _assert_portable_fixture(self) -> None:
        paths = list(self.root.rglob("*"))
        self.assertFalse(any(path.is_symlink() for path in paths))
        forbidden = {".tmp", "__pycache__", ".pytest_cache", "model-artifact"}
        self.assertFalse(
            any(
                forbidden.intersection(path.relative_to(self.root).parts)
                for path in paths
            )
        )
        forbidden_suffixes = {".pt", ".pth", ".ckpt", ".npz", ".so", ".rtb"}
        self.assertFalse(any(path.suffix in forbidden_suffixes for path in paths))
        forbidden_names = {"mtdf", "kalah", "tablebase", "kalah.rtb"}
        self.assertFalse(any(path.name in forbidden_names for path in paths))

    def _invoke_entry_point(self, cwd: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return subprocess.run(
            [
                sys.executable,
                str(self.root / "ml/alphazero_lite/verify_seed422_publication.py"),
                "--root",
                str(self.root),
            ],
            cwd=cwd,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_publication_entry_point_replays_clean_relocated_checkout(self) -> None:
        before = {
            p.relative_to(self.root): digest(p)
            for p in self.root.rglob("*")
            if p.is_file()
        }
        result = self._invoke_entry_point(self.scratch)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["excluded_identity_count"], 266575)
        self.assertEqual(payload["suite_count"], 512)
        self.assertEqual(payload["game_count"], 2048)
        self.assertEqual(
            payload["decision"], "retain_baseline_close_fixed_beta1_intervention"
        )
        self.assertFalse(payload["torch_required"])
        self.assertFalse(payload["runtime_artifacts_required"])
        after = {
            p.relative_to(self.root): digest(p)
            for p in self.root.rglob("*")
            if p.is_file()
        }
        self.assertEqual(after, before, "verifier changed publication evidence")

    def test_pure_public_verifier_helper_matches_published_corrected_analysis(
        self,
    ) -> None:
        result = verify(self.root)
        self.assertEqual(result["excluded_identity_count"], 266575)
        rows = [
            json.loads(line)
            for line in (self.root / DATA / "outcome-ledger.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            analyze(rows),
            json.loads((self.root / DATA / "corrected-analysis.json").read_text()),
        )

    def test_publication_rejects_each_bound_evidence_mutation(self) -> None:
        data = self.root / DATA
        mutations = {
            "historical exclusion evidence": data / "opening-exclusion-proof.json",
            "registered source": self.root / "ml/alphazero_lite/arena.py",
            "source snapshot": data
            / "execution-source-snapshots/ml/alphazero_lite/arena.py",
            "amendment": data / "execution-accounting-amendment.json",
            "training results": data / "training-results.json",
            "runtime and candidate binding": data / "runtime-binding.json",
            "report and outcome binding": data / "outcome-binding.json",
            "ledger": data / "outcome-ledger.jsonl",
        }
        for label, path in mutations.items():
            with self.subTest(evidence=label):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b" ")
                    result = self._invoke_entry_point(self.scratch)
                    self.assertNotEqual(result.returncode, 0)
                finally:
                    path.write_bytes(original)
                self.assertEqual(digest(path), hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
