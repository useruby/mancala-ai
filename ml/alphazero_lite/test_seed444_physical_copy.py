"""Exercise the seed444 CLI against a physical copy from an unrelated cwd."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path("docs/data/seed444-minibatch-denominator-audit")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_physical_copy_cli_is_read_only_from_unrelated_cwd() -> None:
    receipt = json.loads((ROOT / AUDIT / "receipt.json").read_text())
    correction = json.loads((ROOT / AUDIT / "correction-receipt.json").read_text())
    needed = (
        set(receipt["artifact_sha256"])
        | set(receipt["source_sha256"])
        | set(receipt["input_sha256"])
    )
    needed.add(f"{AUDIT}/receipt.json")
    needed.add(f"{AUDIT}/correction-receipt.json")
    needed.add(f"{AUDIT}/correction-receipt-amendment-1.json")
    needed.update(correction["changed_source_sha256"])
    _438 = (
        ROOT / "docs/data/seed438-seed437-gradient-correction/correction-receipt.json"
    )
    seed438_receipt = json.loads(_438.read_text())
    needed.update(seed438_receipt["corrected_sources"])
    with tempfile.TemporaryDirectory(
        prefix="seed444-physical-copy-", dir=ROOT / ".tmp"
    ) as scratch_name:
        scratch = Path(scratch_name)
        copied = scratch / "relocated root"
        unrelated = scratch / "unrelated working directory"
        copied.mkdir()
        unrelated.mkdir()
        for relative in sorted(needed):
            source = ROOT / relative
            destination = copied / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        evidence_paths = [
            copied / relative
            for relative in receipt["artifact_sha256"]
            if relative.endswith((".json", ".npz", ".md"))
        ]
        before = {str(path.relative_to(copied)): _sha(path) for path in evidence_paths}
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT)
        command = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed444_minibatch_denominator",
            "--root",
            str(copied),
        ]
        completed = subprocess.run(
            command,
            cwd=unrelated,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        verification = json.loads(completed.stdout)
        after = {str(path.relative_to(copied)): _sha(path) for path in evidence_paths}
        assert before == after
        assert verification["status"] == "valid"
        report = {
            "schema": "seed444-physical-copy-cli-verification-v1",
            "status": "valid",
            "cwd_was_unrelated": True,
            "copied_root_is_physical_copy": True,
            "command": command,
            "classification": verification["classification"],
            "evidence_hashes_before": before,
            "evidence_hashes_after": after,
            "evidence_immutable": before == after,
        }
        report_path = ROOT / AUDIT / "physical-copy-verification.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        amendment = {
            "schema": "seed444-correction-receipt-amendment-v2",
            "physical_copy_test_source_sha256": _sha(
                ROOT / "ml/alphazero_lite/test_seed444_physical_copy.py"
            ),
            "physical_copy_report_sha256": _sha(report_path),
            "physical_copy_evidence_before_after_identical": True,
            "corrected_verifier_status": verification["status"],
            "amends_publication_receipt_sha256": _sha(ROOT / AUDIT / "receipt.json"),
        }
        (ROOT / AUDIT / "correction-receipt-amendment-2.json").write_text(
            json.dumps(amendment, indent=2, sort_keys=True) + "\n"
        )
