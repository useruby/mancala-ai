"""Run seed452's actual verifier from a relocated evidence/source copy."""

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
OUT = Path("docs/data/seed452-fresh-policy-step-attribution")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed452_verifier_on_physical_copy_from_unrelated_cwd() -> None:
    protocol = json.loads((ROOT / OUT / "protocol.json").read_text())
    receipt = json.loads((ROOT / OUT / "receipt.json").read_text())
    needed = set(protocol["execution_sha256"]) | set(protocol["input_sha256"])
    needed.add("docs/data/seed447-joint-output-cap/ordered-targets.npz")
    needed.update(f"{OUT}/{name}" for name in receipt["files_sha256"])
    needed.add(f"{OUT}/receipt.json")
    with tempfile.TemporaryDirectory(prefix="seed452-copy-", dir=ROOT / ".tmp") as name:
        scratch = Path(name)
        copied, unrelated = scratch / "relocated", scratch / "unrelated-cwd"
        copied.mkdir()
        unrelated.mkdir()
        shutil.copytree(
            ROOT / "ml/alphazero_lite", copied / "ml/alphazero_lite", dirs_exist_ok=True
        )
        for relative in sorted(needed):
            source, destination = ROOT / relative, copied / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        evidence = [copied / f"{OUT}/{name}" for name in receipt["files_sha256"]]
        before = {str(path.relative_to(copied)): _sha(path) for path in evidence}
        env = os.environ.copy()
        env["PYTHONPATH"] = str(copied)
        command = [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed452_fresh_policy_step_attribution",
            "--root",
            str(copied),
        ]
        completed = subprocess.run(
            command,
            cwd=unrelated,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stderr
        after = {str(path.relative_to(copied)): _sha(path) for path in evidence}
        assert before == after
        report = json.loads(completed.stdout)
        assert report["status"] == "valid"
        assert report["rows"] == 64
        assert before == after
