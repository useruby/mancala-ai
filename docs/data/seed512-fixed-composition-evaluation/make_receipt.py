"""Write verification and an integrity receipt for the seed512 publication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite.verify_seed512_evaluation import verify  # noqa: E402

OUT = Path(__file__).resolve().parent
RECEIPT = OUT / "publication-receipt.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def make(root: Path = Path.cwd()) -> dict[str, Any]:
    root = root.resolve()
    verification = verify(root)
    write_json(OUT / "verification.json", verification)
    files = {
        str(path.relative_to(OUT)): sha(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path != RECEIPT
    }
    receipt = {
        "schema": "seed512-publication-receipt-v1",
        "status": "verified_complete",
        "file_count": len(files),
        "files": files,
        "registration_sha256": sha(OUT / "registration.json"),
        "execution_freeze_sha256": sha(OUT / "execution-freeze.json"),
        "suite_sha256": sha(OUT / "openings.jsonl"),
        "outcomes_sha256": sha(OUT / "outcomes.jsonl"),
        "results_sha256": sha(OUT / "results.json"),
        "verification_sha256": sha(OUT / "verification.json"),
    }
    write_json(RECEIPT, receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(make(), indent=2, sort_keys=True))
