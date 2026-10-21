"""Generate a deterministic receipt for the published seed511 run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite.verify_seed511_fixed_composition_screen import verify  # noqa: E402

OUT = Path(__file__).resolve().parent
RECEIPT = OUT / "publication-receipt.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def make(root: Path = Path.cwd()) -> dict[str, Any]:
    root = root.resolve()
    result = verify(root)
    write_json(OUT / "verification.json", result)
    files = {
        str(path.relative_to(OUT)): sha(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path != RECEIPT
    }
    receipt = {
        "schema": "seed511-publication-receipt-v1",
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
