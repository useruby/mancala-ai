"""Generate the append-only seed508 verification supplement receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parent
    repo = out.parents[2]
    paths = (
        out / "verification-supplement-001.md",
        out / "verify_supplement.py",
        out / "publication-receipt-regenerated-001.json",
        out / "publication-receipt.json",
        out / "make_supplement_receipt.py",
        repo / "ml/alphazero_lite/test_seed508_verification_supplement.py",
    )
    manifest = {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    receipt = {
        "schema": "seed508-verification-supplement-receipt-v1",
        "status": "verified",
        "files": manifest,
    }
    (out / "verification-supplement-receipt-001.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": receipt["status"], "files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
