"""Write the append-only v2 supplement receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parent
    repo = out.parents[2]
    paths = (
        out / "verification-supplement-002.md",
        out / "verify_supplement_v2.py",
        out / "make_supplement_receipt_v2.py",
        repo / "ml/alphazero_lite/test_seed508_verification_supplement_v2.py",
    )
    receipt = {
        "schema": "seed508-verification-supplement-receipt-v2",
        "status": "verified",
        "files": {
            path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    (out / "verification-supplement-receipt-002.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": receipt["status"], "files": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
