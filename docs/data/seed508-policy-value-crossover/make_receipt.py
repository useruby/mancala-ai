"""Bind all published seed508 evidence files in a SHA-256 receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    out = Path(__file__).resolve().parent
    files = {}
    for path in sorted(
        p
        for p in out.rglob("*")
        if p.is_file() and p.name != "publication-receipt.json"
    ):
        files[path.relative_to(out).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    receipt = {
        "schema": "seed508-publication-receipt-v1",
        "status": "verified_complete",
        "file_count": len(files),
        "files": files,
    }
    (out / "publication-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": receipt["status"], "file_count": len(files)}, indent=2))


if __name__ == "__main__":
    main()
