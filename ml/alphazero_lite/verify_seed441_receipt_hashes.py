"""Read-only verification of every file hash recorded in the seed441 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    base = root / "docs/data/seed441-seed440-publication-completion"
    receipt = json.loads((base / "receipt.json").read_text(encoding="utf-8"))
    checked: list[str] = []

    def check(relative: str, expected: str, *, path_root: Path = root) -> None:
        path = path_root / relative
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"seed441_receipt_hash_mismatch:{relative}")
        checked.append(
            str(path.relative_to(root)) if path.is_relative_to(root) else relative
        )

    old = root / "docs/data/seed440-recorded-update-attribution"
    for field, name in (
        ("protocol_sha256", "protocol.json"),
        ("evidence_bindings_sha256", "evidence-bindings.json"),
        ("step_ledger_sha256", "step-ledger.json"),
        ("path_hash_archive_sha256", "reconstructed-path-hashes.npz"),
    ):
        check(
            f"docs/data/seed440-recorded-update-attribution/{name}",
            receipt["historical_seed440"][field],
        )

    for relative, expected in receipt["supplemental_files"].items():
        check(f"docs/data/seed441-seed440-publication-completion/{relative}", expected)
    for relative, expected in receipt["source_sha256"].items():
        check(relative, expected)

    bindings = json.loads((old / "evidence-bindings.json").read_text(encoding="utf-8"))
    for section in ("seed435",):
        for item in bindings[section].values():
            check(item["path"], item["sha256"])
    for section in ("seed439_U_gradient_archive", "authoritative_membership"):
        item = bindings[section]
        check(item["path"], item["sha256"])

    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["dependency_sha256"].items():
        check(relative, expected)
    for relative, expected in protocol["source_sha256"].items():
        check(relative, expected)
    return {"status": "valid", "checked_files": len(checked), "binary_archives": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    print(json.dumps(verify(parser.parse_args().root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
