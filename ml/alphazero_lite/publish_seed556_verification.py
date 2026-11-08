"""Publish read-only seed556 verification after the concurrent-order correction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ml.alphazero_lite import verify_seed556_completion_order_correction as verifier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(output: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(output).as_posix(): {
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, value: Any) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)


def publish(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    before = _snapshot(output)
    result = verifier.verify(root, output)
    after = _snapshot(output)
    if before != after:
        raise ValueError("seed556_read_only_verifier_changed_evidence")
    record = {
        **result,
        "read_only_bytes_and_mtimes_unchanged": True,
        "evidence_files_checked": len(before),
        "evidence_snapshot_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "correction_verifier_sha256": _sha(
            root / "ml/alphazero_lite/verify_seed556_completion_order_correction.py"
        ),
    }
    _write_json(output / "verification.json", record)
    receipt = {
        "schema": "seed556_read_only_verification_receipt_v1",
        "status": record["status"],
        "verified_pairs": record["validated_pairs"],
        "verified_games": record["validated_games"],
        "verification_sha256": _sha(output / "verification.json"),
        "evidence_sha256": {name: item["sha256"] for name, item in before.items()},
        "bytes_and_mtimes_unchanged": record["read_only_bytes_and_mtimes_unchanged"],
    }
    _write_json(output / "verification-receipt.json", receipt)
    _write_json(
        output / "execution-progress.json",
        {
            "status": "complete_verified",
            "committed_opening_pairs": record["validated_pairs"],
            "committed_games": record["validated_games"],
            "verification_status": record["status"],
        },
    )
    return record
