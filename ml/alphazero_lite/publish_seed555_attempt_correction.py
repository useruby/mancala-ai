"""Append a receipt for the committed-attempt verifier correction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ml.alphazero_lite import seed555_policy_strength_screen as protocol
from ml.alphazero_lite import verify_seed555_attempt_ledger as verifier


def publish(root: Path, output: Path) -> Path:
    root, output = root.resolve(), output.resolve()
    previous_path = output / "attempt-verification-correction-01.json"
    previous_sha = protocol.sha256(previous_path)
    result = verifier.verify(root, output)
    if result.get("status") != "verified_complete_with_attempt_correction":
        raise ValueError("seed555_attempt_correction_verification_failed")
    attempt = result["attempt_verification"]
    receipt_path = output / "attempt-verification-correction-02.json"
    if receipt_path.exists():
        raise FileExistsError("seed555_attempt_correction_receipt_02_exists")
    receipt = {
        "schema": "seed555_attempt_verification_correction_receipt_v1",
        "status": "verified_complete_with_attempt_correction",
        "supersedes_receipt_sha256": previous_sha,
        "base_read_only_verification_receipt_sha256": protocol.sha256(
            output / "verification-receipt.json"
        ),
        "append_only_source_correction_sha256": protocol.sha256(
            output / "verifier-attempt-correction-01.json"
        ),
        "original_verifier_sha256": protocol.sha256(
            root / "ml/alphazero_lite/verify_seed555_policy_strength_screen.py"
        ),
        "attempt_verifier_sha256": protocol.sha256(
            root / "ml/alphazero_lite/verify_seed555_attempt_ledger.py"
        ),
        "test_source_sha256": protocol.sha256(
            root / "ml/alphazero_lite/test_seed555_attempt_ledger.py"
        ),
        "attempt_verification": {
            "opening_pairs": attempt["opening_pairs"],
            "successful_attempt_records": attempt["successful_attempt_records"],
            "validated_saved_pair_recoveries": attempt[
                "validated_saved_pair_recoveries"
            ],
            "attempts_summary_sha256": attempt["attempts_summary_sha256"],
            "rebound_receipt_attempt_entries_checked": attempt[
                "rebound_receipt_attempt_entries_checked"
            ],
            "failed_attempts": attempt["failed_attempts"],
            "unexplained_missing_attempts": attempt["unexplained_missing_attempts"],
        },
    }
    temporary = receipt_path.with_name(f".{receipt_path.name}.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, receipt_path)
    fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=protocol.ROOT)
    parser.add_argument("--out", type=Path, default=protocol.OUT)
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else args.root / args.out
    path = publish(args.root, output)
    print(json.dumps({"status": "passed", "receipt": str(path)}, indent=2))


if __name__ == "__main__":
    main()
