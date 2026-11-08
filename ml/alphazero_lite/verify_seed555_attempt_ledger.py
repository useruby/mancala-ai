"""Append-only semantic correction for committed seed555 attempt records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed555_policy_strength_screen as protocol
from ml.alphazero_lite import verify_seed555_policy_strength_screen as original


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def verify_attempts(output: Path) -> dict[str, Any]:
    """Validate every committed pair's attempt or explicit recovery evidence."""
    output = output.resolve()
    freeze = json.loads((output / "execution-freeze.json").read_text())
    commands = jsonl(output / "command-preflight.jsonl")
    if len(commands) != protocol.OPENINGS:
        raise ValueError("seed555_attempt_correction_command_count_mismatch")
    attempts_summary = []
    recovery_count = 0
    attempt_count = 0
    for index in range(protocol.OPENINGS):
        batch = output / "batches"
        completion_path = batch / f"completion-{index:04d}.json"
        if not completion_path.is_file():
            raise ValueError(f"seed555_attempt_correction_pair_uncommitted:{index}")
        completion = json.loads(completion_path.read_text())
        command_path = batch / f"command-{index:04d}.json"
        if (
            not command_path.is_file()
            or json.loads(command_path.read_text()) != commands[index]
            or commands[index].get("opening_index") != index
            or completion.get("opening_index") != index
        ):
            raise ValueError(
                f"seed555_attempt_correction_command_binding_mismatch:{index}"
            )
        command_sha = protocol.sha256(command_path)
        if completion.get("command_sha256") != command_sha:
            raise ValueError(
                f"seed555_attempt_correction_completion_command_mismatch:{index}"
            )
        attempt_paths = sorted(batch.glob(f"attempt-{index:04d}-*.json"))
        recovery_path = batch / f"recovery-{index:04d}.json"
        if attempt_paths:
            if (
                len(attempt_paths) != 1
                or attempt_paths[0].name != f"attempt-{index:04d}-00.json"
            ):
                raise ValueError(
                    f"seed555_attempt_correction_attempt_count_mismatch:{index}"
                )
            attempt_path = attempt_paths[0]
            try:
                attempt = json.loads(attempt_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"seed555_attempt_correction_attempt_invalid:{index}"
                ) from error
            if (
                attempt.get("opening_index") != index
                or attempt.get("returncode") != 0
                or attempt.get("command_sha256") != command_sha
            ):
                raise ValueError(
                    f"seed555_attempt_correction_failed_or_rebound_attempt:{index}"
                )
            attempt_count += 1
            attempt_binding = {
                "kind": "successful_arena_attempt",
                "attempt_sha256": protocol.sha256(attempt_path),
                "command_sha256": command_sha,
            }
        else:
            attempt_binding = {
                "kind": "validated_saved_pair_recovery",
                "command_sha256": command_sha,
            }
        if recovery_path.is_file():
            recovery = json.loads(recovery_path.read_text())
            expected_recovery = {
                "opening_index": index,
                "status": "complete_raw_pair_recovered_after_validation",
                "validated_game_records": 2,
                "command_sha256": command_sha,
                "games_sha256": protocol.sha256(batch / f"games-{index:04d}.jsonl"),
                "report_sha256": protocol.sha256(batch / f"report-{index:04d}.json"),
                "recovery_reran_games": False,
            }
            if recovery != expected_recovery:
                raise ValueError(
                    f"seed555_attempt_correction_recovery_mismatch:{index}"
                )
            recovery_count += 1
            attempt_binding["recovery_sha256"] = protocol.sha256(recovery_path)
            attempt_binding["recovery_reran_games"] = False
            if attempt_paths:
                attempt_binding["kind"] = "successful_attempt_with_validated_recovery"
            else:
                attempt_binding["kind"] = "validated_saved_pair_recovery"
        elif not attempt_paths:
            raise ValueError(
                f"seed555_attempt_correction_unexplained_missing_attempt:{index}"
            )
        attempts_summary.append({"opening_index": index, **attempt_binding})

    original_verifier = "ml/alphazero_lite/verify_seed555_policy_strength_screen.py"
    return {
        "status": "committed_attempts_semantically_verified",
        "opening_pairs": protocol.OPENINGS,
        "successful_attempt_records": attempt_count,
        "validated_saved_pair_recoveries": recovery_count,
        "attempts_summary_sha256": hashlib.sha256(
            b"".join(protocol.stable_json(row) + b"\n" for row in attempts_summary)
        ).hexdigest(),
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "execution_freeze_sha256": protocol.sha256(output / "execution-freeze.json"),
        "frozen_original_verifier_sha256": freeze["source_sha256"][original_verifier],
        "current_original_verifier_sha256": protocol.sha256(
            protocol.ROOT / original_verifier
        ),
        "correction_verifier_source_sha256": protocol.sha256(Path(__file__)),
        "attempt_records_checked": attempt_count,
        "failed_attempts": 0,
        "unexplained_missing_attempts": 0,
        "receipt_rebound_semantic_check": "attempt returncode, opening index, and frozen command hash independently validated",
        "attempts_summary": attempts_summary,
    }


def verify_published_receipt(output: Path) -> dict[str, Any]:
    """Check the separate publication receipt after it has been created."""
    output = output.resolve()
    receipt_path = output / "verification-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "independent_read_only_verified_complete"
        or receipt.get("verification_result", {}).get("status") != "verified_complete"
        or receipt.get("verification_result", {}).get("validated_games")
        != protocol.GAMES
        or receipt.get("verification_result", {}).get("validated_pairs")
        != protocol.OPENINGS
        or receipt.get("verification_sha256")
        != protocol.sha256(output / "verification.json")
    ):
        raise ValueError("seed555_attempt_correction_base_receipt_invalid")
    for relative, expected_sha in receipt.get("evidence_sha256", {}).items():
        path = output / relative
        if not path.is_file() or protocol.sha256(path) != expected_sha:
            raise ValueError(
                f"seed555_attempt_correction_evidence_receipt_mismatch:{relative}"
            )
    return {
        "verification_receipt_sha256": protocol.sha256(receipt_path),
        "verification_sha256": protocol.sha256(output / "verification.json"),
        "evidence_entries_checked": len(receipt.get("evidence_sha256", {})),
    }


def verify(root: Path, output: Path) -> dict[str, Any]:
    """Run the base trajectory verifier then the attempt-ledger correction."""
    base = original.verify(root, output)
    if base.get("status") != "verified_complete":
        raise ValueError("seed555_attempt_correction_requires_complete_screen")
    attempts = verify_attempts(output)
    published_receipt = verify_published_receipt(output)
    return {
        "status": "verified_complete_with_attempt_correction",
        "base_verification": {
            "status": base["status"],
            "validated_games": base["validated_games"],
            "validated_pairs": base["validated_pairs"],
            "analysis": base["analysis"],
        },
        "attempt_verification": attempts,
        "published_receipt_verification": published_receipt,
    }


def write_correction_receipt(output: Path, result: dict[str, Any]) -> Path:
    """Append a separate correction receipt; never edit base evidence."""
    output = output.resolve()
    destination = output / "attempt-verification-correction-01.json"
    if destination.exists():
        raise FileExistsError("seed555_attempt_correction_receipt_already_exists")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    pending = destination.with_name(f".{destination.name}.pending")
    pending.write_text(payload)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, destination)
    fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return destination


def write_source_correction_record(root: Path, output: Path) -> Path:
    """Append a narrowly scoped source-hash supersession for this verifier fix."""
    root, output = root.resolve(), output.resolve()
    destination = output / "verifier-attempt-correction-01.json"
    if destination.exists():
        raise FileExistsError("seed555_source_correction_record_already_exists")
    freeze = json.loads((output / "execution-freeze.json").read_text())
    original_verifier = "ml/alphazero_lite/verify_seed555_policy_strength_screen.py"
    correction_source = "ml/alphazero_lite/verify_seed555_attempt_ledger.py"
    previous_commit = "ae349b73b87032419ed30b02b84cac63813764cd"
    previous_bytes = subprocess.run(
        ["git", "show", f"{previous_commit}:{original_verifier}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    previous_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"{previous_commit}:{original_verifier}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    previous_sha = freeze["source_sha256"][original_verifier]
    if hashlib.sha256(previous_bytes).hexdigest() != previous_sha:
        raise ValueError("seed555_previous_verifier_snapshot_mismatch")
    attempts = verify_attempts(output)
    published_receipt = verify_published_receipt(output)
    source_sha = protocol.sha256(root / correction_source)
    record = {
        "schema": "seed555_verifier_attempt_correction_v1",
        "status": "append_only_verifier_correction_verified",
        "correction": "validate committed attempts against opening index, frozen command hash, and zero exit status; missing attempts require a valid saved-pair recovery receipt",
        "previous_verifier_git_commit": previous_commit,
        "previous_verifier_git_blob": previous_blob,
        "previous_verifier_sha256": previous_sha,
        "corrected_verifier_sha256": protocol.sha256(root / original_verifier),
        "correction_source_sha256": source_sha,
        "correction_verifier_source_sha256": source_sha,
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "execution_freeze_sha256": protocol.sha256(output / "execution-freeze.json"),
        "verification_receipt_sha256": protocol.sha256(
            output / "verification-receipt.json"
        ),
        "published_receipt_entries_checked": published_receipt[
            "evidence_entries_checked"
        ],
        "published_outcomes_sha256": protocol.sha256(output / "outcomes.jsonl"),
        "published_analysis_sha256": protocol.sha256(output / "analysis.json"),
        "published_verification_sha256": protocol.sha256(output / "verification.json"),
        "attempts_summary_sha256": attempts["attempts_summary_sha256"],
        "committed_pairs_checked": attempts["opening_pairs"],
        "attempt_records_checked": attempts["attempt_records_checked"],
        "failed_attempts": attempts["failed_attempts"],
        "unexplained_missing_attempts": attempts["unexplained_missing_attempts"],
        "saved_pair_recoveries": attempts["validated_saved_pair_recoveries"],
    }
    _write_json(destination, record)
    return destination


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=protocol.ROOT)
    parser.add_argument("--out", type=Path, default=protocol.OUT)
    parser.add_argument("--record-correction-receipt", action="store_true")
    parser.add_argument("--record-source-correction", action="store_true")
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else args.root / args.out
    if args.record_source_correction:
        result = {
            "status": "source_correction_recorded",
            "path": str(write_source_correction_record(args.root, output)),
        }
    else:
        result = verify(args.root, output)
    if args.record_correction_receipt:
        result["correction_receipt_path"] = str(
            write_correction_receipt(output, result)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
