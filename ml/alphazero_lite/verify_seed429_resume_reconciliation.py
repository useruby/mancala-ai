"""Repair and verify seed429 corrected-launcher chunk-report resume metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def repair(root: Path) -> dict[str, Any]:
    """Bind existing completed chunk outputs without changing any game outcome."""
    root = root.resolve()
    data = root / "docs/data/seed429-canonical-policy-normalization"
    correction_path = data / "arena-correction-receipt.json"
    registration_path = data / "registration.json"
    runtime_path = data / "runtime-binding.json"
    training_path = data / "training-results.json"
    work = root / ".tmp/seed429-canonical-policy-normalization/arena-chunks"
    repaired = []
    lane_game_ids: dict[str, list[int]] = {}
    index_records = {}
    for lane in ("A", "B"):
        lane_dir = work / lane
        index_path = lane_dir / "correction-resume-index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (
            index.get("identities", {}).get("registration_sha256")
            != _sha(registration_path)
            or index["identities"].get("runtime_binding_sha256") != _sha(runtime_path)
            or index["identities"].get("correction_receipt_sha256")
            != _sha(correction_path)
        ):
            raise ValueError(f"resume_reconciliation_binding_mismatch:{lane}")
        game_ids: list[int] = []
        for start_text, record in sorted(
            index["chunks"].items(), key=lambda item: int(item[0])
        ):
            start = int(start_text)
            chunk_path = lane_dir / f"chunk-{start:04d}.json"
            report_path = lane_dir / f"chunk-{start:04d}-report.json"
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                index.get("identities", {}).get("correction_receipt_sha256")
                != _sha(correction_path)
                or report.get("games_played") != int(record["game_count"])
                or report.get("base_seed") != 429
                or report.get("seed_contract") != "azlite_eval_seed_v2"
            ):
                raise ValueError(
                    f"resume_reconciliation_chunk_report_invalid:{lane}:{start}"
                )
            if _sha(report_path) != record["output_sha256"]:
                raise ValueError(
                    f"resume_reconciliation_report_hash_mismatch:{lane}:{start}"
                )
            entries = chunk["worker_result"]["game_entries"]
            before_digest = _canonical_digest(entries)
            if len(entries) != int(record["game_count"]):
                raise ValueError(
                    f"resume_reconciliation_game_count_mismatch:{lane}:{start}"
                )
            game_ids.extend(int(row["game_index"]) for row in entries)
            if "output_sha256" not in chunk:
                chunk["output_sha256"] = _sha(report_path)
                staging = chunk_path.with_suffix(".json.reconcile")
                staging.write_text(
                    json.dumps(chunk, separators=(",", ":")) + "\n", encoding="utf-8"
                )
                os.replace(staging, chunk_path)
            elif chunk["output_sha256"] != _sha(report_path):
                raise ValueError(
                    f"resume_reconciliation_chunk_report_mismatch:{lane}:{start}"
                )
            if (
                _canonical_digest(chunk["worker_result"]["game_entries"])
                != before_digest
            ):
                raise ValueError("resume_reconciliation_outcomes_changed")
            record["sha256"] = _sha(chunk_path)
            repaired.append(
                {
                    "lane": lane,
                    "start_index": start,
                    "game_count": len(entries),
                    "report_sha256": _sha(report_path),
                    "chunk_sha256": record["sha256"],
                    "game_entries_sha256": before_digest,
                }
            )
        if len(game_ids) != len(set(game_ids)):
            raise ValueError(f"resume_reconciliation_duplicate_game:{lane}")
        lane_game_ids[lane] = sorted(game_ids)
        index_records[lane] = (index_path, index)
    # Persist the chunk hash refresh after each lane's chunks have been checked.
    for index_path, index in index_records.values():
        staging = index_path.with_suffix(".json.reconcile")
        staging.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, index_path)
    verifier_path = root / "ml/alphazero_lite/verify_seed429_resume_reconciliation.py"
    receipt = {
        "schema": "seed429-arena-resume-reconciliation-v1",
        "timing": "after initial corrected chunk execution; before resumed continuation",
        "reason": "The corrected launcher bound chunk report hashes in the resume index but omitted the same output_sha256 field from each sibling chunk payload, which the next invocation validates. This metadata repair adds only that report digest and refreshes corresponding chunk digests; worker_result game entries are hash-compared before/after and are unchanged.",
        "registration_sha256": _sha(registration_path),
        "runtime_binding_sha256": _sha(runtime_path),
        "correction_receipt_sha256": _sha(correction_path),
        "training_results_sha256": _sha(training_path),
        "correction_verifier_sha256": _sha(verifier_path),
        "completed_game_count_before_repair": sum(
            len(items) for items in lane_game_ids.values()
        ),
        "lane_game_counts": {lane: len(ids) for lane, ids in lane_game_ids.items()},
        "outcomes_unchanged": True,
        "protocol_changed": False,
        "chunks": repaired,
    }
    receipt_path = data / "arena-resume-reconciliation.json"
    if receipt_path.exists():
        raise ValueError("resume_reconciliation_receipt_is_append_only")
    staging_receipt = receipt_path.with_suffix(".json.staging")
    staging_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(staging_receipt, receipt_path)
    return receipt


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    data = root / "docs/data/seed429-canonical-policy-normalization"
    receipt_path = data / "arena-resume-reconciliation.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_changed"] or not receipt["outcomes_unchanged"]:
        raise ValueError("resume_reconciliation_protocol_or_outcome_change")
    if receipt["registration_sha256"] != _sha(data / "registration.json"):
        raise ValueError("resume_reconciliation_registration_mismatch")
    if receipt["runtime_binding_sha256"] != _sha(data / "runtime-binding.json"):
        raise ValueError("resume_reconciliation_runtime_binding_mismatch")
    if receipt["correction_receipt_sha256"] != _sha(
        data / "arena-correction-receipt.json"
    ):
        raise ValueError("resume_reconciliation_correction_receipt_mismatch")
    game_counts = {"A": 0, "B": 0}
    for chunk in receipt["chunks"]:
        directory = (
            root
            / ".tmp/seed429-canonical-policy-normalization/arena-chunks"
            / chunk["lane"]
        )
        chunk_path = directory / f"chunk-{int(chunk['start_index']):04d}.json"
        report_path = directory / f"chunk-{int(chunk['start_index']):04d}-report.json"
        data_chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        if (
            _sha(chunk_path) != chunk["chunk_sha256"]
            or _sha(report_path) != chunk["report_sha256"]
            or data_chunk.get("output_sha256") != chunk["report_sha256"]
            or _canonical_digest(data_chunk["worker_result"]["game_entries"])
            != chunk["game_entries_sha256"]
        ):
            raise ValueError("resume_reconciliation_chunk_tampered")
        game_counts[chunk["lane"]] += int(chunk["game_count"])
    if game_counts != receipt["lane_game_counts"]:
        raise ValueError("resume_reconciliation_game_count_mismatch")
    return {
        "valid": True,
        "completed_games_before_repair": receipt["completed_game_count_before_repair"],
        "lane_counts": game_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("repair", "verify"))
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    payload = repair(args.root) if args.action == "repair" else verify(args.root)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
