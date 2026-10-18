"""Receipt-bound recovery of the preserved seed488 opening-zero arena pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import seed488_strength_evaluation as protocol
from ml.alphazero_lite.verify_seed488_opening_replay import (
    ATTEMPT,
    OUT,
    _json,
    _rows,
    _verify as verify_saved_pair,
)
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "opening-0000-recovery-receipt.json"
CONTINUATION_BINDING = "opening-0000-continuation-binding-v3.json"
SAVED_HASHES = {
    "games": "34f4d32b0abc864cace899bd3b69b3ef58624be3078e972dbe2a52f08a9a3d1d",
    "report": "d22e47c1ce979fd8067bf9a7148ca637b314817aea5dbfd94aa843b3e6559901",
    "batch": "4dee384ce68692d75ce8b5e0d1ca67cafefc7f87330aeb9005938f6e55a13c7b",
    "parent_suite": "53841cad6c5d34355cd7d6533a0fa435f4164260fd40a2da760a3d78c784cff8",
}
CORRECTED_SOURCES = (
    "ml/alphazero_lite/seed488_strength_evaluation.py",
    "ml/alphazero_lite/verify_seed488_strength_evaluation.py",
    "ml/alphazero_lite/verify_seed488_opening_replay.py",
    "ml/alphazero_lite/seed488_replay_recovery.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    out = root / OUT
    provenance_path = out / "failure-provenance.json"
    provenance = _json(provenance_path)
    paths = {
        out / "registration.json",
        out / "openings.jsonl",
        out / "receipt.json",
        out / "execution-freeze.json",
        provenance_path,
        out / "resume-corrected/opening-0000.jsonl",
        out / "resume-corrected/games-0000.jsonl",
        out / "resume-corrected/report-0000.json",
    }
    paths.update(out / name for name in provenance["preserved_files"])
    recovery_receipt = out / RECEIPT
    if recovery_receipt.exists():
        paths.add(recovery_receipt)
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def _assert_snapshot(snapshot: dict[Path, tuple[bytes, int]]) -> None:
    for path, identity in snapshot.items():
        if (
            not path.is_file()
            or (path.read_bytes(), path.stat().st_mtime_ns) != identity
        ):
            raise ValueError(f"recovery_evidence_changed:{path.name}")


def correction_receipt(root: Path) -> dict[str, Any]:
    """Build the prospective binding after authenticating saved evidence."""
    root = root.resolve()
    snapshot = _snapshot(root)
    try:
        verified = verify_saved_pair(root, allow_prospective_sources=True)
        observed = {
            name: verified[key]
            for name, key in (
                ("games", "game_log_sha256"),
                ("report", "report_sha256"),
                ("batch", "batch_suite_sha256"),
                ("parent_suite", "parent_suite_sha256"),
            )
        }
        if observed != SAVED_HASHES:
            raise ValueError("saved_pair_not_the_published_opening_zero_pair")
        out = root / OUT
        freeze = _json(out / "execution-freeze.json")
        return {
            "schema": "seed488-opening-0000-correction-receipt-v1",
            "status": "validated_for_recovery_import",
            "opening_index": 0,
            "saved_evidence_sha256": SAVED_HASHES,
            "historical_execution": {
                "registration_sha256": freeze["registration_sha256"],
                "execution_freeze_sha256": _sha(out / "execution-freeze.json"),
                "source_sha256": freeze["source_sha256"],
                "execution_namespace": freeze["execution_namespace"],
            },
            "prospective_correction": {
                "source_sha256": {
                    name: _sha(root / name) for name in CORRECTED_SOURCES
                },
                "rules_source": "ml/alphazero_lite/kalah_rules.py",
                "rules_sha256": _sha(root / "ml/alphazero_lite/kalah_rules.py"),
                "coordinates": "opening prefixes are player-relative; trajectory pit indices are absolute",
                "applies_to": "recovery and prospective continuation only",
            },
            "outcomes": verified["games"],
        }
    finally:
        _assert_snapshot(snapshot)


def publish_receipt(root: Path) -> dict[str, Any]:
    out = root.resolve() / OUT
    receipt = correction_receipt(root)
    path = out / RECEIPT
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("recovery_receipt_conflict")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return receipt


def verify_receipt(root: Path) -> dict[str, Any]:
    out = root.resolve() / OUT
    path = out / RECEIPT
    expected = correction_receipt(root)
    if not path.is_file():
        raise ValueError("recovery_receipt_binding")
    published = _json(path)
    if published == expected:
        return expected
    expected_at_recovery = dict(expected)
    expected_at_recovery["prospective_correction"] = dict(
        expected["prospective_correction"]
    )
    expected_at_recovery["prospective_correction"]["source_sha256"] = published.get(
        "prospective_correction", {}
    ).get("source_sha256")
    if published != expected_at_recovery:
        raise ValueError("recovery_receipt_binding")
    binding_path = out / CONTINUATION_BINDING
    if not binding_path.is_file():
        raise ValueError("continuation_source_binding_missing")
    binding = _json(binding_path)
    if (
        binding.get("schema") != "seed488-prospective-continuation-binding-v1"
        or binding.get("recovery_receipt_sha256") != _sha(path)
        or binding.get("source_sha256")
        != expected["prospective_correction"]["source_sha256"]
        or binding.get("rules_sha256")
        != expected["prospective_correction"]["rules_sha256"]
        or binding.get("historical_execution") != expected["historical_execution"]
    ):
        raise ValueError("continuation_source_binding")
    return expected


def publish_continuation_binding(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    receipt_path = out / RECEIPT
    published = _json(receipt_path)
    expected = correction_receipt(root)
    binding = {
        "schema": "seed488-prospective-continuation-binding-v1",
        "status": "prospective_continuation_source_binding",
        "recovery_receipt_sha256": _sha(receipt_path),
        "source_sha256": expected["prospective_correction"]["source_sha256"],
        "rules_source": expected["prospective_correction"]["rules_source"],
        "rules_sha256": expected["prospective_correction"]["rules_sha256"],
        "historical_execution": expected["historical_execution"],
        "historical_recovery_source_sha256": published["prospective_correction"][
            "source_sha256"
        ],
        "scope": "prospective continuation only; original execution source identities remain historical",
    }
    path = out / CONTINUATION_BINDING
    payload = json.dumps(binding, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError("continuation_binding_conflict")
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return binding


def import_pair(root: Path) -> list[dict[str, Any]]:
    """Import this exact two-seat pair once, preserving its original provenance."""
    root = root.resolve()
    out = root / OUT
    receipt = verify_receipt(root)
    batch_dir = out / ATTEMPT
    allowed_batch_files = {
        "opening-0000.jsonl",
        "games-0000.jsonl",
        "report-0000.json",
    }
    foreign_batches = [
        path.name
        for path in batch_dir.iterdir()
        if path.is_file()
        and path.name.startswith(("opening-", "games-", "report-"))
        and path.name not in allowed_batch_files
    ]
    if foreign_batches:
        raise ValueError("unrelated_uncommitted_batch")
    opening = _json(out / ATTEMPT / "opening-0000.jsonl")
    games_path = out / ATTEMPT / "games-0000.jsonl"
    report_path = out / ATTEMPT / "report-0000.json"
    opening_path = out / ATTEMPT / "opening-0000.jsonl"
    games = _rows(games_path)
    registration = _json(out / "registration.json")
    manifest_path = out / "runtime-artifacts.json"
    manifest_hash = _sha(manifest_path)
    batch_identity = protocol.batch_opening_identity(
        registration, opening, opening_path
    )
    runtime_contract = protocol._batch_runtime_contract(
        registration, 0, manifest_hash, batch_identity
    )
    receipt_hash = _sha(out / RECEIPT)
    rows = []
    for position, game in enumerate(games):
        seat = int(game["challenger_player"])
        _replay_game(opening, game, seat)
        winner = {"challenger": "F", "current": "initializer"}.get(
            game["winner"], game["winner"]
        )
        rows.append(
            {
                **game,
                "opening_index": 0,
                "opening_state_hash": opening["state_hash"],
                "F_player": seat,
                "winner": winner,
                "batch_game_position": position,
                "batch_artifact_dir": ATTEMPT,
                "batch_game_log_sha256": _sha(games_path),
                "batch_report_sha256": _sha(report_path),
                "batch_opening_sha256": _sha(opening_path),
                "batch_runtime_contract": runtime_contract,
                "recovery_receipt_sha256": receipt_hash,
                "historical_execution_provenance": receipt["historical_execution"],
            }
        )
    if len(rows) != 2 or {row["F_player"] for row in rows} != {0, 1}:
        raise ValueError("recovery_pair_coverage")
    ledger = out / "outcomes.jsonl"
    if ledger.exists():
        existing = _rows(ledger)
        if len(existing) == 2 and all(
            row.get("recovery_receipt_sha256") == receipt_hash for row in existing
        ):
            raise ValueError("duplicate_recovery_import")
        raise ValueError("unrelated_or_conflicting_outcomes")
    pending = out / "outcomes.jsonl.pending"
    if pending.exists():
        raise ValueError("unrelated_or_conflicting_pending_batch")
    pending.write_bytes(b"".join(_json_bytes(row) + b"\n" for row in rows))
    pending.replace(ledger)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--publish-receipt", action="store_true")
    group.add_argument("--import-pair", action="store_true")
    group.add_argument("--publish-continuation-binding", action="store_true")
    args = parser.parse_args()
    if args.publish_receipt:
        result = publish_receipt(args.root)
    elif args.import_pair:
        result = import_pair(args.root)
    else:
        result = publish_continuation_binding(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
