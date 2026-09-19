#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import (
    ArenaReportValidationError,
    validate_arena_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--min-score", type=float)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="TARGET",
        help="Destination directory (repeatable). Defaults to model-artifact/current",
    )
    parser.add_argument("--gate-report")
    parser.add_argument(
        "--gate-arena-evidence",
        help="Explicit immutable migration record for a gate written before arena_evidence existed.",
    )
    parser.add_argument("--require-lossless", action="store_true")
    parser.add_argument("--max-losses", type=non_negative_int, default=0)
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_identity(checkpoint_path: Path) -> dict[str, str]:
    return {
        "weights_json_sha256": sha256_file(checkpoint_path / "weights.json"),
        "metadata_json_sha256": sha256_file(checkpoint_path / "metadata.json"),
    }


def validate_gate_identity(gate_report: dict, checkpoint_path: Path) -> None:
    gate_identity = gate_report.get("candidate_identity")
    if not isinstance(gate_identity, dict):
        raise SystemExit("gate_candidate_identity_mismatch: missing candidate identity")

    expected_identity = candidate_identity(checkpoint_path)
    if any(
        gate_identity.get(name) != value for name, value in expected_identity.items()
    ):
        raise SystemExit("gate_candidate_identity_mismatch")


def load_json(path: Path, *, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Malformed {description} JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"Malformed {description}: {path}")
    return payload


def validate_gate_policy(gate_report: dict, report: dict) -> float:
    require_lossless = gate_report.get("require_lossless")
    max_losses = gate_report.get("max_losses")
    min_score = gate_report.get("min_arena_score")
    min_games = gate_report.get("min_arena_games")
    if not isinstance(require_lossless, bool):
        raise SystemExit("gate_policy_invalid: require_lossless must be boolean")
    if (
        isinstance(max_losses, bool)
        or not isinstance(max_losses, int)
        or max_losses < 0
    ):
        raise SystemExit(
            "gate_policy_invalid: max_losses must be a non-negative integer"
        )
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise SystemExit("gate_policy_invalid: min_arena_score must be numeric")
    if isinstance(min_games, bool) or not isinstance(min_games, int) or min_games <= 0:
        raise SystemExit("gate_policy_invalid: min_arena_games must be positive")

    try:
        result = validate_arena_report(report=report, min_score=float(min_score))
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error
    if not result["passed"] or result["games_played"] < min_games:
        raise SystemExit(
            "gate_policy_inconsistent: arena evidence does not meet gate threshold"
        )
    if require_lossless and result["losses"] > max_losses:
        raise SystemExit(
            "gate_policy_inconsistent: arena losses exceed gate max_losses"
        )
    if (
        "arena_losses" in gate_report
        and gate_report["arena_losses"] != result["losses"]
    ):
        raise SystemExit("gate_policy_inconsistent: arena losses differ from gate")
    return float(min_score)


def arena_evidence_for_gate(
    gate_report: dict, gate_report_path: Path, migration_path: str | None
) -> tuple[Path, str]:
    evidence = gate_report.get("arena_evidence")
    if evidence is None:
        if migration_path is None:
            raise SystemExit("gate_arena_evidence_missing")
        migration_file = resolve_repo_path(migration_path)
        if not migration_file.is_file():
            raise SystemExit(f"Missing gate arena evidence: {migration_file}")
        migration = load_json(migration_file, description="gate arena evidence")
        evidence = migration.get("arena_evidence")
        if (
            migration.get("schema") != "azlite_gate_arena_evidence_migration_v1"
            or migration.get("gate_report_sha256") != sha256_file(gate_report_path)
            or migration.get("gate_report_path") != gate_report.get("report_path")
        ):
            raise SystemExit("gate_arena_evidence_migration_mismatch")
    if not isinstance(evidence, dict):
        raise SystemExit("gate_arena_evidence_missing")
    path = evidence.get("path")
    expected_hash = evidence.get("sha256")
    if not isinstance(path, str) or not isinstance(expected_hash, str):
        raise SystemExit("gate_arena_evidence_invalid")
    if path != gate_report.get("arena_report_path"):
        raise SystemExit("gate_arena_evidence_path_mismatch")
    arena_path = resolve_repo_path(path)
    if not arena_path.is_file():
        raise SystemExit(f"Missing gate arena report: {arena_path}")
    if sha256_file(arena_path) != expected_hash:
        raise SystemExit("gate_arena_evidence_hash_mismatch")
    return arena_path, expected_hash


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    required_files = ["metadata.json", "weights.json"]
    missing_files = [
        str(checkpoint_path / filename)
        for filename in required_files
        if not (checkpoint_path / filename).is_file()
    ]
    if missing_files:
        raise SystemExit(f"Missing required file: {missing_files[0]}")

    if args.gate_report:
        if args.min_score is not None or args.require_lossless or args.max_losses != 0:
            raise SystemExit("gate_policy_override_forbidden")
        gate_report_path = resolve_repo_path(args.gate_report)
        if not gate_report_path.is_file():
            raise SystemExit(f"Missing gate report: {gate_report_path}")
        gate_report = load_json(gate_report_path, description="gate report")
        if not gate_report.get("passed", False):
            raise SystemExit(f"Gate report did not pass: {gate_report_path}")
        validate_gate_identity(gate_report, checkpoint_path)
        report_path, _ = arena_evidence_for_gate(
            gate_report, gate_report_path, args.gate_arena_evidence
        )
        report = load_json(report_path, description="arena report")
        min_score = validate_gate_policy(gate_report, report)
    else:
        report_path = checkpoint_path / "arena_report.json"
        if not report_path.is_file():
            raise SystemExit(f"Missing required file: {report_path}")
        report = load_json(report_path, description="arena report")
        try:
            result = validate_arena_report(
                report=report,
                min_score=args.min_score if args.min_score is not None else 0.55,
            )
        except ArenaReportValidationError as error:
            raise SystemExit(str(error)) from error
        if not result["passed"]:
            raise SystemExit(
                f"Checkpoint did not meet threshold score={result['score']} min_score={result['min_score']}"
            )
        if args.require_lossless and result["losses"] > args.max_losses:
            raise SystemExit(
                f"lossless requirement failed: losses={result['losses']} max_losses={args.max_losses}"
            )
        min_score = result["min_score"]

    targets_raw = args.target if args.target else ["model-artifact/current"]
    targets = [resolve_repo_path(raw) for raw in targets_raw]

    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        for entry in target.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        for filename in required_files:
            shutil.copy2(checkpoint_path / filename, target / filename)
        shutil.copy2(report_path, target / "arena_report.json")

        print(f"Promoted checkpoint from {checkpoint_path} to {target}")
    print(f"MinScore={min_score}")


if __name__ == "__main__":
    main()
