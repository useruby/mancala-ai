#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.forensic_suite import load_suite


REFERENCE_SCHEMA = "azlite_forensic_references_v1"
DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_REFERENCE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
REQUIRED_ROW_FIELDS = {
    "id",
    "canonical_state",
    "state",
    "reference_move",
    "teacher_value",
    "reference_unstable",
    "observed_reference_moves",
    "seed_samples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--reference-artifact", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--out")
    return parser.parse_args()


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_reference_artifact(
    *, suite_path: str | Path, reference_artifact_path: str | Path
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    suite_by_canonical = {position.canonical_key: position for position in suite}
    suite_ids_by_canonical = {position.canonical_key: position.id for position in suite}

    payload = _load_json(reference_artifact_path)
    errors: list[str] = []
    if payload.get("schema") != REFERENCE_SCHEMA:
        _error(errors, f"reference artifact schema must be {REFERENCE_SCHEMA}")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        _error(errors, "reference artifact must contain a rows list")
        rows = []

    artifact_by_canonical: dict[str, dict[str, Any]] = {}
    extra_row_ids: list[str] = []
    illegal_row_ids: list[str] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _error(errors, f"row {index} must be an object")
            continue

        missing_fields = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing_fields:
            _error(
                errors,
                f"row {index} missing required fields: {', '.join(missing_fields)}",
            )
            continue

        canonical_state = str(row["canonical_state"])
        if canonical_state in artifact_by_canonical:
            _error(
                errors,
                f"duplicate canonical_state in reference artifact: {canonical_state}",
            )
            continue

        expected_position = suite_by_canonical.get(canonical_state)
        if expected_position is None:
            extra_row_ids.append(str(row.get("id", f"row-{index}")))
            _error(
                errors,
                f"row {row.get('id', index)} is not present in the forensic suite",
            )
            artifact_by_canonical[canonical_state] = row
            continue

        if row.get("state") != expected_position.state:
            _error(
                errors,
                f"row {row['id']} state does not match checked-in suite state for its canonical key",
            )

        if str(row["id"]) != suite_ids_by_canonical[canonical_state]:
            _error(
                errors,
                f"row {row['id']} does not match suite id {suite_ids_by_canonical[canonical_state]}",
            )

        observed_reference_moves = row.get("observed_reference_moves")
        if (
            not isinstance(observed_reference_moves, list)
            or not observed_reference_moves
        ):
            _error(
                errors,
                f"row {row['id']} must contain non-empty observed_reference_moves",
            )
            observed_reference_moves = []

        seed_samples = row.get("seed_samples")
        if not isinstance(seed_samples, list) or not seed_samples:
            _error(errors, f"row {row['id']} must contain non-empty seed_samples")
            seed_samples = []

        legal_moves = set(expected_position.legal_moves)
        reference_unstable = bool(row.get("reference_unstable"))
        reference_move = row.get("reference_move")

        if reference_unstable:
            if reference_move is not None:
                _error(
                    errors,
                    f"row {row['id']} is unstable and must not set reference_move",
                )
            if len({int(move) for move in observed_reference_moves}) < 2:
                _error(
                    errors,
                    f"row {row['id']} is unstable but does not expose multiple observed reference moves",
                )
        else:
            if reference_move is None:
                _error(errors, f"row {row['id']} is stable and must set reference_move")
            else:
                normalized_reference_move = int(reference_move)
                if normalized_reference_move not in legal_moves:
                    illegal_row_ids.append(str(row["id"]))
                    _error(
                        errors,
                        f"row {row['id']} reference_move {normalized_reference_move} is not legal for suite state",
                    )
                if normalized_reference_move not in {
                    int(move) for move in observed_reference_moves
                }:
                    _error(
                        errors,
                        f"row {row['id']} reference_move is absent from observed_reference_moves",
                    )

        for sample_index, sample in enumerate(seed_samples):
            if not isinstance(sample, dict):
                _error(
                    errors,
                    f"row {row['id']} seed sample {sample_index} must be an object",
                )
                continue
            if "reference_move" not in sample:
                _error(
                    errors,
                    f"row {row['id']} seed sample {sample_index} is missing reference_move",
                )
                continue
            sample_move = int(sample["reference_move"])
            if sample_move not in legal_moves:
                illegal_row_ids.append(str(row["id"]))
                _error(
                    errors,
                    f"row {row['id']} seed sample {sample_index} move {sample_move} is not legal for suite state",
                )

        artifact_by_canonical[canonical_state] = row

    missing_canonicals = [
        canonical
        for canonical in suite_by_canonical
        if canonical not in artifact_by_canonical
    ]
    missing_row_ids = [
        suite_ids_by_canonical[canonical] for canonical in missing_canonicals
    ]
    for row_id in missing_row_ids:
        _error(errors, f"missing suite reference row: {row_id}")

    return {
        "schema": "azlite_forensic_reference_validation_v1",
        "suite_path": str(Path(suite_path)),
        "reference_artifact_path": str(Path(reference_artifact_path)),
        "suite_row_count": len(suite),
        "reference_row_count": len(rows),
        "missing_row_ids": missing_row_ids,
        "extra_row_ids": sorted(set(extra_row_ids)),
        "illegal_reference_row_ids": sorted(set(illegal_row_ids)),
        "error_count": len(errors),
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    args = parse_args()
    summary = validate_reference_artifact(
        suite_path=args.suite,
        reference_artifact_path=args.reference_artifact,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["valid"]:
        raise SystemExit(json.dumps(summary, indent=2))
    print(
        "validated forensic reference artifact: "
        f"{summary['reference_row_count']} rows against {summary['suite_row_count']} suite positions"
    )


if __name__ == "__main__":
    main()
