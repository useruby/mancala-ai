#!/usr/bin/env python3
"""Generic immutable JSONL dataset transformations for AlphaZero-lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite.export_artifact import sha256_file
from ml.alphazero_lite.self_play import outcome_for_player


class DatasetTransformError(ValueError):
    """Raised when a transformation changes data outside its contract."""


def relabel_value_row(row: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Return a row with its value target derived from the recorded outcome."""
    if mode != "default":
        raise DatasetTransformError(f"unsupported value target relabel mode: {mode}")
    result = dict(row)
    result["value"] = outcome_for_player(row["winner"], row["player"])
    result["value_target_mode"] = mode
    return result


def verify_value_relabel(
    original: dict[str, Any], transformed: dict[str, Any], *, mode: str
) -> None:
    """Ensure a value relabel leaves state, policy, order, and metadata intact."""
    if set(original) != set(transformed) or any(
        original[key] != transformed[key]
        for key in original
        if key not in {"value", "value_target_mode"}
    ):
        raise DatasetTransformError("value_target_relabel_integrity_failed")
    if transformed["value"] != outcome_for_player(
        original["winner"], original["player"]
    ):
        raise DatasetTransformError("value_target_relabel_integrity_failed")
    if transformed["value_target_mode"] != mode:
        raise DatasetTransformError("value_target_relabel_integrity_failed")


def relabel_value_targets(
    source: Path, destination: Path, *, mode: str
) -> dict[str, Any]:
    """Relabel a JSONL artifact and return auditable, deterministic provenance."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = changed = 0
    old_magnitudes: list[float] = []
    new_magnitudes: list[float] = []
    with (
        source.open(encoding="utf-8") as input_file,
        destination.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            if not line.strip():
                continue
            original = json.loads(line)
            transformed = relabel_value_row(original, mode=mode)
            verify_value_relabel(original, transformed, mode=mode)
            output_file.write(
                json.dumps(transformed, sort_keys=True, ensure_ascii=True) + "\n"
            )
            row_count += 1
            changed += original["value"] != transformed["value"]
            old_magnitudes.append(abs(float(original["value"])))
            new_magnitudes.append(abs(float(transformed["value"])))
    return {
        "operation": "relabel-value-targets",
        "mode_transition": {"from": "source_declared", "to": mode},
        "source_sha256": sha256_file(source),
        "transformed_sha256": sha256_file(destination),
        "row_count": row_count,
        "rows_changed": changed,
        "rows_unchanged": row_count - changed,
        "state_policy_equality": "passed",
        "mean_absolute_old_target": sum(old_magnitudes) / row_count
        if row_count
        else None,
        "mean_absolute_new_target": sum(new_magnitudes) / row_count
        if row_count
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    relabel = commands.add_parser("relabel-value-targets")
    relabel.add_argument("--source", type=Path, required=True)
    relabel.add_argument("--out", type=Path, required=True)
    relabel.add_argument("--mode", required=True)
    relabel.add_argument("--audit-out", type=Path)
    args = parser.parse_args()
    audit = relabel_value_targets(args.source, args.out, mode=args.mode)
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.audit_out:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
