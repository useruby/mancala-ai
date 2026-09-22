"""Generic paired-generation comparison records and deterministic bootstrap CIs."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from ml.alphazero_lite.generation_record import load_record as load_generation_record

SCHEMA = "azlite_generation_comparison_v1"


class ComparisonRecordError(ValueError):
    """Raised when a generic comparison record is malformed."""


def paired_bootstrap(
    deltas: list[float], *, seed: int, samples: int = 10_000
) -> dict[str, Any]:
    if not deltas or samples <= 0:
        raise ComparisonRecordError(
            "paired bootstrap requires values and positive samples"
        )
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(deltas) for _ in deltas) for _ in range(samples)
    )
    return {
        "mean": statistics.fmean(deltas),
        "bootstrap_95": {
            "seed": seed,
            "samples": samples,
            "lower": draws[int(samples * 0.025) - 1],
            "upper": draws[int(samples * 0.975) - 1],
        },
    }


def _without(record: dict[str, Any], paths: set[str]) -> dict[str, Any]:
    result = json.loads(json.dumps(record))
    for path in paths:
        current = result
        *parents, leaf = path.split(".")
        for parent in parents:
            current = current.get(parent, {})
        current.pop(leaf, None)
    return result


def _value_at(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for component in path.split("."):
        if isinstance(current, dict):
            current = current[component]
        elif isinstance(current, list):
            current = next(item for item in current if item.get("name") == component)
        else:
            raise ComparisonRecordError(f"invalid controlled difference path: {path}")
    return current


def validate_matched_records(
    left: dict[str, Any], right: dict[str, Any], *, allowed_differences: set[str]
) -> None:
    """Require matched records to differ only in declared treatment/output fields."""
    if left.get("comparison_controls") != right.get("comparison_controls"):
        raise ComparisonRecordError("matched_generation_controls_differ")
    ignored = {
        "generation_id",
        "status",
        # ``comparison_controls`` carries the causal training contract. Metrics and
        # runtime fields in training are outcomes, not matched controls.
        "training",
        "candidate",
        "diagnostics",
        "artifacts",
        "compute",
        "provenance_notes",
        *allowed_differences,
    }
    if _without(left, ignored) != _without(right, ignored):
        raise ComparisonRecordError("matched_generation_controls_differ")


def validate(record: dict[str, Any], *, base_dir: Path) -> None:
    if record.get("schema") != SCHEMA or not record.get("comparison_id"):
        raise ComparisonRecordError("unsupported comparison record")
    pairs = record.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ComparisonRecordError("comparison requires pairs")
    seen: set[str] = set()
    allowed = set(record.get("allowed_record_differences", []))
    controlled_difference = record.get("controlled_difference", {})
    if not isinstance(controlled_difference, dict) or not controlled_difference:
        raise ComparisonRecordError("comparison requires controlled difference")
    for pair in pairs:
        pair_id = str(pair.get("pair_id"))
        if not pair_id or pair_id in seen:
            raise ComparisonRecordError("comparison pair ids must be unique")
        seen.add(pair_id)
        left = load_generation_record(base_dir / pair["baseline_record"])
        right = load_generation_record(base_dir / pair["treatment_record"])
        validate_matched_records(left, right, allowed_differences=allowed)
        for path, expected in controlled_difference.items():
            if _value_at(left, path) != expected.get("baseline") or _value_at(
                right, path
            ) != expected.get("treatment"):
                raise ComparisonRecordError("controlled_difference_not_observed")


def load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    validate(record, base_dir=path.parent)
    return record
