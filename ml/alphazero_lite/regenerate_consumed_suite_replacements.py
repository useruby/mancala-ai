#!/usr/bin/env python3
"""Create an explicitly non-original replacement consumed-suite registry.

This recovery is only for exploratory work when the historical suite files and
their replay exclusions have been lost.  It must never be passed to the sealed
consumed-suite registry or presented as historical-suite equivalence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import consumed_suite_registry as historical
from ml.alphazero_lite.run_pr249_fresh_suite_generalization import all_openings


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def regenerate(*, canonical: Path, out_dir: Path, manifest: Path) -> dict[str, Any]:
    """Regenerate disjoint substitutes without claiming historical identity."""
    if not canonical.is_file():
        raise ValueError("replacement_canonical_suite_missing")
    canonical_sha = suites.suite_sha256(str(canonical))
    if canonical_sha != historical._SPECS[0].sha256:
        raise ValueError("replacement_canonical_suite_hash_mismatch")
    universe = all_openings()
    used = {
        suites.canonical_key(row["state"])
        for row in suites.load_suite_jsonl(str(canonical))
    }
    entries: dict[str, Any] = {
        "canonical": {
            "path": str(canonical),
            "sha256": canonical_sha,
            "status": "historical_original",
        }
    }
    for spec in historical._SPECS[1:]:
        selected = suites.select_diverse(
            [row for row in universe if suites.canonical_key(row["state"]) not in used],
            128,
            spec.seed,
        )
        path = out_dir / f"suite_{spec.label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        keys = {suites.canonical_key(row["state"]) for row in selected}
        if len(keys) != 128 or keys & used:
            raise RuntimeError(f"replacement_suite_not_disjoint:{spec.label}")
        actual = suites.suite_sha256(str(path))
        entries[spec.label] = {
            "path": str(path),
            "sha256": actual,
            "historical_sha256": spec.sha256,
            "historical_match": actual == spec.sha256,
            "seed": spec.seed,
            "status": "replacement_not_original",
        }
        used |= keys
    result = {
        "schema": "consumed_opening_suite_replacement_registry_v1",
        "status": "exploratory_only_not_historical_registry",
        "reason": "Historical suite files and original replay exclusion inputs are unavailable.",
        "selection": {
            "population": "run_pr249_fresh_suite_generalization.all_openings",
            "selector": "build_opening_suite.select_diverse",
            "historical_replay_exclusions": "unavailable_not_applied",
            "cross_replacement_exclusion": "all prior replacement states",
        },
        "entries": entries,
    }
    write_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    regenerate(canonical=args.canonical, out_dir=args.out_dir, manifest=args.manifest)


if __name__ == "__main__":
    main()
