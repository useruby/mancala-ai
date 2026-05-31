#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.build_forensic_references import build_reference_artifact
from ml.alphazero_lite.forensic_suite import load_suite
from ml.alphazero_lite.validate_forensic_reference_artifact import (
    validate_reference_artifact,
)


DEFAULT_SUITE_PATH = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_OUTPUT_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_OVERRIDE_PATH = Path(
    "ml/alphazero_lite/fixtures/incumbent_forensic_reference_audited_overrides_v1.json"
)
DEFAULT_POLICY_SIMULATIONS = 1200
DEFAULT_VALUE_SIMULATIONS = 1800
DEFAULT_SEED = 2040


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--override-artifact", default=str(DEFAULT_OVERRIDE_PATH))
    parser.add_argument(
        "--policy-simulations", type=int, default=DEFAULT_POLICY_SIMULATIONS
    )
    parser.add_argument(
        "--value-simulations", type=int, default=DEFAULT_VALUE_SIMULATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-seed", type=int, action="append", dest="sample_seeds")
    return parser.parse_args()


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_override_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    override_path = Path(path)
    if not override_path.exists():
        return []
    payload = _load_json(override_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("override artifact must contain a rows list")
    return [dict(row) for row in rows]


def build_tracked_reference_artifact(
    *,
    suite_path: str | Path,
    out_path: str | Path,
    override_artifact_path: str | Path | None,
    policy_simulations: int,
    value_simulations: int,
    seed: int,
    sample_seeds: list[int] | None = None,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    ordered_canonicals = [position.canonical_key for position in suite]

    with tempfile.TemporaryDirectory(
        prefix="azlite-tracked-forensic-reference-"
    ) as tmp:
        base_artifact_path = Path(tmp) / "base_reference_artifact.json"
        build_reference_artifact(
            suite_path=suite_path,
            out_path=base_artifact_path,
            policy_simulations=policy_simulations,
            value_simulations=value_simulations,
            seed=seed,
            sample_seeds=sample_seeds,
        )
        generated_artifact = _load_json(base_artifact_path)

    generated_rows = generated_artifact["rows"]
    merged_rows_by_canonical = {
        str(row["canonical_state"]): {**row, "reference_source": "generated"}
        for row in generated_rows
    }

    override_rows = _load_override_rows(override_artifact_path)
    overridden_row_ids: list[str] = []
    for row in override_rows:
        canonical_state = str(row["canonical_state"])
        if canonical_state not in merged_rows_by_canonical:
            raise ValueError(
                f"override row {row.get('id')} does not exist in the checked-in forensic suite"
            )
        merged_rows_by_canonical[canonical_state] = {
            **row,
            "reference_source": "audited_override",
        }
        overridden_row_ids.append(str(row["id"]))

    merged_rows = [
        merged_rows_by_canonical[canonical] for canonical in ordered_canonicals
    ]
    payload = {
        "schema": generated_artifact["schema"],
        "suite_path": generated_artifact["suite_path"],
        "reference": generated_artifact["reference"],
        "meta": {
            "generated_by": "build_tracked_forensic_reference_artifact_v1",
            "override_artifact_path": None
            if override_artifact_path is None
            else str(Path(override_artifact_path)),
            "override_row_ids": overridden_row_ids,
            "generated_reference": generated_artifact["reference"],
        },
        "rows": merged_rows,
    }

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = validate_reference_artifact(
        suite_path=suite_path,
        reference_artifact_path=output_path,
    )
    if not summary["valid"]:
        raise ValueError(json.dumps(summary, indent=2))
    return payload


def main() -> None:
    args = parse_args()
    payload = build_tracked_reference_artifact(
        suite_path=args.suite,
        out_path=args.out,
        override_artifact_path=args.override_artifact,
        policy_simulations=args.policy_simulations,
        value_simulations=args.value_simulations,
        seed=args.seed,
        sample_seeds=args.sample_seeds,
    )
    print(f"wrote {len(payload['rows'])} tracked forensic reference rows to {args.out}")


if __name__ == "__main__":
    main()
