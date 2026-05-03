#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-gate", required=True)
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--exploratory-summary", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_gate_payload(name: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError(f"{name} must include boolean passed")
    return payload


def validate_exploratory_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("exploratory_summary must be a JSON object")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError("exploratory_summary must include boolean passed")
    qualifying_seed_count = payload.get("qualifying_seed_count")
    if isinstance(qualifying_seed_count, bool) or not isinstance(qualifying_seed_count, int):
        raise ValueError("exploratory_summary must include integer qualifying_seed_count")
    required_qualifying_seed_count = payload.get("required_qualifying_seed_count")
    if isinstance(required_qualifying_seed_count, bool) or not isinstance(required_qualifying_seed_count, int):
        raise ValueError("exploratory_summary must include integer required_qualifying_seed_count")
    return payload


def build_decision(
    bucket_gate: dict[str, Any],
    promotion_gate: dict[str, Any],
    exploratory_summary: dict[str, Any],
) -> dict[str, Any]:
    failure_reasons: list[str] = []
    if not bucket_gate.get("passed"):
        failure_reasons.append("bucket_gate_failed")
    if not promotion_gate.get("passed"):
        failure_reasons.append("local_promotion_gate_failed")
    required_qualifying_seed_count = int(exploratory_summary.get("required_qualifying_seed_count", 0))
    if (not exploratory_summary.get("passed")) or int(exploratory_summary.get("qualifying_seed_count", 0)) < required_qualifying_seed_count:
        failure_reasons.append("exploratory_seed_confirmation_failed")

    return {
        "schema": "azlite_tactical_lane_decision_v1",
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "bucket_gate": bucket_gate,
        "promotion_gate": promotion_gate,
        "exploratory_summary": exploratory_summary,
    }


def main() -> int:
    args = parse_args()
    try:
        decision = build_decision(
            validate_gate_payload("bucket_gate", load_json(args.bucket_gate)),
            validate_gate_payload("promotion_gate", load_json(args.promotion_gate)),
            validate_exploratory_summary(load_json(args.exploratory_summary)),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
