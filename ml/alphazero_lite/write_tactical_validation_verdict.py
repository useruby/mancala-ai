#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import check_bucket_promotion_gate


BUCKET_ORDER = [
    "capture_available",
    "high_imbalance",
    "high_value_swing",
    "sparse_endgame",
    "opening_plies_1_8",
    "overall",
]
DERIVED_MAX_METRICS = ["blunder_rate_0_20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selected-artifact", required=True)
    parser.add_argument("--baseline-forensics", required=True)
    parser.add_argument("--candidate-forensics", required=True)
    parser.add_argument("--bucket-gate", required=True)
    parser.add_argument("--regression-report", required=True)
    parser.add_argument("--arena-report", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_bucket_gate(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("bucket_gate must be a JSON object")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError("bucket_gate must include boolean passed")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ValueError("bucket_gate must include checks list")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"bucket_gate checks[{index}] must be an object")
        if not isinstance(check.get("id"), str):
            raise ValueError(f"bucket_gate checks[{index}] must include string id")
        if str(check["id"]).count(".") != 1:
            raise ValueError(
                f"bucket_gate checks[{index}] must include id in <bucket>.<metric> form"
            )
        if check.get("comparison") not in {"max", "min"}:
            raise ValueError(
                f"bucket_gate checks[{index}] must include valid comparison"
            )
        if not isinstance(check.get("passed"), bool):
            raise ValueError(f"bucket_gate checks[{index}] must include boolean passed")
        for field in ("baseline_value", "candidate_value", "threshold"):
            value = check.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"bucket_gate checks[{index}] must include numeric {field}"
                )
    return payload


def validate_regression_report(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("regression_report must be a JSON object")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError("regression_report must include boolean passed")
    return payload


def validate_arena_report(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("arena_report must be a JSON object")
    promotion_decision = payload.get("promotion_decision")
    if not isinstance(promotion_decision, dict):
        raise ValueError("arena_report must include promotion_decision object")
    if not isinstance(promotion_decision.get("passed"), bool):
        raise ValueError("arena_report promotion_decision must include boolean passed")
    return payload


def bucket_gate_metric_specs(bucket_gate: dict[str, Any]) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for check in bucket_gate["checks"]:
        bucket, metric_name = str(check["id"]).split(".", 1)
        specs.append(
            {
                "id": str(check["id"]),
                "bucket": bucket,
                "name": metric_name,
                "comparison": str(check["comparison"]),
            }
        )
    return specs


def compare_metric(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bucket: str,
    name: str,
    comparison: str,
) -> dict[str, Any] | None:
    if bucket == "overall" and name == "blunder_rate_0_20":
        return None

    baseline_value = check_bucket_promotion_gate.metric(baseline, bucket, name)
    candidate_value = check_bucket_promotion_gate.metric(candidate, bucket, name)

    if comparison == "max" and candidate_value > baseline_value:
        return {
            "id": f"{bucket}.{name}",
            "comparison": comparison,
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
        }

    if comparison == "min" and candidate_value < baseline_value:
        return {
            "id": f"{bucket}.{name}",
            "comparison": comparison,
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
        }

    return None


def detect_extra_adverse_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bucket_gate: dict[str, Any],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    canonical_gate = check_bucket_promotion_gate.evaluate_gate(baseline, candidate)
    canonical_specs = bucket_gate_metric_specs(canonical_gate)
    covered = {spec["id"] for spec in bucket_gate_metric_specs(bucket_gate)}

    for spec in canonical_specs:
        delta = compare_metric(
            baseline=baseline,
            candidate=candidate,
            bucket=spec["bucket"],
            name=spec["name"],
            comparison=spec["comparison"],
        )
        if delta is not None and delta["id"] not in covered:
            deltas.append(delta)

    for bucket in BUCKET_ORDER:
        for name in DERIVED_MAX_METRICS:
            delta = compare_metric(
                baseline=baseline,
                candidate=candidate,
                bucket=bucket,
                name=name,
                comparison="max",
            )
            if delta is not None and delta["id"] not in covered:
                deltas.append(delta)

    unique_deltas: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for delta in deltas:
        if delta["id"] in seen_ids:
            continue
        seen_ids.add(delta["id"])
        unique_deltas.append(delta)

    return unique_deltas


def validate_verdict_metric_surfaces(report: dict[str, Any]) -> None:
    check_bucket_promotion_gate.validate_metric_readiness(report)
    for bucket in BUCKET_ORDER:
        if bucket == "overall":
            continue
        for name in DERIVED_MAX_METRICS:
            check_bucket_promotion_gate.metric(report, bucket, name)


def error_payload(
    *,
    args: argparse.Namespace,
    message: str,
    path: str,
) -> dict[str, Any]:
    return {
        "schema": "azlite_tactical_validation_verdict_v1",
        "passed": False,
        "verdict": "fail",
        "failure_reasons": ["input_validation_failed"],
        "rubric": {
            "mode": "strict_binary",
            "strict_binary": True,
            "canonical_blunder_metric": "blunder_rate_0_20",
        },
        "run_dir": args.run_dir,
        "selected_artifact": args.selected_artifact,
        "baseline_forensics_path": args.baseline_forensics,
        "candidate_forensics_path": args.candidate_forensics,
        "bucket_gate_path": args.bucket_gate,
        "regression_report_path": args.regression_report,
        "arena_report_path": args.arena_report,
        "extra_adverse_deltas": [],
        "error": {
            "code": "invalid_input",
            "message": message,
            "path": path,
        },
    }


def build_verdict(
    *,
    run_dir: str,
    selected_artifact: str,
    baseline_forensics_path: str,
    candidate_forensics_path: str,
    bucket_gate_path: str,
    regression_report_path: str,
    arena_report_path: str,
    baseline_forensics: dict[str, Any],
    candidate_forensics: dict[str, Any],
    bucket_gate: dict[str, Any],
    regression_report: dict[str, Any],
    arena_report: dict[str, Any],
) -> dict[str, Any]:
    extra_adverse_deltas = detect_extra_adverse_deltas(
        baseline_forensics,
        candidate_forensics,
        bucket_gate,
    )

    failure_reasons: list[str] = []
    if not bucket_gate["passed"]:
        failure_reasons.append("bucket_gate_failed")
    if not regression_report["passed"]:
        failure_reasons.append("regression_check_failed")
    if not arena_report["promotion_decision"]["passed"]:
        failure_reasons.append("arena_promotion_failed")
    if extra_adverse_deltas:
        failure_reasons.append("extra_adverse_delta_detected")

    passed = not failure_reasons
    return {
        "schema": "azlite_tactical_validation_verdict_v1",
        "passed": passed,
        "verdict": "pass" if passed else "fail",
        "failure_reasons": failure_reasons,
        "rubric": {
            "mode": "strict_binary",
            "strict_binary": True,
            "canonical_blunder_metric": "blunder_rate_0_20",
        },
        "run_dir": run_dir,
        "selected_artifact": selected_artifact,
        "baseline_forensics_path": baseline_forensics_path,
        "candidate_forensics_path": candidate_forensics_path,
        "bucket_gate_path": bucket_gate_path,
        "regression_report_path": regression_report_path,
        "arena_report_path": arena_report_path,
        "extra_adverse_deltas": extra_adverse_deltas,
        "bucket_gate": bucket_gate,
        "regression_report": regression_report,
        "arena_report": arena_report,
    }


def write_report(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        baseline_forensics = (
            check_bucket_promotion_gate.load_validate_and_check_metrics(
                args.baseline_forensics
            )
        )
        candidate_forensics = (
            check_bucket_promotion_gate.load_validate_and_check_metrics(
                args.candidate_forensics
            )
        )
        validate_verdict_metric_surfaces(baseline_forensics)
        validate_verdict_metric_surfaces(candidate_forensics)
        bucket_gate = validate_bucket_gate(load_json(args.bucket_gate))
        regression_report = validate_regression_report(
            load_json(args.regression_report)
        )
        arena_report = validate_arena_report(load_json(args.arena_report))
        verdict = build_verdict(
            run_dir=args.run_dir,
            selected_artifact=args.selected_artifact,
            baseline_forensics_path=args.baseline_forensics,
            candidate_forensics_path=args.candidate_forensics,
            bucket_gate_path=args.bucket_gate,
            regression_report_path=args.regression_report,
            arena_report_path=args.arena_report,
            baseline_forensics=baseline_forensics,
            candidate_forensics=candidate_forensics,
            bucket_gate=bucket_gate,
            regression_report=regression_report,
            arena_report=arena_report,
        )
        write_report(args.out, verdict)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        path = args.arena_report
        reported_error: Exception = error
        for candidate_path, loader in (
            (
                args.baseline_forensics,
                lambda current_path: validate_verdict_metric_surfaces(
                    check_bucket_promotion_gate.load_validate_and_check_metrics(
                        current_path
                    )
                ),
            ),
            (
                args.candidate_forensics,
                lambda current_path: validate_verdict_metric_surfaces(
                    check_bucket_promotion_gate.load_validate_and_check_metrics(
                        current_path
                    )
                ),
            ),
            (
                args.bucket_gate,
                lambda current_path: validate_bucket_gate(load_json(current_path)),
            ),
            (
                args.regression_report,
                lambda current_path: validate_regression_report(
                    load_json(current_path)
                ),
            ),
            (
                args.arena_report,
                lambda current_path: validate_arena_report(load_json(current_path)),
            ),
        ):
            try:
                loader(candidate_path)
            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as specific_error:
                path = candidate_path
                reported_error = specific_error
                break
        write_report(
            args.out,
            error_payload(
                args=args,
                message=str(reported_error),
                path=path,
            ),
        )
        print(str(error), file=sys.stderr)
        return 1

    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
