#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


REQUIRED_BUCKETS = {
    "capture_available",
    "high_imbalance",
    "high_value_swing",
    "sparse_endgame",
    "opening_plies_1_8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-forensics", required=True)
    parser.add_argument("--candidate-forensics", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shared_reference_provenance(report: dict[str, Any]) -> dict[str, Any]:
    reference = report.get("reference")
    if not isinstance(reference, dict) or reference.get("kind") != "shared_artifact":
        raise ValueError("shared reference provenance is required")

    shared_reference = reference.get("shared_reference")
    if not isinstance(shared_reference, dict):
        raise ValueError("shared reference provenance is required")

    policy_simulations = shared_reference.get("policy_simulations")
    value_simulations = shared_reference.get("value_simulations")
    sample_seeds = shared_reference.get("sample_seeds")
    if (
        isinstance(policy_simulations, bool)
        or not isinstance(policy_simulations, int)
        or isinstance(value_simulations, bool)
        or not isinstance(value_simulations, int)
        or not isinstance(sample_seeds, list)
        or not sample_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in sample_seeds
        )
    ):
        raise ValueError("shared reference provenance is malformed")

    return {
        "policy_simulations": policy_simulations,
        "value_simulations": value_simulations,
        "sample_seeds": sample_seeds,
    }


def validate_shared_reference_provenance(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> None:
    baseline_provenance = _shared_reference_provenance(baseline)
    candidate_provenance = _shared_reference_provenance(candidate)
    if baseline_provenance != candidate_provenance:
        raise ValueError("shared reference provenance must match between reports")


def system_metrics(
    report: dict[str, Any], bucket: str, system: str = "challenger"
) -> dict[str, Any]:
    if bucket == "overall":
        systems = report.get("systems")
        if not isinstance(systems, dict) or system not in systems:
            raise ValueError(f"missing systems.{system}")
        overall = systems[system].get("overall")
        if not isinstance(overall, dict):
            raise ValueError(f"missing systems.{system}.overall")
        return overall

    buckets = report.get("buckets")
    if not isinstance(buckets, dict) or bucket not in buckets:
        raise ValueError(f"missing bucket {bucket}")
    bucket_payload = buckets[bucket]
    if not isinstance(bucket_payload, dict):
        raise ValueError(f"invalid bucket payload for {bucket}")
    systems = bucket_payload.get("systems")
    if not isinstance(systems, dict) or system not in systems:
        raise ValueError(f"missing bucket system {bucket}.{system}")
    metrics = systems[system]
    if not isinstance(metrics, dict):
        raise ValueError(f"invalid bucket system payload for {bucket}.{system}")
    return metrics


def metric(report: dict[str, Any], bucket: str, name: str) -> float:
    metrics = system_metrics(report, bucket)
    if name in metrics:
        return float(metrics[name])

    if name == "blunder_rate_0_20":
        positions = report.get("buckets", {}).get(bucket, {}).get("positions")
        if isinstance(positions, bool) or not isinstance(positions, int):
            raise ValueError(f"bucket {bucket} positions must be an integer")
        rows = report.get("systems", {}).get("challenger", {}).get("rows")
        if not isinstance(rows, list):
            raise ValueError("missing systems.challenger.rows")
        bucket_rows = [
            row for row in rows if isinstance(row, dict) and row.get("bucket") == bucket
        ]
        if not bucket_rows:
            raise ValueError(f"missing challenger rows for {bucket}")
        if len(bucket_rows) != positions:
            raise ValueError(f"bucket {bucket} rows must match positions")
        regret_values: list[float] = []
        for row in bucket_rows:
            regret = row.get("regret")
            if isinstance(regret, bool) or not isinstance(regret, (int, float)):
                raise ValueError(f"bucket {bucket} rows must include numeric regret")
            regret_values.append(float(regret))
        stable_regret_values = [
            float(row["regret"])
            for row in bucket_rows
            if not row.get("reference_unstable")
        ]
        if not stable_regret_values:
            raise ValueError(
                f"bucket {bucket} requires at least one stable strict-metric row"
            )
        blunders = sum(1 for regret in stable_regret_values if regret >= 0.20)
        return round(blunders / len(stable_regret_values), 4)

    raise ValueError(f"missing metric {bucket}.{name}")


def add_max_check(
    checks: list[dict[str, Any]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bucket: str,
    name: str,
    allowance: float,
) -> None:
    baseline_value = metric(baseline, bucket, name)
    candidate_value = metric(candidate, bucket, name)
    threshold = baseline_value + allowance
    check_id = f"{bucket}.{name}"
    checks.append(
        {
            "id": check_id,
            "comparison": "max",
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "threshold": threshold,
            "passed": candidate_value <= threshold,
            "metric": check_id,
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
    )


def add_min_check(
    checks: list[dict[str, Any]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bucket: str,
    name: str,
    allowance: float,
) -> None:
    baseline_value = metric(baseline, bucket, name)
    candidate_value = metric(candidate, bucket, name)
    threshold = baseline_value - allowance
    check_id = f"{bucket}.{name}"
    checks.append(
        {
            "id": check_id,
            "comparison": "min",
            "baseline_value": baseline_value,
            "candidate_value": candidate_value,
            "threshold": threshold,
            "passed": candidate_value >= threshold,
            "metric": check_id,
            "baseline": baseline_value,
            "candidate": candidate_value,
        }
    )


def validate_report_shape(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")

    for bucket in REQUIRED_BUCKETS:
        bucket_metrics = system_metrics(report, bucket)
        positions = report.get("buckets", {}).get(bucket, {}).get("positions")
        minimum_positions = 40 if bucket == "opening_plies_1_8" else 20

        if isinstance(positions, bool) or not isinstance(positions, int):
            raise ValueError(f"bucket {bucket} positions must be an integer")
        if positions < minimum_positions:
            raise ValueError(
                f"bucket {bucket} requires at least {minimum_positions} positions"
            )
        if isinstance(bucket_metrics.get("average_regret"), bool) or not isinstance(
            bucket_metrics.get("average_regret"), (int, float)
        ):
            raise ValueError(f"bucket {bucket} missing average_regret")
        if isinstance(bucket_metrics.get("top1_agreement"), bool) or not isinstance(
            bucket_metrics.get("top1_agreement"), (int, float)
        ):
            raise ValueError(f"bucket {bucket} missing top1_agreement")

    overall = system_metrics(report, "overall")
    if isinstance(overall.get("average_regret"), bool) or not isinstance(
        overall.get("average_regret"), (int, float)
    ):
        raise ValueError("overall missing average_regret")
    if isinstance(overall.get("top1_agreement"), bool) or not isinstance(
        overall.get("top1_agreement"), (int, float)
    ):
        raise ValueError("overall missing top1_agreement")


def evaluate_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    validate_report_shape(baseline)
    validate_report_shape(candidate)
    validate_shared_reference_provenance(baseline, candidate)

    checks: list[dict[str, Any]] = []
    add_max_check(
        checks, baseline, candidate, "capture_available", "average_regret", 0.005
    )
    add_max_check(
        checks, baseline, candidate, "capture_available", "blunder_rate_0_20", 0.02
    )
    add_max_check(
        checks, baseline, candidate, "high_imbalance", "average_regret", 0.005
    )
    add_max_check(
        checks, baseline, candidate, "high_value_swing", "average_regret", 0.005
    )
    add_min_check(checks, baseline, candidate, "sparse_endgame", "top1_agreement", 0.03)
    add_max_check(
        checks, baseline, candidate, "opening_plies_1_8", "average_regret", 0.005
    )
    add_max_check(checks, baseline, candidate, "overall", "average_regret", 0.005)
    add_min_check(checks, baseline, candidate, "overall", "top1_agreement", 0.02)

    return {
        "schema": "azlite_bucket_promotion_gate_v1",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def write_report(path: str, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_and_validate_report(path: str) -> dict[str, Any]:
    report = load_json(path)
    validate_report_shape(report)
    _shared_reference_provenance(report)
    return report


def validate_metric_readiness(report: dict[str, Any]) -> None:
    metric(report, "capture_available", "average_regret")
    metric(report, "capture_available", "blunder_rate_0_20")
    metric(report, "high_imbalance", "average_regret")
    metric(report, "high_value_swing", "average_regret")
    metric(report, "sparse_endgame", "top1_agreement")
    metric(report, "opening_plies_1_8", "average_regret")
    metric(report, "overall", "average_regret")
    metric(report, "overall", "top1_agreement")


def load_validate_and_check_metrics(path: str) -> dict[str, Any]:
    report = load_and_validate_report(path)
    validate_metric_readiness(report)
    return report


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_gate(
            load_json(args.baseline_forensics),
            load_json(args.candidate_forensics),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        path = args.candidate_forensics
        reported_error: Exception = error
        try:
            load_validate_and_check_metrics(args.baseline_forensics)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as baseline_error:
            path = args.baseline_forensics
            reported_error = baseline_error
        else:
            try:
                load_validate_and_check_metrics(args.candidate_forensics)
            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as candidate_error:
                reported_error = candidate_error
        result = {
            "schema": "azlite_bucket_promotion_gate_v1",
            "passed": False,
            "checks": [],
            "error": {
                "code": "invalid_forensics",
                "message": str(reported_error),
                "path": path,
            },
        }
        write_report(args.out, result)
        raise SystemExit(1)

    write_report(args.out, result)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
