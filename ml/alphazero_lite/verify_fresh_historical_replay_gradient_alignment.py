"""Portable verifier for published gradient geometry and follow-up classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _close(
    actual: float | None, expected: float | None, *, tolerance: float = 1e-6
) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def recompute(entry: dict[str, Any]) -> dict[str, float | None]:
    dots = entry["source_pairwise_dots"]
    norms = entry["source_norms"]
    fresh = "fresh"
    historical = [name for name in norms if name != fresh]
    f2 = norms[fresh] ** 2
    f_dot_h = sum(dots[fresh][name] for name in historical)
    h2 = sum(dots[left][right] for left in historical for right in historical)
    f_norm, h_norm = math.sqrt(max(0.0, f2)), math.sqrt(max(0.0, h2))
    g_norm = math.sqrt(max(0.0, f2 + h2 + 2.0 * f_dot_h))
    return {
        "dot": f_dot_h,
        "cosine": f_dot_h / (f_norm * h_norm) if f_norm and h_norm else None,
        "fresh_norm": f_norm,
        "historical_norm": h_norm,
        "historical_to_fresh_norm": h_norm / f_norm if f_norm else None,
        "combined_norm": g_norm,
        "cancellation": 1.0 - g_norm / (f_norm + h_norm) if f_norm + h_norm else 0.0,
        "retained_fresh_projection": 1.0 + f_dot_h / f2 if f2 else None,
    }


def meets_rule(metrics: dict[str, Any]) -> bool:
    return (
        metrics["dot"] < 0
        and metrics["historical_norm"] >= metrics["fresh_norm"]
        and metrics["retained_fresh_projection"] is not None
        and metrics["retained_fresh_projection"] <= 0.5
    )


def verify(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "fresh_historical_replay_gradient_alignment_v1":
        raise ValueError("unsupported_gradient_summary_schema")
    for checkpoint_name in ("seed455", "original_o0_e4_secondary"):
        for cohort in payload[checkpoint_name]["results"].values():
            for partition in cohort.values():
                for objective in partition["objectives"].values():
                    for group in objective.values():
                        if (
                            not isinstance(group, dict)
                            or "source_pairwise_dots" not in group
                        ):
                            continue
                        expected = group["fresh_historical"]
                        observed = recompute(group)
                        for key, value in observed.items():
                            if not _close(value, expected[key]):
                                raise ValueError(
                                    f"gradient_metric_mismatch:{checkpoint_name}:{key}"
                                )
    primary = payload["seed455"]["results"][">32"]
    combined = primary["all"]["objectives"]["combined"]["shared_trunk"][
        "fresh_historical"
    ]
    supporting = []
    for index in range(4):
        part = f"partition_{index}"
        combined_part = primary[part]["objectives"]["combined"]["shared_trunk"][
            "fresh_historical"
        ]
        policy_part = primary[part]["objectives"]["policy"]["shared_trunk"][
            "fresh_historical"
        ]
        if meets_rule(combined_part) and meets_rule(policy_part):
            supporting.append(part)
    recommendation = meets_rule(combined) and len(supporting) >= 3
    recorded = payload["classification"]
    if recorded["supporting_partitions"] != supporting:
        raise ValueError("partition_classification_mismatch")
    if recorded["recommend_later_replay_reweighting_experiment"] is not recommendation:
        raise ValueError("recommendation_mismatch")
    expected_class = (
        "supports_separately_registered_replay_reweighting_experiment"
        if recommendation
        else "audit_does_not_support_follow_up"
    )
    if (
        recorded["classification"] != expected_class
        or recorded["strength_claim"] is not False
    ):
        raise ValueError("classification_or_strength_scope_mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    verify(payload)
    manifest_path = args.summary.with_name(args.summary.stem + "-manifest.json")
    if manifest_path.is_file():
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if digest != payload.get("manifest_sha256"):
            raise ValueError("published_manifest_hash_mismatch")
    print("fresh_historical_gradient_summary_verified")


if __name__ == "__main__":
    main()
