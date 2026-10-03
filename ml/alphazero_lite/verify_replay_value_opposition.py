"""Portable verifier for replay-versus-fresh-value gradient evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _close(actual: float, expected: float | None, tolerance: float = 2e-6) -> bool:
    if expected is None:
        return actual is None
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def recompute(group: dict[str, Any]) -> dict[str, float | None]:
    names = list(group["component_norms"])
    dots = group["cross_objective_dot_products"]
    dot = sum(group["source_objective_dot_contributions"].values())
    norm2 = sum(dots[left][right] for left in names for right in names)
    grad_norm = math.sqrt(max(0.0, norm2))
    fresh_norm = group["fresh_value_norm"]
    return {
        "dot": dot,
        "gradient_norm": grad_norm,
        "fresh_value_norm": fresh_norm,
        "cosine": dot / (grad_norm * fresh_norm) if grad_norm and fresh_norm else None,
    }


def _passes(entry: dict[str, Any]) -> bool:
    cosine = entry["groups"]["all_parameters"]["cosine"]
    return cosine is not None and cosine <= -0.05


def verify(payload: dict[str, Any]) -> None:
    if payload.get("schema") != "replay_value_opposition_results_v1":
        raise ValueError("unsupported_results_schema")
    manifest = payload["manifest"]
    if (
        hashlib.sha256(
            json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        ).hexdigest()
        != payload["manifest_sha256"]
    ):
        raise ValueError("manifest_hash_mismatch")
    for checkpoint in ("seed455", "original_o0_e4_secondary"):
        record = payload[checkpoint]
        if record["immutability"] != {
            "checkpoint_file_unchanged": True,
            "parameters_and_buffers_unchanged": True,
            "optimizer_steps": 0,
        }:
            raise ValueError(f"immutability_record_invalid:{checkpoint}")
        for cohort in record["results"].values():
            for part in cohort.values():
                for group_name, group in part["groups"].items():
                    metrics = recompute(group)
                    for metric, expected in metrics.items():
                        actual = group[metric]
                        if expected is None:
                            if actual is not None:
                                raise ValueError(
                                    f"undefined_metric_mismatch:{checkpoint}:{group_name}:{metric}"
                                )
                        elif actual is None or not _close(actual, expected):
                            raise ValueError(
                                f"metric_mismatch:{checkpoint}:{group_name}:{metric}"
                            )
                    if not _close(group["contribution_dot_sum"], metrics["dot"]):
                        raise ValueError(
                            f"contribution_sum_mismatch:{checkpoint}:{group_name}"
                        )
                    dots = group["cross_objective_dot_products"]
                    for left, row in dots.items():
                        for right, value in row.items():
                            if not _close(value, dots[right][left]):
                                raise ValueError("asymmetric_cross_dot")
                            if left == right and not _close(
                                value, group["component_norms"][left] ** 2
                            ):
                                raise ValueError("component_norm_diagonal_mismatch")
                for group_name, metrics in part["fresh_policy_alignment"].items():
                    expected_cosine = (
                        metrics["dot"]
                        / (metrics["gradient_norm"] * metrics["fresh_value_norm"])
                        if metrics["gradient_norm"] and metrics["fresh_value_norm"]
                        else None
                    )
                    if not _close(metrics["cosine"], expected_cosine):
                        raise ValueError(f"fresh_policy_cosine_mismatch:{group_name}")
    seed = payload["seed455"]
    primary = seed["results"][">32"]
    whole_pass = _passes(primary["all"])
    supporting = [
        f"partition_{index}"
        for index in range(4)
        if _passes(primary[f"partition_{index}"])
    ]
    recommend = whole_pass and len(supporting) >= 3
    recorded = payload["classification"]
    if recorded["primary_all_parameters_pass"] is not whole_pass:
        raise ValueError("primary_decision_mismatch")
    if recorded["supporting_partitions"] != supporting:
        raise ValueError("partition_decision_mismatch")
    if recorded["recommend_value_target_provenance_calibration_audit"] is not recommend:
        raise ValueError("recommendation_mismatch")
    expected_class = (
        "recommend_later_value_target_provenance_calibration_audit"
        if recommend
        else "no_robust_net_opposition_signal"
    )
    if (
        recorded["classification"] != expected_class
        or recorded["strength_claim"] is not False
    ):
        raise ValueError("classification_or_scope_mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    verify(json.loads(args.results.read_text(encoding="utf-8")))
    print("replay_value_opposition_results_verified")


if __name__ == "__main__":
    main()
