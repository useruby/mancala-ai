"""Checkpoint-free aggregation for seed427 retrospective validation evidence."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


BUCKETS = {
    ">32": lambda value: value > 32,
    "17-32": lambda value: 17 <= value <= 32,
    "<=16": lambda value: value <= 16,
}


def row_losses(
    logits: list[float],
    value_prediction: float,
    target_policy: list[float],
    target_value: float,
    legal_mask: list[int],
    policy_weight: float,
) -> dict[str, float]:
    """Compute the historical legal-masked cross entropy and Huber loss."""
    if not (len(logits) == len(target_policy) == len(legal_mask) == 6):
        raise ValueError("prediction_shape_mismatch")
    if policy_weight < 0 or not math.isfinite(policy_weight):
        raise ValueError("policy_weight_invalid")
    legal = [index for index, flag in enumerate(legal_mask) if flag]
    if not legal:
        raise ValueError("empty_legal_mask")
    maximum = max(logits[index] for index in legal)
    log_z = maximum + math.log(sum(math.exp(logits[i] - maximum) for i in legal))
    policy_loss = -sum(target_policy[i] * (logits[i] - log_z) for i in legal)
    error = abs(value_prediction - target_value)
    value_loss = 0.5 * error * error if error <= 1.0 else error - 0.5
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "total_loss": policy_loss + 0.3 * value_loss,
        "policy_weight": policy_weight,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate globally; equal-identity means average rows within identity first."""
    result: dict[str, Any] = {}
    scopes = {
        "all": lambda row: True,
        "seen": lambda row: row["membership"]["subset"] == "seen",
        "unseen": lambda row: row["membership"]["subset"] == "unseen",
    }
    for scope, include in scopes.items():
        for bucket, predicate in {"overall": lambda _: True, **BUCKETS}.items():
            selected = [
                row
                for row in rows
                if include(row) and predicate(row["membership"]["active_stones"])
            ]
            exposure = _means(selected)
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in selected:
                groups[row["membership"]["canonical_identity"]].append(row)
            identity_rows = [_means(group) for group in groups.values()]
            equal_identity = {
                key: sum(item[key] for item in identity_rows) / len(identity_rows)
                if identity_rows
                else 0.0
                for key in ("policy_loss", "value_loss", "total_loss")
            }
            result[f"{scope}/{bucket}"] = {
                "weighted_positions": len(selected),
                "policy_weight_denominator": sum(
                    r["losses"]["policy_weight"] for r in selected
                ),
                "canonical_identity_denominator": len(groups),
                "exposure_weighted": exposure,
                "equal_canonical_identity": equal_identity,
            }
    return result


def _means(rows: list[dict[str, Any]]) -> dict[str, float]:
    policy_denominator = sum(row["losses"]["policy_weight"] for row in rows)
    position_denominator = len(rows)
    policy = (
        sum(
            row["losses"]["policy_loss"] * row["losses"]["policy_weight"]
            for row in rows
        )
        / policy_denominator
        if policy_denominator
        else 0.0
    )
    value = (
        sum(row["losses"]["value_loss"] for row in rows) / position_denominator
        if rows
        else 0.0
    )
    return {
        "policy_loss": policy,
        "value_loss": value,
        "total_loss": policy + 0.3 * value,
    }
