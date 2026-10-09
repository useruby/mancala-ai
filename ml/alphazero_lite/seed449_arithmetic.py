"""Pure float64 accounting for seed449 retrospective policy localization."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

STRATA = (
    ("1", lambda n: n == 1),
    ("2-4", lambda n: 2 <= n <= 4),
    (">=5", lambda n: n >= 5),
)


def policy_ce(logits: list[float], target: list[float], mask: list[int]) -> float:
    """Production legal-action softmax CE semantics using stable float64 math."""
    if len(logits) != 6 or len(target) != 6 or len(mask) != 6:
        raise ValueError("policy_shape_mismatch")
    legal = [i for i, flag in enumerate(mask) if flag]
    if not legal or any(flag not in (0, 1) for flag in mask):
        raise ValueError("legal_mask_invalid")
    if any(target[i] != 0.0 for i, flag in enumerate(mask) if not flag):
        raise ValueError("target_mass_on_illegal_action")
    maximum = max(float(logits[i]) for i in legal)
    log_z = maximum + math.log(sum(math.exp(float(logits[i]) - maximum) for i in legal))
    return -sum(float(target[i]) * (float(logits[i]) - log_z) for i in legal)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    return (
        values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    )


def analyze(
    rows: list[dict[str, Any]], initial: str, comparison: str
) -> dict[str, Any]:
    """Reconstruct identity and exposure means, strata, and signed mass."""
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_identity[row["input_identity"]].append(row)
    identity_rows = []
    for identity, group in by_identity.items():
        delta_values = [r["losses"][comparison] - r["losses"][initial] for r in group]
        identity_rows.append(
            {
                "input_identity": identity,
                "canonical_identity": group[0]["canonical_identity"],
                "exposure_count": len(group),
                "compact_rows": [r["compact_row"] for r in group],
                "source_refs": [r["source_ref"] for r in group],
                "initial_mean_loss": sum(r["losses"][initial] for r in group)
                / len(group),
                "comparison_mean_loss": sum(r["losses"][comparison] for r in group)
                / len(group),
                "mean_change": sum(delta_values) / len(group),
                "row_changes": delta_values,
            }
        )
    n_id, n_rows = len(identity_rows), len(rows)
    exp_delta = (
        sum(r["losses"][comparison] - r["losses"][initial] for r in rows) / n_rows
        if n_rows
        else 0.0
    )
    eq_delta = sum(r["mean_change"] for r in identity_rows) / n_id if n_id else 0.0
    mean_n = n_rows / n_id if n_id else 0.0
    covariance = (
        sum(
            (r["exposure_count"] - mean_n) * (r["mean_change"] - eq_delta)
            for r in identity_rows
        )
        / n_id
        if n_id
        else 0.0
    )
    strata: dict[str, Any] = {}
    for name, predicate in STRATA:
        selected = [r for r in identity_rows if predicate(r["exposure_count"])]
        count = sum(r["exposure_count"] for r in selected)
        changes = [change for r in selected for change in r["row_changes"]]
        id_contribution = (
            sum(r["mean_change"] for r in selected) / n_id if n_id else 0.0
        )
        exp_contribution = (
            sum(r["mean_change"] * r["exposure_count"] for r in selected) / n_rows
            if n_rows
            else 0.0
        )
        strata[name] = {
            "identity_count": len(selected),
            "exposure_count": count,
            "exposure_weighted_mean": sum(changes) / count if count else None,
            "equal_identity_mean": sum(r["mean_change"] for r in selected)
            / len(selected)
            if selected
            else None,
            "exposure_signed_contribution": exp_contribution,
            "equal_signed_contribution": id_contribution,
            "median_change": _median(changes),
            "improving_fraction": sum(v < 0 for v in changes) / count
            if count
            else None,
            "worsening_fraction": sum(v > 0 for v in changes) / count
            if count
            else None,
            "unchanged_fraction": sum(v == 0 for v in changes) / count
            if count
            else None,
        }
    changes = [r["losses"][comparison] - r["losses"][initial] for r in rows]
    return {
        "initial": initial,
        "comparison": comparison,
        "exposure_count": n_rows,
        "identity_count": n_id,
        "exposure_mean_change": exp_delta,
        "equal_identity_mean_change": eq_delta,
        "difference": exp_delta - eq_delta,
        "population_covariance": covariance,
        "covariance_over_mean_exposure": covariance / mean_n if mean_n else 0.0,
        "gross_improvement_mass": sum(v for v in changes if v < 0),
        "gross_worsening_mass": sum(v for v in changes if v > 0),
        "net_signed_mass": sum(changes),
        "strata": strata,
        "identities": identity_rows,
    }


def classify(result: dict[str, Any]) -> str:
    """Apply the frozen three-way descriptive classification."""
    strata = result["strata"]
    one = strata["1"]["exposure_weighted_mean"]
    multi_count = strata["2-4"]["exposure_count"] + strata[">=5"]["exposure_count"]
    multi = None
    if multi_count:
        multi = (
            strata["2-4"]["exposure_signed_contribution"]
            + strata[">=5"]["exposure_signed_contribution"]
        ) / (multi_count / result["exposure_count"])
    if one is None or multi is None:
        return "mixed_policy_response"
    exposure = result["exposure_mean_change"]
    equal = result["equal_identity_mean_change"]
    if (
        exposure <= -0.005
        and equal > -0.005
        and result["population_covariance"] < 0
        and one >= 0
        and multi < 0
    ):
        return "frequency_concentrated_policy_gain"
    if -0.005 < equal < 0 and one < 0 and multi < 0:
        return "broad_small_policy_gain"
    return "mixed_policy_response"
