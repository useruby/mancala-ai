"""Pure source decomposition of published seed427 policy-loss differences."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

SOURCES = (
    "fresh",
    "generic_bootstrap",
    "opening_disagreement",
    "random_teacher",
    "stability",
)
COHORTS = ("unseen/>32", "seen/>32", "unseen/17-32", "unseen/<=16")


def decompose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute source contributions with global cohort and identity denominators."""
    output: dict[str, Any] = {}
    for cohort in COHORTS:
        subset, bucket = cohort.split("/")
        selected = [
            row
            for row in rows
            if row["membership"]["subset"] == subset
            and (
                (bucket == ">32" and row["membership"]["active_stones"] > 32)
                or (
                    bucket == "17-32" and 17 <= row["membership"]["active_stones"] <= 32
                )
                or (bucket == "<=16" and row["membership"]["active_stones"] <= 16)
            )
        ]
        weight_total = sum(row["losses"]["policy_weight"] for row in selected)
        identity_totals: dict[str, float] = defaultdict(float)
        by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
        identity_sources: dict[str, set[str]] = defaultdict(set)
        for row in selected:
            identity = row["membership"]["canonical_identity"]
            weight = row["losses"]["policy_weight"]
            identity_totals[identity] += weight
            source = row["membership"]["source"]
            if source not in by_source:
                raise ValueError(f"unregistered_source:{source}")
            by_source[source].append(row)
            identity_sources[identity].add(source)
        identities = len(identity_totals)
        identity_delta_numerators: dict[str, float] = defaultdict(float)
        for row in selected:
            identity_delta_numerators[row["membership"]["canonical_identity"]] += (
                row["e4_policy_loss"] - row["initializer_policy_loss"]
            ) * row["losses"]["policy_weight"]
        identity_deltas = {
            identity: identity_delta_numerators[identity] / denominator
            for identity, denominator in identity_totals.items()
        }
        positive_identity_total = sum(
            value for value in identity_deltas.values() if value > 0
        )
        largest_positive_identity = max(identity_deltas.values(), default=0.0)
        largest_absolute_identity = max(
            identity_deltas.items(), key=lambda item: abs(item[1]), default=(None, 0.0)
        )
        entries = {}
        for source, source_rows in by_source.items():
            weighted_delta = sum(
                (row["e4_policy_loss"] - row["initializer_policy_loss"])
                * row["losses"]["policy_weight"]
                for row in source_rows
            )
            identity_numerators: dict[str, float] = defaultdict(float)
            for row in source_rows:
                identity = row["membership"]["canonical_identity"]
                identity_numerators[identity] += (
                    row["e4_policy_loss"] - row["initializer_policy_loss"]
                ) * row["losses"]["policy_weight"]
            equal_contribution = (
                sum(
                    identity_numerators[identity] / denominator
                    for identity, denominator in identity_totals.items()
                )
                / identities
                if identities
                else 0.0
            )
            entries[source] = {
                "weighted_positions": len(source_rows),
                "canonical_identities": len(
                    {r["membership"]["canonical_identity"] for r in source_rows}
                ),
                "initializer_policy_loss": sum(
                    r["initializer_policy_loss"] * r["losses"]["policy_weight"]
                    for r in source_rows
                )
                / sum(r["losses"]["policy_weight"] for r in source_rows)
                if source_rows
                else None,
                "e4_policy_loss": sum(
                    r["e4_policy_loss"] * r["losses"]["policy_weight"]
                    for r in source_rows
                )
                / sum(r["losses"]["policy_weight"] for r in source_rows)
                if source_rows
                else None,
                "policy_loss_difference": (
                    sum(
                        (r["e4_policy_loss"] - r["initializer_policy_loss"])
                        * r["losses"]["policy_weight"]
                        for r in source_rows
                    )
                    / sum(r["losses"]["policy_weight"] for r in source_rows)
                )
                if source_rows
                else None,
                "exposure_contribution": weighted_delta / weight_total
                if weight_total
                else 0.0,
                "equal_identity_contribution": equal_contribution,
            }
        overlap = {
            f"{left}|{right}": len(
                {
                    identity
                    for identity, memberships in identity_sources.items()
                    if left in memberships and right in memberships
                }
            )
            for index, left in enumerate(SOURCES)
            for right in SOURCES[index + 1 :]
        }
        concentration = {}
        for aggregation in ("exposure_contribution", "equal_identity_contribution"):
            positives = [max(0.0, entries[source][aggregation]) for source in SOURCES]
            total_positive = sum(positives)
            largest = max(positives, default=0.0)
            concentration[aggregation] = {
                "classification": "concentrated"
                if total_positive and largest / total_positive >= 0.75
                else "no single-source concentration",
                "largest_positive_share": largest / total_positive
                if total_positive
                else 0.0,
                "positive_contribution_total": total_positive,
            }
        output[cohort] = {
            "weighted_positions": len(selected),
            "canonical_identities": identities,
            "global_policy_weight_denominator": weight_total,
            "source_contributions": entries,
            "pairwise_shared_identity_counts": overlap,
            "identities_in_multiple_sources": sum(
                len(sources) > 1 for sources in identity_sources.values()
            ),
            "identity_concentration": {
                "positive_delta_total": positive_identity_total,
                "largest_positive_identity_share": (
                    largest_positive_identity / positive_identity_total
                    if positive_identity_total
                    else 0.0
                ),
                "largest_absolute_identity": {
                    "canonical_identity": largest_absolute_identity[0],
                    "equal_identity_delta": largest_absolute_identity[1],
                },
            },
            "concentration": concentration,
        }
    return output
