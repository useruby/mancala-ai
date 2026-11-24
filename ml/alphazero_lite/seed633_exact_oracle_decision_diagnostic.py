"""Retrospective paired exact-oracle analysis of authenticated Seed 633 policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("docs/data/seed633-teacher-target-learning/attempt-0005")
OUTPUT = Path("docs/data/seed633-exact-oracle-decision-diagnostic")
REPLICATES = 20_000
SEED = 633
CE_TOLERANCE = 1e-6
MASS_TOLERANCE = 0.005
REGRET_TOLERANCE = 0.0


def digest(raw: bytes) -> str:
    """Return SHA-256 for an evidence file."""
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    """Serialize a value with stable JSON formatting."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def derive(
    cohort: list[dict[str, Any]], metrics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive per-state exact quality and predeclared directional categories."""
    if len(cohort) != len(metrics):
        raise ValueError("cohort_metric_count_mismatch")
    result = []
    for source, row in zip(cohort, metrics, strict=True):
        if source["canonical_identity"] != row["canonical_identity"]:
            raise ValueError("cohort_metric_identity_mismatch")
        if source["teacher_type"] != "exact_root_tablebase":
            continue
        evidence = source["teacher_evidence"]
        legal = source["legal_moves"]
        optimal = source["exact_optimal_actions"]
        margins = {
            int(action): int(value)
            for action, value in evidence["exact_action_margins"].items()
        }
        if sorted(margins) != legal or sorted(optimal) != sorted(
            action
            for action, value in margins.items()
            if value == max(margins.values())
        ):
            raise ValueError("oracle_action_value_correspondence_invalid")
        per_model = {}
        for model in ("B512", "S615"):
            policy = np.asarray(row[model]["legal_policy"], dtype=np.float64)
            if policy.shape != (6,) or any(policy[action] <= 0 for action in legal):
                raise ValueError("legal_policy_invalid")
            top = int(np.argmax(policy))
            per_model[model] = {
                "teacher_cross_entropy_nats": row[model][
                    "teacher_target_cross_entropy_nats"
                ],
                "exact_optimal_action_mass": float(policy[optimal].sum()),
                "exact_optimal_top_action": top in optimal,
                "top_policy_action": top,
                "top_policy_action_regret": int(max(margins.values()) - margins[top]),
                "margin_scale": "authenticated root-player outcome points; regret is optimal margin minus chosen-action margin",
                "search_selected_action": None,
            }
        ce_delta = (
            per_model["S615"]["teacher_cross_entropy_nats"]
            - per_model["B512"]["teacher_cross_entropy_nats"]
        )
        mass_delta = (
            per_model["S615"]["exact_optimal_action_mass"]
            - per_model["B512"]["exact_optimal_action_mass"]
        )
        agreement_delta = int(per_model["S615"]["exact_optimal_top_action"]) - int(
            per_model["B512"]["exact_optimal_top_action"]
        )
        regret_delta = (
            per_model["S615"]["top_policy_action_regret"]
            - per_model["B512"]["top_policy_action_regret"]
        )
        imitation_better = ce_delta < -CE_TOLERANCE
        quality_direction = (
            -1
            if regret_delta > REGRET_TOLERANCE
            else 1
            if regret_delta < -REGRET_TOLERANCE
            else agreement_delta
        )
        if quality_direction == 0:
            quality_direction = (
                1
                if mass_delta > MASS_TOLERANCE
                else -1
                if mass_delta < -MASS_TOLERANCE
                else 0
            )
        quality_better = quality_direction > 0
        quality_worse = quality_direction < 0
        category = (
            "improved_imitation_and_exact_quality"
            if imitation_better and quality_better
            else "improved_imitation_worse_exact_quality"
            if imitation_better and quality_worse
            else "worse_imitation_improved_exact_quality"
            if ce_delta > CE_TOLERANCE and quality_better
            else "no_meaningful_exact_quality_change"
            if not quality_better and not quality_worse
            else "worse_imitation_and_worse_exact_quality"
        )
        result.append(
            {
                "canonical_identity": source["canonical_identity"],
                "source_opening_group": source["source_opening_group"],
                "membership_class": source["membership_class"],
                "phase": source["phase"],
                "teacher_entropy_nats": row["teacher_entropy_nats"],
                "legal_action_count": len(legal),
                "optimal_action_count": len(optimal),
                "optimal_actions": optimal,
                "legal_actions": legal,
                "exact_action_margins": {str(k): margins[k] for k in sorted(margins)},
                "B512": per_model["B512"],
                "S615": per_model["S615"],
                "S615_minus_B512": {
                    "teacher_cross_entropy_nats": ce_delta,
                    "exact_optimal_action_mass": mass_delta,
                    "exact_optimal_top_action": agreement_delta,
                    "top_policy_action_regret": regret_delta,
                },
                "category": category,
            }
        )
    return result


def bootstrap(
    rows: list[dict[str, Any]], field: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Compute paired source-opening-cluster bootstrap interval for a delta."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["source_opening_group"]].append(float(row["S615_minus_B512"][field]))
    names = sorted(groups)
    if len(names) < 2:
        return {"clusters": len(names), "mean": None, "interval": None}
    values = [np.asarray(groups[name], dtype=np.float64) for name in names]
    sample = rng.integers(0, len(names), size=(REPLICATES, len(names)))
    sums = np.asarray([value.sum() for value in values])[sample].sum(axis=1)
    counts = np.asarray([len(value) for value in values])[sample].sum(axis=1)
    return {
        "clusters": len(names),
        "mean": float(np.mean([value for group in values for value in group])),
        "interval": [
            float(x)
            for x in np.quantile(sums / counts, [0.025, 0.975], method="linear")
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exact-oracle paired metrics across fixed cohort strata."""
    dimensions: dict[str, dict[str, list[dict[str, Any]]]] = {
        "membership_class": defaultdict(list),
        "phase": defaultdict(list),
        "legal_action_count": defaultdict(list),
        "optimality": defaultdict(list),
    }
    for row in rows:
        dimensions["membership_class"][row["membership_class"]].append(row)
        dimensions["phase"][row["phase"]].append(row)
        dimensions["legal_action_count"][str(row["legal_action_count"])].append(row)
        dimensions["optimality"][
            "unique" if row["optimal_action_count"] == 1 else "tied"
        ].append(row)
    entropy = np.asarray([row["teacher_entropy_nats"] for row in rows])
    cuts = np.quantile(entropy, [0.25, 0.5, 0.75], method="linear")
    for row in rows:
        group = str(
            min(
                int(np.searchsorted(cuts, row["teacher_entropy_nats"], side="right"))
                + 1,
                4,
            )
        )
        dimensions.setdefault("teacher_entropy_quartile", defaultdict(list))[
            group
        ].append(row)
    rng = np.random.default_rng(SEED)
    result = {}
    for dimension, groups in dimensions.items():
        result[dimension] = {}
        for name, subset in sorted(groups.items()):
            result[dimension][name] = {
                "states": len(subset),
                "source_opening_clusters": len(
                    {row["source_opening_group"] for row in subset}
                ),
                "metrics": {
                    field: {
                        "B512_mean": float(
                            np.mean([row["B512"][field] for row in subset])
                        ),
                        "S615_mean": float(
                            np.mean([row["S615"][field] for row in subset])
                        ),
                        "S615_minus_B512": float(
                            np.mean([row["S615_minus_B512"][field] for row in subset])
                        ),
                    }
                    for field in (
                        "teacher_cross_entropy_nats",
                        "exact_optimal_action_mass",
                        "exact_optimal_top_action",
                        "top_policy_action_regret",
                    )
                },
                "category_counts": dict(
                    sorted(
                        __import__("collections")
                        .Counter(row["category"] for row in subset)
                        .items()
                    )
                ),
            }
    ce_delta = np.asarray(
        [row["S615_minus_B512"]["teacher_cross_entropy_nats"] for row in rows]
    )
    association = {}
    for field in (
        "exact_optimal_action_mass",
        "exact_optimal_top_action",
        "top_policy_action_regret",
    ):
        quality_delta = np.asarray(
            [row["S615_minus_B512"][field] for row in rows], dtype=np.float64
        )
        association[field] = {
            "pearson_r": float(np.corrcoef(ce_delta, quality_delta)[0, 1])
            if np.std(ce_delta) > 0 and np.std(quality_delta) > 0
            else None,
            "n": len(rows),
            "interpretation": "descriptive paired-state association; no causal or independent-state inference",
        }
    return {
        "exact_oracle_states": len(rows),
        "source_opening_groups": len({row["source_opening_group"] for row in rows}),
        "by_subgroup": result,
        "overall_cluster_bootstrap": {
            field: bootstrap(rows, field, rng)
            for field in (
                "teacher_cross_entropy_nats",
                "exact_optimal_action_mass",
                "exact_optimal_top_action",
                "top_policy_action_regret",
            )
        },
        "protected_holdout_cluster_bootstrap": {
            field: bootstrap(
                [
                    row
                    for row in rows
                    if row["membership_class"] == "neither_training_membership"
                ],
                field,
                rng,
            )
            for field in (
                "teacher_cross_entropy_nats",
                "exact_optimal_action_mass",
                "exact_optimal_top_action",
                "top_policy_action_regret",
            )
        },
        "category_counts": dict(
            sorted(
                __import__("collections")
                .Counter(row["category"] for row in rows)
                .items()
            )
        ),
        "teacher_ce_delta_association_with_exact_quality_delta": association,
    }


def run(source: Path, output: Path) -> dict[str, Any]:
    """Create a frozen, hash-bound diagnostic output directory."""
    if output.exists():
        raise FileExistsError("diagnostic_output_create_only")
    freeze = json.loads((source / "freeze.json").read_bytes())
    if freeze.get("status") != "frozen_before_inference":
        raise ValueError("source_freeze_not_authenticated")
    cohort_raw = (source / "cohort.jsonl").read_bytes()
    metric_raw = (source / "per-state-metrics.jsonl").read_bytes()
    cohort = [json.loads(line) for line in cohort_raw.splitlines() if line]
    metrics = [json.loads(line) for line in metric_raw.splitlines() if line]
    rows = derive(cohort, metrics)
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    plan = {
        "schema": "seed633-exact-oracle-decision-freeze-v1",
        "status": "frozen_before_analysis",
        "analysis_source_sha256": {
            "ml/alphazero_lite/seed633_exact_oracle_decision_diagnostic.py": digest(
                Path(__file__).read_bytes()
            ),
            "ml/alphazero_lite/verify_seed633_exact_oracle_decision_diagnostic.py": digest(
                (
                    ROOT
                    / "ml/alphazero_lite/verify_seed633_exact_oracle_decision_diagnostic.py"
                ).read_bytes()
            ),
        },
        "source_freeze_sha256": digest((source / "freeze.json").read_bytes()),
        "cohort_sha256": digest(cohort_raw),
        "source_metrics_sha256": digest(metric_raw),
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "selected_population": "all Seed 633 exact_root_tablebase states with authenticated outcome and legal action margins",
        "definitions": {
            "delta_direction": "S615 minus B512; negative CE is improved imitation; positive optimal mass and top-action accuracy are better; negative regret is better",
            "policy_decision": "legal-masked unsearched model argmax with NumPy argmax lowest-index tie break",
            "search_decisions": "not observed; no search is rerun",
            "optimal_set": "all actions tied at maximum authenticated exact_action_margin",
            "regret": "max authenticated action margin minus chosen action margin, outcome-point scale",
            "CE_tolerance_nats": CE_TOLERANCE,
            "optimal_mass_tolerance": MASS_TOLERANCE,
            "regret_tolerance_points": REGRET_TOLERANCE,
            "classification_rule": "exact quality improves/worsens if any of optimal mass exceeds tolerance, top action changes correctness, or regret changes strictly; buckets report joint directional categories",
            "bootstrap": {
                "unit": "source-opening group",
                "replicates": REPLICATES,
                "seed": SEED,
                "interval": [0.025, 0.975],
                "interpretation": "descriptive cluster stability only",
            },
            "source_opening_identity": "used only as authenticated historical cluster; no Seed 624 crosswalk assumed",
        },
    }
    summary = summarize(rows)
    output.mkdir(parents=True)
    (output / "freeze.json").write_text(
        json.dumps(plan, sort_keys=True, indent=2) + "\n"
    )
    (output / "per-state-results.jsonl").write_bytes(rows_raw)
    summary["freeze_sha256"] = digest((output / "freeze.json").read_bytes())
    summary["results_sha256"] = digest(rows_raw)
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
