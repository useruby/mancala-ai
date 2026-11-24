"""Independent read-only verifier for the Seed 633 exact-oracle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("docs/data/seed633-teacher-target-learning/attempt-0005")
SOURCE_ANCHORS = {
    "freeze.json": "bedc7d538c9d613afc543f38dd2c1dcf432c293c56aba2c4c637c6088521eb70",
    "cohort.jsonl": "303e6c38feff9868133e84a80b92421e26da480dbea36dd3a37a8f70d33e914d",
    "per-state-metrics.jsonl": "ebba7af50e154ff6e31e3afce3ee325da915295ac11f8a10a73f7a516dd2cccb",
    "summary.json": "b9301d5bfcb7d80cdea8b311247637a8ef21c33820b0eedfe3776a2d3b2ccf29",
    "independent-verification.json": "34aa59dea2547c81aba439aeaef2fb8e8dc56d307f2dbda686d2fb2b45aba639",
}
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 633


def digest(raw: bytes) -> str:
    """Return SHA-256 of evidence bytes."""
    return hashlib.sha256(raw).hexdigest()


def _independent_category(
    ce_delta: float, regret_delta: float, accuracy_delta: int, mass_delta: float
) -> str:
    """Classify directional changes without calling producer analysis code."""
    if ce_delta < -1e-6:
        imitation = "improved"
    elif ce_delta > 1e-6:
        imitation = "worse"
    else:
        imitation = "neutral"
    if regret_delta < 0:
        quality = "improved"
    elif regret_delta > 0:
        quality = "worse"
    elif accuracy_delta > 0:
        quality = "improved"
    elif accuracy_delta < 0:
        quality = "worse"
    elif mass_delta > 0.005:
        quality = "improved"
    elif mass_delta < -0.005:
        quality = "worse"
    else:
        quality = "neutral"
    return f"{imitation}_imitation_and_{quality}_exact_quality"


def _cluster_interval(
    values: list[dict[str, Any]], metric: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Rebuild a paired, source-opening resampled interval independently."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in values:
        buckets[row["source_opening_group"]].append(
            float(row["S615_minus_B512"][metric])
        )
    keys = sorted(buckets)
    if len(keys) < 2:
        return {"clusters": len(keys), "mean": None, "interval": None}
    sums_per_cluster = np.asarray([sum(buckets[key]) for key in keys])
    sizes_per_cluster = np.asarray([len(buckets[key]) for key in keys])
    draws = rng.integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))
    draw_means = sums_per_cluster[draws].sum(axis=1) / sizes_per_cluster[draws].sum(
        axis=1
    )
    flat = [value for key in keys for value in buckets[key]]
    return {
        "clusters": len(keys),
        "mean": float(np.mean(flat)),
        "interval": [
            float(value)
            for value in np.quantile(draw_means, [0.025, 0.975], method="linear")
        ],
    }


def _independent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute subgroup means, correlations, category counts, and CIs independently."""
    strata: dict[str, dict[str, list[dict[str, Any]]]] = {
        key: defaultdict(list)
        for key in (
            "membership_class",
            "phase",
            "legal_action_count",
            "optimality",
        )
    }

    for row in rows:
        strata["membership_class"][row["membership_class"]].append(row)
        strata["phase"][row["phase"]].append(row)
        strata["legal_action_count"][str(row["legal_action_count"])].append(row)
        label = "unique" if row["optimal_action_count"] == 1 else "tied"
        strata["optimality"][label].append(row)
    entropies = np.asarray([row["teacher_entropy_nats"] for row in rows])
    thresholds = np.quantile(entropies, [0.25, 0.5, 0.75], method="linear")
    for index, row in enumerate(rows):
        label = str(
            min(int(np.searchsorted(thresholds, entropies[index], side="right")) + 1, 4)
        )
        strata.setdefault("teacher_entropy_quartile", defaultdict(list))[label].append(
            row
        )
    metric_names = (
        "teacher_cross_entropy_nats",
        "exact_optimal_action_mass",
        "exact_optimal_top_action",
        "top_policy_action_regret",
    )
    summaries = {}
    for dimension, groups in strata.items():
        summaries[dimension] = {}
        for group, sample in sorted(groups.items()):
            summaries[dimension][group] = {
                "states": len(sample),
                "source_opening_clusters": len(
                    {row["source_opening_group"] for row in sample}
                ),
                "metrics": {
                    name: {
                        "B512_mean": float(
                            np.mean([row["B512"][name] for row in sample])
                        ),
                        "S615_mean": float(
                            np.mean([row["S615"][name] for row in sample])
                        ),
                        "S615_minus_B512": float(
                            np.mean([row["S615_minus_B512"][name] for row in sample])
                        ),
                    }
                    for name in metric_names
                },
                "category_counts": dict(
                    sorted(Counter(row["category"] for row in sample).items())
                ),
            }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    interval_metrics = {
        name: _cluster_interval(rows, name, rng) for name in metric_names
    }
    held_out = [
        row for row in rows if row["membership_class"] == "neither_training_membership"
    ]
    heldout_intervals = {
        name: _cluster_interval(held_out, name, rng) for name in metric_names
    }
    imitation_delta = np.asarray(
        [row["S615_minus_B512"]["teacher_cross_entropy_nats"] for row in rows]
    )
    relationships = {}
    for name in metric_names[1:]:
        quality_delta = np.asarray(
            [row["S615_minus_B512"][name] for row in rows], dtype=np.float64
        )
        relationships[name] = {
            "pearson_r": float(np.corrcoef(imitation_delta, quality_delta)[0, 1])
            if np.std(imitation_delta) and np.std(quality_delta)
            else None,
            "n": len(rows),
            "interpretation": "descriptive paired-state association; no causal or independent-state inference",
        }
    return {
        "exact_oracle_states": len(rows),
        "source_opening_groups": len({row["source_opening_group"] for row in rows}),
        "by_subgroup": summaries,
        "overall_cluster_bootstrap": interval_metrics,
        "protected_holdout_cluster_bootstrap": heldout_intervals,
        "category_counts": dict(
            sorted(Counter(row["category"] for row in rows).items())
        ),
        "teacher_ce_delta_association_with_exact_quality_delta": relationships,
    }


def _assert_summary_equivalent(
    observed: Any, expected: Any, path: str = "summary"
) -> None:
    """Compare independently rebuilt summaries, tolerating roundoff only."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or observed.keys() != expected.keys():
            raise ValueError(f"summary_mismatch:{path}")
        for key in expected:
            _assert_summary_equivalent(observed[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"summary_mismatch:{path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _assert_summary_equivalent(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        if not isinstance(observed, (int, float)) or not np.isclose(
            observed, expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"summary_mismatch:{path}")
    elif observed != expected:
        raise ValueError(f"summary_mismatch:{path}")


def verify(source: Path, output: Path) -> dict[str, Any]:
    """Reconstruct exact-oracle rows and summaries, verifying every evidence hash."""
    freeze_raw = (output / "freeze.json").read_bytes()
    freeze = json.loads(freeze_raw)
    if freeze.get("schema") != "seed633-exact-oracle-decision-protocol-record-v2":
        raise ValueError("diagnostic_freeze_schema_invalid")
    if freeze.get("status") != "retrospective_reproduction":
        raise ValueError("diagnostic_protocol_timing_invalid")
    for relative, expected in freeze["analysis_source_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"analysis_source_identity_mismatch:{relative}")
    observed_anchors = {}
    for name, expected in SOURCE_ANCHORS.items():
        actual = digest((source / name).read_bytes())
        if actual != expected:
            raise ValueError(f"historical_source_anchor_mismatch:{name}")
        observed_anchors[name] = actual
    if freeze.get("historical_source_authentication_sha256") != observed_anchors:
        raise ValueError("historical_source_anchor_record_mismatch")
    historical_verification = json.loads(
        (source / "independent-verification.json").read_bytes()
    )
    if (
        historical_verification.get("status") != "verified"
        or historical_verification.get("cohort_rows") != 4521
        or historical_verification.get("independent_numpy_metric_reconstruction")
        is not True
        or historical_verification.get("aggregate_and_bootstrap_reconstruction")
        is not True
    ):
        raise ValueError("historical_source_verification_record_invalid")
    source_freeze_raw = (source / "freeze.json").read_bytes()
    cohort_raw = (source / "cohort.jsonl").read_bytes()
    metrics_raw = (source / "per-state-metrics.jsonl").read_bytes()
    if digest(source_freeze_raw) != freeze["source_freeze_sha256"]:
        raise ValueError("source_freeze_identity_mismatch")
    if (
        digest(cohort_raw) != freeze["cohort_sha256"]
        or digest(metrics_raw) != freeze["source_metrics_sha256"]
    ):
        raise ValueError("source_evidence_identity_mismatch")
    source_freeze = json.loads(source_freeze_raw)
    if freeze["checkpoint_sha256"] != source_freeze["checkpoint_sha256"]:
        raise ValueError("model_identity_mismatch")
    cohort = [json.loads(line) for line in cohort_raw.splitlines() if line]
    metrics = [json.loads(line) for line in metrics_raw.splitlines() if line]
    for row in cohort:
        actual = sorted(
            int(move) for move in KalahGame.from_state(row["state"]).possible_moves()
        )
        if actual != row["legal_moves"]:
            raise ValueError("legal_action_mask_mismatch")
        if row["teacher_type"] == "exact_root_tablebase":
            evidence = row["teacher_evidence"]
            margins = {
                int(action): int(value)
                for action, value in evidence["exact_action_margins"].items()
            }
            winners = sorted(
                action
                for action, value in margins.items()
                if value == max(margins.values())
            )
            if sorted(margins) != actual or winners != row["exact_optimal_actions"]:
                raise ValueError("oracle_label_mismatch")
    expected_rows = []
    for source_row, metric_row in zip(cohort, metrics, strict=True):
        if source_row["teacher_type"] != "exact_root_tablebase":
            continue
        if source_row["canonical_identity"] != metric_row["canonical_identity"]:
            raise ValueError("cohort_metric_identity_mismatch")
        margins = {
            int(action): int(value)
            for action, value in source_row["teacher_evidence"][
                "exact_action_margins"
            ].items()
        }
        legal = source_row["legal_moves"]
        optimal = sorted(
            action
            for action, value in margins.items()
            if value == max(margins.values())
        )
        if sorted(margins) != legal or optimal != source_row["exact_optimal_actions"]:
            raise ValueError("oracle_action_value_correspondence_invalid")
        model_values = {}
        for model in ("B512", "S615"):
            policy = np.asarray(metric_row[model]["legal_policy"], dtype=np.float64)
            action = int(np.argmax(policy))
            model_values[model] = {
                "teacher_cross_entropy_nats": metric_row[model][
                    "teacher_target_cross_entropy_nats"
                ],
                "exact_optimal_action_mass": float(policy[optimal].sum()),
                "exact_optimal_top_action": action in optimal,
                "top_policy_action": action,
                "top_policy_action_regret": max(margins.values()) - margins[action],
                "margin_scale": "authenticated root-player outcome points; regret is optimal margin minus chosen-action margin",
                "search_selected_action": None,
            }
        cross_entropy_delta = (
            model_values["S615"]["teacher_cross_entropy_nats"]
            - model_values["B512"]["teacher_cross_entropy_nats"]
        )
        mass_delta = (
            model_values["S615"]["exact_optimal_action_mass"]
            - model_values["B512"]["exact_optimal_action_mass"]
        )
        agreement_delta = int(model_values["S615"]["exact_optimal_top_action"]) - int(
            model_values["B512"]["exact_optimal_top_action"]
        )
        regret_delta = (
            model_values["S615"]["top_policy_action_regret"]
            - model_values["B512"]["top_policy_action_regret"]
        )
        category = _independent_category(
            cross_entropy_delta, regret_delta, agreement_delta, mass_delta
        )
        expected_rows.append(
            {
                "canonical_identity": source_row["canonical_identity"],
                "source_opening_group": source_row["source_opening_group"],
                "membership_class": source_row["membership_class"],
                "phase": source_row["phase"],
                "teacher_entropy_nats": metric_row["teacher_entropy_nats"],
                "legal_action_count": len(legal),
                "optimal_action_count": len(optimal),
                "optimal_actions": optimal,
                "legal_actions": legal,
                "exact_action_margins": {
                    str(action): margins[action] for action in sorted(margins)
                },
                "B512": model_values["B512"],
                "S615": model_values["S615"],
                "S615_minus_B512": {
                    "teacher_cross_entropy_nats": cross_entropy_delta,
                    "exact_optimal_action_mass": mass_delta,
                    "exact_optimal_top_action": agreement_delta,
                    "top_policy_action_regret": regret_delta,
                },
                "category": category,
            }
        )
    results_raw = (output / "per-state-results.jsonl").read_bytes()
    actual_rows = [json.loads(line) for line in results_raw.splitlines() if line]
    if actual_rows != expected_rows:
        raise ValueError("per_state_analysis_mismatch")
    summary = json.loads((output / "summary.json").read_bytes())
    if digest(results_raw) != summary.get("results_sha256"):
        raise ValueError("results_digest_mismatch")
    if digest(freeze_raw) != summary.get("freeze_sha256"):
        raise ValueError("analysis_freeze_digest_mismatch")
    expected_summary = _independent_summary(expected_rows)
    for field, value in expected_summary.items():
        _assert_summary_equivalent(summary.get(field), value, field)
    return {
        "status": "verified",
        "exact_oracle_states": len(expected_rows),
        "source_opening_clusters": len(
            {row["source_opening_group"] for row in expected_rows}
        ),
        "oracle_action_sets_and_values_verified": True,
        "legal_action_masks_verified": True,
        "model_identities_bound_to_source_freeze": True,
        "historical_source_authentication_anchor_verified": True,
        "all_per_state_and_subgroup_metrics_reconstructed": True,
        "seed624_crosswalk_assumed": False,
        "search_selected_actions_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/data/seed633-exact-oracle-decision-diagnostic",
    )
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = verify(args.source, args.output)
    if args.write_report:
        target = args.output / "independent-verification.json"
        if target.exists():
            raise FileExistsError("verification_report_create_only")
        target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
