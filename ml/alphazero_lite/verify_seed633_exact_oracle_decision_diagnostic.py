"""Independent read-only verifier for the Seed 633 exact-oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed633_exact_oracle_decision_diagnostic import (
    SOURCE,
    digest,
    summarize,
)

ROOT = Path(__file__).resolve().parents[2]


def verify(source: Path, output: Path) -> dict[str, Any]:
    """Reconstruct exact-oracle rows and summaries, verifying every evidence hash."""
    freeze_raw = (output / "freeze.json").read_bytes()
    freeze = json.loads(freeze_raw)
    if freeze.get("schema") != "seed633-exact-oracle-decision-freeze-v1":
        raise ValueError("diagnostic_freeze_schema_invalid")
    for relative, expected in freeze["analysis_source_sha256"].items():
        if digest((ROOT / relative).read_bytes()) != expected:
            raise ValueError(f"analysis_source_identity_mismatch:{relative}")
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
        quality = -1 if regret_delta > 0 else 1 if regret_delta < 0 else agreement_delta
        if quality == 0:
            quality = 1 if mass_delta > 0.005 else -1 if mass_delta < -0.005 else 0
        imitation_better = cross_entropy_delta < -1e-6
        category = (
            "improved_imitation_and_exact_quality"
            if imitation_better and quality > 0
            else "improved_imitation_worse_exact_quality"
            if imitation_better and quality < 0
            else "worse_imitation_improved_exact_quality"
            if cross_entropy_delta > 1e-6 and quality > 0
            else "no_meaningful_exact_quality_change"
            if quality == 0
            else "worse_imitation_and_worse_exact_quality"
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
    expected_summary = summarize(expected_rows)
    for field, value in expected_summary.items():
        if summary.get(field) != value:
            raise ValueError(f"summary_mismatch:{field}")
    return {
        "status": "verified",
        "exact_oracle_states": len(expected_rows),
        "source_opening_clusters": len(
            {row["source_opening_group"] for row in expected_rows}
        ),
        "oracle_action_sets_and_values_verified": True,
        "legal_action_masks_verified": True,
        "model_identities_bound_to_source_freeze": True,
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
