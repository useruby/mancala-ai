#!/usr/bin/env python3
# ruff: noqa: E402
"""Correct PR #275 records from stored rows and the required limited repeat."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_generation3_turn_completion_puct import (
    BUDGET,
    REGISTRY,
    SEEDS,
    WEIGHTS_SHA,
    aggregate,
    classify,
    compare_deterministic_result,
    run_lane,
    score,
    sliced_metrics,
)

FULL_PATH = ROOT / "docs/data/alphazero-lite-generation3-turn-completion-puct-full.json"
SUMMARY_PATH = (
    ROOT / "docs/data/alphazero-lite-generation3-turn-completion-puct-summary.json"
)
REPORT_PATH = ROOT / "docs/alphazero-lite-generation3-turn-completion-puct-results.md"
PR274_COMMIT = "569cfebbc270b23bc2f7cf87996d0b0d5bd897a1"
PR274_ARTIFACTS = {
    "docs/alphazero-lite-generation3-implicit-minimax-puct-results.md": "97056d75001711d9e2e94464b4782b567021abfa68f2158f9004cc026f5e8db9",
    "docs/data/alphazero-lite-generation3-implicit-minimax-puct-full.json": "fcca76448ab599b1b120f0e956fad6253c904c493a63802974a77893105fcfe9",
    "docs/data/alphazero-lite-generation3-implicit-minimax-puct-summary.json": "f2f05005777be9b0ca44a7c64a7f9403bcb3edc7ca61d573ba4f572fcf20c644",
}


def condition(name: str, observed: object, threshold: object, passed: bool) -> dict:
    """Format one machine-readable preregistration gate."""
    return {
        "condition": name,
        "observed": observed,
        "threshold": threshold,
        "passed": passed,
    }


def stored_rows_are_consistent(payload: dict) -> bool:
    """Verify all stored scores still match their immutable references."""
    for row in payload["results"]:
        for lane in ("baseline", "candidate"):
            for result in row[lane].values():
                rescored = score(result, row["action_references"])
                if any(
                    rescored[key] != result[key]
                    for key in (
                        "selected_reference_value",
                        "regret",
                        "best_reference_action_agreement",
                        "top_two_agreement",
                        "catastrophic_miss",
                        "legal_selection_status",
                    )
                ):
                    return False
    return True


def repeat_rows(rows: list[dict]) -> dict:
    """Run exactly the preregistered 12-state, three-seed, two-lane repeat."""
    evaluator = ArtifactEvaluator(ROOT / "model-artifact/current")
    comparisons = []
    for row in rows[:12]:
        for seed in SEEDS:
            for lane, candidate in (("baseline", False), ("candidate", True)):
                repeated = score(
                    run_lane(
                        KalahGame.from_state(row["state"]), evaluator, seed, candidate
                    ),
                    row["action_references"],
                )
                comparison = compare_deterministic_result(
                    row[lane][str(seed)], repeated
                )
                comparisons.append(
                    {
                        "state_hash": row["state_hash"],
                        "seed": seed,
                        "lane": lane,
                    }
                    | comparison
                )
    return {
        "first_canonically_sorted_states": 12,
        "seeds": list(SEEDS),
        "lanes": ["baseline", "candidate"],
        "neural_budget": BUDGET,
        "comparisons": comparisons,
        "determinism_passed": all(
            comparison["equal_excluding_runtime"] for comparison in comparisons
        ),
    }


def gate_matrix(
    payload: dict,
    aggregate_metrics: dict,
    metrics_by_slice: dict,
    *,
    artifact_ok: bool,
    registry_ok: bool,
    repeat_ok: bool,
    stored_rows_ok: bool,
) -> list[dict]:
    """Record every qualification, invariant, and subset guard from the plan."""
    rows = payload["results"]
    budget_ok = all(
        item["evaluator_calls"] == BUDGET
        for row in rows
        for lane in ("baseline", "candidate")
        for item in row[lane].values()
    )
    legal_ok = all(
        item["legal_selection_status"]
        for row in rows
        for lane in ("baseline", "candidate")
        for item in row[lane].values()
    )
    candidate_invariants_ok = all(
        item.get("defensive_cap_events", 0) == 0
        and item.get("repeated_state_events", 0) == 0
        and item.get("incomplete_due_to_budget", 0) <= 1
        for row in rows
        for item in row["candidate"].values()
    )
    primary, full = aggregate_metrics["root_extra_turn"], aggregate_metrics["full"]
    gates = [
        condition(
            "artifact_identity",
            sha256_file(ROOT / "model-artifact/current/weights.json"),
            WEIGHTS_SHA,
            artifact_ok,
        ),
        condition(
            "registry_unchanged",
            sha256_file(REGISTRY),
            payload["consumed_suite_registry_sha256_before"],
            registry_ok,
        ),
        condition("stored_rows_match_references", stored_rows_ok, True, stored_rows_ok),
        condition("legal_selection", legal_ok, True, legal_ok),
        condition(
            "candidate_extension_invariants",
            candidate_invariants_ok,
            True,
            candidate_invariants_ok,
        ),
        condition("deterministic_repeat", repeat_ok, True, repeat_ok),
        condition(
            "budget_contract",
            budget_ok,
            f"{BUDGET} evaluator calls per lane",
            budget_ok,
        ),
        condition(
            "root_extra_turn_baseline_regret_positive",
            primary["baseline"]["mean_regret"],
            "> 0",
            primary["baseline"]["mean_regret"] > 0,
        ),
        condition(
            "root_extra_turn_regret_reduction",
            primary["candidate"]["mean_regret"] / primary["baseline"]["mean_regret"],
            "<= 0.75",
            primary["candidate"]["mean_regret"]
            <= 0.75 * primary["baseline"]["mean_regret"],
        ),
        condition(
            "root_extra_turn_bootstrap_upper",
            primary["paired_hierarchical_bootstrap"]["upper_95"],
            "< 0",
            primary["paired_hierarchical_bootstrap"]["upper_95"] < 0,
        ),
        condition(
            "root_extra_turn_agreement",
            primary["candidate"]["best_reference_action_agreement"]
            - primary["baseline"]["best_reference_action_agreement"],
            ">= 0",
            primary["candidate"]["best_reference_action_agreement"]
            >= primary["baseline"]["best_reference_action_agreement"],
        ),
        condition(
            "root_extra_turn_catastrophic_misses",
            primary["candidate"]["catastrophic_miss_rate"]
            - primary["baseline"]["catastrophic_miss_rate"],
            "<= 0",
            primary["candidate"]["catastrophic_miss_rate"]
            <= primary["baseline"]["catastrophic_miss_rate"],
        ),
        condition(
            "full_regret_reduction",
            full["candidate"]["mean_regret"] / full["baseline"]["mean_regret"],
            "<= 0.9",
            full["candidate"]["mean_regret"] <= 0.9 * full["baseline"]["mean_regret"],
        ),
        condition(
            "full_bootstrap_upper",
            full["paired_hierarchical_bootstrap"]["upper_95"],
            "< 0",
            full["paired_hierarchical_bootstrap"]["upper_95"] < 0,
        ),
        condition(
            "full_agreement",
            full["candidate"]["best_reference_action_agreement"]
            - full["baseline"]["best_reference_action_agreement"],
            ">= 0",
            full["candidate"]["best_reference_action_agreement"]
            >= full["baseline"]["best_reference_action_agreement"],
        ),
        condition(
            "full_catastrophic_misses",
            full["candidate"]["catastrophic_miss_rate"]
            - full["baseline"]["catastrophic_miss_rate"],
            "<= 0",
            full["candidate"]["catastrophic_miss_rate"]
            <= full["baseline"]["catastrophic_miss_rate"],
        ),
        condition(
            "full_runtime",
            full["candidate"]["p95_runtime_seconds"]
            / full["baseline"]["p95_runtime_seconds"],
            "<= 1.25",
            full["candidate"]["p95_runtime_seconds"]
            <= 1.25 * full["baseline"]["p95_runtime_seconds"],
        ),
    ]
    for field, groups in metrics_by_slice.items():
        for label, metrics in groups.items():
            eligible = metrics["n_distinct_states"] >= 8
            passes = not eligible or metrics["catastrophic_miss_rate_delta"] <= 0.02
            gates.append(
                condition(
                    f"subset_catastrophic_miss:{field}={label}",
                    {
                        "n_distinct_states": metrics["n_distinct_states"],
                        "catastrophic_miss_rate_delta": metrics[
                            "catastrophic_miss_rate_delta"
                        ],
                    },
                    "eligible n_distinct_states >= 8 requires delta <= 0.02",
                    passes,
                )
            )
    return gates


def corrected_payload(payload: dict) -> dict:
    """Build the corrected record without regenerating corpus or references."""
    if (
        payload["corpus_sha256"]
        != hashlib.sha256(
            json.dumps(
                payload["corpus"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    ):
        raise RuntimeError("stored corpus identity differs")
    if sha256_file(ROOT / "model-artifact/current/weights.json") != WEIGHTS_SHA:
        raise RuntimeError("frozen artifact differs")
    for path, expected_hash in PR274_ARTIFACTS.items():
        if sha256_file(ROOT / path) != expected_hash:
            raise RuntimeError(f"PR #274 artifact cannot be restored exactly: {path}")

    artifact_ok = (
        payload["artifact_sha256_before"]
        == payload["artifact_sha256_after"]
        == WEIGHTS_SHA
    )
    registry_hash = sha256_file(REGISTRY)
    registry_ok = (
        payload["consumed_suite_registry_sha256_before"]
        == payload["consumed_suite_registry_sha256_after"]
        == registry_hash
    )
    stored_rows_ok = stored_rows_are_consistent(payload)
    deterministic_repeat = repeat_rows(payload["results"])
    aggregate_metrics = aggregate(payload["results"])
    all_slices = sliced_metrics(payload["results"])
    gates = gate_matrix(
        payload,
        aggregate_metrics,
        all_slices,
        artifact_ok=artifact_ok,
        registry_ok=registry_ok,
        repeat_ok=deterministic_repeat["determinism_passed"],
        stored_rows_ok=stored_rows_ok,
    )
    invariants_ok = all(
        gate["passed"]
        for gate in gates
        if gate["condition"]
        in {
            "artifact_identity",
            "registry_unchanged",
            "stored_rows_match_references",
            "legal_selection",
            "candidate_extension_invariants",
            "deterministic_repeat",
        }
    )
    budget_ok = next(
        gate["passed"] for gate in gates if gate["condition"] == "budget_contract"
    )
    classification = classify(
        aggregate_metrics,
        all_slices,
        invariants_ok=invariants_ok,
        budget_ok=budget_ok,
    )
    return payload | {
        "schema_version": "turn_completion_puct_v2_correction",
        "classification": classification,
        "original_experiment_implementation_commit": payload["experiment_code_commit"],
        "corrective_implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "correction": {
            "previous_classification": payload["classification"],
            "corrected_classification": classification,
            "reason": "The original runner omitted complete sliced metrics and the preregistered regresses_subsets branch. The stored 64-state root-extra-turn group has a catastrophic-miss-rate delta of 0.0625, exceeding the strict 0.02 maximum.",
            "no_search_or_reference_result_changed": True,
            "pr274_merge_commit": PR274_COMMIT,
            "pr274_artifact_hashes": PR274_ARTIFACTS,
        },
        "aggregate_metrics": aggregate_metrics,
        "sliced_metrics": all_slices,
        "deterministic_repeat": deterministic_repeat,
        "gate_matrix": gates,
        "telemetry_completeness": {
            "policy_priors_collected": False,
            "exploration_values_collected": False,
            "telemetry_complete": False,
            "explanation": "policy_priors and exploration_values were not captured by the original PR #275 run; they are not fabricated by this correction.",
        },
        "invariants": payload["invariants"]
        | {
            "artifact_unchanged": artifact_ok,
            "registry_unchanged": registry_ok,
            "stored_rows_match_references": stored_rows_ok,
            "determinism_passed": deterministic_repeat["determinism_passed"],
            "invariants_ok": invariants_ok,
            "budget_ok": budget_ok,
        },
    }


def write_corrected_artifacts(payload: dict, output_root: Path = ROOT) -> None:
    """Write deterministic corrected artifacts to the requested repository root."""
    full = output_root / FULL_PATH.relative_to(ROOT)
    summary = output_root / SUMMARY_PATH.relative_to(ROOT)
    report = output_root / REPORT_PATH.relative_to(ROOT)
    full.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary.write_text(
        json.dumps(payload | {"results": "see full result"}, indent=2, sort_keys=True)
        + "\n"
    )
    report.write_text(
        "# Generation-3 Turn-Completion PUCT Results\n\n"
        f"**Previous classification:** `{payload['correction']['previous_classification']}`\n\n"
        f"**Corrected classification:** `{payload['classification']}`\n\n"
        "Turn completion regressed its intended root-extra-turn subset: its catastrophic-miss rate increased from 0.34375 to 0.40625, a delta of 0.0625. Full-corpus regret also increased. The candidate is not eligible for a 1,200-budget evaluation or arena, and the turn-completion lane is closed. No search or reference result was changed by this correction.\n\n"
        "The original PR #275 run did not capture policy priors or exploration values; this correction does not fabricate them.\n\n"
        "```json\n"
        + json.dumps(
            {
                key: payload[key]
                for key in (
                    "classification",
                    "correction",
                    "aggregate_metrics",
                    "sliced_metrics",
                    "deterministic_repeat",
                    "gate_matrix",
                    "telemetry_completeness",
                    "invariants",
                )
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n"
    )


def main() -> None:
    payload = json.loads(FULL_PATH.read_text())
    corrected = corrected_payload(payload)
    write_corrected_artifacts(corrected)
    print(corrected["classification"])


if __name__ == "__main__":
    main()
