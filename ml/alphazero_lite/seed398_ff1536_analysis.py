"""Pure analysis and provenance-matrix construction for the FF1536 diagnostic."""

from __future__ import annotations

import random
from typing import Any


def calculate(
    registration: dict[str, Any],
    new_rows: list[dict[str, Any]],
    seed455_baseline: list[dict[str, Any]],
    e4_baseline: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return analysis and 128 case records without mutating any input."""
    plan = registration["reuse_plan"]
    new_by_case: dict[tuple[int, str], dict[str, Any]] = {}
    for row in new_rows:
        key = row["opening_index"], row["reference"]
        if key in new_by_case:
            raise ValueError("duplicate_new_case")
        new_by_case[key] = row
    baseline = {
        "seed455": _index(seed455_baseline),
        "original_O0_E4": _index(e4_baseline),
    }
    cases: list[dict[str, Any]] = []
    gains: dict[tuple[int, str], float] = {}
    for item in plan:
        key = item["opening_index"], item["reference"]
        if item["source"] == "reused":
            outcome = item["outcome"]
        else:
            if key not in new_by_case:
                raise ValueError("missing_new_case")
            outcome = new_by_case[key]["outcome"]
        if outcome["trajectory"][0]["action_relative"] != item["forced_action"]:
            raise ValueError("forced_action_mismatch")
        ref = item["reference"]
        ff = baseline[ref].get((item["opening_index"], "FF", 1536))
        ss = baseline[ref].get((item["opening_index"], "SS", 1536))
        if ff is None or ss is None:
            raise ValueError("missing_baseline_case")
        gain = outcome["score"] - ff["score"]
        gains[key] = gain
        cases.append(
            {
                "opening_index": item["opening_index"],
                "state_hash": item["state_hash"],
                "reference": ref,
                "forced_action": item["forced_action"],
                "source": item["source"],
                "source_ledger": item.get("source_ledger"),
                "source_ledger_sha256": item.get("source_ledger_sha256"),
                "source_case_identity": item.get("source_case_identity"),
                "score": outcome["score"],
                "store_margin_root_perspective": outcome[
                    "store_margin_root_perspective"
                ],
                "trajectory": outcome["trajectory"],
                "gain_vs_ff384": gain,
                "store_margin_change_vs_ff384": outcome["store_margin_root_perspective"]
                - ff["store_margin_root_perspective"],
                "score_difference_vs_ss384": outcome["score"] - ss["score"],
                "store_margin_difference_vs_ss384": outcome[
                    "store_margin_root_perspective"
                ]
                - ss["store_margin_root_perspective"],
            }
        )
    expected = {
        (s["opening_index"], r)
        for s in registration["action_mapping"]
        for r in baseline
    }
    if (
        len(cases) != 128
        or {(c["opening_index"], c["reference"]) for c in cases} != expected
    ):
        raise ValueError("provenance_matrix_coverage_mismatch")
    if len(new_by_case) != 8 or set(new_by_case) != {
        (item["opening_index"], item["reference"])
        for item in plan
        if item["source"] == "new"
    }:
        raise ValueError("new_case_set_mismatch")

    state_rows = []
    by_state: dict[int, dict[str, float]] = {}
    for state in registration["action_mapping"]:
        index = state["opening_index"]
        per_reference = {reference: gains[(index, reference)] for reference in baseline}
        by_state[index] = per_reference
        state_rows.append(
            {
                "opening_index": index,
                "state_hash": state["state_hash"],
                "reference_gains": per_reference,
                "paired_gain": sum(per_reference.values()) / 2,
            }
        )
    paired = [row["paired_gain"] for row in state_rows]
    rng = random.Random(411)
    boot = sorted(
        sum(paired[rng.randrange(64)] for _ in range(64)) / 64 for _ in range(10000)
    )
    mean = sum(paired) / 64
    lower = boot[249]
    ref_means = {
        reference: sum(row[reference] for row in by_state.values()) / 64
        for reference in baseline
    }
    report = {
        "schema": "seed398-ff1536-confirmation-analysis-v1",
        "exploratory": True,
        "primary": {
            "mean_gain": mean,
            "bootstrap_95_interval": [lower, boot[9749]],
            "resamples": 10000,
            "seed": 411,
            "units": 64,
        },
        "reference_mean_gains": ref_means,
        "state_results": state_rows,
        "reference_store_margin_change_means_vs_ff384": {
            reference: sum(
                row["store_margin_change_vs_ff384"]
                for row in cases
                if row["reference"] == reference
            )
            / 64
            for reference in baseline
        },
        "reference_mean_score_difference_vs_ss384": {
            reference: sum(
                row["score_difference_vs_ss384"]
                for row in cases
                if row["reference"] == reference
            )
            / 64
            for reference in baseline
        },
        "reference_mean_store_margin_difference_vs_ss384": {
            reference: sum(
                row["store_margin_difference_vs_ss384"]
                for row in cases
                if row["reference"] == reference
            )
            / 64
            for reference in baseline
        },
        "decision": "nominate_fresh_confirmation_experiment"
        if mean >= 0.03
        and lower > 0
        and all(value >= 0 for value in ref_means.values())
        else "no_supported_budget_follow_up",
        "interpretation": "Exploratory retrospective comparison using previously observed outcomes. No promotion or overall-strength evidence.",
    }
    return report, cases


def _index(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    result = {}
    for row in rows:
        key = row["opening_index"], row["action"], row["budget"]
        if key in result:
            raise ValueError("duplicate_baseline_case")
        result[key] = row
    if len(result) != 256:
        raise ValueError("baseline_case_count_mismatch")
    return result
