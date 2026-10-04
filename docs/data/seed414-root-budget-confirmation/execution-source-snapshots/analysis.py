"""Pure paired analysis for the seed-414 root-budget confirmation."""

from __future__ import annotations

import random
from typing import Any

REFERENCES = ("seed455", "original_O0_E4")
ROOT_BUDGETS = (384, 1536)


def calculate(
    registration: dict[str, Any],
    probes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = registration["states"]
    probe_by_index = _unique(
        probes, lambda row: row["opening_index"], "duplicate_root_probe"
    )
    trajectory_by_key = _unique(
        trajectories,
        lambda row: (row["opening_index"], row["reference"], row["forced_action"]),
        "duplicate_physical_trajectory",
    )
    alias_by_key = _unique(
        aliases,
        lambda row: (row["opening_index"], row["reference"], row["root_budget"]),
        "duplicate_logical_alias",
    )
    if len(probe_by_index) != 128 or set(probe_by_index) != {
        state["opening_index"] for state in states
    }:
        raise ValueError("root_probe_coverage_mismatch")
    expected_aliases = {
        (state["opening_index"], ref, budget)
        for state in states
        for ref in REFERENCES
        for budget in ROOT_BUDGETS
    }
    if set(alias_by_key) != expected_aliases:
        raise ValueError("logical_case_coverage_mismatch")
    matrix: list[dict[str, Any]] = []
    paired_gains: list[float] = []
    gains_by_reference = {ref: [] for ref in REFERENCES}
    margins_by_reference = {ref: [] for ref in REFERENCES}
    changes = 0
    root_seconds: list[float] = []
    state_results: list[dict[str, Any]] = []

    for state in states:
        index = state["opening_index"]
        probe = probe_by_index[index]
        actions = {
            budget: int(probe["snapshots"][str(budget)]["action"])
            for budget in ROOT_BUDGETS
        }
        changes += actions[384] != actions[1536]
        root_seconds.append(float(probe["elapsed_wall_seconds"]))
        state_gains = []
        state_record: dict[str, Any] = {
            "opening_index": index,
            "state_hash": state["state_hash"],
            "actions": {str(key): value for key, value in actions.items()},
            "action_changed": actions[384] != actions[1536],
            "reference_results": {},
        }
        for ref in REFERENCES:
            results = {}
            for budget in ROOT_BUDGETS:
                alias = alias_by_key[(index, ref, budget)]
                action = actions[budget]
                if int(alias["action"]) != action:
                    raise ValueError("alias_action_does_not_match_probe")
                if alias.get("alias_of"):
                    source_budget = int(alias["alias_of"]["root_budget"])
                    if actions[source_budget] != action:
                        raise ValueError("invalid_alias_target_action")
                key = (index, ref, action)
                if key not in trajectory_by_key:
                    raise ValueError("missing_physical_continuation")
                outcome = trajectory_by_key[key]["outcome"]
                if outcome["state_hash"] != state["state_hash"]:
                    raise ValueError("continuation_state_identity_mismatch")
                results[budget] = outcome
            gain = float(results[1536]["score"]) - float(results[384]["score"])
            margin_change = int(results[1536]["store_margin_root_perspective"]) - int(
                results[384]["store_margin_root_perspective"]
            )
            gains_by_reference[ref].append(gain)
            margins_by_reference[ref].append(margin_change)
            state_gains.append(gain)
            state_record["reference_results"][ref] = {
                "gain": gain,
                "margin_change": margin_change,
                "scores": {str(b): results[b]["score"] for b in ROOT_BUDGETS},
                "margins": {
                    str(b): results[b]["store_margin_root_perspective"]
                    for b in ROOT_BUDGETS
                },
                "trajectory_identities": {
                    str(b): alias_by_key[(index, ref, b)]["trajectory_identity"]
                    for b in ROOT_BUDGETS
                },
            }
            matrix.extend(
                _logical_row(
                    state,
                    probe,
                    ref,
                    budget,
                    alias_by_key[(index, ref, budget)],
                    trajectory_by_key,
                )
                for budget in ROOT_BUDGETS
            )
        paired_gains.append(sum(state_gains) / len(REFERENCES))
        state_record["paired_gain"] = paired_gains[-1]
        state_record["zero_gain_observation"] = not state_record["action_changed"]
        # Keep compact state aggregates; the full reference-by-budget table is below.
        state_record.pop("reference_results")
        state_record["reference_gains"] = {
            ref: gains_by_reference[ref][-1] for ref in REFERENCES
        }
        matrix_state = state_record
        state_results.append(matrix_state)

    if len(matrix) != 512 or len(state_results) != 128:
        raise ValueError("paired_matrix_accounting_mismatch")
    randomizer = random.Random(414)
    boot = sorted(
        sum(paired_gains[randomizer.randrange(128)] for _ in range(128)) / 128
        for _ in range(10000)
    )
    mean = sum(paired_gains) / 128
    low, high = boot[249], boot[9749]
    ref_means = {ref: sum(values) / 128 for ref, values in gains_by_reference.items()}
    decision = (
        "confirmation_criteria_met"
        if mean >= 0.03 and low > 0 and all(value >= 0 for value in ref_means.values())
        else "confirmation_criteria_not_met"
    )
    report = {
        "schema": "seed414-root-budget-confirmation-analysis-v1",
        "decision": decision,
        "primary": {
            "metric": "per-state mean across reference-specific FF1536-minus-FF384 score gains",
            "mean_gain": mean,
            "bootstrap_95_interval": [low, high],
            "resamples": 10000,
            "seed": 414,
            "paired_units": 128,
        },
        "reference_mean_gains": ref_means,
        "reference_mean_margin_changes": {
            ref: sum(values) / 128 for ref, values in margins_by_reference.items()
        },
        "action_change_rate": changes / 128,
        "action_changes": changes,
        "search_timing": {
            "root_total_seconds": sum(root_seconds),
            "root_mean_seconds": sum(root_seconds) / len(root_seconds),
            "continuation_total_seconds": {
                ref: sum(
                    float(row["outcome"]["elapsed_wall_seconds"])
                    for row in trajectory_by_key.values()
                    if row["reference"] == ref
                )
                for ref in REFERENCES
            },
            "physical_continuations": len(trajectory_by_key),
        },
        "state_results": state_results,
        "interpretation": "First-action quality diagnostic under two frozen continuation references; not overall strength or promotion evidence.",
    }
    return report, matrix


def _logical_row(
    state: dict[str, Any],
    probe: dict[str, Any],
    ref: str,
    budget: int,
    alias: dict[str, Any],
    physical: dict[tuple[int, str, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (state["opening_index"], ref, int(alias["action"]))
    trajectory = physical[key]["outcome"]
    return {
        "opening_index": state["opening_index"],
        "state_hash": state["state_hash"],
        "reference": ref,
        "root_budget": budget,
        "root_action": alias["action"],
        "probe_seed": probe["seed"],
        "probe_seed_context_hash": probe["seed_context_hash"],
        "trajectory_identity": alias["trajectory_identity"],
        "alias_of": alias["alias_of"],
        "score": trajectory["score"],
        "margin": trajectory["store_margin_root_perspective"],
        "elapsed_wall_seconds": trajectory["elapsed_wall_seconds"],
    }


def _unique(
    rows: list[dict[str, Any]], key_fn, error: str
) -> dict[Any, dict[str, Any]]:
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise ValueError(error)
        result[key] = row
    return result
