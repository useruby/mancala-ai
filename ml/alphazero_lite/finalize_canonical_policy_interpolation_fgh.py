#!/usr/bin/env python3
# ruff: noqa: E402
"""Attach robust F-H diagnostics to the canonical interpolation report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.alphazero_lite.run_canonical_policy_interpolation_reconciliation import (
    write_final_outputs,
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def bootstrap(values: list[float], seed: int = 42) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(10_000, len(data)))].mean(1)
    return {
        "n": len(values),
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "samples": 10_000,
    }


def phase_f(search: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in search:
        if "@" in row["candidate"]:
            by_candidate[row["candidate"]].append(row)
    for candidate, group in sorted(by_candidate.items()):
        input_js = np.asarray([row["input_js"] for row in group])
        output_js = np.asarray([row["output_js"] for row in group])
        valid = input_js >= 1e-6
        q_margin = np.asarray([row["current_root_q_margin"] for row in group])
        visit_margin = np.asarray([row["current_visit_margin"] for row in group])

        def quartile_rates(values: np.ndarray) -> list[float]:
            cuts = np.quantile(values, [0.25, 0.5, 0.75])
            return (
                [
                    float(
                        np.mean(
                            [
                                row["selected_move_changed"]
                                for row, value in zip(group, values)
                                if value <= cuts[0]
                                if True
                            ]
                        )
                    )
                ]
                if False
                else [
                    float(
                        np.mean(
                            [
                                row["selected_move_changed"]
                                for row, value in zip(group, values)
                                if (i == 0 and value <= cuts[0])
                                or (
                                    i > 0
                                    and value > cuts[i - 1]
                                    and (i == 3 or value <= cuts[i])
                                )
                            ]
                        )
                    )
                    for i in range(4)
                ]
            )

        amplified = ((output_js >= 5 * input_js) & valid).astype(float)
        output[candidate] = {
            "median_input_js": float(np.median(input_js)),
            "median_output_js": float(np.median(output_js)),
            "median_log10_output_over_input_input_js_ge_1e6": float(
                np.median(
                    np.log10(np.maximum(output_js[valid], 1e-300) / input_js[valid])
                )
            )
            if valid.any()
            else None,
            "fraction_output_js_ge_5x_input_js": bootstrap(amplified.tolist()),
            "selected_move_change_rate": bootstrap(
                [float(row["selected_move_changed"]) for row in group]
            ),
            "move_change_rate_by_prior_js_quartile": quartile_rates(input_js),
            "move_change_rate_by_current_q_margin_quartile": quartile_rates(q_margin),
            "move_change_rate_by_current_visit_margin_quartile": quartile_rates(
                visit_margin
            ),
        }
    return output


def phase_h(forced: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forced:
        groups[
            f"{row['candidate']}|{row['original_budget']}|{row['continuation_budget']}"
        ].append(row)
    for key, group in groups.items():
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_state[row["state_hash"]].append(row)
        outcome = [
            float(np.mean([item["outcome_delta"] for item in value]))
            for value in by_state.values()
        ]
        margin = [
            float(np.mean([item["store_margin_delta"] for item in value]))
            for value in by_state.values()
        ]
        result[key] = {
            "unique_states": len(by_state),
            "outcome_delta": bootstrap(outcome),
            "store_margin_delta": bootstrap(margin),
            "better_fraction": float(np.mean(np.asarray(outcome) > 0)),
            "worse_fraction": float(np.mean(np.asarray(outcome) < 0)),
            "tie_fraction": float(np.mean(np.asarray(outcome) == 0)),
        }
    return result


def phase_g(
    search: list[dict[str, Any]], traces: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    trace_keys = {
        (row["candidate"], row["budget"], row["state_hash"]) for row in traces
    }
    matched = [
        row
        for row in search
        if (row["candidate"], row["budget"], row["state_hash"]) in trace_keys
    ]
    near = [row for row in matched if row["current_root_q_margin"] <= threshold]
    outside = [row for row in matched if row["current_root_q_margin"] > threshold]
    # Compare divergence probabilities by resampling unique state-condition rows.
    low_rows = [
        row
        for row in search
        if "@" in row["candidate"] and row["current_root_q_margin"] <= threshold
    ]
    high_rows = [
        row
        for row in search
        if "@" in row["candidate"] and row["current_root_q_margin"] > threshold
    ]
    rng = np.random.default_rng(4242)
    low, high = (
        np.asarray([row["selected_move_changed"] for row in low_rows], float),
        np.asarray([row["selected_move_changed"] for row in high_rows], float),
    )
    draws = low[rng.integers(0, len(low), size=(10_000, len(low)))].mean(1) - high[
        rng.integers(0, len(high), size=(10_000, len(high)))
    ].mean(1)
    by_alpha: dict[str, list[float]] = defaultdict(list)
    for row in traces:
        if row.get("simulation") is not None and "@" in row["candidate"]:
            by_alpha[row["candidate"]].append(float(row["simulation"]))
    prior_quartiles: dict[str, list[float]] = {}
    for candidate in sorted(
        {row["candidate"] for row in search if "@" in row["candidate"]}
    ):
        group = [row for row in search if row["candidate"] == candidate]
        cuts = np.quantile(
            [row["prior_delta_on_candidate_move"] for row in group], [0.25, 0.5, 0.75]
        )
        prior_quartiles[candidate] = [
            float(
                np.mean(
                    [
                        row["selected_move_changed"]
                        for row in group
                        if (i == 0 and row["prior_delta_on_candidate_move"] <= cuts[0])
                        or (
                            i > 0
                            and row["prior_delta_on_candidate_move"] > cuts[i - 1]
                            and (
                                i == 3
                                or row["prior_delta_on_candidate_move"] <= cuts[i]
                            )
                        )
                    ]
                )
            )
            for i in range(4)
        ]
    return {
        "first_divergences_near_tied": {
            "count": len(near),
            "fraction": len(near) / max(len(matched), 1),
        },
        "first_divergences_outside_near_tied": {
            "count": len(outside),
            "fraction": len(outside) / max(len(matched), 1),
        },
        "low_minus_high_q_margin_divergence_probability_ci95": {
            "mean": float(low.mean() - high.mean()),
            "lower_95": float(np.quantile(draws, 0.025)),
            "upper_95": float(np.quantile(draws, 0.975)),
            "samples": 10_000,
        },
        "first_divergence_simulation_index_by_alpha": {
            key: float(np.median(value)) for key, value in by_alpha.items()
        },
        "divergence_probability_by_prior_delta_quartile": prior_quartiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_canonical_policy_fgh_v2")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    search, traces, forced = (
        rows(workdir / name)
        for name in (
            "search_response_records.jsonl",
            "first_divergence_records.jsonl",
            "forced_move_quality_records.jsonl",
        )
    )
    summary_path = (
        REPO_ROOT
        / "docs/data/alphazero-lite-canonical-policy-interpolation-summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    q_threshold = float(
        np.quantile([row["current_root_q_margin"] for row in search], 0.25)
    )
    flip_counts = {
        direction: [
            sum(
                row["selected_move_changed"]
                for row in search
                if row["candidate"] == f"{direction}@{alpha:.2f}"
            )
            for alpha in (0.25, 0.5, 0.75, 1.0)
        ]
        for direction in ("D384", "D1200")
    }
    monotonic = {key: values == sorted(values) for key, values in flip_counts.items()}
    h = phase_h(forced)
    conflict = any(
        value["unique_states"] >= 64 and value["outcome_delta"]["lower_95"] > 0
        for value in h.values()
    ) and any(
        value["unique_states"] >= 64 and value["outcome_delta"]["upper_95"] < 0
        for value in h.values()
    )
    summary["phase_f"] = {
        "state_count": 512,
        "method": "10,000-sample state bootstrap",
        "by_candidate": phase_f(search),
    }
    summary["phase_g"] = {
        "first_divergence_count": len(traces),
        "near_tie_threshold_current_root_q_margin_bottom_quartile": q_threshold,
        "alpha_move_change_counts": flip_counts,
        "alpha_monotonic_by_direction": monotonic,
        "classification": "search_prior_bifurcation_confirmed"
        if all(monotonic.values())
        else "policy_prior_search_sensitivity_confirmed",
        "note": "The old maximum output/input ratio is excluded because near-zero input-JS denominators are non-robust.",
        **phase_g(search, traces, q_threshold),
    }
    summary["phase_h"] = {
        "method": "10,000-sample unique-state bootstrap",
        "by_direction_alpha_origin_budget_continuation_budget": h,
        "classification": "budget_specific_policy_target_conflict"
        if conflict
        else "interpolation_results_statistically_inconclusive",
    }
    summary["classifications"] = [
        summary["classification"],
        summary["phase_g"]["classification"],
        summary["phase_h"]["classification"],
    ]
    write_final_outputs(summary)
    print(json.dumps({"classification": summary["classifications"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
