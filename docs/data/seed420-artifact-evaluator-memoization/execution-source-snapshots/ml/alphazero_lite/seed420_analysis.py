"""Read-only, dependency-light accounting and decision rules for seed420."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict


EXPECTED_ROOTS = 32
EXPECTED_BUDGETS = (384, 1536)
REPETITIONS = 3


def nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty_metric")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _outputs(record: dict) -> tuple[dict, dict]:
    result = record["result"]
    if record["kind"] == "parity":
        return result["off"], result["on"]
    return result["off"], result["on"]


def validate_ledger(rows: list[dict], cohort: list[dict]) -> None:
    if len(cohort) != EXPECTED_ROOTS:
        raise ValueError("cohort_count_mismatch")
    root_phases = {row["state_hash"]: row["phase"] for row in cohort}
    expected_keys = {
        (root, budget, kind, repetition)
        for root in root_phases
        for budget in EXPECTED_BUDGETS
        for kind, count in (("parity", 1), ("timed", REPETITIONS))
        for repetition in range(count)
    }
    observed: set[tuple[str, int, str, int]] = set()
    for row in rows:
        key = (
            row["root_hash"],
            int(row["budget"]),
            row["kind"],
            int(row["repetition"]),
        )
        if key not in expected_keys:
            raise ValueError("substituted_or_unregistered_pair")
        if key in observed:
            raise ValueError("duplicate_pair")
        observed.add(key)
        if row["phase"] != root_phases[row["root_hash"]]:
            raise ValueError("pair_phase_mismatch")
        off, on = _outputs(row)
        if off != on:
            raise ValueError("search_output_mismatch")
        if row["kind"] == "parity":
            result = row["result"]
            if result["evaluation_requests"] != len(
                result.get("off_request_trace", [])
            ):
                raise ValueError("partial_instrumented_requests")
            if result["off_request_trace"] != result["on_request_trace"]:
                raise ValueError("instrumented_request_mismatch")
            if result["neural_calls_off"] != result["evaluation_requests"]:
                raise ValueError("instrumented_off_call_count_mismatch")
            if result["neural_calls_on"] != result["cache"]["misses"]:
                raise ValueError("instrumented_on_call_count_mismatch")
        else:
            if set(row.get("latency_ms", {})) != {"off", "on"}:
                raise ValueError("partial_timed_pair")
            order = row.get("execution_order")
            expected_order = ["off", "on"] if key[3] % 2 == 0 else ["on", "off"]
            if order != expected_order:
                raise ValueError("execution_order_mismatch")
            if not row.get("cache") or row["cache"].get("requests", 0) <= 0:
                raise ValueError("timed_cache_accounting_missing")
            if set(row.get("request_counts", {})) != {"off", "on"}:
                raise ValueError("timed_request_counts_missing")
            if set(row.get("neural_call_counts", {})) != {"off", "on"}:
                raise ValueError("timed_neural_call_counts_missing")
            if row["neural_call_counts"]["off"] != row["request_counts"]["off"]:
                raise ValueError("timed_off_calls_mismatch")
            if row["neural_call_counts"]["on"] != row["cache"].get("neural_calls"):
                raise ValueError("timed_on_calls_mismatch")
    if observed != expected_keys:
        raise ValueError(f"incomplete_pairs:{len(expected_keys - observed)}")


def _bootstrap_cluster(values: dict[str, float]) -> list[float]:
    roots = sorted(values)
    rng = random.Random(420)
    samples = []
    for _ in range(10_000):
        sample = [values[roots[rng.randrange(len(roots))]] for _ in roots]
        samples.append(statistics.fmean(sample))
    return samples


def analyze(rows: list[dict], cohort: list[dict]) -> dict:
    validate_ledger(rows, cohort)
    timed = [row for row in rows if row["kind"] == "timed"]
    output = {}
    for budget in EXPECTED_BUDGETS:
        budget_rows = [row for row in timed if int(row["budget"]) == budget]
        by_root: dict[str, list[dict]] = defaultdict(list)
        for row in budget_rows:
            by_root[row["root_hash"]].append(row)
        root_results = {}
        root_speedups = {}
        for root, rep_rows in by_root.items():
            off = statistics.median(row["latency_ms"]["off"] for row in rep_rows)
            on = statistics.median(row["latency_ms"]["on"] for row in rep_rows)
            root_speedups[root] = statistics.fmean(
                (row["latency_ms"]["off"] - row["latency_ms"]["on"])
                / row["latency_ms"]["off"]
                for row in rep_rows
            )
            root_results[root] = {
                "off_median_ms": off,
                "on_median_ms": on,
                "paired_relative_speedup": root_speedups[root],
            }
        phases = {}

        def stats(rows_for_stats: list[dict]) -> dict[str, int]:
            return {
                "cache_hits": sum(
                    int(row["cache"].get("hits", 0)) for row in rows_for_stats
                ),
                "cache_misses": sum(
                    int(row["cache"].get("misses", 0)) for row in rows_for_stats
                ),
                "cache_evictions": sum(
                    int(row["cache"].get("evictions", 0)) for row in rows_for_stats
                ),
                "cache_neural_calls": sum(
                    int(row["neural_call_counts"]["on"]) for row in rows_for_stats
                ),
                "cache_requests": sum(
                    int(row["request_counts"]["on"]) for row in rows_for_stats
                ),
                "baseline_neural_calls": sum(
                    int(row["neural_call_counts"]["off"]) for row in rows_for_stats
                ),
                "baseline_requests": sum(
                    int(row["request_counts"]["off"]) for row in rows_for_stats
                ),
                "peak_entries_max": max(
                    int(row["cache"].get("peak_entries", 0)) for row in rows_for_stats
                ),
            }

        for phase in (">32", "17-32"):
            roots = [row["state_hash"] for row in cohort if row["phase"] == phase]
            phase_rows = [row for row in budget_rows if row["phase"] == phase]
            off_latencies = [row["latency_ms"]["off"] for row in phase_rows]
            on_latencies = [row["latency_ms"]["on"] for row in phase_rows]
            phases[phase] = {
                "roots": len(roots),
                "mean_paired_relative_speedup": statistics.fmean(
                    root_speedups[root] for root in roots
                ),
                "mean_off_latency_ms": statistics.fmean(off_latencies),
                "mean_on_latency_ms": statistics.fmean(on_latencies),
                "off_p95_nearest_rank_ms": nearest_rank(off_latencies, 0.95),
                "on_p95_nearest_rank_ms": nearest_rank(on_latencies, 0.95),
                "per_root_median_latency_ms": {
                    root: root_results[root] for root in roots
                },
                "evaluation_accounting": stats(phase_rows),
            }
        off_all = [row["latency_ms"]["off"] for row in budget_rows]
        on_all = [row["latency_ms"]["on"] for row in budget_rows]
        output[str(budget)] = {
            "mean_paired_relative_speedup": statistics.fmean(root_speedups.values()),
            "mean_off_latency_ms": statistics.fmean(off_all),
            "mean_on_latency_ms": statistics.fmean(on_all),
            "off_p95_nearest_rank_ms": nearest_rank(off_all, 0.95),
            "on_p95_nearest_rank_ms": nearest_rank(on_all, 0.95),
            "per_root_median_latency_ms": root_results,
            "by_phase": phases,
            "cache_totals": {
                key: sum(int(row["cache"].get(key, 0)) for row in budget_rows)
                for key in (
                    "hits",
                    "misses",
                    "evictions",
                    "neural_calls",
                    "requests",
                )
            },
            "peak_entries_max": max(
                int(row["cache"].get("peak_entries", 0)) for row in budget_rows
            ),
            "timed_request_totals": {
                condition: sum(
                    int(row["request_counts"][condition]) for row in budget_rows
                )
                for condition in ("off", "on")
            },
            "timed_neural_call_totals": {
                condition: sum(
                    int(row["neural_call_counts"][condition]) for row in budget_rows
                )
                for condition in ("off", "on")
            },
        }
        if budget == 384:
            samples = _bootstrap_cluster(root_speedups)
            output[str(budget)]["bootstrap_95_percentile_interval"] = [
                nearest_rank(samples, 0.025),
                nearest_rank(samples, 0.975),
            ]
    low = output["384"]
    low_interval = low["bootstrap_95_percentile_interval"]
    advance = (
        low["mean_paired_relative_speedup"] >= 0.15
        and low_interval[0] > 0
        and all(
            phase["mean_paired_relative_speedup"] >= -0.05
            for phase in low["by_phase"].values()
        )
        and output["1536"]["mean_paired_relative_speedup"] >= 0
    )
    return {
        "schema": "seed420-analysis-v1",
        "complete_pairs": True,
        "search_count": 512,
        "by_budget": output,
        "decision": "advance_to_separately_proposed_integration_task"
        if advance
        else "stop_memoization_branch",
        "interpretation": "performance-only retrospective cohort; does not establish playing-strength improvement",
    }
