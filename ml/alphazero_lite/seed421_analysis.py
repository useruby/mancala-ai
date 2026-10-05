"""Strict, read-only validation and analysis for the seed421 correction."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict

import numpy as np

from .kalah_rules import KalahGame

ROOTS = 32
BUDGETS = (384, 1536)
REPETITIONS = 3
CAPACITY = 4096


def seed_for(root: str, budget: int, repetition: int) -> int:
    return int(
        hashlib.sha256(f"420:rng:{root}:{budget}:{repetition}".encode()).hexdigest()[
            :16
        ],
        16,
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def validate_output(output: dict, root: dict, budget: int) -> None:
    legal = KalahGame.from_state(root["state"]).possible_moves()
    visits = output.get("visits")
    if (
        not isinstance(visits, list)
        or len(visits) != 6
        or any(type(count) is not int or count < 0 for count in visits)
    ):
        raise ValueError("invalid_visits")
    if sum(visits) != budget or output.get("root_visit_count") != budget:
        raise ValueError("visit_budget_mismatch")
    if any(count and move not in legal for move, count in enumerate(visits)):
        raise ValueError("illegal_visit_support")
    action = output.get("selected_action")
    if type(action) is not int or action not in legal:
        raise ValueError("illegal_selected_action")
    policy = np.asarray(output.get("policy"), dtype=np.float32)
    target = np.asarray(visits, dtype=np.float32) / np.float32(budget)
    if policy.shape != (6,) or not np.allclose(policy, target, rtol=1e-6, atol=1e-7):
        raise ValueError("policy_visit_mismatch")
    q_values = output.get("q_values")
    if not isinstance(q_values, dict) or any(
        str(move) not in q_values for move in legal if visits[move]
    ):
        raise ValueError("q_value_support_mismatch")
    if any(not _finite(value) for value in [output.get("root_q"), *q_values.values()]):
        raise ValueError("nonfinite_q_value")


def validate_cache(cache: dict) -> None:
    names = ("hits", "misses", "requests", "neural_calls", "evictions", "peak_entries")
    if any(type(cache.get(name)) is not int or cache[name] < 0 for name in names):
        raise ValueError("invalid_cache_accounting")
    if cache["hits"] + cache["misses"] != cache["requests"]:
        raise ValueError("cache_request_count_mismatch")
    if cache["misses"] != cache["neural_calls"]:
        raise ValueError("cache_call_count_mismatch")
    if cache["peak_entries"] > CAPACITY or cache["evictions"] != max(
        0, cache["misses"] - CAPACITY
    ):
        raise ValueError("cache_occupancy_mismatch")


def validate_ledger(
    rows: list[dict], cohort: list[dict], registration_sha256: str, *, complete: bool
) -> None:
    if len(cohort) != ROOTS:
        raise ValueError("cohort_count_mismatch")
    roots = {root["state_hash"]: root for root in cohort}
    expected = {
        (root, budget, kind, repetition)
        for root in roots
        for budget in BUDGETS
        for kind, count in (("parity", 1), ("timed", REPETITIONS))
        for repetition in range(count)
    }
    observed = set()
    for row in rows:
        key = (
            row.get("root_hash"),
            row.get("budget"),
            row.get("kind"),
            row.get("repetition"),
        )
        if key not in expected:
            raise ValueError("unregistered_pair")
        if key in observed:
            raise ValueError("duplicate_pair")
        observed.add(key)
        root_hash, budget, kind, repetition = key
        root = roots[root_hash]
        if row.get("phase") != root["phase"]:
            raise ValueError("phase_mismatch")
        if row.get("seed") != seed_for(
            root_hash, budget, -1 if kind == "parity" else repetition
        ):
            raise ValueError("seed_mismatch")
        if row.get("registration_sha256") != registration_sha256:
            raise ValueError("registration_identity_mismatch")
        result = row.get("result", {})
        off, on = result.get("off"), result.get("on")
        if not isinstance(off, dict) or not isinstance(on, dict):
            raise ValueError("output_mismatch")
        validate_output(off, root, budget)
        validate_output(on, root, budget)
        if off != on:
            raise ValueError("output_mismatch")
        if kind == "parity":
            n = result.get("evaluation_requests")
            traces = (result.get("off_request_trace"), result.get("on_request_trace"))
            if (
                type(n) is not int
                or n <= 0
                or any(
                    not isinstance(trace, list) or len(trace) != n for trace in traces
                )
            ):
                raise ValueError("invalid_instrumented_trace")
            if traces[0] != traces[1] or result.get("neural_calls_off") != n:
                raise ValueError("instrumented_request_mismatch")
            if any(
                item.get("root_identity") != root_hash
                for trace in traces
                for item in trace
            ):
                raise ValueError("instrumented_root_identity_mismatch")
            cache = result.get("cache", {})
            validate_cache(cache)
            if (
                cache["requests"] != n
                or result.get("neural_calls_on") != cache["neural_calls"]
            ):
                raise ValueError("instrumented_accounting_mismatch")
        else:
            expected_order = ["off", "on"] if repetition % 2 == 0 else ["on", "off"]
            if row.get("execution_order") != expected_order:
                raise ValueError("execution_order_mismatch")
            latencies = row.get("latency_ms", {})
            if set(latencies) != {"off", "on"} or any(
                not _finite(x) or x <= 0 for x in latencies.values()
            ):
                raise ValueError("invalid_timing")
            requests = row.get("request_counts", {})
            calls = row.get("neural_call_counts", {})
            cache = row.get("cache", {})
            validate_cache(cache)
            if (
                requests.get("off") != requests.get("on")
                or requests.get("on") != cache["requests"]
            ):
                raise ValueError("timed_request_count_mismatch")
            if (
                calls.get("off") != requests.get("off")
                or calls.get("on") != cache["neural_calls"]
            ):
                raise ValueError("timed_call_count_mismatch")
    if complete and observed != expected:
        raise ValueError(f"incomplete_evidence:{len(expected - observed)}")


def analyze(rows: list[dict], cohort: list[dict], registration_sha256: str) -> dict:
    validate_ledger(rows, cohort, registration_sha256, complete=True)
    timed = [row for row in rows if row["kind"] == "timed"]
    by_budget = {}
    for budget in BUDGETS:
        rows_budget = [row for row in timed if row["budget"] == budget]
        by_root: dict[str, list[dict]] = defaultdict(list)
        for row in rows_budget:
            by_root[row["root_hash"]].append(row)
        root_speedups = {
            root: statistics.fmean(
                (pair["latency_ms"]["off"] - pair["latency_ms"]["on"])
                / pair["latency_ms"]["off"]
                for pair in pairs
            )
            for root, pairs in by_root.items()
        }
        entry = {
            "mean_paired_relative_speedup": statistics.fmean(root_speedups.values())
        }
        entry["by_phase"] = {}
        for phase in (">32", "17-32"):
            phase_roots = [r["state_hash"] for r in cohort if r["phase"] == phase]
            entry["by_phase"][phase] = statistics.fmean(
                root_speedups[r] for r in phase_roots
            )
        entry["mean_off_latency_ms"] = statistics.fmean(
            r["latency_ms"]["off"] for r in rows_budget
        )
        entry["mean_on_latency_ms"] = statistics.fmean(
            r["latency_ms"]["on"] for r in rows_budget
        )
        entry["off_p95_nearest_rank_ms"] = sorted(
            r["latency_ms"]["off"] for r in rows_budget
        )[math.ceil(0.95 * len(rows_budget)) - 1]
        entry["on_p95_nearest_rank_ms"] = sorted(
            r["latency_ms"]["on"] for r in rows_budget
        )[math.ceil(0.95 * len(rows_budget)) - 1]
        by_budget[str(budget)] = entry
    rng = random.Random(420)
    roots = sorted(
        by_root for by_root in {r["root_hash"] for r in timed if r["budget"] == 384}
    )
    speed = {}
    for root in roots:
        pairs = [r for r in timed if r["budget"] == 384 and r["root_hash"] == root]
        speed[root] = statistics.fmean(
            (p["latency_ms"]["off"] - p["latency_ms"]["on"]) / p["latency_ms"]["off"]
            for p in pairs
        )
    samples = [
        statistics.fmean(speed[roots[rng.randrange(len(roots))]] for _ in roots)
        for _ in range(10_000)
    ]
    low = sorted(samples)[math.ceil(0.025 * len(samples)) - 1]
    high = sorted(samples)[math.ceil(0.975 * len(samples)) - 1]
    by_budget["384"]["bootstrap_95_percentile_interval"] = [low, high]
    advance = (
        by_budget["384"]["mean_paired_relative_speedup"] >= 0.15
        and low > 0
        and all(v >= -0.05 for v in by_budget["384"]["by_phase"].values())
        and by_budget["1536"]["mean_paired_relative_speedup"] >= 0
    )
    return {
        "schema": "seed421-analysis-v1",
        "complete_pairs": True,
        "search_count": 512,
        "by_budget": by_budget,
        "decision": "advance_to_separately_proposed_integration_task"
        if advance
        else "stop_memoization_branch",
    }
