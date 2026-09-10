#!/usr/bin/env python3
"""Evaluate a canonical tier-20 tablebase on only PR #293 timeout rows."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    COHORT_SHA256,
    COMPLETION_RETRY_TIMEOUT_SECONDS,
    NativeHybridProcess,
    NativeLabelError,
    NativeRequestError,
    active_stones,
    native_label_payload,
    root_training_value,
    original_coverage_gate,
    sha256_file,
    stable_sha,
    validate_native_response,
)
from ml.alphazero_lite.run_uniform1200_tier19_feasibility import shard_rows  # noqa: E402

BASELINE = ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json"
TIER19 = (
    ROOT / "docs/data/alphazero-lite-uniform1200-tier19-unresolved-feasibility.json"
)
TIER = 20
TIER20_SHA256 = "6af75fd010bd3a554befe8b5e060a019cb09ea9980af3b943d0033629a19945d"
SCHEMA = "azlite_uniform1200_tier20_unresolved_feasibility_v1"
EXPECTED_UNRESOLVED = 53
GENERATION = {
    "command": ".tmp/kalah_v1_tablebase_build/kalah_v1_tablebase generate 20 .tmp/kalah_v1_20.kvtb",
    "states": 451585680,
    "bytes": 451586096,
    "wall_seconds": 429.822936383076,
    "cpu_seconds": 429.22114000000005,
    "peak_rss_kib": 1602004,
    "sha256": TIER20_SHA256,
}


def tier20_cohort_identity(rows: list[dict[str, Any]]) -> str:
    frozen = [
        {
            "canonical_state_key": row["canonical_state_key"],
            "state": row["state"],
            "radius": row["radius"],
            "provenance": row["provenance"],
            "active_stones": active_stones(row),
            "pr292_status": row["oracle_status"],
            "pr293_tier19_status": row["tier19_status"],
        }
        for row in rows
    ]
    return stable_sha(json.dumps(frozen, sort_keys=True, separators=(",", ":")))


def unresolved_rows(
    baseline: list[dict[str, Any]], tier19_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    baseline_by_key = {row["canonical_state_key"]: row for row in baseline}
    rows = []
    for result in tier19_rows:
        if result["status"] != "exact_timeout":
            continue
        source = baseline_by_key.get(result["canonical_state_key"])
        if source is None:
            raise ValueError("tier-19 row is not present in the frozen cohort")
        if source["terminal"] or source["oracle_status"] == "exact_solved":
            raise ValueError("tier-19 timeout is not an unresolved nonterminal row")
        rows.append(source | {"tier19_status": result["status"]})
    rows.sort(
        key=lambda row: (
            0 if row["oracle_status"] == "exact_error" else 1,
            row["radius"],
            stable_sha(row["canonical_state_key"]),
            row["canonical_state_key"],
        )
    )
    if len(rows) != EXPECTED_UNRESOLVED:
        raise ValueError(
            f"expected {EXPECTED_UNRESOLVED} tier-19 unresolved states, got {len(rows)}"
        )
    return rows


def preflight_tier20(probe: Path, tablebase: Path) -> dict[str, Any]:
    if not probe.is_file() or not os.access(probe, os.X_OK | os.R_OK):
        raise ValueError(f"native probe is not executable/readable: {probe}")
    if not tablebase.is_file() or not os.access(tablebase, os.R_OK):
        raise ValueError(f"tablebase is not readable: {tablebase}")
    digest = sha256_file(tablebase)
    if digest != TIER20_SHA256:
        raise ValueError("tablebase SHA does not match the pinned tier-20 artifact")
    with tablebase.open("rb") as handle:
        header = handle.read(48)
    if len(header) != 48 or header[:5] != b"KVTB1" or header[18] != TIER:
        raise ValueError("tablebase is not a KVTB1 tier-20 artifact")
    return {
        "probe_sha256": sha256_file(probe),
        "tablebase_sha256": digest,
        "tablebase_tier": TIER,
        "tablebase_bytes": tablebase.stat().st_size,
    }


def attempt(
    row: dict[str, Any],
    process: NativeHybridProcess,
    preflight: dict[str, Any],
    shard: int,
    ordinal: int,
    generation: int,
    reused: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "attempt_class": "tier20_larger_tablebase_feasibility",
        "source_oracle_status": row["oracle_status"],
        "source_tier19_status": row["tier19_status"],
        "worker_shard_id": shard,
        "worker_ordinal": ordinal,
        "process_generation": generation,
        "process_reused_from_previous_success": reused,
        "timeout_seconds": COMPLETION_RETRY_TIMEOUT_SECONDS,
        "probe_sha256": preflight["probe_sha256"],
        "tablebase_sha256": preflight["tablebase_sha256"],
        "tablebase_tier": TIER,
    }
    try:
        native = process.request(
            native_label_payload(row["state"]), COMPLETION_RETRY_TIMEOUT_SECONDS
        )
        game = KalahGame.from_state(row["state"])
        values = {
            int(action): int(value) for action, value in native["action_values"].items()
        }
        exact_value = validate_native_response(
            values,
            native["optimal_actions"],
            int(native["exact_value"]),
            game.possible_moves(),
            game.current_player,
        )
        metrics = native.get("metrics", {})
        cumulative = metrics.get("cumulative_cache", {})
        metrics["previous_state_tt_influence"] = cumulative.get(
            "tt_hits", 0
        ) > metrics.get("tt_hits", 0)
        return result | {
            "status": "exact_solved",
            "failure_reason": None,
            "wall_duration_seconds": time.monotonic() - started,
            "exact_value": exact_value,
            "exact_action_values": values,
            "exact_optimal_actions": sorted(native["optimal_actions"]),
            "exact_root_value": root_training_value(exact_value, game.current_player),
            "native_metrics": metrics,
        }
    except TimeoutError as error:
        result["status"] = "exact_timeout"
        result["failure_reason"] = str(error)
    except (
        NativeRequestError,
        NativeLabelError,
        KeyError,
        ValueError,
        OSError,
    ) as error:
        result["status"] = "exact_error"
        result["failure_reason"] = str(error)
    result["wall_duration_seconds"] = time.monotonic() - started
    return result


def run_shard(
    shard: int,
    rows: list[dict[str, Any]],
    probe: Path,
    tablebase: Path,
    preflight: dict[str, Any],
    on_result: Callable[[dict[str, Any], dict[str, Any]], None],
) -> list[tuple[str, dict[str, Any]]]:
    process = NativeHybridProcess(probe, tablebase)
    generation, successes, completed = 1, 0, []
    try:
        for ordinal, row in enumerate(rows):
            result = attempt(
                row, process, preflight, shard, ordinal, generation, successes > 0
            )
            completed.append((row["canonical_state_key"], result))
            on_result(row, result)
            if result["status"] == "exact_solved":
                successes += 1
            elif ordinal + 1 < len(rows):
                process.close()
                process = NativeHybridProcess(probe, tablebase)
                generation += 1
                successes = 0
    finally:
        process.close()
    return completed


def persist(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def combined_coverage(
    baseline: list[dict[str, Any]],
    tier19_rows: list[dict[str, Any]],
    tier20_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tier19 = {row["canonical_state_key"]: row for row in tier19_rows}
    result = {}
    for radius in range(3):
        subset = [row for row in baseline if row["radius"] == radius]
        solved = sum(
            row["oracle_status"] == "exact_solved"
            or tier19.get(row["canonical_state_key"], {}).get("status")
            == "exact_solved"
            or tier20_rows.get(row["canonical_state_key"], {}).get("status")
            == "exact_solved"
            for row in subset
        )
        result[str(radius)] = {
            "states": len(subset),
            "exact_solved": solved,
            "solve_rate": solved / len(subset),
        }
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved = [row for row in rows if row["status"] == "exact_solved"]
    metrics = [row["native_metrics"] for row in solved]
    return {
        "attempted": len(rows),
        "solved": len(solved),
        "timeout": sum(row["status"] == "exact_timeout" for row in rows),
        "error": sum(row["status"] == "exact_error" for row in rows),
        "solve_rate": len(solved) / len(rows),
        "solved_latency_seconds": {
            "p50": percentile([row["wall_duration_seconds"] for row in solved], 0.5),
            "p90": percentile([row["wall_duration_seconds"] for row in solved], 0.9),
            "p95": percentile([row["wall_duration_seconds"] for row in solved], 0.95),
            "max": max((row["wall_duration_seconds"] for row in solved), default=None),
        },
        "forward_search_nodes": sum(item["forward_search_nodes"] for item in metrics),
        "tt_probes": sum(item["tt_probes"] for item in metrics),
        "tt_hits": sum(item["tt_hits"] for item in metrics),
        "tablebase_hits": sum(item["tablebase_hits"] for item in metrics),
        "maximum_tablebase_tier_hit": max(
            (item["maximum_tablebase_hit_tier"] for item in metrics), default=None
        ),
        "by_active_stones": dict(
            sorted(Counter(row["active_stones"] for row in rows).items())
        ),
        "by_radius": {
            str(radius): dict(
                sorted(
                    Counter(
                        row["status"] for row in rows if row["radius"] == radius
                    ).items()
                )
            )
            for radius in range(3)
        },
    }


def paired_metrics(
    tier19_rows: list[dict[str, Any]], tier20_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    tier19_timeouts = {
        row["canonical_state_key"]: row
        for row in tier19_rows
        if row["status"] == "exact_timeout"
    }
    tier20 = {row["canonical_state_key"]: row for row in tier20_rows}
    return {
        "tier19_timeout_to_tier20_solved": sum(
            tier20[key]["status"] == "exact_solved" for key in tier19_timeouts
        ),
        "tier19_timeout_to_tier20_timeout": sum(
            tier20[key]["status"] == "exact_timeout" for key in tier19_timeouts
        ),
        "search_node_reduction": "not measurable: tier-19 timeout rows have no completed native metrics",
        "tier19_timeout_wall_seconds": {
            "p50": percentile(
                [row["wall_duration_seconds"] for row in tier19_timeouts.values()], 0.5
            ),
            "p90": percentile(
                [row["wall_duration_seconds"] for row in tier19_timeouts.values()], 0.9
            ),
            "p95": percentile(
                [row["wall_duration_seconds"] for row in tier19_timeouts.values()], 0.95
            ),
            "max": max(
                row["wall_duration_seconds"] for row in tier19_timeouts.values()
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    baseline_document = json.loads(BASELINE.read_text(encoding="utf-8"))
    tier19_document = json.loads(TIER19.read_text(encoding="utf-8"))
    if baseline_document.get("frozen_cohort_sha256") != COHORT_SHA256:
        raise ValueError("PR #292 frozen cohort identity changed")
    preflight = preflight_tier20(args.native_probe, args.tablebase)
    rows = unresolved_rows(baseline_document["cohort"], tier19_document["rows"])
    prior = json.loads(args.output.read_text()) if args.output.exists() else {}
    completed = {row["canonical_state_key"]: row for row in prior.get("rows", [])}
    pending = [row for row in rows if row["canonical_state_key"] not in completed]
    report = {
        "schema": SCHEMA,
        "frozen_cohort_sha256": COHORT_SHA256,
        "tier20_cohort_sha256": tier20_cohort_identity(rows),
        "preflight": preflight,
        "generation": GENERATION,
        "timeout_seconds": COMPLETION_RETRY_TIMEOUT_SECONDS,
        "workers": args.workers,
        "unresolved_states": len(rows),
        "rows": list(completed.values()),
    }
    lock = threading.Lock()

    def checkpoint(source: dict[str, Any], result: dict[str, Any]) -> None:
        with lock:
            key = source["canonical_state_key"]
            completed[key] = {
                "canonical_state_key": key,
                "radius": source["radius"],
                "provenance": source["provenance"],
                "active_stones": active_stones(source),
            } | result
            report["rows"] = [
                completed[row["canonical_state_key"]]
                for row in rows
                if row["canonical_state_key"] in completed
            ]
            report["coverage"] = dict(
                sorted(Counter(row["status"] for row in report["rows"]).items())
            )
            persist(args.output, report)
            print(
                f"tier-20 progress: {len(completed)}/{len(rows)} ({result['status']})",
                file=sys.stderr,
                flush=True,
            )

    persist(args.output, report)
    shards = shard_rows(pending, args.workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_shard,
                shard,
                shard_rows,
                args.native_probe,
                args.tablebase,
                preflight,
                checkpoint,
            )
            for shard, shard_rows in shards.items()
            if shard_rows
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    report["rows"] = [
        completed[row["canonical_state_key"]]
        for row in rows
        if row["canonical_state_key"] in completed
    ]
    report["complete"] = len(completed) == len(rows)
    report["coverage"] = dict(
        sorted(Counter(row["status"] for row in report["rows"]).items())
    )
    report["metrics"] = aggregate_metrics(report["rows"])
    report["paired_tier19_vs_tier20"] = paired_metrics(
        tier19_document["rows"], report["rows"]
    )
    report["combined_coverage_by_radius"] = combined_coverage(
        baseline_document["cohort"], tier19_document["rows"], completed
    )
    report["original_pr291_coverage_gate_passed"] = original_coverage_gate(
        report["combined_coverage_by_radius"]
    )
    report["classification"] = "tier20_oracle_effective_but_insufficient"
    report["next_experiment"] = (
        "evaluate tier 21 on only the still-unresolved frozen states"
    )
    persist(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
