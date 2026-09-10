#!/usr/bin/env python3
"""Evaluate tier-19 native exact-oracle feasibility on frozen unresolved rows.

This runner is deliberately separate from the PR #292 audit artifact. It never
updates its cohort, exact labels, or attempt histories.
"""

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

from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    CANONICAL_TABLEBASE_SHA256,
    COHORT_SHA256,
    COMPLETION_RETRY_TIMEOUT_SECONDS,
    NativeHybridProcess,
    NativeLabelError,
    NativeRequestError,
    active_stones,
    native_label_payload,
    primary_anchor,
    root_training_value,
    sha256_file,
    stable_sha,
    validate_native_response,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402


BASELINE = ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json"
TIER19_SHA256 = "53cf399e8a00e0eb4bbd9d7ae20d528f6e2d91f9350269047de49d2c4a4099fd"
TIER = 19
SCHEMA = "azlite_uniform1200_tier19_unresolved_feasibility_v1"


def preflight_tier19(probe: Path, tablebase: Path) -> dict[str, Any]:
    if not probe.is_file() or not os.access(probe, os.X_OK | os.R_OK):
        raise ValueError(f"native probe is not executable/readable: {probe}")
    if not tablebase.is_file() or not os.access(tablebase, os.R_OK):
        raise ValueError(f"tablebase is not readable: {tablebase}")
    digest = sha256_file(tablebase)
    if digest != TIER19_SHA256:
        raise ValueError("tablebase SHA does not match the pinned tier-19 artifact")
    with tablebase.open("rb") as handle:
        header = handle.read(48)
    if len(header) != 48 or header[:5] != b"KVTB1" or header[18] != TIER:
        raise ValueError("tablebase is not a KVTB1 tier-19 artifact")
    return {
        "probe_sha256": sha256_file(probe),
        "tier18_tablebase_sha256": CANONICAL_TABLEBASE_SHA256,
        "tablebase_sha256": digest,
        "tablebase_tier": TIER,
        "tablebase_bytes": tablebase.stat().st_size,
    }


def unresolved_rows(cohort: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in cohort
            if not row["terminal"]
            and row.get("oracle_status") in {"exact_error", "exact_timeout"}
        ],
        key=lambda row: (
            0 if row["oracle_status"] == "exact_error" else 1,
            primary_anchor(row),
            row["radius"],
            stable_sha(row["canonical_state_key"]),
            row["canonical_state_key"],
        ),
    )


def shard_rows(
    rows: list[dict[str, Any]], workers: int
) -> dict[int, list[dict[str, Any]]]:
    shards = {worker: [] for worker in range(workers)}
    for row in rows:
        shard = int(stable_sha(primary_anchor(row)), 16) % workers
        shards[shard].append(row)
    for shard in shards.values():
        shard.sort(
            key=lambda row: (
                0 if row["oracle_status"] == "exact_error" else 1,
                primary_anchor(row),
                row["radius"],
                stable_sha(row["canonical_state_key"]),
                row["canonical_state_key"],
            )
        )
    return shards


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
        "attempt_class": "tier19_larger_tablebase_feasibility",
        "source_oracle_status": row["oracle_status"],
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("frozen_cohort_sha256") != COHORT_SHA256:
        raise ValueError("PR #292 frozen cohort identity changed")
    preflight = preflight_tier19(args.native_probe, args.tablebase)
    rows = unresolved_rows(baseline["cohort"])
    prior = json.loads(args.output.read_text()) if args.output.exists() else {}
    completed = {row["canonical_state_key"]: row for row in prior.get("rows", [])}
    rows_by_key = {row["canonical_state_key"]: row for row in rows}
    pending = [row for row in rows if row["canonical_state_key"] not in completed]
    report = {
        "schema": SCHEMA,
        "frozen_cohort_sha256": COHORT_SHA256,
        "baseline_oracle_coverage": baseline["oracle_coverage"],
        "preflight": preflight,
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
                "active_stones": active_stones(rows_by_key[key]),
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
            for key, result in future.result():
                completed[key] = {
                    "canonical_state_key": key,
                    "active_stones": active_stones(rows_by_key[key]),
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
    report["complete"] = len(completed) == len(rows)
    report["coverage"] = dict(
        sorted(Counter(row["status"] for row in report["rows"]).items())
    )
    report["combined_coverage_by_radius"] = {
        str(radius): {
            "states": len(subset),
            "exact_solved": sum(
                row["oracle_status"] == "exact_solved"
                or completed.get(row["canonical_state_key"], {}).get("status")
                == "exact_solved"
                for row in subset
            ),
        }
        for radius in range(3)
        for subset in [[row for row in baseline["cohort"] if row["radius"] == radius]]
    }
    for item in report["combined_coverage_by_radius"].values():
        item["solve_rate"] = item["exact_solved"] / item["states"]
    persist(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
