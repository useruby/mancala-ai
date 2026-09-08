#!/usr/bin/env python3
"""Label a frozen source-state cohort with the native hybrid exact teacher.

This runner never trains or promotes. It labels each frozen source state
with the PR #281 native MTD(f) hybrid (exact player-zero margins),
converts labels with ``exact_teacher_labeling`` (no MCTS fallback), and
writes train/holdout JSONL rows plus a machine-readable summary.

Timeouts and errors remain explicit failed/unsolved attempts, persisted in
the failures file. The run is resumable: completed source ids found in the
existing train/holdout/failures outputs are skipped without relabeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ml.alphazero_lite import exact_teacher_labeling as exact
from ml.alphazero_lite.native_mtdf_probe import payload
from ml.alphazero_lite.run_native_hybrid_feasibility import (
    NativeHybridProcess,
    NativeRequestError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-holdout", type=Path, required=True)
    parser.add_argument("--out-failures", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh-process", action="store_true")
    return parser.parse_args(argv)


def split_membership(
    train_path: Path, holdout_path: Path, failures_path: Path
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    completed: dict[str, str] = {}
    existing: dict[str, dict[str, Any]] = {}
    for path, lane in (
        (train_path, "train"),
        (holdout_path, "holdout"),
        (failures_path, "failures"),
    ):
        if not path.is_file():
            continue
        for row in exact.read_jsonl(path):
            source_id = str(row.get("source_id", ""))
            if not source_id or source_id in completed:
                continue
            completed[source_id] = lane
            existing[source_id] = row
    return completed, existing


def load_sources(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = exact.read_jsonl(path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = load_sources(args.source_states, args.limit)
    train_rows, holdout_rows = exact.deterministic_split(
        sources, train_split=args.train_split
    )
    train_ids = {row["source_id"] for row in train_rows}
    completed, existing = split_membership(
        args.out_train, args.out_holdout, args.out_failures
    )
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    pending = [row for row in sources if row["source_id"] not in completed]
    latencies: list[float] = []
    cpu_latencies: list[float] = []
    solved = sum(1 for lane in completed.values() if lane in ("train", "holdout"))
    failed = sum(1 for lane in completed.values() if lane == "failures")
    reason_counts: dict[str, int] = {}
    for row in existing.values():
        if str(row.get("status", "")) in ("failed", "timeout"):
            reason = str(row.get("error", "unknown"))[:80]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        elif "label_wall_seconds" in row:
            latencies.append(float(row["label_wall_seconds"]))
            cpu_latencies.append(float(row.get("label_cpu_seconds", 0.0)))
    process = (
        None
        if args.fresh_process
        else NativeHybridProcess(args.native_probe, args.artifact)
    )
    teacher_provenance: dict[str, Any] = {
        "identity": exact.TEACHER_IDENTITY,
        "version": exact.TEACHER_VERSION,
        "native_probe_sha256": exact.sha256_file(args.native_probe),
        "tablebase_artifact": str(args.artifact),
        "tablebase_artifact_sha256": exact.sha256_file(args.artifact),
        "solver_timeout_seconds": float(args.timeout),
        "fresh_process": bool(args.fresh_process),
        "policy_target_convention": "uniform_over_exact_optimal_set",
        "value_target_convention": (
            "sign_of_root_perspective_margin_pm1_from_player_zero_margin"
        ),
        "value_semantics": (
            "native exact_value/action_values are player-zero final stone "
            "margins; root extremum max(player0)/min(player1); training value "
            "is sign from root-player perspective (+1 win/0 draw/-1 loss)"
        ),
    }
    try:
        for index, source in enumerate(pending):
            worker = (
                NativeHybridProcess(args.native_probe, args.artifact)
                if args.fresh_process
                else process
            )
            assert worker is not None
            label_started_wall = time.monotonic()
            label_started_cpu = time.process_time()
            try:
                state = exact.exact_state_from_game_state(source["state"])
                result = worker.request(payload(state, "label"), args.timeout)
                repeat = worker.request(payload(state, "label"), args.timeout)
                if (
                    result["action_values"] != repeat["action_values"]
                    or result["optimal_actions"] != repeat["optimal_actions"]
                    or result["exact_value"] != repeat["exact_value"]
                ):
                    raise AssertionError("non-deterministic native label")
                actions = {int(k): int(v) for k, v in result["action_values"].items()}
                row = exact.build_training_row(
                    source=source,
                    action_values=actions,
                    optimal_actions=[int(a) for a in result["optimal_actions"]],
                    exact_value=int(result["exact_value"]),
                    label_wall_seconds=time.monotonic() - label_started_wall,
                    label_cpu_seconds=time.process_time() - label_started_cpu,
                    teacher_provenance=teacher_provenance,
                )
                lane = "train" if source["source_id"] in train_ids else "holdout"
                exact.append_jsonl(
                    args.out_train if lane == "train" else args.out_holdout, row
                )
                completed[source["source_id"]] = lane
                existing[source["source_id"]] = row
                solved += 1
                latencies.append(float(row["label_wall_seconds"]))
                cpu_latencies.append(float(row["label_cpu_seconds"]))
            except (
                TimeoutError,
                NativeRequestError,
                AssertionError,
                ValueError,
            ) as exc:
                status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                failed_row = exact.build_failed_row(
                    source=source,
                    status=status,
                    error=str(exc),
                    teacher_provenance=teacher_provenance,
                )
                exact.append_jsonl(args.out_failures, failed_row)
                completed[source["source_id"]] = "failures"
                existing[source["source_id"]] = failed_row
                failed += 1
                reason = str(exc)[:80]
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if not args.fresh_process:
                    worker.replace()
            finally:
                if args.fresh_process:
                    worker.close()
            if (index + 1) % 500 == 0:
                print(
                    f"labeled {index + 1}/{len(pending)} "
                    f"solved={solved} failed={failed}",
                    flush=True,
                )
    finally:
        if process is not None:
            process.close()
    wall_seconds = time.monotonic() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    train_out = [r for sid, r in existing.items() if completed[sid] == "train"]
    holdout_out = [r for sid, r in existing.items() if completed[sid] == "holdout"]
    failure_out = [r for sid, r in existing.items() if completed[sid] == "failures"]
    train_keys = {str(r["canonical_state"]) for r in train_out}
    holdout_keys = {str(r["canonical_state"]) for r in holdout_out}
    total = len(sources)
    summary: dict[str, Any] = {
        "schema": "native_hybrid_exact_label_production_v1",
        "requested_unique_states": total,
        "solved_labels": solved,
        "failed_labels": failed,
        "solve_rate": (solved / total) if total else 0.0,
        "failure_reason_counts": reason_counts,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "labels_per_cpu_hour": (
            (solved / cpu_seconds * 3600.0) if cpu_seconds > 0 else 0.0
        ),
        "label_latency_wall_seconds": exact.summarize_latencies(latencies),
        "label_latency_cpu_seconds": exact.summarize_latencies(cpu_latencies),
        "train_rows": len(train_out),
        "holdout_rows": len(holdout_out),
        "train_sha256": (
            exact.sha256_file(args.out_train) if args.out_train.is_file() else None
        ),
        "holdout_sha256": (
            exact.sha256_file(args.out_holdout) if args.out_holdout.is_file() else None
        ),
        "failures_sha256": (
            exact.sha256_file(args.out_failures)
            if args.out_failures.is_file()
            else None
        ),
        "source_states_path": str(args.source_states),
        "source_states_sha256": exact.sha256_file(args.source_states),
        "train_holdout_canonical_overlap": len(train_keys & holdout_keys),
        "train_canonical_hash": hashlib.sha256(
            "\n".join(sorted(train_keys)).encode()
        ).hexdigest(),
        "holdout_canonical_hash": hashlib.sha256(
            "\n".join(sorted(holdout_keys)).encode()
        ).hexdigest(),
        "teacher_provenance": teacher_provenance,
        "timeout_seconds": float(args.timeout),
        "train_split": float(args.train_split),
        "complete": len(completed) == total,
    }
    _ = failure_out
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"solved={solved} failed={failed} solve_rate={summary['solve_rate']:.4f}")
    print(f"wall_seconds={wall_seconds:.1f} cpu_seconds={cpu_seconds:.1f}")
    print(f"summary_written={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
