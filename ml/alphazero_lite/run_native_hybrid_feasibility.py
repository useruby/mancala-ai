"""Run the frozen exact-teacher corpus through the native tablebase hybrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any

from ml.alphazero_lite.native_mtdf_probe import payload
from ml.alphazero_lite.exact_kalah_solver import ExactState
from ml.alphazero_lite.run_exact_solver_feasibility_preflight import (
    BUCKETS,
    DEFAULT_SEED,
    generate_feasibility_corpus,
)


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_CORPUS = (
    ROOT / "docs/data/alphazero-lite-kalah-v1-tier18-hybrid-feasibility.json"
)


class NativeRequestError(RuntimeError):
    pass


class NativeHybridProcess:
    """One JSONL native process; failed requests never leak into the next one."""

    def __init__(self, executable: Path, artifact: Path) -> None:
        self.executable = executable
        self.artifact = artifact
        self.process: subprocess.Popen[str] | None = None
        self.start()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [str(self.executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ | {"NATIVE_CANONICAL_KVTB": str(self.artifact)},
        )

    def request(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise NativeRequestError("native process is not running")
        assert self.process.stdin and self.process.stdout
        started = time.monotonic()
        self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(max(0.0, timeout - (time.monotonic() - started))):
                raise TimeoutError("native request exceeded timeout")
            line = self.process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise NativeRequestError("native process closed its JSONL stream")
        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            raise NativeRequestError(f"malformed native JSON: {error}") from error
        if "error" in result:
            raise NativeRequestError(result["error"])
        return result

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process = None

    def replace(self) -> None:
        self.close()
        self.start()


def canonical_inputs(corpus: list[dict[str, Any]]) -> bytes:
    rows = [
        {
            key: row[key]
            for key in ("id", "state", "stones_remaining", "training_eligible")
        }
        for row in corpus
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


def verify_frozen_corpus(corpus: list[dict[str, Any]]) -> str:
    if len(corpus) != 96 or any(row["training_eligible"] for row in corpus):
        raise AssertionError("frozen corpus must contain 96 ineligible rows")
    for low, high in BUCKETS:
        if sum(low <= row["stones_remaining"] <= high for row in corpus) != 32:
            raise AssertionError(f"frozen corpus bucket {low}-{high} is not 32 states")
    historical = json.loads(HISTORICAL_CORPUS.read_text(encoding="utf-8"))["corpus"]
    if [row["state"] for row in corpus] != [row["state"] for row in historical]:
        raise AssertionError("generated corpus differs from the frozen PR-era inputs")
    return hashlib.sha256(canonical_inputs(corpus)).hexdigest()


def _result_row(
    source: dict[str, Any], result: dict[str, Any] | None, error: str | None
) -> dict[str, Any]:
    row = source | {
        "training_eligible": False,
        "error": error,
        "exact": result is not None,
    }
    if result is None:
        return row | {
            "reproducible": False,
            "final_margin": None,
            "action_final_margins": {},
            "optimal_actions": [],
        }
    actions = {int(key): value for key, value in result["action_values"].items()}
    player = source["state"]["current_player"]
    root = max(actions.values()) if player == 0 else min(actions.values())
    if (
        root != result["exact_value"]
        or sorted(action for action, value in actions.items() if value == root)
        != result["optimal_actions"]
    ):
        raise AssertionError(f"native root/action inconsistency for {source['id']}")
    metrics = result["metrics"]
    metrics["previous_state_tt_influence"] = (
        metrics["cumulative_cache"]["tt_hits"] > metrics["tt_hits"]
    )
    return row | {
        "reproducible": True,
        "final_margin": root,
        "action_final_margins": actions,
        "optimal_actions": result["optimal_actions"],
        "metrics": metrics,
    }


def run_mode(
    corpus: list[dict[str, Any]],
    executable: Path,
    artifact: Path,
    timeout: float,
    fresh: bool,
    existing: dict[str, Any],
) -> list[dict[str, Any]]:
    completed = {row["id"]: row for row in existing.get("corpus", [])}
    process = None if fresh else NativeHybridProcess(executable, artifact)
    try:
        for source in corpus:
            if source["id"] in completed:
                continue
            worker = NativeHybridProcess(executable, artifact) if fresh else process
            assert worker is not None
            result = None
            error = None
            started = time.monotonic()
            try:
                state = ExactState.from_game_state(source["state"])
                result = worker.request(payload(state, "label"), timeout)
                repeat = worker.request(payload(state, "label"), timeout)
                if (
                    result["action_values"] != repeat["action_values"]
                    or result["optimal_actions"] != repeat["optimal_actions"]
                ):
                    raise AssertionError("non-deterministic native label")
                result["metrics"]["request_wall_time_seconds"] = (
                    time.monotonic() - started
                )
            except (TimeoutError, NativeRequestError, AssertionError) as exc:
                error = str(exc)
                if not fresh:
                    worker.replace()
            finally:
                if fresh:
                    worker.close()
            completed[source["id"]] = _result_row(source, result, error)
            existing["corpus"] = [
                completed[row["id"]] for row in corpus if row["id"] in completed
            ]
            yield_rows = existing["on_update"]
            yield_rows(existing)
    finally:
        if process is not None:
            process.close()
    return [completed[row["id"]] for row in corpus]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    corpus = generate_feasibility_corpus(DEFAULT_SEED)
    digest = verify_frozen_corpus(corpus)
    report: dict[str, Any] = {
        "schema": "native_hybrid_feasibility_v1",
        "seed": DEFAULT_SEED,
        "timeout_seconds": args.timeout,
        "mode": "fresh" if args.fresh else "warm",
        "input_sha256": digest,
        "corpus": [],
    }

    def persist(value: dict[str, Any]) -> None:
        value.pop("on_update", None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        value["on_update"] = persist

    report["on_update"] = persist
    rows = run_mode(
        corpus, args.native_probe, args.artifact, args.timeout, args.fresh, report
    )
    report["corpus"] = rows
    report.pop("on_update", None)
    report["complete"] = len(rows) == 96
    persist(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
