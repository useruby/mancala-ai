"""Protocol-correction rerun for the frozen seed420 memoization cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.memoized_evaluator import MemoizedEvaluator
from ml.alphazero_lite.seed420_experiment import search, verify_artifact
from ml.alphazero_lite.seed421_analysis import analyze, seed_for, validate_ledger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed421-memoization-timing-correction"
OLD = ROOT / "docs/data/seed420-artifact-evaluator-memoization"
ARTIFACT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
SOURCE_FILES = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/eval_cache.py",
    "ml/alphazero_lite/memoized_evaluator.py",
    "ml/alphazero_lite/seed420_experiment.py",
    "ml/alphazero_lite/seed421_experiment.py",
    "ml/alphazero_lite/seed421_analysis.py",
    "ml/alphazero_lite/seed421_verify.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CountingEvaluator:
    """Unt instrumented baseline: forward each call and increment one integer."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = 0

    def evaluate(self, game):
        self.calls += 1
        return self.evaluator.evaluate(game)


def freeze() -> None:
    if DATA.exists():
        raise FileExistsError("seed421_registration_refuses_overwrite")
    verify_artifact()
    original = json.loads((OLD / "registration.json").read_text())
    cohort_bytes = (OLD / "cohort.json").read_bytes()
    DATA.mkdir(parents=True)
    (DATA / "cohort.json").write_bytes(cohort_bytes)
    (DATA / "cohort-inputs").mkdir()
    for name in (
        "seed416-openings.jsonl",
        "seed416-outcome-ledger.jsonl",
        "seed416-registration-v3.json",
        "seed416-evaluation-binding.json",
    ):
        (DATA / "cohort-inputs" / name).write_bytes(
            (OLD / "cohort-inputs" / name).read_bytes()
        )
    snapshots = DATA / "execution-source-snapshots"
    for relative in SOURCE_FILES:
        dst = snapshots / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / relative).read_bytes())
    registration = {
        "schema": "seed421-memoization-registration-v1",
        "status": "frozen-before-searches",
        "protocol": "correction rerun of seed420; not a sample extension or capacity sweep",
        "seed420_registration_sha256": sha(OLD / "registration.json"),
        "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        "source_hashes": original["source_hashes"],
        "seed455_artifact_files": original["seed455_artifact_files"],
        "artifact_identity": original["artifact_identity"],
        "search": {k: v for k, v in original["search"].items() if k != "commands"},
        "decision_rules": original["decision_rules"],
        "execution_source_hashes": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "timing_scope": "external complete-search wall time; tracing disabled in both timed conditions",
    }
    (DATA / "registration.json").write_text(
        json.dumps(registration, sort_keys=True, indent=2) + "\n"
    )


def _parity(root, budget, identity, evaluator):
    seed = seed_for(root["state_hash"], budget, -1)
    off_model = TraceEvaluator(evaluator)
    on = MemoizedEvaluator(
        evaluator, artifact_identity=identity, input_encoding=evaluator.input_encoding
    )
    on.request_trace = []
    off = search(off_model, root["state"], budget, seed)
    cached = search(on, root["state"], budget, seed)
    if off != cached:
        raise ValueError("parity_output_mismatch")
    on_trace = [
        {k: v for k, v in row.items() if k != "cache_hit"} for row in on.request_trace
    ]
    if off_model.trace != on_trace:
        raise ValueError("parity_request_mismatch")
    for item in off_model.trace:
        item["root_identity"] = root["state_hash"]
    for item in on_trace:
        item["root_identity"] = root["state_hash"]
    return {
        "off": off,
        "on": cached,
        "evaluation_requests": len(off_model.trace),
        "off_request_trace": off_model.trace,
        "on_request_trace": on_trace,
        "neural_calls_off": off_model.calls,
        "neural_calls_on": on.neural_calls,
        "cache": on.cache_stats,
    }


class TraceEvaluator:
    def __init__(self, evaluator):
        self.evaluator, self.calls, self.trace = evaluator, 0, []

    def evaluate(self, game):
        policy, value = self.evaluator.evaluate(game)
        state = json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"))
        self.trace.append(
            {
                "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                "policy": policy.tolist(),
                "value": float(value),
            }
        )
        self.calls += 1
        return policy, value


def append(record):
    with (DATA / "benchmark-ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run():
    reg_path = DATA / "registration.json"
    registration = json.loads(reg_path.read_text())
    reg_sha = sha(reg_path)
    if sha(DATA / "cohort.json") != registration["cohort_sha256"]:
        raise ValueError("cohort_hash_mismatch")
    for relative, expected in registration["execution_source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"execution_source_changed:{relative}")
    verify_artifact()
    cohort = json.loads((DATA / "cohort.json").read_text())
    evaluator = ArtifactEvaluator(ARTIFACT)
    evaluator.evaluate(KalahGame.from_state(cohort[0]["state"]))
    ledger_path = DATA / "benchmark-ledger.jsonl"
    rows = (
        [json.loads(line) for line in ledger_path.read_text().splitlines()]
        if ledger_path.exists()
        else []
    )
    validate_ledger(rows, cohort, reg_sha, complete=False)
    completed = {
        (r["root_hash"], r["budget"], r["kind"], r["repetition"]) for r in rows
    }
    identity = registration["artifact_identity"]
    for root in cohort:
        for budget in (384, 1536):
            parity_key = (root["state_hash"], budget, "parity", 0)
            if parity_key not in completed:
                result = _parity(root, budget, identity, evaluator)
                append(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "parity",
                        "repetition": 0,
                        "seed": seed_for(root["state_hash"], budget, -1),
                        "registration_sha256": reg_sha,
                        "result": result,
                    }
                )
            for repetition in range(3):
                key = (root["state_hash"], budget, "timed", repetition)
                if key in completed:
                    continue
                seed = seed_for(root["state_hash"], budget, repetition)
                order = ["off", "on"] if repetition % 2 == 0 else ["on", "off"]
                outputs, times, cache, requests, calls = {}, {}, {}, {}, {}
                for condition in order:
                    if condition == "off":
                        measured = CountingEvaluator(evaluator)
                    else:
                        measured = MemoizedEvaluator(
                            evaluator,
                            artifact_identity=identity,
                            input_encoding=evaluator.input_encoding,
                        )
                    started = time.perf_counter_ns()
                    outputs[condition] = search(measured, root["state"], budget, seed)
                    times[condition] = (time.perf_counter_ns() - started) / 1e6
                    if condition == "off":
                        requests[condition] = calls[condition] = measured.calls
                    else:
                        requests[condition], calls[condition], cache = (
                            measured.requests,
                            measured.neural_calls,
                            measured.cache_stats,
                        )
                if outputs["off"] != outputs["on"]:
                    raise ValueError("timed_output_mismatch")
                append(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "timed",
                        "repetition": repetition,
                        "seed": seed,
                        "registration_sha256": reg_sha,
                        "execution_order": order,
                        "latency_ms": times,
                        "result": outputs,
                        "cache": cache,
                        "request_counts": requests,
                        "neural_call_counts": calls,
                    }
                )
                rows.append(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "timed",
                        "repetition": repetition,
                        "seed": seed,
                        "registration_sha256": reg_sha,
                        "execution_order": order,
                        "latency_ms": times,
                        "result": outputs,
                        "cache": cache,
                        "request_counts": requests,
                        "neural_call_counts": calls,
                    }
                )


def publish():
    cohort = json.loads((DATA / "cohort.json").read_text())
    rows = [
        json.loads(line)
        for line in (DATA / "benchmark-ledger.jsonl").read_text().splitlines()
    ]
    report = analyze(rows, cohort, sha(DATA / "registration.json"))
    report.update(
        {
            "ledger_sha256": sha(DATA / "benchmark-ledger.jsonl"),
            "registration_sha256": sha(DATA / "registration.json"),
        }
    )
    (DATA / "analysis.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "analyze"))
    command = parser.parse_args().command
    {"freeze": freeze, "run": run, "analyze": publish}[command]()


if __name__ == "__main__":
    main()
