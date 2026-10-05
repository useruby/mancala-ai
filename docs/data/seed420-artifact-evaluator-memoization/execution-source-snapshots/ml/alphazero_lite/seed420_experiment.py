"""Frozen seed420 memoization benchmark runner and integrity-bound accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.memoized_evaluator import MemoizedEvaluator
from ml.alphazero_lite.seed420_analysis import analyze
from ml.alphazero_lite.seed420_cohort import derive_cohort
from ml.alphazero_lite.self_play import PUCT


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed420-artifact-evaluator-memoization"
ARTIFACT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
SUITE = ROOT / "docs/data/seed416-policy-target-softening/openings-v3.jsonl"
GAMES = ROOT / "docs/data/seed416-policy-target-softening/outcome-ledger.jsonl"
SOURCE_FILES = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/eval_cache.py",
    "ml/alphazero_lite/memoized_evaluator.py",
    "ml/alphazero_lite/seed420_cohort.py",
    "ml/alphazero_lite/seed420_analysis.py",
    "ml/alphazero_lite/seed420_experiment.py",
    "ml/alphazero_lite/seed420_verify.py",
)
EXPECTED_ARTIFACT_FILES = {
    "weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
    "metadata.json": "8b9f4d02395271accd5accb5cde20ec6ddc4ecd63ded192b0b84d9ecf6c649ac",
    "search_policy.json": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
}
CAPACITY = 4096
BUDGETS = (384, 1536)
REPETITIONS = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def identity_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_artifact() -> dict[str, str]:
    actual = {name: sha256(ARTIFACT / name) for name in EXPECTED_ARTIFACT_FILES}
    if actual != EXPECTED_ARTIFACT_FILES:
        raise ValueError("seed455_artifact_binding_mismatch")
    return actual


def freeze() -> None:
    if DATA.exists():
        raise FileExistsError("seed420_registration_refuses_overwrite")
    artifact_files = verify_artifact()
    suite = jsonl(SUITE)
    games = jsonl(GAMES)
    cohort = derive_cohort(suite, games)
    source_hashes = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    seed_contexts = [
        {
            "root_hash": root["state_hash"],
            "budget": budget,
            "parity_seed": seed_for(root["state_hash"], budget, -1),
            "timed_seeds": [
                seed_for(root["state_hash"], budget, repetition)
                for repetition in range(REPETITIONS)
            ],
        }
        for root in cohort
        for budget in BUDGETS
    ]
    source_values = {
        "suite": sha256(SUITE),
        "trajectory_ledger": sha256(GAMES),
        "seed416_registration": sha256(
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ),
        "evaluation_binding": sha256(
            ROOT / "docs/data/seed416-policy-target-softening/evaluation-binding.json"
        ),
    }
    DATA.mkdir(parents=True)
    snapshots = DATA / "execution-source-snapshots"
    snapshots.mkdir()
    for relpath in SOURCE_FILES:
        destination = snapshots / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relpath).read_bytes())
    inputs = DATA / "cohort-inputs"
    inputs.mkdir()
    (inputs / "seed416-openings.jsonl").write_bytes(SUITE.read_bytes())
    (inputs / "seed416-outcome-ledger.jsonl").write_bytes(GAMES.read_bytes())
    (inputs / "seed416-registration-v3.json").write_bytes(
        (
            ROOT / "docs/data/seed416-policy-target-softening/registration-v3.json"
        ).read_bytes()
    )
    (inputs / "seed416-evaluation-binding.json").write_bytes(
        (
            ROOT / "docs/data/seed416-policy-target-softening/evaluation-binding.json"
        ).read_bytes()
    )
    cohort_bytes = (json.dumps(cohort, sort_keys=True, indent=2) + "\n").encode()
    (DATA / "cohort.json").write_bytes(cohort_bytes)
    registration = {
        "schema": "seed420-memoization-registration-v1",
        "status": "frozen-before-searches",
        "interpretation": "retrospective performance cohort; not a strength holdout",
        "source_hashes": source_values,
        "execution_source_hashes": source_hashes,
        "seed455_artifact_files": artifact_files,
        "artifact_identity": identity_hash(artifact_files),
        "cohort_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        "selection": {
            "source": "#416 A lane; challenger seat zero; replay absolute trajectory actions",
            "eligibility": "nonterminal with at least two legal actions; >32 then 17-32 active pit stones",
            "count_per_phase": 16,
            "deduplication": "canonical complete state includes both pit rows, both stores, and current_player",
            "ordering": "ascending SHA256('420:'+state_hash), then state_hash",
            "opening_rule": "at most one root per opening globally; >32 phase selected first",
        },
        "search": {
            "budgets": list(BUDGETS),
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
            "dirichlet_noise": False,
            "exact_leaf_solve": False,
            "exact_root_contract": "roots are selected under the frozen seed455 root-16 contract; PUCT leaves are not solved",
            "cache_capacity": CAPACITY,
            "instrumented_pairs_per_root_budget": 1,
            "timed_pairs_per_root_budget": REPETITIONS,
            "timed_order": "rep 0 off-on, rep 1 on-off, rep 2 off-on",
            "rng_seed": "int(SHA256('420:rng:'+state_hash+':'+budget+':'+rep)[:16],16)",
            "seed_contexts": seed_contexts,
            "commands": [
                "PYTHONPATH=. python -m ml.alphazero_lite.seed420_experiment freeze",
                "PYTHONPATH=. python -m ml.alphazero_lite.seed420_experiment run",
                "PYTHONPATH=. python -m ml.alphazero_lite.seed420_experiment analyze",
            ],
        },
        "decision_rules": {
            "equivalence_required": True,
            "advance": "all equivalence checks pass; 384 mean paired relative speedup >= 0.15 and root-cluster bootstrap 95% lower bound > 0; neither phase speedup < -0.05; 1536 mean paired relative speedup >= 0",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 420,
            "cluster": "root; repetitions remain paired within root",
        },
        "reported_metrics": [
            "cache hits, misses, evictions, underlying calls, request counts, peak entries",
            "per-root median latency across repetitions, paired relative speedup, arithmetic mean latency, nearest-rank p95",
            "10,000 root-cluster bootstrap resamples with seed 420",
        ],
        "timing_scope": "external wall time around complete PUCT root search, including PUCT construction, state conversion, cache reset, key serialization, lookup, and search; tracing is outside timed calls",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_model": cpu_model,
            "numpy": np.__version__,
            "thread_settings": {
                key: os.environ.get(key, "unset")
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "warmup": "artifact load, numerical runtime import, and one network evaluation before timed runs",
        },
    }
    (DATA / "registration.json").write_text(
        json.dumps(registration, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


class RecordingEvaluator:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.requests: list[dict] = []
        self.calls = 0

    def reset_telemetry(self):
        self.requests.clear()
        self.calls = 0

    def evaluate(self, game):
        policy, value = self.evaluator.evaluate(game)
        state_text = json.dumps(game.to_state(), sort_keys=True, separators=(",", ":"))
        self.requests.append(
            {
                "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
                "policy": np.asarray(policy).tolist(),
                "value": float(value),
            }
        )
        self.calls += 1
        return policy, value


def seed_for(state_hash: str, budget: int, repetition: int) -> int:
    text = f"420:rng:{state_hash}:{budget}:{repetition}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def search(evaluator, root_state: dict, budget: int, seed: int) -> dict:
    engine = PUCT(
        evaluator,
        simulations=budget,
        c_puct=1.25,
        rng=random.Random(seed),
        fpu_mode="zero",
        root_fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, root = engine.run(KalahGame.from_state(root_state))
    summary = engine.root_summary()
    total = float(visits.sum())
    policy = (visits / total).tolist() if total else visits.tolist()
    return {
        "selected_action": int(summary["selected_move"]),
        "visits": [int(value) for value in visits],
        "policy": policy,
        "q_values": {
            str(row["move"]): float(row["q_value"]) for row in summary["child_stats"]
        },
        "root_visit_count": int(root.visit_count),
        "root_q": float(root.q_value),
    }


def parity(root: dict, budget: int, identity: str, evaluator) -> dict:
    seed = seed_for(root["state_hash"], budget, -1)
    off_model = RecordingEvaluator(evaluator)
    wrapped = MemoizedEvaluator(
        evaluator,
        artifact_identity=identity,
        input_encoding=evaluator.input_encoding,
    )
    wrapped.request_trace = []
    off = search(off_model, root["state"], budget, seed)
    on = search(wrapped, root["state"], budget, seed)
    if off != on:
        raise ValueError("instrumented_search_output_mismatch")
    normalized_on_trace = [
        {key: value for key, value in row.items() if key != "cache_hit"}
        for row in wrapped.request_trace
    ]
    if off_model.requests != normalized_on_trace:
        raise ValueError("instrumented_evaluation_request_mismatch")
    return {
        "off": off,
        "on": on,
        "evaluation_requests": len(off_model.requests),
        "off_request_trace": off_model.requests,
        "on_request_trace": normalized_on_trace,
        "neural_calls_off": off_model.calls,
        "neural_calls_on": wrapped.neural_calls,
        "cache": wrapped.cache_stats,
    }


def append_record(record: dict) -> None:
    ledger = DATA / "benchmark-ledger.jsonl"
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def record_correctness_failure(
    root: dict, budget: int, kind: str, error: Exception
) -> None:
    path = DATA / "correctness-failure.json"
    payload = {
        "schema": "seed420-correctness-failure-v1",
        "root_hash": root["state_hash"],
        "phase": root["phase"],
        "budget": budget,
        "pair_kind": kind,
        "error": str(error),
        "decision": "stop_memoization_branch",
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run() -> None:
    if (DATA / "correctness-failure.json").exists():
        raise ValueError("registered_correctness_failure_exists")
    reg_path = DATA / "registration.json"
    if not reg_path.exists():
        raise FileNotFoundError("seed420_registration_required")
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    verify_artifact()
    for relpath, expected in registration["execution_source_hashes"].items():
        if sha256(ROOT / relpath) != expected:
            raise ValueError(f"execution_source_changed:{relpath}")
    cohort = json.loads((DATA / "cohort.json").read_text(encoding="utf-8"))
    if (
        hashlib.sha256((DATA / "cohort.json").read_bytes()).hexdigest()
        != registration["cohort_sha256"]
    ):
        raise ValueError("frozen_cohort_hash_mismatch")
    evaluator = ArtifactEvaluator(ARTIFACT)
    wrapped_identity = registration["artifact_identity"]
    # Warm the loaded network before any latency measurement.
    evaluator.evaluate(KalahGame.from_state(cohort[0]["state"]))
    existing = DATA / "benchmark-ledger.jsonl"
    completed = set()
    if existing.exists():
        for row in jsonl(existing):
            key = (row["root_hash"], row["budget"], row["kind"], row["repetition"])
            if key in completed:
                raise ValueError("duplicate_ledger_pair")
            completed.add(key)
    for root in cohort:
        for budget in BUDGETS:
            key = (root["state_hash"], budget, "parity", 0)
            if key not in completed:
                try:
                    result = parity(root, budget, wrapped_identity, evaluator)
                except Exception as error:
                    record_correctness_failure(root, budget, "parity", error)
                    raise
                append_record(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "parity",
                        "repetition": 0,
                        "seed": seed_for(root["state_hash"], budget, -1),
                        "result": result,
                    }
                )
            for repetition in range(REPETITIONS):
                key = (root["state_hash"], budget, "timed", repetition)
                if key in completed:
                    continue
                seed = seed_for(root["state_hash"], budget, repetition)
                order = ("off", "on") if repetition % 2 == 0 else ("on", "off")
                outputs = {}
                times = {}
                cache_stats = {}
                request_counts = {}
                neural_call_counts = {}
                for condition in order:
                    if condition == "off":
                        measured = RecordingEvaluator(evaluator)
                    else:
                        measured = MemoizedEvaluator(
                            evaluator,
                            artifact_identity=wrapped_identity,
                            input_encoding=evaluator.input_encoding,
                        )
                    started = time.perf_counter_ns()
                    outputs[condition] = search(measured, root["state"], budget, seed)
                    times[condition] = (time.perf_counter_ns() - started) / 1e6
                    if condition == "on":
                        cache_stats = getattr(measured, "cache_stats")
                        request_counts[condition] = measured.requests
                        neural_call_counts[condition] = measured.neural_calls
                    else:
                        request_counts[condition] = measured.calls
                        neural_call_counts[condition] = getattr(measured, "calls")
                if outputs["off"] != outputs["on"]:
                    error = ValueError("timed_search_output_mismatch")
                    record_correctness_failure(root, budget, "timed", error)
                    raise error
                append_record(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "timed",
                        "repetition": repetition,
                        "seed": seed,
                        "execution_order": list(order),
                        "latency_ms": times,
                        "result": outputs,
                        "cache": cache_stats,
                        "request_counts": request_counts,
                        "neural_call_counts": neural_call_counts,
                    }
                )


def publish_analysis() -> dict:
    registration = json.loads((DATA / "registration.json").read_text(encoding="utf-8"))
    ledger = jsonl(DATA / "benchmark-ledger.jsonl")
    cohort = json.loads((DATA / "cohort.json").read_text(encoding="utf-8"))
    report = analyze(ledger, cohort)
    report["ledger_sha256"] = sha256(DATA / "benchmark-ledger.jsonl")
    report["registration_sha256"] = sha256(DATA / "registration.json")
    report["execution_order"] = registration["search"]["timed_order"]
    (DATA / "analysis.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "analyze"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        run()
    else:
        publish_analysis()


if __name__ == "__main__":
    main()
