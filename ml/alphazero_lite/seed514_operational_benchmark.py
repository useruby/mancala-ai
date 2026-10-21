"""Seed514 local operational benchmark for the frozen P composition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/data/seed512-fixed-composition-evaluation"
OUT = ROOT / "docs/data/seed514-operational-benchmark-v2"
REGISTRATION = SOURCE / "registration.json"
_REGISTRATION = json.loads(REGISTRATION.read_text())
_SEARCH_OPTIONS = dict(_REGISTRATION["runtime"]["search_options"])
SIMULATIONS = 1200
REPETITIONS = 5
WARMUPS = 1
CPUCT = 1.25
COHORT_COUNTS = {"gt32": 32, "17to32": 16, "le16": 16}
TEST_SOURCE = ROOT / "ml/alphazero_lite/test_seed514_operational_benchmark.py"
VERIFY_SOURCE = ROOT / "ml/alphazero_lite/verify_seed514_operational_benchmark.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def state_id(state: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(state).encode()).hexdigest()


def verify_registered_identities(registration: dict[str, Any]) -> dict[str, Any]:
    """Verify component files and exact-root runtime assets against seed512."""
    verified: dict[str, Any] = {"components": {}, "runtime_assets": {}}
    for name, component in registration["components"].items():
        artifact = ROOT / component["artifact"]
        actual = {
            filename: sha256(artifact / filename) for filename in component["files"]
        }
        if actual != component["files"]:
            raise ValueError(f"seed514_component_identity:{name}")
        verified["components"][name] = actual
    for name, asset in registration["runtime_assets"].items():
        if name == "exact_root_tablebase":
            pointer = ROOT / asset["artifact"]
            materialized = Path(asset["runtime_materialized_path"])
            valid = (
                sha256(pointer) == asset["source_pointer_sha256"]
                and sha256(materialized) == asset["sha256"]
            )
            resolved_path = materialized
        else:
            resolved_path = ROOT / asset["artifact"]
            valid = sha256(resolved_path) == asset["sha256"]
        if not valid:
            raise ValueError(f"seed514_runtime_asset_identity:{name}")
        verified["runtime_assets"][name] = {
            "path": str(resolved_path),
            "sha256": asset["sha256"],
        }
    return verified


def stone_bucket(state: dict[str, Any]) -> str:
    stones = sum(state["player_pits"]) + sum(state["opponent_pits"])
    if stones > 32:
        return "gt32"
    if stones >= 17:
        return "17to32"
    return "le16"


def reconstruct_cohort(source: Path = SOURCE) -> list[dict[str, Any]]:
    """Select unique nonterminal trajectory states in hash order, stratified by stones."""
    candidates: dict[str, dict[str, Any]] = {}
    files = sorted((source / "batches").glob("games-*.jsonl"))
    if not files:
        raise ValueError("seed514_seed512_game_batches_missing")
    for path in files:
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            game = KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            prefix = row.get("opening_prefix_moves", [])
            trajectory = [
                int(move) for move in str(row["trajectory"]).split(",") if move
            ]
            for prefix_move in prefix:
                if game.over():
                    break
                if not game.move(game.pit_index(int(prefix_move))):
                    raise ValueError(
                        f"seed514_invalid_archived_prefix:{row['game_index']}"
                    )
            for trajectory_ply, absolute_move in enumerate(trajectory):
                if game.over():
                    break
                state = game.to_state()
                identity = state_id(state)
                bucket = stone_bucket(state)
                if bucket in COHORT_COUNTS:
                    source_ply = len(prefix) + trajectory_ply
                    reference = {
                        "source_game_index": int(row["game_index"]),
                        "source_opening_index": int(row["opening_index"]),
                        "ply": source_ply,
                    }
                    order = hashlib.sha256(
                        f"seed514-cohort-v1:{identity}:{reference['source_game_index']}:{source_ply}".encode()
                    ).hexdigest()
                    previous = candidates.get(identity)
                    item = {
                        "state": state,
                        "state_sha256": identity,
                        "bucket": bucket,
                        "selection_order_sha256": order,
                        **reference,
                    }
                    if previous is None or order < previous["selection_order_sha256"]:
                        candidates[identity] = item
                if not game.move(absolute_move):
                    raise ValueError(
                        f"seed514_invalid_archived_move:{row['game_index']}:{source_ply}"
                    )
    selected: list[dict[str, Any]] = []
    for bucket, count in COHORT_COUNTS.items():
        rows = sorted(
            (item for item in candidates.values() if item["bucket"] == bucket),
            key=lambda item: (item["selection_order_sha256"], item["state_sha256"]),
        )
        if len(rows) < count:
            raise ValueError(f"seed514_cohort_shortfall:{bucket}:{len(rows)}:{count}")
        selected.extend(rows[:count])
    return selected


def configs(registration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifact_root = (
        ROOT / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"
    )
    baseline = artifact_root / "initializer"
    policy_artifact = artifact_root / "E4"
    start = time.perf_counter_ns()
    baseline_evaluator = arena.ArtifactEvaluator(baseline)
    baseline_load_seconds = (time.perf_counter_ns() - start) / 1e9
    start = time.perf_counter_ns()
    policy_evaluator = arena.ArtifactEvaluator(policy_artifact)
    composed_evaluator = arena.ComposedArtifactEvaluator(
        baseline_evaluator,
        policy_evaluator,
        policy_source="candidate",
        value_source="current",
    )
    p_load_seconds = (time.perf_counter_ns() - start) / 1e9
    return {
        "baseline": {
            "evaluator": baseline_evaluator,
            "artifact_path": baseline,
            "policy_source": "initializer",
            "value_source": "initializer",
            "loading_seconds": baseline_load_seconds,
        },
        "P": {
            "evaluator": composed_evaluator,
            "artifact_path": baseline,
            "policy_source": "E4",
            "value_source": "initializer",
            "loading_seconds": baseline_load_seconds + p_load_seconds,
            "incremental_loading_seconds": p_load_seconds,
        },
    }


def run_search(
    state: dict[str, Any], config: dict[str, Any], seed: int, *, timing: bool = True
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter_ns()
    result = arena.evaluate_artifact_position(
        evaluator=config["evaluator"],
        artifact_path=config["artifact_path"],
        state=state,
        simulations=SIMULATIONS,
        seed=seed,
        c_puct=CPUCT,
        search_options=_SEARCH_OPTIONS,
        exact_root_solve_threshold=16,
        exact_solve_stone_threshold=None,
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9 if timing else 0.0
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    if result["legal_moves"] != legal or result["selected_move"] not in legal:
        raise ValueError("seed514_illegal_search_result")
    return result, elapsed


def repeatability_signature(result: dict[str, Any]) -> dict[str, Any]:
    """Drop wall-clock telemetry while retaining all deterministic search outputs."""

    def strip_telemetry(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_telemetry(item)
                for key, item in value.items()
                if "latency" not in key.lower()
                and "timing" not in key.lower()
                and not key.lower().endswith("_time_ms")
            }
        if isinstance(value, list):
            return [strip_telemetry(item) for item in value]
        return value

    return strip_telemetry(result)


def paired_ratio_rows(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in measurements:
        key = (row["state_sha256"], row["repetition"], row["seed"])
        by_key.setdefault(key, {})[row["config"]] = float(row["seconds"])
    output = []
    for (identity, repetition, seed), pair in sorted(by_key.items()):
        if set(pair) != {"baseline", "P"}:
            raise ValueError("seed514_incomplete_paired_measurement")
        output.append(
            {
                "state_sha256": identity,
                "repetition": repetition,
                "seed": seed,
                "baseline_seconds": pair["baseline"],
                "P_seconds": pair["P"],
                "P_over_baseline": pair["P"] / pair["baseline"],
            }
        )
    return output


def per_state_ratios(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in measurements:
        grouped.setdefault((row["state_sha256"], row["config"]), []).append(
            float(row["seconds"])
        )
    identities = sorted({key[0] for key in grouped})
    return [
        {
            "state_sha256": identity,
            "baseline_median_seconds": statistics.median(
                grouped[(identity, "baseline")]
            ),
            "P_median_seconds": statistics.median(grouped[(identity, "P")]),
            "P_over_baseline": statistics.median(grouped[(identity, "P")])
            / statistics.median(grouped[(identity, "baseline")]),
        }
        for identity in identities
    ]


def summarize(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = paired_ratio_rows(measurements)
    summary: dict[str, Any] = {
        "configs": {},
        "paired_latency_ratios": ratios,
        "per_state_latency_ratios": per_state_ratios(measurements),
    }
    for name in ("baseline", "P"):
        values = [
            float(row["seconds"]) for row in measurements if row["config"] == name
        ]
        summary["configs"][name] = {
            "searches": len(values),
            "p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)),
            "throughput_searches_per_second": len(values) / sum(values),
        }
    ratio_values = [row["P_over_baseline"] for row in ratios]
    summary["paired_ratio_p50"] = float(np.percentile(ratio_values, 50))
    summary["paired_ratio_p95"] = float(np.percentile(ratio_values, 95))
    summary["search_path_counts"] = {
        name: {
            "exact_root": sum(
                row["exact_root"] for row in measurements if row["config"] == name
            ),
            "puct": sum(
                not row["exact_root"] for row in measurements if row["config"] == name
            ),
        }
        for name in ("baseline", "P")
    }
    return summary


def memory_measurement() -> dict[str, Any]:
    """Measure process RSS separately, without tracing or profiling searches."""
    status = Path("/proc/self/status").read_text()
    rss_kib = next(
        int(line.split()[1])
        for line in status.splitlines()
        if line.startswith("VmRSS:")
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 if sys.platform != "darwin" else 1
    return {
        "combined_models_rss_bytes": rss_kib * 1024,
        "process_peak_rss_bytes": int(peak * scale),
    }


def freeze_protocol(output: Path = OUT) -> dict[str, Any]:
    """Freeze selected states and finalized source/runtime identities before timing."""
    output.mkdir(parents=True, exist_ok=True)
    cohort = reconstruct_cohort()
    cohort_path = output / "cohort.json"
    cohort_path.write_text(json.dumps(cohort, indent=2, sort_keys=True) + "\n")
    archive_files = sorted((SOURCE / "batches").glob("games-*.jsonl"))
    payload = {
        "schema": "seed514-protocol-freeze-v1",
        "cohort_sha256": sha256(cohort_path),
        "selection": "unique nonterminal trajectory states; bucket counts gt32=32, 17to32=16, le16=16; sort by SHA256(seed514-cohort-v1:<state_sha256>:<game_index>:<ply>) then state identity; no outcomes/model outputs",
        "seed512_archive": {
            "registration_sha256": sha256(REGISTRATION),
            "game_batches": {
                str(path.relative_to(ROOT)): sha256(path) for path in archive_files
            },
        },
        "artifacts": verify_registered_identities(_REGISTRATION),
        "protocol": {
            "simulations": SIMULATIONS,
            "c_puct": CPUCT,
            "workers": 1,
            "warmups_per_config_state": WARMUPS,
            "measured_repetitions": REPETITIONS,
            "seed_contract": "514000 plus cohort order index, shared by baseline and P",
            "measured_order": "baseline,P when (state_index+repetition) is even, else P,baseline",
            "root_dispatch": "evaluate_artifact_position; exact_root_solve_threshold=16; exact leaf solve disabled",
            "timing": "perf_counter_ns around complete evaluate_artifact_position call, including exact-root dispatch",
            "memory": "process RSS sampled separately outside timed searches; no allocation tracing or profiler",
        },
        "source_sha256": {
            "ml/alphazero_lite/seed514_operational_benchmark.py": sha256(
                Path(__file__)
            ),
            "ml/alphazero_lite/test_seed514_operational_benchmark.py": sha256(
                TEST_SOURCE
            ),
            "ml/alphazero_lite/verify_seed514_operational_benchmark.py": sha256(
                VERIFY_SOURCE
            ),
            "ml/alphazero_lite/arena.py": sha256(Path(arena.__file__)),
        },
    }
    freeze_path = output / "protocol-freeze.json"
    freeze_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run(output: Path = OUT) -> None:
    registration = _REGISTRATION
    freeze_path = output / "protocol-freeze.json"
    cohort_path = output / "cohort.json"
    if not freeze_path.exists() or not cohort_path.exists():
        raise ValueError("seed514_protocol_not_frozen")
    freeze = json.loads(freeze_path.read_text())
    if sha256(cohort_path) != freeze["cohort_sha256"]:
        raise ValueError("seed514_frozen_cohort_identity")
    expected_sources = freeze["source_sha256"]
    current_sources = {
        "ml/alphazero_lite/seed514_operational_benchmark.py": sha256(Path(__file__)),
        "ml/alphazero_lite/test_seed514_operational_benchmark.py": sha256(TEST_SOURCE),
        "ml/alphazero_lite/verify_seed514_operational_benchmark.py": sha256(
            VERIFY_SOURCE
        ),
        "ml/alphazero_lite/arena.py": sha256(Path(arena.__file__)),
    }
    if current_sources != expected_sources:
        raise ValueError("seed514_frozen_source_identity")
    if (output / "raw-measurements.jsonl").exists() or (
        output / "evidence.json"
    ).exists():
        raise ValueError("seed514_measurements_already_exist")
    cohort = json.loads(cohort_path.read_text())
    cohort_sha = sha256(cohort_path)
    artifact_identities = verify_registered_identities(registration)
    models = configs(registration)
    settings = registration["runtime"]
    seeds = {item["state_sha256"]: 514000 + index for index, item in enumerate(cohort)}
    for item in cohort:
        game = KalahGame.from_state(item["state"])
        if game.over():
            raise ValueError("seed514_terminal_cohort_state")
        seed = seeds[item["state_sha256"]]
        baseline_output, _ = run_search(item["state"], models["baseline"], seed)
        p_output, _ = run_search(item["state"], models["P"], seed)
        game = KalahGame.from_state(item["state"])
        initializer_policy, initializer_value = models["baseline"][
            "evaluator"
        ].evaluate(game)
        e4_policy, _ = models["P"]["evaluator"].candidate_evaluator.evaluate(game)
        composed_policy, composed_value = models["P"]["evaluator"].evaluate(game)
        if not np.allclose(composed_policy, e4_policy, rtol=0.0, atol=1e-7):
            raise ValueError("seed514_policy_component_routing")
        if not np.isclose(composed_value, initializer_value, rtol=0.0, atol=1e-7):
            raise ValueError("seed514_value_component_routing")
        if not np.allclose(
            baseline_output["policy"], initializer_policy, rtol=0.0, atol=1e-6
        ):
            raise ValueError("seed514_baseline_output_mismatch")
        if not np.isfinite(np.asarray(p_output["policy"])).all():
            raise ValueError("seed514_nonfinite_policy")
        repeated_baseline, _ = run_search(item["state"], models["baseline"], seed)
        repeated_p, _ = run_search(item["state"], models["P"], seed)
        if repeatability_signature(baseline_output) != repeatability_signature(
            repeated_baseline
        ) or repeatability_signature(p_output) != repeatability_signature(repeated_p):
            raise ValueError("seed514_fixed_seed_search_not_repeatable")
        stones = sum(item["state"]["player_pits"] + item["state"]["opponent_pits"])
        if stones <= 16 and (
            "exact_root_decision" not in baseline_output
            or "exact_root_decision" not in p_output
        ):
            raise ValueError("seed514_exact_root_dispatch_missing")
    measurements: list[dict[str, Any]] = []
    for item_index, item in enumerate(cohort):
        seed = seeds[item["state_sha256"]]
        warmup_order = ("baseline", "P") if item_index % 2 == 0 else ("P", "baseline")
        for name in warmup_order:
            run_search(item["state"], models[name], seed)
        for repetition in range(REPETITIONS):
            order = (
                ("baseline", "P")
                if (item_index + repetition) % 2 == 0
                else ("P", "baseline")
            )
            for name in order:
                result, seconds = run_search(item["state"], models[name], seed)
                measurements.append(
                    {
                        "state_sha256": item["state_sha256"],
                        "config": name,
                        "repetition": repetition,
                        "seed": seed,
                        "seconds": seconds,
                        "exact_root": "exact_root_decision" in result,
                        "puct_simulations_executed": result.get(
                            "exact_root_decision", {}
                        ).get("puct_simulations_executed", SIMULATIONS),
                    }
                )
    (output / "raw-measurements.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in measurements)
    )
    memory = memory_measurement()
    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": __import__("torch").__version__,
        "dependencies": sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        "torch_threads": {
            "intraop": __import__("torch").get_num_threads(),
            "interop": __import__("torch").get_num_interop_threads(),
        },
        "cpu_count": os.cpu_count(),
        "thread_settings": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "command": f"{sys.executable} -m ml.alphazero_lite.seed514_operational_benchmark",
    }
    evidence = {
        "schema": "seed514-operational-benchmark-v1",
        "interpretation": "local examined diagnostic measurements on this host; not holdout or production qualification",
        "protocol": {
            "simulations": SIMULATIONS,
            "c_puct": CPUCT,
            "workers": 1,
            "warmups_per_config_state": WARMUPS,
            "repetitions": REPETITIONS,
            "order": "alternate by state-index plus repetition parity",
            "seed_contract": "514000 plus frozen cohort index",
            "runtime": settings,
        },
        "cohort_sha256": cohort_sha,
        "protocol_freeze_sha256": sha256(freeze_path),
        "seed512_registration_sha256": sha256(REGISTRATION),
        "artifact_identities": artifact_identities,
        "environment": env,
        "model_loading_seconds": {
            name: config["loading_seconds"] for name, config in models.items()
        },
        "memory": {
            "measurement": "combined model process RSS, sampled outside timed searches",
            **memory,
        },
        "summary": summarize(measurements),
        "source_sha256": {
            "ml/alphazero_lite/seed514_operational_benchmark.py": sha256(
                Path(__file__)
            ),
            "ml/alphazero_lite/verify_seed514_operational_benchmark.py": sha256(
                VERIFY_SOURCE
            ),
            "ml/alphazero_lite/arena.py": sha256(Path(arena.__file__)),
        },
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="freeze cohort and identities without searches",
    )
    args = parser.parse_args()
    if args.freeze:
        freeze_protocol(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
