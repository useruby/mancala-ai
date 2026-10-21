"""Frozen seed518 parity gate and paired selective-head timing screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import arena, seed514_operational_benchmark as seed514
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.selective_head_evaluator import SelectiveHeadArtifactEvaluator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/data/seed518-selective-head"
COHORT_SOURCE = ROOT / "docs/data/seed514-operational-benchmark-v2/cohort.json"
EXCLUDED_TELEMETRY = [
    "exact_root_decision.root_latency_ms",
    "root_latency_ms",
]
FINAL_FREEZE = "amended-timing-freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_runtime_assets(
    registration: dict[str, Any], root: Path
) -> dict[str, dict[str, str]]:
    """Authenticate the checked-out runtime bytes and resolve paths used by search."""
    assets = registration["runtime_assets"]
    table = assets["exact_root_tablebase"]
    checked_out_table = root / table["artifact"]
    if not checked_out_table.is_file():
        raise ValueError(f"seed518_tablebase_missing:{checked_out_table}")
    local_sha = sha256(checked_out_table)
    object_sha = table["sha256"]
    pointer_sha = table["source_pointer_sha256"]
    if local_sha == object_sha:
        table_path = checked_out_table.resolve()
    elif local_sha == pointer_sha:
        materialized = Path(table["runtime_materialized_path"])
        if not materialized.is_file():
            raise ValueError(f"seed518_materialized_tablebase_missing:{materialized}")
        actual_object_sha = sha256(materialized)
        if actual_object_sha != object_sha:
            raise ValueError(
                f"seed518_materialized_tablebase_identity:{materialized}:"
                f"expected={object_sha}:actual={actual_object_sha}"
            )
        table_path = materialized.resolve()
    else:
        raise ValueError(
            f"seed518_checked_out_tablebase_identity:{checked_out_table}:"
            f"expected_pointer_or_object={pointer_sha}|{object_sha}:actual={local_sha}"
        )

    probe = assets["exact_root_native_probe"]
    probe_path = root / probe["artifact"]
    if not probe_path.is_file():
        raise ValueError(f"seed518_native_probe_missing:{probe_path}")
    probe_sha = sha256(probe_path)
    if probe_sha != probe["sha256"]:
        raise ValueError(
            f"seed518_native_probe_identity:{probe_path}:"
            f"expected={probe['sha256']}:actual={probe_sha}"
        )
    return {
        "exact_root_tablebase": {"path": str(table_path), "sha256": sha256(table_path)},
        "exact_root_native_probe": {
            "path": str(probe_path.resolve()),
            "sha256": probe_sha,
        },
        "checked_out_tablebase": {
            "path": str(checked_out_table.resolve()),
            "sha256": local_sha,
        },
    }


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def strip_declared_telemetry(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(result))
    for dotted in EXCLUDED_TELEMETRY:
        keys = dotted.split(".")
        target = cleaned
        for key in keys[:-1]:
            target = target.get(key, {})
        target.pop(keys[-1], None)
    return cleaned


def compare_search_outputs(
    reference: dict[str, Any], selective: dict[str, Any]
) -> list[str]:
    """Return exact differing paths, ignoring only registered telemetry paths."""
    left = strip_declared_telemetry(reference)
    right = strip_declared_telemetry(selective)
    differences: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if type(a) is not type(b):
            differences.append(path)
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{path}.{key}" if path else key
                if key not in a or key not in b:
                    differences.append(child)
                else:
                    walk(a[key], b[key], child)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(f"{path}.length")
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
        elif a != b:
            differences.append(path)

    walk(left, right, "")
    return differences


def configurations() -> dict[str, dict[str, Any]]:
    base = ROOT / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"
    started = time.perf_counter_ns()
    initializer = arena.ArtifactEvaluator(base / "initializer")
    e4 = arena.ArtifactEvaluator(base / "E4")
    loaded = (time.perf_counter_ns() - started) / 1e9
    reference = arena.ComposedArtifactEvaluator(
        initializer, e4, policy_source="candidate", value_source="current"
    )
    selective = SelectiveHeadArtifactEvaluator(e4, initializer)
    return {
        "reference_P": {"evaluator": reference, "loading_seconds": loaded},
        "selective_P": {"evaluator": selective, "loading_seconds": loaded},
    }


def dispatch(
    state: dict[str, Any], config: dict[str, Any], seed: int
) -> dict[str, Any]:
    return arena.evaluate_artifact_position(
        evaluator=config["evaluator"],
        artifact_path=ROOT
        / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts/initializer",
        state=state,
        simulations=1200,
        seed=seed,
        c_puct=1.25,
        search_options=seed514._SEARCH_OPTIONS,
        exact_root_solve_threshold=16,
        exact_solve_stone_threshold=None,
    )


def run(output: Path = OUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / FINAL_FREEZE
    if not freeze_path.exists():
        raise ValueError("seed518_protocol_not_frozen")
    freeze = json.loads(freeze_path.read_text())
    sources = {
        "ml/alphazero_lite/arena.py": sha256(Path(arena.__file__)),
        "ml/alphazero_lite/seed518_selective_head.py": sha256(Path(__file__)),
        "ml/alphazero_lite/selective_head_evaluator.py": sha256(
            Path(
                __import__(
                    SelectiveHeadArtifactEvaluator.__module__, fromlist=["__file__"]
                ).__file__
            )
        ),
        "ml/alphazero_lite/verify_seed518_selective_head.py": sha256(
            ROOT / "ml/alphazero_lite/verify_seed518_selective_head.py"
        ),
        "ml/alphazero_lite/test_selective_head_evaluator.py": sha256(
            ROOT / "ml/alphazero_lite/test_selective_head_evaluator.py"
        ),
        "ml/alphazero_lite/test_seed518_runtime_identity.py": sha256(
            ROOT / "ml/alphazero_lite/test_seed518_runtime_identity.py"
        ),
        "ml/alphazero_lite/test_seed518_comparison_contract.py": sha256(
            ROOT / "ml/alphazero_lite/test_seed518_comparison_contract.py"
        ),
        "ml/alphazero_lite/self_play.py": sha256(
            ROOT / "ml/alphazero_lite/self_play.py"
        ),
    }
    if sources != freeze["source_sha256"]:
        raise ValueError("seed518_frozen_source_identity")
    registration = json.loads(
        (
            ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
        ).read_text()
    )
    if resolve_runtime_assets(registration, ROOT) != freeze["runtime_identities"]:
        raise ValueError("seed518_frozen_runtime_identity")
    cohort = json.loads(COHORT_SOURCE.read_text())
    models = configurations()
    parity: list[dict[str, Any]] = []
    for index, item in enumerate(cohort):
        seed = 518000 + index
        evaluator_outputs = {}
        search_outputs = {}
        for name, config in models.items():
            game = KalahGame.from_state(item["state"])
            policy, value = config["evaluator"].evaluate(game)
            evaluator_outputs[name] = {
                "policy_f32_hex": np.asarray(policy, dtype=np.float32).tobytes().hex(),
                "value_f64_hex": np.asarray(value, dtype=np.float64).tobytes().hex(),
            }
            search_outputs[name] = strip_declared_telemetry(
                dispatch(item["state"], config, seed)
            )
        mismatches = []
        for output_kind, outputs in (
            ("evaluator_outputs", evaluator_outputs),
            ("search_outputs", search_outputs),
        ):
            if canonical(outputs["reference_P"]) != canonical(outputs["selective_P"]):
                mismatches.append(output_kind)
        parity.append(
            {
                "state_sha256": item["state_sha256"],
                "seed": seed,
                "evaluator_outputs": evaluator_outputs,
                "search_outputs": search_outputs,
                "mismatches": mismatches,
            }
        )
    (output / "parity-outputs.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n"
    )
    mismatch_rows = [row for row in parity if row["mismatches"]]
    if mismatch_rows:
        raise ValueError(
            f"seed518_parity_mismatch:{json.dumps(mismatch_rows, sort_keys=True)}"
        )
    measurements = []
    for index, item in enumerate(cohort):
        seed = 518000 + index
        for name in (
            ("reference_P", "selective_P")
            if index % 2 == 0
            else ("selective_P", "reference_P")
        ):
            dispatch(item["state"], models[name], seed)
        for repetition in range(5):
            order = (
                ("reference_P", "selective_P")
                if (index + repetition) % 2 == 0
                else ("selective_P", "reference_P")
            )
            for name in order:
                start = time.perf_counter_ns()
                result = dispatch(item["state"], models[name], seed)
                seconds = (time.perf_counter_ns() - start) / 1e9
                measurements.append(
                    {
                        "state_sha256": item["state_sha256"],
                        "bucket": item["bucket"],
                        "config": name,
                        "repetition": repetition,
                        "seed": seed,
                        "seconds": seconds,
                        "exact_root": "exact_root_decision" in result,
                        "puct_simulations_executed": result.get(
                            "exact_root_decision", {}
                        ).get("puct_simulations_executed", 1200),
                        "dispatch_path": "exact_root"
                        if "exact_root_decision" in result
                        else "puct",
                        "order_position": order.index(name),
                    }
                )
    (output / "raw-measurements.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in measurements)
    )
    evidence = {
        "schema": "seed518-evidence-v1",
        "cohort_sha256": sha256(COHORT_SOURCE),
        "protocol_freeze_sha256": sha256(freeze_path),
        "parity_sha256": sha256(output / "parity-outputs.json"),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "dependencies": sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            ),
            "cpu_count": os.cpu_count(),
            "command": f"{sys.executable} -m ml.alphazero_lite.seed518_selective_head",
        },
        "loading_seconds": {
            name: config["loading_seconds"] for name, config in models.items()
        },
        "timed_searches": len(measurements),
        "summary": summarize(measurements),
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )


def freeze(output: Path = OUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    components = {}
    base = ROOT / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"
    for name in ("initializer", "E4"):
        components[name] = {
            p.name: sha256(p) for p in sorted((base / name).iterdir()) if p.is_file()
        }
    registration = json.loads(
        (
            ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
        ).read_text()
    )
    runtime = resolve_runtime_assets(registration, ROOT)
    source_paths = [
        ROOT / k
        for k in (
            "ml/alphazero_lite/arena.py",
            "ml/alphazero_lite/seed518_selective_head.py",
            "ml/alphazero_lite/selective_head_evaluator.py",
            "ml/alphazero_lite/verify_seed518_selective_head.py",
            "ml/alphazero_lite/test_selective_head_evaluator.py",
            "ml/alphazero_lite/test_seed518_runtime_identity.py",
            "ml/alphazero_lite/test_seed518_comparison_contract.py",
            "ml/alphazero_lite/self_play.py",
        )
    ]
    payload = {
        "schema": "seed518-protocol-freeze-v1",
        "cohort_sha256": sha256(COHORT_SOURCE),
        "amendment_sha256": sha256(output / "comparison-contract-amendment.json"),
        "original_failed_freeze_sha256": sha256(output / "protocol-freeze.json"),
        "original_failed_parity_sha256": sha256(output / "parity-outputs.json"),
        "cohort_source": str(COHORT_SOURCE.relative_to(ROOT)),
        "components": components,
        "runtime_identities": runtime,
        "registration_sha256": sha256(
            ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
        ),
        "architecture": {
            n: json.loads((base / n / "metadata.json").read_text()).get("architecture")
            for n in ("initializer", "E4")
        },
        "routing": {
            "policy": "E4 trunk + policy branch",
            "value": "initializer trunk + value branch",
        },
        "dependencies": sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        ),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "protocol": {
            "simulations": 1200,
            "c_puct": 1.25,
            "workers": 1,
            "seed_contract": "518000 + frozen cohort index",
            "warmups": 1,
            "repetitions": 5,
            "order": "alternating by state-index plus repetition parity",
            "exact_root_solve_threshold": 16,
            "exact_solve_stone_threshold": None,
            "telemetry_exclusions": EXCLUDED_TELEMETRY,
            "timing": "perf_counter_ns around full evaluate_artifact_position dispatch",
        },
    }
    path = output / FINAL_FREEZE
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def summarize(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("reference_P", "selective_P")
    by_state: dict[str, dict[str, list[float]]] = {}
    for row in measurements:
        by_state.setdefault(row["state_sha256"], {name: [] for name in names})[
            row["config"]
        ].append(float(row["seconds"]))
    states = [
        {
            "state_sha256": state,
            "reference_median_seconds": statistics.median(values[names[0]]),
            "selective_median_seconds": statistics.median(values[names[1]]),
            "selective_over_reference": statistics.median(values[names[1]])
            / statistics.median(values[names[0]]),
        }
        for state, values in sorted(by_state.items())
    ]
    out: dict[str, Any] = {"configs": {}, "buckets": {}, "per_state": states}
    for name in names:
        values = [
            float(row["seconds"]) for row in measurements if row["config"] == name
        ]
        out["configs"][name] = {
            "count": len(values),
            "p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)),
            "throughput_searches_per_second": len(values) / sum(values),
        }
    for bucket in ("gt32", "17to32", "le16"):
        out["buckets"][bucket] = {}
        for path in ("puct", "exact_root", "all"):
            rows = [
                row
                for row in measurements
                if row["bucket"] == bucket
                and (path == "all" or row["dispatch_path"] == path)
            ]
            paired: dict[tuple[str, int], dict[str, float]] = {}
            for row in rows:
                paired.setdefault((row["state_sha256"], row["repetition"]), {})[
                    row["config"]
                ] = float(row["seconds"])
            ratios = [
                pair[names[1]] / pair[names[0]]
                for pair in paired.values()
                if set(pair) == set(names)
            ]
            config_stats = {}
            for name in names:
                vals = [float(row["seconds"]) for row in rows if row["config"] == name]
                config_stats[name] = {
                    "count": len(vals),
                    "p50_seconds": float(np.percentile(vals, 50)) if vals else None,
                    "p95_seconds": float(np.percentile(vals, 95)) if vals else None,
                    "throughput_searches_per_second": len(vals) / sum(vals)
                    if vals
                    else None,
                }
            out["buckets"][bucket][path] = {
                "count": len(rows),
                "paired_count": len(ratios),
                "configs": config_stats,
                "paired_ratio_p50": float(np.percentile(ratios, 50))
                if ratios
                else None,
                "paired_ratio_p95": float(np.percentile(ratios, 95))
                if ratios
                else None,
            }
    puct_ratios = {
        bucket: out["buckets"][bucket]["puct"]["paired_ratio_p50"]
        for bucket in ("gt32", "17to32")
    }
    out["decision"] = {
        "parity_required": True,
        "minimum_improvement": 0.05,
        "puct_bucket_median_ratios": puct_ratios,
        "advance": all(
            value is not None and value <= 0.95 for value in puct_ratios.values()
        ),
    }
    return out


def reanalyze_archived_parity(output: Path = OUT) -> dict[str, Any]:
    """Recompare saved outputs; never dispatches or modifies original results."""
    parity_path = output / "parity-outputs.json"
    rows = json.loads(parity_path.read_text())
    per_state = []
    for row in rows:
        eval_equal = (
            row["evaluator_outputs"]["reference_P"]
            == row["evaluator_outputs"]["selective_P"]
        )
        search_differences = compare_search_outputs(
            row["search_outputs"]["reference_P"],
            row["search_outputs"]["selective_P"],
        )
        per_state.append(
            {
                "state_sha256": row["state_sha256"],
                "evaluator_bitwise_equal": eval_equal,
                "search_differences_after_declared_exclusions": search_differences,
                "passed": eval_equal and not search_differences,
            }
        )
    result = {
        "schema": "seed518-post-execution-reanalysis-v1",
        "interpretation": "post-execution reanalysis of archived outputs; no searches rerun",
        "amendment": "comparison-contract-amendment.json",
        "parity_archive_sha256": sha256(parity_path),
        "states": per_state,
        "evaluator_bitwise_equal_count": sum(
            r["evaluator_bitwise_equal"] for r in per_state
        ),
        "search_equal_under_amended_contract_count": sum(
            not r["search_differences_after_declared_exclusions"] for r in per_state
        ),
        "passed": all(r["passed"] for r in per_state),
    }
    path = output / "post-execution-reanalysis.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_timing(output: Path = OUT) -> None:
    """Run fresh timing only after successful amended archived reanalysis."""
    freeze_path = output / FINAL_FREEZE
    freeze = json.loads(freeze_path.read_text())
    cohort = json.loads(COHORT_SOURCE.read_text())
    amendment_path = output / "comparison-contract-amendment.json"
    if sha256(amendment_path) != freeze["amendment_sha256"]:
        raise ValueError("seed518_amendment_freeze_identity")
    reanalysis = json.loads((output / "post-execution-reanalysis.json").read_text())
    if not reanalysis["passed"] or reanalysis["parity_archive_sha256"] != sha256(
        output / "parity-outputs.json"
    ):
        raise ValueError("seed518_amended_parity_gate_failed")
    source_paths = freeze["source_sha256"]
    for relative, expected in source_paths.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"seed518_frozen_source_identity:{relative}")
    if sha256(COHORT_SOURCE) != freeze["cohort_sha256"]:
        raise ValueError("seed518_frozen_cohort_identity")
    registration = json.loads(
        (
            ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
        ).read_text()
    )
    if resolve_runtime_assets(registration, ROOT) != freeze["runtime_identities"]:
        raise ValueError("seed518_frozen_runtime_identity")
    if (output / "raw-measurements.jsonl").exists() or (
        output / "evidence.json"
    ).exists():
        raise FileExistsError("seed518_measurements_already_exist")
    models = configurations()
    measurements = []
    for index, item in enumerate(cohort):
        seed = 518000 + index
        for name in (
            ("reference_P", "selective_P")
            if index % 2 == 0
            else ("selective_P", "reference_P")
        ):
            dispatch(item["state"], models[name], seed)
        for repetition in range(5):
            order = (
                ("reference_P", "selective_P")
                if (index + repetition) % 2 == 0
                else ("selective_P", "reference_P")
            )
            for position, name in enumerate(order):
                started = time.perf_counter_ns()
                result = dispatch(item["state"], models[name], seed)
                seconds = (time.perf_counter_ns() - started) / 1e9
                measurements.append(
                    {
                        "state_sha256": item["state_sha256"],
                        "bucket": item["bucket"],
                        "config": name,
                        "repetition": repetition,
                        "seed": seed,
                        "seconds": seconds,
                        "dispatch_path": "exact_root"
                        if "exact_root_decision" in result
                        else "puct",
                        "puct_simulations_executed": result.get(
                            "exact_root_decision", {}
                        ).get("puct_simulations_executed", 1200),
                        "order_position": position,
                    }
                )
    raw = output / "raw-measurements.jsonl"
    with raw.open("x") as stream:
        stream.write(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in measurements)
        )
    evidence = {
        "schema": "seed518-evidence-v1",
        "interpretation": "local examined diagnostic timing; not production qualification",
        "cohort_sha256": sha256(COHORT_SOURCE),
        "protocol_freeze_sha256": sha256(freeze_path),
        "parity_sha256": sha256(output / "parity-outputs.json"),
        "post_execution_reanalysis_sha256": sha256(
            output / "post-execution-reanalysis.json"
        ),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "dependencies": sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            ),
            "thread_settings": {
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "command": "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m ml.alphazero_lite.seed518_selective_head --timing-only",
        },
        "loading_seconds": {key: val["loading_seconds"] for key, val in models.items()},
        "timed_searches": len(measurements),
        "summary": summarize(measurements),
    }
    with (output / "evidence.json").open("x") as stream:
        stream.write(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.output)
    elif args.timing_only:
        run_timing(args.output)
    else:
        run(args.output)


if __name__ == "__main__":
    main()
