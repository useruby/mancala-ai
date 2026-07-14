#!/usr/bin/env python3
"""Run control_ep2 PUCT policy-head promotion preflight.

Does not train, generate self-play, promote, or overwrite model-artifact/current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.build_opening_suite import (  # noqa: E402
    deduplicate_openings,
    enumerate_legal_prefixes,
    load_suite_jsonl,
    select_diverse,
    stratify_openings,
    write_suite_jsonl,
)
from ml.alphazero_lite.seat_aware_arena import (  # noqa: E402
    compute_seat_split_metrics,
    parse_game_jsonl,
)

VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
DEFAULT_GATE_BUDGET_PAIRS = "384:256,1200:1200,1200:256,256:768"
DEFAULT_H2H_BUDGET_PAIRS = "384:384,1200:1200,384:256,256:384"
DEFAULT_HELDOUT_SEEDS = "43,44,45"
DEFAULT_BOOTSTRAP_SAMPLES = 10000
EXPECTED_CONTROL_CHECKPOINT_SHA256 = (
    "619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9"
)
EXPECTED_CONTROL_ARTIFACT_SHA256 = (
    "34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad"
)

BENCHMARK_LABEL_TO_BUDGET = {
    "standard": "384:256",
    "challenger_768_vs_256": "768:256",
    "equal_768": "768:768",
    "equal_high": "1200:1200",
    "1200_vs_256": "1200:256",
    "current_high_asymmetry": "256:768",
}
GATE_LABEL_TO_BUDGET = {
    "standard": "384:256",
    "equal_high": "1200:1200",
    "challenger_high": "1200:256",
    "current_high_asymmetry": "256:768",
    "384_vs_384": "384:384",
    "256_vs_384": "256:384",
}


def _python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def resolve_artifact_dir(
    *, requested_dir: Path, checkpoint_path: Path, label: str
) -> Path:
    candidates = [
        requested_dir,
        checkpoint_path.parent / f"artifact_{checkpoint_path.parent.name}",
    ]
    for candidate in candidates:
        if (candidate / "weights.json").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing {label} artifact weights; searched: {searched}")


def build_input_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if path.suffix == ".jsonl":
        summary["rows"] = count_jsonl_rows(path)
    return summary


def verify_expected_hash(
    path: Path, expected_hash: str | None, label: str
) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    result = {
        "path": str(path),
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "matches_expected": expected_hash is None or actual_hash == expected_hash,
    }
    if expected_hash is not None and actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return result


def benchmark_candidate_name(candidate_path: Path) -> str:
    name = candidate_path.name
    for part in candidate_path.parts:
        if part.startswith("replicate_seed_"):
            return part
        if part in (
            "curriculum_ep1",
            "curriculum_ep2",
            "curriculum_lr1e5_ep1",
            "curriculum_lr1e5_ep2",
        ):
            return part
    if "iter0_candidate" in name:
        return "iter0_reference"
    if "control_ep1" in name:
        return "control_ep1"
    if "control_ep2" in name:
        return "control_ep2"
    return name


def prefix_key(prefix_moves: list[int]) -> str:
    return ",".join(str(int(move)) for move in prefix_moves)


def suite_prefix_keys(entries: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for entry in entries:
        moves = entry.get("prefix_moves")
        if isinstance(moves, list):
            keys.add(prefix_key(moves))
        for alternate in entry.get("alternate_prefixes", []) or []:
            if isinstance(alternate, list):
                keys.add(prefix_key(alternate))
    return keys


def suite_distribution(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ply_counts = Counter(int(entry["ply"]) for entry in entries)
    first_move_counts = Counter(
        str(entry.get("first_move_family", "none")) for entry in entries
    )
    side_counts = Counter(int(entry.get("side_to_move", 0)) for entry in entries)
    return {
        "rows": len(entries),
        "ply_counts": {str(key): ply_counts[key] for key in sorted(ply_counts)},
        "first_move_family_counts": dict(sorted(first_move_counts.items())),
        "side_to_move_counts": {
            str(key): side_counts[key] for key in sorted(side_counts)
        },
    }


def build_heldout_suite(
    *,
    fixed_suite_path: Path,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    fixed_entries = load_suite_jsonl(str(fixed_suite_path))
    fixed_prefixes = suite_prefix_keys(fixed_entries)
    target_size = len(fixed_entries)
    target_ply_counts = Counter(int(entry["ply"]) for entry in fixed_entries)

    all_prefixes: list[dict[str, Any]] = []
    for max_ply in (2, 4, 6):
        all_prefixes.extend(enumerate_legal_prefixes(max_ply))
    unique, _duplicates, _duplicate_count = deduplicate_openings(all_prefixes)
    unique = stratify_openings(unique)

    filtered = []
    for entry in unique:
        entry_keys = {prefix_key(entry["prefix_moves"])}
        for alternate in entry.get("alternate_prefixes", []) or []:
            entry_keys.add(prefix_key(alternate))
        if entry_keys.isdisjoint(fixed_prefixes):
            filtered.append(entry)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for ply, target_count in sorted(target_ply_counts.items()):
        ply_pool = [entry for entry in filtered if int(entry["ply"]) == ply]
        quota = min(int(target_count), len(ply_pool))
        picked = select_diverse(ply_pool, quota, seed + ply * 1000)
        for entry in picked:
            key = prefix_key(entry["prefix_moves"])
            if key in selected_keys:
                continue
            selected.append(entry)
            selected_keys.add(key)

    if len(selected) < target_size:
        deficit = target_size - len(selected)
        remainder = [
            entry
            for entry in filtered
            if prefix_key(entry["prefix_moves"]) not in selected_keys
        ]
        selected.extend(select_diverse(remainder, deficit, seed + 99991))

    if len(selected) != target_size:
        raise RuntimeError(
            f"heldout suite size mismatch for seed {seed}: expected {target_size}, got {len(selected)}"
        )

    rng = np.random.default_rng(seed)
    order = list(range(len(selected)))
    rng.shuffle(order)
    selected = [selected[index] for index in order]
    write_suite_jsonl(selected, str(out_path))

    heldout_entries = load_suite_jsonl(str(out_path))
    overlap = sorted(suite_prefix_keys(heldout_entries) & fixed_prefixes)
    if overlap:
        raise RuntimeError(
            f"heldout suite overlaps fixed suite by {len(overlap)} prefixes; first={overlap[0]}"
        )
    optional_keys = {"alternate_prefixes"}
    schema_matches = (
        set(heldout_entries[0].keys()) - optional_keys
        == set(fixed_entries[0].keys()) - optional_keys
    )
    return {
        "path": str(out_path),
        "seed": seed,
        "sha256": sha256_file(out_path),
        "rows": len(heldout_entries),
        "schema_matches_fixed_large": schema_matches,
        "fixed_distribution": suite_distribution(fixed_entries),
        "heldout_distribution": suite_distribution(heldout_entries),
        "overlap_with_fixed_prefixes": 0,
        "generation": "deterministic_legal_prefix_resample",
    }


def ensure_heldout_suite(
    *,
    fixed_suite_path: Path,
    seed: int,
    suite_dir: Path,
) -> dict[str, Any]:
    out_path = suite_dir / f"heldout_seed{seed}_large.jsonl"
    if out_path.is_file():
        try:
            return build_heldout_suite(
                fixed_suite_path=fixed_suite_path,
                seed=seed,
                out_path=out_path,
            )
        except Exception:
            pass
    return build_heldout_suite(
        fixed_suite_path=fixed_suite_path,
        seed=seed,
        out_path=out_path,
    )


def run_opening_suite_benchmark(
    *,
    workdir: str,
    suite: str,
    current: str,
    candidates: str,
    budget_pairs: str,
    games_per_opening: int,
    seed: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        workdir,
        "--suite",
        suite,
        "--current",
        current,
        "--candidates",
        candidates,
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--root-policy-mode",
        "deterministic",
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
    ]
    print(f"[suite-eval] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout + 300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed: {result.stderr[-2000:]}")
    return json.loads(
        (Path(workdir) / "temperature_benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_default_gate(
    *,
    candidate_path: str,
    current_path: str,
    out: str,
    seed: int,
    workers: int,
    games: int,
    budget_pairs: str,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        candidate_path,
        "--current-path",
        current_path,
        "--out",
        out,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        budget_pairs,
    ]
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


def find_candidate_report(
    report: dict[str, Any], candidate: str
) -> dict[str, Any] | None:
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            for candidate_report in seed_report.get("candidate_reports", []):
                if candidate_report.get("candidate") == candidate:
                    return candidate_report
    return None


def benchmark_budget_results(
    candidate_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, budget_result in candidate_report.get("budget_results", {}).items():
        budget_label = str(label)
        budget_pair = BENCHMARK_LABEL_TO_BUDGET.get(budget_label, budget_label)
        results[budget_pair] = {
            "ds": budget_result.get("ds"),
            "p0_score": budget_result.get("p0_score"),
            "p1_score": budget_result.get("p1_score"),
            "disadvantaged_seat_score": budget_result.get("disadvantaged_seat_score"),
            "margin_mean": budget_result.get("margin_mean"),
            "margin_median": budget_result.get("margin_median"),
            "duplicate_trajectory_count": budget_result.get(
                "duplicate_trajectory_count"
            ),
            "duplicate_trajectory_rate": budget_result.get("duplicate_trajectory_rate"),
            "total_games": budget_result.get("total_games"),
            "per_opening_metrics": budget_result.get("per_opening_metrics", []),
            "move_time_mean_ms": budget_result.get("move_time_mean_ms"),
            "move_time_p95_ms": budget_result.get("move_time_p95_ms"),
            "search_profile_hash": budget_result.get("search_profile_hash"),
            "search_profile": budget_result.get("search_profile"),
        }
    return results


def gate_budget_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, budget_result in report.get("budget_results", {}).items():
        budget_label = str(label)
        budget_pair = GATE_LABEL_TO_BUDGET.get(budget_label, budget_label)
        results[budget_pair] = {
            "arena_score": budget_result.get("arena_score"),
            "disadvantaged_seat_score": budget_result.get("disadvantaged_seat_score"),
            "challenger_starts_0_score": budget_result.get(
                "challenger_starts_0", {}
            ).get("score"),
            "challenger_starts_1_score": budget_result.get(
                "challenger_starts_1", {}
            ).get("score"),
            "duplicate_trajectory_count": budget_result.get(
                "duplicate_trajectory_count"
            ),
            "total_games": budget_result.get("total_games"),
        }
    return results


def load_gate_eval_metrics(eval_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not eval_dir.is_dir():
        return results
    for budget_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
        budget_pair = GATE_LABEL_TO_BUDGET.get(budget_dir.name, budget_dir.name)
        alternating_jsonl = budget_dir / "alternating_games.jsonl"
        starts_0_jsonl = budget_dir / "starts_0" / "games.jsonl"
        starts_1_jsonl = budget_dir / "starts_1" / "games.jsonl"
        alternating_metrics = (
            compute_seat_split_metrics(parse_game_jsonl(str(alternating_jsonl)))
            if alternating_jsonl.is_file()
            else {}
        )
        starts_0_metrics = (
            compute_seat_split_metrics(parse_game_jsonl(str(starts_0_jsonl)))
            if starts_0_jsonl.is_file()
            else {}
        )
        starts_1_metrics = (
            compute_seat_split_metrics(parse_game_jsonl(str(starts_1_jsonl)))
            if starts_1_jsonl.is_file()
            else {}
        )
        results[budget_pair] = {
            "arena_score": (
                (
                    float(
                        alternating_metrics.get("challenger_starts_0", {}).get(
                            "score", 0.0
                        )
                    )
                    + float(
                        alternating_metrics.get("challenger_starts_1", {}).get(
                            "score", 0.0
                        )
                    )
                )
                / 2.0
                if alternating_metrics
                else None
            ),
            "disadvantaged_seat_score": alternating_metrics.get(
                "disadvantaged_seat_score"
            ),
            "challenger_starts_0_score": starts_0_metrics.get(
                "challenger_starts_0", {}
            ).get("score"),
            "challenger_starts_1_score": starts_1_metrics.get(
                "challenger_starts_1", {}
            ).get("score"),
            "duplicate_trajectory_count": alternating_metrics.get(
                "duplicate_trajectory_count"
            ),
            "total_games": alternating_metrics.get("total_games"),
        }
    return results


def suite_result_rows(
    *, report: dict[str, Any], candidates: list[dict[str, str]], suite_name: str
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "suite_name": suite_name,
        "suite_path": report.get("suite_path"),
        "suite_sha256": report.get("suite_sha256"),
        "suite_size": report.get("suite_size"),
        "candidates": {},
    }
    for candidate in candidates:
        candidate_report = find_candidate_report(report, candidate["report_name"])
        if candidate_report is None:
            raise RuntimeError(
                f"missing candidate {candidate['report_name']} in suite {suite_name}"
            )
        rows["candidates"][candidate["name"]] = {
            "candidate_path": candidate["artifact_dir"],
            "candidate_sha256": candidate_report.get("candidate_sha256"),
            "budget_results": benchmark_budget_results(candidate_report),
        }
    rows["suite_delta_vs_canonical"] = {}
    rows["suite_delta_e2_vs_e1"] = {}
    return rows


def pooled_per_opening_differences(
    *,
    suite_rows: dict[str, Any],
    candidate_a: str,
    candidate_b: str,
    budget_pair: str,
    metric_key: str,
) -> list[float]:
    pooled: list[float] = []
    for suite_name, suite_row in suite_rows.items():
        budget_a = (
            suite_row["candidates"][candidate_a]["budget_results"]
            .get(budget_pair, {})
            .get("per_opening_metrics", [])
        )
        budget_b = (
            suite_row["candidates"][candidate_b]["budget_results"]
            .get(budget_pair, {})
            .get("per_opening_metrics", [])
        )
        metrics_a = {entry["opening_prefix"]: entry for entry in budget_a}
        metrics_b = {entry["opening_prefix"]: entry for entry in budget_b}
        common = sorted(set(metrics_a) & set(metrics_b))
        if not common:
            raise RuntimeError(
                f"no per-opening overlap for {suite_name} {candidate_a} vs {candidate_b} {budget_pair}"
            )
        for opening_prefix in common:
            pooled.append(
                float(metrics_a[opening_prefix][metric_key])
                - float(metrics_b[opening_prefix][metric_key])
            )
    return pooled


def bootstrap_ci(
    diffs: list[float], *, seed: int, samples: int = DEFAULT_BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    if not diffs:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": samples, "n": 0}
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(samples, len(arr)))
    bootstrap_means = arr[indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.percentile(bootstrap_means, 2.5)),
        "upper": float(np.percentile(bootstrap_means, 97.5)),
        "samples": samples,
        "n": int(len(arr)),
    }


def aggregate_candidate_metrics(
    *,
    suite_rows: dict[str, Any],
    candidate_name: str,
    canonical_name: str,
    e1_name: str,
    e2_name: str,
    budget_pairs: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for budget_pair in budget_pairs:
        suite_ds: dict[str, float] = {}
        suite_deltas_vs_canonical: dict[str, float] = {}
        suite_deltas_e2_vs_e1: dict[str, float] = {}
        for suite_name, suite_row in suite_rows.items():
            suite_candidates = suite_row["candidates"]
            ds = float(
                suite_candidates[candidate_name]["budget_results"][budget_pair]["ds"]
            )
            canonical_ds = float(
                suite_candidates[canonical_name]["budget_results"][budget_pair]["ds"]
            )
            e2_ds = float(
                suite_candidates[e2_name]["budget_results"][budget_pair]["ds"]
            )
            e1_ds = float(
                suite_candidates[e1_name]["budget_results"][budget_pair]["ds"]
            )
            suite_ds[suite_name] = ds
            suite_deltas_vs_canonical[suite_name] = ds - canonical_ds
            suite_deltas_e2_vs_e1[suite_name] = e2_ds - e1_ds
        ds_values = list(suite_ds.values())
        summary[budget_pair] = {
            "suite_ds": suite_ds,
            "suite_delta_vs_canonical": suite_deltas_vs_canonical,
            "suite_delta_e2_vs_e1": suite_deltas_e2_vs_e1,
            "mean_ds": statistics.fmean(ds_values),
            "worst_suite_ds": min(ds_values),
            "stddev_ds": statistics.stdev(ds_values) if len(ds_values) > 1 else 0.0,
            "mean_delta_vs_canonical": statistics.fmean(
                suite_deltas_vs_canonical.values()
            ),
        }
    return summary


def annotate_suite_deltas(
    *,
    suite_rows: dict[str, Any],
    canonical_name: str,
    e1_name: str,
    e2_name: str,
    budget_pairs: list[str],
) -> None:
    for suite_row in suite_rows.values():
        delta_vs_canonical: dict[str, dict[str, float]] = {}
        for candidate_name, candidate_row in suite_row["candidates"].items():
            budget_delta: dict[str, float] = {}
            for budget_pair in budget_pairs:
                candidate_ds = float(candidate_row["budget_results"][budget_pair]["ds"])
                canonical_ds = float(
                    suite_row["candidates"][canonical_name]["budget_results"][
                        budget_pair
                    ]["ds"]
                )
                budget_delta[budget_pair] = candidate_ds - canonical_ds
            delta_vs_canonical[candidate_name] = budget_delta
        suite_row["suite_delta_vs_canonical"] = delta_vs_canonical

        e2_vs_e1: dict[str, float] = {}
        for budget_pair in budget_pairs:
            e2_ds = float(
                suite_row["candidates"][e2_name]["budget_results"][budget_pair]["ds"]
            )
            e1_ds = float(
                suite_row["candidates"][e1_name]["budget_results"][budget_pair]["ds"]
            )
            e2_vs_e1[budget_pair] = e2_ds - e1_ds
        suite_row["suite_delta_e2_vs_e1"] = e2_vs_e1


def opening_outcome_counts(
    *, suite_rows: dict[str, Any], budget_pair: str, candidate_a: str, candidate_b: str
) -> dict[str, int]:
    diffs = pooled_per_opening_differences(
        suite_rows=suite_rows,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        budget_pair=budget_pair,
        metric_key="disadvantaged_seat_score",
    )
    counts = {candidate_a: 0, candidate_b: 0, "tie": 0}
    for diff in diffs:
        if diff > 1e-12:
            counts[candidate_a] += 1
        elif diff < -1e-12:
            counts[candidate_b] += 1
        else:
            counts["tie"] += 1
    return counts


def candidate_gate_preserved(report: dict[str, Any]) -> bool:
    return report.get("classification") == "high_search_breakthrough"


def heldout_mean_delta(
    suite_rows: dict[str, Any],
    candidate_name: str,
    canonical_name: str,
    budget_pair: str,
) -> float:
    heldout_names = [name for name in suite_rows if name.startswith("heldout_seed")]
    values = []
    for suite_name in heldout_names:
        candidate_ds = float(
            suite_rows[suite_name]["candidates"][candidate_name]["budget_results"][
                budget_pair
            ]["ds"]
        )
        canonical_ds = float(
            suite_rows[suite_name]["candidates"][canonical_name]["budget_results"][
                budget_pair
            ]["ds"]
        )
        values.append(candidate_ds - canonical_ds)
    return statistics.fmean(values) if values else 0.0


def large_p0_p1_regression(
    suite_rows: dict[str, Any], candidate_name: str, canonical_name: str
) -> bool:
    heldout_names = [name for name in suite_rows if name.startswith("heldout_seed")]
    candidate_gaps = []
    canonical_gaps = []
    for suite_name in heldout_names:
        candidate_budget = suite_rows[suite_name]["candidates"][candidate_name][
            "budget_results"
        ]["384:256"]
        canonical_budget = suite_rows[suite_name]["candidates"][canonical_name][
            "budget_results"
        ]["384:256"]
        candidate_gaps.append(
            float(candidate_budget["p1_score"]) - float(candidate_budget["p0_score"])
        )
        canonical_gaps.append(
            float(canonical_budget["p1_score"]) - float(canonical_budget["p0_score"])
        )
    if not candidate_gaps:
        return False
    return statistics.fmean(candidate_gaps) - statistics.fmean(canonical_gaps) > 0.10


def classify_preflight(summary: dict[str, Any]) -> str:
    aggregate = summary["aggregate_robustness"]
    bootstrap = summary["bootstrap_cis"]
    gates = summary["default_gate"]
    candidates = summary["candidate_names"]
    canonical = candidates["canonical_ref"]
    e1 = candidates["puct_policy_head_e1"]
    e2 = candidates["puct_policy_head_e2"]
    e1_gate = candidate_gate_preserved(gates[e1])
    e2_gate = candidate_gate_preserved(gates[e2])

    e1_std = aggregate[e1]["384:256"]
    e2_std = aggregate[e2]["384:256"]
    e1_high = aggregate[e1]["1200:1200"]
    e2_high = aggregate[e2]["1200:1200"]
    e2_vs_e1_std = bootstrap["e2_minus_e1_384_256"]

    if (
        e1_std["mean_delta_vs_canonical"] >= 0.10
        and bootstrap["e1_minus_canonical_384_256"]["lower"] > 0.03
        and e1_high["mean_delta_vs_canonical"] >= -0.03
        and e1_gate
        and not (
            e2_std["mean_delta_vs_canonical"] >= 0.10
            and e2_vs_e1_std["mean"] >= 0.05
            and e2_vs_e1_std["lower"] > 0.0
        )
    ):
        return "select_e1_for_promotion_followup"

    if (
        e2_std["mean_delta_vs_canonical"] >= 0.10
        and bootstrap["e2_minus_canonical_384_256"]["lower"] > 0.03
        and e2_high["mean_delta_vs_canonical"] >= -0.05
        and e2_gate
        and e2_vs_e1_std["mean"] >= 0.05
        and e2_vs_e1_std["lower"] > 0.0
    ):
        return "select_e2_for_promotion_followup"

    e1_heldout = heldout_mean_delta(summary["suite_results"], e1, canonical, "384:256")
    e2_heldout = heldout_mean_delta(summary["suite_results"], e2, canonical, "384:256")
    if (
        (e1_heldout <= 0.0 and e2_heldout <= 0.0)
        or not e1_gate
        or not e2_gate
        or large_p0_p1_regression(summary["suite_results"], e1, canonical)
        or large_p0_p1_regression(summary["suite_results"], e2, canonical)
    ):
        return "no_promotion_candidate"

    inconclusive = (
        bootstrap["e1_minus_canonical_384_256"]["lower"] <= 0.03
        or bootstrap["e2_minus_canonical_384_256"]["lower"] <= 0.03
        or e2_vs_e1_std["lower"] <= 0.0 <= e2_vs_e1_std["upper"]
    )
    fixed_large = summary["suite_results"]["fixed_large"]["candidates"]
    fixed_strong = float(fixed_large[e1]["budget_results"]["384:256"]["ds"]) > float(
        fixed_large[canonical]["budget_results"]["384:256"]["ds"]
    ) and float(fixed_large[e2]["budget_results"]["384:256"]["ds"]) > float(
        fixed_large[canonical]["budget_results"]["384:256"]["ds"]
    )
    if fixed_strong and inconclusive:
        return "needs_more_heldout_openings"

    return "no_promotion_candidate"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    candidate_names = summary["candidate_names"]
    suite_rows = summary["suite_results"]
    aggregate = summary["aggregate_robustness"]
    bootstrap = summary["bootstrap_cis"]
    gates = summary["default_gate"]
    head_to_head = summary["direct_head_to_head"]

    artifact_rows = []
    for candidate in summary["candidates"]:
        artifact_rows.append(
            [
                candidate["name"],
                candidate["checkpoint_path"],
                candidate["checkpoint_sha256"],
                candidate["artifact_dir"],
                candidate["artifact_weights_sha256"],
            ]
        )

    suite_hash_rows = []
    for suite_name, suite_info in summary["suite_inputs"].items():
        suite_hash_rows.append(
            [suite_name, suite_info["path"], suite_info["sha256"], suite_info["rows"]]
        )

    fixed_rows = []
    heldout_rows = []
    for suite_name, suite_row in suite_rows.items():
        target_rows = fixed_rows if suite_name == "fixed_large" else heldout_rows
        for candidate in (
            candidate_names["canonical_ref"],
            candidate_names["puct_policy_head_e1"],
            candidate_names["puct_policy_head_e2"],
        ):
            budget = suite_row["candidates"][candidate]["budget_results"]["384:256"]
            target_rows.append(
                [
                    suite_name,
                    candidate,
                    fmt(budget["ds"]),
                    f"{budget['p0_score']:.4f} / {budget['p1_score']:.4f}",
                    budget["duplicate_trajectory_count"],
                ]
            )

    aggregate_rows = []
    for candidate in (
        candidate_names["canonical_ref"],
        candidate_names["puct_policy_head_e1"],
        candidate_names["puct_policy_head_e2"],
    ):
        std = aggregate[candidate]["384:256"]
        high = aggregate[candidate]["1200:1200"]
        aggregate_rows.append(
            [
                candidate,
                fmt(std["mean_ds"]),
                fmt(std["worst_suite_ds"]),
                f"{std['stddev_ds']:.4f}",
                fmt(std["mean_delta_vs_canonical"]),
                fmt(high["mean_ds"]),
                fmt(high["mean_delta_vs_canonical"]),
            ]
        )

    ci_rows = []
    for label, ci in bootstrap.items():
        ci_rows.append(
            [label, fmt(ci["mean"]), fmt(ci["lower"]), fmt(ci["upper"]), ci["n"]]
        )

    h2h_rows = []
    for matchup_name, report in head_to_head.items():
        for budget_pair, budget in report["budget_results"].items():
            h2h_rows.append(
                [
                    matchup_name,
                    budget_pair,
                    fmt(budget["arena_score"]),
                    fmt(budget["disadvantaged_seat_score"]),
                    fmt(budget["challenger_starts_0_score"]),
                    fmt(budget["challenger_starts_1_score"]),
                    budget["duplicate_trajectory_count"],
                ]
            )

    gate_rows = []
    for candidate, report in gates.items():
        gate_rows.append(
            [
                candidate,
                report["classification"],
                "yes" if candidate_gate_preserved(report) else "no",
                fmt(
                    report["budget_results"]
                    .get("384:256", {})
                    .get("disadvantaged_seat_score")
                ),
                fmt(
                    report["budget_results"]
                    .get("1200:1200", {})
                    .get("disadvantaged_seat_score")
                ),
            ]
        )

    opening_counts = summary["opening_outcome_counts_384_256_disadvantaged"]
    lines = [
        "# AlphaZero-Lite Control EP2 PUCT Head Preflight Results",
        "",
        f"**Date**: {summary['date']}",
        f"**Classification**: `{summary['classification']}`",
        f"**Schema**: `{summary['schema']}`",
        "",
        "## Summary",
        "",
        f"Recommendation: `{summary['recommendation']}`.",
        "",
        "## Artifact Hashes",
        "",
        markdown_table(
            [
                "Candidate",
                "Checkpoint",
                "Checkpoint SHA256",
                "Artifact",
                "Weights SHA256",
            ],
            artifact_rows,
        ),
        "",
        "## Suite Hashes",
        "",
        markdown_table(["Suite", "Path", "SHA256", "Rows"], suite_hash_rows),
        "",
        "## Fixed-Suite Results",
        "",
        markdown_table(
            [
                "Suite",
                "Candidate",
                "384:256 DS",
                "384:256 P0 / P1",
                "Duplicate trajectories",
            ],
            fixed_rows,
        ),
        "",
        "## Held-Out-Suite Results",
        "",
        markdown_table(
            [
                "Suite",
                "Candidate",
                "384:256 DS",
                "384:256 P0 / P1",
                "Duplicate trajectories",
            ],
            heldout_rows,
        ),
        "",
        "## Aggregate Robustness Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean DS 384:256",
                "Worst-suite DS 384:256",
                "Stddev DS 384:256",
                "Mean delta vs canonical 384:256",
                "Mean DS 1200:1200",
                "Mean delta vs canonical 1200:1200",
            ],
            aggregate_rows,
        ),
        "",
        "## Paired Bootstrap CI Table",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%", "Openings"], ci_rows
        ),
        "",
        "## Direct e1/e2 Head-to-Head Table",
        "",
        markdown_table(
            [
                "Matchup",
                "Budget",
                "Arena score",
                "Disadvantaged-seat score",
                "Starts 0",
                "Starts 1",
                "Duplicate trajectories",
            ],
            h2h_rows,
        ),
        "",
        "## Default Gate Classification Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Classification",
                "High-search preserved",
                "384:256 disadvantaged",
                "1200:1200 disadvantaged",
            ],
            gate_rows,
        ),
        "",
        "## Opening-Level e1 vs e2 Count",
        "",
        markdown_table(
            ["Budget", "e1 better", "e2 better", "Tie"],
            [
                [
                    "384:256 disadvantaged seat",
                    opening_counts[candidate_names["puct_policy_head_e1"]],
                    opening_counts[candidate_names["puct_policy_head_e2"]],
                    opening_counts["tie"],
                ]
            ],
        ),
        "",
        "## Final Classification",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Recommendation: `{summary['recommendation']}`",
        f"- Rationale: {summary['decision_rationale']}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_control_ep2_puct_head_preflight"
    )
    parser.add_argument(
        "--control-checkpoint",
        default="/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz",
    )
    parser.add_argument(
        "--control-artifact",
        default="/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2",
    )
    parser.add_argument(
        "--puct-e1-checkpoint",
        default="/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz",
    )
    parser.add_argument(
        "--puct-e1-artifact",
        default="/tmp/azlite_control_ep2_puct_smoke/artifacts/puct_policy_head_e1",
    )
    parser.add_argument(
        "--puct-e2-checkpoint",
        default="/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e2/checkpoint_epoch2.npz",
    )
    parser.add_argument(
        "--puct-e2-artifact",
        default="/tmp/azlite_control_ep2_puct_smoke/artifacts/puct_policy_head_e2",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument("--heldout-seeds", default=DEFAULT_HELDOUT_SEEDS)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite-timeout", type=int, default=14400)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--gate-budget-pairs", default=DEFAULT_GATE_BUDGET_PAIRS)
    parser.add_argument("--gate-games", type=int, default=60)
    parser.add_argument("--head-to-head-budget-pairs", default=DEFAULT_H2H_BUDGET_PAIRS)
    parser.add_argument("--head-to-head-games", type=int, default=240)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--expected-control-checkpoint-sha256",
        default=EXPECTED_CONTROL_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-control-artifact-sha256",
        default=EXPECTED_CONTROL_ARTIFACT_SHA256,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games_per_opening > 2:
        raise ValueError("games_per_opening must not exceed 2 for this preflight")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    suite_dir = workdir / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)

    control_checkpoint = Path(args.control_checkpoint)
    control_artifact = Path(args.control_artifact)
    puct_e1_checkpoint = Path(args.puct_e1_checkpoint)
    puct_e1_artifact = resolve_artifact_dir(
        requested_dir=Path(args.puct_e1_artifact),
        checkpoint_path=puct_e1_checkpoint,
        label="puct e1",
    )
    puct_e2_checkpoint = Path(args.puct_e2_checkpoint)
    puct_e2_artifact = resolve_artifact_dir(
        requested_dir=Path(args.puct_e2_artifact),
        checkpoint_path=puct_e2_checkpoint,
        label="puct e2",
    )
    current_artifact = Path(args.current)
    fixed_large_suite = Path(args.fixed_large_suite)

    require_existing_file(control_checkpoint, "control checkpoint")
    require_existing_file(control_artifact / "weights.json", "control artifact weights")
    require_existing_file(puct_e1_checkpoint, "puct e1 checkpoint")
    require_existing_file(puct_e1_artifact / "weights.json", "puct e1 artifact weights")
    require_existing_file(puct_e2_checkpoint, "puct e2 checkpoint")
    require_existing_file(puct_e2_artifact / "weights.json", "puct e2 artifact weights")
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(fixed_large_suite, "fixed large suite")

    candidate_specs = [
        {
            "name": "canonical_ref",
            "report_name": benchmark_candidate_name(control_artifact),
            "checkpoint_path": str(control_checkpoint),
            "artifact_dir": str(control_artifact),
            "checkpoint_sha256": sha256_file(control_checkpoint),
            "artifact_weights_sha256": sha256_file(control_artifact / "weights.json"),
        },
        {
            "name": "puct_policy_head_e1",
            "report_name": benchmark_candidate_name(puct_e1_artifact),
            "checkpoint_path": str(puct_e1_checkpoint),
            "artifact_dir": str(puct_e1_artifact),
            "checkpoint_sha256": sha256_file(puct_e1_checkpoint),
            "artifact_weights_sha256": sha256_file(puct_e1_artifact / "weights.json"),
        },
        {
            "name": "puct_policy_head_e2",
            "report_name": benchmark_candidate_name(puct_e2_artifact),
            "checkpoint_path": str(puct_e2_checkpoint),
            "artifact_dir": str(puct_e2_artifact),
            "checkpoint_sha256": sha256_file(puct_e2_checkpoint),
            "artifact_weights_sha256": sha256_file(puct_e2_artifact / "weights.json"),
        },
    ]
    candidate_names = {spec["name"]: spec["name"] for spec in candidate_specs}

    suite_inputs: dict[str, dict[str, Any]] = {
        "fixed_large": build_input_summary(fixed_large_suite),
    }
    heldout_suite_paths: list[Path] = []
    for seed_text in args.heldout_seeds.split(","):
        seed_value = int(seed_text.strip())
        heldout_info = ensure_heldout_suite(
            fixed_suite_path=fixed_large_suite,
            seed=seed_value,
            suite_dir=suite_dir,
        )
        suite_inputs[f"heldout_seed{seed_value}_large"] = heldout_info
        heldout_suite_paths.append(Path(heldout_info["path"]))

    input_summary = {
        "control_checkpoint": verify_expected_hash(
            control_checkpoint,
            args.expected_control_checkpoint_sha256,
            "control checkpoint",
        ),
        "control_artifact_weights": verify_expected_hash(
            control_artifact / "weights.json",
            args.expected_control_artifact_sha256,
            "control artifact",
        ),
        "puct_e1_checkpoint": build_input_summary(puct_e1_checkpoint),
        "puct_e1_artifact_weights": build_input_summary(
            puct_e1_artifact / "weights.json"
        ),
        "puct_e2_checkpoint": build_input_summary(puct_e2_checkpoint),
        "puct_e2_artifact_weights": build_input_summary(
            puct_e2_artifact / "weights.json"
        ),
        "current_artifact_weights": build_input_summary(
            current_artifact / "weights.json"
        ),
    }

    candidate_paths = ",".join(spec["artifact_dir"] for spec in candidate_specs)
    suite_plan = [
        ("fixed_large", fixed_large_suite),
        *[
            (path.stem.replace("_large", "") + "_large", path)
            for path in heldout_suite_paths
        ],
    ]

    suite_results: dict[str, Any] = {}
    for suite_name, suite_path in suite_plan:
        eval_dir = workdir / f"eval_{suite_name}"
        report_path = eval_dir / "temperature_benchmark_report.json"
        if report_path.is_file():
            report = load_json(report_path)
        else:
            report = run_opening_suite_benchmark(
                workdir=str(eval_dir),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.suite_timeout,
            )
        suite_results[suite_name] = suite_result_rows(
            report=report,
            candidates=candidate_specs,
            suite_name=suite_name,
        )

    budget_pairs = args.budget_pairs.split(",")
    annotate_suite_deltas(
        suite_rows=suite_results,
        canonical_name=candidate_names["canonical_ref"],
        e1_name=candidate_names["puct_policy_head_e1"],
        e2_name=candidate_names["puct_policy_head_e2"],
        budget_pairs=budget_pairs,
    )
    aggregate_robustness = {
        candidate_names[spec["name"]]: aggregate_candidate_metrics(
            suite_rows=suite_results,
            candidate_name=candidate_names[spec["name"]],
            canonical_name=candidate_names["canonical_ref"],
            e1_name=candidate_names["puct_policy_head_e1"],
            e2_name=candidate_names["puct_policy_head_e2"],
            budget_pairs=budget_pairs,
        )
        for spec in candidate_specs
    }

    bootstrap_cis = {
        "e1_minus_canonical_384_256": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e1"],
                candidate_b=candidate_names["canonical_ref"],
                budget_pair="384:256",
                metric_key="ds",
            ),
            seed=args.seed + 101,
            samples=args.bootstrap_samples,
        ),
        "e1_minus_canonical_1200_1200": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e1"],
                candidate_b=candidate_names["canonical_ref"],
                budget_pair="1200:1200",
                metric_key="ds",
            ),
            seed=args.seed + 102,
            samples=args.bootstrap_samples,
        ),
        "e2_minus_canonical_384_256": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e2"],
                candidate_b=candidate_names["canonical_ref"],
                budget_pair="384:256",
                metric_key="ds",
            ),
            seed=args.seed + 103,
            samples=args.bootstrap_samples,
        ),
        "e2_minus_canonical_1200_1200": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e2"],
                candidate_b=candidate_names["canonical_ref"],
                budget_pair="1200:1200",
                metric_key="ds",
            ),
            seed=args.seed + 104,
            samples=args.bootstrap_samples,
        ),
        "e2_minus_e1_384_256": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e2"],
                candidate_b=candidate_names["puct_policy_head_e1"],
                budget_pair="384:256",
                metric_key="ds",
            ),
            seed=args.seed + 105,
            samples=args.bootstrap_samples,
        ),
        "e2_minus_e1_1200_1200": bootstrap_ci(
            pooled_per_opening_differences(
                suite_rows=suite_results,
                candidate_a=candidate_names["puct_policy_head_e2"],
                candidate_b=candidate_names["puct_policy_head_e1"],
                budget_pair="1200:1200",
                metric_key="ds",
            ),
            seed=args.seed + 106,
            samples=args.bootstrap_samples,
        ),
    }

    opening_outcomes = opening_outcome_counts(
        suite_rows=suite_results,
        budget_pair="384:256",
        candidate_a=candidate_names["puct_policy_head_e1"],
        candidate_b=candidate_names["puct_policy_head_e2"],
    )

    default_gate: dict[str, Any] = {}
    gate_dir = workdir / "default_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    for spec in candidate_specs:
        gate_out = gate_dir / f"{spec['name']}.json"
        if gate_out.is_file():
            gate_report = load_json(gate_out)
        else:
            gate_report = run_default_gate(
                candidate_path=spec["artifact_dir"],
                current_path=str(current_artifact),
                out=str(gate_out),
                seed=args.seed,
                workers=args.workers,
                games=args.gate_games,
                budget_pairs=args.gate_budget_pairs,
            )
        gate_eval_dir = gate_dir / Path(spec["artifact_dir"]).name
        default_gate[spec["name"]] = {
            "classification": gate_report.get("classification"),
            "standard_alternating_score": gate_report.get("standard_alternating_score"),
            "budget_results": load_gate_eval_metrics(gate_eval_dir),
        }

    direct_head_to_head: dict[str, Any] = {}
    h2h_dir = workdir / "head_to_head"
    h2h_dir.mkdir(parents=True, exist_ok=True)
    for challenger_name, current_name in (
        ("puct_policy_head_e1", "puct_policy_head_e2"),
        ("puct_policy_head_e2", "puct_policy_head_e1"),
    ):
        challenger_spec = next(
            spec for spec in candidate_specs if spec["name"] == challenger_name
        )
        current_spec = next(
            spec for spec in candidate_specs if spec["name"] == current_name
        )
        h2h_out = h2h_dir / f"{challenger_name}_vs_{current_name}.json"
        if h2h_out.is_file():
            report = load_json(h2h_out)
        else:
            report = run_default_gate(
                candidate_path=challenger_spec["artifact_dir"],
                current_path=current_spec["artifact_dir"],
                out=str(h2h_out),
                seed=args.seed,
                workers=args.workers,
                games=args.head_to_head_games,
                budget_pairs=args.head_to_head_budget_pairs,
            )
        h2h_eval_dir = h2h_dir / Path(challenger_spec["artifact_dir"]).name
        direct_head_to_head[f"{challenger_name}_vs_{current_name}"] = {
            "classification": report.get("classification"),
            "budget_results": load_gate_eval_metrics(h2h_eval_dir),
        }

    summary = {
        "schema": "azlite_control_ep2_puct_head_preflight_v1",
        "date": time.strftime("%Y-%m-%d"),
        "status": "completed",
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "root_policy_mode": "deterministic",
        "games_per_opening": args.games_per_opening,
        "budget_pairs": budget_pairs,
        "candidate_names": candidate_names,
        "inputs": input_summary,
        "suite_inputs": suite_inputs,
        "candidates": candidate_specs,
        "suite_results": suite_results,
        "aggregate_robustness": aggregate_robustness,
        "bootstrap_cis": bootstrap_cis,
        "default_gate": default_gate,
        "direct_head_to_head": direct_head_to_head,
        "opening_outcome_counts_384_256_disadvantaged": opening_outcomes,
        "guardrails": {
            "training": False,
            "new_self_play": False,
            "promotion": False,
            "overwrite_current": False,
            "replay_changes": False,
            "architecture_change": False,
            "residual_v4": False,
            "seed_sweep_training": False,
        },
    }
    summary["classification"] = classify_preflight(summary)
    summary["recommendation"] = summary["classification"]
    summary["decision_rationale"] = (
        "Uses fixed plus held-out opening robustness, paired per-opening bootstrap CIs, "
        "default gate behavior, and direct e1/e2 head-to-head checks without any training or promotion step."
    )

    summary_path = workdir / "summary_metrics.json"
    write_json(summary_path, summary)
    docs_path = (
        REPO_ROOT / "docs/alphazero-lite-control-ep2-puct-head-preflight-results.md"
    )
    write_markdown_report(summary, docs_path)
    print(f"[report] {summary_path}", flush=True)
    print(f"[doc] {docs_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
