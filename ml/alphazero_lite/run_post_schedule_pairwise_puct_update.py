#!/usr/bin/env python3
"""Post-schedule pairwise PUCT policy-head update experiment."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    ASYM_1200_256_BUDGET,
    EQ_768_BUDGET,
    EQ_1200_BUDGET,
    PRIMARY_BUDGET,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    find_candidate_report,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    build_input_summary,
    canonical_state_hash,
    raw_margin,
    read_jsonl,
    require_existing_file,
    sha256_file,
    top_policy_move,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    compute_param_delta_norm,
    export_checkpoint,
    heldout_summary,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_search_profile,
    encode_state,
    policy_from_visits,
)

SUMMARY_SCHEMA = "azlite_post_schedule_pairwise_puct_update_v1"
SUMMARY_FILENAME = "summary_metrics.json"
PAIRWISE_FILENAME = "validated_pairwise_targets.jsonl"
ANCHOR_FILENAME = "behavior_anchor_rows.jsonl"
PROBE_FILENAME = "probe_rows.jsonl"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-post-schedule-pairwise-puct-update-results.md"
)
TARGET_PAIRWISE_ROWS = 2000
MIN_PAIRWISE_ROWS = 1000
TARGET_ANCHOR_ROWS = 8000
MIN_ANCHOR_ROWS = 4000
TARGET_STABILITY_PROBE_ROWS = 2000
TARGET_BROAD_ANCHOR_PROBE_ROWS = 4000
DEFAULT_BUDGETS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--default-c-puct", type=float, default=1.25)
    parser.add_argument("--cpuct-schedule", required=True)
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--pairwise-margins", default="0.05,0.10")
    parser.add_argument("--behavior-anchor-weights", default="4,8")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one float is required")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_for_entry(entry: dict[str, Any]) -> str:
    ply = int(entry.get("ply", len(entry.get("prefix_moves", []))))
    state = entry["state"]
    seeds_remaining = sum(int(v) for v in state["player_pits"] + state["opponent_pits"])
    if ply <= 8:
        return "opening"
    if seeds_remaining <= 16:
        return "late"
    return "mid"


def seat_context_for_entry(entry: dict[str, Any]) -> str:
    return f"P{int(entry['state'].get('current_player', entry.get('side_to_move', 0)))}"


def load_suite_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        suite_name = path.stem
        for entry in read_jsonl(path):
            row = dict(entry)
            row["suite_name"] = suite_name
            row["phase"] = phase_for_entry(row)
            row["seat_context"] = seat_context_for_entry(row)
            row["state_hash"] = canonical_state_hash(row["state"])
            rows.append(row)
    return rows


def unique_state_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        state_hash = str(row["state_hash"])
        existing = grouped.get(state_hash)
        if existing is None:
            grouped[state_hash] = {
                **row,
                "suite_names": [str(row["suite_name"])],
            }
            continue
        suite_names = set(existing.get("suite_names", []))
        suite_names.add(str(row["suite_name"]))
        existing["suite_names"] = sorted(suite_names)
    return list(grouped.values())


def search_record(
    *,
    evaluator: ArtifactEvaluator,
    state_row: dict[str, Any],
    budget_pair: str,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    challenger_sims, current_sims = [int(part) for part in budget_pair.split(":", 1)]
    effective_c_puct = resolve_budget_cpuct(
        schedule=cpuct_schedule,
        challenger_simulations=challenger_sims,
        current_simulations=current_sims,
        default_c_puct=default_c_puct,
    )
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    profile = build_search_profile(
        kind="arena_eval",
        player_mode="puct",
        simulations=challenger_sims,
        c_puct=effective_c_puct,
        search_options=search_options,
        extra_fields={
            "challenger_simulations": challenger_sims,
            "current_simulations": current_sims,
        },
    )
    game = KalahGame.from_state(state_row["state"])
    legal_moves = [int(move) for move in game.possible_moves()]
    logits, raw_policy, raw_value = artifact_forward_details(evaluator, game)
    raw_top1 = top_policy_move(raw_policy, legal_moves)
    visits, _root = PUCT(
        evaluator,
        simulations=challenger_sims,
        c_puct=effective_c_puct,
        rng=random.Random(seed),
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    ).run(game)
    puct_policy = policy_from_visits(visits, legal_moves, 0.0)
    puct_top1 = top_policy_move(puct_policy, legal_moves)
    top1_visit_share = float(puct_policy[puct_top1]) if puct_top1 is not None else 0.0
    return {
        "state_hash": str(state_row["state_hash"]),
        "suite_name": str(state_row["suite_name"]),
        "suite_names": list(state_row.get("suite_names", [state_row["suite_name"]])),
        "state": state_row["state"],
        "encoded_state": encode_state(state_row["state"], input_encoding="kalah_v3"),
        "opening_prefix": list(state_row.get("prefix_moves", [])),
        "budget_pair": budget_pair,
        "phase": str(state_row["phase"]),
        "seat_context": str(state_row["seat_context"]),
        "legal_moves": legal_moves,
        "legal_mask_valid": bool(legal_moves),
        "raw_logits": [float(value) for value in logits],
        "raw_policy": [float(value) for value in raw_policy],
        "raw_value": float(raw_value),
        "raw_top1": raw_top1,
        "raw_top2_margin": float(raw_margin(raw_policy, legal_moves)),
        "puct_policy": [float(value) for value in puct_policy],
        "puct_top1": puct_top1,
        "puct_top1_visit_share": top1_visit_share,
        "effective_c_puct": float(effective_c_puct),
        "search_profile": profile,
        "search_profile_hash": str(profile["hash"]),
        "record_id": f"{state_row['state_hash']}::{budget_pair}",
    }


def build_search_records(
    *,
    state_rows: list[dict[str, Any]],
    budget_pairs: list[str],
    evaluator: ArtifactEvaluator,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for state_row in state_rows:
        for index, budget_pair in enumerate(budget_pairs):
            records.append(
                search_record(
                    evaluator=evaluator,
                    state_row=state_row,
                    budget_pair=budget_pair,
                    default_c_puct=default_c_puct,
                    cpuct_schedule=cpuct_schedule,
                    tactical_root_bias=tactical_root_bias,
                    seed=seed + index,
                )
            )
    return records


def group_records_by_state_hash(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["state_hash"])].append(record)
    for state_hash in grouped:
        grouped[state_hash].sort(key=lambda row: str(row["budget_pair"]))
    return grouped


def stable_across_budget_pairs(records: list[dict[str, Any]]) -> bool:
    top1s = {
        row.get("puct_top1") for row in records if row.get("puct_top1") is not None
    }
    return len(top1s) == 1 and len(records) > 1


def pairwise_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["budget_pair"] == PRIMARY_BUDGET),
        -int(row["seat_context"] == "P0"),
        -float(row["puct_top1_visit_share"]),
        float(row["raw_top2_margin"]),
        str(row["suite_name"]),
        str(row["state_hash"]),
    )


def sample_unique_pairwise_rows(
    records: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]], target: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in sorted(records, key=pairwise_sort_key):
        state_hash = str(row["state_hash"])
        if state_hash in seen_hashes:
            continue
        state_records = grouped[state_hash]
        if not (
            stable_across_budget_pairs(state_records)
            or float(row["puct_top1_visit_share"]) >= 0.55
        ):
            continue
        selected.append(row)
        seen_hashes.add(state_hash)
        if len(selected) >= target:
            break
    return selected


def sample_rows(
    rows: list[dict[str, Any]],
    target: int,
    *,
    key_fields: tuple[str, ...],
    unique_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while grouped and len(selected) < target:
        empty_keys: list[tuple[Any, ...]] = []
        for key in sorted(grouped):
            bucket = grouped[key]
            while bucket and str(bucket[0][unique_key]) in seen:
                bucket.pop(0)
            if not bucket:
                empty_keys.append(key)
                continue
            row = bucket.pop(0)
            selected.append(row)
            seen.add(str(row[unique_key]))
            if len(selected) >= target:
                break
        for key in empty_keys:
            grouped.pop(key, None)
    return selected


def build_pairwise_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_hash": str(row["state_hash"]),
        "state": list(row["encoded_state"]),
        "puct_move": int(row["puct_top1"]),
        "raw_move": int(row["raw_top1"]),
        "suite_name": str(row["suite_name"]),
        "budget_pair": str(row["budget_pair"]),
        "phase": str(row["phase"]),
        "seat_context": str(row["seat_context"]),
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "puct_top1_visit_share": float(row["puct_top1_visit_share"]),
        "raw_top2_margin": float(row["raw_top2_margin"]),
        "effective_c_puct": float(row["effective_c_puct"]),
        "search_profile_hash": str(row["search_profile_hash"]),
    }


def build_anchor_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_hash": str(row["state_hash"]),
        "state": list(row["encoded_state"]),
        "policy": [float(value) for value in row["raw_policy"]],
        "value": float(row["raw_value"]),
        "player": int(row["state"]["current_player"]),
        "move_index": int(len(row.get("opening_prefix", []))),
        "winner": None,
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "policy_target_mode": "sharpened",
        "policy_target_actual_mode": "sharpened",
        "value_target_mode": "sharpened",
        "bucket": "pairwise_behavior_anchor",
        "suite_name": str(row["suite_name"]),
        "budget_pair": str(row["budget_pair"]),
        "phase": str(row["phase"]),
        "seat_context": str(row["seat_context"]),
        "raw_top1": row["raw_top1"],
        "puct_top1": row["puct_top1"],
        "search_profile_hash": str(row["search_profile_hash"]),
    }


def build_probe_row(row: dict[str, Any], probe_kind: str) -> dict[str, Any]:
    return {
        "probe_kind": probe_kind,
        "state_hash": str(row["state_hash"]),
        "state": row["state"],
        "encoded_state": list(row["encoded_state"]),
        "budget_pair": str(row["budget_pair"]),
        "phase": str(row["phase"]),
        "seat_context": str(row["seat_context"]),
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "current_policy": [float(value) for value in row["raw_policy"]],
        "current_top1": row["raw_top1"],
        "puct_top1": row["puct_top1"],
        "search_profile_hash": str(row["search_profile_hash"]),
    }


def build_dataset(
    *,
    records: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pairwise_candidates = [
        row
        for row in records
        if row["legal_mask_valid"]
        and row["raw_top1"] is not None
        and row["puct_top1"] is not None
        and row["raw_top1"] != row["puct_top1"]
    ]
    pairwise_rows = sample_unique_pairwise_rows(
        pairwise_candidates, grouped, TARGET_PAIRWISE_ROWS
    )
    stability_candidates = [
        row
        for row in records
        if row["legal_mask_valid"]
        and row["raw_top1"] is not None
        and row["puct_top1"] is not None
        and row["raw_top1"] == row["puct_top1"]
        and (
            stable_across_budget_pairs(grouped[str(row["state_hash"])])
            or float(row["puct_top1_visit_share"]) >= 0.55
        )
    ]
    stability_rows = sample_rows(
        sorted(stability_candidates, key=pairwise_sort_key),
        TARGET_STABILITY_PROBE_ROWS,
        key_fields=("budget_pair", "phase", "seat_context", "suite_name"),
        unique_key="state_hash",
    )
    anchor_rows = sample_rows(
        sorted(records, key=pairwise_sort_key),
        TARGET_ANCHOR_ROWS,
        key_fields=("budget_pair", "phase", "seat_context", "suite_name"),
        unique_key="record_id",
    )
    broad_anchor_probe_rows = sample_rows(
        sorted(anchor_rows, key=pairwise_sort_key),
        TARGET_BROAD_ANCHOR_PROBE_ROWS,
        key_fields=("budget_pair", "phase", "seat_context", "suite_name"),
        unique_key="record_id",
    )
    if len(pairwise_rows) < MIN_PAIRWISE_ROWS:
        raise RuntimeError(
            f"validated pairwise row count below minimum: {len(pairwise_rows)} < {MIN_PAIRWISE_ROWS}"
        )
    if len(anchor_rows) < MIN_ANCHOR_ROWS:
        raise RuntimeError(
            f"behavior anchor row count below minimum: {len(anchor_rows)} < {MIN_ANCHOR_ROWS}"
        )
    return {
        "pairwise_rows": pairwise_rows,
        "stability_rows": stability_rows,
        "anchor_rows": anchor_rows,
        "broad_anchor_probe_rows": broad_anchor_probe_rows,
    }


def run_train_command(
    *,
    pairwise_file: Path,
    anchor_file: Path,
    init_checkpoint: Path,
    out_path: Path,
    margin: float,
    anchor_weight: int,
    epochs: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--pairwise-target-files",
        str(pairwise_file),
        "--pairwise-loss-weight",
        "1.0",
        "--pairwise-margin",
        str(margin),
        "--behavior-anchor-files",
        str(anchor_file),
        "--behavior-loss-weight",
        str(anchor_weight),
        "--init-checkpoint",
        str(init_checkpoint),
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--hidden-sizes",
        "96,3",
        "--epochs",
        str(epochs),
        "--batch-size",
        "512",
        "--lr",
        "1e-5",
        "--value-loss-weight",
        "0.0",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "0",
        "--out",
        str(out_path),
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--lr-scheduler",
        "none",
        "--seed",
        str(seed),
        "--trainable-scope",
        "policy_head",
        "--save-epochs",
        str(epochs),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed: {result.stderr[-2000:]}")
    metrics: dict[str, Any] = {"stdout": result.stdout, "stderr": result.stderr}
    for line in result.stdout.splitlines():
        line = line.strip()
        for key in (
            "policy_loss",
            "value_loss",
            "pairwise_loss",
            "behavior_anchor_loss",
            "total_loss",
            "best_val_loss",
        ):
            token = f"{key}="
            if line.startswith(token):
                metrics[key] = float(line.split("=", 1)[1])
    return metrics


def run_opening_suite_benchmark_schedule(
    *,
    workdir: Path,
    suite: Path,
    current: Path,
    candidate_dirs: list[Path],
    budget_pairs: str,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(workdir),
        "--suite",
        str(suite),
        "--current",
        str(current),
        "--candidates",
        ",".join(str(path) for path in candidate_dirs),
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        "2",
        "--seed",
        str(seed),
        "--root-policy-mode",
        "deterministic",
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--c-puct",
        str(default_c_puct),
        "--c-puct-schedule-json",
        json.dumps(cpuct_schedule, sort_keys=True),
        "--tactical-root-bias",
        str(tactical_root_bias),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout + 300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"benchmark failed: {result.stderr[-2000:]}")
    return load_json(workdir / "temperature_benchmark_report.json")


def run_default_gate_schedule(
    *,
    candidate_path: Path,
    current_path: Path,
    out_path: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "script/ai/seat_aware_promotion_gate"),
        "--candidate-path",
        str(candidate_path),
        "--current-path",
        str(current_path),
        "--out",
        str(out_path),
        "--games",
        "60",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        DEFAULT_BUDGETS,
        "--c-puct",
        str(default_c_puct),
        "--c-puct-schedule-json",
        json.dumps(cpuct_schedule, sort_keys=True),
        "--root-policy-mode",
        "deterministic",
        "--tactical-root-bias",
        str(tactical_root_bias),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    return load_json(out_path)


def candidate_spec(
    *,
    name: str,
    artifact_dir: Path,
    checkpoint_path: Path | None,
    epochs: int,
    margin: float | None,
    anchor_weight: int | None,
    report_name: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate": name,
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "epochs": epochs,
        "pairwise_margin": margin,
        "behavior_anchor_weight": anchor_weight,
        "report_candidate_name": report_name or name,
    }


def evaluate_probe_candidate(
    *,
    artifact_path: Path,
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    pairwise_successes = 0
    pairwise_margins: list[float] = []
    pairwise_margin_improvements: list[float] = []
    pairwise_total = 0
    puct_agree_total = 0
    stability_total = 0
    stability_preserved = 0
    broad_total = 0
    broad_changed = 0
    anchor_kls: list[float] = []
    changed_breakdown: dict[str, int] = defaultdict(int)
    changed_384_256_p0 = 0
    changed_384_256_p0_total = 0
    for row in probe_rows:
        game = KalahGame.from_state(row["state"])
        logits, candidate_policy, _value = artifact_forward_details(evaluator, game)
        logits_array = np.asarray(logits, dtype=np.float64)
        legal_moves = [int(move) for move in row["legal_moves"]]
        candidate_top1 = top_policy_move(candidate_policy, legal_moves)
        current_top1 = int(row["current_top1"])
        puct_top1 = int(row["puct_top1"])
        if candidate_top1 == puct_top1:
            puct_agree_total += 1
        if row["probe_kind"] == "pairwise":
            pairwise_total += 1
            margin = float(logits_array[puct_top1] - logits_array[current_top1])
            current_margin = float(
                row["current_policy"][puct_top1] - row["current_policy"][current_top1]
            )
            if margin > 0.0:
                pairwise_successes += 1
            pairwise_margins.append(margin)
            pairwise_margin_improvements.append(margin - current_margin)
            continue
        if row["probe_kind"] == "stability":
            stability_total += 1
            if candidate_top1 == current_top1:
                stability_preserved += 1
        if row["probe_kind"] == "broad_anchor":
            broad_total += 1
            safe_current = np.clip(
                np.asarray(row["current_policy"], dtype=np.float64), 1e-12, 1.0
            )
            safe_current /= np.sum(safe_current)
            safe_candidate = np.clip(
                np.asarray(candidate_policy, dtype=np.float64), 1e-12, 1.0
            )
            safe_candidate /= np.sum(safe_candidate)
            anchor_kls.append(
                float(np.sum(safe_current * np.log(safe_current / safe_candidate)))
            )
            if candidate_top1 != current_top1:
                broad_changed += 1
                changed_breakdown[f"{row['phase']}::{row['seat_context']}"] += 1
                if row["budget_pair"] == PRIMARY_BUDGET and row["seat_context"] == "P0":
                    changed_384_256_p0 += 1
            if row["budget_pair"] == PRIMARY_BUDGET and row["seat_context"] == "P0":
                changed_384_256_p0_total += 1
    return {
        "pairwise_target_success_rate": pairwise_successes / max(pairwise_total, 1),
        "mean_target_margin": statistics.fmean(pairwise_margins)
        if pairwise_margins
        else 0.0,
        "mean_target_margin_improvement": statistics.fmean(pairwise_margin_improvements)
        if pairwise_margin_improvements
        else 0.0,
        "stability_top1_preservation": stability_preserved / max(stability_total, 1),
        "broad_anchor_top1_changed_rate": broad_changed / max(broad_total, 1),
        "mean_anchor_kl_current_to_candidate": statistics.fmean(anchor_kls)
        if anchor_kls
        else 0.0,
        "changed_top1_rate_by_phase_and_seat_context": dict(
            sorted(changed_breakdown.items())
        ),
        "changed_top1_rate_384_256_p0": changed_384_256_p0
        / max(changed_384_256_p0_total, 1),
        "effective_search_profile_aware_puct_agreement": puct_agree_total
        / max(len(probe_rows), 1),
    }


def suite_rows_from_report(
    *,
    report: dict[str, Any],
    eval_candidates: list[dict[str, Any]],
    suite_name: str,
) -> dict[str, Any]:
    rows = {"suite_name": suite_name, "candidates": {}}
    for candidate in eval_candidates:
        candidate_report = find_candidate_report(
            report, candidate["report_candidate_name"]
        )
        if candidate_report is None:
            raise RuntimeError(
                f"missing candidate {candidate['report_candidate_name']} in {suite_name} report"
            )
        rows["candidates"][candidate["candidate"]] = {
            "budget_results": benchmark_budget_results(candidate_report)
        }
    return rows


def aggregate_budget_summary(
    *,
    suite_rows: dict[str, Any],
    candidate_name: str,
    current_name: str,
    budget_pairs: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for budget_pair in budget_pairs:
        deltas: list[float] = []
        p0_scores: list[float] = []
        p1_scores: list[float] = []
        duplicates: list[float] = []
        for suite_row in suite_rows.values():
            candidate_budget = suite_row["candidates"][candidate_name][
                "budget_results"
            ][budget_pair]
            current_budget = suite_row["candidates"][current_name]["budget_results"][
                budget_pair
            ]
            deltas.append(float(candidate_budget["ds"]) - float(current_budget["ds"]))
            p0_scores.append(float(candidate_budget["p0_score"]))
            p1_scores.append(float(candidate_budget["p1_score"]))
            duplicates.append(
                float(candidate_budget.get("duplicate_trajectory_count", 0.0))
            )
        summary[budget_pair] = {
            "mean_delta_vs_current_ref": statistics.fmean(deltas) if deltas else 0.0,
            "mean_p0_score": statistics.fmean(p0_scores) if p0_scores else 0.0,
            "mean_p1_score": statistics.fmean(p1_scores) if p1_scores else 0.0,
            "mean_duplicate_trajectory_count": statistics.fmean(duplicates)
            if duplicates
            else 0.0,
        }
    return summary


def classify(summary_rows: list[dict[str, Any]]) -> str:
    successful = False
    probe_moved = False
    eval_regressed = False
    for row in summary_rows:
        if row["candidate"] == "current_ref" or row.get("aborted"):
            continue
        probe = row["probe_metrics"]
        heldout = row.get("heldout_summary", {})
        agg = row.get("heldout_aggregate", {})
        ci = row.get("bootstrap_cis", {}).get(
            f"{row['candidate']}_minus_current_ref_384_256", {}
        )
        probe_gain = float(probe["pairwise_target_success_rate"]) - float(
            row["current_probe_reference"]["pairwise_target_success_rate"]
        )
        if probe_gain >= 0.03:
            probe_moved = True
        if (
            float(heldout.get("mean_ds_384_256", 0.0)) >= 0.05
            and float(ci.get("lower", 0.0)) > 0.01
            and float(agg.get(EQ_768_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0))
            >= -0.05
            and float(agg.get(EQ_1200_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0))
            >= -0.03
            and float(
                agg.get(ASYM_1200_256_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0)
            )
            >= -0.03
            and float(probe["broad_anchor_top1_changed_rate"]) <= 0.03
            and float(probe["stability_top1_preservation"]) >= 0.99
            and row.get("gate_classification") in {None, "high_search_breakthrough"}
        ):
            successful = True
        if probe_gain >= 0.03 and float(heldout.get("mean_ds_384_256", 0.0)) < 0.05:
            eval_regressed = True
    if successful:
        return "pairwise_update_success"
    if not probe_moved:
        return "pairwise_update_too_weak"
    if eval_regressed:
        return "pairwise_update_still_harmful"
    return "policy_head_learning_blocked"


def write_report(*, summary: dict[str, Any], report_path: Path) -> None:
    probe_rows = []
    aborted_rows = []
    training_rows = []
    fixed_large_rows = []
    heldout_rows = []
    bootstrap_rows = []
    p0_rows = []
    for row in summary["candidates"]:
        probe = row["probe_metrics"]
        probe_rows.append(
            [
                row["candidate"],
                fmt(probe["pairwise_target_success_rate"]),
                fmt(probe["mean_target_margin_improvement"]),
                fmt(probe["stability_top1_preservation"]),
                fmt(probe["broad_anchor_top1_changed_rate"]),
                fmt(probe["mean_anchor_kl_current_to_candidate"]),
                fmt(probe["effective_search_profile_aware_puct_agreement"]),
                str(row.get("aborted", False)),
            ]
        )
        if row.get("aborted"):
            aborted_rows.append(
                [row["candidate"], "; ".join(row.get("abort_reasons", []))]
            )
        training_rows.append(
            [
                row["candidate"],
                fmt(row.get("pairwise_loss")),
                fmt(row.get("behavior_anchor_loss")),
                fmt(row.get("total_loss")),
                fmt(row.get("validation_loss")),
                row.get("checkpoint_sha256", "n/a"),
                row.get("artifact_weights_sha256", "n/a"),
                fmt(row.get("delta_norm_vs_current_checkpoint")),
            ]
        )
        if "fixed_large_budget_results" in row:
            fixed_large_rows.append(
                [
                    row["candidate"],
                    fmt(row["fixed_large_budget_results"][PRIMARY_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"]["768:256"]["ds"]),
                    fmt(row["fixed_large_budget_results"][EQ_768_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"][EQ_1200_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"][ASYM_1200_256_BUDGET]["ds"]),
                    fmt(row["fixed_large_budget_results"]["256:768"]["ds"]),
                ]
            )
        if "heldout_summary" in row:
            heldout_rows.append(
                [
                    row["candidate"],
                    fmt(row["heldout_summary"].get("mean_ds_384_256")),
                    fmt(row["heldout_summary"].get("worst_suite_ds_384_256")),
                ]
            )
        for budget_pair in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            EQ_1200_BUDGET,
            ASYM_1200_256_BUDGET,
        ):
            ci = row.get("bootstrap_cis", {}).get(
                f"{row['candidate']}_minus_current_ref_{budget_pair.replace(':', '_')}"
            )
            if ci is None:
                continue
            bootstrap_rows.append(
                [
                    f"{row['candidate']}_minus_current_ref_{budget_pair.replace(':', '_')}",
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )
        if "heldout_aggregate" in row:
            p0_rows.append(
                [
                    row["candidate"],
                    fmt(row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]),
                    fmt(row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]),
                    fmt(
                        row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]
                        - row["heldout_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]
                    ),
                    fmt(
                        row["heldout_aggregate"][PRIMARY_BUDGET][
                            "mean_duplicate_trajectory_count"
                        ]
                    ),
                ]
            )
    lines = [
        "# AlphaZero-Lite Post-Schedule Pairwise PUCT Update Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Inputs",
        "",
        f"- Current artifact weights SHA256: `{summary['inputs']['current_weights']['actual_sha256']}`",
        f"- Current source checkpoint SHA256: `{summary['inputs']['init_checkpoint']['sha256']}`",
        "",
        "## Promoted Search Schedule Confirmation",
        "",
        f"- Schedule manifest: `{json.dumps(summary['search_schedule'], sort_keys=True)}`",
        f"- Root policy mode: `{summary['search_options']['root_policy_mode']}`",
        f"- Tactical root bias: `{summary['search_options']['tactical_root_bias']}`",
        "",
        "## Dataset Build",
        "",
        f"- Pairwise target row count: `{summary['dataset']['pairwise_target_rows']}`",
        f"- Anchor row count: `{summary['dataset']['anchor_rows']}`",
        f"- Probe composition: `{json.dumps(summary['dataset']['probe_composition'], sort_keys=True)}`",
        "",
        "## Pairwise Loss Implementation Summary",
        "",
        "- `train.py` now accepts optional pairwise target files, pairwise loss weight, and margin.",
        "- Pairwise loss: `softplus(margin - (logit[puct_move] - logit[raw_move]))`.",
        "- Behavior anchors remain policy-only and keep the current distribution via cross-entropy.",
        "- Training remains restricted to `trainable_scope=policy_head`, `residual_v3`, `kalah_v3`, `hidden_sizes=96,3`.",
        "",
        "## Probe Metrics Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Pairwise success",
                "Margin improve",
                "Stability preserve",
                "Broad changed",
                "Anchor KL",
                "PUCT agree",
                "Aborted",
            ],
            probe_rows,
        ),
        "",
        "## Aborted-Candidate Table",
        "",
        markdown_table(["Candidate", "Reasons"], aborted_rows or [["none", "n/a"]]),
        "",
        "## Training Losses",
        "",
        markdown_table(
            [
                "Candidate",
                "Pairwise loss",
                "Behavior loss",
                "Total loss",
                "Validation loss",
                "Checkpoint SHA256",
                "Artifact weights SHA256",
                "Delta norm",
            ],
            training_rows,
        ),
        "",
        "## Fixed Large DS Table",
        "",
        markdown_table(
            [
                "Candidate",
                "384:256",
                "768:256",
                "768:768",
                "1200:1200",
                "1200:256",
                "256:768",
            ],
            fixed_large_rows,
        ),
        "",
        "## Held-Out Mean/Worst-Suite Table",
        "",
        markdown_table(
            ["Candidate", "Held-out mean 384:256", "Held-out worst-suite 384:256"],
            heldout_rows,
        ),
        "",
        "## Bootstrap CI",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%"], bootstrap_rows
        ),
        "",
        "## P0/P1 Split For 384:256",
        "",
        markdown_table(
            ["Candidate", "Mean P0", "Mean P1", "Gap", "Mean duplicates"], p0_rows
        ),
        "",
        "## Gate Classification",
        "",
        *[
            f"- {row['candidate']}: `{row.get('gate_classification', 'not_run')}`"
            for row in summary["candidates"]
        ],
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_dir = Path(args.current)
    current_weights_path = current_dir / "weights.json"
    init_checkpoint = Path(args.init_checkpoint)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    all_eval_suites = [fixed_large_suite, *heldout_suites]
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)
    budget_pairs = parse_budget_pairs(DEFAULT_BUDGETS)
    margins = parse_float_list(args.pairwise_margins)
    anchor_weights = parse_int_list(args.behavior_anchor_weights)

    for path, label in (
        (current_weights_path, "current weights"),
        (init_checkpoint, "init checkpoint"),
        (medium_suite, "medium suite"),
        (fixed_large_suite, "fixed large suite"),
    ):
        require_existing_file(path, label)
    for suite in heldout_suites:
        require_existing_file(suite, f"heldout suite {suite.name}")

    inputs = {
        "current_weights": verify_expected_hash(
            current_weights_path,
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact": build_input_summary(current_weights_path),
        "init_checkpoint": build_input_summary(init_checkpoint),
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suites
        },
    }

    evaluator = ArtifactEvaluator(current_dir)
    suite_rows = load_suite_rows(all_eval_suites)
    unique_states = unique_state_rows(suite_rows)
    records = build_search_records(
        state_rows=unique_states,
        budget_pairs=budget_pairs,
        evaluator=evaluator,
        default_c_puct=float(args.default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.tactical_root_bias),
        seed=args.seed,
    )
    grouped_records = group_records_by_state_hash(records)
    dataset = build_dataset(records=records, grouped=grouped_records)

    pairwise_file = workdir / PAIRWISE_FILENAME
    anchor_file = workdir / ANCHOR_FILENAME
    probe_file = workdir / PROBE_FILENAME
    write_jsonl(
        pairwise_file,
        [build_pairwise_training_row(row) for row in dataset["pairwise_rows"]],
    )
    write_jsonl(
        anchor_file, [build_anchor_training_row(row) for row in dataset["anchor_rows"]]
    )
    probe_rows = [
        *(
            build_probe_row(row, "pairwise")
            for row in dataset["pairwise_rows"][:TARGET_PAIRWISE_ROWS]
        ),
        *(
            build_probe_row(row, "stability")
            for row in dataset["stability_rows"][:TARGET_STABILITY_PROBE_ROWS]
        ),
        *(
            build_probe_row(row, "broad_anchor")
            for row in dataset["broad_anchor_probe_rows"][
                :TARGET_BROAD_ANCHOR_PROBE_ROWS
            ]
        ),
    ]
    write_jsonl(probe_file, probe_rows)

    current_candidate = candidate_spec(
        name="current_ref",
        artifact_dir=current_dir,
        checkpoint_path=init_checkpoint,
        epochs=0,
        margin=None,
        anchor_weight=None,
        report_name=current_dir.name,
    )
    candidates: list[dict[str, Any]] = [current_candidate]
    for margin in margins:
        for anchor_weight in anchor_weights:
            if margin == 0.05 and anchor_weight != 4:
                continue
            for epochs in (1, 2):
                candidate_name = f"pairwise_margin{int(round(margin * 100)):03d}_kl{anchor_weight}_e{epochs}"
                checkpoint_path = workdir / candidate_name / f"{candidate_name}.npz"
                artifact_dir = workdir / candidate_name / "artifact"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                metrics = run_train_command(
                    pairwise_file=pairwise_file,
                    anchor_file=anchor_file,
                    init_checkpoint=init_checkpoint,
                    out_path=checkpoint_path,
                    margin=margin,
                    anchor_weight=anchor_weight,
                    epochs=epochs,
                    seed=args.seed,
                    timeout=args.timeout,
                )
                export_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    out_dir=str(artifact_dir),
                    version=candidate_name,
                    policy_loss=float(metrics.get("pairwise_loss", 0.0)),
                    value_loss=0.0,
                )
                delta_norm, _relative = compute_param_delta_norm(
                    checkpoint_path, init_checkpoint
                )
                candidates.append(
                    {
                        **candidate_spec(
                            name=candidate_name,
                            artifact_dir=artifact_dir,
                            checkpoint_path=checkpoint_path,
                            epochs=epochs,
                            margin=margin,
                            anchor_weight=anchor_weight,
                        ),
                        **metrics,
                        "validation_loss": float(metrics.get("best_val_loss", 0.0)),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "artifact_weights_sha256": sha256_file(
                            artifact_dir / "weights.json"
                        ),
                        "delta_norm_vs_current_checkpoint": delta_norm,
                    }
                )

    current_probe = evaluate_probe_candidate(
        artifact_path=current_dir, probe_rows=probe_rows
    )
    for row in candidates:
        artifact_path = Path(row["artifact_dir"])
        row["probe_metrics"] = evaluate_probe_candidate(
            artifact_path=artifact_path,
            probe_rows=probe_rows,
        )
        row["current_probe_reference"] = current_probe

    best_anchor_kl = min(
        float(row["probe_metrics"]["mean_anchor_kl_current_to_candidate"])
        for row in candidates
        if row["candidate"] != "current_ref"
    )
    for row in candidates:
        if row["candidate"] == "current_ref":
            row["aborted"] = False
            row["abort_reasons"] = []
            continue
        probe = row["probe_metrics"]
        reasons: list[str] = []
        if float(probe["broad_anchor_top1_changed_rate"]) > 0.03:
            reasons.append("broad-anchor top1 changed rate > 3%")
        if float(probe["stability_top1_preservation"]) < 0.99:
            reasons.append("stability preservation < 99%")
        if (
            float(probe["pairwise_target_success_rate"])
            - float(current_probe["pairwise_target_success_rate"])
            < 0.03
        ):
            reasons.append("pairwise success gain < +3 percentage points")
        if float(probe["mean_anchor_kl_current_to_candidate"]) > (best_anchor_kl * 2.0):
            reasons.append("anchor KL materially exceeds best lower-drift lane")
        row["aborted"] = bool(reasons)
        row["abort_reasons"] = reasons

    eval_candidates = [
        row
        for row in candidates
        if row["candidate"] == "current_ref" or not row.get("aborted")
    ]
    eval_candidate_dirs = [Path(row["artifact_dir"]) for row in eval_candidates]

    suite_reports: dict[str, dict[str, Any]] = {}
    suite_rows_map: dict[str, dict[str, Any]] = {}
    for suite_path in [medium_suite, fixed_large_suite, *heldout_suites]:
        report = run_opening_suite_benchmark_schedule(
            workdir=workdir / f"bench_{suite_path.stem}",
            suite=suite_path,
            current=current_dir,
            candidate_dirs=eval_candidate_dirs,
            budget_pairs=DEFAULT_BUDGETS,
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
        suite_reports[suite_path.stem] = report
        suite_rows_map[suite_path.stem] = suite_rows_from_report(
            report=report,
            eval_candidates=eval_candidates,
            suite_name=suite_path.stem,
        )

    heldout_suite_rows = {
        name: rows
        for name, rows in suite_rows_map.items()
        if name not in {medium_suite.stem, fixed_large_suite.stem}
    }
    fixed_large_rows = {fixed_large_suite.stem: suite_rows_map[fixed_large_suite.stem]}

    for row in candidates:
        if row["candidate"] not in {
            candidate["candidate"] for candidate in eval_candidates
        }:
            continue
        candidate_name = row["candidate"]
        fixed_report = find_candidate_report(
            suite_reports[fixed_large_suite.stem], row["report_candidate_name"]
        )
        row["fixed_large_budget_results"] = benchmark_budget_results(fixed_report)
        row["heldout_summary"] = heldout_summary(
            suite_reports, row["report_candidate_name"]
        )
        row["heldout_summary"] = {
            "available": bool(row["heldout_summary"].get("available", True)),
            "mean_ds_384_256": float(
                row["heldout_summary"].get("mean_ds_384_256", 0.0)
            ),
            "worst_suite_ds_384_256": float(
                row["heldout_summary"].get("worst_suite_ds_384_256", 0.0)
            ),
        }
        row["heldout_aggregate"] = aggregate_budget_summary(
            suite_rows=heldout_suite_rows,
            candidate_name=candidate_name,
            current_name="current_ref",
            budget_pairs=budget_pairs,
        )
        row["fixed_large_aggregate"] = aggregate_budget_summary(
            suite_rows=fixed_large_rows,
            candidate_name=candidate_name,
            current_name="current_ref",
            budget_pairs=budget_pairs,
        )
        row["bootstrap_cis"] = {}
        if candidate_name != "current_ref":
            for budget_pair in (
                PRIMARY_BUDGET,
                EQ_768_BUDGET,
                EQ_1200_BUDGET,
                ASYM_1200_256_BUDGET,
            ):
                diffs = pooled_per_opening_differences(
                    suite_rows=heldout_suite_rows,
                    candidate_a=candidate_name,
                    candidate_b="current_ref",
                    budget_pair=budget_pair,
                    metric_key="disadvantaged_seat_score",
                )
                row["bootstrap_cis"][
                    f"{candidate_name}_minus_current_ref_{budget_pair.replace(':', '_')}"
                ] = bootstrap_ci(diffs, seed=args.seed, samples=args.bootstrap_samples)

    for row in candidates:
        if row["candidate"] == "current_ref":
            gate_report = run_default_gate_schedule(
                candidate_path=Path(row["artifact_dir"]),
                current_path=current_dir,
                out_path=workdir / "gate_current_ref.json",
                default_c_puct=float(args.default_c_puct),
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=float(args.tactical_root_bias),
                seed=args.seed,
                workers=args.workers,
            )
            row["gate_classification"] = gate_report.get("classification")
            continue
        if row.get("aborted"):
            row["gate_classification"] = "not_run"
            continue
        robust = (
            float(row["heldout_summary"].get("mean_ds_384_256", 0.0)) >= 0.05
            and float(
                row["heldout_aggregate"][EQ_768_BUDGET]["mean_delta_vs_current_ref"]
            )
            >= -0.05
            and float(
                row["heldout_aggregate"][EQ_1200_BUDGET]["mean_delta_vs_current_ref"]
            )
            >= -0.03
            and float(
                row["heldout_aggregate"][ASYM_1200_256_BUDGET][
                    "mean_delta_vs_current_ref"
                ]
            )
            >= -0.03
        )
        if not robust:
            row["gate_classification"] = "not_run"
            continue
        gate_report = run_default_gate_schedule(
            candidate_path=Path(row["artifact_dir"]),
            current_path=current_dir,
            out_path=workdir / f"gate_{row['candidate']}.json",
            default_c_puct=float(args.default_c_puct),
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=float(args.tactical_root_bias),
            seed=args.seed,
            workers=args.workers,
        )
        row["gate_classification"] = gate_report.get("classification")

    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": inputs,
        "search_schedule": schedule_definition(
            default_c_puct=float(args.default_c_puct),
            schedule=cpuct_schedule,
        ),
        "search_options": {
            "root_policy_mode": "deterministic",
            "tactical_root_bias": float(args.tactical_root_bias),
            "search_mode": "full",
        },
        "dataset": {
            "pairwise_target_rows": len(dataset["pairwise_rows"]),
            "anchor_rows": len(dataset["anchor_rows"]),
            "probe_composition": {
                "pairwise": min(len(dataset["pairwise_rows"]), TARGET_PAIRWISE_ROWS),
                "stability": min(
                    len(dataset["stability_rows"]), TARGET_STABILITY_PROBE_ROWS
                ),
                "broad_anchor": min(
                    len(dataset["broad_anchor_probe_rows"]),
                    TARGET_BROAD_ANCHOR_PROBE_ROWS,
                ),
            },
        },
        "candidates": candidates,
    }
    summary["classification"] = classify(candidates)
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(summary=summary, report_path=REPORT_PATH)


if __name__ == "__main__":
    main()
