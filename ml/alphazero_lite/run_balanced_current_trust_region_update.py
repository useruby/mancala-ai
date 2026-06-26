#!/usr/bin/env python3
"""Trust-region policy-head update for validated PUCT targets."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_balanced_current_puct_target_counterfactual import (  # noqa: E402
    disagreement_candidate,
    stability_candidate,
    unique_rows_by_state_hash,
)
from ml.alphazero_lite.run_balanced_opening_puct_replay import (  # noqa: E402
    ASYM_1200_256_BUDGET,
    EQ_1200_BUDGET,
    EQ_768_BUDGET,
    PRIMARY_BUDGET,
    aggregate_budget_summary,
    effective_sampling_fractions,
    large_suite_rows,
    markdown_table,
    parse_budget_pairs,
    parse_csv_paths,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    gate_budget_results,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    build_input_summary,
    require_existing_file,
    sha256_file,
    verify_expected_hash,
    write_json,
    write_jsonl,
)
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    compute_param_delta_norm,
    export_checkpoint,
    heldout_summary,
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_policy_target_from_distribution,
    encode_state,
)

SUMMARY_SCHEMA = "azlite_balanced_current_trust_region_update_v1"
SUMMARY_FILENAME = "summary_metrics.json"
VALIDATED_REPLAY_FILENAME = "validated_puct_disagreement_replay.jsonl"
BEHAVIOR_ANCHOR_FILENAME = "behavior_anchor_replay.jsonl"
PROBE_STATE_FILENAME = "probe_state_set.jsonl"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-balanced-current-trust-region-update-results.md"
)
TARGET_DISAGREEMENT_ROWS = 2000
MIN_DISAGREEMENT_ROWS = 1000
TARGET_ANCHOR_ROWS = 8000
MIN_ANCHOR_ROWS = 4000
TARGET_STABILITY_PROBE_ROWS = 2000
TARGET_BROAD_PROBE_ROWS = 4000
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
TARGET_POLICY_MODE = "sharpened"
TARGET_VALUE_MODE = "sharpened"
GATE_DELTA_THRESHOLD = 0.05


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--expected-init-checkpoint-sha256", required=True)
    parser.add_argument("--consensus-state-table", required=True)
    parser.add_argument("--counterfactual-summary", required=True)
    parser.add_argument("--generic-bootstrap", required=True)
    parser.add_argument("--random-teacher", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--behavior-kl-weights", default="1,4,8")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def legal_mask_valid(policy: list[float], legal_moves: list[int]) -> bool:
    if len(policy) != 6:
        return False
    total = sum(float(value) for value in policy)
    if abs(total - 1.0) > 1e-6:
        return False
    legal_set = {int(move) for move in legal_moves}
    return all(
        move in legal_set or float(probability) <= 1e-6
        for move, probability in enumerate(policy)
    )


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def kl_divergence(p: list[float], q: list[float], legal_moves: list[int]) -> float:
    eps = 1e-12
    total = 0.0
    for move in legal_moves:
        p_value = max(float(p[move]), eps)
        q_value = max(float(q[move]), eps)
        total += p_value * math.log(p_value / q_value)
    return total


def state_hash_sort_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["state_hash"]), -int(row.get("occurrence_count", 0)))


def priority_disagreement_rows(
    rows: list[dict[str, Any]], counterfactual_hashes: set[str], rng: random.Random
) -> list[dict[str, Any]]:
    preferred = [row for row in rows if str(row["state_hash"]) in counterfactual_hashes]
    fallback = [
        row for row in rows if str(row["state_hash"]) not in counterfactual_hashes
    ]
    rng.shuffle(preferred)
    rng.shuffle(fallback)
    preferred.sort(
        key=lambda row: (
            -int(str(row["state_hash"]) in counterfactual_hashes),
            -float(row["searches"]["1200"]["top1_visit_share"]),
            -float(row["searches"]["1200"]["kl_search_raw"]),
            -float(row.get("raw_margin", 0.0)),
            str(row["state_hash"]),
        )
    )
    fallback.sort(
        key=lambda row: (
            -float(row["searches"]["1200"]["top1_visit_share"]),
            -float(row["searches"]["1200"]["kl_search_raw"]),
            -float(row.get("raw_margin", 0.0)),
            str(row["state_hash"]),
        )
    )
    return preferred + fallback


def sample_round_robin(
    rows: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    *,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
    skip_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    skip_hashes = set() if skip_hashes is None else set(skip_hashes)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    queues: dict[tuple[Any, ...], deque[dict[str, Any]]] = {}
    for key, members in grouped.items():
        rng.shuffle(members)
        queues[key] = deque(members)
    ordered_keys = list(queues)
    rng.shuffle(ordered_keys)
    selected: list[dict[str, Any]] = []
    selected_hashes = set(skip_hashes)
    while ordered_keys and len(selected) < target:
        next_keys: list[tuple[Any, ...]] = []
        for key in ordered_keys:
            queue = queues[key]
            while queue and str(queue[0]["state_hash"]) in selected_hashes:
                queue.popleft()
            if not queue:
                continue
            row = queue.popleft()
            state_hash = str(row["state_hash"])
            if state_hash in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(state_hash)
            if queue:
                next_keys.append(key)
            if len(selected) >= target:
                break
        ordered_keys = next_keys
    return selected


def broad_anchor_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("suite_name", "unknown")),
        str(row.get("phase", "unknown")),
        str(row.get("dominant_seat_context", "unknown")),
        int(row.get("raw_top1", -1)),
    )


def build_supervised_row(row: dict[str, Any]) -> dict[str, Any]:
    target_policy = build_policy_target_from_distribution(
        row["searches"]["1200"]["policy"], mode=TARGET_POLICY_MODE
    )
    return {
        "state_hash": str(row["state_hash"]),
        "state": encode_state(row["state"], input_encoding="kalah_v3"),
        "policy": [float(value) for value in target_policy],
        "value": float(row["searches"]["1200"]["value"]),
        "player": int(row["state"]["current_player"]),
        "move_index": int(row.get("move_index", 0)),
        "winner": None,
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "policy_target_mode": TARGET_POLICY_MODE,
        "policy_target_actual_mode": TARGET_POLICY_MODE,
        "value_target_mode": TARGET_VALUE_MODE,
        "bucket": "validated_puct_disagreement_replay",
        "teacher_source": "puct_consensus_1200",
        "suite_name": str(row.get("suite_name", "unknown")),
        "suite_names": list(row.get("suite_names", [])),
        "opening_prefix": list(row.get("opening_prefix", [])),
        "phase": str(row.get("phase", "unknown")),
        "dominant_seat_context": str(row.get("dominant_seat_context", "unknown")),
        "raw_top1": row.get("raw_top1"),
        "search_top1": row["searches"]["1200"].get("top1"),
        "search_top1_visit_share": float(
            row["searches"]["1200"].get("top1_visit_share", 0.0)
        ),
        "raw_margin": float(row.get("raw_margin", 0.0)),
    }


def build_behavior_anchor_row(
    row: dict[str, Any], *, anchor_kind: str
) -> dict[str, Any]:
    current_policy = [float(value) for value in row["raw_policy"]]
    return {
        "state_hash": str(row["state_hash"]),
        "state": encode_state(row["state"], input_encoding="kalah_v3"),
        "policy": current_policy,
        "value": float(row.get("raw_value", 0.0)),
        "player": int(row["state"]["current_player"]),
        "move_index": int(row.get("move_index", 0)),
        "winner": None,
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "policy_target_mode": TARGET_POLICY_MODE,
        "policy_target_actual_mode": TARGET_POLICY_MODE,
        "value_target_mode": TARGET_VALUE_MODE,
        "bucket": "behavior_anchor_replay",
        "anchor_kind": anchor_kind,
        "suite_name": str(row.get("suite_name", "unknown")),
        "suite_names": list(row.get("suite_names", [])),
        "opening_prefix": list(row.get("opening_prefix", [])),
        "phase": str(row.get("phase", "unknown")),
        "dominant_seat_context": str(row.get("dominant_seat_context", "unknown")),
        "raw_top1": row.get("raw_top1"),
        "puct_top1": row["searches"]["1200"].get("top1"),
        "search_top1_visit_share": float(
            row["searches"]["1200"].get("top1_visit_share", 0.0)
        ),
        "raw_margin": float(row.get("raw_margin", 0.0)),
    }


def build_probe_row(row: dict[str, Any], *, probe_kind: str) -> dict[str, Any]:
    return {
        "probe_kind": probe_kind,
        "state_hash": str(row["state_hash"]),
        "state": row["state"],
        "encoded_state": encode_state(row["state"], input_encoding="kalah_v3"),
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "current_policy": [float(value) for value in row["raw_policy"]],
        "puct_policy": [float(value) for value in row["searches"]["1200"]["policy"]],
        "current_top1": row.get("raw_top1"),
        "puct_top1": row["searches"]["1200"].get("top1"),
        "phase": str(row.get("phase", "unknown")),
        "seat_context": str(row.get("dominant_seat_context", "unknown")),
        "suite_name": str(row.get("suite_name", "unknown")),
    }


def evaluate_probe_candidate(
    candidate_name: str,
    artifact_path: Path,
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    disagreement_total = 0
    disagreement_match = 0
    disagreement_kls: list[float] = []
    stability_total = 0
    stability_preserved = 0
    broad_total = 0
    broad_preserved = 0
    anchor_kls: list[float] = []
    anchor_changed_total = 0
    broad_changed_total = 0
    changed_breakdown: dict[str, int] = defaultdict(int)
    for row in probe_rows:
        game = KalahGame.from_state(row["state"])
        _logits, candidate_policy, _value = artifact_forward_details(evaluator, game)
        legal_moves = [int(move) for move in row["legal_moves"]]
        current_top1 = top_policy_move(row["current_policy"], legal_moves)
        candidate_top1 = top_policy_move(candidate_policy, legal_moves)
        puct_top1 = top_policy_move(row["puct_policy"], legal_moves)
        if row["probe_kind"] == "disagreement":
            disagreement_total += 1
            if candidate_top1 == puct_top1:
                disagreement_match += 1
            disagreement_kls.append(
                kl_divergence(candidate_policy, row["puct_policy"], legal_moves)
            )
            continue
        anchor_kls.append(
            kl_divergence(row["current_policy"], candidate_policy, legal_moves)
        )
        top1_changed = candidate_top1 != current_top1
        anchor_changed_total += int(top1_changed)
        if row["probe_kind"] == "stability":
            stability_total += 1
            if candidate_top1 == current_top1:
                stability_preserved += 1
        elif row["probe_kind"] == "broad_anchor":
            broad_total += 1
            if candidate_top1 == current_top1:
                broad_preserved += 1
            if top1_changed:
                broad_changed_total += 1
                changed_breakdown[f"{row['phase']}::{row['seat_context']}"] += 1
    return {
        "candidate": candidate_name,
        "disagreement_top1_agreement_with_puct": disagreement_match
        / max(disagreement_total, 1),
        "disagreement_kl_candidate_to_puct": statistics.fmean(disagreement_kls)
        if disagreement_kls
        else 0.0,
        "stability_top1_agreement_with_current": stability_preserved
        / max(stability_total, 1),
        "broad_anchor_top1_agreement_with_current": broad_preserved
        / max(broad_total, 1),
        "mean_anchor_kl_current_to_candidate": statistics.fmean(anchor_kls)
        if anchor_kls
        else 0.0,
        "anchor_top1_changed_rate": anchor_changed_total
        / max(stability_total + broad_total, 1),
        "broad_anchor_top1_changed_rate": broad_changed_total / max(broad_total, 1),
        "changed_top1_phase_seat_breakdown": dict(sorted(changed_breakdown.items())),
        "disagreement_states": disagreement_total,
        "stability_states": stability_total,
        "broad_anchor_states": broad_total,
    }


def build_candidate_spec(
    *,
    name: str,
    artifact_dir: Path,
    checkpoint_path: Path,
    epochs: int,
    behavior_loss_weight: int,
    train_metrics: dict[str, Any] | None,
    report_candidate_name: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "report_candidate_name": report_candidate_name or name,
        "artifact_dir": str(artifact_dir),
        "checkpoint_path": str(checkpoint_path),
        "epochs": epochs,
        "behavior_loss_weight": behavior_loss_weight,
        "train_metrics": train_metrics,
        "trainable_scope": "policy_head" if epochs > 0 else "none",
    }


def run_train_command(
    *,
    data_files: str,
    replay_weights: str,
    behavior_anchor_files: str,
    behavior_loss_weight: int,
    init_checkpoint: Path,
    out_path: Path,
    epochs: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--data-files",
        data_files,
        "--replay-weights",
        replay_weights,
        "--behavior-anchor-files",
        behavior_anchor_files,
        "--behavior-loss-weight",
        str(behavior_loss_weight),
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
        "--value-loss",
        "huber",
        "--value-loss-weight",
        "0.3",
        "--grad-clip",
        "1.0",
        "--save-top-k",
        "0",
        "--out",
        str(out_path),
        "--policy-target-mode",
        TARGET_POLICY_MODE,
        "--value-target-mode",
        TARGET_VALUE_MODE,
        "--lr-scheduler",
        "none",
        "--seed",
        str(seed),
        "--trainable-scope",
        "policy_head",
        "--save-epochs",
        str(epochs),
    ]
    start = time.time()
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
    metrics: dict[str, Any] = {
        "training_elapsed_s": time.time() - start,
        "behavior_loss_weight": int(behavior_loss_weight),
    }
    for line in result.stdout.splitlines():
        line = line.strip()
        for key in (
            "policy_loss",
            "value_loss",
            "behavior_anchor_loss",
            "total_loss",
            "best_val_loss",
        ):
            token = f"{key}="
            if line.startswith(token):
                metrics[key] = float(line.split("=", 1)[1])
        if line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            metrics[f"epoch_{parts[0].split('=', 1)[1]}_path"] = parts[1].split("=", 1)[
                1
            ]
    for line in result.stderr.splitlines():
        for key in ("trainable_params", "frozen_params", "total_params"):
            token = f"{key}="
            if token in line:
                for part in line.split():
                    if part.startswith(token):
                        metrics[key] = int(part.split("=", 1)[1])
    return metrics


def load_or_run_suite_report(
    *,
    workdir: Path,
    suite: Path,
    current: Path,
    eval_candidates: list[dict[str, Any]],
    seed: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    report_path = workdir / "temperature_benchmark_report.json"
    expected_candidates = {
        str(candidate["report_candidate_name"]) for candidate in eval_candidates
    }
    if report_path.is_file():
        cached = load_json(report_path)
        cached_candidates = {
            str(candidate_report.get("candidate"))
            for candidate_report in cached.get("temperature_reports", [])[0]
            .get("seed_reports", [])[0]
            .get("candidate_reports", [])
        }
        if expected_candidates.issubset(cached_candidates):
            return cached
    return run_opening_suite_benchmark(
        workdir=str(workdir),
        suite=str(suite),
        current=str(current),
        candidates=",".join(
            str(candidate["artifact_dir"]) for candidate in eval_candidates
        ),
        budget_pairs=DEFAULT_BUDGET_PAIRS,
        games_per_opening=2,
        seed=seed,
        workers=workers,
        timeout=timeout,
    )


def load_or_run_gate_report(
    *,
    path: Path,
    candidate_path: Path,
    current_path: Path,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if path.is_file():
        return load_json(path)
    return run_default_gate(
        candidate_path=str(candidate_path),
        current_path=str(current_path),
        out=str(path),
        seed=seed,
        workers=workers,
    )


def classify_run(
    *,
    current_probe: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> str:
    successful = []
    probe_improved_eval_regressed = False
    all_anchor_good_but_weak = True
    any_too_weak = False
    for row in candidate_rows:
        if row.get("aborted"):
            any_too_weak = True
            continue
        probe = row["probe_metrics"]
        agg = row.get("large_suite_aggregate", {})
        delta_384 = float(
            agg.get(PRIMARY_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0)
        )
        delta_768 = float(
            agg.get(EQ_768_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0)
        )
        delta_1200 = float(
            agg.get(EQ_1200_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0)
        )
        delta_1200_256 = float(
            agg.get(ASYM_1200_256_BUDGET, {}).get("mean_delta_vs_current_ref", 0.0)
        )
        ci = row.get("bootstrap_cis", {}).get(
            f"{row['candidate']}_minus_balanced_current_ref_384_256", {}
        )
        probe_gain = float(probe["disagreement_top1_agreement_with_puct"]) - float(
            current_probe["disagreement_top1_agreement_with_puct"]
        )
        if probe_gain >= 0.03 and delta_384 < 0.05:
            probe_improved_eval_regressed = True
        if probe_gain >= 0.03:
            all_anchor_good_but_weak = False
        if (
            delta_384 >= 0.05
            and float(ci.get("lower", 0.0)) > 0.01
            and delta_768 >= -0.08
            and delta_1200 >= -0.03
            and delta_1200_256 >= -0.03
            and float(probe["stability_top1_agreement_with_current"]) >= 0.98
            and float(probe["broad_anchor_top1_changed_rate"]) <= 0.05
            and row.get("gate_classification") in {None, "high_search_breakthrough"}
        ):
            successful.append(row["candidate"])
        if float(probe["broad_anchor_top1_changed_rate"]) > 0.05 or (
            delta_384 < 0.05 and (delta_768 < 0.0 or delta_1200 < 0.0)
        ):
            any_too_weak = True
    if successful:
        return "trust_region_update_success"
    if probe_improved_eval_regressed:
        return "update_format_still_harmful"
    if all_anchor_good_but_weak:
        return "trust_region_too_strong"
    if any_too_weak:
        return "trust_region_too_weak"
    return "supervised_policy_head_path_blocked"


def write_report(
    *,
    summary: dict[str, Any],
    report_path: Path,
) -> None:
    candidate_rows = summary["candidates"]
    probe_headers = [
        "Candidate",
        "PUCT agree",
        "Cand||PUCT KL",
        "Stability preserve",
        "Broad preserve",
        "Anchor KL",
        "Broad changed",
        "Aborted",
    ]
    probe_table_rows = [
        [
            row["candidate"],
            fmt(row["probe_metrics"]["disagreement_top1_agreement_with_puct"]),
            fmt(row["probe_metrics"]["disagreement_kl_candidate_to_puct"]),
            fmt(row["probe_metrics"]["stability_top1_agreement_with_current"]),
            fmt(row["probe_metrics"]["broad_anchor_top1_agreement_with_current"]),
            fmt(row["probe_metrics"]["mean_anchor_kl_current_to_candidate"]),
            fmt(row["probe_metrics"]["broad_anchor_top1_changed_rate"]),
            str(row.get("aborted", False)),
        ]
        for row in candidate_rows
    ]
    aborted_rows = [
        [row["candidate"], "; ".join(row.get("abort_reasons", []))]
        for row in candidate_rows
        if row.get("aborted")
    ]
    fixed_large_rows = [
        [
            row["candidate"],
            fmt(row["fixed_large_budget_results"][PRIMARY_BUDGET]["ds"]),
            fmt(row["fixed_large_budget_results"]["768:256"]["ds"]),
            fmt(row["fixed_large_budget_results"][EQ_768_BUDGET]["ds"]),
            fmt(row["fixed_large_budget_results"][EQ_1200_BUDGET]["ds"]),
            fmt(row["fixed_large_budget_results"][ASYM_1200_256_BUDGET]["ds"]),
            fmt(row["fixed_large_budget_results"]["256:768"]["ds"]),
        ]
        for row in candidate_rows
        if "fixed_large_budget_results" in row
    ]
    heldout_rows = [
        [
            row["candidate"],
            fmt(row["heldout_summary"].get("mean_ds_384_256")),
            fmt(row["heldout_summary"].get("worst_suite_ds_384_256")),
        ]
        for row in candidate_rows
        if "heldout_summary" in row
    ]
    bootstrap_rows = []
    for row in candidate_rows:
        for budget_pair in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            EQ_1200_BUDGET,
            ASYM_1200_256_BUDGET,
        ):
            key = f"{row['candidate']}_minus_balanced_current_ref_{budget_pair.replace(':', '_')}"
            ci = row.get("bootstrap_cis", {}).get(key)
            if ci is None:
                continue
            bootstrap_rows.append(
                [
                    key,
                    fmt(ci.get("mean")),
                    fmt(ci.get("lower")),
                    fmt(ci.get("upper")),
                ]
            )
    p0_rows = [
        [
            row["candidate"],
            fmt(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]),
            fmt(row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]),
            fmt(
                row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p1_score"]
                - row["large_suite_aggregate"][PRIMARY_BUDGET]["mean_p0_score"]
            ),
            fmt(
                row["large_suite_aggregate"][PRIMARY_BUDGET][
                    "mean_duplicate_trajectory_count"
                ]
            ),
        ]
        for row in candidate_rows
        if "large_suite_aggregate" in row
    ]
    gate_lines = [
        f"- {row['candidate']}: `{row.get('gate_classification', 'not_run')}`"
        for row in candidate_rows
    ]
    report = [
        "# AlphaZero-Lite Balanced Current Trust-Region Update Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Inputs",
        "",
        f"- Current artifact weights SHA256: `{summary['inputs']['current_weights']['actual_sha256']}`",
        f"- Current source checkpoint SHA256: `{summary['inputs']['init_checkpoint']['actual_sha256']}`",
        "",
        "## Dataset Build",
        "",
        f"- Validated disagreement rows: `{summary['dataset']['validated_disagreement_rows']}`",
        f"- Behavior anchor rows: `{summary['dataset']['behavior_anchor_rows']}`",
        f"- Probe-state composition: `{json.dumps(summary['dataset']['probe_composition'], sort_keys=True)}`",
        "",
        "## Probe Metrics",
        "",
        markdown_table(probe_headers, probe_table_rows),
        "",
        "## Aborted Candidates",
        "",
        markdown_table(["Candidate", "Reasons"], aborted_rows or [["none", "n/a"]]),
        "",
        "## Training Losses And Artifacts",
        "",
        markdown_table(
            [
                "Candidate",
                "Policy loss",
                "Value loss",
                "Behavior loss",
                "Total loss",
                "Validation loss",
                "Checkpoint SHA256",
                "Artifact weights SHA256",
                "Delta norm",
            ],
            [
                [
                    row["candidate"],
                    fmt(row.get("policy_loss")),
                    fmt(row.get("value_loss")),
                    fmt(row.get("behavior_anchor_loss")),
                    fmt(row.get("total_loss")),
                    fmt(row.get("validation_loss")),
                    row.get("checkpoint_sha256", "n/a"),
                    row.get("artifact_weights_sha256", "n/a"),
                    fmt(row.get("delta_norm_vs_current_checkpoint")),
                ]
                for row in candidate_rows
            ],
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
        "## Held-Out Mean/Worst DS Table",
        "",
        markdown_table(
            ["Candidate", "Held-out mean 384:256", "Held-out worst-suite 384:256"],
            heldout_rows,
        ),
        "",
        "## Bootstrap CI Table",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%"], bootstrap_rows
        ),
        "",
        "## P0/P1 Split At 384:256",
        "",
        markdown_table(
            ["Candidate", "Mean P0", "Mean P1", "Gap", "Mean duplicates"], p0_rows
        ),
        "",
        "## Gate",
        "",
        *gate_lines,
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_dir = Path(args.current)
    current_weights_path = current_dir / "weights.json"
    init_checkpoint = Path(args.init_checkpoint)
    consensus_state_table = Path(args.consensus_state_table)
    counterfactual_summary = Path(args.counterfactual_summary)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    behavior_kl_weights = parse_int_list(args.behavior_kl_weights)
    budget_pairs = parse_budget_pairs(DEFAULT_BUDGET_PAIRS)

    for path, label in (
        (current_weights_path, "current weights"),
        (init_checkpoint, "init checkpoint"),
        (consensus_state_table, "consensus state table"),
        (counterfactual_summary, "counterfactual summary"),
        (generic_bootstrap, "generic bootstrap"),
        (random_teacher, "random teacher"),
        (medium_suite, "medium suite"),
        (fixed_large_suite, "fixed large suite"),
    ):
        require_existing_file(path, label)
    for suite_path in heldout_suites:
        require_existing_file(suite_path, f"heldout suite {suite_path.name}")

    input_summary = {
        "current_weights": verify_expected_hash(
            current_weights_path,
            args.expected_current_weights_sha256,
            "current weights",
        ),
        "init_checkpoint": verify_expected_hash(
            init_checkpoint,
            args.expected_init_checkpoint_sha256,
            "init checkpoint",
        ),
        "consensus_state_table": build_input_summary(consensus_state_table),
        "counterfactual_summary": build_input_summary(counterfactual_summary),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": [build_input_summary(path) for path in heldout_suites],
    }

    rng = random.Random(args.seed)
    consensus_rows = unique_rows_by_state_hash(read_jsonl(consensus_state_table))
    counterfactual = load_json(counterfactual_summary)
    disagreement_results = read_jsonl(
        Path(counterfactual["outputs"]["disagreement_results"])
    )
    counterfactual_hashes = {str(row["state_hash"]) for row in disagreement_results}

    disagreement_candidates = [
        row for row in consensus_rows if disagreement_candidate(row)
    ]
    disagreement_candidates = priority_disagreement_rows(
        disagreement_candidates, counterfactual_hashes, rng
    )
    disagreement_selected = disagreement_candidates[:TARGET_DISAGREEMENT_ROWS]
    if len(disagreement_selected) < MIN_DISAGREEMENT_ROWS:
        raise RuntimeError(
            f"validated disagreement rows below minimum: {len(disagreement_selected)}"
        )

    disagreement_hashes = {str(row["state_hash"]) for row in disagreement_selected}
    stability_candidates = [
        row
        for row in consensus_rows
        if stability_candidate(row)
        and str(row["state_hash"]) not in disagreement_hashes
    ]
    stability_selected = sample_round_robin(
        stability_candidates,
        TARGET_STABILITY_PROBE_ROWS,
        rng,
        key_fn=broad_anchor_group_key,
    )

    used_anchor_hashes = disagreement_hashes | {
        str(row["state_hash"]) for row in stability_selected
    }
    broad_anchor_candidates = [
        row
        for row in consensus_rows
        if bool(row.get("legal_mask_valid"))
        and len(row.get("legal_moves", [])) > 1
        and legal_mask_valid(row.get("raw_policy", []), row.get("legal_moves", []))
        and str(row["state_hash"]) not in used_anchor_hashes
    ]
    broad_anchor_selected = sample_round_robin(
        broad_anchor_candidates,
        TARGET_ANCHOR_ROWS - len(stability_selected),
        rng,
        key_fn=broad_anchor_group_key,
    )
    behavior_anchor_rows = stability_selected + broad_anchor_selected
    if len(behavior_anchor_rows) < MIN_ANCHOR_ROWS:
        raise RuntimeError(
            f"behavior anchor rows below minimum: {len(behavior_anchor_rows)}"
        )

    probe_broad_rows = broad_anchor_selected[:TARGET_BROAD_PROBE_ROWS]
    probe_rows = [
        *[
            build_probe_row(row, probe_kind="disagreement")
            for row in disagreement_selected
        ],
        *[build_probe_row(row, probe_kind="stability") for row in stability_selected],
        *[build_probe_row(row, probe_kind="broad_anchor") for row in probe_broad_rows],
    ]

    validated_replay_rows = [build_supervised_row(row) for row in disagreement_selected]
    behavior_anchor_replay_rows = [
        build_behavior_anchor_row(
            row,
            anchor_kind=(
                "stability"
                if str(row["state_hash"]) not in disagreement_hashes
                and stability_candidate(row)
                else "broad"
            ),
        )
        for row in behavior_anchor_rows
    ]

    validated_replay_path = workdir / VALIDATED_REPLAY_FILENAME
    behavior_anchor_path = workdir / BEHAVIOR_ANCHOR_FILENAME
    probe_path = workdir / PROBE_STATE_FILENAME
    write_jsonl(validated_replay_path, validated_replay_rows)
    write_jsonl(behavior_anchor_path, behavior_anchor_replay_rows)
    write_jsonl(probe_path, probe_rows)

    current_candidate = build_candidate_spec(
        name="balanced_current_ref",
        artifact_dir=current_dir,
        checkpoint_path=init_checkpoint,
        epochs=0,
        behavior_loss_weight=0,
        train_metrics=None,
        report_candidate_name="current",
    )

    candidates: list[dict[str, Any]] = [current_candidate]
    pr129_dir = Path(
        "/tmp/azlite_balanced_current_puct_consensus/consensus_w8s4_policy_head_e1"
    )
    if (pr129_dir / "weights.json").is_file() and (pr129_dir / "model.npz").is_file():
        candidates.append(
            build_candidate_spec(
                name="pr129_consensus_e1_ref",
                artifact_dir=pr129_dir,
                checkpoint_path=pr129_dir / "model.npz",
                epochs=0,
                behavior_loss_weight=0,
                train_metrics=None,
                report_candidate_name=pr129_dir.name,
            )
        )

    data_files = f"{generic_bootstrap},{random_teacher},{validated_replay_path}"
    replay_weights = "4,1,8"
    training_candidates: list[dict[str, Any]] = []
    for behavior_weight in behavior_kl_weights:
        for epochs in (1, 2):
            name = f"trust_region_kl{behavior_weight}_e{epochs}"
            checkpoint_path = workdir / f"{name}.npz"
            artifact_dir = workdir / name
            metrics_path = workdir / f"{name}_train_metrics.json"
            if (
                checkpoint_path.is_file()
                and (artifact_dir / "weights.json").is_file()
                and metrics_path.is_file()
            ):
                train_metrics = load_json(metrics_path)
            else:
                train_metrics = run_train_command(
                    data_files=data_files,
                    replay_weights=replay_weights,
                    behavior_anchor_files=str(behavior_anchor_path),
                    behavior_loss_weight=behavior_weight,
                    init_checkpoint=init_checkpoint,
                    out_path=checkpoint_path,
                    epochs=epochs,
                    seed=args.seed,
                    timeout=args.timeout,
                )
                export_checkpoint(
                    checkpoint_path=str(checkpoint_path),
                    out_dir=str(artifact_dir),
                    version=name,
                    policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                    value_loss=float(train_metrics.get("value_loss", 0.0)),
                )
                write_json(metrics_path, train_metrics)
            training_candidates.append(
                build_candidate_spec(
                    name=name,
                    artifact_dir=artifact_dir,
                    checkpoint_path=checkpoint_path,
                    epochs=epochs,
                    behavior_loss_weight=behavior_weight,
                    train_metrics=train_metrics,
                    report_candidate_name=artifact_dir.name,
                )
            )
    candidates.extend(training_candidates)

    probe_metrics_by_candidate = {
        candidate["name"]: evaluate_probe_candidate(
            candidate["name"], Path(candidate["artifact_dir"]), probe_rows
        )
        for candidate in candidates
    }
    current_probe = probe_metrics_by_candidate["balanced_current_ref"]
    best_lower_weight_kl: dict[int, float] = {}
    aborted: dict[str, list[str]] = {}
    for candidate in training_candidates:
        reasons: list[str] = []
        probe_metrics = probe_metrics_by_candidate[candidate["name"]]
        probe_gain = float(
            probe_metrics["disagreement_top1_agreement_with_puct"]
        ) - float(current_probe["disagreement_top1_agreement_with_puct"])
        if float(probe_metrics["broad_anchor_top1_changed_rate"]) > 0.05:
            reasons.append("broad-anchor top-1 changed rate > 5%")
        if float(probe_metrics["stability_top1_agreement_with_current"]) < 0.98:
            reasons.append("stability top-1 preservation < 98%")
        lower_weights = [
            weight
            for weight in best_lower_weight_kl
            if weight < candidate["behavior_loss_weight"]
        ]
        if lower_weights:
            best_lower = min(best_lower_weight_kl[weight] for weight in lower_weights)
            if float(probe_metrics["mean_anchor_kl_current_to_candidate"]) > (
                2.0 * best_lower
            ):
                reasons.append("mean anchor KL exceeds 2x best lower-KL lane")
        if probe_gain < 0.03:
            reasons.append("disagreement top-1 agreement with PUCT improved by < +3pp")
        aborted[candidate["name"]] = reasons
        best_lower_weight_kl[candidate["behavior_loss_weight"]] = min(
            best_lower_weight_kl.get(candidate["behavior_loss_weight"], float("inf")),
            float(probe_metrics["mean_anchor_kl_current_to_candidate"]),
        )

    eval_candidates = [
        candidate
        for candidate in candidates
        if candidate["name"] in {"balanced_current_ref", "pr129_consensus_e1_ref"}
        or not aborted.get(candidate["name"])
    ]
    medium_report = load_or_run_suite_report(
        workdir=workdir / "eval_medium",
        suite=medium_suite,
        current=current_dir,
        eval_candidates=eval_candidates,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout,
    )
    fixed_large_report = load_or_run_suite_report(
        workdir=workdir / "eval_fixed_large",
        suite=fixed_large_suite,
        current=current_dir,
        eval_candidates=eval_candidates,
        seed=args.seed,
        workers=args.workers,
        timeout=args.timeout,
    )
    heldout_reports: dict[str, dict[str, Any]] = {}
    for suite_path in heldout_suites:
        heldout_reports[suite_path.stem] = load_or_run_suite_report(
            workdir=workdir / f"eval_{suite_path.stem}",
            suite=suite_path,
            current=current_dir,
            eval_candidates=eval_candidates,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    reports_for_large = {"fixed_large": fixed_large_report, **heldout_reports}
    suite_rows = large_suite_rows(reports=reports_for_large, candidates=eval_candidates)

    gate_targets = {"balanced_current_ref"}
    current_aggregate = aggregate_budget_summary(
        suite_rows, "balanced_current_ref", PRIMARY_BUDGET, "balanced_current_ref"
    )
    for candidate in eval_candidates:
        if not str(candidate["name"]).startswith("trust_region_"):
            continue
        aggregate = aggregate_budget_summary(
            suite_rows,
            str(candidate["name"]),
            PRIMARY_BUDGET,
            "balanced_current_ref",
        )
        if (
            float(aggregate["mean_ds"])
            >= float(current_aggregate["mean_ds"]) + GATE_DELTA_THRESHOLD
        ):
            gate_targets.add(str(candidate["name"]))

    gate_reports: dict[str, dict[str, Any]] = {}
    for candidate in eval_candidates:
        if str(candidate["name"]) not in gate_targets:
            continue
        gate_reports[str(candidate["name"])] = load_or_run_gate_report(
            path=workdir / "gate" / f"{candidate['name']}.json",
            candidate_path=Path(candidate["artifact_dir"]),
            current_path=current_dir,
            seed=args.seed,
            workers=args.workers,
        )

    summary_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        row: dict[str, Any] = {
            "candidate": candidate["name"],
            "epochs": candidate["epochs"],
            "behavior_loss_weight": candidate["behavior_loss_weight"],
            "probe_metrics": probe_metrics_by_candidate[candidate["name"]],
            "aborted": bool(aborted.get(candidate["name"])),
            "abort_reasons": aborted.get(candidate["name"], []),
        }
        artifact_dir = Path(candidate["artifact_dir"])
        checkpoint_path = Path(candidate["checkpoint_path"])
        row["checkpoint_sha256"] = sha256_file(checkpoint_path)
        row["artifact_weights_sha256"] = sha256_file(artifact_dir / "weights.json")
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, init_checkpoint
        )
        row["delta_norm_vs_current_checkpoint"] = delta_norm
        row["relative_delta_pct_vs_current_checkpoint"] = relative_delta_pct
        if candidate.get("train_metrics"):
            row.update(candidate["train_metrics"])
            row["validation_loss"] = candidate["train_metrics"].get("best_val_loss")
        if candidate in eval_candidates:
            row["medium_budget_results"] = benchmark_budget_results(
                next(
                    report
                    for report in medium_report["temperature_reports"][0][
                        "seed_reports"
                    ][0]["candidate_reports"]
                    if report.get("candidate") == candidate["report_candidate_name"]
                )
            )
            row["fixed_large_budget_results"] = benchmark_budget_results(
                next(
                    report
                    for report in fixed_large_report["temperature_reports"][0][
                        "seed_reports"
                    ][0]["candidate_reports"]
                    if report.get("candidate") == candidate["report_candidate_name"]
                )
            )
            row["heldout_summary"] = heldout_summary(
                heldout_reports, candidate["report_candidate_name"]
            )
            row["large_suite_aggregate"] = {
                budget_pair: {
                    **aggregate_budget_summary(
                        suite_rows,
                        str(candidate["name"]),
                        budget_pair,
                        "balanced_current_ref",
                    ),
                    "mean_delta_vs_current_ref": aggregate_budget_summary(
                        suite_rows,
                        str(candidate["name"]),
                        budget_pair,
                        "balanced_current_ref",
                    )["mean_delta_vs_promoted_current_ref"],
                }
                for budget_pair in budget_pairs
            }
            row["bootstrap_cis"] = {}
            for budget_pair in (
                PRIMARY_BUDGET,
                EQ_768_BUDGET,
                EQ_1200_BUDGET,
                ASYM_1200_256_BUDGET,
            ):
                diffs = pooled_per_opening_differences(
                    suite_rows=suite_rows,
                    candidate_a=str(candidate["name"]),
                    candidate_b="balanced_current_ref",
                    budget_pair=budget_pair,
                    metric_key="ds",
                )
                row["bootstrap_cis"][
                    f"{candidate['name']}_minus_balanced_current_ref_{budget_pair.replace(':', '_')}"
                ] = bootstrap_ci(
                    diffs, seed=args.seed, samples=DEFAULT_BOOTSTRAP_SAMPLES
                )
            gate_report = gate_reports.get(str(candidate["name"]))
            row["gate_classification"] = (
                gate_report.get("classification")
                if gate_report is not None
                else "not_run"
            )
            if gate_report is not None:
                row["gate_budget_results"] = gate_budget_results(gate_report)
        summary_candidates.append(row)

    classification = classify_run(
        current_probe=current_probe,
        candidate_rows=[
            row
            for row in summary_candidates
            if str(row["candidate"]).startswith("trust_region_")
        ],
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "classification": classification,
        "inputs": input_summary,
        "dataset": {
            "validated_disagreement_rows": len(validated_replay_rows),
            "behavior_anchor_rows": len(behavior_anchor_replay_rows),
            "probe_composition": {
                "disagreement": TARGET_DISAGREEMENT_ROWS,
                "stability": len(stability_selected),
                "broad_anchor": len(probe_broad_rows),
            },
            "effective_supervised_replay_sampling_fractions": effective_sampling_fractions(
                {
                    "generic_bootstrap": input_summary["generic_bootstrap"]["rows"],
                    "random_teacher": input_summary["random_teacher"]["rows"],
                    "validated_puct_disagreement": len(validated_replay_rows),
                },
                {
                    "generic_bootstrap": 4,
                    "random_teacher": 1,
                    "validated_puct_disagreement": 8,
                },
            ),
        },
        "outputs": {
            "workdir": str(workdir),
            "validated_replay": str(validated_replay_path),
            "behavior_anchor_replay": str(behavior_anchor_path),
            "probe_state_set": str(probe_path),
            "report": str(REPORT_PATH),
        },
        "candidates": summary_candidates,
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    write_report(summary=summary, report_path=REPORT_PATH)


if __name__ == "__main__":
    main()
