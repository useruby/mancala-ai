#!/usr/bin/env python3
"""Run balanced opening PUCT replay with equal-budget stability anchors."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    find_candidate_report,
    gate_budget_results,
    pooled_per_opening_differences,
)
from ml.alphazero_lite.run_pr123_weighted_candidate_preflight import (  # noqa: E402
    SOURCE_DOC,
    checkpoint_hints,
    candidate_dir_hints,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    EXPECTED_INIT_CHECKPOINT_SHA256,
    EXPECTED_PROMOTED_WEIGHTS_SHA256,
    TARGET_POLICY_MODE,
    TARGET_VALUE_MODE,
    artifact_forward_details,
    build_input_summary,
    canonical_state_hash,
    partition_batches,
    policy_entropy,
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
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_policy_target_from_distribution,
    build_search_options,
    derive_self_play_value_target,
    encode_state,
)

REPORT_PATH = REPO_ROOT / "docs/alphazero-lite-balanced-opening-puct-replay-results.md"
SUMMARY_SCHEMA = "azlite_balanced_opening_puct_replay_v1"
PRIMARY_BUDGET = "384:256"
EQ_768_BUDGET = "768:768"
EQ_1200_BUDGET = "1200:1200"
ASYM_1200_256_BUDGET = "1200:256"
TARGET_STABILITY_ROWS = 2000
STABILITY_MIN_ROWS = 1000
STABILITY_PREFIX_CAP = 12
STABILITY_TOP_MOVE_CAP = 420
STABILITY_PHASE_CAPS = {"opening": 1200, "mid": 600, "late": 200}
PHASE_CHOICES = (
    "stability",
    "train",
    "eval-fixed",
    "eval-heldout",
    "gate",
    "report",
    "all",
)
STABILITY_TRACE_FILENAME = "stability_candidates.jsonl"
STABILITY_SELECTED_FILENAME = "equal_budget_stability_selected.jsonl"
STABILITY_REPLAY_FILENAME = "equal_budget_stability_replay.jsonl"
TRAINING_MANIFEST_FILENAME = "training_manifest.json"
CANDIDATE_MANIFEST_FILENAME = "candidate_manifest.json"
SUMMARY_FILENAME = "summary_metrics.json"
PR123_SUMMARY_PATH = Path(
    "/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/summary_metrics.json"
)
PR123_WORKDIR = Path(
    "/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation"
)

PR123_REFERENCES = {
    "pr123_w8_e1_ref": {
        "source_name": "disagreement_w8_policy_head_e1",
        "expected_checkpoint_sha256": "cd9ef83902516283d680fec0d3986cea832bf467042aaabd814ecd5026ec1e0e",
        "expected_artifact_weights_sha256": "2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad",
    },
    "pr123_w4_e2_ref": {
        "source_name": "disagreement_w4_policy_head_e2",
        "expected_checkpoint_sha256": "04ec8f9420459359b21b7f2d3b5ab5c6f42bc054711c3e13d9538dc8456ab616",
        "expected_artifact_weights_sha256": "65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177",
    },
}


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


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_weight_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        disagreement, stability = item.split(":", 1)
        pairs.append((int(disagreement), int(stability)))
    if not pairs:
        raise ValueError("at least one disagreement:stability pair is required")
    return pairs


def parse_budget_pairs(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_candidate_filter(text: str | None) -> list[str] | None:
    if text is None:
        return None
    names = [item.strip() for item in text.split(",") if item.strip()]
    return names or None


def parse_equal_budget_pairs(include_384: bool) -> list[tuple[str, int]]:
    pairs = [(EQ_768_BUDGET, 768), (EQ_1200_BUDGET, 1200)]
    if include_384:
        pairs.insert(0, ("384:384", 384))
    return pairs


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def artifact_paths(workdir: Path) -> dict[str, Path]:
    return {
        "stability_trace": workdir / STABILITY_TRACE_FILENAME,
        "stability_selected": workdir / STABILITY_SELECTED_FILENAME,
        "stability_replay": workdir / STABILITY_REPLAY_FILENAME,
        "training_manifest": workdir / TRAINING_MANIFEST_FILENAME,
        "candidate_manifest": workdir / CANDIDATE_MANIFEST_FILENAME,
        "reports_dir": workdir / "reports",
        "gates_dir": workdir / "gates",
        "summary": workdir / SUMMARY_FILENAME,
    }


def candidate_manifest_lookup(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {str(candidate["name"]): candidate for candidate in candidates}


def report_candidate_lookup(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(candidate["report_candidate_name"]): candidate for candidate in candidates
    }


def phase_requested(target_phase: str, current_phase: str) -> bool:
    return target_phase == "all" or target_phase == current_phase


def should_reuse_existing(path: Path, *, resume: bool, skip_existing: bool) -> bool:
    return path.exists() and (resume or skip_existing)


def default_candidate_filter_names(candidates: list[dict[str, Any]]) -> list[str]:
    return [str(candidate["name"]) for candidate in candidates]


def filter_candidates(
    candidates: list[dict[str, Any]], candidate_filter: list[str] | None
) -> list[dict[str, Any]]:
    if candidate_filter is None:
        return list(candidates)
    wanted = set(candidate_filter)
    filtered = [
        candidate for candidate in candidates if str(candidate["name"]) in wanted
    ]
    missing = sorted(wanted - {str(candidate["name"]) for candidate in filtered})
    if missing:
        raise RuntimeError(f"unknown candidate-filter entries: {', '.join(missing)}")
    return filtered


def suite_report_path(workdir: Path, suite_name: str) -> Path:
    return workdir / "reports" / f"{suite_name}.json"


def gate_report_path(workdir: Path, candidate_name: str) -> Path:
    return workdir / "gates" / f"{candidate_name}.json"


def flatten_candidate_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            rows.extend(seed_report.get("candidate_reports", []))
    return rows


def merge_suite_reports(
    *,
    base_report: dict[str, Any] | None,
    new_report: dict[str, Any],
    candidate_order: list[str],
) -> dict[str, Any]:
    if base_report is None:
        merged = json.loads(json.dumps(new_report))
    else:
        merged = json.loads(json.dumps(base_report))
        candidate_map = {
            str(candidate_report.get("candidate")): candidate_report
            for candidate_report in flatten_candidate_reports(merged)
        }
        for candidate_report in flatten_candidate_reports(new_report):
            candidate_map[str(candidate_report.get("candidate"))] = candidate_report
        merged["suite_path"] = new_report.get("suite_path")
        merged["suite_sha256"] = new_report.get("suite_sha256")
        merged["suite_size"] = new_report.get("suite_size")
        merged["current_sha256"] = new_report.get("current_sha256")
        merged["root_policy_mode"] = new_report.get("root_policy_mode")
        ordered_reports: list[dict[str, Any]] = []
        remaining = dict(candidate_map)
        for candidate_name in candidate_order:
            if candidate_name in remaining:
                ordered_reports.append(remaining.pop(candidate_name))
        ordered_reports.extend(remaining[key] for key in sorted(remaining))
        merged["temperature_reports"][0]["seed_reports"][0]["candidate_reports"] = (
            ordered_reports
        )
    return merged


def report_has_candidate(report: dict[str, Any], report_candidate_name: str) -> bool:
    return find_candidate_report(report, report_candidate_name) is not None


def report_has_all_candidates(
    report: dict[str, Any], candidates: list[dict[str, Any]]
) -> bool:
    return all(
        report_has_candidate(report, str(candidate["report_candidate_name"]))
        for candidate in candidates
    )


def source_summary_candidates(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(candidate.get("candidate")): candidate
        for candidate in summary.get("candidates", [])
        if isinstance(candidate, dict)
    }


def discover_reference_candidate(
    *,
    name: str,
    source_name: str,
    workdir: Path,
    summary: dict[str, Any],
    expected_checkpoint_sha256: str,
    expected_artifact_weights_sha256: str,
) -> dict[str, Any]:
    summary_entry = source_summary_candidates(summary).get(source_name, {})

    artifact_candidates: list[Path] = []
    summary_artifact = summary_entry.get("artifact_dir")
    if isinstance(summary_artifact, str) and summary_artifact:
        artifact_candidates.append(Path(summary_artifact))
    artifact_candidates.extend(candidate_dir_hints(workdir, source_name))

    artifact_dir: Path | None = None
    for candidate_dir in artifact_candidates:
        weights_path = candidate_dir / "weights.json"
        if not weights_path.is_file():
            continue
        if sha256_file(weights_path) == expected_artifact_weights_sha256:
            artifact_dir = candidate_dir
            break
    if artifact_dir is None:
        raise FileNotFoundError(f"missing artifact for {name} ({source_name})")

    checkpoint_candidates: list[Path] = []
    summary_checkpoint = summary_entry.get("checkpoint_path")
    if isinstance(summary_checkpoint, str) and summary_checkpoint:
        checkpoint_candidates.append(Path(summary_checkpoint))
    checkpoint_candidates.extend(checkpoint_hints(workdir, source_name))

    checkpoint_path: Path | None = None
    for candidate_checkpoint in checkpoint_candidates:
        if not candidate_checkpoint.is_file():
            continue
        if sha256_file(candidate_checkpoint) == expected_checkpoint_sha256:
            checkpoint_path = candidate_checkpoint
            break
    if checkpoint_path is None:
        raise FileNotFoundError(f"missing checkpoint for {name} ({source_name})")

    verify_expected_hash(
        artifact_dir / "weights.json",
        expected_artifact_weights_sha256,
        f"{name} artifact weights",
    )
    verify_expected_hash(
        checkpoint_path, expected_checkpoint_sha256, f"{name} checkpoint"
    )
    return {
        "name": name,
        "report_candidate_name": artifact_dir.name,
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": expected_artifact_weights_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "source": SOURCE_DOC,
    }


def run_train_save_both_epochs(
    *,
    data_files: str,
    replay_weights: str,
    init_checkpoint: str,
    out: str,
    top_k_dir: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "ml/alphazero_lite/train.py"),
        "--data-files",
        data_files,
        "--replay-weights",
        replay_weights,
        "--init-checkpoint",
        init_checkpoint,
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
        "--top-k-dir",
        top_k_dir,
        "--out",
        out,
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
        "1,2",
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)
    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed: {result.stderr[-2000:]}")
    metrics: dict[str, Any] = {
        "training_elapsed_s": elapsed,
        "trainable_scope": "policy_head",
    }
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("policy_loss="):
            metrics["policy_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("value_loss="):
            metrics["value_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("best_val_loss="):
            metrics["best_val_loss"] = float(line.split("=", 1)[1])
        elif line.startswith("saved_epoch_checkpoint_epoch="):
            parts = line.split()
            epoch = parts[0].split("=", 1)[1]
            metrics[f"epoch_{epoch}_path"] = parts[1].split("=", 1)[1]
    for line in result.stderr.splitlines():
        for key in ("trainable_params", "frozen_params", "total_params"):
            token = f"{key}="
            if token in line:
                for part in line.split():
                    if part.startswith(token):
                        metrics[key] = int(part.split("=", 1)[1])
    return metrics


def collect_equal_budget_trace(
    *,
    suite_name: str,
    budget_label: str,
    opening_entry: dict[str, Any],
    challenger_player: int,
    evaluator: ArtifactEvaluator,
    simulations: int,
    c_puct: float,
    seed: int,
    game_index: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(opening_entry["state"])
    total_ply = int(
        opening_entry.get("ply", len(opening_entry.get("prefix_moves", [])))
    )
    trajectory: list[int] = []
    encountered_states: list[dict[str, Any]] = []
    reusable_roots: dict[int, Any] = {0: None, 1: None}
    eval_search_options = build_eval_search_options(root_policy_mode="deterministic")
    rng = np.random.default_rng(seed + game_index)

    while not game.over():
        legal_moves = game.possible_moves()
        if len(legal_moves) <= 1:
            break
        state = game.to_state()
        logits, raw_policy, _raw_value = artifact_forward_details(evaluator, game)
        raw_top1 = top_policy_move(raw_policy, legal_moves)
        search = PUCT(
            evaluator=evaluator,
            simulations=int(simulations),
            c_puct=float(c_puct),
            rng=random.Random(int(rng.integers(0, 2**31 - 1))),
            root=reusable_roots[game.current_player],
            fpu_mode=str(eval_search_options["fpu_mode"]),
            reuse_subtree=bool(eval_search_options["reuse_subtree"]),
            normalize_values=bool(eval_search_options["normalize_values"]),
            root_policy_mode=str(eval_search_options["root_policy_mode"]),
            tactical_root_bias=float(eval_search_options["tactical_root_bias"]),
            root_temperature=float(eval_search_options["root_temperature"]),
        )
        visits, root = search.run(game)
        search_policy = np.zeros(6, dtype=np.float32)
        total_visits = float(sum(float(visits[move]) for move in legal_moves))
        if total_visits <= 0.0:
            for move in legal_moves:
                search_policy[move] = 1.0 / len(legal_moves)
        else:
            for move in legal_moves:
                search_policy[move] = float(visits[move] / total_visits)
        search_top1 = top_policy_move(search_policy.tolist(), legal_moves)
        search_top1_visit_share = (
            float(search_policy[search_top1]) if search_top1 is not None else 0.0
        )
        encountered_states.append(
            {
                "suite_name": suite_name,
                "opening_prefix": [
                    int(move) for move in opening_entry.get("prefix_moves", [])
                ],
                "opening_prefix_text": ",".join(
                    str(move) for move in opening_entry.get("prefix_moves", [])
                ),
                "initial_suite_ply": int(opening_entry.get("ply", 0)),
                "move_index": int(total_ply),
                "phase": "opening"
                if total_ply <= 8
                else ("late" if sum(int(v) for v in game.pits) <= 16 else "mid"),
                "side_to_move": int(game.current_player),
                "challenger_player": int(challenger_player),
                "state": state,
                "state_hash": canonical_state_hash(state),
                "legal_moves": [int(move) for move in legal_moves],
                "raw_logits": [float(value) for value in logits],
                "raw_policy": [float(value) for value in raw_policy],
                "raw_top1": raw_top1,
                "raw_margin": float(raw_margin(raw_policy, legal_moves)),
                "search_policy": [float(value) for value in search_policy.tolist()],
                "search_value": float(root.q_value if root is not None else 0.0),
                "search_top1": search_top1,
                "search_top1_visit_share": float(search_top1_visit_share),
                "search_entropy": policy_entropy(search_policy.tolist(), legal_moves),
                "budget_label": budget_label,
                "simulations": int(simulations),
                "top1_changed": bool(search_top1 != raw_top1),
            }
        )
        move = search.select_root_move(root, legal_moves)
        trajectory.append(int(game.pit_index(move)))
        if not game.move(game.pit_index(move)):
            break
        if game.current_player == challenger_player and root is not None:
            reusable_roots[challenger_player] = root.child_for_action(move)
        else:
            reusable_roots[challenger_player] = None
        if game.current_player != challenger_player:
            reusable_roots[1 - challenger_player] = None
        total_ply += 1
    return {
        "trajectory": ",".join(str(move) for move in trajectory),
        "states": encountered_states,
    }


def collect_stability_candidates(
    *,
    suite_specs: list[tuple[str, Path]],
    artifact_path: Path,
    equal_budgets: list[tuple[str, int]],
    c_puct: float,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analyzed_rows: list[dict[str, Any]] = []
    trajectory_counter: Counter[str] = Counter()
    budget_state_counts: Counter[str] = Counter()
    tasks = [
        {
            "suite_name": suite_name,
            "suite_path": str(suite_path),
            "budget_label": budget_label,
            "simulations": simulations,
            "artifact_path": str(artifact_path),
            "c_puct": float(c_puct),
            "seed": int(seed + (task_index * 10_000)),
        }
        for task_index, (budget_label, simulations) in enumerate(equal_budgets)
        for suite_name, suite_path in suite_specs
    ]
    total_game_count = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(tasks)))
    ) as pool:
        futures = [pool.submit(_collect_stability_task, task=task) for task in tasks]
        for future in futures:
            result = future.result()
            analyzed_rows.extend(result["rows"])
            trajectory_counter.update(result["trajectory_counter"])
            budget_state_counts.update(result["budget_state_counts"])
            total_game_count += int(result["game_count"])
    duplicate_trajectory_count = sum(
        count for count in trajectory_counter.values() if count > 1
    )
    return analyzed_rows, {
        "equal_budget_labels": [label for label, _sims in equal_budgets],
        "analyzed_row_count": len(analyzed_rows),
        "duplicate_trajectory_count": duplicate_trajectory_count,
        "duplicate_trajectory_rate": duplicate_trajectory_count
        / max(total_game_count, 1),
        "budget_state_counts": dict(budget_state_counts),
        "games": total_game_count,
    }


def _collect_stability_task(*, task: dict[str, Any]) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(Path(str(task["artifact_path"])))
    rows: list[dict[str, Any]] = []
    trajectory_counter: Counter[str] = Counter()
    budget_state_counts: Counter[str] = Counter()
    openings = read_jsonl(Path(str(task["suite_path"])))
    game_count = 0
    for opening_index, opening_entry in enumerate(openings):
        for challenger_player in (0, 1):
            trace = collect_equal_budget_trace(
                suite_name=str(task["suite_name"]),
                budget_label=str(task["budget_label"]),
                opening_entry=opening_entry,
                challenger_player=challenger_player,
                evaluator=evaluator,
                simulations=int(task["simulations"]),
                c_puct=float(task["c_puct"]),
                seed=int(task["seed"]),
                game_index=(opening_index * 2) + challenger_player,
            )
            rows.extend(trace["states"])
            trajectory_counter[trace["trajectory"]] += 1
            budget_state_counts[str(task["budget_label"])] += len(trace["states"])
            game_count += 1
    return {
        "rows": rows,
        "trajectory_counter": dict(trajectory_counter),
        "budget_state_counts": dict(budget_state_counts),
        "game_count": game_count,
    }


def stability_selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    tier_order = {"preferred": 0, "fallback": 1, "confidence": 2}
    return (
        tier_order[str(row["selection_tier"])],
        -float(row["search_top1_visit_share"]),
        -int(row["simulations"]),
        -float(row["raw_margin"]),
        int(row["move_index"]),
        str(row["state_hash"]),
    )


def select_stability_rows(
    analyzed_rows: list[dict[str, Any]],
    disagreement_hashes: set[str],
    *,
    target_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped_best_by_hash: dict[str, dict[str, Any]] = {}
    raw_overlap = 0
    filtered_late_forced = 0
    for row in analyzed_rows:
        state_hash = str(row["state_hash"])
        if state_hash in disagreement_hashes:
            raw_overlap += 1
            continue
        if len(row["legal_moves"]) <= 1:
            filtered_late_forced += 1
            continue
        if row["search_top1"] != row["raw_top1"]:
            continue
        if (
            float(row["search_top1_visit_share"]) >= 0.55
            and float(row["raw_margin"]) >= 0.05
        ):
            row["selection_tier"] = "preferred"
        elif float(row["search_top1_visit_share"]) >= 0.45:
            row["selection_tier"] = "fallback"
        else:
            row["selection_tier"] = "confidence"
        existing = deduped_best_by_hash.get(state_hash)
        if existing is None or stability_selection_key(row) < stability_selection_key(
            existing
        ):
            deduped_best_by_hash[state_hash] = row

    candidates = sorted(deduped_best_by_hash.values(), key=stability_selection_key)
    prefix_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    top_move_counts: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    skipped_for_caps: list[dict[str, Any]] = []
    for row in candidates:
        prefix_key = str(row.get("opening_prefix_text") or "")
        phase_key = str(row["phase"])
        top_move = int(row["search_top1"]) if row["search_top1"] is not None else -1
        if prefix_counts[prefix_key] >= STABILITY_PREFIX_CAP:
            skipped_for_caps.append(row)
            continue
        if phase_counts[phase_key] >= STABILITY_PHASE_CAPS.get(phase_key, target_rows):
            skipped_for_caps.append(row)
            continue
        if top_move >= 0 and top_move_counts[top_move] >= STABILITY_TOP_MOVE_CAP:
            skipped_for_caps.append(row)
            continue
        selected.append(row)
        prefix_counts[prefix_key] += 1
        phase_counts[phase_key] += 1
        if top_move >= 0:
            top_move_counts[top_move] += 1
        if len(selected) >= target_rows:
            break
    if len(selected) < target_rows:
        for row in skipped_for_caps:
            state_hash = str(row["state_hash"])
            if any(str(existing["state_hash"]) == state_hash for existing in selected):
                continue
            selected.append(row)
            if len(selected) >= target_rows:
                break

    entropy_values = [float(row["search_entropy"]) for row in selected]
    return selected, {
        "target_rows": target_rows,
        "selected_rows": len(selected),
        "unique_rows": len({str(row["state_hash"]) for row in selected}),
        "preferred_rows": sum(
            1 for row in selected if row["selection_tier"] == "preferred"
        ),
        "fallback_rows": sum(
            1 for row in selected if row["selection_tier"] == "fallback"
        ),
        "confidence_rows": sum(
            1 for row in selected if row["selection_tier"] == "confidence"
        ),
        "raw_overlap_with_mined_disagreement_rows": raw_overlap,
        "selected_overlap_with_mined_disagreement_rows": 0,
        "filtered_late_forced_rows": filtered_late_forced,
        "phase_counts": dict(phase_counts),
        "top_move_counts": {
            str(key): value for key, value in sorted(top_move_counts.items())
        },
        "prefix_cap": STABILITY_PREFIX_CAP,
        "top_move_cap": STABILITY_TOP_MOVE_CAP,
        "phase_caps": STABILITY_PHASE_CAPS,
        "target_entropy": {
            "mean": statistics.fmean(entropy_values) if entropy_values else 0.0,
            "p10": float(np.percentile(np.asarray(entropy_values), 10))
            if entropy_values
            else 0.0,
            "p50": float(np.percentile(np.asarray(entropy_values), 50))
            if entropy_values
            else 0.0,
            "p90": float(np.percentile(np.asarray(entropy_values), 90))
            if entropy_values
            else 0.0,
        },
    }


def build_stability_replay_row(
    row: dict[str, Any], evaluator: ArtifactEvaluator
) -> dict[str, Any]:
    game = KalahGame.from_state(row["state"])
    root_player = int(game.current_player)
    policy_target = build_policy_target_from_distribution(
        row["search_policy"], mode=TARGET_POLICY_MODE
    )
    search_options = build_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    rng = np.random.default_rng(int(row["continuation_seed"]))
    while not game.over():
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        search = PUCT(
            evaluator=evaluator,
            simulations=int(row["simulations"]),
            c_puct=float(row["c_puct"]),
            rng=random.Random(int(rng.integers(0, 2**31 - 1))),
            fpu_mode=str(search_options["fpu_mode"]),
            reuse_subtree=bool(search_options["reuse_subtree"]),
            normalize_values=bool(search_options["normalize_values"]),
            root_policy_mode=str(search_options["root_policy_mode"]),
            tactical_root_bias=float(search_options["tactical_root_bias"]),
            root_temperature=float(search_options["root_temperature"]),
        )
        _visits, root = search.run(game)
        move = search.select_root_move(root, legal_moves)
        if not game.move(game.pit_index(move)):
            break
    if game.winner is None:
        outcome = 0.0
    elif int(game.winner) == root_player:
        outcome = 1.0
    else:
        outcome = -1.0
    return {
        "state_hash": row["state_hash"],
        "state": encode_state(row["state"], input_encoding="kalah_v3"),
        "policy": [float(value) for value in policy_target],
        "value": derive_self_play_value_target(
            outcome_value=outcome,
            search_value=float(row["search_value"]),
            move_index=int(row["move_index"]),
            mode=TARGET_VALUE_MODE,
        ),
        "player": root_player,
        "move_index": int(row["move_index"]),
        "winner": game.winner,
        "legal_moves": [int(move) for move in row["legal_moves"]],
        "top_target_move": row["search_top1"],
        "policy_target_mode": TARGET_POLICY_MODE,
        "policy_target_actual_mode": TARGET_POLICY_MODE,
        "value_target_mode": TARGET_VALUE_MODE,
        "teacher_source": "equal_budget_stability_anchor",
        "bucket": "equal_budget_stability_anchor",
        "suite_name": row["suite_name"],
        "suite_names": [row["suite_name"]],
        "opening_prefix": list(row.get("opening_prefix", [])),
        "phase": row["phase"],
        "selection_tier": row["selection_tier"],
        "search_top1": row["search_top1"],
        "raw_top1": row["raw_top1"],
        "search_top1_visit_share": row["search_top1_visit_share"],
        "raw_margin": row["raw_margin"],
        "search_entropy": row["search_entropy"],
        "budget_label": row["budget_label"],
        "simulations": row["simulations"],
        "continuation_outcome": outcome,
        "continuation_winner": game.winner,
    }


def build_stability_replay(
    *,
    selected_rows: list[dict[str, Any]],
    artifact_path: Path,
    c_puct: float,
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        jobs.append(
            {
                **row,
                "artifact_path": str(artifact_path),
                "c_puct": float(c_puct),
                "continuation_seed": int(seed + 500_000 + index),
            }
        )
    batches = partition_batches(jobs, workers)
    replay_rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(batches)))
    ) as pool:
        futures = [
            pool.submit(_build_stability_replay_batch, batch=batch) for batch in batches
        ]
        for future in futures:
            replay_rows.extend(future.result())
    replay_rows.sort(key=lambda row: str(row["state_hash"]))
    return replay_rows


def _build_stability_replay_batch(
    *, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not batch:
        return []
    evaluator = ArtifactEvaluator(Path(str(batch[0]["artifact_path"])))
    return [build_stability_replay_row(row, evaluator) for row in batch]


def effective_sampling_fractions(
    row_counts: dict[str, int], weights: dict[str, int]
) -> dict[str, float]:
    total = sum(int(weights[key]) * int(row_counts[key]) for key in weights)
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: (int(weights[key]) * int(row_counts[key])) / total for key in weights}


def load_candidate_policy_shift_rows(
    replay_rows: list[dict[str, Any]],
    state_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replay_row in replay_rows:
        state_hash = str(replay_row["state_hash"])
        state_row = state_lookup[state_hash]
        rows.append(
            {
                "state_hash": state_hash,
                "state": state_row["state"],
                "legal_moves": [int(move) for move in state_row["legal_moves"]],
                "phase": str(state_row["phase"]),
                "raw_margin": float(state_row["raw_margin"]),
                "raw_top1": state_row["raw_top1"],
                "search_top1": state_row["search_top1"],
                "search_policy": state_row["search_policy"],
            }
        )
    return rows


def anchored_top1_preserved_rate(
    candidate_artifact: Path, stability_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(candidate_artifact)
    preserved = 0
    by_phase: dict[str, dict[str, int]] = defaultdict(
        lambda: {"preserved": 0, "total": 0}
    )
    for row in stability_rows:
        game = KalahGame.from_state(row["state"])
        _logits, raw_policy, _value = artifact_forward_details(evaluator, game)
        top1 = top_policy_move(raw_policy, row["legal_moves"])
        preserved_now = top1 == row["search_top1"]
        phase = str(row["phase"])
        by_phase[phase]["total"] += 1
        if preserved_now:
            preserved += 1
            by_phase[phase]["preserved"] += 1
    total = len(stability_rows)
    return {
        "states": total,
        "top1_preserved_rate": preserved / total if total else 0.0,
        "by_phase": {
            phase: {
                "states": counts["total"],
                "top1_preserved_rate": counts["preserved"] / counts["total"]
                if counts["total"]
                else 0.0,
            }
            for phase, counts in sorted(by_phase.items())
        },
    }


def large_suite_rows(
    *, reports: dict[str, dict[str, Any]], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    suite_rows: dict[str, Any] = {}
    for suite_name, report in reports.items():
        suite_rows[suite_name] = {
            "suite_name": suite_name,
            "suite_path": report.get("suite_path"),
            "suite_sha256": report.get("suite_sha256"),
            "suite_size": report.get("suite_size"),
            "candidates": {},
        }
        for candidate in candidates:
            candidate_report = find_candidate_report(
                report, str(candidate["report_candidate_name"])
            )
            if candidate_report is None:
                raise RuntimeError(
                    f"missing candidate report for {candidate['name']} in {suite_name}"
                )
            suite_rows[suite_name]["candidates"][candidate["name"]] = {
                "candidate_path": candidate["artifact_dir"],
                "candidate_sha256": candidate_report.get("candidate_sha256"),
                "budget_results": benchmark_budget_results(candidate_report),
            }
    return suite_rows


def aggregate_budget_summary(
    suite_rows: dict[str, Any],
    candidate_name: str,
    budget_pair: str,
    reference_name: str,
) -> dict[str, Any]:
    ds_values: list[float] = []
    delta_values: list[float] = []
    p0_values: list[float] = []
    p1_values: list[float] = []
    duplicate_values: list[float] = []
    worst_suite: str | None = None
    worst_ds = float("inf")
    for suite_name, suite_row in suite_rows.items():
        candidate_budget = suite_row["candidates"][candidate_name]["budget_results"][
            budget_pair
        ]
        reference_budget = suite_row["candidates"][reference_name]["budget_results"][
            budget_pair
        ]
        ds = float(candidate_budget["ds"])
        delta = ds - float(reference_budget["ds"])
        ds_values.append(ds)
        delta_values.append(delta)
        p0_values.append(float(candidate_budget["p0_score"]))
        p1_values.append(float(candidate_budget["p1_score"]))
        duplicate_values.append(
            float(candidate_budget.get("duplicate_trajectory_count", 0))
        )
        if ds < worst_ds:
            worst_ds = ds
            worst_suite = suite_name
    return {
        "mean_ds": statistics.fmean(ds_values) if ds_values else 0.0,
        "mean_delta_vs_promoted_current_ref": statistics.fmean(delta_values)
        if delta_values
        else 0.0,
        "worst_suite_ds": worst_ds if ds_values else 0.0,
        "worst_suite_name": worst_suite,
        "mean_p0_score": statistics.fmean(p0_values) if p0_values else 0.0,
        "mean_p1_score": statistics.fmean(p1_values) if p1_values else 0.0,
        "mean_duplicate_trajectory_count": statistics.fmean(duplicate_values)
        if duplicate_values
        else 0.0,
    }


def classify_balanced_run(
    *,
    candidate_rows: list[dict[str, Any]],
    current_name: str,
    pr123_w8_name: str,
) -> str:
    balanced_rows = [
        row for row in candidate_rows if str(row["candidate"]).startswith("balanced_")
    ]
    if not balanced_rows:
        return "incomplete"
    pr123_w8_row = next(
        row for row in candidate_rows if row["candidate"] == pr123_w8_name
    )
    pr123_w8_delta_768 = float(
        pr123_w8_row["large_suite_aggregate"][EQ_768_BUDGET][
            "mean_delta_vs_promoted_current_ref"
        ]
    )
    for row in balanced_rows:
        agg = row["large_suite_aggregate"]
        delta_384 = float(agg[PRIMARY_BUDGET]["mean_delta_vs_promoted_current_ref"])
        delta_768 = float(agg[EQ_768_BUDGET]["mean_delta_vs_promoted_current_ref"])
        delta_1200 = float(agg[EQ_1200_BUDGET]["mean_delta_vs_promoted_current_ref"])
        delta_1200_256 = float(
            agg[ASYM_1200_256_BUDGET]["mean_delta_vs_promoted_current_ref"]
        )
        ci_384 = row["bootstrap_cis"][
            f"{row['candidate']}_minus_{current_name}_384_256"
        ]
        ci_768 = row["bootstrap_cis"][
            f"{row['candidate']}_minus_{current_name}_768_768"
        ]
        gate = row.get("default_gate")
        gate_ok = not gate or gate.get("classification") == "high_search_breakthrough"
        if (
            delta_384 >= 0.15
            and float(ci_384["lower"]) > 0.08
            and delta_768 >= -0.10
            and float(ci_768["lower"]) >= -0.15
            and delta_1200 >= -0.03
            and delta_1200_256 >= -0.03
            and gate_ok
        ):
            return "balanced_replay_success"
    for row in balanced_rows:
        agg = row["large_suite_aggregate"]
        delta_384 = float(agg[PRIMARY_BUDGET]["mean_delta_vs_promoted_current_ref"])
        delta_768 = float(agg[EQ_768_BUDGET]["mean_delta_vs_promoted_current_ref"])
        if delta_384 >= 0.12 and abs(delta_768) <= (abs(pr123_w8_delta_768) / 2.0):
            return "partial_balance_tradeoff"
    if any(
        float(
            row["large_suite_aggregate"][PRIMARY_BUDGET][
                "mean_delta_vs_promoted_current_ref"
            ]
        )
        < 0.08
        and float(
            row["large_suite_aggregate"][EQ_768_BUDGET][
                "mean_delta_vs_promoted_current_ref"
            ]
        )
        >= -0.10
        for row in balanced_rows
    ):
        return "stability_anchor_dominates"
    return "disagreement_overfit_confirmed"


def load_candidate_manifest_or_fail(workdir: Path) -> dict[str, Any]:
    path = artifact_paths(workdir)["candidate_manifest"]
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate manifest: {path}")
    return load_json(path)


def load_training_manifest_or_fail(workdir: Path) -> dict[str, Any]:
    path = artifact_paths(workdir)["training_manifest"]
    if not path.is_file():
        raise FileNotFoundError(f"missing training manifest: {path}")
    return load_json(path)


def load_stability_artifacts_or_fail(workdir: Path) -> dict[str, Any]:
    paths = artifact_paths(workdir)
    trace_path = paths["stability_trace"]
    selected_path = paths["stability_selected"]
    replay_path = paths["stability_replay"]
    for path, label in (
        (trace_path, "stability candidate trace"),
        (selected_path, "selected stability rows"),
        (replay_path, "stability replay"),
    ):
        require_existing_file(path, label)
    return {
        "trace_rows": read_jsonl(trace_path),
        "selected_rows": read_jsonl(selected_path),
        "replay_rows": read_jsonl(replay_path),
    }


def build_candidate_specs(
    *,
    workdir: Path,
    current_artifact: Path,
    init_checkpoint: Path,
    current_weights_sha256: str,
    pr123_summary: dict[str, Any],
    row_counts: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [
        {
            "name": "promoted_current_ref",
            "report_candidate_name": "current",
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": current_weights_sha256,
            "checkpoint_path": str(init_checkpoint),
            "checkpoint_sha256": sha256_file(init_checkpoint),
            "epochs": 0,
            "source": "promoted_current_ref",
        }
    ]
    for name, reference_spec in PR123_REFERENCES.items():
        candidates.append(
            discover_reference_candidate(
                name=name,
                source_name=str(reference_spec["source_name"]),
                workdir=PR123_WORKDIR,
                summary=pr123_summary,
                expected_checkpoint_sha256=str(
                    reference_spec["expected_checkpoint_sha256"]
                ),
                expected_artifact_weights_sha256=str(
                    reference_spec["expected_artifact_weights_sha256"]
                ),
            )
            | {"epochs": 0}
        )

    training_manifest = load_training_manifest_or_fail(workdir)
    for lane in training_manifest.get("lanes", []):
        train_metrics = lane.get("train_metrics")
        for epoch_key in ("e1", "e2"):
            epoch_info = lane.get(epoch_key)
            if not isinstance(epoch_info, dict):
                continue
            candidates.append(
                {
                    "name": str(epoch_info["name"]),
                    "report_candidate_name": str(epoch_info["report_candidate_name"]),
                    "artifact_dir": str(epoch_info["artifact_dir"]),
                    "checkpoint_path": str(epoch_info["checkpoint_path"]),
                    "checkpoint_sha256": str(epoch_info["checkpoint_sha256"]),
                    "artifact_weights_sha256": str(
                        epoch_info["artifact_weights_sha256"]
                    ),
                    "epochs": int(epoch_info["epochs"]),
                    "source": str(lane["lane_base"]),
                    "replay_weights": dict(lane["replay_weights"]),
                    "effective_replay_sampling_fractions": effective_sampling_fractions(
                        row_counts,
                        {
                            "generic_bootstrap": int(
                                lane["replay_weights"]["generic_bootstrap"]
                            ),
                            "random_teacher": int(
                                lane["replay_weights"]["random_teacher"]
                            ),
                            "mined_disagreement": int(
                                lane["replay_weights"]["mined_disagreement"]
                            ),
                            "stability_anchor": int(
                                lane["replay_weights"]["stability_anchor"]
                            ),
                        },
                    ),
                    "train_metrics": train_metrics,
                }
            )
    return candidates


def build_training_manifest(
    *,
    workdir: Path,
    init_checkpoint: Path,
    generic_bootstrap: Path,
    random_teacher: Path,
    mined_disagreement_replay: Path,
    stability_replay_path: Path,
    weight_pairs: list[tuple[int, int]],
    seed: int,
    candidate_filter: list[str] | None,
    resume: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    manifest = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "workdir": str(workdir),
        "seed": seed,
        "data_files": [
            str(generic_bootstrap),
            str(random_teacher),
            str(mined_disagreement_replay),
            str(stability_replay_path),
        ],
        "lanes": [],
    }
    allowed_names = set(candidate_filter or [])
    for disagreement_weight, stability_weight in weight_pairs:
        lane_base = f"balanced_w{disagreement_weight}s{stability_weight}_policy_head"
        lane_dir = workdir / lane_base
        lane_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_epoch1 = lane_dir / "checkpoint_epoch1.npz"
        checkpoint_epoch2 = lane_dir / "checkpoint_epoch2.npz"
        artifact_epoch1 = lane_dir / f"artifact_{lane_base}_e1"
        artifact_epoch2 = lane_dir / f"artifact_{lane_base}_e2"
        metrics_path = lane_dir / "train_metrics.json"
        replay_weights = {
            "generic_bootstrap": 4,
            "random_teacher": 1,
            "mined_disagreement": disagreement_weight,
            "stability_anchor": stability_weight,
        }
        lane_candidate_names = {f"{lane_base}_e1", f"{lane_base}_e2"}
        lane_selected = not allowed_names or bool(allowed_names & lane_candidate_names)
        outputs_ready = (
            checkpoint_epoch1.is_file()
            and checkpoint_epoch2.is_file()
            and (artifact_epoch1 / "weights.json").is_file()
            and (artifact_epoch2 / "weights.json").is_file()
            and metrics_path.is_file()
        )
        if not lane_selected and not outputs_ready:
            continue
        train_metrics = load_json(metrics_path) if metrics_path.is_file() else None
        if lane_selected and not (outputs_ready and (resume or skip_existing)):
            train_metrics = run_train_save_both_epochs(
                data_files=(
                    f"{generic_bootstrap},{random_teacher},{mined_disagreement_replay},{stability_replay_path}"
                ),
                replay_weights=(f"4,1,{disagreement_weight},{stability_weight}"),
                init_checkpoint=str(init_checkpoint),
                out=str(lane_dir / "checkpoint.npz"),
                top_k_dir=str(lane_dir),
                epochs=2,
                seed=seed,
            )
            export_checkpoint(
                checkpoint_path=str(checkpoint_epoch1),
                out_dir=str(artifact_epoch1),
                version=f"{lane_base}_e1",
                policy_loss=float((train_metrics or {}).get("policy_loss", 0.0)),
                value_loss=float((train_metrics or {}).get("value_loss", 0.0)),
            )
            export_checkpoint(
                checkpoint_path=str(checkpoint_epoch2),
                out_dir=str(artifact_epoch2),
                version=f"{lane_base}_e2",
                policy_loss=float((train_metrics or {}).get("policy_loss", 0.0)),
                value_loss=float((train_metrics or {}).get("value_loss", 0.0)),
            )
            write_json(metrics_path, train_metrics or {})
        for path, label in (
            (checkpoint_epoch1, f"{lane_base} epoch1 checkpoint"),
            (checkpoint_epoch2, f"{lane_base} epoch2 checkpoint"),
            (artifact_epoch1 / "weights.json", f"{lane_base} epoch1 artifact"),
            (artifact_epoch2 / "weights.json", f"{lane_base} epoch2 artifact"),
        ):
            require_existing_file(path, label)
        if train_metrics is None:
            train_metrics = load_json(metrics_path)
        lane_record = {
            "lane_base": lane_base,
            "replay_weights": replay_weights,
            "train_metrics": train_metrics,
            "e1": {
                "name": f"{lane_base}_e1",
                "report_candidate_name": artifact_epoch1.name,
                "artifact_dir": str(artifact_epoch1),
                "artifact_weights_sha256": sha256_file(
                    artifact_epoch1 / "weights.json"
                ),
                "checkpoint_path": str(checkpoint_epoch1),
                "checkpoint_sha256": sha256_file(checkpoint_epoch1),
                "epochs": 1,
            },
            "e2": {
                "name": f"{lane_base}_e2",
                "report_candidate_name": artifact_epoch2.name,
                "artifact_dir": str(artifact_epoch2),
                "artifact_weights_sha256": sha256_file(
                    artifact_epoch2 / "weights.json"
                ),
                "checkpoint_path": str(checkpoint_epoch2),
                "checkpoint_sha256": sha256_file(checkpoint_epoch2),
                "epochs": 2,
            },
        }
        manifest["lanes"].append(lane_record)
        write_json(artifact_paths(workdir)["training_manifest"], manifest)
    return manifest


def evaluate_suite_candidates(
    *,
    workdir: Path,
    suite_name: str,
    suite_path: Path,
    current_artifact: Path,
    candidates: list[dict[str, Any]],
    budget_pairs: str,
    games_per_opening: int,
    seed: int,
    workers: int,
    timeout: int,
    candidate_filter: list[str] | None,
    resume: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    report_path = suite_report_path(workdir, suite_name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing_report = load_json(report_path) if report_path.is_file() else None
    filtered_candidates = filter_candidates(candidates, candidate_filter)
    pending_candidates = filtered_candidates
    if existing_report is not None and (resume or skip_existing):
        pending_candidates = [
            candidate
            for candidate in filtered_candidates
            if not report_has_candidate(
                existing_report, str(candidate["report_candidate_name"])
            )
        ]
    if not pending_candidates and existing_report is not None:
        return existing_report
    temp_workdir = workdir / "eval_cache" / suite_name
    temp_workdir.mkdir(parents=True, exist_ok=True)
    new_report = run_opening_suite_benchmark(
        workdir=str(temp_workdir),
        suite=str(suite_path),
        current=str(current_artifact),
        candidates=",".join(
            str(candidate["artifact_dir"]) for candidate in pending_candidates
        ),
        budget_pairs=budget_pairs,
        games_per_opening=games_per_opening,
        seed=seed,
        workers=workers,
        timeout=timeout,
    )
    merged = merge_suite_reports(
        base_report=existing_report,
        new_report=new_report,
        candidate_order=[
            str(candidate["report_candidate_name"]) for candidate in candidates
        ],
    )
    write_json(report_path, merged)
    return merged


def gate_targets_from_suite_rows(
    suite_rows: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[str]:
    promoted_mean_384 = aggregate_budget_summary(
        suite_rows, "promoted_current_ref", PRIMARY_BUDGET, "promoted_current_ref"
    )["mean_ds"]
    gate_targets = ["promoted_current_ref", "pr123_w8_e1_ref", "pr123_w4_e2_ref"]
    for candidate in candidates:
        if not str(candidate["name"]).startswith("balanced_"):
            continue
        mean_384 = aggregate_budget_summary(
            suite_rows, str(candidate["name"]), PRIMARY_BUDGET, "promoted_current_ref"
        )["mean_ds"]
        if float(mean_384) >= float(promoted_mean_384) + 0.10:
            gate_targets.append(str(candidate["name"]))
    return gate_targets


def classify_from_cached_artifacts(
    *,
    stability_selected_rows: list[dict[str, Any]],
    required_heldout_reports: dict[str, Path],
    summary_candidates: list[dict[str, Any]],
) -> str:
    unique_stability_rows = len(
        {str(row["state_hash"]) for row in stability_selected_rows}
    )
    if unique_stability_rows < STABILITY_MIN_ROWS:
        return "runner_still_blocked"
    if any(not path.is_file() for path in required_heldout_reports.values()):
        return "runner_unblocked_partial_result"
    return classify_balanced_run(
        candidate_rows=summary_candidates,
        current_name="promoted_current_ref",
        pr123_w8_name="pr123_w8_e1_ref",
    )


def render_report(summary: dict[str, Any]) -> str:
    candidate_rows = summary["candidates"]
    large_budgets = summary["budget_pairs"]
    stability = summary["stability_replay"]
    completion = summary.get("experiment_completion", {})
    balanced_candidates = [
        row
        for row in candidate_rows
        if str(row.get("candidate", "")).startswith("balanced_")
    ]
    best_balanced = None
    if balanced_candidates:
        best_balanced = max(
            balanced_candidates,
            key=lambda row: float(
                row.get("large_suite_aggregate", {})
                .get(PRIMARY_BUDGET, {})
                .get("mean_delta_vs_promoted_current_ref", float("-inf"))
            ),
        )
    lines = [
        "# AlphaZero-Lite Balanced Opening PUCT Replay Results",
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['classification']}`",
        "",
        "## Completion",
        "",
        f"- original PR #125 balanced experiment: `{completion.get('status', 'unknown')}`",
        f"- fixed-suite evaluation complete: `{completion.get('fixed_eval_complete', False)}`",
        f"- held-out evaluation complete: `{completion.get('heldout_eval_complete', False)}`",
        f"- gate complete: `{completion.get('gate_complete', False)}`",
        "",
        "## Decision",
        "",
    ]
    if best_balanced is not None:
        best_agg = best_balanced.get("large_suite_aggregate", {})
        best_384 = best_agg.get(PRIMARY_BUDGET, {})
        best_768 = best_agg.get(EQ_768_BUDGET, {})
        best_1200 = best_agg.get(EQ_1200_BUDGET, {})
        lines.extend(
            [
                f"- best balanced lane: `{best_balanced['candidate']}`",
                f"- mean delta vs promoted current at `{PRIMARY_BUDGET}`: `{fmt(float(best_384.get('mean_delta_vs_promoted_current_ref', 0.0)))}`",
                f"- mean delta vs promoted current at `{EQ_768_BUDGET}`: `{fmt(float(best_768.get('mean_delta_vs_promoted_current_ref', 0.0)))}`",
                f"- mean delta vs promoted current at `{EQ_1200_BUDGET}`: `{fmt(float(best_1200.get('mean_delta_vs_promoted_current_ref', 0.0)))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Stability Replay",
            "",
            f"- selected rows: `{stability['selection_summary']['selected_rows']}`",
            f"- unique rows: `{stability['selection_summary']['unique_rows']}`",
            f"- overlap with mined disagreement before exclusion: `{stability['selection_summary']['raw_overlap_with_mined_disagreement_rows']}`",
            f"- stability entropy mean: `{stability['selection_summary']['target_entropy']['mean']:.4f}`",
            "",
            "## Candidate Aggregate Table",
            "",
        ]
    )
    rows: list[list[Any]] = []
    for row in candidate_rows:
        large_suite_aggregate = row.get("large_suite_aggregate", {})
        agg_384 = large_suite_aggregate.get(
            PRIMARY_BUDGET,
            {
                "mean_ds": 0.0,
                "mean_delta_vs_promoted_current_ref": 0.0,
            },
        )
        agg_768 = large_suite_aggregate.get(
            EQ_768_BUDGET,
            {"mean_delta_vs_promoted_current_ref": 0.0},
        )
        agg_1200 = large_suite_aggregate.get(
            EQ_1200_BUDGET,
            {"mean_delta_vs_promoted_current_ref": 0.0},
        )
        rows.append(
            [
                row["candidate"],
                fmt(float(agg_384["mean_ds"])),
                fmt(float(agg_384["mean_delta_vs_promoted_current_ref"])),
                fmt(float(agg_768["mean_delta_vs_promoted_current_ref"])),
                fmt(float(agg_1200["mean_delta_vs_promoted_current_ref"])),
                fmt(
                    float(
                        row["stability_anchor_top1_preserved_rate"][
                            "top1_preserved_rate"
                        ]
                    )
                ),
                fmt(
                    float(
                        row["mined_state_policy_shift"][
                            "top1_changed_rate_vs_promoted_current"
                        ]
                    )
                ),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Candidate",
                "Mean DS 384:256",
                "Delta 384:256",
                "Delta 768:768",
                "Delta 1200:1200",
                "Stability preserved",
                "Mined top-1 changed",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("## Bootstrap CI Table")
    lines.append("")
    ci_rows: list[list[Any]] = []
    for row in candidate_rows:
        for key, ci in sorted(row.get("bootstrap_cis", {}).items()):
            ci_rows.append(
                [
                    key,
                    fmt(float(ci["mean"])),
                    fmt(float(ci["lower"])),
                    fmt(float(ci["upper"])),
                    ci["n"],
                ]
            )
    lines.append(
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%", "Openings"], ci_rows
        )
    )
    lines.append("")
    lines.append("## Large-Suite Budgets")
    lines.append("")
    for budget_pair in large_budgets:
        budget_rows: list[list[Any]] = []
        for row in candidate_rows:
            aggregate = row.get("large_suite_aggregate", {}).get(
                budget_pair,
                {
                    "mean_ds": 0.0,
                    "worst_suite_ds": 0.0,
                    "mean_p0_score": 0.0,
                    "mean_p1_score": 0.0,
                    "mean_duplicate_trajectory_count": 0.0,
                },
            )
            budget_rows.append(
                [
                    row["candidate"],
                    fmt(float(aggregate["mean_ds"])),
                    fmt(float(aggregate["worst_suite_ds"])),
                    fmt(float(aggregate["mean_p0_score"])),
                    fmt(float(aggregate["mean_p1_score"])),
                    fmt(float(aggregate["mean_duplicate_trajectory_count"])),
                ]
            )
        lines.append(f"### `{budget_pair}`")
        lines.append("")
        lines.append(
            markdown_table(
                [
                    "Candidate",
                    "Mean DS",
                    "Worst-suite DS",
                    "Mean P0",
                    "Mean P1",
                    "Mean duplicates",
                ],
                budget_rows,
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workdir", default="/tmp/azlite_balanced_opening_puct_replay")
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=EXPECTED_PROMOTED_WEIGHTS_SHA256,
    )
    parser.add_argument(
        "--init-checkpoint",
        default="/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz",
    )
    parser.add_argument(
        "--expected-init-checkpoint-sha256",
        default=EXPECTED_INIT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--generic-bootstrap",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl",
    )
    parser.add_argument(
        "--random-teacher",
        default="/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument(
        "--mined-disagreement-replay",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement/opening_puct_disagreement_replay.jsonl",
    )
    parser.add_argument(
        "--disagreement-audit",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json",
    )
    parser.add_argument(
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--disagreement-stability-weight-pairs", default="8:4,8:8,4:8")
    parser.add_argument("--candidate-filter", default=None)
    parser.add_argument(
        "--budget-pairs", default="384:256,768:256,768:768,1200:1200,1200:256,256:768"
    )
    parser.add_argument("--include-384-equal-budget", action="store_true")
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--max-stability-rows", type=int, default=TARGET_STABILITY_ROWS)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--stability-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(workdir)
    stability_workers = args.stability_workers or args.workers

    current_artifact = Path(args.current)
    init_checkpoint = Path(args.init_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    mined_disagreement_replay = Path(args.mined_disagreement_replay)
    disagreement_audit = Path(args.disagreement_audit)
    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suite_paths = parse_csv_paths(args.heldout_suites)
    budget_pairs = parse_budget_pairs(args.budget_pairs)
    weight_pairs = parse_weight_pairs(args.disagreement_stability_weight_pairs)
    equal_budgets = parse_equal_budget_pairs(args.include_384_equal_budget)
    candidate_filter = parse_candidate_filter(args.candidate_filter)

    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(
        current_artifact / "metadata.json", "current artifact metadata"
    )
    require_existing_file(init_checkpoint, "init checkpoint")
    require_existing_file(generic_bootstrap, "generic bootstrap")
    require_existing_file(random_teacher, "random teacher")
    require_existing_file(mined_disagreement_replay, "mined disagreement replay")
    require_existing_file(disagreement_audit, "disagreement audit")
    require_existing_file(fixed_large_suite, "fixed large suite")
    require_existing_file(medium_suite, "medium suite")
    for path in heldout_suite_paths:
        require_existing_file(path, f"heldout suite {path.name}")

    input_summary = {
        "current_artifact_weights": verify_expected_hash(
            current_artifact / "weights.json",
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact_metadata": build_input_summary(
            current_artifact / "metadata.json"
        ),
        "init_checkpoint": verify_expected_hash(
            init_checkpoint,
            args.expected_init_checkpoint_sha256,
            "init checkpoint",
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "mined_disagreement_replay": build_input_summary(mined_disagreement_replay),
        "disagreement_audit": build_input_summary(disagreement_audit),
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suite_paths
        },
    }

    suite_specs = [("fixed_large", fixed_large_suite)] + [
        (path.stem, path) for path in heldout_suite_paths
    ]
    disagreement_rows = read_jsonl(mined_disagreement_replay)
    disagreement_hashes = {str(row["state_hash"]) for row in disagreement_rows}
    pr123_summary = (
        load_json(PR123_SUMMARY_PATH) if PR123_SUMMARY_PATH.is_file() else {}
    )

    stability_selected_rows: list[dict[str, Any]] = []
    stability_replay_rows: list[dict[str, Any]] = []
    stability_trace_rows: list[dict[str, Any]] = []
    stability_summary: dict[str, Any] = {}
    collection_metadata: dict[str, Any] = {}
    if phase_requested(args.phase, "stability"):
        reuse_stability = all(
            should_reuse_existing(
                path, resume=args.resume, skip_existing=args.skip_existing
            )
            for path in (
                paths["stability_trace"],
                paths["stability_selected"],
                paths["stability_replay"],
            )
        )
        if reuse_stability:
            stability_trace_rows = read_jsonl(paths["stability_trace"])
            stability_selected_rows = read_jsonl(paths["stability_selected"])
            stability_replay_rows = read_jsonl(paths["stability_replay"])
            if stability_trace_rows:
                collection_metadata = dict(
                    stability_trace_rows[0].get("_collection_metadata", {})
                )
            if stability_selected_rows:
                stability_summary = dict(
                    stability_selected_rows[0].get("_selection_summary", {})
                )
        else:
            analyzed_rows, collection_metadata = collect_stability_candidates(
                suite_specs=suite_specs,
                artifact_path=current_artifact,
                equal_budgets=equal_budgets,
                c_puct=args.c_puct,
                seed=args.seed,
                workers=stability_workers,
            )
            stability_selected_rows, stability_summary = select_stability_rows(
                analyzed_rows,
                disagreement_hashes,
                target_rows=args.max_stability_rows,
            )
            analyzed_rows.sort(key=lambda row: str(row["state_hash"]))
            stability_trace_rows = [{**row} for row in analyzed_rows]
            if stability_trace_rows:
                stability_trace_rows[0]["_collection_metadata"] = collection_metadata
            write_jsonl(paths["stability_trace"], stability_trace_rows)
            selected_rows_with_meta = [{**row} for row in stability_selected_rows]
            if selected_rows_with_meta:
                selected_rows_with_meta[0]["_selection_summary"] = stability_summary
            write_jsonl(paths["stability_selected"], selected_rows_with_meta)
            stability_replay_rows = build_stability_replay(
                selected_rows=stability_selected_rows,
                artifact_path=current_artifact,
                c_puct=args.c_puct,
                seed=args.seed,
                workers=stability_workers,
            )
            write_jsonl(paths["stability_replay"], stability_replay_rows)
        unique_selected = len(
            {str(row["state_hash"]) for row in stability_selected_rows}
        )
        if (
            unique_selected < STABILITY_MIN_ROWS
            and args.max_stability_rows >= STABILITY_MIN_ROWS
        ):
            summary = {
                "schema": SUMMARY_SCHEMA,
                "status": "aborted_before_training",
                "classification": "runner_still_blocked",
                "workdir": str(workdir),
                "inputs": input_summary,
                "stability_replay": {
                    "path": str(paths["stability_replay"]),
                    "collection_metadata": collection_metadata,
                    "selection_summary": stability_summary,
                },
            }
            write_json(paths["summary"], summary)
            return 1
    else:
        stability = load_stability_artifacts_or_fail(workdir)
        stability_trace_rows = stability["trace_rows"]
        stability_selected_rows = stability["selected_rows"]
        stability_replay_rows = stability["replay_rows"]
        if stability_trace_rows:
            collection_metadata = dict(
                stability_trace_rows[0].get("_collection_metadata", {})
            )
        if stability_selected_rows:
            stability_summary = dict(
                stability_selected_rows[0].get("_selection_summary", {})
            )

    if args.phase == "stability":
        return 0

    stability_replay_path = paths["stability_replay"]

    stability_state_lookup = {
        str(row["state_hash"]): row for row in stability_selected_rows
    }

    row_counts = {
        "generic_bootstrap": int(input_summary["generic_bootstrap"]["rows"]),
        "random_teacher": int(input_summary["random_teacher"]["rows"]),
        "mined_disagreement": int(input_summary["mined_disagreement_replay"]["rows"]),
        "stability_anchor": len(stability_replay_rows),
    }

    if phase_requested(args.phase, "train"):
        build_training_manifest(
            workdir=workdir,
            init_checkpoint=init_checkpoint,
            generic_bootstrap=generic_bootstrap,
            random_teacher=random_teacher,
            mined_disagreement_replay=mined_disagreement_replay,
            stability_replay_path=stability_replay_path,
            weight_pairs=weight_pairs,
            seed=args.seed,
            candidate_filter=candidate_filter,
            resume=args.resume,
            skip_existing=args.skip_existing,
        )
    else:
        load_training_manifest_or_fail(workdir)

    candidate_specs = build_candidate_specs(
        workdir=workdir,
        current_artifact=current_artifact,
        init_checkpoint=init_checkpoint,
        current_weights_sha256=args.expected_current_weights_sha256,
        pr123_summary=pr123_summary,
        row_counts=row_counts,
    )
    write_json(
        paths["candidate_manifest"],
        {
            "schema": SUMMARY_SCHEMA,
            "status": "completed",
            "workdir": str(workdir),
            "row_counts": row_counts,
            "candidates": candidate_specs,
        },
    )
    if args.phase == "train":
        return 0

    medium_report: dict[str, Any] | None = None
    large_report: dict[str, Any] | None = None
    heldout_reports: dict[str, dict[str, Any]] = {}
    if phase_requested(args.phase, "eval-fixed"):
        medium_report = evaluate_suite_candidates(
            workdir=workdir,
            suite_name="medium",
            suite_path=medium_suite,
            current_artifact=current_artifact,
            candidates=candidate_specs,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
            candidate_filter=candidate_filter,
            resume=args.resume,
            skip_existing=args.skip_existing,
        )
        large_report = evaluate_suite_candidates(
            workdir=workdir,
            suite_name="fixed_large",
            suite_path=fixed_large_suite,
            current_artifact=current_artifact,
            candidates=candidate_specs,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
            candidate_filter=candidate_filter,
            resume=args.resume,
            skip_existing=args.skip_existing,
        )
    else:
        if suite_report_path(workdir, "medium").is_file():
            medium_report = load_json(suite_report_path(workdir, "medium"))
        if suite_report_path(workdir, "fixed_large").is_file():
            large_report = load_json(suite_report_path(workdir, "fixed_large"))
    if args.phase == "eval-fixed":
        return 0

    if phase_requested(args.phase, "eval-heldout"):
        for suite_name, suite_path in suite_specs[1:]:
            heldout_reports[suite_name] = evaluate_suite_candidates(
                workdir=workdir,
                suite_name=suite_name,
                suite_path=suite_path,
                current_artifact=current_artifact,
                candidates=candidate_specs,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
                candidate_filter=candidate_filter,
                resume=args.resume,
                skip_existing=args.skip_existing,
            )
    else:
        for suite_name, _suite_path in suite_specs[1:]:
            report_path = suite_report_path(workdir, suite_name)
            if report_path.is_file():
                heldout_reports[suite_name] = load_json(report_path)
    if args.phase == "eval-heldout":
        return 0

    large_reports_map = {"fixed_large": large_report, **heldout_reports}
    available_reports = {
        name: report
        for name, report in large_reports_map.items()
        if report is not None and report_has_all_candidates(report, candidate_specs)
    }
    suite_rows = (
        large_suite_rows(reports=available_reports, candidates=candidate_specs)
        if available_reports
        else {}
    )
    gate_targets = (
        gate_targets_from_suite_rows(suite_rows, candidate_specs)
        if "fixed_large" in suite_rows
        else []
    )
    gate_reports: dict[str, dict[str, Any]] = {}
    if phase_requested(args.phase, "gate"):
        paths["gates_dir"].mkdir(parents=True, exist_ok=True)
        for candidate in candidate_specs:
            candidate_name = str(candidate["name"])
            if candidate_name not in gate_targets or (
                candidate_filter is not None
                and candidate_name not in set(candidate_filter)
            ):
                continue
            out_path = gate_report_path(workdir, candidate_name)
            gate_reports[candidate_name] = (
                load_json(out_path)
                if should_reuse_existing(
                    out_path, resume=args.resume, skip_existing=args.skip_existing
                )
                else run_default_gate(
                    candidate_path=str(candidate["artifact_dir"]),
                    current_path=str(current_artifact),
                    out=str(out_path),
                    seed=args.seed,
                    workers=args.workers,
                )
            )
    for candidate_name in gate_targets:
        out_path = gate_report_path(workdir, candidate_name)
        if out_path.is_file() and candidate_name not in gate_reports:
            gate_reports[candidate_name] = load_json(out_path)
    if args.phase == "gate":
        return 0

    stability_policy_rows = load_candidate_policy_shift_rows(
        stability_replay_rows, stability_state_lookup
    )
    disagreement_policy_rows = load_candidate_policy_shift_rows(
        disagreement_rows,
        {
            str(row["state_hash"]): row
            for row in load_json(disagreement_audit).get("states", [])
            if str(row.get("state_hash", "")) in disagreement_hashes
        },
    )

    summary_candidates: list[dict[str, Any]] = []
    required_heldout_reports = {
        suite_name: suite_report_path(workdir, suite_name)
        for suite_name, _path in suite_specs[1:]
    }
    for candidate in candidate_specs:
        checkpoint_path = Path(str(candidate["checkpoint_path"]))
        artifact_dir = Path(str(candidate["artifact_dir"]))
        checkpoint_sha = sha256_file(checkpoint_path)
        artifact_sha = sha256_file(artifact_dir / "weights.json")
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, init_checkpoint
        )
        row: dict[str, Any] = {
            "candidate": candidate["name"],
            "report_candidate_name": candidate["report_candidate_name"],
            "source": candidate["source"],
            "epochs": candidate["epochs"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "artifact_dir": str(artifact_dir),
            "artifact_weights_sha256": artifact_sha,
            "delta_norm_vs_promoted_e1": delta_norm,
            "relative_delta_pct_vs_promoted_e1": relative_delta_pct,
            "mined_state_policy_shift": {
                "states": 0,
                "top1_changed_rate_vs_promoted_current": 0.0,
            },
            "stability_anchor_top1_preserved_rate": anchored_top1_preserved_rate(
                artifact_dir, stability_policy_rows
            ),
        }
        if disagreement_policy_rows:
            evaluator = ArtifactEvaluator(artifact_dir)
            changed = 0
            for shift_row in disagreement_policy_rows:
                game = KalahGame.from_state(shift_row["state"])
                _logits, raw_policy, _value = artifact_forward_details(evaluator, game)
                top1 = top_policy_move(raw_policy, shift_row["legal_moves"])
                if top1 != shift_row["raw_top1"]:
                    changed += 1
            row["mined_state_policy_shift"] = {
                "states": len(disagreement_policy_rows),
                "top1_changed_rate_vs_promoted_current": changed
                / len(disagreement_policy_rows),
            }
        train_metrics = candidate.get("train_metrics")
        if isinstance(train_metrics, dict) and train_metrics:
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["validation_loss"] = train_metrics.get("best_val_loss")
        if "effective_replay_sampling_fractions" in candidate:
            row["effective_replay_sampling_fractions"] = candidate[
                "effective_replay_sampling_fractions"
            ]
        if medium_report is not None and report_has_candidate(
            medium_report, str(candidate["report_candidate_name"])
        ):
            report = find_candidate_report(
                medium_report, str(candidate["report_candidate_name"])
            )
            row["medium_budget_results"] = benchmark_budget_results(report)
        if large_report is not None and report_has_candidate(
            large_report, str(candidate["report_candidate_name"])
        ):
            report = find_candidate_report(
                large_report, str(candidate["report_candidate_name"])
            )
            row["large_budget_results"] = benchmark_budget_results(report)
        row["heldout_summary"] = (
            heldout_summary(heldout_reports, str(candidate["report_candidate_name"]))
            if heldout_reports
            else {"available": False}
        )
        row["large_suite_aggregate"] = (
            {
                budget_pair: aggregate_budget_summary(
                    suite_rows,
                    str(candidate["name"]),
                    budget_pair,
                    "promoted_current_ref",
                )
                for budget_pair in budget_pairs
            }
            if suite_rows
            else {}
        )
        row["bootstrap_cis"] = {}
        for budget_pair in (PRIMARY_BUDGET, EQ_768_BUDGET, EQ_1200_BUDGET):
            if not suite_rows:
                continue
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a=str(candidate["name"]),
                candidate_b="promoted_current_ref",
                budget_pair=budget_pair,
                metric_key="ds",
            )
            row["bootstrap_cis"][
                f"{candidate['name']}_minus_promoted_current_ref_{budget_pair.replace(':', '_')}"
            ] = bootstrap_ci(diffs, seed=args.seed, samples=DEFAULT_BOOTSTRAP_SAMPLES)
        if suite_rows and str(candidate["name"]) != "pr123_w8_e1_ref":
            for budget_pair in (PRIMARY_BUDGET, EQ_768_BUDGET):
                diffs = pooled_per_opening_differences(
                    suite_rows=suite_rows,
                    candidate_a=str(candidate["name"]),
                    candidate_b="pr123_w8_e1_ref",
                    budget_pair=budget_pair,
                    metric_key="ds",
                )
                row["bootstrap_cis"][
                    f"{candidate['name']}_minus_pr123_w8_e1_ref_{budget_pair.replace(':', '_')}"
                ] = bootstrap_ci(
                    diffs, seed=args.seed + 1, samples=DEFAULT_BOOTSTRAP_SAMPLES
                )
        if str(candidate["name"]) in gate_reports:
            row["default_gate"] = {
                "classification": gate_reports[str(candidate["name"])].get(
                    "classification"
                ),
                "budget_results": gate_budget_results(
                    gate_reports[str(candidate["name"])]
                ),
            }
        summary_candidates.append(row)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "classification": classify_from_cached_artifacts(
            stability_selected_rows=stability_selected_rows,
            required_heldout_reports=required_heldout_reports,
            summary_candidates=summary_candidates,
        ),
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "stability_workers": stability_workers,
        "phase": args.phase,
        "budget_pairs": budget_pairs,
        "games_per_opening": args.games_per_opening,
        "inputs": input_summary,
        "guardrails": {
            "promotion": False,
            "overwrite_current": False,
            "new_self_play": False,
            "classic_mcts_replay": False,
            "full_network_training": False,
            "last_block_policy_training": False,
            "architecture_change": False,
            "residual_v4": False,
            "lr_change": False,
            "threshold_tuning_after_results": False,
        },
        "stability_replay": {
            "path": str(stability_replay_path),
            "row_count": len(stability_replay_rows),
            "unique_row_count": len(
                {str(row["state_hash"]) for row in stability_replay_rows}
            ),
            "collection_metadata": collection_metadata,
            "selection_summary": stability_summary,
            "top1_distribution_by_pit": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        int(row["search_top1"])
                        for row in stability_selected_rows
                        if row["search_top1"] is not None
                    ).items()
                )
            },
        },
        "row_counts": row_counts,
        "gate_targets": gate_targets,
        "experiment_completion": {
            "status": "completed"
            if all(path.is_file() for path in required_heldout_reports.values())
            else "partially_completed",
            "fixed_eval_complete": large_report is not None,
            "heldout_eval_complete": all(
                path.is_file() for path in required_heldout_reports.values()
            ),
            "gate_complete": all(
                gate_report_path(workdir, candidate_name).is_file()
                for candidate_name in gate_targets
            )
            if gate_targets
            else False,
        },
        "candidates": summary_candidates,
    }
    write_json(paths["summary"], summary)
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
