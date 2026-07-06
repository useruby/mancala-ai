#!/usr/bin/env python3
"""Search-teacher student distillation preflight for larger residual models.

Does not promote, overwrite `model-artifact/current`, or mutate current weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    benchmark_budget_results,
    bootstrap_ci,
    candidate_gate_preserved,
    find_candidate_report,
    pooled_per_opening_differences,
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.run_promoted_current_opening_puct_disagreement import (  # noqa: E402
    artifact_forward_details,
    canonical_state_hash,
    policy_entropy,
    read_jsonl,
    search_policy_from_visits,
    top_policy_move,
    write_jsonl,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_eval_search_options,
    build_search_profile,
    encode_state,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    checkpoint_from_model,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    input_size_for_encoding,
    select_device,
    set_seed,
)


SUMMARY_SCHEMA = "azlite_search_teacher_student_preflight_v1"
DATASET_FILENAME = "search_teacher_dataset.jsonl"
DATASET_AUDIT_FILENAME = "search_teacher_dataset_audit.json"
SUMMARY_FILENAME = "summary_metrics.json"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-search-teacher-student-preflight-results.md"
)
RUN_MANIFEST_FILENAME = "run_manifest.json"
DEFAULT_BUDGET_CONTEXTS = ((384, 256), (768, 768), (1200, 1200), (1200, 256))
DEFAULT_SUITE_EVAL_BUDGETS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
DEFAULT_GATE_BUDGETS = "384:256,1200:1200,1200:256,256:768"
PRIMARY_BUDGET = "384:256"
EQ_768_BUDGET = "768:768"
EQ_1200_BUDGET = "1200:1200"
ASYM_1200_256_BUDGET = "1200:256"
MAX_MOVES = 200
DEFAULT_TARGET_ROWS = 50000
MIN_TARGET_ROWS = 20000
DEFAULT_GENERATION_ROWS = 60000
DEFAULT_PREFIX_CAP = 24
DEFAULT_MAX_TRAJECTORY_PLIES = 12
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 2
DEFAULT_LR = 3e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_VALUE_LOSS = "huber"
DEFAULT_HUBER_DELTA = 1.0
DEFAULT_VALUE_LOSS_WEIGHT = 1.0
DEFAULT_BOOTSTRAP_SAMPLES = 10000
DEFAULT_WORKERS = 24
KL_EPSILON = 1e-8
FAILURE_DOC_PATH = (
    REPO_ROOT / "docs/alphazero-lite-current-384-256-failure-family-trace-results.md"
)
EARLY_STOP_TOP1_THRESHOLD = 0.50


@dataclass(frozen=True)
class StudentSpec:
    name: str
    model_type: str
    trunk_size: int
    residual_block_count: int

    @property
    def hidden_sizes(self) -> tuple[int, int]:
        return (self.trunk_size, self.residual_block_count)


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log_progress(message: str) -> None:
    print(f"[search-teacher-preflight] {message}", flush=True)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in str(text).split(",") if item.strip()]


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify_expected_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    result = {
        "path": str(path),
        "actual_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "matches_expected": actual_hash == expected_hash,
    }
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return result


def parse_student_architecture(text: str) -> StudentSpec:
    raw = str(text).strip()
    parts = raw.split("_")
    if len(parts) < 3 or parts[0] != "residual" or parts[1] != "v3":
        raise ValueError(f"unsupported student architecture: {raw}")
    shape = parts[2]
    if "x" not in shape:
        raise ValueError(f"student architecture must look like residual_v3_96x3: {raw}")
    trunk_size_str, block_count_str = shape.split("x", 1)
    return StudentSpec(
        name=raw,
        model_type="residual_v3",
        trunk_size=int(trunk_size_str),
        residual_block_count=int(block_count_str),
    )


def parse_budget_pair(text: str) -> tuple[int, int]:
    left, right = str(text).split(":", 1)
    return int(left), int(right)


def budget_label(challenger_sims: int, current_sims: int) -> str:
    return f"{int(challenger_sims)}:{int(current_sims)}"


def suite_entry_key(entry: dict[str, Any]) -> str:
    return ",".join(str(int(move)) for move in entry.get("prefix_moves", []))


def state_outcome_from_game(game: KalahGame, current_player: int) -> float:
    if not game.over():
        return 0.0
    player0 = int(game.captured_seeds[0])
    player1 = int(game.captured_seeds[1])
    if player0 == player1:
        return 0.0
    winner = 0 if player0 > player1 else 1
    return 1.0 if int(current_player) == winner else -1.0


def phase_for_state(entry: dict[str, Any], game: KalahGame) -> str:
    total_ply = int(entry.get("ply", 0))
    if total_ply <= 8:
        return "opening"
    seeds_remaining = sum(int(value) for value in game.pits)
    if seeds_remaining <= 16:
        return "late"
    return "mid"


def masked_policy(
    policy: list[float] | np.ndarray, legal_moves: list[int]
) -> list[float]:
    masked = np.zeros(6, dtype=np.float32)
    if not legal_moves:
        return masked.tolist()
    policy_array = np.asarray(policy, dtype=np.float64)
    total = float(np.sum(policy_array[legal_moves]))
    if total <= 0.0:
        masked[legal_moves] = 1.0 / len(legal_moves)
        return masked.tolist()
    masked[legal_moves] = policy_array[legal_moves] / total
    return masked.tolist()


def kl_divergence(p: list[float], q: list[float], legal_moves: list[int]) -> float:
    if not legal_moves:
        return 0.0
    p_arr = np.clip(np.asarray(p, dtype=np.float64), KL_EPSILON, 1.0)
    q_arr = np.clip(np.asarray(q, dtype=np.float64), KL_EPSILON, 1.0)
    p_arr = p_arr[legal_moves]
    q_arr = q_arr[legal_moves]
    p_arr /= np.sum(p_arr)
    q_arr /= np.sum(q_arr)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def load_failure_state_prefixes(doc_path: Path) -> set[str]:
    if not doc_path.is_file():
        return set()
    prefixes: set[str] = set()
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 3:
            continue
        state_hash = parts[1]
        if len(state_hash) == 12 and all(ch in "0123456789abcdef" for ch in state_hash):
            prefixes.add(state_hash)
    return prefixes


def build_run_manifest(
    *,
    current_weights_sha256: str,
    fixed_large_suite: Path,
    heldout_suites: list[Path],
    medium_suite: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    target_rows: int,
    max_trajectory_plies: int,
    prefix_cap: int,
    student_architectures: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "current_weights_sha256": current_weights_sha256,
        "fixed_large_suite": {
            "path": str(fixed_large_suite),
            "sha256": sha256_file(fixed_large_suite),
        },
        "heldout_suites": [
            {"path": str(path), "sha256": sha256_file(path)} for path in heldout_suites
        ],
        "medium_suite": {
            "path": str(medium_suite),
            "sha256": sha256_file(medium_suite),
        },
        "teacher": {
            "default_c_puct": float(default_c_puct),
            "cpuct_schedule": dict(sorted(cpuct_schedule.items())),
            "tactical_root_bias": float(tactical_root_bias),
        },
        "dataset": {
            "target_rows": int(target_rows),
            "max_trajectory_plies": int(max_trajectory_plies),
            "prefix_cap": int(prefix_cap),
        },
        "students": {
            "architectures": list(student_architectures),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "grad_clip": float(grad_clip),
        },
        "seed": int(seed),
    }


def manifests_match(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    return existing == expected


def build_teacher_search_profile(
    *,
    simulations: int,
    challenger_sims: int,
    current_sims: int,
    effective_c_puct: float,
    tactical_root_bias: float,
    artifact_path: str,
) -> dict[str, Any]:
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    return build_search_profile(
        kind="teacher_dataset",
        player_mode="puct",
        simulations=simulations,
        c_puct=effective_c_puct,
        search_options=search_options,
        extra_fields={
            "artifact": str(artifact_path),
            "challenger_simulations": int(challenger_sims),
            "current_simulations": int(current_sims),
        },
    )


def evaluate_teacher_state(
    *,
    evaluator: ArtifactEvaluator,
    game: KalahGame,
    challenger_sims: int,
    current_sims: int,
    effective_c_puct: float,
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    raw_policy, raw_logits, raw_value = artifact_forward_details(evaluator, game)
    result = evaluate_artifact_position(
        evaluator=evaluator,
        state=game.to_state(),
        simulations=int(challenger_sims),
        seed=int(seed),
        c_puct=float(effective_c_puct),
        search_options=search_options,
        ablation_mode="full",
    )
    legal_moves = [int(move) for move in result["legal_moves"]]
    puct_policy = search_policy_from_visits(result["visits"], legal_moves)
    return {
        "legal_moves": legal_moves,
        "legal_mask": [1 if move in legal_moves else 0 for move in range(6)],
        "teacher_raw_policy": masked_policy(raw_policy, legal_moves),
        "teacher_raw_logits": [float(value) for value in raw_logits],
        "teacher_puct_policy": puct_policy,
        "teacher_selected_move": int(result["selected_move"])
        if result.get("selected_move") is not None
        else None,
        "teacher_root_value": float(result.get("value", raw_value)),
        "teacher_search_root_value": float(result.get("search_root_value", 0.0)),
        "teacher_raw_value": float(raw_value),
        "teacher_visit_counts": [float(value) for value in result.get("visits", [])],
        "selection_breakdown": result.get("selection_breakdown"),
        "visit_snapshots": result.get("visit_snapshots"),
        "root_evaluation_raw_value": result.get("root_evaluation_raw_value"),
        "root_evaluation_transformed_value": result.get(
            "root_evaluation_transformed_value"
        ),
    }


def deterministic_teacher_trajectory(
    *,
    evaluator: ArtifactEvaluator,
    suite_entry: dict[str, Any],
    challenger_sims: int,
    current_sims: int,
    effective_c_puct: float,
    tactical_root_bias: float,
    max_plies: int,
    seed: int,
    search_profile_hash: str,
    suite_name: str,
) -> list[dict[str, Any]]:
    game = KalahGame.from_state(suite_entry["state"])
    opening_prefix = suite_entry_key(suite_entry)
    first_move_family = str(suite_entry.get("first_move_family", "unknown"))
    seat_context = (
        "challenger_player_0"
        if int(game.current_player) == 0
        else "challenger_player_1"
    )
    rows: list[dict[str, Any]] = []

    for local_ply in range(max_plies):
        if game.over():
            break
        current_state = game.to_state()
        teacher = evaluate_teacher_state(
            evaluator=evaluator,
            game=game,
            challenger_sims=challenger_sims,
            current_sims=current_sims,
            effective_c_puct=effective_c_puct,
            tactical_root_bias=tactical_root_bias,
            seed=seed + local_ply,
        )
        legal_moves = teacher["legal_moves"]
        selected_move = teacher["teacher_selected_move"]
        if selected_move is None or selected_move not in legal_moves:
            break
        state_hash = canonical_state_hash(current_state)
        phase = phase_for_state(
            {**suite_entry, "ply": int(suite_entry.get("ply", 0)) + local_ply}, game
        )
        rows.append(
            {
                "suite_name": suite_name,
                "state_hash": state_hash,
                "state": encode_state(
                    current_state, input_encoding=evaluator.input_encoding
                ),
                "raw_state": current_state,
                "legal_mask": teacher["legal_mask"],
                "teacher_raw_policy": teacher["teacher_raw_policy"],
                "teacher_puct_policy": teacher["teacher_puct_policy"],
                "teacher_selected_move": selected_move,
                "teacher_root_value": teacher["teacher_root_value"],
                "teacher_search_root_value": teacher["teacher_search_root_value"],
                "teacher_raw_value": teacher["teacher_raw_value"],
                "phase": phase,
                "seat_context": seat_context,
                "opening_prefix": opening_prefix,
                "prefix_length": len(suite_entry.get("prefix_moves", [])),
                "first_move_family": first_move_family,
                "budget_context": budget_label(challenger_sims, current_sims),
                "challenger_simulations": int(challenger_sims),
                "current_simulations": int(current_sims),
                "effective_c_puct": float(effective_c_puct),
                "search_profile_hash": search_profile_hash,
                "search_profile_kind": "teacher_dataset",
                "source_suite": suite_name,
                "ply_from_suite_root": int(local_ply),
                "absolute_ply": int(suite_entry.get("ply", 0)) + int(local_ply),
                "value_target": None,
                "value_target_source": None,
                "final_outcome": None,
                "selection_breakdown": teacher["selection_breakdown"],
                "visit_snapshots": teacher["visit_snapshots"],
                "root_evaluation_raw_value": teacher["root_evaluation_raw_value"],
                "root_evaluation_transformed_value": teacher[
                    "root_evaluation_transformed_value"
                ],
            }
        )
        if not game.move(game.pit_index(int(selected_move))):
            break

    final_outcomes = [
        state_outcome_from_game(game, int(row["raw_state"]["current_player"]))
        for row in rows
    ]
    for row, final_outcome in zip(rows, final_outcomes, strict=True):
        row["final_outcome"] = float(final_outcome)
        row["value_target"] = float(final_outcome)
        row["value_target_source"] = "deterministic_continuation"
    return rows


def build_dataset(
    *,
    workdir: Path,
    current_artifact: Path,
    fixed_large_suite: Path,
    heldout_suites: list[Path],
    medium_suite: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    target_rows: int,
    seed: int,
    max_trajectory_plies: int,
    prefix_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dataset_path = workdir / DATASET_FILENAME
    dataset_audit_path = workdir / DATASET_AUDIT_FILENAME
    if dataset_path.is_file() and dataset_audit_path.is_file():
        existing_rows = read_jsonl(dataset_path)
        existing_audit = load_json(dataset_audit_path)
        if int(existing_audit.get("row_count", 0)) >= min(
            MIN_TARGET_ROWS, max(int(target_rows), 1)
        ):
            log_progress(
                f"reusing existing dataset rows={len(existing_rows)} path={dataset_path}"
            )
            evaluator = ArtifactEvaluator(current_artifact)
            probe_rows = build_probe_rows(
                evaluator=evaluator,
                suite_paths=[medium_suite],
                current_artifact=current_artifact,
                default_c_puct=default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=tactical_root_bias,
                max_trajectory_plies=max_trajectory_plies,
                seed=seed + 999,
            )
            inputs = {
                "dataset_path": str(dataset_path),
                "dataset_audit_path": str(dataset_audit_path),
                "dataset_rows": len(existing_rows),
                "probe_rows": len(probe_rows),
                "training_suites": [
                    str(path) for path in [fixed_large_suite, *heldout_suites]
                ],
                "probe_suites": [str(medium_suite)],
                "budget_targets": dict(
                    sorted(
                        Counter(
                            str(row["budget_context"]) for row in existing_rows
                        ).items()
                    )
                ),
                "prefix_cap": int(prefix_cap),
                "max_trajectory_plies": int(max_trajectory_plies),
                "reused_existing_dataset": True,
            }
            return existing_rows, probe_rows, inputs

    evaluator = ArtifactEvaluator(current_artifact)
    log_progress(
        "building teacher dataset "
        f"target_rows={target_rows} max_plies={max_trajectory_plies} prefix_cap={prefix_cap}"
    )
    training_suite_paths = [fixed_large_suite, *heldout_suites]
    probe_suite_paths = [medium_suite]
    suite_entries: list[tuple[str, dict[str, Any]]] = []
    for suite_path in training_suite_paths:
        suite_name = suite_path.stem
        for entry in read_jsonl(suite_path):
            suite_entries.append((suite_name, entry))

    random.Random(seed).shuffle(suite_entries)

    rows: list[dict[str, Any]] = []
    generated_rows_target = max(int(target_rows), MIN_TARGET_ROWS)
    budget_target = max(generated_rows_target // len(DEFAULT_BUDGET_CONTEXTS), 1)
    budget_counts = Counter()
    prefix_counts = Counter()
    queue = deque(suite_entries)
    exhausted_rounds = 0

    while queue and len(rows) < max(DEFAULT_GENERATION_ROWS, generated_rows_target):
        suite_name, entry = queue.popleft()
        opening_prefix = suite_entry_key(entry)
        if prefix_counts[opening_prefix] >= prefix_cap:
            exhausted_rounds += 1
            if exhausted_rounds > len(suite_entries) * len(DEFAULT_BUDGET_CONTEXTS):
                break
            continue
        exhausted_rounds = 0
        for challenger_sims, current_sims in DEFAULT_BUDGET_CONTEXTS:
            budget_key = budget_label(challenger_sims, current_sims)
            if (
                budget_counts[budget_key] >= budget_target
                and len(rows) >= MIN_TARGET_ROWS
            ):
                continue
            effective_c_puct = resolve_budget_cpuct(
                schedule=cpuct_schedule,
                challenger_simulations=challenger_sims,
                current_simulations=current_sims,
                default_c_puct=default_c_puct,
            )
            profile = build_teacher_search_profile(
                simulations=challenger_sims,
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                effective_c_puct=effective_c_puct,
                tactical_root_bias=tactical_root_bias,
                artifact_path=str(current_artifact),
            )
            trajectory_rows = deterministic_teacher_trajectory(
                evaluator=evaluator,
                suite_entry=entry,
                challenger_sims=challenger_sims,
                current_sims=current_sims,
                effective_c_puct=effective_c_puct,
                tactical_root_bias=tactical_root_bias,
                max_plies=max_trajectory_plies,
                seed=seed + len(rows) + challenger_sims + current_sims,
                search_profile_hash=str(profile["hash"]),
                suite_name=suite_name,
            )
            remaining_prefix = max(0, prefix_cap - prefix_counts[opening_prefix])
            if remaining_prefix <= 0:
                break
            selected_rows = trajectory_rows[:remaining_prefix]
            rows.extend(selected_rows)
            prefix_counts[opening_prefix] += len(selected_rows)
            budget_counts[budget_key] += len(selected_rows)
            if len(rows) % 5000 == 0:
                log_progress(
                    "dataset generation "
                    f"rows={len(rows)} unique_states={len({str(row['state_hash']) for row in rows})}"
                )
            if len(rows) >= max(DEFAULT_GENERATION_ROWS, generated_rows_target):
                break
        if len(rows) >= max(DEFAULT_GENERATION_ROWS, generated_rows_target):
            break

    minimum_required_rows = min(MIN_TARGET_ROWS, max(int(target_rows), 1))
    if len(rows) < minimum_required_rows:
        raise RuntimeError(
            f"teacher dataset too small: generated {len(rows)} rows, require at least {minimum_required_rows}"
        )

    balanced_rows = select_balanced_rows(
        rows, target_rows=generated_rows_target, seed=seed
    )
    write_jsonl(dataset_path, balanced_rows)
    audit = audit_dataset(
        rows=balanced_rows,
        failure_state_prefixes=load_failure_state_prefixes(FAILURE_DOC_PATH),
    )
    write_json(workdir / DATASET_AUDIT_FILENAME, audit)
    log_progress(
        f"dataset ready rows={len(balanced_rows)} unique_states={audit['unique_state_count']}"
    )

    probe_rows = build_probe_rows(
        evaluator=evaluator,
        suite_paths=probe_suite_paths,
        current_artifact=current_artifact,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        max_trajectory_plies=max_trajectory_plies,
        seed=seed + 999,
    )

    inputs = {
        "dataset_path": str(dataset_path),
        "dataset_audit_path": str(workdir / DATASET_AUDIT_FILENAME),
        "dataset_rows": len(balanced_rows),
        "probe_rows": len(probe_rows),
        "training_suites": [str(path) for path in training_suite_paths],
        "probe_suites": [str(path) for path in probe_suite_paths],
        "budget_targets": dict(sorted(budget_counts.items())),
        "prefix_cap": int(prefix_cap),
        "max_trajectory_plies": int(max_trajectory_plies),
    }
    return balanced_rows, probe_rows, inputs


def select_balanced_rows(
    rows: list[dict[str, Any]], *, target_rows: int, seed: int
) -> list[dict[str, Any]]:
    if len(rows) <= target_rows:
        return list(rows)
    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["budget_context"]),
            str(row["phase"]),
            str(row["seat_context"]),
            str(row["first_move_family"]),
        )
        grouped[key].append(row)
    ordered_keys = list(grouped)
    rng.shuffle(ordered_keys)
    for key in ordered_keys:
        rng.shuffle(grouped[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < target_rows:
        added = False
        for key in ordered_keys:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            added = True
            if len(selected) >= target_rows:
                break
        if not added:
            break
    return selected


def build_probe_rows(
    *,
    evaluator: ArtifactEvaluator,
    suite_paths: list[Path],
    current_artifact: Path,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    max_trajectory_plies: int,
    seed: int,
) -> list[dict[str, Any]]:
    probe_rows: list[dict[str, Any]] = []
    for suite_path in suite_paths:
        suite_name = suite_path.stem
        entries = read_jsonl(suite_path)
        for entry in entries:
            for challenger_sims, current_sims in DEFAULT_BUDGET_CONTEXTS:
                effective_c_puct = resolve_budget_cpuct(
                    schedule=cpuct_schedule,
                    challenger_simulations=challenger_sims,
                    current_simulations=current_sims,
                    default_c_puct=default_c_puct,
                )
                profile = build_teacher_search_profile(
                    simulations=challenger_sims,
                    challenger_sims=challenger_sims,
                    current_sims=current_sims,
                    effective_c_puct=effective_c_puct,
                    tactical_root_bias=tactical_root_bias,
                    artifact_path=str(current_artifact),
                )
                probe_rows.extend(
                    deterministic_teacher_trajectory(
                        evaluator=evaluator,
                        suite_entry=entry,
                        challenger_sims=challenger_sims,
                        current_sims=current_sims,
                        effective_c_puct=effective_c_puct,
                        tactical_root_bias=tactical_root_bias,
                        max_plies=max_trajectory_plies,
                        seed=seed
                        + challenger_sims
                        + current_sims
                        + int(entry.get("ply", 0)),
                        search_profile_hash=str(profile["hash"]),
                        suite_name=suite_name,
                    )
                )
    return probe_rows


def audit_dataset(
    *, rows: list[dict[str, Any]], failure_state_prefixes: set[str]
) -> dict[str, Any]:
    row_count = len(rows)
    state_hashes = [str(row["state_hash"]) for row in rows]
    unique_state_count = len(set(state_hashes))
    phase_counts = Counter(str(row["phase"]) for row in rows)
    seat_counts = Counter(str(row["seat_context"]) for row in rows)
    budget_counts = Counter(str(row["budget_context"]) for row in rows)
    top_move_counts = Counter(str(row["teacher_selected_move"]) for row in rows)
    value_buckets = Counter()
    policy_entropies: list[float] = []
    visit_top_shares: list[float] = []
    raw_vs_puct_disagreements = 0
    overlap_failures = 0
    prefix_counts = Counter(str(row["opening_prefix"]) for row in rows)

    for row in rows:
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        raw_policy = list(row["teacher_raw_policy"])
        puct_policy = list(row["teacher_puct_policy"])
        raw_top = top_policy_move(raw_policy, legal_moves)
        puct_top = top_policy_move(puct_policy, legal_moves)
        if raw_top != puct_top:
            raw_vs_puct_disagreements += 1
        policy_entropies.append(policy_entropy(puct_policy, legal_moves))
        visit_top_shares.append(max(float(puct_policy[move]) for move in legal_moves))
        value = float(row.get("final_outcome", row.get("teacher_root_value", 0.0)))
        if value > 0.0:
            value_buckets["positive"] += 1
        elif value < 0.0:
            value_buckets["negative"] += 1
        else:
            value_buckets["draw"] += 1
        if str(row["state_hash"])[:12] in failure_state_prefixes:
            overlap_failures += 1

    duplicate_rows = row_count - unique_state_count
    duplicate_rate = duplicate_rows / max(row_count, 1)
    return {
        "schema": SUMMARY_SCHEMA,
        "row_count": row_count,
        "unique_state_count": unique_state_count,
        "duplicate_state_rate": duplicate_rate,
        "duplicate_state_rows": duplicate_rows,
        "phase_distribution": dict(sorted(phase_counts.items())),
        "seat_distribution": dict(sorted(seat_counts.items())),
        "budget_context_distribution": dict(sorted(budget_counts.items())),
        "opening_prefix_distribution_top20": dict(prefix_counts.most_common(20)),
        "policy_entropy": {
            "mean": float(statistics.fmean(policy_entropies))
            if policy_entropies
            else 0.0,
            "median": float(statistics.median(policy_entropies))
            if policy_entropies
            else 0.0,
        },
        "visit_share_distribution": {
            "mean_top_share": float(statistics.fmean(visit_top_shares))
            if visit_top_shares
            else 0.0,
            "median_top_share": float(statistics.median(visit_top_shares))
            if visit_top_shares
            else 0.0,
        },
        "top_move_distribution": dict(sorted(top_move_counts.items())),
        "value_outcome_distribution": dict(sorted(value_buckets.items())),
        "teacher_raw_vs_puct_top1_disagreement_rate": raw_vs_puct_disagreements
        / max(row_count, 1),
        "overlap_with_pr149_failure_states": {
            "matching_rows": int(overlap_failures),
            "matching_unique_states": len(
                {
                    str(row["state_hash"])
                    for row in rows
                    if str(row["state_hash"])[:12] in failure_state_prefixes
                }
            ),
            "state_hash_prefixes_available": len(failure_state_prefixes),
        },
    }


def prepare_training_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    p = np.asarray([row["teacher_puct_policy"] for row in rows], dtype=np.float32)
    v = np.asarray([[row["value_target"]] for row in rows], dtype=np.float32)
    legal_mask = np.asarray([row["legal_mask"] for row in rows], dtype=np.float32)
    replay_indexes = np.arange(len(rows), dtype=np.int64)
    return {
        "x": x,
        "p": p,
        "v": v,
        "legal_mask": legal_mask,
        "replay_indexes": replay_indexes,
    }


def masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(legal_mask <= 0.0, -1e9)
    return compute_policy_cross_entropy(masked_logits, targets)


def predict_student(
    model: PolicyValueNet,
    x: np.ndarray,
    legal_mask: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    policies: list[np.ndarray] = []
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device)
            maskb = torch.from_numpy(legal_mask[start:end]).to(device)
            logits, value = model(xb)
            masked_logits = logits.masked_fill(maskb <= 0.0, -1e9)
            policy = torch.softmax(masked_logits, dim=1)
            policies.append(policy.detach().cpu().numpy())
            values.append(value.detach().cpu().numpy())
    return np.concatenate(policies, axis=0), np.concatenate(values, axis=0)


def export_checkpoint_artifact(
    *,
    checkpoint_path: Path,
    export_dir: Path,
    version: str,
    model_type: str,
    input_encoding: str,
    policy_loss: float,
    value_loss: float,
) -> Path:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(export_dir),
        "--version",
        version,
        "--self-play-games",
        "0",
        "--policy-loss",
        str(policy_loss),
        "--value-loss",
        str(value_loss),
        "--model-type",
        model_type,
        "--rules-version",
        "kalah_v1",
        "--input-encoding",
        input_encoding,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"artifact export failed: {result.stderr[-2000:]}")
    return export_dir


def current_checkpoint_path(current_artifact: Path, workdir: Path) -> Path:
    for name in ("checkpoint.npz", "model.npz"):
        candidate = current_artifact / name
        if candidate.is_file():
            return candidate
    weights_path = current_artifact / "weights.json"
    if weights_path.is_file():
        return materialize_weights_json_checkpoint(
            weights_path=weights_path,
            out_path=workdir / "current_materialized_checkpoint.npz",
        )
    raise FileNotFoundError(
        f"current artifact has no checkpoint payload: {current_artifact}"
    )


def train_student_candidate(
    *,
    spec: StudentSpec,
    train_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    workdir: Path,
    input_encoding: str,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    lane_dir = workdir / spec.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = lane_dir / f"checkpoint_epoch{epochs}.npz"
    final_artifact = lane_dir / f"artifact_epoch{epochs}"
    if final_checkpoint.is_file() and (final_artifact / "weights.json").is_file():
        log_progress(
            f"reusing trained student candidate={spec.name} artifact={final_artifact}"
        )
        probe_metrics = evaluate_probe_candidate(
            artifact_path=final_artifact,
            candidate_name=spec.name,
            probe_rows=probe_rows,
        )
        checkpoints = []
        for epoch in range(1, epochs + 1):
            epoch_checkpoint = lane_dir / f"checkpoint_epoch{epoch}.npz"
            epoch_artifact = lane_dir / f"artifact_epoch{epoch}"
            if (
                epoch_checkpoint.is_file()
                and (epoch_artifact / "weights.json").is_file()
            ):
                checkpoints.append(
                    {
                        "epoch": epoch,
                        "checkpoint_path": str(epoch_checkpoint),
                        "checkpoint_sha256": sha256_file(epoch_checkpoint),
                        "artifact_dir": str(epoch_artifact),
                        "artifact_weights_sha256": sha256_file(
                            epoch_artifact / "weights.json"
                        ),
                    }
                )
        return {
            "name": spec.name,
            "kind": "student",
            "architecture": {
                "model_type": spec.model_type,
                "trunk_size": spec.trunk_size,
                "residual_block_count": spec.residual_block_count,
                "hidden_sizes": list(spec.hidden_sizes),
            },
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "grad_clip": grad_clip,
                "seed": seed,
                "rows": len(train_rows),
                "reused_existing_training": True,
            },
            "training_metrics": [],
            "checkpoints": checkpoints,
            "artifact_dir": str(final_artifact),
            "artifact_weights_sha256": sha256_file(final_artifact / "weights.json"),
            "probe_metrics": probe_metrics,
        }

    log_progress(
        f"training student candidate={spec.name} rows={len(train_rows)} epochs={epochs}"
    )
    arrays = prepare_training_arrays(train_rows)
    model = PolicyValueNet(
        hidden_sizes=spec.hidden_sizes,
        model_type=spec.model_type,
        input_size=input_size_for_encoding(input_encoding),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    set_seed(seed)
    training_metrics: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    best_probe_metrics: dict[str, Any] | None = None

    x_t = torch.from_numpy(arrays["x"]).to(device)
    p_t = torch.from_numpy(arrays["p"]).to(device)
    v_t = torch.from_numpy(arrays["v"]).to(device)
    legal_mask_t = torch.from_numpy(arrays["legal_mask"]).to(device)

    for epoch in range(1, epochs + 1):
        log_progress(f"student {spec.name} epoch={epoch} started")
        model.train()
        perm = torch.randperm(x_t.size(0), device=device)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        gradient_norms: list[float] = []
        for start in range(0, x_t.size(0), batch_size):
            idx = perm[start : start + batch_size]
            xb = x_t[idx]
            pb = p_t[idx]
            vb = v_t[idx]
            maskb = legal_mask_t[idx]
            logits, value = model(xb)
            policy_loss = masked_cross_entropy(logits, pb, maskb).mean()
            value_loss = compute_value_loss_vector(
                value,
                vb,
                value_loss=DEFAULT_VALUE_LOSS,
                huber_delta=DEFAULT_HUBER_DELTA,
            ).mean()
            total_loss = policy_loss + (DEFAULT_VALUE_LOSS_WEIGHT * value_loss)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_squared = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_squared += float(
                        torch.sum(parameter.grad.detach() ** 2).item()
                    )
            gradient_norms.append(
                float(math.sqrt(grad_squared)) if grad_squared > 0.0 else 0.0
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            total_losses.append(float(total_loss.detach().cpu().item()))

        epoch_checkpoint = lane_dir / f"checkpoint_epoch{epoch}.npz"
        np.savez(epoch_checkpoint, **checkpoint_from_model(model))
        export_dir = lane_dir / f"artifact_epoch{epoch}"
        export_checkpoint_artifact(
            checkpoint_path=epoch_checkpoint,
            export_dir=export_dir,
            version=f"{spec.name}_epoch{epoch}",
            model_type=spec.model_type,
            input_encoding=input_encoding,
            policy_loss=float(statistics.fmean(policy_losses))
            if policy_losses
            else 0.0,
            value_loss=float(statistics.fmean(value_losses)) if value_losses else 0.0,
        )
        checkpoints.append(
            {
                "epoch": epoch,
                "checkpoint_path": str(epoch_checkpoint),
                "checkpoint_sha256": sha256_file(epoch_checkpoint),
                "artifact_dir": str(export_dir),
                "artifact_weights_sha256": sha256_file(export_dir / "weights.json"),
            }
        )
        probe_metrics = evaluate_probe_candidate(
            artifact_path=export_dir,
            candidate_name=spec.name,
            probe_rows=probe_rows,
        )
        best_probe_metrics = probe_metrics
        training_metrics.append(
            {
                "epoch": epoch,
                "policy_loss": float(statistics.fmean(policy_losses))
                if policy_losses
                else 0.0,
                "value_loss": float(statistics.fmean(value_losses))
                if value_losses
                else 0.0,
                "total_loss": float(statistics.fmean(total_losses))
                if total_losses
                else 0.0,
                "gradient_norm": float(statistics.fmean(gradient_norms))
                if gradient_norms
                else 0.0,
                "probe_top1_agreement": probe_metrics["top1_agreement"],
                "probe_policy_kl": probe_metrics["policy_kl"],
                "probe_value_sign_accuracy": probe_metrics["value_sign_accuracy"],
            }
        )
        log_progress(
            f"student {spec.name} epoch={epoch} policy_loss={training_metrics[-1]['policy_loss']:.4f} "
            f"value_loss={training_metrics[-1]['value_loss']:.4f} "
            f"probe_top1={probe_metrics['top1_agreement']:.4f} "
            f"probe_kl={probe_metrics['policy_kl']:.4f}"
        )
        if (
            epoch < epochs
            and probe_metrics["top1_agreement"] < EARLY_STOP_TOP1_THRESHOLD
        ):
            log_progress(
                f"student {spec.name} early-stopped after epoch={epoch} "
                f"probe_top1={probe_metrics['top1_agreement']:.4f}"
            )
            break

    final_artifact = Path(checkpoints[-1]["artifact_dir"])
    probe_metrics = best_probe_metrics or evaluate_probe_candidate(
        artifact_path=final_artifact,
        candidate_name=spec.name,
        probe_rows=probe_rows,
    )
    return {
        "name": spec.name,
        "kind": "student",
        "architecture": {
            "model_type": spec.model_type,
            "trunk_size": spec.trunk_size,
            "residual_block_count": spec.residual_block_count,
            "hidden_sizes": list(spec.hidden_sizes),
        },
        "training": {
            "epochs": len(training_metrics),
            "batch_size": batch_size,
            "lr": lr,
            "grad_clip": grad_clip,
            "seed": seed,
            "rows": len(train_rows),
        },
        "training_metrics": training_metrics,
        "checkpoints": checkpoints,
        "artifact_dir": str(final_artifact),
        "artifact_weights_sha256": sha256_file(final_artifact / "weights.json"),
        "probe_metrics": probe_metrics,
    }


def evaluate_probe_candidate(
    *, artifact_path: Path, candidate_name: str, probe_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(artifact_path)
    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    legal_mask = np.asarray([row["legal_mask"] for row in probe_rows], dtype=np.float32)
    legal_moves_list = [
        [move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1]
        for row in probe_rows
    ]
    student_policies = []
    student_values = []
    for row in probe_rows:
        game = KalahGame.from_state(row["raw_state"])
        policy, value = evaluator.evaluate(game)
        student_policies.append(
            masked_policy(
                policy.tolist(),
                [move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1],
            )
        )
        student_values.append(float(value))
    del x, legal_mask

    top1_matches = 0
    legal_failures = 0
    kls: list[float] = []
    value_abs_errors: list[float] = []
    sign_matches = 0
    entropies: list[float] = []
    phase_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    seat_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    failure_matches = 0
    failure_total = 0

    for row, legal_moves, student_policy, student_value in zip(
        probe_rows, legal_moves_list, student_policies, student_values, strict=True
    ):
        if any(
            student_policy[move] > 1e-7 for move in range(6) if move not in legal_moves
        ):
            legal_failures += 1
        teacher_top = top_policy_move(row["teacher_puct_policy"], legal_moves)
        student_top = top_policy_move(student_policy, legal_moves)
        if teacher_top == student_top:
            top1_matches += 1
        kls.append(
            kl_divergence(row["teacher_puct_policy"], student_policy, legal_moves)
        )
        target_value = float(row["value_target"])
        value_abs_errors.append(abs(student_value - target_value))
        if (student_value >= 0.0) == (target_value >= 0.0):
            sign_matches += 1
        entropies.append(policy_entropy(student_policy, legal_moves))
        phase_rows[str(row["phase"])].append(
            {"match": 1.0 if teacher_top == student_top else 0.0, "kl": kls[-1]}
        )
        seat_rows[str(row["seat_context"])].append(
            {"match": 1.0 if teacher_top == student_top else 0.0, "kl": kls[-1]}
        )
        if str(row["state_hash"])[:12] in load_failure_state_prefixes(FAILURE_DOC_PATH):
            failure_total += 1
            if teacher_top == student_top:
                failure_matches += 1

    current_baseline_kl = statistics.fmean(
        kl_divergence(
            row["teacher_puct_policy"], row["teacher_raw_policy"], legal_moves
        )
        for row, legal_moves in zip(probe_rows, legal_moves_list, strict=True)
    )
    mean_entropy = float(statistics.fmean(entropies)) if entropies else 0.0
    teacher_mean_entropy = float(
        statistics.fmean(
            policy_entropy(row["teacher_puct_policy"], legal_moves)
            for row, legal_moves in zip(probe_rows, legal_moves_list, strict=True)
        )
    )
    metrics = {
        "candidate": candidate_name,
        "rows": len(probe_rows),
        "top1_agreement": top1_matches / max(len(probe_rows), 1),
        "policy_kl": float(statistics.fmean(kls)) if kls else 0.0,
        "current_raw_policy_baseline_kl": current_baseline_kl,
        "legal_mask_failures": legal_failures,
        "value_mae": float(statistics.fmean(value_abs_errors))
        if value_abs_errors
        else 0.0,
        "value_sign_accuracy": sign_matches / max(len(probe_rows), 1),
        "mean_output_entropy": mean_entropy,
        "teacher_mean_entropy": teacher_mean_entropy,
        "entropy_ratio_vs_teacher": (mean_entropy / teacher_mean_entropy)
        if teacher_mean_entropy > 0.0
        else 0.0,
        "phase_metrics": {
            key: {
                "rows": len(values),
                "top1_agreement": float(statistics.fmean(v["match"] for v in values)),
                "policy_kl": float(statistics.fmean(v["kl"] for v in values)),
            }
            for key, values in sorted(phase_rows.items())
        },
        "seat_metrics": {
            key: {
                "rows": len(values),
                "top1_agreement": float(statistics.fmean(v["match"] for v in values)),
                "policy_kl": float(statistics.fmean(v["kl"] for v in values)),
            }
            for key, values in sorted(seat_rows.items())
        },
        "pr149_top_cost_state_agreement": {
            "rows": failure_total,
            "top1_agreement": failure_matches / max(failure_total, 1),
        },
    }
    metrics["abort_reasons"] = probe_abort_reasons(metrics)
    metrics["aborted"] = bool(metrics["abort_reasons"])
    return metrics


def probe_abort_reasons(metrics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(metrics.get("legal_mask_failures", 0)) > 0:
        reasons.append("legal-mask failures")
    if float(metrics.get("top1_agreement", 0.0)) < 0.55:
        reasons.append("teacher top-1 agreement < 55%")
    if float(metrics.get("policy_kl", 0.0)) > float(
        metrics.get("current_raw_policy_baseline_kl", 0.0)
    ):
        reasons.append("policy KL worse than current raw policy baseline")
    if float(metrics.get("value_sign_accuracy", 0.0)) < 0.55:
        reasons.append("value sign accuracy < 55%")
    entropy_ratio = float(metrics.get("entropy_ratio_vs_teacher", 0.0))
    mean_entropy = float(metrics.get("mean_output_entropy", 0.0))
    if mean_entropy < 0.05 or entropy_ratio < 0.25:
        reasons.append("output entropy collapsed")
    return reasons


def evaluate_current_ref_probe(
    current_artifact: Path, probe_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return evaluate_probe_candidate(
        artifact_path=current_artifact,
        candidate_name="current_ref",
        probe_rows=probe_rows,
    )


def _normalize_candidate_report_label(label: str) -> str:
    normalized = str(label).strip()
    if normalized in {"current", "current_ref"}:
        return "current_ref"
    return normalized


def _matching_candidate_report(
    *,
    report: dict[str, Any],
    candidate_name: str,
    artifact_path: Path,
    current_artifact: Path,
) -> dict[str, Any] | None:
    for lookup_name in (artifact_path.name, candidate_name, "current", "current_ref"):
        candidate_report = find_candidate_report(report, lookup_name)
        if candidate_report is None:
            continue
        candidate_path = str(candidate_report.get("candidate_path", ""))
        candidate_sha = str(candidate_report.get("candidate_sha256", ""))
        if candidate_sha and candidate_sha != sha256_file(
            artifact_path / "weights.json"
        ):
            continue
        if candidate_path and candidate_path not in {
            str(artifact_path),
            str(current_artifact),
        }:
            continue
        return candidate_report
    return None


def discover_cached_benchmark_report(
    *,
    suite_path: Path,
    current_artifact: Path,
    candidate_artifacts: list[tuple[str, Path]],
    seed: int,
) -> tuple[Path, dict[str, Any]] | None:
    tmp_root = Path("/tmp")
    if not tmp_root.is_dir():
        return None
    candidate_sha_by_name = {
        name: sha256_file(path / "weights.json") for name, path in candidate_artifacts
    }
    expected_candidate_names = {name for name, _path in candidate_artifacts}
    expected_current_sha = sha256_file(current_artifact / "weights.json")
    expected_suite_sha = sha256_file(suite_path)
    for report_path in sorted(tmp_root.rglob("temperature_benchmark_report.json")):
        try:
            report = load_json(report_path)
        except Exception:
            continue
        if Path(str(report.get("suite_path", ""))) != suite_path:
            continue
        if str(report.get("suite_sha256", "")) != expected_suite_sha:
            continue
        if str(report.get("root_policy_mode", "")) != "deterministic":
            continue
        if report.get("c_puct_schedule") != schedule_definition(
            default_c_puct=1.25, schedule=parse_cpuct_schedule_json('{"768:768":0.90}')
        ):
            continue
        temperature_reports = report.get("temperature_reports", [])
        if len(temperature_reports) != 1:
            continue
        seed_reports = temperature_reports[0].get("seed_reports", [])
        if len(seed_reports) != 1 or int(seed_reports[0].get("seed", -1)) != int(seed):
            continue
        candidate_reports = seed_reports[0].get("candidate_reports", [])
        normalized_names = {
            _normalize_candidate_report_label(
                str(candidate_report.get("candidate", ""))
            )
            for candidate_report in candidate_reports
        }
        if normalized_names != expected_candidate_names:
            continue
        matched = True
        for candidate_name, artifact_path in candidate_artifacts:
            candidate_report = _matching_candidate_report(
                report=report,
                candidate_name=candidate_name,
                artifact_path=artifact_path,
                current_artifact=current_artifact,
            )
            if candidate_report is None:
                matched = False
                break
            if candidate_name == "current_ref":
                if (
                    str(candidate_report.get("current_sha256", ""))
                    != expected_current_sha
                ):
                    matched = False
                    break
            if (
                str(candidate_report.get("candidate_sha256", ""))
                != candidate_sha_by_name[candidate_name]
            ):
                matched = False
                break
        if matched:
            return report_path, report
    return None


def discover_cached_gate_report(
    *, current_artifact: Path, seed: int
) -> tuple[Path, dict[str, Any]] | None:
    tmp_root = Path("/tmp")
    if not tmp_root.is_dir():
        return None
    current_sha = sha256_file(current_artifact / "weights.json")
    for report_path in sorted(tmp_root.rglob("*.json")):
        if report_path.name not in {
            "gate_report.json",
            "current_ref.json",
            "gate_current_ref.json",
            "balanced_current_ref.json",
        }:
            continue
        try:
            report = load_json(report_path)
        except Exception:
            continue
        if report.get("schema") != "azlite_seat_aware_promotion_gate_v1":
            continue
        if str(report.get("candidate_path", "")) != str(current_artifact):
            continue
        if str(report.get("current_path", "")) != str(current_artifact):
            continue
        if str(report.get("candidate_sha256", "")) != current_sha:
            continue
        if str(report.get("current_sha256", "")) != current_sha:
            continue
        del seed  # deterministic gate reports do not currently persist seed in top-level schema
        return report_path, report
    return None


def run_suite_evaluations(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_artifacts: list[tuple[str, Path]],
    suite_paths: list[Path],
    workers: int,
    seed: int,
) -> dict[str, Any]:
    suite_rows: dict[str, Any] = {}
    candidates_csv = ",".join(str(path) for _name, path in candidate_artifacts)
    for suite_path in suite_paths:
        suite_name = suite_path.stem
        suite_workdir = workdir / "suite_eval" / suite_name
        suite_workdir.mkdir(parents=True, exist_ok=True)
        cached_report = discover_cached_benchmark_report(
            suite_path=suite_path,
            current_artifact=current_artifact,
            candidate_artifacts=candidate_artifacts,
            seed=seed,
        )
        if cached_report is not None:
            _cached_path, report = cached_report
            log_progress(f"reusing suite benchmark suite={suite_name}")
        else:
            log_progress(f"running suite benchmark suite={suite_name}")
            report = run_opening_suite_benchmark(
                workdir=str(suite_workdir),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidates_csv,
                budget_pairs=DEFAULT_SUITE_EVAL_BUDGETS,
                games_per_opening=2,
                seed=seed,
                workers=workers,
                timeout=7200,
            )
        suite_rows[suite_name] = {
            "suite_name": suite_name,
            "suite_path": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "suite_size": count_jsonl_rows(suite_path),
            "candidates": {},
        }
        for candidate_name, artifact_path in candidate_artifacts:
            candidate_report = _matching_candidate_report(
                report=report,
                candidate_name=candidate_name,
                artifact_path=artifact_path,
                current_artifact=current_artifact,
            )
            if candidate_report is None:
                raise RuntimeError(
                    f"missing benchmark report for {candidate_name} in {suite_name}"
                )
            suite_rows[suite_name]["candidates"][candidate_name] = {
                "candidate_path": str(artifact_path),
                "candidate_sha256": candidate_report.get("candidate_sha256"),
                "budget_results": benchmark_budget_results(candidate_report),
            }
    return suite_rows


def aggregate_suite_metrics(
    *, suite_rows: dict[str, Any], candidate_names: list[str]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    heldout_names = [name for name in suite_rows if name.startswith("heldout_seed")]
    for candidate_name in candidate_names:
        candidate_summary: dict[str, Any] = {
            "fixed_large": {},
            "heldout": {},
            "p0_p1_384_256": None,
            "duplicate_trajectory_count": None,
            "runtime_cost": None,
        }
        fixed_large = suite_rows.get("large_eval") or suite_rows.get("large")
        if fixed_large is not None:
            candidate_summary["fixed_large"] = {
                budget: fixed_large["candidates"][candidate_name]["budget_results"][
                    budget
                ]
                for budget in fixed_large["candidates"][candidate_name][
                    "budget_results"
                ]
            }
        for budget in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            EQ_1200_BUDGET,
            ASYM_1200_256_BUDGET,
        ):
            values = [
                float(
                    suite_rows[suite_name]["candidates"][candidate_name][
                        "budget_results"
                    ][budget]["ds"]
                )
                for suite_name in heldout_names
            ]
            candidate_summary["heldout"][budget] = {
                "mean_ds": float(statistics.fmean(values)) if values else 0.0,
                "worst_suite_ds": min(values) if values else 0.0,
            }
        p0_values = []
        p1_values = []
        duplicates = []
        move_time_mean = []
        move_time_p95 = []
        for suite_name in heldout_names:
            budget_result = suite_rows[suite_name]["candidates"][candidate_name][
                "budget_results"
            ][PRIMARY_BUDGET]
            p0_values.append(float(budget_result["p0_score"]))
            p1_values.append(float(budget_result["p1_score"]))
            duplicates.append(
                float(budget_result.get("duplicate_trajectory_count") or 0.0)
            )
            if budget_result.get("move_time_mean_ms") is not None:
                move_time_mean.append(float(budget_result["move_time_mean_ms"]))
            if budget_result.get("move_time_p95_ms") is not None:
                move_time_p95.append(float(budget_result["move_time_p95_ms"]))
        candidate_summary["p0_p1_384_256"] = {
            "mean_p0": float(statistics.fmean(p0_values)) if p0_values else 0.0,
            "mean_p1": float(statistics.fmean(p1_values)) if p1_values else 0.0,
            "gap": (
                float(statistics.fmean(p1_values)) - float(statistics.fmean(p0_values))
            )
            if p0_values and p1_values
            else 0.0,
        }
        candidate_summary["duplicate_trajectory_count"] = (
            float(statistics.fmean(duplicates)) if duplicates else 0.0
        )
        candidate_summary["runtime_cost"] = {
            "mean_move_time_ms": float(statistics.fmean(move_time_mean))
            if move_time_mean
            else None,
            "mean_p95_move_time_ms": float(statistics.fmean(move_time_p95))
            if move_time_p95
            else None,
        }
        summary[candidate_name] = candidate_summary
    return summary


def compute_bootstrap_comparisons(
    *, suite_rows: dict[str, Any], candidate_names: list[str], seed: int
) -> dict[str, dict[str, Any]]:
    heldout_only = {
        name: row for name, row in suite_rows.items() if name.startswith("heldout_seed")
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for candidate_name in candidate_names:
        if candidate_name == "current_ref":
            continue
        for budget in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            EQ_1200_BUDGET,
            ASYM_1200_256_BUDGET,
        ):
            diffs = pooled_per_opening_differences(
                suite_rows=heldout_only,
                candidate_a=candidate_name,
                candidate_b="current_ref",
                budget_pair=budget,
                metric_key="disadvantaged_seat_score",
            )
            comparisons[
                f"{candidate_name}_minus_current_ref_{budget.replace(':', '_')}"
            ] = bootstrap_ci(diffs, seed=seed, samples=DEFAULT_BOOTSTRAP_SAMPLES)
    return comparisons


def maybe_run_gate(
    *,
    workdir: Path,
    current_artifact: Path,
    candidate_name: str,
    candidate_path: Path,
    suite_summary: dict[str, Any],
    workers: int,
    seed: int,
) -> dict[str, Any] | None:
    if candidate_name == "current_ref":
        cached_gate = discover_cached_gate_report(
            current_artifact=current_artifact,
            seed=seed,
        )
        if cached_gate is not None:
            cached_path, report = cached_gate
            log_progress(f"reusing current_ref gate report path={cached_path}")
            return {
                "ran": False,
                "reused_cached_report": True,
                "report_path": str(cached_path),
                "classification": report.get("classification"),
                "preserved": candidate_gate_preserved(report),
            }
        out_path = workdir / "gate" / f"{candidate_name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log_progress("running current_ref gate")
        report = run_default_gate(
            candidate_path=str(candidate_path),
            current_path=str(current_artifact),
            out=str(out_path),
            seed=seed,
            workers=workers,
            games=120,
            budget_pairs=DEFAULT_GATE_BUDGETS,
        )
        return {
            "ran": True,
            "report_path": str(out_path),
            "classification": report.get("classification"),
            "preserved": candidate_gate_preserved(report),
        }

    heldout = suite_summary[candidate_name]["heldout"]
    current_heldout = suite_summary["current_ref"]["heldout"]
    delta_384 = float(heldout[PRIMARY_BUDGET]["mean_ds"]) - float(
        current_heldout[PRIMARY_BUDGET]["mean_ds"]
    )
    delta_768 = float(heldout[EQ_768_BUDGET]["mean_ds"]) - float(
        current_heldout[EQ_768_BUDGET]["mean_ds"]
    )
    delta_1200 = float(heldout[EQ_1200_BUDGET]["mean_ds"]) - float(
        current_heldout[EQ_1200_BUDGET]["mean_ds"]
    )
    delta_1200_256 = float(heldout[ASYM_1200_256_BUDGET]["mean_ds"]) - float(
        current_heldout[ASYM_1200_256_BUDGET]["mean_ds"]
    )
    if not (
        delta_384 >= 0.05
        and delta_768 >= -0.05
        and delta_1200 >= -0.03
        and delta_1200_256 >= -0.03
    ):
        log_progress(
            f"skipping gate for {candidate_name} heldout_deltas="
            f"{delta_384:+.4f},{delta_768:+.4f},{delta_1200:+.4f},{delta_1200_256:+.4f}"
        )
        return {
            "ran": False,
            "reason": "did not clear held-out robustness gate for explicit gate run",
        }
    out_path = workdir / "gate" / f"{candidate_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_progress(f"running gate for candidate={candidate_name}")
    report = run_default_gate(
        candidate_path=str(candidate_path),
        current_path=str(current_artifact),
        out=str(out_path),
        seed=seed,
        workers=workers,
        games=120,
        budget_pairs=DEFAULT_GATE_BUDGETS,
    )
    return {
        "ran": True,
        "report_path": str(out_path),
        "classification": report.get("classification"),
        "preserved": candidate_gate_preserved(report),
    }


def classify_results(
    *,
    candidates: dict[str, dict[str, Any]],
    suite_summary: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    same_size = candidates.get("residual_v3_96x3") or candidates.get(
        "residual_v3_96x3_student"
    )
    current_probe = candidates["current_ref"]["probe_metrics"]
    current_suite_summary = suite_summary.get("current_ref")
    if same_size is not None:
        same_probe = same_size["probe_metrics"]
        same_summary = suite_summary.get(same_size["name"])
        if (
            current_suite_summary is None
            or same_summary is None
            or same_probe["aborted"]
        ):
            labels.append("distillation_pipeline_broken")
        else:
            current_heldout = current_suite_summary["heldout"]
            same_heldout = same_summary["heldout"]
            if same_probe["top1_agreement"] < 0.55 or any(
                float(same_heldout[budget]["mean_ds"])
                - float(current_heldout[budget]["mean_ds"])
                < -0.05
                for budget in (
                    PRIMARY_BUDGET,
                    EQ_768_BUDGET,
                    EQ_1200_BUDGET,
                    ASYM_1200_256_BUDGET,
                )
            ):
                labels.append("distillation_pipeline_broken")
            same_budget_deltas = [
                float(same_heldout[budget]["mean_ds"])
                - float(current_heldout[budget]["mean_ds"])
                for budget in (
                    PRIMARY_BUDGET,
                    EQ_768_BUDGET,
                    EQ_1200_BUDGET,
                    ASYM_1200_256_BUDGET,
                )
            ]
            if min(same_budget_deltas) >= -0.05:
                labels.append("dataset_objective_can_reproduce_current")

    larger_promising = False
    larger_present = False
    for candidate_name, candidate in candidates.items():
        if candidate_name in {
            "current_ref",
            "residual_v3_96x3",
            "residual_v3_96x3_student",
        }:
            continue
        larger_present = True
        probe = candidate["probe_metrics"]
        candidate_summary = suite_summary.get(candidate_name)
        if candidate_summary is None or current_suite_summary is None:
            continue
        heldout = suite_summary[candidate_name]["heldout"]
        current_heldout = current_suite_summary["heldout"]
        delta_384 = float(heldout[PRIMARY_BUDGET]["mean_ds"]) - float(
            current_heldout[PRIMARY_BUDGET]["mean_ds"]
        )
        delta_768 = float(heldout[EQ_768_BUDGET]["mean_ds"]) - float(
            current_heldout[EQ_768_BUDGET]["mean_ds"]
        )
        delta_1200 = float(heldout[EQ_1200_BUDGET]["mean_ds"]) - float(
            current_heldout[EQ_1200_BUDGET]["mean_ds"]
        )
        delta_1200_256 = float(heldout[ASYM_1200_256_BUDGET]["mean_ds"]) - float(
            current_heldout[ASYM_1200_256_BUDGET]["mean_ds"]
        )
        same_probe = (
            same_size["probe_metrics"] if same_size is not None else current_probe
        )
        if (
            not probe["aborted"]
            and delta_384 >= 0.0
            and delta_768 >= -0.05
            and delta_1200 >= -0.03
            and delta_1200_256 >= -0.03
            and probe["top1_agreement"] > same_probe["top1_agreement"]
            and probe["policy_kl"] < same_probe["policy_kl"]
        ):
            larger_promising = True
    if larger_promising:
        labels.append("student_architecture_promising")
    elif larger_present:
        labels.append("architecture_change_not_justified")

    if same_size is not None:
        same_summary = suite_summary.get(same_size["name"])
        if same_summary is None or current_suite_summary is None:
            return labels
        same_heldout = same_summary["heldout"]
        current_heldout = current_suite_summary["heldout"]
        imitated = (
            same_size["probe_metrics"]["top1_agreement"] >= 0.60
            and same_size["probe_metrics"]["policy_kl"]
            <= same_size["probe_metrics"]["current_raw_policy_baseline_kl"]
        )
        reproduced = (
            min(
                float(same_heldout[budget]["mean_ds"])
                - float(current_heldout[budget]["mean_ds"])
                for budget in (
                    PRIMARY_BUDGET,
                    EQ_768_BUDGET,
                    EQ_1200_BUDGET,
                    ASYM_1200_256_BUDGET,
                )
            )
            >= -0.05
        )
        improved = any(
            float(suite_summary[name]["heldout"][PRIMARY_BUDGET]["mean_ds"])
            > float(current_heldout[PRIMARY_BUDGET]["mean_ds"])
            for name in suite_summary
            if name not in {"current_ref", same_size["name"]}
        )
        if imitated and not reproduced and not improved:
            labels.append("teacher_distillation_not_enough")
    return labels


def build_report(
    *,
    summary: dict[str, Any],
    current_hash: str,
    teacher_profile: dict[str, Any],
    dataset_audit: dict[str, Any],
) -> str:
    candidates = summary["candidates"]
    suite_summary = summary["suite_summary"]
    bootstrap = summary["bootstrap_cis"]
    gate = summary["gate_results"]
    current_runtime_cost = suite_summary.get("current_ref", {}).get("runtime_cost", {})
    current_heldout = suite_summary.get("current_ref", {}).get("heldout", {})

    def suite_entry(name: str) -> dict[str, Any]:
        return suite_summary.get(name, {})

    lines = [
        "# AlphaZero-Lite Search-Teacher Student Preflight Results",
        "",
        f"**Classification**: `{','.join(summary['classifications']) or 'unclassified'}`",
        "",
        "## Current Artifact Hash",
        "",
        f"- current weights SHA256: `{current_hash}`",
        "",
        "## Teacher Runtime Profile",
        "",
        f"- profile: `{json.dumps(teacher_profile, sort_keys=True)}`",
        "",
        "## Dataset Audit",
        "",
        f"- row count: `{dataset_audit['row_count']}`",
        f"- unique state count: `{dataset_audit['unique_state_count']}`",
        f"- duplicate state rate: `{dataset_audit['duplicate_state_rate']:.4f}`",
        f"- phase distribution: `{json.dumps(dataset_audit['phase_distribution'], sort_keys=True)}`",
        f"- seat distribution: `{json.dumps(dataset_audit['seat_distribution'], sort_keys=True)}`",
        f"- budget-context distribution: `{json.dumps(dataset_audit['budget_context_distribution'], sort_keys=True)}`",
        f"- teacher raw-vs-PUCT top-1 disagreement rate: `{dataset_audit['teacher_raw_vs_puct_top1_disagreement_rate']:.4f}`",
        f"- overlap with PR #149 failure states: `{json.dumps(dataset_audit['overlap_with_pr149_failure_states'], sort_keys=True)}`",
        "",
        "## Model Architecture Table",
        "",
        markdown_table(
            ["Candidate", "Kind", "Model", "Trunk", "Blocks", "Weights SHA256"],
            [
                [
                    name,
                    entry.get("kind", "reference"),
                    entry.get("architecture", {}).get("model_type", "residual_v3"),
                    entry.get("architecture", {}).get("trunk_size", 96),
                    entry.get("architecture", {}).get("residual_block_count", 3),
                    entry.get("artifact_weights_sha256", "n/a"),
                ]
                for name, entry in candidates.items()
            ],
        ),
        "",
        "## Training Loss Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Epoch",
                "Policy loss",
                "Value loss",
                "Total loss",
                "Grad norm",
            ],
            [
                [
                    name,
                    metric["epoch"],
                    fmt(metric["policy_loss"]),
                    fmt(metric["value_loss"]),
                    fmt(metric["total_loss"]),
                    fmt(metric["gradient_norm"]),
                ]
                for name, entry in candidates.items()
                for metric in entry.get("training_metrics", [])
            ],
        ),
        "",
        "## Probe Imitation Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Top-1",
                "KL",
                "Baseline KL",
                "Legal failures",
                "Value MAE",
                "Sign acc",
                "Entropy",
                "Aborted",
            ],
            [
                [
                    name,
                    fmt(entry["probe_metrics"]["top1_agreement"]),
                    fmt(entry["probe_metrics"]["policy_kl"]),
                    fmt(entry["probe_metrics"]["current_raw_policy_baseline_kl"]),
                    entry["probe_metrics"]["legal_mask_failures"],
                    fmt(entry["probe_metrics"]["value_mae"]),
                    fmt(entry["probe_metrics"]["value_sign_accuracy"]),
                    fmt(entry["probe_metrics"]["mean_output_entropy"]),
                    entry["probe_metrics"]["aborted"],
                ]
                for name, entry in candidates.items()
            ],
        ),
        "",
        "## Aborted-Candidate Table",
        "",
        markdown_table(
            ["Candidate", "Reasons"],
            [
                [name, "; ".join(entry["probe_metrics"].get("abort_reasons", []))]
                for name, entry in candidates.items()
                if entry["probe_metrics"].get("aborted")
            ]
            or [["none", "n/a"]],
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
            [
                [
                    name,
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("384:256", {})
                        .get("ds")
                    ),
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("768:256", {})
                        .get("ds")
                    ),
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("768:768", {})
                        .get("ds")
                    ),
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("1200:1200", {})
                        .get("ds")
                    ),
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("1200:256", {})
                        .get("ds")
                    ),
                    fmt(
                        suite_entry(name)
                        .get("fixed_large", {})
                        .get("256:768", {})
                        .get("ds")
                    ),
                ]
                for name in candidates
            ],
        ),
        "",
        "## Held-Out Mean/Worst-Suite DS Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Held-out mean 384:256",
                "Delta vs current",
                "Held-out worst-suite 384:256",
            ],
            [
                [
                    name,
                    fmt(
                        suite_entry(name)
                        .get("heldout", {})
                        .get(PRIMARY_BUDGET, {})
                        .get("mean_ds")
                    ),
                    fmt(
                        float(
                            suite_entry(name)
                            .get("heldout", {})
                            .get(PRIMARY_BUDGET, {})
                            .get("mean_ds", 0.0)
                        )
                        - float(
                            current_heldout.get(PRIMARY_BUDGET, {}).get("mean_ds", 0.0)
                        )
                        if name in suite_summary and PRIMARY_BUDGET in current_heldout
                        else None
                    ),
                    fmt(
                        suite_entry(name)
                        .get("heldout", {})
                        .get(PRIMARY_BUDGET, {})
                        .get("worst_suite_ds")
                    ),
                ]
                for name in candidates
            ],
        ),
        "",
        "## Bootstrap CI For Candidate Minus Current",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%"],
            [
                [key, fmt(value["mean"]), fmt(value["lower"]), fmt(value["upper"])]
                for key, value in sorted(bootstrap.items())
            ]
            or [["none", "n/a", "n/a", "n/a"]],
        ),
        "",
        "## P0/P1 Split For 384:256",
        "",
        markdown_table(
            ["Candidate", "Mean P0", "Mean P1", "Gap"],
            [
                [
                    name,
                    fmt(suite_entry(name).get("p0_p1_384_256", {}).get("mean_p0")),
                    fmt(suite_entry(name).get("p0_p1_384_256", {}).get("mean_p1")),
                    fmt(suite_entry(name).get("p0_p1_384_256", {}).get("gap")),
                ]
                for name in candidates
            ],
        ),
        "",
        "## Duplicate Trajectory Count",
        "",
        markdown_table(
            ["Candidate", "Mean duplicates"],
            [
                [name, fmt(suite_entry(name).get("duplicate_trajectory_count"))]
                for name in candidates
            ],
        ),
        "",
        "## Runtime Cost Comparison",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean move latency ms",
                "Mean p95 latency ms",
                "Relative slowdown",
            ],
            [
                [
                    name,
                    fmt(
                        suite_entry(name)
                        .get("runtime_cost", {})
                        .get("mean_move_time_ms"),
                        digits=2,
                    ),
                    fmt(
                        suite_entry(name)
                        .get("runtime_cost", {})
                        .get("mean_p95_move_time_ms"),
                        digits=2,
                    ),
                    fmt(
                        (
                            (
                                float(
                                    suite_entry(name)
                                    .get("runtime_cost", {})
                                    .get("mean_move_time_ms")
                                )
                                - float(current_runtime_cost["mean_move_time_ms"])
                            )
                            / max(
                                float(current_runtime_cost["mean_move_time_ms"]), 1e-9
                            )
                        )
                        if suite_entry(name)
                        .get("runtime_cost", {})
                        .get("mean_move_time_ms")
                        is not None
                        and current_runtime_cost.get("mean_move_time_ms") is not None
                        else None,
                        digits=3,
                    ),
                ]
                for name in candidates
            ],
        ),
        "",
        "## Gate Classification If Run",
        "",
    ]
    for name, result in gate.items():
        if result is None:
            lines.append(f"- {name}: `not_run`")
            continue
        if not result.get("ran"):
            lines.append(f"- {name}: `not_run` reason=`{result.get('reason', 'n/a')}`")
            continue
        lines.append(
            f"- {name}: `{result.get('classification')}` preserved=`{result.get('preserved')}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--teacher-default-c-puct", type=float, default=1.25)
    parser.add_argument("--teacher-cpuct-schedule", required=True)
    parser.add_argument("--teacher-tactical-root-bias", type=float, default=0.0)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument(
        "--student-architectures",
        default="residual_v3_96x3,residual_v3_128x4",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument(
        "--max-trajectory-plies", type=int, default=DEFAULT_MAX_TRAJECTORY_PLIES
    )
    parser.add_argument("--prefix-cap", type=int, default=DEFAULT_PREFIX_CAP)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log_progress(f"starting workdir={workdir}")

    current_artifact = Path(args.current)
    current_weights = current_artifact / "weights.json"
    current_metadata = current_artifact / "metadata.json"
    require_existing_file(current_weights, "current artifact weights")
    require_existing_file(current_metadata, "current artifact metadata")
    current_hash_info = verify_expected_hash(
        current_weights,
        args.expected_current_weights_sha256,
        "current artifact weights",
    )
    metadata = load_json(current_metadata)
    input_encoding = str(metadata.get("input_encoding", "kalah_v3"))
    if input_encoding != "kalah_v3":
        raise RuntimeError(
            f"expected current artifact input_encoding kalah_v3, got {input_encoding}"
        )
    architecture = metadata.get("architecture", {})
    if str(architecture.get("model_type", "")) != "residual_v3":
        raise RuntimeError(
            f"expected current artifact model_type residual_v3, got {architecture.get('model_type')}"
        )

    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    require_existing_file(fixed_large_suite, "fixed large suite")
    require_existing_file(medium_suite, "medium suite")
    for path in heldout_suites:
        require_existing_file(path, f"heldout suite {path.name}")

    cpuct_schedule = parse_cpuct_schedule_json(args.teacher_cpuct_schedule)
    teacher_profile = {
        "artifact": str(current_artifact),
        "tactical_root_bias": float(args.teacher_tactical_root_bias),
        "default_c_puct": float(args.teacher_default_c_puct),
        "c_puct_schedule": schedule_definition(
            default_c_puct=float(args.teacher_default_c_puct), schedule=cpuct_schedule
        ),
        "root_policy_mode": "deterministic",
        "root_prior_transform": None,
        "value_transform": None,
    }

    train_rows, probe_rows, dataset_inputs = build_dataset(
        workdir=workdir,
        current_artifact=current_artifact,
        fixed_large_suite=fixed_large_suite,
        heldout_suites=heldout_suites,
        medium_suite=medium_suite,
        default_c_puct=float(args.teacher_default_c_puct),
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=float(args.teacher_tactical_root_bias),
        target_rows=int(args.target_rows),
        seed=int(args.seed),
        max_trajectory_plies=int(args.max_trajectory_plies),
        prefix_cap=int(args.prefix_cap),
    )
    dataset_audit = load_json(workdir / DATASET_AUDIT_FILENAME)
    log_progress(
        f"probe set ready rows={len(probe_rows)} dataset_rows={dataset_audit['row_count']}"
    )

    device = select_device(args.device)
    candidates: dict[str, dict[str, Any]] = {
        "current_ref": {
            "name": "current_ref",
            "kind": "reference",
            "architecture": {
                "model_type": "residual_v3",
                "trunk_size": int(architecture.get("trunk_size", 96)),
                "residual_block_count": int(
                    architecture.get("residual_block_count", 3)
                ),
            },
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": current_hash_info["actual_sha256"],
            "probe_metrics": evaluate_current_ref_probe(current_artifact, probe_rows),
            "training_metrics": [],
            "checkpoints": [],
        }
    }

    for spec in [
        parse_student_architecture(item)
        for item in args.student_architectures.split(",")
        if item.strip()
    ]:
        candidate = train_student_candidate(
            spec=spec,
            train_rows=train_rows,
            probe_rows=probe_rows,
            workdir=workdir,
            input_encoding=input_encoding,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            grad_clip=float(args.grad_clip),
            seed=int(args.seed),
            device=device,
        )
        candidates[spec.name] = candidate

    evaluated_candidates: list[tuple[str, Path]] = [("current_ref", current_artifact)]
    for name, entry in candidates.items():
        if name == "current_ref":
            continue
        if entry["probe_metrics"].get("aborted"):
            log_progress(
                f"candidate {name} aborted after probe reasons={'; '.join(entry['probe_metrics'].get('abort_reasons', []))}"
            )
            continue
        evaluated_candidates.append((name, Path(entry["artifact_dir"])))
    log_progress(
        f"probe stage complete evaluated_candidates={','.join(name for name, _ in evaluated_candidates)}"
    )

    suite_rows: dict[str, Any] = {}
    suite_summary: dict[str, Any] = {}
    bootstrap_cis: dict[str, dict[str, Any]] = {}
    gate_results: dict[str, dict[str, Any] | None] = {}
    if len(evaluated_candidates) > 1:
        suite_paths = [medium_suite, fixed_large_suite, *heldout_suites]
        suite_rows = run_suite_evaluations(
            workdir=workdir,
            current_artifact=current_artifact,
            candidate_artifacts=evaluated_candidates,
            suite_paths=suite_paths,
            workers=int(args.workers),
            seed=int(args.seed),
        )
        suite_summary = aggregate_suite_metrics(
            suite_rows=suite_rows,
            candidate_names=[name for name, _path in evaluated_candidates],
        )
        bootstrap_cis = compute_bootstrap_comparisons(
            suite_rows=suite_rows,
            candidate_names=[name for name, _path in evaluated_candidates],
            seed=int(args.seed),
        )

        for name, path in evaluated_candidates:
            gate_results[name] = maybe_run_gate(
                workdir=workdir,
                current_artifact=current_artifact,
                candidate_name=name,
                candidate_path=path,
                suite_summary=suite_summary,
                workers=int(args.workers),
                seed=int(args.seed),
            )
    else:
        log_progress(
            "skipping suite evaluation and gate because no student survived probe"
        )
        gate_results["current_ref"] = {
            "ran": False,
            "reason": "skipped because no student survived the probe gate",
        }
    for name, entry in candidates.items():
        if name in gate_results:
            continue
        if entry["probe_metrics"].get("aborted"):
            gate_results[name] = {
                "ran": False,
                "reason": "aborted after probe",
            }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": {
            "workdir": str(workdir),
            "current": str(current_artifact),
            "expected_current_weights_sha256": args.expected_current_weights_sha256,
            "medium_suite": str(medium_suite),
            "fixed_large_suite": str(fixed_large_suite),
            "heldout_suites": [str(path) for path in heldout_suites],
            "teacher_runtime_profile": teacher_profile,
            "dataset": dataset_inputs,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "grad_clip": float(args.grad_clip),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "suite_eval_skipped": len(evaluated_candidates) == 1,
        },
        "current_hash": current_hash_info,
        "dataset_audit": dataset_audit,
        "candidates": candidates,
        "suite_rows": suite_rows,
        "suite_summary": suite_summary,
        "bootstrap_cis": bootstrap_cis,
        "gate_results": gate_results,
        "classifications": classify_results(
            candidates=candidates, suite_summary=suite_summary
        ),
    }
    write_json(workdir / SUMMARY_FILENAME, summary)
    report_text = build_report(
        summary=summary,
        current_hash=current_hash_info["actual_sha256"],
        teacher_profile=teacher_profile,
        dataset_audit=dataset_audit,
    )
    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    log_progress(
        f"finished classifications={','.join(summary['classifications'])} summary={workdir / SUMMARY_FILENAME}"
    )
    print(
        json.dumps(
            {
                "summary_path": str(workdir / SUMMARY_FILENAME),
                "report_path": str(REPORT_PATH),
                "classifications": summary["classifications"],
                "evaluated_candidates": [name for name, _path in evaluated_candidates],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
