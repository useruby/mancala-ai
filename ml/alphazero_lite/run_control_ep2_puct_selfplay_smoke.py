#!/usr/bin/env python3
"""Run a control_ep2 PUCT self-play smoke experiment.

Generates a small neural-guided PUCT replay from canonical control_ep2,
audits it, trains policy-head-only continuations, and evaluates them with the
fixed deterministic opening-suite benchmark.

Does not promote and does not overwrite model-artifact/current.
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
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,256:768"
LABEL_TO_BUDGET = {
    "standard": "384:256",
    "challenger_768_vs_256": "768:256",
    "equal_768": "768:768",
    "equal_high": "1200:1200",
    "current_high_asymmetry": "256:768",
}
EXPECTED_CONTROL_CHECKPOINT_SHA256 = (
    "619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9"
)
EXPECTED_CONTROL_ARTIFACT_SHA256 = (
    "34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad"
)
NEAR_ONE_HOT_THRESHOLD = 0.99
KL_EPSILON = 1e-12


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def decode_state(features: list[float]) -> dict[str, Any]:
    return {
        "player_pits": [
            max(0, int(round(float(features[idx]) * 48.0))) for idx in range(6)
        ],
        "opponent_pits": [
            max(0, int(round(float(features[6 + idx]) * 48.0))) for idx in range(6)
        ],
        "player_store": max(0, int(round(float(features[12]) * 48.0))),
        "opponent_store": max(0, int(round(float(features[13]) * 48.0))),
        "current_player": int(round(float(features[14]))),
    }


def canonical_state_key(state: dict[str, Any]) -> str:
    payload = {
        "player_pits": [int(value) for value in state["player_pits"]],
        "opponent_pits": [int(value) for value in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": int(state["current_player"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def phase_for_row(row: dict[str, Any]) -> str:
    state = decode_state(list(row["state"]))
    ply = int(row.get("move_index", 0))
    stones_in_pits = sum(state["player_pits"]) + sum(state["opponent_pits"])
    if ply <= 8:
        return "opening"
    if stones_in_pits <= 16:
        return "late"
    return "mid"


def legal_moves_from_row_state(row: dict[str, Any]) -> list[int]:
    from ml.alphazero_lite.kalah_rules import KalahGame

    state = decode_state(list(row["state"]))
    return [int(move) for move in KalahGame.from_state(state).possible_moves()]


def select_rows_for_game(
    rows: list[dict[str, Any]], *, max_positions_per_game: int, seed: int
) -> list[dict[str, Any]]:
    if len(rows) <= max_positions_per_game:
        return list(rows)

    targets = {
        "opening": max(round(max_positions_per_game * 0.34), 1),
        "mid": max(round(max_positions_per_game * 0.33), 1),
    }
    targets["late"] = max(
        max_positions_per_game - targets["opening"] - targets["mid"], 1
    )
    by_phase: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_phase[phase_for_row(row)].append((index, row))

    rng = random.Random(seed)
    selected_indices: set[int] = set()
    leftovers: list[tuple[int, dict[str, Any]]] = []
    for phase_name in ("opening", "mid", "late"):
        phase_rows = list(by_phase.get(phase_name, []))
        target = int(targets[phase_name])
        if len(phase_rows) <= target:
            selected_indices.update(index for index, _ in phase_rows)
            continue
        sampled = rng.sample(phase_rows, target)
        selected_indices.update(index for index, _ in sampled)
        sampled_index_set = {index for index, _ in sampled}
        leftovers.extend(
            (index, row) for index, row in phase_rows if index not in sampled_index_set
        )

    if len(selected_indices) < max_positions_per_game:
        remaining = max_positions_per_game - len(selected_indices)
        for index, _row in sorted(leftovers, key=lambda item: item[0])[:remaining]:
            selected_indices.add(index)

    return [row for index, row in enumerate(rows) if index in selected_indices]


def cap_replay_rows(
    raw_rows: list[dict[str, Any]], *, max_positions_per_game: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        games[int(row["game_index"])].append(row)

    capped_rows: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    original_counts: list[int] = []
    for game_index in sorted(games):
        rows = games[game_index]
        selected = select_rows_for_game(
            rows,
            max_positions_per_game=max_positions_per_game,
            seed=seed + game_index,
        )
        for row in selected:
            row = dict(row)
            row["selected_positions_per_game_cap"] = int(max_positions_per_game)
            row["selected_from_game_position_count"] = int(len(rows))
            capped_rows.append(row)
        selected_counts.append(len(selected))
        original_counts.append(len(rows))

    return capped_rows, {
        "games": len(games),
        "original_rows": len(raw_rows),
        "capped_rows": len(capped_rows),
        "original_game_position_count_mean": statistics.fmean(original_counts)
        if original_counts
        else 0.0,
        "selected_game_position_count_mean": statistics.fmean(selected_counts)
        if selected_counts
        else 0.0,
    }


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    safe_p = np.clip(p.astype(np.float64), KL_EPSILON, 1.0)
    safe_q = np.clip(q.astype(np.float64), KL_EPSILON, 1.0)
    safe_p /= np.sum(safe_p)
    safe_q /= np.sum(safe_q)
    return float(np.sum(safe_p * np.log(safe_p / safe_q)))


def audit_replay(
    *,
    replay_path: Path,
    requested_games: int,
    control_artifact: Path,
    control_checkpoint: Path,
) -> dict[str, Any]:
    from ml.alphazero_lite.arena import ArtifactEvaluator

    rows = read_jsonl(replay_path)
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        games[int(row.get("game_index", -1))].append(row)

    policy_entropies: list[float] = []
    invalid_legal_mask_rows = 0
    invalid_value_rows = 0
    one_hot_rows = 0
    near_one_hot_rows = 0
    top1_counts: Counter[int] = Counter()
    value_targets: list[float] = []
    outcome_counts: Counter[str] = Counter()
    state_keys: list[str] = []
    root_visit_totals: list[int] = []
    search_profile_hashes: Counter[str] = Counter()

    evaluator = ArtifactEvaluator(control_artifact)
    compare_rows = rows[:1000]
    top1_agreements = 0
    top1_changed = 0
    changed_by_phase: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    kl_search_raw: list[float] = []
    kl_raw_search: list[float] = []

    for row in rows:
        legal_moves = [int(move) for move in row.get("legal_moves") or []]
        derived_legal_moves = legal_moves_from_row_state(row)
        policy = np.asarray(row["policy"], dtype=np.float64)
        phase = phase_for_row(row)
        if legal_moves != derived_legal_moves:
            invalid_legal_mask_rows += 1
        if not np.isclose(float(np.sum(policy)), 1.0, atol=1e-6):
            invalid_legal_mask_rows += 1
        for move in range(6):
            if move not in derived_legal_moves and policy[move] > 1e-6:
                invalid_legal_mask_rows += 1
                break
        policy_entropy = 0.0
        for move in derived_legal_moves:
            probability = float(policy[move])
            if probability > 0.0:
                policy_entropy -= probability * math.log(probability, 2)
        policy_entropies.append(policy_entropy)
        max_probability = float(np.max(policy))
        if (
            np.count_nonzero(np.abs(policy - 1.0) <= 1e-9) == 1
            and np.count_nonzero(policy > 1e-9) == 1
        ):
            one_hot_rows += 1
        if max_probability >= NEAR_ONE_HOT_THRESHOLD:
            near_one_hot_rows += 1
        top1_move = top_policy_move(policy.tolist(), derived_legal_moves)
        if top1_move is not None:
            top1_counts[top1_move] += 1
        value = float(row["value"])
        value_targets.append(value)
        if value < -1.0 or value > 1.0 or not math.isfinite(value):
            invalid_value_rows += 1
        winner = row.get("winner")
        player = int(row.get("player", 0))
        if winner is None:
            outcome_counts["draw"] += 1
        elif int(winner) == player:
            outcome_counts["win"] += 1
        else:
            outcome_counts["loss"] += 1
        state_keys.append(canonical_state_key(decode_state(list(row["state"]))))
        if isinstance(row.get("root_visit_counts"), list):
            root_visit_totals.append(
                sum(int(value) for value in row["root_visit_counts"])
            )
        search_profile_hash = row.get("search_profile_hash")
        if isinstance(search_profile_hash, str) and search_profile_hash:
            search_profile_hashes[search_profile_hash] += 1

    for row in compare_rows:
        from ml.alphazero_lite.kalah_rules import KalahGame

        state = decode_state(list(row["state"]))
        game = KalahGame.from_state(state)
        legal_moves = [int(move) for move in game.possible_moves()]
        raw_policy, _raw_value = evaluator.evaluate(game)
        search_policy = np.asarray(row["policy"], dtype=np.float64)
        raw_top1 = top_policy_move(raw_policy.tolist(), legal_moves)
        search_top1 = top_policy_move(search_policy.tolist(), legal_moves)
        phase = phase_for_row(row)
        phase_counts[phase] += 1
        if raw_top1 == search_top1:
            top1_agreements += 1
        else:
            top1_changed += 1
            changed_by_phase[phase] += 1
        legal_raw = raw_policy[legal_moves]
        legal_search = search_policy[legal_moves]
        kl_search_raw.append(kl_divergence(legal_search, legal_raw))
        kl_raw_search.append(kl_divergence(legal_raw, legal_search))

    unique_state_count = len(set(state_keys))
    duplicate_state_rate = 1.0 - (unique_state_count / max(len(state_keys), 1))
    trajectory_hashes = [
        rows_for_game[0].get("trajectory_hash")
        for rows_for_game in games.values()
        if rows_for_game
    ]
    trajectory_counter = Counter(
        str(value) for value in trajectory_hashes if isinstance(value, str) and value
    )
    duplicate_trajectory_count = sum(
        count for count in trajectory_counter.values() if count > 1
    )
    completed_games = sum(
        1
        for rows_for_game in games.values()
        if rows_for_game and bool(rows_for_game[0].get("game_completed"))
    )
    game_lengths = [
        int(rows_for_game[0].get("game_length", len(rows_for_game)))
        for rows_for_game in games.values()
        if rows_for_game
    ]
    mean_kl_search_raw = statistics.fmean(kl_search_raw) if kl_search_raw else 0.0
    top_move_change_rate = top1_changed / max(len(compare_rows), 1)
    if mean_kl_search_raw >= 0.01 and top_move_change_rate >= 0.10:
        puct_alignment_classification = "puct_policy_improvement_signal"
    elif mean_kl_search_raw < 0.005 or top_move_change_rate < 0.10:
        puct_alignment_classification = "puct_policy_collapse"
    else:
        puct_alignment_classification = "inconclusive"

    audit = {
        "replay_path": str(replay_path),
        "row_count": len(rows),
        "unique_row_count": unique_state_count,
        "games_completed": completed_games,
        "requested_games": requested_games,
        "completed_game_fraction": completed_games / max(requested_games, 1),
        "average_game_length": statistics.fmean(game_lengths) if game_lengths else 0.0,
        "policy_target_entropy": {
            "mean": statistics.fmean(policy_entropies) if policy_entropies else 0.0,
            "p50": percentile(policy_entropies, 50),
            "p90": percentile(policy_entropies, 90),
        },
        "legal_move_mask": {
            "invalid_rows": invalid_legal_mask_rows,
            "valid": invalid_legal_mask_rows == 0,
        },
        "value_target_range": {
            "invalid_rows": invalid_value_rows,
            "valid": invalid_value_rows == 0,
        },
        "one_hot_policy_fraction": one_hot_rows / max(len(rows), 1),
        "near_one_hot_policy_fraction": near_one_hot_rows / max(len(rows), 1),
        "top1_move_distribution_by_pit": {
            str(move): {
                "count": count,
                "fraction": count / max(len(rows), 1),
            }
            for move, count in sorted(top1_counts.items())
        },
        "value_targets": {
            "mean": statistics.fmean(value_targets) if value_targets else 0.0,
            "std": statistics.pstdev(value_targets) if len(value_targets) > 1 else 0.0,
            "win_fraction": outcome_counts["win"] / max(len(rows), 1),
            "loss_fraction": outcome_counts["loss"] / max(len(rows), 1),
            "draw_fraction": outcome_counts["draw"] / max(len(rows), 1),
        },
        "duplicate_state_rate": duplicate_state_rate,
        "duplicate_state_count": len(rows) - unique_state_count,
        "duplicate_trajectory": {
            "available": bool(trajectory_counter),
            "count": duplicate_trajectory_count,
            "rate": duplicate_trajectory_count / max(len(trajectory_counter), 1)
            if trajectory_counter
            else None,
        },
        "root_visit_total": {
            "available": bool(root_visit_totals),
            "mean": statistics.fmean(root_visit_totals) if root_visit_totals else None,
            "p50": percentile([float(value) for value in root_visit_totals], 50),
            "p90": percentile([float(value) for value in root_visit_totals], 90),
        },
        "search_profile_hashes": dict(search_profile_hashes),
        "control_checkpoint_sha256": sha256_file(control_checkpoint),
        "control_artifact_weights_sha256": sha256_file(
            control_artifact / "weights.json"
        ),
        "raw_policy_comparison_first_1000": {
            "rows_compared": len(compare_rows),
            "top1_agreement_rate": top1_agreements / max(len(compare_rows), 1),
            "mean_kl_search_policy_to_raw_policy": mean_kl_search_raw,
            "mean_kl_raw_policy_to_search_policy": statistics.fmean(kl_raw_search)
            if kl_raw_search
            else 0.0,
            "changed_top_move_fraction": top_move_change_rate,
            "changed_top_move_fraction_by_phase": {
                phase: changed_by_phase[phase] / max(phase_counts[phase], 1)
                for phase in ("opening", "mid", "late")
                if phase_counts[phase] > 0
            },
        },
        "puct_alignment_classification": puct_alignment_classification,
    }
    audit["abort_before_training"] = (
        invalid_legal_mask_rows > 0
        or invalid_value_rows > 0
        or audit["completed_game_fraction"] < 0.9
        or unique_state_count < 5000
    )
    return audit


def run_self_play_generation(
    *,
    out_path: Path,
    checkpoint: Path,
    games: int,
    simulations: int,
    c_puct: float,
    workers: int,
    seed: int,
) -> None:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/self_play.py"),
        "--out",
        str(out_path),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--checkpoint",
        str(checkpoint),
        "--input-encoding",
        "kalah_v3",
        "--simulations",
        str(simulations),
        "--c-puct",
        str(c_puct),
        "--workers",
        str(workers),
        "--player-mode",
        "puct",
        "--root-policy-mode",
        "visit_count",
        "--policy-target-mode",
        "sharpened",
        "--value-target-mode",
        "sharpened",
        "--write-root-target-telemetry",
        "--write-game-metadata",
    ]
    print(f"[selfplay] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"self_play failed: {result.stderr[-2000:]}")


def run_train(
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
        _python(),
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
        str(epochs),
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


def export_checkpoint(
    *,
    checkpoint_path: str,
    out_dir: str,
    version: str,
    policy_loss: float,
    value_loss: float,
) -> None:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        checkpoint_path,
        "--out-dir",
        out_dir,
        "--version",
        version,
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--rules-version",
        "kalah_v3",
        "--policy-loss",
        str(policy_loss),
        "--value-loss",
        str(value_loss),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export_artifact failed: {result.stderr[-2000:]}")


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
    print(f"[eval] {' '.join(cmd)}", flush=True)
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


def run_default_gate(
    *,
    candidate_path: str,
    current_path: str,
    out: str,
    seed: int,
    workers: int,
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
        "60",
        "--seed",
        str(seed),
        "--workers",
        str(workers),
        "--budget-pairs",
        DEFAULT_BUDGET_PAIRS,
    ]
    print(f"[gate] {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gate failed: {result.stderr[-2000:]}")
    return json.loads(Path(out).read_text(encoding="utf-8"))


def compute_param_delta_norm(
    npz_path: Path, reference_npz: Path
) -> tuple[float, float]:
    checkpoint = np.load(npz_path)
    reference = np.load(reference_npz)
    total_sq = 0.0
    ref_sq = 0.0
    for key in sorted(reference.files):
        if key not in checkpoint:
            continue
        delta = checkpoint[key] - reference[key]
        total_sq += float((delta**2).sum())
        ref_sq += float((reference[key] ** 2).sum())
    delta_norm = float(total_sq**0.5)
    ref_norm = float(ref_sq**0.5)
    relative_delta_pct = (delta_norm / ref_norm * 100.0) if ref_norm > 0 else 0.0
    return delta_norm, relative_delta_pct


def find_candidate_report(
    report: dict[str, Any], candidate: str
) -> dict[str, Any] | None:
    for temperature_report in report.get("temperature_reports", []):
        for seed_report in temperature_report.get("seed_reports", []):
            for candidate_report in seed_report.get("candidate_reports", []):
                if candidate_report.get("candidate") == candidate:
                    return candidate_report
    return None


def candidate_standard_ds(report: dict[str, Any], candidate: str) -> float:
    candidate_report = find_candidate_report(report, candidate)
    if candidate_report is None:
        return float("-inf")
    standard = candidate_report.get("budget_results", {}).get("standard", {})
    ds = standard.get("ds")
    return float(ds) if ds is not None else float("-inf")


def budget_results_by_pair(
    candidate_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for label, pair in LABEL_TO_BUDGET.items():
        budget_result = candidate_report.get("budget_results", {}).get(label)
        if not budget_result:
            continue
        results[pair] = {
            "ds": budget_result.get("ds"),
            "p0_score": budget_result.get("p0_score"),
            "p1_score": budget_result.get("p1_score"),
            "disadvantaged_seat_score": budget_result.get("disadvantaged_seat_score"),
            "margin_mean": budget_result.get("margin_mean"),
            "margin_median": budget_result.get("margin_median"),
            "duplicate_trajectory_count": budget_result.get(
                "duplicate_trajectory_count"
            ),
            "total_games": budget_result.get("total_games"),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/azlite_control_ep2_puct_smoke")
    parser.add_argument(
        "--control-checkpoint",
        default="/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz",
    )
    parser.add_argument(
        "--control-artifact",
        default="/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--generic-bootstrap",
        default="/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl",
    )
    parser.add_argument(
        "--random-teacher",
        default="/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl",
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--max-positions-per-game", type=int, default=24)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval-medium", action="store_true")
    parser.add_argument("--skip-eval-large", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
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
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    control_checkpoint = Path(args.control_checkpoint)
    control_artifact = Path(args.control_artifact)
    current_artifact = Path(args.current)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    medium_suite = Path(args.medium_suite)
    large_suite = Path(args.large_suite)
    raw_replay_path = workdir / "control_ep2_puct_selfplay_raw.jsonl"
    replay_path = workdir / "control_ep2_puct_selfplay.jsonl"
    audit_path = workdir / "puct_replay_audit.json"

    require_existing_file(control_checkpoint, "control checkpoint")
    require_existing_file(control_artifact / "weights.json", "control artifact weights")
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(generic_bootstrap, "generic bootstrap replay")
    require_existing_file(random_teacher, "random teacher replay")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(large_suite, "large suite")

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
        "current_artifact_weights": build_input_summary(
            current_artifact / "weights.json"
        ),
        "generic_bootstrap": build_input_summary(generic_bootstrap),
        "random_teacher": build_input_summary(random_teacher),
        "medium_suite": build_input_summary(medium_suite),
        "large_suite": build_input_summary(large_suite),
    }

    if not args.skip_generation:
        run_self_play_generation(
            out_path=raw_replay_path,
            checkpoint=control_checkpoint,
            games=args.games,
            simulations=args.simulations,
            c_puct=args.c_puct,
            workers=args.workers,
            seed=args.seed,
        )
        raw_rows = read_jsonl(raw_replay_path)
        capped_rows, cap_summary = cap_replay_rows(
            raw_rows,
            max_positions_per_game=args.max_positions_per_game,
            seed=args.seed,
        )
        write_jsonl(replay_path, capped_rows)
        write_json(workdir / "replay_cap_summary.json", cap_summary)
    else:
        require_existing_file(raw_replay_path, "raw PUCT replay")
        require_existing_file(replay_path, "capped PUCT replay")

    audit = audit_replay(
        replay_path=replay_path,
        requested_games=args.games,
        control_artifact=control_artifact,
        control_checkpoint=control_checkpoint,
    )
    write_json(audit_path, audit)
    if bool(audit.get("abort_before_training")):
        summary = {
            "schema": "azlite_control_ep2_puct_selfplay_smoke_v1",
            "status": "aborted_before_training",
            "workdir": str(workdir),
            "inputs": input_summary,
            "replay_audit": audit,
        }
        write_json(workdir / "summary_metrics.json", summary)
        print(f"[abort] replay audit failed: {audit_path}", flush=True)
        return 1

    lane_specs = [
        {
            "name": "puct_policy_head_e1",
            "epochs": 1,
            "data_files": f"{generic_bootstrap},{random_teacher},{replay_path}",
            "replay_weights": "4,1,1",
        },
        {
            "name": "puct_policy_head_e2",
            "epochs": 2,
            "data_files": f"{generic_bootstrap},{random_teacher},{replay_path}",
            "replay_weights": "4,1,1",
        },
    ]
    lanes: list[dict[str, Any]] = [
        {
            "name": "canonical_ref",
            "report_candidate_name": "control_ep2",
            "epochs": 0,
            "checkpoint_path": str(control_checkpoint),
            "artifact_dir": str(control_artifact),
            "trainable_scope": "none",
            "source": "canonical_ref",
        }
    ]

    for lane_spec in lane_specs:
        lane_dir = workdir / lane_spec["name"]
        lane_dir.mkdir(parents=True, exist_ok=True)
        epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{lane_spec['epochs']}.npz"
        export_dir = lane_dir / f"artifact_{lane_spec['name']}"
        train_metrics: dict[str, Any] | None = None
        if not args.skip_training:
            checkpoint_out = lane_dir / "checkpoint.npz"
            train_metrics = run_train(
                data_files=str(lane_spec["data_files"]),
                replay_weights=str(lane_spec["replay_weights"]),
                init_checkpoint=str(control_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=int(lane_spec["epochs"]),
                seed=args.seed,
            )
            export_checkpoint(
                checkpoint_path=str(epoch_checkpoint_path),
                out_dir=str(export_dir),
                version=str(lane_spec["name"]),
                policy_loss=float(train_metrics.get("policy_loss", 0.0)),
                value_loss=float(train_metrics.get("value_loss", 0.0)),
            )
        else:
            require_existing_file(
                epoch_checkpoint_path, f"checkpoint for {lane_spec['name']}"
            )
            require_existing_file(
                export_dir / "weights.json", f"artifact for {lane_spec['name']}"
            )

        lanes.append(
            {
                "name": lane_spec["name"],
                "report_candidate_name": export_dir.name,
                "epochs": lane_spec["epochs"],
                "checkpoint_path": str(epoch_checkpoint_path),
                "artifact_dir": str(export_dir),
                "trainable_scope": "policy_head",
                "source": "puct_policy_head",
                "replay_weights": lane_spec["replay_weights"],
                "data_files": str(lane_spec["data_files"]).split(","),
                "train_metrics": train_metrics,
            }
        )

    candidate_paths = ",".join(str(lane["artifact_dir"]) for lane in lanes)
    medium_report_path = workdir / "eval_medium" / "temperature_benchmark_report.json"
    if args.skip_eval_medium:
        medium_report = json.loads(medium_report_path.read_text(encoding="utf-8"))
    else:
        medium_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_medium"),
            suite=str(medium_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    large_report_path = workdir / "eval_large" / "temperature_benchmark_report.json"
    if args.skip_eval_large:
        large_report = json.loads(large_report_path.read_text(encoding="utf-8"))
    else:
        large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_large"),
            suite=str(large_suite),
            current=str(current_artifact),
            candidates=candidate_paths,
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )

    canonical_large_384_256 = candidate_standard_ds(large_report, "control_ep2")
    gate_target_names = ["canonical_ref"]
    for lane in lanes:
        if lane["name"] == "canonical_ref":
            continue
        candidate_score = candidate_standard_ds(
            large_report, str(lane["report_candidate_name"])
        )
        if candidate_score >= canonical_large_384_256:
            gate_target_names.append(str(lane["name"]))

    gate_reports: dict[str, dict[str, Any]] = {}
    gate_dir = workdir / "eval_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_gate:
        for lane_name in gate_target_names:
            gate_reports[lane_name] = json.loads(
                (gate_dir / f"{lane_name}_default_gate.json").read_text(
                    encoding="utf-8"
                )
            )
    else:
        for lane in lanes:
            lane_name = str(lane["name"])
            if lane_name not in gate_target_names:
                continue
            gate_reports[lane_name] = run_default_gate(
                candidate_path=str(lane["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_dir / f"{lane_name}_default_gate.json"),
                seed=args.seed,
                workers=args.workers,
            )

    summary_candidates: list[dict[str, Any]] = []
    for lane in lanes:
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, control_checkpoint
        )
        row: dict[str, Any] = {
            "candidate": lane["name"],
            "report_candidate_name": lane["report_candidate_name"],
            "source": lane.get("source"),
            "epochs": lane["epochs"],
            "trainable_scope": lane.get("trainable_scope"),
            "checkpoint_path": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
            "delta_norm_vs_canonical_control_ep2": delta_norm,
            "relative_delta_pct_vs_canonical_control_ep2": relative_delta_pct,
        }
        train_metrics = lane.get("train_metrics")
        if isinstance(train_metrics, dict):
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["validation_loss"] = train_metrics.get("best_val_loss")
            row["training_elapsed_s"] = train_metrics.get("training_elapsed_s")
        medium_candidate_report = find_candidate_report(
            medium_report, str(lane["report_candidate_name"])
        )
        if medium_candidate_report is not None:
            row["medium_budget_results"] = budget_results_by_pair(
                medium_candidate_report
            )
        large_candidate_report = find_candidate_report(
            large_report, str(lane["report_candidate_name"])
        )
        if large_candidate_report is not None:
            row["large_budget_results"] = budget_results_by_pair(large_candidate_report)
        gate_report = gate_reports.get(str(lane["name"]))
        if gate_report is not None:
            row["default_gate"] = {
                "classification": gate_report.get("classification"),
                "standard_alternating_score": gate_report.get(
                    "standard_alternating_score"
                ),
            }
            row["high_search_breakthrough_preserved"] = (
                gate_report.get("classification") == "high_search_breakthrough"
            )
        else:
            row["high_search_breakthrough_preserved"] = None
        summary_candidates.append(row)

    puct_candidates = [
        row
        for row in summary_candidates
        if str(row["candidate"]).startswith("puct_policy_head_")
    ]
    best_puct_candidate = max(
        puct_candidates,
        key=lambda row: float(
            row.get("large_budget_results", {})
            .get("384:256", {})
            .get("ds", float("-inf"))
        ),
    )
    best_large_384_256 = float(
        best_puct_candidate["large_budget_results"]["384:256"]["ds"]
    )
    best_large_1200_1200 = float(
        best_puct_candidate["large_budget_results"]["1200:1200"]["ds"]
    )
    best_large_256_768 = float(
        best_puct_candidate["large_budget_results"]["256:768"]["ds"]
    )
    best_p0 = float(best_puct_candidate["large_budget_results"]["384:256"]["p0_score"])
    best_p1 = float(best_puct_candidate["large_budget_results"]["384:256"]["p1_score"])
    canonical_row = next(
        row for row in summary_candidates if row["candidate"] == "canonical_ref"
    )
    canonical_large_1200_1200 = float(
        canonical_row["large_budget_results"]["1200:1200"]["ds"]
    )
    canonical_large_256_768 = float(
        canonical_row["large_budget_results"]["256:768"]["ds"]
    )
    canonical_p0 = float(canonical_row["large_budget_results"]["384:256"]["p0_score"])
    canonical_p1 = float(canonical_row["large_budget_results"]["384:256"]["p1_score"])
    delta_vs_canonical = best_large_384_256 - canonical_large_384_256
    worsened_p0_p1_imbalance = abs(best_p0 - best_p1) > abs(canonical_p0 - canonical_p1)
    gate_status = best_puct_candidate.get("high_search_breakthrough_preserved")

    if best_large_384_256 >= canonical_large_384_256 and gate_status is True:
        decision = "keep_for_full_network_followup"
    elif (
        delta_vs_canonical >= -0.01
        and (
            best_large_1200_1200 > canonical_large_1200_1200
            or best_large_256_768 > canonical_large_256_768
        )
        and not worsened_p0_p1_imbalance
    ):
        decision = "borderline_rerun_with_larger_puct_replay"
    else:
        decision = "reject_puct_generation_recipe"

    summary = {
        "schema": "azlite_control_ep2_puct_selfplay_smoke_v1",
        "status": "completed",
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "budget_pairs": args.budget_pairs.split(","),
        "games_per_opening": args.games_per_opening,
        "inputs": input_summary,
        "guardrails": {
            "model_type": "residual_v3",
            "input_encoding": "kalah_v3",
            "hidden_sizes": "96,3",
            "trainable_scope": "policy_head",
            "promotion": "disabled",
            "overwrite_current": False,
            "full_network_training": False,
        },
        "replay_generation": {
            "raw_replay_path": str(raw_replay_path),
            "replay_path": str(replay_path),
            "games": args.games,
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "max_positions_per_game": args.max_positions_per_game,
            "player_mode": "puct",
            "root_policy_mode": "visit_count",
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
        },
        "replay_audit": audit,
        "gate_targets": gate_target_names,
        "decision": decision,
        "canonical_large_384_256_ds": canonical_large_384_256,
        "best_puct_candidate": best_puct_candidate["candidate"],
        "best_puct_large_384_256_ds": best_large_384_256,
        "best_puct_delta_vs_canonical_large_384_256_ds": delta_vs_canonical,
        "best_puct_large_1200_1200_ds": best_large_1200_1200,
        "best_puct_large_256_768_ds": best_large_256_768,
        "best_puct_large_384_256_p0_score": best_p0,
        "best_puct_large_384_256_p1_score": best_p1,
        "best_puct_high_search_breakthrough_preserved": gate_status,
        "candidates": summary_candidates,
    }
    summary_path = workdir / "summary_metrics.json"
    write_json(summary_path, summary)
    print(f"[report] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
