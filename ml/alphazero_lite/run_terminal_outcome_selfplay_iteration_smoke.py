#!/usr/bin/env python3
"""Terminal-outcome self-play iteration smoke for AlphaZero-lite.

Generates a small true self-play replay with PUCT visit policies and terminal
outcome value targets, trains conservative current-initialized candidates,
runs probes, conditionally evaluates opening suites, and writes a summary plus
markdown report.

Does not promote and does not overwrite model-artifact/current.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.cpuct_schedule import (  # noqa: E402
    DEFAULT_RUNTIME_C_PUCT,
    parse_cpuct_schedule_json,
    resolve_budget_cpuct,
    schedule_definition,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint  # noqa: E402
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    DEFAULT_BOOTSTRAP_SAMPLES,
    benchmark_budget_results,
    bootstrap_ci,
    pooled_per_opening_differences,
    run_default_gate,
    run_opening_suite_benchmark,
)
from ml.alphazero_lite.run_current_init_policy_only_distillation_preflight import (  # noqa: E402
    evaluate_policy_probe_metrics,
    evaluate_raw_outputs,
    matching_candidate_report,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    build_eval_search_options,
    build_search_profile,
    encode_state,
    outcome_for_player,
    sample_move,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    _count_parameters,
    checkpoint_from_model,
    input_size_for_encoding,
    load_checkpoint_into_model,
    train,
)

VENV_PYTHON = REPO_ROOT / ".venv/bin/python"
SUMMARY_SCHEMA = "azlite_terminal_outcome_selfplay_iteration_smoke_v1"
REPORT_PATH = (
    REPO_ROOT
    / "docs/alphazero-lite-terminal-outcome-selfplay-iteration-smoke-results.md"
)
DEFAULT_BUDGET_PAIRS = (
    "384:256",
    "768:256",
    "768:768",
    "1200:1200",
    "1200:256",
    "256:768",
)
SEARCH_PROBE_BUDGETS = ("384:256", "768:768", "1200:1200", "1200:256")
SELFPLAY_DIAGNOSTIC_FRACTION = 0.25
MIN_COMPLETED_GAMES = 512
TRAIN_FRACTION = 0.85
PROBE_ANCHOR_CAP = 1024
VALUE_PROBE_CAP = 2048
POLICY_CHANGE_LIMITS = {
    "value_head_only": 0.05,
    "policy_value_heads": 0.12,
}
SEARCH_CHANGE_LIMIT = 0.10
ENTROPY_EPSILON = 1e-12
MAX_DUPLICATE_STATE_RATE = 0.98
MIN_CORRELATION_ABS = 0.02
MAX_DRAW_RATE = 0.90
ROOT_VALUE_LR = 3e-5
HEADS_LR = 1e-5
ALL_MODEL_LR = 1e-6
VALUE_LOSS_WEIGHT = 1.0
VALUE_LOSS = "huber"
HUBER_DELTA = 1.0
MODEL_TYPE = "residual_v3"
INPUT_ENCODING = "kalah_v3"
HIDDEN_SIZES = (96, 3)
DOC_TITLE = "# AlphaZero-Lite Terminal-Outcome Self-Play Iteration Smoke Results"
MIN_PRE_SUITE_SIGN_GAIN = 0.01
MIN_PRE_SUITE_CORR_GAIN = 0.01
DRIFT_DIAGNOSTIC_BUDGET = "768:768"
MAX_DRIFT_DIAGNOSTIC_ROWS = 64


@dataclass(frozen=True)
class LaneSpec:
    name: str
    epochs: int
    lr: float
    scope: str
    family: str
    value_loss_weight: float = VALUE_LOSS_WEIGHT
    value_loss: str = VALUE_LOSS
    huber_delta: float = HUBER_DELTA
    optional: bool = False


LANE_SPECS = {
    "value_head_only_e1": LaneSpec(
        name="value_head_only_e1",
        epochs=1,
        lr=1e-4,
        scope="value_head_only",
        family="value_head_only",
        value_loss_weight=1.5,
    ),
    "value_head_only_e2": LaneSpec(
        name="value_head_only_e2",
        epochs=2,
        lr=5e-5,
        scope="value_head_only",
        family="value_head_only",
        value_loss_weight=1.25,
    ),
    "value_head_only_e2_low_lr": LaneSpec(
        name="value_head_only_e2_low_lr",
        epochs=2,
        lr=2.5e-5,
        scope="value_head_only",
        family="value_head_only",
        value_loss_weight=1.0,
    ),
    "policy_value_heads_e1": LaneSpec(
        name="policy_value_heads_e1",
        epochs=1,
        lr=HEADS_LR,
        scope="policy_value_heads",
        family="policy_value_heads",
        value_loss_weight=0.6,
    ),
    "policy_value_heads_e2": LaneSpec(
        name="policy_value_heads_e2",
        epochs=2,
        lr=HEADS_LR,
        scope="policy_value_heads",
        family="policy_value_heads",
        value_loss_weight=0.6,
    ),
    "all_model_low_lr_e1": LaneSpec(
        name="all_model_low_lr_e1",
        epochs=1,
        lr=ALL_MODEL_LR,
        scope="all",
        family="all_model",
        optional=True,
    ),
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_input_summary(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    if path.suffix == ".jsonl":
        payload["rows"] = count_jsonl_rows(path)
    return payload


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:+.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_lane_names(text: str) -> list[str]:
    names = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [name for name in names if name not in LANE_SPECS]
    if invalid:
        raise ValueError(f"unsupported lanes: {invalid}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_terminal_outcome_selfplay_iteration"
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument("--expected-current-weights-sha256", required=True)
    parser.add_argument("--medium-suite", required=True)
    parser.add_argument("--fixed-large-suite", required=True)
    parser.add_argument("--heldout-suites", required=True)
    parser.add_argument("--selfplay-games", type=int, default=1024)
    parser.add_argument("--selfplay-sims", type=int, default=384)
    parser.add_argument("--default-c-puct", type=float, default=DEFAULT_RUNTIME_C_PUCT)
    parser.add_argument("--cpuct-schedule", default='{"768:768":0.90}')
    parser.add_argument("--tactical-root-bias", type=float, default=0.0)
    parser.add_argument(
        "--lanes",
        default="value_head_only_e1,value_head_only_e2,value_head_only_e2_low_lr,policy_value_heads_e1,policy_value_heads_e2",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=14400)
    return parser.parse_args()


def partition_counts(total: int, workers: int) -> list[int]:
    workers = max(1, workers)
    base = total // workers
    remainder = total % workers
    return [
        base + (1 if index < remainder else 0)
        for index in range(workers)
        if base + (1 if index < remainder else 0) > 0
    ]


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


def legal_mask_for_moves(legal_moves: list[int]) -> list[int]:
    mask = [0] * 6
    for move in legal_moves:
        mask[int(move)] = 1
    return mask


def canonical_state_key(state: dict[str, Any]) -> str:
    payload = {
        "player_pits": [int(value) for value in state["player_pits"]],
        "opponent_pits": [int(value) for value in state["opponent_pits"]],
        "player_store": int(state["player_store"]),
        "opponent_store": int(state["opponent_store"]),
        "current_player": int(state["current_player"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def phase_for_state(*, ply: int, state: dict[str, Any]) -> str:
    stones_in_pits = sum(int(value) for value in state["player_pits"]) + sum(
        int(value) for value in state["opponent_pits"]
    )
    if ply <= 8:
        return "opening"
    if stones_in_pits <= 16:
        return "late"
    return "mid"


def policy_entropy(policy: list[float], legal_moves: list[int]) -> float:
    entropy = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return entropy


def safe_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def top_policy_move(policy: list[float], legal_moves: list[int]) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def masked_policy(policy: list[float], legal_moves: list[int]) -> list[float]:
    masked = [0.0] * 6
    total = sum(float(policy[move]) for move in legal_moves)
    if total <= 0.0:
        share = 1.0 / max(len(legal_moves), 1)
        for move in legal_moves:
            masked[move] = share
        return masked
    for move in legal_moves:
        masked[move] = float(policy[move]) / total
    return masked


def split_replay_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_game: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_index"])].append(row)
    game_indexes = sorted(by_game)
    train_game_count = max(1, int(len(game_indexes) * TRAIN_FRACTION))
    train_indexes = set(game_indexes[:train_game_count])
    train_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for game_index in game_indexes:
        target = train_rows if game_index in train_indexes else heldout_rows
        target.extend(by_game[game_index])
    if not heldout_rows and train_rows:
        spill = max(1, len(train_rows) // 10)
        heldout_rows = train_rows[-spill:]
        train_rows = train_rows[:-spill]
    return train_rows, heldout_rows


def sample_probe_rows(
    rows: list[dict[str, Any]], *, seed: int, cap: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["phase"]), str(row["seat_context"]))].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    per_bucket = max(1, cap // max(len(grouped), 1))
    for key in sorted(grouped):
        bucket = list(grouped[key])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bucket])
    if len(selected) < min(cap, len(rows)):
        seen = {id(row) for row in selected}
        remainder = [row for row in rows if id(row) not in seen]
        rng.shuffle(remainder)
        selected.extend(remainder[: max(0, cap - len(selected))])
    return selected[:cap]


def materialize_current_checkpoint(current_artifact: Path, workdir: Path) -> Path:
    checkpoint_path = workdir / "current_init_checkpoint.npz"
    if checkpoint_path.is_file():
        return checkpoint_path
    return materialize_weights_json_checkpoint(
        weights_path=current_artifact / "weights.json", out_path=checkpoint_path
    )


def _run_suite_job(
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
    return run_opening_suite_benchmark(
        workdir=workdir,
        suite=suite,
        current=current,
        candidates=candidates,
        budget_pairs=budget_pairs,
        games_per_opening=games_per_opening,
        seed=seed,
        workers=workers,
        timeout=timeout,
    )


def run_suite_jobs_parallel(
    *, jobs: list[dict[str, Any]], max_parallel_jobs: int
) -> dict[str, dict[str, Any]]:
    if not jobs:
        return {}
    if max_parallel_jobs <= 1 or len(jobs) == 1:
        return {str(job["name"]): _run_suite_job(**job["kwargs"]) for job in jobs}
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_parallel_jobs, len(jobs))
    ) as executor:
        future_map = {
            executor.submit(_run_suite_job, **job["kwargs"]): str(job["name"])
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            results[future_map[future]] = future.result()
    return results


def search_profile_for_selfplay(
    simulations: int, c_puct: float, tactical_root_bias: float
) -> dict[str, Any]:
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    return build_search_profile(
        kind="self_play",
        player_mode="puct",
        simulations=simulations,
        c_puct=c_puct,
        search_options=search_options,
    )


def _generate_replay_chunk(
    *,
    current_artifact: str,
    start_game_index: int,
    games: int,
    primary_sims: int,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(current_artifact))
    rows: list[dict[str, Any]] = []
    runtime_profiles: dict[str, dict[str, Any]] = {}
    completed_games = 0
    diagnostic_games = 0
    for game_index in range(start_game_index, start_game_index + games):
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        rng = random.Random((seed * 1_000_003) + game_index)
        positions: list[dict[str, Any]] = []
        opening_prefix: list[int] = []
        use_diagnostic_lane = (game_index % 4) == 0
        sims = 768 if use_diagnostic_lane else primary_sims
        if use_diagnostic_lane:
            diagnostic_games += 1
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=sims,
            current_simulations=sims,
            default_c_puct=default_c_puct,
        )
        runtime_profile = runtime_profiles.get(f"{sims}:{sims}")
        if runtime_profile is None:
            runtime_profile = search_profile_for_selfplay(
                simulations=sims,
                c_puct=effective_c_puct,
                tactical_root_bias=tactical_root_bias,
            )
            runtime_profiles[f"{sims}:{sims}"] = runtime_profile
        for ply in range(200):
            if game.over():
                break
            state = game.to_state()
            legal_moves = [int(move) for move in game.possible_moves()]
            if not legal_moves:
                break
            search_result = evaluate_artifact_position(
                evaluator=evaluator,
                state=state,
                simulations=sims,
                seed=(seed * 10_007) + (game_index * 257) + ply,
                c_puct=effective_c_puct,
                search_options=build_eval_search_options(
                    root_policy_mode="deterministic",
                    tactical_root_bias=tactical_root_bias,
                ),
                ablation_mode="full",
            )
            visit_policy = masked_policy(list(search_result["visits"]), legal_moves)
            root_value = search_result.get("search_root_value")
            if root_value is None:
                root_value = search_result.get("value", 0.0)
            temperature = 0.67 if ply < 8 else 0.0
            if temperature > 0.0:
                scaled = np.asarray(visit_policy, dtype=np.float64)
                powered = np.zeros_like(scaled)
                for move in legal_moves:
                    powered[move] = float(scaled[move]) ** (1.0 / temperature)
                total = float(np.sum(powered[legal_moves]))
                sampling_policy = (
                    visit_policy
                    if total <= 0.0
                    else masked_policy(powered.tolist(), legal_moves)
                )
                selected_move = int(
                    sample_move(sampling_policy, legal_moves=legal_moves, rng=rng)
                )
            else:
                selected_move = int(
                    top_policy_move(visit_policy, legal_moves) or legal_moves[0]
                )
            encoded_state = encode_state(state, input_encoding=INPUT_ENCODING)
            positions.append(
                {
                    "state": encoded_state,
                    "raw_state": state,
                    "legal_moves": legal_moves,
                    "legal_mask": legal_mask_for_moves(legal_moves),
                    "policy": [float(value) for value in visit_policy],
                    "selected_move": selected_move,
                    "root_value": float(root_value),
                    "search_root_value": float(
                        search_result.get("search_root_value", root_value)
                    ),
                    "ply": ply,
                    "phase": phase_for_state(ply=ply, state=state),
                    "player": int(state["current_player"]),
                    "seat": int(state["current_player"]),
                    "seat_context": f"player_{int(state['current_player'])}",
                    "opening_prefix": list(opening_prefix),
                    "effective_c_puct": float(effective_c_puct),
                    "simulations": sims,
                    "search_profile": runtime_profile,
                    "search_profile_hash": str(runtime_profile["hash"]),
                    "root_entropy": policy_entropy(visit_policy, legal_moves),
                }
            )
            if not game.move(game.pit_index(selected_move)):
                break
            opening_prefix.append(selected_move)
        if game.over():
            completed_games += 1
        winner = game.winner
        final_state = game.to_state()
        final_margin_p0 = int(final_state["player_store"]) - int(
            final_state["opponent_store"]
        )
        game_length = len(positions)
        trajectory_hash = hashlib.sha256(
            json.dumps(
                {"states": [row["state"] for row in positions], "winner": winner},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        for row in positions:
            player = int(row["player"])
            final_outcome = outcome_for_player(winner, player)
            signed_margin = final_margin_p0 if player == 0 else -final_margin_p0
            rows.append(
                {
                    "state": row["state"],
                    "policy": row["policy"],
                    "value": float(final_outcome),
                    "player": player,
                    "seat": int(row["seat"]),
                    "seat_context": str(row["seat_context"]),
                    "legal_moves": row["legal_moves"],
                    "legal_mask": row["legal_mask"],
                    "selected_move": int(row["selected_move"]),
                    "root_value": float(row["root_value"]),
                    "search_root_value": float(row["search_root_value"]),
                    "final_outcome": float(final_outcome),
                    "final_margin": int(signed_margin),
                    "final_stores": {
                        "player_0": int(final_state["player_store"]),
                        "player_1": int(final_state["opponent_store"]),
                    },
                    "ply": int(row["ply"]),
                    "move_index": int(row["ply"]),
                    "phase": str(row["phase"]),
                    "opening_prefix": row["opening_prefix"],
                    "effective_c_puct": float(row["effective_c_puct"]),
                    "simulations": int(row["simulations"]),
                    "search_profile": row["search_profile"],
                    "search_profile_hash": row["search_profile_hash"],
                    "game_index": int(game_index),
                    "game_length": int(game_length),
                    "game_completed": bool(game.over()),
                    "winner": winner,
                    "trajectory_hash": trajectory_hash,
                    "raw_state": row["raw_state"],
                    "root_entropy": float(row["root_entropy"]),
                }
            )
    return rows, {
        "completed_games": completed_games,
        "diagnostic_lane_games": diagnostic_games,
        "runtime_profiles": runtime_profiles,
    }


def _evaluate_search_outputs_chunk(
    *,
    artifact_path_text: str,
    chunk_rows: list[dict[str, Any]],
    row_offset: int,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path_text))
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    chunk_outputs: list[dict[str, Any]] = []
    for index, row in enumerate(chunk_rows):
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=int(row["challenger_simulations"]),
            current_simulations=int(row["current_simulations"]),
            default_c_puct=default_c_puct,
        )
        result = evaluate_artifact_position(
            evaluator=evaluator,
            state=row["raw_state"],
            simulations=int(row["challenger_simulations"]),
            seed=seed + row_offset + index,
            c_puct=effective_c_puct,
            search_options=search_options,
            ablation_mode="full",
        )
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        search_policy = masked_policy(list(result["visits"]), legal_moves)
        selected_move = int(result["selected_move"])
        selected_visit_share = (
            float(search_policy[selected_move]) if selected_move in legal_moves else 0.0
        )
        chunk_outputs.append(
            {
                "selected_move": selected_move,
                "search_policy": search_policy,
                "selected_visit_share": selected_visit_share,
                "effective_c_puct": effective_c_puct,
                "root_value": float(
                    result.get("search_root_value", result.get("value", 0.0))
                ),
            }
        )
    return chunk_outputs


def generate_replay(
    *,
    current_artifact: Path,
    out_path: Path,
    games: int,
    primary_sims: int,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.time()
    chunk_counts = partition_counts(games, workers)
    rows: list[dict[str, Any]] = []
    completed_games = 0
    diagnostic_games = 0
    runtime_profiles: dict[str, dict[str, Any]] = {}
    start_game_index = 0
    if len(chunk_counts) == 1:
        shard_rows, shard_meta = _generate_replay_chunk(
            current_artifact=str(current_artifact),
            start_game_index=0,
            games=chunk_counts[0],
            primary_sims=primary_sims,
            default_c_puct=default_c_puct,
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=tactical_root_bias,
            seed=seed,
        )
        rows.extend(shard_rows)
        completed_games += int(shard_meta["completed_games"])
        diagnostic_games += int(shard_meta["diagnostic_lane_games"])
        runtime_profiles.update(shard_meta["runtime_profiles"])
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(chunk_counts)
        ) as executor:
            futures = []
            for chunk_games in chunk_counts:
                futures.append(
                    executor.submit(
                        _generate_replay_chunk,
                        current_artifact=str(current_artifact),
                        start_game_index=start_game_index,
                        games=chunk_games,
                        primary_sims=primary_sims,
                        default_c_puct=default_c_puct,
                        cpuct_schedule=cpuct_schedule,
                        tactical_root_bias=tactical_root_bias,
                        seed=seed,
                    )
                )
                start_game_index += chunk_games
            for future in concurrent.futures.as_completed(futures):
                shard_rows, shard_meta = future.result()
                rows.extend(shard_rows)
                completed_games += int(shard_meta["completed_games"])
                diagnostic_games += int(shard_meta["diagnostic_lane_games"])
                runtime_profiles.update(shard_meta["runtime_profiles"])
    rows.sort(key=lambda row: (int(row["game_index"]), int(row["ply"])))
    write_jsonl(out_path, rows)
    return rows, {
        "requested_games": int(games),
        "completed_games": int(completed_games),
        "diagnostic_lane_games": int(diagnostic_games),
        "generation_elapsed_s": time.time() - start,
        "runtime_profiles": runtime_profiles,
    }


def replay_audit(
    rows: list[dict[str, Any]], generation_meta: dict[str, Any]
) -> dict[str, Any]:
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    state_counts: Counter[str] = Counter()
    trajectory_counts: Counter[str] = Counter()
    illegal_policy_count = 0
    invalid_legal_mask_count = 0
    policy_entropies: list[float] = []
    value_targets: list[float] = []
    root_values: list[float] = []
    terminal_outcomes: list[float] = []
    phase_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    top_move_counts: Counter[str] = Counter()
    outcome_buckets: Counter[str] = Counter()
    margins: list[int] = []
    for row in rows:
        games[int(row["game_index"])].append(row)
        legal_moves = [int(move) for move in row.get("legal_moves", [])]
        legal_mask = list(row.get("legal_mask", []))
        policy = list(row["policy"])
        state_key = canonical_state_key(row["raw_state"])
        state_counts[state_key] += 1
        trajectory_counts[str(row["trajectory_hash"])] += 1
        if len(legal_mask) != 6 or any(flag not in (0, 1) for flag in legal_mask):
            invalid_legal_mask_count += 1
        else:
            expected_mask = legal_mask_for_moves(legal_moves)
            if legal_mask != expected_mask:
                invalid_legal_mask_count += 1
        if len(policy) != 6 or not np.isclose(sum(policy), 1.0, atol=1e-6):
            illegal_policy_count += 1
        else:
            for move in range(6):
                if move not in legal_moves and float(policy[move]) > 1e-6:
                    illegal_policy_count += 1
                    break
        policy_entropies.append(policy_entropy(policy, legal_moves))
        value = float(row["value"])
        root_value = float(row["root_value"])
        outcome = float(row["final_outcome"])
        value_targets.append(value)
        root_values.append(root_value)
        terminal_outcomes.append(outcome)
        phase_counts[str(row["phase"])] += 1
        seat_counts[str(row["seat_context"])] += 1
        top_move = top_policy_move(policy, legal_moves)
        if top_move is not None:
            top_move_counts[str(top_move)] += 1
        margins.append(int(row["final_margin"]))
        if outcome > 0.0:
            outcome_buckets["win"] += 1
        elif outcome < 0.0:
            outcome_buckets["loss"] += 1
        else:
            outcome_buckets["draw"] += 1
    unique_states = len(state_counts)
    duplicate_state_rate = 0.0 if not rows else 1.0 - (unique_states / len(rows))
    duplicate_trajectory_count = sum(
        count for count in trajectory_counts.values() if count > 1
    )
    completed_games = sum(
        1
        for game_rows in games.values()
        if game_rows and bool(game_rows[0]["game_completed"])
    )
    avg_game_length = (
        statistics.fmean(len(game_rows) for game_rows in games.values())
        if games
        else 0.0
    )
    root_value_corr = safe_correlation(root_values, terminal_outcomes)
    draw_rate = outcome_buckets["draw"] / max(len(rows), 1)
    extreme_duplicates = duplicate_state_rate > MAX_DUPLICATE_STATE_RATE
    uncorrelated = abs(root_value_corr) < MIN_CORRELATION_ABS
    abort_reasons: list[str] = []
    if completed_games < MIN_COMPLETED_GAMES:
        abort_reasons.append(f"completed games < {MIN_COMPLETED_GAMES}")
    if illegal_policy_count > 0:
        abort_reasons.append("invalid policies > 0")
    if invalid_legal_mask_count > 0:
        abort_reasons.append("invalid legal mask count > 0")
    if extreme_duplicates:
        abort_reasons.append("duplicate state rate extreme")
    if draw_rate > MAX_DRAW_RATE:
        abort_reasons.append("terminal outcomes >90% draws")
    if uncorrelated:
        abort_reasons.append("root value and terminal outcome are uncorrelated")
    return {
        "schema": "azlite_terminal_outcome_replay_audit_v1",
        "generation": generation_meta,
        "games_completed": completed_games,
        "rows": len(rows),
        "unique_states": unique_states,
        "duplicate_state_rate": duplicate_state_rate,
        "duplicate_trajectory_count": duplicate_trajectory_count,
        "illegal_policy_count": illegal_policy_count,
        "invalid_legal_mask_count": invalid_legal_mask_count,
        "policy_entropy_distribution": {
            "mean": statistics.fmean(policy_entropies) if policy_entropies else 0.0,
            "p25": percentile(policy_entropies, 25),
            "p50": percentile(policy_entropies, 50),
            "p75": percentile(policy_entropies, 75),
            "p90": percentile(policy_entropies, 90),
            "min": min(policy_entropies) if policy_entropies else 0.0,
            "max": max(policy_entropies) if policy_entropies else 0.0,
        },
        "outcome_distribution": dict(sorted(outcome_buckets.items())),
        "value_target_distribution": {
            "mean": statistics.fmean(value_targets) if value_targets else 0.0,
            "p25": percentile(value_targets, 25),
            "p50": percentile(value_targets, 50),
            "p75": percentile(value_targets, 75),
            "negative": sum(1 for value in value_targets if value < 0),
            "positive": sum(1 for value in value_targets if value > 0),
            "zero": sum(1 for value in value_targets if value == 0),
        },
        "root_value_vs_terminal_outcome": {
            "correlation": root_value_corr,
            "root_value_mean": statistics.fmean(root_values) if root_values else 0.0,
            "outcome_mean": statistics.fmean(terminal_outcomes)
            if terminal_outcomes
            else 0.0,
        },
        "phase_distribution": dict(sorted(phase_counts.items())),
        "seat_distribution": dict(sorted(seat_counts.items())),
        "top_move_distribution": dict(sorted(top_move_counts.items())),
        "average_game_length": avg_game_length,
        "final_margin_distribution": {
            "mean": statistics.fmean(margins) if margins else 0.0,
            "p50": percentile([float(value) for value in margins], 50),
            "p90": percentile([float(value) for value in margins], 90),
        },
        "resign_or_early_stop": {
            "resign_supported": False,
            "incomplete_games": len(games) - completed_games,
        },
        "abort_before_training": bool(abort_reasons),
        "abort_reasons": abort_reasons,
    }


def export_checkpoint(
    *,
    checkpoint_path: Path,
    out_dir: Path,
    version: str,
    policy_loss: float,
    value_loss: float,
) -> None:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--version",
        version,
        "--model-type",
        MODEL_TYPE,
        "--input-encoding",
        INPUT_ENCODING,
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"export_artifact failed: {result.stderr[-2000:]}")


def apply_custom_scope(model: PolicyValueNet, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    if scope == "value_head_only":
        assert model.value_hidden_layer is not None
        model.value_hidden_layer.weight.requires_grad = True
        model.value_hidden_layer.bias.requires_grad = True
        model.value_head.weight.requires_grad = True
        model.value_head.bias.requires_grad = True
        return
    if scope == "policy_value_heads":
        assert model.policy_hidden_layer is not None
        assert model.value_hidden_layer is not None
        model.policy_hidden_layer.weight.requires_grad = True
        model.policy_hidden_layer.bias.requires_grad = True
        model.policy_head.weight.requires_grad = True
        model.policy_head.bias.requires_grad = True
        model.value_hidden_layer.weight.requires_grad = True
        model.value_hidden_layer.bias.requires_grad = True
        model.value_head.weight.requires_grad = True
        model.value_head.bias.requires_grad = True
        return
    raise ValueError(f"unsupported custom scope: {scope}")


def save_checkpoint(path: Path, model: PolicyValueNet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_from_model(model)
    np.savez(path, **checkpoint)


def train_candidate(
    *,
    lane_spec: LaneSpec,
    train_rows: list[dict[str, Any]],
    init_checkpoint: Path,
    lane_dir: Path,
    seed: int,
) -> dict[str, Any]:
    replay_path = lane_dir / "train_replay.jsonl"
    write_jsonl(replay_path, train_rows)
    from ml.alphazero_lite.train import load_jsonl

    x, p_target, v_target = load_jsonl(
        replay_path,
        policy_target_mode="default",
        value_target_mode="default",
    )
    model = PolicyValueNet(
        hidden_sizes=HIDDEN_SIZES,
        model_type=MODEL_TYPE,
        input_size=input_size_for_encoding(INPUT_ENCODING),
    )
    load_checkpoint_into_model(model, init_checkpoint)
    apply_custom_scope(model, lane_spec.scope)
    total_params, trainable_params = _count_parameters(model)
    checkpoint_path = lane_dir / f"checkpoint_epoch{lane_spec.epochs}.npz"
    start = time.time()
    policy_loss, value_loss, best_val_loss = train(
        model,
        x,
        p_target,
        v_target,
        epochs=lane_spec.epochs,
        batch_size=512,
        lr=lane_spec.lr,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        value_loss_weight=lane_spec.value_loss_weight,
        value_loss=lane_spec.value_loss,
        huber_delta=lane_spec.huber_delta,
        val_split=0.1,
        grad_clip=1.0,
        save_top_k=0,
        lr_scheduler="none",
    )
    elapsed = time.time() - start
    save_checkpoint(checkpoint_path, model)
    artifact_dir = lane_dir / f"artifact_{lane_spec.name}"
    export_checkpoint(
        checkpoint_path=checkpoint_path,
        out_dir=artifact_dir,
        version=lane_spec.name,
        policy_loss=float(policy_loss),
        value_loss=float(value_loss),
    )
    return {
        "name": lane_spec.name,
        "report_candidate_name": artifact_dir.name,
        "epochs": lane_spec.epochs,
        "trainable_scope": lane_spec.scope,
        "family": lane_spec.family,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
        "training_metrics": {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "best_val_loss": float(best_val_loss),
            "total_loss": float(
                (model.last_train_metrics or {}).get(
                    "total_loss",
                    float(policy_loss) + (VALUE_LOSS_WEIGHT * float(value_loss)),
                )
            ),
            "grad_norm": (model.last_train_metrics or {}).get("grad_norm"),
            "training_elapsed_s": elapsed,
            "lr": lane_spec.lr,
            "value_loss_weight": lane_spec.value_loss_weight,
            "value_loss_kind": lane_spec.value_loss,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "total_params": total_params,
        },
    }


def checkpoint_key_families(arrays: dict[str, np.ndarray]) -> dict[str, list[str]]:
    trunk = sorted(
        key
        for key in arrays
        if key.startswith("w_input")
        or key.startswith("b_input")
        or key.startswith("w_residual_")
        or key.startswith("b_residual_")
    )
    policy = sorted(
        key
        for key in arrays
        if key in {"w_policy_hidden", "b_policy_hidden", "w_policy", "b_policy"}
    )
    value = sorted(
        key
        for key in arrays
        if key in {"w_value_hidden", "b_value_hidden", "w_value", "b_value"}
    )
    other = sorted(key for key in arrays if key not in set(trunk + policy + value))
    return {"trunk": trunk, "policy": policy, "value": value, "other": other}


def load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def changed_key_audit(
    *,
    reference_checkpoint: Path,
    candidate_checkpoint: Path,
    allowed_families: set[str],
) -> dict[str, Any]:
    reference = load_checkpoint_arrays(reference_checkpoint)
    candidate = load_checkpoint_arrays(candidate_checkpoint)
    families = checkpoint_key_families(reference)
    changed: dict[str, list[str]] = {
        "trunk": [],
        "policy": [],
        "value": [],
        "other": [],
    }
    max_abs_diff = 0.0
    for family_name, keys in families.items():
        for key in keys:
            diff = float(np.max(np.abs(candidate[key] - reference[key])))
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 0.0:
                changed[family_name].append(key)
    disallowed = {
        family
        for family, keys in changed.items()
        if keys and family not in allowed_families
    }
    return {
        "passes": not disallowed,
        "max_abs_diff": max_abs_diff,
        "changed_keys": changed,
        "allowed_families": sorted(allowed_families),
        "disallowed_changed_families": sorted(disallowed),
    }


def build_policy_probe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probe_rows: list[dict[str, Any]] = []
    for row in rows:
        probe_rows.append(
            {
                "raw_state": row["raw_state"],
                "legal_mask": row["legal_mask"],
                "teacher_puct_policy": row["policy"],
                "budget_context": "selfplay_anchor",
                "phase": row["phase"],
                "seat_context": row["seat_context"],
            }
        )
    return probe_rows


def build_search_probe_rows(
    rows: list[dict[str, Any]], budgets: tuple[str, ...]
) -> list[dict[str, Any]]:
    probe_rows: list[dict[str, Any]] = []
    for row in rows:
        state_hash = canonical_state_key(row["raw_state"])
        for budget_pair in budgets:
            challenger_sims, current_sims = (
                int(part) for part in budget_pair.split(":", 1)
            )
            probe_rows.append(
                {
                    "state_hash": state_hash,
                    "raw_state": row["raw_state"],
                    "legal_mask": row["legal_mask"],
                    "budget_context": budget_pair,
                    "seat_context": row["seat_context"],
                    "phase": row["phase"],
                    "opening_prefix": list(row.get("opening_prefix", [])),
                    "game_index": int(row.get("game_index", -1)),
                    "ply": int(row.get("ply", 0)),
                    "challenger_simulations": challenger_sims,
                    "current_simulations": current_sims,
                    "teacher_selected_move": row["selected_move"],
                    "teacher_policy": row["policy"],
                }
            )
    return probe_rows


def evaluate_search_outputs(
    *,
    artifact_path: Path,
    rows: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1 or len(rows) <= 1:
        return _evaluate_search_outputs_chunk(
            artifact_path_text=str(artifact_path),
            chunk_rows=rows,
            row_offset=0,
            default_c_puct=default_c_puct,
            cpuct_schedule=cpuct_schedule,
            tactical_root_bias=tactical_root_bias,
            seed=seed,
        )

    chunk_counts = partition_counts(len(rows), workers)
    outputs: list[dict[str, Any]] = []
    start_index = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(chunk_counts)
    ) as executor:
        futures = []
        for chunk_size in chunk_counts:
            chunk_rows = rows[start_index : start_index + chunk_size]
            futures.append(
                executor.submit(
                    _evaluate_search_outputs_chunk,
                    artifact_path_text=str(artifact_path),
                    chunk_rows=chunk_rows,
                    row_offset=start_index,
                    default_c_puct=default_c_puct,
                    cpuct_schedule=cpuct_schedule,
                    tactical_root_bias=tactical_root_bias,
                    seed=seed,
                )
            )
            start_index += chunk_size
        for future in futures:
            outputs.extend(future.result())
    return outputs


def evaluate_search_probe_metrics(
    *,
    rows: list[dict[str, Any]],
    candidate_outputs: list[dict[str, Any]],
    current_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    changed_rates: list[float] = []
    visit_kls: list[float] = []
    selected_share_deltas: list[float] = []
    teacher_agreements: list[float] = []
    root_value_abs_deltas: list[float] = []
    changed_root_value_abs_deltas: list[float] = []
    by_budget: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    p0_p1_384: dict[str, list[float]] = defaultdict(list)
    for row, candidate, current in zip(
        rows, candidate_outputs, current_outputs, strict=True
    ):
        legal_moves = [
            move for move, flag in enumerate(row["legal_mask"]) if int(flag) == 1
        ]
        candidate_policy = np.asarray(candidate["search_policy"], dtype=np.float64)
        current_policy = np.asarray(current["search_policy"], dtype=np.float64)
        changed = (
            1.0
            if int(candidate["selected_move"]) != int(current["selected_move"])
            else 0.0
        )
        teacher_agree = (
            1.0
            if int(candidate["selected_move"]) == int(row["teacher_selected_move"])
            else 0.0
        )
        kl = float(
            np.sum(
                np.clip(current_policy[legal_moves], 1e-8, 1.0)
                * np.log(
                    np.clip(current_policy[legal_moves], 1e-8, 1.0)
                    / np.clip(candidate_policy[legal_moves], 1e-8, 1.0)
                )
            )
        )
        changed_rates.append(changed)
        teacher_agreements.append(teacher_agree)
        visit_kls.append(kl)
        root_value_abs_delta = abs(
            float(candidate.get("root_value", 0.0))
            - float(current.get("root_value", 0.0))
        )
        root_value_abs_deltas.append(root_value_abs_delta)
        if changed > 0.0:
            changed_root_value_abs_deltas.append(root_value_abs_delta)
        selected_share_deltas.append(
            float(candidate["selected_visit_share"])
            - float(current["selected_visit_share"])
        )
        budget = str(row["budget_context"])
        by_budget[budget]["changed"].append(changed)
        by_budget[budget]["teacher"].append(teacher_agree)
        by_budget[budget]["visit_kl"].append(kl)
        by_budget[budget]["root_value_abs_delta"].append(root_value_abs_delta)
        if changed > 0.0:
            by_budget[budget]["changed_root_value_abs_delta"].append(
                root_value_abs_delta
            )
        if budget == "384:256":
            p0_p1_384[str(row["seat_context"])].append(changed)
    return {
        "rows": len(rows),
        "search_selected_move_changed_rate_vs_current": statistics.fmean(changed_rates)
        if changed_rates
        else 0.0,
        "search_selected_move_agreement_with_replay": statistics.fmean(
            teacher_agreements
        )
        if teacher_agreements
        else 0.0,
        "root_visit_kl_vs_current_ref": statistics.fmean(visit_kls)
        if visit_kls
        else 0.0,
        "root_value_abs_delta_vs_current_ref": statistics.fmean(root_value_abs_deltas)
        if root_value_abs_deltas
        else 0.0,
        "changed_row_root_value_abs_delta_vs_current_ref": statistics.fmean(
            changed_root_value_abs_deltas
        )
        if changed_root_value_abs_deltas
        else 0.0,
        "root_selected_visit_share_delta_vs_current_ref": statistics.fmean(
            selected_share_deltas
        )
        if selected_share_deltas
        else 0.0,
        "changed_move_rate_by_budget_context": {
            budget: {
                "rows": len(values["changed"]),
                "changed_rate_vs_current": statistics.fmean(values["changed"]),
                "search_agreement_with_replay": statistics.fmean(values["teacher"]),
                "root_visit_kl_vs_current_ref": statistics.fmean(values["visit_kl"]),
                "root_value_abs_delta_vs_current_ref": statistics.fmean(
                    values["root_value_abs_delta"]
                ),
                "changed_row_root_value_abs_delta_vs_current_ref": statistics.fmean(
                    values["changed_root_value_abs_delta"]
                )
                if values["changed_root_value_abs_delta"]
                else 0.0,
            }
            for budget, values in sorted(by_budget.items())
        },
        "p0_p1_split_384_256": {
            "player_0": statistics.fmean(p0_p1_384.get("player_0", [0.0]))
            if p0_p1_384.get("player_0")
            else 0.0,
            "player_1": statistics.fmean(p0_p1_384.get("player_1", [0.0]))
            if p0_p1_384.get("player_1")
            else 0.0,
        },
    }


def _selection_digest(
    search_result: dict[str, Any], *, top_n: int = 3
) -> list[dict[str, Any]]:
    breakdown = search_result.get("selection_breakdown")
    if not isinstance(breakdown, dict):
        return []
    moves = breakdown.get("moves")
    if not isinstance(moves, list):
        return []
    sorted_moves = sorted(
        [move for move in moves if isinstance(move, dict)],
        key=lambda move: (
            -float(move.get("selection_score", 0.0)),
            int(move.get("move", 0)),
        ),
    )
    digest: list[dict[str, Any]] = []
    for move in sorted_moves[:top_n]:
        digest.append(
            {
                "move": int(move.get("move", 0)),
                "selection_score": float(move.get("selection_score", 0.0)),
                "q_value": float(move.get("q_value", 0.0)),
                "prior": float(move.get("prior", 0.0)),
                "visits": int(move.get("visits", 0)),
                "u_value": float(move.get("u_value", 0.0)),
            }
        )
    return digest


def build_value_search_drift_diagnostic(
    *,
    candidate_name: str,
    candidate_artifact: Path,
    current_artifact: Path,
    rows: list[dict[str, Any]],
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    seed: int,
) -> dict[str, Any]:
    target_rows = [
        row for row in rows if str(row.get("budget_context")) == DRIFT_DIAGNOSTIC_BUDGET
    ]
    candidate_evaluator = ArtifactEvaluator(candidate_artifact)
    current_evaluator = ArtifactEvaluator(current_artifact)
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=tactical_root_bias,
    )
    changed_cases: list[dict[str, Any]] = []
    root_value_abs_deltas: list[float] = []
    for index, row in enumerate(target_rows):
        effective_c_puct = resolve_budget_cpuct(
            schedule=cpuct_schedule,
            challenger_simulations=int(row["challenger_simulations"]),
            current_simulations=int(row["current_simulations"]),
            default_c_puct=default_c_puct,
        )
        current_result = evaluate_artifact_position(
            evaluator=current_evaluator,
            state=row["raw_state"],
            simulations=int(row["challenger_simulations"]),
            seed=seed + index,
            c_puct=effective_c_puct,
            search_options=search_options,
            ablation_mode="full",
        )
        candidate_result = evaluate_artifact_position(
            evaluator=candidate_evaluator,
            state=row["raw_state"],
            simulations=int(row["challenger_simulations"]),
            seed=seed + index,
            c_puct=effective_c_puct,
            search_options=search_options,
            ablation_mode="full",
        )
        current_root_value = float(
            current_result.get("search_root_value", current_result.get("value", 0.0))
        )
        candidate_root_value = float(
            candidate_result.get(
                "search_root_value", candidate_result.get("value", 0.0)
            )
        )
        root_value_abs_delta = abs(candidate_root_value - current_root_value)
        root_value_abs_deltas.append(root_value_abs_delta)
        current_selected = int(current_result.get("selected_move", 0))
        candidate_selected = int(candidate_result.get("selected_move", 0))
        if current_selected == candidate_selected:
            continue
        changed_cases.append(
            {
                "game_index": int(row.get("game_index", -1)),
                "ply": int(row.get("ply", 0)),
                "phase": str(row.get("phase")),
                "seat_context": str(row.get("seat_context")),
                "opening_prefix": list(row.get("opening_prefix", [])),
                "raw_state": dict(row["raw_state"]),
                "teacher_selected_move": int(row.get("teacher_selected_move", 0)),
                "current_selected_move": current_selected,
                "candidate_selected_move": candidate_selected,
                "current_root_value": current_root_value,
                "candidate_root_value": candidate_root_value,
                "root_value_delta": candidate_root_value - current_root_value,
                "root_value_abs_delta": root_value_abs_delta,
                "current_top_moves": _selection_digest(current_result),
                "candidate_top_moves": _selection_digest(candidate_result),
            }
        )
    changed_cases.sort(
        key=lambda row: (
            -float(row["root_value_abs_delta"]),
            int(row["game_index"]),
            int(row["ply"]),
        )
    )
    return {
        "candidate": candidate_name,
        "budget": DRIFT_DIAGNOSTIC_BUDGET,
        "rows": len(target_rows),
        "changed_rows": len(changed_cases),
        "changed_rate": len(changed_cases) / max(len(target_rows), 1),
        "mean_root_value_abs_delta": statistics.fmean(root_value_abs_deltas)
        if root_value_abs_deltas
        else 0.0,
        "mean_changed_row_root_value_abs_delta": statistics.fmean(
            float(row["root_value_abs_delta"]) for row in changed_cases
        )
        if changed_cases
        else 0.0,
        "top_changed_cases": changed_cases[:MAX_DRIFT_DIAGNOSTIC_ROWS],
    }


def value_probe_metrics(
    rows: list[dict[str, Any]], outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = [float(row["final_outcome"]) for row in rows]
    predictions = [float(output["value"]) for output in outputs]
    mae = (
        statistics.fmean(
            abs(pred - target)
            for pred, target in zip(predictions, targets, strict=True)
        )
        if targets
        else 0.0
    )
    sign_accuracy = (
        statistics.fmean(
            1.0 if (pred > 0) == (target > 0) and (pred < 0) == (target < 0) else 0.0
            for pred, target in zip(predictions, targets, strict=True)
        )
        if targets
        else 0.0
    )
    corr = safe_correlation(predictions, targets)
    return {
        "rows": len(rows),
        "value_mae_vs_terminal_outcome": mae,
        "value_sign_accuracy_vs_terminal_outcome": sign_accuracy,
        "root_value_correlation_vs_terminal_outcome": corr,
    }


def candidate_probe_gate(
    *,
    candidate_name: str,
    candidate_family: str,
    current_ref_value_probe: dict[str, Any],
    policy_probe: dict[str, Any],
    search_probe: dict[str, Any],
    value_probe: dict[str, Any],
    changed_key_probe: dict[str, Any],
    current_policy_entropy: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(policy_probe.get("legal_failures", 0)) != 0:
        reasons.append("legal failures != 0")
    if not changed_key_probe.get("passes", False):
        reasons.append("changed-key audit failed")
    if candidate_name != "current_ref":
        required_gain = (
            0.03
            if candidate_family
            in {"value_head_only", "policy_value_heads", "all_model"}
            else 0.0
        )
        gain = float(
            value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
        ) - float(
            current_ref_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
        )
        if (
            candidate_family in {"value_head_only", "policy_value_heads", "all_model"}
            and gain < required_gain
        ):
            reasons.append("value sign accuracy gain < +0.03 vs current_ref")
        if candidate_family == "value_head_only":
            limit = POLICY_CHANGE_LIMITS["value_head_only"]
            if float(policy_probe.get("changed_raw_top1_rate_vs_current", 0.0)) > limit:
                reasons.append(f"policy top-1 changed rate > {limit:.2f}")
        elif candidate_family in {"policy_value_heads", "all_model"}:
            limit = POLICY_CHANGE_LIMITS["policy_value_heads"]
            if float(policy_probe.get("changed_raw_top1_rate_vs_current", 0.0)) > limit:
                reasons.append(f"policy top-1 changed rate > {limit:.2f}")
        for budget in ("768:768", "1200:1200"):
            changed_rate = float(
                search_probe.get("changed_move_rate_by_budget_context", {})
                .get(budget, {})
                .get("changed_rate_vs_current", 0.0)
            )
            if changed_rate > SEARCH_CHANGE_LIMIT:
                reasons.append(
                    f"{budget} search changed rate > {SEARCH_CHANGE_LIMIT:.2f}"
                )
    if float(policy_probe.get("candidate_entropy", 0.0)) + 1e-12 < max(
        0.2, current_policy_entropy * 0.75
    ):
        reasons.append("entropy collapse")
    return {"passed": not reasons, "reasons": reasons}


def candidate_shows_presuite_promise(
    *, candidate_name: str, candidate_rows: dict[str, dict[str, Any]]
) -> bool:
    if candidate_name == "current_ref":
        return False
    current_value_probe = candidate_rows["current_ref"]["value_probe"]
    candidate_value_probe = candidate_rows[candidate_name]["value_probe"]
    sign_gain = float(
        candidate_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
    ) - float(current_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0))
    corr_gain = float(
        candidate_value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
    ) - float(
        current_value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
    )
    return sign_gain >= MIN_PRE_SUITE_SIGN_GAIN or corr_gain >= MIN_PRE_SUITE_CORR_GAIN


def candidate_shows_followup_value_promise(
    *, candidate_name: str, candidate_rows: dict[str, dict[str, Any]]
) -> bool:
    if candidate_name == "current_ref":
        return False
    current_value_probe = candidate_rows["current_ref"]["value_probe"]
    candidate_value_probe = candidate_rows[candidate_name]["value_probe"]
    sign_gain = float(
        candidate_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
    ) - float(current_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0))
    corr_gain = float(
        candidate_value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
    ) - float(
        current_value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
    )
    mae_gain = float(
        current_value_probe.get("value_mae_vs_terminal_outcome", 0.0)
    ) - float(candidate_value_probe.get("value_mae_vs_terminal_outcome", 0.0))
    return sign_gain >= 0.015 or corr_gain >= 0.015 or mae_gain >= 0.02


def select_best_value_head_candidate(
    *, candidate_rows: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    candidate_names = [
        name
        for name in (
            "value_head_only_e1",
            "value_head_only_e2",
            "value_head_only_e2_low_lr",
        )
        if name in candidate_rows
        and candidate_rows[name].get("artifact_weights_sha256")
    ]
    if not candidate_names:
        return None
    current_value_probe = candidate_rows["current_ref"]["value_probe"]

    def score(name: str) -> tuple[float, float, float, float, float]:
        row = candidate_rows[name]
        value_probe = row["value_probe"]
        search_probe = row["search_probe"]
        sign_gain = float(
            value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
        ) - float(
            current_value_probe.get("value_sign_accuracy_vs_terminal_outcome", 0.0)
        )
        corr_gain = float(
            value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
        ) - float(
            current_value_probe.get("root_value_correlation_vs_terminal_outcome", 0.0)
        )
        mae = -float(value_probe.get("value_mae_vs_terminal_outcome", 0.0))
        eq768 = -float(
            search_probe.get("changed_move_rate_by_budget_context", {})
            .get("768:768", {})
            .get("changed_rate_vs_current", 1.0)
        )
        overall = -float(
            search_probe.get("search_selected_move_changed_rate_vs_current", 1.0)
        )
        return (sign_gain, corr_gain, eq768, overall, mae)

    selected_name = max(candidate_names, key=score)
    selected = dict(candidate_rows[selected_name])
    selected["name"] = "value_head_only_best_probe"
    selected["source_candidate"] = selected_name
    selected["report_candidate_name"] = f"best_probe_{selected_name}"
    selected["selection_score"] = {
        "sign_gain": score(selected_name)[0],
        "corr_gain": score(selected_name)[1],
        "neg_eq768_changed_rate": score(selected_name)[2],
        "neg_overall_changed_rate": score(selected_name)[3],
        "neg_value_mae": score(selected_name)[4],
    }
    return selected


def suite_rows_for_report(
    *,
    report: dict[str, Any],
    current_artifact: Path,
    candidate_artifacts: list[tuple[str, Path]],
    suite_path: Path,
) -> dict[str, Any]:
    suite_rows: dict[str, Any] = {
        suite_path.stem: {
            "suite_name": suite_path.stem,
            "suite_path": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "suite_size": count_jsonl_rows(suite_path),
            "candidates": {},
        }
    }
    for candidate_name, artifact_path in candidate_artifacts:
        candidate_report = matching_candidate_report(
            report=report,
            candidate_name=candidate_name,
            artifact_path=artifact_path,
            current_artifact=current_artifact,
        )
        if candidate_report is None:
            raise RuntimeError(
                f"missing benchmark report for {candidate_name} in {suite_path.stem}"
            )
        suite_rows[suite_path.stem]["candidates"][candidate_name] = {
            "candidate_path": str(artifact_path),
            "candidate_sha256": candidate_report.get("candidate_sha256"),
            "budget_results": benchmark_budget_results(candidate_report),
        }
    return suite_rows


def candidate_suite_metrics(
    suite_rows: dict[str, Any], candidate_name: str
) -> dict[str, Any]:
    suite_name = next(iter(suite_rows))
    return suite_rows[suite_name]["candidates"][candidate_name]["budget_results"]


def build_bootstrap_table(
    *,
    suite_rows: dict[str, Any],
    candidate_names: list[str],
    current_name: str,
    budgets: tuple[str, ...],
    seed: int,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for candidate_name in candidate_names:
        if candidate_name == current_name:
            continue
        table[candidate_name] = {}
        for budget in budgets:
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a=candidate_name,
                candidate_b=current_name,
                budget_pair=budget,
                metric_key="ds",
            )
            table[candidate_name][budget] = {
                "candidate_minus_current": bootstrap_ci(
                    diffs,
                    seed=seed + len(table[candidate_name]),
                    samples=DEFAULT_BOOTSTRAP_SAMPLES,
                )
            }
    return table


def classify_result(summary: dict[str, Any]) -> str:
    replay_audit = summary["replay_audit"]
    if replay_audit.get("abort_before_training"):
        return "selfplay_training_pipeline_not_ready"
    current_row = summary["candidate_rows"]["current_ref"]
    current_value_sign = float(
        current_row["value_probe"]["value_sign_accuracy_vs_terminal_outcome"]
    )
    value_lane = summary["candidate_rows"].get("value_head_only_e1")
    if value_lane is None:
        return "selfplay_training_pipeline_not_ready"
    value_probe = value_lane["value_probe"]
    value_improved = float(
        value_probe["value_sign_accuracy_vs_terminal_outcome"]
    ) >= current_value_sign + 0.03 and float(
        value_probe["root_value_correlation_vs_terminal_outcome"]
    ) > float(current_row["value_probe"]["root_value_correlation_vs_terminal_outcome"])
    fixed_large = summary.get("fixed_large_summary", {})
    heldout = summary.get("heldout_summary", {})
    gate = summary.get("gate_result")
    if gate and gate.get("run") and gate.get("passed"):
        return "outcome_selfplay_candidate"
    if value_improved:
        value_fixed = fixed_large.get("value_head_only_e1")
        value_medium = summary.get("medium_summary", {}).get("value_head_only_e1")
        if value_fixed and value_medium:
            low_search_gain = (
                float(value_fixed["384:256"]["candidate_minus_current"]) >= 0.03
                or float(value_medium["384:256"]["candidate_minus_current"]) >= 0.03
            )
            high_search_ok = (
                float(value_fixed["768:768"]["candidate_minus_current"]) >= -0.05
                and float(value_fixed["1200:1200"]["candidate_minus_current"]) >= -0.03
            )
            if low_search_gain and high_search_ok:
                return "terminal_outcome_value_promising"
    if any(
        row["probe_gate"]["passed"]
        for name, row in summary["candidate_rows"].items()
        if name != "current_ref"
    ):
        if heldout:
            for name, row in heldout.items():
                if name == "current_ref":
                    continue
                low = float(row["384:256"]["candidate_minus_current"])
                eq = float(row["768:768"]["candidate_minus_current"])
                high = float(row["1200:1200"]["candidate_minus_current"])
                if low >= 0.05 and eq < -0.05:
                    return "outcome_selfplay_tradeoff"
                if low >= 0.05 and high < -0.03:
                    return "outcome_selfplay_tradeoff"
        return "outcome_value_probe_good_but_strength_neutral"
    return "selfplay_training_pipeline_not_ready"


def build_candidate_eval_delta_map(
    suite_rows: dict[str, Any], candidate_names: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    suite_name = next(iter(suite_rows))
    current_results = suite_rows[suite_name]["candidates"]["current_ref"][
        "budget_results"
    ]
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for candidate_name in candidate_names:
        candidate_results = suite_rows[suite_name]["candidates"][candidate_name][
            "budget_results"
        ]
        rows[candidate_name] = {}
        for budget in DEFAULT_BUDGET_PAIRS:
            candidate_budget = candidate_results.get(budget, {})
            current_budget = current_results.get(budget, {})
            candidate_ds = float(candidate_budget.get("ds", 0.0))
            current_ds = float(current_budget.get("ds", 0.0))
            rows[candidate_name][budget] = {
                "ds": candidate_ds,
                "candidate_minus_current": candidate_ds - current_ds,
                "p0_score": float(candidate_budget.get("p0_score", 0.0)),
                "p1_score": float(candidate_budget.get("p1_score", 0.0)),
                "duplicate_trajectory_count": float(
                    candidate_budget.get("duplicate_trajectory_count", 0.0) or 0.0
                ),
            }
    return rows


def candidate_artifacts_from_rows(
    candidate_rows: dict[str, dict[str, Any]],
) -> list[tuple[str, Path]]:
    return [
        (name, Path(str(row["artifact_dir"]))) for name, row in candidate_rows.items()
    ]


def train_and_probe_followup_lane(
    *,
    lane_name: str,
    candidate_rows: dict[str, dict[str, Any]],
    train_rows: list[dict[str, Any]],
    init_checkpoint: Path,
    workdir: Path,
    policy_probe_rows: list[dict[str, Any]],
    search_probe_rows: list[dict[str, Any]],
    value_probe_rows: list[dict[str, Any]],
    current_raw_outputs: list[dict[str, Any]],
    current_search_outputs: list[dict[str, Any]],
    current_ref_value_probe: dict[str, Any],
    current_policy_entropy: float,
    default_c_puct: float,
    cpuct_schedule: dict[str, float],
    tactical_root_bias: float,
    workers: int,
    seed: int,
    allowed_families: set[str],
) -> dict[str, Any]:
    lane_spec = LANE_SPECS[lane_name]
    lane_dir = workdir / lane_name
    lane_dir.mkdir(parents=True, exist_ok=True)
    row = train_candidate(
        lane_spec=lane_spec,
        train_rows=train_rows,
        init_checkpoint=init_checkpoint,
        lane_dir=lane_dir,
        seed=seed,
    )
    artifact_path = Path(str(row["artifact_dir"]))
    checkpoint_path = Path(str(row["checkpoint_path"]))
    raw_outputs = evaluate_raw_outputs(
        artifact_path=artifact_path, rows=policy_probe_rows
    )["outputs"]
    search_outputs = evaluate_search_outputs(
        artifact_path=artifact_path,
        rows=search_probe_rows,
        default_c_puct=default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=tactical_root_bias,
        seed=seed,
        workers=workers,
    )
    value_outputs = evaluate_raw_outputs(
        artifact_path=artifact_path,
        rows=build_policy_probe_rows(value_probe_rows),
    )["outputs"]
    row["policy_probe"] = evaluate_policy_probe_metrics(
        rows=policy_probe_rows,
        candidate_outputs=raw_outputs,
        current_outputs=current_raw_outputs,
    )
    row["search_probe"] = evaluate_search_probe_metrics(
        rows=search_probe_rows,
        candidate_outputs=search_outputs,
        current_outputs=current_search_outputs,
    )
    row["value_probe"] = value_probe_metrics(value_probe_rows, value_outputs)
    row["changed_key_probe"] = changed_key_audit(
        reference_checkpoint=init_checkpoint,
        candidate_checkpoint=checkpoint_path,
        allowed_families=allowed_families,
    )
    row["probe_gate"] = candidate_probe_gate(
        candidate_name=lane_name,
        candidate_family=row["family"],
        current_ref_value_probe=current_ref_value_probe,
        policy_probe=row["policy_probe"],
        search_probe=row["search_probe"],
        value_probe=row["value_probe"],
        changed_key_probe=row["changed_key_probe"],
        current_policy_entropy=current_policy_entropy,
    )
    candidate_rows[lane_name] = row
    return row


def build_report(summary: dict[str, Any]) -> str:
    medium_summary = summary.get("medium_summary", {})
    fixed_large_summary = summary.get("fixed_large_summary", {})
    heldout_summary = summary.get("heldout_summary", {})
    candidate_rows = summary["candidate_rows"]
    candidate_table_rows = []
    training_rows = []
    probe_rows = []
    search_drift_rows = []
    aborted_rows = []
    for name, row in candidate_rows.items():
        candidate_table_rows.append(
            [
                name,
                row.get("trainable_scope", "none"),
                row.get("epochs", 0),
                row.get("artifact_weights_sha256", "n/a"),
                row["probe_gate"]["passed"],
            ]
        )
        training = row.get("training_metrics", {})
        training_rows.append(
            [
                name,
                fmt(float(training.get("policy_loss", 0.0))),
                fmt(float(training.get("value_loss", 0.0))),
                fmt(float(training.get("total_loss", 0.0))),
                fmt(training.get("grad_norm")),
            ]
        )
        probe_rows.append(
            [
                name,
                int(row["policy_probe"].get("legal_failures", 0)),
                fmt(float(row["policy_probe"].get("policy_kl", 0.0))),
                fmt(
                    float(
                        row["policy_probe"].get("changed_raw_top1_rate_vs_current", 0.0)
                    )
                ),
                fmt(
                    float(row["value_probe"].get("value_mae_vs_terminal_outcome", 0.0))
                ),
                fmt(
                    float(
                        row["value_probe"].get(
                            "value_sign_accuracy_vs_terminal_outcome", 0.0
                        )
                    )
                ),
                fmt(
                    float(
                        row["value_probe"].get(
                            "root_value_correlation_vs_terminal_outcome", 0.0
                        )
                    )
                ),
                row["probe_gate"]["passed"],
            ]
        )
        search_drift_rows.append(
            [
                name,
                fmt(
                    float(
                        row["search_probe"].get(
                            "root_value_abs_delta_vs_current_ref", 0.0
                        )
                    )
                ),
                fmt(
                    float(
                        row["search_probe"].get(
                            "changed_row_root_value_abs_delta_vs_current_ref", 0.0
                        )
                    )
                ),
                fmt(
                    float(
                        row["search_probe"]
                        .get("changed_move_rate_by_budget_context", {})
                        .get("768:768", {})
                        .get("root_value_abs_delta_vs_current_ref", 0.0)
                    )
                ),
                fmt(
                    float(
                        row["search_probe"]
                        .get("changed_move_rate_by_budget_context", {})
                        .get("768:768", {})
                        .get("changed_row_root_value_abs_delta_vs_current_ref", 0.0)
                    )
                ),
                fmt(
                    float(
                        row["search_probe"]
                        .get("changed_move_rate_by_budget_context", {})
                        .get("768:768", {})
                        .get("changed_rate_vs_current", 0.0)
                    )
                ),
            ]
        )
        if not row["probe_gate"]["passed"]:
            aborted_rows.append(
                [name, "; ".join(row["probe_gate"]["reasons"]) or "not carried"]
            )
    medium_rows = []
    for name, budget_map in sorted(medium_summary.items()):
        medium_rows.append(
            [
                name,
                *(
                    fmt(float(budget_map[budget]["candidate_minus_current"]))
                    for budget in DEFAULT_BUDGET_PAIRS
                ),
            ]
        )
    fixed_rows = []
    for name, budget_map in sorted(fixed_large_summary.items()):
        fixed_rows.append(
            [
                name,
                *(
                    fmt(float(budget_map[budget]["candidate_minus_current"]))
                    for budget in DEFAULT_BUDGET_PAIRS
                ),
            ]
        )
    heldout_rows = []
    for name, budget_map in sorted(heldout_summary.items()):
        heldout_rows.append(
            [
                name,
                *(
                    fmt(float(budget_map[budget]["candidate_minus_current"]))
                    for budget in DEFAULT_BUDGET_PAIRS
                ),
            ]
        )
    bootstrap_rows = []
    for candidate_name, budgets in sorted(summary.get("bootstrap_cis", {}).items()):
        for budget, payload in sorted(budgets.items()):
            ci = payload["candidate_minus_current"]
            bootstrap_rows.append(
                [
                    candidate_name,
                    budget,
                    fmt(ci["mean"]),
                    fmt(ci["lower"]),
                    fmt(ci["upper"]),
                ]
            )
    p0_rows = []
    for name, row in sorted(candidate_rows.items()):
        split = row["search_probe"].get("p0_p1_split_384_256", {})
        p0_rows.append(
            [
                name,
                fmt(float(split.get("player_0", 0.0))),
                fmt(float(split.get("player_1", 0.0))),
            ]
        )
    duplicate_rows = []
    for name, medium in sorted(medium_summary.items()):
        duplicate_rows.append(
            [name, fmt(float(medium["384:256"].get("duplicate_trajectory_count", 0.0)))]
        )
    runtime_rows = [
        [phase, fmt(float(seconds), 2)]
        for phase, seconds in sorted(summary.get("runtime_cost", {}).items())
    ]
    drift_case_rows = []
    drift_diagnostic = summary.get("value_search_drift_diagnostic") or {}
    for case in drift_diagnostic.get("top_changed_cases", [])[:10]:
        drift_case_rows.append(
            [
                int(case.get("game_index", -1)),
                int(case.get("ply", 0)),
                str(case.get("phase")),
                str(case.get("seat_context")),
                int(case.get("current_selected_move", 0)),
                int(case.get("candidate_selected_move", 0)),
                fmt(float(case.get("current_root_value", 0.0))),
                fmt(float(case.get("candidate_root_value", 0.0))),
                fmt(float(case.get("root_value_delta", 0.0))),
            ]
        )
    gate_result = summary.get("gate_result")
    gate_lines = ["- gate run: `False`"]
    if gate_result:
        gate_lines = [
            f"- gate run: `{gate_result.get('run', False)}`",
            f"- gate passed: `{gate_result.get('passed', False)}`",
            f"- gate candidate: `{gate_result.get('candidate')}`",
        ]
    lines = [
        DOC_TITLE,
        "",
        f"**Date**: {date.today().isoformat()}",
        "",
        f"**Classification**: `{summary['final_classification']}`",
        "",
        "## Current Artifact Hash",
        "",
        f"- current weights SHA256: `{summary['inputs']['current_artifact_weights']['actual_sha256']}`",
        "",
        "## Promoted Runtime Profile Confirmation",
        "",
        f"- default runtime schedule: `{json.dumps(summary['runtime_profile_confirmation']['schedule_manifest'], sort_keys=True)}`",
        f"- self-play runtime profiles: `{json.dumps(summary['runtime_profile_confirmation']['selfplay_profiles'], sort_keys=True)}`",
        "",
        "## Self-Play Replay Audit",
        "",
        f"- replay rows: `{summary['replay_audit']['rows']}`",
        f"- games completed: `{summary['replay_audit']['games_completed']}`",
        f"- unique states: `{summary['replay_audit']['unique_states']}`",
        f"- duplicate state rate: `{summary['replay_audit']['duplicate_state_rate']:.4f}`",
        f"- invalid policies: `{summary['replay_audit']['illegal_policy_count']}`",
        f"- invalid legal masks: `{summary['replay_audit']['invalid_legal_mask_count']}`",
        f"- outcome distribution: `{json.dumps(summary['replay_audit']['outcome_distribution'], sort_keys=True)}`",
        f"- phase distribution: `{json.dumps(summary['replay_audit']['phase_distribution'], sort_keys=True)}`",
        f"- seat distribution: `{json.dumps(summary['replay_audit']['seat_distribution'], sort_keys=True)}`",
        "",
        "## Root-Value Vs Terminal-Outcome Calibration",
        "",
        f"- correlation: `{summary['replay_audit']['root_value_vs_terminal_outcome']['correlation']:+.4f}`",
        f"- root value mean: `{summary['replay_audit']['root_value_vs_terminal_outcome']['root_value_mean']:+.4f}`",
        f"- terminal outcome mean: `{summary['replay_audit']['root_value_vs_terminal_outcome']['outcome_mean']:+.4f}`",
        "",
        "## Candidate Table",
        "",
        markdown_table(
            ["Candidate", "Scope", "Epochs", "Artifact SHA", "Probe pass"],
            candidate_table_rows,
        ),
        "",
        "## Training Loss Table",
        "",
        markdown_table(
            ["Candidate", "Policy loss", "Value loss", "Total loss", "Grad norm"],
            training_rows,
        ),
        "",
        "## Probe Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Legal fails",
                "Policy KL",
                "Top-1 change",
                "Value MAE",
                "Sign acc",
                "Value corr",
                "Passed",
            ],
            probe_rows,
        ),
        "",
        "## Search Drift Diagnostics",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean |dV|",
                "Changed-row |dV|",
                "768 |dV|",
                "768 changed-row |dV|",
                "768 changed rate",
            ],
            search_drift_rows,
        ),
        "",
        "## Value-Search Drift Cases",
        "",
        f"- candidate: `{drift_diagnostic.get('candidate')}`",
        f"- budget: `{drift_diagnostic.get('budget')}`",
        f"- changed rows: `{drift_diagnostic.get('changed_rows')}` / `{drift_diagnostic.get('rows')}`",
        f"- mean |dV|: `{fmt(float(drift_diagnostic.get('mean_root_value_abs_delta', 0.0)))}`",
        f"- changed-row mean |dV|: `{fmt(float(drift_diagnostic.get('mean_changed_row_root_value_abs_delta', 0.0)))}`",
        "",
        markdown_table(
            [
                "Game",
                "Ply",
                "Phase",
                "Seat",
                "Current move",
                "Cand move",
                "Current V",
                "Cand V",
                "Delta V",
            ],
            drift_case_rows or [["n/a", "", "", "", "", "", "", "", ""]],
        ),
        "",
        "## Aborted-Candidate Table",
        "",
        markdown_table(["Candidate", "Reasons"], aborted_rows or [["none", "n/a"]]),
        "",
        "## Medium DS Table",
        "",
        markdown_table(
            ["Candidate", *DEFAULT_BUDGET_PAIRS],
            medium_rows or [["not_run", *(["n/a"] * len(DEFAULT_BUDGET_PAIRS))]],
        ),
        "",
        "## Fixed-Large DS Table",
        "",
        markdown_table(
            ["Candidate", *DEFAULT_BUDGET_PAIRS],
            fixed_rows or [["not_run", *(["n/a"] * len(DEFAULT_BUDGET_PAIRS))]],
        ),
        "",
    ]
    if heldout_rows:
        lines.extend(
            [
                "## Held-Out Table",
                "",
                markdown_table(["Candidate", *DEFAULT_BUDGET_PAIRS], heldout_rows),
                "",
            ]
        )
    lines.extend(
        [
            "## Bootstrap CIs",
            "",
            "- orientation: `candidate_minus_current`",
            "",
            markdown_table(
                ["Candidate", "Budget", "Mean", "Lower 95%", "Upper 95%"],
                bootstrap_rows or [["not_run", "", "", "", ""]],
            ),
            "",
            "## P0/P1 Split For 384:256",
            "",
            markdown_table(["Candidate", "P0 changed", "P1 changed"], p0_rows),
            "",
            "## Duplicate Trajectory Count",
            "",
            markdown_table(
                ["Candidate", "384:256 duplicates"],
                duplicate_rows or [["not_run", "n/a"]],
            ),
            "",
            "## Runtime Cost",
            "",
            markdown_table(
                ["Phase", "Seconds"], runtime_rows or [["not_recorded", "n/a"]]
            ),
            "",
            "## Gate Result",
            "",
            *gate_lines,
            "",
            "## Final Classification",
            "",
            f"- result: `{summary['final_classification']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    current_artifact = Path(args.current)
    medium_suite = Path(args.medium_suite)
    fixed_large_suite = Path(args.fixed_large_suite)
    heldout_suites = parse_csv_paths(args.heldout_suites)
    lane_names = parse_lane_names(args.lanes)
    cpuct_schedule = parse_cpuct_schedule_json(args.cpuct_schedule)

    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(
        current_artifact / "metadata.json", "current artifact metadata"
    )
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(fixed_large_suite, "fixed large suite")
    for suite_path in heldout_suites:
        require_existing_file(suite_path, f"heldout suite {suite_path.name}")

    init_checkpoint = materialize_current_checkpoint(current_artifact, workdir)
    input_summary = {
        "current_artifact_weights": verify_expected_hash(
            current_artifact / "weights.json",
            args.expected_current_weights_sha256,
            "current artifact weights",
        ),
        "current_artifact_metadata": build_input_summary(
            current_artifact / "metadata.json"
        ),
        "init_checkpoint": build_input_summary(init_checkpoint),
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suites
        },
    }

    timings: dict[str, float] = {}
    replay_path = workdir / "replay.jsonl"
    replay_audit_path = workdir / "replay_audit.json"
    summary_path = workdir / "summary_metrics.json"

    start = time.time()
    replay_rows, generation_meta = generate_replay(
        current_artifact=current_artifact,
        out_path=replay_path,
        games=args.selfplay_games,
        primary_sims=args.selfplay_sims,
        default_c_puct=args.default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=args.tactical_root_bias,
        seed=args.seed,
        workers=args.workers,
    )
    timings["selfplay_generation"] = time.time() - start
    audit = replay_audit(replay_rows, generation_meta)
    write_json(replay_audit_path, audit)
    if audit["abort_before_training"]:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "aborted_before_training",
            "inputs": input_summary,
            "replay_path": str(replay_path),
            "replay_audit": audit,
            "runtime_profile_confirmation": {
                "schedule_manifest": schedule_definition(
                    default_c_puct=args.default_c_puct, schedule=cpuct_schedule
                ),
                "selfplay_profiles": generation_meta["runtime_profiles"],
            },
            "runtime_cost": timings,
            "candidate_rows": {},
            "bootstrap_cis": {},
            "final_classification": "selfplay_training_pipeline_not_ready",
        }
        write_json(summary_path, summary)
        REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
        return 1

    train_rows, heldout_rows = split_replay_rows(replay_rows)
    train_replay_path = workdir / "train_replay.jsonl"
    heldout_replay_path = workdir / "heldout_replay.jsonl"
    write_jsonl(train_replay_path, train_rows)
    write_jsonl(heldout_replay_path, heldout_rows)

    probe_anchor_rows = sample_probe_rows(
        heldout_rows or replay_rows, seed=args.seed, cap=PROBE_ANCHOR_CAP
    )
    value_probe_rows = sample_probe_rows(
        heldout_rows or replay_rows, seed=args.seed + 1, cap=VALUE_PROBE_CAP
    )
    policy_probe_rows = build_policy_probe_rows(probe_anchor_rows)
    search_probe_rows = build_search_probe_rows(probe_anchor_rows, SEARCH_PROBE_BUDGETS)

    candidate_rows: dict[str, dict[str, Any]] = {
        "current_ref": {
            "name": "current_ref",
            "report_candidate_name": "current_ref",
            "epochs": 0,
            "trainable_scope": "none",
            "family": "reference",
            "checkpoint_path": str(init_checkpoint),
            "checkpoint_sha256": sha256_file(init_checkpoint),
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": sha256_file(current_artifact / "weights.json"),
            "training_metrics": {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": 0.0,
                "grad_norm": None,
                "training_elapsed_s": 0.0,
            },
        }
    }

    initial_lane_names = [
        name
        for name in lane_names
        if name
        not in {
            "value_head_only_e2",
            "value_head_only_e2_low_lr",
            "policy_value_heads_e2",
        }
    ]
    if "all_model_low_lr_e1" in initial_lane_names:
        initial_lane_names = [
            name for name in initial_lane_names if name != "all_model_low_lr_e1"
        ]

    start = time.time()
    for lane_name in initial_lane_names:
        lane_spec = LANE_SPECS[lane_name]
        lane_dir = workdir / lane_name
        lane_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows[lane_name] = train_candidate(
            lane_spec=lane_spec,
            train_rows=train_rows,
            init_checkpoint=init_checkpoint,
            lane_dir=lane_dir,
            seed=args.seed,
        )
    timings["training_initial"] = time.time() - start

    current_raw_outputs = evaluate_raw_outputs(
        artifact_path=current_artifact, rows=policy_probe_rows
    )["outputs"]
    current_search_outputs = evaluate_search_outputs(
        artifact_path=current_artifact,
        rows=search_probe_rows,
        default_c_puct=args.default_c_puct,
        cpuct_schedule=cpuct_schedule,
        tactical_root_bias=args.tactical_root_bias,
        seed=args.seed,
        workers=args.workers,
    )
    current_value_outputs = evaluate_raw_outputs(
        artifact_path=current_artifact,
        rows=build_policy_probe_rows(value_probe_rows),
    )["outputs"]

    for name, row in list(candidate_rows.items()):
        artifact_path = Path(str(row["artifact_dir"]))
        checkpoint_path = Path(str(row["checkpoint_path"]))
        raw_outputs = (
            current_raw_outputs
            if name == "current_ref"
            else evaluate_raw_outputs(
                artifact_path=artifact_path, rows=policy_probe_rows
            )["outputs"]
        )
        search_outputs = (
            current_search_outputs
            if name == "current_ref"
            else evaluate_search_outputs(
                artifact_path=artifact_path,
                rows=search_probe_rows,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                seed=args.seed,
                workers=args.workers,
            )
        )
        value_outputs = (
            current_value_outputs
            if name == "current_ref"
            else evaluate_raw_outputs(
                artifact_path=artifact_path,
                rows=build_policy_probe_rows(value_probe_rows),
            )["outputs"]
        )
        policy_probe = evaluate_policy_probe_metrics(
            rows=policy_probe_rows,
            candidate_outputs=raw_outputs,
            current_outputs=current_raw_outputs,
        )
        search_probe = evaluate_search_probe_metrics(
            rows=search_probe_rows,
            candidate_outputs=search_outputs,
            current_outputs=current_search_outputs,
        )
        value_probe = value_probe_metrics(value_probe_rows, value_outputs)
        if name == "current_ref":
            changed_key_probe = {
                "passes": True,
                "max_abs_diff": 0.0,
                "changed_keys": {"trunk": [], "policy": [], "value": [], "other": []},
                "allowed_families": [],
                "disallowed_changed_families": [],
            }
        elif row["family"] == "value_head_only":
            changed_key_probe = changed_key_audit(
                reference_checkpoint=init_checkpoint,
                candidate_checkpoint=checkpoint_path,
                allowed_families={"value"},
            )
        elif row["family"] == "policy_value_heads":
            changed_key_probe = changed_key_audit(
                reference_checkpoint=init_checkpoint,
                candidate_checkpoint=checkpoint_path,
                allowed_families={"policy", "value"},
            )
        else:
            changed_key_probe = changed_key_audit(
                reference_checkpoint=init_checkpoint,
                candidate_checkpoint=checkpoint_path,
                allowed_families={"trunk", "policy", "value", "other"},
            )
        row["policy_probe"] = policy_probe
        row["search_probe"] = search_probe
        row["value_probe"] = value_probe
        row["changed_key_probe"] = changed_key_probe

    current_ref_value_probe = candidate_rows["current_ref"]["value_probe"]
    current_policy_entropy = float(
        candidate_rows["current_ref"]["policy_probe"]["candidate_entropy"]
    )
    for name, row in candidate_rows.items():
        row["probe_gate"] = candidate_probe_gate(
            candidate_name=name,
            candidate_family=row["family"],
            current_ref_value_probe=current_ref_value_probe,
            policy_probe=row["policy_probe"],
            search_probe=row["search_probe"],
            value_probe=row["value_probe"],
            changed_key_probe=row["changed_key_probe"],
            current_policy_entropy=current_policy_entropy,
        )

    if "value_head_only_e2" in lane_names:
        e1_followup_ready = candidate_shows_followup_value_promise(
            candidate_name="value_head_only_e1", candidate_rows=candidate_rows
        )
        if e1_followup_ready:
            start = time.time()
            value_e1_checkpoint = Path(
                str(candidate_rows["value_head_only_e1"]["checkpoint_path"])
            )
            train_and_probe_followup_lane(
                lane_name="value_head_only_e2",
                candidate_rows=candidate_rows,
                train_rows=train_rows,
                init_checkpoint=value_e1_checkpoint,
                workdir=workdir,
                policy_probe_rows=policy_probe_rows,
                search_probe_rows=search_probe_rows,
                value_probe_rows=value_probe_rows,
                current_raw_outputs=current_raw_outputs,
                current_search_outputs=current_search_outputs,
                current_ref_value_probe=current_ref_value_probe,
                current_policy_entropy=current_policy_entropy,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                workers=args.workers,
                seed=args.seed,
                allowed_families={"value"},
            )
            timings["training_value_e2"] = time.time() - start
        else:
            candidate_rows["value_head_only_e2"] = {
                "name": "value_head_only_e2",
                "report_candidate_name": "value_head_only_e2_not_trained",
                "epochs": 2,
                "trainable_scope": "value_head_only",
                "family": "value_head_only",
                "checkpoint_path": str(
                    workdir / "value_head_only_e2" / "checkpoint_epoch2.npz"
                ),
                "artifact_dir": str(workdir / "value_head_only_e2" / "artifact"),
                "training_metrics": {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "total_loss": 0.0,
                    "grad_norm": None,
                },
                "policy_probe": candidate_rows["value_head_only_e1"]["policy_probe"],
                "search_probe": candidate_rows["value_head_only_e1"]["search_probe"],
                "value_probe": candidate_rows["value_head_only_e1"]["value_probe"],
                "changed_key_probe": candidate_rows["value_head_only_e1"][
                    "changed_key_probe"
                ],
                "probe_gate": {
                    "passed": False,
                    "reasons": [
                        "value_head_only_e1 did not show enough follow-up value movement; e2 not trained"
                    ],
                },
                "skipped_training": True,
                "artifact_weights_sha256": None,
                "checkpoint_sha256": None,
            }

    if "value_head_only_e2_low_lr" in lane_names:
        e1_followup_ready = candidate_shows_followup_value_promise(
            candidate_name="value_head_only_e1", candidate_rows=candidate_rows
        )
        if e1_followup_ready:
            start = time.time()
            value_e1_checkpoint = Path(
                str(candidate_rows["value_head_only_e1"]["checkpoint_path"])
            )
            train_and_probe_followup_lane(
                lane_name="value_head_only_e2_low_lr",
                candidate_rows=candidate_rows,
                train_rows=train_rows,
                init_checkpoint=value_e1_checkpoint,
                workdir=workdir,
                policy_probe_rows=policy_probe_rows,
                search_probe_rows=search_probe_rows,
                value_probe_rows=value_probe_rows,
                current_raw_outputs=current_raw_outputs,
                current_search_outputs=current_search_outputs,
                current_ref_value_probe=current_ref_value_probe,
                current_policy_entropy=current_policy_entropy,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                workers=args.workers,
                seed=args.seed,
                allowed_families={"value"},
            )
            timings["training_value_e2_low_lr"] = time.time() - start
        else:
            candidate_rows["value_head_only_e2_low_lr"] = {
                "name": "value_head_only_e2_low_lr",
                "report_candidate_name": "value_head_only_e2_low_lr_not_trained",
                "epochs": 2,
                "trainable_scope": "value_head_only",
                "family": "value_head_only",
                "checkpoint_path": str(
                    workdir / "value_head_only_e2_low_lr" / "checkpoint_epoch2.npz"
                ),
                "artifact_dir": str(workdir / "value_head_only_e2_low_lr" / "artifact"),
                "training_metrics": {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "total_loss": 0.0,
                    "grad_norm": None,
                },
                "policy_probe": candidate_rows["value_head_only_e1"]["policy_probe"],
                "search_probe": candidate_rows["value_head_only_e1"]["search_probe"],
                "value_probe": candidate_rows["value_head_only_e1"]["value_probe"],
                "changed_key_probe": candidate_rows["value_head_only_e1"][
                    "changed_key_probe"
                ],
                "probe_gate": {
                    "passed": False,
                    "reasons": [
                        "value_head_only_e1 did not show enough follow-up value movement; low-lr e2 not trained"
                    ],
                },
                "skipped_training": True,
                "artifact_weights_sha256": None,
                "checkpoint_sha256": None,
            }

    if "policy_value_heads_e2" in lane_names:
        e1_gate = candidate_rows.get("policy_value_heads_e1", {}).get("probe_gate", {})
        if e1_gate.get("passed"):
            start = time.time()
            lane_name = "policy_value_heads_e2"
            lane_spec = LANE_SPECS[lane_name]
            lane_dir = workdir / lane_name
            lane_dir.mkdir(parents=True, exist_ok=True)
            heads_e1_checkpoint = Path(
                str(candidate_rows["policy_value_heads_e1"]["checkpoint_path"])
            )
            candidate_rows[lane_name] = train_candidate(
                lane_spec=lane_spec,
                train_rows=train_rows,
                init_checkpoint=heads_e1_checkpoint,
                lane_dir=lane_dir,
                seed=args.seed,
            )
            timings["training_e2"] = time.time() - start
            artifact_path = Path(str(candidate_rows[lane_name]["artifact_dir"]))
            checkpoint_path = Path(str(candidate_rows[lane_name]["checkpoint_path"]))
            raw_outputs = evaluate_raw_outputs(
                artifact_path=artifact_path, rows=policy_probe_rows
            )["outputs"]
            search_outputs = evaluate_search_outputs(
                artifact_path=artifact_path,
                rows=search_probe_rows,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                seed=args.seed,
                workers=args.workers,
            )
            value_outputs = evaluate_raw_outputs(
                artifact_path=artifact_path,
                rows=build_policy_probe_rows(value_probe_rows),
            )["outputs"]
            candidate_rows[lane_name]["policy_probe"] = evaluate_policy_probe_metrics(
                rows=policy_probe_rows,
                candidate_outputs=raw_outputs,
                current_outputs=current_raw_outputs,
            )
            candidate_rows[lane_name]["search_probe"] = evaluate_search_probe_metrics(
                rows=search_probe_rows,
                candidate_outputs=search_outputs,
                current_outputs=current_search_outputs,
            )
            candidate_rows[lane_name]["value_probe"] = value_probe_metrics(
                value_probe_rows, value_outputs
            )
            candidate_rows[lane_name]["changed_key_probe"] = changed_key_audit(
                reference_checkpoint=init_checkpoint,
                candidate_checkpoint=checkpoint_path,
                allowed_families={"policy", "value"},
            )
            candidate_rows[lane_name]["probe_gate"] = candidate_probe_gate(
                candidate_name=lane_name,
                candidate_family=candidate_rows[lane_name]["family"],
                current_ref_value_probe=current_ref_value_probe,
                policy_probe=candidate_rows[lane_name]["policy_probe"],
                search_probe=candidate_rows[lane_name]["search_probe"],
                value_probe=candidate_rows[lane_name]["value_probe"],
                changed_key_probe=candidate_rows[lane_name]["changed_key_probe"],
                current_policy_entropy=current_policy_entropy,
            )
        else:
            candidate_rows["policy_value_heads_e2"] = {
                "name": "policy_value_heads_e2",
                "report_candidate_name": "policy_value_heads_e2_not_trained",
                "epochs": 2,
                "trainable_scope": "policy_value_heads",
                "family": "policy_value_heads",
                "checkpoint_path": str(
                    workdir / "policy_value_heads_e2" / "checkpoint_epoch2.npz"
                ),
                "artifact_dir": str(workdir / "policy_value_heads_e2" / "artifact"),
                "training_metrics": {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "total_loss": 0.0,
                    "grad_norm": None,
                },
                "policy_probe": candidate_rows["policy_value_heads_e1"]["policy_probe"],
                "search_probe": candidate_rows["policy_value_heads_e1"]["search_probe"],
                "value_probe": candidate_rows["policy_value_heads_e1"]["value_probe"],
                "changed_key_probe": candidate_rows["policy_value_heads_e1"][
                    "changed_key_probe"
                ],
                "probe_gate": {
                    "passed": False,
                    "reasons": [
                        "policy_value_heads_e1 probes were not stable; e2 not trained"
                    ],
                },
                "skipped_training": True,
                "artifact_weights_sha256": None,
                "checkpoint_sha256": None,
            }

    if "all_model_low_lr_e1" in lane_names:
        head_lanes_ok = any(
            candidate_rows.get(name, {}).get("probe_gate", {}).get("passed")
            for name in ("value_head_only_e1", "policy_value_heads_e1")
        )
        if head_lanes_ok:
            lane_name = "all_model_low_lr_e1"
            lane_dir = workdir / lane_name
            lane_dir.mkdir(parents=True, exist_ok=True)
            candidate_rows[lane_name] = train_candidate(
                lane_spec=LANE_SPECS[lane_name],
                train_rows=train_rows,
                init_checkpoint=init_checkpoint,
                lane_dir=lane_dir,
                seed=args.seed,
            )
            artifact_path = Path(str(candidate_rows[lane_name]["artifact_dir"]))
            checkpoint_path = Path(str(candidate_rows[lane_name]["checkpoint_path"]))
            raw_outputs = evaluate_raw_outputs(
                artifact_path=artifact_path, rows=policy_probe_rows
            )["outputs"]
            search_outputs = evaluate_search_outputs(
                artifact_path=artifact_path,
                rows=search_probe_rows,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                seed=args.seed,
                workers=args.workers,
            )
            value_outputs = evaluate_raw_outputs(
                artifact_path=artifact_path,
                rows=build_policy_probe_rows(value_probe_rows),
            )["outputs"]
            candidate_rows[lane_name]["policy_probe"] = evaluate_policy_probe_metrics(
                rows=policy_probe_rows,
                candidate_outputs=raw_outputs,
                current_outputs=current_raw_outputs,
            )
            candidate_rows[lane_name]["search_probe"] = evaluate_search_probe_metrics(
                rows=search_probe_rows,
                candidate_outputs=search_outputs,
                current_outputs=current_search_outputs,
            )
            candidate_rows[lane_name]["value_probe"] = value_probe_metrics(
                value_probe_rows, value_outputs
            )
            candidate_rows[lane_name]["changed_key_probe"] = changed_key_audit(
                reference_checkpoint=init_checkpoint,
                candidate_checkpoint=checkpoint_path,
                allowed_families={"trunk", "policy", "value", "other"},
            )
            candidate_rows[lane_name]["probe_gate"] = candidate_probe_gate(
                candidate_name=lane_name,
                candidate_family=candidate_rows[lane_name]["family"],
                current_ref_value_probe=current_ref_value_probe,
                policy_probe=candidate_rows[lane_name]["policy_probe"],
                search_probe=candidate_rows[lane_name]["search_probe"],
                value_probe=candidate_rows[lane_name]["value_probe"],
                changed_key_probe=candidate_rows[lane_name]["changed_key_probe"],
                current_policy_entropy=current_policy_entropy,
            )
        else:
            candidate_rows["all_model_low_lr_e1"] = {
                "name": "all_model_low_lr_e1",
                "report_candidate_name": "all_model_low_lr_e1_not_trained",
                "epochs": 1,
                "trainable_scope": "all",
                "family": "all_model",
                "checkpoint_path": str(
                    workdir / "all_model_low_lr_e1" / "checkpoint_epoch1.npz"
                ),
                "artifact_dir": str(workdir / "all_model_low_lr_e1" / "artifact"),
                "training_metrics": {
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "total_loss": 0.0,
                    "grad_norm": None,
                },
                "policy_probe": candidate_rows["current_ref"]["policy_probe"],
                "search_probe": candidate_rows["current_ref"]["search_probe"],
                "value_probe": candidate_rows["current_ref"]["value_probe"],
                "changed_key_probe": candidate_rows["current_ref"]["changed_key_probe"],
                "probe_gate": {
                    "passed": False,
                    "reasons": [
                        "head-only lanes did not clear probe stability; optional all-model lane not trained"
                    ],
                },
                "skipped_training": True,
                "artifact_weights_sha256": None,
                "checkpoint_sha256": None,
            }

    best_value_head_candidate = select_best_value_head_candidate(
        candidate_rows=candidate_rows
    )
    if best_value_head_candidate is not None:
        candidate_rows["value_head_only_best_probe"] = best_value_head_candidate

    value_search_drift_diagnostic: dict[str, Any] | None = None
    if "value_head_only_best_probe" in candidate_rows:
        best_artifact = Path(
            str(candidate_rows["value_head_only_best_probe"]["artifact_dir"])
        )
        if (best_artifact / "weights.json").is_file():
            value_search_drift_diagnostic = build_value_search_drift_diagnostic(
                candidate_name="value_head_only_best_probe",
                candidate_artifact=best_artifact,
                current_artifact=current_artifact,
                rows=search_probe_rows,
                default_c_puct=args.default_c_puct,
                cpuct_schedule=cpuct_schedule,
                tactical_root_bias=args.tactical_root_bias,
                seed=args.seed,
            )

    carried_medium = ["current_ref"]
    for name, row in candidate_rows.items():
        if name == "current_ref":
            continue
        if name.startswith("value_head_only_") and name != "value_head_only_best_probe":
            continue
        if row.get("probe_gate", {}).get("passed") and row.get(
            "artifact_weights_sha256"
        ):
            carried_medium.append(name)
    fixed_large_summary: dict[str, Any] = {}
    fixed_large_suite_rows: dict[str, Any] = {}
    medium_summary: dict[str, Any] = {}
    medium_suite_rows: dict[str, Any] = {}
    bootstrap_cis: dict[str, dict[str, Any]] = {}
    heldout_summary: dict[str, Any] = {}
    heldout_suite_rows: dict[str, Any] = {}
    gate_result: dict[str, Any] | None = None

    promising_probe_candidates = [
        name
        for name in carried_medium
        if candidate_shows_presuite_promise(
            candidate_name=name, candidate_rows=candidate_rows
        )
    ]
    if promising_probe_candidates:
        medium_candidate_artifacts = candidate_artifacts_from_rows(
            {name: candidate_rows[name] for name in carried_medium}
        )
        start = time.time()
        medium_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_medium"),
            suite=str(medium_suite),
            current=str(current_artifact),
            candidates=",".join(
                str(path) for _name, path in medium_candidate_artifacts
            ),
            budget_pairs=",".join(DEFAULT_BUDGET_PAIRS),
            games_per_opening=2,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
        timings["eval_medium"] = time.time() - start
        medium_suite_rows = suite_rows_for_report(
            report=medium_report,
            current_artifact=current_artifact,
            candidate_artifacts=medium_candidate_artifacts,
            suite_path=medium_suite,
        )
        medium_summary = build_candidate_eval_delta_map(
            medium_suite_rows, carried_medium
        )

        fixed_large_candidates = ["current_ref"]
        medium_ranked: list[tuple[float, str]] = []
        for name in carried_medium:
            if name == "current_ref":
                continue
            budget_map = medium_summary[name]
            if (
                float(budget_map["384:256"]["candidate_minus_current"]) >= 0.03
                and float(budget_map["768:768"]["candidate_minus_current"]) >= -0.03
                and float(budget_map["1200:1200"]["candidate_minus_current"]) >= -0.03
            ):
                medium_ranked.append(
                    (float(budget_map["384:256"]["candidate_minus_current"]), name)
                )
        medium_ranked.sort(reverse=True)
        fixed_large_candidates.extend(name for _score, name in medium_ranked[:2])

        fixed_large_artifacts = candidate_artifacts_from_rows(
            {name: candidate_rows[name] for name in fixed_large_candidates}
        )
        start = time.time()
        fixed_large_report = run_opening_suite_benchmark(
            workdir=str(workdir / "eval_fixed_large"),
            suite=str(fixed_large_suite),
            current=str(current_artifact),
            candidates=",".join(str(path) for _name, path in fixed_large_artifacts),
            budget_pairs=",".join(DEFAULT_BUDGET_PAIRS),
            games_per_opening=2,
            seed=args.seed,
            workers=args.workers,
            timeout=args.timeout,
        )
        timings["eval_fixed_large"] = time.time() - start
        fixed_large_suite_rows = suite_rows_for_report(
            report=fixed_large_report,
            current_artifact=current_artifact,
            candidate_artifacts=fixed_large_artifacts,
            suite_path=fixed_large_suite,
        )
        fixed_large_summary = build_candidate_eval_delta_map(
            fixed_large_suite_rows, fixed_large_candidates
        )
        bootstrap_cis.update(
            build_bootstrap_table(
                suite_rows=fixed_large_suite_rows,
                candidate_names=fixed_large_candidates,
                current_name="current_ref",
                budgets=DEFAULT_BUDGET_PAIRS,
                seed=args.seed,
            )
        )
        heldout_candidate_names = ["current_ref"]
        for name in fixed_large_candidates:
            if name == "current_ref":
                continue
            candidate_budget = fixed_large_summary.get(name, {})
            if not candidate_budget:
                continue
            if (
                float(candidate_budget["384:256"]["candidate_minus_current"]) >= 0.03
                and float(candidate_budget["768:768"]["candidate_minus_current"])
                >= -0.05
                and float(candidate_budget["1200:1200"]["candidate_minus_current"])
                >= -0.03
                and float(candidate_budget["1200:256"]["candidate_minus_current"])
                >= -0.03
            ):
                heldout_candidate_names.append(name)

        if len(heldout_candidate_names) > 1:
            heldout_artifacts = candidate_artifacts_from_rows(
                {name: candidate_rows[name] for name in heldout_candidate_names}
            )
            heldout_parallel_jobs = min(len(heldout_suites), max(1, args.workers // 4))
            heldout_job_workers = max(1, args.workers // heldout_parallel_jobs)
            heldout_jobs = [
                {
                    "name": suite_path.stem,
                    "kwargs": {
                        "workdir": str(workdir / "eval_heldout" / suite_path.stem),
                        "suite": str(suite_path),
                        "current": str(current_artifact),
                        "candidates": ",".join(
                            str(path) for _name, path in heldout_artifacts
                        ),
                        "budget_pairs": ",".join(DEFAULT_BUDGET_PAIRS),
                        "games_per_opening": 2,
                        "seed": args.seed,
                        "workers": heldout_job_workers,
                        "timeout": args.timeout,
                    },
                }
                for suite_path in heldout_suites
            ]
            start = time.time()
            heldout_reports = run_suite_jobs_parallel(
                jobs=heldout_jobs, max_parallel_jobs=heldout_parallel_jobs
            )
            timings["eval_heldout_total"] = time.time() - start
            for suite_path in heldout_suites:
                report = heldout_reports[suite_path.stem]
                heldout_suite_rows.update(
                    suite_rows_for_report(
                        report=report,
                        current_artifact=current_artifact,
                        candidate_artifacts=heldout_artifacts,
                        suite_path=suite_path,
                    )
                )
            for candidate_name in heldout_candidate_names:
                candidate_budget_summary: dict[str, dict[str, float]] = {}
                for budget in DEFAULT_BUDGET_PAIRS:
                    candidate_diffs: list[float] = []
                    for _suite_name, suite_row in heldout_suite_rows.items():
                        candidate_ds = float(
                            suite_row["candidates"][candidate_name]["budget_results"][
                                budget
                            ]["ds"]
                        )
                        current_ds = float(
                            suite_row["candidates"]["current_ref"]["budget_results"][
                                budget
                            ]["ds"]
                        )
                        candidate_diffs.append(candidate_ds - current_ds)
                    candidate_budget_summary[budget] = {
                        "candidate_minus_current": statistics.fmean(candidate_diffs)
                        if candidate_diffs
                        else 0.0
                    }
                heldout_summary[candidate_name] = candidate_budget_summary
            bootstrap_cis.update(
                build_bootstrap_table(
                    suite_rows=heldout_suite_rows,
                    candidate_names=heldout_candidate_names,
                    current_name="current_ref",
                    budgets=DEFAULT_BUDGET_PAIRS,
                    seed=args.seed + 1000,
                )
            )

        gate_candidates = []
        for name in heldout_candidate_names:
            if name == "current_ref":
                continue
            heldout_budget = heldout_summary.get(name, {})
            ci_384 = (
                bootstrap_cis.get(name, {})
                .get("384:256", {})
                .get("candidate_minus_current")
            )
            if not heldout_budget or ci_384 is None:
                continue
            if (
                float(heldout_budget["384:256"]["candidate_minus_current"]) >= 0.05
                and float(ci_384["lower"]) > 0.01
                and float(heldout_budget["768:768"]["candidate_minus_current"]) >= -0.05
                and float(heldout_budget["1200:1200"]["candidate_minus_current"])
                >= -0.03
                and float(heldout_budget["1200:256"]["candidate_minus_current"])
                >= -0.03
            ):
                gate_candidates.append(name)
        if gate_candidates:
            best_candidate = max(
                gate_candidates,
                key=lambda name: float(
                    heldout_summary[name]["384:256"]["candidate_minus_current"]
                ),
            )
            gate_report = run_default_gate(
                candidate_path=str(candidate_rows[best_candidate]["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(workdir / "gate_result.json"),
                seed=args.seed,
                workers=args.workers,
                games=60,
                budget_pairs="384:256,768:256,768:768,1200:1200,1200:256,256:768",
            )
            gate_budgets = gate_report.get("budget_results", {})
            gate_passed = gate_report.get("classification") is not None
            gate_result = {
                "run": True,
                "candidate": best_candidate,
                "classification": gate_report.get("classification"),
                "budget_results": gate_budgets,
                "passed": bool(gate_passed),
            }
        else:
            gate_result = {"run": False, "candidate": None, "passed": False}
    else:
        gate_result = {
            "run": False,
            "candidate": None,
            "passed": False,
            "reason": "no probe-passing candidate cleared pre-suite promise thresholds",
        }
        timings["suite_eval_skipped_after_probe"] = 1.0

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "workdir": str(workdir),
        "inputs": input_summary,
        "replay_path": str(replay_path),
        "train_replay_path": str(train_replay_path),
        "heldout_replay_path": str(heldout_replay_path),
        "replay_audit": audit,
        "runtime_profile_confirmation": {
            "schedule_manifest": schedule_definition(
                default_c_puct=args.default_c_puct, schedule=cpuct_schedule
            ),
            "selfplay_profiles": generation_meta["runtime_profiles"],
        },
        "candidate_rows": candidate_rows,
        "value_search_drift_diagnostic": value_search_drift_diagnostic,
        "medium_summary": medium_summary,
        "fixed_large_summary": fixed_large_summary,
        "heldout_summary": heldout_summary,
        "bootstrap_cis": bootstrap_cis,
        "gate_result": gate_result,
        "runtime_cost": timings,
    }
    summary["final_classification"] = classify_result(summary)
    write_json(summary_path, summary)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
