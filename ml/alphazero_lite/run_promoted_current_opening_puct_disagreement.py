#!/usr/bin/env python3
"""Mine opening-suite PUCT disagreements from promoted current.

Audits evaluation-relevant states encountered on the deterministic opening-suite
distribution, mines disagreement-focused replay rows, trains policy-head-only
continuations from the promoted e1 checkpoint, and evaluates the resulting
artifacts without promoting or overwriting current.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_promoted_current_puct_iter2_smoke import (  # noqa: E402
    DEFAULT_BUDGET_PAIRS,
    budget_results_by_pair,
    candidate_standard_ds,
    compute_param_delta_norm,
    export_checkpoint,
    find_candidate_report,
    heldout_summary,
    run_default_gate,
    run_opening_suite_benchmark,
    run_train,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_policy_target_from_distribution,
    build_search_options,
    derive_self_play_value_target,
    encode_state,
)

EXPECTED_PROMOTED_WEIGHTS_SHA256 = (
    "6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece"
)
EXPECTED_INIT_CHECKPOINT_SHA256 = (
    "a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357"
)
KL_EPSILON = 1e-12
POLICY_SIZE = 6
STANDARD_BUDGET = "384:256"
DEFAULT_TARGET_ROWS = 2000
MIN_TRAINABLE_ROWS = 1000
TARGET_POLICY_MODE = "sharpened"
TARGET_VALUE_MODE = "sharpened"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


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


def top_policy_move(
    policy: list[float] | np.ndarray, legal_moves: list[int]
) -> int | None:
    if not legal_moves:
        return None
    return min(legal_moves, key=lambda move: (-float(policy[move]), move))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    safe_p = np.clip(np.asarray(p, dtype=np.float64), KL_EPSILON, 1.0)
    safe_q = np.clip(np.asarray(q, dtype=np.float64), KL_EPSILON, 1.0)
    safe_p /= np.sum(safe_p)
    safe_q /= np.sum(safe_q)
    return float(np.sum(safe_p * np.log(safe_p / safe_q)))


def canonical_state_hash(state: dict[str, Any]) -> str:
    encoded = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def search_policy_from_visits(
    visits: list[float] | np.ndarray, legal_moves: list[int]
) -> list[float]:
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    if not legal_moves:
        return policy.tolist()
    visits_array = np.asarray(visits, dtype=np.float64)
    total = float(np.sum(visits_array[legal_moves]))
    if total <= 0:
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            policy[move] = uniform
        return policy.tolist()
    for move in legal_moves:
        policy[move] = float(visits_array[move] / total)
    return policy.tolist()


def policy_entropy(policy: list[float] | np.ndarray, legal_moves: list[int]) -> float:
    entropy = 0.0
    for move in legal_moves:
        probability = float(policy[move])
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return entropy


def raw_margin(policy: list[float] | np.ndarray, legal_moves: list[int]) -> float:
    if len(legal_moves) < 2:
        return 1.0
    ranked = sorted((float(policy[move]) for move in legal_moves), reverse=True)
    return float(ranked[0] - ranked[1])


def phase_label(*, total_ply: int, game: KalahGame) -> str:
    if total_ply <= 8:
        return "opening"
    seeds_remaining = sum(int(value) for value in game.pits)
    if seeds_remaining <= 16:
        return "late"
    return "mid"


def margin_bucket(margin: float) -> str:
    if margin < 0.02:
        return "margin < 0.02"
    if margin < 0.05:
        return "0.02 <= margin < 0.05"
    if margin < 0.10:
        return "0.05 <= margin < 0.10"
    return "margin >= 0.10"


def load_suite(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def artifact_forward_details(
    evaluator: ArtifactEvaluator, game: KalahGame
) -> tuple[list[float], list[float], float]:
    x = np.asarray(
        encode_state(game.to_state(), input_encoding=evaluator.input_encoding),
        dtype=np.float32,
    )
    if evaluator.residual_blocks:
        assert evaluator.w_input is not None
        assert evaluator.b_input is not None
        hidden = np.maximum(0.0, (x @ evaluator.w_input) + evaluator.b_input)
        for (w1, b1), (w2, b2) in evaluator.residual_blocks:
            residual = hidden
            hidden = np.maximum(0.0, (hidden @ w1) + b1)
            hidden = np.maximum(0.0, (hidden @ w2) + b2 + residual)
    else:
        hidden = x
        for w, b in evaluator.hidden_layers:
            hidden = np.maximum(0.0, (hidden @ w) + b)
    policy_hidden = hidden
    value_hidden = hidden
    if evaluator.model_type in ("residual_v3", "residual_v4_move_factorized"):
        assert evaluator.w_policy_hidden is not None
        assert evaluator.b_policy_hidden is not None
        assert evaluator.w_value_hidden is not None
        assert evaluator.b_value_hidden is not None
        policy_hidden = np.maximum(
            0.0, (hidden @ evaluator.w_policy_hidden) + evaluator.b_policy_hidden
        )
        value_hidden = np.maximum(
            0.0, (hidden @ evaluator.w_value_hidden) + evaluator.b_value_hidden
        )
    if evaluator.move_factorized:
        assert evaluator.w_policy_move is not None
        assert evaluator.b_policy_move is not None
        logits = np.asarray(
            [
                float((policy_hidden @ w_move).reshape(-1)[0] + b_move.reshape(-1)[0])
                for w_move, b_move in zip(
                    evaluator.w_policy_move, evaluator.b_policy_move
                )
            ],
            dtype=np.float32,
        )
    else:
        assert evaluator.w_policy is not None
        assert evaluator.b_policy is not None
        logits = (policy_hidden @ evaluator.w_policy) + evaluator.b_policy
    raw_probs = np.zeros(POLICY_SIZE, dtype=np.float32)
    legal_moves = game.possible_moves()
    if legal_moves:
        shifted = logits - np.max(logits[legal_moves])
        exp_values = np.exp(shifted)
        raw_probs[legal_moves] = exp_values[legal_moves]
        total = float(np.sum(raw_probs[legal_moves]))
        if total <= 0.0:
            raw_probs[legal_moves] = 1.0 / len(legal_moves)
        else:
            raw_probs[legal_moves] /= total
    value = float(
        np.tanh((value_hidden @ evaluator.w_value + evaluator.b_value).reshape(-1)[0])
    )
    return (
        logits.astype(np.float32).tolist(),
        raw_probs.astype(np.float32).tolist(),
        value,
    )


def build_eval_game_trace(
    *,
    suite_name: str,
    opening_entry: dict[str, Any],
    challenger_player: int,
    evaluator: ArtifactEvaluator,
    challenger_simulations: int,
    current_simulations: int,
    eval_search_options: dict[str, Any],
    c_puct: float,
    seed: int,
    game_index: int,
) -> dict[str, Any]:
    game = KalahGame.from_state(opening_entry["state"])
    total_ply = int(
        opening_entry.get("ply", len(opening_entry.get("prefix_moves", [])))
    )
    encountered_states: list[dict[str, Any]] = []
    trajectory: list[int] = []
    reusable_roots: dict[int, Any] = {0: None, 1: None}
    rng = np.random.default_rng(seed + game_index)
    while not game.over():
        legal_moves = game.possible_moves()
        if not legal_moves:
            break
        acting_role = (
            "challenger" if game.current_player == challenger_player else "current"
        )
        state = game.to_state()
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
                "phase": phase_label(total_ply=total_ply, game=game),
                "side_to_move": int(game.current_player),
                "challenger_player": int(challenger_player),
                "acting_role": acting_role,
                "state": state,
                "state_hash": canonical_state_hash(state),
                "legal_moves": [int(move) for move in legal_moves],
            }
        )
        simulations = (
            challenger_simulations
            if acting_role == "challenger"
            else current_simulations
        )
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
    challenger_store = int(game.captured_seeds[challenger_player])
    current_store = int(game.captured_seeds[1 - challenger_player])
    if challenger_store > current_store:
        winner = "challenger"
    elif challenger_store < current_store:
        winner = "current"
    else:
        winner = "draw"
    return {
        "suite_name": suite_name,
        "opening_prefix": [int(move) for move in opening_entry.get("prefix_moves", [])],
        "challenger_player": int(challenger_player),
        "winner": winner,
        "margin": challenger_store - current_store,
        "trajectory": ",".join(str(move) for move in trajectory),
        "states": encountered_states,
    }


def collect_evaluation_states(
    *,
    suite_specs: list[tuple[str, Path]],
    artifact_path: Path,
    c_puct: float,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evaluator = ArtifactEvaluator(artifact_path)
    eval_search_options = build_eval_search_options(root_policy_mode="deterministic")
    state_index: dict[str, dict[str, Any]] = {}
    per_suite_games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    poor_p0_hashes: set[str] = set()
    trajectory_counter: Counter[str] = Counter()
    total_game_index = 0
    for suite_name, suite_path in suite_specs:
        for opening_index, opening_entry in enumerate(load_suite(suite_path)):
            del opening_index
            for challenger_player in (0, 1):
                trace = build_eval_game_trace(
                    suite_name=suite_name,
                    opening_entry=opening_entry,
                    challenger_player=challenger_player,
                    evaluator=evaluator,
                    challenger_simulations=384,
                    current_simulations=256,
                    eval_search_options=eval_search_options,
                    c_puct=c_puct,
                    seed=seed,
                    game_index=total_game_index,
                )
                total_game_index += 1
                per_suite_games[suite_name].append(trace)
                trajectory_counter[trace["trajectory"]] += 1
                poor_p0 = (
                    suite_name == "fixed_large"
                    and trace["challenger_player"] == 0
                    and trace["winner"] != "challenger"
                )
                for state_entry in trace["states"]:
                    if poor_p0:
                        poor_p0_hashes.add(str(state_entry["state_hash"]))
                    existing = state_index.get(str(state_entry["state_hash"]))
                    if existing is None:
                        state_index[str(state_entry["state_hash"])] = {
                            **state_entry,
                            "suite_names": [state_entry["suite_name"]],
                            "occurrence_count": 1,
                            "seat_contexts": Counter([str(state_entry["acting_role"])]),
                            "challenger_player_contexts": Counter(
                                [str(state_entry["challenger_player"])]
                            ),
                            "poor_384_256_p0_game": poor_p0,
                        }
                        continue
                    existing["occurrence_count"] = int(existing["occurrence_count"]) + 1
                    if state_entry["suite_name"] not in existing["suite_names"]:
                        existing["suite_names"].append(state_entry["suite_name"])
                    existing["seat_contexts"][str(state_entry["acting_role"])] += 1
                    existing["challenger_player_contexts"][
                        str(state_entry["challenger_player"])
                    ] += 1
                    existing["poor_384_256_p0_game"] = bool(
                        existing["poor_384_256_p0_game"] or poor_p0
                    )
    for state_entry in state_index.values():
        state_entry["seat_contexts"] = dict(state_entry["seat_contexts"])
        state_entry["challenger_player_contexts"] = dict(
            state_entry["challenger_player_contexts"]
        )
        state_entry["suite_names"].sort()
    duplicate_trajectory_count = sum(
        count for count in trajectory_counter.values() if count > 1
    )
    metadata = {
        "suite_game_counts": {
            suite_name: len(games) for suite_name, games in per_suite_games.items()
        },
        "total_games": sum(len(games) for games in per_suite_games.values()),
        "duplicate_trajectory_count": duplicate_trajectory_count,
        "duplicate_trajectory_rate": duplicate_trajectory_count
        / max(sum(len(games) for games in per_suite_games.values()), 1),
        "poor_384_256_p0_game_state_count": len(poor_p0_hashes),
    }
    return state_index, metadata


def analyze_state_batch(
    *,
    batch: list[dict[str, Any]],
    artifact_path: str,
    simulations: int,
    c_puct: float,
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path))
    search_options = build_search_options(
        root_policy_mode="visit_count", tactical_root_bias=0.0
    )
    results: list[dict[str, Any]] = []
    for index, state_entry in enumerate(batch):
        game = KalahGame.from_state(state_entry["state"])
        legal_moves = [int(move) for move in game.possible_moves()]
        logits, raw_policy, raw_value = artifact_forward_details(evaluator, game)
        raw_top1 = top_policy_move(raw_policy, legal_moves)
        search = PUCT(
            evaluator=evaluator,
            simulations=int(simulations),
            c_puct=float(c_puct),
            rng=random.Random(seed + index),
            fpu_mode=str(search_options["fpu_mode"]),
            reuse_subtree=bool(search_options["reuse_subtree"]),
            normalize_values=bool(search_options["normalize_values"]),
            root_policy_mode=str(search_options["root_policy_mode"]),
            tactical_root_bias=float(search_options["tactical_root_bias"]),
            root_temperature=float(search_options["root_temperature"]),
        )
        visits, root = search.run(game)
        search_policy = search_policy_from_visits(visits, legal_moves)
        search_top1 = top_policy_move(search_policy, legal_moves)
        legal_raw = np.asarray(
            [raw_policy[move] for move in legal_moves], dtype=np.float64
        )
        legal_search = np.asarray(
            [search_policy[move] for move in legal_moves], dtype=np.float64
        )
        search_visit_total = int(
            sum(int(round(float(visits[move]))) for move in legal_moves)
        )
        search_top1_visit_share = (
            float(search_policy[search_top1]) if search_top1 is not None else 0.0
        )
        preferred = (
            search_top1 is not None
            and raw_top1 is not None
            and search_top1 != raw_top1
            and search_top1_visit_share >= 0.55
            and len(legal_moves) > 1
        )
        results.append(
            {
                **state_entry,
                "legal_moves": legal_moves,
                "raw_logits": [float(value) for value in logits],
                "raw_policy": [float(value) for value in raw_policy],
                "raw_value": float(raw_value),
                "raw_top1": raw_top1,
                "raw_margin": raw_margin(raw_policy, legal_moves),
                "search_policy": [float(value) for value in search_policy],
                "search_value": float(root.q_value if root is not None else 0.0),
                "search_top1": search_top1,
                "top1_changed": bool(search_top1 != raw_top1),
                "kl_search_raw": kl_divergence(legal_search, legal_raw),
                "kl_raw_search": kl_divergence(legal_raw, legal_search),
                "search_entropy": policy_entropy(search_policy, legal_moves),
                "visit_count_total": search_visit_total,
                "search_top1_visit_share": float(search_top1_visit_share),
                "high_confidence_disagreement": bool(
                    search_top1 != raw_top1 and search_top1_visit_share >= 0.55
                ),
                "trainable_candidate": bool(len(legal_moves) > 1),
                "preferred_disagreement": bool(preferred),
            }
        )
    return results


def partition_batches(
    items: list[dict[str, Any]], workers: int
) -> list[list[dict[str, Any]]]:
    workers = max(1, min(workers, len(items) or 1))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    for index, item in enumerate(items):
        buckets[index % workers].append(item)
    return [bucket for bucket in buckets if bucket]


def summarize_disagreement_audit(
    analyzed_states: list[dict[str, Any]], collection_metadata: dict[str, Any]
) -> dict[str, Any]:
    total = len(analyzed_states)
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_margin_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    poor_p0_rows: list[dict[str, Any]] = []
    for row in analyzed_states:
        for suite_name in row.get("suite_names", []):
            by_suite[str(suite_name)].append(row)
        by_phase[str(row["phase"])].append(row)
        dominant_seat = max(
            row.get("seat_contexts", {"unknown": 1}).items(),
            key=lambda item: (int(item[1]), item[0]),
        )[0]
        by_seat[str(dominant_seat)].append(row)
        by_margin_bucket[margin_bucket(float(row["raw_margin"]))].append(row)
        if bool(row.get("poor_384_256_p0_game")):
            poor_p0_rows.append(row)

    def disagreement_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        changed = sum(1 for row in rows if row["top1_changed"])
        return {
            "states": len(rows),
            "changed_top1": changed,
            "rate": changed / max(len(rows), 1),
        }

    def mean_kl(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        if not rows:
            return {"kl_search_raw": None, "kl_raw_search": None}
        return {
            "kl_search_raw": statistics.fmean(
                float(row["kl_search_raw"]) for row in rows
            ),
            "kl_raw_search": statistics.fmean(
                float(row["kl_raw_search"]) for row in rows
            ),
        }

    phase_seat_mean_kl: dict[str, dict[str, Any]] = {}
    for phase_name, phase_rows in by_phase.items():
        phase_seat_mean_kl[phase_name] = {}
        for seat_name, seat_rows in by_seat.items():
            filtered = [row for row in phase_rows if row in seat_rows]
            phase_seat_mean_kl[phase_name][seat_name] = mean_kl(filtered)

    audit_tables = {
        "overall_disagreement_rate": disagreement_rate(analyzed_states),
        "disagreement_rate_by_suite": {
            suite_name: disagreement_rate(rows)
            for suite_name, rows in sorted(by_suite.items())
        },
        "disagreement_rate_by_phase": {
            phase_name: disagreement_rate(rows)
            for phase_name, rows in sorted(by_phase.items())
        },
        "disagreement_rate_by_seat_context": {
            seat_name: disagreement_rate(rows)
            for seat_name, rows in sorted(by_seat.items())
        },
        "disagreement_rate_poor_384_256_p0_games": disagreement_rate(poor_p0_rows),
        "top_raw_margin_buckets": {
            bucket: {
                **disagreement_rate(rows),
                "mean_kl_search_raw": statistics.fmean(
                    float(row["kl_search_raw"]) for row in rows
                )
                if rows
                else None,
            }
            for bucket, rows in by_margin_bucket.items()
        },
        "mean_kl_by_phase_and_seat": phase_seat_mean_kl,
        "high_confidence_disagreements": sum(
            1 for row in analyzed_states if row["high_confidence_disagreement"]
        ),
        "collection_metadata": collection_metadata,
    }
    return {
        "tables": audit_tables,
        "states": analyzed_states,
        "summary": {
            "unique_state_count": total,
            "high_confidence_disagreements": audit_tables[
                "high_confidence_disagreements"
            ],
        },
    }


def select_replay_rows(
    analyzed_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        row for row in analyzed_states if bool(row.get("trainable_candidate"))
    ]
    for row in candidates:
        row["selection_tier"] = "skip"
        if bool(row["preferred_disagreement"]):
            row["selection_tier"] = "preferred_disagreement"
        elif row["top1_changed"]:
            row["selection_tier"] = "fallback_kl"
        elif (
            row["search_top1"] == row["raw_top1"]
            and float(row["raw_margin"]) < 0.05
            and float(row["search_top1_visit_share"]) >= 0.70
        ):
            row["selection_tier"] = "fallback_sharpen"
    candidates = [row for row in candidates if row["selection_tier"] != "skip"]
    candidates.sort(
        key=lambda row: (
            {"preferred_disagreement": 0, "fallback_kl": 1, "fallback_sharpen": 2}[
                row["selection_tier"]
            ],
            -float(row["search_top1_visit_share"]),
            -float(row["kl_search_raw"]),
            float(row["raw_margin"]),
            int(row["move_index"]),
            str(row["state_hash"]),
        )
    )
    target_rows = min(len(candidates), DEFAULT_TARGET_ROWS)
    phase_caps = {"opening": 1200, "mid": 600, "late": 200}
    prefix_cap = 12
    top_move_cap = 420
    prefix_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    top_move_counts: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    skipped_for_caps = []
    for row in candidates:
        prefix_key = str(row.get("opening_prefix_text") or "")
        phase_key = str(row["phase"])
        top_move = int(row["search_top1"]) if row["search_top1"] is not None else -1
        if prefix_counts[prefix_key] >= prefix_cap:
            skipped_for_caps.append(row)
            continue
        if phase_counts[phase_key] >= phase_caps.get(phase_key, target_rows):
            skipped_for_caps.append(row)
            continue
        if top_move >= 0 and top_move_counts[top_move] >= top_move_cap:
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
            selected.append(row)
            if len(selected) >= target_rows:
                break
    summary = {
        "target_rows": DEFAULT_TARGET_ROWS,
        "selected_rows": len(selected),
        "preferred_rows": sum(
            1 for row in selected if row["selection_tier"] == "preferred_disagreement"
        ),
        "fallback_kl_rows": sum(
            1 for row in selected if row["selection_tier"] == "fallback_kl"
        ),
        "fallback_sharpen_rows": sum(
            1 for row in selected if row["selection_tier"] == "fallback_sharpen"
        ),
        "phase_counts": dict(phase_counts),
        "top_move_counts": {
            str(key): value for key, value in sorted(top_move_counts.items())
        },
        "prefix_cap": prefix_cap,
        "top_move_cap": top_move_cap,
        "phase_caps": phase_caps,
    }
    return selected, summary


def continue_state_batch(
    *,
    batch: list[dict[str, Any]],
    artifact_path: str,
    simulations: int,
    c_puct: float,
    seed: int,
) -> list[dict[str, Any]]:
    evaluator = ArtifactEvaluator(Path(artifact_path))
    search_options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0
    )
    rows: list[dict[str, Any]] = []
    for index, state_entry in enumerate(batch):
        game = KalahGame.from_state(state_entry["state"])
        root_player = int(game.current_player)
        rng = np.random.default_rng(seed + index)
        search_value = float(state_entry["search_value"])
        policy_target = build_policy_target_from_distribution(
            state_entry["search_policy"], mode=TARGET_POLICY_MODE
        )
        outcome = 0.0
        while not game.over():
            legal_moves = game.possible_moves()
            if not legal_moves:
                break
            search = PUCT(
                evaluator=evaluator,
                simulations=int(simulations),
                c_puct=float(c_puct),
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
        rows.append(
            {
                "state_hash": state_entry["state_hash"],
                "state": encode_state(state_entry["state"], input_encoding="kalah_v3"),
                "policy": [float(value) for value in policy_target],
                "value": derive_self_play_value_target(
                    outcome_value=outcome,
                    search_value=search_value,
                    move_index=int(state_entry["move_index"]),
                    mode=TARGET_VALUE_MODE,
                ),
                "player": root_player,
                "move_index": int(state_entry["move_index"]),
                "winner": game.winner,
                "legal_moves": [int(move) for move in state_entry["legal_moves"]],
                "top_target_move": state_entry["search_top1"],
                "policy_target_mode": TARGET_POLICY_MODE,
                "policy_target_actual_mode": TARGET_POLICY_MODE,
                "value_target_mode": TARGET_VALUE_MODE,
                "teacher_source": "opening_puct_disagreement",
                "bucket": "opening_puct_disagreement",
                "suite_names": list(state_entry.get("suite_names", [])),
                "suite_name": state_entry["suite_name"],
                "opening_prefix": list(state_entry.get("opening_prefix", [])),
                "phase": state_entry["phase"],
                "selection_tier": state_entry["selection_tier"],
                "search_top1": state_entry["search_top1"],
                "raw_top1": state_entry["raw_top1"],
                "search_top1_visit_share": state_entry["search_top1_visit_share"],
                "raw_margin": state_entry["raw_margin"],
                "kl_search_raw": state_entry["kl_search_raw"],
                "continuation_outcome": outcome,
                "continuation_winner": game.winner,
            }
        )
    return rows


def mined_state_policy_shift(
    candidate_artifact: Path, mined_states: list[dict[str, Any]]
) -> dict[str, Any]:
    evaluator = ArtifactEvaluator(candidate_artifact)
    changed = 0
    aligns_with_search = 0
    for row in mined_states:
        game = KalahGame.from_state(row["state"])
        _logits, raw_policy, _value = artifact_forward_details(evaluator, game)
        top1 = top_policy_move(raw_policy, row["legal_moves"])
        if top1 != row["raw_top1"]:
            changed += 1
        if top1 == row["search_top1"]:
            aligns_with_search += 1
    total = max(len(mined_states), 1)
    return {
        "states": len(mined_states),
        "top1_changed_rate_vs_promoted_current": changed / total,
        "top1_matches_search_rate": aligns_with_search / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement",
    )
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
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--heldout-suites",
        default=(
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl"
        ),
    )
    parser.add_argument("--simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    current_artifact = Path(args.current)
    init_checkpoint = Path(args.init_checkpoint)
    generic_bootstrap = Path(args.generic_bootstrap)
    random_teacher = Path(args.random_teacher)
    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suite_paths = [
        Path(item.strip())
        for item in str(args.heldout_suites).split(",")
        if item.strip()
    ]
    heldout_suite_paths = [path for path in heldout_suite_paths if path.is_file()]
    iter2_ref_checkpoint = Path(
        "/tmp/azlite_promoted_current_puct_iter2/iter2_puct_policy_head_e2/checkpoint_epoch2.npz"
    )
    iter2_ref_artifact = Path(
        "/tmp/azlite_promoted_current_puct_iter2/iter2_puct_policy_head_e2/artifact_iter2_puct_policy_head_e2"
    )
    audit_path = workdir / "opening_state_disagreement_audit.json"
    replay_path = workdir / "opening_puct_disagreement_replay.jsonl"

    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(
        current_artifact / "metadata.json", "current artifact metadata"
    )
    require_existing_file(init_checkpoint, "init checkpoint")
    require_existing_file(generic_bootstrap, "generic bootstrap replay")
    require_existing_file(random_teacher, "random teacher replay")
    require_existing_file(fixed_large_suite, "fixed large suite")
    require_existing_file(medium_suite, "medium suite")
    require_existing_file(iter2_ref_checkpoint, "iter2 reference checkpoint")

    if not iter2_ref_artifact.joinpath("weights.json").is_file():
        export_checkpoint(
            checkpoint_path=str(iter2_ref_checkpoint),
            out_dir=str(iter2_ref_artifact),
            version="iter2_puct_policy_head_e2_ref",
            policy_loss=0.0,
            value_loss=0.0,
        )

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
        "medium_suite": build_input_summary(medium_suite),
        "fixed_large_suite": build_input_summary(fixed_large_suite),
        "heldout_suites": {
            path.stem: build_input_summary(path) for path in heldout_suite_paths
        },
        "iter2_selfplay_e2_checkpoint": build_input_summary(iter2_ref_checkpoint),
        "iter2_selfplay_e2_artifact_weights": build_input_summary(
            iter2_ref_artifact / "weights.json"
        ),
    }

    suite_specs = [("fixed_large", fixed_large_suite)] + [
        (path.stem, path) for path in heldout_suite_paths
    ]
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        analyzed_states = list(audit.get("states", []))
    else:
        state_index, collection_metadata = collect_evaluation_states(
            suite_specs=suite_specs,
            artifact_path=current_artifact,
            c_puct=args.c_puct,
            seed=args.seed,
        )
        unique_states = list(state_index.values())
        state_batches = partition_batches(unique_states, args.workers)
        analyzed_states = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(state_batches)))
        ) as pool:
            futures = [
                pool.submit(
                    analyze_state_batch,
                    batch=batch,
                    artifact_path=str(current_artifact),
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    seed=args.seed + batch_index * 1000,
                )
                for batch_index, batch in enumerate(state_batches)
            ]
            for future in futures:
                analyzed_states.extend(future.result())
        analyzed_states.sort(key=lambda row: str(row["state_hash"]))
        audit = summarize_disagreement_audit(analyzed_states, collection_metadata)

    selected_states, selection_summary = select_replay_rows(analyzed_states)
    audit["tables"]["trainable_disagreement_rows_after_dedupe"] = len(selected_states)
    audit["tables"]["selection_summary"] = selection_summary
    write_json(audit_path, audit)

    if len(selected_states) < MIN_TRAINABLE_ROWS:
        summary = {
            "schema": "azlite_promoted_current_opening_puct_disagreement_v1",
            "status": "aborted_before_training",
            "classification": "eval_search_saturated"
            if (
                float(audit["tables"]["overall_disagreement_rate"]["rate"]) < 0.05
                and int(audit["tables"]["high_confidence_disagreements"]) < 500
            )
            else "insufficient_trainable_rows",
            "workdir": str(workdir),
            "inputs": input_summary,
            "opening_state_disagreement_audit": audit,
        }
        write_json(workdir / "summary_metrics.json", summary)
        return 1

    if replay_path.is_file():
        replay_rows = read_jsonl(replay_path)
    else:
        continuation_batches = partition_batches(selected_states, args.workers)
        replay_rows = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, min(args.workers, len(continuation_batches)))
        ) as pool:
            futures = [
                pool.submit(
                    continue_state_batch,
                    batch=batch,
                    artifact_path=str(current_artifact),
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    seed=args.seed + 50_000 + batch_index * 1000,
                )
                for batch_index, batch in enumerate(continuation_batches)
            ]
            for future in futures:
                replay_rows.extend(future.result())
        replay_rows.sort(key=lambda row: str(row["state_hash"]))
        write_jsonl(replay_path, replay_rows)

    lane_specs = [
        {
            "name": "opening_disagreement_policy_head_e1",
            "epochs": 1,
        },
        {
            "name": "opening_disagreement_policy_head_e2",
            "epochs": 2,
        },
    ]
    lanes: list[dict[str, Any]] = [
        {
            "name": "promoted_current_ref",
            "report_candidate_name": "current",
            "epochs": 0,
            "checkpoint_path": str(init_checkpoint),
            "artifact_dir": str(current_artifact),
            "trainable_scope": "none",
            "source": "promoted_current_ref",
        },
        {
            "name": "iter2_selfplay_e2_ref",
            "report_candidate_name": iter2_ref_artifact.name,
            "epochs": 0,
            "checkpoint_path": str(iter2_ref_checkpoint),
            "artifact_dir": str(iter2_ref_artifact),
            "trainable_scope": "none",
            "source": "iter2_selfplay_e2_ref",
        },
    ]

    for lane_spec in lane_specs:
        lane_dir = workdir / lane_spec["name"]
        lane_dir.mkdir(parents=True, exist_ok=True)
        epoch_checkpoint_path = lane_dir / f"checkpoint_epoch{lane_spec['epochs']}.npz"
        export_dir = lane_dir / f"artifact_{lane_spec['name']}"
        train_metrics: dict[str, Any] | None = None
        if (
            export_dir.joinpath("weights.json").is_file()
            and epoch_checkpoint_path.is_file()
        ):
            train_metrics = None
        elif not args.skip_training:
            checkpoint_out = lane_dir / "checkpoint.npz"
            train_metrics = run_train(
                data_files=f"{generic_bootstrap},{random_teacher},{replay_path}",
                replay_weights="4,1,1",
                init_checkpoint=str(init_checkpoint),
                out=str(checkpoint_out),
                top_k_dir=str(lane_dir),
                epochs=int(lane_spec["epochs"]),
                seed=args.seed,
            )
            export_checkpoint(
                checkpoint_path=str(epoch_checkpoint_path),
                out_dir=str(export_dir),
                version=str(lane_spec["name"]),
                policy_loss=float((train_metrics or {}).get("policy_loss", 0.0)),
                value_loss=float((train_metrics or {}).get("value_loss", 0.0)),
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
                "source": "opening_puct_disagreement",
                "replay_weights": "4,1,1",
                "data_files": [
                    str(generic_bootstrap),
                    str(random_teacher),
                    str(replay_path),
                ],
                "train_metrics": train_metrics,
            }
        )

    candidate_paths = ",".join(str(lane["artifact_dir"]) for lane in lanes)
    medium_report: dict[str, Any] | None = None
    large_report: dict[str, Any] | None = None
    heldout_reports: dict[str, dict[str, Any]] = {}
    if not args.skip_eval:
        medium_report_path = (
            workdir / "eval_medium" / "temperature_benchmark_report.json"
        )
        if medium_report_path.is_file():
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
        if large_report_path.is_file():
            large_report = json.loads(large_report_path.read_text(encoding="utf-8"))
        else:
            large_report = run_opening_suite_benchmark(
                workdir=str(workdir / "eval_large"),
                suite=str(fixed_large_suite),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.timeout,
            )
        for suite_name, suite_path in suite_specs[1:]:
            heldout_report_path = (
                workdir / f"eval_{suite_name}" / "temperature_benchmark_report.json"
            )
            if heldout_report_path.is_file():
                heldout_reports[suite_name] = json.loads(
                    heldout_report_path.read_text(encoding="utf-8")
                )
            else:
                heldout_reports[suite_name] = run_opening_suite_benchmark(
                    workdir=str(workdir / f"eval_{suite_name}"),
                    suite=str(suite_path),
                    current=str(current_artifact),
                    candidates=candidate_paths,
                    budget_pairs=args.budget_pairs,
                    games_per_opening=args.games_per_opening,
                    seed=args.seed,
                    workers=args.workers,
                    timeout=args.timeout,
                )

    gate_reports: dict[str, dict[str, Any]] = {}
    promoted_ref_score = candidate_standard_ds(large_report or {}, "current")
    gate_targets = ["promoted_current_ref"]
    for lane in lanes:
        if lane["name"] == "promoted_current_ref" or large_report is None:
            continue
        score = candidate_standard_ds(large_report, str(lane["report_candidate_name"]))
        if score >= promoted_ref_score + 0.01:
            gate_targets.append(str(lane["name"]))
    if not args.skip_gate and large_report is not None:
        gate_dir = workdir / "eval_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        for lane in lanes:
            lane_name = str(lane["name"])
            if lane_name not in gate_targets:
                continue
            gate_reports[lane_name] = run_default_gate(
                candidate_path=str(lane["artifact_dir"]),
                current_path=str(current_artifact),
                out=str(gate_dir / f"{lane_name}_default_gate.json"),
                seed=args.seed,
                workers=args.workers,
            )

    mined_policy_shifts = {
        str(lane["name"]): mined_state_policy_shift(
            Path(str(lane["artifact_dir"])), selected_states
        )
        for lane in lanes
    }

    summary_candidates: list[dict[str, Any]] = []
    for lane in lanes:
        checkpoint_path = Path(str(lane["checkpoint_path"]))
        artifact_dir = Path(str(lane["artifact_dir"]))
        delta_norm, relative_delta_pct = compute_param_delta_norm(
            checkpoint_path, init_checkpoint
        )
        row: dict[str, Any] = {
            "candidate": lane["name"],
            "report_candidate_name": lane["report_candidate_name"],
            "source": lane["source"],
            "epochs": lane["epochs"],
            "checkpoint_path": str(checkpoint_path),
            "artifact_dir": str(artifact_dir),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_weights_sha256": sha256_file(artifact_dir / "weights.json"),
            "delta_norm_vs_promoted_e1": delta_norm,
            "relative_delta_pct_vs_promoted_e1": relative_delta_pct,
            "mined_state_policy_shift": mined_policy_shifts[str(lane["name"])],
        }
        train_metrics = lane.get("train_metrics")
        if isinstance(train_metrics, dict):
            row["policy_loss"] = train_metrics.get("policy_loss")
            row["value_loss"] = train_metrics.get("value_loss")
            row["validation_loss"] = train_metrics.get("best_val_loss")
        if medium_report is not None:
            candidate_report = find_candidate_report(
                medium_report, str(lane["report_candidate_name"])
            )
            if candidate_report is not None:
                row["medium_budget_results"] = budget_results_by_pair(candidate_report)
        if large_report is not None:
            candidate_report = find_candidate_report(
                large_report, str(lane["report_candidate_name"])
            )
            if candidate_report is not None:
                row["large_budget_results"] = budget_results_by_pair(candidate_report)
        if heldout_reports:
            row["heldout_summary"] = heldout_summary(
                heldout_reports, str(lane["report_candidate_name"])
            )
        if lane["name"] in gate_reports:
            row["default_gate"] = {
                "classification": gate_reports[lane["name"]].get("classification")
            }
        summary_candidates.append(row)

    promoted_row = next(
        row for row in summary_candidates if row["candidate"] == "promoted_current_ref"
    )
    trained_rows = [
        row
        for row in summary_candidates
        if row["candidate"].startswith("opening_disagreement_")
    ]
    best_trained = max(
        trained_rows,
        key=lambda row: float(
            row.get("large_budget_results", {})
            .get(STANDARD_BUDGET, {})
            .get("ds", float("-inf"))
        ),
    )
    promoted_large = promoted_row.get("large_budget_results", {})
    best_large = best_trained.get("large_budget_results", {})
    promoted_384 = float(
        promoted_large.get(STANDARD_BUDGET, {}).get("ds", float("-inf"))
    )
    best_384 = float(best_large.get(STANDARD_BUDGET, {}).get("ds", float("-inf")))
    best_1200_1200 = float(best_large.get("1200:1200", {}).get("ds", float("-inf")))
    promoted_1200_1200 = float(
        promoted_large.get("1200:1200", {}).get("ds", float("-inf"))
    )
    best_1200_256 = float(best_large.get("1200:256", {}).get("ds", float("-inf")))
    promoted_1200_256 = float(
        promoted_large.get("1200:256", {}).get("ds", float("-inf"))
    )
    heldout_promoted = (
        heldout_summary(heldout_reports, "current")
        if heldout_reports
        else {"available": False}
    )
    heldout_best = (
        heldout_summary(heldout_reports, str(best_trained["report_candidate_name"]))
        if heldout_reports
        else {"available": False}
    )
    disagreement_rate = float(audit["tables"]["overall_disagreement_rate"]["rate"])
    high_conf_disagreements = int(audit["tables"]["high_confidence_disagreements"])
    trained_top1_changed = float(
        best_trained["mined_state_policy_shift"][
            "top1_changed_rate_vs_promoted_current"
        ]
    )
    heldout_mean_delta = None
    if heldout_promoted.get("available") and heldout_best.get("available"):
        heldout_mean_delta = float(
            heldout_best["mean_ds_384_256"] - heldout_promoted["mean_ds_384_256"]
        )
    classification = "no_useful_signal"
    if disagreement_rate < 0.05 and high_conf_disagreements < 500:
        classification = "eval_search_saturated"
    elif (
        disagreement_rate >= 0.10
        and trained_top1_changed < 0.05
        and best_384 <= promoted_384
    ):
        classification = "training_update_too_weak"
    elif (
        best_384 - promoted_384 >= 0.03
        and best_1200_1200 - promoted_1200_1200 >= -0.03
        and best_1200_256 - promoted_1200_256 >= -0.03
        and heldout_mean_delta is not None
        and heldout_mean_delta >= 0.02
    ):
        classification = "targeted_replay_improvement"
    elif best_384 - promoted_384 > 0.0 and (
        heldout_mean_delta is None or heldout_mean_delta <= 0.0
    ):
        classification = "targeted_replay_overfit"
    elif (
        disagreement_rate >= 0.05
        and trained_top1_changed > 0.0
        and best_384 <= promoted_384
    ):
        classification = "no_useful_signal"

    summary = {
        "schema": "azlite_promoted_current_opening_puct_disagreement_v1",
        "status": "completed",
        "classification": classification,
        "workdir": str(workdir),
        "seed": args.seed,
        "workers": args.workers,
        "budget_pairs": args.budget_pairs.split(","),
        "games_per_opening": args.games_per_opening,
        "inputs": input_summary,
        "guardrails": {
            "promotion": False,
            "overwrite_current": False,
            "full_network_training": False,
            "last_block_policy_training": False,
            "residual_v4": False,
            "architecture_change": False,
            "classic_mcts_replay": False,
            "seed_sweep": False,
            "threshold_change": False,
        },
        "opening_state_disagreement_audit": audit,
        "mined_state_count": len(selected_states),
        "training_row_count": len(replay_rows),
        "replay_path": str(replay_path),
        "gate_targets": gate_targets,
        "heldout_evaluation": {
            "promoted_current_ref": heldout_promoted,
            "best_trained": heldout_best,
        },
        "candidates": summary_candidates,
    }
    write_json(workdir / "summary_metrics.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
