#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.alphazero_lite.arena import ArtifactEvaluator, build_eval_search_options, evaluate_artifact_position
from ml.alphazero_lite.forensic_suite import REQUIRED_BUCKETS, canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame

FIXTURE_SCHEMA = "azlite_forensic_suite_v1"
TARGET_SUITE_SIZE = 224
DEFAULT_OUTPUT = ROOT_DIR / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
MAX_PLIES = 32
MAX_STATES_PER_DEPTH = 256
PROXY_ARTIFACT = ROOT_DIR / "ml/alphazero_lite/fixtures/incumbent_forensic_proxy_current"
CURRENT_PROXY_SIMULATIONS = 256
CHALLENGER_PROXY_SIMULATIONS = 384
PROXY_SEARCH_OPTIONS = build_eval_search_options()
PROXY_DISAGREEMENT_BUCKET = "incumbent_proxy_disagreement"


def _initial_state() -> dict[str, Any]:
    return {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }


def _legal_moves(state: dict[str, Any]) -> list[int]:
    return KalahGame.from_state(state).possible_moves()


def _side_view(state: dict[str, Any], player: int | None = None) -> tuple[list[int], list[int], int, int]:
    current_player = int(state["current_player"] if player is None else player)
    if current_player == 0:
        return (
            list(state["player_pits"]),
            list(state["opponent_pits"]),
            int(state["player_store"]),
            int(state["opponent_store"]),
        )
    return (
        list(state["opponent_pits"]),
        list(state["player_pits"]),
        int(state["opponent_store"]),
        int(state["player_store"]),
    )


def _move_features(game: KalahGame, move: int) -> dict[str, bool | int]:
    simulated = game.clone()
    original_player = simulated.current_player
    absolute_move = simulated.pit_index(move)
    if not simulated.move(absolute_move):
        raise ValueError(f"illegal move {move} for state {game.to_state()}")
    state_after = simulated.to_state()
    side_pits_after, _, side_store_after, _ = _side_view(state_after, player=original_player)
    player_store_gain = side_store_after - game.captured_seeds[original_player]
    return {
        "extra_turn": simulated.current_player == original_player and not simulated.over(),
        "capture": player_store_gain > 1,
        "swing": player_store_gain >= 4,
        "post_move_empty_pits": sum(1 for value in side_pits_after if value == 0),
    }


def _choose_bucket(state: dict[str, Any], ply: int) -> str | None:
    game = KalahGame.from_state(state)
    legal_moves = game.possible_moves()
    if not legal_moves:
        raise ValueError("forensic suite rows require at least one legal move")

    side_pits, opponent_pits, side_store, opponent_store = _side_view(state)
    total_side = sum(side_pits)
    total_opponent = sum(opponent_pits)
    empty_player = sum(1 for value in side_pits if value == 0)
    empty_opponent = sum(1 for value in opponent_pits if value == 0)
    features = [_move_features(game, move) for move in legal_moves]

    if total_side + total_opponent <= 16:
        return "sparse_endgame"
    if any(bool(feature["swing"]) for feature in features):
        return "high_value_swing"
    if total_side <= 6 or total_opponent <= 6 or empty_player >= 4 or empty_opponent >= 4:
        return "starvation_pressure"
    if abs(side_store - opponent_store) >= 8:
        return "high_imbalance"
    if any(bool(feature["capture"]) for feature in features):
        return "capture_available"
    if ply <= 8:
        return "opening_plies_1_8"
    if any(bool(feature["extra_turn"]) for feature in features):
        return "early_extra_turn"
    return None


def _best_child_score(summary: dict[str, Any], move: int | None) -> float | None:
    if move is None:
        return None
    for child in summary.get("child_stats", []):
        if child.get("move") == move:
            if "win_rate" in child:
                return float(child.get("win_rate", 0.0))
            return float(child.get("q_value", 0.0))
    return None


@lru_cache(maxsize=1)
def _proxy_evaluator() -> ArtifactEvaluator:
    return ArtifactEvaluator(PROXY_ARTIFACT)


@lru_cache(maxsize=20_000)
def _proxy_root_summary(state_key: str, simulations: int) -> dict[str, Any]:
    state = json.loads(state_key)
    return evaluate_artifact_position(
        artifact_path=PROXY_ARTIFACT,
        evaluator=_proxy_evaluator(),
        state=state,
        simulations=simulations,
        seed=42,
        c_puct=1.25,
        search_options=PROXY_SEARCH_OPTIONS,
    )


def _is_proxy_disagreement_candidate(state: dict[str, Any], ply: int) -> bool:
    if ply <= 8:
        return False
    state_key = canonical_state_key(state)
    current_summary = _proxy_root_summary(state_key, CURRENT_PROXY_SIMULATIONS)
    challenger_summary = _proxy_root_summary(state_key, CHALLENGER_PROXY_SIMULATIONS)
    current_move = current_summary.get("selected_move")
    challenger_move = challenger_summary.get("selected_move")
    if current_move is None or challenger_move is None or current_move == challenger_move:
        return False
    current_quality = _best_child_score(current_summary, current_move)
    challenger_quality = _best_child_score(challenger_summary, challenger_move)
    if current_quality is None or challenger_quality is None:
        return False
    return current_quality > challenger_quality


def _phase_for_state(state: dict[str, Any], ply: int) -> str:
    stones_in_pits = sum(state["player_pits"]) + sum(state["opponent_pits"])
    if ply <= 8:
        return "opening"
    if stones_in_pits <= 16:
        return "late"
    return "mid"


def _row_from_state(*, state: dict[str, Any], bucket: str, phase: str, source: str, row_id: str, tags: list[str]) -> dict[str, Any]:
    return {
        "id": row_id,
        "state": state,
        "side_to_move": int(state["current_player"]),
        "legal_moves": _legal_moves(state),
        "phase": phase,
        "bucket": bucket,
        "tags": tags,
        "source": source,
    }


def _seed_rows(candidates: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        _row_from_state(
            state=_initial_state(),
            bucket="opening_plies_1_8",
            phase="opening",
            source="seed",
            row_id="opening_plies_1_8-001",
            tags=["opening_plies_1_8", "seed"],
        )
    )

    seed_targets = {
        "early_extra_turn": 3,
        "capture_available": 3,
        "high_imbalance": 3,
        "starvation_pressure": 3,
        "sparse_endgame": 3,
        "high_value_swing": 3,
        PROXY_DISAGREEMENT_BUCKET: 3,
    }
    seed_counts = {bucket: 0 for bucket in seed_targets}
    for ply, state in candidates:
        if seed_counts[PROXY_DISAGREEMENT_BUCKET] < seed_targets[PROXY_DISAGREEMENT_BUCKET] and _is_proxy_disagreement_candidate(state, ply):
            seed_counts[PROXY_DISAGREEMENT_BUCKET] += 1
            bucket = PROXY_DISAGREEMENT_BUCKET
            rows.append(
                _row_from_state(
                    state=state,
                    bucket=bucket,
                    phase=_phase_for_state(state, ply),
                    source="seed",
                    row_id=f"{bucket}-{seed_counts[bucket]:03d}",
                    tags=[bucket, "seed", f"ply_{ply}"],
                )
            )
            if all(seed_counts[name] >= target for name, target in seed_targets.items()):
                break
            continue

        bucket = _choose_bucket(state, ply)
        if bucket not in seed_targets:
            continue
        if seed_counts[bucket] >= seed_targets[bucket]:
            continue
        seed_counts[bucket] += 1
        rows.append(
            _row_from_state(
                state=state,
                bucket=bucket,
                phase=_phase_for_state(state, ply),
                source="seed",
                row_id=f"{bucket}-{seed_counts[bucket]:03d}",
                tags=[bucket, "seed", f"ply_{ply}"],
            )
        )
        if all(seed_counts[name] >= target for name, target in seed_targets.items()):
            break
    return rows


def _candidate_states() -> list[tuple[int, dict[str, Any]]]:
    initial = KalahGame.from_state(_initial_state())
    frontier: list[tuple[int, KalahGame]] = [(0, initial)]
    seen: set[str] = set()
    candidates: list[tuple[int, dict[str, Any]]] = []

    for depth in range(MAX_PLIES + 1):
        depth_nodes = [item for item in frontier if item[0] == depth]
        if not depth_nodes:
            continue
        depth_nodes = depth_nodes[:MAX_STATES_PER_DEPTH]
        for ply, game in depth_nodes:
            state = game.to_state()
            key = canonical_state_key(state)
            if key in seen:
                continue
            seen.add(key)
            if game.possible_moves():
                candidates.append((ply, state))
            if ply >= MAX_PLIES or game.over():
                continue
            for move in game.possible_moves():
                next_game = game.clone()
                next_game.move(next_game.pit_index(move))
                frontier.append((ply + 1, next_game))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]["player_store"] + item[1]["opponent_store"],
            item[1]["player_pits"],
            item[1]["opponent_pits"],
            item[1]["current_player"],
        )
    )
    return candidates


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in sorted(REQUIRED_BUCKETS)}
    seen: set[str] = set()
    bucket_ids: dict[str, int] = {bucket: 0 for bucket in sorted(REQUIRED_BUCKETS)}

    def next_row_id(bucket: str) -> str:
        bucket_ids[bucket] += 1
        return f"{bucket}-{bucket_ids[bucket]:03d}"

    def add_row(row: dict[str, Any]) -> bool:
        key = canonical_state_key(row["state"])
        if key in seen:
            return False
        seen.add(key)
        rows.append(row)
        rows_by_bucket[row["bucket"]].append(row)
        return True

    candidates = _candidate_states()
    for row in _seed_rows(candidates):
        row["id"] = next_row_id(row["bucket"])
        add_row(row)

    for ply, state in candidates:
        if len(rows_by_bucket[PROXY_DISAGREEMENT_BUCKET]) >= 32:
            break
        if not _is_proxy_disagreement_candidate(state, ply):
            continue
        add_row(
            _row_from_state(
                state=state,
                bucket=PROXY_DISAGREEMENT_BUCKET,
                phase=_phase_for_state(state, ply),
                source="generated",
                row_id=next_row_id(PROXY_DISAGREEMENT_BUCKET),
                tags=[PROXY_DISAGREEMENT_BUCKET, "generated", f"ply_{ply}"],
            )
        )

    bucket_minimums = {bucket: 24 for bucket in REQUIRED_BUCKETS}
    bucket_minimums[PROXY_DISAGREEMENT_BUCKET] = 32

    for bucket in sorted(REQUIRED_BUCKETS):
        for ply, state in candidates:
            if len(rows_by_bucket[bucket]) >= bucket_minimums[bucket]:
                break
            chosen_bucket = _choose_bucket(state, ply)
            if chosen_bucket != bucket:
                continue
            add_row(
                _row_from_state(
                    state=state,
                    bucket=bucket,
                    phase=_phase_for_state(state, ply),
                    source="generated",
                    row_id=next_row_id(bucket),
                    tags=[bucket, "generated", f"ply_{ply}"],
                )
            )

    for ply, state in candidates:
        if len(rows) >= TARGET_SUITE_SIZE:
            break
        bucket = _choose_bucket(state, ply)
        if bucket is None:
            continue
        add_row(
            _row_from_state(
                state=state,
                bucket=bucket,
                phase=_phase_for_state(state, ply),
                source="generated",
                row_id=next_row_id(bucket),
                tags=[bucket, "generated", f"ply_{ply}"],
            )
        )

    present_buckets = {row["bucket"] for row in rows}
    missing = REQUIRED_BUCKETS - present_buckets
    if missing:
        raise SystemExit(f"builder is missing required buckets: {sorted(missing)}")
    if len(rows) < 200:
        raise SystemExit("builder must output at least 200 forensic rows")
    return rows


def build_fixture_text(rows: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(build_rows() if rows is None else rows, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    output_path.write_text(build_fixture_text(rows), encoding="utf-8")
    bucket_counts = Counter(row["bucket"] for row in rows)
    print(f"wrote forensic suite to {output_path}")
    print(json.dumps({"schema": FIXTURE_SCHEMA, "rows": len(rows), "buckets": dict(sorted(bucket_counts.items()))}, sort_keys=True))


if __name__ == "__main__":
    main()
