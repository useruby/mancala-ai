"""Torch-free validation and analysis for the fixed seed422 Adam ablation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame

ADAM_COMMON = {
    "lr": 0.001,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "scheduler": "none",
}


def optimizer_options(lane: str) -> dict[str, Any]:
    if lane not in {"A", "B"}:
        raise ValueError("lane_must_be_A_or_B")
    return {
        **ADAM_COMMON,
        "betas": (0.9, 0.999) if lane == "A" else (0.0, 0.999),
    }


def make_adam(model: Any, lane: str) -> Any:
    import torch

    options = optimizer_options(lane)
    return torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=options["lr"],
        betas=options["betas"],
        eps=options["eps"],
        weight_decay=options["weight_decay"],
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def replay_opening(row: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame([4] * 12, [0, 0], 0)
    for action in row["prefix_moves"]:
        if game.over() or not 0 <= int(action) < 6:
            raise ValueError("opening_prefix_invalid")
        if not game.move(game.pit_index(int(action))):
            raise ValueError("opening_prefix_illegal")
    state = game.to_state()
    if state != row["state"] or canonical_state_hash(state) != row["state_hash"]:
        raise ValueError("opening_state_mismatch")
    if game.over() or sum(game.pits) <= 32:
        raise ValueError("opening_ineligible")
    return state


def validate_suite(
    suite: list[dict[str, Any]], excluded: set[str], expected_count: int = 512
) -> None:
    identities = [canonical_state_hash(replay_opening(row)) for row in suite]
    if len(identities) != expected_count or len(set(identities)) != expected_count:
        raise ValueError("suite_count_or_unique_identity_failure")
    if set(identities) & excluded:
        raise ValueError("suite_overlaps_historical_exclusions")


def validate_game(row: dict[str, Any], opening: dict[str, Any]) -> None:
    game_record = row["game"]
    opening_id = int(row["opening_id"])
    within = int(game_record["game_within_opening"])
    if (
        int(game_record["opening_index"]) != opening_id
        or game_record["opening_prefix_moves"] != opening["prefix_moves"]
        or within not in (0, 1)
        or int(game_record["challenger_player"]) != within
        or int(game_record["game_index"]) != opening_id * 2 + within
    ):
        raise ValueError("game_opening_or_seat_identity_invalid")
    if game_record["opening_state_hash"] != opening["state_hash"]:
        raise ValueError("game_opening_hash_mismatch")
    board = KalahGame.from_state(opening["state"])
    actions = [int(x) for x in game_record["trajectory"].split(",") if x]
    if len(actions) != int(game_record["game_length"]):
        raise ValueError("game_length_mismatch")
    for action in actions:
        if board.over() or not board.move(action):
            raise ValueError("game_absolute_trajectory_illegal")
    if not board.over():
        raise ValueError("game_trajectory_not_terminal")
    seat = int(game_record["challenger_player"])
    margin = board.captured_seeds[seat] - board.captured_seeds[1 - seat]
    if margin != int(game_record["margin"]):
        raise ValueError("game_margin_mismatch")
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if winner != game_record["winner"]:
        raise ValueError("game_winner_mismatch")
    score = 1.0 if winner == "challenger" else 0.5 if winner == "draw" else 0.0
    if score != float(row["opponent_score"]):
        raise ValueError("game_score_mismatch")


def validate_ledger(rows: list[dict[str, Any]], suite: list[dict[str, Any]]) -> None:
    expected = {
        (lane, opening, seat)
        for lane in ("A", "B")
        for opening in range(len(suite))
        for seat in (0, 1)
    }
    observed: Counter[tuple[str, int, int]] = Counter()
    for row in rows:
        lane = str(row["lane"])
        opening = int(row["opening_id"])
        game = row["game"]
        seat = int(game["challenger_player"])
        if lane not in {"A", "B"} or (lane, opening, seat) not in expected:
            raise ValueError("ledger_identity_invalid")
        validate_game(row, suite[opening])
        observed[(lane, opening, seat)] += 1
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise ValueError("ledger_exactly_once_accounting_failure")


def opening_scores(rows: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lane"]), int(row["opening_id"]))].append(
            float(row["opponent_score"])
        )
    scores: dict[str, dict[int, float]] = {"A": {}, "B": {}}
    for (lane, opening), values in grouped.items():
        if len(values) != 2:
            raise ValueError("opening_seat_pair_incomplete")
        scores[lane][opening] = sum(values) / 2.0
    if set(scores["A"]) != set(scores["B"]):
        raise ValueError("paired_opening_identity_mismatch")
    return scores


def bootstrap_paired(
    differences: list[float], *, samples: int = 10_000, seed: int = 422
) -> tuple[float, float]:
    if not differences:
        raise ValueError("empty_opening_cluster_sample")
    rng = np.random.default_rng(seed)
    n = len(differences)
    indexes = rng.integers(0, n, size=(samples, n))
    means = np.asarray(differences, dtype=np.float64)[indexes].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2048:
        raise ValueError("expected_exactly_2048_games")
    scores = opening_scores(rows)
    differences = [scores["B"][i] - scores["A"][i] for i in sorted(scores["A"])]
    b_cluster_scores = [scores["B"][i] for i in sorted(scores["B"])]
    delta = sum(differences) / len(differences)
    ci = bootstrap_paired(differences)
    b_ci = bootstrap_paired(b_cluster_scores)
    lane_summary: dict[str, Any] = {}
    for lane in ("A", "B"):
        lane_rows = [row for row in rows if row["lane"] == lane]
        points = [float(row["opponent_score"]) for row in lane_rows]
        wins = sum(score == 1.0 for score in points)
        draws = sum(score == 0.5 for score in points)
        losses = sum(score == 0.0 for score in points)
        seat_scores = {
            str(seat): sum(
                float(row["opponent_score"])
                for row in lane_rows
                if int(row["game"]["challenger_player"]) == seat
            )
            / 1024
            for seat in (0, 1)
        }
        lane_summary[lane] = {
            "score": sum(points) / len(points),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "seat_scores": seat_scores,
        }
    advance = (
        delta >= 0.03
        and ci[0] > 0
        and lane_summary["B"]["score"] >= 0.53
        and b_ci[0] > 0.5
    )
    return {
        "schema": "seed422-analysis-v1",
        "primary": "paired_B_minus_A_opening_average_score",
        "paired_delta": delta,
        "paired_delta_ci95": list(ci),
        "B_opening_cluster_score_ci95": list(b_ci),
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 422,
        "cluster_count": len(differences),
        "lanes": lane_summary,
        "opening_matrix": {
            str(index): {lane: scores[lane][index] for lane in ("A", "B")}
            for index in sorted(scores["A"])
        },
        "decision": "advance_to_independent_confirmation"
        if advance
        else "retain_baseline_close_fixed_beta1_intervention",
        "thresholds": {
            "delta_minimum": 0.03,
            "delta_lower_bound_gt": 0.0,
            "B_score_minimum": 0.53,
            "B_score_lower_bound_gt": 0.5,
        },
    }
