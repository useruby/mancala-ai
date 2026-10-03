"""Shared validation for hash-bound seed461 arena evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.alphazero_lite.arena import canonical_game_state_hash
from ml.alphazero_lite.build_opening_suite import INITIAL_STATE, canonical_key
from ml.alphazero_lite.kalah_rules import KalahGame


def validate_arena_evidence(
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    openings: list[dict[str, Any]],
    run: str,
    candidate: dict[str, Any],
    opponent: dict[str, Any],
    evaluation: dict[str, Any],
) -> np.ndarray:
    """Validate the complete report/game contract and return per-opening scores."""
    expected_games = evaluation.get(
        "games_per_candidate", evaluation.get("games_per_model")
    )
    if expected_games is None:
        raise ValueError(f"registered_game_count_missing:{run}")
    if (
        len(rows) != expected_games
        or report.get("schema") != "arena_v1"
        or report.get("games") != expected_games
        or report.get("games_played") != expected_games
    ):
        raise ValueError(f"game_count_mismatch:{run}")
    notes = report.get("notes", {})
    expected_notes = {
        "challenger_path": candidate["artifact"],
        "current_path": opponent["artifact"],
        "suite_sha256": evaluation["suite"]["sha256"],
        "seed": evaluation["arena_seed"],
        "base_seed": evaluation["arena_seed"],
        "seed_contract": evaluation["seed_contract"],
        "challenger_simulations": evaluation["simulations_per_side"],
        "current_simulations": evaluation["simulations_per_side"],
    }
    for key, expected in expected_notes.items():
        if notes.get(key) != expected:
            raise ValueError(f"report_identity_mismatch:{run}:{key}")
    profile = notes.get("search_profile", {})
    if (
        profile.get("c_puct") != evaluation["c_puct"]
        or profile.get("simulations") != evaluation["simulations_per_side"]
    ):
        raise ValueError(f"report_search_contract_mismatch:{run}")
    for key, expected in candidate["runtime_contract"].items():
        if notes.get(key) != expected or profile.get(key) != expected:
            raise ValueError(f"report_runtime_contract_mismatch:{run}:{key}")

    grouped: dict[int, list[dict[str, Any]]] = {}
    actual_identities: dict[int, tuple[str, str, int]] = {}
    versioned_indices: set[int] = set()
    for index, opening in enumerate(openings):
        game = KalahGame.from_state(INITIAL_STATE)
        from ml.alphazero_lite.arena import apply_opening_moves

        prefix = [int(move) for move in opening.get("prefix_moves", [])]
        versioned = opening.get("opening_contract") == "arena_player_relative_v2"
        if versioned:
            versioned_indices.add(index)
        if not versioned and "state" not in opening:
            continue
        applied = apply_opening_moves(game, prefix)
        identity = canonical_game_state_hash(game)
        if applied != len(prefix):
            raise ValueError(
                f"opening_prefix_truncated:{run}:{index}:{applied}/{len(prefix)}"
            )
        if "state" in opening and canonical_key(game.to_state()) != canonical_key(
            opening["state"]
        ):
            raise ValueError(f"opening_declared_state_mismatch:{run}:{index}")
        if opening.get("opening_contract") == "arena_player_relative_v2":
            if canonical_key(game.to_state()) != opening.get("state_hash"):
                raise ValueError(f"opening_declared_hash_mismatch:{run}:{index}")
        actual_identities[index] = (identity, canonical_key(game.to_state()), applied)
    if len({actual_identities[index][1] for index in versioned_indices}) != len(
        versioned_indices
    ):
        raise ValueError(f"duplicate_actual_opening_state:{run}")
    wins = losses = draws = 0
    for row in rows:
        winner = row.get("winner")
        if winner not in {"challenger", "current", "draw"}:
            raise ValueError(f"unknown_winner:{run}")
        wins += winner == "challenger"
        losses += winner == "current"
        draws += winner == "draw"
        index = row.get("opening_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(openings)
            or row.get("opening_prefix_moves") != openings[index]["prefix_moves"]
        ):
            raise ValueError(f"opening_identity_mismatch:{run}")
        if index in versioned_indices:
            actual_hash, actual_canonical, actual_length = actual_identities[index]
            if row.get("opening_state_hash") != actual_hash:
                raise ValueError(f"actual_opening_state_identity_mismatch:{run}")
            if row.get("opening_applied_prefix_length") != actual_length:
                raise ValueError(f"opening_applied_prefix_length_mismatch:{run}")
            if row.get(
                "opening_contract"
            ) != "arena_player_relative_v2" or actual_canonical != openings[index].get(
                "state_hash"
            ):
                raise ValueError(f"actual_opening_contract_mismatch:{run}")
        grouped.setdefault(index, []).append(row)
    if set(grouped) != set(range(len(openings))):
        raise ValueError(f"opening_coverage_mismatch:{run}")
    scores = np.empty(len(openings), dtype=np.float64)
    for index, pair in grouped.items():
        seats = [row.get("challenger_player") for row in pair]
        if (
            len(pair) != 2
            or any(seat not in (0, 1) for seat in seats)
            or set(seats) != {0, 1}
        ):
            raise ValueError(f"seat_pairing_mismatch:{run}:{index}")
        scores[index] = (
            sum(
                1.0
                if row["winner"] == "challenger"
                else 0.5
                if row["winner"] == "draw"
                else 0.0
                for row in pair
            )
            / 2
        )
    if (report.get("wins"), report.get("losses"), report.get("draws")) != (
        wins,
        losses,
        draws,
    ):
        raise ValueError(f"report_outcome_accounting_mismatch:{run}")
    if not np.isclose(scores.mean(), report.get("score", float("nan"))):
        raise ValueError(f"report_score_mismatch:{run}")
    return scores
