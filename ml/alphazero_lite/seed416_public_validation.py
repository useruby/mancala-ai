"""Pure replay validation for the public seed416 evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame


def state_hash(state: dict[str, Any]) -> str:
    payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_openings(suite: list[dict[str, Any]], proof: dict[str, Any]) -> None:
    excluded = set(proof["excluded_identities"])
    hashes: set[str] = set()
    for row in suite:
        if row.get("opening_contract") != "arena_player_relative_v2" or int(
            row.get("ply", -1)
        ) != len(row["prefix_moves"]):
            raise ValueError("opening_contract_or_prefix_length_invalid")
        game = KalahGame([4] * 12, [0, 0], 0)
        absolute_prefix: list[int] = []
        for action in row["prefix_moves"]:
            if game.over() or action not in game.possible_moves():
                raise ValueError("opening_prefix_illegal")
            absolute_prefix.append(game.pit_index(int(action)))
            if not game.move(game.pit_index(int(action))):
                raise ValueError("opening_prefix_illegal")
        actual = game.to_state()
        if actual != row["state"] or state_hash(actual) != row["state_hash"]:
            raise ValueError("opening_state_reconciliation_invalid")
        if (
            int(row["side_to_move"]) != game.current_player
            or int(row["store_diff"])
            != game.captured_seeds[game.current_player]
            - game.captured_seeds[1 - game.current_player]
            or int(row["pit_sum"]) != sum(game.pits)
            or row.get("source_prefix_moves_absolute") != absolute_prefix
        ):
            raise ValueError("opening_declared_metadata_invalid")
        key = row["state_hash"]
        if key in hashes:
            raise ValueError("opening_duplicate")
        hashes.add(key)
        if game.over() or not game.possible_moves():
            raise ValueError("opening_ineligible")
        if (
            sum(game.pits) <= 32
            or sum(game.pits[:6]) == 0
            or sum(game.pits[6:]) == 0
            or key in excluded
        ):
            raise ValueError("opening_exclusion_or_phase_invalid")
    if len(suite) != 512 or len(hashes) != 512:
        raise ValueError("suite_count_or_uniqueness_invalid")
    if len(excluded) != proof["combined_union_count"] or proof["suite_overlap"] != 0:
        raise ValueError("exclusion_proof_count_invalid")


def validate_game(row: dict[str, Any], opening: dict[str, Any]) -> None:
    game_row = row["game"]
    if game_row.get("opening_index") != int(row["opening_id"]):
        raise ValueError("outcome_opening_identity_invalid")
    if game_row.get("opening_state_hash") != opening["state_hash"]:
        raise ValueError("outcome_opening_hash_invalid")
    if game_row.get("opening_prefix_moves") != opening["prefix_moves"]:
        raise ValueError("outcome_opening_prefix_invalid")
    within = int(game_row["game_within_opening"])
    if (
        within not in (0, 1)
        or int(game_row["game_index"]) != int(row["opening_id"]) * 2 + within
        or int(game_row["challenger_player"]) != within
        or int(game_row["opening_applied_prefix_length"])
        != len(opening["prefix_moves"])
        or game_row.get("opening_contract") != "arena_player_relative_v2"
    ):
        raise ValueError("outcome_lane_opening_seat_identity_invalid")
    seat = int(game_row["challenger_player"])
    if seat not in (0, 1):
        raise ValueError("outcome_seat_invalid")
    state = opening["state"]
    board = KalahGame.from_state(state)
    trajectory = [int(x) for x in game_row["trajectory"].split(",") if x]
    if len(trajectory) != int(game_row["game_length"]):
        raise ValueError("outcome_game_length_invalid")
    for absolute in trajectory:
        if board.over():
            raise ValueError("outcome_moves_after_terminal")
        actor = board.current_player
        if absolute // 6 != actor or not board.move(absolute):
            raise ValueError("outcome_action_invalid")
    if not board.over():
        raise ValueError("outcome_trajectory_not_terminal")
    margin = board.captured_seeds[seat] - board.captured_seeds[1 - seat]
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if margin != int(game_row["margin"]):
        raise ValueError("outcome_terminal_margin_invalid")
    if winner != game_row["winner"]:
        raise ValueError("outcome_terminal_winner_invalid")
    score = {"challenger": 1.0, "draw": 0.5, "current": 0.0}[winner]
    if score != float(row["opponent_score"]):
        raise ValueError("outcome_score_mismatch")


def validate_records(rows: list[dict[str, Any]], suite: list[dict[str, Any]]) -> None:
    for row in rows:
        lane, opening_id = row["lane"], int(row["opening_id"])
        if lane not in {"A", "B"} or not 0 <= opening_id < len(suite):
            raise ValueError("outcome_identity_invalid")
        validate_game(row, suite[opening_id])
    validate_record_accounting(rows)


def validate_record_accounting(rows: list[dict[str, Any]]) -> None:
    observed: Counter[tuple[str, int, int]] = Counter()
    for row in rows:
        lane, opening_id = row["lane"], int(row["opening_id"])
        seat = int(row["game"]["challenger_player"])
        key = (lane, opening_id, seat)
        observed[key] += 1
    expected = {
        (lane, i, seat) for lane in ("A", "B") for i in range(512) for seat in (0, 1)
    }
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise ValueError("outcome_lane_opening_seat_accounting_invalid")


def lane_games_sha256(rows: list[dict[str, Any]], lane: str) -> str:
    """Hash original arena serialization, preserving public ledger order."""
    payload = "".join(
        json.dumps(row["game"]) + "\n" for row in rows if row["lane"] == lane
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
