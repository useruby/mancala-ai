from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed416_public_validation import (
    lane_games_sha256,
    state_hash,
    validate_game,
    validate_record_accounting,
)


def replayable_fixture(pits: list[int] | None = None) -> tuple[dict, dict]:
    board = KalahGame(pits if pits is not None else [4] * 12, [0, 0], 0)
    opening_state = board.to_state()
    opening = {
        "prefix_moves": [],
        "state": opening_state,
        "state_hash": state_hash(opening_state),
    }
    trajectory = []
    while not board.over():
        move = board.possible_moves()[0]
        absolute = board.pit_index(move)
        trajectory.append(absolute)
        assert board.move(absolute)
    seat = 0
    margin = board.captured_seeds[seat] - board.captured_seeds[1 - seat]
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    game = {
        "game_index": 0,
        "opening_index": 0,
        "game_within_opening": 0,
        "opening_prefix_moves": [],
        "opening_state_hash": opening["state_hash"],
        "opening_applied_prefix_length": 0,
        "opening_contract": "arena_player_relative_v2",
        "challenger_player": seat,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": winner,
    }
    score = {"challenger": 1.0, "draw": 0.5, "current": 0.0}[winner]
    return opening, {
        "lane": "A",
        "opening_id": "0",
        "opponent_score": score,
        "game": game,
    }


def test_valid_trajectory_replays() -> None:
    opening, row = replayable_fixture()
    validate_game(row, opening)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("margin", 100, "outcome_terminal_margin_invalid"),
        ("opening_index", 1, "outcome_opening_identity_invalid"),
        ("opening_state_hash", "0" * 64, "outcome_opening_hash_invalid"),
    ],
)
def test_invalid_public_game_is_rejected(field: str, value: object, error: str) -> None:
    opening, row = replayable_fixture()
    corrupted = copy.deepcopy(row)
    corrupted["game"][field] = value
    with pytest.raises(ValueError, match=error):
        validate_game(corrupted, opening)


def test_wrong_actor_is_rejected() -> None:
    opening, row = replayable_fixture()
    trajectory = row["game"]["trajectory"].split(",")
    trajectory[0] = "6"
    row["game"]["trajectory"] = ",".join(trajectory)
    with pytest.raises(ValueError, match="outcome_action_invalid"):
        validate_game(row, opening)


def test_empty_pit_action_is_rejected() -> None:
    pits = [0, 4, 4, 4, 4, 4] + [4] * 6
    opening, row = replayable_fixture(pits)
    trajectory = row["game"]["trajectory"].split(",")
    trajectory[0] = "0"
    row["game"]["trajectory"] = ",".join(trajectory)
    with pytest.raises(ValueError, match="outcome_action_invalid"):
        validate_game(row, opening)


def test_public_game_bytes_remain_bound_after_ledger_hash_rewrite() -> None:
    _, row = replayable_fixture()
    rows = [row]
    bound_hash = lane_games_sha256(rows, "A")
    edited = copy.deepcopy(rows)
    edited[0]["game"]["trajectory"] = "6"
    rewritten_ledger_hash = (
        __import__("hashlib")
        .sha256((json.dumps(edited[0]) + "\n").encode())
        .hexdigest()
    )
    assert (
        rewritten_ledger_hash
        != __import__("hashlib")
        .sha256((json.dumps(rows[0]) + "\n").encode())
        .hexdigest()
    )
    assert lane_games_sha256(edited, "A") != bound_hash


def test_duplicate_lane_opening_seat_is_rejected() -> None:
    rows = [
        {
            "lane": lane,
            "opening_id": str(opening),
            "game": {"challenger_player": seat},
        }
        for lane in ("A", "B")
        for opening in range(512)
        for seat in (0, 1)
    ]
    rows[-1]["game"]["challenger_player"] = 0
    with pytest.raises(
        ValueError, match="outcome_lane_opening_seat_accounting_invalid"
    ):
        validate_record_accounting(rows)


def test_public_verifier_imports_without_training_or_arena_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ml.alphazero_lite.verify_seed416_policy_target_softening; "
            "assert not {'torch', 'ml.alphazero_lite.train', "
            "'ml.alphazero_lite.arena'} & sys.modules.keys()",
        ],
        check=True,
    )
