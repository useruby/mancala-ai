"""Synthetic integrity and trajectory tests for Seed 599 evaluation readiness."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import seed599_evaluation_readiness as protocol
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed599_evaluation import (
    INITIAL,
    validate_assignment_binding,
    validate_assignment_coverage,
    verify_game,
)


def synthetic_game() -> tuple[dict, dict]:
    game = KalahGame.from_state(INITIAL)
    trajectory: list[int] = []
    while not game.over() and len(trajectory) < 200:
        legal = game.possible_moves()
        move = game.pit_index(legal[0])
        trajectory.append(move)
        game.move(move)
    margin = int(game.captured_seeds[0] - game.captured_seeds[1])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    return game.to_state(), {
        "challenger_player": 0,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": winner,
    }


class Seed599EvaluationReadinessTests(unittest.TestCase):
    def test_tampered_checkpoint_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            path.write_bytes(b"registered checkpoint")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            path.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "identity_mismatch:checkpoint"):
                protocol.require_sha(path, expected, "checkpoint")

    def test_missing_game_assignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "missing_or_duplicate_game_assignments"
        ):
            validate_assignment_coverage({("A", 0): [1.0, 0.0]}, 2)

    def test_invalid_seat_assignment_is_rejected(self) -> None:
        opening = {"opening_id": "synthetic", "canonical_identity": "abc"}
        assignment = {
            "arm": "A",
            "opening_index": 0,
            "opening_id": "synthetic",
            "opening_canonical_identity": "abc",
        }
        with self.assertRaisesRegex(ValueError, "assignment_binding_or_seats"):
            validate_assignment_binding(
                assignment,
                "A",
                0,
                opening,
                [{"challenger_player": 0}, {"challenger_player": 0}],
            )

    def test_altered_opening_identity_is_rejected(self) -> None:
        opening = {"opening_id": "synthetic", "canonical_identity": "registered"}
        assignment = {
            "arm": "A",
            "opening_index": 0,
            "opening_id": "synthetic",
            "opening_canonical_identity": "altered",
        }
        with self.assertRaisesRegex(ValueError, "assignment_binding_or_seats"):
            validate_assignment_binding(
                assignment,
                "A",
                0,
                opening,
                [{"challenger_player": 0}, {"challenger_player": 1}],
            )

    def test_legal_synthetic_trajectory_and_outcome_mismatch(self) -> None:
        _state, row = synthetic_game()
        opening = {"opening_id": "synthetic", "prefix_moves": [], "state": INITIAL}
        self.assertEqual(
            row["winner"], verify_game(Path("."), "A", opening, 0, row)["winner"]
        )
        altered = dict(row, winner="draw" if row["winner"] != "draw" else "challenger")
        with self.assertRaisesRegex(ValueError, "trajectory_outcome_mismatch"):
            verify_game(Path("."), "A", opening, 0, altered)

    def test_mismatched_assignment_outcome_is_rejected(self) -> None:
        _state, game = synthetic_game()
        opening = {
            "opening_id": "synthetic",
            "canonical_identity": "registered",
        }
        assignment = {
            "arm": "A",
            "opening_index": 0,
            "opening_id": "synthetic",
            "opening_canonical_identity": "registered",
            "challenger_seats": [0, 1],
            "game_count": 2,
            "execution_environment": protocol.SEARCH_ENVIRONMENT,
            "games": [
                {
                    "position": position,
                    "challenger_player": seat,
                    "winner": game["winner"],
                    "margin": game["margin"],
                    "trajectory_sha256": hashlib.sha256(
                        game["trajectory"].encode()
                    ).hexdigest(),
                }
                for position, seat in enumerate((0, 1))
            ],
        }
        game_rows = [dict(game, challenger_player=seat) for seat in (0, 1)]
        assignment["games"][1]["winner"] = "draw"
        with self.assertRaisesRegex(ValueError, "assignment_outcome_ledger_mismatch"):
            validate_assignment_binding(assignment, "A", 0, opening, game_rows)


if __name__ == "__main__":
    unittest.main()
