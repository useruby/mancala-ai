"""Synthetic integrity and trajectory tests for Seed 599 evaluation readiness."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite import seed599_evaluation_readiness as protocol
from ml.alphazero_lite.seed599_evaluation_readiness import execution_authorized
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed599_evaluation import (
    INITIAL,
    bootstrap_decision_metrics,
    classify_decision,
    summarize_trajectory_overlap,
    validate_assignment_binding,
    validate_assignment_coverage,
    validate_game_opening_provenance,
    validate_report_outcomes,
    verify_game,
)


def synthetic_opening() -> dict:
    from ml.alphazero_lite.self_play import encode_state

    state = dict(INITIAL)
    encoded = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return {
        "opening_id": "synthetic-opening",
        "canonical_identity": hashlib.sha256(protocol.canonical(state)).hexdigest(),
        "float32_identity": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "prefix_moves": [],
        "state": state,
    }


def synthetic_game(
    opening: dict | None = None, *, seat: int = 0, position: int = 0
) -> tuple[dict, dict]:
    opening = opening or synthetic_opening()
    game = KalahGame.from_state(opening["state"])
    trajectory: list[int] = []
    while not game.over() and len(trajectory) < 200:
        legal = game.possible_moves()
        move = game.pit_index(legal[0])
        trajectory.append(move)
        game.move(move)
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    return game.to_state(), {
        "game_index": position,
        "game_within_opening": position,
        "opening_index": 0,
        "opening_state_hash": opening["canonical_identity"],
        "opening_prefix_moves": opening["prefix_moves"],
        "opening_applied_prefix_length": len(opening["prefix_moves"]),
        "opening_contract": opening.get(
            "opening_contract", "legacy_player_relative_v1"
        ),
        "challenger_player": seat,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": winner,
    }


class Seed599EvaluationReadinessTests(unittest.TestCase):
    def _write_authorization(self, root: Path, freeze_bytes: bytes) -> dict:
        output = root / protocol.OUT
        output.mkdir(parents=True, exist_ok=True)
        freeze_path = output / "execution-freeze.json"
        freeze_path.write_bytes(freeze_bytes)
        freeze = json.loads(freeze_bytes)
        authorization = {
            "schema": "seed599-evaluation-owner-authorization-v1",
            "status": "authorized-for-evaluation",
            "execution_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
            "authorization": {"authorized_by": "project owner", "run_games": True},
        }
        (output / "evaluation-authorization.json").write_text(
            json.dumps(authorization), encoding="utf-8"
        )
        return freeze

    def test_exact_file_byte_authorization_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_bytes = (
                json.dumps(
                    {
                        "schema": "seed599-evaluation-execution-freeze-v1",
                        "status": "frozen-awaiting-separate-project-owner-authorization",
                        "execution_authorized": False,
                    },
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            freeze = self._write_authorization(root, freeze_bytes)
            self.assertTrue(execution_authorized(root, freeze))

    def test_authorization_bound_to_stale_freeze_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = (
                json.dumps(
                    {
                        "schema": "seed599-evaluation-execution-freeze-v1",
                        "status": "frozen-awaiting-separate-project-owner-authorization",
                        "execution_authorized": False,
                    },
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            self._write_authorization(root, original)
            freeze_path = root / protocol.OUT / "execution-freeze.json"
            current_bytes = original + b" "
            freeze_path.write_bytes(current_bytes)
            self.assertFalse(execution_authorized(root, json.loads(current_bytes)))

    def test_malformed_authorization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_bytes = (
                json.dumps(
                    {
                        "schema": "seed599-evaluation-execution-freeze-v1",
                        "status": "frozen-awaiting-separate-project-owner-authorization",
                        "execution_authorized": False,
                    },
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            freeze = self._write_authorization(root, freeze_bytes)
            (root / protocol.OUT / "evaluation-authorization.json").write_text(
                "{", encoding="utf-8"
            )
            self.assertFalse(execution_authorized(root, freeze))

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
        opening = synthetic_opening()
        _state, row = synthetic_game(opening)
        self.assertEqual(
            row["winner"],
            verify_game(Path("."), "A", opening, 0, 0, row)["winner"],
        )
        altered = dict(row, winner="draw" if row["winner"] != "draw" else "challenger")
        with self.assertRaisesRegex(ValueError, "trajectory_outcome_mismatch"):
            verify_game(Path("."), "A", opening, 0, 0, altered)

    def test_registered_opening_root_provenance_replays(self) -> None:
        opening = protocol.read_json(protocol.ROOT / protocol.OPENINGS)[0]
        _state, row = synthetic_game(opening)
        result = verify_game(Path("."), "A", opening, 0, 0, row)
        self.assertIn("canonical_positions", result)
        self.assertIn("float32_positions", result)

    def test_mismatched_assignment_outcome_is_rejected(self) -> None:
        opening_state = synthetic_opening()
        _state, game = synthetic_game(opening_state)
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

    def test_incorrect_game_assignment_fields_are_rejected(self) -> None:
        opening = synthetic_opening()
        _state, row = synthetic_game(opening, seat=0, position=0)
        for field in ("game_index", "game_within_opening", "opening_index"):
            with self.subTest(field=field):
                altered = dict(row)
                altered[field] = 1
                with self.assertRaisesRegex(
                    ValueError, "game_opening_provenance_mismatch"
                ):
                    validate_game_opening_provenance(opening, 0, 0, altered)

    def test_opening_prefix_provenance_tampering_is_rejected(self) -> None:
        opening = synthetic_opening()
        _state, row = synthetic_game(opening)
        row["opening_prefix_moves"] = [0]
        with self.assertRaisesRegex(ValueError, "game_opening_provenance_mismatch"):
            validate_game_opening_provenance(opening, 0, 0, row)

    def test_opening_state_hash_provenance_tampering_is_rejected(self) -> None:
        opening = synthetic_opening()
        _state, row = synthetic_game(opening)
        row["opening_state_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "game_opening_provenance_mismatch"):
            validate_game_opening_provenance(opening, 0, 0, row)

    def test_report_counts_must_match_game_outcomes(self) -> None:
        _state, first = synthetic_game(seat=0, position=0)
        _state, second = synthetic_game(seat=1, position=1)
        games = [first, second]
        report: dict[str, int | float] = {
            "games_played": 2,
            "wins": sum(row["winner"] == "challenger" for row in games),
            "losses": sum(row["winner"] == "current" for row in games),
            "draws": sum(row["winner"] == "draw" for row in games),
        }
        report["score"] = (report["wins"] + 0.5 * report["draws"]) / report[
            "games_played"
        ]
        validate_report_outcomes(report, games)
        report["wins"] += 1
        with self.assertRaisesRegex(ValueError, "arena_report_outcome_mismatch"):
            validate_report_outcomes(report, games)

    def test_positive_and_negative_training_overlap(self) -> None:
        observed = {
            "A": {"canonical": {"c1", "c2"}, "float32": {"f1", "f2"}},
            "B": {"canonical": {"c3"}, "float32": {"f3"}},
        }
        membership = {
            "A": {"canonical_identities": {"c1"}, "float32_identities": {"f1"}},
            "B": {"canonical_identities": {"c2"}, "float32_identities": {"f9"}},
        }
        result = summarize_trajectory_overlap(observed, membership)
        overlap = result["comparisons"]["A"]["vs_training_A"]
        self.assertEqual(overlap["canonical"]["overlapping_distinct_positions"], 1)
        self.assertEqual(overlap["canonical"]["non_overlapping_distinct_positions"], 1)
        self.assertEqual(overlap["float32"]["overlapping_distinct_positions"], 1)
        self.assertFalse(result["gate"])
        negative = summarize_trajectory_overlap(
            {
                "A": {"canonical": {"x"}, "float32": {"y"}},
                "B": {"canonical": {"z"}, "float32": {"w"}},
            },
            membership,
        )
        self.assertEqual(
            negative["comparisons"]["A"]["vs_training_A"]["canonical"][
                "overlapping_distinct_positions"
            ],
            0,
        )

    def test_small_deterministic_bootstrap_and_categories(self) -> None:
        result = bootstrap_decision_metrics(
            np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64),
            resamples=64,
            seed=599,
        )
        self.assertEqual(result["A_score"], 0.0)
        self.assertEqual(result["B_score"], 1.0)
        self.assertEqual(result["B_one_sided_95_lower"], 1.0)
        self.assertEqual(result["B_minus_A_two_sided_95_interval"], [1.0, 1.0])
        self.assertEqual(result["outcome_category"], "both_conditions_pass")
        self.assertEqual(
            {
                classify_decision(True, True),
                classify_decision(True, False),
                classify_decision(False, True),
                classify_decision(False, False),
            },
            {
                "both_conditions_pass",
                "strength_only",
                "coverage_only",
                "neither_condition_passes",
            },
        )


if __name__ == "__main__":
    unittest.main()
