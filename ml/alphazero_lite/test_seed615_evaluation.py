"""Synthetic-only Seed 615 evaluation contract tests; never starts registered games."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite import seed615_evaluation as runner
from ml.alphazero_lite import verify_seed615_evaluation as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


class _BoundaryGame:
    """Small deterministic game double for exact max-ply boundary regressions."""

    def __init__(self, terminal_at: int | None):
        self.current_player = 0
        self.move_count = 0
        self.terminal_at = terminal_at
        self.captured_seeds = [30, 18]

    def over(self) -> bool:
        return self.terminal_at is not None and self.move_count >= self.terminal_at

    def possible_moves(self) -> list[int]:
        return [] if self.over() else list(range(6))

    def pit_index(self, relative_move: int) -> int:
        return self.current_player * 6 + relative_move

    def move(self, absolute_move: int) -> bool:
        if self.over() or absolute_move // 6 != self.current_player:
            return False
        self.move_count += 1
        self.current_player = 1 - self.current_player
        return True

    def to_state(self) -> dict[str, int]:
        return {"current_player": self.current_player, "move_count": self.move_count}


def synthetic_boundary_case(
    *, terminal_at: int | None, continuation_plies: int
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    def factory() -> _BoundaryGame:
        return _BoundaryGame(terminal_at)

    game = factory()
    prefix = [0] * 12
    for move in prefix:
        game.move(game.pit_index(move))
    opening_hash = hashlib.sha256(runner.canonical(game.to_state())).hexdigest()
    opening = {"opening_id": "boundary-opening", "prefix_moves": prefix}
    assignment = {
        "candidate": "S615",
        "candidate_seat": 0,
        "game_id": "boundary-game",
        "game_seed": 31,
        "opening_id": "boundary-opening",
        "opening_identity": opening_hash,
        "opening_index": 0,
        "opponent": "P-E4",
    }
    trajectory = []
    for _ in range(continuation_plies):
        if game.over():
            break
        absolute_move = game.pit_index(0)
        trajectory.append(absolute_move)
        game.move(absolute_move)
    margin = game.captured_seeds[0] - game.captured_seeds[1]
    row = {
        "game_index": 0,
        "game_within_opening": 0,
        "opening_index": 0,
        "opening_state_hash": opening_hash,
        "opening_prefix_moves": prefix,
        "opening_applied_prefix_length": len(prefix),
        "challenger_player": 0,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": "challenger" if margin > 0 else "current" if margin < 0 else "draw",
    }
    return assignment, opening, row


def synthetic_analysis_records() -> list[dict[str, object]]:
    rows = []
    for opening in range(1024):
        for candidate in ("B512", "S615"):
            for seat in (0, 1):
                win = (opening + seat) % (2 if candidate == "B512" else 3) != 0
                margin = 1 if win else -1
                rows.append(
                    {
                        "opening_index": opening,
                        "candidate": candidate,
                        "candidate_seat": seat,
                        "margin": margin,
                        "score": 1.0 if win else 0.0,
                    }
                )
    return rows


def synthetic_opening() -> dict[str, object]:
    initial = {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    for seed in range(615, 715):
        game = KalahGame.from_state(dict(initial))
        randomizer = random.Random(seed)
        prefix = []
        while not game.over() and len(prefix) < 12:
            move = randomizer.choice(game.possible_moves())
            prefix.append(move)
            game.move(game.pit_index(move))
        if len(prefix) == 12 and not game.over():
            state = game.to_state()
            return {
                "opening_id": "synthetic-opening-0000",
                "prefix_moves": prefix,
                "canonical_identity": hashlib.sha256(
                    runner.canonical(state)
                ).hexdigest(),
                "state": state,
            }
    raise AssertionError("no nonterminal synthetic opening found")


def synthetic_game(seat: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    opening = synthetic_opening()
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    for move in opening["prefix_moves"]:
        game.move(game.pit_index(int(move)))
    opening_hash = hashlib.sha256(runner.canonical(game.to_state())).hexdigest()
    trajectory = []
    while not game.over() and len(trajectory) < 200:
        move = game.pit_index(game.possible_moves()[0])
        trajectory.append(move)
        game.move(move)
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    assignment = {
        "candidate": "S615",
        "candidate_seat": seat,
        "game_id": "synthetic-game",
        "game_seed": 987,
        "opening_id": opening["opening_id"],
        "opening_identity": opening_hash,
        "opening_index": 0,
        "opponent": "P-E4",
    }
    row = {
        "game_index": 0,
        "game_within_opening": 0,
        "opening_index": 0,
        "opening_state_hash": opening_hash,
        "opening_prefix_moves": opening["prefix_moves"],
        "opening_applied_prefix_length": 12,
        "challenger_player": seat,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": "challenger" if margin > 0 else "current" if margin < 0 else "draw",
    }
    return assignment, {"opening": opening, "game": row}


class Seed615EvaluationTests(unittest.TestCase):
    def test_exact_registered_search_contract(self) -> None:
        verifier.validate_search(dict(runner.SEARCH))
        altered = dict(runner.SEARCH, simulations=127)
        with self.assertRaisesRegex(ValueError, "search_contract"):
            verifier.validate_search(altered)

    def test_wrong_candidate_identity_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_checkpoint_identity"):
            verifier.validate_identity(
                "0" * 64, runner.EXPECTED["S615"], "candidate_checkpoint"
            )

    def test_registered_opponent_is_always_serving_model(self) -> None:
        assignment = {"candidate": "S615", "candidate_seat": 1, "game_seed": 9}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opening.jsonl"
            path.write_text('{"prefix_moves":[] }\n', encoding="utf-8")
            command = runner._arena_command(
                Path("/root"), assignment, path, Path(directory)
            )
        options = dict(zip(command, command[1:]))
        self.assertEqual(
            options["--current-policy-artifact"], options["--current-value-artifact"]
        )
        self.assertEqual(
            options["--challenger-value-artifact"], options["--current-value-artifact"]
        )
        self.assertNotEqual(
            options["--challenger-policy-artifact"],
            options["--challenger-value-artifact"],
        )

    def test_wrong_serving_identity_cannot_match_registered_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "serving_model_identity"):
            verifier.validate_identity(
                "0" * 64, runner.EXPECTED["P-E4"], "serving_model"
            )

    def test_assignment_seed_reconstruction_detects_seed_mutation(self) -> None:
        opening = synthetic_opening()
        opening["opening_id"] = "seed615-opening-0000"
        expected = runner.expected_assignments([opening])[0]
        changed = dict(expected, game_seed=expected["game_seed"] + 1)
        with self.assertRaisesRegex(ValueError, "assignment_manifest"):
            verifier.validate_assignment_identity(changed, expected)

    def test_wrong_registered_opening_manifest_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "registered_opening_manifest"):
            runner.validate_registered_digest(
                "0" * 64, runner.EXPECTED["openings"], "opening_manifest"
            )

    def test_assignment_count_and_duplicates_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "assignment_manifest"):
            verifier.validate_assignment_rows([], [{}] * 4096)

    def test_incorrect_seat_fails_game_verification(self) -> None:
        assignment, data = synthetic_game(0)
        altered = dict(data["game"], challenger_player=1)
        with self.assertRaisesRegex(ValueError, "seat_or_opening"):
            verifier.verify_game(assignment, data["opening"], altered)

    def test_illegal_game_transition_fails(self) -> None:
        assignment, data = synthetic_game()
        row = dict(data["game"])
        row["trajectory"] = "99," + row["trajectory"]
        row["game_length"] += 1
        with self.assertRaisesRegex(ValueError, "illegal_trajectory"):
            verifier.verify_game(assignment, data["opening"], row)

    def test_wrong_terminal_margin_fails(self) -> None:
        assignment, data = synthetic_game()
        row = dict(data["game"], margin=int(data["game"]["margin"]) + 1)
        with self.assertRaisesRegex(ValueError, "margin_mismatch"):
            verifier.verify_game(assignment, data["opening"], row)

    def test_candidate_value_artifact_must_not_replace_serving_value(self) -> None:
        assignment = {"candidate": "B512", "candidate_seat": 0, "game_seed": 12}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opening.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            command = runner._arena_command(
                Path("/root"), assignment, path, Path(directory)
            )
            verifier.validate_arena_command(
                command, assignment, Path("/root"), path, Path(directory)
            )
            candidate = str(Path("/root") / runner.OUT / "runtime-artifacts/B512")
            command[command.index("--challenger-value-artifact") + 1] = candidate
            with self.assertRaisesRegex(ValueError, "arena_command_contract"):
                verifier.validate_arena_command(
                    command, assignment, Path("/root"), path, Path(directory)
                )

    def test_tablebase_asset_size_and_digest_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "tablebase"
            asset.write_bytes(b"LFS pointer")
            with self.assertRaisesRegex(ValueError, "materialized_tablebase_asset"):
                runner.validate_asset(
                    asset,
                    runner.EXPECTED["tablebase"],
                    709635072,
                    "materialized_tablebase",
                )
            with self.assertRaisesRegex(ValueError, "native_probe_asset"):
                runner.validate_asset(
                    Path(directory) / "missing",
                    runner.EXPECTED["native_probe"],
                    106776,
                    "native_probe",
                )

    def test_authorization_requires_explicit_freeze_binding(self) -> None:
        valid = {
            "schema": "seed615-evaluation-authorization-v1",
            "status": "authorized-for-evaluation",
            "evaluation_freeze_sha256": "a" * 64,
            "authorization": {
                "type": "explicit-project-owner-authorization",
                "authorized_by": "synthetic-owner",
                "run_games": True,
            },
        }
        verifier.validate_authorization_record(valid, "a" * 64)
        valid["evaluation_freeze_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "freeze_binding"):
            verifier.validate_authorization_record(valid, "a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "authorization_missing"):
                verifier.verify_authorization(Path(directory))

    def test_bootstrap_contract_rejects_wrong_shape_and_resample_count(self) -> None:
        values = np.zeros(1024)
        with self.assertRaisesRegex(ValueError, "bootstrap_contract"):
            runner.paired_bootstrap(values[:10], values[:10])
        with self.assertRaisesRegex(ValueError, "bootstrap_contract"):
            runner.paired_bootstrap(values, values, resamples=10)

    def test_bootstrap_matches_independent_synthetic_reference(self) -> None:
        left = np.linspace(0.1, 0.9, 1024)
        right = np.linspace(0.2, 0.8, 1024)
        observed = runner.paired_bootstrap(left, right)
        rng = np.random.default_rng(615)
        indexes = rng.integers(0, 1024, size=(20000, 1024))
        delta = np.mean((right - left)[indexes], axis=1)
        absolute = np.mean(right[indexes], axis=1)
        expected = (
            [float(x) for x in np.quantile(delta, [0.025, 0.975], method="linear")],
            float(np.quantile(absolute, 0.05, method="linear")),
        )
        self.assertEqual(observed, expected)

    def test_incorrect_bootstrap_quantiles_are_rejected(self) -> None:
        statistics = {
            "clusters": 1024,
            "resamples": 20000,
            "seed": 615,
            "generator": "numpy.random.default_rng(615)",
            "quantile_method": "numpy.quantile(method='linear')",
            "primary_quantiles": [0.025, 0.975],
            "primary_strict_lower_endpoint_gt": 0.0,
            "serving_lower_quantile": 0.05,
            "serving_strict_lower_bound_gt": 0.50,
            "paired_opening_indices": True,
            "preserve_both_seat_outcomes": True,
            "interpretations": {
                "absolute_only": "absolute strength established; incremental improvement not established",
                "both_pass": "incremental improvement and absolute strength established",
                "incremental_only": "improvement over B512 established; absolute strength not established; do not promote automatically",
                "neither": "neither claim established",
            },
        }
        verifier.validate_statistics_contract(statistics)
        statistics["primary_quantiles"] = [0.05, 0.95]
        with self.assertRaisesRegex(ValueError, "statistical_contract"):
            verifier.validate_statistics_contract(statistics)

    def test_analysis_rejects_incomplete_or_duplicate_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete_game_ledger"):
            runner.analyze_scores([])
        duplicate = [
            {"opening_index": 0, "candidate": "B512", "candidate_seat": 0, "margin": 0}
        ] * 4096
        with self.assertRaisesRegex(ValueError, "missing_or_duplicate"):
            runner.analyze_scores(duplicate)

    def test_analysis_rejects_broken_seat_pairing(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_or_duplicate"):
            runner.analyze_scores(
                [
                    {
                        "opening_index": 0,
                        "candidate": "S615",
                        "candidate_seat": 0,
                        "margin": 0,
                    }
                ]
                * 4096
            )

    def test_analysis_is_blocked_until_full_verified_ledger(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete_game_ledger"):
            runner.analyze_scores([{}] * 4095)

    def test_result_artifacts_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            runner.atomic_create(path, b"original")
            with self.assertRaises(FileExistsError):
                runner.atomic_create(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")

    def test_wrong_freeze_binding_is_rejected(self) -> None:
        record = {
            "schema": "seed615-evaluation-authorization-v1",
            "status": "authorized-for-evaluation",
            "evaluation_freeze_sha256": "f" * 64,
            "authorization": {
                "type": "explicit-project-owner-authorization",
                "authorized_by": "synthetic-owner",
                "run_games": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "freeze_binding"):
            verifier.validate_authorization_record(
                record, hashlib.sha256(b"freeze").hexdigest()
            )

    def test_result_hash_tamper_is_detected(self) -> None:
        result = {
            "game_sha256": hashlib.sha256(b"game").hexdigest(),
            "report_sha256": hashlib.sha256(b"report").hexdigest(),
            "command_sha256": hashlib.sha256(b"command").hexdigest(),
        }
        verifier.validate_result_hashes(result, b"game", b"report", b"command")
        with self.assertRaisesRegex(ValueError, "modified_or_conflicting"):
            verifier.validate_result_hashes(result, b"tampered", b"report", b"command")

    def test_inconsistent_capped_game_is_rejected(self) -> None:
        assignment, data = synthetic_game()
        row = dict(data["game"], trajectory="", game_length=0)
        with self.assertRaisesRegex(ValueError, "nonterminal_game_before_cap"):
            verifier.verify_game(assignment, data["opening"], row)

    def test_game_cap_boundary_distinguishes_terminal_and_capped(self) -> None:
        cases = (
            ("terminal_before_cap", 20, 200, False),
            ("terminal_on_cap", 212, 200, False),
            ("nonterminal_at_cap", None, 200, True),
        )
        for label, terminal_at, continuation, expected_capped in cases:
            with self.subTest(label=label):
                assignment, opening, game_row = synthetic_boundary_case(
                    terminal_at=terminal_at,
                    continuation_plies=continuation,
                )
                verified = verifier.verify_game(
                    assignment,
                    opening,
                    game_row,
                    game_factory=lambda: _BoundaryGame(terminal_at),
                )
                self.assertEqual(
                    verified["game_length"],
                    continuation if label != "terminal_before_cap" else 8,
                )
                self.assertIs(verified["capped"], expected_capped)

    def test_nonterminal_game_ending_before_cap_is_rejected(self) -> None:
        assignment, opening, game_row = synthetic_boundary_case(
            terminal_at=None,
            continuation_plies=199,
        )
        with self.assertRaisesRegex(ValueError, "nonterminal_game_before_cap"):
            verifier.verify_game(
                assignment,
                opening,
                game_row,
                game_factory=lambda: _BoundaryGame(None),
            )

    def test_independent_analysis_rejects_corrupted_statistical_value(self) -> None:
        outcomes = synthetic_analysis_records()
        reference = verifier.reference_analysis(outcomes)
        document = verifier.analysis_document(reference, capped_games=0)
        document["results"]["S615_score"] += 0.01
        with self.assertRaisesRegex(ValueError, "independent_analysis_mismatch"):
            verifier.validate_analysis_document(document, reference, capped_games=0)

    def test_independent_analysis_rejects_corrupted_metadata_and_schema(self) -> None:
        reference = verifier.reference_analysis(synthetic_analysis_records())
        valid = verifier.analysis_document(reference, capped_games=0)
        invalid_metadata = json.loads(json.dumps(valid))
        invalid_metadata["bootstrap"]["seed"] = 604
        with self.assertRaisesRegex(ValueError, "independent_analysis_mismatch"):
            verifier.validate_analysis_document(
                invalid_metadata, reference, capped_games=0
            )
        invalid_schema = {**valid, "unexpected": True}
        with self.assertRaisesRegex(ValueError, "analysis_schema_mismatch"):
            verifier.validate_analysis_document(
                invalid_schema, reference, capped_games=0
            )

    def test_independent_analysis_rejects_corrupted_gate_and_promotion_decision(
        self,
    ) -> None:
        reference = verifier.reference_analysis(synthetic_analysis_records())
        valid = verifier.analysis_document(reference, capped_games=0)
        invalid_gate = json.loads(json.dumps(valid))
        invalid_gate["results"]["incremental_improvement_pass"] = False
        with self.assertRaisesRegex(ValueError, "independent_analysis_mismatch"):
            verifier.validate_analysis_document(invalid_gate, reference, capped_games=0)
        invalid_promotion = json.loads(json.dumps(valid))
        invalid_promotion["results"]["automatic_promotion"] = True
        with self.assertRaisesRegex(ValueError, "independent_analysis_mismatch"):
            verifier.validate_analysis_document(
                invalid_promotion, reference, capped_games=0
            )

    def test_independent_bootstrap_does_not_call_production_analysis(self) -> None:
        outcomes = synthetic_analysis_records()
        with mock.patch.object(
            runner, "analyze_scores", side_effect=AssertionError("production called")
        ):
            reference = verifier.reference_analysis(outcomes)
        self.assertEqual(reference["interpretation"], "both_pass")

    def test_create_only_attempt_policy_does_not_delete_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt-0001"
            attempt.mkdir()
            game = attempt / "game.jsonl"
            game.write_text("partial\n", encoding="utf-8")
            self.assertTrue(game.exists())
            self.assertEqual(game.read_text(encoding="utf-8"), "partial\n")
            with self.assertRaisesRegex(ValueError, "incomplete_attempt"):
                runner.validate_no_uncommitted_attempt(attempt)

    def test_preexisting_authorization_results_or_analysis_rejected_in_preflight(
        self,
    ) -> None:
        runner.validate_preflight_artifact_state(False, False, False)
        for state in ((True, False, False), (False, True, False), (False, False, True)):
            with self.assertRaisesRegex(ValueError, "unexpected_registered"):
                runner.validate_preflight_artifact_state(*state)


if __name__ == "__main__":
    unittest.main()
