"""Synthetic-only tests for Seed 604 evaluation contracts."""

from __future__ import annotations

import unittest
import tempfile
import hashlib
from pathlib import Path

import numpy as np

from ml.alphazero_lite import arena, seed604_evaluation as runner
from ml.alphazero_lite import verify_seed604_evaluation as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


def synthetic_game_row(seat: int = 0) -> dict[str, object]:
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    opening_state = game.to_state()
    trajectory: list[int] = []
    observed_extra_turn = False
    while not game.over() and len(trajectory) < 200:
        player = game.current_player
        move = game.pit_index(game.possible_moves()[0])
        trajectory.append(move)
        if not game.move(move):
            raise AssertionError("synthetic legal move rejected")
        observed_extra_turn |= game.current_player == player
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    opening_identity = hashlib.sha256(verifier.canonical(opening_state)).hexdigest()
    return {
        "game_index": 0,
        "game_within_opening": 0,
        "opening_index": 0,
        "opening_state_hash": opening_identity,
        "opening_prefix_moves": [],
        "opening_applied_prefix_length": 0,
        "challenger_player": seat,
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": "challenger" if margin > 0 else "current" if margin < 0 else "draw",
        "synthetic_extra_turn_observed": observed_extra_turn,
    }


class _Evaluator:
    def __init__(self, policy: list[float], value: float):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.value = value

    def evaluate(self, game: object) -> tuple[np.ndarray, float]:
        del game
        return self.policy, self.value


class Seed604EvaluationTests(unittest.TestCase):
    def test_assignment_manifest_has_exact_four_assignments_per_opening(self) -> None:
        rows = runner.build_assignments()
        self.assertEqual(len(rows), 4096)
        self.assertEqual(len({row["game_id"] for row in rows}), 4096)
        self.assertEqual(rows[0]["candidate_seat"], 0)
        self.assertEqual(rows[1]["candidate_seat"], 1)
        self.assertEqual(rows[2]["candidate"], "B2048")

    def test_missing_and_duplicated_assignment_fail(self) -> None:
        expected = [{"game_id": str(index)} for index in range(4096)]
        with self.assertRaisesRegex(ValueError, "assignment_manifest"):
            verifier.validate_assignment_ledger(expected[:-1], expected)
        duplicated = list(expected)
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ValueError, "assignment_manifest"):
            verifier.validate_assignment_ledger(duplicated, expected)

    def test_wrong_checkpoint_identity_fails(self) -> None:
        expected = {"game_id": "synthetic", "candidate_checkpoint_sha256": "a" * 64}
        altered = {**expected, "candidate_checkpoint_sha256": "b" * 64}
        with self.assertRaisesRegex(ValueError, "checkpoint_mismatch"):
            verifier.validate_assignment(altered, expected)

    def test_value_source_must_remain_serving_p_e4(self) -> None:
        assignment = {"candidate": "B2048", "candidate_seat": 1, "game_seed": 17}
        with tempfile.TemporaryDirectory() as directory:
            opening_path = Path(directory) / "synthetic-opening.jsonl"
            opening_path.write_text('{"prefix_moves":[]}\n', encoding="utf-8")
            argv = runner._arena_command(
                Path("/synthetic-root"), assignment, opening_path, Path("/out")
            )
            verifier.validate_candidate_value_sources(
                argv, assignment, Path("/synthetic-root")
            )
            value_index = argv.index("--challenger-value-artifact") + 1
            argv[value_index] = argv[argv.index("--challenger-policy-artifact") + 1]
            with self.assertRaisesRegex(ValueError, "composition"):
                verifier.validate_candidate_value_sources(
                    argv, assignment, Path("/synthetic-root")
                )

    def test_seat_inversion_and_illegal_transition_fail(self) -> None:
        row = synthetic_game_row(0)
        assignment = {
            "candidate_seat": 0,
            "prefix_moves": [],
            "opening_identity": row["opening_state_hash"],
        }
        verifier.verify_game(assignment, row)
        wrong_seat = {**row, "challenger_player": 1}
        with self.assertRaisesRegex(ValueError, "seat_inversion"):
            verifier.verify_game(assignment, wrong_seat)
        bad_trajectory = {
            **row,
            "trajectory": "6," + str(row["trajectory"]).split(",", 1)[1],
        }
        with self.assertRaisesRegex(ValueError, "player_to_move|illegal_trajectory"):
            verifier.verify_game(assignment, bad_trajectory)

    def test_extra_turn_does_not_change_candidate_or_value_source(self) -> None:
        candidate = _Evaluator([1, 2, 3, 4, 5, 6], 0.91)
        serving = _Evaluator([6, 5, 4, 3, 2, 1], -0.37)
        composed = arena.ComposedArtifactEvaluator(
            serving, candidate, policy_source="candidate", value_source="current"
        )
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        extra_turn_seen = False
        while not game.over():
            policy, value = composed.evaluate(game)
            np.testing.assert_array_equal(policy, candidate.policy)
            self.assertEqual(value, -0.37)
            old_player = game.current_player
            game.move(game.pit_index(game.possible_moves()[0]))
            extra_turn_seen |= old_player == game.current_player
        self.assertTrue(extra_turn_seen)

    def test_search_noise_and_modified_configuration_fail(self) -> None:
        valid = dict(verifier.EXPECTED_SEARCH)
        verifier.validate_search_contract(valid)
        valid["noise"] = True
        with self.assertRaisesRegex(ValueError, "search_contract"):
            verifier.validate_search_contract(valid)

    def test_statistical_quantiles_and_gates_are_frozen(self) -> None:
        expected = {
            "unit": "opening group",
            "clusters": 1024,
            "resamples": 20000,
            "seed": 604,
            "quantile_method": "linear",
            "primary_lower_quantile": 0.05,
            "primary_strict_lower_bound_gt": 0.5,
            "secondary_quantiles": [0.025, 0.975],
            "secondary_strict_lower_endpoint_gt": 0.0,
            "interpretation": {
                "both_pass": "absolute strength and additional optimization benefit established",
                "primary_only": "absolute strength established; incremental improvement not established",
                "secondary_only": "incremental improvement established; absolute strength not established",
                "neither": "neither claim established",
            },
        }
        verifier.validate_statistics_contract(expected)
        expected["primary_lower_quantile"] = 0.025
        with self.assertRaisesRegex(ValueError, "statistical_contract"):
            verifier.validate_statistics_contract(expected)
        valid = dict(verifier.EXPECTED_SEARCH)
        valid["simulations"] = 129
        with self.assertRaisesRegex(ValueError, "search_contract"):
            verifier.validate_search_contract(valid)

    def test_capped_or_terminal_margin_mismatch_fails(self) -> None:
        row = synthetic_game_row()
        assignment = {
            "candidate_seat": 0,
            "prefix_moves": [],
            "opening_identity": row["opening_state_hash"],
        }
        altered = {**row, "margin": int(row["margin"]) + 1}
        with self.assertRaisesRegex(ValueError, "outcome_or_capped"):
            verifier.verify_game(assignment, altered)

    def test_capped_game_uses_store_margin_from_candidate_seat(self) -> None:
        self.assertEqual(
            verifier.store_margin_outcome((27, 21), 0), (6, "challenger", 1.0)
        )
        self.assertEqual(
            verifier.store_margin_outcome((27, 21), 1), (-6, "current", 0.0)
        )
        self.assertEqual(verifier.store_margin_outcome((24, 24), 1), (0, "draw", 0.5))

    def test_native_and_tablebase_asset_identity_requires_bytes(self) -> None:
        payload = b"synthetic tablebase object"
        import hashlib

        identity = hashlib.sha256(payload).hexdigest()
        verifier.validate_asset_bytes(payload, identity, len(payload))
        with self.assertRaisesRegex(ValueError, "asset_size_or_hash"):
            verifier.validate_asset_bytes(payload, "0" * 64, len(payload))
        with self.assertRaisesRegex(ValueError, "asset_size_or_hash"):
            verifier.validate_asset_bytes(payload, identity, len(payload) + 1)

    def test_incomplete_game_ledger_fails_before_analysis(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete_game_ledger"):
            verifier.analyze_independently([{}] * 4095)

    def test_paired_bootstrap_matches_small_independent_calculation(self) -> None:
        left = np.asarray([0.0, 0.5, 1.0, 0.25])
        right = np.asarray([1.0, 0.5, 0.0, 0.75])
        observed = verifier.paired_bootstrap_reference(
            left, right, resamples=37, seed=91
        )
        rng = np.random.default_rng(91)
        selected = rng.integers(0, 4, size=(37, 4))
        primary = np.asarray([np.mean(right[index]) for index in selected])
        difference = right - left
        paired = np.asarray([np.mean(difference[index]) for index in selected])
        expected = (
            float(np.quantile(primary, 0.05, method="linear")),
            [float(x) for x in np.quantile(paired, [0.025, 0.975], method="linear")],
        )
        self.assertEqual(observed, expected)

    def test_strict_preregistered_thresholds(self) -> None:
        self.assertEqual(runner.interpret_gates(0.5, 0.0)["interpretation"], "neither")
        self.assertEqual(
            runner.interpret_gates(0.51, 0.0)["interpretation"], "primary_only"
        )
        self.assertEqual(
            runner.interpret_gates(0.5, 0.01)["interpretation"], "secondary_only"
        )
        self.assertEqual(
            runner.interpret_gates(0.51, 0.01)["interpretation"], "both_pass"
        )

    def test_repeated_execution_without_authorization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "authorization_required"):
            runner.run(Path("/synthetic-empty-root"))

    def test_modified_result_artifact_is_rejected(self) -> None:
        game, report, command = b"game", b"report", b"command"
        result = {
            "game_sha256": __import__("hashlib").sha256(game).hexdigest(),
            "report_sha256": __import__("hashlib").sha256(report).hexdigest(),
            "command_sha256": __import__("hashlib").sha256(command).hexdigest(),
        }
        verifier.validate_result_hashes(result, game, report, command)
        with self.assertRaisesRegex(ValueError, "modified_or_conflicting"):
            verifier.validate_result_hashes(result, b"modified", report, command)

    def test_create_only_artifact_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            runner.atomic_create(path, b"first")
            with self.assertRaises(FileExistsError):
                runner.atomic_create(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")


if __name__ == "__main__":
    unittest.main()
