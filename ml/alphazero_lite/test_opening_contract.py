from __future__ import annotations

import unittest
import json
from pathlib import Path

from ml.alphazero_lite.build_opening_suite import (
    absolute_prefix_to_relative,
    canonical_key,
    enumerate_legal_prefixes,
    export_arena_entry,
    validate_arena_entries,
)


class OpeningContractTests(unittest.TestCase):
    def test_absolute_conversion_and_arena_replay(self) -> None:
        prefixes = enumerate_legal_prefixes(5)
        self.assertTrue(
            any(
                absolute != absolute_prefix_to_relative(absolute)
                for absolute in (entry["prefix_moves"] for entry in prefixes)
            )
        )
        # Explicitly find an extra-turn prefix and verify the next action uses
        # the same player's relative coordinates.
        from ml.alphazero_lite.kalah_rules import KalahGame

        for entry in prefixes:
            moves = entry["prefix_moves"]
            game = KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            relative = []
            same_player = False
            previous = game.current_player
            for absolute in moves:
                relative.append(absolute - game.current_player * 6)
                game.move(absolute)
                same_player |= game.current_player == previous
                previous = game.current_player
            if same_player and len(moves) > 1:
                self.assertEqual(relative, absolute_prefix_to_relative(moves))
                break
        else:
            self.fail("enumeration did not include an extra-turn prefix")

    def test_v2_entries_replay_to_declared_unique_states(self) -> None:
        entries = enumerate_legal_prefixes(4)
        unique = {}
        for entry in entries:
            unique.setdefault(canonical_key(entry["state"]), entry)
        exported = [export_arena_entry(entry) for entry in unique.values()]
        self.assertEqual(len(validate_arena_entries(exported)), len(exported))

    def test_published_391_and_392_suites_convert_completely(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for seed in (391, 392):
            path = (
                root / f"docs/data/order38615-a5-confirmation-seed{seed}-openings.jsonl"
            )
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            converted = []
            for row in rows:
                item = dict(row)
                item["prefix_moves"] = absolute_prefix_to_relative(row["prefix_moves"])
                item["opening_contract"] = "arena_player_relative_v2"
                item["state_hash"] = canonical_key(row["state"])
                converted.append(item)
            self.assertEqual(len(validate_arena_entries(converted)), 512)

    def test_canonical_prefilter_256_prefixes_remain_valid(self) -> None:
        from ml.alphazero_lite.arena import apply_opening_moves
        from ml.alphazero_lite.kalah_rules import KalahGame
        from ml.alphazero_lite.build_opening_suite import INITIAL_STATE

        root = Path(__file__).resolve().parents[2]
        path = (
            root
            / "docs/data/alphazero-lite-production-prefilter-calibration-openings-v1.jsonl"
        )
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        self.assertEqual(len(rows), 256)
        actual = set()
        for row in rows:
            game = KalahGame.from_state(INITIAL_STATE)
            self.assertEqual(apply_opening_moves(game, row["prefix_moves"]), 4)
            identity = canonical_key(game.to_state())
            self.assertEqual(identity, row["canonical_resulting_state_hash"])
            actual.add(identity)
        self.assertEqual(len(actual), 256)

    def test_state_mismatch_and_actual_duplicates_are_rejected(self) -> None:
        rows = enumerate_legal_prefixes(3)
        one_per_state = {}
        for row in rows:
            one_per_state.setdefault(canonical_key(row["state"]), row)
        valid = [export_arena_entry(row) for row in one_per_state.values()]
        mismatched = dict(valid[0])
        mismatched["state"] = valid[-1]["state"]
        with self.assertRaisesRegex(ValueError, "state does not match"):
            validate_arena_entries([mismatched])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_arena_entries([valid[0], valid[0]])

    def test_evidence_rejects_wrong_actual_state_with_matching_prefix_array(
        self,
    ) -> None:
        from ml.alphazero_lite import arena
        from ml.alphazero_lite.seed461_arena_validation import validate_arena_evidence

        chosen = []
        for row in enumerate_legal_prefixes(1):
            chosen.append(export_arena_entry(row))
            if len(chosen) == 2:
                break
        games = []
        for opening_index, opening in enumerate(chosen):
            game = arena.KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            arena.apply_opening_moves(game, opening["prefix_moves"])
            for seat in (0, 1):
                games.append(
                    {
                        "opening_index": opening_index,
                        "opening_prefix_moves": opening["prefix_moves"],
                        "opening_state_hash": arena.canonical_game_state_hash(game),
                        "opening_applied_prefix_length": len(opening["prefix_moves"]),
                        "opening_contract": "arena_player_relative_v2",
                        "challenger_player": seat,
                        "winner": "challenger",
                    }
                )
        report = {
            "schema": "arena_v1",
            "games": 4,
            "games_played": 4,
            "wins": 4,
            "draws": 0,
            "losses": 0,
            "score": 1.0,
            "notes": {
                "challenger_path": "candidate",
                "current_path": "opponent",
                "suite_sha256": "suite",
                "seed": 7,
                "base_seed": 7,
                "seed_contract": "seed",
                "challenger_simulations": 384,
                "current_simulations": 384,
                "search_profile": {"c_puct": 1.25, "simulations": 384},
            },
        }
        suite = {"sha256": "suite"}
        candidate = {"artifact": "candidate", "runtime_contract": {}}
        opponent = {"artifact": "opponent"}
        evaluation = {
            "games_per_candidate": 4,
            "suite": suite,
            "arena_seed": 7,
            "seed_contract": "seed",
            "simulations_per_side": 384,
            "c_puct": 1.25,
        }
        validate_arena_evidence(
            report, games, chosen, "test", candidate, opponent, evaluation
        )
        corrupted = [dict(row) for row in games]
        corrupted[0]["opening_state_hash"] = "wrong-state"
        with self.assertRaisesRegex(
            ValueError, "actual_opening_state_identity_mismatch"
        ):
            validate_arena_evidence(
                report, corrupted, chosen, "test", candidate, opponent, evaluation
            )
        incompatible = [dict(row) for row in games]
        incompatible[0]["opening_contract"] = "legacy_player_relative_v1"
        with self.assertRaisesRegex(ValueError, "actual_opening_contract_mismatch"):
            validate_arena_evidence(
                report, incompatible, chosen, "test", candidate, opponent, evaluation
            )

    def test_illegal_or_truncated_prefix_is_rejected(self) -> None:
        entry = export_arena_entry(enumerate_legal_prefixes(1)[0])
        entry["prefix_moves"] = [99]
        with self.assertRaises(ValueError):
            validate_arena_entries([entry])


if __name__ == "__main__":
    unittest.main()
