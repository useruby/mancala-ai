from __future__ import annotations

import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

from ml.alphazero_lite.build_opening_suite import (
    absolute_prefix_to_relative,
    canonical_key,
    enumerate_legal_prefixes,
    export_arena_entry,
    validate_arena_entries,
)


class OpeningContractTests(unittest.TestCase):
    def test_exclusion_contract_detects_both_collision_views_and_duplicates(
        self,
    ) -> None:
        from ml.alphazero_lite.opening_exclusion_contract import (
            opening_identities,
            validate_exclusions,
            validate_suite_set,
        )

        unique = {}
        for row in enumerate_legal_prefixes(2):
            unique.setdefault(canonical_key(row["state"]), row)
        entries = [export_arena_entry(row) for row in unique.values()]
        declared, actual = opening_identities(entries)
        with self.assertRaisesRegex(ValueError, "declared_exclusion_overlap"):
            validate_exclusions(entries, {declared[0]}, set())
        with self.assertRaisesRegex(ValueError, "actual_exclusion_overlap"):
            validate_exclusions(entries, set(), {actual[0]})
        with self.assertRaisesRegex(ValueError, "duplicate_start"):
            validate_exclusions([entries[0], entries[0]], set(), set())
        with self.assertRaisesRegex(ValueError, "actual_exclusion_overlap"):
            validate_exclusions(entries[1:], set(), {actual[1]})
        permissive_manifest = {
            "declared_state_identities": [],
            "actual_state_identities": [],
        }
        with self.assertRaisesRegex(ValueError, "cross_suite_opening_overlap"):
            validate_suite_set(
                {"left": entries[:2], "right": [entries[1]]}, permissive_manifest
            )

    def test_manifest_recomputation_rejects_changed_manifest_and_source_bytes(
        self,
    ) -> None:
        from ml.alphazero_lite.opening_exclusion_contract import (
            create_manifest,
            verify_manifest,
        )

        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            source = Path(directory) / "source.jsonl"
            manifest_path = Path(directory) / "manifest.json"
            entry = export_arena_entry(enumerate_legal_prefixes(1)[0])
            source.write_text(json.dumps(entry) + "\n")
            manifest = create_manifest(
                [{"path": str(source), "kind": "historical_suite"}]
            )
            manifest_path.write_text(json.dumps(manifest))
            verify_manifest(manifest_path)
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "stale_or_mutated"):
                verify_manifest(manifest_path)
            source.write_text(json.dumps(entry) + "\n")
            manifest["excluded_state_count"] += 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "stale_or_mutated"):
                verify_manifest(manifest_path)

    def test_runner_rejects_changed_suite_or_stale_resume_binding_before_arena(
        self,
    ) -> None:
        from ml.alphazero_lite import run_corrected_order38615_diagnostic as runner

        root = Path(__file__).resolve().parents[2]
        committed_registration = (
            root / "docs/data/order38615-a5-frozen-diagnostic-v4/registration.json"
        )
        registration = json.loads(committed_registration.read_text())
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            temp = Path(directory)
            source_suite = root / registration["evaluation"]["suites"]["395"]["path"]
            copied_suite = temp / "suite.jsonl"
            copied_suite.write_bytes(source_suite.read_bytes())
            registration["evaluation"]["suites"]["395"]["path"] = str(copied_suite)
            reg_path = temp / "registration.json"
            bind_path = temp / "binding.json"
            reg_path.write_text(json.dumps(registration))
            binding = {
                "registration_sha256": runner.sha(reg_path),
                "source_candidate_binding_sha256": runner.sha(runner.SOURCE_BIND),
                "exclusion_manifest_sha256": registration["exclusion_proof"][
                    "manifest_sha256"
                ],
                "reports": {},
            }
            bind_path.write_text(json.dumps(binding))
            copied_suite.write_bytes(copied_suite.read_bytes() + b"\n")
            frozen_manifest = json.loads(
                (
                    root
                    / "docs/data/order38615-a5-frozen-diagnostic-v4/opening-exclusion-manifest.json"
                ).read_text()
            )
            with (
                patch.object(runner, "REG", reg_path),
                patch.object(runner, "BIND", bind_path),
                patch.object(runner, "WORK", temp / "work"),
                patch.object(runner, "verify_manifest", return_value=frozen_manifest),
                patch.object(runner.subprocess, "run") as arena_call,
            ):
                with self.assertRaisesRegex(ValueError, "suite_hash_mismatch:395"):
                    runner.run()
                arena_call.assert_not_called()

            registration["evaluation"]["suites"]["395"]["path"] = str(source_suite)
            reg_path.write_text(json.dumps(registration))
            binding["registration_sha256"] = runner.sha(reg_path)
            binding["exclusion_manifest_sha256"] = "stale"
            bind_path.write_text(json.dumps(binding))
            with (
                patch.object(runner, "REG", reg_path),
                patch.object(runner, "BIND", bind_path),
                patch.object(runner, "WORK", temp / "work"),
                patch.object(runner.subprocess, "run") as arena_call,
            ):
                with self.assertRaisesRegex(
                    ValueError, "exclusion_manifest_binding_mismatch"
                ):
                    runner.run()
                arena_call.assert_not_called()

    def test_generic_arena_preflight_fails_before_worker_pool(self) -> None:
        from ml.alphazero_lite import arena

        root = Path(__file__).resolve().parents[2]
        candidate = root / ".tmp/seed461-e2-e4-average/artifacts/order_38615_A"
        opponent = root / ".tmp/seed461-order-confirmation/opponent-artifact"
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            temp = Path(directory)
            suite = temp / "truncated.jsonl"
            suite.write_text(json.dumps({"prefix_moves": [99]}) + "\n")
            argv = [
                "arena.py",
                "--challenger",
                str(candidate),
                "--current",
                str(opponent),
                "--games",
                "2",
                "--workers",
                "2",
                "--opening-prefixes-jsonl",
                str(suite),
                "--out",
                str(temp / "out.json"),
            ]
            with (
                patch("sys.argv", argv),
                patch.object(arena.concurrent.futures, "ProcessPoolExecutor") as pool,
            ):
                with self.assertRaisesRegex(ValueError, "truncated_prefix"):
                    arena.main()
                pool.assert_not_called()

    def test_interrupted_runner_restarts_only_after_all_bindings_validate(self) -> None:
        from ml.alphazero_lite import run_corrected_order38615_diagnostic as runner

        root = Path(__file__).resolve().parents[2]
        registration = json.loads(
            (
                root / "docs/data/order38615-a5-frozen-diagnostic-v4/registration.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory(dir=root / ".tmp") as directory:
            temp = Path(directory)
            reg_path = temp / "registration.json"
            bind_path = temp / "binding.json"
            reg_path.write_text(json.dumps(registration))
            binding = {
                "registration_sha256": runner.sha(reg_path),
                "source_candidate_binding_sha256": runner.sha(runner.SOURCE_BIND),
                "exclusion_manifest_sha256": registration["exclusion_proof"][
                    "manifest_sha256"
                ],
                "reports": {"395": {"state": "running"}},
            }
            bind_path.write_text(json.dumps(binding))
            frozen_manifest = json.loads(
                (
                    root
                    / "docs/data/order38615-a5-frozen-diagnostic-v4/opening-exclusion-manifest.json"
                ).read_text()
            )
            with (
                patch.object(runner, "REG", reg_path),
                patch.object(runner, "BIND", bind_path),
                patch.object(runner, "WORK", temp / "work"),
                patch.object(runner, "verify_manifest", return_value=frozen_manifest),
                patch.object(
                    runner.subprocess,
                    "run",
                    side_effect=RuntimeError("arena_launch_marker"),
                ) as arena_call,
            ):
                with self.assertRaisesRegex(RuntimeError, "arena_launch_marker"):
                    runner.run()
                arena_call.assert_called_once()

    def test_formatted_delivery_sources_do_not_pass_the_old_launch_binding(
        self,
    ) -> None:
        from ml.alphazero_lite import run_corrected_order38615_diagnostic as runner

        with patch.object(runner.subprocess, "run") as arena_call:
            with self.assertRaisesRegex(
                ValueError, "exclusion_manifest_stale_or_mutated"
            ):
                runner.run()
            arena_call.assert_not_called()

    def test_publication_snapshots_do_not_weaken_launch_source_hashes(self) -> None:
        from ml.alphazero_lite.opening_exclusion_contract import verify_manifest
        from ml.alphazero_lite.publish_order38615_a5_diagnostic import (
            MANIFEST,
            validate_publication,
        )

        result = validate_publication()
        self.assertEqual(result["status"], "verified_post_execution_publication")
        with self.assertRaisesRegex(ValueError, "exclusion_manifest_stale_or_mutated"):
            verify_manifest(MANIFEST)

    def test_corrected_393_394_declared_collisions_are_rejected(self) -> None:
        from ml.alphazero_lite.opening_exclusion_contract import (
            historical_opening_identities,
            validate_exclusions,
        )
        from ml.alphazero_lite.build_opening_suite import load_suite_jsonl

        root = Path(__file__).resolve().parents[2]
        registration = json.loads(
            (
                root / "docs/data/order38615-corrected-diagnostic/registration.json"
            ).read_text()
        )
        historical = registration["exclusion_proof"]["prior_evaluations"].values()
        declared: set[str] = set()
        actual: set[str] = set()
        for record in historical:
            rows = load_suite_jsonl(str(root / record["path"]))
            prior_declared, prior_actual = historical_opening_identities(rows)
            declared.update(prior_declared)
            actual.update(prior_actual)
        for seed, expected in ((393, 14), (394, 12)):
            spec = registration["evaluation"]["suites"][str(seed)]
            rows = load_suite_jsonl(str(root / spec["path"]))
            from ml.alphazero_lite.opening_exclusion_contract import opening_identities

            candidate_declared, candidate_actual = opening_identities(rows)
            self.assertEqual(len(set(candidate_declared) & declared), expected)
            self.assertEqual(len(set(candidate_actual) & actual), 0)
            with self.assertRaisesRegex(ValueError, "declared_exclusion_overlap"):
                validate_exclusions(rows, declared, actual)

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
