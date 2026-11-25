"""Small deterministic fixtures for Seed 624 generation and selection contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import shutil
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch


from ml.alphazero_lite import seed624_prospective_benchmark as seed624
from ml.alphazero_lite import seed628_authenticated_screening as seed628
from ml.alphazero_lite import verify_seed628_authenticated_screening as seed628_verifier
from ml.alphazero_lite import verify_seed624_prospective_benchmark as independent
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed624_prospective_benchmark import _apply_actions
from ml.alphazero_lite.verify_seed624_prospective_benchmark import (
    validate_candidate_rank,
)


def verify_bundle(bundle_path: Path) -> dict:
    """Keep population tests offline; PR evidence has dedicated unit tests."""
    with patch.object(
        independent, "verify_registration_pr_authorization", return_value="useruby"
    ):
        return independent.verify_bundle(bundle_path)


class Seed624SyntheticTests(unittest.TestCase):
    def _seal_freeze(self, freeze: dict) -> tuple[bytes, dict, bytes]:
        from ml.alphazero_lite.seed636_authorization import (
            APPROVED_DECISIONS,
            EXECUTION_SCOPE,
            REQUIRED_PROTOCOL_DOCUMENTS,
            SCHEMA as AUTHORIZATION_SCHEMA,
            sha256,
        )

        freeze = dict(freeze, status="frozen")
        freeze_raw = seed624.canonical_bytes(freeze)
        documents = {
            name: sha256((seed624.ROOT / name).read_bytes())
            for name in sorted(REQUIRED_PROTOCOL_DOCUMENTS)
        }
        registration = {
            "schema": "mancala-experiment-registration-v1",
            "experiment_id": "seed636-prospective-population",
            "base_commit": seed624.current_git_commit(),
            "authorization_contract": AUTHORIZATION_SCHEMA,
            "execution_scope": EXECUTION_SCOPE,
            "random_seeds": [seed624.MASTER_SEED],
            "protocol_documents": documents,
            "input_artifacts": {},
            "approved_decisions": APPROVED_DECISIONS,
            "resource_budgets": {
                "wall_time_limit_seconds": freeze["production_resource_limits"][
                    "wall_seconds"
                ],
                "cpu_time_limit_seconds": freeze["production_resource_limits"][
                    "cpu_seconds"
                ],
                "process_rss_limit_bytes": freeze["production_resource_limits"][
                    "process_rss_bytes"
                ],
                "output_and_temporary_storage_limit_bytes": freeze[
                    "production_resource_limits"
                ]["output_and_temporary_storage_bytes"],
                "execution_concurrency": freeze["production_resource_limits"][
                    "execution_concurrency"
                ],
                "selection_optimizer_workers": 1,
            },
            "scientific_acceptance": {
                "openings": 256,
                "continuation_games": 2048,
                "selected_states": 512,
                "stone_bin_quotas": {
                    "17": 103,
                    "18": 103,
                    "19": 102,
                    "20": 102,
                    "21": 102,
                },
                "minimum_eligible_candidate_supply": {
                    "17": 206,
                    "18": 206,
                    "19": 204,
                    "20": 204,
                    "21": 204,
                },
                "minimum_distinct_opening_clusters": 128,
                "max_states_per_opening": 4,
                "max_states_per_continuation_game": 2,
                "exposure_classification": "known-exposure-screened; inherited P-E4 exposure unresolved",
                "deduplication": ["canonical_state", "production_float32_input"],
                "authenticated_known_exposure_screening": True,
                "independent_reproduction_required": True,
                "no_inference_mcts_exact_solving_tablebase_probes_training_or_promotion": True,
                "failure_policy": "preserve_evidence_and_stop_without_parameter_change_or_automatic_retry",
            },
        }
        registration_raw = json.dumps(registration, sort_keys=True).encode()
        registration_sha = sha256(registration_raw)
        registration_path = (
            "docs/experiments/seed636-prospective-population/registration.json"
        )
        authorization_record = {
            "schema": AUTHORIZATION_SCHEMA,
            "repository": "useruby/mancala-ai",
            "experiment_id": registration["experiment_id"],
            "execution_scope": EXECUTION_SCOPE,
            "approved_decisions": APPROVED_DECISIONS,
            "registration_sha256": registration_sha,
            "registration_path": registration_path,
            "pr_number": 84,
            "pr_url": "https://github.com/useruby/mancala-ai/pull/84",
            "merge_commit": seed624.current_git_commit(),
            "merged_by": "useruby",
            "merge_timestamp": "2026-11-25T12:30:00Z",
        }
        attempt_manifest = {
            "schema": "mancala-experiment-attempt-v1",
            "status": "frozen",
            "experiment_id": registration["experiment_id"],
            "attempt_id": "synthetic-attempt",
            "source_commit": freeze.get("git_commit"),
            "command": [
                "python",
                "-m",
                "ml.alphazero_lite.seed624_prospective_benchmark",
            ],
            "configuration": {
                "execution_scope": EXECUTION_SCOPE,
                "registration_sha256": registration_sha,
                "seed624_freeze": freeze,
            },
            "input_artifacts": {},
            "registration_path": registration_path,
            "registration_sha256": registration_sha,
            "worker_count": freeze["production_resource_limits"][
                "execution_concurrency"
            ],
            "resource_budget": {
                "wall_time_limit_seconds": freeze["production_resource_limits"][
                    "wall_seconds"
                ],
                "additional_limits": {
                    "cpu_seconds": freeze["production_resource_limits"]["cpu_seconds"],
                    "process_rss_bytes": freeze["production_resource_limits"][
                        "process_rss_bytes"
                    ],
                    "output_and_temporary_storage_bytes": freeze[
                        "production_resource_limits"
                    ]["output_and_temporary_storage_bytes"],
                },
            },
        }
        lifecycle = {
            "experiment_id": registration["experiment_id"],
            "attempt_id": "synthetic-attempt",
            "attempt_manifest_sha256": sha256(
                json.dumps(attempt_manifest, sort_keys=True).encode()
            ),
            "attempt_manifest_raw_utf8": json.dumps(attempt_manifest, sort_keys=True),
            "registration_sha256": registration_sha,
            "registration_path": registration_path,
            "registration_raw_utf8": registration_raw.decode(),
        }
        self._fixture_lifecycle = lifecycle
        return (
            freeze_raw,
            authorization_record,
            seed624.canonical_bytes(authorization_record),
        )

    def test_seed_derivation_and_opening_are_reproducible(self) -> None:
        self.assertEqual(
            seed624.derive_seed(seed624.MASTER_SEED, 3, 0, "opening"),
            seed624.derive_seed(seed624.MASTER_SEED, 3, 0, "opening"),
        )
        self.assertEqual(
            seed624.generate_opening(seed624.MASTER_SEED, 3),
            seed624.generate_opening(seed624.MASTER_SEED, 3),
        )

    def test_opening_index_outside_reserve_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "opening_index_outside_frozen_reserve"):
            seed624.generate_opening(seed624.MASTER_SEED, seed624.OPENING_RESERVE)

    def test_duplicate_openings_exhaust_the_fixed_reserve(self) -> None:
        rows = [
            {"canonical_identity": "same", "production_float32_identity": "same-float"}
            for _ in range(3)
        ]
        with self.assertRaises(seed624.ShortfallError) as error:
            seed624.choose_distinct_openings(
                lambda index: rows[index], reserve=3, target=2
            )
        self.assertEqual(error.exception.report["available"], 1)
        self.assertEqual(error.exception.report["consumed_seed_indexes"], [0, 1, 2])
        self.assertEqual(len(error.exception.report["rejections"]), 2)

    def test_invalid_prefix_consumes_its_index_records_failure_and_skips(self) -> None:
        invalid_seed = seed624.derive_seed(seed624.MASTER_SEED, 0, 0, "opening")
        terminal_state = KalahGame(
            pits=[0] * 12, captured_seeds=[24, 24], current_player=0
        ).to_state()
        cid, fid = seed624.identities(terminal_state)

        def factory(index: int) -> dict:
            if index == 0:
                raise seed624.InvalidOpeningPrefix(
                    {
                        "opening_index": 0,
                        "seed": invalid_seed,
                        "status": "rejected_invalid_prefix",
                        "reason": "terminal_before_eight_actions",
                        "actions": [5],
                        "terminal": True,
                        "state": terminal_state,
                        "canonical_identity": cid,
                        "production_float32_identity": fid,
                    }
                )
            return seed624.generate_opening(seed624.MASTER_SEED, index)

        selection = seed624.choose_distinct_openings(factory, reserve=2, target=1)
        self.assertEqual(selection["consumed_seed_indexes"], [0, 1])
        self.assertEqual(selection["openings"][0]["opening_index"], 1)
        self.assertEqual(selection["attempts"][0]["status"], "rejected_invalid_prefix")
        self.assertEqual(
            selection["rejections"][0]["reason"], "terminal_before_eight_actions"
        )

    def test_independent_reserve_replay_skips_and_records_invalid_prefix(self) -> None:
        invalid = {
            "opening_index": 0,
            "seed": independent._seed(seed624.MASTER_SEED, 0, 0, "opening"),
            "status": "rejected_invalid_prefix",
            "reason": "terminal_before_eight_actions",
            "actions": [5],
            "terminal": True,
            "state": {},
            "canonical_identity": "invalid-canonical",
            "production_float32_identity": "invalid-float",
        }
        opening = seed624.generate_opening(seed624.MASTER_SEED, 1)
        valid_attempt = {
            "opening_index": 1,
            "seed": opening["seed"],
            "status": "valid_prefix",
            "canonical_identity": opening["canonical_identity"],
            "production_float32_identity": opening["production_float32_identity"],
        }
        with patch.object(
            independent,
            "_replay_opening_seed",
            side_effect=[(invalid, None), (valid_attempt, opening)],
        ):
            selected, attempts, rejections = independent._reconstruct_opening_reserve(
                [0, 1], 1
            )
        self.assertEqual(selected, [opening])
        self.assertEqual(
            [row["status"] for row in attempts], ["rejected_invalid_prefix", "accepted"]
        )
        self.assertEqual(rejections, [invalid])

    def test_opening_seed_generation_preserves_eight_action_prefix(self) -> None:
        opening = seed624.generate_opening(seed624.MASTER_SEED, 0)
        self.assertEqual(len(opening["prefix_actions"]), 8)
        game = seed624.standard_game()
        for action in opening["prefix_actions"]:
            self.assertIn(action, game.possible_moves())
            self.assertTrue(game.move(game.pit_index(action)))
        self.assertEqual(game.to_state(), opening["state"])

    def test_within_game_bin_uses_minimum_rank_across_all_visits(self) -> None:
        opening = seed624.generate_opening(seed624.MASTER_SEED, 0)
        action_records = []
        game, rows = seed624.generate_game(
            opening,
            seed624.MASTER_SEED,
            1,
            on_action=action_records.append,
        )
        self.assertEqual([row["action"] for row in action_records], game["actions"])
        self.assertEqual(
            [row["continuation_ply"] for row in action_records],
            list(range(len(game["actions"]))),
        )
        candidate = next(row for row in rows if row["active_pit_stones"] == 19)
        visits = candidate["within_game_bin_visits"]
        self.assertEqual(len(visits), 3)
        self.assertEqual(candidate["rank"], min(row["rank"] for row in visits))
        self.assertNotEqual(
            candidate["continuation_ply"],
            min(row["continuation_ply"] for row in visits),
            "fixture must prove the earliest visit is not necessarily the minimum rank",
        )

    def test_independent_replay_rejects_illegal_action_and_terminal_transition(
        self,
    ) -> None:
        state = KalahGame(
            pits=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            captured_seeds=[0, 47],
            current_player=0,
        )
        with self.assertRaisesRegex(ValueError, "illegal_trajectory_action"):
            _apply_actions(state, [0], label="illegal")
        terminal = KalahGame(
            pits=[0] * 12,
            captured_seeds=[24, 24],
            current_player=0,
        )
        with self.assertRaisesRegex(ValueError, "terminal_transition"):
            _apply_actions(terminal, [0], label="terminal")

    def test_canonical_duplicate_keeps_occurrence_links(self) -> None:
        rows = [
            {
                "canonical_identity": "same",
                "production_float32_identity": "f1",
                "rank": "1",
                "opening_index": 0,
                "game_id": "g0",
                "continuation_ply": 2,
            },
            {
                "canonical_identity": "same",
                "production_float32_identity": "f1",
                "rank": "2",
                "opening_index": 1,
                "game_id": "g1",
                "continuation_ply": 3,
            },
        ]
        deduped = seed624.deduplicate_candidates(rows)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(deduped[0]["occurrences"]), 2)

    def test_float32_identity_collision_deduplicates_distinct_canonical_states(
        self,
    ) -> None:
        rows = [
            {
                "canonical_identity": "c1",
                "production_float32_identity": "same-float",
                "rank": "1",
                "opening_index": 0,
                "game_id": "g0",
                "continuation_ply": 2,
            },
            {
                "canonical_identity": "c2",
                "production_float32_identity": "same-float",
                "rank": "2",
                "opening_index": 1,
                "game_id": "g1",
                "continuation_ply": 3,
            },
        ]
        self.assertEqual(len(seed624.deduplicate_candidates(rows)), 1)

    def test_membership_overlap_requires_both_authenticated_identity_contracts(
        self,
    ) -> None:
        state = seed624.standard_game().to_state()
        cid, fid = seed624.identities(state)
        member = {
            "canonical_state": state,
            "canonical_identity": cid,
            "production_float32_identity": fid,
        }
        evidence = {}
        for name in ("B512", "S615", "protected_holdout"):
            evidence[name] = {
                "authenticated": True,
                "rows": [member]
                if name == "B512"
                else [
                    {
                        "canonical_state": seed624.generate_opening(
                            seed624.MASTER_SEED, 0
                        )["state"],
                        "canonical_identity": seed624.generate_opening(
                            seed624.MASTER_SEED, 0
                        )["canonical_identity"],
                        "production_float32_identity": seed624.generate_opening(
                            seed624.MASTER_SEED, 0
                        )["production_float32_identity"],
                    }
                ],
            }
            evidence[name]["sha256"] = seed624.sha256(
                seed624.canonical_bytes(evidence[name]["rows"])
            )
        candidate = {
            "state": state,
            "canonical_identity": cid,
            "production_float32_identity": fid,
            "opening_identity": "opening",
            "game_id": "game",
        }
        screening_records = []
        result = seed624.screen_known_exposure(
            [candidate],
            evidence,
            expected_snapshot_hashes={
                name: source["sha256"] for name, source in evidence.items()
            },
            on_screened=screening_records.append,
        )[0]
        self.assertEqual(screening_records, [result])
        self.assertEqual(result["classification"], "confirmed_known_exposure")
        self.assertEqual(result["inherited_PE4_exposure"], "unresolved")
        evidence["protected_holdout"]["rows"] = [member]
        evidence["protected_holdout"]["sha256"] = seed624.sha256(
            seed624.canonical_bytes([member])
        )
        heldout_match = seed624.screen_known_exposure(
            [candidate],
            evidence,
            expected_snapshot_hashes={
                name: source["sha256"] for name, source in evidence.items()
            },
        )[0]
        self.assertTrue(heldout_match["exposure"]["protected_holdout"])

    def test_missing_or_corrupt_membership_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "invalid_or_unauthenticated_membership"
        ):
            seed624.screen_known_exposure([], {})
        bad = {"authenticated": True, "rows": [], "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "missing_membership_rows"):
            seed624.screen_known_exposure(
                [],
                {name: bad for name in ("B512", "S615", "protected_holdout")},
                expected_snapshot_hashes={
                    name: "0" * 64 for name in ("B512", "S615", "protected_holdout")
                },
            )

    def test_selection_is_deterministic_and_enforces_caps_on_small_fixture(
        self,
    ) -> None:
        candidates = []
        for stones in seed624.BIN_ORDER:
            for occurrence in range(2):
                candidates.append(
                    {
                        "active_pit_stones": stones,
                        "rank": hashlib.sha256(
                            f"{stones}-{occurrence}".encode()
                        ).hexdigest(),
                        "canonical_identity": f"{stones}-{occurrence}",
                        "game_id": f"g-{stones}-{occurrence}",
                        "opening_index": len(candidates),
                        "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                    }
                )
        targets = {stones: 1 for stones in seed624.BIN_ORDER}
        with (
            patch.object(seed624, "STONE_TARGETS", targets),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 1),
        ):
            first = seed624.select_cohort(candidates)
            second = seed624.select_cohort(candidates)
        self.assertEqual(
            [row["canonical_identity"] for row in first["states"]],
            [row["canonical_identity"] for row in second["states"]],
        )
        self.assertEqual(first["counts_by_stone"], targets)
        self.assertLessEqual(first["max_per_game"], 2)
        self.assertLessEqual(first["max_per_opening"], 4)
        for stones in seed624.BIN_ORDER:
            pool = sorted(
                (row for row in candidates if row["active_pit_stones"] == stones),
                key=lambda row: (row["rank"], row["canonical_identity"]),
            )
            chosen = next(
                row for row in first["states"] if row["active_pit_stones"] == stones
            )
            self.assertEqual(
                chosen["canonical_identity"], pool[0]["canonical_identity"]
            )

    def test_opening_cap_infeasibility_does_not_relax_constraints(self) -> None:
        candidates = [
            {
                "active_pit_stones": stones,
                "rank": f"{stones}-{occurrence}",
                "canonical_identity": f"{stones}-{occurrence}",
                "game_id": f"g-{stones}-{occurrence}",
                "opening_index": 0,
                "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
            }
            for stones in seed624.BIN_ORDER
            for occurrence in range(2)
        ]
        with (
            patch.object(
                seed624, "STONE_TARGETS", {stones: 1 for stones in seed624.BIN_ORDER}
            ),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 1),
        ):
            with self.assertRaises(seed624.ShortfallError) as error:
                seed624.select_cohort(candidates)
        self.assertEqual(error.exception.report["reason"], "aggregate_cluster_capacity")

    def test_game_cap_infeasibility_does_not_relax_constraints(self) -> None:
        candidates = [
            {
                "active_pit_stones": stones,
                "rank": f"{stones}-{game_index}",
                "canonical_identity": f"{stones}-{game_index}",
                "game_id": f"g-{game_index}",
                "opening_index": game_index,
                "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
            }
            for stones in seed624.BIN_ORDER
            for game_index in range(2)
        ]
        with (
            patch.object(
                seed624, "STONE_TARGETS", {stones: 1 for stones in seed624.BIN_ORDER}
            ),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 1),
        ):
            with self.assertRaises(seed624.ShortfallError) as error:
                seed624.select_cohort(candidates)
        self.assertEqual(error.exception.report["reason"], "aggregate_cluster_capacity")

    def test_canonical_identity_breaks_candidate_rank_ties(self) -> None:
        candidates = []
        for index, (stones, identity) in enumerate(
            (stones, identity)
            for stones in seed624.BIN_ORDER
            for identity in ("z", "a")
        ):
            candidates.append(
                {
                    "active_pit_stones": stones,
                    "rank": "same-rank",
                    "canonical_identity": f"{stones}-{identity}",
                    "game_id": f"g-{stones}-{identity}",
                    "opening_index": index,
                    "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                }
            )
        targets = {stones: 1 for stones in seed624.BIN_ORDER}
        with (
            patch.object(seed624, "STONE_TARGETS", targets),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 1),
        ):
            result = seed624.select_cohort(candidates)
        self.assertEqual(
            {
                row["active_pit_stones"]: row["canonical_identity"]
                for row in result["states"]
            },
            {stones: f"{stones}-a" for stones in seed624.BIN_ORDER},
        )

    def test_hard_supply_shortfall_is_structured(self) -> None:
        with self.assertRaises(seed624.ShortfallError) as error:
            seed624.select_cohort([])
        self.assertEqual(error.exception.report["reason"], "hard_supply")

    def test_realistic_scale_feasible_and_infeasible_optimizer_pools_are_bounded(
        self,
    ) -> None:
        targets = seed624.STONE_TARGETS

        def pool(opening_count: int) -> list[dict]:
            rows = []
            for stones, target in targets.items():
                for ordinal in range(4 * target):
                    digest = hashlib.sha256(
                        f"synthetic:{stones}:{ordinal}".encode()
                    ).hexdigest()
                    opening = ordinal % opening_count
                    rows.append(
                        {
                            "active_pit_stones": stones,
                            "rank": digest,
                            "canonical_identity": f"state-{stones}-{ordinal}",
                            "production_float32_identity": f"float-{stones}-{ordinal}",
                            "game_id": f"game-{stones}-{ordinal}",
                            "opening_index": opening,
                            "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                            "state": {
                                "player_pits": [stones - 1, 1, 0, 0, 0, 0],
                                "opponent_pits": [0] * 6,
                                "player_store": 0,
                                "opponent_store": 48 - stones,
                                "current_player": 0,
                            },
                            "legal_actions": [0, 1],
                            "trajectory": [ordinal % 6 for _ in range(64)],
                            "within_game_bin_visits": [
                                {"continuation_ply": 32, "rank": digest},
                                {"continuation_ply": 33, "rank": digest},
                            ],
                        }
                    )
            return rows

        feasible_rows = pool(256)
        started = time.perf_counter()
        feasible = seed624.select_cohort(feasible_rows)
        feasible_elapsed = time.perf_counter() - started
        self.assertEqual(len(feasible["states"]), 512)
        self.assertEqual(
            feasible["eligible_supply"],
            {stone: 4 * target for stone, target in targets.items()},
        )
        self.assertLessEqual(
            feasible["optimizer"]["nodes"], seed624.MAX_SELECTION_NODES
        )
        self.assertLessEqual(
            feasible["optimizer"]["peak_working_bytes"],
            seed624.MAX_SELECTION_WORKING_BYTES,
        )
        self.assertLessEqual(feasible_elapsed, seed624.MAX_SELECTION_SECONDS)

        infeasible_rows = pool(100)
        started = time.perf_counter()
        with self.assertRaises(seed624.ShortfallError) as shortfall:
            seed624.select_cohort(infeasible_rows)
        infeasible_elapsed = time.perf_counter() - started
        self.assertEqual(
            shortfall.exception.report["reason"], "aggregate_cluster_capacity"
        )
        self.assertEqual(len(infeasible_rows), sum(targets.values()) * 4)
        self.assertLess(infeasible_elapsed, 5.0)

    def test_adversarial_feasible_pool_backtracks_from_early_rank(self) -> None:
        targets = {stone: 1 for stone in seed624.BIN_ORDER}
        # The rank-first path fills opening 0 and saturates game A. The final
        # bin has no feasible candidate until the preceding bin takes rank z.
        assignments = {
            17: [("a", "game-a", 0), ("z", "game-b", 1)],
            18: [("a", "game-a", 0), ("z", "game-c", 1)],
            19: [("a", "game-d", 0), ("z", "game-e", 1)],
            20: [("a", "game-f", 0), ("z", "game-g", 1)],
            21: [("a", "game-h", 0), ("z", "game-a", 1)],
        }
        candidates = [
            {
                "active_pit_stones": stone,
                "rank": rank,
                "canonical_identity": f"{stone}-{game_id}",
                "game_id": game_id,
                "opening_index": opening,
                "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
            }
            for stone, rows in assignments.items()
            for rank, game_id, opening in rows
        ]
        with (
            patch.object(seed624, "STONE_TARGETS", targets),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 1),
        ):
            selected = seed624.select_cohort(candidates)
        picked = {row["active_pit_stones"]: row["rank"] for row in selected["states"]}
        self.assertEqual(picked[20], "z")
        self.assertGreater(selected["optimizer"]["nodes"], len(targets) + 1)

    def test_adversarial_infeasible_pool_passes_aggregate_gates_and_exhausts(
        self,
    ) -> None:
        bins = (17, 18, 19, 20, 21, 22)
        targets = {stone: 2 for stone in bins}
        candidates = []
        shared_games = ("g0", "g1", "g2", "g3")
        game_opening = {"g0": 0, "g1": 0, "g2": 1, "g3": 1}
        for stone in bins[:5]:
            for rank_index, game_id in enumerate(shared_games):
                candidates.append(
                    {
                        "active_pit_stones": stone,
                        "rank": f"{rank_index:02d}",
                        "canonical_identity": f"{stone}-{game_id}",
                        "game_id": game_id,
                        "opening_index": game_opening[game_id],
                        "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                    }
                )
        for rank_index, game_id in enumerate(("g4", "g5", "g6", "g7")):
            candidates.append(
                {
                    "active_pit_stones": 22,
                    "rank": f"{rank_index:02d}",
                    "canonical_identity": f"22-{game_id}",
                    "game_id": game_id,
                    "opening_index": 2 if game_id in {"g4", "g5"} else 3,
                    "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                }
            )

        # All cheap necessary capacity bounds pass, but bins 17-21 need ten
        # states from g0-g3, whose combined game capacity is only eight.
        game_capacity = sum(
            min(2, count)
            for count in Counter(row["game_id"] for row in candidates).values()
        )
        opening_capacity = sum(
            min(4, count)
            for count in Counter(row["opening_index"] for row in candidates).values()
        )
        self.assertEqual(game_capacity, sum(targets.values()))
        self.assertEqual(opening_capacity, sum(targets.values()))
        with (
            patch.object(seed624, "STONE_TARGETS", targets),
            patch.object(seed624, "BIN_ORDER", bins),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 4),
        ):
            with self.assertRaises(seed624.ShortfallError) as error:
                seed624.select_cohort(candidates)
        self.assertEqual(error.exception.report["reason"], "constraints_infeasible")
        self.assertGreater(error.exception.report["nodes"], 128)

    def test_adversarial_selector_node_and_time_limits_are_deterministic(self) -> None:
        bins = (17, 18, 19, 20, 21, 22)
        targets = {stone: 2 for stone in bins}
        infeasible_candidates = []
        for stone in bins[:5]:
            for rank_index, game_id in enumerate(("g0", "g1", "g2", "g3")):
                infeasible_candidates.append(
                    {
                        "active_pit_stones": stone,
                        "rank": f"{rank_index:02d}",
                        "canonical_identity": f"{stone}-{game_id}",
                        "game_id": game_id,
                        "opening_index": rank_index // 2,
                        "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                    }
                )
        for rank_index, game_id in enumerate(("g4", "g5", "g6", "g7")):
            infeasible_candidates.append(
                {
                    "active_pit_stones": 22,
                    "rank": f"{rank_index:02d}",
                    "canonical_identity": f"22-{game_id}",
                    "game_id": game_id,
                    "opening_index": 2 if game_id in {"g4", "g5"} else 3,
                    "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
                }
            )
        feasible_assignments = {
            17: [("a", "game-a", 0), ("z", "game-b", 1)],
            18: [("a", "game-a", 0), ("z", "game-c", 1)],
            19: [("a", "game-d", 0), ("z", "game-e", 1)],
            20: [("a", "game-f", 0), ("z", "game-g", 1)],
            21: [("a", "game-h", 0), ("z", "game-a", 1)],
        }
        feasible_candidates = [
            {
                "active_pit_stones": stone,
                "rank": rank,
                "canonical_identity": f"{stone}-{game_id}",
                "game_id": game_id,
                "opening_index": opening,
                "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
            }
            for stone, rows in feasible_assignments.items()
            for rank, game_id, opening in rows
        ]
        cases = (
            (
                feasible_candidates,
                seed624.BIN_ORDER,
                {b: 1 for b in seed624.BIN_ORDER},
                1,
            ),
            (infeasible_candidates, bins, targets, 4),
        )
        for pool, pool_bins, pool_targets, min_openings in cases:
            with (
                patch.object(seed624, "STONE_TARGETS", pool_targets),
                patch.object(seed624, "BIN_ORDER", pool_bins),
                patch.object(seed624, "MIN_OPENING_CLUSTERS", min_openings),
                patch.object(seed624, "MAX_SELECTION_NODES", 8),
            ):
                node_reports = []
                for _ in range(2):
                    with self.assertRaises(seed624.ShortfallError) as error:
                        seed624.select_cohort(pool)
                    node_reports.append(error.exception.report)
            self.assertEqual(node_reports[0]["reason"], "optimizer_resource_limit")
            self.assertEqual(node_reports[0]["limit"], "node_limit")
            self.assertEqual(node_reports[0]["nodes"], 9)
            self.assertEqual(node_reports[0]["nodes"], node_reports[1]["nodes"])
            with patch.object(independent, "MAX_SELECTION_NODES", 8):
                with self.assertRaisesRegex(
                    ValueError, "verifier_optimizer_node_limit"
                ):
                    independent._reconstruct_selection(pool, pool_targets, min_openings)
            with patch(
                "ml.alphazero_lite.verify_seed624_prospective_benchmark.time.monotonic",
                side_effect=[0.0, 31.0, 31.0],
            ):
                with self.assertRaisesRegex(
                    ValueError, "verifier_optimizer_resource_limit"
                ):
                    independent._reconstruct_selection(pool, pool_targets, min_openings)

            for _ in range(2):
                with (
                    patch.object(seed624, "STONE_TARGETS", pool_targets),
                    patch.object(seed624, "BIN_ORDER", pool_bins),
                    patch.object(seed624, "MIN_OPENING_CLUSTERS", min_openings),
                    patch.object(seed624, "MAX_SELECTION_NODES", 250_000),
                    patch.object(seed624, "MAX_SELECTION_SECONDS", 30),
                    patch(
                        "ml.alphazero_lite.seed624_prospective_benchmark.time.monotonic",
                        side_effect=[0.0, 31.0, 31.0],
                    ),
                ):
                    with self.assertRaises(seed624.ShortfallError) as timeout_error:
                        seed624.select_cohort(pool)
                self.assertEqual(
                    timeout_error.exception.report["reason"],
                    "optimizer_resource_limit",
                )
                self.assertEqual(
                    timeout_error.exception.report["limit"], "time_seconds"
                )
                self.assertEqual(timeout_error.exception.report["nodes"], 1)

    def test_invalid_authorization_cannot_start_generation(self) -> None:
        with self.assertRaisesRegex(
            PermissionError, "lifecycle_frozen_started_attempt_required"
        ):
            seed624.generate_population({}, Path("unused"))

    def test_selection_shortfall_keeps_synthetic_generation_and_screening_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as temporary:
            output = Path(temporary) / "population"
            output.mkdir()
            opening_attempt = {"opening_index": 0, "status": "accepted"}
            game = {"game_id": "o0000-g00", "actions": [1, 2, 3]}
            candidate = {
                "canonical_identity": "state-a",
                "active_pit_stones": 17,
                "game_id": "o0000-g00",
                "opening_index": 0,
                "classification": "known_exposure_nonmatch_inherited_PE4_unresolved",
            }
            exposed = {
                "canonical_identity": "state-b",
                "active_pit_stones": 18,
                "game_id": "o0000-g01",
                "opening_index": 0,
                "classification": "confirmed_known_exposure",
            }
            seed624._write_json_create_only(
                output / "opening-attempts.json", [opening_attempt]
            )
            seed624._write_jsonl_create_only(output / "games.jsonl", [game])
            seed624._write_jsonl_create_only(
                output / "candidate-occurrences.jsonl", [candidate, exposed]
            )
            seed624._write_jsonl_create_only(
                output / "candidate-inventory.jsonl", [candidate, exposed]
            )
            shortfall = seed624.ShortfallError(
                {"schema": "seed624-selection-shortfall-v1", "reason": "hard_supply"}
            )
            with patch.object(seed624, "select_cohort", side_effect=shortfall):
                with self.assertRaises(seed624.ShortfallError):
                    seed624._persist_screened_stages_and_select(
                        output, [candidate, exposed]
                    )
            self.assertEqual(
                json.loads((output / "opening-attempts.json").read_text()),
                [opening_attempt],
            )
            self.assertEqual(json.loads((output / "games.jsonl").read_text()), game)
            self.assertEqual(
                [
                    json.loads(line)
                    for line in (output / "candidate-occurrences.jsonl")
                    .read_text()
                    .splitlines()
                ],
                [candidate, exposed],
            )
            self.assertEqual(
                json.loads((output / "failure.json").read_text())["classification"],
                "insufficient_candidate_supply",
            )
            self.assertFalse(
                json.loads((output / "selection-diagnostics.json").read_text())[
                    "qualified_population"
                ]
            )

    def test_shortfall_classification_distinguishes_supply_math_resource_and_verifier(
        self,
    ) -> None:
        cases = {
            "hard_supply": "insufficient_candidate_supply",
            "opening_reserve_exhausted": "insufficient_candidate_supply",
            "constraints_infeasible": "mathematical_infeasibility",
            "aggregate_cluster_capacity": "mathematical_infeasibility",
            "optimizer_resource_limit": "resource_exhaustion",
            "candidate_count_resource_limit": "resource_exhaustion",
            "identity_replay_mismatch": "verification_failure",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(
                    seed624._shortfall_classification({"reason": reason}), expected
                )

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_failure_receipts_are_durable_create_only_and_classified(self) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            authorization = self._receipt_authorization(Path(directory))
            receipt = Path(authorization["failure_receipt_path"])
            shortfall = seed624.ShortfallError(
                {"reason": "constraints_infeasible", "nodes": 17}
            )
            with patch.object(seed624, "generate_population", side_effect=shortfall):
                with self.assertRaises(seed624.ShortfallError):
                    seed624.execute_with_failure_receipts(
                        authorization, Path(directory) / "partial"
                    )
            report = json.loads(receipt.read_text())
            self.assertEqual(
                report["schema"], "seed624-authenticated-execution-failure-v1"
            )
            self.assertEqual(report["failure_category"], "mathematical_infeasibility")
            self.assertEqual(report["authorized_run_identity"], "synthetic-run-1")
            self.assertEqual(
                report["execution_freeze_sha256"],
                authorization["execution_freeze_sha256"],
            )
            self.assertEqual(report["diagnostic"]["report"]["nodes"], 17)
            with self.assertRaises(FileExistsError):
                seed624.write_failure_receipt(
                    authorization=authorization,
                    category="unexpected_execution_exception",
                    diagnostic={},
                )

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_mismatched_run_identity_and_receipt_path_are_unauthenticated(self) -> None:
        for field, wrong_value in (
            ("authorized_run_identity", "caller-run-forgery"),
            ("failure_receipt_path", "caller-selected-receipt.json"),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory(dir=".tmp") as directory,
            ):
                authorization = self._receipt_authorization(Path(directory))
                authorization[field] = wrong_value
                requested = Path(directory) / "preflight.json"
                with self.assertRaisesRegex(
                    PermissionError, "failure_receipt_.*mismatch"
                ):
                    seed624.execute_with_failure_receipts(
                        authorization,
                        Path(directory) / "unused",
                        receipt_path=requested,
                    )
                record = json.loads(requested.read_text())
                self.assertEqual(
                    record["schema"], "seed624-unauthenticated-preflight-failure-v1"
                )
                self.assertIsNone(record["authorized_run_identity"])
                self.assertIsNone(record["execution_freeze_sha256"])

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_unauthenticated_preflight_is_not_attributed_to_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            authorization = self._receipt_authorization(Path(directory))
            authorization["execution_freeze_sha256"] = "f" * 64
            path = Path(directory) / "preflight.json"
            with self.assertRaisesRegex(
                PermissionError, "execution_freeze_hash_mismatch"
            ):
                seed624.execute_with_failure_receipts(
                    authorization, Path(directory) / "unused", receipt_path=path
                )
            record = json.loads(path.read_text())
            self.assertEqual(
                record["schema"], "seed624-unauthenticated-preflight-failure-v1"
            )
            self.assertEqual(record["failure_category"], "preflight_failure")
            self.assertIsNone(record["authorized_run_identity"])

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_resource_exhaustion_and_supply_have_distinct_receipt_categories(
        self,
    ) -> None:
        resource = seed624.ShortfallError({"reason": "optimizer_resource_limit"})
        supply = seed624.ShortfallError({"reason": "hard_supply"})
        opening = seed624.ShortfallError({"reason": "opening_reserve_exhausted"})
        self.assertEqual(
            seed624._failure_category(resource), "selector_resource_exhaustion"
        )
        self.assertEqual(
            seed624._failure_category(supply), "insufficient_stone_bin_supply"
        )
        self.assertEqual(
            seed624._failure_category(opening), "opening_reserve_exhaustion"
        )

    @unittest.skip("resource terminations are retained by the Seed 635 lifecycle")
    def test_synthetic_wall_limit_termination_writes_supervisor_receipt(self) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            authorization = self._receipt_authorization(Path(directory))
            try:
                subprocess.run(
                    ["python", "-c", "import time; time.sleep(2)"],
                    check=True,
                    timeout=0.05,
                )
            except subprocess.TimeoutExpired:
                seed624.write_failure_receipt(
                    authorization=authorization,
                    category="external_resource_limit_termination",
                    diagnostic={"limit": "wall_seconds", "exit_status": "terminated"},
                )
            receipt_data = json.loads(
                Path(authorization["failure_receipt_path"]).read_text()
            )
            self.assertEqual(receipt_data["diagnostic"]["exit_status"], "terminated")
            self.assertEqual(
                receipt_data["failure_category"],
                "external_resource_limit_termination",
            )

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_receipt_write_failure_preserves_original_execution_exception(self) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            authorization = self._receipt_authorization(Path(directory))
            original = seed624.ShortfallError({"reason": "hard_supply"})
            with (
                patch.object(seed624, "generate_population", side_effect=original),
                patch.object(
                    seed624,
                    "write_failure_receipt",
                    side_effect=OSError("synthetic receipt disk failure"),
                ),
            ):
                with self.assertRaises(seed624.ShortfallError) as raised:
                    seed624.execute_with_failure_receipts(
                        authorization, Path(directory) / "partial"
                    )
            self.assertIs(raised.exception, original)
            self.assertTrue(
                any(
                    "synthetic receipt disk failure" in note
                    for note in original.__notes__
                )
            )

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_failure_category_enum_is_shared_with_proposed_freeze(self) -> None:
        categories = set(
            seed624.proposed_freeze_schema()["required"]["failure_receipt_contract"][
                "categories"
            ]
        )
        self.assertEqual(categories, seed624.FAILURE_CATEGORIES)

    @unittest.skip("failure records are now owned by the Seed 635 lifecycle runner")
    def test_durable_receipt_fsyncs_file_and_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "durability.json"
            actual_fsync = seed624.os.fsync
            with patch.object(seed624.os, "fsync", wraps=actual_fsync) as fsync:
                seed624._create_durable_json(path, {"synthetic": True})
            self.assertEqual(fsync.call_count, 2)

    def _empty_replay_bundle(self) -> dict:
        opening = seed624.generate_opening(seed624.MASTER_SEED, 0)
        games = []
        occurrences = []
        for trajectory_index in range(seed624.GAMES_PER_OPENING):
            game, game_candidates = seed624.generate_game(
                opening, seed624.MASTER_SEED, trajectory_index
            )
            game["opening_prefix_actions"] = opening["prefix_actions"]
            games.append(game)
            occurrences.extend(game_candidates)
        state = seed624.standard_game().to_state()
        cid, fid = seed624.identities(state)
        member = {
            "canonical_state": state,
            "canonical_identity": cid,
            "production_float32_identity": fid,
        }
        evidence = {
            name: {"authenticated": True, "rows": [member]}
            for name in ("B512", "S615", "protected_holdout")
        }
        for source in evidence.values():
            source["sha256"] = seed624.sha256(seed624.canonical_bytes(source["rows"]))
        evidence["source_provenance"] = {
            "authenticated": True,
            "crosswalk_status": "complete",
            "B512": [],
            "S615": [],
            "protected_holdout": [],
        }
        provenance_hash = seed624.sha256(
            seed624.canonical_bytes(
                {
                    key: value
                    for key, value in evidence["source_provenance"].items()
                    if key != "authenticated"
                }
            )
        )
        evidence["source_provenance"]["sha256"] = provenance_hash
        for source in evidence.values():
            if isinstance(source, dict) and "rows" in source:
                source["authenticated"] = True
        membership_hashes = {
            name: source["sha256"]
            for name, source in evidence.items()
            if name in ("B512", "S615", "protected_holdout")
        }
        screened = seed624.screen_known_exposure(
            seed624.deduplicate_candidates(occurrences),
            evidence,
            expected_snapshot_hashes=membership_hashes,
            expected_provenance_hash=provenance_hash,
        )
        freeze = {
            "schema": "seed624-execution-freeze-v1",
            "status": "authorized",
            "production_resource_limits": {
                "wall_seconds": 3600,
                "cpu_seconds": 3600,
                "process_rss_bytes": 4 * 1024**3,
                "execution_concurrency": 1,
                "output_and_temporary_storage_bytes": 2 * 1024**3,
            },
            "project_owner_identity": "synthetic-fixture-owner",
            "execution_scope": seed624.EXECUTION_SCOPE,
            "git_commit": seed624.current_git_commit(),
            "source_file_sha256": seed624.current_source_hashes(),
            "runtime": seed624.runtime_identity(),
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_target": 1,
                "opening_reserve": 512,
                "invalid_prefix_policy": seed624.OPENING_INVALID_PREFIX_POLICY,
                "prefix_actions": 8,
                "games_per_opening": 8,
                "continuation_action_limit": 200,
                "seed_contract": seed624.SEED_CONTRACT,
            },
            "rules_identity": {
                "rule_id": "historical-kalah-rules",
                "rule_version": "repository-kalah-rules-v1",
                "source_sha256": seed624.current_source_hashes()[
                    "ml/alphazero_lite/kalah_rules.py"
                ],
            },
            "encoder_identity": {
                "encoder": "kalah_v3",
                "source_sha256": seed624.current_source_hashes()[
                    "ml/alphazero_lite/input_encodings.py"
                ],
            },
            "output_schema": seed624.SCHEMA,
            "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
            "stone_bin_weights": seed624.STONE_BIN_WEIGHTS,
            "membership_snapshot_sha256": membership_hashes,
            "membership_snapshot_file_sha256": {
                name.removesuffix(".json"): digest
                for name, digest in seed628.PINNED_FILES.items()
                if name.endswith(".json")
                and name != "source-identity-lineage-manifest.json"
            },
            "membership_source_manifest_sha256": seed628.PINNED_FILES[
                "source-identity-lineage-manifest.json"
            ],
            "source_provenance_sha256": provenance_hash,
            "selection": {
                "targets": {str(stones): 0 for stones in seed624.BIN_ORDER},
                "max_per_game": 2,
                "max_per_opening": 4,
                "minimum_openings": 0,
                "objective": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity",
                "optimizer": "seed624-dfs-bounded-v2",
                "candidate_limit": seed624.MAX_SELECTION_CANDIDATES,
                "input_memory_limit_bytes": seed624.MAX_SELECTION_INPUT_BYTES,
                "workers": 1,
                "runtime_limit_seconds": seed624.MAX_SELECTION_SECONDS,
                "node_limit": seed624.MAX_SELECTION_NODES,
                "time_check_interval_nodes": seed624.SELECTION_TIME_CHECK_INTERVAL_NODES,
                "working_memory_limit_bytes": seed624.MAX_SELECTION_WORKING_BYTES,
            },
        }
        freeze_raw, authorization_receipt, _receipt_raw = self._seal_freeze(freeze)
        empty_targets = {stones: 0 for stones in seed624.BIN_ORDER}
        return {
            "schema": seed624.SCHEMA,
            "authorization": {
                "project_owner_identity": "useruby",
                "execution_freeze_sha256": seed624.sha256(freeze_raw),
                "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
                "registration_pr_authorization": authorization_receipt,
                "lifecycle": self._fixture_lifecycle,
            },
            "source_identity": {
                "git_commit": freeze["git_commit"],
                "runtime": freeze["runtime"],
                "files": {
                    name: {
                        "sha256": digest,
                        "source": (seed624.ROOT / name).read_text(),
                    }
                    for name, digest in freeze["source_file_sha256"].items()
                },
            },
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_seeds_consumed": [0],
                "invalid_prefix_policy": seed624.OPENING_INVALID_PREFIX_POLICY,
                "opening_attempts": [
                    {
                        "opening_index": 0,
                        "seed": opening["seed"],
                        "status": "accepted",
                        "canonical_identity": opening["canonical_identity"],
                        "production_float32_identity": opening[
                            "production_float32_identity"
                        ],
                    }
                ],
                "opening_rejections": [],
            },
            "openings": [opening],
            "opening_attempts_sha256": seed624.sha256(
                seed624.canonical_bytes(
                    [
                        {
                            "opening_index": 0,
                            "seed": opening["seed"],
                            "status": "accepted",
                            "canonical_identity": opening["canonical_identity"],
                            "production_float32_identity": opening[
                                "production_float32_identity"
                            ],
                        }
                    ]
                )
            ),
            "opening_inventory_sha256": seed624.sha256(
                seed624.canonical_bytes([opening])
            ),
            "games": games,
            "game_inventory_sha256": seed624.sha256(seed624.canonical_bytes(games)),
            "candidates": screened,
            "candidate_inventory_sha256": seed624.sha256(
                seed624.canonical_bytes(screened)
            ),
            "aggregate_counts": {
                "openings": 1,
                "opening_seed_indexes_consumed": 1,
                "invalid_opening_prefixes": 0,
                "duplicate_openings": 0,
                "games": 8,
                "candidates": len(screened),
                "confirmed_known_exposure": 0,
                "known_exposure_nonmatch_inherited_PE4_unresolved": len(screened),
                "invalid_or_insufficiently_authenticated_evidence": 0,
            },
            "evidence": evidence,
            "selection": {
                "states": [],
                "targets": empty_targets,
                "selection_contract": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity; bounded exhaustive feasibility; seed624-dfs-bounded-v2",
                "optimizer": {
                    "name": "seed624-dfs-bounded-v2",
                    "version": 2,
                    "workers": 1,
                    "runtime_limit_seconds": seed624.MAX_SELECTION_SECONDS,
                    "candidate_limit": seed624.MAX_SELECTION_CANDIDATES,
                    "input_memory_limit_bytes": seed624.MAX_SELECTION_INPUT_BYTES,
                    "input_bytes": seed624.deep_size_bytes(screened),
                    "node_limit": seed624.MAX_SELECTION_NODES,
                    "time_check_interval_nodes": seed624.SELECTION_TIME_CHECK_INTERVAL_NODES,
                    "working_memory_limit_bytes": seed624.MAX_SELECTION_WORKING_BYTES,
                    "peak_working_bytes": 0,
                    "elapsed_seconds": 0.0,
                    "nodes": 6,
                },
                "selected_inventory_sha256": seed624.sha256(
                    seed624.canonical_bytes([])
                ),
                "counts_by_stone": {str(stones): 0 for stones in seed624.BIN_ORDER},
                "max_per_opening": 0,
                "max_per_game": 0,
                "minimum_openings": 0,
                "distinct_openings": 0,
                "eligible_supply": {
                    str(stones): sum(
                        row["active_pit_stones"] == stones
                        and row["classification"]
                        == "known_exposure_nonmatch_inherited_PE4_unresolved"
                        for row in screened
                    )
                    for stones in seed624.BIN_ORDER
                },
                "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
                "stone_bin_weights": seed624.STONE_BIN_WEIGHTS,
            },
        }

    def test_independent_verifier_rejects_unpinned_synthetic_membership_evidence(
        self,
    ) -> None:
        bundle = self._empty_replay_bundle()
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            physical_copy = Path(directory) / "copied-evidence" / "population.json"
            physical_copy.parent.mkdir()
            shutil.copyfile(path, physical_copy)
            with self.assertRaisesRegex(
                ValueError, "production_evidence_does_not_match_pinned_seed627_chain"
            ):
                verify_bundle(physical_copy)

    def test_independent_verifier_rejects_skipping_first_valid_opening(self) -> None:
        bundle = self._empty_replay_bundle()
        bundle["openings"] = [seed624.generate_opening(seed624.MASTER_SEED, 1)]
        bundle["generation"]["opening_seeds_consumed"] = [0, 1]
        bundle["opening_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(bundle["openings"])
        )
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "skipped-opening.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "opening_reserve_consumed_after_target_reached"
            ):
                verify_bundle(path)

    def test_independent_verifier_reconstructs_and_rejects_multivisit_fixture(
        self,
    ) -> None:
        openings = [seed624.generate_opening(seed624.MASTER_SEED, i) for i in (0, 1)]
        games = []
        occurrences = []
        for opening in openings:
            for trajectory_index in range(seed624.GAMES_PER_OPENING):
                game, rows = seed624.generate_game(
                    opening, seed624.MASTER_SEED, trajectory_index
                )
                game["opening_prefix_actions"] = opening["prefix_actions"]
                games.append(game)
                occurrences.extend(rows)
        unique = seed624.deduplicate_candidates(occurrences)
        state = seed624.standard_game().to_state()
        cid, fid = seed624.identities(state)
        member = {
            "canonical_state": state,
            "canonical_identity": cid,
            "production_float32_identity": fid,
        }
        evidence = {
            name: {"authenticated": True, "rows": [member]}
            for name in ("B512", "S615", "protected_holdout")
        }
        for value in evidence.values():
            value["sha256"] = seed624.sha256(seed624.canonical_bytes(value["rows"]))
        provenance = {
            "authenticated": True,
            "crosswalk_status": "complete",
            "B512": [],
            "S615": [],
            "protected_holdout": [],
        }
        provenance["sha256"] = seed624.sha256(
            seed624.canonical_bytes(
                {
                    key: value
                    for key, value in provenance.items()
                    if key != "authenticated"
                }
            )
        )
        evidence["source_provenance"] = provenance
        for source in evidence.values():
            if isinstance(source, dict) and "rows" in source:
                source["authenticated"] = True
        target_hashes = {
            name: evidence[name]["sha256"]
            for name in ("B512", "S615", "protected_holdout")
        }
        prov_hash = provenance["sha256"]
        freeze = {
            "schema": "seed624-execution-freeze-v1",
            "status": "authorized",
            "production_resource_limits": {
                "wall_seconds": 3600,
                "cpu_seconds": 3600,
                "process_rss_bytes": 4 * 1024**3,
                "execution_concurrency": 1,
                "output_and_temporary_storage_bytes": 2 * 1024**3,
            },
            "project_owner_identity": "synthetic-fixture-owner",
            "execution_scope": seed624.EXECUTION_SCOPE,
            "git_commit": seed624.current_git_commit(),
            "source_file_sha256": seed624.current_source_hashes(),
            "runtime": seed624.runtime_identity(),
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_target": 2,
                "opening_reserve": 512,
                "invalid_prefix_policy": seed624.OPENING_INVALID_PREFIX_POLICY,
                "prefix_actions": 8,
                "games_per_opening": 8,
                "continuation_action_limit": 200,
                "seed_contract": seed624.SEED_CONTRACT,
            },
            "rules_identity": {
                "rule_id": "historical-kalah-rules",
                "rule_version": "repository-kalah-rules-v1",
                "source_sha256": seed624.current_source_hashes()[
                    "ml/alphazero_lite/kalah_rules.py"
                ],
            },
            "encoder_identity": {
                "encoder": "kalah_v3",
                "source_sha256": seed624.current_source_hashes()[
                    "ml/alphazero_lite/input_encodings.py"
                ],
            },
            "output_schema": seed624.SCHEMA,
            "sampling_contract": "deterministic standardized selected-state population; fixed 1/5 weights per stone bin; no design-unbiased inference or inclusion-probability claim",
            "stone_bin_weights": seed624.STONE_BIN_WEIGHTS,
            "selection": {
                "targets": {str(stones): 1 for stones in seed624.BIN_ORDER},
                "max_per_game": 2,
                "max_per_opening": 4,
                "minimum_openings": 2,
                "objective": "lexicographically smallest feasible per-bin rank vector; bin order 17,18,19,20,21; rank ties by canonical identity",
                "optimizer": "seed624-dfs-bounded-v2",
                "candidate_limit": seed624.MAX_SELECTION_CANDIDATES,
                "input_memory_limit_bytes": seed624.MAX_SELECTION_INPUT_BYTES,
                "workers": 1,
                "runtime_limit_seconds": seed624.MAX_SELECTION_SECONDS,
                "node_limit": seed624.MAX_SELECTION_NODES,
                "time_check_interval_nodes": seed624.SELECTION_TIME_CHECK_INTERVAL_NODES,
                "working_memory_limit_bytes": seed624.MAX_SELECTION_WORKING_BYTES,
            },
            "membership_snapshot_sha256": target_hashes,
            "membership_snapshot_file_sha256": {
                name.removesuffix(".json"): digest
                for name, digest in seed628.PINNED_FILES.items()
                if name.endswith(".json")
                and name != "source-identity-lineage-manifest.json"
            },
            "membership_source_manifest_sha256": seed628.PINNED_FILES[
                "source-identity-lineage-manifest.json"
            ],
            "source_provenance_sha256": prov_hash,
        }
        freeze_raw, authorization_receipt, _receipt_raw = self._seal_freeze(freeze)
        screened = seed624.screen_known_exposure(
            unique,
            evidence,
            expected_snapshot_hashes=target_hashes,
            expected_provenance_hash=prov_hash,
        )
        targets = {stones: 1 for stones in seed624.BIN_ORDER}
        with (
            patch.object(seed624, "STONE_TARGETS", targets),
            patch.object(seed624, "MIN_OPENING_CLUSTERS", 2),
        ):
            selection = seed624.select_cohort(screened)
        bundle = {
            "schema": seed624.SCHEMA,
            "authorization": {
                "project_owner_identity": "useruby",
                "execution_freeze_sha256": seed624.sha256(freeze_raw),
                "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
                "registration_pr_authorization": authorization_receipt,
                "lifecycle": self._fixture_lifecycle,
            },
            "source_identity": {
                "git_commit": freeze["git_commit"],
                "runtime": freeze["runtime"],
                "files": {
                    name: {
                        "sha256": digest,
                        "source": (seed624.ROOT / name).read_text(),
                    }
                    for name, digest in freeze["source_file_sha256"].items()
                },
            },
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_seeds_consumed": [0, 1],
                "invalid_prefix_policy": seed624.OPENING_INVALID_PREFIX_POLICY,
                "opening_attempts": [
                    {
                        "opening_index": opening["opening_index"],
                        "seed": opening["seed"],
                        "status": "accepted",
                        "canonical_identity": opening["canonical_identity"],
                        "production_float32_identity": opening[
                            "production_float32_identity"
                        ],
                    }
                    for opening in openings
                ],
                "opening_rejections": [],
            },
            "openings": openings,
            "opening_attempts_sha256": seed624.sha256(
                seed624.canonical_bytes(
                    [
                        {
                            "opening_index": opening["opening_index"],
                            "seed": opening["seed"],
                            "status": "accepted",
                            "canonical_identity": opening["canonical_identity"],
                            "production_float32_identity": opening[
                                "production_float32_identity"
                            ],
                        }
                        for opening in openings
                    ]
                )
            ),
            "opening_inventory_sha256": seed624.sha256(
                seed624.canonical_bytes(openings)
            ),
            "games": games,
            "game_inventory_sha256": seed624.sha256(seed624.canonical_bytes(games)),
            "candidates": screened,
            "candidate_inventory_sha256": seed624.sha256(
                seed624.canonical_bytes(screened)
            ),
            "aggregate_counts": {
                "openings": 2,
                "opening_seed_indexes_consumed": 2,
                "invalid_opening_prefixes": 0,
                "duplicate_openings": 0,
                "games": len(games),
                "candidates": len(screened),
                "confirmed_known_exposure": sum(
                    row["classification"] == "confirmed_known_exposure"
                    for row in screened
                ),
                "known_exposure_nonmatch_inherited_PE4_unresolved": sum(
                    row["classification"]
                    == "known_exposure_nonmatch_inherited_PE4_unresolved"
                    for row in screened
                ),
                "invalid_or_insufficiently_authenticated_evidence": sum(
                    row["classification"]
                    == "invalid_or_insufficiently_authenticated_evidence"
                    for row in screened
                ),
            },
            "evidence": evidence,
            "selection": selection,
        }
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with patch.object(
                seed628_verifier,
                "reconstruct_evidence",
                return_value=bundle["evidence"],
            ):
                report = verify_bundle(path)
        self.assertEqual(report["selected_count"], 5)
        self.assertEqual(report["opening_clusters"], 2)
        multi_visit = next(
            row
            for row in bundle["candidates"]
            if len(row["within_game_bin_visits"]) > 1
        )
        tampered = json.loads(json.dumps(bundle))
        tampered_row = next(
            row
            for row in tampered["candidates"]
            if row["canonical_identity"] == multi_visit["canonical_identity"]
        )
        tampered_row["within_game_bin_visits"][0]["rank"] = "0" * 64
        tampered["candidate_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(tampered["candidates"])
        )
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "tampered-population.json"
            path.write_text(json.dumps(tampered, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "candidate_occurrence_or_deduplication_mismatch"
            ):
                verify_bundle(path)

    def test_independent_verifier_rejects_modified_trajectory_even_if_hash_rebound(
        self,
    ) -> None:
        bundle = self._empty_replay_bundle()
        bundle["games"][0]["actions"][0] = (bundle["games"][0]["actions"][0] + 1) % 6
        bundle["games"][0]["full_actions"] = (
            bundle["games"][0]["opening_prefix_actions"] + bundle["games"][0]["actions"]
        )
        bundle["game_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(bundle["games"])
        )
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "continuation_random_stream_mismatch"
            ):
                verify_bundle(path)

    def test_independent_verifier_rejects_rehashed_early_nonterminal_truncation(
        self,
    ) -> None:
        bundle = self._empty_replay_bundle()
        game_row = bundle["games"][0]
        game_row["actions"] = game_row["actions"][:10]
        game_row["full_actions"] = (
            game_row["opening_prefix_actions"] + game_row["actions"]
        )
        game = KalahGame.from_state(
            next(
                row["state"]
                for row in bundle["openings"]
                if row["opening_index"] == game_row["opening_index"]
            )
        )
        for action in game_row["actions"]:
            self.assertTrue(game.move(game.pit_index(action)))
        game_row["final_state"] = game.to_state()
        game_row["terminal"] = game.over()
        game_row["action_limit_reached"] = False
        bundle["opening_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(bundle["openings"])
        )
        bundle["game_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(bundle["games"])
        )
        bundle["candidate_inventory_sha256"] = seed624.sha256(
            seed624.canonical_bytes(bundle["candidates"])
        )
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "truncated-population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "nonterminal_continuation_truncated_early"
            ):
                verify_bundle(path)

    def test_changed_rank_is_rejected_by_independent_rank_contract(self) -> None:
        row = {
            "opening_identity": "opening",
            "game_id": "game",
            "active_pit_stones": 17,
            "canonical_identity": "state",
        }
        row["rank"] = seed624._candidate_rank(
            seed624.MASTER_SEED, "opening", "game", 17, "state"
        )
        validate_candidate_rank(row)
        row["rank"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate_rank_mismatch"):
            validate_candidate_rank(row)

    def test_false_training_disjointness_claim_fails_closed(self) -> None:
        bundle = self._empty_replay_bundle()
        bundle["selection"]["training_disjoint"] = True
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "false_training_disjointness_claim"
            ):
                verify_bundle(path)

    def test_invalid_sampling_weights_are_rejected(self) -> None:
        bundle = self._empty_replay_bundle()
        bundle["selection"]["sampling_contract"] = "inverse-probability weights: 2.0"
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "invalid_sampling_weight_or_estimand_claim"
            ):
                verify_bundle(path)

    def test_malformed_fixed_bin_weights_are_rejected(self) -> None:
        bundle = self._empty_replay_bundle()
        bundle["selection"]["stone_bin_weights"]["17"] = 0.5
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "bad-bin-weights.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(
                ValueError, "invalid_sampling_weight_or_estimand_claim"
            ):
                verify_bundle(path)

    def test_modified_source_hash_is_rejected(self) -> None:
        bundle = self._empty_replay_bundle()
        source_name = next(iter(bundle["source_identity"]["files"]))
        bundle["source_identity"]["files"][source_name]["source"] += "# altered\n"
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "source_file_hash_mismatch"):
                verify_bundle(path)


if __name__ == "__main__":
    unittest.main()
