"""Small deterministic fixtures for Seed 624 generation and selection contracts."""

from __future__ import annotations

import hashlib
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ml.alphazero_lite import seed624_prospective_benchmark as seed624
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.verify_seed624_prospective_benchmark import _apply_actions
from ml.alphazero_lite.verify_seed624_prospective_benchmark import (
    validate_candidate_rank,
    verify_owner_approval,
    verify_bundle,
)


class Seed624SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._owner_private_key = Ed25519PrivateKey.generate()
        public_key = cls._owner_private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        cls._owner_key_id = hashlib.sha256(public_key).hexdigest()[:16]
        cls._trusted_key = {
            "project_owner_identity": "synthetic-fixture-owner",
            "public_key_hex": public_key.hex(),
        }
        seed624.TRUSTED_PROJECT_OWNER_KEYS[cls._owner_key_id] = cls._trusted_key
        import ml.alphazero_lite.verify_seed624_prospective_benchmark as independent

        independent.TRUSTED_PROJECT_OWNER_KEYS[cls._owner_key_id] = cls._trusted_key

    @classmethod
    def tearDownClass(cls) -> None:
        seed624.TRUSTED_PROJECT_OWNER_KEYS.clear()
        import ml.alphazero_lite.verify_seed624_prospective_benchmark as independent

        independent.TRUSTED_PROJECT_OWNER_KEYS.clear()

    def _seal_freeze(self, freeze: dict) -> tuple[bytes, dict, bytes]:
        freeze_raw = seed624.canonical_bytes(freeze)
        record = {
            "schema": "seed624-owner-approval-v1",
            "project_owner_identity": "synthetic-fixture-owner",
            "key_id": self._owner_key_id,
            "freeze_sha256": seed624.sha256(freeze_raw),
            "execution_scope": seed624.EXECUTION_SCOPE,
        }
        signature = self._owner_private_key.sign(seed624.canonical_bytes(record))
        record["signature_ed25519"] = base64.b64encode(signature).decode("ascii")
        return freeze_raw, record, seed624.canonical_bytes(record)

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
        _game, rows = seed624.generate_game(opening, seed624.MASTER_SEED, 1)
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
        result = seed624.screen_known_exposure(
            [candidate],
            evidence,
            expected_snapshot_hashes={
                name: source["sha256"] for name, source in evidence.items()
            },
        )[0]
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

    def test_invalid_authorization_cannot_start_generation(self) -> None:
        with self.assertRaisesRegex(PermissionError, "execution_freeze_file_required"):
            seed624.generate_population({}, Path("unused"))

    def test_entrypoint_rejects_caller_supplied_untrusted_owner_key(self) -> None:
        attacker_key = Ed25519PrivateKey.generate()
        attacker_public_key = attacker_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            freeze_path = Path(directory) / "freeze.json"
            approval_path = Path(directory) / "owner-approval.json"
            freeze = {
                "schema": "seed624-execution-freeze-v1",
                "status": "authorized",
                "project_owner_identity": "synthetic-fixture-owner",
                "execution_scope": seed624.EXECUTION_SCOPE,
            }
            freeze_raw = seed624.canonical_bytes(freeze)
            freeze_path.write_bytes(freeze_raw)
            record = {
                "schema": "seed624-owner-approval-v1",
                "project_owner_identity": "synthetic-fixture-owner",
                "key_id": hashlib.sha256(attacker_public_key).hexdigest()[:16],
                "freeze_sha256": seed624.sha256(freeze_raw),
                "execution_scope": seed624.EXECUTION_SCOPE,
            }
            record["signature_ed25519"] = base64.b64encode(
                attacker_key.sign(seed624.canonical_bytes(record))
            ).decode("ascii")
            approval_path.write_bytes(seed624.canonical_bytes(record))
            with self.assertRaisesRegex(
                PermissionError, "owner_approval_signing_key_not_trusted"
            ):
                seed624.generate_population(
                    {
                        "execution_freeze_path": str(freeze_path),
                        "execution_freeze_sha256": seed624.sha256(freeze_raw),
                        "owner_approval_record_path": str(approval_path),
                        "trusted_public_key_hex": attacker_public_key.hex(),
                    },
                    Path(directory) / "no-population",
                )

    def test_owner_approval_requires_trusted_signature_exact_digest_and_scope(
        self,
    ) -> None:
        freeze_digest = "ab" * 32
        record = {
            "schema": "seed624-owner-approval-v1",
            "project_owner_identity": "synthetic-fixture-owner",
            "key_id": self._owner_key_id,
            "freeze_sha256": freeze_digest,
            "execution_scope": seed624.EXECUTION_SCOPE,
        }
        signature = self._owner_private_key.sign(seed624.canonical_bytes(record))
        record["signature_ed25519"] = base64.b64encode(signature).decode("ascii")
        self.assertEqual(
            seed624.verify_owner_approval_record(
                record, freeze_digest, seed624.EXECUTION_SCOPE
            ),
            "synthetic-fixture-owner",
        )
        with self.assertRaisesRegex(PermissionError, "owner_approval_record_binding"):
            seed624.verify_owner_approval_record(
                record, "cd" * 32, seed624.EXECUTION_SCOPE
            )
        with self.assertRaisesRegex(ValueError, "owner_approval_binding_mismatch"):
            verify_owner_approval(record, freeze_digest, "different-scope")
        corrupted_signature = dict(record)
        corrupted_signature["signature_ed25519"] = base64.b64encode(
            b"invalid-signature"
        ).decode("ascii")
        with self.assertRaisesRegex(
            PermissionError, "owner_approval_signature_invalid"
        ):
            seed624.verify_owner_approval_record(
                corrupted_signature, freeze_digest, seed624.EXECUTION_SCOPE
            )
        untrusted = dict(record, key_id="caller-supplied-key")
        with self.assertRaisesRegex(
            PermissionError, "owner_approval_signing_key_not_trusted"
        ):
            seed624.verify_owner_approval_record(
                untrusted, freeze_digest, seed624.EXECUTION_SCOPE
            )

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
            "authenticated": False,
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
                source["authenticated"] = False
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
            "project_owner_identity": "synthetic-fixture-owner",
            "execution_scope": seed624.EXECUTION_SCOPE,
            "git_commit": seed624.current_git_commit(),
            "source_file_sha256": seed624.current_source_hashes(),
            "runtime": seed624.runtime_identity(),
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_target": 1,
                "opening_reserve": 512,
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
                "working_memory_limit_bytes": seed624.MAX_SELECTION_WORKING_BYTES,
            },
        }
        freeze_raw, approval, approval_raw = self._seal_freeze(freeze)
        empty_targets = {stones: 0 for stones in seed624.BIN_ORDER}
        return {
            "schema": seed624.SCHEMA,
            "authorization": {
                "project_owner_identity": "synthetic-fixture-owner",
                "execution_freeze_sha256": seed624.sha256(freeze_raw),
                "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
                "owner_approval_record": approval,
                "owner_approval_record_raw_utf8": approval_raw.decode("utf-8"),
                "owner_approval_record_sha256": seed624.sha256(approval_raw),
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
            },
            "openings": [opening],
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

    def test_independent_verifier_reconstructs_copied_synthetic_bundle(self) -> None:
        bundle = self._empty_replay_bundle()
        with tempfile.TemporaryDirectory(dir=".tmp") as directory:
            path = Path(directory) / "population.json"
            path.write_text(json.dumps(bundle, sort_keys=True))
            report = verify_bundle(path)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["games"], 8)
        self.assertFalse(report["training_disjoint"])

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
            "authenticated": False,
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
                source["authenticated"] = False
        target_hashes = {
            name: evidence[name]["sha256"]
            for name in ("B512", "S615", "protected_holdout")
        }
        prov_hash = provenance["sha256"]
        freeze = {
            "schema": "seed624-execution-freeze-v1",
            "status": "authorized",
            "project_owner_identity": "synthetic-fixture-owner",
            "execution_scope": seed624.EXECUTION_SCOPE,
            "git_commit": seed624.current_git_commit(),
            "source_file_sha256": seed624.current_source_hashes(),
            "runtime": seed624.runtime_identity(),
            "generation": {
                "master_seed": seed624.MASTER_SEED,
                "opening_target": 2,
                "opening_reserve": 512,
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
                "working_memory_limit_bytes": seed624.MAX_SELECTION_WORKING_BYTES,
            },
            "membership_snapshot_sha256": target_hashes,
            "source_provenance_sha256": prov_hash,
        }
        freeze_raw, approval, approval_raw = self._seal_freeze(freeze)
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
                "project_owner_identity": "synthetic-fixture-owner",
                "execution_freeze_sha256": seed624.sha256(freeze_raw),
                "execution_freeze_raw_utf8": freeze_raw.decode("utf-8"),
                "owner_approval_record": approval,
                "owner_approval_record_raw_utf8": approval_raw.decode("utf-8"),
                "owner_approval_record_sha256": seed624.sha256(approval_raw),
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
            },
            "openings": openings,
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
