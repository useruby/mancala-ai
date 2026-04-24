import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.opening_cache import canonical_key, canonical_payload, load_opening_cache
from ml.alphazero_lite.generate_opening_cache import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_GENERATION_GATE,
    DEFAULT_SEARCH_PROFILE,
    build_search_profile,
    build_opening_cache_artifact,
    default_teacher,
    write_opening_cache_artifact,
)


class OpeningCacheTest(unittest.TestCase):
    def opening_state(self):
        return {
            "current_player": 0,
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
        }

    def lookup_entry(self):
        return {
            "state": self.opening_state(),
            "side_to_move": 0,
            "selected_move": 2,
            "policy": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "value": 0.4,
            "budget": {
                "baseline_simulations": 1200,
                "probe_simulations": 1200,
                "chosen_simulations": 1200,
                "final_simulations": 1200,
                "root_latency_ms": 8.0,
            },
            "provenance": {
                "teacher_kind": "classic_mcts",
                "search_profile_hash": "abc",
                "hit_count_in_sources": 3,
                "example_origins": ["self_play:test"],
            },
        }

    def lookup_entry_for_state(self, state):
        entry = self.lookup_entry()
        entry["state"] = state
        return entry

    def canonical_payload_json(self, state):
        canonical_payload = {
            "current_player": int(state["current_player"]),
            "player_pits": [int(value) for value in state["player_pits"]],
            "opponent_pits": [int(value) for value in state["opponent_pits"]],
            "player_store": int(state["player_store"]),
            "opponent_store": int(state["opponent_store"]),
        }
        return json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def test_canonical_key_is_stable_hash_for_canonical_state_payload(self):
        state = {
            "current_player": 0,
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "debug_only": "ignored",
        }

        equivalent_state = {
            "current_player": 0,
            "opponent_store": 0,
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "player_pits": [4, 4, 4, 4, 4, 4],
        }

        self.assertEqual(canonical_key(state), canonical_key(equivalent_state))
        self.assertRegex(canonical_key(state), r"^[0-9a-f]{64}$")

    def test_load_opening_cache_rejects_unknown_schema(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            load_opening_cache(
                {
                    "schema": "azlite_opening_cache_v999",
                    "entries": {},
                }
            )

    def test_load_opening_cache_rejects_non_mapping_entries(self):
        with self.assertRaisesRegex(ValueError, "entries"):
            load_opening_cache(
                {
                    "schema": "azlite_opening_cache_v1",
                    "entries": [],
                }
            )

    def test_load_opening_cache_preserves_opening_gate_metadata(self):
        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "entries": {},
            }
        )

        self.assertEqual({"max_ply": 10, "min_stones_in_pits": 36}, cache.opening_gate)

    def test_load_opening_cache_coerces_numeric_opening_gate_metadata(self):
        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": "10", "min_stones_in_pits": "36"},
                "entries": {},
            }
        )

        self.assertEqual({"max_ply": 10, "min_stones_in_pits": 36}, cache.opening_gate)

    def test_load_opening_cache_rejects_invalid_opening_gate_metadata(self):
        for opening_gate in (
            {"max_ply": None, "min_stones_in_pits": 36},
            {"max_ply": "ten", "min_stones_in_pits": 36},
        ):
            with self.subTest(opening_gate=opening_gate):
                with self.assertRaisesRegex(ValueError, "opening_gate"):
                    load_opening_cache(
                        {
                            "schema": "azlite_opening_cache_v1",
                            "opening_gate": opening_gate,
                            "entries": {},
                        }
                    )

    def test_load_opening_cache_preserves_search_profile_metadata(self):
        search_profile = {
            "version": "v1",
            "kind": "opening_cache_teacher",
            "player_mode": "classic_mcts",
            "classic_mcts_simulations": 1200,
            "hash": "cached-profile-hash",
        }

        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "search_profile": search_profile,
                "entries": {},
            }
        )

        self.assertEqual(search_profile, cache.search_profile)

    def test_lookup_returns_matching_entry_for_hashed_key_artifact(self):
        state = self.opening_state()
        entry = self.lookup_entry()
        hashed_key = canonical_key(state)

        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "entries": {hashed_key: entry},
            }
        )

        self.assertIn(hashed_key, cache._entries_by_key)
        self.assertEqual(entry, cache.lookup(state, ply=0))

    def test_lookup_returns_matching_entry_for_legacy_raw_canonical_json_key_artifact(self):
        state = self.opening_state()
        entry = self.lookup_entry()
        hashed_key = canonical_key(state)
        legacy_key = self.canonical_payload_json(state)

        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "entries": {legacy_key: entry},
            }
        )

        self.assertIn(hashed_key, cache._entries_by_key)
        self.assertNotIn(legacy_key, cache._entries_by_key)
        self.assertEqual(entry, cache.lookup(state, ply=0))

    def test_load_opening_cache_rejects_mismatched_key_and_entry_state(self):
        state = self.opening_state()
        entry = self.lookup_entry()
        mismatched_state = {
            **state,
            "player_store": 1,
            "opponent_pits": [4, 4, 4, 4, 4, 3],
        }

        with self.assertRaisesRegex(ValueError, "Opening cache entry key does not match entry state"):
            load_opening_cache(
                {
                    "schema": "azlite_opening_cache_v1",
                    "entries": {canonical_key(mismatched_state): entry},
                }
            )

    def test_load_opening_cache_rejects_duplicate_normalized_keys(self):
        state = self.opening_state()
        hashed_entry = self.lookup_entry()
        legacy_entry = self.lookup_entry()
        hashed_key = canonical_key(state)
        legacy_key = self.canonical_payload_json(state)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_opening_cache(
                {
                    "schema": "azlite_opening_cache_v1",
                    "entries": {
                        legacy_key: legacy_entry,
                        hashed_key: hashed_entry,
                    },
                }
            )

    def test_lookup_returns_none_when_state_exceeds_max_ply(self):
        state = self.opening_state()

        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 2, "min_stones_in_pits": 36},
                "entries": {canonical_key(state): self.lookup_entry()},
            }
        )

        self.assertIsNone(cache.lookup(state, ply=3))

    def test_lookup_returns_none_when_state_falls_below_min_stones_in_pits(self):
        state = {
            "current_player": 0,
            "player_pits": [1, 1, 1, 1, 1, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 18,
            "opponent_store": 18,
        }

        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "entries": {canonical_key(state): self.lookup_entry_for_state(state)},
            }
        )

        self.assertIsNone(cache.lookup(state, ply=0))

    def test_lookup_returns_none_when_state_is_missing(self):
        cache = load_opening_cache(
            {
                "schema": "azlite_opening_cache_v1",
                "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                "entries": {},
            }
        )

        self.assertIsNone(cache.lookup(self.opening_state(), ply=0))


class OpeningCacheGeneratorTest(unittest.TestCase):
    maxDiff = None

    def opening_state(self):
        return {
            "current_player": 0,
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
        }

    def second_state(self):
        return {
            "current_player": 1,
            "player_pits": [4, 4, 0, 5, 5, 5],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 1,
        }

    def low_stones_state(self):
        return {
            "current_player": 0,
            "player_pits": [1, 1, 1, 1, 1, 1],
            "opponent_pits": [1, 1, 1, 1, 1, 1],
            "player_store": 18,
            "opponent_store": 18,
        }

    def teacher_result(self, move, *, value, root_latency_ms):
        return {
            "selected_move": move,
            "policy": [1.0 if index == move else 0.0 for index in range(6)],
            "value": value,
            "budget": {
                "baseline_simulations": 1200,
                "probe_simulations": 1200,
                "chosen_simulations": 1200,
                "final_simulations": 1200,
                "root_latency_ms": root_latency_ms,
            },
        }

    def test_default_teacher_constructs_classic_mcts_with_fixed_seed(self):
        state = self.opening_state()
        search = mock.Mock()
        search.root_summary.return_value = {
            "selected_move": 3,
            "budget": self.teacher_result(3, value=0.0, root_latency_ms=5.0)["budget"],
        }
        search.search_root.return_value = mock.Mock(visits=0, wins=0)

        with mock.patch("ml.alphazero_lite.generate_opening_cache.ClassicMCTS", return_value=search) as classic_mcts:
            default_teacher(state)

        classic_mcts.assert_called_once()
        args, kwargs = classic_mcts.call_args
        self.assertIsInstance(args[0], KalahGame)
        self.assertEqual(1200, kwargs["simulations"])
        self.assertEqual(0, kwargs["seed"])

    def test_default_teacher_uses_classic_mcts_root_value(self):
        state = self.opening_state()
        search = mock.Mock()
        search.root_summary.return_value = {
            "selected_move": 3,
            "budget": self.teacher_result(3, value=0.0, root_latency_ms=5.0)["budget"],
        }
        search.search_root.return_value = mock.Mock(visits=10, wins=8)

        with mock.patch("ml.alphazero_lite.generate_opening_cache.ClassicMCTS", return_value=search):
            result = default_teacher(state)

        self.assertAlmostEqual(0.6, result["value"])

    def test_build_search_profile_hash_changes_when_seed_changes(self):
        seed_zero_profile = build_search_profile(seed=0)
        seed_one_profile = build_search_profile(seed=1)

        self.assertEqual(0, seed_zero_profile["deterministic_seed"])
        self.assertEqual(1, seed_one_profile["deterministic_seed"])
        self.assertNotEqual(seed_zero_profile["hash"], seed_one_profile["hash"])

    def test_build_opening_cache_artifact_deduplicates_qualifying_states_and_sorts_entries(self):
        opening_state = self.opening_state()
        second_state = self.second_state()
        teacher_calls = []

        def teacher(state):
            key = canonical_key(state)
            teacher_calls.append(key)
            if key == canonical_key(opening_state):
                return self.teacher_result(2, value=0.4, root_latency_ms=8.0)
            return self.teacher_result(4, value=-0.2, root_latency_ms=11.0)

        artifact = build_opening_cache_artifact(
            [
                {"state": second_state, "ply": 2, "origin": "game-2"},
                {"state": opening_state, "ply": 0, "origin": "game-1"},
                {"state": opening_state, "ply": 1, "origin": "game-3"},
                {"state": self.low_stones_state(), "ply": 3, "origin": "game-4"},
                {"state": opening_state, "ply": 11, "origin": "game-5"},
            ],
            teacher=teacher,
        )

        self.assertEqual("azlite_opening_cache_v1", artifact["schema"])
        self.assertEqual(DEFAULT_GENERATION_GATE, artifact["opening_gate"])
        self.assertEqual(DEFAULT_SEARCH_PROFILE, artifact["search_profile"])
        self.assertEqual(sorted(teacher_calls), teacher_calls)
        self.assertEqual(2, len(artifact["entries"]))

        entry_keys = list(artifact["entries"].keys())
        self.assertEqual(sorted(entry_keys), entry_keys)

        opening_entry = artifact["entries"][canonical_key(opening_state)]
        self.assertEqual(2, opening_entry["provenance"]["hit_count_in_sources"])
        self.assertEqual(["game-1", "game-3"], opening_entry["provenance"]["example_origins"])

        second_entry = artifact["entries"][canonical_key(second_state)]
        self.assertEqual(1, second_entry["provenance"]["hit_count_in_sources"])
        self.assertEqual(["game-2"], second_entry["provenance"]["example_origins"])

        self.assertEqual(
            {
                "source_positions": 5,
                "qualifying_positions": 3,
                "retained_positions": 2,
                "teacher_evaluations": 2,
            },
            artifact["generation_metrics"],
        )

    def test_build_opening_cache_artifact_stores_canonical_state_payload(self):
        source_state = {
            "current_player": "0",
            "player_pits": ["4", "4", "4", "4", "4", "4"],
            "opponent_pits": ["4", "4", "4", "4", "4", "4"],
            "player_store": "0",
            "opponent_store": "0",
            "debug_only": "ignored",
        }

        artifact = build_opening_cache_artifact(
            [{"state": source_state, "ply": 0, "origin": "seed-game"}],
            teacher=lambda state: self.teacher_result(1, value=0.25, root_latency_ms=7.0),
        )

        self.assertEqual(
            canonical_payload(source_state),
            artifact["entries"][canonical_key(source_state)]["state"],
        )

    def test_build_opening_cache_artifact_derives_provenance_teacher_kind_from_search_profile(self):
        opening_state = self.opening_state()
        profile = {
            "hash": "profile-hash",
            "kind": "custom_opening_teacher",
            "player_mode": "scripted_teacher",
        }

        artifact = build_opening_cache_artifact(
            [{"state": opening_state, "ply": 0, "origin": "seed-game"}],
            teacher=lambda state: self.teacher_result(1, value=0.25, root_latency_ms=7.0),
            search_profile=profile,
        )

        self.assertEqual(
            "scripted_teacher",
            artifact["entries"][canonical_key(opening_state)]["provenance"]["teacher_kind"],
        )
        self.assertEqual(profile["hash"], artifact["entries"][canonical_key(opening_state)]["provenance"]["search_profile_hash"])

    def test_build_opening_cache_artifact_rejects_mismatched_search_profile_for_default_teacher(self):
        with self.assertRaisesRegex(ValueError, "search_profile"):
            build_opening_cache_artifact(
                [{"state": self.opening_state(), "ply": 0, "origin": "seed-game"}],
                search_profile=build_search_profile(seed=1),
            )

    def test_build_opening_cache_artifact_caps_example_origins_with_sorted_sample(self):
        opening_state = self.opening_state()

        artifact = build_opening_cache_artifact(
            [
                {"state": opening_state, "ply": 0, "origin": "game-d"},
                {"state": opening_state, "ply": 0, "origin": "game-b"},
                {"state": opening_state, "ply": 0, "origin": "game-e"},
                {"state": opening_state, "ply": 0, "origin": "game-a"},
                {"state": opening_state, "ply": 0, "origin": "game-c"},
            ],
            teacher=lambda state: self.teacher_result(1, value=0.25, root_latency_ms=7.0),
        )

        entry = artifact["entries"][canonical_key(opening_state)]

        self.assertEqual(5, entry["provenance"]["hit_count_in_sources"])
        self.assertEqual(["game-a", "game-b", "game-c"], entry["provenance"]["example_origins"])

    def test_write_opening_cache_artifact_persists_deterministic_json_to_default_path(self):
        opening_state = self.opening_state()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / DEFAULT_ARTIFACT_PATH
            artifact = write_opening_cache_artifact(
                [{"state": opening_state, "ply": 0, "origin": "seed-game"}],
                teacher=lambda state: self.teacher_result(1, value=0.25, root_latency_ms=7.0),
                artifact_path=artifact_path,
            )

            self.assertTrue(artifact_path.exists())
            self.assertEqual(Path("storage/ai/alphazero_lite/opening_cache_v1.json"), DEFAULT_ARTIFACT_PATH)
            written_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact, written_payload)
            self.assertEqual([canonical_key(opening_state)], list(written_payload["entries"].keys()))

    def test_write_opening_cache_artifact_threads_opening_gate_and_search_profile(self):
        opening_state = self.opening_state()
        opening_gate = {"max_ply": 0, "min_stones_in_pits": 36}
        search_profile = {
            "hash": "custom-hash",
            "kind": "custom_teacher",
            "player_mode": "scripted_teacher",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "opening_cache.json"
            artifact = write_opening_cache_artifact(
                [{"state": opening_state, "ply": 1, "origin": "seed-game"}],
                teacher=lambda state: self.teacher_result(1, value=0.25, root_latency_ms=7.0),
                opening_gate=opening_gate,
                search_profile=search_profile,
                artifact_path=artifact_path,
            )

            self.assertEqual(opening_gate, artifact["opening_gate"])
            self.assertEqual(search_profile, artifact["search_profile"])
            self.assertEqual({}, artifact["entries"])


if __name__ == "__main__":
    unittest.main()
