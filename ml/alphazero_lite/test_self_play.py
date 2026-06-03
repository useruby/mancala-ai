import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import hashlib
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite.eval_cache import EvalCache
from ml.alphazero_lite import self_play
from ml.alphazero_lite import opponent_pool
from ml.alphazero_lite.input_encodings import (
    BASE_FEATURE_ORDER,
    KALAH_V3_EXTRA_FEATURE_ORDER,
)
from ml.alphazero_lite.kalah_rules import KalahGame


VALUE_TARGET_PARITY_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "test"
    / "fixtures"
    / "ai"
    / "value_target_parity_cases.json"
)


def kalah_v3_feature_index(name: str) -> int:
    return len(BASE_FEATURE_ORDER) + KALAH_V3_EXTRA_FEATURE_ORDER.index(name)


class SelfPlayScriptTest(unittest.TestCase):
    def executable_python(self) -> str:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return sys.executable

    def test_executable_python_skips_non_executable_candidates(self):
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / ".venv/bin/python",
            repo_root.parents[1] / ".venv/bin/python",
        ]
        executable_fallback = candidates[1]

        def fake_is_file(self):
            return self in candidates

        def fake_access(path, mode):
            return path == executable_fallback and mode == os.X_OK

        with (
            mock.patch.object(Path, "is_file", fake_is_file),
            mock.patch("os.access", side_effect=fake_access),
        ):
            self.assertEqual(str(executable_fallback), self.executable_python())

    def test_executable_python_falls_back_to_sys_executable(self):
        expected = __import__("sys").executable

        with mock.patch.object(Path, "is_file", return_value=False):
            self.assertEqual(expected, self.executable_python())

    def _checkpoint_evaluator_test_game(self, *, current_player=0):
        return KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": current_player,
            }
        )

    def _write_checkpoint(
        self, directory: Path, name: str, *, input_features: int
    ) -> Path:
        checkpoint_path = directory / name
        np.savez(
            checkpoint_path,
            w_hidden_1=np.zeros((input_features, 2), dtype=np.float32),
            b_hidden_1=np.zeros(2, dtype=np.float32),
            w_hidden_2=np.zeros((2, 2), dtype=np.float32),
            b_hidden_2=np.zeros(2, dtype=np.float32),
            w_policy=np.zeros((2, 6), dtype=np.float32),
            b_policy=np.zeros(6, dtype=np.float32),
            w_value=np.zeros((2, 1), dtype=np.float32),
            b_value=np.zeros(1, dtype=np.float32),
        )
        return checkpoint_path

    def test_eval_cache_stores_and_retrieves_values_by_key(self):
        cache = EvalCache(max_entries=2)

        cache.put("state-a", ("priors-a", 0.25))

        self.assertEqual(("priors-a", 0.25), cache.get("state-a"))

    def test_eval_cache_evicts_least_recently_used_entry_when_max_entries_exceeded(
        self,
    ):
        cache = EvalCache(max_entries=2)

        cache.put("state-a", "value-a")
        cache.put("state-b", "value-b")
        self.assertEqual("value-a", cache.get("state-a"))
        cache.put("state-c", "value-c")

        self.assertIsNone(cache.get("state-b"))
        self.assertEqual("value-a", cache.get("state-a"))
        self.assertEqual("value-c", cache.get("state-c"))

    def test_eval_cache_tracks_hit_and_miss_counters(self):
        cache = EvalCache(max_entries=2)

        cache.put("state-a", "value-a")

        self.assertEqual("value-a", cache.get("state-a"))
        self.assertIsNone(cache.get("missing-state"))
        self.assertEqual(1, cache.hits)
        self.assertEqual(1, cache.misses)
        self.assertEqual(1, cache.size)

    def test_canonical_value_target_default_uses_outcome_value(self):
        target = self_play.canonical_value_target(
            outcome_value=1.0,
            search_value=-0.4,
            move_index=0,
            mode="default",
        )

        self.assertEqual(1.0, target)

    def test_canonical_value_target_sharpened_keeps_outcome_sign_on_disagreement(self):
        target = self_play.canonical_value_target(
            outcome_value=1.0,
            search_value=-0.4,
            move_index=0,
            mode="sharpened",
        )

        self.assertAlmostEqual(0.64, target, places=6)

    def test_canonical_value_target_phase_aware_sharpens_by_move_bucket(self):
        target = self_play.canonical_value_target(
            outcome_value=1.0,
            search_value=-0.4,
            move_index=30,
            mode="phase_aware_sharpened",
        )

        self.assertAlmostEqual(0.784, target, places=6)

    def test_canonical_value_target_hybrid_blends_outcome_and_search_magnitudes(self):
        target = self_play.canonical_value_target(
            outcome_value=1.0,
            search_value=-0.4,
            move_index=0,
            mode="hybrid",
        )

        self.assertAlmostEqual(0.55, target, places=6)

    def test_canonical_value_target_matches_shared_parity_fixture(self):
        parity_cases = json.loads(
            VALUE_TARGET_PARITY_FIXTURE.read_text(encoding="utf-8")
        )

        for parity_case in parity_cases:
            with self.subTest(case=parity_case["name"]):
                target = self_play.canonical_value_target(
                    outcome_value=parity_case["outcome_value"],
                    search_value=parity_case["search_value"],
                    move_index=parity_case["move_index"],
                    mode=parity_case["value_target_mode"],
                )

                self.assertAlmostEqual(parity_case["expected_target"], target, places=9)

    def test_canonical_value_target_returns_zero_for_draws(self):
        target = self_play.canonical_value_target(
            outcome_value=0.0,
            search_value=0.8,
            move_index=30,
            mode="hybrid",
        )

        self.assertEqual(0.0, target)

    def test_parse_args_accepts_hybrid_value_target_mode(self):
        with mock.patch(
            "sys.argv",
            ["self_play.py", "--out", "tmp.jsonl", "--value-target-mode", "hybrid"],
        ):
            args = self_play.parse_args()

        self.assertEqual("hybrid", args.value_target_mode)

    def test_parse_args_accepts_phase_aware_value_target_mode(self):
        with mock.patch(
            "sys.argv",
            [
                "self_play.py",
                "--out",
                "tmp.jsonl",
                "--value-target-mode",
                "phase_aware_sharpened",
            ],
        ):
            args = self_play.parse_args()

        self.assertEqual("phase_aware_sharpened", args.value_target_mode)

    def test_parse_args_accepts_sharpened_value_target_mode(self):
        with mock.patch(
            "sys.argv",
            ["self_play.py", "--out", "tmp.jsonl", "--value-target-mode", "sharpened"],
        ):
            args = self_play.parse_args()

        self.assertEqual("sharpened", args.value_target_mode)

    def test_parse_args_accepts_sharpened_policy_target_mode(self):
        with mock.patch(
            "sys.argv",
            ["self_play.py", "--out", "tmp.jsonl", "--policy-target-mode", "sharpened"],
        ):
            args = self_play.parse_args()

        self.assertEqual("sharpened", args.policy_target_mode)

    def test_parse_args_accepts_denoised_policy_target_noise_mode(self):
        with mock.patch(
            "sys.argv",
            [
                "self_play.py",
                "--out",
                "tmp.jsonl",
                "--policy-target-noise-mode",
                "denoised",
                "--write-root-target-telemetry",
            ],
        ):
            args = self_play.parse_args()

        self.assertEqual("denoised", args.policy_target_noise_mode)
        self.assertTrue(args.write_root_target_telemetry)

    def test_parse_args_accepts_opening_min_simulations(self):
        with mock.patch(
            "sys.argv",
            [
                "self_play.py",
                "--out",
                "tmp.jsonl",
                "--opening-min-simulations",
                "384",
                "--opening-min-simulations-plies",
                "8",
            ],
        ):
            args = self_play.parse_args()

        self.assertEqual(384, args.opening_min_simulations)
        self.assertEqual(8, args.opening_min_simulations_plies)

    def test_parse_args_accepts_opponent_pool_config(self):
        with mock.patch(
            "sys.argv",
            [
                "self_play.py",
                "--out",
                "tmp.jsonl",
                "--opponent-pool-config",
                "pool.json",
            ],
        ):
            args = self_play.parse_args()

        self.assertEqual("pool.json", args.opponent_pool_config)

    def test_parse_args_accepts_opening_cache_path(self):
        with mock.patch(
            "sys.argv",
            [
                "self_play.py",
                "--out",
                "tmp.jsonl",
                "--opening-cache",
                "opening-cache.json",
            ],
        ):
            args = self_play.parse_args()

        self.assertEqual("opening-cache.json", args.opening_cache)

    def test_parse_args_accepts_evaluator_cache_size(self):
        with mock.patch(
            "sys.argv",
            ["self_play.py", "--out", "tmp.jsonl", "--evaluator-cache-size", "128"],
        ):
            args = self_play.parse_args()

        self.assertEqual(128, args.evaluator_cache_size)

    def test_parse_args_rejects_negative_evaluator_cache_size(self):
        with mock.patch(
            "sys.argv",
            ["self_play.py", "--out", "tmp.jsonl", "--evaluator-cache-size", "-1"],
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                self_play.parse_args()

    def test_load_opponent_checkpoints_requires_checkpoints_field(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "pool.json"
            config_path.write_text(json.dumps({"unexpected": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must include a checkpoints field"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_rejects_empty_checkpoints_list(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "pool.json"
            config_path.write_text(json.dumps({"checkpoints": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_rejects_missing_file(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "missing.json"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "pool.json"
            config_path.write_text("{ not-json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_requires_object_root(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "pool.json"
            config_path.write_text(
                json.dumps(["not", "an", "object"]), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "root must be a JSON object"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_rejects_missing_checkpoint_path(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            config_path = Path(tmp) / "pool.json"
            config_path.write_text(
                json.dumps({"checkpoints": ["missing/model.npz"]}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "references missing checkpoint"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_rejects_directory_entry(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_dir = tmp_path / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            config_path = tmp_path / "pool.json"
            config_path.write_text(
                json.dumps({"checkpoints": [str(checkpoint_dir)]}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "non-file checkpoint"):
                opponent_pool.load_opponent_checkpoints(str(config_path))

    def test_load_opponent_checkpoints_normalizes_absolute_paths(self):
        with tempfile.TemporaryDirectory(prefix="azlite-opponent-pool-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "model.npz"
            checkpoint_path.write_text("", encoding="utf-8")
            weird_absolute = checkpoint_path.parent / "." / checkpoint_path.name
            config_path = tmp_path / "pool.json"
            config_path.write_text(
                json.dumps({"checkpoints": [str(weird_absolute)]}), encoding="utf-8"
            )

            loaded = opponent_pool.load_opponent_checkpoints(str(config_path))
            self.assertEqual([str(checkpoint_path.resolve())], loaded)

    def test_opponent_pool_sampling_is_repeatable_for_seeded_sequence(self):
        checkpoints = ["opp_a.npz", "opp_b.npz", "opp_c.npz"]

        first = [
            opponent_pool.sample_opponent_checkpoint(
                checkpoints,
                base_seed=11,
                game_index=game_index,
                worker_id=2,
            )
            for game_index in range(8)
        ]
        second = [
            opponent_pool.sample_opponent_checkpoint(
                checkpoints,
                base_seed=11,
                game_index=game_index,
                worker_id=2,
            )
            for game_index in range(8)
        ]

        self.assertEqual(first, second)

    def test_build_search_profile_hash_is_deterministic_for_same_inputs(self):
        search_options = self_play.build_search_options(
            fpu_mode="parent_q",
            reuse_subtree=True,
            normalize_values=True,
            root_policy_mode="deterministic",
            tactical_root_bias=0.2,
        )

        first = self_play.build_search_profile(
            kind="self_play",
            player_mode="puct",
            simulations=96,
            c_puct=1.25,
            search_options=search_options,
        )
        second = self_play.build_search_profile(
            kind="self_play",
            player_mode="puct",
            simulations=96,
            c_puct=1.25,
            search_options=search_options,
        )

        self.assertEqual(first["hash"], second["hash"])

    def test_build_search_profile_hash_changes_when_semantic_field_changes(self):
        search_options = self_play.build_search_options(
            fpu_mode="parent_q",
            reuse_subtree=True,
            normalize_values=True,
            root_policy_mode="deterministic",
            tactical_root_bias=0.2,
        )

        baseline = self_play.build_search_profile(
            kind="self_play",
            player_mode="puct",
            simulations=96,
            c_puct=1.25,
            search_options=search_options,
        )
        changed = self_play.build_search_profile(
            kind="self_play",
            player_mode="puct",
            simulations=128,
            c_puct=1.25,
            search_options=search_options,
        )

        self.assertNotEqual(baseline["hash"], changed["hash"])

    def test_build_search_profile_uses_classic_fields_for_classic_mode(self):
        profile = self_play.build_search_profile(
            kind="self_play",
            player_mode="classic_mcts",
            simulations=96,
            c_puct=1.25,
            search_options=self_play.build_search_options(),
        )

        self.assertEqual("classic_mcts", profile["player_mode"])
        self.assertEqual(96, profile["classic_mcts_simulations"])
        self.assertNotIn("simulations", profile)
        self.assertNotIn("c_puct", profile)
        self.assertNotIn("search_options", profile)

    def test_build_search_options_normalizes_partial_value_trust_schedule_with_defaults(
        self,
    ):
        options = self_play.build_search_options(
            value_trust_schedule={
                "opening": 0.8,
            }
        )

        self.assertEqual(
            {
                "enabled": False,
                "opening": 0.8,
                "midgame": 1.0,
                "late": 1.0,
            },
            options["value_trust_schedule"],
        )

    def test_run_self_play_worker_profile_includes_opponent_pool_fingerprint(self):
        evaluator_paths = []

        class FakeEvaluator(self_play.Evaluator):
            def __init__(self, checkpoint_path, *, input_encoding):
                evaluator_paths.append((str(checkpoint_path), input_encoding))

            def evaluate(self, game):
                del game
                priors = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return priors, 0.0

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opponents-fingerprint-"
        ) as tmp:
            tmp_path = Path(tmp)
            shard_path = tmp_path / "worker.jsonl"
            opponent_pool_config_path = tmp_path / "pool.json"
            (tmp_path / "opp_1.npz").write_text("", encoding="utf-8")
            nested_dir = tmp_path / "nested"
            nested_dir.mkdir(parents=True, exist_ok=True)
            (nested_dir / "opp_2.npz").write_text("", encoding="utf-8")
            opponent_pool_config_path.write_text(
                json.dumps({"checkpoints": ["opp_1.npz", "nested/opp_2.npz"]}),
                encoding="utf-8",
            )

            with mock.patch.object(self_play, "CheckpointEvaluator", FakeEvaluator):
                self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=7,
                    seed_pool=[7],
                    checkpoint="primary.npz",
                    input_encoding="kalah_v1",
                    simulations=8,
                    c_puct=1.25,
                    temperature_threshold=10,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=2,
                    shard_path=str(shard_path),
                    opponent_pool_config=str(opponent_pool_config_path),
                )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(rows), 0)

            resolved_checkpoints = [
                str((tmp_path / "opp_1.npz").resolve()),
                str((tmp_path / "nested/opp_2.npz").resolve()),
            ]
            expected_fingerprint = hashlib.sha256(
                json.dumps(
                    resolved_checkpoints, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
            ).hexdigest()

            self.assertEqual(
                expected_fingerprint,
                rows[0]["search_profile"]["opponent_pool_fingerprint"],
            )

    def test_encode_state_kalah_v3_exposes_extra_turn_and_capture_signals(self):
        extra_turn_state = {
            "player_pits": [0, 0, 0, 0, 1, 1],
            "opponent_pits": [0, 2, 0, 0, 0, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        capture_state = {
            "player_pits": [1, 0, 0, 2, 0, 0],
            "opponent_pits": [0, 0, 0, 0, 5, 1],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        extra_turn_features = self_play.encode_state(
            extra_turn_state, input_encoding="kalah_v3"
        )
        capture_features = self_play.encode_state(
            capture_state, input_encoding="kalah_v3"
        )

        extra_turn_index = kalah_v3_feature_index("player_extra_turn_available")
        capture_index = kalah_v3_feature_index("player_capture_available")

        self.assertEqual(1.0, extra_turn_features[extra_turn_index])
        self.assertEqual(0.0, extra_turn_features[capture_index])
        self.assertEqual(0.0, capture_features[extra_turn_index])
        self.assertEqual(1.0, capture_features[capture_index])

    def test_encode_state_kalah_v3_does_not_count_terminal_store_move_as_extra_turn(
        self,
    ):
        state = {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [0, 0, 0, 0, 0, 0],
            "player_store": 10,
            "opponent_store": 9,
            "current_player": 0,
        }

        game = KalahGame.from_state(state)
        self.assertTrue(game.move(game.pit_index(5)))
        self.assertTrue(game.over())
        self.assertFalse(
            self_play._is_immediate_extra_turn(KalahGame.from_state(state), 5)
        )

        features = self_play.encode_state(state, input_encoding="kalah_v3")
        extra_turn_index = kalah_v3_feature_index("player_extra_turn_available")
        self.assertEqual(0.0, features[extra_turn_index])

    def test_encode_state_kalah_v3_exposes_structure_and_endgame_pressure(self):
        tactical_state = {
            "player_pits": [0, 0, 7, 0, 0, 1],
            "opponent_pits": [0, 1, 0, 0, 0, 1],
            "player_store": 14,
            "opponent_store": 10,
            "current_player": 0,
        }
        quiet_state = {
            "player_pits": [2, 2, 2, 2, 2, 2],
            "opponent_pits": [2, 2, 2, 2, 2, 2],
            "player_store": 14,
            "opponent_store": 10,
            "current_player": 0,
        }

        tactical_features = self_play.encode_state(
            tactical_state, input_encoding="kalah_v3"
        )
        quiet_features = self_play.encode_state(quiet_state, input_encoding="kalah_v3")

        player_empty_index = kalah_v3_feature_index("player_empty_pit_ratio")
        player_high_seed_index = kalah_v3_feature_index("player_high_seed_ratio")
        player_one_to_store_index = kalah_v3_feature_index("player_one_to_store_ratio")
        player_remaining_index = kalah_v3_feature_index("player_remaining_stones")
        opponent_extra_turn_index = kalah_v3_feature_index(
            "opponent_extra_turn_available"
        )
        opponent_empty_index = kalah_v3_feature_index("opponent_empty_pit_ratio")
        opponent_remaining_index = kalah_v3_feature_index("opponent_remaining_stones")

        self.assertAlmostEqual(4.0 / 6.0, tactical_features[player_empty_index])
        self.assertAlmostEqual(1.0 / 6.0, tactical_features[player_high_seed_index])
        self.assertAlmostEqual(1.0 / 6.0, tactical_features[player_one_to_store_index])
        self.assertAlmostEqual(8.0 / 48.0, tactical_features[player_remaining_index])
        self.assertEqual(1.0, tactical_features[opponent_extra_turn_index])
        self.assertAlmostEqual(4.0 / 6.0, tactical_features[opponent_empty_index])
        self.assertAlmostEqual(2.0 / 48.0, tactical_features[opponent_remaining_index])
        self.assertLess(
            tactical_features[player_remaining_index],
            quiet_features[player_remaining_index],
        )
        self.assertGreater(
            tactical_features[player_empty_index], quiet_features[player_empty_index]
        )

    def test_checkpoint_evaluator_accepts_kalah_v3_placeholder_encoding(self):
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        checkpoint = {
            "w_hidden_1": np.zeros((27, 2), dtype=np.float32),
            "b_hidden_1": np.zeros(2, dtype=np.float32),
            "w_hidden_2": np.zeros((2, 2), dtype=np.float32),
            "b_hidden_2": np.zeros(2, dtype=np.float32),
            "w_policy": np.zeros((2, 6), dtype=np.float32),
            "b_policy": np.zeros(6, dtype=np.float32),
            "w_value": np.zeros((2, 1), dtype=np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.npz"
            np.savez(checkpoint_path, **checkpoint)

            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v3"
            )

            with mock.patch(
                "ml.alphazero_lite.self_play.encode_state", wraps=self_play.encode_state
            ) as encode_state:
                policy, value = evaluator.evaluate(game)

            encode_state.assert_called_once()
            self.assertEqual(
                "kalah_v3", encode_state.call_args.kwargs["input_encoding"]
            )
            self.assertEqual(6, len(policy))
            self.assertEqual(0.0, value)

    def test_checkpoint_evaluator_uses_selected_input_encoding(self):
        game = self._checkpoint_evaluator_test_game()

        checkpoint = {
            "w_hidden_1": np.zeros((15, 2), dtype=np.float32),
            "b_hidden_1": np.zeros(2, dtype=np.float32),
            "w_hidden_2": np.zeros((2, 2), dtype=np.float32),
            "b_hidden_2": np.zeros(2, dtype=np.float32),
            "w_policy": np.zeros((2, 6), dtype=np.float32),
            "b_policy": np.zeros(6, dtype=np.float32),
            "w_value": np.zeros((2, 1), dtype=np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.npz"
            np.savez(checkpoint_path, **checkpoint)

            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v2"
            )

            with mock.patch(
                "ml.alphazero_lite.self_play.encode_state", wraps=self_play.encode_state
            ) as encode_state:
                evaluator.evaluate(game)

            encode_state.assert_called_once()
            self.assertEqual(
                "kalah_v2", encode_state.call_args.kwargs["input_encoding"]
            )

    def test_checkpoint_evaluator_cache_computes_once_for_same_state_when_enabled(self):
        game = self._checkpoint_evaluator_test_game()

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = self._write_checkpoint(
                Path(tmp), "checkpoint.npz", input_features=15
            )
            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v1", cache_size=8
            )

            with mock.patch(
                "ml.alphazero_lite.self_play.encode_state", wraps=self_play.encode_state
            ) as encode_state:
                first_policy, first_value = evaluator.evaluate(game)
                second_policy, second_value = evaluator.evaluate(game)

            encode_state.assert_called_once()
            np.testing.assert_allclose(first_policy, second_policy)
            self.assertEqual(first_value, second_value)
            self.assertEqual(
                {"enabled": True, "hits": 1, "misses": 1, "size": 1},
                evaluator.cache_stats,
            )

    def test_checkpoint_evaluator_cached_policy_mutation_does_not_affect_later_evaluations(
        self,
    ):
        game = self._checkpoint_evaluator_test_game()

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = self._write_checkpoint(
                Path(tmp), "checkpoint.npz", input_features=15
            )
            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v1", cache_size=8
            )

            first_policy, _first_value = evaluator.evaluate(game)
            first_policy[0] = 0.75
            first_policy[1] = 0.25

            second_policy, _second_value = evaluator.evaluate(game)

            self.assertEqual(1.0 / 6.0, second_policy[0])
            self.assertEqual(1.0 / 6.0, second_policy[1])

    def test_checkpoint_evaluator_cache_key_changes_with_input_encoding_and_checkpoint_identity(
        self,
    ):
        game = self._checkpoint_evaluator_test_game()

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            tmp_path = Path(tmp)
            checkpoint_v1 = self._write_checkpoint(
                tmp_path, "checkpoint_v1.npz", input_features=15
            )
            checkpoint_v3 = self._write_checkpoint(
                tmp_path, "checkpoint_v3.npz", input_features=27
            )
            duplicate_checkpoint_v1 = self._write_checkpoint(
                tmp_path, "checkpoint_v1_copy.npz", input_features=15
            )

            evaluator_v1 = self_play.CheckpointEvaluator(
                checkpoint_v1, input_encoding="kalah_v1", cache_size=8
            )
            evaluator_v3 = self_play.CheckpointEvaluator(
                checkpoint_v3, input_encoding="kalah_v3", cache_size=8
            )
            duplicate_evaluator_v1 = self_play.CheckpointEvaluator(
                duplicate_checkpoint_v1,
                input_encoding="kalah_v1",
                cache_size=8,
            )

            with mock.patch(
                "ml.alphazero_lite.self_play.encode_state", wraps=self_play.encode_state
            ) as encode_state:
                evaluator_v1.evaluate(game)
                evaluator_v3.evaluate(game)
                duplicate_evaluator_v1.evaluate(game)

            self.assertEqual(3, encode_state.call_count)
            self.assertNotEqual(
                evaluator_v1._cache_key_for(game), evaluator_v3._cache_key_for(game)
            )
            self.assertNotEqual(
                evaluator_v1._cache_key_for(game),
                duplicate_evaluator_v1._cache_key_for(game),
            )
            self.assertEqual(
                {"enabled": True, "hits": 0, "misses": 1, "size": 1},
                evaluator_v1.cache_stats,
            )
            self.assertEqual(
                {"enabled": True, "hits": 0, "misses": 1, "size": 1},
                evaluator_v3.cache_stats,
            )
            self.assertEqual(
                {"enabled": True, "hits": 0, "misses": 1, "size": 1},
                duplicate_evaluator_v1.cache_stats,
            )

    def test_checkpoint_evaluator_disabled_cache_preserves_old_behavior(self):
        game = self._checkpoint_evaluator_test_game()

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = self._write_checkpoint(
                Path(tmp), "checkpoint.npz", input_features=15
            )
            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v1"
            )

            with mock.patch(
                "ml.alphazero_lite.self_play.encode_state", wraps=self_play.encode_state
            ) as encode_state:
                evaluator.evaluate(game)
                evaluator.evaluate(game)

            self.assertEqual(2, encode_state.call_count)
            self.assertEqual(
                {"enabled": False, "hits": 0, "misses": 0, "size": 0},
                evaluator.cache_stats,
            )

    def test_checkpoint_evaluator_applies_residual_v3_specialized_heads(self):
        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        checkpoint = {
            "w_input": np.array(
                [[1.0, 2.0]] * 12 + [[0.0, 0.0]] * 3,
                dtype=np.float32,
            ),
            "b_input": np.zeros(2, dtype=np.float32),
            "w_residual_1_1": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_1": np.zeros(2, dtype=np.float32),
            "w_residual_1_2": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_2": np.zeros(2, dtype=np.float32),
            "w_policy_hidden": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "b_policy_hidden": np.zeros(3, dtype=np.float32),
            "w_value_hidden": np.array(
                [
                    [1.0],
                    [1.0],
                ],
                dtype=np.float32,
            ),
            "b_value_hidden": np.zeros(1, dtype=np.float32),
            "w_policy": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "b_policy": np.zeros(6, dtype=np.float32),
            "w_value": np.array([[1.0 / 6.0]], dtype=np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }
        expected_policy = self_play.softmax(
            np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        expected_value = float(np.tanh(0.5))

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.npz"
            np.savez(checkpoint_path, **checkpoint)

            evaluator = self_play.CheckpointEvaluator(
                checkpoint_path, input_encoding="kalah_v1"
            )
            policy, value = evaluator.evaluate(game)

        np.testing.assert_allclose(expected_policy, policy, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(expected_value, value)

    def test_checkpoint_evaluator_rejects_partial_residual_v3_specialized_heads(self):
        checkpoint = {
            "w_input": np.zeros((15, 2), dtype=np.float32),
            "b_input": np.zeros(2, dtype=np.float32),
            "w_residual_1_1": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_1": np.zeros(2, dtype=np.float32),
            "w_residual_1_2": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_2": np.zeros(2, dtype=np.float32),
            "w_policy_hidden": np.zeros((2, 3), dtype=np.float32),
            "b_policy_hidden": np.zeros(3, dtype=np.float32),
            "w_policy": np.zeros((3, 6), dtype=np.float32),
            "b_policy": np.zeros(6, dtype=np.float32),
            "w_value": np.zeros((1, 1), dtype=np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.npz"
            np.savez(checkpoint_path, **checkpoint)

            with self.assertRaisesRegex(ValueError, "missing specialized head weights"):
                self_play.CheckpointEvaluator(
                    checkpoint_path, input_encoding="kalah_v1"
                )

    def test_checkpoint_evaluator_rejects_shape_mismatched_residual_v3_specialized_heads(
        self,
    ):
        checkpoint = {
            "w_input": np.zeros((15, 2), dtype=np.float32),
            "b_input": np.zeros(2, dtype=np.float32),
            "w_residual_1_1": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_1": np.zeros(2, dtype=np.float32),
            "w_residual_1_2": np.zeros((2, 2), dtype=np.float32),
            "b_residual_1_2": np.zeros(2, dtype=np.float32),
            "w_policy_hidden": np.zeros((3, 3), dtype=np.float32),
            "b_policy_hidden": np.zeros(3, dtype=np.float32),
            "w_value_hidden": np.zeros((2, 1), dtype=np.float32),
            "b_value_hidden": np.zeros(1, dtype=np.float32),
            "w_policy": np.zeros((3, 6), dtype=np.float32),
            "b_policy": np.zeros(6, dtype=np.float32),
            "w_value": np.zeros((1, 1), dtype=np.float32),
            "b_value": np.zeros(1, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.npz"
            np.savez(checkpoint_path, **checkpoint)

            with self.assertRaisesRegex(ValueError, r"w_policy_hidden must have shape"):
                self_play.CheckpointEvaluator(
                    checkpoint_path, input_encoding="kalah_v1"
                )

    def test_puct_rejects_unsupported_root_policy_mode(self):
        with self.assertRaisesRegex(ValueError, "unsupported root_policy_mode"):
            self_play.PUCT(
                evaluator=self_play.HeuristicEvaluator(),
                simulations=1,
                c_puct=1.25,
                rng=random.Random(7),
                root_policy_mode="softmax",
            )

    def test_self_play_defaults_keep_exploration_oriented_root_settings(self):
        options = self_play.build_search_options()

        self.assertEqual("visit_count", options["root_policy_mode"])
        self.assertEqual(0.0, options["tactical_root_bias"])

    def test_build_policy_target_keeps_default_temperature_weighted_behavior(self):
        visits = np.array([70, 20, 10, 0, 0, 0], dtype=np.float32)
        legal_moves = [0, 1, 2]

        target = self_play.build_policy_target(
            visits, legal_moves=legal_moves, temperature=1.0
        )
        default_policy = self_play.policy_from_visits(
            visits, legal_moves=legal_moves, temperature=1.0
        )

        self.assertEqual(default_policy, target)

    def test_build_policy_target_sharpened_preserves_legality_and_increases_top_move_separation(
        self,
    ):
        visits = np.array([70, 20, 10, 0, 0, 0], dtype=np.float32)
        legal_moves = [0, 1, 2]

        default_target = self_play.build_policy_target(
            visits, legal_moves=legal_moves, temperature=1.0
        )
        sharpened_target = self_play.build_policy_target(
            visits,
            legal_moves=legal_moves,
            temperature=1.0,
            mode="sharpened",
        )

        self.assertAlmostEqual(1.0, sum(sharpened_target), places=6)
        self.assertEqual([0.0, 0.0, 0.0], sharpened_target[3:])
        self.assertGreater(sharpened_target[0], default_target[0])
        self.assertLess(sharpened_target[1], default_target[1])
        self.assertLess(sharpened_target[2], default_target[2])
        self.assertGreater(
            sharpened_target[0] - sharpened_target[1],
            default_target[0] - default_target[1],
        )

    def test_build_policy_target_sharpened_preserves_low_temperature_top_move_separation(
        self,
    ):
        visits = np.array([70, 20, 10, 0, 0, 0], dtype=np.float32)
        legal_moves = [0, 1, 2]

        default_target = self_play.build_policy_target(
            visits, legal_moves=legal_moves, temperature=0.0
        )
        sharpened_target = self_play.build_policy_target(
            visits,
            legal_moves=legal_moves,
            temperature=0.0,
            mode="sharpened",
        )

        self.assertEqual([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], default_target)
        self.assertAlmostEqual(1.0, sum(sharpened_target), places=6)
        self.assertEqual([0.0, 0.0, 0.0], sharpened_target[3:])
        self.assertGreaterEqual(sharpened_target[0], default_target[0])
        self.assertGreaterEqual(
            sharpened_target[0] - sharpened_target[1],
            default_target[0] - default_target[1],
        )

    def test_build_policy_target_from_distribution_sharpens_cached_teacher_policy(self):
        policy = [0.0, 0.75, 0.25, 0.0, 0.0, 0.0]

        target = self_play.build_policy_target_from_distribution(
            policy, mode="sharpened"
        )

        np.testing.assert_allclose(
            np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array(target, dtype=np.float32),
            atol=1e-6,
        )

    def test_build_value_target_keeps_default_outcome_behavior(self):
        self.assertEqual(0.4, self_play.build_value_target(0.4))
        self.assertEqual(-0.4, self_play.build_value_target(-0.4))

    def test_build_value_target_sharpened_is_bounded_and_more_decisive(self):
        positive_default = self_play.build_value_target(0.4)
        negative_default = self_play.build_value_target(-0.4)

        positive_sharpened = self_play.build_value_target(0.4, mode="sharpened")
        negative_sharpened = self_play.build_value_target(-0.4, mode="sharpened")

        self.assertGreater(positive_sharpened, positive_default)
        self.assertLess(negative_sharpened, negative_default)
        self.assertGreaterEqual(positive_sharpened, 0.0)
        self.assertLessEqual(positive_sharpened, 1.0)
        self.assertGreaterEqual(negative_sharpened, -1.0)
        self.assertLessEqual(negative_sharpened, 0.0)

    def test_build_value_target_sharpened_preserves_order_and_endpoints(self):
        targets = [
            self_play.build_value_target(value, mode="sharpened")
            for value in (-1.0, -0.4, 0.0, 0.4, 1.0)
        ]

        self.assertEqual(-1.0, targets[0])
        self.assertEqual(0.0, targets[2])
        self.assertEqual(1.0, targets[4])
        self.assertLess(targets[0], targets[1])
        self.assertLess(targets[1], targets[2])
        self.assertLess(targets[2], targets[3])
        self.assertLess(targets[3], targets[4])

    def test_phase_aware_value_target_bucket_selection_uses_fixed_move_ranges(self):
        self.assertEqual("early", self_play.value_target_bucket_for_move_index(0))
        self.assertEqual("early", self_play.value_target_bucket_for_move_index(9))
        self.assertEqual("mid", self_play.value_target_bucket_for_move_index(10))
        self.assertEqual("mid", self_play.value_target_bucket_for_move_index(29))
        self.assertEqual("late", self_play.value_target_bucket_for_move_index(30))

    def test_hybrid_value_target_bucket_selection_uses_fixed_move_ranges(self):
        self.assertEqual("early", self_play.value_target_bucket_for_move_index(0))
        self.assertEqual("mid", self_play.value_target_bucket_for_move_index(10))
        self.assertEqual("late", self_play.value_target_bucket_for_move_index(30))

    def test_phase_aware_value_target_becomes_more_decisive_across_buckets(self):
        early = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.4,
            move_index=0,
            mode="phase_aware_sharpened",
        )
        mid = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.4,
            move_index=10,
            mode="phase_aware_sharpened",
        )
        late = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.4,
            move_index=30,
            mode="phase_aware_sharpened",
        )

        self.assertGreater(early, 0.4)
        self.assertGreater(mid, early)
        self.assertGreater(late, mid)
        self.assertLessEqual(late, 1.0)

    def test_hybrid_value_target_blends_search_more_early_and_outcome_more_late(self):
        early = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.2,
            move_index=0,
            mode="hybrid",
        )
        mid = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.2,
            move_index=10,
            mode="hybrid",
        )
        late = self_play.derive_self_play_value_target(
            outcome_value=1.0,
            search_value=0.2,
            move_index=30,
            mode="hybrid",
        )

        self.assertGreater(early, 0.2)
        self.assertGreater(mid, early)
        self.assertGreater(late, mid)
        self.assertLess(late, 1.0)

    def test_hybrid_value_target_returns_zero_for_draws(self):
        value = self_play.derive_self_play_value_target(
            outcome_value=0.0,
            search_value=0.8,
            move_index=30,
            mode="hybrid",
        )

        self.assertEqual(0.0, value)

    def test_tactical_root_bias_is_root_only_during_live_search(self):
        seen_states = []

        class FakeGame:
            def __init__(self, state="root", current_player=0):
                self.state = state
                self.current_player = current_player
                self.winner = None

            def clone(self):
                return FakeGame(self.state, self.current_player)

            def possible_moves(self):
                return [0, 1] if self.state in {"root", "child"} else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                if self.state == "root":
                    self.state = "child"
                    self.current_player = 1
                    return True
                if self.state == "child":
                    self.state = f"leaf_{absolute_move}"
                    self.current_player = 0
                    return True
                return False

            def over(self):
                return self.state.startswith("leaf")

            @property
            def pits(self):
                return [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            @property
            def captured_seeds(self):
                return [0, 0]

            def to_state(self):
                return {
                    "player_pits": [1, 1, 0, 0, 0, 0],
                    "opponent_pits": [0, 0, 0, 0, 0, 0],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                    "state": self.state,
                }

        class ScriptedEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                return np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.0

        search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            tactical_root_bias=0.1,
        )

        original_apply = search.apply_tactical_root_bias

        def tracking_apply(game, priors):
            seen_states.append(game.state)
            return original_apply(game, priors)

        search.apply_tactical_root_bias = tracking_apply
        search.run(FakeGame())

        self.assertEqual(["root"], seen_states)

    def test_cli_generates_trajectory_rows_with_valid_contract(self):
        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "3",
                    "--seed",
                    "7",
                    "--simulations",
                    "24",
                    "--temperature-threshold",
                    "6",
                    "--dirichlet-alpha",
                    "0.3",
                    "--workers",
                    "2",
                    "--seed-sweep",
                    "7,8,9",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())

            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(rows), 3)

            unique_states = set()
            for row in rows:
                self.assertEqual("default", row["policy_target_mode"])
                self.assertEqual("v1", row["search_profile"]["version"])
                self.assertEqual("self_play", row["search_profile"]["kind"])
                self.assertEqual(
                    row["search_profile"]["hash"], row["search_profile_hash"]
                )
                self.assertEqual(15, len(row["state"]))
                self.assertEqual(6, len(row["policy"]))
                self.assertAlmostEqual(1.0, sum(row["policy"]), places=4)
                self.assertTrue(all(prob >= 0.0 for prob in row["policy"]))
                self.assertGreaterEqual(row["value"], -1.0)
                self.assertLessEqual(row["value"], 1.0)
                unique_states.add(tuple(row["state"]))

            self.assertGreater(len(unique_states), 2)

    def test_cli_emits_machine_readable_cache_metrics(self):
        with tempfile.TemporaryDirectory(prefix="azlite-self-play-metrics-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"
            checkpoint_path = self._write_checkpoint(
                tmp_path, "checkpoint.npz", input_features=15
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--seed",
                    "7",
                    "--simulations",
                    "8",
                    "--workers",
                    "1",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--evaluator-cache-size",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertRegex(result.stdout, r"cache_enabled=true")
            self.assertRegex(result.stdout, r"cache_hits=\d+")
            self.assertRegex(result.stdout, r"cache_misses=\d+")
            self.assertRegex(result.stdout, r"cache_hit_rate=\d+(?:\.\d+)?")

    def test_cli_emits_cache_disabled_metric_when_evaluator_cache_is_off(self):
        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-metrics-disabled-"
        ) as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--seed",
                    "7",
                    "--simulations",
                    "8",
                    "--workers",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertRegex(result.stdout, r"cache_enabled=false")
            self.assertRegex(result.stdout, r"cache_hits=0")
            self.assertRegex(result.stdout, r"cache_misses=0")

    def test_cli_keeps_cache_disabled_metric_false_without_checkpoint_evaluator(self):
        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-metrics-no-checkpoint-"
        ) as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--seed",
                    "7",
                    "--simulations",
                    "8",
                    "--workers",
                    "1",
                    "--evaluator-cache-size",
                    "8",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertRegex(result.stdout, r"cache_enabled=false")
            self.assertRegex(result.stdout, r"cache_hits=0")
            self.assertRegex(result.stdout, r"cache_misses=0")

    def test_cli_emits_requested_value_target_mode_in_rows(self):
        with tempfile.TemporaryDirectory(prefix="azlite-self-play-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play_value_target.jsonl"

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--seed",
                    "7",
                    "--simulations",
                    "8",
                    "--workers",
                    "1",
                    "--value-target-mode",
                    "sharpened",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())

            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertGreater(len(rows), 0)
            self.assertTrue(
                all(row["value_target_mode"] == "sharpened" for row in rows)
            )

    def test_cli_uses_indexed_hidden_layers_when_checkpoint_has_aliases(self):
        with tempfile.TemporaryDirectory(prefix="azlite-self-play-ckpt-") as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"

            # Valid deep path uses w_hidden_1..3 and projects to 64 before heads.
            # Legacy aliases w1/w2 are intentionally mismatched and must be ignored.
            import numpy as np

            np.savez(
                checkpoint_path,
                w_hidden_1=np.zeros((15, 128), dtype=np.float32),
                b_hidden_1=np.zeros((128,), dtype=np.float32),
                w_hidden_2=np.zeros((128, 128), dtype=np.float32),
                b_hidden_2=np.zeros((128,), dtype=np.float32),
                w_hidden_3=np.zeros((128, 64), dtype=np.float32),
                b_hidden_3=np.zeros((64,), dtype=np.float32),
                w1=np.zeros((15, 128), dtype=np.float32),
                b1=np.zeros((128,), dtype=np.float32),
                w2=np.zeros((128, 128), dtype=np.float32),
                b2=np.zeros((128,), dtype=np.float32),
                w_policy=np.zeros((64, 6), dtype=np.float32),
                b_policy=np.zeros((6,), dtype=np.float32),
                w_value=np.zeros((64, 1), dtype=np.float32),
                b_value=np.zeros((1,), dtype=np.float32),
            )

            result = subprocess.run(
                [
                    self.executable_python(),
                    "ml/alphazero_lite/self_play.py",
                    "--out",
                    str(out_path),
                    "--games",
                    "1",
                    "--seed",
                    "11",
                    "--simulations",
                    "8",
                    "--workers",
                    "1",
                    "--checkpoint",
                    str(checkpoint_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            self.assertTrue(out_path.exists())
            rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(row["policy_target_mode"] == "default" for row in rows))

    def test_run_self_play_worker_reuses_tree_root_between_plies_when_enabled(self):
        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = self.moves_played % 2
                return absolute_move == 0

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": self.moves_played,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRoot:
            def __init__(self, child=None, q_value=0.0):
                self.child = child
                self.q_value = q_value

            def child_for_action(self, action):
                if action != 0:
                    raise AssertionError(f"unexpected action {action}")
                return self.child

        created_roots = []
        second_root = FakeRoot()
        first_root = FakeRoot(child=second_root)
        roots_to_return = [first_root, second_root]

        class FakePUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng
                del search_options
                created_roots.append(root)

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                root = roots_to_return[len(created_roots) - 1]
                visits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, root

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=4,
                        shard_path=str(shard_path),
                        tree_reuse_enabled=True,
                    )

            self.assertEqual(2, result["rows_written"])
            self.assertEqual([None, second_root], created_roots)

    def test_run_self_play_worker_actually_reuses_subtree_when_tree_reuse_path_enabled(
        self,
    ):
        created_roots = []
        reused_roots = []
        real_puct = self_play.PUCT

        class TrackingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                created_roots.append(root)
                self._root = root
                self._delegate = real_puct(
                    evaluator=evaluator,
                    simulations=simulations,
                    c_puct=c_puct,
                    rng=rng,
                    root=root,
                    **search_options,
                )

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                visits, root = self._delegate.run(
                    game,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                )
                reused_roots.append(root is self._root and self._root is not None)
                return visits, root

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(self_play, "PUCT", TrackingPUCT):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    seed_pool=[42],
                    checkpoint=None,
                    input_encoding="kalah_v1",
                    simulations=8,
                    c_puct=1.25,
                    temperature_threshold=10,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=2,
                    shard_path=str(shard_path),
                    tree_reuse_enabled=True,
                )

        self.assertEqual(2, result["rows_written"])
        self.assertEqual(2, len(created_roots))
        self.assertIsNone(created_roots[0])
        self.assertIsNotNone(created_roots[1])
        self.assertEqual([False, True], reused_roots)

    def test_run_self_play_worker_persists_and_reuses_subtree_when_reuse_subtree_enabled(
        self,
    ):
        created_roots = []
        reused_roots = []
        real_puct = self_play.PUCT

        class TrackingPUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                created_roots.append(root)
                self._root = root
                self._delegate = real_puct(
                    evaluator=evaluator,
                    simulations=simulations,
                    c_puct=c_puct,
                    rng=rng,
                    root=root,
                    **search_options,
                )

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                visits, root = self._delegate.run(
                    game,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                )
                reused_roots.append(root is self._root and self._root is not None)
                return visits, root

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(self_play, "PUCT", TrackingPUCT):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    seed_pool=[42],
                    checkpoint=None,
                    input_encoding="kalah_v1",
                    simulations=8,
                    c_puct=1.25,
                    temperature_threshold=10,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=2,
                    shard_path=str(shard_path),
                    reuse_subtree=True,
                )

        self.assertEqual(2, result["rows_written"])
        self.assertEqual(2, len(created_roots))
        self.assertIsNone(created_roots[0])
        self.assertIsNotNone(created_roots[1])
        self.assertEqual([False, True], reused_roots)

    def test_run_self_play_worker_reuses_opening_cache_teacher_outputs_on_hit(self):
        cached_policy = [0.0, 0.75, 0.25, 0.0, 0.0, 0.0]
        cached_value = 0.5
        lookup_calls = []
        cached_teacher_profile = {
            "version": "v1",
            "kind": "opening_cache_teacher",
            "player_mode": "classic_mcts",
            "classic_mcts_simulations": 1200,
            "hash": "cached-profile-hash",
        }

        class FakeOpeningCache:
            search_profile = cached_teacher_profile

            def lookup(self, state, *, ply):
                lookup_calls.append((state, ply))
                return {
                    "policy": cached_policy,
                    "value": cached_value,
                    "provenance": {
                        "teacher_kind": "classic_mcts",
                        "search_profile_hash": cached_teacher_profile["hash"],
                    },
                }

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [1] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 1

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-cache-hit-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(
                    self_play,
                    "PUCT",
                    side_effect=AssertionError("PUCT should not run on cache hit"),
                ):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        opening_cache=FakeOpeningCache(),
                        policy_target_mode="sharpened",
                        value_target_mode="sharpened",
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(1, len(lookup_calls))
        np.testing.assert_allclose(
            np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array(rows[0]["policy"], dtype=np.float32),
            atol=1e-6,
        )
        self.assertEqual("opening_cache", rows[0]["teacher_source"])
        self.assertEqual("sharpened", rows[0]["policy_target_mode"])
        self.assertEqual("sharpened", rows[0]["policy_target_actual_mode"])
        self.assertEqual(cached_teacher_profile, rows[0]["teacher_search_profile"])
        self.assertEqual(
            cached_teacher_profile["hash"], rows[0]["teacher_search_profile_hash"]
        )
        self.assertAlmostEqual(
            self_play.canonical_value_target(
                outcome_value=1.0,
                search_value=cached_value,
                mode="sharpened",
            ),
            rows[0]["value"],
            places=6,
        )

    def test_run_self_play_worker_lazy_loads_opening_cache_teacher_profile_on_hit(self):
        cached_policy = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        cached_value = 0.5
        cached_teacher_profile = {
            "version": "v1",
            "kind": "opening_cache_teacher",
            "player_mode": "classic_mcts",
            "classic_mcts_simulations": 1200,
            "hash": "cached-profile-hash",
        }
        opening_state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        opening_cache_payload = {
            "schema": "azlite_opening_cache_v1",
            "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
            "search_profile": cached_teacher_profile,
            "entries": {
                hashlib.sha256(
                    json.dumps(
                        opening_state,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(): {
                    "state": opening_state,
                    "side_to_move": 0,
                    "selected_move": 1,
                    "policy": cached_policy,
                    "value": cached_value,
                    "budget": {
                        "baseline_simulations": 1200,
                        "probe_simulations": 1200,
                        "chosen_simulations": 1200,
                        "final_simulations": 1200,
                        "root_latency_ms": 8.0,
                    },
                    "provenance": {
                        "teacher_kind": "classic_mcts",
                        "search_profile_hash": cached_teacher_profile["hash"],
                        "hit_count_in_sources": 1,
                        "example_origins": ["self_play:test"],
                    },
                }
            },
        }

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [1] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 1

            def to_state(self):
                return dict(opening_state)

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-cache-lazy-hit-"
        ) as tmp:
            tmp_path = Path(tmp)
            shard_path = tmp_path / "worker.jsonl"
            opening_cache_path = tmp_path / "opening_cache.json"
            opening_cache_path.write_text(
                json.dumps(opening_cache_payload), encoding="utf-8"
            )

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(
                    self_play,
                    "PUCT",
                    side_effect=AssertionError("PUCT should not run on cache hit"),
                ):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        opening_cache_path=str(opening_cache_path),
                        value_target_mode="sharpened",
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(cached_policy, rows[0]["policy"])
        self.assertEqual("opening_cache", rows[0]["teacher_source"])
        self.assertEqual("default", rows[0]["policy_target_actual_mode"])
        self.assertEqual(cached_teacher_profile, rows[0]["teacher_search_profile"])
        self.assertEqual(
            cached_teacher_profile["hash"], rows[0]["teacher_search_profile_hash"]
        )
        self.assertAlmostEqual(
            self_play.canonical_value_target(
                outcome_value=1.0,
                search_value=cached_value,
                mode="sharpened",
            ),
            rows[0]["value"],
            places=6,
        )

    def test_run_self_play_worker_marks_teacher_source_on_opening_cache_miss(self):
        lookup_calls = []

        class FakeOpeningCache:
            def lookup(self, state, *, ply):
                lookup_calls.append((state, ply))
                return None

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 0

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRoot:
            q_value = 0.4

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng, root, search_options

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                visits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, FakeRoot()

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-cache-miss-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        opening_cache=FakeOpeningCache(),
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(1, len(lookup_calls))
        self.assertEqual("puct", rows[0]["teacher_source"])
        self.assertEqual(rows[0]["search_profile"], rows[0]["teacher_search_profile"])
        self.assertEqual(
            rows[0]["search_profile_hash"], rows[0]["teacher_search_profile_hash"]
        )

    def test_run_self_play_worker_cache_miss_uses_gameplay_sampling_policy_not_sharpened_target(
        self,
    ):
        lookup_calls = []
        created_games = []

        class FakeOpeningCache:
            def lookup(self, state, *, ply):
                lookup_calls.append((state, ply))
                return None

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0
                self.selected_moves = []

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0, 1] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.selected_moves.append(absolute_move)
                self.moves_played += 1
                return True

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRoot:
            q_value = 0.4

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng, root, search_options

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                visits = np.array([70.0, 30.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, FakeRoot()

        class FakeRandom:
            def __init__(self, seed):
                self.seed = seed

            def random(self):
                return 0.8

            def randint(self, a, b):
                del a, b
                return 0

        def build_game(_state):
            game = FakeGame()
            created_games.append(game)
            return game

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-cache-miss-sampling-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", side_effect=build_game
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    with mock.patch.object(self_play.random, "Random", FakeRandom):
                        result = self_play.run_self_play_worker(
                            worker_id=0,
                            start_index=0,
                            games=1,
                            seed=42,
                            seed_pool=[42],
                            checkpoint=None,
                            input_encoding="kalah_v1",
                            simulations=8,
                            c_puct=1.25,
                            temperature_threshold=10,
                            temperature=1.0,
                            temperature_late=1.0,
                            dirichlet_alpha=0.3,
                            dirichlet_epsilon=0.25,
                            max_moves=2,
                            shard_path=str(shard_path),
                            opening_cache=FakeOpeningCache(),
                            policy_target_mode="sharpened",
                        )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(1, len(lookup_calls))
        self.assertEqual([1], created_games[0].selected_moves)
        self.assertEqual("sharpened", rows[0]["policy_target_mode"])
        self.assertEqual("sharpened", rows[0]["policy_target_actual_mode"])
        self.assertGreater(rows[0]["policy"][0], 0.8)
        self.assertLess(rows[0]["policy"][1], 0.2)

    def test_run_self_play_worker_denoised_targets_preserve_sampling_noise_metadata(
        self,
    ):
        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0, 1] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move in {0, 1}

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRoot:
            q_value = 0.25

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            run_calls = []

            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng, root, search_options

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game
                FakePUCT.run_calls.append((dirichlet_alpha, dirichlet_epsilon))
                if dirichlet_alpha is None:
                    visits = np.array(
                        [10.0, 90.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )
                else:
                    visits = np.array(
                        [90.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
                    )
                return visits, FakeRoot()

            def root_summary(self):
                return {
                    "root_prior_telemetry": {
                        "before": [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
                        "after": [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
                    }
                }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-denoised-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        policy_target_noise_mode="denoised",
                        write_root_target_telemetry=True,
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual([(0.3, 0.25), (None, 0.0)], FakePUCT.run_calls)
        self.assertEqual("denoised", rows[0]["policy_target_noise_mode"])
        self.assertTrue(rows[0]["action_sampling_noise_enabled"])
        self.assertEqual(0.0, rows[0]["target_dirichlet_epsilon"])
        self.assertEqual(0.25, rows[0]["sampling_dirichlet_epsilon"])
        self.assertEqual(1, rows[0]["top_target_move"])
        self.assertEqual([10, 90, 0, 0, 0, 0], rows[0]["root_visit_counts"])
        self.assertIn("root_policy_before_root_prior_transform", rows[0])
        self.assertIn("root_policy_after_root_prior_transform", rows[0])
        self.assertEqual(rows[0]["stored_policy_target"], rows[0]["policy"])

    def test_run_self_play_worker_uses_opponent_pool_checkpoint_for_player_one(self):
        evaluator_paths = []

        class FakeEvaluator(self_play.Evaluator):
            def __init__(self, checkpoint_path, *, input_encoding):
                evaluator_paths.append((str(checkpoint_path), input_encoding))

            def evaluate(self, game):
                del game
                priors = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return priors, 0.0

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-opponents-") as tmp:
            tmp_path = Path(tmp)
            shard_path = tmp_path / "worker.jsonl"
            opponent_pool_config_path = tmp_path / "pool.json"
            (tmp_path / "opp_1.npz").write_text("", encoding="utf-8")
            (tmp_path / "opp_2.npz").write_text("", encoding="utf-8")
            opponent_pool_config_path.write_text(
                json.dumps({"checkpoints": ["opp_1.npz", "opp_2.npz"]}),
                encoding="utf-8",
            )

            with mock.patch.object(self_play, "CheckpointEvaluator", FakeEvaluator):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=2,
                    seed=7,
                    seed_pool=[7],
                    checkpoint="primary.npz",
                    input_encoding="kalah_v1",
                    simulations=8,
                    c_puct=1.25,
                    temperature_threshold=10,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=2,
                    shard_path=str(shard_path),
                    opponent_pool_config=str(opponent_pool_config_path),
                )

        self.assertGreater(result["rows_written"], 0)
        loaded_paths = [path for path, _encoding in evaluator_paths]
        self.assertIn("primary.npz", loaded_paths)
        self.assertTrue(
            any(
                path.endswith("opp_1.npz") or path.endswith("opp_2.npz")
                for path in loaded_paths
            )
        )

    def test_search_preserves_value_sign_when_child_keeps_turn(self):
        class FakeGame:
            def __init__(self, current_player):
                self.current_player = current_player

        parent = self_play.Node(game=FakeGame(0), expanded=True)
        child = self_play.Node(game=FakeGame(0), expanded=False)
        parent.children = {0: child}
        puct = self_play.PUCT(
            evaluator=mock.Mock(), simulations=8, c_puct=1.25, rng=random.Random(42)
        )

        with mock.patch.object(self_play, "terminal_value", return_value=None):
            with mock.patch.object(puct, "_select_child", return_value=child):
                with mock.patch.object(
                    puct, "_expand", return_value=(np.zeros(6, dtype=np.float32), 0.5)
                ):
                    value = puct._search(parent)

        self.assertEqual(0.5, value)
        self.assertEqual(1, child.visit_count)
        self.assertEqual(0.5, child.value_sum)

    def test_puct_value_only_mode_uses_flat_legal_priors_and_preserves_evaluator_value(
        self,
    ):
        game = self._checkpoint_evaluator_test_game()
        node = self_play.Node(game=game)
        evaluator_priors = np.array([0.55, 0.05, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        class ScriptedEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                return evaluator_priors.copy(), 0.75

        search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            ablation_mode="value_only",
        )

        priors, value = search._expand(
            node,
            apply_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            is_root=False,
        )

        expected_priors = np.full(6, 1.0 / 6.0, dtype=np.float32)
        np.testing.assert_allclose(expected_priors, priors)
        self.assertEqual(0.75, value)

    def test_puct_policy_only_mode_preserves_evaluator_priors_and_uses_neutral_value(
        self,
    ):
        game = self._checkpoint_evaluator_test_game()
        node = self_play.Node(game=game)
        evaluator_priors = np.array([0.55, 0.05, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        class ScriptedEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                return evaluator_priors.copy(), 0.75

        search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            ablation_mode="policy_only",
        )

        priors, value = search._expand(
            node,
            apply_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            is_root=False,
        )

        np.testing.assert_allclose(evaluator_priors, priors, atol=1e-7)
        self.assertEqual(0.0, value)

    def test_puct_full_mode_preserves_evaluator_priors_and_value(self):
        game = self._checkpoint_evaluator_test_game()
        node = self_play.Node(game=game)
        evaluator_priors = np.array([0.55, 0.05, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        class ScriptedEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                return evaluator_priors.copy(), 0.75

        search = self_play.PUCT(
            evaluator=ScriptedEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            ablation_mode="full",
        )

        priors, value = search._expand(
            node,
            apply_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            is_root=False,
        )

        np.testing.assert_allclose(evaluator_priors, priors, atol=1e-7)
        self.assertEqual(0.75, value)

    def test_puct_value_trust_schedule_changes_child_selection_score(self):
        class FakeGame:
            current_player = 0
            pits = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        parent = self_play.Node(game=FakeGame(), expanded=True, visit_count=10)
        high_q_child = self_play.Node(
            game=FakeGame(), prior=0.1, visit_count=2, value_sum=1.4
        )
        high_u_child = self_play.Node(
            game=FakeGame(), prior=0.9, visit_count=10, value_sum=3.0
        )
        parent.children = {0: high_q_child, 1: high_u_child}

        default_search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
        )
        scheduled_search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            value_trust_schedule={
                "enabled": True,
                "opening": 0.1,
                "midgame": 1.0,
                "late": 1.0,
            },
        )

        self.assertIs(high_q_child, default_search._select_child(parent))
        self.assertIs(high_u_child, scheduled_search._select_child(parent))

    def test_select_child_uses_fast_path_without_building_telemetry_entries(self):
        class FakeGame:
            current_player = 0
            pits = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        parent = self_play.Node(game=FakeGame(), expanded=True, visit_count=10)
        best_child = self_play.Node(
            game=FakeGame(), prior=0.9, visit_count=1, value_sum=0.0
        )
        other_child = self_play.Node(
            game=FakeGame(), prior=0.1, visit_count=2, value_sum=1.4
        )
        parent.children = {0: other_child, 1: best_child}

        search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
        )

        with mock.patch.object(
            search,
            "_selection_entries",
            side_effect=AssertionError("telemetry path should not run"),
        ):
            selected_child = search._select_child(parent)

        self.assertIs(best_child, selected_child)

    def test_root_summary_includes_selection_breakdown(self):
        class FakeGame:
            current_player = 0
            pits = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        root = self_play.Node(
            game=FakeGame(), expanded=True, visit_count=12, value_sum=6.0
        )
        visited_child = self_play.Node(
            game=FakeGame(), prior=0.1, visit_count=2, value_sum=1.4
        )
        unvisited_child = self_play.Node(
            game=FakeGame(), prior=0.9, visit_count=0, value_sum=0.0
        )
        tied_prior_child = self_play.Node(
            game=FakeGame(), prior=0.9, visit_count=0, value_sum=0.0
        )
        root.children = {2: tied_prior_child, 1: unvisited_child, 0: visited_child}

        search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            fpu_mode="parent_q",
        )
        search._last_root = root

        summary = search.root_summary()

        self.assertIn("selection_breakdown", summary)
        breakdown = summary["selection_breakdown"]
        self.assertEqual("parent_q", breakdown["fpu_mode"])
        self.assertEqual(1.0, breakdown["value_trust_multiplier"])
        self.assertEqual(0.5, breakdown["parent_q_value"])
        self.assertEqual(0, breakdown["selected_move"])
        self.assertEqual(1, breakdown["reference_move"])
        self.assertEqual(1, breakdown["highest_prior_move"])
        self.assertEqual("highest_prior_move", breakdown["reference_move_kind"])
        self.assertEqual(
            "highest current PUCT selection score using deterministic telemetry ordering",
            breakdown["next_simulation_move_kind"],
        )
        self.assertEqual(1, breakdown["next_simulation_move"])
        self.assertEqual(1, breakdown["policy_top_move"])
        self.assertEqual(0, breakdown["visit_top_move"])
        self.assertEqual(0, breakdown["q_top_move"])

        self.assertEqual([0, 1, 2], [entry["move"] for entry in breakdown["moves"]])

        moves = {entry["move"]: entry for entry in breakdown["moves"]}
        self.assertEqual({0, 1, 2}, set(moves))
        self.assertAlmostEqual(0.7, moves[0]["q_value"])
        self.assertAlmostEqual(0.7, moves[0]["selection_q_value"])
        self.assertFalse(moves[0]["used_fpu"])
        self.assertIsNone(moves[0]["fpu_value"])
        self.assertEqual(0.0, moves[1]["q_value"])
        self.assertEqual(0.5, moves[1]["selection_q_value"])
        self.assertTrue(moves[1]["used_fpu"])
        self.assertEqual(0.5, moves[1]["fpu_value"])
        self.assertEqual(0.9, moves[2]["prior"])
        self.assertEqual(moves[1]["selection_score"], moves[2]["selection_score"])

    def test_root_summary_includes_sparse_visit_snapshots(self):
        class DeterministicEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = self_play.PUCT(
            evaluator=DeterministicEvaluator(),
            simulations=8,
            c_puct=1.25,
            rng=random.Random(7),
        )

        visits, _root = search.run(game)
        summary = search.root_summary()

        self.assertIn("visit_snapshots", summary)
        self.assertEqual(4, len(summary["visit_snapshots"]))
        self.assertEqual(
            [
                {"simulation": 1, "visits": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                {"simulation": 2, "visits": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                {"simulation": 4, "visits": [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                {"simulation": 8, "visits": [8.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            ],
            [
                {"simulation": snapshot["simulation"], "visits": snapshot["visits"]}
                for snapshot in summary["visit_snapshots"]
            ],
        )
        self.assertEqual([8.0, 0.0, 0.0, 0.0, 0.0, 0.0], visits.tolist())

    def test_root_summary_visit_snapshots_include_per_move_root_telemetry(self):
        class FakeGame:
            current_player = 0
            pits = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

            def clone(self):
                return self

        root = self_play.Node(game=FakeGame(), expanded=True)
        root.children = {
            0: self_play.Node(game=FakeGame(), prior=0.2),
            1: self_play.Node(game=FakeGame(), prior=0.6),
            2: self_play.Node(game=FakeGame(), prior=0.2),
        }

        search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=4,
            c_puct=1.25,
            rng=random.Random(7),
            fpu_mode="parent_q",
        )

        def scripted_search(node):
            self.assertIs(root, node)
            simulation_index = root.visit_count + 1
            if simulation_index == 1:
                root.children[0].visit_count = 1
                root.children[0].value_sum = 0.8
            elif simulation_index == 2:
                root.children[1].visit_count = 1
                root.children[1].value_sum = 0.6
            elif simulation_index == 3:
                root.children[1].visit_count = 2
                root.children[1].value_sum = 1.0
            elif simulation_index == 4:
                root.children[2].visit_count = 1
                root.children[2].value_sum = -0.2
            else:
                self.fail(f"unexpected simulation index {simulation_index}")
            return 0.5

        with (
            mock.patch.object(search, "_root_for", return_value=root),
            mock.patch.object(
                search, "_expand", return_value=(np.zeros(6, dtype=np.float32), 0.0)
            ),
            mock.patch.object(search, "_search", side_effect=scripted_search),
        ):
            visits, _returned_root = search.run(root.game)

        self.assertEqual([1.0, 2.0, 1.0, 0.0, 0.0, 0.0], visits.tolist())

        summary = search.root_summary()
        snapshots = summary["visit_snapshots"]

        self.assertEqual([1, 2, 4], [snapshot["simulation"] for snapshot in snapshots])

        first_snapshot = snapshots[0]
        self.assertEqual(0, first_snapshot["selected_move"])
        self.assertEqual(1, first_snapshot["reference_move_by_prior"])
        self.assertEqual(2, first_snapshot["reference_move_rank_by_visits"])
        self.assertEqual(2, first_snapshot["reference_move_rank_by_q"])
        self.assertEqual(1, first_snapshot["reference_move_rank_by_selection_score"])
        self.assertEqual([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], first_snapshot["visits"])

        first_snapshot_moves = {
            entry["move"]: entry for entry in first_snapshot["moves"]
        }
        self.assertEqual({0, 1, 2}, set(first_snapshot_moves))
        self.assertEqual(0.2, first_snapshot_moves[0]["prior"])
        self.assertEqual(1, first_snapshot_moves[0]["visit_count"])
        self.assertAlmostEqual(0.8, first_snapshot_moves[0]["q_value"])
        self.assertAlmostEqual(0.8, first_snapshot_moves[0]["selection_q_value"])
        self.assertAlmostEqual(0.125, first_snapshot_moves[0]["u_component"])
        self.assertAlmostEqual(0.925, first_snapshot_moves[0]["selection_score"])
        self.assertFalse(first_snapshot_moves[0]["used_fpu"])
        self.assertEqual(0.6, first_snapshot_moves[1]["prior"])
        self.assertEqual(0, first_snapshot_moves[1]["visit_count"])
        self.assertEqual(0.0, first_snapshot_moves[1]["q_value"])
        self.assertAlmostEqual(0.5, first_snapshot_moves[1]["selection_q_value"])
        self.assertAlmostEqual(0.75, first_snapshot_moves[1]["u_component"])
        self.assertAlmostEqual(1.25, first_snapshot_moves[1]["selection_score"])
        self.assertTrue(first_snapshot_moves[1]["used_fpu"])
        self.assertEqual(0.2, first_snapshot_moves[2]["prior"])
        self.assertEqual(0, first_snapshot_moves[2]["visit_count"])
        self.assertEqual(0.0, first_snapshot_moves[2]["q_value"])
        self.assertAlmostEqual(0.5, first_snapshot_moves[2]["selection_q_value"])
        self.assertAlmostEqual(0.25, first_snapshot_moves[2]["u_component"])
        self.assertAlmostEqual(0.75, first_snapshot_moves[2]["selection_score"])
        self.assertTrue(first_snapshot_moves[2]["used_fpu"])

        final_snapshot = snapshots[-1]
        self.assertEqual(1, final_snapshot["selected_move"])
        self.assertEqual(1, final_snapshot["reference_move_by_prior"])
        self.assertEqual(1, final_snapshot["reference_move_rank_by_visits"])
        self.assertEqual(2, final_snapshot["reference_move_rank_by_q"])
        self.assertEqual(2, final_snapshot["reference_move_rank_by_selection_score"])
        self.assertEqual([1.0, 2.0, 1.0, 0.0, 0.0, 0.0], final_snapshot["visits"])

    def test_root_summary_visit_snapshot_ranks_follow_engine_tiebreakers(self):
        class FakeGame:
            current_player = 0
            pits = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

        root = self_play.Node(
            game=FakeGame(), expanded=True, visit_count=3, value_sum=1.5
        )
        root.children = {
            0: self_play.Node(game=FakeGame(), prior=0.2, visit_count=1, value_sum=0.6),
            1: self_play.Node(
                game=FakeGame(), prior=0.5, visit_count=1, value_sum=-0.2
            ),
            2: self_play.Node(
                game=FakeGame(), prior=0.8, visit_count=1, value_sum=-0.049519052838329
            ),
        }

        default_search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=3,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="visit_count",
        )
        default_snapshot = default_search._build_root_visit_snapshot(
            root, simulation_index=3
        )

        self.assertEqual(2, default_snapshot["reference_move_by_prior"])
        self.assertEqual(3, default_snapshot["reference_move_rank_by_visits"])
        self.assertEqual(2, default_snapshot["reference_move_rank_by_selection_score"])
        self.assertEqual(0, default_search.select_root_move(root, [0, 1, 2]))

        deterministic_search = self_play.PUCT(
            evaluator=mock.Mock(),
            simulations=3,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="deterministic",
        )
        deterministic_snapshot = deterministic_search._build_root_visit_snapshot(
            root, simulation_index=3
        )

        self.assertEqual(2, deterministic_snapshot["reference_move_by_prior"])
        self.assertEqual(2, deterministic_snapshot["reference_move_rank_by_visits"])
        self.assertEqual(0, deterministic_search.select_root_move(root, [0, 1, 2]))

    def test_run_self_play_worker_sharpened_row_preserves_winner_sign_when_search_disagrees(
        self,
    ):
        default_row, sharpened_row = self._emit_value_target_rows_for_search_value(
            search_value=-0.4, winner=0
        )

        self.assertEqual(1.0, default_row["value"])
        self.assertAlmostEqual(0.64, sharpened_row["value"], places=6)
        self.assertGreater(sharpened_row["value"], 0.0)
        self.assertLess(sharpened_row["value"], default_row["value"])

    def test_run_self_play_worker_sharpened_row_preserves_loser_sign_when_search_disagrees(
        self,
    ):
        default_row, sharpened_row = self._emit_value_target_rows_for_search_value(
            search_value=0.4, winner=1
        )

        self.assertEqual(-1.0, default_row["value"])
        self.assertAlmostEqual(-0.64, sharpened_row["value"], places=6)
        self.assertLess(sharpened_row["value"], 0.0)
        self.assertGreater(sharpened_row["value"], default_row["value"])

    def test_run_self_play_worker_phase_aware_rows_preserve_move_index_and_mode(self):
        rows = self._emit_rows_for_value_target_mode(
            search_values=[0.4, 0.4],
            winner=0,
            value_target_mode="phase_aware_sharpened",
        )

        self.assertEqual([0, 1], [row["move_index"] for row in rows])
        self.assertTrue(
            all(row["value_target_mode"] == "phase_aware_sharpened" for row in rows)
        )

    def test_run_self_play_worker_hybrid_rows_preserve_move_index_and_mode(self):
        rows = self._emit_rows_for_value_target_mode(
            search_values=[0.4, 0.4],
            winner=0,
            value_target_mode="hybrid",
        )

        self.assertEqual([0, 1], [row["move_index"] for row in rows])
        self.assertTrue(all(row["value_target_mode"] == "hybrid" for row in rows))

    def test_run_self_play_worker_phase_aware_rows_preserve_winner_sign_with_bucket_strength(
        self,
    ):
        rows = self._emit_rows_for_value_target_mode(
            search_values=[-0.4],
            winner=0,
            value_target_mode="phase_aware_sharpened",
        )

        self.assertEqual(0, rows[0]["winner"])
        self.assertGreater(rows[0]["value"], 0.0)
        self.assertLess(rows[0]["value"], 1.0)

    def test_run_self_play_worker_hybrid_rows_stay_bounded_and_deterministic_when_signals_disagree(
        self,
    ):
        first_rows = self._emit_rows_for_value_target_mode(
            search_values=[-0.8, -0.8, -0.8],
            winner=0,
            value_target_mode="hybrid",
        )
        second_rows = self._emit_rows_for_value_target_mode(
            search_values=[-0.8, -0.8, -0.8],
            winner=0,
            value_target_mode="hybrid",
        )

        self.assertEqual(
            [row["value"] for row in first_rows], [row["value"] for row in second_rows]
        )
        self.assertTrue(all(-1.0 <= row["value"] <= 1.0 for row in first_rows))
        self.assertGreater(first_rows[0]["value"], 0.0)
        self.assertLess(first_rows[0]["value"], 1.0)

    def _emit_value_target_rows_for_search_value(
        self, *, search_value: float, winner: int
    ):
        default_row = self._emit_rows_for_value_target_mode(
            search_values=[search_value],
            winner=winner,
            value_target_mode="default",
        )[0]
        sharpened_row = self._emit_rows_for_value_target_mode(
            search_values=[search_value],
            winner=winner,
            value_target_mode="sharpened",
        )[0]

        return default_row, sharpened_row

    def _emit_rows_for_value_target_mode(
        self, *, search_values: list[float], winner: int, value_target_mode: str
    ):
        run_state = {"run_count": 0}

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = winner

            def over(self):
                return self.moves_played >= len(search_values)

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = self.moves_played % 2
                return absolute_move == 0

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }

        class FakeRoot:
            def __init__(self, q_value):
                self.q_value = q_value

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng, root, search_options

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                visits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                root = FakeRoot(q_value=search_values[run_state["run_count"]])
                run_state["run_count"] += 1
                return visits, root

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-value-row-") as tmp:
            out_path = Path(tmp) / f"{value_target_mode}.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=len(search_values) + 1,
                        shard_path=str(out_path),
                        value_target_mode=value_target_mode,
                    )

            return [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def test_main_merges_worker_shards_without_read_text(self):
        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text(
                    json.dumps({"worker_id": kwargs["worker_id"]}) + "\n",
                    encoding="utf-8",
                )
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 1,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-merge-") as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "2",
                "--workers",
                "2",
                "--seed",
                "7",
                "--simulations",
                "8",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError("merge should stream shard lines"),
                ):
                    with mock.patch("sys.argv", argv):
                        self_play.main()

            merged_rows = [
                json.loads(line)
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([{"worker_id": 0}, {"worker_id": 1}], merged_rows)

    def test_main_passes_search_options_to_worker_submissions(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-options-") as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
                "--fpu-mode",
                "parent_q",
                "--reuse-subtree",
                "--normalize-values",
                "--root-policy-mode",
                "deterministic",
                "--tactical-root-bias",
                "0.2",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertEqual("parent_q", submitted_kwargs[0]["fpu_mode"])
        self.assertTrue(submitted_kwargs[0]["reuse_subtree"])
        self.assertTrue(submitted_kwargs[0]["normalize_values"])
        self.assertEqual("deterministic", submitted_kwargs[0]["root_policy_mode"])
        self.assertEqual(0.2, submitted_kwargs[0]["tactical_root_bias"])

    def test_main_passes_exploration_defaults_to_worker_submissions(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-default-options-"
        ) as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertEqual("visit_count", submitted_kwargs[0]["root_policy_mode"])
        self.assertEqual(0.0, submitted_kwargs[0]["tactical_root_bias"])
        self.assertEqual("default", submitted_kwargs[0]["policy_target_mode"])

    def test_main_passes_policy_target_mode_to_worker_submissions(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-policy-target-"
        ) as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
                "--policy-target-mode",
                "sharpened",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertEqual("sharpened", submitted_kwargs[0]["policy_target_mode"])

    def test_main_passes_opening_cache_path_to_worker_submissions(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-cache-main-"
        ) as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "self_play.jsonl"
            opening_state = {
                "current_player": 0,
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
            }
            opening_cache_path = tmp_path / "opening_cache.json"
            opening_cache_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_opening_cache_v1",
                        "opening_gate": {"max_ply": 10, "min_stones_in_pits": 36},
                        "entries": {
                            hashlib.sha256(
                                json.dumps(
                                    opening_state,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=True,
                                ).encode("utf-8")
                            ).hexdigest(): {
                                "state": opening_state,
                                "side_to_move": 0,
                                "selected_move": 0,
                                "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                "value": 0.25,
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
                                    "hit_count_in_sources": 1,
                                    "example_origins": ["test"],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
                "--opening-cache",
                str(opening_cache_path),
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertEqual(
            str(opening_cache_path), submitted_kwargs[0]["opening_cache_path"]
        )
        self.assertNotIn("opening_cache", submitted_kwargs[0])

    def test_main_passes_value_target_mode_to_worker_submissions(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-value-target-"
        ) as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
                "--value-target-mode",
                "sharpened",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertEqual("sharpened", submitted_kwargs[0]["value_target_mode"])

    def test_main_only_passes_value_trust_schedule_when_configured(self):
        submitted_kwargs = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                del exc_type, exc, tb
                return False

            def submit(self, fn, **kwargs):
                del fn
                submitted_kwargs.append(kwargs)
                shard_path = Path(kwargs["shard_path"])
                shard_path.write_text("", encoding="utf-8")
                return FakeFuture(
                    {
                        "worker_id": kwargs["worker_id"],
                        "rows_written": 0,
                        "shard_path": str(shard_path),
                    }
                )

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-value-trust-main-"
        ) as tmp:
            out_path = Path(tmp) / "self_play.jsonl"
            argv = [
                "self_play.py",
                "--out",
                str(out_path),
                "--games",
                "1",
                "--workers",
                "1",
                "--seed",
                "7",
                "--simulations",
                "8",
            ]

            with mock.patch.object(
                self_play.concurrent.futures, "ProcessPoolExecutor", FakeExecutor
            ):
                with mock.patch("sys.argv", argv):
                    self_play.main()

        self.assertEqual(1, len(submitted_kwargs))
        self.assertNotIn("value_trust_schedule", submitted_kwargs[0])


class ClassicMCTSHelpersInSelfPlayTest(unittest.TestCase):
    def test_visits_from_classic_mcts_root_extracts_child_visit_counts(self):
        """visits_from_classic_mcts_root should return a list[float] of length 6."""
        from ml.alphazero_lite.self_play import visits_from_classic_mcts_root
        from unittest.mock import MagicMock

        child0 = MagicMock()
        child0.visits = 50
        child2 = MagicMock()
        child2.visits = 30
        root = MagicMock()
        root.children = {0: child0, 2: child2}
        result = visits_from_classic_mcts_root(root)
        self.assertEqual([50.0, 0.0, 30.0, 0.0, 0.0, 0.0], result)

    def test_value_from_classic_mcts_root_converts_win_rate_to_pm1(self):
        """value_from_classic_mcts_root should map [0,1] win rate to [-1,1]."""
        from ml.alphazero_lite.self_play import value_from_classic_mcts_root
        from unittest.mock import MagicMock

        root = MagicMock()
        root.visits = 100
        root.wins = 75.0
        result = value_from_classic_mcts_root(root)
        self.assertAlmostEqual(0.5, result, places=6)

    def test_value_from_classic_mcts_root_returns_zero_for_unvisited(self):
        from ml.alphazero_lite.self_play import value_from_classic_mcts_root
        from unittest.mock import MagicMock

        root = MagicMock()
        root.visits = 0
        self.assertEqual(0.0, value_from_classic_mcts_root(root))


class ClassicMCTSSelfPlayWorkerTest(unittest.TestCase):
    def test_run_self_play_worker_passes_value_trust_schedule_to_puct(self):
        captured_search_options = []

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 0

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRoot:
            q_value = 0.4

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            def __init__(
                self,
                *,
                evaluator,
                simulations,
                c_puct,
                rng,
                root=None,
                **search_options,
            ):
                del evaluator, simulations, c_puct, rng, root
                captured_search_options.append(dict(search_options))

            def run(self, game, *, dirichlet_alpha=None, dirichlet_epsilon=0.25):
                del game, dirichlet_alpha, dirichlet_epsilon
                visits = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                return visits, FakeRoot()

        value_trust_schedule = {
            "enabled": True,
            "opening": 0.8,
            "midgame": 1.0,
            "late": 1.15,
        }

        with tempfile.TemporaryDirectory(prefix="azlite-self-play-value-trust-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "PUCT", FakePUCT):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        value_trust_schedule=value_trust_schedule,
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(
            value_trust_schedule, captured_search_options[0]["value_trust_schedule"]
        )
        self.assertNotIn("teacher_root_summary", rows[0])

    def test_classic_mcts_rows_include_root_summary_value_trust_metadata(self):
        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 0

            def clone(self):
                cloned = FakeGame()
                cloned.moves_played = self.moves_played
                cloned.current_player = self.current_player
                cloned.winner = self.winner
                return cloned

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeClassicRoot:
            visits = 10
            wins = 7.0
            children = {}

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed, value_trust_schedule=None):
                del game, simulations, seed, value_trust_schedule

            def search_root(self):
                return FakeClassicRoot()

            def root_summary(self):
                return {
                    "selected_move": 0,
                    "child_stats": [{"move": 0, "visits": 10, "win_rate": 0.7}],
                    "value_trust": {
                        "enabled": True,
                        "phase_bucket": "opening",
                        "effective_multiplier": 0.8,
                        "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                    },
                }

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-classic-summary-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "ClassicMCTS", FakeClassicMCTS):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        player_mode="classic_mcts",
                        value_trust_schedule={
                            "enabled": True,
                            "opening": 0.8,
                            "midgame": 1.0,
                            "late": 1.15,
                        },
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(
            {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
            rows[0]["teacher_root_summary"]["value_trust"],
        )

    def test_classic_mcts_rows_omit_root_summary_by_default(self):
        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.winner = 0

            def over(self):
                return self.moves_played >= 1

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                return absolute_move == 0

            def clone(self):
                cloned = FakeGame()
                cloned.moves_played = self.moves_played
                cloned.current_player = self.current_player
                cloned.winner = self.winner
                return cloned

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeClassicRoot:
            visits = 10
            wins = 7.0
            children = {}

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed, value_trust_schedule=None):
                del game, simulations, seed, value_trust_schedule

            def search_root(self):
                return FakeClassicRoot()

            def root_summary(self):
                return {
                    "selected_move": 0,
                    "child_stats": [{"move": 0, "visits": 10, "win_rate": 0.7}],
                    "value_trust": {
                        "enabled": True,
                        "phase_bucket": "opening",
                        "effective_multiplier": 0.8,
                        "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                    },
                }

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-classic-default-summary-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play.KalahGame, "from_state", return_value=FakeGame()
            ):
                with mock.patch.object(self_play, "ClassicMCTS", FakeClassicMCTS):
                    result = self_play.run_self_play_worker(
                        worker_id=0,
                        start_index=0,
                        games=1,
                        seed=42,
                        seed_pool=[42],
                        checkpoint=None,
                        input_encoding="kalah_v1",
                        simulations=8,
                        c_puct=1.25,
                        temperature_threshold=10,
                        temperature=0.0,
                        temperature_late=0.0,
                        dirichlet_alpha=0.3,
                        dirichlet_epsilon=0.25,
                        max_moves=2,
                        shard_path=str(shard_path),
                        player_mode="classic_mcts",
                    )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertNotIn("teacher_root_summary", rows[0])

    def test_puct_rows_include_real_teacher_root_summary_value_trust_metadata(self):
        class TinyDeterministicEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-puct-summary-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play, "HeuristicEvaluator", TinyDeterministicEvaluator
            ):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    seed_pool=[42],
                    checkpoint=None,
                    input_encoding="kalah_v1",
                    simulations=4,
                    c_puct=1.25,
                    temperature_threshold=0,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=1,
                    shard_path=str(shard_path),
                    player_mode="puct",
                    value_trust_schedule={
                        "enabled": True,
                        "opening": 0.8,
                        "midgame": 1.0,
                        "late": 1.15,
                    },
                )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(
            {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
            rows[0]["teacher_root_summary"]["value_trust"],
        )
        self.assertEqual(0, rows[0]["teacher_root_summary"]["selected_move"])
        self.assertEqual(0, rows[0]["teacher_root_summary"]["child_stats"][0]["move"])

    def test_puct_rows_omit_value_trust_metadata_by_default(self):
        class TinyDeterministicEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-puct-default-summary-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play, "HeuristicEvaluator", TinyDeterministicEvaluator
            ):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    seed_pool=[42],
                    checkpoint=None,
                    input_encoding="kalah_v1",
                    simulations=4,
                    c_puct=1.25,
                    temperature_threshold=0,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=1,
                    shard_path=str(shard_path),
                    player_mode="puct",
                )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertNotIn("teacher_root_summary", rows[0])

    def test_puct_rows_record_opening_min_simulations_when_applied(self):
        class TinyDeterministicEvaluator(self_play.Evaluator):
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        with tempfile.TemporaryDirectory(
            prefix="azlite-self-play-opening-min-sims-"
        ) as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            with mock.patch.object(
                self_play, "HeuristicEvaluator", TinyDeterministicEvaluator
            ):
                result = self_play.run_self_play_worker(
                    worker_id=0,
                    start_index=0,
                    games=1,
                    seed=42,
                    seed_pool=[42],
                    checkpoint=None,
                    input_encoding="kalah_v1",
                    simulations=4,
                    opening_min_simulations=9,
                    opening_min_simulations_plies=1,
                    c_puct=1.25,
                    temperature_threshold=0,
                    temperature=0.0,
                    temperature_late=0.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    max_moves=1,
                    shard_path=str(shard_path),
                    player_mode="puct",
                )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(1, result["rows_written"])
        self.assertEqual(9, rows[0]["simulations"])
        self.assertEqual("9", rows[0]["search_profile"]["opening_min_simulations"])
        self.assertEqual(
            "1", rows[0]["search_profile"]["opening_min_simulations_plies"]
        )

    def test_classic_mcts_player_mode_ignores_opponent_pool_config(self):
        with tempfile.TemporaryDirectory(prefix="azlite-self-play-worker-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"
            pool_path = Path(tmp) / "pool.json"
            pool_path.write_text(json.dumps({"invalid": True}), encoding="utf-8")

            result = self_play.run_self_play_worker(
                worker_id=0,
                start_index=0,
                games=1,
                seed=7,
                seed_pool=[7],
                checkpoint=None,
                input_encoding="kalah_v3",
                simulations=30,
                c_puct=1.25,
                temperature_threshold=5,
                temperature=1.0,
                temperature_late=0.1,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                max_moves=100,
                shard_path=str(shard_path),
                player_mode="classic_mcts",
                opponent_pool_config=str(pool_path),
                policy_target_mode="default",
                value_target_mode="default",
            )

            self.assertGreater(result["rows_written"], 0)

    def test_classic_mcts_player_mode_produces_valid_rows(self):
        """run_self_play_worker with player_mode=classic_mcts should write valid JSONL rows."""
        import tempfile
        import json
        from ml.alphazero_lite.self_play import run_self_play_worker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            shard_path = f.name
        result = run_self_play_worker(
            worker_id=0,
            start_index=0,
            games=2,
            seed=42,
            seed_pool=[42],
            checkpoint=None,
            input_encoding="kalah_v3",
            simulations=50,
            c_puct=1.25,
            temperature_threshold=5,
            temperature=1.0,
            temperature_late=0.1,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            max_moves=200,
            shard_path=shard_path,
            player_mode="classic_mcts",
            policy_target_mode="sharpened",
            value_target_mode="sharpened",
        )
        self.assertGreater(result["rows_written"], 0)
        import pathlib

        rows = [
            json.loads(line)
            for line in pathlib.Path(shard_path).read_text().splitlines()
            if line.strip()
        ]
        for row in rows:
            self.assertEqual(6, len(row["policy"]))
            self.assertAlmostEqual(1.0, sum(row["policy"]), places=5)
            self.assertGreaterEqual(row["value"], -1.0)
            self.assertLessEqual(row["value"], 1.0)

    def test_classic_mcts_player_mode_ignores_checkpoint(self):
        """classic_mcts player mode should produce output even when checkpoint is None."""
        import tempfile
        from ml.alphazero_lite.self_play import run_self_play_worker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            shard_path = f.name
        # Should not raise even though checkpoint=None
        result = run_self_play_worker(
            worker_id=0,
            start_index=0,
            games=1,
            seed=7,
            seed_pool=[7],
            checkpoint=None,
            input_encoding="kalah_v3",
            simulations=30,
            c_puct=1.25,
            temperature_threshold=5,
            temperature=1.0,
            temperature_late=0.1,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            max_moves=200,
            shard_path=shard_path,
            player_mode="classic_mcts",
            policy_target_mode="default",
            value_target_mode="default",
        )
        self.assertGreater(result["rows_written"], 0)

    def test_puct_player_mode_still_works_as_default(self):
        """run_self_play_worker without player_mode (default puct) should still work."""
        import tempfile
        from ml.alphazero_lite.self_play import run_self_play_worker

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            shard_path = f.name
        result = run_self_play_worker(
            worker_id=0,
            start_index=0,
            games=1,
            seed=1,
            seed_pool=[1],
            checkpoint=None,
            input_encoding="kalah_v3",
            simulations=10,
            c_puct=1.25,
            temperature_threshold=5,
            temperature=1.0,
            temperature_late=0.1,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            max_moves=200,
            shard_path=shard_path,
            # player_mode not passed → default "puct"
            policy_target_mode="default",
            value_target_mode="default",
        )
        self.assertGreater(result["rows_written"], 0)


class CanonicalValueTargetZeroSearchValueTest(unittest.TestCase):
    """Regression test: zero search_value with sharpened mode must not produce zero when outcome is non-zero."""

    def test_zero_search_value_sharpened_returns_outcome_for_winner(self):
        """When search_value=0.0 (e.g. classic MCTS at 50/50), sharpened mode should return outcome_value directly."""
        from ml.alphazero_lite.self_play import canonical_value_target

        # Player wins (outcome=+1), but MCTS had 50/50 → search_value=0.0
        result = canonical_value_target(
            outcome_value=1.0, search_value=0.0, mode="sharpened"
        )
        self.assertEqual(1.0, result)

    def test_zero_search_value_sharpened_returns_outcome_for_loser(self):
        from ml.alphazero_lite.self_play import canonical_value_target

        result = canonical_value_target(
            outcome_value=-1.0, search_value=0.0, mode="sharpened"
        )
        self.assertEqual(-1.0, result)

    def test_negative_zero_search_value_sharpened_returns_outcome(self):
        from ml.alphazero_lite.self_play import canonical_value_target

        # -0.0 is also a zero — should still fall back to outcome
        result = canonical_value_target(
            outcome_value=1.0, search_value=-0.0, mode="sharpened"
        )
        self.assertEqual(1.0, result)

    def test_nonzero_search_value_still_sharpens(self):
        from ml.alphazero_lite.self_play import canonical_value_target

        # Non-zero search value: sharpening should still happen (result != raw outcome)
        result_sharpened = canonical_value_target(
            outcome_value=1.0, search_value=0.5, mode="sharpened"
        )
        result_default = canonical_value_target(
            outcome_value=1.0, search_value=0.5, mode="default"
        )
        # Both should be positive, but sharpened != default
        self.assertGreater(result_sharpened, 0.0)
        self.assertGreater(result_default, 0.0)

    def test_draw_outcome_with_zero_search_value_returns_zero(self):
        from ml.alphazero_lite.self_play import canonical_value_target

        # Draw (outcome=0.0) with zero search_value → should still be 0.0
        result = canonical_value_target(
            outcome_value=0.0, search_value=0.0, mode="sharpened"
        )
        self.assertEqual(0.0, result)


class StartStateModeTest(unittest.TestCase):
    def test_random_symmetric_start_state_includes_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-start-state-random-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            result = self_play.run_self_play_worker(
                worker_id=0,
                start_index=0,
                games=1,
                seed=7,
                seed_pool=[7],
                checkpoint=None,
                input_encoding="kalah_v3",
                simulations=10,
                c_puct=1.25,
                temperature_threshold=5,
                temperature=1.0,
                temperature_late=0.1,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                max_moves=20,
                shard_path=str(shard_path),
                start_state_mode="random_symmetric_total24",
            )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertGreater(result["rows_written"], 0)
        self.assertTrue(rows)
        self.assertEqual("random_symmetric_total24", rows[0]["start_state_mode"])
        self.assertIn("start_state_hash", rows[0])
        self.assertEqual(sum(rows[0]["start_distribution"]), 24)
        self.assertIn(rows[0]["start_player"], (0, 1))

    def test_preset_pool_start_state_includes_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-start-state-preset-") as tmp:
            tmp_path = Path(tmp)
            shard_path = tmp_path / "worker.jsonl"
            pool_path = tmp_path / "pool.jsonl"
            pool_path.write_text(
                json.dumps(
                    {
                        "player_pits": [0, 4, 4, 4, 4, 8],
                        "opponent_pits": [0, 4, 4, 4, 4, 8],
                        "player_store": 0,
                        "opponent_store": 0,
                        "current_player": 1,
                        "metadata": {
                            "preset_id": "balanced-001",
                            "bucket": "near_zero",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = self_play.run_self_play_worker(
                worker_id=0,
                start_index=0,
                games=1,
                seed=7,
                seed_pool=[7],
                checkpoint=None,
                input_encoding="kalah_v3",
                simulations=10,
                c_puct=1.25,
                temperature_threshold=5,
                temperature=1.0,
                temperature_late=0.1,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                max_moves=20,
                shard_path=str(shard_path),
                start_state_mode="preset_pool",
                start_state_pool_path=str(pool_path),
            )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertGreater(result["rows_written"], 0)
        self.assertTrue(rows)
        self.assertEqual("preset_pool", rows[0]["start_state_mode"])
        self.assertEqual(1, rows[0]["start_player"])
        self.assertEqual("balanced-001", rows[0]["start_state_metadata"]["preset_id"])

    def test_standard_start_state_omits_start_state_metadata(self):
        with tempfile.TemporaryDirectory(prefix="azlite-start-state-standard-") as tmp:
            shard_path = Path(tmp) / "worker.jsonl"

            result = self_play.run_self_play_worker(
                worker_id=0,
                start_index=0,
                games=1,
                seed=7,
                seed_pool=[7],
                checkpoint=None,
                input_encoding="kalah_v3",
                simulations=10,
                c_puct=1.25,
                temperature_threshold=5,
                temperature=1.0,
                temperature_late=0.1,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                max_moves=20,
                shard_path=str(shard_path),
            )

            rows = [
                json.loads(line)
                for line in shard_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertGreater(result["rows_written"], 0)
        self.assertTrue(rows)
        self.assertNotIn("start_state_mode", rows[0])
        self.assertNotIn("start_state_hash", rows[0])


if __name__ == "__main__":
    unittest.main()
