import unittest

from ml.alphazero_lite.endgame_tablebase import EndgameTablebase
from ml.alphazero_lite.kalah_rules import KalahGame


class SpyTablebase:
    def __init__(self, *, cached_value=None, lookup_value=None):
        self.cached_value = cached_value
        self.lookup_value = lookup_value
        self.lookup_calls = 0
        self.lookup_cached_calls = 0

    def lookup_cached(self, game, perspective_player):
        del game, perspective_player
        self.lookup_cached_calls += 1
        return self.cached_value

    def lookup(self, game, perspective_player):
        del game, perspective_player
        self.lookup_calls += 1
        return self.lookup_value


class ClassicMCTSTest(unittest.TestCase):
    def _guided_action_after_one_simulation(self, search, *, root_visits, child_stats):
        from ml.alphazero_lite.classic_mcts import Node

        root = Node(search.game.clone(), visits=root_visits)
        root.children = {
            action: Node(search.game.clone(), root, visits=visits, wins=wins)
            for action, visits, wins in child_stats
        }
        before_visits = {
            action: child.visits for action, child in root.children.items()
        }
        search.simulate_playout = lambda game: 0.5

        search.run_search(root, 1, allow_early_stop=False)

        selected_actions = [
            action
            for action, child in root.children.items()
            if child.visits == before_visits[action] + 1
        ]
        self.assertEqual(1, len(selected_actions))
        return selected_actions[0], {
            action: (child.visits, child.wins)
            for action, child in root.children.items()
        }

    def test_root_summary_selected_move_stays_aligned_with_choose_move_on_same_instance(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=8, seed=11)
        call_count = 0

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            nonlocal call_count
            del simulations_to_run, allow_early_stop
            call_count += 1
            for action, child in node.children.items():
                child.visits = 1
                child.wins = 0.0
            if call_count == 1:
                node.children[0].wins = 1.0
            else:
                node.children[5].wins = 1.0
            return 1

        search.run_search = fake_run_search

        selected_move = search.choose_move()
        summary = search.root_summary()

        self.assertEqual(selected_move, summary["selected_move"])
        self.assertEqual(1, call_count)

    def test_root_summary_exposes_child_visit_and_win_rate(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=32, seed=11)

        summary = search.root_summary()

        self.assertIn("selected_move", summary)
        self.assertTrue(summary["child_stats"])
        self.assertIn("visits", summary["child_stats"][0])
        self.assertIn("win_rate", summary["child_stats"][0])

    def test_root_summary_reports_fixed_budget_metadata_when_dynamic_budget_disabled(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)

        summary = search.root_summary()

        self.assertIn("budget", summary)
        self.assertEqual(False, summary["budget"]["dynamic_budget_enabled"])
        self.assertEqual(16, summary["budget"]["baseline_simulations"])
        self.assertEqual(16, summary["budget"]["final_simulations"])
        self.assertEqual("fixed_budget", summary["budget"]["trigger"])

    def test_constructor_rejects_unknown_value_trust_schedule_phase_key(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "value_trust_schedule keys must be enabled, opening, midgame, and late",
        ):
            MCTS(
                game,
                simulations=16,
                seed=7,
                value_trust_schedule={
                    "enabled": True,
                    "opening": 0.8,
                    "midgame": 1.0,
                    "endgame": 1.15,
                },
            )

    def test_constructor_rejects_non_boolean_value_trust_enabled_flag(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        with self.assertRaisesRegex(
            ValueError, "value_trust_schedule enabled must be a boolean"
        ):
            MCTS(
                game,
                simulations=16,
                seed=7,
                value_trust_schedule={
                    "enabled": "false",
                    "opening": 0.8,
                    "midgame": 1.0,
                    "late": 1.15,
                },
            )

    def test_constructor_rejects_non_positive_value_trust_multiplier(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        with self.assertRaisesRegex(
            ValueError, "value_trust_schedule opening must be > 0"
        ):
            MCTS(
                game,
                simulations=16,
                seed=7,
                value_trust_schedule={
                    "enabled": True,
                    "opening": 0.0,
                    "midgame": 1.0,
                    "late": 1.15,
                },
            )

    def test_root_summary_reports_disabled_value_trust_schedule_metadata_by_default(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)

        summary = search.root_summary()

        self.assertIn("value_trust", summary)
        self.assertEqual(False, summary["value_trust"]["enabled"])
        self.assertEqual(
            {"opening": 1.0, "midgame": 1.0, "late": 1.0},
            summary["value_trust"]["schedule"],
        )
        self.assertEqual("opening", summary["value_trust"]["phase_bucket"])
        self.assertEqual(1.0, summary["value_trust"]["effective_multiplier"])

    def test_root_summary_reports_custom_value_trust_schedule_metadata_for_late_phase(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 1, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": True,
                "opening": 0.8,
                "midgame": 1.0,
                "late": 1.15,
            },
        )

        summary = search.root_summary()

        self.assertEqual(True, summary["value_trust"]["enabled"])
        self.assertEqual("late", summary["value_trust"]["phase_bucket"])
        self.assertEqual(1.15, summary["value_trust"]["effective_multiplier"])
        self.assertEqual(
            {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            summary["value_trust"]["schedule"],
        )

    def test_late_phase_value_trust_schedule_increases_value_term_contribution_in_search_guidance(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 1, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        baseline = MCTS(game, simulations=16, seed=7)
        boosted = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": True,
                "opening": 1.0,
                "midgame": 1.0,
                "late": 1.5,
            },
        )
        child_stats = [
            (0, 50, 40.0),
            (1, 20, 11.6),
        ]

        baseline_action, _ = self._guided_action_after_one_simulation(
            baseline,
            root_visits=70,
            child_stats=child_stats,
        )
        boosted_action, _ = self._guided_action_after_one_simulation(
            boosted,
            root_visits=70,
            child_stats=child_stats,
        )

        self.assertEqual(1, baseline_action)
        self.assertEqual(0, boosted_action)

    def test_opening_phase_value_trust_schedule_reduces_value_term_contribution_in_search_guidance(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        baseline = MCTS(game, simulations=16, seed=7)
        dampened = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": True,
                "opening": 0.5,
                "midgame": 1.0,
                "late": 1.0,
            },
        )
        child_stats = [
            (0, 50, 40.0),
            (1, 20, 9.0),
        ]

        baseline_action, _ = self._guided_action_after_one_simulation(
            baseline,
            root_visits=70,
            child_stats=child_stats,
        )
        dampened_action, _ = self._guided_action_after_one_simulation(
            dampened,
            root_visits=70,
            child_stats=child_stats,
        )

        self.assertEqual(0, baseline_action)
        self.assertEqual(1, dampened_action)

    def test_midgame_value_trust_schedule_uses_midgame_multiplier_in_search_guidance(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [2, 2, 2, 2, 2, 2],
                "opponent_pits": [2, 2, 2, 2, 2, 2],
                "player_store": 12,
                "opponent_store": 12,
                "current_player": 0,
            }
        )
        baseline = MCTS(game, simulations=16, seed=7)
        boosted = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": True,
                "opening": 0.5,
                "midgame": 1.5,
                "late": 0.5,
            },
        )
        child_stats = [
            (0, 50, 40.0),
            (1, 20, 11.6),
        ]

        baseline_action, _ = self._guided_action_after_one_simulation(
            baseline,
            root_visits=70,
            child_stats=child_stats,
        )
        boosted_action, _ = self._guided_action_after_one_simulation(
            boosted,
            root_visits=70,
            child_stats=child_stats,
        )

        self.assertEqual(1, baseline_action)
        self.assertEqual(0, boosted_action)

    def test_disabled_value_trust_schedule_preserves_existing_search_guidance_behavior(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 1, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        baseline = MCTS(game, simulations=16, seed=7)
        disabled = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": False,
                "opening": 0.5,
                "midgame": 1.0,
                "late": 1.5,
            },
        )
        child_stats = [
            (0, 50, 40.0),
            (1, 20, 11.6),
        ]

        baseline_action, baseline_stats = self._guided_action_after_one_simulation(
            baseline,
            root_visits=70,
            child_stats=child_stats,
        )
        disabled_action, disabled_stats = self._guided_action_after_one_simulation(
            disabled,
            root_visits=70,
            child_stats=child_stats,
        )

        self.assertEqual(baseline_action, disabled_action)
        self.assertEqual(baseline_stats, disabled_stats)

    def test_root_summary_reports_unit_effective_multiplier_when_schedule_disabled(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 1],
                "opponent_pits": [2, 2, 2, 2, 2, 2],
                "player_store": 11,
                "opponent_store": 11,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            value_trust_schedule={
                "enabled": False,
                "opening": 0.5,
                "midgame": 1.7,
                "late": 2.0,
            },
        )

        summary = search.root_summary()

        self.assertEqual(False, summary["value_trust"]["enabled"])
        self.assertEqual("midgame", summary["value_trust"]["phase_bucket"])
        self.assertEqual(1.0, summary["value_trust"]["effective_multiplier"])
        self.assertEqual(
            {"opening": 0.5, "midgame": 1.7, "late": 2.0},
            summary["value_trust"]["schedule"],
        )

    def test_constructor_rejects_invalid_dynamic_budget_range(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        with self.assertRaises(ValueError):
            MCTS(
                game,
                simulations=16,
                seed=7,
                dynamic_budget_enabled=True,
                dynamic_budget_probe_simulations=8,
                dynamic_budget_min_simulations=20,
                dynamic_budget_max_simulations=12,
            )

    def test_constructor_stores_valid_dynamic_budget_config_and_initial_summary_metadata(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=12,
            dynamic_budget_max_simulations=24,
        )

        summary = search.root_summary()

        self.assertEqual(True, search.dynamic_budget_config.enabled)
        self.assertEqual(8, search.dynamic_budget_config.probe_simulations)
        self.assertEqual(12, search.dynamic_budget_config.min_simulations)
        self.assertEqual(24, search.dynamic_budget_config.max_simulations)
        self.assertEqual(True, summary["budget"]["dynamic_budget_enabled"])
        self.assertEqual(8, summary["budget"]["probe_simulations"])
        self.assertGreaterEqual(summary["budget"]["final_simulations"], 12)
        self.assertLessEqual(summary["budget"]["final_simulations"], 24)
        self.assertNotEqual("fixed_budget", summary["budget"]["trigger"])

    def test_root_summary_reports_zero_final_simulations_when_root_search_does_not_run(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)

        summary = search.root_summary()

        self.assertEqual(0, summary["budget"]["final_simulations"])

    def test_dynamic_root_summary_refreshes_no_move_root_metadata(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=12,
            dynamic_budget_max_simulations=24,
        )

        summary = search.root_summary()

        self.assertIsNone(summary["selected_move"])
        self.assertEqual([], summary["child_stats"])
        self.assertEqual(True, summary["budget"]["dynamic_budget_enabled"])
        self.assertIsNone(summary["budget"]["chosen_simulations"])
        self.assertEqual(0, summary["budget"]["final_simulations"])
        self.assertEqual("late", summary["budget"]["phase_bucket"])
        self.assertEqual(0.0, summary["budget"]["entropy"])
        self.assertEqual(1.0, summary["budget"]["top_move_margin"])
        self.assertEqual(0.0, summary["budget"]["child_value_variance"])
        self.assertEqual("late_low_uncertainty", summary["budget"]["trigger"])

    def test_dynamic_root_summary_reports_null_chosen_simulations_for_single_legal_move(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=12,
            dynamic_budget_max_simulations=24,
        )

        summary = search.root_summary()

        self.assertEqual(4, summary["selected_move"])
        self.assertEqual(1, len(summary["child_stats"]))
        self.assertIsNone(summary["budget"]["chosen_simulations"])
        self.assertEqual(0, summary["budget"]["final_simulations"])

    def test_root_phase_bucket_uses_remaining_seed_ranges(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        early = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        mid = KalahGame.from_state(
            {
                "player_pits": [2, 2, 2, 2, 2, 2],
                "opponent_pits": [2, 2, 2, 2, 2, 2],
                "player_store": 12,
                "opponent_store": 12,
                "current_player": 0,
            }
        )
        late = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [0, 1, 0, 0, 0, 0],
                "player_store": 23,
                "opponent_store": 23,
                "current_player": 0,
            }
        )
        search = MCTS(early, simulations=16, seed=7)

        self.assertEqual("early", search.phase_bucket_for(early))
        self.assertEqual("mid", search.phase_bucket_for(mid))
        self.assertEqual("late", search.phase_bucket_for(late))

    def test_root_phase_bucket_respects_boundary_values(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        late_boundary = KalahGame.from_state(
            {
                "player_pits": [1, 1, 1, 1, 1, 1],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 18,
                "opponent_store": 18,
                "current_player": 0,
            }
        )
        thirteen = KalahGame.from_state(
            {
                "player_pits": [2, 1, 1, 1, 1, 1],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 17,
                "opponent_store": 18,
                "current_player": 0,
            }
        )
        twenty_four = KalahGame.from_state(
            {
                "player_pits": [2, 2, 2, 2, 2, 2],
                "opponent_pits": [2, 2, 2, 2, 2, 2],
                "player_store": 12,
                "opponent_store": 12,
                "current_player": 0,
            }
        )
        twenty_five = KalahGame.from_state(
            {
                "player_pits": [3, 2, 2, 2, 2, 2],
                "opponent_pits": [2, 2, 2, 2, 2, 2],
                "player_store": 11,
                "opponent_store": 12,
                "current_player": 0,
            }
        )
        search = MCTS(late_boundary, simulations=16, seed=7)

        self.assertEqual("late", search.phase_bucket_for(late_boundary))
        self.assertEqual("mid", search.phase_bucket_for(thirteen))
        self.assertEqual("mid", search.phase_bucket_for(twenty_four))
        self.assertEqual("early", search.phase_bucket_for(twenty_five))

    def test_root_signal_helpers_measure_entropy_margin_and_variance(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=8, wins=6.0),
            1: Node(game.clone(), root, visits=4, wins=2.0),
            2: Node(game.clone(), root, visits=4, wins=1.0),
        }

        self.assertAlmostEqual(0.9464, round(search.root_entropy(root), 4))
        self.assertAlmostEqual(0.25, round(search.top_move_margin(root), 4))
        self.assertAlmostEqual(0.0417, round(search.child_value_variance(root), 4))

    def test_root_entropy_includes_unvisited_children_in_support(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=4, wins=3.0),
            1: Node(game.clone(), root, visits=1, wins=1.0),
            2: Node(game.clone(), root, visits=0, wins=0.0),
        }

        self.assertAlmostEqual(0.4555, round(search.root_entropy(root), 4))

    def test_root_entropy_defaults_to_zero_for_zero_visit_root(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=0, wins=0.0),
            1: Node(game.clone(), root, visits=0, wins=0.0),
        }

        self.assertEqual(0.0, search.root_entropy(root))

    def test_sparse_multi_child_root_defaults_top_move_margin_to_zero(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=5, wins=4.0),
            1: Node(game.clone(), root, visits=0, wins=0.0),
            2: Node(game.clone(), root, visits=0, wins=0.0),
        }

        self.assertEqual(0.0, search.top_move_margin(root))

    def test_sparse_multi_child_root_uses_conservative_variance_default(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=5, wins=4.0),
            1: Node(game.clone(), root, visits=0, wins=0.0),
            2: Node(game.clone(), root, visits=0, wins=0.0),
        }

        self.assertEqual(0.25, search.child_value_variance(root))

    def test_search_root_runs_probe_then_full_budget_when_dynamic_budget_enabled(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )
        calls = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            calls.append(simulations_to_run)
            for child in node.children.values():
                child.visits = 4
                child.wins = 2.0
            return simulations_to_run

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        search.search_root()

        self.assertEqual([8, 16], calls)
        self.assertEqual(24, search.root_summary()["budget"]["final_simulations"])

    def test_root_summary_runs_search_for_fresh_dynamic_budget_positions(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )
        calls = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            calls.append(simulations_to_run)
            for action, child in node.children.items():
                child.visits = 0
                child.wins = 0.0
            node.children[3].visits = simulations_to_run
            node.children[3].wins = float(simulations_to_run)
            return simulations_to_run

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        summary = search.root_summary()

        self.assertEqual([8, 16], calls)
        self.assertEqual(3, summary["selected_move"])
        self.assertTrue(summary["child_stats"])
        self.assertEqual(24, summary["budget"]["final_simulations"])

    def test_search_root_dynamic_budget_summary_uses_actual_simulations_run(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )
        calls = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            calls.append(simulations_to_run)
            for child in node.children.values():
                child.visits = 4
                child.wins = 2.0
            if len(calls) == 1:
                return simulations_to_run
            return 5

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        search.search_root()

        self.assertEqual([8, 16], calls)
        self.assertEqual(13, search.root_summary()["budget"]["final_simulations"])

    def test_search_root_reports_chosen_and_executed_budget_separately(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )
        calls = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            calls.append(simulations_to_run)
            for child in node.children.values():
                child.visits = 4
                child.wins = 2.0
            return simulations_to_run if len(calls) == 1 else 5

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        budget = search.root_summary()["budget"]

        self.assertEqual([8, 16], calls)
        self.assertEqual(24, budget["chosen_simulations"])
        self.assertEqual(13, budget["final_simulations"])

    def test_search_root_records_root_latency_in_budget_summary(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            for child in node.children.values():
                child.visits = max(child.visits, 1)
                child.wins = float(child.visits)
            return simulations_to_run

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        summary = search.root_summary()

        self.assertGreater(summary["budget"]["root_latency_ms"], 0.0)

    def test_search_root_dynamic_budget_summary_keeps_probe_state_metadata(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=48,
        )

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            if simulations_to_run == 8:
                node.children[0].visits = 4
                node.children[0].wins = 2.0
                node.children[1].visits = 4
                node.children[1].wins = 2.0
                for action, child in node.children.items():
                    if action not in (0, 1):
                        child.visits = 0
                        child.wins = 0.0
                return 8

            node.children[0].visits = 12
            node.children[0].wins = 12.0
            node.children[1].visits = 1
            node.children[1].wins = 0.0
            for action, child in node.children.items():
                if action not in (0, 1):
                    child.visits = 0
                    child.wins = 0.0
            return simulations_to_run

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 24

        search.search_root()
        budget = search.root_summary()["budget"]

        self.assertAlmostEqual(0.3869, round(budget["entropy"], 4))
        self.assertEqual(0.0, budget["top_move_margin"])
        self.assertEqual(0.0, budget["child_value_variance"])
        self.assertEqual("early_low_margin", budget["trigger"])

    def test_choose_dynamic_budget_never_returns_less_than_probe_budget(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=9,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=4,
            dynamic_budget_max_simulations=40,
        )
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=20, wins=20.0),
            1: Node(game.clone(), root, visits=1, wins=0.8),
        }

        budget = search.choose_dynamic_budget(root)

        self.assertGreaterEqual(budget, 8)

    def test_choose_dynamic_budget_clamps_to_configured_range(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=40,
        )
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=6, wins=3.0),
            1: Node(game.clone(), root, visits=5, wins=2.5),
            2: Node(game.clone(), root, visits=5, wins=2.0),
        }

        budget = search.choose_dynamic_budget(root)

        self.assertGreaterEqual(budget, 16)
        self.assertLessEqual(budget, 40)

    def test_choose_dynamic_budget_uses_configured_signal_weights(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=8, wins=6.0),
            1: Node(game.clone(), root, visits=8, wins=4.0),
            2: Node(game.clone(), root, visits=8, wins=3.0),
        }
        default_search = MCTS(
            game,
            simulations=40,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=8,
            dynamic_budget_max_simulations=96,
        )
        tuned_search = MCTS(
            game,
            simulations=40,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=8,
            dynamic_budget_max_simulations=96,
            dynamic_budget_entropy_weight=0.0,
            dynamic_budget_low_margin_threshold=0.0,
            dynamic_budget_low_margin_weight=0.0,
            dynamic_budget_variance_weight=0.0,
        )

        self.assertEqual(51, default_search.choose_dynamic_budget(root))
        self.assertEqual(34, tuned_search.choose_dynamic_budget(root))

    def test_dynamic_budget_trigger_label_uses_configured_low_margin_threshold(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=32,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=16,
            dynamic_budget_max_simulations=40,
            dynamic_budget_low_margin_threshold=0.15,
        )
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=20, wins=12.0),
            1: Node(game.clone(), root, visits=1, wins=0.48),
            2: Node(game.clone(), root, visits=1, wins=0.1),
        }

        self.assertAlmostEqual(0.12, search.top_move_margin(root), places=2)
        self.assertEqual(
            "early_low_margin_high_variance", search.dynamic_budget_trigger_label(root)
        )

    def test_signal_helpers_default_for_single_child_root(self):
        from ml.alphazero_lite.classic_mcts import MCTS, Node

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=16, seed=7)
        root = Node(game.clone())
        root.children = {
            0: Node(game.clone(), root, visits=5, wins=4.0),
        }

        self.assertEqual(0.0, search.root_entropy(root))
        self.assertEqual(1.0, search.top_move_margin(root))
        self.assertEqual(0.0, search.child_value_variance(root))

    def test_search_root_rebuilds_when_underlying_game_state_changes(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=8, seed=11)
        observed_states = []

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del simulations_to_run, allow_early_stop
            observed_states.append(node.game.to_state())
            for child in node.children.values():
                child.visits = 1
                child.wins = 0.0
            return 1

        search.run_search = fake_run_search

        search.choose_move()
        game.move(game.pit_index(0))
        search.root_summary()

        self.assertEqual(2, len(observed_states))
        self.assertNotEqual(observed_states[0], observed_states[1])

    def test_root_summary_rebuild_resets_dynamic_budget_metadata_for_single_move_root(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(
            game,
            simulations=16,
            seed=7,
            dynamic_budget_enabled=True,
            dynamic_budget_probe_simulations=8,
            dynamic_budget_min_simulations=12,
            dynamic_budget_max_simulations=24,
        )

        def fake_run_search(node, simulations_to_run, *, allow_early_stop):
            del allow_early_stop
            for action, child in node.children.items():
                if action in (0, 1):
                    child.visits = 4
                    child.wins = 2.0
                else:
                    child.visits = 0
                    child.wins = 0.0
            return simulations_to_run

        search.run_search = fake_run_search
        search.choose_dynamic_budget = lambda node: 16

        search.search_root()
        game.player_pits = [0, 0, 0, 0, 1, 0]
        game.opponent_pits = [4, 4, 4, 4, 4, 4]
        game.pits = game.player_pits + game.opponent_pits
        summary = search.root_summary()

        self.assertEqual(0, summary["budget"]["final_simulations"])
        self.assertEqual("early", summary["budget"]["phase_bucket"])
        self.assertEqual(0.0, summary["budget"]["entropy"])
        self.assertEqual(1.0, summary["budget"]["top_move_margin"])
        self.assertEqual(0.0, summary["budget"]["child_value_variance"])
        self.assertEqual("early_low_uncertainty", summary["budget"]["trigger"])

    def test_search_root_refreshes_player_when_underlying_game_turn_changes(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = MCTS(game, simulations=1, seed=11)

        self.assertEqual(0, search.player)
        search.search_root()

        game.move(game.pit_index(4))
        self.assertEqual(1, game.current_player)

        search.search_root()

        self.assertEqual(1, search.player)

    def test_choose_move_returns_none_without_legal_moves(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        self.assertIsNone(MCTS(game, simulations=32, seed=42).choose_move())

    def test_choose_move_is_deterministic_for_same_seed(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        first = MCTS(KalahGame.from_state(state), simulations=64, seed=42).choose_move()
        second = MCTS(
            KalahGame.from_state(state), simulations=64, seed=42
        ).choose_move()

        self.assertEqual(first, second)

    def test_choose_playout_move_prefers_extra_turn(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 1, 0],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        self.assertEqual(
            4, MCTS(game, simulations=8, seed=42).choose_playout_move(game)
        )

    def test_simulate_playout_runtime_mode_skips_exact_solver_for_sparse_nonterminal_state(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(0, tablebase.lookup_calls)

    def test_simulate_playout_uses_exact_solver_when_evaluation_mode_threshold_applies(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )

        self.assertEqual(0.75, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)

    def test_simulate_playout_skips_cached_probe_when_exact_solver_applies(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )

        self.assertEqual(0.75, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)
        self.assertEqual(0, tablebase.lookup_cached_calls)

    def test_simulate_playout_falls_back_to_rollout_when_exact_solver_returns_none(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=None)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=2,
        )
        playout_move_calls = 0

        def fake_choose_playout_move(playout_game):
            nonlocal playout_move_calls
            self.assertIsNot(playout_game, game)
            playout_move_calls += 1
            return 5

        mcts.choose_playout_move = fake_choose_playout_move

        self.assertTrue(mcts.exact_solve_applies(game))
        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(1, tablebase.lookup_calls)
        self.assertEqual(1, playout_move_calls)

    def test_simulate_playout_skips_exact_solver_above_threshold(self):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = SpyTablebase(lookup_value=0.75)

        mcts = MCTS(
            game,
            simulations=4,
            seed=42,
            endgame_tablebase=tablebase,
            exact_solve_enabled=True,
            exact_solve_stone_threshold=1,
        )

        self.assertEqual(0.5, mcts.simulate_playout(game))
        self.assertEqual(0, tablebase.lookup_calls)

    def test_simulate_playout_returns_terminal_tablebase_value_for_no_legal_move_state(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 0],
                "opponent_pits": [1, 1, 1, 1, 1, 1],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )

        tablebase = EndgameTablebase()
        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.0, mcts.simulate_playout(game))

    def test_simulate_playout_uses_recorded_tablebase_hit_when_exact_solver_disabled(
        self,
    ):
        from ml.alphazero_lite.classic_mcts import MCTS

        game = KalahGame.from_state(
            {
                "player_pits": [0, 0, 0, 0, 0, 1],
                "opponent_pits": [1, 0, 0, 0, 0, 0],
                "player_store": 20,
                "opponent_store": 20,
                "current_player": 0,
            }
        )
        tablebase = EndgameTablebase()
        tablebase.record(game, perspective_player=0, value=0.75)

        mcts = MCTS(game, simulations=4, seed=42, endgame_tablebase=tablebase)

        self.assertEqual(0.75, mcts.simulate_playout(game))
