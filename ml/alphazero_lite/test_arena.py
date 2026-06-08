import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np

from ml.alphazero_lite import arena
from ml.alphazero_lite import self_play
from ml.alphazero_lite.kalah_rules import KalahGame


class ArenaAblationEvaluationTest(unittest.TestCase):
    def test_evaluate_artifact_position_passes_normalized_ablation_mode_to_puct(self):
        captured = []

        class FakePUCT:
            def __init__(self, **kwargs):
                captured.append(kwargs["ablation_mode"])

            def run(self, game):
                del game
                return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), None

            def select_root_move(self, root, legal_moves):
                del root
                return legal_moves[0]

        with mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT):
            result = arena.evaluate_artifact_position(
                evaluator=mock.Mock(),
                state={
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                simulations=32,
                seed=42,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
                ablation_mode="value_only",
            )

        self.assertEqual("value_only", captured[0])
        self.assertEqual(0, result["selected_move"])
        self.assertEqual([0, 1, 2, 3, 4, 5], result["legal_moves"])

    def test_evaluate_artifact_position_policy_only_reports_neutral_value(self):
        class FakeEvaluator:
            def evaluate(self, game):
                del game
                return np.full(6, 1.0 / 6.0, dtype=np.float32), 0.25

        summary = arena.evaluate_artifact_position(
            artifact_path="ignored",
            evaluator=FakeEvaluator(),
            state={
                "player_pits": [4, 4, 4, 4, 4, 4],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            },
            simulations=0,
            seed=42,
            c_puct=1.25,
            search_options=arena.build_eval_search_options(),
            ablation_mode="policy_only",
        )

        self.assertEqual(0.0, summary["value"])

    def test_evaluate_artifact_position_puct_preserves_value_trust_root_summary_metadata(
        self,
    ):
        root_value_trust = {
            "enabled": True,
            "phase_bucket": "opening",
            "effective_multiplier": 0.8,
            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
        }

        class FakeEvaluator:
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        class FakeRoot:
            def __init__(self):
                self.children = {}

        class FakePUCT:
            def __init__(self, **kwargs):
                del kwargs

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root, legal_moves
                return 0

            def root_summary(self):
                return {"value_trust": root_value_trust}

        with mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT):
            summary = arena.evaluate_artifact_position(
                evaluator=FakeEvaluator(),
                state={
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                simulations=32,
                seed=42,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(
                    value_trust_schedule={
                        "enabled": True,
                        "opening": 0.8,
                        "midgame": 1.0,
                        "late": 1.15,
                    }
                ),
            )

        self.assertEqual(root_value_trust, summary["value_trust"])

    def test_evaluate_artifact_position_puct_preserves_selection_breakdown_and_visit_snapshots(
        self,
    ):
        expected_selection_breakdown = {
            "selected_move": 0,
            "reference_move": 1,
            "highest_prior_move": 1,
            "reference_move_kind": "highest_prior_move",
            "next_simulation_move_kind": "highest current PUCT selection score using deterministic telemetry ordering",
            "next_simulation_move": 0,
            "policy_top_move": 1,
            "visit_top_move": 0,
            "q_top_move": 0,
            "moves": [
                {
                    "move": 0,
                    "prior": 0.2,
                    "visit_count": 2,
                    "q_value": 0.7,
                    "selection_q_value": 0.7,
                    "u_component": 0.55,
                    "selection_score": 1.25,
                    "used_fpu": False,
                    "fpu_value": None,
                },
                {
                    "move": 1,
                    "prior": 0.6,
                    "visit_count": 1,
                    "q_value": 0.6,
                    "selection_q_value": 0.6,
                    "u_component": 0.4,
                    "selection_score": 1.0,
                    "used_fpu": False,
                    "fpu_value": None,
                },
                {
                    "move": 2,
                    "prior": 0.2,
                    "visit_count": 0,
                    "q_value": 0.0,
                    "selection_q_value": 0.5,
                    "u_component": 0.25,
                    "selection_score": 0.75,
                    "used_fpu": True,
                    "fpu_value": 0.5,
                },
            ],
        }
        expected_visit_snapshots = [
            {
                "simulation": 1,
                "visits": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "moves": [
                    {
                        "move": 0,
                        "prior": 0.2,
                        "visit_count": 1,
                        "q_value": 0.8,
                        "selection_q_value": 0.8,
                        "u_component": 0.125,
                        "selection_score": 0.925,
                        "used_fpu": False,
                        "fpu_value": None,
                    },
                    {
                        "move": 1,
                        "prior": 0.6,
                        "visit_count": 0,
                        "q_value": 0.0,
                        "selection_q_value": 0.5,
                        "u_component": 0.75,
                        "selection_score": 1.25,
                        "used_fpu": True,
                        "fpu_value": 0.5,
                    },
                    {
                        "move": 2,
                        "prior": 0.2,
                        "visit_count": 0,
                        "q_value": 0.0,
                        "selection_q_value": 0.5,
                        "u_component": 0.25,
                        "selection_score": 0.75,
                        "used_fpu": True,
                        "fpu_value": 0.5,
                    },
                ],
                "selected_move": 0,
                "reference_move_by_prior": 1,
                "reference_move_rank_by_visits": 2,
                "reference_move_rank_by_q": 2,
                "reference_move_rank_by_selection_score": 1,
            },
            {
                "simulation": 4,
                "visits": [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                "moves": [
                    {
                        "move": 0,
                        "prior": 0.2,
                        "visit_count": 1,
                        "q_value": 0.8,
                        "selection_q_value": 0.8,
                        "u_component": 0.3125,
                        "selection_score": 1.1125,
                        "used_fpu": False,
                        "fpu_value": None,
                    },
                    {
                        "move": 1,
                        "prior": 0.6,
                        "visit_count": 2,
                        "q_value": 0.5,
                        "selection_q_value": 0.5,
                        "u_component": 0.625,
                        "selection_score": 1.125,
                        "used_fpu": False,
                        "fpu_value": None,
                    },
                    {
                        "move": 2,
                        "prior": 0.2,
                        "visit_count": 1,
                        "q_value": -0.2,
                        "selection_q_value": -0.2,
                        "u_component": 0.3125,
                        "selection_score": 0.1125,
                        "used_fpu": False,
                        "fpu_value": None,
                    },
                ],
                "selected_move": 1,
                "reference_move_by_prior": 1,
                "reference_move_rank_by_visits": 1,
                "reference_move_rank_by_q": 2,
                "reference_move_rank_by_selection_score": 2,
            },
        ]

        class FakeEvaluator:
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        class FakeRoot:
            def __init__(self):
                self.children = {}

        class FakePUCT:
            def __init__(self, **kwargs):
                del kwargs

                self._selection_breakdown = deepcopy(expected_selection_breakdown)
                self._visit_snapshots = deepcopy(expected_visit_snapshots)

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root, legal_moves
                return 0

            def root_summary(self):
                return {
                    "selection_breakdown": self._selection_breakdown,
                    "visit_snapshots": self._visit_snapshots,
                }

        with mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT):
            summary = arena.evaluate_artifact_position(
                evaluator=FakeEvaluator(),
                state={
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                simulations=32,
                seed=42,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
            )

        self.assertEqual(expected_selection_breakdown, summary["selection_breakdown"])
        self.assertEqual(expected_visit_snapshots, summary["visit_snapshots"])

        summary["selection_breakdown"]["moves"][0]["selection_score"] = -1.0
        summary["visit_snapshots"][0]["moves"][0]["selection_score"] = -1.0

        self.assertEqual(
            1.25, expected_selection_breakdown["moves"][0]["selection_score"]
        )
        self.assertEqual(
            0.925, expected_visit_snapshots[0]["moves"][0]["selection_score"]
        )

    def test_evaluate_artifact_position_puct_omits_value_trust_when_root_summary_lacks_it(
        self,
    ):
        class FakeEvaluator:
            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        class FakeRoot:
            def __init__(self):
                self.children = {}

        class FakePUCT:
            def __init__(self, **kwargs):
                del kwargs

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root, legal_moves
                return 0

            def root_summary(self):
                return {}

        with mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT):
            summary = arena.evaluate_artifact_position(
                evaluator=FakeEvaluator(),
                state={
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                simulations=32,
                seed=42,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
            )

        self.assertNotIn("value_trust", summary)

    def test_evaluate_artifact_position_classic_only_uses_classic_mcts(self):
        class FakeChild:
            def __init__(self, visits, wins):
                self.visits = visits
                self.wins = wins

        class FakeRoot:
            def __init__(self):
                self.visits = 10
                self.wins = 7.5
                self.children = {
                    1: FakeChild(visits=2, wins=1.0),
                    4: FakeChild(visits=8, wins=7.0),
                }

        captured = []

        class FakeClassicMCTS:
            def __init__(self, game, *, simulations, seed):
                del game
                captured.append((simulations, seed))

            def search_root(self):
                return FakeRoot()

            def root_summary(self):
                return {
                    "selected_move": 4,
                    "child_stats": [
                        {"move": 1, "visits": 2, "win_rate": 0.5},
                        {"move": 4, "visits": 8, "win_rate": 0.875},
                    ],
                }

        with (
            mock.patch(
                "ml.alphazero_lite.arena.PUCT",
                side_effect=AssertionError("classic_only should not use PUCT"),
            ),
            mock.patch(
                "ml.alphazero_lite.arena.ClassicMCTS", FakeClassicMCTS, create=True
            ),
        ):
            summary = arena.evaluate_artifact_position(
                artifact_path="ignored",
                evaluator=mock.Mock(),
                state={
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
                simulations=64,
                seed=17,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
                ablation_mode="classic_only",
            )

        self.assertEqual([(64, 17)], captured)
        self.assertEqual(4, summary["selected_move"])
        self.assertAlmostEqual(0.5, summary["value"])


class ArenaScriptTest(unittest.TestCase):
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

    def write_tactical_bias_artifact(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix="azlite-tactical-artifact-")
        self.addCleanup(temp_dir.cleanup)
        artifact_dir = Path(temp_dir.name)
        (artifact_dir / "weights.json").write_text(
            json.dumps(
                {
                    "w1": [[0.1], [0.0]] + [[0.0]] * 13,
                    "b1": [0.0],
                    "w2": [[1.0]],
                    "b2": [0.0],
                    "w_policy": [[0.2, -10.0, -10.0, -10.0, -10.0, 0.0]],
                    "b_policy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "w_value": [[0.0]],
                    "b_value": [0.0],
                }
            ),
            encoding="utf-8",
        )
        return artifact_dir

    def write_basic_artifact(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory(prefix="azlite-artifact-")
        self.addCleanup(temp_dir.cleanup)
        artifact_dir = Path(temp_dir.name)
        (artifact_dir / "weights.json").write_text(
            json.dumps(
                {
                    "w1": [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]]
                    + [[0.0, 0.0, 0.0, 0.0]] * 13,
                    "b1": [0.0, 0.0, 0.0, 0.0],
                    "w2": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "b2": [0.0, 0.0, 0.0, 0.0],
                    "w_policy": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    ],
                    "b_policy": [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                    "w_value": [[0.25], [0.25], [0.25], [0.25]],
                    "b_value": [0.0],
                }
            ),
            encoding="utf-8",
        )
        return artifact_dir

    def test_evaluate_position_returns_selected_move_priors_and_value(self):
        artifact_dir = self.write_basic_artifact()
        position = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        summary = arena.evaluate_artifact_position(
            artifact_path=artifact_dir,
            state=position,
            simulations=16,
            seed=7,
            c_puct=1.25,
            search_options=arena.build_eval_search_options(),
        )

        self.assertIn("selected_move", summary)
        self.assertIn("policy", summary)
        self.assertIn("value", summary)
        self.assertIn("child_stats", summary)

    def test_evaluate_artifact_position_classic_only_reports_budget_metadata(self):
        from ml.alphazero_lite import arena

        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed):
                del game, simulations, seed

            def search_root(self):
                class FakeChild:
                    visits = 12
                    wins = 9.0

                class FakeRoot:
                    visits = 12
                    wins = 9.0
                    children = {2: FakeChild()}

                return FakeRoot()

            def root_summary(self):
                return {
                    "selected_move": 2,
                    "child_stats": [{"move": 2, "visits": 12, "win_rate": 0.75}],
                    "budget": {
                        "dynamic_budget_enabled": True,
                        "baseline_simulations": 96,
                        "probe_simulations": 24,
                        "final_simulations": 144,
                        "phase_bucket": "mid",
                        "entropy": 0.81,
                        "top_move_margin": 0.04,
                        "child_value_variance": 0.03,
                        "trigger": "mid_high_entropy_low_margin",
                        "root_latency_ms": 7.5,
                    },
                }

        with unittest.mock.patch(
            "ml.alphazero_lite.arena.ClassicMCTS", FakeClassicMCTS, create=True
        ):
            result = arena.evaluate_artifact_position(
                artifact_path=None,
                evaluator=None,
                state=state,
                simulations=96,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
                ablation_mode="classic_only",
            )

        self.assertIn("budget", result)
        self.assertEqual(144, result["budget"]["final_simulations"])
        self.assertEqual("mid_high_entropy_low_margin", result["budget"]["trigger"])
        self.assertEqual(7.5, result["budget"]["root_latency_ms"])

    def test_real_arena_report_includes_budget_summary_when_classic_only_probe_data_present(
        self,
    ):
        report = {
            "schema": "arena_v1",
            "games_played": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "promotion_decision": {"passed": False},
            "notes": {},
            "positions": [
                {
                    "challenger_summary": {
                        "budget": {
                            "dynamic_budget_enabled": True,
                            "baseline_simulations": 96,
                            "probe_simulations": 24,
                            "final_simulations": 128,
                            "phase_bucket": "late",
                            "entropy": 0.82,
                            "top_move_margin": 0.05,
                            "child_value_variance": 0.03,
                            "trigger": "late_high_entropy",
                            "root_latency_ms": 7.0,
                        }
                    },
                    "current_summary": {
                        "budget": {
                            "dynamic_budget_enabled": False,
                            "baseline_simulations": 96,
                            "probe_simulations": 96,
                            "final_simulations": 96,
                            "phase_bucket": "fixed",
                            "entropy": 0.0,
                            "top_move_margin": 0.0,
                            "child_value_variance": 0.0,
                            "trigger": "fixed_budget",
                            "root_latency_ms": 6.0,
                        }
                    },
                },
                {
                    "challenger_summary": {
                        "budget": {
                            "dynamic_budget_enabled": True,
                            "baseline_simulations": 96,
                            "probe_simulations": 24,
                            "final_simulations": 144,
                            "phase_bucket": "late",
                            "entropy": 0.88,
                            "top_move_margin": 0.03,
                            "child_value_variance": 0.04,
                            "trigger": "late_high_entropy",
                            "root_latency_ms": 9.0,
                        }
                    },
                    "current_summary": {
                        "budget": {
                            "dynamic_budget_enabled": False,
                            "baseline_simulations": 96,
                            "probe_simulations": 96,
                            "final_simulations": 96,
                            "phase_bucket": "fixed",
                            "entropy": 0.0,
                            "top_move_margin": 0.0,
                            "child_value_variance": 0.0,
                            "trigger": "fixed_budget",
                            "root_latency_ms": 6.5,
                        }
                    },
                },
            ],
        }

        summary = arena.budget_summary_for(report)

        self.assertEqual(136.0, summary["mean_final_simulations"])
        self.assertEqual(8.9, summary["p95_root_latency_ms"])
        self.assertEqual({"late_high_entropy": 2}, summary["trigger_counts"])

    def test_budget_summary_ignores_partial_or_invalid_budget_fields(self):
        report = {
            "positions": [
                {
                    "challenger_summary": {
                        "budget": {
                            "trigger": "fixed_budget",
                            "final_simulations": "not-a-number",
                            "root_latency_ms": 7.0,
                        }
                    }
                },
                {
                    "challenger_summary": {
                        "budget": {
                            "trigger": "late_high_entropy",
                            "final_simulations": 144,
                        }
                    }
                },
            ],
            "notes": {},
        }

        summary = arena.budget_summary_for(report)

        self.assertEqual(144.0, summary["mean_final_simulations"])
        self.assertEqual(7.0, summary["p95_root_latency_ms"])
        self.assertEqual(
            {"fixed_budget": 1, "late_high_entropy": 1}, summary["trigger_counts"]
        )

    def test_budget_summary_does_not_infer_hard_suite_buckets_from_synthetic_positions(
        self,
    ):
        report = {
            "positions": [
                {
                    "challenger_summary": {
                        "budget": {
                            "final_simulations": 128,
                            "root_latency_ms": 7.0,
                            "trigger": "early_high_entropy",
                            "phase_bucket": "early",
                        }
                    }
                },
                {
                    "challenger_summary": {
                        "budget": {
                            "final_simulations": 96,
                            "root_latency_ms": 6.0,
                            "trigger": "mid_high_entropy",
                            "phase_bucket": "mid",
                        }
                    }
                },
                {
                    "challenger_summary": {
                        "budget": {
                            "final_simulations": 64,
                            "root_latency_ms": 5.0,
                            "trigger": "late_high_entropy",
                            "phase_bucket": "late",
                        }
                    }
                },
            ],
            "notes": {},
        }

        summary = arena.budget_summary_for(report)

        self.assertNotIn("hard_suite_buckets", summary)

    def test_attach_budget_summary_emits_stable_empty_hard_suite_bucket_schema_without_worker_data(
        self,
    ):
        report = {
            "positions": [
                {
                    "challenger_summary": {
                        "budget": {
                            "final_simulations": 128,
                            "root_latency_ms": 7.0,
                            "trigger": "early_high_entropy",
                            "phase_bucket": "early",
                        }
                    }
                }
            ],
            "notes": {},
        }

        emitted = arena.attach_budget_summary(report)

        self.assertEqual(
            {
                "opening": {"games": 0, "score": None},
                "midgame": {"games": 0, "score": None},
                "late": {"games": 0, "score": None},
            },
            emitted["hard_suite_buckets"],
        )

    def test_evaluate_artifact_position_classic_only_synthesizes_fixed_budget_metadata(
        self,
    ):
        from ml.alphazero_lite import arena

        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed):
                del game, simulations, seed

            def search_root(self):
                class FakeChild:
                    visits = 10
                    wins = 6.0

                class FakeRoot:
                    visits = 10
                    wins = 6.0
                    children = {3: FakeChild()}

                return FakeRoot()

            def root_summary(self):
                return {
                    "selected_move": 3,
                    "child_stats": [{"move": 3, "visits": 10, "win_rate": 0.6}],
                }

        with unittest.mock.patch(
            "ml.alphazero_lite.arena.ClassicMCTS", FakeClassicMCTS, create=True
        ):
            result = arena.evaluate_artifact_position(
                artifact_path=None,
                evaluator=None,
                state=state,
                simulations=96,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
                ablation_mode="classic_only",
            )

        self.assertIn("budget", result)
        self.assertFalse(result["budget"]["dynamic_budget_enabled"])
        self.assertEqual(96, result["budget"]["baseline_simulations"])
        self.assertEqual(96, result["budget"]["chosen_simulations"])
        self.assertEqual(96, result["budget"]["final_simulations"])
        self.assertEqual("fixed_budget", result["budget"]["trigger"])

    def test_budget_summary_returns_none_for_missing_numeric_metrics(self):
        report = {
            "positions": [
                {
                    "challenger_summary": {
                        "budget": {
                            "trigger": "fixed_budget",
                            "final_simulations": "not-a-number",
                            "root_latency_ms": "not-a-number",
                        }
                    }
                }
            ],
            "notes": {},
        }

        summary = arena.budget_summary_for(report)

        self.assertIsNone(summary["mean_final_simulations"])
        self.assertIsNone(summary["p95_root_latency_ms"])
        self.assertEqual({"fixed_budget": 1}, summary["trigger_counts"])

    def test_evaluate_position_policy_matches_tactical_root_bias_adjusted_root(self):
        artifact_dir = self.write_tactical_bias_artifact()
        position = {
            "player_pits": [1, 0, 0, 0, 0, 2],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }

        with mock.patch(
            "ml.alphazero_lite.arena.encode_state", return_value=[1.0] * 15
        ):
            summary = arena.evaluate_artifact_position(
                artifact_path=artifact_dir,
                state=position,
                simulations=0,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(
                    root_policy_mode="deterministic", tactical_root_bias=2.0
                ),
            )

        self.assertEqual(0, summary["selected_move"])
        self.assertEqual(summary["selected_move"], int(np.argmax(summary["policy"])))
        self.assertGreater(summary["policy"][0], 0.8)
        self.assertLess(summary["policy"][5], 0.2)

    def test_evaluate_position_accepts_preloaded_evaluator(self):
        artifact_dir = self.write_basic_artifact()
        position = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        evaluator = arena.ArtifactEvaluator(artifact_dir)

        with mock.patch(
            "ml.alphazero_lite.arena.ArtifactEvaluator",
            side_effect=AssertionError("should not reload artifact"),
        ):
            summary = arena.evaluate_artifact_position(
                artifact_path=artifact_dir,
                evaluator=evaluator,
                state=position,
                simulations=16,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
            )

        self.assertIn("selected_move", summary)
        self.assertIn("value", summary)

    def test_evaluate_position_reuses_root_evaluation_without_extra_forward_pass(self):
        position = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        evaluate_calls = 0

        class FakeEvaluator:
            def evaluate(self, game):
                nonlocal evaluate_calls
                del game
                evaluate_calls += 1
                return np.full(6, 1.0 / 6.0, dtype=np.float32), 0.25

        summary = arena.evaluate_artifact_position(
            artifact_path="ignored",
            evaluator=FakeEvaluator(),
            state=position,
            simulations=0,
            seed=7,
            c_puct=1.25,
            search_options=arena.build_eval_search_options(),
        )

        self.assertEqual(1, evaluate_calls)
        self.assertEqual(0.25, summary["value"])

    def test_deterministic_root_policy_breaks_tied_root_visits_using_search_signal(
        self,
    ):
        root = arena.PUCT(
            evaluator=arena.ArtifactEvaluator,
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
        )._root_for(
            KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
        )
        root.children[0] = self_play.Node(
            game=self_play.KalahGame.from_state(root.game.to_state()),
            prior=0.2,
            visit_count=5,
            value_sum=1.0,
        )
        root.children[1] = self_play.Node(
            game=self_play.KalahGame.from_state(root.game.to_state()),
            prior=0.6,
            visit_count=5,
            value_sum=3.0,
        )

        visit_count_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="visit_count",
        )
        deterministic_search = self_play.PUCT(
            evaluator=self_play.HeuristicEvaluator(),
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            root_policy_mode="deterministic",
        )

        self.assertEqual(0, visit_count_search.select_root_move(root, [0, 1]))
        self.assertEqual(1, deterministic_search.select_root_move(root, [0, 1]))

    def test_tactical_root_bias_only_adjusts_immediate_tactical_moves(self):
        game = KalahGame.from_state(
            {
                "player_pits": [1, 0, 0, 0, 0, 1],
                "opponent_pits": [4, 4, 4, 4, 4, 4],
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        search = arena.PUCT(
            evaluator=arena.ArtifactEvaluator,
            simulations=1,
            c_puct=1.25,
            rng=random.Random(7),
            tactical_root_bias=0.1,
        )
        base_priors = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.45], dtype=np.float32)

        biased_priors = search.apply_tactical_root_bias(game, base_priors)

        self.assertGreater(biased_priors[5], base_priors[5])
        self.assertLess(biased_priors[0], base_priors[0])
        self.assertEqual(0.0, biased_priors[1])
        self.assertAlmostEqual(1.0, float(np.sum(biased_priors)), places=5)

    def test_run_arena_worker_uses_eval_search_defaults(self):
        captured_search_options = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

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

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
            )

        self.assertEqual("deterministic", result["search_options"]["root_policy_mode"])
        self.assertEqual(0.1, result["search_options"]["tactical_root_bias"])
        self.assertEqual(
            {**result["search_options"], "root_prior_override": None},
            captured_search_options[0],
        )
        self.assertEqual("v1", result["search_profile"]["version"])
        self.assertEqual("arena_eval", result["search_profile"]["kind"])
        self.assertEqual(
            result["search_profile"]["hash"], result["search_profile_hash"]
        )

    def test_artifact_evaluator_loads_residual_v2_weights_and_selected_input_encoding(
        self,
    ):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-unit-") as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "input_encoding": "kalah_v2",
                        "architecture": {
                            "model_type": "residual_v2",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w_input": [[0.5, 0.0], [0.0, 0.5]] + [[0.0, 0.0]] * 13,
                        "b_input": [0.0, 0.0],
                        "w_residual_1_1": [[1.0, 0.0], [0.0, 1.0]],
                        "b_residual_1_1": [0.0, 0.0],
                        "w_residual_1_2": [[1.0, 0.0], [0.0, 1.0]],
                        "b_residual_1_2": [0.0, 0.0],
                        "w_policy": [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        ],
                        "b_policy": [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                        "w_value": [[0.25], [0.25]],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            evaluator = arena.ArtifactEvaluator(artifact_dir)
            game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            encoded_state = [0.2] * 15

            with mock.patch(
                "ml.alphazero_lite.arena.encode_state", return_value=encoded_state
            ) as encode_state:
                priors, value = evaluator.evaluate(game)

            self.assertEqual("kalah_v2", evaluator.input_encoding)
            self.assertEqual("residual_v2", evaluator.model_type)
            self.assertEqual((6,), priors.shape)
            self.assertAlmostEqual(1.0, float(np.sum(priors)), places=5)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)
            encode_state.assert_called_once_with(
                game.to_state(), input_encoding="kalah_v2"
            )

    def test_artifact_evaluator_uses_specialized_residual_v3_heads(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-unit-") as tmp:
            artifact_dir = Path(tmp)
            (artifact_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "input_encoding": "kalah_v3",
                        "architecture": {
                            "model_type": "residual_v3",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (artifact_dir / "weights.json").write_text(
                json.dumps(
                    {
                        "w_input": [[1.0, 0.0], [0.0, 1.0]],
                        "b_input": [0.0, 0.0],
                        "w_residual_1_1": [[0.0, 0.0], [0.0, 0.0]],
                        "b_residual_1_1": [0.0, 0.0],
                        "w_residual_1_2": [[0.0, 0.0], [0.0, 0.0]],
                        "b_residual_1_2": [0.0, 0.0],
                        "w_policy_hidden": [[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]],
                        "b_policy_hidden": [0.0, 0.0, 0.0],
                        "w_policy": [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        ],
                        "b_policy": [0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
                        "w_value_hidden": [[1.0], [1.0]],
                        "b_value_hidden": [0.0],
                        "w_value": [[0.5]],
                        "b_value": [0.0],
                    }
                ),
                encoding="utf-8",
            )

            evaluator = arena.ArtifactEvaluator(artifact_dir)
            game = KalahGame.from_state(
                {
                    "player_pits": [4, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )

            with mock.patch(
                "ml.alphazero_lite.arena.encode_state", return_value=[1.0, 2.0]
            ):
                priors, value = evaluator.evaluate(game)

            expected_logits = np.array(
                [2.0, 6.0, 1.0, -10.0, -10.0, -10.0], dtype=np.float32
            )
            expected_logits = expected_logits - np.max(expected_logits)
            expected_priors = np.exp(expected_logits)
            expected_priors /= np.sum(expected_priors)

            np.testing.assert_allclose(priors, expected_priors, rtol=1e-6, atol=1e-6)
            self.assertAlmostEqual(float(np.tanh(1.5)), value, places=6)

    def test_run_arena_worker_preserves_search_options(self):
        captured_search_options = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

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
                captured_search_options.append(search_options)

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                fpu_mode="parent_q",
                reuse_subtree=True,
                normalize_values=True,
                root_policy_mode="deterministic",
                tactical_root_bias=0.2,
            )

        # run_arena_worker produces result with game outcome fields
        self.assertIn("wins", result)
        self.assertIn("losses", result)
        self.assertIn("draws", result)
        expected_search_options_arg = {
            "fpu_mode": "parent_q",
            "reuse_subtree": True,
            "normalize_values": True,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.2,
            "root_prior_override": None,
        }
        self.assertEqual(
            expected_search_options_arg,
            captured_search_options[0],
        )

    def test_run_arena_worker_preserves_value_trust_schedule_in_search_options(self):
        captured_search_options = []
        value_trust_schedule = {
            "enabled": True,
            "opening": 0.8,
            "midgame": 1.0,
            "late": 1.15,
        }

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

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
                captured_search_options.append(search_options)

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                value_trust_schedule=value_trust_schedule,
            )

        self.assertEqual(
            value_trust_schedule, result["search_options"]["value_trust_schedule"]
        )
        self.assertEqual(
            value_trust_schedule, captured_search_options[0]["value_trust_schedule"]
        )

    def test_run_arena_worker_emits_root_summary_value_trust_not_raw_schedule(self):
        value_trust_schedule = {
            "enabled": True,
            "opening": 0.8,
            "midgame": 1.0,
            "late": 1.15,
        }
        root_value_trust = {
            "enabled": True,
            "phase_bucket": "opening",
            "effective_multiplier": 0.8,
            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
        }

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

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

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0

                class FakeRoot:
                    q_value = 0.3

                    def child_for_action(self, action):
                        del action
                        return None

                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root, legal_moves
                return 0

            def root_summary(self):
                return {"value_trust": root_value_trust}

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                value_trust_schedule=value_trust_schedule,
            )

        self.assertEqual(root_value_trust, result["value_trust_summary"])
        self.assertNotEqual(value_trust_schedule, result["value_trust_summary"])

    def test_real_puct_run_arena_worker_emits_value_trust_summary(self):
        value_trust_schedule = {
            "enabled": True,
            "opening": 0.8,
            "midgame": 1.0,
            "late": 1.15,
        }

        class TinyDeterministicEvaluator(self_play.Evaluator):
            def __init__(self, _artifact_dir):
                pass

            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        with mock.patch(
            "ml.alphazero_lite.arena.ArtifactEvaluator", TinyDeterministicEvaluator
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=4,
                current_simulations=4,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                value_trust_schedule=value_trust_schedule,
            )

        self.assertEqual(
            {
                "enabled": True,
                "phase_bucket": "opening",
                "effective_multiplier": 0.8,
                "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
            },
            result["value_trust_summary"],
        )

    def test_real_puct_run_arena_worker_omits_value_trust_summary_by_default(self):
        class TinyDeterministicEvaluator(self_play.Evaluator):
            def __init__(self, _artifact_dir):
                pass

            def evaluate(self, game):
                del game
                priors = np.zeros(6, dtype=np.float32)
                priors[0] = 1.0
                return priors, 0.25

        with mock.patch(
            "ml.alphazero_lite.arena.ArtifactEvaluator", TinyDeterministicEvaluator
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=4,
                current_simulations=4,
                seed=42,
                c_puct=1.25,
                max_moves=1,
            )

        self.assertIsNone(result["value_trust_summary"])

    def test_evaluate_artifact_position_classic_only_preserves_value_trust_root_summary_metadata(
        self,
    ):
        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        value_trust = {
            "enabled": True,
            "phase_bucket": "midgame",
            "effective_multiplier": 1.15,
            "schedule": {"opening": 0.8, "midgame": 1.15, "late": 1.3},
        }

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed, value_trust_schedule=None):
                del game, simulations, seed, value_trust_schedule

            def search_root(self):
                class FakeChild:
                    visits = 10
                    wins = 6.0

                class FakeRoot:
                    visits = 10
                    wins = 6.0
                    children = {3: FakeChild()}

                return FakeRoot()

            def root_summary(self):
                return {
                    "selected_move": 3,
                    "child_stats": [{"move": 3, "visits": 10, "win_rate": 0.6}],
                    "value_trust": value_trust,
                }

        with unittest.mock.patch(
            "ml.alphazero_lite.arena.ClassicMCTS", FakeClassicMCTS, create=True
        ):
            result = arena.evaluate_artifact_position(
                artifact_path=None,
                evaluator=None,
                state=state,
                simulations=96,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(
                    value_trust_schedule=value_trust["schedule"]
                    | {"enabled": value_trust["enabled"]}
                ),
                ablation_mode="classic_only",
            )

        self.assertEqual(value_trust, result["value_trust"])

    def test_evaluate_artifact_position_classic_only_omits_value_trust_by_default(self):
        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        root_value_trust = {
            "enabled": True,
            "phase_bucket": "opening",
            "effective_multiplier": 0.8,
            "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
        }

        class FakeClassicMCTS:
            def __init__(self, game, simulations, seed, value_trust_schedule=None):
                del game, simulations, seed, value_trust_schedule

            def search_root(self):
                class FakeChild:
                    visits = 10
                    wins = 6.0

                class FakeRoot:
                    visits = 10
                    wins = 6.0
                    children = {3: FakeChild()}

                return FakeRoot()

            def root_summary(self):
                return {
                    "selected_move": 3,
                    "child_stats": [{"move": 3, "visits": 10, "win_rate": 0.6}],
                    "value_trust": root_value_trust,
                }

        with unittest.mock.patch(
            "ml.alphazero_lite.arena.ClassicMCTS", FakeClassicMCTS, create=True
        ):
            result = arena.evaluate_artifact_position(
                artifact_path=None,
                evaluator=None,
                state=state,
                simulations=96,
                seed=7,
                c_puct=1.25,
                search_options=arena.build_eval_search_options(),
                ablation_mode="classic_only",
            )

        self.assertNotIn("value_trust", result)

    def test_run_arena_worker_reuses_selected_child_between_moves_when_enabled(self):
        created_roots = []

        class FakeArtifactEvaluator:
            def __init__(self, _artifact_dir):
                pass

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [0, 0]
                self.pits = [4] * 12

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = 0
                self.pits = [4] * 12
                return absolute_move == 0

        class FakeRoot:
            def __init__(self, child=None):
                self.child = child

            def child_for_action(self, action):
                if action != 0:
                    raise AssertionError(f"unexpected action {action}")
                return self.child

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
                del evaluator, simulations, c_puct, rng, search_options
                created_roots.append(root)

            def run(self, game):
                del game
                root = roots_to_return[len(created_roots) - 1]
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, root

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()
            ),
        ):
            arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
                reuse_subtree=True,
            )

        self.assertEqual([None, second_root], created_roots)

    def test_run_arena_worker_does_not_reuse_subtree_across_evaluator_switches(self):
        created = []

        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [0, 0]
                self.pits = [4] * 12

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = self.moves_played % 2
                self.pits = [4] * 12
                return absolute_move == 0

        class FakeRoot:
            def __init__(self, child=None):
                self.child = child

            def child_for_action(self, action):
                if action != 0:
                    raise AssertionError(f"unexpected action {action}")
                return self.child

        first_root = FakeRoot(child=FakeRoot())
        second_root = FakeRoot()
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
                del simulations, c_puct, rng, search_options
                created.append((evaluator.name, root))

            def run(self, game):
                del game
                root = roots_to_return[len(created) - 1]
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, root

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()
            ),
        ):
            arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
                reuse_subtree=True,
            )

        self.assertEqual([("challenger", None), ("current", None)], created)

    def test_run_arena_worker_assigns_each_game_to_a_single_hard_suite_bucket(self):
        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [1, 0]
                self.pits = [3] * 10

            def over(self):
                return self.moves_played >= 4

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                del absolute_move
                phase_sums = [30, 18, 9, 9]
                self.moves_played += 1
                self.current_player = self.moves_played % 2
                self.pits = [1] * phase_sums[self.moves_played - 1]
                return True

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

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()
            ),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
            )

        self.assertEqual(
            1, sum(bucket["games"] for bucket in result["hard_suite_buckets"].values())
        )
        self.assertEqual(
            {"games": 0, "score": None}, result["hard_suite_buckets"]["opening"]
        )
        self.assertEqual(
            {"games": 1, "score": None}, result["hard_suite_buckets"]["midgame"]
        )
        self.assertEqual(
            {"games": 0, "score": None}, result["hard_suite_buckets"]["late"]
        )

    def test_record_completed_game_bucket_prefers_deepest_phase_reached(self):
        buckets = arena.empty_hard_suite_buckets()

        arena.record_completed_game_bucket(buckets, {"early", "mid", "late"})

        self.assertEqual({"games": 0, "score": None}, buckets["opening"])
        self.assertEqual({"games": 0, "score": None}, buckets["midgame"])
        self.assertEqual({"games": 1, "score": None}, buckets["late"])

    def test_run_arena_worker_emits_runtime_opening_cache_metrics_from_real_hits_and_misses(
        self,
    ):
        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeOpeningCache:
            opening_gate = {"max_ply": 10, "min_stones_in_pits": 36}

            def lookup(self, state, *, ply):
                del state
                if ply == 0:
                    return {
                        "selected_move": 0,
                        "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "value": 0.8,
                    }
                return None

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [1, 0]
                self.pits = [4] * 12

            def over(self):
                return self.moves_played >= 2

            def possible_moves(self):
                return [0] if not self.over() else []

            def pit_index(self, move):
                return move

            def move(self, absolute_move):
                self.moves_played += 1
                self.current_player = 0
                self.pits = [4] * 12 if self.moves_played == 1 else [0] * 12
                return absolute_move == 0

            def to_state(self):
                return {
                    "player_pits": [4, 4, 4, 4, 4, 4]
                    if self.moves_played == 0
                    else [3, 4, 4, 4, 4, 4],
                    "opponent_pits": [4, 4, 4, 4, 4, 4],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": self.current_player,
                }

        class FakeRootChild:
            visit_count = 1
            q_value = 0.3
            prior = 1.0

        class FakeRoot:
            visit_count = 1
            value_sum = 0.3
            children = {0: FakeRootChild()}

            @property
            def q_value(self):
                return self.value_sum / self.visit_count

            def child_for_action(self, action):
                del action
                return None

        class FakePUCT:
            run_calls = 0

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

            def run(self, game):
                del game
                FakePUCT.run_calls += 1
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root
                return legal_moves[0]

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()
            ),
            mock.patch(
                "ml.alphazero_lite.arena.time.perf_counter",
                side_effect=[1.0, 1.003, 2.0, 2.001, 3.0, 3.011],
            ),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=4,
                opening_cache=FakeOpeningCache(),
            )

        self.assertEqual(1, FakePUCT.run_calls)
        self.assertEqual(1, result["opening_cache_hits"])
        self.assertEqual(1, result["opening_cache_misses"])
        self.assertEqual(0.8, result["opening_cache_hit_quality_sum"])
        self.assertEqual(0.3, result["opening_cache_miss_quality_sum"])
        self.assertAlmostEqual(3.0, result["opening_cache_hit_latency_ms"][0], places=6)
        self.assertAlmostEqual(
            12.0, result["opening_cache_miss_latency_ms"][0], places=6
        )

    def test_run_arena_worker_miss_latency_includes_lookup_overhead_in_total_runtime_cost(
        self,
    ):
        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeOpeningCache:
            def lookup(self, state, *, ply):
                del state, ply
                return None

        class FakeRootChild:
            visit_count = 1
            q_value = 0.4
            prior = 1.0

        class FakeRoot:
            visit_count = 1
            value_sum = 0.4
            children = {0: FakeRootChild()}

            @property
            def q_value(self):
                return self.value_sum / self.visit_count

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

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root
                return legal_moves[0]

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.time.perf_counter",
                side_effect=[1.0, 1.002, 2.0, 2.011],
            ),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                opening_cache=FakeOpeningCache(),
            )

        self.assertEqual(1, result["opening_cache_misses"])
        self.assertAlmostEqual(
            13.0, result["opening_cache_miss_latency_ms"][0], places=6
        )
        self.assertAlmostEqual(13.0, result["move_durations_ms"][0], places=6)

    def test_run_arena_worker_does_not_count_gate_rejected_positions_as_opening_cache_misses(
        self,
    ):
        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeOpeningCache:
            opening_gate = {"max_ply": 10, "min_stones_in_pits": 49}

            def __init__(self):
                self.lookup_calls = []

            def lookup(self, state, *, ply):
                self.lookup_calls.append((state, ply))
                return None

        class FakeRootChild:
            visit_count = 1
            q_value = 0.4
            prior = 1.0

        class FakeRoot:
            visit_count = 1
            value_sum = 0.4
            children = {0: FakeRootChild()}

            @property
            def q_value(self):
                return self.value_sum / self.visit_count

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

            def run(self, game):
                del game
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, FakeRoot()

            def select_root_move(self, root, legal_moves):
                del root
                return legal_moves[0]

        cache = FakeOpeningCache()
        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.time.perf_counter",
                side_effect=[1.0, 1.001, 2.0, 2.011],
            ),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                opening_cache=cache,
            )

        self.assertEqual([], cache.lookup_calls)
        self.assertEqual(0, result["opening_cache_hits"])
        self.assertEqual(0, result["opening_cache_misses"])

    def test_run_arena_worker_accepts_string_selected_move_when_it_is_legal(self):
        class FakeArtifactEvaluator:
            def __init__(self, artifact_dir):
                self.name = str(artifact_dir)

        class FakeOpeningCache:
            opening_gate = {"max_ply": 10, "min_stones_in_pits": 36}

            def lookup(self, state, *, ply):
                del state, ply
                return {
                    "selected_move": "0",
                    "policy": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "value": 0.8,
                }

        class FakeGame:
            def __init__(self):
                self.moves_played = 0
                self.current_player = 0
                self.captured_seeds = [1, 0]

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

        class FakePUCT:
            run_calls = 0

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

            def run(self, game):
                del game
                FakePUCT.run_calls += 1
                visits = np.zeros(6, dtype=np.float32)
                visits[0] = 1.0
                return visits, None

            def select_root_move(self, root, legal_moves):
                del root
                return legal_moves[0]

        with (
            mock.patch(
                "ml.alphazero_lite.arena.ArtifactEvaluator", FakeArtifactEvaluator
            ),
            mock.patch("ml.alphazero_lite.arena.PUCT", FakePUCT),
            mock.patch(
                "ml.alphazero_lite.arena.KalahGame.from_state", return_value=FakeGame()
            ),
            mock.patch(
                "ml.alphazero_lite.arena.time.perf_counter",
                side_effect=[1.0, 1.003, 2.0, 2.011],
            ),
        ):
            result = arena.run_arena_worker(
                worker_id=0,
                start_index=0,
                games=1,
                challenger_path="challenger",
                current_path="current",
                challenger_simulations=32,
                current_simulations=16,
                seed=42,
                c_puct=1.25,
                max_moves=1,
                opening_cache=FakeOpeningCache(),
            )

        self.assertEqual(0, FakePUCT.run_calls)
        self.assertEqual(1, result["opening_cache_hits"])
        self.assertEqual(0, result["opening_cache_misses"])

    def test_aggregate_worker_reports_emits_stable_hard_suite_bucket_schema(self):
        report = arena.aggregate_worker_reports(
            games=3,
            min_score=0.55,
            challenger_path=Path("challenger"),
            current_path=Path("current"),
            challenger_simulations=32,
            current_simulations=16,
            seed=42,
            workers=2,
            search_options=arena.build_eval_search_options(),
            results=[
                {
                    "wins": 2,
                    "losses": 0,
                    "draws": 0,
                    "move_durations_ms": [10.0, 20.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": {
                        "opening": {"games": 2, "score": None},
                        "midgame": {"games": 1, "score": None},
                        "late": {"games": 0, "score": None},
                    },
                },
                {
                    "wins": 0,
                    "losses": 1,
                    "draws": 0,
                    "move_durations_ms": [30.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": {
                        "opening": {"games": 0, "score": None},
                        "midgame": {"games": 1, "score": None},
                        "late": {"games": 3, "score": None},
                    },
                },
            ],
        )

        self.assertEqual(
            {
                "opening": {"games": 2, "score": None},
                "midgame": {"games": 2, "score": None},
                "late": {"games": 3, "score": None},
            },
            report["hard_suite_buckets"],
        )

    def test_aggregate_worker_reports_includes_confidence_interval_fields(self):
        report = arena.aggregate_worker_reports(
            games=4,
            min_score=0.55,
            challenger_path=Path("challenger"),
            current_path=Path("current"),
            challenger_simulations=32,
            current_simulations=16,
            seed=42,
            workers=1,
            search_options=arena.build_eval_search_options(),
            results=[
                {
                    "wins": 2,
                    "losses": 1,
                    "draws": 1,
                    "move_durations_ms": [10.0, 12.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": arena.empty_hard_suite_buckets(),
                }
            ],
        )

        self.assertEqual(4, report["games"])
        self.assertEqual(0.625, report["score"])
        self.assertEqual(0.55, report["threshold"])
        self.assertAlmostEqual(0.075, report["threshold_margin"])
        self.assertIn("confidence_interval_95", report)
        self.assertIn("lower", report["confidence_interval_95"])
        self.assertIn("upper", report["confidence_interval_95"])
        self.assertIsInstance(report["unstable_decision"], bool)

    def test_aggregate_worker_reports_emits_opening_cache_summary_from_runtime_metrics(
        self,
    ):
        report = arena.aggregate_worker_reports(
            games=4,
            min_score=0.55,
            challenger_path=Path("challenger"),
            current_path=Path("current"),
            challenger_simulations=32,
            current_simulations=16,
            seed=42,
            workers=2,
            search_options=arena.build_eval_search_options(),
            results=[
                {
                    "wins": 2,
                    "losses": 0,
                    "draws": 0,
                    "move_durations_ms": [10.0, 14.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": arena.empty_hard_suite_buckets(),
                    "opening_cache_hits": 2,
                    "opening_cache_misses": 2,
                    "opening_cache_hit_quality_sum": 1.4,
                    "opening_cache_miss_quality_sum": 1.0,
                    "opening_cache_hit_latency_ms": [3.0, 5.0],
                    "opening_cache_miss_latency_ms": [11.0, 13.0],
                },
                {
                    "wins": 0,
                    "losses": 2,
                    "draws": 0,
                    "move_durations_ms": [12.0, 18.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": arena.empty_hard_suite_buckets(),
                    "opening_cache_hits": 0,
                    "opening_cache_misses": 0,
                    "opening_cache_hit_quality_sum": 0.0,
                    "opening_cache_miss_quality_sum": 0.0,
                    "opening_cache_hit_latency_ms": [],
                    "opening_cache_miss_latency_ms": [],
                },
            ],
        )

        self.assertEqual(0.5, report["opening_cache_summary"]["runtime_hit_rate"])
        self.assertIsNone(report["opening_cache_summary"]["training_hit_rate"])
        self.assertEqual(
            0.2, report["opening_cache_summary"]["opening_bucket_quality_delta"]
        )
        self.assertEqual(-8.0, report["opening_cache_summary"]["latency_delta_ms"])

    def test_aggregate_worker_reports_falls_back_to_configured_value_trust_schedule_when_runtime_summary_is_missing(
        self,
    ):
        configured_schedule = {
            "enabled": True,
            "opening": 0.8,
            "midgame": 1.0,
            "late": 1.15,
        }

        report = arena.aggregate_worker_reports(
            games=2,
            min_score=0.55,
            challenger_path=Path("challenger"),
            current_path=Path("current"),
            challenger_simulations=32,
            current_simulations=16,
            seed=42,
            workers=1,
            search_options=arena.build_eval_search_options(
                value_trust_schedule=configured_schedule
            ),
            results=[
                {
                    "wins": 1,
                    "losses": 1,
                    "draws": 0,
                    "move_durations_ms": [10.0, 12.0],
                    "search_profile": {"hash": "abc"},
                    "search_profile_hash": "abc",
                    "hard_suite_buckets": arena.empty_hard_suite_buckets(),
                    "value_trust_summary": None,
                }
            ],
        )

        self.assertEqual(
            {
                "enabled": True,
                "phase_bucket": None,
                "effective_multiplier": None,
                "schedule": {
                    "opening": 0.8,
                    "midgame": 1.0,
                    "late": 1.15,
                },
                "source": "configured_schedule",
            },
            report["value_trust_summary"],
        )

    def test_cli_generates_validator_compatible_arena_report(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text(
                "\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8"
            )

            train = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            for out_dir, version in (
                (challenger_dir, "azlite-challenger"),
                (current_dir, "azlite-current"),
            ):
                export = subprocess.run(
                    [
                        python,
                        "ml/alphazero_lite/export_artifact.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--out-dir",
                        str(out_dir),
                        "--version",
                        version,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, export.returncode, msg=export.stderr)

            arena = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "6",
                    "--challenger-simulations",
                    "32",
                    "--current-simulations",
                    "32",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--min-score",
                    "0.55",
                    "--workers",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, arena.returncode, msg=arena.stderr)
            self.assertTrue(out_path.exists())

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("budget_summary", report)
            self.assertEqual(32.0, report["budget_summary"]["mean_final_simulations"])
            self.assertEqual(
                {"fixed_budget": 6}, report["budget_summary"]["trigger_counts"]
            )
            self.assertGreaterEqual(
                report["budget_summary"]["p95_root_latency_ms"], 0.0
            )
            validate = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/validate_arena_report.py",
                    "--report",
                    str(out_path),
                    "--min-score",
                    "0.55",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, validate.returncode, msg=validate.stderr)
            self.assertIn("arena_report_valid", validate.stdout)

    def test_validate_arena_report_strict_ci_mode_fails_when_lower_bound_is_below_threshold(
        self,
    ):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-") as tmp:
            report_path = Path(tmp) / "arena_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "arena_v1",
                        "games_played": 4,
                        "wins": 3,
                        "losses": 1,
                        "draws": 0,
                        "promotion_decision": {"passed": True},
                    }
                ),
                encoding="utf-8",
            )

            validate = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/validate_arena_report.py",
                    "--report",
                    str(report_path),
                    "--min-score",
                    "0.55",
                    "--require-pass",
                    "--require-ci-above-threshold",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(1, validate.returncode)
            self.assertIn("arena_prefilter_failed", validate.stderr)

    def test_cli_records_multi_worker_distribution_when_workers_requested(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-") as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "data.jsonl"
            checkpoint_path = tmp_path / "checkpoint.npz"
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"

            row = {
                "state": [0.1] * 15,
                "policy": [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                "value": 1.0,
            }
            data_path.write_text(
                "\n".join([json.dumps(row) for _ in range(64)]) + "\n", encoding="utf-8"
            )

            train = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/train.py",
                    "--data",
                    str(data_path),
                    "--out",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "32",
                    "--device",
                    "cpu",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, train.returncode, msg=train.stderr)

            for out_dir, version in (
                (challenger_dir, "azlite-challenger"),
                (current_dir, "azlite-current"),
            ):
                export = subprocess.run(
                    [
                        python,
                        "ml/alphazero_lite/export_artifact.py",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--out-dir",
                        str(out_dir),
                        "--version",
                        version,
                    ],
                    cwd=Path(__file__).resolve().parents[2],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, export.returncode, msg=export.stderr)

            arena = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--seed",
                    "42",
                    "--out",
                    str(out_path),
                    "--min-score",
                    "0.55",
                    "--workers",
                    "2",
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, arena.returncode, msg=arena.stderr)

            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(2, report["notes"]["workers_requested"])
            self.assertEqual(2, report["notes"]["workers_used"])
            self.assertEqual([3, 2], report["notes"]["worker_game_counts"])

    def test_stub_cli_emits_budget_summary_for_downstream_parity(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()

            result = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("budget_summary", report)
            self.assertEqual(0.8, report["score"])
            self.assertEqual(16.0, report["budget_summary"]["mean_final_simulations"])
            self.assertEqual(
                {"fixed_budget": 5}, report["budget_summary"]["trigger_counts"]
            )
            self.assertEqual(
                5,
                sum(
                    bucket["games"] for bucket in report["hard_suite_buckets"].values()
                ),
            )

    def test_stub_cli_records_value_trust_schedule_in_search_options(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()

            result = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--out",
                    str(out_path),
                    "--value-trust-enabled",
                    "--value-trust-opening",
                    "0.8",
                    "--value-trust-midgame",
                    "1.0",
                    "--value-trust-late",
                    "1.15",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "enabled": True,
                    "opening": 0.8,
                    "midgame": 1.0,
                    "late": 1.15,
                },
                report["notes"]["search_options"]["value_trust_schedule"],
            )

    def test_stub_cli_emits_value_trust_summary_for_benchmark_passthrough(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()

            result = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--out",
                    str(out_path),
                    "--value-trust-enabled",
                    "--value-trust-opening",
                    "0.8",
                    "--value-trust-midgame",
                    "1.0",
                    "--value-trust-late",
                    "1.15",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {
                    "enabled": True,
                    "phase_bucket": "opening",
                    "effective_multiplier": 0.8,
                    "schedule": {"opening": 0.8, "midgame": 1.0, "late": 1.15},
                },
                report["value_trust_summary"],
            )

    def test_stub_cli_derives_pass_and_stdout_from_report_score(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()

            result = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "1",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--out",
                    str(out_path),
                    "--min-score",
                    "0.55",
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(0.5, report["score"])
            self.assertFalse(report["promotion_decision"]["passed"])
            self.assertIn("score=0.5000 passed=False", result.stdout)

    def test_stub_cli_emits_opening_cache_summary_with_training_pass_through(self):
        python = self.executable_python()
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            training_summary_path = tmp_path / "training_summary.json"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()
            training_summary_path.write_text(
                json.dumps({"training_hit_rate": 0.4}), encoding="utf-8"
            )

            result = subprocess.run(
                [
                    python,
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--opening-cache-training-summary",
                    str(training_summary_path),
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("opening_cache_summary", report)
            self.assertEqual(0.4, report["opening_cache_summary"]["training_hit_rate"])

    def test_stub_cli_runs_without_numpy_on_plain_python(self):
        with tempfile.TemporaryDirectory(prefix="azlite-arena-stub-") as tmp:
            tmp_path = Path(tmp)
            challenger_dir = tmp_path / "challenger"
            current_dir = tmp_path / "current"
            out_path = tmp_path / "arena_report.json"
            challenger_dir.mkdir()
            current_dir.mkdir()

            result = subprocess.run(
                [
                    "python3",
                    "ml/alphazero_lite/arena.py",
                    "--challenger",
                    str(challenger_dir),
                    "--current",
                    str(current_dir),
                    "--games",
                    "5",
                    "--challenger-simulations",
                    "16",
                    "--current-simulations",
                    "16",
                    "--out",
                    str(out_path),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "AZLITE_ARENA_STUB": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("arena_v1", report["schema"])
            self.assertTrue(report["promotion_decision"]["passed"])


if __name__ == "__main__":
    unittest.main()
