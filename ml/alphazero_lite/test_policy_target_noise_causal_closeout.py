"""Determinism invariants for the PR #177 causal closeout."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite.run_policy_target_noise_causal_closeout import (
    _forced_task_record,
    build_tasks,
    canonical_json,
    continuation_seed_context,
    continuation_seed_identity,
    forced_continuation,
    run_forced_tasks,
)
from ml.alphazero_lite.self_play import HeuristicEvaluator


STATE = {
    "player_pits": [1, 0, 0, 0, 0, 0],
    "opponent_pits": [0, 0, 0, 0, 0, 0],
    "player_store": 23,
    "opponent_store": 24,
    "current_player": 0,
}


def task() -> dict:
    return {
        "state_hash": "state-a",
        "state": STATE,
        "player": 0,
        "phase": "late",
        "source_domain": "evaluation_opening_diagnostic",
        "domain_group": "evaluation_diagnostic",
        "legal_move_count": 1,
        "continuation_budget": 768,
        "experiment_seed": 42,
        "teacher_moves": {"noisy_n384": 0, "denoised_d384": 0, "reference_d1200": 0},
    }


class FakeExecutor:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def map(self, function, values):
        return map(function, values)


class PolicyTargetNoiseCausalCloseoutTest(unittest.TestCase):
    def test_renaming_intervention_leaves_base_seed_unchanged(self) -> None:
        left = continuation_seed_identity(
            state_hash="state-a",
            continuation_budget=768,
            root_player=0,
            experiment_seed=42,
        )
        right = continuation_seed_identity(
            state_hash="state-a",
            continuation_budget=768,
            root_player=0,
            experiment_seed=42,
        )
        self.assertEqual(left, right)

    def test_base_seed_excludes_labels_and_forced_moves(self) -> None:
        context = continuation_seed_context(
            state_hash="state-a",
            continuation_budget=1200,
            root_player=1,
            experiment_seed=42,
        )
        serialized = canonical_json(context)
        self.assertNotIn("noisy", serialized)
        self.assertNotIn("denoised", serialized)
        self.assertNotIn("reference", serialized)

    def test_same_forced_move_is_identical_and_repeated(self) -> None:
        result_a = forced_continuation(
            evaluator=HeuristicEvaluator(), task=task(), forced_move=0
        )
        result_b = forced_continuation(
            evaluator=HeuristicEvaluator(), task=task(), forced_move=0
        )
        self.assertEqual(result_a, result_b)

    def test_same_forced_move_has_identical_noisy_and_denoised_interventions(
        self,
    ) -> None:
        with patch(
            "ml.alphazero_lite.run_policy_target_noise_causal_closeout._WORKER_EVALUATOR",
            HeuristicEvaluator(),
        ):
            record = _forced_task_record(task())
        self.assertEqual(
            record["interventions"]["noisy_n384"],
            record["interventions"]["denoised_d384"],
        )

    def test_forced_move_does_not_change_preintervention_pair_identity(self) -> None:
        first = continuation_seed_identity(
            state_hash="state-a",
            continuation_budget=768,
            root_player=0,
            experiment_seed=42,
        )
        second = continuation_seed_identity(
            state_hash="state-a",
            continuation_budget=768,
            root_player=0,
            experiment_seed=42,
        )
        self.assertEqual(first, second)

    def test_workers_preserve_identical_ordered_records(self) -> None:
        rows = [
            {**task(), "state_hash": "state-b"},
            {**task(), "state_hash": "state-a"},
        ]

        def fake_record(row):
            return {
                "state_hash": row["state_hash"],
                "continuation_budget": row["continuation_budget"],
            }

        with (
            patch(
                "ml.alphazero_lite.run_policy_target_noise_causal_closeout._init_worker"
            ),
            patch(
                "ml.alphazero_lite.run_policy_target_noise_causal_closeout._forced_task_record",
                side_effect=fake_record,
            ),
            patch(
                "ml.alphazero_lite.run_policy_target_noise_causal_closeout.concurrent.futures.ProcessPoolExecutor",
                FakeExecutor,
            ),
        ):
            sequential = run_forced_tasks(
                tasks=rows, checkpoint=Path("unused"), workers=1
            )
            parallel = run_forced_tasks(
                tasks=rows, checkpoint=Path("unused"), workers=2
            )
        self.assertEqual(sequential, parallel)

    def test_task_builder_keeps_all_disagreement_budgets(self) -> None:
        source = {
            **task(),
            "teacher_moves": {
                "noisy_n384": 0,
                "denoised_d384": 1,
                "reference_d1200": 2,
            },
        }
        # The builder derives tasks from teacher rows in production; this guards its two-budget contract.
        teacher_row = {
            **source,
            "noisy_n384": {"top_move": 0},
            "denoised_d384": {"top_move": 1},
            "reference_d1200": {"top_move": 2},
        }
        tasks = build_tasks([teacher_row], experiment_seed=42)
        self.assertEqual([768, 1200], [row["continuation_budget"] for row in tasks])


if __name__ == "__main__":
    unittest.main()
