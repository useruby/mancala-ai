"""Contracts for the diagnostic-only affine value-scale causal preflight."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_value_scale_causal_preflight import (
    AffineValueEvaluator,
    CACHE_SCHEMA,
    cache_matches,
    canonical_manifest,
    classify,
    load_cache,
    q_rank,
    save_cache,
)


class _Current:
    def evaluate(self, _game):
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32), 0.8


class ValueScaleCausalPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = canonical_manifest(
            kind="continuation_labels",
            current_weights_sha256="weights-a",
            state_corpus_hash="corpus-a",
            base_seed=42,
            search_budget=1200,
            c_puct=1.25,
            search_options={"normalize_values": False},
        )

    def test_affine_evaluator_preserves_policy_and_clips_value(self) -> None:
        baseline = _Current()
        transformed = AffineValueEvaluator(baseline, a=3.0, b=0.0)
        policy, value = transformed.evaluate(KalahGame([4] * 12, [0, 0], 0))
        expected_policy, _ = baseline.evaluate(None)
        np.testing.assert_array_equal(expected_policy, policy)
        self.assertEqual(1.0, value)

    def test_cache_requires_full_provenance_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            save_cache(path, self.manifest, [{"state_hash": "a"}])
            self.assertEqual([{"state_hash": "a"}], load_cache(path, self.manifest))
            for field, replacement in (
                ("current_weights_sha256", "weights-b"),
                ("state_corpus_hash", "corpus-b"),
                ("base_seed", 43),
                ("search_budget", 768),
                ("c_puct", 1.5),
                ("search_options", {"normalize_values": True}),
                ("schema", "old-schema"),
            ):
                changed = dict(self.manifest)
                changed[field] = replacement
                self.assertIsNone(load_cache(path, changed), field)

    def test_cache_without_matching_manifest_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(json.dumps({"rows": []}), encoding="utf-8")
            self.assertIsNone(load_cache(path, self.manifest))
            self.assertFalse(cache_matches({"schema": CACHE_SCHEMA}, self.manifest))

    def test_q_rank_ignores_changed_child_statistics_with_same_move_order(self) -> None:
        baseline = {
            "child_stats": [
                {"move": 3, "q_value": 0.8, "visits": 4},
                {"move": 1, "q_value": 0.2, "visits": 9},
            ]
        }
        changed_stats = {
            "child_stats": [
                {"move": 3, "q_value": 0.7, "visits": 100},
                {"move": 1, "q_value": -0.4, "visits": 1},
            ]
        }
        swapped = {
            "child_stats": [
                {"move": 3, "q_value": 0.1},
                {"move": 1, "q_value": 0.2},
            ]
        }
        self.assertEqual([3, 1], q_rank(baseline))
        self.assertEqual(q_rank(baseline), q_rank(changed_stats))
        self.assertNotEqual(q_rank(baseline), q_rank(swapped))

    def test_q_rank_breaks_ties_by_move_id(self) -> None:
        self.assertEqual(
            [1, 3],
            q_rank(
                {
                    "child_stats": [
                        {"move": 3, "q_value": 0.5},
                        {"move": 1, "q_value": 0.5},
                    ]
                }
            ),
        )

    def test_primary_margin_rejection_beats_secondary_control_conflict(self) -> None:
        def treatment(mean: float, lower: float, upper: float) -> dict:
            return {
                "causal": {
                    str(budget): {
                        "normalized_final_margin_delta": {
                            "mean": mean,
                            "lower": lower,
                            "upper": upper,
                            "unique_states": 64,
                        }
                    }
                    for budget in (768, 1200)
                }
            }

        report = {
            "fresh": {
                "D1200": {
                    "treatments": {
                        "margin_affine": treatment(-0.1, -0.2, -0.01),
                        "outcome_affine": treatment(0.1, 0.0, 0.2),
                    }
                }
            }
        }
        evaluation = {
            "current_value": {"margin": {"mae": 1.0}},
            "margin_affine": {"margin": {"mae": 0.8}},
        }
        self.assertEqual(
            "margin_value_semantics_rejected_for_search", classify(evaluation, report)
        )


if __name__ == "__main__":
    unittest.main()
