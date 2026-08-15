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
    load_cache,
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


if __name__ == "__main__":
    unittest.main()
