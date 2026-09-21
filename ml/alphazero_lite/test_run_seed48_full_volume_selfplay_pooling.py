from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.alphazero_lite import run_seed48_fixed_volume_selfplay_pooling as pooling
from ml.alphazero_lite import run_seed48_full_volume_selfplay_pooling as full_volume


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/alphazero-lite-seed48-full-volume-selfplay-pooling"


class FullVolumeSelfplayPoolingTest(unittest.TestCase):
    def test_plan_pins_full_volume_contract(self):
        plan = json.loads((DATA / "plan.json").read_text(encoding="utf-8"))

        self.assertEqual(
            "azlite_seed48_full_volume_selfplay_pooling_v1", plan["schema"]
        )
        self.assertEqual([443, 1001, 1003, 1009, 1013], plan["training_seed_group"])
        self.assertEqual([1, 4, 1, 8, 4], plan["replay_weights"])
        self.assertEqual(5, len(plan["selfplay_sources"]))
        self.assertEqual(
            [401, 407, 413, 419, 443],
            [source["seed"] for source in plan["selfplay_sources"]],
        )
        self.assertEqual(
            [
                "ee5fedce1fe4fea1fa2bdf51dce8b26a54014e665115a5578b7af440de9c8b44",
                "f3a27c48185ed527b2c0c817b49479bb0ff401a9617d7945ee96e42400163baa",
                "2fe9549255f17698d227864511be5acb2dc167c322612a546cb399217f168c85",
                "83befab937e2b8eebe27577f428a9fc8cdc8498eb639ddaf22415b690fdaddca",
                "a8046517cd6da07a7ddff8efa03fea27b5a53f5a9146bcf0632150a651e74895",
            ],
            [source["sha256"] for source in plan["selfplay_sources"]],
        )
        self.assertEqual(
            "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c",
            plan["parent_weights_sha256"],
        )
        self.assertEqual(4, plan["training"]["epochs"])
        self.assertFalse(plan["canonical_gate_run"])
        self.assertFalse(plan["promotion_performed"])
        self.assertFalse(plan["candidate_selection_performed"])

    def test_manifest_is_balanced_and_reproducible(self):
        manifest = json.loads((DATA / "pool_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(347, manifest["pool_seed"])
        self.assertEqual(353_455, manifest["pool_row_count"])
        self.assertEqual(5, len(manifest["sources"]))
        self.assertTrue(
            all(source["source_row_count"] >= 70_691 for source in manifest["sources"])
        )
        self.assertTrue(
            all(
                source["selected_row_count"] == 70_691 for source in manifest["sources"]
            )
        )
        self.assertEqual(
            353_455,
            sum(source["selected_row_count"] for source in manifest["sources"]),
        )

    def test_bootstrap_and_nonpromotion_contract_are_deterministic(self):
        result = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
        deltas = list(result["paired_arena_deltas"].values())

        original_seed = pooling.POOL_SEED
        try:
            pooling.POOL_SEED = 347
            self.assertEqual(pooling.bootstrap(deltas), pooling.bootstrap(deltas))
        finally:
            pooling.POOL_SEED = original_seed
        self.assertEqual(10_000, result["paired_bootstrap_95"]["samples"])
        self.assertEqual(347, result["paired_bootstrap_95"]["seed"])
        self.assertFalse(result["canonical_gate_run"])
        self.assertFalse(result["promotion_performed"])
        self.assertFalse(result["candidate_selection_performed"])
        self.assertTrue(callable(full_volume.main))


if __name__ == "__main__":
    unittest.main()
