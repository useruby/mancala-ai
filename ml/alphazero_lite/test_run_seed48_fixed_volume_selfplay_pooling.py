from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import run_seed48_fixed_volume_selfplay_pooling as pooling


class FixedVolumeSelfplayPoolingTest(unittest.TestCase):
    def test_plan_pins_sources_parent_replay_and_training_seeds(self):
        plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-fixed-volume-selfplay-pooling/plan.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(pooling.SCHEMA, plan["schema"])
        self.assertEqual(
            pooling.SOURCE_SEEDS, tuple(row["seed"] for row in plan["selfplay_sources"])
        )
        self.assertEqual(
            (
                "ee5fedce1fe4fea1fa2bdf51dce8b26a54014e665115a5578b7af440de9c8b44",
                "f3a27c48185ed527b2c0c817b49479bb0ff401a9617d7945ee96e42400163baa",
                "2fe9549255f17698d227864511be5acb2dc167c322612a546cb399217f168c85",
                "83befab937e2b8eebe27577f428a9fc8cdc8498eb639ddaf22415b690fdaddca",
                "a8046517cd6da07a7ddff8efa03fea27b5a53f5a9146bcf0632150a651e74895",
            ),
            tuple(row["sha256"] for row in plan["selfplay_sources"]),
        )
        self.assertEqual(
            "935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c",
            plan["parent_weights_sha256"],
        )
        self.assertEqual([1, 4, 1, 8, 4], plan["replay_weights"])
        self.assertEqual(pooling.TRAINING_SEEDS, tuple(plan["training_seed_group"]))
        self.assertFalse(plan["canonical_gate_run"])
        self.assertFalse(plan["promotion_performed"])
        self.assertFalse(plan["candidate_selection_performed"])
        baseline = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-seed48-challenger-variance/results.json"
            ).read_text(encoding="utf-8")
        )["plan"]
        for field in (
            "parent_artifact",
            "parent_weights_sha256",
            "parent_init_checkpoint",
            "training",
            "replay_weights",
            "replay_sources",
            "training_seed_group",
            "exact_corpus",
            "evaluation",
        ):
            self.assertEqual(baseline[field], plan[field])

    def test_pool_regenerates_byte_for_byte_with_equal_contributions(self):
        with tempfile.TemporaryDirectory(prefix="azlite-fixed-volume-pool-") as temp:
            root = Path(temp)
            sources = []
            for seed in pooling.SOURCE_SEEDS:
                path = root / f"source-{seed}.jsonl"
                rows = [
                    {
                        "state": [seed, index],
                        "move_index": index,
                        "game_index": index // 2,
                    }
                    for index in range(4)
                ]
                payload = "".join(json.dumps(row) + "\n" for row in rows)
                path.write_text(payload, encoding="utf-8")
                sources.append(
                    {
                        "seed": seed,
                        "path": path.name,
                        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                    }
                )
            plan = {"selfplay_sources": sources}
            with (
                patch.object(pooling, "ROOT", root),
                patch.object(pooling, "ROWS_PER_SOURCE", 2),
            ):
                first = pooling.construct_pool(plan, root / "first.jsonl")
                second = pooling.construct_pool(plan, root / "second.jsonl")

            self.assertEqual(10, first["pool_row_count"])
            self.assertEqual(first["pool_sha256"], second["pool_sha256"])
            self.assertEqual(
                (root / "first.jsonl").read_bytes(),
                (root / "second.jsonl").read_bytes(),
            )
            self.assertTrue(
                all(source["selected_row_count"] == 2 for source in first["sources"])
            )
            self.assertEqual(
                5, len(first["diagnostics"]["source_contribution_by_ply_bucket"])
            )

    def test_training_commands_differ_only_in_pooled_data_path(self):
        plan = {
            "replay_sources": [{"path": f"replay-{index}.jsonl"} for index in range(4)],
            "parent_init_checkpoint": "parent.npz",
            "training": {
                "epochs": 4,
                "batch_size": 512,
                "lr_scheduler": "none",
                "hidden_sizes": "96,3",
                "model_type": "residual_v3",
                "input_encoding": "kalah_v3",
                "value_loss": "huber",
                "huber_delta": 1.0,
                "value_loss_weight": 0.3,
                "val_split": 0.1,
                "grad_clip": 1.0,
                "policy_target_mode": "sharpened",
                "value_target_mode": "sharpened",
            },
        }
        with patch.object(pooling, "ROOT", Path("/experiment")):
            first = pooling.training_command(
                Path("/data/pool.jsonl"), Path("/out/a.npz"), plan, 443
            )
            second = pooling.training_command(
                Path("/data/pool.jsonl"), Path("/out/b.npz"), plan, 1001
            )

        for flag in (
            "--epochs",
            "--batch-size",
            "--replay-weights",
            "--hidden-sizes",
            "--model-type",
            "--value-loss",
        ):
            self.assertEqual(
                first[first.index(flag) + 1], second[second.index(flag) + 1]
            )
        self.assertNotIn("canonical", " ".join(first).lower())
        self.assertNotIn("promote", " ".join(first).lower())

    def test_bootstrap_is_deterministic(self):
        deltas = [-0.02, 0.0, 0.01, 0.03, 0.05]
        self.assertEqual(pooling.bootstrap(deltas), pooling.bootstrap(deltas))


if __name__ == "__main__":
    unittest.main()
