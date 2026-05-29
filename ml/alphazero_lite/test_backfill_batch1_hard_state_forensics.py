from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from ml.alphazero_lite import backfill_batch1_hard_state_forensics as module


class BackfillBatch1HardStateForensicsTest(TestCase):
    def test_validate_suite_path_rejects_holdout_suite(self) -> None:
        root = module.repo_root()
        with self.assertRaisesRegex(
            ValueError, "must not be the hard-state validation holdout"
        ):
            module.validate_suite_path(root, root / module.DEFAULT_HOLDOUT_SUITE)

    def test_build_plan_blocks_without_train_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            gate_root = tmp_path / "gates"
            variant_dir = gate_root / "exp-a"
            variant_dir.mkdir(parents=True)
            report_path = variant_dir / "local_promotion.json"
            report_path.write_text(
                json.dumps({"candidate_path": "/tmp/candidate-a"}), encoding="utf-8"
            )

            plan = module.build_plan(
                root=module.repo_root(),
                gate_reports=[report_path],
                out_root=tmp_path / "out",
                suite_path=None,
                current_path="storage/ai/alphazero_lite/current",
                mcts_simulations=1200,
                teacher_simulations=1800,
                artifact_simulations=384,
                seed=42,
            )

            self.assertEqual(
                "no train-only forensic suite supplied", plan["blocked_reason"]
            )
            self.assertEqual(
                "blocked_missing_train_suite", plan["variants"][0]["status"]
            )
