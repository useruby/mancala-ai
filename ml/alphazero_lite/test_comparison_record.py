import unittest
from pathlib import Path

from ml.alphazero_lite.comparison_record import (
    ComparisonRecordError,
    _value_at,
    load_record,
    paired_bootstrap,
    validate_matched_records,
)


class ComparisonRecordTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        self.assertEqual(
            paired_bootstrap([0.1, -0.2, 0.3], seed=353),
            paired_bootstrap([0.1, -0.2, 0.3], seed=353),
        )

    def test_matched_records_reject_changed_training_controls(self):
        baseline = {
            "generation_id": "a",
            "training": {
                "config": {"batch_size": 512, "value_target_mode": "sharpened"}
            },
            "candidate": {},
            "diagnostics": {},
            "artifacts": [],
            "compute": {},
            "provenance_notes": [],
            "status": "candidate_ready",
            "comparison_controls": {"batch_size": 512},
        }
        treatment = {
            **baseline,
            "generation_id": "b",
            "training": {"config": {"batch_size": 256, "value_target_mode": "default"}},
            "comparison_controls": {"batch_size": 256},
        }
        with self.assertRaisesRegex(ComparisonRecordError, "controls"):
            validate_matched_records(
                baseline,
                treatment,
                allowed_differences={"training.config.value_target_mode"},
            )

    def test_committed_comparison_references_valid_diagnostic_records(self):
        root = Path(__file__).resolve().parents[2]
        comparison = load_record(
            root
            / "docs/data/alphazero-lite-generation-comparisons"
            / "value-target-default-vs-sharpened.json"
        )
        self.assertEqual(6, len(comparison["pairs"]))
        self.assertFalse(comparison["scope"]["canonical_gate_run"])
        self.assertFalse(comparison["scope"]["promotion_performed"])

    def test_controlled_difference_uses_named_replay_source(self):
        baseline = {
            "replay": {"sources": [{"name": "fresh", "value_target_mode": "sharpened"}]}
        }
        treatment = {
            "replay": {"sources": [{"name": "fresh", "value_target_mode": "default"}]}
        }
        self.assertEqual(
            "sharpened", _value_at(baseline, "replay.sources.fresh.value_target_mode")
        )
        self.assertEqual(
            "default", _value_at(treatment, "replay.sources.fresh.value_target_mode")
        )


if __name__ == "__main__":
    unittest.main()
