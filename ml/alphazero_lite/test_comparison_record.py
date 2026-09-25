import unittest
import json
import tempfile
from pathlib import Path

from ml.alphazero_lite.comparison_record import (
    ComparisonRecordError,
    _value_at,
    load_record,
    paired_bootstrap,
    canonical_gate_candidate,
    validate_matched_records,
)
from ml.alphazero_lite.generation_record import load_record as load_generation_record


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

    def test_seed461_sibling_comparison_preserves_rejected_a_and_blocks_b_gate(self):
        root = Path(__file__).resolve().parents[2]
        comparison = load_record(
            root
            / "docs/data/alphazero-lite-generation-comparisons"
            / "seed461-exact-root-target-selected-vs-uniform.json"
        )
        self.assertEqual(
            "exact_root_optimal_set_targets_improve_policy_only",
            comparison["conclusion"]["classification"],
        )
        self.assertFalse(comparison["scope"]["canonical_gate_run"])
        comparison_dir = root / "docs/data/alphazero-lite-generation-comparisons"
        baseline = load_generation_record(
            comparison_dir / comparison["pairs"][0]["baseline_record"]
        )
        treatment = load_generation_record(
            comparison_dir / comparison["pairs"][0]["treatment_record"]
        )
        self.assertEqual("rejected", baseline["status"])
        self.assertEqual("not_run", treatment["self_play"]["status"])
        self.assertEqual(
            "4f575f92e83965949388b83863097c795b2182c54fb4017eca709cbd07c77014",
            treatment["candidate"]["weights"]["sha256"],
        )
        with self.assertRaisesRegex(ComparisonRecordError, "not_qualified"):
            canonical_gate_candidate(comparison, base_dir=comparison_dir)

    def test_opening_disagreement_refresh_comparison_is_matched_and_not_promotable(
        self,
    ):
        root = Path(__file__).resolve().parents[2]
        comparison_dir = root / "docs/data/alphazero-lite-generation-comparisons"
        comparison = load_record(
            comparison_dir
            / "seed461-opening-disagreement-historical-vs-seed455-refresh.json"
        )
        self.assertEqual(
            "opening_disagreement_refresh_no_clear_benefit",
            comparison["conclusion"]["classification"],
        )
        self.assertEqual(
            "historical",
            comparison["controlled_difference"][
                "replay.sources.opening_disagreement.policy_teacher"
            ]["baseline"],
        )
        self.assertFalse(comparison["scope"]["canonical_gate_run"])
        with self.assertRaisesRegex(ComparisonRecordError, "not_qualified"):
            canonical_gate_candidate(comparison, base_dir=comparison_dir)

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

    def test_canonical_gate_requires_qualifying_comparison_and_treatment_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = {
                "schema": "azlite_generation_record_v1",
                "generation_id": "a",
                "status": "rejected",
                "parent": {"weights_sha256": "a" * 64},
                "self_play": {
                    "status": "not_run",
                    "config": {},
                    "seeds": [],
                    "artifact": None,
                    "compute": {},
                },
                "replay": {"sources": []},
                "training": {
                    "status": "not_run",
                    "config": {},
                    "seed": None,
                    "optimizer_updates": None,
                    "checkpoint_policy": None,
                    "metrics": {},
                    "compute": {},
                },
                "candidate": {
                    "version": "a",
                    "checkpoint": None,
                    "weights": {"sha256": "b" * 64},
                    "metadata": {"sha256": "c" * 64},
                },
                "diagnostics": {},
                "canonical_evaluation": {},
                "promotion": {
                    "decision": "rejected",
                    "failure_reasons": [{"code": "x"}],
                    "gate_report": None,
                },
                "compute": {},
                "artifacts": [],
                "provenance_notes": [],
                "comparison_controls": {"seed": 461},
            }
            treatment = {
                **baseline,
                "generation_id": "b",
                "status": "evaluated",
                "candidate": {
                    "version": "b",
                    "checkpoint": None,
                    "weights": {"sha256": "d" * 64},
                    "metadata": {"sha256": "e" * 64},
                },
                "promotion": {
                    "decision": "not_evaluated",
                    "failure_reasons": [],
                    "gate_report": None,
                },
            }
            (root / "a.json").write_text(json.dumps(baseline), encoding="utf-8")
            (root / "b.json").write_text(json.dumps(treatment), encoding="utf-8")
            comparison = {
                "schema": "azlite_generation_comparison_v1",
                "comparison_id": "x",
                "allowed_record_differences": ["promotion"],
                "controlled_difference": {
                    "comparison_controls.seed": {"baseline": 461, "treatment": 461}
                },
                "pairs": [
                    {
                        "pair_id": "461",
                        "baseline_record": "a.json",
                        "treatment_record": "b.json",
                    }
                ],
                "evidence": {
                    "path": "unavailable",
                    "required": False,
                    "sha256": "a" * 64,
                },
                "scope": {"canonical_gate_run": False},
                "conclusion": {
                    "classification": "exact_root_optimal_set_targets_regress"
                },
            }
            with self.assertRaisesRegex(ComparisonRecordError, "not_qualified"):
                canonical_gate_candidate(comparison, base_dir=root)
            comparison["conclusion"]["classification"] = (
                "exact_root_optimal_set_targets_improve_seed461"
            )
            self.assertEqual(
                "d" * 64,
                canonical_gate_candidate(comparison, base_dir=root)["weights"][
                    "sha256"
                ],
            )


if __name__ == "__main__":
    unittest.main()
