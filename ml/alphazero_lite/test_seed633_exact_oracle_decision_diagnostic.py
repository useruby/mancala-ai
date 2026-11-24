"""Regression checks for exact-oracle state/action handling."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed633_exact_oracle_decision_diagnostic import (
    SOURCE_ANCHOR_SHA256,
    authenticate_source,
    derive,
    directional_category,
)
from ml.alphazero_lite.verify_seed633_exact_oracle_decision_diagnostic import verify

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_SOURCE = ROOT / "docs/data/seed633-teacher-target-learning/attempt-0005"
CORRECTED_OUTPUT = (
    ROOT / "docs/data/seed633-exact-oracle-decision-diagnostic/attempt-0003"
)


class ExactOracleDecisionDiagnosticTests(unittest.TestCase):
    """Protect tied-optimum handling, regret, and cohort integrity."""

    def setUp(self) -> None:
        self.cohort = [
            {
                "canonical_identity": "state-a",
                "teacher_type": "exact_root_tablebase",
                "teacher_evidence": {"exact_action_margins": {"0": 5, "1": 5, "2": -1}},
                "legal_moves": [0, 1, 2],
                "exact_optimal_actions": [0, 1],
                "source_opening_group": "source:opening",
                "membership_class": "neither_training_membership",
                "phase": "at_most_16",
            }
        ]
        self.metrics = [
            {
                "canonical_identity": "state-a",
                "teacher_entropy_nats": 0.5,
                "B512": {
                    "legal_policy": [0.4, 0.3, 0.3, 0.0, 0.0, 0.0],
                    "teacher_target_cross_entropy_nats": 0.5,
                },
                "S615": {
                    "legal_policy": [0.3, 0.4, 0.3, 0.0, 0.0, 0.0],
                    "teacher_target_cross_entropy_nats": 0.4,
                },
            }
        ]

    def test_all_tied_optimal_actions_count_as_correct(self) -> None:
        result = derive(self.cohort, self.metrics)[0]
        self.assertEqual(result["B512"]["exact_optimal_top_action"], True)
        self.assertEqual(result["S615"]["exact_optimal_top_action"], True)
        self.assertEqual(result["S615"]["top_policy_action_regret"], 0)

    def test_altered_oracle_label_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["exact_optimal_actions"] = [0]
        with self.assertRaisesRegex(
            ValueError, "oracle_action_value_correspondence_invalid"
        ):
            derive(cohort, self.metrics)

    def test_altered_legal_action_mask_rejected(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["legal_moves"] = [0, 1]
        with self.assertRaisesRegex(
            ValueError, "oracle_action_value_correspondence_invalid"
        ):
            derive(cohort, self.metrics)

    def test_model_policy_identity_order_mismatch_rejected(self) -> None:
        metrics = copy.deepcopy(self.metrics)
        metrics[0]["canonical_identity"] = "other-state"
        with self.assertRaisesRegex(ValueError, "cohort_metric_identity_mismatch"):
            derive(self.cohort, metrics)

    def test_cohort_membership_is_preserved(self) -> None:
        cohort = copy.deepcopy(self.cohort)
        cohort[0]["membership_class"] = "S615_only"
        self.assertEqual(
            derive(cohort, self.metrics)[0]["membership_class"], "S615_only"
        )

    def test_all_imitation_and_quality_directions_are_classified(self) -> None:
        imitation_deltas = {"improved": -0.1, "worse": 0.1, "neutral": 0.0}
        quality_deltas = {"improved": -1, "worse": 1, "neutral": 0}
        for imitation, ce_delta in imitation_deltas.items():
            for quality, regret_delta in quality_deltas.items():
                with self.subTest(imitation=imitation, quality=quality):
                    actual = directional_category(ce_delta, regret_delta, 0, 0.0)
                    self.assertEqual(
                        actual, f"{imitation}_imitation_and_{quality}_exact_quality"
                    )

    def test_neutral_teacher_imitation_is_neutral_not_deterioration(self) -> None:
        self.assertEqual(
            directional_category(0.0, 0.0, 0, 0.0),
            "neutral_imitation_and_neutral_exact_quality",
        )

    def test_source_anchor_rejects_self_consistent_substitution(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            altered_source = Path(temporary)
            for name in SOURCE_ANCHOR_SHA256:
                (altered_source / name).symlink_to(HISTORICAL_SOURCE / name)
            for name in SOURCE_ANCHOR_SHA256:
                with self.subTest(identity=name):
                    target = altered_source / name
                    target.unlink()
                    target.write_text(
                        json.dumps(
                            {
                                "source_sha256": {
                                    anchor: "attacker-controlled"
                                    for anchor in SOURCE_ANCHOR_SHA256
                                }
                            }
                        )
                    )
                    with self.assertRaisesRegex(
                        ValueError, f"historical_source_anchor_mismatch:{name}"
                    ):
                        authenticate_source(altered_source)
                    target.unlink()
                    target.symlink_to(HISTORICAL_SOURCE / name)

    def test_verifier_rejects_altered_source_freeze_even_with_rewritten_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            altered_source = Path(temporary)
            (altered_source / "freeze.json").write_text(
                json.dumps({"source_sha256": {"cohort.jsonl": "rewritten"}})
            )
            with self.assertRaisesRegex(
                ValueError, "historical_source_anchor_mismatch:freeze.json"
            ):
                verify(altered_source, CORRECTED_OUTPUT)

    def test_verifier_reconstructs_summary_and_rejects_altered_statistic(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temporary:
            output = Path(temporary)
            for filename in ("freeze.json", "per-state-results.jsonl"):
                (output / filename).symlink_to(CORRECTED_OUTPUT / filename)
            summary = json.loads((CORRECTED_OUTPUT / "summary.json").read_bytes())
            summary["overall_cluster_bootstrap"]["exact_optimal_action_mass"][
                "mean"
            ] += 0.01
            (output / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(
                ValueError, "summary_mismatch:overall_cluster_bootstrap"
            ):
                verify(HISTORICAL_SOURCE, output)


if __name__ == "__main__":
    unittest.main()
