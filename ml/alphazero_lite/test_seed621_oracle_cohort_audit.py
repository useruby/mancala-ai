"""Focused negative tests for Seed 621 report invariants."""

from __future__ import annotations

import copy
import unittest

from ml.alphazero_lite.seed557_retrospective import canonical_hash
from ml.alphazero_lite.seed621_oracle_cohort_audit import verify_report


def sample_report() -> dict:
    state = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    return {
        "inventory": [
            {
                "state_id": canonical_hash(state),
                "state": state,
                "exact_action_margins_to_move_perspective": {
                    str(move): 0 for move in range(6)
                },
                "eligible_by_exact_membership_rule": True,
                "contamination_status": "unresolved_nonmembership_not_proven",
                "exclusion_reasons": [],
            }
        ]
    }


class Seed621AuditNegativeTests(unittest.TestCase):
    def test_rejects_duplicate_state_inventory(self) -> None:
        report = sample_report()
        report["inventory"].append(copy.deepcopy(report["inventory"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_inventory_state"):
            verify_report(report)

    def test_rejects_changed_state_identity(self) -> None:
        report = sample_report()
        report["inventory"][0]["state"]["current_player"] = 1
        with self.assertRaisesRegex(ValueError, "state_identity_mismatch"):
            verify_report(report)

    def test_rejects_incomplete_legal_action_coverage(self) -> None:
        report = sample_report()
        report["inventory"][0]["exact_action_margins_to_move_perspective"].pop("5")
        with self.assertRaisesRegex(ValueError, "oracle_coverage_mismatch"):
            verify_report(report)

    def test_rejects_claim_that_unresolved_nonmembership_is_clear(self) -> None:
        report = sample_report()
        report["inventory"][0]["contamination_status"] = "clear"
        with self.assertRaisesRegex(ValueError, "explicitly_unresolved"):
            verify_report(report)

    def test_rejects_eligible_state_with_exclusion_reason(self) -> None:
        report = sample_report()
        report["inventory"][0]["exclusion_reasons"] = ["protected_holdout_overlap"]
        with self.assertRaisesRegex(ValueError, "eligible_state_has_exclusion_reason"):
            verify_report(report)


if __name__ == "__main__":
    unittest.main()
