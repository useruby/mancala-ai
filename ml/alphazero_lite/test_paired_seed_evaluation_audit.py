import unittest

from ml.alphazero_lite.run_paired_seed_evaluation_audit import (
    BUDGETS,
    compliance_table,
    effective_cpuct,
    historical_seed,
)


class PairedSeedEvaluationAuditTests(unittest.TestCase):
    def test_historical_control_leaks_lane_identity(self):
        self.assertNotEqual(
            historical_seed(base_seed=42, lane="current", opening_index=0, ply=0),
            historical_seed(base_seed=42, lane="candidate", opening_index=0, ply=0),
        )

    def test_profile_and_budget_definitions_are_fixed(self):
        self.assertEqual(6, len(BUDGETS))
        self.assertEqual(0.9, effective_cpuct("768:768", 1.25, {"768:768": 0.9}))
        self.assertEqual(1.25, effective_cpuct("384:256", 1.25, {"768:768": 0.9}))

    def test_compliance_marks_standard_arena_as_canonical(self):
        table = {row["path"]: row["status"] for row in compliance_table()}
        self.assertEqual("compliant: per-search v1 seed context", table["arena.py"])
        self.assertEqual("compliant", table["run_paired_seed_evaluation_audit.py"])
