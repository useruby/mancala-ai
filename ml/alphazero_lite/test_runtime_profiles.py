import unittest

from ml.alphazero_lite.runtime_profiles import (
    resolve_runtime_profile,
    runtime_profile_definition,
)


class RuntimeProfileTests(unittest.TestCase):
    def profile(self, overrides=None):
        return runtime_profile_definition(
            name="budget_conditioned_tactical_profile",
            default_tactical_root_bias=0.10,
            tactical_root_bias_overrides=overrides or {"384:256": 0.00},
            default_c_puct=1.25,
            c_puct_overrides={"768:768": 0.90},
        )

    def test_tactical_override_changes_profile_hash(self):
        self.assertNotEqual(
            self.profile()["hash"], self.profile({"768:768": 0.00})["hash"]
        )
        self.assertNotEqual(
            self.profile()["runtime_treatment_hash"],
            self.profile({"768:768": 0.00})["runtime_treatment_hash"],
        )

    def test_identity_fields_do_not_change_treatment_hash(self):
        first = self.profile()
        second = runtime_profile_definition(
            name="display-only-renamed-profile",
            default_tactical_root_bias=0.10,
            tactical_root_bias_overrides={"384:256": 0.00},
            default_c_puct=1.25,
            c_puct_overrides={"768:768": 0.90},
        )
        self.assertNotEqual(first["hash"], second["hash"])
        self.assertEqual(
            first["runtime_treatment_hash"], second["runtime_treatment_hash"]
        )

    def test_cpuct_changes_treatment_hash(self):
        changed = runtime_profile_definition(
            name="budget_conditioned_tactical_profile",
            default_tactical_root_bias=0.10,
            tactical_root_bias_overrides={"384:256": 0.00},
            default_c_puct=1.30,
            c_puct_overrides={"768:768": 0.90},
        )
        self.assertNotEqual(
            self.profile()["runtime_treatment_hash"], changed["runtime_treatment_hash"]
        )

    def test_unknown_or_malformed_budget_keys_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.profile({"not-a-budget": 0.0})
        with self.assertRaises(ValueError):
            resolve_runtime_profile(self.profile(), "768:not-a-number")


if __name__ == "__main__":
    unittest.main()
