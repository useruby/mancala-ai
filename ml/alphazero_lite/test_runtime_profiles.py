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

    def test_unknown_or_malformed_budget_keys_fail_loudly(self):
        with self.assertRaises(ValueError):
            self.profile({"not-a-budget": 0.0})
        with self.assertRaises(ValueError):
            resolve_runtime_profile(self.profile(), "768:not-a-number")


if __name__ == "__main__":
    unittest.main()
