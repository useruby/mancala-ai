import unittest
from pathlib import Path

from ml.alphazero_lite.runtime_search_policy import (
    RuntimeSearchPolicyError,
    resolve_strength_comparison_runtime_contract,
)


class RuntimeSearchPolicyComparisonTest(unittest.TestCase):
    artifact_dir = Path(__file__).resolve().parents[2] / "model-artifact/current"

    def test_incumbent_contract_resolves_production_native_identities(self):
        contract = resolve_strength_comparison_runtime_contract(
            current_artifact=self.artifact_dir,
            challenger_artifact=self.artifact_dir,
        )
        assert contract is not None
        self.assertEqual(
            "puct_exact_root_hybrid", contract["runtime_search_policy_mode"]
        )
        self.assertEqual(16, contract["exact_root_solve_threshold"])
        self.assertEqual(
            "native_kvtb_root_action_probe_v1", contract["exact_root_solver"]
        )
        self.assertEqual(
            "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
            contract["exact_root_native_probe_sha256"],
        )
        self.assertEqual(
            "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
            contract["exact_root_tablebase_sha256"],
        )
        self.assertEqual("final_score_margin", contract["exact_root_objective"])
        self.assertEqual(
            "highest_legal_network_prior_then_lowest_move_index",
            contract["exact_root_tie_rule"],
        )

    def test_mismatched_threshold_or_native_path_fails_closed(self):
        with self.assertRaisesRegex(
            RuntimeSearchPolicyError, "arena_runtime_search_policy_mismatch"
        ):
            resolve_strength_comparison_runtime_contract(
                current_artifact=self.artifact_dir,
                challenger_artifact=self.artifact_dir,
                exact_root_threshold=15,
            )
        with self.assertRaisesRegex(
            RuntimeSearchPolicyError, "arena_runtime_search_policy_mismatch"
        ):
            resolve_strength_comparison_runtime_contract(
                current_artifact=self.artifact_dir,
                challenger_artifact=self.artifact_dir,
                native_probe="missing-native-probe",
            )
