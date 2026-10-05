import unittest
import hashlib
import tempfile
from pathlib import Path

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed418_native_root_handoff import (
    canonical_hash,
)
from ml.alphazero_lite.seed418_analysis import analyze, choose_exact_action
from ml.alphazero_lite.verify_seed418_native_root_handoff import verify_bound_files


class Seed418NativeRootHandoffTest(unittest.TestCase):
    def test_canonical_state_hash_is_key_order_independent(self):
        state = KalahGame([1] * 12, [0, 0], 0).to_state()
        self.assertEqual(
            canonical_hash(state), canonical_hash(dict(reversed(list(state.items()))))
        )

    def test_exact_tie_uses_highest_prior_then_lowest_index(self):
        self.assertEqual(
            3,
            choose_exact_action({1: 4, 3: 4}, [0, 0.2, 0, 0.8, 0, 0], [1, 3]),
        )
        self.assertEqual(
            1,
            choose_exact_action(
                {1: 4, 3: 4, 4: -1}, [0, 0.2, 0, 0.2, 0.9, 0], [1, 3, 4]
            ),
        )
        self.assertEqual(
            1, choose_exact_action({1: 4, 3: 4}, [0, 0.2, 0, 0.2, 0, 0], [1, 3])
        )

    def test_exact_action_requires_complete_legal_coverage(self):
        with self.assertRaisesRegex(ValueError, "exact_action_coverage_mismatch"):
            choose_exact_action({1: 0}, [0] * 6, [1, 3])

    def test_terminal_state_is_not_given_a_fake_action(self):
        with self.assertRaisesRegex(ValueError, "terminal_state_has_no_legal_actions"):
            choose_exact_action({}, [0] * 6, [])

    def test_analysis_rejects_incomplete_cases_and_counts_wdl_inferiority(self):
        with self.assertRaisesRegex(ValueError, "incomplete_exact_coverage"):
            analyze([])
        rows = [
            {
                "coverage_complete": True,
                "action_margins": {"0": 2, "1": -1},
                "legal_moves": [0, 1],
                "search_selected_move": 1,
                "active_pit_stones": 18,
                "native_decision_latency_ms": 10.0,
            }
            for _ in range(128)
        ]
        result = analyze(rows)
        self.assertEqual(3.0, result["mean_final_margin_regret"])
        self.assertEqual(1.0, result["wdl_inferior_choice_rate"])
        self.assertEqual(
            "advance_to_separately_registered_runtime_experiment", result["decision"]
        )

    def test_verifier_rejects_altered_published_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text("original")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_bound_files(Path(directory), {"analysis.json": expected})
            path.write_text("altered")
            with self.assertRaisesRegex(ValueError, "published_evidence_hash_mismatch"):
                verify_bound_files(Path(directory), {"analysis.json": expected})


if __name__ == "__main__":
    unittest.main()
