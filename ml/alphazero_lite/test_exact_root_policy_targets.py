import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.exact_root_policy_targets import (
    OPTIMAL_SET_UNIFORM,
    SELECTED_ONE_HOT,
    exact_root_policy_target,
    relabel_exact_root_policy_jsonl,
)


class ExactRootPolicyTargetTest(unittest.TestCase):
    def test_targets_are_deterministic_and_legal(self):
        self.assertEqual(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            exact_root_policy_target(
                selected_action=1, optimal_actions=[1], mode=OPTIMAL_SET_UNIFORM
            ),
        )
        self.assertEqual(
            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            exact_root_policy_target(
                selected_action=2, optimal_actions=[0, 2], mode=OPTIMAL_SET_UNIFORM
            ),
        )
        target = exact_root_policy_target(
            selected_action=4, optimal_actions=[1, 3, 4], mode=OPTIMAL_SET_UNIFORM
        )
        self.assertEqual([0.0, 1 / 3, 0.0, 1 / 3, 1 / 3, 0.0], target)
        self.assertEqual(1.0, sum(target))
        self.assertEqual(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            exact_root_policy_target(
                selected_action=2, optimal_actions=[0, 2], mode=SELECTED_ONE_HOT
            ),
        )

    def test_relabel_only_changes_tied_exact_rows(self):
        rows = [
            self._row([0], 0),
            self._row([1, 4], 4),
            {"state": [3], "policy": [0, 0, 1, 0, 0, 0], "value": 0.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "output.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit = relabel_exact_root_policy_jsonl(source, output)
            transformed = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(3, audit["source_rows"])
        self.assertEqual(1, audit["policy_rows_changed"])
        self.assertEqual(rows[0], transformed[0])
        self.assertEqual([0.0, 0.5, 0.0, 0.0, 0.5, 0.0], transformed[1]["policy"])
        self.assertEqual(rows[2], transformed[2])

    @staticmethod
    def _row(optimal, selected):
        margins = {str(action): 1 for action in optimal}
        margins.setdefault("5", 0)
        policy = [0.0] * 6
        policy[selected] = 1.0
        return {
            "state": [1],
            "policy": policy,
            "stored_policy_target": policy,
            "value": 0.3,
            "teacher_source": "exact_root_tablebase",
            "exact_selected_action": selected,
            "exact_optimal_actions": optimal,
            "exact_action_margins": margins,
            "policy_target_actual_mode": "exact_root_one_hot",
        }
