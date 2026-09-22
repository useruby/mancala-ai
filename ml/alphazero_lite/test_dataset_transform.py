import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.dataset_transform import relabel_value_targets
from ml.alphazero_lite.self_play import outcome_for_player


class DatasetTransformTest(unittest.TestCase):
    def test_default_relabel_uses_outcome_and_preserves_data_deterministically(self):
        rows = [
            {
                "state": [1],
                "policy": [1, 0, 0, 0, 0, 0],
                "value": 0.2,
                "value_target_mode": "sharpened",
                "winner": 0,
                "player": 0,
                "move_index": 1,
                "metadata": {"x": 1},
            },
            {
                "state": [2],
                "policy": [0, 1, 0, 0, 0, 0],
                "value": -0.2,
                "value_target_mode": "sharpened",
                "winner": None,
                "player": 1,
                "move_index": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            first, second = (
                Path(directory) / "first.jsonl",
                Path(directory) / "second.jsonl",
            )
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            audit = relabel_value_targets(source, first, mode="default")
            repeat = relabel_value_targets(source, second, mode="default")
            transformed = [
                json.loads(line)
                for line in first.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(audit["transformed_sha256"], repeat["transformed_sha256"])
        self.assertEqual("passed", audit["state_policy_equality"])
        self.assertEqual(2, audit["row_count"])
        for original, derived in zip(rows, transformed, strict=True):
            self.assertEqual(original["state"], derived["state"])
            self.assertEqual(original["policy"], derived["policy"])
            self.assertEqual(
                outcome_for_player(original["winner"], original["player"]),
                derived["value"],
            )


if __name__ == "__main__":
    unittest.main()
