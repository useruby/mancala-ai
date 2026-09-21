from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite import run_value_target_default_vs_sharpened as experiment
from ml.alphazero_lite.self_play import outcome_for_player
from ml.alphazero_lite.train import load_jsonl_replay


class ValueTargetDefaultVsSharpenedTest(unittest.TestCase):
    def row(
        self, *, winner: int | None = 0, player: int = 0, move_index: int = 3
    ) -> dict:
        return {
            "state": [0.1] * 15,
            "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "value": 0.36,
            "policy_target_mode": "sharpened",
            "value_target_mode": "sharpened",
            "winner": winner,
            "player": player,
            "move_index": move_index,
            "teacher_search_metadata": {"simulations": 1200},
            "game_metadata": {"game_index": 4},
        }

    def test_default_relabel_uses_canonical_outcome_helper(self):
        row = self.row(winner=1, player=0)
        self.assertEqual(outcome_for_player(1, 0), experiment.relabel_row(row)["value"])

    def test_win_loss_draw_perspective_is_correct(self):
        self.assertEqual(
            1.0, experiment.relabel_row(self.row(winner=0, player=0))["value"]
        )
        self.assertEqual(
            -1.0, experiment.relabel_row(self.row(winner=1, player=0))["value"]
        )
        self.assertEqual(
            0.0, experiment.relabel_row(self.row(winner=None, player=0))["value"]
        )

    def test_extra_turn_metadata_does_not_change_final_outcome_perspective(self):
        row = self.row(winner=0, player=0)
        row["extra_turn"] = True
        self.assertEqual(1.0, experiment.relabel_row(row)["value"])

    def test_relabel_preserves_state_policy_order_and_all_other_fields(self):
        rows = [self.row(move_index=2), self.row(winner=1, player=1, move_index=9)]
        with tempfile.TemporaryDirectory() as directory:
            source, derived = (
                Path(directory) / "input.jsonl",
                Path(directory) / "output.jsonl",
            )
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            experiment.relabel_dataset(source, derived)
            output = [
                json.loads(line)
                for line in derived.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(
            [row["move_index"] for row in rows], [row["move_index"] for row in output]
        )
        for original, relabelled in zip(rows, output, strict=True):
            self.assertEqual(
                json.dumps(original["state"]), json.dumps(relabelled["state"])
            )
            self.assertEqual(
                json.dumps(original["policy"]), json.dumps(relabelled["policy"])
            )
            experiment.verify_relabel_pair(original, relabelled)

    def test_only_value_and_value_target_mode_can_change(self):
        original, relabelled = self.row(), experiment.relabel_row(self.row())
        self.assertEqual(
            {"value", "value_target_mode"},
            {key for key in original if original[key] != relabelled[key]},
        )

    def test_relabel_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text(json.dumps(self.row()) + "\n", encoding="utf-8")
            first, second = (
                Path(directory) / "first.jsonl",
                Path(directory) / "second.jsonl",
            )
            experiment.relabel_dataset(source, first)
            experiment.relabel_dataset(source, second)
            self.assertEqual(
                experiment.sha256_file(first), experiment.sha256_file(second)
            )

    def test_training_config_changes_only_fresh_value_target_mode(self):
        plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-value-target-default-vs-sharpened/plan.json"
            ).read_text(encoding="utf-8")
        )
        command = experiment.training_command(
            Path("fresh-default.jsonl"), Path("checkpoint.npz"), plan
        )
        self.assertEqual("sharpened", plan["training"]["value_target_mode"])
        self.assertEqual("default", command[command.index("--value-target-mode") + 1])
        self.assertEqual(
            "sharpened", command[command.index("--policy-target-mode") + 1]
        )
        self.assertEqual(
            "default,sharpened,sharpened,sharpened,sharpened",
            command[command.index("--replay-value-target-modes") + 1],
        )

    def test_every_treatment_command_has_exactly_1084_updates(self):
        plan = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "docs/data/alphazero-lite-value-target-default-vs-sharpened/plan.json"
            ).read_text(encoding="utf-8")
        )
        command = experiment.training_command(
            Path("fresh-default.jsonl"), Path("checkpoint.npz"), plan
        )
        self.assertEqual("1084", command[command.index("--max-optimizer-updates") + 1])
        self.assertEqual("final", command[command.index("--final-checkpoint") + 1])

    def test_loader_validates_default_fresh_and_sharpened_history_per_source(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh, history = (
                Path(directory) / "fresh.jsonl",
                Path(directory) / "history.jsonl",
            )
            fresh_row = self.row()
            fresh_row.update({"value": 1.0, "value_target_mode": "default"})
            history_row = self.row()
            fresh.write_text(json.dumps(fresh_row) + "\n", encoding="utf-8")
            history.write_text(json.dumps(history_row) + "\n", encoding="utf-8")
            _x, _p, values, indexes = load_jsonl_replay(
                [fresh, history],
                [1, 1],
                policy_target_mode="sharpened",
                value_target_mode="default",
                replay_value_target_modes=["default", "sharpened"],
            )
        self.assertEqual(1.0, values[0, 0])
        self.assertAlmostEqual(0.36, values[1, 0])
        self.assertEqual([0, 1], indexes.tolist())

    def test_canonical_promotion_suites_cannot_be_invoked(self):
        source = Path(experiment.__file__).read_text(encoding="utf-8")
        self.assertNotIn("promotion_gate", source)
        self.assertNotIn("canonical_prefilter", source)


if __name__ == "__main__":
    unittest.main()
