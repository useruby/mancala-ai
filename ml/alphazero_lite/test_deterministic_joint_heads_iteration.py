"""Focused tests for the deterministic joint-head iteration protocol."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    canonical_manifest_hash,
    classify_completed_evidence,
    compare_runs,
    freeze_joint_heads,
    game_split,
    probe_gate,
    select_search_probe_indexes,
    sha256_file,
    verify_manifest,
    write_json,
)
from ml.alphazero_lite.train import PolicyValueNet, input_size_for_encoding


class DeterministicJointHeadsIterationTest(unittest.TestCase):
    def _rows(self) -> list[dict]:
        rows = []
        for game in range(20):
            for ply in range(3):
                rows.append(
                    {
                        "game_index": game,
                        "value": [-1.0, 0.0, 1.0][game % 3],
                        "seat_context": f"player_{game % 2}",
                        "game_length": 20 + game,
                        "state": [0.0] * 27,
                        "policy": [1.0, 0, 0, 0, 0, 0],
                    }
                )
        return rows

    def test_game_split_is_seeded_disjoint_and_game_level(self) -> None:
        train, validation = game_split(self._rows(), 42)
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual((train, validation), game_split(self._rows(), 42))

    def test_freeze_joint_heads_leaves_only_specialized_heads_trainable(self) -> None:
        model = PolicyValueNet(
            (96, 3), "residual_v3", input_size_for_encoding("kalah_v3")
        )
        names = freeze_joint_heads(model)
        self.assertEqual(8, len(names))
        self.assertTrue(all("residual_layers" not in name for name in names))
        self.assertTrue(
            all(
                parameter.requires_grad == (name in names)
                for name, parameter in model.named_parameters()
            )
        )

    def test_reproduction_accepts_matching_hashes_and_tolerant_predictions(
        self,
    ) -> None:
        base = {
            "parameter_sha256": {"x": "a"},
            "checkpoint_sha256": "one",
            "artifact_weights_sha256": "two",
            "captures": {
                "initialization": {
                    "parameter_sha256": {"x": "a"},
                    "optimizer_state_sha256": "optimizer",
                    "checkpoint_sha256": "checkpoint",
                }
            },
            "validation_logits": np.array([[1.0, 0.0]]),
            "validation_values": np.array([[0.2]]),
        }
        identical = {
            **base,
            "validation_logits": base["validation_logits"].copy(),
            "validation_values": base["validation_values"].copy(),
        }
        self.assertTrue(compare_runs(base, identical)["passes"])
        close = {
            **identical,
            "parameter_sha256": {"x": "b"},
            "validation_values": np.array([[0.20000001]]),
        }
        self.assertTrue(compare_runs(base, close)["passes"])
        divergent = {**close, "validation_logits": np.array([[0.0, 1.0]])}
        self.assertFalse(compare_runs(base, divergent)["passes"])

    def test_manifest_hash_includes_replay_path_and_verification_checks_files(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            replay, split, plan = (
                root / "replay.jsonl",
                root / "split.npy",
                root / "plan.npy",
            )
            replay.write_text("row\n", encoding="utf-8")
            split.write_bytes(b"split")
            plan.write_bytes(b"plan")
            manifest = {
                "replay_path": str(replay),
                "files": {
                    str(path): sha256_file(path) for path in (replay, split, plan)
                },
            }
            manifest["manifest_sha256_excluding_this_field"] = canonical_manifest_hash(
                manifest
            )
            manifest_path = root / "manifest.json"
            write_json(manifest_path, manifest)
            self.assertEqual(str(replay), verify_manifest(manifest_path)["replay_path"])
            changed_path = dict(manifest)
            changed_path["replay_path"] = "different"
            self.assertNotEqual(
                canonical_manifest_hash(manifest), canonical_manifest_hash(changed_path)
            )
            for path in (replay, split, plan):
                path.write_bytes(path.read_bytes() + b"!")
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    verify_manifest(manifest_path)
                path.write_bytes(path.read_bytes()[:-1])

    def test_search_probe_selection_is_deterministic_unique_and_stratified(
        self,
    ) -> None:
        rows = []
        for index in range(400):
            rows.append(
                {
                    "game_index": index // 2,
                    "value": [-1.0, 0.0, 1.0][index % 3],
                    "seat_context": f"player_{index % 2}",
                    "phase": ("opening", "midgame", "late")[index % 3],
                    "game_length": 20 + index % 100,
                    "state": [float(index)] * 27,
                }
            )
        source = np.arange(len(rows), dtype=np.int64)
        first, first_manifest = select_search_probe_indexes(rows, source, 42)
        second, second_manifest = select_search_probe_indexes(rows, source, 42)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(len(first), len(set(first_manifest["state_hashes"])))
        self.assertIn("player_0", " ".join(first_manifest["stratum_counts"]))
        self.assertIn("opening", " ".join(first_manifest["stratum_counts"]))
        self.assertIn("-1", " ".join(first_manifest["stratum_counts"]))

    def test_probe_gate_and_completed_evidence_require_all_stages(self) -> None:
        policy = {
            "legal_failures": 0,
            "policy_kl": 0.1,
            "teacher_puct_top1_agreement": 0.9,
            "changed_raw_top1_rate_vs_current": 0.01,
        }
        current = {"policy_kl": 0.2, "teacher_puct_top1_agreement": 0.8}
        values, current_values = (
            {"mae": 0.1, "sign_accuracy": 0.8},
            {"mae": 0.2, "sign_accuracy": 0.8},
        )
        search = {
            "changed_move_rate_by_budget_context": {
                key: {"changed_rate_vs_current": 0.01}
                for key in ("768:768", "1200:1200", "1200:256")
            }
        }
        self.assertTrue(
            probe_gate(
                policy, current, values, current_values, search, {"passes": True}
            )
        )
        self.assertFalse(
            probe_gate(
                policy, current, values, current_values, search, {"passes": False}
            )
        )
        self.assertEqual(
            "deterministic_joint_heads_experiment_incomplete",
            classify_completed_evidence(
                {"reproducibility": True, "probes": True, "medium": None}
            ),
        )
        self.assertEqual(
            "deterministic_joint_heads_candidate",
            classify_completed_evidence(
                {
                    "reproducibility": True,
                    "probes": True,
                    "medium": True,
                    "fixed_large": True,
                    "heldout": True,
                    "deterministic_gate": True,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
