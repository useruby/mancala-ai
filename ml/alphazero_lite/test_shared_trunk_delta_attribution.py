"""Tests for PR #191 tensor-family attribution invariants."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    _clustered_bootstrap,
    _context_c_puct,
    assert_decomposition,
    byte_identical,
    decoded_validation_manifest,
    hybrid_state,
    phase_c_metrics,
    phase_g_classification,
    load_complete_phase_f_arena_records,
    phase_f_suite_provenance,
    tensor_family,
)
from ml.alphazero_lite.self_play import encode_state


class SharedTrunkDeltaAttributionTest(unittest.TestCase):
    def test_tensor_family_covers_residual_v3_parameters(self) -> None:
        self.assertEqual("trunk", tensor_family("input_layer.weight"))
        self.assertEqual("trunk", tensor_family("residual_layers.2.1.bias"))
        self.assertEqual("heads", tensor_family("policy_hidden_layer.weight"))
        self.assertEqual("heads", tensor_family("value_head.bias"))

    def test_hybrids_copy_exact_source_tensors(self) -> None:
        current = {
            "input_layer.weight": torch.tensor([1.0]),
            "policy_head.weight": torch.tensor([2.0]),
        }
        joint = {
            "input_layer.weight": torch.tensor([3.0]),
            "policy_head.weight": torch.tensor([4.0]),
        }
        models = {
            "C": hybrid_state(
                current, joint, trunk_from_joint=False, heads_from_joint=False
            ),
            "T": hybrid_state(
                current, joint, trunk_from_joint=True, heads_from_joint=False
            ),
            "JH": hybrid_state(
                current, joint, trunk_from_joint=False, heads_from_joint=True
            ),
            "J": hybrid_state(
                current, joint, trunk_from_joint=True, heads_from_joint=True
            ),
        }
        assert_decomposition(current, joint, models)
        self.assertTrue(
            byte_identical(
                models["T"]["policy_head.weight"], current["policy_head.weight"]
            )
        )
        self.assertTrue(
            byte_identical(
                models["JH"]["input_layer.weight"], current["input_layer.weight"]
            )
        )
        self.assertTrue(
            byte_identical(
                models["J"]["policy_head.weight"], joint["policy_head.weight"]
            )
        )

    def test_manifest_rejects_non_round_trippable_validation_rows(self) -> None:
        state = {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
        row = {"state": encode_state(state, input_encoding="kalah_v3")}
        invalid = {"state": [0.0] * 27}
        with self.assertRaisesRegex(RuntimeError, "verified unique"):
            decoded_validation_manifest([invalid, row], np.asarray([0, 1]))

    def test_phase_c_metrics_report_directed_search_contrasts(self) -> None:
        records = []
        for model, move, root_value, rank, margin, visits in (
            ("C", 1, 0.1, 1, 12, [8.0, 12.0]),
            ("T", 0, 0.3, 2, 8, [14.0, 6.0]),
            ("JH", 1, 0.2, 1, 10, [7.0, 13.0]),
            ("J", 0, 0.4, 2, 6, [15.0, 5.0]),
            ("H", 1, 0.0, 1, 11, [9.0, 11.0]),
        ):
            records.append(
                {
                    "state_hash": "a",
                    "context": "768:768",
                    "model": model,
                    "selected_move": move,
                    "root_value": root_value,
                    "visit_policy": visits,
                    "selected_child_q_rank": rank,
                    "top1_top2_visit_margin": margin,
                }
            )
        metrics = phase_c_metrics(records)
        contrast = metrics["768:768"]["direct_contrasts"]["T-C"]
        self.assertEqual("B-minus-A", contrast["orientation"])
        self.assertEqual(1.0, contrast["selected_move_change_rate"])
        self.assertAlmostEqual(0.2, contrast["mean_signed_root_value_delta"])
        self.assertEqual(1.0, contrast["mean_selected_child_q_rank_change"])
        self.assertEqual(-4.0, contrast["mean_visit_margin_change"])
        self.assertGreater(contrast["mean_visit_policy_js"], 0.0)

    def test_context_c_puct_is_fixed_by_context(self) -> None:
        self.assertEqual(0.90, _context_c_puct("768:768"))
        self.assertEqual(1.25, _context_c_puct("768:256"))

    def test_clustered_bootstrap_is_deterministic(self) -> None:
        self.assertEqual(
            _clustered_bootstrap([1.0, -1.0], seed=191),
            _clustered_bootstrap([1.0, -1.0], seed=191),
        )
        result = _clustered_bootstrap([1.0, -1.0, 0.0], seed=191)
        self.assertEqual(10_000, result["samples"])
        self.assertEqual(1 / 3, result["better_fraction"])
        self.assertEqual(1 / 3, result["worse_fraction"])
        self.assertEqual(1 / 3, result["tie_fraction"])

    def test_phase_g_reports_mixed_additive_harm_without_conflict_or_tradeoff(
        self,
    ) -> None:
        def effect(value: float, lower: float, upper: float) -> dict[str, object]:
            return {
                "paired_candidate_effect": value,
                "opening_bootstrap_ci": {"lower_95": lower, "upper_95": upper},
            }

        result = phase_g_classification(
            {},
            {
                "current": {
                    "mean": {
                        "grad_policy_norm": 3.88,
                        "weighted_grad_value_norm": 0.489,
                    },
                    "cosine_bootstrap_95": {"lower": -0.0618, "upper": 0.0134},
                }
            },
            {
                "enabled": True,
                "metrics": {
                    "384:256": {
                        "direct_contrasts": {"T-C": effect(-0.1504, -0.1914, -0.1094)}
                    },
                    "1200:1200": {
                        "direct_contrasts": {
                            "T-C": effect(-0.0527, -0.0840, -0.0234),
                            "JH-C": effect(-0.0703, -0.0898, -0.0508),
                        }
                    },
                },
            },
        )

        self.assertIsNone(result["primary_classification"])
        self.assertIn(
            "policy_gradient_dominates_trunk_learning", result["classifications"]
        )
        self.assertIn("mixed_additive_harm", result["classifications"])
        self.assertNotIn("gradient_conflict", result["classifications"])
        self.assertNotIn("tradeoff", result["classifications"])

    def test_phase_f_reuses_verified_persisted_canonical_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persisted_text = "".join(
                json.dumps({"prefix_moves": [index % 6]}) + "\n" for index in range(128)
            )
            source_text = "".join(
                json.dumps({"prefix_moves": [index % 6], "source": "canonical"}) + "\n"
                for index in range(128)
            )
            source = root / "medium_eval.jsonl"
            source.write_text(source_text, encoding="utf-8")
            persisted = root / "starts_0" / "opening_suite.jsonl"
            persisted.parent.mkdir()
            persisted.write_text(persisted_text, encoding="utf-8")
            source_hash = hashlib.sha256(source_text.encode()).hexdigest()
            (persisted.parent / "metadata.json").write_text(
                json.dumps(
                    {
                        "cache_manifest": {
                            "suite_path": str(source),
                            "suite_sha256": source_hash,
                            "suite_size": 128,
                        }
                    }
                ),
                encoding="utf-8",
            )

            provenance = phase_f_suite_provenance(persisted)

            self.assertEqual(str(persisted), provenance["reused_suite_path"])
            self.assertEqual(
                hashlib.sha256(persisted_text.encode()).hexdigest(),
                provenance["reused_suite_sha256"],
            )
            self.assertEqual(str(source), provenance["persisted_source_path"])
            self.assertEqual(source_hash, provenance["persisted_source_sha256"])
            self.assertEqual(128, provenance["unique_openings"])

    def test_phase_f_arena_evidence_requires_unique_expected_games(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arena_jsonl = Path(directory) / "arena.jsonl"
            arena_jsonl.write_text(
                "".join(
                    json.dumps({"game_index": index}) + "\n" for index in range(256)
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                256,
                len(
                    load_complete_phase_f_arena_records(arena_jsonl, expected_games=256)
                    or []
                ),
            )

            with arena_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"game_index": 0}) + "\n")
            self.assertIsNone(
                load_complete_phase_f_arena_records(arena_jsonl, expected_games=256)
            )
