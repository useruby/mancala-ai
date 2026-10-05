from __future__ import annotations

import hashlib
import json
import unittest

import torch

from ml.alphazero_lite.run_seed422_adam_memory import make_adam
from ml.alphazero_lite.seed422_adam_memory_analysis import (
    analyze,
    bootstrap_paired,
    optimizer_options,
)
from ml.alphazero_lite.seed422_exclusions import build_union, make_suite
from ml.alphazero_lite import build_opening_suite as suites


class Seed422AdamMemoryTests(unittest.TestCase):
    def test_only_beta_one_differs_between_optimizer_lanes(self):
        a = optimizer_options("A")
        b = optimizer_options("B")
        self.assertEqual(a["betas"], (0.9, 0.999))
        self.assertEqual(b["betas"], (0.0, 0.999))
        self.assertEqual(
            {key: value for key, value in a.items() if key != "betas"},
            {key: value for key, value in b.items() if key != "betas"},
        )
        self.assertEqual(a["lr"], 0.001)
        self.assertEqual(a["eps"], 1e-8)
        self.assertEqual(a["weight_decay"], 0.0)
        self.assertEqual(a["scheduler"], "none")

    def test_lane_a_is_identical_to_default_adam_updates(self):
        class Holder(torch.nn.Module):
            def __init__(self, parameter):
                super().__init__()
                self.weight = parameter

        reference = torch.nn.Parameter(torch.tensor([1.25, -0.75]))
        treatment = torch.nn.Parameter(reference.detach().clone())
        default = torch.optim.Adam([reference], lr=0.001, weight_decay=0.0)
        actual = make_adam(Holder(treatment), "A")
        for gradient in ([0.2, -0.3], [-0.4, 0.1], [0.05, 0.8]):
            reference.grad = torch.tensor(gradient)
            treatment.grad = torch.tensor(gradient)
            default.step()
            actual.step()
            self.assertTrue(torch.equal(reference, treatment))
        self.assertEqual(
            actual.param_groups[0]["betas"], default.param_groups[0]["betas"]
        )
        self.assertEqual(actual.param_groups[0]["eps"], default.param_groups[0]["eps"])

    def test_bootstrap_is_fixed_seed_and_percentile(self):
        sample = [0.0, 1.0, 0.5, -0.5]
        self.assertEqual(bootstrap_paired(sample), bootstrap_paired(sample))
        low, high = bootstrap_paired(sample)
        self.assertLessEqual(low, high)

    def test_decision_uses_paired_delta_and_b_score_lower_bound(self):
        rows = []
        for opening in range(512):
            b_score = 1.0 if opening < 51 else 0.5
            for lane, score in (("A", 0.5), ("B", b_score)):
                for seat in (0, 1):
                    rows.append(
                        {
                            "lane": lane,
                            "opening_id": opening,
                            "opponent_score": score,
                            "game": {"challenger_player": seat},
                        }
                    )
        report = analyze(rows)
        self.assertAlmostEqual(report["paired_delta"], 51 / 512 * 0.5)
        self.assertEqual(report["decision"], "advance_to_independent_confirmation")
        self.assertEqual(len(report["opening_matrix"]), 512)
        self.assertEqual(report["lanes"]["B"]["wins"], 102)
        self.assertEqual(report["lanes"]["B"]["draws"], 922)

    def test_analysis_refuses_adaptive_or_incomplete_game_count(self):
        with self.assertRaisesRegex(ValueError, "2048"):
            analyze([])

    def test_corrected_historical_union_and_new_suite(self):
        excluded, proof = build_union()
        self.assertEqual(len(excluded), 266575)
        self.assertEqual(
            proof["identity_set_sha256"],
            "f6713bebfe1ded09ed15e45f17a11566d0012f03fe7719138fdc832bbb1339f7",
        )
        suite = make_suite(excluded)
        identities = suites.validate_arena_entries(suite)
        self.assertEqual(len(identities), 512)
        self.assertEqual(len(set(identities)), 512)
        self.assertFalse(set(identities) & excluded)
        serialized = "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in suite
        ).encode()
        self.assertEqual(
            hashlib.sha256(serialized).hexdigest(),
            "74619aed20aaa9be4f6c0a1c1cb74263a782d8200c4c9e0907f66e7ca8b301a2",
        )


if __name__ == "__main__":
    unittest.main()
