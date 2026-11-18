"""Synthetic fail-closed checks for Seed 615 source-group selection."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from ml.alphazero_lite.seed599_replay_coverage import (
    validate_training_heldout_disjointness,
)
from ml.alphazero_lite.seed615_opening_diversity import select_balanced
from ml.alphazero_lite.verify_seed615_opening_diversity import (
    EXPECTED_BOOTSTRAP_SPEC,
    validate_opening_prefix,
    validate_registration_contract,
)


def row(identity: str, group: str, phase: str = "above_32") -> dict[str, object]:
    return {
        "canonical_identity": identity,
        "float32_identity": f"f32-{identity}",
        "source": "synthetic",
        "opening_id": group,
        "phase": phase,
    }


def valid_registration() -> dict[str, Any]:
    return {
        "seed": 615,
        "status": "proposed-for-review-not-authorized",
        "feasibility": {
            "candidate_membership": "docs/data/seed615-source-opening-diversity/candidate-membership.jsonl",
            "candidate_membership_sha256": "membership",
        },
        "treatment": {"membership_sha256": "membership"},
        "evaluation": {
            "status": "prospectively-registered-not-authorized",
            "cohort": {
                "path": "docs/data/seed615-source-opening-diversity/evaluation-openings.jsonl",
                "assignments_path": "docs/data/seed615-source-opening-diversity/evaluation-assignments.jsonl",
                "canonical_manifest_sha256": "openings",
                "assignment_manifest_sha256": "assignments",
            },
            "bootstrap": copy.deepcopy(EXPECTED_BOOTSTRAP_SPEC),
            "primary_confirmatory_test": {
                "interval": "paired two-sided 95% opening-cluster bootstrap confidence interval",
                "contrast": "treatment minus fixed B512 seat-averaged playing score",
                "resamples": 20000,
                "bootstrap_seed": 615,
                "quantiles": [0.025, 0.975],
                "pass_rule": "lower endpoint strictly greater than zero",
            },
            "separate_serving_strength_gate": {
                "bound": "one-sided 95% paired opening-cluster bootstrap lower confidence bound",
                "contrast": "treatment versus serving P-E4 seat-averaged score",
                "resamples": 20000,
                "bootstrap_seed": 615,
                "lower_quantile": 0.05,
                "pass_rule": "lower bound strictly greater than 0.50",
            },
        },
        "authorization": {"currently_authorized": False},
    }


class SyntheticGame:
    def __init__(self, *, terminal_after: int | None = None) -> None:
        self.moves = 0
        self.terminal_after = terminal_after

    def over(self) -> bool:
        return self.terminal_after is not None and self.moves >= self.terminal_after

    def possible_moves(self) -> list[int]:
        return [0] if not self.over() else []

    def pit_index(self, action: int) -> int:
        return action

    def move(self, action: int) -> None:
        self.moves += 1

    def to_state(self) -> dict[str, int]:
        return {"moves": self.moves}


class Seed615SelectionTests(unittest.TestCase):
    def test_round_robin_balances_groups_and_is_deterministic(self) -> None:
        candidates = {
            f"state-{group}-{index}": row(f"state-{group}-{index}", group)
            for group, capacity in (("g0", 6), ("g1", 4), ("g2", 2))
            for index in range(capacity)
        }
        first = select_balanced(candidates, {"above_32": 7})
        second = select_balanced(candidates, {"above_32": 7})
        self.assertEqual(first, second)
        counts = {}
        for item in first:
            counts[item["opening_id"]] = counts.get(item["opening_id"], 0) + 1
        self.assertEqual(sorted(counts.values()), [2, 2, 3])

    def test_insufficient_total_phase_capacity_fails_closed(self) -> None:
        candidates = {"one": row("one", "g0")}
        with self.assertRaisesRegex(ValueError, "insufficient_phase_positions"):
            select_balanced(candidates, {"above_32": 2})

    def test_float32_collision_fails_closed(self) -> None:
        first = row("canonical-a", "g0")
        second = row("canonical-b", "g1")
        second["float32_identity"] = first["float32_identity"]
        with self.assertRaisesRegex(ValueError, "duplicate_float32_identity"):
            select_balanced({"a": first, "b": second}, {"above_32": 2})

    def test_canonical_duplicate_fails_closed(self) -> None:
        first = row("same", "g0")
        second = row("same", "g1")
        with self.assertRaisesRegex(ValueError, "duplicate_canonical_identity"):
            select_balanced({"a": first, "b": second}, {"above_32": 2})

    def test_heldout_canonical_overlap_fails_closed(self) -> None:
        candidate = [row("same", "train")]
        control = [row("control", "train")]
        heldout = [row("same", "heldout")]
        with self.assertRaisesRegex(ValueError, "heldout_treatment_canonical_overlap"):
            validate_training_heldout_disjointness(candidate, control, heldout)

    def test_heldout_float32_overlap_fails_closed(self) -> None:
        candidate = [row("candidate", "train")]
        control = [row("control", "train")]
        heldout = [row("heldout", "test")]
        heldout[0]["float32_identity"] = candidate[0]["float32_identity"]
        with self.assertRaisesRegex(ValueError, "heldout_treatment_float32_overlap"):
            validate_training_heldout_disjointness(candidate, control, heldout)

    def test_evaluation_opening_with_wrong_length_fails(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "evaluation_opening_prefix_length_mismatch"
        ):
            validate_opening_prefix([0] * 11)

    def test_evaluation_opening_ending_terminal_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation_opening_terminal"):
            validate_opening_prefix(
                [0] * 12,
                game_factory=lambda: SyntheticGame(terminal_after=12),
            )

    def test_evaluation_opening_with_illegal_move_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "evaluation_opening_prefix_illegal"):
            validate_opening_prefix([0] * 11 + [1], game_factory=SyntheticGame)

    def test_registration_statistical_settings_are_authenticated(self) -> None:
        changes = (
            lambda registration: registration["evaluation"]["bootstrap"].update(
                resamples=19999
            ),
            lambda registration: registration["evaluation"]["bootstrap"].update(
                random_generator="numpy.random.RandomState"
            ),
            lambda registration: registration["evaluation"]["bootstrap"].update(
                quantile_method="numpy.quantile(method='midpoint')"
            ),
            lambda registration: registration["evaluation"][
                "primary_confirmatory_test"
            ].update(quantiles=[0.05, 0.95]),
            lambda registration: registration["evaluation"][
                "primary_confirmatory_test"
            ].update(pass_rule="lower endpoint at least zero"),
            lambda registration: registration["evaluation"][
                "separate_serving_strength_gate"
            ].update(lower_quantile=0.025),
            lambda registration: registration["evaluation"][
                "separate_serving_strength_gate"
            ].update(pass_rule="lower bound at least 0.50"),
            lambda registration: registration["evaluation"][
                "separate_serving_strength_gate"
            ].update(bound="one-sided 90% lower bound"),
        )
        for mutate in changes:
            with self.subTest(mutate=mutate):
                registration = valid_registration()
                mutate(registration)
                with self.assertRaises(ValueError):
                    validate_registration_contract(
                        registration,
                        membership_digest="membership",
                        opening_digest="openings",
                        assignment_digest="assignments",
                    )

    def test_registration_membership_or_evaluation_hash_change_fails(self) -> None:
        for path in (
            ("feasibility", "candidate_membership"),
            ("feasibility", "candidate_membership_sha256"),
            ("treatment", "membership_sha256"),
            ("evaluation", "cohort", "path"),
            ("evaluation", "cohort", "canonical_manifest_sha256"),
            ("evaluation", "cohort", "assignments_path"),
            ("evaluation", "cohort", "assignment_manifest_sha256"),
        ):
            with self.subTest(path=path):
                registration = valid_registration()
                target: dict[str, Any] = registration
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = "altered"
                with self.assertRaises(ValueError):
                    validate_registration_contract(
                        registration,
                        membership_digest="membership",
                        opening_digest="openings",
                        assignment_digest="assignments",
                    )

    def test_registration_authorization_change_fails(self) -> None:
        registration = valid_registration()
        registration["authorization"]["currently_authorized"] = True
        with self.assertRaisesRegex(ValueError, "registration_authorized_or_invalid"):
            validate_registration_contract(
                registration,
                membership_digest="membership",
                opening_digest="openings",
                assignment_digest="assignments",
            )


if __name__ == "__main__":
    unittest.main()
