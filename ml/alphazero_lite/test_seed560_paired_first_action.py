"""Semantic regression tests for the paired first-action evidence contract."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ml.alphazero_lite.seed557_retrospective import canonical_hash
from ml.alphazero_lite.seed560_paired_first_action import (
    INITIALIZER,
    OUT,
    P_ARTIFACT,
    _commit_root_bundle,
    _read_resume_evidence,
    authenticated_inputs,
    sha256,
)
from ml.alphazero_lite.verify_seed560_paired_first_action import (
    validate_continuation,
    verify_source_bindings,
    verify_search_choices,
)


class Seed560PairedFirstActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = json.loads((OUT / "cohort-manifest.json").read_text())["states"]
        cls.roots = [
            json.loads(line)
            for line in (OUT / "root-searches.jsonl").read_text().splitlines()
        ]
        cls.outcomes = [
            json.loads(line)
            for line in (OUT / "continuations.jsonl").read_text().splitlines()
        ]

    def test_identical_actions_reuse_one_continuation(self):
        by_key = {(row["opening_index"], row["budget"]): row for row in self.outcomes}
        matches = [
            i
            for i, root in enumerate(self.roots)
            if root["actions"]["128"] == root["actions"]["512"]
        ]
        self.assertTrue(matches)
        for index in matches:
            self.assertEqual(
                by_key[(index, 128)]["outcome"], by_key[(index, 512)]["outcome"]
            )

    def test_extra_turn_trajectory_is_replayed_without_alternating_actor_assumption(
        self,
    ):
        by_key = {(row["opening_index"], row["budget"]): row for row in self.outcomes}
        selected = None
        for index, root in enumerate(self.roots):
            row = by_key[(index, 128)]
            actor = self.cohort[index]["state"]["current_player"]
            sequence = [
                actor,
                *[step["actor"] for step in row["outcome"]["trajectory"]],
            ]
            if any(left == right for left, right in zip(sequence, sequence[1:])):
                selected = (self.cohort[index], row)
                break
        self.assertIsNotNone(selected)
        validate_continuation(selected[0], selected[1], selected[1]["outcome"])

    def test_continuation_seed_is_bound_to_root_and_ply(self):
        row = next(row for row in self.outcomes if row["outcome"]["trajectory"])
        corrupted = json.loads(json.dumps(row))
        corrupted["outcome"]["trajectory"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed_or_runtime_binding"):
            validate_continuation(
                self.cohort[row["opening_index"]], corrupted, corrupted["outcome"]
            )

    def test_contradictory_terminal_evidence_is_rejected(self):
        row = next(row for row in self.outcomes if row["outcome"]["trajectory"])
        corrupted = json.loads(json.dumps(row))
        corrupted["outcome"]["margin"] += 1
        with self.assertRaisesRegex(ValueError, "terminal_evidence_contradiction"):
            validate_continuation(
                self.cohort[row["opening_index"]], corrupted, corrupted["outcome"]
            )

    def _search_evidence_fixture(self, index=0):
        cohort = [self.cohort[index]]
        roots = [self.roots[index]]
        arms = [row for row in self.outcomes if row["opening_index"] == index]
        expected = {}
        root = roots[0]
        state = cohort[0]["state"]
        for budget in (128, 512):
            expected[(canonical_hash(state), budget, root["seed"])] = root["actions"][
                str(budget)
            ]
        from ml.alphazero_lite.kalah_rules import KalahGame

        for arm in arms:
            game = KalahGame.from_state(cohort[0]["state"])
            self.assertTrue(game.move(game.pit_index(arm["action"])))
            for step in arm["outcome"]["trajectory"]:
                key = (step["state_hash"], 128, step["seed"])
                self.assertIn(expected.get(key, step["action"]), (step["action"],))
                expected[key] = step["action"]
                self.assertTrue(game.move(game.pit_index(step["action"])))

        def search_action(state, simulations, seed):
            return expected[(canonical_hash(state), simulations, seed)]

        roots[0]["opening_index"] = 0
        for arm in arms:
            arm["opening_index"] = 0
        return cohort, roots, arms, search_action

    def test_reexecuted_search_rejects_swapped_root_and_continuation_labels(self):
        index = next(
            i
            for i, row in enumerate(self.roots)
            if row["actions"]["128"] != row["actions"]["512"]
        )
        cohort, roots, arms, search_action = self._search_evidence_fixture(index)
        corrupted_roots = json.loads(json.dumps(roots))
        corrupted_arms = json.loads(json.dumps(arms))
        actions = corrupted_roots[0]["actions"]
        actions["128"], actions["512"] = actions["512"], actions["128"]
        for row in corrupted_arms:
            row["budget"] = 640 - row["budget"]
        with self.assertRaisesRegex(ValueError, "root_search_choice_mismatch"):
            verify_search_choices(
                cohort, corrupted_roots, corrupted_arms, search_action
            )

    def test_reexecuted_search_rejects_a_legal_but_wrong_continuation_action(self):
        index = next(
            i
            for i, row in enumerate(self.roots)
            if row["actions"]["128"] != row["actions"]["512"]
        )
        cohort, roots, arms, search_action = self._search_evidence_fixture(index)
        changed = json.loads(json.dumps(arms))
        row = next(row for row in changed if row["budget"] == 128)
        step = row["outcome"]["trajectory"][0]
        from ml.alphazero_lite.kalah_rules import KalahGame

        game = KalahGame.from_state(cohort[0]["state"])
        self.assertTrue(game.move(game.pit_index(row["action"])))
        legal_alternatives = [
            move for move in game.possible_moves() if move != step["action"]
        ]
        self.assertTrue(legal_alternatives)
        step["action"] = legal_alternatives[0]
        with self.assertRaisesRegex(ValueError, "continuation_search_choice_mismatch"):
            verify_search_choices(cohort, roots, changed, search_action)

    def test_saved_root_without_both_arms_is_rejected_before_resume(self):
        root = {
            "opening_index": 0,
            "state_hash": "state",
            "actions": {"128": 1, "512": 2},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)
            (output / "root-searches.jsonl").write_text(
                json.dumps(root) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "incomplete_saved_root_bundle"):
                _read_resume_evidence(output)
            self.assertFalse((output / "root-commits").exists())

    def test_complete_legacy_prefix_migrates_before_a_new_atomic_bundle(self):
        def root_record(index):
            return {
                "opening_index": index,
                "state_hash": f"state-{index}",
                "actions": {"128": 1, "512": 2},
            }

        def arm_records(index):
            return [
                {
                    "opening_index": index,
                    "state_hash": f"state-{index}",
                    "budget": budget,
                    "action": action,
                }
                for budget, action in ((128, 1), (512, 2))
            ]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)
            legacy_root = root_record(0)
            legacy_arms = arm_records(0)
            (output / "root-searches.jsonl").write_text(
                json.dumps(legacy_root, sort_keys=True) + "\n", encoding="utf-8"
            )
            (output / "continuations.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in legacy_arms),
                encoding="utf-8",
            )
            roots, outcomes, done = _read_resume_evidence(output)
            self.assertEqual(done, {0})
            self.assertTrue((output / "root-commits/opening-0000.json").is_file())

            _commit_root_bundle(output, root_record(1), arm_records(1))
            roots, outcomes, done = _read_resume_evidence(output)
            self.assertEqual(done, {0, 1})
            self.assertEqual([row["opening_index"] for row in roots], [0, 1])
            self.assertEqual(len(outcomes), 4)
            self.assertEqual(_read_resume_evidence(output)[2], {0, 1})

    def test_incomplete_legacy_tail_rejection_writes_no_new_bundles_or_ledgers(self):
        def root_record(index):
            return {
                "opening_index": index,
                "state_hash": f"state-{index}",
                "actions": {"128": 1, "512": 2},
            }

        def arm_records(index):
            return [
                {
                    "opening_index": index,
                    "state_hash": f"state-{index}",
                    "budget": budget,
                    "action": action,
                }
                for budget, action in ((128, 1), (512, 2))
            ]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)
            committed_root = root_record(0)
            committed_arms = arm_records(0)
            _commit_root_bundle(output, committed_root, committed_arms)
            aggregate_roots = [committed_root, root_record(1), root_record(2)]
            aggregate_arms = [
                *committed_arms,
                *arm_records(1),
                arm_records(2)[0],
            ]
            (output / "root-searches.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in aggregate_roots
                ),
                encoding="utf-8",
            )
            (output / "continuations.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in aggregate_arms
                ),
                encoding="utf-8",
            )

            def snapshot():
                return {
                    path.relative_to(output).as_posix(): path.read_bytes()
                    for path in output.rglob("*")
                    if path.is_file()
                }

            before = snapshot()
            with self.assertRaisesRegex(ValueError, "incomplete_legacy_root_bundle:2"):
                _read_resume_evidence(output)
            self.assertEqual(snapshot(), before)

    def test_verifier_accepts_a_fresh_freeze_with_current_runner_hash(self):
        runner_path = Path(__file__).with_name("seed560_paired_first_action.py")
        fresh_digest = sha256(runner_path.read_bytes())
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            verify_source_bindings(
                Path(temporary),
                {
                    "source_sha256": {
                        runner_path.relative_to(
                            Path(__file__).parents[2]
                        ).as_posix(): fresh_digest
                    }
                },
            )

    def test_historical_runner_hash_exception_is_not_accepted_outside_run0004(self):
        runner_path = Path(__file__).with_name("seed560_paired_first_action.py")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            with self.assertRaisesRegex(ValueError, "fresh_freeze_runner_source_hash"):
                verify_source_bindings(
                    Path(temporary),
                    {
                        "source_sha256": {
                            runner_path.relative_to(
                                Path(__file__).parents[2]
                            ).as_posix(): "70b07ad5e9af020606f83dc45fe7e603b04d8aa3167fcd1eba0fc581ab3b9bf7"
                        }
                    },
                )

    def test_atomic_root_bundle_recovers_torn_aggregate_ledgers(self):
        root = {
            "opening_index": 0,
            "state_hash": "state",
            "actions": {"128": 1, "512": 2},
        }
        arms = [
            {"opening_index": 0, "state_hash": "state", "budget": 128, "action": 1},
            {"opening_index": 0, "state_hash": "state", "budget": 512, "action": 2},
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = Path(temporary)
            _commit_root_bundle(output, root, arms)
            (output / "root-searches.jsonl").write_text(
                json.dumps(root, sort_keys=True) + "\n", encoding="utf-8"
            )
            roots, outcomes, done = _read_resume_evidence(output)
            self.assertEqual(done, {0})
            self.assertEqual(roots, [root])
            self.assertEqual(outcomes, arms)

    def test_artifact_hashes_must_match_authenticated_seed556_components(self):
        authenticated = authenticated_inputs()
        self.assertEqual(set(authenticated["components"]), {"P", "initializer"})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            copied_policy = Path(temporary) / "P-E4"
            shutil.copytree(P_ARTIFACT, copied_policy)
            model = copied_policy / "model.npz"
            model.write_bytes(model.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                ValueError, "unauthenticated_serving_artifact:P"
            ):
                authenticated_inputs({"P": copied_policy, "initializer": INITIALIZER})


if __name__ == "__main__":
    unittest.main()
