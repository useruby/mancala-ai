from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import broader_opening_coverage_pilot as experiment
from ml.alphazero_lite import run_varied_opponent_pilot_v2 as v2
from ml.alphazero_lite import varied_opponent_pilot as pilot


class BroaderOpeningFreezeTests(unittest.TestCase):
    def test_deterministic_matching_and_identity_checks(self):
        controls = json.loads(experiment.CONTROL_PATH.read_text())
        pool = [
            json.loads(line)
            for line in experiment.POOL_PATH.read_text().splitlines()
            if line
        ]
        controls, treatment, report = experiment.select_matched_openings(controls, pool)
        self.assertEqual(len(controls), 62)
        self.assertEqual(len(treatment), 62)
        self.assertEqual(report["unmatched_count"], 2)
        self.assertEqual(
            [row["control_id"] for row in report["unmatched_controls"]],
            ["opening-0002", "opening-0022"],
        )
        self.assertTrue(report["identity_disjointness"]["canonical"])
        self.assertTrue(report["identity_disjointness"]["production_kalah_v3_float32"])
        for control, matched in zip(controls, treatment, strict=True):
            self.assertEqual(
                experiment.signature(control), experiment.signature(matched)
            )
        control_ids = [experiment.identity_pair(row["state"]) for row in controls]
        treatment_ids = [experiment.identity_pair(row["state"]) for row in treatment]
        self.assertEqual(len({item[0] for item in treatment_ids}), 62)
        self.assertEqual(len({item[1] for item in treatment_ids}), 62)
        self.assertFalse({x[0] for x in control_ids} & {x[0] for x in treatment_ids})
        self.assertFalse({x[1] for x in control_ids} & {x[1] for x in treatment_ids})

    def test_supplied_opening_starts_are_excluded_in_both_identity_schemes(self):
        freeze = json.loads((experiment.DATA / "freeze.json").read_text())
        starts = freeze["exclusion"]["supplied_opening_positions"]
        for arm in ("control_manifest", "treatment_manifest"):
            opening = next(
                row for row in freeze[arm] if row["signature"]["active_stones"] > 32
            )
            row = {
                "canonical_state": opening["state"],
                "stone_count": opening["signature"]["active_stones"],
            }
            self.assertNotIn(
                opening["canonical_identity"],
                experiment.coverage_identity_set(
                    [row], "canonical", set(starts["canonical"])
                ),
            )
            self.assertNotIn(
                opening["production_float32_identity"],
                experiment.coverage_identity_set(
                    [row], "kalah_v3_float32", set(starts["kalah_v3_float32"])
                ),
            )

    def test_treatment_games_against_p_use_the_p_evaluator(self):
        specs = [
            {
                "game_id": f"treatment-P{seat}",
                "pair_id": "treatment-pair",
                "arm": "treatment",
                "opening_id": "treatment-opening",
                "opening_moves": [],
                "seed": 71,
                "designated_p_seat": seat,
                "opponent": "P",
                "search_workers": 1,
            }
            for seat in (0, 1)
        ]
        observed = []

        class Probe:
            def __init__(self, *_args, **_kwargs):
                pass

            def warm(self):
                pass

            def close(self):
                pass

        def play(_spec, **kwargs):
            observed.append(kwargs["opponent_evaluator"])
            return {"spec": _spec}

        with (
            patch.object(pilot, "play_game", side_effect=play),
            patch.object(pilot, "verify_pair_result"),
        ):
            evaluators = self._evaluators()
            v2.play_pair_with_probe(
                {"game_specs": specs}, evaluators, probe_factory=Probe
            )
        self.assertEqual(observed, [evaluators.p, evaluators.p])

    @staticmethod
    def _evaluators():
        return type(
            "Evaluators",
            (),
            {
                "runtime_policy": {
                    "runtime_assets": {
                        "native_probe": {"path": "probe"},
                        "tablebase": {"path": "tablebase"},
                    },
                    "effective_search_policy": {"exact_root_threshold": 16},
                },
                "p": object(),
                "initializer": object(),
                "t512": object(),
            },
        )()

    def test_v2_resume_skips_a_verified_synthetic_pair(self):
        spec_pair = [
            {
                "game_id": f"synthetic-P{seat}",
                "pair_id": "synthetic-pair",
                "arm": "control",
                "opening_id": "synthetic-opening",
                "opening_moves": [],
                "seed": 55,
                "designated_p_seat": seat,
                "opponent": "P",
                "search_workers": 1,
            }
            for seat in (0, 1)
        ]
        job = {"game_specs": spec_pair}
        identity = {
            "schema": "synthetic-v2",
            "freeze_sha256": "a" * 64,
            "runner_source": v2.RUNNER_RELATIVE_PATH,
            "runner_sha256": "b" * 64,
            "probe_lifetime": v2.PAIR_LIFETIME,
            "checkpoint_schema": "varied_opponent_pair_checkpoint_v1",
        }
        result = {
            "pair_id": "synthetic-pair",
            "arm": "control",
            "games": [],
            "pair_process_cpu_time_ns": 7,
            "native_child_cpu_time_ns": 3,
            "pair_cpu_time_ns": 10,
            "execution_identity": identity,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs, attempts = root / "pairs", root / "attempts"
            pairs.mkdir()
            attempts.mkdir()
            v2.bind_run_directory(pairs, attempts, identity)
            with patch.object(pilot, "verify_pair_result"):
                checkpoint = v2.commit_v2_pair(
                    pairs, job, result, identity=identity, attempt_number=1
                )
            attempt = pilot.pair_attempt_path(attempts, "synthetic-pair", 1)
            pilot.write_json_atomic(
                attempt,
                {
                    "pair_id": "synthetic-pair",
                    "status": "completed",
                    "assignment": job,
                    "execution_identity": identity,
                    "checkpoint_sha256": pilot.sha256(checkpoint),
                    "pair_cpu_time_ns": 10,
                },
            )
            with patch.object(pilot, "verify_pair_result"):
                committed = v2.run_checkpointed_pairs(
                    [job],
                    root,
                    pairs,
                    attempts,
                    identity,
                    max_workers=1,
                )
            self.assertEqual([row["pair_id"] for row in committed], ["synthetic-pair"])
            self.assertEqual(len(list(pairs.glob("*.json"))), 1)
            self.assertEqual(len(list(attempts.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
