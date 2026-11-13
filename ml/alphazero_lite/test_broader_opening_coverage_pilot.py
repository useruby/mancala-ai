from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from ml.alphazero_lite import broader_opening_coverage_pilot as experiment
from ml.alphazero_lite import run_varied_opponent_pilot_v2 as v2
from ml.alphazero_lite import varied_opponent_pilot as pilot


class BroaderOpeningFreezeTests(unittest.TestCase):
    def test_prior_frozen_runner_sources_keep_registered_bytes(self):
        self.assertEqual(
            experiment.sha(
                experiment.ROOT / "ml/alphazero_lite/run_varied_opponent_pilot_v2.py"
            ),
            "ba93a0aeeade6298c5a4f0d2a48f0d42125e9af8ff157361c1d5a585fcb0b1e7",
        )
        self.assertEqual(
            experiment.sha(
                experiment.ROOT / "ml/alphazero_lite/varied_opponent_pilot.py"
            ),
            "41f125962f0c07777f0c4d4d6c12c456a0495c5e5e006fb891ebb5f862a81289",
        )

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

        def play(_spec, **kwargs):
            observed.append(kwargs["opponent_evaluator"])
            return {"spec": _spec}

        with (
            patch.object(pilot, "play_game", side_effect=play),
            patch.object(experiment, "verify_pilot_pair_result"),
            patch.object(v2, "warmed_probe", return_value=nullcontext(object())),
        ):
            evaluators = self._evaluators()
            experiment.play_pair_with_probe({"game_specs": specs}, evaluators)
        self.assertEqual(observed, [evaluators.p, evaluators.p])

    def test_treatment_retained_rows_must_say_p_before_verifier_adaptation(self):
        specs = [
            {
                "game_id": f"treatment-row-P{seat}",
                "pair_id": "treatment-row-pair",
                "arm": "treatment",
                "opening_id": "treatment-row-opening",
                "opening_moves": [],
                "seed": 72,
                "designated_p_seat": seat,
                "opponent": "P",
                "search_workers": 1,
            }
            for seat in (0, 1)
        ]
        job = {"game_specs": specs}
        result = {
            "pair_id": "treatment-row-pair",
            "arm": "treatment",
            "games": [{"spec": spec, "rows": [{"opponent": "P"}]} for spec in specs],
        }
        with patch.object(pilot, "verify_pair_result") as verifier:
            experiment.verify_pilot_pair_result(job, result)
            verifier.assert_called_once()
            verifier.reset_mock()
            mutated = json.loads(json.dumps(result))
            mutated["games"][0]["rows"][0]["opponent"] = "initializer"
            with self.assertRaisesRegex(
                ValueError, "retained_pilot_row_opponent_mismatch"
            ):
                experiment.verify_pilot_pair_result(job, mutated)
            verifier.assert_not_called()

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
            with patch.object(experiment, "verify_pilot_pair_result"):
                checkpoint = experiment.commit_pilot_pair(
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
            with patch.object(experiment, "verify_pilot_pair_result"):
                committed = experiment.run_checkpointed_pairs(
                    [job],
                    root,
                    pairs,
                    attempts,
                    identity,
                )
            self.assertEqual([row["pair_id"] for row in committed], ["synthetic-pair"])
            self.assertEqual(len(list(pairs.glob("*.json"))), 1)
            self.assertEqual(len(list(attempts.glob("*.json"))), 1)

    def test_novelty_inventory_mutation_is_rejected(self):
        original_digest = experiment.sha(experiment.INVENTORY_PATH)
        with tempfile.TemporaryDirectory() as directory:
            mutated = Path(directory) / "identity-inventory.json"
            mutated.write_bytes(experiment.INVENTORY_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                ValueError, "broader_pilot_novelty_inventory_hash_mismatch"
            ):
                experiment.load_inventory(mutated, expected_sha256=original_digest)

    def test_inventory_hashing_and_parsing_share_one_byte_read(self):
        payload = b'{"scheme":"canonical","identity":"known"}\n'
        with tempfile.TemporaryDirectory() as directory:
            inventory_path = Path(directory) / "inventory.jsonl"
            inventory_path.write_bytes(payload)
            expected = experiment.digest_bytes(payload)
            original_read_bytes = Path.read_bytes
            reads = []

            def counted_read_bytes(path):
                if path == inventory_path:
                    reads.append(path)
                return original_read_bytes(path)

            with (
                patch.object(Path, "read_bytes", counted_read_bytes),
                patch.object(
                    Path,
                    "read_text",
                    side_effect=AssertionError("inventory should parse hashed bytes"),
                ),
            ):
                result = experiment.load_inventory(
                    inventory_path, expected_sha256=expected
                )
        self.assertEqual(reads, [inventory_path])
        self.assertEqual(result["canonical"], {"known"})

    def test_resource_scope_requires_all_frozen_live_limits(self):
        freeze = json.loads((experiment.DATA / "freeze.json").read_text())
        limits = freeze["runtime_limits"]
        valid = {
            "cpu_quota_percent": 1800.0,
            "cpu_weight": 100,
            "memory_max_bytes": 38654705664,
            "allowed_cpus": set(range(14, 32)),
            "cpuset_source": "cgroup_cpuset",
            "process_affinity": set(range(14, 32)),
            "nice": 19,
        }
        self.assertEqual(
            experiment.enforce_live_resources(limits, observed=valid), valid
        )
        fallback = {**valid, "cpuset_source": "taskset_affinity_fallback"}
        self.assertEqual(
            experiment.enforce_live_resources(limits, observed=fallback), fallback
        )
        for key, value in (
            ("cpu_quota_percent", 1900.0),
            ("cpu_weight", 99),
            ("memory_max_bytes", 38654705663),
            ("allowed_cpus", set(range(13, 32))),
            ("process_affinity", set(range(14, 31))),
            ("nice", 18),
        ):
            with self.subTest(key=key):
                mismatched = {**valid, key: value}
                with self.assertRaisesRegex(
                    ValueError, "broader_pilot_live_resource_scope_mismatch"
                ):
                    experiment.enforce_live_resources(limits, observed=mismatched)

    def test_live_resource_reader_uses_cgroup_and_process_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_cgroup = root / "proc-cgroup"
            proc_cgroup.write_text("0::/user.slice/pilot.scope\n")
            cgroup = root / "cgroup" / "user.slice/pilot.scope"
            cgroup.mkdir(parents=True)
            (cgroup / "cpu.max").write_text("1800000 100000\n")
            (cgroup / "cpu.weight").write_text("100\n")
            (cgroup / "memory.max").write_text("38654705664\n")
            (cgroup / "cpuset.cpus.effective").write_text("14-31\n")
            snapshot = experiment.live_resource_snapshot(
                proc_cgroup=proc_cgroup,
                cgroup_root=root / "cgroup",
                affinity=set(range(14, 32)),
                nice=19,
            )
        self.assertEqual(
            snapshot,
            {
                "cpu_quota_percent": 1800.0,
                "cpu_weight": 100,
                "memory_max_bytes": 38654705664,
                "allowed_cpus": set(range(14, 32)),
                "cpuset_source": "cgroup_cpuset",
                "process_affinity": set(range(14, 32)),
                "nice": 19,
            },
        )

    def test_live_resource_reader_uses_registered_taskset_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc_cgroup = root / "proc-cgroup"
            proc_cgroup.write_text("0::/user.slice/pilot.scope\n")
            cgroup = root / "cgroup" / "user.slice/pilot.scope"
            cgroup.mkdir(parents=True)
            (cgroup / "cpu.max").write_text("1800000 100000\n")
            (cgroup / "cpu.weight").write_text("100\n")
            (cgroup / "memory.max").write_text("38654705664\n")
            snapshot = experiment.live_resource_snapshot(
                proc_cgroup=proc_cgroup,
                cgroup_root=root / "cgroup",
                affinity=set(range(14, 32)),
                nice=19,
            )
        self.assertEqual(snapshot["cpuset_source"], "taskset_affinity_fallback")
        self.assertEqual(snapshot["allowed_cpus"], set(range(14, 32)))
        self.assertEqual(
            experiment.enforce_live_resources(
                json.loads((experiment.DATA / "freeze.json").read_text())[
                    "runtime_limits"
                ],
                observed=snapshot,
            ),
            snapshot,
        )

    def test_run_rejects_out_of_scope_before_writing_evidence(self):
        freeze = json.loads((experiment.DATA / "freeze.json").read_text())
        observed = {
            "cpu_quota_percent": 1800.0,
            "cpu_weight": 1,
            "memory_max_bytes": 38654705664,
            "allowed_cpus": set(range(14, 32)),
            "cpuset_source": "cgroup_cpuset",
            "process_affinity": set(range(14, 32)),
            "nice": 19,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "pairs"
            args = [
                "broader-opening-coverage-pilot",
                "run",
                "--freeze",
                str(experiment.DATA / "freeze.json"),
                "--expected-freeze-sha256",
                freeze["freeze_sha256"],
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
            with (
                patch("sys.argv", args),
                patch.object(
                    experiment, "live_resource_snapshot", return_value=observed
                ),
                self.assertRaisesRegex(
                    ValueError, "broader_pilot_live_resource_scope_mismatch"
                ),
            ):
                experiment.main()
            self.assertFalse(checkpoint_dir.exists())
            self.assertFalse((Path(directory) / "attempts").exists())


if __name__ == "__main__":
    unittest.main()
