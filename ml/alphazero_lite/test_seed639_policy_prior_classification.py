"""Synthetic regression tests for Seed 639 policy comparison contracts."""

from __future__ import annotations

import unittest
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from ml.alphazero_lite import seed639_policy_prior_classification as diagnostic
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier
from ml.alphazero_lite import experiment_lifecycle
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state


def synthetic_population():
    rows = []
    index = 0
    for stones, quota in diagnostic.STONE_BIN_COUNTS.items():
        pit_total = stones - 1
        for first in range(1, pit_total - 1):
            for second in range(1, pit_total - first):
                third = pit_total - first - second
                if third <= 0:
                    continue
                state = {
                    "player_pits": [first, second, third, 0, 0, 0],
                    "opponent_pits": [1, 0, 0, 0, 0, 0],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
                game = KalahGame.from_state(state)
                rows.append(
                    {
                        "state": state,
                        "canonical_identity": diagnostic.sha256_bytes(
                            diagnostic.canonical_json(state)
                        ),
                        "production_float32_identity": diagnostic.sha256_bytes(
                            np.asarray(
                                encode_state(state, input_encoding="kalah_v3"),
                                dtype="<f4",
                            ).tobytes()
                        ),
                        "active_pit_stones": stones,
                        "legal_actions": [
                            int(action) for action in game.possible_moves()
                        ],
                        "game_id": f"game-{index:04d}",
                        "opening_identity": hashlib.sha256(
                            f"opening-{index:04d}".encode()
                        ).hexdigest(),
                        "opening_index": index,
                        "continuation_ply": index + 1,
                    }
                )
                index += 1
                if sum(row["active_pit_stones"] == stones for row in rows) == quota:
                    break
            if sum(row["active_pit_stones"] == stones for row in rows) == quota:
                break
    if len(rows) != 512:
        raise AssertionError(f"synthetic_population_size:{len(rows)}")
    return rows


class FixedEvaluator:
    def __init__(self, policy, fail_on_call=None):
        self.policy = np.asarray(policy, dtype=np.float32)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def evaluate(self, _game):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected_inference_failure")
        return self.policy.copy(), 0.0


def run_git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class Seed639PolicyPriorClassificationTests(unittest.TestCase):
    def test_historical_authentication_uses_registered_subprocess_runtime(self):
        from unittest.mock import patch

        from ml.alphazero_lite import seed639_authorization as authorization

        expected = authorization.HISTORICAL_VERIFIER_RUNTIME
        source_root = Path(__file__).resolve().parents[2]
        runtime = {
            "python_executable": expected["interpreter"],
            "python_version": expected["python_version"],
            "numpy_version": expected["numpy_version"],
            "torch_version": expected["torch_version"],
            "device": expected["device"],
            "cuda_available": expected["cuda_available"],
            "thread_environment": expected["thread_environment"],
            "torch_threads": expected["torch_threads"],
            "deterministic_algorithms": expected["deterministic_algorithms"],
            "runtime_lock_path": expected["runtime_lock_path"],
            "runtime_lock_sha256": expected["runtime_lock_sha256"],
            "seed615_dependency_lock_path": expected["seed615_dependency_lock_path"],
            "seed615_dependency_lock_sha256": expected[
                "seed615_dependency_lock_sha256"
            ],
            "worktree_seed615_dependency_lock_path": str(
                source_root / expected["worktree_dependency_lock"]
            ),
            "worktree_seed615_dependency_lock_sha256": expected[
                "worktree_dependency_lock_sha256"
            ],
        }
        response = {
            "schema": "seed639-seed615-historical-authentication-v1",
            "runtime": runtime,
            "authentication": {
                "freeze_sha256": diagnostic.SEED615_EVALUATION_FREEZE_SHA256,
                "checkpoint_sha256": diagnostic.MODEL_ARTIFACTS,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "population.tar.zst"
            archive.write_bytes(b"synthetic archive input")
            historical_root = root / "seed615"
            historical_root.mkdir()
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=json.dumps(response), stderr=""
            )
            with (
                patch.dict(os.environ, expected["thread_environment"]),
                patch.object(
                    diagnostic.subprocess, "run", return_value=completed
                ) as run,
            ):
                result = diagnostic.authenticate_seed615_in_subprocess(
                    seed615_root=historical_root,
                    archive_path=archive,
                    registered_runtime=expected,
                )

        self.assertEqual(result["historical_runtime"], runtime)
        command = run.call_args.args[0]
        self.assertEqual(command[0], expected["interpreter"])
        self.assertEqual(command[1:3], ["-m", expected["entrypoint_module"]])
        self.assertEqual(
            command[command.index("--seed615-root") + 1], str(historical_root)
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 14_400)
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONHASHSEED"], "615")

    def test_attempt_0002_amendment_binds_only_the_corrected_runtime(self):
        from ml.alphazero_lite import seed639_authorization as authorization

        root = Path(__file__).resolve().parents[2]
        registration = json.loads(
            (
                root
                / "docs/experiments/seed639-prospective-policy-prior-classification/registration.json"
            ).read_bytes()
        )
        authorization._validate_registration(registration)
        self.assertEqual(
            registration["execution_amendment"]["authorized_attempt_ids"],
            ["attempt-0002"],
        )

        altered = json.loads(json.dumps(registration))
        altered["execution_amendment"]["historical_verifier_runtime"][
            "torch_version"
        ] = "2.13.0+cu130"
        with self.assertRaisesRegex(PermissionError, "execution_amendment_invalid"):
            authorization._validate_registration(altered)

        authorization.authorize_attempt_identity(
            registration, "attempt-0002", "execute"
        )
        authorization.authorize_attempt_identity(registration, "attempt-0001", "verify")
        for attempt_id in ("attempt-0001", "attempt-0003", "attempt-0004"):
            with self.subTest(attempt_id=attempt_id):
                with self.assertRaisesRegex(
                    PermissionError, "attempt_identity_not_authorized"
                ):
                    authorization.authorize_attempt_identity(
                        registration, attempt_id, "execute"
                    )
        for attempt_id in ("attempt-0003", "attempt-0004"):
            with self.subTest(verification_attempt_id=attempt_id):
                with self.assertRaisesRegex(
                    PermissionError, "attempt_identity_not_authorized"
                ):
                    authorization.authorize_attempt_identity(
                        registration, attempt_id, "verify"
                    )

    def test_preflight_rehashes_every_registered_input_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "registered-input.json"
            input_path.write_text('{"input":1}\n')
            registered = {
                str(input_path): diagnostic.sha256_bytes(input_path.read_bytes())
            }
            self.assertEqual(
                diagnostic.rehash_registered_inputs(registered), registered
            )
            input_path.write_text('{"input":2}\n')
            with self.assertRaisesRegex(
                ValueError, "preflight_registered_input_mismatch"
            ):
                diagnostic.rehash_registered_inputs(registered)

    def test_preflight_validates_full_population_without_evaluator_construction(self):
        summary, bindings_digest, bindings = diagnostic.validate_preflight_population(
            synthetic_population()
        )
        self.assertEqual(summary["states_authenticated"], 512)
        self.assertEqual(
            summary["stone_bin_counts"],
            {"17": 103, "18": 103, "19": 102, "20": 102, "21": 102},
        )
        self.assertEqual(len(bindings), 512)
        self.assertEqual(
            summary["state_bindings_sha256"],
            diagnostic.sha256_bytes(diagnostic.canonical_json(bindings)),
        )
        self.assertEqual(bindings_digest, summary["state_bindings_sha256"])

    def test_read_only_attempt_0002_preflight_retains_complete_receipt_without_inference(
        self,
    ):
        from unittest.mock import patch

        from ml.alphazero_lite import seed639_authorization as authorization

        root = Path(__file__).resolve().parents[2]
        registration_path = (
            root
            / "docs/experiments/seed639-prospective-policy-prior-classification/registration.json"
        )
        registration = json.loads(registration_path.read_bytes())
        original_run = subprocess.run

        def git_run(command, **kwargs):
            if "status" in command:
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return original_run(command, **kwargs)

        def authorize(_path, source_commit):
            return {
                "merge_commit": source_commit,
                "registration_sha256": diagnostic.sha256_bytes(
                    registration_path.read_bytes()
                ),
                "pr_number": 641,
                "merged_by": "useruby",
            }

        authentication = {
            "selected_population_rows": synthetic_population(),
            "checkpoint_identity": registration["checkpoint_identity"],
            "seed636_archive_verification": {
                "experiment_id": "seed636-prospective-population",
                "attempt_id": "attempt-0001",
                "verified": True,
                "files": [],
            },
            "seed615_authentication": {"verification": {"passed": True}},
            "input_sha256": {},
            "exposure_classification": "known-exposure-screened; inherited P-E4 exposure unresolved",
        }

        with tempfile.TemporaryDirectory(dir=root / ".tmp") as temporary:
            receipt_path = Path(temporary) / "attempt-0002-preflight.json"
            with (
                patch.object(
                    authorization, "PREFLIGHT_RECEIPT_PATH", str(receipt_path)
                ),
                patch.object(
                    authorization, "authorize_registration", side_effect=authorize
                ),
                patch.object(diagnostic.subprocess, "run", side_effect=git_run),
                patch.object(
                    diagnostic,
                    "runtime_identity",
                    return_value={"python_version": "3.14.7"},
                ),
                patch.object(
                    diagnostic, "authenticate_inputs", return_value=authentication
                ),
                patch.object(
                    diagnostic,
                    "build_authenticated_evaluators",
                    side_effect=AssertionError(
                        "preflight_must_not_construct_evaluators"
                    ),
                ) as evaluator_factory,
            ):
                result = diagnostic.run_attempt_0002_preflight()

            self.assertTrue(receipt_path.is_file())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["population_preflight"]["states_authenticated"], 512
            )
            self.assertFalse(result["prospective_model_inference_performed"])
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(
                result["receipt_sha256"],
                diagnostic.sha256_bytes(receipt_path.read_bytes()),
            )
            evaluator_factory.assert_not_called()

    def test_preflight_receipt_is_hash_and_identity_bound(self):
        from unittest.mock import patch

        from ml.alphazero_lite import seed639_authorization as authorization

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            inputs = {"/registered/input": "a" * 64}
            receipt = {
                "schema": "seed639-attempt-0002-read-only-preflight-v1",
                "status": "passed",
                "experiment_id": authorization.EXPERIMENT_ID,
                "attempt_id": "attempt-0002",
                "source_commit": "b" * 40,
                "registration_sha256": "c" * 64,
                "registered_input_artifacts": inputs,
                "checkpoint_identity": authorization.CHECKPOINT_IDENTITY,
                "selected_inventory_sha256": diagnostic.SELECTED_INVENTORY_SHA256,
                "checks": {
                    "owner_merge_authorized": True,
                    "registered_inputs_rehashed": True,
                    "historical_authentication_passed": True,
                    "checkpoint_identity_matches_registration": True,
                    "population_validated": True,
                    "archive_reading_verified": True,
                },
                "prospective_model_inference_performed": False,
                "population_preflight": {
                    "states_authenticated": 512,
                    "canonical_identity_count": 512,
                    "production_float32_identity_count": 512,
                    "stone_bin_counts": {
                        "17": 103,
                        "18": 103,
                        "19": 102,
                        "20": 102,
                        "21": 102,
                    },
                    "continuation_games": 512,
                    "opening_clusters": 512,
                },
                "population_state_bindings_sha256": "d" * 64,
            }
            raw = json.dumps(receipt, sort_keys=True).encode("ascii") + b"\n"
            path.write_bytes(raw)
            with patch.object(authorization, "PREFLIGHT_RECEIPT_PATH", str(path)):
                verified = authorization.validate_preflight_receipt(
                    {"path": str(path), "sha256": diagnostic.sha256_bytes(raw)},
                    attempt_id="attempt-0002",
                    source_commit="b" * 40,
                    registration_sha256="c" * 64,
                    input_artifacts=inputs,
                    checkpoint_identity=authorization.CHECKPOINT_IDENTITY,
                )
                self.assertEqual(verified, receipt)
                with self.assertRaisesRegex(
                    PermissionError, "preflight_receipt_hash_mismatch"
                ):
                    authorization.validate_preflight_receipt(
                        {"path": str(path), "sha256": "0" * 64},
                        attempt_id="attempt-0002",
                        source_commit="b" * 40,
                        registration_sha256="c" * 64,
                        input_artifacts=inputs,
                        checkpoint_identity=authorization.CHECKPOINT_IDENTITY,
                    )

    def test_attempt_0002_freeze_refuses_to_freeze_without_preflight_receipt(self):
        from unittest.mock import patch

        from ml.alphazero_lite import experiment_lifecycle
        from ml.alphazero_lite import seed639_authorization as authorization

        repository = Path(__file__).resolve().parents[2]
        registration_path = authorization.REGISTRATION_PATH
        registration = json.loads((repository / registration_path).read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            worktree = Path(temporary)
            local_registration = worktree / registration_path
            local_registration.parent.mkdir(parents=True)
            local_registration.write_bytes(
                (repository / registration_path).read_bytes()
            )
            source_commit = "a" * 40
            manifest_registration_sha = diagnostic.sha256_bytes(
                local_registration.read_bytes()
            )
            spec = {
                "attempt_id": "attempt-0002",
                "configuration": {
                    "historical_verifier_runtime": authorization.HISTORICAL_VERIFIER_RUNTIME
                },
                "input_artifacts": registration["input_artifacts"],
                "random_seeds": [],
                "outputs": [
                    "seed639-output",
                    "seed639-output/preflight-receipt.json",
                ],
                "publication_outputs": ["seed639-output/preflight-receipt.json"],
            }
            with (
                patch.object(
                    experiment_lifecycle,
                    "load_worktree_registration",
                    return_value=(registration, registration_path),
                ),
                patch.object(
                    authorization,
                    "authorize_registration",
                    return_value={
                        "merge_commit": source_commit,
                        "registration_sha256": manifest_registration_sha,
                    },
                ),
                patch.object(
                    diagnostic.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        args=[], returncode=0, stdout=source_commit, stderr=""
                    ),
                ),
                patch.object(experiment_lifecycle, "freeze_attempt") as freeze_attempt,
            ):
                with self.assertRaisesRegex(
                    PermissionError, "preflight_receipt_binding_invalid"
                ):
                    diagnostic.freeze_attempt_0002(worktree, spec)
            freeze_attempt.assert_not_called()

    def test_legal_masking_zeroes_illegal_actions_and_normalizes(self):
        result = diagnostic.compare_legal_priors(
            [0.1, 0.2, 0.3, 0.1, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
            [0, 2, 5],
        )
        np.testing.assert_allclose(
            result["B512_legal_probabilities"], [1 / 6, 0, 1 / 2, 0, 0, 1 / 3]
        )
        self.assertEqual(result["B512_legal_probabilities"][1], 0)
        self.assertEqual(sum(result["S615_legal_probabilities"]), 1.0)
        self.assertAlmostEqual(result["total_variation"], 1 / 6)

    def test_first_legal_index_wins_a_top_action_tie(self):
        result = diagnostic.compare_legal_priors(
            [0.5, 0.5, 0, 0, 0, 0],
            [0.5, 0.5, 0, 0, 0, 0],
            [0, 1],
        )
        self.assertEqual(result["B512_top_action"], 0)
        self.assertFalse(result["top_action_disagreement"])

    def test_invalid_probabilities_and_legal_actions_fail_closed(self):
        invalid = (
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.3],
            [0.2, 0.2, 0.2, 0.2, 0.2, -0.1],
            [float("nan"), 0.2, 0.2, 0.2, 0.2, 0.2],
        )
        for vector in invalid:
            with self.subTest(vector=vector):
                with self.assertRaisesRegex(
                    ValueError, "invalid_raw_probability_vector"
                ):
                    diagnostic.compare_legal_priors(vector, [1 / 6] * 6, [0, 1])
        with self.assertRaisesRegex(ValueError, "invalid_legal_action_set"):
            diagnostic.compare_legal_priors([1 / 6] * 6, [1 / 6] * 6, [6])

    def test_subset_caps_and_hash_tie_break_are_deterministic(self):
        rows = []
        for index in range(80):
            rows.append(
                {
                    "canonical_identity": f"state-{index:03d}",
                    "game_id": f"game-{index:03d}",
                    "opening_identity": f"opening-{index // 5:03d}",
                    "total_variation": 0.75 if index < 70 else 0.25,
                }
            )
        first = diagnostic.select_diagnostic_subset(rows)
        second = diagnostic.select_diagnostic_subset(rows)
        self.assertEqual(first, second)
        self.assertTrue(first["available"])
        self.assertEqual(first["selected_count"], 64)
        self.assertEqual(first["distinct_continuation_games"], 64)
        self.assertTrue(
            all(count <= 4 for count in first["states_per_opening_cluster"].values())
        )
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["total_variation"],
                diagnostic.sha256_bytes(row["canonical_identity"].encode()),
                row["canonical_identity"],
            ),
        )
        expected = []
        opening_counts = {}
        for row in ranked:
            opening = row["opening_identity"]
            if opening_counts.get(opening, 0) >= 4:
                continue
            expected.append(row)
            opening_counts[opening] = opening_counts.get(opening, 0) + 1
            if len(expected) == 64:
                break
        self.assertEqual(
            first["selected_state_ids"],
            [row["canonical_identity"] for row in expected],
        )

    def test_subset_unavailable_below_distinct_game_gate(self):
        rows = [
            {
                "canonical_identity": f"state-{index}",
                "game_id": f"game-{index % 31}",
                "opening_identity": f"opening-{index}",
                "total_variation": 0.5,
            }
            for index in range(64)
        ]
        result = diagnostic.select_diagnostic_subset(rows)
        self.assertFalse(result["available"])
        self.assertEqual(result["selected_state_ids"], [])

    def test_duplicate_source_games_are_excluded_from_subset(self):
        rows = [
            {
                "canonical_identity": f"state-{index:03d}",
                "game_id": f"game-{index:03d}"
                if index < 60 or index >= 80
                else f"game-{index - 60:03d}",
                "opening_identity": f"opening-{index // 4:03d}",
                "total_variation": 1.0 - index / 1000,
            }
            for index in range(84)
        ]
        result = diagnostic.select_diagnostic_subset(rows)
        self.assertTrue(result["available"])
        selected_games = [row["game_id"] for row in result["selected_rows"]]
        self.assertEqual(len(selected_games), len(set(selected_games)))
        self.assertEqual(result["selected_count"], 64)

    def test_invalid_population_is_rejected_before_any_inference(self):
        rows = synthetic_population()
        rows[-1]["canonical_identity"] = "0" * 64
        b512 = FixedEvaluator([1 / 6] * 6)
        s615 = FixedEvaluator([1 / 6] * 6)
        with self.assertRaisesRegex(ValueError, "canonical_identity_mismatch"):
            diagnostic.evaluate_population(rows, b512, s615)
        self.assertEqual(b512.calls, 0)
        self.assertEqual(s615.calls, 0)

    def test_substituted_checkpoint_weights_metadata_and_evaluator_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "B512"
            model.mkdir()
            model_npz = b"synthetic-npz-checkpoint"
            weights = b'{"synthetic":"weights"}'
            metadata = {
                "framework": "numpy",
                "input_encoding": "kalah_v3",
                "artifacts": {
                    "weights_sha256": hashlib.sha256(model_npz).hexdigest(),
                    "weights_json_sha256": hashlib.sha256(weights).hexdigest(),
                },
            }
            (model / "model.npz").write_bytes(model_npz)
            (model / "weights.json").write_bytes(weights)
            (model / "metadata.json").write_text(json.dumps(metadata))
            expected = {
                "model.npz": hashlib.sha256(model_npz).hexdigest(),
                "weights.json": hashlib.sha256(weights).hexdigest(),
                "metadata.json": hashlib.sha256(
                    (model / "metadata.json").read_bytes()
                ).hexdigest(),
            }
            self.assertEqual(
                diagnostic.verify_numpy_artifact_files(model, expected, "synthetic"),
                expected,
            )
            for filename in expected:
                original = (model / filename).read_bytes()
                (model / filename).write_bytes(original + b"substitution")
                with self.subTest(filename=filename):
                    with self.assertRaisesRegex(
                        ValueError, "numpy_artifact_identity_mismatch"
                    ):
                        diagnostic.verify_numpy_artifact_files(
                            model, expected, "synthetic"
                        )
                (model / filename).write_bytes(original)

        class LookalikeEvaluator:
            input_encoding = "kalah_v3"

        with self.assertRaisesRegex(TypeError, "evaluator_type_mismatch"):
            diagnostic.assert_evaluator_identity(
                LookalikeEvaluator(), FixedEvaluator, "B512"
            )

    def test_independent_verifier_rejects_changed_vector_with_recomputed_metrics(self):
        original_b = np.asarray([0.4, 0.3, 0.2, 0.1, 0, 0], dtype=np.float64)
        original_s = np.asarray([0.1, 0.2, 0.3, 0.4, 0, 0], dtype=np.float64)
        row = diagnostic.compare_legal_priors(original_b, original_s, [0, 1, 2, 3])
        altered_b = np.asarray([0.3, 0.4, 0.2, 0.1, 0, 0], dtype=np.float64)
        row.update(diagnostic.compare_legal_priors(altered_b, original_s, [0, 1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "vector_mismatch"):
            verifier.require_exact_probability_vector(
                "B512", 0, original_b, row["B512_legal_probabilities"]
            )

    def test_independent_verifier_rejects_incorrect_aggregate(self):
        expected = {"states": 512, "mean_total_variation": 0.25}
        with self.assertRaisesRegex(ValueError, "aggregate_summary_mismatch"):
            verifier.require_exact_summary(
                {"states": 512, "mean_total_variation": 0.2}, expected
            )

    def test_unmerged_and_nonowner_registration_prs_are_rejected(self):
        from ml.alphazero_lite import seed639_authorization as authorization

        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            run_git(repo, "init", "--initial-branch=main")
            registration = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "docs/experiments/seed639-prospective-policy-prior-classification/registration.json"
                ).read_bytes()
            )
            registration.pop("execution_amendment", None)
            for runtime_input in authorization.AMENDMENT_RUNTIME_INPUTS:
                registration["input_artifacts"].pop(runtime_input, None)
            registration_path = repo / authorization.REGISTRATION_PATH
            source_root = Path(__file__).resolve().parents[2]
            for protocol in authorization.REQUIRED_PROTOCOLS:
                target = repo / protocol
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / protocol, target)
            run_git(repo, "add", ".")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Seed 639 test",
                    "-c",
                    "user.email=seed639-test@example.invalid",
                    "commit",
                    "-m",
                    "synthetic registration merge fixture",
                ],
                check=True,
                capture_output=True,
            )
            base_commit = run_git(repo, "rev-parse", "HEAD")
            registration_path.parent.mkdir(parents=True)
            registration_path.write_text(json.dumps(registration))
            run_git(repo, "add", authorization.REGISTRATION_PATH)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Seed 639 test",
                    "-c",
                    "user.email=seed639-test@example.invalid",
                    "commit",
                    "-m",
                    "introduce experiment registration",
                ],
                check=True,
                capture_output=True,
            )
            source_commit = run_git(repo, "rev-parse", "HEAD")
            pull = {
                "number": 639,
                "state": "open",
                "merged_at": None,
                "merged_by": {"login": "useruby"},
                "merge_commit_sha": source_commit,
                "base": {
                    "sha": base_commit,
                    "ref": "main",
                    "repo": {"full_name": "useruby/mancala-ai"},
                },
                "html_url": "https://example.invalid/pull/639",
            }
            with self.assertRaisesRegex(
                PermissionError, "owner_merged_registration_not_unique"
            ):
                authorization.authorize_registration(
                    registration_path,
                    source_commit,
                    associated_prs=[{"number": 639}],
                    pr_fetcher=lambda _number: pull,
                )
            pull = {**pull, "state": "closed", "merged_at": "2026-11-26T00:00:00Z"}
            accepted = authorization.authorize_registration(
                registration_path,
                source_commit,
                associated_prs=[{"number": 639}],
                pr_fetcher=lambda _number: pull,
            )
            self.assertEqual(accepted["merge_commit"], source_commit)
            pull = {**pull, "state": "closed", "merged_at": "2026-11-26T00:00:00Z"}
            pull["merged_by"] = {"login": "untrusted-contributor"}
            with self.assertRaisesRegex(
                PermissionError, "owner_merged_registration_not_unique"
            ):
                authorization.authorize_registration(
                    registration_path,
                    source_commit,
                    associated_prs=[{"number": 639}],
                    pr_fetcher=lambda _number: pull,
                )

            registered_commit = source_commit
            (repo / "unrelated.txt").write_text("unrelated change\n")
            run_git(repo, "add", "unrelated.txt")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "user.name=Seed 639 test",
                    "-c",
                    "user.email=seed639-test@example.invalid",
                    "commit",
                    "-m",
                    "unrelated change inheriting registration",
                ],
                check=True,
                capture_output=True,
            )
            inherited_commit = run_git(repo, "rev-parse", "HEAD")
            pull = {
                **pull,
                "state": "closed",
                "merged_at": "2026-11-26T00:00:00Z",
                "merged_by": {"login": "useruby"},
                "merge_commit_sha": inherited_commit,
                "base": {
                    "sha": registered_commit,
                    "ref": "main",
                    "repo": {"full_name": "useruby/mancala-ai"},
                },
            }
            with self.assertRaisesRegex(
                PermissionError, "owner_merged_registration_not_unique"
            ):
                authorization.authorize_registration(
                    registration_path,
                    inherited_commit,
                    associated_prs=[{"number": 639}],
                    pr_fetcher=lambda _number: pull,
                )

    def test_synthetic_execution_and_independent_verification_through_lifecycle(self):
        source_root = Path(__file__).resolve().parents[2]
        files = (
            "experiment_lifecycle.py",
            "kalah_rules.py",
            "seed639_policy_prior_classification.py",
            "verify_seed639_policy_prior_classification.py",
            "test_seed639_policy_prior_classification.py",
        )
        worker_source = """
import json
import sys
from pathlib import Path
from ml.alphazero_lite.test_seed639_policy_prior_classification import synthetic_population, FixedEvaluator
from ml.alphazero_lite import seed639_policy_prior_classification as producer
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier

attempt = Path(sys.argv[1])
output = attempt / "synthetic-output"
output.mkdir()
rows = synthetic_population()
preflight = producer.validate_population(rows)
records_dir = output / "state-records"
records_dir.mkdir()
b = FixedEvaluator([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
s = FixedEvaluator([0.10, 0.20, 0.25, 0.30, 0.10, 0.05])
records = producer.persist_inference_records(rows, b, s, records_dir)
subset = producer.select_diagnostic_subset(records)
summary = producer.summarize_population(records)
relationship = producer.summarize_subset_relationship(records, subset)
verified = verifier.verify_policy_records(rows, records)
subset_verified = verifier.verify_subset_selection(records, subset["selected_state_ids"])
for index, (row, record) in enumerate(zip(rows, records, strict=True)):
    from ml.alphazero_lite.kalah_rules import KalahGame
    game = KalahGame.from_state(row["state"])
    legal = [int(action) for action in game.possible_moves()]
    mask = [action in legal for action in range(6)]
    for model, evaluator in (("B512", b), ("S615", s)):
        vector, _ = evaluator.evaluate(game)
        vector = vector.astype("float64")
        for action, is_legal in enumerate(mask):
            if not is_legal:
                vector[action] = 0
        vector /= vector.sum()
        verifier.require_exact_probability_vector(model, index, vector, record[f"{model}_legal_probabilities"])
result = {"population_preflight": preflight, "independent_record_verification": verified,
          "subset_verification": subset_verified, "states": len(records),
          "model_state_evaluations": b.calls + s.calls, "summary": summary,
          "subset": {"available": subset["available"], "selected_state_ids": subset["selected_state_ids"]},
          "relationship": relationship}
(output / "summary.json").write_text(json.dumps(summary, sort_keys=True))
(output / "subset.json").write_text(json.dumps(subset, sort_keys=True))
(output / "relationship.json").write_text(json.dumps(relationship, sort_keys=True))
(output / "verification.json").write_text(json.dumps(result, sort_keys=True))
"""
        independent_source = """
import json
import sys
from pathlib import Path
from ml.alphazero_lite.test_seed639_policy_prior_classification import synthetic_population, FixedEvaluator
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier
attempt = Path(sys.argv[1])
output = attempt / "synthetic-output"
rows = synthetic_population()
records = [json.loads(path.read_bytes()) for path in sorted((output / "state-records").glob("state-*.json"))]
verified = verifier.verify_policy_records(rows, records)
summary = json.loads((output / "summary.json").read_bytes())
verifier.require_exact_summary(summary, verifier._summary_independent(records))
subset_doc = json.loads((output / "subset.json").read_bytes())
subset_check = verifier.verify_subset_selection(records, subset_doc["selected_state_ids"])
b = FixedEvaluator([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
s = FixedEvaluator([0.10, 0.20, 0.25, 0.30, 0.10, 0.05])
from ml.alphazero_lite.kalah_rules import KalahGame
for index, (row, record) in enumerate(zip(rows, records, strict=True)):
    game = KalahGame.from_state(row["state"])
    legal = [int(action) for action in game.possible_moves()]
    for model, evaluator in (("B512", b), ("S615", s)):
        vector, _ = evaluator.evaluate(game)
        vector = vector.astype("float64")
        vector[[action for action in range(6) if action not in legal]] = 0
        vector /= vector.sum()
        verifier.require_exact_probability_vector(model, index, vector, record[f"{model}_legal_probabilities"])
print(json.dumps({"passed": True, "states": len(records), "model_state_evaluations": b.calls + s.calls,
                  "record_verification": verified, "subset_verification": subset_check}, sort_keys=True))
"""
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repo"
            repository.mkdir()
            run_git(repository, "init", "--initial-branch=main")
            (repository / ".gitignore").write_text(
                "/.worktrees/\n**/__pycache__/\n*.pyc\n"
            )
            package = repository / "ml/alphazero_lite"
            package.mkdir(parents=True)
            (repository / "ml/__init__.py").write_text("")
            (package / "__init__.py").write_text("")
            for filename in files:
                shutil.copyfile(
                    source_root / "ml/alphazero_lite" / filename,
                    package / filename,
                )
            (package / "self_play.py").write_text(
                "import numpy as np\n"
                "def encode_state(state, input_encoding='kalah_v3'):\n"
                "    if input_encoding != 'kalah_v3': raise ValueError('encoding')\n"
                "    return np.asarray(state['player_pits'] + state['opponent_pits'] + [state['player_store'], state['opponent_store'], state['current_player']], dtype=np.float32)\n"
            )
            (repository / "synthetic_worker.py").write_text(worker_source)
            (repository / "synthetic_verifier.py").write_text(independent_source)
            run_git(repository, "add", ".")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=Seed 639 synthetic",
                    "-c",
                    "user.email=synthetic@example.invalid",
                    "commit",
                    "-m",
                    "synthetic base",
                ],
                check=True,
                capture_output=True,
            )
            base = run_git(repository, "rev-parse", "HEAD")
            experiment_id = "seed639-synthetic-lifecycle"
            worktree = experiment_lifecycle.create_worktree(
                repository,
                experiment_id,
                "experiment/seed639-synthetic-lifecycle",
                base,
                hypothesis="Synthetic only.",
                objective="Exercise lifecycle plumbing on synthetic states.",
                experiment_type="retrospective_diagnostic",
                random_seeds=[],
                input_artifacts={},
                checkpoint_identity=None,
                acceptance_criteria=["synthetic completion and verification"],
                worktree_root=repository / ".worktrees",
            )
            run_git(
                worktree,
                "add",
                ".experiment/.gitignore",
                "docs/experiments",
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(worktree),
                    "-c",
                    "user.name=Seed 639 synthetic",
                    "-c",
                    "user.email=synthetic@example.invalid",
                    "commit",
                    "-m",
                    "synthetic registration",
                ],
                check=True,
                capture_output=True,
            )
            if dirty := run_git(worktree, "status", "--short"):
                raise AssertionError(f"synthetic_worktree_dirty_before_freeze:{dirty}")
            attempt_id = "attempt-0001"
            attempt_dir = worktree / ".experiment/runtime/attempts" / attempt_id
            freeze = experiment_lifecycle.freeze_attempt(
                worktree,
                attempt_id=attempt_id,
                command=[
                    sys.executable,
                    "synthetic_worker.py",
                    str(attempt_dir),
                ],
                configuration={"execution_scope": "synthetic-test"},
                runtime_versions={"python_version": sys.version.split()[0]},
                device={"type": "cpu"},
                worker_count=1,
                expected_runtime_seconds=60,
                wall_time_limit_seconds=180,
                minimum_free_disk_bytes=0,
                outputs=["synthetic-output"],
                input_artifacts={},
                random_seeds=[],
            )
            terminal = experiment_lifecycle.execute_attempt(
                freeze,
                expected_manifest_sha256=experiment_lifecycle.sha256_file(
                    freeze / "manifest.json"
                ),
            )
            self.assertEqual(
                terminal["status"],
                "completed",
                f"{terminal}\n{(freeze / 'stderr.log').read_text()}",
            )
            lifecycle_receipt = experiment_lifecycle.verify_attempt(freeze)
            self.assertTrue(lifecycle_receipt["integrity_verified"])
            self.assertEqual(lifecycle_receipt["outputs_verified"], 1)
            independent = subprocess.run(
                [sys.executable, "synthetic_verifier.py", str(freeze)],
                cwd=worktree,
                check=True,
                capture_output=True,
                text=True,
            )
            independent_result = json.loads(independent.stdout)
            self.assertTrue(independent_result["passed"])
            self.assertEqual(independent_result["states"], 512)
            self.assertEqual(independent_result["model_state_evaluations"], 1024)

    def test_actual_run_attempt_and_verify_attempt_with_synthetic_numpy_artifacts(self):
        from ml.alphazero_lite import seed639_authorization as authorization

        source_root = Path(__file__).resolve().parents[2]
        scratch = source_root / ".tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            repository = Path(temporary) / "repo"
            repository.mkdir()
            run_git(repository, "init", "--initial-branch=main")
            (repository / ".gitignore").write_text(
                "/.worktrees/\n**/__pycache__/\n*.pyc\n"
            )
            shutil.copytree(
                source_root / "ml/alphazero_lite",
                repository / "ml/alphazero_lite",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
            (repository / "ml/__init__.py").write_text("")
            for relative in authorization.REQUIRED_PROTOCOLS:
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, target)

            synthetic_rows = synthetic_population()
            inventory_sha = hashlib.sha256(
                diagnostic.canonical_json(synthetic_rows)
            ).hexdigest()
            seed615_root = repository / "synthetic-seed615"
            eval_root = (
                seed615_root
                / "docs/data/seed615-source-opening-diversity/evaluation-preparation"
            )
            artifact_root = eval_root / "runtime-artifacts"
            model_hashes = {}
            runtime_records = {}
            for model, biases in (
                ("B512", [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
                ("S615", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ):
                model_directory = artifact_root / model
                model_directory.mkdir(parents=True)
                model_bytes = f"synthetic-{model}-npz".encode()
                weights_bytes = json.dumps(
                    {
                        "w1": [[0.0, 0.0] for _ in range(27)],
                        "b1": [0.0, 0.0],
                        "w2": [[0.0, 0.0], [0.0, 0.0]],
                        "b2": [0.0, 0.0],
                        "w_policy": [[0.0] * 6, [0.0] * 6],
                        "b_policy": biases,
                        "w_value": [[0.0], [0.0]],
                        "b_value": [0.0],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                model_sha = hashlib.sha256(model_bytes).hexdigest()
                weights_sha = hashlib.sha256(weights_bytes).hexdigest()
                metadata = {
                    "schema_version": "azlite_model_v1",
                    "game": "kalah",
                    "rules_version": "kalah_v1",
                    "input_encoding": "kalah_v3",
                    "architecture": {"model_type": "mlp_v1"},
                    "framework": "numpy",
                    "artifacts": {
                        "weights_file": "weights.json",
                        "weights_fallback_file": "model.npz",
                        "weights_sha256": model_sha,
                        "weights_json_sha256": weights_sha,
                    },
                }
                metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
                (model_directory / "model.npz").write_bytes(model_bytes)
                (model_directory / "weights.json").write_bytes(weights_bytes)
                (model_directory / "metadata.json").write_bytes(metadata_bytes)
                metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
                model_hashes[model] = {
                    "model.npz": model_sha,
                    "weights.json": weights_sha,
                    "metadata.json": metadata_sha,
                }
                runtime_records[model] = {
                    "model_npz_sha256": model_sha,
                    "weights_json_sha256": weights_sha,
                    "metadata_sha256": metadata_sha,
                }

            freeze_path = eval_root / "execution-freeze-v4.json"
            freeze_path.write_text(
                json.dumps({"runtime_artifacts": runtime_records}, sort_keys=True)
            )
            freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            p_e4_sha = diagnostic.INHERITED_P_E4_SHA256
            checkpoint_identity = {
                "B512": model_hashes["B512"]["model.npz"],
                "S615": model_hashes["S615"]["model.npz"],
                "historical_seed615_evaluation_freeze_sha256": freeze_sha,
            }

            seed636_attempt = (
                repository / "docs/data/seed636-prospective-population/attempt-0001"
            )
            seed636_attempt.mkdir(parents=True)
            population_member = diagnostic.canonical_json(
                {"selection": {"states": synthetic_rows}}
            )
            member_sha = hashlib.sha256(population_member).hexdigest()
            selected_manifest = {
                "schema": "seed636-selected-state-manifest-v1",
                "selected_count": 512,
                "selected_inventory_sha256": inventory_sha,
                "exposure_classification": "known-exposure-screened; inherited P-E4 exposure unresolved",
                "distinct_opening_clusters": 512,
                "max_per_game": 1,
                "max_per_opening": 1,
                "states": [
                    {
                        "i": index,
                        **{
                            name: row[name]
                            for name in (
                                "canonical_identity",
                                "production_float32_identity",
                                "active_pit_stones",
                                "game_id",
                                "opening_identity",
                                "opening_index",
                                "continuation_ply",
                            )
                        },
                    }
                    for index, row in enumerate(synthetic_rows)
                ],
            }
            selected_manifest_path = seed636_attempt / "selected-state-manifest.json"
            selected_manifest_path.write_text(
                json.dumps(selected_manifest, sort_keys=True)
            )
            selected_manifest_sha = hashlib.sha256(
                selected_manifest_path.read_bytes()
            ).hexdigest()
            selected_verification = {
                "schema": "seed636-selected-state-manifest-verification-v1",
                "status": "verified",
                "selected_count": 512,
                "selected_inventory_sha256": inventory_sha,
                "manifest_sha256": selected_manifest_sha,
                "population_member_sha256": member_sha,
                "checks": {
                    key: True
                    for key in (
                        "archive_hash",
                        "archive_sidecar_identity",
                        "canonical_ids_unique",
                        "exposure_binding",
                        "float32_ids_unique",
                        "game_cap",
                        "opening_cap",
                        "population_member_hash",
                        "projection_matches_archived_rows",
                        "selected_count",
                        "selected_inventory_hash",
                        "sidecar_hash",
                        "stone_quotas",
                    )
                },
            }
            selected_verification_path = (
                seed636_attempt / "selected-state-manifest-verification.json"
            )
            selected_verification_path.write_text(
                json.dumps(selected_verification, sort_keys=True)
            )
            archive_path = repository / "synthetic-archive/seed636-attempt-0001.tar.zst"
            archive_path.parent.mkdir(parents=True)
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("population/population.json")
                info.size = len(population_member)
                archive.addfile(info, io.BytesIO(population_member))
            sidecar_path = (
                repository
                / "synthetic-archive/seed636-attempt-0001.archive-manifest.json"
            )
            sidecar_path.write_text('{"synthetic":true}\n')

            input_paths = {
                str(freeze_path): hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                **{
                    str(artifact_root / model / filename): digest
                    for model, files_for_model in model_hashes.items()
                    for filename, digest in files_for_model.items()
                },
                str(selected_manifest_path): selected_manifest_sha,
                str(selected_verification_path): hashlib.sha256(
                    selected_verification_path.read_bytes()
                ).hexdigest(),
                str(archive_path): hashlib.sha256(
                    archive_path.read_bytes()
                ).hexdigest(),
                str(sidecar_path): hashlib.sha256(
                    sidecar_path.read_bytes()
                ).hexdigest(),
            }
            external_hashes = {
                "selected_manifest": input_paths[str(selected_manifest_path)],
                "selected_verification": input_paths[str(selected_verification_path)],
                "archive": input_paths[str(archive_path)],
                "archive_manifest": input_paths[str(sidecar_path)],
                "population_member": member_sha,
            }
            fixture = {
                "model_hashes": model_hashes,
                "input_hashes": external_hashes,
                "checkpoint_identity": checkpoint_identity,
                "selected_inventory_sha256": inventory_sha,
                "seed615_freeze_sha256": input_paths[str(freeze_path)],
                "p_e4_sha256": p_e4_sha,
                "archive_member_size": len(population_member),
            }
            fixture_path = repository / "synthetic-entrypoint-fixture.json"
            fixture_path.write_text(json.dumps(fixture, sort_keys=True))

            controls_source = """
import json
from collections import Counter
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
from ml.alphazero_lite import arena, experiment_lifecycle, seed639_authorization, seed639_policy_prior_classification as producer
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier

@contextmanager
def controls(attempt_dir, fixture_path):
    attempt = Path(attempt_dir)
    fixture = json.loads(Path(fixture_path).read_bytes())
    manifest = json.loads((attempt / "manifest.json").read_bytes())
    pull = {"number": 639, "state": "closed", "merged_at": "2026-11-26T00:00:00Z",
            "merged_by": {"login": "useruby"}, "merge_commit_sha": manifest["source_commit"],
            "base": {"sha": manifest["base_commit"], "ref": "main",
                     "repo": {"full_name": "useruby/mancala-ai"}},
            "html_url": "https://example.invalid/pull/639"}
    def gh(endpoint):
        if "/commits/" in endpoint: return [{"number": 639}]
        if endpoint.endswith("/pulls/639"): return pull
        raise AssertionError(endpoint)
    def historical_auth(*, registered_runtime, **_kwargs):
        return {"freeze_sha256": fixture["seed615_freeze_sha256"],
                "checkpoint_sha256": {"B512": fixture["model_hashes"]["B512"]["model.npz"],
                                      "S615": fixture["model_hashes"]["S615"]["model.npz"],
                                      "P-E4": fixture["p_e4_sha256"]},
                "historical_runtime": registered_runtime}
    def archive_verify(_archive, _sidecar):
        return {"experiment_id": "seed636-prospective-population", "attempt_id": "attempt-0001",
                "files": [{"path": "population/population.json",
                           "size_bytes": fixture["archive_member_size"],
                           "sha256": fixture["input_hashes"]["population_member"]}]}
    original = arena.ArtifactEvaluator
    class CountingEvaluator(original):
        calls = Counter()
        def __init__(self, directory):
            super().__init__(directory)
            self.label = Path(directory).name
        def evaluate(self, game):
            type(self).calls[self.label] += 1
            return super().evaluate(game)
    patches = [
        patch.object(seed639_authorization, "_github_json", side_effect=gh),
        patch.object(experiment_lifecycle, "verify_archive", side_effect=archive_verify),
        patch.object(producer, "authenticate_seed615_in_subprocess", side_effect=historical_auth),
        patch.object(arena, "ArtifactEvaluator", CountingEvaluator),
        patch.object(producer, "REGISTERED_INPUT_SHA256", fixture["input_hashes"]),
        patch.object(producer, "MODEL_ARTIFACTS", fixture["model_hashes"]),
        patch.object(producer, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]),
        patch.object(producer, "SEED615_EVALUATION_FREEZE_SHA256", fixture["seed615_freeze_sha256"]),
        patch.object(seed639_authorization, "CHECKPOINT_IDENTITY", fixture["checkpoint_identity"]),
        patch.object(seed639_authorization, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]),
        patch.object(verifier, "EXPECTED_MODEL_FILES", fixture["model_hashes"]),
        patch.object(verifier, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]),
        patch.object(verifier, "EXPECTED_SELECTED_MANIFEST_SHA256", fixture["input_hashes"]["selected_manifest"]),
        patch.object(verifier, "EXPECTED_SELECTED_VERIFICATION_SHA256", fixture["input_hashes"]["selected_verification"]),
        patch.object(verifier, "EXPECTED_SEED615_FREEZE_SHA256", fixture["seed615_freeze_sha256"]),
    ]
    with ExitStack() as stack:
        for item in patches: stack.enter_context(item)
        yield fixture, CountingEvaluator
"""
            worker_source = """
import json, sys
from ml.alphazero_lite import seed639_policy_prior_classification as producer
from synthetic_entrypoint_controls import controls
with controls(sys.argv[1], sys.argv[2]) as (fixture, evaluator):
    result = producer.run_attempt(__import__("pathlib").Path(sys.argv[1]))
    print(json.dumps({"result": result, "evaluator_calls": dict(evaluator.calls)}, sort_keys=True))
"""
            verify_source = """
import json, sys
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier
from synthetic_entrypoint_controls import controls
with controls(sys.argv[1], sys.argv[2]) as (fixture, evaluator):
    result = verifier.verify_attempt(__import__("pathlib").Path(sys.argv[1]))
    print(json.dumps({"result": result, "evaluator_calls": dict(evaluator.calls)}, sort_keys=True))
"""
            (repository / "synthetic_entrypoint_controls.py").write_text(
                controls_source
            )
            (
                repository / "synthetic_seed639_policy_prior_classification_worker.py"
            ).write_text(worker_source)
            (repository / "synthetic_seed639_verify_worker.py").write_text(
                verify_source
            )
            run_git(repository, "add", ".")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=Seed639 test",
                    "-c",
                    "user.email=seed639@example.invalid",
                    "commit",
                    "-m",
                    "synthetic base",
                ],
                check=True,
                capture_output=True,
            )
            base_commit = run_git(repository, "rev-parse", "HEAD")

            registration = json.loads(
                (
                    source_root
                    / "docs/experiments/seed639-prospective-policy-prior-classification/registration.json"
                ).read_bytes()
            )
            registration.pop("execution_amendment", None)
            for runtime_input in authorization.AMENDMENT_RUNTIME_INPUTS:
                registration["input_artifacts"].pop(runtime_input, None)
            reserved = {
                "schema",
                "experiment_id",
                "experiment_type",
                "authorization_reference",
                "hypothesis",
                "objective",
                "base_commit",
                "branch",
                "random_seeds",
                "input_artifacts",
                "checkpoint_identity",
                "acceptance_criteria",
                "registered_at",
            }
            metadata = {
                key: value for key, value in registration.items() if key not in reserved
            }
            metadata["scientific_acceptance"] = {
                **metadata["scientific_acceptance"],
                "selected_inventory_sha256": inventory_sha,
            }
            worktree = experiment_lifecycle.create_worktree(
                repository,
                authorization.EXPERIMENT_ID,
                "experiment/seed639-actual-entrypoints-synthetic",
                base_commit,
                hypothesis=registration["hypothesis"],
                objective=registration["objective"],
                experiment_type=registration["experiment_type"],
                random_seeds=[],
                input_artifacts=input_paths,
                checkpoint_identity=checkpoint_identity,
                acceptance_criteria=registration["acceptance_criteria"],
                worktree_root=repository / ".worktrees",
                registration_metadata=metadata,
            )
            run_git(worktree, "add", ".experiment/.gitignore", "docs/experiments")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(worktree),
                    "-c",
                    "user.name=Seed639 test",
                    "-c",
                    "user.email=seed639@example.invalid",
                    "commit",
                    "-m",
                    "introduce registration",
                ],
                check=True,
                capture_output=True,
            )
            permanent_registration = worktree / authorization.REGISTRATION_PATH
            registration_sha = hashlib.sha256(
                permanent_registration.read_bytes()
            ).hexdigest()
            environment = diagnostic.PINNED_ENVIRONMENT.copy()
            configuration = {
                "registration_sha256": registration_sha,
                "execution_scope": authorization.EXECUTION_SCOPE,
                "input_encoding": "kalah_v3",
                "inference_batch_size": 1,
                "evaluation_cache": False,
                "execution_concurrency": 1,
                "environment": environment,
                "numerical_backend": "NumPy ArtifactEvaluator",
                "thread_configuration": {
                    "environment": environment,
                    "threads_per_pool": 1,
                },
                "historical_verifier_runtime": authorization.HISTORICAL_VERIFIER_RUNTIME,
            }
            attempt = worktree / ".experiment/runtime/attempts/attempt-0001"
            freeze = experiment_lifecycle.freeze_attempt(
                worktree,
                attempt_id="attempt-0001",
                command=[
                    sys.executable,
                    "synthetic_seed639_policy_prior_classification_worker.py",
                    str(attempt),
                    str(fixture_path),
                ],
                configuration=configuration,
                runtime_versions={
                    "python_version": platform.python_version(),
                    "numpy": np.__version__,
                    "numerical_backend": "NumPy ArtifactEvaluator",
                },
                device={"type": "cpu"},
                worker_count=1,
                expected_runtime_seconds=600,
                wall_time_limit_seconds=14_400,
                minimum_free_disk_bytes=5_368_709_120,
                outputs=["seed639-output"],
                input_artifacts=input_paths,
                random_seeds=[],
                additional_resource_limits={
                    "cpu_time_limit_seconds": 10_800,
                    "process_rss_limit_bytes": 8_589_934_592,
                    "output_and_temporary_storage_limit_bytes": 5_368_709_120,
                },
                environment=environment,
            )
            executed = experiment_lifecycle.execute_attempt(
                freeze,
                expected_manifest_sha256=experiment_lifecycle.sha256_file(
                    freeze / "manifest.json"
                ),
            )
            self.assertEqual(executed["status"], "completed")
            producer_report = json.loads(
                (freeze / "stdout.log").read_bytes().splitlines()[-1]
            )
            self.assertEqual(
                producer_report["evaluator_calls"], {"B512": 512, "S615": 512}
            )
            self.assertTrue(
                experiment_lifecycle.verify_attempt(freeze)["integrity_verified"]
            )

            verify_environment = os.environ.copy() | environment
            independent = subprocess.run(
                [
                    sys.executable,
                    "synthetic_seed639_verify_worker.py",
                    str(freeze),
                    str(fixture_path),
                ],
                cwd=worktree,
                check=True,
                capture_output=True,
                text=True,
                env=verify_environment,
            )
            verify_report = json.loads(independent.stdout)
            self.assertTrue(verify_report["result"]["passed"])
            self.assertEqual(verify_report["result"]["states_verified"], 512)
            self.assertEqual(
                verify_report["result"]["model_state_evaluations_reproduced"], 1024
            )
            self.assertEqual(
                verify_report["evaluator_calls"], {"B512": 512, "S615": 512}
            )

    def test_actual_run_and_independent_verify_entrypoints_with_synthetic_artifacts(
        self,
    ):
        from ml.alphazero_lite import seed639_authorization as authorization

        source_root = Path(__file__).resolve().parents[2]
        worktree_temp_root = source_root / ".tmp"
        created_scratch_parent = not worktree_temp_root.exists()
        worktree_temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=worktree_temp_root) as temporary:
            repository = Path(temporary) / "repo"
            repository.mkdir()
            run_git(repository, "init", "--initial-branch=main")
            (repository / ".gitignore").write_text(
                "/.worktrees/\n**/__pycache__/\n*.pyc\n"
            )
            shutil.copytree(
                source_root / "ml/alphazero_lite",
                repository / "ml/alphazero_lite",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
            (repository / "ml/__init__.py").write_text("")
            for relative in authorization.REQUIRED_PROTOCOLS:
                target = repository / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, target)

            seed615_root = repository / "synthetic-seed615"
            eval_root = (
                seed615_root
                / "docs/data/seed615-source-opening-diversity/evaluation-preparation"
            )
            artifacts_root = eval_root / "runtime-artifacts"
            rows = synthetic_population()
            selected_inventory_sha256 = hashlib.sha256(
                diagnostic.canonical_json(rows)
            ).hexdigest()
            model_hashes = {}
            model_records = {}
            for model, biases in (
                ("B512", [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
                ("S615", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ):
                artifact_dir = artifacts_root / model
                artifact_dir.mkdir(parents=True)
                model_bytes = f"synthetic-{model}-checkpoint".encode()
                weights_doc = {
                    "w1": [[0.0, 0.0] for _ in range(27)],
                    "b1": [0.0, 0.0],
                    "w2": [[0.0, 0.0], [0.0, 0.0]],
                    "b2": [0.0, 0.0],
                    "w_policy": [[0.0] * 6, [0.0] * 6],
                    "b_policy": biases,
                    "w_value": [[0.0], [0.0]],
                    "b_value": [0.0],
                }
                weights_bytes = json.dumps(
                    weights_doc, sort_keys=True, separators=(",", ":")
                ).encode()
                model_sha = hashlib.sha256(model_bytes).hexdigest()
                weights_sha = hashlib.sha256(weights_bytes).hexdigest()
                metadata_doc = {
                    "schema_version": "azlite_model_v1",
                    "game": "kalah",
                    "rules_version": "kalah_v1",
                    "input_encoding": "kalah_v3",
                    "architecture": {"model_type": "mlp_v1"},
                    "framework": "numpy",
                    "artifacts": {
                        "weights_file": "weights.json",
                        "weights_fallback_file": "model.npz",
                        "weights_sha256": model_sha,
                        "weights_json_sha256": weights_sha,
                    },
                }
                (artifact_dir / "model.npz").write_bytes(model_bytes)
                (artifact_dir / "weights.json").write_bytes(weights_bytes)
                (artifact_dir / "metadata.json").write_text(
                    json.dumps(metadata_doc, sort_keys=True)
                )
                metadata_sha = hashlib.sha256(
                    (artifact_dir / "metadata.json").read_bytes()
                ).hexdigest()
                model_hashes[model] = {
                    "model.npz": model_sha,
                    "weights.json": weights_sha,
                    "metadata.json": metadata_sha,
                }
                model_records[model] = {
                    "model_npz_sha256": model_sha,
                    "weights_json_sha256": weights_sha,
                    "metadata_sha256": metadata_sha,
                }

            freeze_path = eval_root / "execution-freeze-v4.json"
            freeze_path.write_text(
                json.dumps({"runtime_artifacts": model_records}, sort_keys=True)
            )
            freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            p_e4_sha = diagnostic.INHERITED_P_E4_SHA256
            checkpoint_identity = {
                "B512": model_hashes["B512"]["model.npz"],
                "S615": model_hashes["S615"]["model.npz"],
                "historical_seed615_evaluation_freeze_sha256": freeze_sha,
            }

            seed636_attempt = (
                repository / "docs/data/seed636-prospective-population/attempt-0001"
            )
            seed636_attempt.mkdir(parents=True)
            selected_manifest = {
                "schema": "seed636-selected-state-manifest-v1",
                "selected_count": 512,
                "selected_inventory_sha256": selected_inventory_sha256,
                "exposure_classification": "known-exposure-screened; inherited P-E4 exposure unresolved",
                "distinct_opening_clusters": 512,
                "max_per_game": 1,
                "max_per_opening": 1,
                "states": [
                    {
                        "i": index,
                        **{
                            key: row[key]
                            for key in (
                                "canonical_identity",
                                "production_float32_identity",
                                "active_pit_stones",
                                "game_id",
                                "opening_identity",
                                "opening_index",
                                "continuation_ply",
                            )
                        },
                    }
                    for index, row in enumerate(rows)
                ],
            }
            selected_manifest_path = seed636_attempt / "selected-state-manifest.json"
            selected_manifest_path.write_text(
                json.dumps(selected_manifest, sort_keys=True)
            )
            selected_manifest_sha = hashlib.sha256(
                selected_manifest_path.read_bytes()
            ).hexdigest()
            population_bundle = {"selection": {"states": rows}}
            population_bytes = diagnostic.canonical_json(population_bundle)
            population_sha = hashlib.sha256(population_bytes).hexdigest()
            archive_path = repository / "synthetic-archive/seed636-attempt-0001.tar.zst"
            archive_path.parent.mkdir(parents=True)
            with tarfile.open(archive_path, "w:gz") as archive:
                member_info = tarfile.TarInfo("population/population.json")
                member_info.size = len(population_bytes)
                archive.addfile(member_info, io.BytesIO(population_bytes))
            archive_manifest_path = (
                repository
                / "synthetic-archive/seed636-attempt-0001.archive-manifest.json"
            )
            archive_manifest_path.write_text('{"synthetic":true}\n')
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            archive_manifest_sha = hashlib.sha256(
                archive_manifest_path.read_bytes()
            ).hexdigest()
            check_names = (
                "archive_hash",
                "archive_sidecar_identity",
                "canonical_ids_unique",
                "exposure_binding",
                "float32_ids_unique",
                "game_cap",
                "opening_cap",
                "population_member_hash",
                "projection_matches_archived_rows",
                "selected_count",
                "selected_inventory_hash",
                "sidecar_hash",
                "stone_quotas",
            )
            selected_verification = {
                "schema": "seed636-selected-state-manifest-verification-v1",
                "status": "verified",
                "selected_count": 512,
                "selected_inventory_sha256": selected_inventory_sha256,
                "manifest_sha256": selected_manifest_sha,
                "population_member_sha256": population_sha,
                "checks": {name: True for name in check_names},
            }
            selected_verification_path = (
                seed636_attempt / "selected-state-manifest-verification.json"
            )
            selected_verification_path.write_text(
                json.dumps(selected_verification, sort_keys=True)
            )

            input_paths = {
                str(freeze_path): freeze_sha,
                **{
                    str(artifacts_root / model / filename): digest
                    for model, model_files in model_hashes.items()
                    for filename, digest in model_files.items()
                },
                str(selected_manifest_path): selected_manifest_sha,
                str(selected_verification_path): hashlib.sha256(
                    selected_verification_path.read_bytes()
                ).hexdigest(),
                str(archive_path): archive_sha,
                str(archive_manifest_path): archive_manifest_sha,
            }
            fixture = {
                "model_hashes": model_hashes,
                "input_hashes": {
                    "selected_manifest": selected_manifest_sha,
                    "selected_verification": input_paths[
                        str(selected_verification_path)
                    ],
                    "archive": archive_sha,
                    "archive_manifest": archive_manifest_sha,
                    "population_member": population_sha,
                },
                "checkpoint_identity": checkpoint_identity,
                "selected_inventory_sha256": selected_inventory_sha256,
                "seed615_freeze_sha256": freeze_sha,
                "p_e4_sha256": p_e4_sha,
                "archive_member_size": len(population_bytes),
            }
            fixture_path = repository / "synthetic-seed639-fixture.json"
            fixture_path.write_text(json.dumps(fixture, sort_keys=True))

            worker_source = """
import json, sys
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from ml.alphazero_lite import experiment_lifecycle, seed639_authorization, seed639_policy_prior_classification as producer
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier
from ml.alphazero_lite import arena

attempt = Path(sys.argv[1])
fixture = json.loads(Path(sys.argv[2]).read_bytes())
manifest = json.loads((attempt / "manifest.json").read_bytes())
pull = {"number": 639, "state": "closed", "merged_at": "2026-11-26T00:00:00Z",
        "merged_by": {"login": "useruby"}, "merge_commit_sha": manifest["source_commit"],
        "base": {"sha": manifest["base_commit"], "ref": "main",
                 "repo": {"full_name": "useruby/mancala-ai"}},
        "html_url": "https://example.invalid/pull/639"}
def gh(endpoint):
    if "/commits/" in endpoint:
        return [{"number": 639}]
    if endpoint.endswith("/pulls/639"):
        return pull
    raise AssertionError(endpoint)
def historical_auth(*, registered_runtime, **_kwargs):
    return {"freeze_sha256": fixture["seed615_freeze_sha256"],
            "checkpoint_sha256": {"B512": fixture["model_hashes"]["B512"]["model.npz"],
                                  "S615": fixture["model_hashes"]["S615"]["model.npz"],
                                  "P-E4": fixture["p_e4_sha256"]},
            "historical_runtime": registered_runtime}
def archive_verify(_archive, _sidecar):
    return {"experiment_id": "seed636-prospective-population", "attempt_id": "attempt-0001",
            "files": [{"path": "population/population.json",
                       "size_bytes": fixture["archive_member_size"],
                       "sha256": fixture["input_hashes"]["population_member"]}]}
original_evaluator = arena.ArtifactEvaluator
class CountingEvaluator(original_evaluator):
    calls = Counter()
    def __init__(self, artifact_dir):
        super().__init__(artifact_dir)
        self.model_name = Path(artifact_dir).name
    def evaluate(self, game):
        type(self).calls[self.model_name] += 1
        return super().evaluate(game)
with patch.object(seed639_authorization, "_github_json", side_effect=gh), \\
     patch.object(experiment_lifecycle, "verify_archive", side_effect=archive_verify), \\
     patch.object(producer, "authenticate_seed615_in_subprocess", side_effect=historical_auth), \\
     patch.object(arena, "ArtifactEvaluator", CountingEvaluator), \\
     patch.object(producer, "REGISTERED_INPUT_SHA256", fixture["input_hashes"]), \\
     patch.object(producer, "MODEL_ARTIFACTS", fixture["model_hashes"]), \\
     patch.object(producer, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]), \\
     patch.object(producer, "SEED615_EVALUATION_FREEZE_SHA256", fixture["seed615_freeze_sha256"]), \\
     patch.object(seed639_authorization, "CHECKPOINT_IDENTITY", fixture["checkpoint_identity"]), \\
     patch.object(seed639_authorization, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]):
    result = producer.run_attempt(attempt)
    print(json.dumps({"result": result, "producer_evaluator_calls": dict(CountingEvaluator.calls)}, sort_keys=True))
"""
            verifier_source = """
import json, sys
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from ml.alphazero_lite import experiment_lifecycle, seed639_authorization, seed639_policy_prior_classification as producer
from ml.alphazero_lite import verify_seed639_policy_prior_classification as verifier
from ml.alphazero_lite import arena

attempt = Path(sys.argv[1])
fixture = json.loads(Path(sys.argv[2]).read_bytes())
manifest = json.loads((attempt / "manifest.json").read_bytes())
pull = {"number": 639, "state": "closed", "merged_at": "2026-11-26T00:00:00Z",
        "merged_by": {"login": "useruby"}, "merge_commit_sha": manifest["source_commit"],
        "base": {"sha": manifest["base_commit"], "ref": "main",
                 "repo": {"full_name": "useruby/mancala-ai"}},
        "html_url": "https://example.invalid/pull/639"}
def gh(endpoint):
    if "/commits/" in endpoint:
        return [{"number": 639}]
    if endpoint.endswith("/pulls/639"):
        return pull
    raise AssertionError(endpoint)
def historical_auth(*, registered_runtime, **_kwargs):
    return {"freeze_sha256": fixture["seed615_freeze_sha256"],
            "checkpoint_sha256": {"B512": fixture["model_hashes"]["B512"]["model.npz"],
                                  "S615": fixture["model_hashes"]["S615"]["model.npz"],
                                  "P-E4": fixture["p_e4_sha256"]},
            "historical_runtime": registered_runtime}
def archive_verify(_archive, _sidecar):
    return {"experiment_id": "seed636-prospective-population", "attempt_id": "attempt-0001",
            "files": [{"path": "population/population.json",
                       "size_bytes": fixture["archive_member_size"],
                       "sha256": fixture["input_hashes"]["population_member"]}]}
original_evaluator = arena.ArtifactEvaluator
class CountingEvaluator(original_evaluator):
    calls = Counter()
    def __init__(self, artifact_dir):
        super().__init__(artifact_dir)
        self.model_name = Path(artifact_dir).name
    def evaluate(self, game):
        type(self).calls[self.model_name] += 1
        return super().evaluate(game)
with patch.object(seed639_authorization, "_github_json", side_effect=gh), \\
     patch.object(experiment_lifecycle, "verify_archive", side_effect=archive_verify), \\
     patch.object(producer, "authenticate_seed615_in_subprocess", side_effect=historical_auth), \\
     patch.object(arena, "ArtifactEvaluator", CountingEvaluator), \\
     patch.object(producer, "REGISTERED_INPUT_SHA256", fixture["input_hashes"]), \\
     patch.object(producer, "MODEL_ARTIFACTS", fixture["model_hashes"]), \\
     patch.object(producer, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]), \\
     patch.object(producer, "SEED615_EVALUATION_FREEZE_SHA256", fixture["seed615_freeze_sha256"]), \\
     patch.object(verifier, "EXPECTED_MODEL_FILES", fixture["model_hashes"]), \\
     patch.object(verifier, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]), \\
     patch.object(verifier, "EXPECTED_SELECTED_MANIFEST_SHA256", fixture["input_hashes"]["selected_manifest"]), \\
     patch.object(verifier, "EXPECTED_SELECTED_VERIFICATION_SHA256", fixture["input_hashes"]["selected_verification"]), \\
     patch.object(verifier, "EXPECTED_SEED615_FREEZE_SHA256", fixture["seed615_freeze_sha256"]), \\
     patch.object(seed639_authorization, "CHECKPOINT_IDENTITY", fixture["checkpoint_identity"]), \\
     patch.object(seed639_authorization, "SELECTED_INVENTORY_SHA256", fixture["selected_inventory_sha256"]):
    result = verifier.verify_attempt(attempt)
    print(json.dumps({"result": result, "verifier_evaluator_calls": dict(CountingEvaluator.calls)}, sort_keys=True))
"""
            (
                repository / "synthetic_seed639_policy_prior_classification_worker.py"
            ).write_text(worker_source)
            (repository / "synthetic_seed639_verify_worker.py").write_text(
                verifier_source
            )
            run_git(repository, "add", ".")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=Seed639 test",
                    "-c",
                    "user.email=seed639@example.invalid",
                    "commit",
                    "-m",
                    "synthetic base",
                ],
                check=True,
                capture_output=True,
            )
            base_commit = run_git(repository, "rev-parse", "HEAD")

            synthetic_registration = json.loads(
                (
                    source_root
                    / "docs/experiments/seed639-prospective-policy-prior-classification/registration.json"
                ).read_bytes()
            )
            synthetic_registration.pop("execution_amendment", None)
            for runtime_input in authorization.AMENDMENT_RUNTIME_INPUTS:
                synthetic_registration["input_artifacts"].pop(runtime_input, None)
            reserved = {
                "schema",
                "experiment_id",
                "experiment_type",
                "authorization_reference",
                "hypothesis",
                "objective",
                "base_commit",
                "branch",
                "random_seeds",
                "input_artifacts",
                "checkpoint_identity",
                "acceptance_criteria",
                "registered_at",
            }
            registration_metadata = {
                key: value
                for key, value in synthetic_registration.items()
                if key not in reserved
            }
            registration_metadata["scientific_acceptance"] = {
                **registration_metadata["scientific_acceptance"],
                "selected_inventory_sha256": selected_inventory_sha256,
            }
            checkpoint_identity = {
                "B512": model_hashes["B512"]["model.npz"],
                "S615": model_hashes["S615"]["model.npz"],
                "historical_seed615_evaluation_freeze_sha256": freeze_sha,
            }
            worktree = experiment_lifecycle.create_worktree(
                repository,
                authorization.EXPERIMENT_ID,
                "experiment/seed639-actual-entrypoints-synthetic",
                base_commit,
                hypothesis=synthetic_registration["hypothesis"],
                objective=synthetic_registration["objective"],
                experiment_type="retrospective_diagnostic",
                random_seeds=[],
                input_artifacts=input_paths,
                checkpoint_identity=checkpoint_identity,
                acceptance_criteria=synthetic_registration["acceptance_criteria"],
                worktree_root=repository / ".worktrees",
                registration_metadata=registration_metadata,
            )
            run_git(worktree, "add", ".experiment/.gitignore", "docs/experiments")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(worktree),
                    "-c",
                    "user.name=Seed639 test",
                    "-c",
                    "user.email=seed639@example.invalid",
                    "commit",
                    "-m",
                    "introduce registration",
                ],
                check=True,
                capture_output=True,
            )
            registration_path = worktree / authorization.REGISTRATION_PATH
            registration_sha = hashlib.sha256(
                registration_path.read_bytes()
            ).hexdigest()
            environment = diagnostic.PINNED_ENVIRONMENT.copy()
            configuration = {
                "registration_sha256": registration_sha,
                "execution_scope": authorization.EXECUTION_SCOPE,
                "input_encoding": "kalah_v3",
                "inference_batch_size": 1,
                "evaluation_cache": False,
                "execution_concurrency": 1,
                "environment": environment,
                "numerical_backend": "NumPy ArtifactEvaluator",
                "thread_configuration": {
                    "environment": environment,
                    "threads_per_pool": 1,
                },
                "historical_verifier_runtime": authorization.HISTORICAL_VERIFIER_RUNTIME,
            }
            attempt_id = "attempt-0001"
            attempt_path = worktree / ".experiment/runtime/attempts" / attempt_id
            attempt = experiment_lifecycle.freeze_attempt(
                worktree,
                attempt_id=attempt_id,
                command=[
                    sys.executable,
                    "synthetic_seed639_policy_prior_classification_worker.py",
                    str(attempt_path),
                    str(fixture_path),
                ],
                configuration=configuration,
                runtime_versions={
                    "python_version": platform.python_version(),
                    "numpy": np.__version__,
                    "numerical_backend": "NumPy ArtifactEvaluator",
                },
                device={"type": "cpu"},
                worker_count=1,
                expected_runtime_seconds=600,
                wall_time_limit_seconds=14_400,
                minimum_free_disk_bytes=5_368_709_120,
                outputs=["seed639-output"],
                input_artifacts=input_paths,
                random_seeds=[],
                additional_resource_limits={
                    "cpu_time_limit_seconds": 10_800,
                    "process_rss_limit_bytes": 8_589_934_592,
                    "output_and_temporary_storage_limit_bytes": 5_368_709_120,
                },
                environment=environment,
            )
            terminal = experiment_lifecycle.execute_attempt(
                attempt,
                expected_manifest_sha256=experiment_lifecycle.sha256_file(
                    attempt / "manifest.json"
                ),
            )
            self.assertEqual(
                terminal["status"],
                "completed",
                f"{terminal}\n{(attempt / 'stderr.log').read_text()}",
            )
            self.assertTrue(
                experiment_lifecycle.verify_attempt(attempt)["integrity_verified"]
            )
            test_environment = os.environ.copy() | environment
            verify = subprocess.run(
                [
                    sys.executable,
                    "synthetic_seed639_verify_worker.py",
                    str(attempt),
                    str(fixture_path),
                ],
                cwd=worktree,
                check=True,
                capture_output=True,
                text=True,
                env=test_environment,
            )
            verified = json.loads(verify.stdout)
            self.assertTrue(verified["result"]["passed"])
            self.assertEqual(verified["result"]["states_verified"], 512)
            self.assertEqual(
                verified["result"]["model_state_evaluations_reproduced"], 1024
            )
            self.assertEqual(
                verified["verifier_evaluator_calls"], {"B512": 512, "S615": 512}
            )
        if created_scratch_parent:
            worktree_temp_root.rmdir()

    def test_inference_failure_keeps_incremental_records_and_failure_receipt(self):
        population = synthetic_population()[:3]
        population = [
            {**row, "active_pit_stones": row["active_pit_stones"]} for row in population
        ]
        b512 = FixedEvaluator([1 / 6] * 6)
        s615 = FixedEvaluator([1 / 6] * 6, fail_on_call=2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            records = output / "state-records"
            records.mkdir()
            with self.assertRaisesRegex(RuntimeError, "injected_inference_failure"):
                diagnostic.persist_inference_records(population, b512, s615, records)
            diagnostic.write_failure_receipt(
                output,
                attempt_id="synthetic-attempt",
                source_commit="f" * 40,
                error=RuntimeError("injected_inference_failure"),
                started_at=0.0,
                completed_records=0,
            )
            self.assertTrue((records / "state-0000.json").is_file())
            self.assertFalse((records / "state-0001.json").exists())
            receipt = json.loads((output / "failure.json").read_bytes())
            self.assertEqual(receipt["partial_record_count"], 1)
            self.assertEqual(receipt["exception_type"], "RuntimeError")

    def test_summary_uses_equal_fixed_bin_weights(self):
        rows = []
        for stones, disagreement in zip((17, 18, 19, 20, 21), (0, 1, 0, 1, 0)):
            for index in range(102):
                rows.append(
                    {
                        "active_pit_stones": stones,
                        "total_variation": float(disagreement),
                        "top_action_disagreement": bool(disagreement),
                        "B512_entropy_bits": float(disagreement),
                        "S615_entropy_bits": float(disagreement),
                        "entropy_difference_S615_minus_B512_bits": float(disagreement),
                    }
                )
        rows.extend(
            [
                {
                    "active_pit_stones": 17,
                    "total_variation": 0.0,
                    "top_action_disagreement": False,
                    "B512_entropy_bits": 0.0,
                    "S615_entropy_bits": 0.0,
                    "entropy_difference_S615_minus_B512_bits": 0.0,
                },
                {
                    "active_pit_stones": 18,
                    "total_variation": 1.0,
                    "top_action_disagreement": True,
                    "B512_entropy_bits": 1.0,
                    "S615_entropy_bits": 1.0,
                    "entropy_difference_S615_minus_B512_bits": 1.0,
                },
            ]
        )
        summary = diagnostic.summarize_population(rows)
        self.assertEqual(summary["population_size"], 512)
        self.assertAlmostEqual(
            summary["fixed_one_fifth_bin_standardized"]["top_action_disagreement_rate"],
            0.4,
        )
        self.assertEqual(len(summary["overall"]["total_variation_quantiles"]), 8)


if __name__ == "__main__":
    unittest.main()
