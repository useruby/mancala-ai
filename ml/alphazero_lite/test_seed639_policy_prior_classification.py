"""Synthetic regression tests for Seed 639 policy comparison contracts."""

from __future__ import annotations

import unittest
import hashlib
import json
import shutil
import subprocess
import sys
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
            registration_path = repo / authorization.REGISTRATION_PATH
            registration_path.parent.mkdir(parents=True)
            registration_path.write_text(json.dumps(registration))
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
            source_commit = run_git(repo, "rev-parse", "HEAD")
            pull = {
                "number": 639,
                "state": "open",
                "merged_at": None,
                "merged_by": {"login": "useruby"},
                "merge_commit_sha": source_commit,
                "base": {
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
            (repository / ".gitignore").write_text("/.worktrees/\n")
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
            self.assertEqual(terminal["status"], "completed")
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
