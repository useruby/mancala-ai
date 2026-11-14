"""Synthetic guards for #588 execution and result verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pytest

from ml.alphazero_lite import run_seed588_raw_prior as runner
from ml.alphazero_lite import seed588_raw_prior_analysis as analysis_impl
from ml.alphazero_lite.verify_seed588_raw_prior_result import (
    VerificationError,
    _root_metrics,
    validate_membership,
    verify_frozen_artifacts,
    verify_frozen_implementation,
    verify_result,
)
from ml.alphazero_lite.seed588_raw_prior_diagnostic import _jsonl


def _authorization(scratch_dir: Path) -> tuple[Path, str, dict[str, object]]:
    freeze_path = runner.ROOT / runner.FREEZE_PATH.relative_to(runner.ROOT)
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    authorization = {
        "schema": runner.AUTH_SCHEMA,
        "status": "reviewed_and_authorized",
        "review_url": "https://example.invalid/review/approved",
        "reviewed_by": "synthetic-independent-reviewer",
        "execution_freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "source_and_implementation_sha256": runner.frozen_dependencies(),
        "source_evidence_sha256": runner.EXPECTED_INPUTS,
        "artifacts": runner.EXPECTED_ARTIFACTS,
        "runtime": freeze["runtime"],
    }
    path = scratch_dir / "reviewed-authorization.json"
    raw = json.dumps(authorization, sort_keys=True).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), authorization


@pytest.fixture
def scratch_dir():
    path = runner.ROOT / ".tmp" / f"seed588-test-{uuid.uuid4().hex}"
    path.mkdir(parents=False)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_executor_requires_external_authorization_before_evaluator(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluator must not be constructed")

    monkeypatch.setattr(runner.arena, "ArtifactEvaluator", forbidden)
    output = scratch_dir / "new-package"
    with pytest.raises(runner.ExecutionBlocked, match="required"):
        runner.execute(
            root=runner.ROOT,
            output=output,
            authorization_path=None,
            trusted_authorization_sha256=None,
        )
    assert not output.exists()


def test_authorization_rejects_altered_execution_freeze(
    scratch_dir: Path,
) -> None:
    authorization = {
        "schema": runner.AUTH_SCHEMA,
        "status": "reviewed_and_authorized",
        "review_url": "https://example.invalid/review",
        "reviewed_by": "synthetic",
        "execution_freeze_sha256": "0" * 64,
    }
    path = scratch_dir / "authorization.json"
    raw = json.dumps(authorization).encode()
    path.write_bytes(raw)
    with pytest.raises(runner.ExecutionBlocked, match="execution_freeze_hash_mismatch"):
        runner.verify_authorization(
            root=runner.ROOT,
            authorization_path=path,
            trusted_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_result_verifier_rejects_input_encoding_hash_change() -> None:
    freeze = json.loads(runner.FREEZE_PATH.read_bytes())
    freeze["implementation_sha256"]["ml/alphazero_lite/input_encodings.py"] = "0" * 64
    with pytest.raises(
        VerificationError, match="checked_out_implementation_hash_mismatch"
    ):
        verify_frozen_implementation(runner.ROOT, freeze)


def test_result_verifier_rejects_altered_frozen_artifact_identity() -> None:
    freeze = json.loads(runner.FREEZE_PATH.read_bytes())
    freeze["artifacts"]["T512"]["model.npz"] = "0" * 64
    with pytest.raises(VerificationError, match="freeze_artifact_identity:T512"):
        verify_frozen_artifacts(freeze)


def test_prior_parity_rejects_action_index_shift_and_excess_error() -> None:
    archived = [0.0, 0.75, 0.25, 0.0, 0.0, 0.0]
    assert runner.compare_parity(archived.copy(), archived, [1, 2]) == 0.0
    shifted = [0.75, 0.0, 0.25, 0.0, 0.0, 0.0]
    shifted_row = {
        "legal_mask_exact": all(shifted[action] == 0.0 for action in (0, 3, 4, 5)),
        "max_abs_difference": runner.compare_parity(shifted, archived, [1, 2]),
    }
    with pytest.raises(VerificationError, match="P_archived_prior_parity_failed"):
        runner.require_parity_pass(
            [shifted_row]
            + [
                {"legal_mask_exact": True, "max_abs_difference": 0.0}
                for _ in range(511)
            ]
        )
    changed = archived.copy()
    changed[1] += 2e-6
    with pytest.raises(VerificationError, match="P_archived_prior_parity_failed"):
        runner.require_parity_pass(
            [
                {
                    "legal_mask_exact": True,
                    "max_abs_difference": runner.compare_parity(
                        changed, archived, [1, 2]
                    ),
                }
            ]
            + [
                {"legal_mask_exact": True, "max_abs_difference": 0.0}
                for _ in range(511)
            ]
        )


@pytest.mark.parametrize("parity_fails", [False, True])
def test_executor_runs_all_p_parity_before_any_t512(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch, parity_fails: bool
) -> None:
    roots = []
    archived = {}
    oracles = {}
    for index in range(512):
        state_hash = f"root-{index}"
        roots.append(
            {
                "state_hash": state_hash,
                "opening_index": index,
                "state": {
                    "player_pits": [1, 1, 0, 0, 0, 0],
                    "opponent_pits": [1, 0, 0, 0, 0, 0],
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                },
            }
        )
        archived[state_hash] = {"legal_moves": [0, 1], "priors": [0.5, 0.5, 0, 0, 0, 0]}
        oracles[state_hash] = {"action_margins": {"0": 2, "1": 1}}
    monkeypatch.setattr(
        runner, "_read_inputs", lambda _root: (roots, archived, oracles)
    )
    freeze = json.loads(runner.FREEZE_PATH.read_bytes())
    monkeypatch.setattr(
        runner,
        "verify_authorization",
        lambda **_kwargs: {
            "source_and_implementation_sha256": runner.frozen_dependencies(),
            "runtime": freeze["runtime"],
        },
    )
    monkeypatch.setattr(runner, "analyze", lambda _rows: {"synthetic": True})
    events: list[str] = []

    class StubEvaluator:
        def __init__(self, artifact_dir: Path) -> None:
            self.model = artifact_dir.name
            self.calls = 0

        def evaluate(self, _game):
            self.calls += 1
            events.append(self.model)
            if self.model == "P-E4":
                prior = (
                    [0.6, 0.4, 0, 0, 0, 0]
                    if parity_fails and self.calls == 1
                    else [0.5, 0.5, 0, 0, 0, 0]
                )
                return prior, 0.0
            return [0.75, 0.25, 0, 0, 0, 0], 0.0

    monkeypatch.setattr(runner.arena, "ArtifactEvaluator", StubEvaluator)
    output = scratch_dir / f"synthetic-{parity_fails}"
    if parity_fails:
        with pytest.raises(VerificationError, match="P_archived_prior_parity_failed"):
            runner.execute(
                root=runner.ROOT,
                output=output,
                authorization_path=None,
                trusted_authorization_sha256=None,
            )
        assert events == ["P-E4"] * 512
        assert not output.exists()
    else:
        result = runner.execute(
            root=runner.ROOT,
            output=output,
            authorization_path=None,
            trusted_authorization_sha256=None,
        )
        assert result["p_parity_forward_evaluations"] == 512
        assert result["t512_forward_evaluations"] == 512
        assert events == ["P-E4"] * 512 + ["T512"] * 512
        assert (output / "run-manifest.json").is_file()


@pytest.mark.parametrize(
    "records, message",
    [
        (
            [{"state_hash": f"r{i}"} for i in range(512)] + [{"state_hash": "r0"}],
            "count",
        ),
        ([{"state_hash": f"r{i}"} for i in range(511)], "count"),
        (
            [{"state_hash": f"r{i}"} for i in range(511)] + [{"state_hash": "other"}],
            "membership",
        ),
    ],
)
def test_membership_rejects_duplicate_missing_and_wrong_root(
    records: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(VerificationError, match=message):
        validate_membership(records, {f"r{i}" for i in range(512)}, "synthetic")


def test_independent_result_verifier_detects_prediction_tampering(
    scratch_dir: Path,
) -> None:
    package = scratch_dir / "run"
    package.mkdir()
    authorization_path, authorization_digest, authorization = _authorization(
        scratch_dir
    )
    filenames = (
        "p-parity.jsonl",
        "t512-predictions.jsonl",
        "root-results.jsonl",
        "analysis.json",
    )
    for name in filenames:
        (package / name).write_text("synthetic evidence\n")
    manifest = {
        "schema": "seed588-raw-prior-run-v1",
        "execution_freeze_sha256": runner.sha256_file(
            runner.ROOT / runner.FREEZE_PATH.relative_to(runner.ROOT)
        ),
        "source_evidence_sha256": runner.EXPECTED_INPUTS,
        "artifacts": runner.EXPECTED_ARTIFACTS,
        "implementation_sha256": authorization["source_and_implementation_sha256"],
        "p_parity_passed": True,
        "authorization_sha256": authorization_digest,
        "runtime": authorization["runtime"],
        "p_parity_forward_evaluations": 512,
        "t512_forward_evaluations": 512,
        "mcts": False,
        "games": False,
        "training": False,
        "result_files_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in filenames
        },
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    (package / "run-manifest.json").write_bytes(raw)
    trusted_digest = hashlib.sha256(raw).hexdigest()
    (package / "t512-predictions.jsonl").write_text("tampered\n")
    with pytest.raises(
        VerificationError, match="trusted_authorization_digest_mismatch"
    ):
        verify_result(
            root=runner.ROOT,
            package=package,
            trusted_manifest_sha256=trusted_digest,
            authorization_path=authorization_path,
            trusted_authorization_sha256="0" * 64,
        )
    with pytest.raises(
        VerificationError, match="result_file_hash:t512-predictions.jsonl"
    ):
        verify_result(
            root=runner.ROOT,
            package=package,
            trusted_manifest_sha256=trusted_digest,
            authorization_path=authorization_path,
            trusted_authorization_sha256=authorization_digest,
        )


def test_independent_result_verifier_recomputes_complete_synthetic_package(
    scratch_dir: Path,
) -> None:
    data = runner.ROOT / "docs/data/seed559-fixed-position-feasibility/run-0003"
    cohort = json.loads((data / "cohort-manifest.json").read_bytes())["states"]
    archived = {
        row["state_hash"]: row
        for row in _jsonl((data / "search-records.jsonl").read_bytes())
    }
    oracles = {
        row["state_hash"]: row
        for row in _jsonl((data / "oracle-records.jsonl").read_bytes())
    }
    package = scratch_dir / "complete-synthetic-package"
    package.mkdir()
    authorization_path, authorization_digest, authorization = _authorization(
        scratch_dir
    )
    parity, predictions, root_rows = [], [], []
    for root in cohort:
        key = root["state_hash"]
        source = archived[key]
        priors = [float(value) for value in source["priors"]]
        legal = [int(move) for move in source["legal_moves"]]
        parity.append(
            {
                "state_hash": key,
                "opening_index": root["opening_index"],
                "p_forward_priors": priors,
                "archived_p_priors": priors,
                "max_abs_difference": 0.0,
                "legal_mask_exact": True,
                "legal_relative_actions": legal,
            }
        )
        predictions.append(
            {
                "state_hash": key,
                "opening_index": root["opening_index"],
                "t512_priors": priors,
            }
        )
        margins = {
            int(move): int(value)
            for move, value in oracles[key]["action_margins"].items()
        }
        root_rows.append(
            {
                "state_hash": key,
                "opening_index": int(root["opening_index"]),
                "metrics": _root_metrics(priors, priors, legal, margins),
            }
        )
    analysis = analysis_impl.analyze(root_rows)
    for name, rows in (
        ("p-parity.jsonl", parity),
        ("t512-predictions.jsonl", predictions),
        ("root-results.jsonl", root_rows),
    ):
        (package / name).write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )
    (package / "analysis.json").write_text(json.dumps(analysis, sort_keys=True) + "\n")
    manifest = {
        "schema": "seed588-raw-prior-run-v1",
        "execution_freeze_sha256": runner.sha256_file(runner.FREEZE_PATH),
        "implementation_sha256": authorization["source_and_implementation_sha256"],
        "source_evidence_sha256": runner.EXPECTED_INPUTS,
        "artifacts": runner.EXPECTED_ARTIFACTS,
        "authorization_sha256": authorization_digest,
        "runtime": authorization["runtime"],
        "p_parity_forward_evaluations": 512,
        "t512_forward_evaluations": 512,
        "parity_max_abs_difference": 0.0,
        "p_parity_passed": True,
        "mcts": False,
        "games": False,
        "training": False,
        "result_files_sha256": {
            name: runner.sha256_file(package / name)
            for name in (
                "p-parity.jsonl",
                "t512-predictions.jsonl",
                "root-results.jsonl",
                "analysis.json",
            )
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (package / "run-manifest.json").write_bytes(manifest_bytes)
    result = verify_result(
        root=runner.ROOT,
        package=package,
        trusted_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        authorization_path=authorization_path,
        trusted_authorization_sha256=authorization_digest,
    )
    assert result["status"] == "verified"
    assert result["p_parity_roots"] == result["t512_roots"] == 512
    assert result["analysis"] == analysis
