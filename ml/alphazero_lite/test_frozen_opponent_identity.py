"""Tests for preflight binding of frozen opponent files and native runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite.frozen_opponent_identity import (
    validate_frozen_opponent_identity,
)
from ml.alphazero_lite import run_seed461_cross_order_e4_average_arena as runner


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registered_opponent_identity_mutations_are_rejected(tmp_path):
    artifact = tmp_path / "opponent"
    artifact.mkdir()
    files = {}
    for name in ("weights.json", "metadata.json", "search_policy.json"):
        files[name] = artifact / name
        files[name].write_text(f"frozen {name}")
    probe, tablebase = tmp_path / "probe", tmp_path / "tablebase"
    probe.write_bytes(b"native probe")
    tablebase.write_bytes(b"native tablebase")
    opponent = {
        "artifact": str(artifact),
        "weights_sha256": digest(files["weights.json"]),
        "metadata_sha256": digest(files["metadata.json"]),
        "sidecar_sha256": digest(files["search_policy.json"]),
        "native_probe_sha256": digest(probe),
        "tablebase_sha256": digest(tablebase),
    }
    contract = {
        "exact_root_native_probe": str(probe),
        "exact_root_native_probe_sha256": digest(probe),
        "exact_root_tablebase": str(tablebase),
        "exact_root_tablebase_sha256": digest(tablebase),
    }
    validate_frozen_opponent_identity(artifact, opponent, contract)

    for name, message in (
        ("weights.json", "opponent_hash_mismatch"),
        ("metadata.json", "opponent_hash_mismatch"),
        ("search_policy.json", "opponent_hash_mismatch"),
    ):
        original = files[name].read_bytes()
        files[name].write_bytes(original + b" changed")
        with pytest.raises(ValueError, match=message):
            validate_frozen_opponent_identity(artifact, opponent, contract)
        files[name].write_bytes(original)

    probe.write_bytes(b"changed probe")
    with pytest.raises(ValueError, match="runtime_identity_mismatch"):
        validate_frozen_opponent_identity(artifact, opponent, contract)
    probe.write_bytes(b"native probe")
    tablebase.write_bytes(b"changed tablebase")
    with pytest.raises(ValueError, match="runtime_identity_mismatch"):
        validate_frozen_opponent_identity(artifact, opponent, contract)

    with pytest.raises(ValueError, match="opponent_path_mismatch"):
        validate_frozen_opponent_identity(tmp_path / "different", opponent, contract)


def _runner_fixture(tmp_path, monkeypatch, completed):
    data, work, artifact = tmp_path / "data", tmp_path / "work", tmp_path / "opponent"
    data.mkdir()
    artifact.mkdir()
    suite = data / "suite.jsonl"
    suite.write_text(
        "".join(json.dumps({"prefix_moves": [i]}) + "\n" for i in range(256))
    )
    opponent_files = {}
    for name in ("weights.json", "metadata.json", "search_policy.json"):
        opponent_files[name] = artifact / name
        opponent_files[name].write_text(name)
    probe, tablebase = tmp_path / "probe", tmp_path / "tablebase"
    probe.write_text("probe")
    tablebase.write_text("tablebase")
    contract = {
        "exact_root_native_probe": str(probe),
        "exact_root_native_probe_sha256": digest(probe),
        "exact_root_tablebase": str(tablebase),
        "exact_root_tablebase_sha256": digest(tablebase),
        "runtime_search_policy_sha256": digest(opponent_files["search_policy.json"]),
    }
    opponent = {
        "artifact": str(artifact),
        "weights_sha256": digest(opponent_files["weights.json"]),
        "metadata_sha256": digest(opponent_files["metadata.json"]),
        "sidecar_sha256": digest(opponent_files["search_policy.json"]),
        "native_probe_sha256": digest(probe),
        "tablebase_sha256": digest(tablebase),
    }
    registration = data / "registration.json"
    reg = {
        "training": {"training_record_sha256": "training"},
        "evaluation": {
            "suite": {"sha256": digest(suite)},
            "opponent_binding": opponent,
            "runtime_contract": contract,
            "games_per_model": 512,
            "arena_seed": 390,
            "seed_contract": "azlite_eval_seed_v2",
            "simulations_per_side": 384,
            "c_puct": 1.25,
        },
    }
    registration.write_text(json.dumps(reg))
    candidates = {}
    for run in runner.RUNS:
        candidate = tmp_path / "candidates" / run
        candidate.mkdir(parents=True)
        checkpoint = candidate / "checkpoint.npz"
        checkpoint.write_text(run)
        (candidate / "model.npz").write_text(run)
        candidates[run] = {
            "artifact": str(candidate),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "artifact_sha256": {"model.npz": digest(candidate / "model.npz")},
            "runtime_contract": contract,
        }
    candidate_path = data / "candidates.json"
    candidate_binding = {
        "schema": "seed461-cross-order-e4-average-candidate-binding-v1",
        "registration_sha256": digest(registration),
        "suite_sha256": digest(suite),
        "source_training_sha256": "training",
        "opponent": opponent,
        "runtime_contract": contract,
        "candidates": candidates,
    }
    candidate_path.write_text(json.dumps(candidate_binding))
    binding_path = data / "binding.json"
    if completed:
        reports = {}
        for run in runner.RUNS:
            report, games = work / f"{run}.json", work / f"{run}.jsonl"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text("bound report")
            games.write_text("bound games")
            reports[run] = {
                "report": str(report),
                "report_sha256": digest(report),
                "games": str(games),
                "games_sha256": digest(games),
            }
        binding_path.write_text(
            json.dumps(
                {
                    "schema": "seed461-cross-order-e4-average-evaluation-binding-v1",
                    "registration_sha256": digest(registration),
                    "candidate_binding_sha256": digest(candidate_path),
                    "suite_sha256": digest(suite),
                    "opponent": opponent,
                    "candidates": candidates,
                    "reports": reports,
                }
            )
        )
    monkeypatch.setattr(runner, "REG", registration)
    monkeypatch.setattr(runner, "SUITE", suite)
    monkeypatch.setattr(runner, "CANDIDATES", candidate_path)
    monkeypatch.setattr(runner, "BINDING", binding_path)
    monkeypatch.setattr(runner, "WORK", work)
    monkeypatch.setattr(runner, "OPPONENT", artifact)
    monkeypatch.setattr(
        runner, "resolve_strength_comparison_runtime_contract", lambda **_: contract
    )
    return opponent_files, probe, tablebase, artifact


@pytest.mark.parametrize(
    "identity",
    (
        "weights.json",
        "metadata.json",
        "search_policy.json",
        "native_probe",
        "tablebase",
        "artifact_path",
    ),
)
@pytest.mark.parametrize("completed_resume", (False, True))
def test_identity_mutations_reject_before_any_game_launch(
    tmp_path, monkeypatch, identity, completed_resume
):
    files, probe, tablebase, artifact = _runner_fixture(
        tmp_path, monkeypatch, completed_resume
    )
    if identity in files:
        files[identity].write_text("mutated")
    elif identity == "native_probe":
        probe.write_text("mutated")
    elif identity == "tablebase":
        tablebase.write_text("mutated")
    else:
        monkeypatch.setattr(runner, "OPPONENT", tmp_path / "different-opponent")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("identity mutation launched arena"),
    )
    with pytest.raises(ValueError):
        runner.main()
