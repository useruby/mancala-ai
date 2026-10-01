"""Recovery and preflight coverage for the two-suite confirmation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import run_order38615_a5_confirmation as runner
from ml.alphazero_lite.order38615_confirmation_validation import (
    validate_confirmation_inputs,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    artifact = tmp_path / "candidate"
    artifact.mkdir()
    files = {}
    for name in ("model.npz", "weights.json", "metadata.json", "search_policy.json"):
        (artifact / name).write_text(name)
        files[name] = digest(artifact / name)
    opponent_path = tmp_path / "opponent"
    opponent_path.mkdir()
    for name in ("weights.json", "metadata.json", "search_policy.json"):
        (opponent_path / name).write_text(name)
    native, tablebase = tmp_path / "native", tmp_path / "tablebase"
    native.write_text("native")
    tablebase.write_text("tablebase")
    contract = {
        "exact_root_native_probe": str(native),
        "exact_root_native_probe_sha256": digest(native),
        "exact_root_tablebase": str(tablebase),
        "exact_root_tablebase_sha256": digest(tablebase),
    }
    opponent = {
        "artifact": str(opponent_path),
        "weights_sha256": digest(opponent_path / "weights.json"),
        "metadata_sha256": digest(opponent_path / "metadata.json"),
        "sidecar_sha256": digest(opponent_path / "search_policy.json"),
        "native_probe_sha256": digest(native),
        "tablebase_sha256": digest(tablebase),
    }
    suites = {}
    for seed in (391, 392):
        suite = data / f"{seed}.jsonl"
        suite.write_text(
            "".join(json.dumps({"prefix_moves": [i]}) + "\n" for i in range(512))
        )
        suites[str(seed)] = {"path": str(suite), "sha256": digest(suite), "seed": seed}
    checkpoint = artifact / "model.npz"
    candidate = {
        "artifact": str(artifact),
        "artifact_sha256": files,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "runtime_contract": contract,
    }
    evaluation = {
        "candidate": "fixture",
        "artifact": candidate,
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "opponent_binding": opponent,
        "runtime_contract": contract,
        "suites": suites,
        "games_per_suite": 1024,
        "games_per_opening": 2,
        "simulations_per_side": 384,
        "c_puct": 1.25,
        "seed_contract": "azlite_eval_seed_v2",
    }
    reg_path, candidate_path, binding_path = (
        data / "reg.json",
        data / "candidate.json",
        data / "binding.json",
    )
    reg_path.write_text(json.dumps({"evaluation": evaluation}))
    candidate_path.write_text(
        json.dumps(
            {
                "registration_sha256": digest(reg_path),
                "opponent": opponent,
                "runtime_contract": contract,
                "candidate": candidate,
            }
        )
    )
    binding_path.write_text(
        json.dumps(
            {
                "registration_sha256": digest(reg_path),
                "candidate_binding_sha256": digest(candidate_path),
                "opponent": opponent,
                "reports": {},
            }
        )
    )
    monkeypatch.setattr(runner, "REG", reg_path)
    monkeypatch.setattr(runner, "CANDIDATE", candidate_path)
    monkeypatch.setattr(runner, "BINDING", binding_path)
    monkeypatch.setattr(runner, "WORK", work)
    return (
        json.loads(reg_path.read_text()),
        json.loads(candidate_path.read_text()),
        binding_path,
        work,
    )


def fake_evidence(report: Path, games: Path) -> None:
    games.write_text("{}\n" * 1024)
    report.write_text(json.dumps({"games_played": 1024}))


def test_opponent_identity_rejected_before_launch_or_resume(tmp_path, monkeypatch):
    _, candidate, binding_path, work = setup(tmp_path, monkeypatch)
    binding = json.loads(binding_path.read_text())
    for seed in (391, 392):
        report, games = work / f"seed{seed}.json", work / f"seed{seed}-games.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        fake_evidence(report, games)
        binding["reports"][str(seed)] = {
            "report": str(report),
            "report_sha256": digest(report),
            "games": str(games),
            "games_sha256": digest(games),
        }
    binding["status"] = "completed_2048_games"
    binding_path.write_text(json.dumps(binding))
    opponent_artifact = Path(candidate["opponent"]["artifact"])
    (opponent_artifact / "weights.json").write_text("changed")
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(ValueError, match="opponent_hash_mismatch:weights.json"):
        runner.run()
    assert json.loads(binding_path.read_text()) == binding


def test_partial_recovery_runs_only_missing_suite_and_preserves_prior(
    tmp_path, monkeypatch
):
    reg, candidate, binding_path, work = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "inputs", lambda *_: None)
    monkeypatch.setattr(runner.validation, "validate_arena_evidence", lambda *a: None)
    seed = "391"
    report, games = work / f"seed{seed}.json", work / f"seed{seed}-games.jsonl"
    report.parent.mkdir(parents=True)
    fake_evidence(report, games)
    binding = json.loads(binding_path.read_text())
    binding["reports"][seed] = {
        "state": "running",
        "report": str(report),
        "games": str(games),
    }
    binding_path.write_text(json.dumps(binding))
    before = report.read_bytes(), games.read_bytes()
    calls = []

    def launch(command, **_):
        calls.append(command)
        fake_evidence(
            Path(command[command.index("--out") + 1]),
            Path(command[command.index("--game-jsonl") + 1]),
        )

    monkeypatch.setattr(runner.subprocess, "run", launch)
    runner.run()
    assert len(calls) == 1
    assert (report.read_bytes(), games.read_bytes()) == before
    assert set(json.loads(binding_path.read_text())["reports"]) == {"391", "392"}


def test_completed_resume_preserves_evidence_without_launch(tmp_path, monkeypatch):
    _, _, binding_path, work = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "inputs", lambda *_: None)
    monkeypatch.setattr(runner.validation, "validate_arena_evidence", lambda *a: None)
    binding = json.loads(binding_path.read_text())
    for seed in (391, 392):
        report, games = work / f"seed{seed}.json", work / f"seed{seed}-games.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        fake_evidence(report, games)
        binding["reports"][str(seed)] = {
            "report": str(report),
            "report_sha256": digest(report),
            "games": str(games),
            "games_sha256": digest(games),
        }
    binding["status"] = "completed_2048_games"
    binding_path.write_text(json.dumps(binding))
    paths = [
        binding_path,
        *(Path(v[k]) for v in binding["reports"].values() for k in ("report", "games")),
    ]
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("must not launch")
    )
    runner.run()
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_stale_registration_header_rejected_before_launch(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    binding = json.loads(runner.BINDING.read_text())
    binding["registration_sha256"] = "stale"
    runner.BINDING.write_text(json.dumps(binding))
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(
        ValueError, match="evaluation_binding_registration_hash_mismatch"
    ):
        runner.run()
    assert not runner.WORK.exists()


def test_stale_candidate_registration_header_rejected_before_launch(
    tmp_path, monkeypatch
):
    setup(tmp_path, monkeypatch)
    candidate = json.loads(runner.CANDIDATE.read_text())
    candidate["registration_sha256"] = "stale"
    runner.CANDIDATE.write_text(json.dumps(candidate))
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(
        ValueError, match="candidate_binding_registration_hash_mismatch"
    ):
        runner.run()
    assert not runner.WORK.exists()


def test_changed_suite_rejected_even_when_opening_prefix_is_unchanged(
    tmp_path, monkeypatch
):
    setup(tmp_path, monkeypatch)
    suite = Path(
        json.loads(runner.REG.read_text())["evaluation"]["suites"]["391"]["path"]
    )
    original = suite.read_text()
    suite.write_text(
        original.replace('"prefix_moves": [0]', '"prefix_moves": [0], "extra": true', 1)
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(ValueError, match="registered_suite_hash_mismatch"):
        runner.run()
    assert not runner.WORK.exists()


def test_candidate_runtime_identity_mismatch_rejected(tmp_path, monkeypatch):
    reg, _, binding_path, _ = setup(tmp_path, monkeypatch)
    candidate_path = runner.CANDIDATE
    candidate = json.loads(candidate_path.read_text())
    candidate["candidate"]["runtime_contract"]["exact_root_solve_threshold"] = 8
    candidate_path.write_text(json.dumps(candidate))
    binding = json.loads(binding_path.read_text())
    binding["candidate_binding_sha256"] = digest(candidate_path)
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="candidate_binding_identity_mismatch"):
        validate_confirmation_inputs(
            runner.REG,
            candidate_path,
            binding_path,
            reg,
            candidate,
            binding,
        )
