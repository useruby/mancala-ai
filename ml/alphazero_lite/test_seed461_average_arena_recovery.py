"""Recovery-path tests exercise the hash-bound averaging arena runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import run_seed461_e2_e4_average_arena as runner


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    suite = data / "suite.jsonl"
    openings = [{"prefix_moves": [i]} for i in range(256)]
    suite.write_text("".join(json.dumps(row) + "\n" for row in openings))
    reg_path, candidates_path, binding_path = (
        data / "registration.json",
        data / "candidates.json",
        data / "binding.json",
    )
    opponent = {"artifact": str(tmp_path / "opponent")}
    contract = {"runtime": "test-runtime"}
    evaluation = {
        "suite": {"sha256": digest(suite)},
        "games_per_candidate": 512,
        "arena_seed": 389,
        "seed_contract": "azlite_eval_seed_v2",
        "simulations_per_side": 384,
        "c_puct": 1.25,
    }
    reg = {"evaluation": {**evaluation, "opponent_binding": opponent}}
    reg_path.write_text(json.dumps(reg))
    candidates = {}
    for order in range(38611, 38616):
        for arm in ("A", "B"):
            name = f"order_{order}_{arm}"
            artifact = tmp_path / "artifacts" / name
            artifact.mkdir(parents=True)
            checkpoint = artifact / "checkpoint"
            checkpoint.write_text(name)
            (artifact / "model.npz").write_text(name)
            candidates[name] = {
                "artifact": str(artifact),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest(checkpoint),
                "artifact_sha256": {"model.npz": digest(artifact / "model.npz")},
                "runtime_contract": contract,
            }
    candidate_binding = {
        "schema": "seed461-e2-e4-average-candidate-binding-v1",
        "registration_sha256": digest(reg_path),
        "source_training_sha256": "source-training",
        "suite_sha256": digest(suite),
        "opponent": opponent,
        "candidates": candidates,
    }
    candidates_path.write_text(json.dumps(candidate_binding))
    monkeypatch.setattr(runner, "REG", reg_path)
    monkeypatch.setattr(runner, "SUITE", suite)
    monkeypatch.setattr(runner, "CANDIDATES", candidates_path)
    monkeypatch.setattr(runner, "BINDING", binding_path)
    monkeypatch.setattr(runner, "WORK", work)
    monkeypatch.setattr(runner, "OPPONENT", Path(opponent["artifact"]))
    monkeypatch.setattr(runner, "validate_inputs", lambda *_: None)
    return reg, candidate_binding, openings, binding_path, work


def write_evidence(
    report: Path, games: Path, candidate, opponent, evaluation, openings
):
    games.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, opening in enumerate(openings):
        for seat in (0, 1):
            rows.append(
                {
                    "opening_index": index,
                    "opening_prefix_moves": opening["prefix_moves"],
                    "challenger_player": seat,
                    "winner": "challenger" if seat == 0 else "current",
                }
            )
    games.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runtime = candidate["runtime_contract"]
    profile = {
        "c_puct": evaluation["c_puct"],
        "simulations": evaluation["simulations_per_side"],
        **runtime,
    }
    notes = {
        "challenger_path": candidate["artifact"],
        "current_path": opponent["artifact"],
        "suite_sha256": evaluation["suite"]["sha256"],
        "seed": evaluation["arena_seed"],
        "base_seed": evaluation["arena_seed"],
        "seed_contract": evaluation["seed_contract"],
        "challenger_simulations": evaluation["simulations_per_side"],
        "current_simulations": evaluation["simulations_per_side"],
        "search_profile": profile,
        **runtime,
    }
    report.write_text(
        json.dumps(
            {
                "schema": "arena_v1",
                "games": 512,
                "games_played": 512,
                "wins": 256,
                "losses": 256,
                "draws": 0,
                "score": 0.5,
                "notes": notes,
            }
        )
    )


def prepare_binding(reg, candidates, suite, path, reports=None):
    binding_data = {
        "schema": "seed461-e2-e4-average-evaluation-binding-v1",
        "registration_sha256": digest(runner.REG),
        "candidate_binding_sha256": digest(runner.CANDIDATES),
        "suite_sha256": digest(runner.SUITE),
        "opponent": reg["evaluation"]["opponent_binding"],
        "candidates": candidates["candidates"],
        "reports": reports or {},
    }
    if reports and len(reports) == 10:
        binding_data["status"] = "completed_fixed_5120_games"
    path.write_text(json.dumps(binding_data))


def stub_arena(monkeypatch, candidates, opponent, evaluation, openings, work):
    calls = []
    by_artifact = {row["artifact"]: row for row in candidates["candidates"].values()}

    def launch(command, **_kwargs):
        calls.append(command)
        candidate = by_artifact[command[command.index("--challenger") + 1]]
        report = Path(command[command.index("--out") + 1])
        games = Path(command[command.index("--game-jsonl") + 1])
        write_evidence(report, games, candidate, opponent, evaluation, openings)

    monkeypatch.setattr(runner.subprocess, "run", launch)
    return calls


def test_valid_interrupted_run_recovers_without_launching(tmp_path, monkeypatch):
    reg, candidates, openings, binding, work = fixture(tmp_path, monkeypatch)
    name = "order_38611_A"
    report, games = (
        work / "arena" / f"{name}.json",
        work / "arena" / f"{name}-games.jsonl",
    )
    write_evidence(
        report,
        games,
        candidates["candidates"][name],
        reg["evaluation"]["opponent_binding"],
        reg["evaluation"],
        openings,
    )
    prepare_binding(
        reg,
        candidates,
        openings,
        binding,
        {name: {"state": "running", "report": str(report), "games": str(games)}},
    )
    calls = stub_arena(
        monkeypatch,
        candidates,
        reg["evaluation"]["opponent_binding"],
        reg["evaluation"],
        openings,
        work,
    )
    runner.main()
    assert len(calls) == 9
    stored = json.loads(binding.read_text())["reports"][name]
    assert stored["report_sha256"] == digest(report)
    assert stored["games_sha256"] == digest(games)


def test_completed_resume_launches_nothing_and_preserves_evidence(
    tmp_path, monkeypatch
):
    reg, candidates, openings, binding, work = fixture(tmp_path, monkeypatch)
    reports = {}
    evidence_files = []
    for name, candidate in candidates["candidates"].items():
        report, games = (
            work / "arena" / f"{name}.json",
            work / "arena" / f"{name}-games.jsonl",
        )
        write_evidence(
            report,
            games,
            candidate,
            reg["evaluation"]["opponent_binding"],
            reg["evaluation"],
            openings,
        )
        reports[name] = {
            "report": str(report),
            "report_sha256": digest(report),
            "games": str(games),
            "games_sha256": digest(games),
        }
        evidence_files.extend((report, games))
    prepare_binding(reg, candidates, openings, binding, reports)
    before = {path: path.read_bytes() for path in [binding, *evidence_files]}
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("must resume")
    )
    runner.main()
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_partial_resume_runs_only_missing_and_preserves_completed(
    tmp_path, monkeypatch
):
    reg, candidates, openings, binding, work = fixture(tmp_path, monkeypatch)
    name, candidate = next(iter(candidates["candidates"].items()))
    report, games = (
        work / "arena" / f"{name}.json",
        work / "arena" / f"{name}-games.jsonl",
    )
    write_evidence(
        report,
        games,
        candidate,
        reg["evaluation"]["opponent_binding"],
        reg["evaluation"],
        openings,
    )
    completed = {
        "report": str(report),
        "report_sha256": digest(report),
        "games": str(games),
        "games_sha256": digest(games),
    }
    prepare_binding(reg, candidates, openings, binding, {name: completed})
    before = report.read_bytes(), games.read_bytes()
    calls = stub_arena(
        monkeypatch,
        candidates,
        reg["evaluation"]["opponent_binding"],
        reg["evaluation"],
        openings,
        work,
    )
    runner.main()
    assert len(calls) == 9
    assert (report.read_bytes(), games.read_bytes()) == before


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("winner", "unknown_winner"),
        ("identity", "opening_identity_mismatch"),
        ("coverage", "game_count_mismatch"),
        ("seat", "seat_pairing_mismatch"),
        ("report_identity", "report_identity_mismatch"),
    ],
)
def test_invalid_interrupted_evidence_is_not_bound(
    tmp_path, monkeypatch, corruption, message
):
    reg, candidates, openings, binding, work = fixture(tmp_path, monkeypatch)
    name = "order_38611_A"
    report, games = (
        work / "arena" / f"{name}.json",
        work / "arena" / f"{name}-games.jsonl",
    )
    write_evidence(
        report,
        games,
        candidates["candidates"][name],
        reg["evaluation"]["opponent_binding"],
        reg["evaluation"],
        openings,
    )
    rows = [json.loads(row) for row in games.read_text().splitlines()]
    if corruption == "winner":
        rows[0]["winner"] = "unknown"
    elif corruption == "identity":
        rows[0]["opening_prefix_moves"] = [-1]
    elif corruption == "coverage":
        rows.pop()
    elif corruption == "seat":
        rows[1]["challenger_player"] = 0
    else:
        report_data = json.loads(report.read_text())
        report_data["notes"]["current_path"] = "wrong-opponent"
        report.write_text(json.dumps(report_data))
    games.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prepare_binding(
        reg,
        candidates,
        openings,
        binding,
        {name: {"state": "running", "report": str(report), "games": str(games)}},
    )
    before = json.loads(binding.read_text())
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("must reject")
    )
    with pytest.raises(ValueError, match=message):
        runner.main()
    assert json.loads(binding.read_text()) == before
