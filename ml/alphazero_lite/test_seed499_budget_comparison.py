"""Synthetic execution, recovery, and identity tests for seed499."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import seed499_budget_comparison as protocol
from ml.alphazero_lite.seed499_budget_analysis import load_rows


@pytest.fixture
def frozen_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(protocol, "OUT", Path(".tmp") / f"seed499-test-{tmp_path.name}")
    out = protocol.ROOT / protocol.OUT
    if out.exists():
        shutil.rmtree(out)
    protocol.prepare(protocol.ROOT)
    return out


def _historical_batch(index: int) -> tuple[dict, list[dict], dict]:
    data = protocol.ROOT / protocol.HISTORICAL
    opening = load_rows(data / "openings.jsonl")[index]
    games = load_rows(data / "resume-corrected" / f"games-{index:04d}.jsonl")
    report = json.loads(
        (data / "resume-corrected" / f"report-{index:04d}.json").read_text()
    )
    report["notes"]["challenger_simulations"] = 128
    report["notes"]["current_simulations"] = 128
    return opening, games, report


def _make_batch(root: Path, index: int) -> None:
    out = root / protocol.OUT
    batch = out / "batches"
    batch.mkdir(exist_ok=True)
    opening, games, report = _historical_batch(index)
    opener = batch / f"opening-{index:04d}.jsonl"
    opener.write_bytes(protocol.base.canonical_json(opening) + b"\n")
    (batch / f"games-{index:04d}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in games)
    )
    (batch / f"report-{index:04d}.json").write_text(json.dumps(report))


def _stub_arena(
    monkeypatch: pytest.MonkeyPatch, root: Path, calls: list[int], stop: int
):
    def command(_root: Path, index: int, _opening: dict) -> list[str]:
        return [str(index)]

    def run(args: list[str], *, cwd: Path, check: bool) -> None:
        index = int(args[0])
        calls.append(index)
        if index == stop:
            raise RuntimeError("synthetic_interruption")
        _make_batch(root, index)

    monkeypatch.setattr(protocol, "_arena_command", command)
    monkeypatch.setattr(protocol.subprocess, "run", run)


def test_runner_resume_skips_committed_pair_and_preserves_interruption(
    frozen_fixture: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[int] = []
    root = protocol.ROOT
    _stub_arena(monkeypatch, root, calls, stop=1)
    with pytest.raises(RuntimeError, match="synthetic_interruption"):
        protocol.run(root, execute=True)
    assert calls == [0, 1]
    ledger_rows = load_rows(frozen_fixture / "outcomes.jsonl")
    assert len(ledger_rows) == 2
    preserved_games = frozen_fixture / "batches/games-0000.jsonl"
    before = preserved_games.read_bytes()

    calls.clear()
    _stub_arena(monkeypatch, root, calls, stop=2)
    with pytest.raises(RuntimeError, match="synthetic_interruption"):
        protocol.run(root, execute=True)
    assert calls == [1, 2]
    assert preserved_games.read_bytes() == before
    assert len(load_rows(frozen_fixture / "outcomes.jsonl")) == 4


def test_runner_rejects_partial_and_duplicate_seat_pairs(
    frozen_fixture: Path, monkeypatch: pytest.MonkeyPatch
):
    out = frozen_fixture
    opening, games, _ = _historical_batch(0)
    out.joinpath("outcomes.jsonl").write_text(json.dumps({"opening_index": 0}) + "\n")
    with pytest.raises(ValueError, match="partial_committed_pair"):
        protocol.run(protocol.ROOT, execute=True)

    rows = [
        {
            **games[0],
            "opening_index": 0,
            "opening_state_hash": opening["state_hash"],
            "F_player": 0,
            "batch_game_position": 0,
            "batch_game_sha256": "x",
            "batch_report_sha256": "x",
            "batch_opening_sha256": "x",
        },
        {
            **games[1],
            "opening_index": 0,
            "opening_state_hash": opening["state_hash"],
            "F_player": 0,
            "batch_game_position": 1,
            "batch_game_sha256": "x",
            "batch_report_sha256": "x",
            "batch_opening_sha256": "x",
        },
    ]
    out.joinpath("outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="committed_pair_coverage:0"):
        protocol.run(protocol.ROOT, execute=True)


def test_runner_rejects_runtime_artifact_mutation(
    frozen_fixture: Path,
):
    artifact = frozen_fixture / "runtime-artifacts/initializer/weights.json"
    original = artifact.read_bytes()
    artifact.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="seed499_runtime_artifact_hash"):
        protocol.run(protocol.ROOT, execute=True)
    artifact.write_bytes(original)


def test_uncommitted_outcome_batch_is_preserved_and_never_rerun(
    frozen_fixture: Path, monkeypatch: pytest.MonkeyPatch
):
    root = protocol.ROOT
    out = frozen_fixture
    calls: list[int] = []

    def command(_root: Path, index: int, opening: dict) -> list[str]:
        batch = out / "batches"
        batch.mkdir(exist_ok=True)
        (batch / f"opening-{index:04d}.jsonl").write_bytes(
            protocol.base.canonical_json(opening) + b"\n"
        )
        return [str(index)]

    def interrupted(args: list[str], *, cwd: Path, check: bool) -> None:
        index = int(args[0])
        calls.append(index)
        if index == 0:
            _, games, _ = _historical_batch(index)
            path = out / "batches/games-0000.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in games))
            raise RuntimeError("synthetic_crash_after_outcome_log")

    monkeypatch.setattr(protocol, "_arena_command", command)
    monkeypatch.setattr(protocol.subprocess, "run", interrupted)
    with pytest.raises(RuntimeError, match="synthetic_crash_after_outcome_log"):
        protocol.run(root, execute=True)
    preserved = out / "batches/games-0000.jsonl"
    before = preserved.read_bytes()
    with pytest.raises(ValueError, match="interrupted_outcome_batch_preserved:0"):
        protocol.run(root, execute=True)
    assert calls == [0]
    assert preserved.read_bytes() == before


def test_replay_and_registered_runtime_source_mutations_are_rejected(
    frozen_fixture: Path,
):
    opening, games, _ = _historical_batch(0)
    altered = dict(games[0], trajectory="99")
    with pytest.raises(ValueError):
        protocol._replay(opening, altered, int(games[0]["challenger_player"]))

    runtime_source = protocol.ROOT / "model-artifact/current/search_policy.json"
    original = runtime_source.read_bytes()
    runtime_source.write_bytes(original + b" ")
    try:
        with pytest.raises(
            ValueError,
            match="runtime_identity:model-artifact/current/search_policy.json",
        ):
            protocol.run(protocol.ROOT, execute=True)
    finally:
        runtime_source.write_bytes(original)
