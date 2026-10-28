"""Tests for the immutable seed542 generation plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import run_seed542_generation as generation


def test_shard_seed_and_source_assignment() -> None:
    assert list(range(542000, 542064)) == [542000 + i for i in range(64)]
    assert sum(seed < 542048 for seed in range(542000, 542064)) == 48
    assert sum(seed >= 542048 for seed in range(542000, 542064)) == 16
    assert generation.command_for(542000)[-1] == "542000"
    assert any("shard-542000.jsonl" in arg for arg in generation.command_for(542000))
    assert generation.command_for(542047)[-1] == "542047"
    assert generation.command_for(542048)[-1] == "542048"
    assert generation.command_for(542063)[-1] == "542063"


def test_command_freezes_seed500_settings() -> None:
    command = generation.command_for(542000)
    assert command[:4] == [
        generation.PYTHON,
        "-m",
        "ml.alphazero_lite.self_play",
        "--out",
    ]
    expected = {
        "--games": "64",
        "--seed": "542000",
        "--input-encoding": "kalah_v3",
        "--player-mode": "puct",
        "--simulations": "128",
        "--c-puct": "1.25",
        "--temperature-threshold": "10",
        "--temperature": "1.0",
        "--temperature-late": "0.1",
        "--dirichlet-alpha": "0.3",
        "--dirichlet-epsilon": "0.25",
        "--schedule-progress-mode": "none",
        "--max-moves": "200",
        "--workers": "1",
        "--policy-target-mode": "default",
        "--exact-root-policy-target-mode": "selected_one_hot",
        "--policy-target-noise-mode": "denoised",
        "--value-target-mode": "default",
        "--start-state-mode": "standard_4x6",
    }
    for option, value in expected.items():
        assert command[command.index(option) + 1] == value
    assert "--write-game-metadata" in command
    assert "--runtime-search-policy-artifact" in command
    assert "model-artifact/current" in command


def test_resume_refuses_to_overwrite_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus"
    out.mkdir()
    target = out / "shard-542000.jsonl"
    target.write_bytes(b"preserve these bytes")
    monkeypatch.setattr(generation, "OUT", out)
    monkeypatch.setattr(generation, "INVENTORY", out / "inventory.jsonl")
    monkeypatch.setattr(generation, "CHECKPOINT", tmp_path / "initializer.npz")
    generation.CHECKPOINT.write_bytes(b"initializer")
    with pytest.raises(FileExistsError):
        generation.run_seed(542000)
    assert target.read_bytes() == b"preserve these bytes"
    assert not (out / "inventory.jsonl").exists()


def test_inventory_is_append_only_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "execution-inventory.jsonl"
    monkeypatch.setattr(generation, "INVENTORY", path)
    generation.record({"event": "started", "seed": 542000})
    generation.record({"event": "finished", "seed": 542000, "exit_status": 0})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"event": "started", "seed": 542000},
        {"event": "finished", "seed": 542000, "exit_status": 0},
    ]


def test_resume_preserves_existing_output_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus"
    out.mkdir()
    existing = out / "shard-542000.jsonl"
    existing.write_bytes(b"partial evidence")
    monkeypatch.setattr(generation, "OUT", out)
    monkeypatch.setattr(generation, "INVENTORY", out / "inventory.jsonl")
    attempted: list[int] = []
    monkeypatch.setattr(generation, "run_seed", attempted.append)
    generation.main()
    assert attempted == list(range(542001, 542064))
    assert existing.read_bytes() == b"partial evidence"
    first_record = json.loads((out / "inventory.jsonl").read_text().splitlines()[0])
    assert first_record["event"] == "preserved_existing_output_skipped"
    assert first_record["output_sha256"] == generation.digest(existing)
