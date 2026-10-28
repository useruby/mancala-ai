"""Tests for the immutable seed542 generation plan."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from ml.alphazero_lite import run_seed542_generation as generation


def test_shard_seed_and_source_assignment() -> None:
    assert list(range(542000, 542064)) == [542000 + i for i in range(64)]
    assert sum(seed < 542048 for seed in range(542000, 542064)) == 48
    assert sum(seed >= 542048 for seed in range(542000, 542064)) == 16
    assert (
        generation.command_for(542000)[
            generation.command_for(542000).index("--seed") + 1
        ]
        == "542000"
    )
    assert any("shard-542000.jsonl" in arg for arg in generation.command_for(542000))
    for seed in (542047, 542048, 542063):
        command = generation.command_for(seed)
        assert command[command.index("--seed") + 1] == str(seed)


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
    monkeypatch.setattr(generation, "preflight", lambda **_kwargs: {})
    with pytest.raises(RuntimeError, match="generation_refuses_non_unattempted_seed"):
        generation.run_seed(542000)
    assert target.read_bytes() == b"preserve these bytes"
    assert not generation.CONTINUATION_INVENTORY.exists()


def test_inventory_is_append_only_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "execution-inventory.jsonl"
    monkeypatch.setattr(generation, "INVENTORY", path)
    generation.record({"event": "started", "seed": 542000}, inventory=path)
    generation.record(
        {"event": "finished", "seed": 542000, "exit_status": 0}, inventory=path
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [
        {"event": "started", "seed": 542000},
        {"event": "finished", "seed": 542000, "exit_status": 0},
    ]


def test_resume_preserves_existing_output_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    out = root / "docs/data/seed542-selfplay-4096"
    out.mkdir(parents=True)
    completed = out / "shard-542000.jsonl"
    completed_bytes = b"completed evidence"
    completed.write_bytes(completed_bytes)
    interrupted_log = out / "shard-542001.log"
    interrupted_log.write_bytes(b"do not reopen with xb")
    staging = out / "worker_0.jsonl"
    staging.write_bytes(b"preserved partial evidence")
    original_inventory = out / "execution-inventory.jsonl"
    original_events = [
        {"event": "started", "seed": 542000},
        {
            "event": "finished",
            "seed": 542000,
            "exit_status": 0,
            "output_sha256": generation.digest(completed),
        },
        {"event": "started", "seed": 542001},
    ]
    original_inventory.write_text(
        "".join(json.dumps(event) + "\n" for event in original_events)
    )
    (out / "execution-interruption.json").write_text(
        json.dumps({"seed": 542001, "staging_path": str(staging)})
    )
    original_inventory_bytes = original_inventory.read_bytes()
    monkeypatch.setattr(generation, "ROOT", root)
    monkeypatch.setattr(generation, "OUT", out)
    monkeypatch.setattr(generation, "INVENTORY", original_inventory)
    monkeypatch.setattr(
        generation, "CONTINUATION_INVENTORY", out / "continuation-inventory.jsonl"
    )
    monkeypatch.setattr(
        generation,
        "preflight",
        lambda **_kwargs: {
            "resume_policy": {"interrupted_or_ambiguous_attempts": "preserve and skip"}
        },
    )
    attempted: list[int] = []
    monkeypatch.setattr(
        generation, "run_seed", lambda seed, **_kwargs: attempted.append(seed)
    )
    plan = generation.main(execute=True)
    assert plan[0] == {"seed": 542000, "state": "completed"}
    assert plan[1] == {"seed": 542001, "state": "interrupted"}
    assert attempted == list(range(542002, 542064))
    assert completed.read_bytes() == completed_bytes
    assert interrupted_log.read_bytes() == b"do not reopen with xb"
    assert staging.read_bytes() == b"preserved partial evidence"
    assert original_inventory.read_bytes() == original_inventory_bytes
    continuation_events = [
        json.loads(line)
        for line in (out / "continuation-inventory.jsonl").read_text().splitlines()
    ]
    assert continuation_events == [
        {
            "event": "continuation_skip_preserved",
            "seed": 542000,
            "state": "completed",
            "policy": "preserve and skip",
            "timestamp": continuation_events[0]["timestamp"],
        },
        {
            "event": "continuation_skip_preserved",
            "seed": 542001,
            "state": "interrupted",
            "policy": "preserve and skip",
            "timestamp": continuation_events[1]["timestamp"],
        },
    ]


def test_preflight_rejects_simulation_drift_before_any_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = list(generation.COMMAND_TEMPLATE)
    changed[changed.index("128")] = "256"
    launched: list[list[str]] = []
    monkeypatch.setattr(generation, "COMMAND_TEMPLATE", changed)
    monkeypatch.setattr(
        generation.subprocess, "run", lambda command, **_: launched.append(command)
    )
    with pytest.raises(
        RuntimeError, match="generation_preflight_command_template_drift"
    ):
        generation.preflight()
    assert launched == []


def test_preflight_authenticates_frozen_source_and_runtime_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_python = tmp_path / "python"
    fake_python.write_bytes(b"executable fixture")
    fake_python.chmod(0o755)
    monkeypatch.setattr(generation.sys, "executable", str(fake_python))
    template = list(generation.COMMAND_TEMPLATE)
    template[0] = str(fake_python)
    monkeypatch.setattr(generation, "COMMAND_TEMPLATE", template)

    source_hashes = {}
    for relative in sorted(generation.SOURCE_PATHS):
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source:{relative}".encode())
        source_hashes[relative] = generation.digest(source)
    behavior = {}
    behavior["model_version"] = "seed48-nextgen-s455-default-value-iter1"
    for name, relative in generation.BEHAVIOR_PATHS.items():
        asset = tmp_path / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(f"asset:{name}".encode())
        behavior[name] = {"path": relative, "sha256": generation.digest(asset)}
    environment = {
        "python": "3.14.7",
        "platform": "test-platform",
        "numpy": "2.5.3",
        "torch": "2.14.0+cu130",
    }
    freeze = {
        "source_sha256": source_hashes,
        "behavior": behavior,
        "environment": environment,
        "seed_plan": {
            "first_seed": 542000,
            "last_seed_inclusive": 542063,
            "games_per_shard": 64,
            "training_seed_count": 48,
            "training_last_seed_inclusive": 542047,
            "development_seed_count": 16,
            "development_first_seed": 542048,
        },
        "command_template": template,
        "resume_policy": generation.RESUME_POLICY,
        "resource_scope": generation.RESOURCE_SCOPE,
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    generation.preflight(
        root=tmp_path,
        freeze_path=freeze_path,
        expected_freeze_sha256=freeze_hash,
        command_template=template,
        current_environment=environment,
    )

    weights_path = tmp_path / generation.BEHAVIOR_PATHS["weights"]
    weights_path.write_bytes(b"drifted weights")
    with pytest.raises(
        RuntimeError,
        match="generation_preflight_artifact_sha256_mismatch:model-artifact/current/weights.json",
    ):
        generation.preflight(
            root=tmp_path,
            freeze_path=freeze_path,
            expected_freeze_sha256=freeze_hash,
            command_template=template,
            current_environment=environment,
        )

    weights_path.write_bytes("asset:weights".encode())
    source = tmp_path / sorted(generation.SOURCE_PATHS)[0]
    source.write_bytes(b"drifted source")
    with pytest.raises(
        RuntimeError, match="generation_preflight_source_sha256_mismatch"
    ):
        generation.preflight(
            root=tmp_path,
            freeze_path=freeze_path,
            expected_freeze_sha256=freeze_hash,
            command_template=template,
            current_environment=environment,
        )


def test_run_seed_preflights_before_start_record_or_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "corpus"
    out.mkdir()
    inventory = out / "continuation-inventory.jsonl"
    launched: list[list[str]] = []

    def fail_preflight() -> None:
        raise RuntimeError("generation_preflight_command_template_drift")

    monkeypatch.setattr(generation, "OUT", out)
    monkeypatch.setattr(generation, "CONTINUATION_INVENTORY", inventory)
    monkeypatch.setattr(generation, "preflight", fail_preflight)
    monkeypatch.setattr(
        generation.subprocess, "run", lambda command, **_: launched.append(command)
    )
    with pytest.raises(
        RuntimeError, match="generation_preflight_command_template_drift"
    ):
        generation.run_seed(542063)
    assert launched == []
    assert not inventory.exists()
    assert not (out / "shard-542063.log").exists()
