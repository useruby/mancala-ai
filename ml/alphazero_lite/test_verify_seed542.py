"""Tamper tests for the read-only seed542 verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from ml.alphazero_lite import verify_seed542 as verifier


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    monkeypatch.setattr(verifier, "CORPUS", tmp_path)
    inventory = []
    continuation_inventory = []
    for seed in [542000, *range(542002, 542065)]:
        path = tmp_path / f"shard-{seed}.jsonl"
        data = f"shard {seed}\n".encode()
        path.write_bytes(data)
        log_path = tmp_path / f"shard-{seed}.log"
        log_bytes = f"log {seed}\n".encode()
        log_path.write_bytes(log_bytes)
        games = 13 if seed == 542064 else 64
        command = (
            ["python", "--games", str(games)]
            if seed == 542000
            else verifier.amended_runner.command_for(seed, games)
        )
        event = {
            "event": "finished",
            "seed": seed,
            "source": "development" if 542048 <= seed <= 542063 else "training",
            "command": command,
            "command_text": " ".join(command),
            "requested_games": games,
            "completed_games": games,
            "capped_games": 0,
            "incomplete_or_unstarted_games": 0,
            "exit_status": 0,
            "output_sha256": hashlib.sha256(data).hexdigest(),
            "output_bytes": len(data),
            "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
            "elapsed_seconds": 1.0,
        }
        (inventory if seed == 542000 else continuation_inventory).append(event)
    (tmp_path / "execution-inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inventory)
    )
    (tmp_path / "continuation-inventory.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in continuation_inventory)
    )
    artifacts = {
        "accounting.json": {"status": "complete", "rows": 12},
        "development-membership.json": {"included": ["d:1"]},
        "training-protocol.json": {"schedule_sha256": "abc"},
    }
    source_manifest = {"status": "test source manifest", "sources": []}
    (tmp_path / "source-manifest.json").write_text(json.dumps(source_manifest))
    amendment = {
        "status": "frozen continuation amendment",
        "command_template": verifier.amended_runner.command_template(),
    }
    amendment_bytes = json.dumps(amendment).encode()
    (tmp_path / "continuation-amendment.json").write_bytes(amendment_bytes)
    monkeypatch.setattr(
        verifier.amended_runner,
        "AMENDMENT_SHA256",
        hashlib.sha256(amendment_bytes).hexdigest(),
    )
    for name, payload in artifacts.items():
        (tmp_path / name).write_text(json.dumps(payload))
    monkeypatch.setattr(verifier.rebuild, "build_artifacts", lambda: artifacts)
    monkeypatch.setattr(
        verifier.rebuild.seed542_amendment,
        "build_source_manifest",
        lambda *_args: source_manifest,
    )
    return artifacts


def test_verifier_is_read_only_and_rebuilds_expected_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _fixture(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = verifier.verify()
    assert set(result) == {
        *artifacts,
        "source-manifest.json",
        "continuation-amendment.json",
    }
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


@pytest.mark.parametrize(
    ("filename", "bad_payload"),
    [
        ("accounting.json", {"rows": 11}),
        ("development-membership.json", {"included": []}),
        ("training-protocol.json", {"schedule_sha256": "corrupt"}),
    ],
)
def test_verifier_rejects_corrupted_archived_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    bad_payload: dict,
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / filename).write_text(json.dumps(bad_payload))
    with pytest.raises(ValueError, match=f"archived_evidence_mismatch:{filename}"):
        verifier.verify()


def test_verifier_rejects_corrupt_inventory_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    path = tmp_path / "shard-542000.jsonl"
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="inventory_shard_hash_mismatch:542000"):
        verifier.verify()


def test_verifier_rejects_mutated_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "source-manifest.json").write_text('{"sources": []}')
    with pytest.raises(
        ValueError, match="archived_evidence_mismatch:source-manifest.json"
    ):
        verifier.verify()


def test_verifier_rejects_mutated_inventory_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    path = tmp_path / "execution-inventory.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[0]["command"][2] = "13"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(
        ValueError, match="inventory_finished_game_count_mismatch:542000"
    ):
        verifier.verify_inventory()


def test_verifier_rejects_amended_behavior_command_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    path = tmp_path / "continuation-inventory.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    command = events[0]["command"]
    command[command.index("128")] = "256"
    events[0]["command_text"] = " ".join(command)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="amended_inventory_command_mismatch:542002"):
        verifier.verify()


def _copy_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict]:
    source = verifier.CORPUS
    shutil.copyfile(source / "accounting.json", tmp_path / "accounting.json")
    shutil.copyfile(source / "partial-status.json", tmp_path / "partial-status.json")
    shutil.copyfile(
        source / "execution-interruption.json",
        tmp_path / "execution-interruption.json",
    )
    shutil.copyfile(
        source / "execution-inventory.jsonl", tmp_path / "execution-inventory.jsonl"
    )
    shutil.copyfile(source / "shard-542000.jsonl", tmp_path / "shard-542000.jsonl")
    shutil.copyfile(source / "shard-542000.log", tmp_path / "shard-542000.log")
    staging_rel = Path("azlite-self-play-shards-etoy6j_i/worker_0.jsonl")
    (tmp_path / staging_rel).parent.mkdir(parents=True)
    shutil.copyfile(source / staging_rel, tmp_path / staging_rel)
    monkeypatch.setattr(verifier, "CORPUS", tmp_path)
    return (
        json.loads((tmp_path / "accounting.json").read_text()),
        json.loads((tmp_path / "partial-status.json").read_text()),
    )


def test_partial_verifier_rebuilds_totals_and_cohorts_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_partial_publication(tmp_path, monkeypatch)
    paths = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {path.relative_to(tmp_path): path.read_bytes() for path in paths}
    verified = verifier.verify_partial()
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert "completed-shard.sha256" in verified
    assert "partial-staging.sha256" in verified


def test_partial_verifier_rejects_mutated_totals_even_when_both_are_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accounting, status = _copy_partial_publication(tmp_path, monkeypatch)
    accounting["game_status"]["completed"] += 1
    status["completed_games"] += 1
    (tmp_path / "accounting.json").write_text(json.dumps(accounting))
    (tmp_path / "partial-status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="aggregate_game_accounting_mismatch"):
        verifier.verify_partial()


@pytest.mark.parametrize("section", ["completed_shards", "partial_shards"])
def test_partial_verifier_rejects_mutated_cohort_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    accounting, _ = _copy_partial_publication(tmp_path, monkeypatch)
    accounting[section][0][
        "teacher_and_active_stone_cohorts"
        if section == "completed_shards"
        else "teacher_and_active_stone_cohorts_of_staged_rows"
    ]["PUCT_>32"]["rows"] += 1
    (tmp_path / "accounting.json").write_text(json.dumps(accounting))
    with pytest.raises(
        ValueError,
        match="(completed_shard_accounting_mismatch|partial_shard_accounting_mismatch)",
    ):
        verifier.verify_partial()
