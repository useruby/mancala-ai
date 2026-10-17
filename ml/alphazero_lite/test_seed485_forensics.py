"""Tests for authenticated and explicitly incomplete seed461 forensics."""

from __future__ import annotations

import json
import hashlib
import tarfile
from pathlib import Path

import pytest

from ml.alphazero_lite.seed485_forensics import (
    ARCHIVE,
    build_inventory,
    verify_inventory,
)


def test_committed_inventory_authenticates_all_receipt_matched_streams() -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root)
    assert inventory["stream_count"] == 58
    assert inventory["receipt_matched_stream_count"] == 58
    assert inventory["registration_bound_stream_count"] == 48
    assert inventory["writer_contract_stream_count"] == 0
    assert inventory["status"] == "incomplete_writer_and_rules_bindings"
    assert all(
        stream["writer_action_semantics"] == "unknown"
        for stream in inventory["streams"]
    )
    assert all(
        stream["rules_semantics"] == "unknown" for stream in inventory["streams"]
    )
    assert any(stream["first_game_registered_root"] for stream in inventory["streams"])
    packet = inventory["first_failure_packet"]
    assert packet["registered_root"]["current_player"] == 0
    assert packet["legal_absolute_pits_under_registered_orientation"] == [0, 1, 2, 4, 5]
    assert packet["first_trajectory_value"] == 8
    assert (
        packet["comparison_contract"]["binding_to_this_run"]
        == "not established; comparison only"
    )


def test_inventory_verifier_rejects_rebound_or_changed_rows(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    assert verify_inventory(root, path)["stream_count"] == 58
    inventory["streams"][0]["games_sha256"] = "0" * 64
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="reconstruction_mismatch"):
        verify_inventory(root, path)
    inventory = build_inventory(root)
    inventory["streams"][0]["games_path"] = "rebound.jsonl"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="reconstruction_mismatch"):
        verify_inventory(root, path)


def test_inventory_exposes_missing_execution_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root)
    assert inventory["generation_authorized"] is False
    assert all(stream["writer_source"] is None for stream in inventory["streams"])
    assert all(stream["rules_source"] is None for stream in inventory["streams"])
    assert all(
        stream["trajectory_encoding"].endswith("not authenticated for this run")
        for stream in inventory["streams"]
    )


def test_inventory_rejects_altered_suite_and_log_associations(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory))
    inventory["streams"][0]["suite_sha256_registered"] = "f" * 64
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="reconstruction_mismatch"):
        verify_inventory(root, path)


def _write_fixture(root: Path) -> tuple[Path, Path, bytes]:
    data = root / "docs/data"
    data.mkdir(parents=True)
    registration = {
        "evaluation": {
            "suite": {
                "path": "/original/docs/data/seed461-fixture-openings.jsonl",
                "sha256": "suitehash",
            }
        }
    }
    reg_bytes = json.dumps(registration).encode()
    (data / "seed461-fixture-registration.json").write_bytes(reg_bytes)
    suite = b'{"state":{"player_pits":[1,0,0,0,0,0],"opponent_pits":[0,0,0,0,0,0],"player_store":0,"opponent_store":0,"current_player":0}}\n'
    (data / "seed461-fixture-openings.jsonl").write_bytes(suite)
    log = b'{"game_index":0,"opening_index":0,"game_within_opening":0,"challenger_player":0,"opening_prefix_moves":[],"trajectory":"0","margin":1,"winner":"challenger","game_length":1}\n'
    log_hash = hashlib.sha256(log).hexdigest()
    reg_hash = hashlib.sha256(reg_bytes).hexdigest()
    binding = {
        "registration_sha256": reg_hash,
        "suite_sha256": hashlib.sha256(suite).hexdigest(),
        "candidates": {
            "fixture_A": {"arm": "A", "artifact_sha256": {"model": "artifact"}}
        },
        "reports": {
            "fixture_A": {
                "games": "/original/.tmp/seed461-fixture/arena/fixture_A-games.jsonl",
                "games_sha256": log_hash,
                "report_sha256": "reporthash",
            }
        },
    }
    (data / "seed461-fixture-evaluation-binding.json").write_text(json.dumps(binding))
    archive_path = root / ARCHIVE
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        import io

        info = tarfile.TarInfo("seed461-fixture/arena/fixture_A-games.jsonl")
        info.size = len(log)
        archive.addfile(info, io.BytesIO(log))
    return data / "seed461-fixture-evaluation-binding.json", archive_path, log


def test_source_receipt_rebound_and_altered_archive_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixture"
    binding_path, archive_path, log = _write_fixture(root)
    inventory = build_inventory(root, expected_streams=1)
    assert inventory["streams"][0]["writer_action_semantics"] == "unknown"
    binding = json.loads(binding_path.read_text())
    binding["reports"]["fixture_A"]["games_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="game_log_hash_mismatch"):
        build_inventory(root, expected_streams=1)
    binding["reports"]["fixture_A"]["games_sha256"] = hashlib.sha256(log).hexdigest()
    binding_path.write_text(json.dumps(binding))
    archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        import io

        corrupted = log.replace(b'"trajectory":"0"', b'"trajectory":"1"')
        info = tarfile.TarInfo("seed461-fixture/arena/fixture_A-games.jsonl")
        info.size = len(corrupted)
        archive.addfile(info, io.BytesIO(corrupted))
    with pytest.raises(ValueError, match="game_log_hash_mismatch"):
        build_inventory(root, expected_streams=1)
