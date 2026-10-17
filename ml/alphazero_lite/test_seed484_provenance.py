"""Tests for PR/experiment separation in the seed484 provenance graph."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from ml.alphazero_lite.seed484_provenance import (
    EXPECTED_MAPPINGS,
    PROVENANCE_PATH,
    load_provenance,
    replay_seed461_game,
    replay_seed461_opening,
    seed467_source_receipt,
    verify_identity_snapshot,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import (
    canonical_identity,
    float32_identity,
)


def test_repository_provenance_baseline_passes() -> None:
    root = Path(__file__).resolve().parents[2]
    graph = load_provenance(root)
    assert {
        row["pull_request"]: row["experiment_seed"] for row in graph["pr_mappings"]
    } == EXPECTED_MAPPINGS


def _copy_graph(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[2]
    graph = json.loads((source_root / PROVENANCE_PATH).read_text())
    for row in graph["pr_mappings"]:
        for evidence in row["evidence"]:
            path = tmp_path / evidence
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture evidence\n")
    for evidence in [
        *graph["bound_receipts"],
        "docs/data/seed484-seed467-recovered-identities.json.gz",
    ]:
        target_evidence = tmp_path / evidence
        target_evidence.parent.mkdir(parents=True, exist_ok=True)
        target_evidence.write_bytes((source_root / evidence).read_bytes())
    target = tmp_path / PROVENANCE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph))
    return target


def test_pr_number_used_as_experiment_seed_is_rejected(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"][0]["experiment_seed"] = 458
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_pr_seed_mapping_mismatch"):
        load_provenance(tmp_path)


def test_pr_seed_mapping_requires_matching_merge_commit(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"][0]["merge_commit"] = "0" * 40
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_pr_commit_binding_mismatch"):
        load_provenance(tmp_path)


def test_preparation_or_correction_cannot_claim_new_population(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"][0]["population_action"] = "new_probe_population"
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_pr_population_claim_mismatch"):
        load_provenance(tmp_path)


def test_activity_claim_is_bound_to_pr_evidence(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"][0]["activity_evidence"] = "The preparation PR launched probes."
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_pr_activity_evidence_mismatch"):
        load_provenance(tmp_path)


def test_omitted_mapping_is_rejected(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"].pop()
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_pr_inventory_mismatch"):
        load_provenance(tmp_path)


def test_missing_concrete_evidence_is_rejected(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["pr_mappings"][0]["evidence"] = ["missing.json"]
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_evidence_unavailable"):
        load_provenance(tmp_path)


def test_changed_registration_receipt_is_rejected(tmp_path: Path) -> None:
    _copy_graph(tmp_path)
    registration = (
        tmp_path / "docs/data/seed457-fresh-subgroup-projection/registration.json"
    )
    registration.write_text(registration.read_text() + "changed\n")
    with pytest.raises(ValueError, match="provenance_receipt_hash_mismatch"):
        load_provenance(tmp_path)


def test_missing_receipt_binding_is_rejected(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["bound_receipts"].pop(
        "docs/data/seed457-fresh-subgroup-projection/registration.json"
    )
    target.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match="provenance_receipt_inventory_mismatch"):
        load_provenance(tmp_path)


def test_apparent_pr_seed_gap_cannot_be_relabelled_as_empty(tmp_path: Path) -> None:
    target = _copy_graph(tmp_path)
    graph = json.loads(target.read_text())
    graph["apparent_seed_number_gaps"][0]["status"] = "evidence_backed_empty"
    target.write_text(json.dumps(graph))
    with pytest.raises(
        ValueError, match="provenance_apparent_gap_classification_mismatch"
    ):
        load_provenance(tmp_path)


def test_seed461_absolute_opening_prefix_reaches_registered_root() -> None:
    opening = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/data/seed461-cosine-lr-ablation-openings.jsonl"
        )
        .read_text()
        .splitlines()[0]
    )
    assert replay_seed461_opening(opening["prefix_moves"]) == opening["state"]


def test_seed461_replay_uses_absolute_actions_and_reconciles_seat() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    row = {
        "trajectory": "6",
        "challenger_player": 1,
        "margin": 1,
        "winner": "challenger",
    }
    assert replay_seed461_game(start, row) == [start]


def test_seed461_relative_action_is_not_silently_converted() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    with pytest.raises(ValueError, match="seed461_absolute_action_illegal"):
        replay_seed461_game(
            start,
            {
                "trajectory": "0",
                "challenger_player": 1,
                "margin": 1,
                "winner": "challenger",
            },
        )


def test_seed461_extra_turn_keeps_current_seat() -> None:
    game = KalahGame([1, 0, 0, 0, 0, 1] + [1] * 6, [0, 0], 0)
    assert game.move(5)
    assert game.current_player == 0
    assert game.pits[0] == 1
    assert game.move(0)


def test_seed461_full_replay_preserves_extra_turn_and_seat_outcome() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 1, 1],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    game = KalahGame.from_state(start)
    actions = [5]
    players = [game.current_player]
    assert game.move(5)
    players.append(game.current_player)
    assert game.move(4)
    actions.append(4)
    while not game.over():
        players.append(game.current_player)
        relative = game.possible_moves()[0]
        action = game.pit_index(relative)
        actions.append(action)
        assert game.move(action)
        assert len(actions) <= 200
    margin = game.captured_seeds[1] - game.captured_seeds[0]
    row = {
        "trajectory": ",".join(map(str, actions)),
        "challenger_player": 1,
        "margin": margin,
        "winner": "draw" if margin == 0 else "challenger" if margin > 0 else "current",
    }
    replayed = replay_seed461_game(start, row)
    assert players[0] == players[1] == 0
    assert replayed[0] == start


def test_recovered_seed461_first_action_conflict_is_not_transformed() -> None:
    import tarfile

    root = Path(__file__).resolve().parents[2]
    with tarfile.open(
        root / "docs/data/seed483-recovered-seed461-arena-games.tar.gz"
    ) as archive:
        member = archive.getmember(
            "seed461-cosine-lr-ablation/arena-e4/order_38611_A-games.jsonl"
        )
        member_file = archive.extractfile(member)
        assert member_file is not None
        game_row = json.loads(member_file.readline())
    opening = json.loads(
        (root / "docs/data/seed461-cosine-lr-ablation-openings.jsonl")
        .read_text()
        .splitlines()[0]
    )
    assert game_row["trajectory"].split(",", maxsplit=1)[0] == "8"
    with pytest.raises(ValueError, match="seed461_absolute_action_illegal:0:8"):
        replay_seed461_game(opening["state"], game_row)


def test_seed461_replay_rejects_actions_after_terminal() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    with pytest.raises(ValueError, match="seed461_trajectory_after_terminal"):
        replay_seed461_game(
            start,
            {
                "trajectory": "6,6",
                "challenger_player": 1,
                "margin": 1,
                "winner": "challenger",
            },
        )


def test_seed461_replay_reconciles_recorded_outcome() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    row = {
        "trajectory": "6",
        "challenger_player": 0,
        "margin": -1,
        "winner": "challenger",
    }
    with pytest.raises(ValueError, match="seed461_recorded_winner_mismatch"):
        replay_seed461_game(start, row)


def test_seed461_replay_reconciles_recorded_game_length() -> None:
    start = {
        "player_pits": [0, 0, 0, 0, 0, 0],
        "opponent_pits": [1, 0, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    row = {
        "trajectory": "6",
        "challenger_player": 1,
        "margin": 1,
        "winner": "challenger",
        "game_length": 2,
    }
    with pytest.raises(ValueError, match="seed461_recorded_game_length_mismatch"):
        replay_seed461_game(start, row)


def test_seed467_snapshot_baseline_fixture_passes(tmp_path: Path, monkeypatch) -> None:
    state = {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    row = {
        "state": state,
        "canonical_identity": canonical_identity(state),
        "float32_identity": float32_identity(state),
        "provenance": {
            "experiment": 467,
            "source": ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
            "line": 0,
            "game_index": 0,
            "move_index": 0,
        },
    }
    payload = {
        "schema": "seed484-seed467-recovered-identities-v1",
        "source_binding": {
            "path": ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
            "sha256": "6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76",
            "size_bytes": 749547469,
            "row_count": 1,
        },
        "records": [row],
    }
    import gzip

    path = tmp_path / "snapshot.json.gz"
    packed = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(packed)
    import hashlib

    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_SNAPSHOT_SHA256",
        hashlib.sha256(packed).hexdigest(),
    )
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_GENERATION_SHA256",
        "3c989868860beed12b9b465f691e4d31466afc75594bfcba6a87d3b96c719f3d",
    )
    monkeypatch.setattr("ml.alphazero_lite.seed484_provenance.SEED467_ROW_COUNT", 1)
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_DISTINCT_PAIRS", 1
    )
    payload["source_binding"]["receipt_sha256"] = (
        "3c989868860beed12b9b465f691e4d31466afc75594bfcba6a87d3b96c719f3d"
    )
    packed = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(packed)
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_SNAPSHOT_SHA256",
        hashlib.sha256(packed).hexdigest(),
    )
    monkeypatch.setattr("ml.alphazero_lite.seed484_provenance.SEED467_ROW_COUNT", 1)
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_DISTINCT_PAIRS", 1
    )
    assert verify_identity_snapshot(path)["distinct_pairs"] == 1


def test_seed467_snapshot_receipt_rebound_does_not_hide_state_corruption(
    tmp_path: Path, monkeypatch
) -> None:
    state = {
        "player_pits": [4] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    row = {
        "state": state,
        "canonical_identity": "0" * 64,
        "float32_identity": float32_identity(state),
        "provenance": {
            "experiment": 467,
            "source": ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
            "line": 0,
            "game_index": 0,
            "move_index": 0,
        },
    }
    payload = {
        "schema": "seed484-seed467-recovered-identities-v1",
        "source_binding": {
            "path": ".tmp/seed455-nextgen-s467-parent-anchor-w010/runs/seed455-nextgen-s467-parent-anchor-w010-iter1/self_play.jsonl",
            "sha256": "6f7038dccf282fb9365a031a06f8c5a8af4a2b8a6eda32725fe125b06d654a76",
            "size_bytes": 749547469,
            "row_count": 1,
            "receipt_sha256": "3c989868860beed12b9b465f691e4d31466afc75594bfcba6a87d3b96c719f3d",
        },
        "records": [row],
    }
    import gzip
    import hashlib

    path = tmp_path / "snapshot.json.gz"
    packed = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(packed)
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_SNAPSHOT_SHA256",
        hashlib.sha256(packed).hexdigest(),
    )
    monkeypatch.setattr("ml.alphazero_lite.seed484_provenance.SEED467_ROW_COUNT", 1)
    monkeypatch.setattr(
        "ml.alphazero_lite.seed484_provenance.SEED467_DISTINCT_PAIRS", 1
    )
    with pytest.raises(ValueError, match="seed467_snapshot_identity_mismatch"):
        verify_identity_snapshot(path)


def _write_seed467_source_fixture(tmp_path: Path, source: bytes, monkeypatch) -> Path:
    from ml.alphazero_lite import seed484_provenance as provenance

    monkeypatch_values = {
        "SEED467_SOURCE": ".tmp/test/self_play.jsonl",
        "SEED467_GENERATION": "docs/generation.json",
        "SEED467_SHA256": hashlib.sha256(source).hexdigest(),
        "SEED467_SIZE": len(source),
    }
    for name, value in monkeypatch_values.items():
        monkeypatch.setattr(provenance, name, value)
    source_path = tmp_path / provenance.SEED467_SOURCE
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source)
    generation_path = tmp_path / provenance.SEED467_GENERATION
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    generation_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": provenance.SEED467_SOURCE,
                        "role": "self_play",
                        "sha256": provenance.SEED467_SHA256,
                        "size_bytes": provenance.SEED467_SIZE,
                    }
                ]
            }
        )
    )
    return source_path


def test_seed467_exact_source_bytes_pass_receipt_verification(
    tmp_path: Path, monkeypatch
) -> None:
    source = b'{"state":[],"game_index":0,"move_index":0}\n'
    _write_seed467_source_fixture(tmp_path, source, monkeypatch)
    result = seed467_source_receipt(tmp_path)
    assert result["sha256"] == hashlib.sha256(source).hexdigest()


def test_seed467_altered_source_bytes_fail_receipt_verification(
    tmp_path: Path, monkeypatch
) -> None:
    source = b'{"state":[],"game_index":0,"move_index":0}\n'
    path = _write_seed467_source_fixture(tmp_path, source, monkeypatch)
    path.write_bytes(source + b"altered")
    with pytest.raises(ValueError, match="seed467_recovered_source_identity_mismatch"):
        seed467_source_receipt(tmp_path)


def test_seed467_missing_population_is_not_empty(tmp_path: Path, monkeypatch) -> None:
    import ml.alphazero_lite.seed484_provenance as provenance

    generation = tmp_path / provenance.SEED467_GENERATION
    generation.parent.mkdir(parents=True, exist_ok=True)
    generation.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": provenance.SEED467_SOURCE,
                        "role": "self_play",
                        "sha256": provenance.SEED467_SHA256,
                        "size_bytes": provenance.SEED467_SIZE,
                    }
                ]
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="seed467_source_unavailable"):
        seed467_source_receipt(tmp_path)
