"""Tests for bounded, receipt-linked seed461 recovery."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from ml.alphazero_lite.seed486_recovery import build_ledger, verify_ledger


def test_recovery_links_original_game_bytes_without_inventing_source_binding() -> None:
    root = Path(__file__).resolve().parents[2]
    ledger = build_ledger(root)
    assert ledger["stream_count"] == 58
    assert ledger["generation_authorized"] is False
    first = ledger["streams"][0]
    assert first["local_artifacts"][0]["role"] == "game_log"
    assert first["local_artifacts"][0]["sha256"] == first["receipt"]["game_log_sha256"]
    assert first["working_revision"] is None
    assert first["arena_rules_source_binding"] is None
    assert first["explicit_state_trace_rows"] == 0


def test_ledger_verification_rejects_altered_receipts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "ledger.json"
    ledger = build_ledger(root)
    path.write_text(json.dumps(ledger))
    assert verify_ledger(root, path)["stream_count"] == 58
    ledger["streams"][0]["receipt"]["game_log_sha256"] = "0" * 64
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="receipt_mismatch|reconstruction_mismatch"):
        verify_ledger(root, path)


def test_receipt_bound_archive_rows_do_not_claim_explicit_states() -> None:
    root = Path(__file__).resolve().parents[2]
    with tarfile.open(
        root / "docs/data/seed483-recovered-seed461-arena-games.tar.gz", "r:gz"
    ) as archive:
        for member in archive.getmembers():
            if member.isfile() and "seed461" in member.name:
                stream = archive.extractfile(member)
                assert stream is not None
                for line in stream:
                    row = json.loads(line)
                    assert not any(
                        key in row
                        for key in ("states", "state_trace", "explicit_states")
                    )


def test_all_archive_receipts_are_portable_and_required_artifacts_are_named() -> None:
    root = Path(__file__).resolve().parents[2]
    ledger = build_ledger(root)
    assert all(
        stream["receipt"]["archive_member_sha256"]
        == stream["receipt"]["game_log_sha256"]
        for stream in ledger["streams"]
    )
    assert ledger["required_artifact_request"]
