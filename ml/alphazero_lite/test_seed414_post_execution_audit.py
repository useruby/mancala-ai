"""Tests for the post-execution seed414 audit boundary conditions."""

from __future__ import annotations

import json

import pytest

from ml.alphazero_lite import seed414_post_execution_audit as audit


def test_published_omitted_state_coverage_and_immutability() -> None:
    before = {
        path: path.read_bytes()
        for path in (audit.ORIGINAL_REGISTRATION, audit.ORIGINAL_PROOF, audit.SUITE)
    }
    proof, receipt = audit.build_audit()
    assert proof["additional_identity_count"] == 63471
    assert proof["corrected_union_count"] == 171434
    assert proof["frozen_suite_collision_count"] == 0
    assert receipt["audit_timing"] == "after_execution"
    assert receipt["decision"] == "confirmation_criteria_not_met"
    assert {path: path.read_bytes() for path in before} == before


def test_missing_or_altered_ledger_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(audit, "LEDGER", missing)
    with pytest.raises(FileNotFoundError):
        audit.verify_ledger_binding()

    altered = tmp_path / "altered.jsonl"
    altered.write_text("{}\n")
    monkeypatch.setattr(audit, "LEDGER", altered)
    with pytest.raises(ValueError, match="binding_mismatch"):
        audit.verify_ledger_binding()


def test_malformed_replay_rejected() -> None:
    with pytest.raises(ValueError, match="ledger_row_count_mismatch"):
        audit.replay_ledger([{}])


def test_suite_collision_detection_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audit,
        "replay_ledger",
        lambda rows: (set(), {"games": 4096, "additional_unique_identities": 0}),
    )
    with pytest.raises(ValueError, match="expected_counts_mismatch"):
        audit.build_audit()


def test_collision_report_names_overlapping_identity() -> None:
    from ml.alphazero_lite import build_opening_suite as suites

    state = suites.INITIAL_STATE
    identity = suites.canonical_key(state)
    assert audit.find_suite_collisions({identity}, [{"state": state}]) == [identity]


def test_published_receipt_binds_corrected_proof() -> None:
    receipt = json.loads(audit.RECEIPT.read_text())
    assert receipt["bindings"]["corrected_proof_sha256"] == audit.digest(
        audit.CORRECTED
    )
    assert receipt["statement"].startswith("This audit occurred after execution.")
