"""Fail-closed tests for source coverage and exclusion-union receipts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite.seed482_fresh_confirmation import (
    _canonical_json,
    canonical_identity,
    float32_identity,
    sha_bytes,
)
from ml.alphazero_lite.seed483_exclusion_closure import (
    authorize_generation,
    verify_coverage_document,
)
from ml.alphazero_lite import seed483_exclusion_closure as exclusion_closure


STATE = {
    "player_pits": [4, 4, 4, 4, 4, 4],
    "opponent_pits": [4, 4, 4, 4, 4, 4],
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def _document(root: Path, *, contribution: str = "reconstructed_identities") -> dict:
    source = root / "source.json"
    source.write_text("{}\n")
    row = {
        "experiment": 427,
        "source_artifacts": [
            {"path": source.name, "sha256": sha_bytes(source.read_bytes())}
        ],
        "contribution": contribution,
        "identities": [
            {
                "state": STATE,
                "canonical_identity": canonical_identity(STATE),
                "float32_identity": float32_identity(STATE),
            }
        ],
    }
    rows = [row]
    rows.extend(
        {
            "experiment": seed,
            "source_artifacts": [
                {"path": source.name, "sha256": sha_bytes(source.read_bytes())}
            ],
            "contribution": "reconstructed_identities",
            "identities": row["identities"],
        }
        for seed in range(428, 481)
    )
    body = {"schema": "test", "status": "complete", "experiments": rows}
    body["receipt_sha256"] = sha_bytes(_canonical_json(body))
    return body


def _rebound(document: dict) -> None:
    document.pop("receipt_sha256", None)
    document["receipt_sha256"] = sha_bytes(_canonical_json(document))


def test_valid_baseline_coverage_fixture_passes(tmp_path: Path) -> None:
    document = _document(tmp_path)
    pair = (canonical_identity(STATE), float32_identity(STATE))
    verify_coverage_document(document, tmp_path, {pair})


def test_missing_source_coverage_fails_closed(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["experiments"][0]["source_artifacts"][0]["path"] = "missing.json"
    _rebound(document)
    with pytest.raises(FileNotFoundError):
        verify_coverage_document(
            document, tmp_path, {(canonical_identity(STATE), float32_identity(STATE))}
        )


def test_false_subset_claim_is_rejected(tmp_path: Path, monkeypatch) -> None:
    document = _document(tmp_path)
    row = document["experiments"][0]
    row["contribution"] = "verified_subset"
    valid_pair = (canonical_identity(STATE), float32_identity(STATE))
    population_pairs = {valid_pair}
    monkeypatch.setattr(
        exclusion_closure,
        "_population_identity_pairs",
        lambda _root, _ref: population_pairs,
    )
    row["subset_proof"] = {
        "population_ref": "fixture_population",
        "population_source": "source.json",
        "population_pair_count": 1,
        "population_pairs_sha256": sha_bytes(
            json.dumps([list(valid_pair)], separators=(",", ":")).encode()
        ),
    }
    outside = dict(STATE, player_store=1)
    row["identities"] = [
        {
            "state": outside,
            "canonical_identity": canonical_identity(outside),
            "float32_identity": float32_identity(outside),
        }
    ]
    _rebound(document)
    with pytest.raises(ValueError, match="coverage_identity_outside_subset_population"):
        verify_coverage_document(
            document,
            tmp_path,
            {(canonical_identity(STATE), float32_identity(STATE))},
        )


def test_rebound_receipt_with_corrupt_coverage_is_rejected(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["experiments"][0]["contribution"] = "unrecognized_status"
    # Rebinding the outer receipt cannot make malformed/false evidence pass.
    _rebound(document)
    with pytest.raises(ValueError, match="coverage_contribution_invalid"):
        verify_coverage_document(
            document,
            tmp_path,
            {(canonical_identity(STATE), float32_identity(STATE))},
        )


def test_stale_receipt_is_rejected_after_union_identity_mutation(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path)
    document["experiments"][0]["identities"][0]["float32_identity"] = "0" * 64
    with pytest.raises(ValueError, match="coverage_receipt_mismatch"):
        verify_coverage_document(
            document,
            tmp_path,
            {(canonical_identity(STATE), float32_identity(STATE))},
        )


def test_successful_and_failed_checks_leave_source_bytes_and_mtimes_unchanged(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path)
    source = tmp_path / "source.json"
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    pair = (canonical_identity(STATE), float32_identity(STATE))
    verify_coverage_document(document, tmp_path, {pair})
    with pytest.raises(ValueError, match="coverage_identity_outside_final_union"):
        verify_coverage_document(document, tmp_path, set())
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_generation_authorization_reconstructs_complete_fixture(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document["independent_verification"] = "passed"
    pair = (canonical_identity(STATE), float32_identity(STATE))
    union_digest = sha_bytes(json.dumps([list(pair)], separators=(",", ":")).encode())
    document["reconstructed_union"] = {"union_sha256": union_digest}
    _rebound(document)
    union = {
        "status": "complete",
        "union_sha256": union_digest,
        "final_identity_pairs_sha256": union_digest,
    }
    authorize_generation(document, union, root=tmp_path, union_pairs={pair})

    incomplete = dict(document, status="incomplete")
    _rebound(incomplete)
    with pytest.raises(ValueError, match="historical_probe_coverage_not_verified"):
        authorize_generation(incomplete, union, root=tmp_path, union_pairs={pair})
    with pytest.raises(ValueError, match="historical_probe_union_binding_mismatch"):
        authorize_generation(
            document,
            {**union, "union_sha256": "changed"},
            root=tmp_path,
            union_pairs={pair},
        )
    changed_state = dict(STATE, opponent_store=1)
    changed_pair = (canonical_identity(changed_state), float32_identity(changed_state))
    with pytest.raises(ValueError, match="historical_probe_union_binding_mismatch"):
        authorize_generation(
            document,
            union,
            root=tmp_path,
            union_pairs={pair, changed_pair},
        )
