"""Tests for seed542 continuation source and partial-game provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import seed542_amendment


ROOT = Path(__file__).resolve().parents[2]
STAGING = (
    ROOT
    / "docs/data/seed542-selfplay-4096/azlite-self-play-shards-etoy6j_i/worker_0.jsonl"
)


def _manifest() -> dict:
    sources = [
        {
            "path": f"full-{seed}.jsonl",
            "sha256": "a" * 64,
            "role": "training",
            "games": 64,
            "source_seed": seed,
        }
        for seed in [542000, *range(542002, 542048)]
    ]
    sources.extend(
        [
            {
                "path": "recovered.jsonl",
                "sha256": "b" * 64,
                "role": "training",
                "games": 51,
                "source_seed": 542001,
                "original_source_path": "original.jsonl",
                "original_source_sha256": "e" * 64,
                "provenance_path": "provenance.jsonl",
                "provenance_sha256": "f" * 64,
            },
            {
                "path": "supplement.jsonl",
                "sha256": "c" * 64,
                "role": "training",
                "games": 13,
                "source_seed": 542064,
            },
        ]
    )
    sources.extend(
        {
            "path": f"development-{index}.jsonl",
            "sha256": "d" * 64,
            "role": "development",
            "games": 64,
            "source_seed": 542048 + index,
        }
        for index in range(16)
    )
    return {"sources": sources}


def test_amended_manifest_reconciles_exact_partition_counts() -> None:
    seed542_amendment.validate_source_manifest(_manifest())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove_supplement", "amended_source_totals_mismatch"),
        ("supplement_role", "amended_source_totals_mismatch"),
        ("duplicate_path", "amended_source_path_duplicate"),
    ],
)
def test_amended_manifest_rejects_accounting_drift(mutation: str, message: str) -> None:
    manifest = _manifest()
    if mutation == "remove_supplement":
        manifest["sources"] = [
            s for s in manifest["sources"] if s["path"] != "supplement.jsonl"
        ]
    elif mutation == "supplement_role":
        next(s for s in manifest["sources"] if s["path"] == "supplement.jsonl")[
            "role"
        ] = "development"
    else:
        manifest["sources"][-1]["path"] = manifest["sources"][-2]["path"]
    with pytest.raises(ValueError, match=message):
        seed542_amendment.validate_source_manifest(manifest)


def test_extraction_preserves_only_complete_original_rows_with_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "worker_0.jsonl"
    source.write_bytes(STAGING.read_bytes())
    derivative = tmp_path / "recovered.jsonl"
    provenance = tmp_path / "recovered-provenance.jsonl"
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = seed542_amendment.extract_complete_games(source, derivative, provenance)
    checked = seed542_amendment.verify_extraction(source, derivative, provenance)

    assert result["complete_games"] == 51
    assert result["excluded_incomplete_game_indices"] == [51]
    assert result["excluded_unstarted_game_indices"] == list(range(52, 64))
    assert checked["rows"] == 1768
    assert checked["source_sha256"] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    provenance_rows = [json.loads(line) for line in provenance.read_text().splitlines()]
    assert len(provenance_rows) == 1768
    assert all(row["source_sha256"] == before for row in provenance_rows)
    assert all(row["original_line"] >= 1 for row in provenance_rows)


def test_extraction_verifier_rejects_changed_derivative_row(tmp_path: Path) -> None:
    source = tmp_path / "worker_0.jsonl"
    source.write_bytes(STAGING.read_bytes())
    derivative = tmp_path / "recovered.jsonl"
    provenance = tmp_path / "recovered-provenance.jsonl"
    seed542_amendment.extract_complete_games(source, derivative, provenance)
    data = derivative.read_bytes()
    derivative.write_bytes(data.replace(b'"value":', b'"value" :', 1))
    with pytest.raises(
        ValueError,
        match="recovered_provenance_derivative_line_hash_mismatch|recovered_derivative_row_bytes_changed",
    ):
        seed542_amendment.verify_extraction(source, derivative, provenance)
