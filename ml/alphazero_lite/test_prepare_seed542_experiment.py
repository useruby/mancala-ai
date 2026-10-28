"""Focused tests for seed542 membership and fixed-update sampling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import prepare_seed542_experiment as preparation


def _row(path: Path, identity: str, *, extra_feature: float = 0.0) -> dict:
    state = [0.0] * 27
    state[:12] = [1 / 48] * 12
    state[15] = extra_feature
    return {
        "state": state,
        "_identity": identity,
        "_path": path,
        "_line": 1,
        "_source_sha256": "a" * 64,
        "_line_sha256": "b" * 64,
    }


def test_membership_excludes_canonical_and_exact_input_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"state": _row(old, "x")["state"]}) + "\n")
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "SEED500", tmp_path)
    dev = [
        _row(tmp_path / "dev.jsonl", "dev:1", extra_feature=1.0),
        _row(tmp_path / "dev.jsonl", "dev:2"),
    ]
    # The old path convention is six seed500 files; supply the identical rows.
    for seed in range(503000, 503006):
        (tmp_path / f"shard-{seed}.jsonl").write_text(old.read_text())
    result = preparation.build_membership(dev, [])
    assert result["counts"]["included_rows"] == 0
    assert result["counts"]["excluded_rows"] == 2
    assert all(not item["included"] for item in result["membership"])
    assert result["counts"]["remaining_canonical_overlap"] == 0
    assert result["counts"]["remaining_float32_overlap"] == 0


def test_matched_update_lengths_retain_seed504_short_batches() -> None:
    lengths = preparation.batch_lengths()
    assert len(lengths) == 108
    assert lengths == ([512] * 26 + [412]) * 4
    assert sum(lengths) == 54896


def test_expanded_sampling_fails_without_replacement_capacity() -> None:
    control = [{"_identity": f"C:{i}"} for i in range(13724)]
    expanded = [{"_identity": f"D:{i}"} for i in range(54895)]
    # Schedule construction must reject an undersized D population.
    with pytest.raises(ValueError, match="expanded_population_cannot_support"):
        preparation.schedules(control, expanded)


def test_corrupted_control_permutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = [{"_identity": f"C:{i}"} for i in range(13724)]
    expanded = [{"_identity": f"D:{i}"} for i in range(54896)]
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: "[]" if path.name == "permutations.json" else "",
    )
    with pytest.raises(ValueError, match="seed504_frozen_permutation_mismatch"):
        preparation.schedules(control, expanded)
