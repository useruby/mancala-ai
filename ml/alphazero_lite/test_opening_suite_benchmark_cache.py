from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.run_opening_suite_seat_benchmark import (
    METRIC_SCHEMA_VERSION,
    cache_is_reusable,
    record_files_sha256,
)


def _manifest() -> dict:
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "suite_sha256": "suite-a",
        "candidate_sha256": "candidate-a",
        "current_sha256": "control-a",
        "budget_pair": "384:256",
        "c_puct": 1.25,
        "tactical_root_bias": 0.0,
        "seed_contract": "azlite_eval_seed_v2",
        "base_seed": 42,
        "games_per_opening": 2,
        "root_policy_mode": "deterministic",
    }


def _metrics(candidate: list[Path], control: list[Path]) -> dict:
    return {
        "paired_candidate_effect": 0.01,
        "cache_manifest": {
            **_manifest(),
            "candidate_record_sha256": record_files_sha256(candidate),
            "current_control_record_sha256": record_files_sha256(control),
        },
    }


def _files(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    candidate = [tmp_path / "candidate-0.jsonl", tmp_path / "candidate-1.jsonl"]
    control = [tmp_path / "control-0.jsonl", tmp_path / "control-1.jsonl"]
    for path in candidate + control:
        path.write_text('{"opening_index": 0}\n', encoding="utf-8")
    return candidate, control


def test_pre_v2_metrics_are_invalidated(tmp_path: Path) -> None:
    candidate, control = _files(tmp_path)
    assert not cache_is_reusable(
        {"cache_context": _manifest()},
        _manifest(),
        candidate_record_paths=candidate,
        control_record_paths=control,
    )


def test_missing_control_records_are_invalidated(tmp_path: Path) -> None:
    candidate, control = _files(tmp_path)
    control[1].unlink()
    assert not cache_is_reusable(
        _metrics(candidate, control),
        _manifest(),
        candidate_record_paths=candidate,
        control_record_paths=control,
    )


def test_changed_suite_artifact_or_runtime_invalidates_cache(tmp_path: Path) -> None:
    candidate, control = _files(tmp_path)
    metrics = _metrics(candidate, control)
    for key, value in (
        ("suite_sha256", "suite-b"),
        ("candidate_sha256", "candidate-b"),
        ("current_sha256", "control-b"),
        ("c_puct", 0.9),
    ):
        expected = {**_manifest(), key: value}
        assert not cache_is_reusable(
            metrics,
            expected,
            candidate_record_paths=candidate,
            control_record_paths=control,
        )


def test_corrupted_control_output_invalidates_cache(tmp_path: Path) -> None:
    candidate, control = _files(tmp_path)
    metrics = _metrics(candidate, control)
    control[0].write_text('{"corrupted": true}\n', encoding="utf-8")
    assert not cache_is_reusable(
        metrics,
        _manifest(),
        candidate_record_paths=candidate,
        control_record_paths=control,
    )


def test_valid_v2_cache_is_reused(tmp_path: Path) -> None:
    candidate, control = _files(tmp_path)
    assert cache_is_reusable(
        _metrics(candidate, control),
        _manifest(),
        candidate_record_paths=candidate,
        control_record_paths=control,
    )
