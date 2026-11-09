"""Regression tests for seed559 evidence integrity and immutable runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import seed559_fixed_position_feasibility as diagnostic
from ml.alphazero_lite import verify_seed559_fixed_position_feasibility as verifier

RUN = diagnostic.DEFAULT_ANALYSIS_DATA


def _copy_run(tmp_path: Path) -> Path:
    destination = tmp_path / "evidence-copy"
    shutil.copytree(RUN, destination)
    return destination


def test_verifier_recomputes_latency_from_per_root_measurements(
    tmp_path: Path,
) -> None:
    copied = _copy_run(tmp_path)
    timings_path = copied / "timings.jsonl"
    timings = [json.loads(line) for line in timings_path.read_text().splitlines()]
    for row in timings:
        row["exact_warm_decision_ms"] = 1000.0
    timings_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in timings)
    )

    with pytest.raises(ValueError, match="latency_summary_does_not_match_per_root"):
        verifier.verify(copied)


def test_verifier_recomputes_confidence_interval_and_gate(tmp_path: Path) -> None:
    copied = _copy_run(tmp_path)
    analysis_path = copied / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["baseline_regret_opening_bootstrap_95_ci"] = [0.0, 0.0]
    analysis["gate"]["exact_mean_latency_at_most_200_ms"] = False
    analysis_path.write_text(json.dumps(analysis, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="analysis_latency_interval_or_gate"):
        verifier.verify(copied)


def test_evaluator_artifact_tamper_rejected_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied_dirs = {name: tmp_path / name for name in diagnostic.EVALUATOR_ARTIFACTS}
    expected = {}
    for name, source in diagnostic.EVALUATOR_ARTIFACTS.items():
        shutil.copytree(source, copied_dirs[name])
        expected[name] = {
            filename: hashlib.sha256(
                (copied_dirs[name] / filename).read_bytes()
            ).hexdigest()
            for filename in diagnostic.RUNTIME_FILENAMES
        }
    metadata = copied_dirs["initializer"] / "metadata.json"
    metadata.write_bytes(metadata.read_bytes() + b" ")
    evaluator_constructor = Mock(side_effect=AssertionError("evaluator constructed"))
    monkeypatch.setattr(diagnostic.arena, "ArtifactEvaluator", evaluator_constructor)

    with pytest.raises(ValueError, match="evaluator_artifact_identity_mismatch"):
        diagnostic.build_evaluators(copied_dirs, expected)
    evaluator_constructor.assert_not_called()


def test_existing_output_directory_is_rejected_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "already-published"
    output.mkdir()
    sentinel = output / "sentinel.bin"
    sentinel.write_bytes(b"preserve-me")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(
        diagnostic,
        "select",
        lambda: (_ for _ in ()).throw(AssertionError("selection must not run")),
    )

    with pytest.raises(FileExistsError, match="diagnostic output must be new"):
        diagnostic.freeze_and_run(output)

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before
