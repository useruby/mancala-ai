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
    tmp_path.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "evidence-copy"
    shutil.copytree(RUN, destination)
    return destination


def _make_passing_gate_fixture(data: Path) -> None:
    """Synthesize internally consistent action margins that clear both efficacy gates."""
    search_path = data / "search-records.jsonl"
    oracle_path = data / "oracle-records.jsonl"
    searches = {
        row["state_hash"]: row
        for row in map(json.loads, search_path.read_text().splitlines())
    }
    oracles = {
        row["state_hash"]: row
        for row in map(json.loads, oracle_path.read_text().splitlines())
    }
    for state_hash, search in searches.items():
        legal = search["legal_moves"]
        baseline = int(search["baseline_action"])
        alternatives = [move for move in legal if move != baseline]
        if not alternatives:
            raise AssertionError("fixture root must have at least two legal actions")
        margins = {str(move): (-2 if move == baseline else 1) for move in legal}
        priors = [float(value) for value in search["priors"]]
        exact = max(alternatives, key=lambda move: (1, priors[move], -move))
        oracles[state_hash]["action_margins"] = margins
        search["exact_action"] = exact
    oracle_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in oracles.values())
    )
    search_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches.values())
    )
    analysis = diagnostic.analyze_published_run(data)
    roots = analysis.pop("root_records")
    (data / "root-metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in roots)
    )
    (data / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2) + "\n"
    )


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


def test_fresh_package_verifier_accepts_both_gate_outcomes(tmp_path: Path) -> None:
    failing_package = _copy_run(tmp_path / "failed")
    failed = verifier.verify(failing_package)
    assert failed["decision"] == "no_cutoff_21_arena_proposal"

    passing_package = _copy_run(tmp_path / "passed")
    _make_passing_gate_fixture(passing_package)
    passed = verifier.verify(passing_package)
    assert passed["decision"] == "supports_proposing_separate_cutoff_21_arena_only"


def test_fresh_shortfall_output_writes_count_supplement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fresh-diagnostic"
    audit = {"selected_roots": 0, "shortfall": 512, "ranking": "seed559 test"}
    supplement = {"schema": "seed559-cohort-count-audit-supplement-v1"}
    monkeypatch.setattr(diagnostic, "select", lambda: ([], audit))
    monkeypatch.setattr(diagnostic, "count_audit_supplement", lambda: supplement)

    diagnostic.freeze_and_run(output)

    assert (
        json.loads((output / "count-audit-supplement.json").read_text()) == supplement
    )
    assert (
        json.loads((output / "decision.json").read_text())["decision"] == "no_decision"
    )


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
