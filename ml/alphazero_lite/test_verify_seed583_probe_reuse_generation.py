"""Tests for independent digest and checkpoint-summary reconciliation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import validate_varied_opponent_probe_reuse as validation
from ml.alphazero_lite import verify_seed583_probe_reuse_generation as verifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed583-probe-reuse-generation-validation"


def _checkpoint_report() -> tuple[dict, dict]:
    freeze = json.loads((DATA / "freeze.json").read_text())
    results = json.loads((DATA / "results.json").read_text())
    report = validation.verify_evidence(freeze, DATA / "evidence")
    return report, results


def test_published_summary_must_match_checkpoint_accounting() -> None:
    report, results = _checkpoint_report()
    verifier.reconcile_results_report(report, results)

    changed_rows = copy.deepcopy(results)
    changed_rows["outputs"]["retained_training_rows_per_runner"] += 1
    with pytest.raises(ValueError, match="results_output_summary_mismatch"):
        verifier.reconcile_results_report(report, changed_rows)

    changed_cpu = copy.deepcopy(results)
    changed_cpu["cpu_time_seconds"]["v2"]["reaped_native_child_cpu"] += 1
    with pytest.raises(ValueError, match="results_cpu_summary_mismatch"):
        verifier.reconcile_results_report(report, changed_cpu)

    changed_gate = copy.deepcopy(results)
    changed_gate["adoption_gate"]["passed"] = False
    with pytest.raises(ValueError, match="results_adoption_gate_mismatch"):
        verifier.reconcile_results_report(report, changed_gate)


def test_results_cannot_supply_its_own_expected_inventory_digest() -> None:
    freeze = DATA / "freeze.json"
    with pytest.raises(ValueError, match="raw_evidence_inventory_mismatch"):
        verifier.verify(
            freeze,
            DATA / "evidence",
            DATA / "results.json",
            expected_freeze_sha256="edbc5c72031064dde8cb5491efc4e9e2130300acc35325cfcd6ca92bc8371b77",
            expected_evidence_inventory_sha256="0" * 64,
        )
