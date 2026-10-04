"""Focused tests for the E4 reference diagnostic's frozen analysis rules."""

from __future__ import annotations

import json

from ml.alphazero_lite import seed398_e4_reference_diagnostic as diagnostic


def test_registration_binds_exact_e4_and_seed455_state_order() -> None:
    registration = diagnostic.register()
    source = json.loads((diagnostic.SOURCE / "registration.json").read_text())
    assert registration["states"] == source["states"]
    assert registration["reference"]["checkpoint_sha256"] == (
        "4bc05f8284d5adb5beac5090dbf2c09686c84831131cf3ceede606587ec127f4"
    )
    assert registration["search"]["budgets"] == [1536, 384]


def test_incomplete_analysis_evidence_is_rejected() -> None:
    registration = json.loads((diagnostic.DATA / "registration.json").read_text())
    try:
        diagnostic.analyze(registration, [])
    except ValueError as exc:
        assert str(exc) == "incomplete_outcome_evidence"
    else:
        raise AssertionError("incomplete evidence was accepted")
