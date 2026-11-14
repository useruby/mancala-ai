"""Synthetic corruption checks for the #588 read-only source verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ml.alphazero_lite.seed588_raw_prior_diagnostic import (
    EXPECTED_ARTIFACTS,
    VerificationError,
    verify,
    verify_documents,
)


def _fixture():
    root = Path(__file__).resolve().parents[2]
    result = verify(root)
    data = root / "docs/data/seed559-fixed-position-feasibility/run-0003"
    cohort = json.loads((data / "cohort-manifest.json").read_bytes())
    freeze = json.loads((data / "execution-freeze.json").read_bytes())
    oracle = [
        json.loads(line)
        for line in (data / "oracle-records.jsonl").read_bytes().splitlines()
    ]
    search = [
        json.loads(line)
        for line in (data / "search-records.jsonl").read_bytes().splitlines()
    ]
    assert result["status"] == "verified"
    return cohort, freeze, oracle, search


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("active", "active_pit_stones"),
        ("oracle", "oracle_action_margins"),
        ("prior", "archived_prior_not_normalized"),
        ("artifact", "T512_freeze_artifact_identity"),
    ],
)
def test_rejects_source_mutations(mutation: str, message: str) -> None:
    cohort, freeze, oracle, search = _fixture()
    cohort, freeze, oracle, search = map(
        copy.deepcopy, (cohort, freeze, oracle, search)
    )
    root = cohort["states"][0]
    if mutation == "active":
        root["active_pit_stones"] += 1
    elif mutation == "oracle":
        oracle[0]["action_margins"].pop(str(oracle[0]["legal_moves"][0]))
    elif mutation == "prior":
        search[0]["priors"] = [0.0] * 6
    else:
        freeze["authenticated_policies"]["T512"] = dict(EXPECTED_ARTIFACTS["P"])
    with pytest.raises(VerificationError, match=message):
        verify_documents(cohort, freeze, oracle, search)
