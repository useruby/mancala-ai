"""Regression tests for the A16 lineage closeout contract."""

import copy
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import validate_a16_lineage_closeout as closeout


def ledger() -> dict[str, object]:
    with (closeout.REPO_ROOT / closeout.LEDGER_PATH).open() as handle:
        return json.load(handle)


def write_ledger(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(value))
    return path


def test_closeout_validates_committed_evidence() -> None:
    closeout.validate()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["frozen_pr266"].update({"head_sha": "bad"}), "hashes"),
        (
            lambda value: value["experiments"][1].update(
                {"published_classification": "changed"}
            ),
            "classification",
        ),
        (
            lambda value: value["experiments"].append(
                copy.deepcopy(value["experiments"][0])
            ),
            "duplicate",
        ),
        (
            lambda value: value["branch_eligibility"].update(
                {"more_epochs_or_optimizer_steps": True}
            ),
            "eligible",
        ),
        (
            lambda value: value["experiments"][0].update({"lifecycle": "proposed"}),
            "descendant",
        ),
        (
            lambda value: value["experiments"][0].update(
                {"source_document": "docs/missing.md"}
            ),
            "missing source",
        ),
        (
            lambda value: value["experiments"][1]["consumed_suites"].update(
                {"labels": ["ZZ"]}
            ),
            "registry",
        ),
    ],
)
def test_closeout_rejects_contract_drift(
    tmp_path: Path, mutate: object, message: str
) -> None:
    value = ledger()
    mutate(value)  # type: ignore[operator]

    with pytest.raises(RuntimeError, match=message):
        closeout.validate(ledger_path=write_ledger(tmp_path, value))
