"""Receipt-rebound mutation tests for the seed477 correction verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import seed477_counterfactual_accounting as correction
from ml.alphazero_lite import verify_seed477_counterfactual_accounting as verifier


@pytest.fixture
def published_copy(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    root.mkdir()
    paths = [*correction.SOURCES, *correction.INPUTS.values()]
    for directory in (correction.BASE, correction.PRIOR):
        receipt = json.loads((correction.ROOT / directory / "receipt.json").read_text())
        paths.extend(directory / name for name in receipt if name != "status")
    for relative in paths:
        source = correction.ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    correction.publish(root)
    verifier.verify(root)
    return root


def _rebind(root: Path) -> None:
    out = root / correction.OUT
    receipt = json.loads((out / "receipt.json").read_text())
    for name in ("registration.json", "evidence.json", "results.md"):
        receipt[name] = correction.sha256(out / name)
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "restored_counterfactual_fields",
        "altered_actual_accounting",
        "changed_direction_value",
        "missing_record",
        "duplicate_record",
        "changed_categories_or_counts",
        "incorrect_historical_binding",
    ],
)
def test_receipt_rebound_verifier_entrypoint_rejects_semantic_corruption(
    published_copy: Path, mutation: str
) -> None:
    root = published_copy
    out = root / correction.OUT
    report_path = out / "evidence.json"
    report = json.loads(report_path.read_text())
    if mutation == "restored_counterfactual_fields":
        report["steps"][0]["directions"]["exposure"]["full_negative"][
            "observed_loss_change"
        ] = 0.0
    elif mutation == "altered_actual_accounting":
        report["steps"][0]["directions"]["exposure"]["recorded_adam"][
            "observed_loss_change"
        ] += 0.125
    elif mutation == "changed_direction_value":
        report["steps"][0]["directions"]["exposure"]["full_negative"]["norm"] += 0.125
    elif mutation == "missing_record":
        report["steps"].pop(0)
    elif mutation == "duplicate_record":
        report["steps"].append(report["steps"][-1])
    elif mutation == "changed_categories_or_counts":
        report["steps"][0]["directions"]["exposure"]["category"] = "both_descent"
        report["totals"]["exposure"]["category_counts"]["both_descent"] += 1
    elif mutation == "incorrect_historical_binding":
        registration_path = out / "registration.json"
        registration = json.loads(registration_path.read_text())
        registration["original_seed476_receipt_sha256"] = "0" * 64
        registration_path.write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n"
        )
        _rebind(root)
        with pytest.raises(ValueError, match="historical_binding"):
            verifier.verify(root)
        return
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _rebind(root)
    with pytest.raises(ValueError):
        verifier.verify(root)


def test_receipt_rebound_baseline_passes(published_copy: Path) -> None:
    verifier.verify(published_copy)
