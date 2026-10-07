"""Tamper, relocation, and immutability checks for seed435 evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ml.alphazero_lite.verify_seed435_publication import verify

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed435-adam-direction-screen"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_publication(root: Path) -> Path:
    relocated = root / "checkout"
    shutil.copytree(
        ROOT / "ml",
        relocated / "ml",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    relatives = (
        "docs/data/seed435-adam-direction-screen",
        "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz",
        "docs/data/seed416-policy-target-softening/registration-v3.json",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/source-row-split.json.gz",
        "docs/data/seed416-policy-target-softening/training-freeze-v3/epoch-permutations.json.gz",
        "docs/data/seed426-canonical-overlap/sources",
        "docs/data/seed426-canonical-overlap/row-accounting.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-verification-receipt.json",
        "docs/data/seed426-canonical-overlap/seed427-supplemental-verification-receipt.json",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-manifest.json",
        "docs/data/seed426-canonical-overlap/seed427-evaluation-results.json",
        "docs/data/seed432-policy-target-compatibility/manifest.json",
        "docs/data/seed433-seed432-census-correction/correction-receipt.json",
    )
    for relative in relatives:
        source, target = ROOT / relative, relocated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return relocated


def test_relocated_read_only_cli_and_publication_immutability(tmp_path: Path) -> None:
    before = {path.name: _sha(path) for path in DATA.iterdir() if path.is_file()}
    expected = verify(ROOT)
    relocated = _copy_publication(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed435_publication",
            "--root",
            str(relocated),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(relocated)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["classification"] == expected["classification"]
    assert before == {
        path.name: _sha(path) for path in DATA.iterdir() if path.is_file()
    }


@pytest.mark.parametrize(
    ("relative", "mutate", "error"),
    [
        (
            "docs/data/seed435-adam-direction-screen/A-predictions.npz",
            "byte",
            "prediction_archive_binding",
        ),
        (
            "docs/data/seed426-canonical-overlap/seed427-validation-membership.jsonl.gz",
            "byte",
            "seed427_receipt_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/A-updates.json",
            "radius",
            "original_evidence_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/B-updates.json",
            "order",
            "original_evidence_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/A-audit-deltas.npz",
            "byte",
            "audit_telemetry_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/post-execution-telemetry.json",
            "group",
            "audit_telemetry_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/A-B-audit-delta-cosines.npz",
            "byte",
            "audit_telemetry_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/results.json",
            "classification",
            "original_evidence_binding",
        ),
        (
            "docs/data/seed435-adam-direction-screen/supplemental-receipt.json",
            "metric",
            "recomputed_results_receipt",
        ),
    ],
)
def test_altered_evidence_is_rejected(
    tmp_path: Path, relative: str, mutate: str, error: str
) -> None:
    relocated = _copy_publication(tmp_path)
    path = relocated / relative
    if mutate == "byte":
        path.write_bytes(path.read_bytes() + b"x")
    elif mutate in {"radius", "order", "classification"}:
        payload = json.loads(path.read_text())
        if mutate == "radius":
            payload[0]["requested_radius"] += 0.1
        elif mutate == "order":
            payload[0]["batch"]["expanded_positions"][0] += 1
        else:
            payload["decision"]["classification"] = (
                "advance_to_separately_preregistered_strength_experiment"
            )
        path.write_text(json.dumps(payload))
    elif mutate == "group":
        payload = json.loads(path.read_text())
        first = next(iter(payload["group_assignment"]))
        payload["group_assignment"][first] = "value_head"
        path.write_text(json.dumps(payload))
    elif mutate == "metric":
        payload = json.loads(path.read_text())
        payload["recomputed_results"]["metrics"]["A"]["full_training_objective"] += 1
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        verify(relocated)


def test_production_decision_requires_every_fixed_gate() -> None:
    from ml.alphazero_lite.seed435_adam_direction import decide

    initial = {
        "full_training_objective": 1.0,
        "exposure_weighted": {"policy_ce": 1.0, "value_mse": 1.0},
        "equal_input": {"policy_ce": 1.0, "value_mse": 1.0},
    }
    control = {
        "full_training_objective": 0.9,
        "exposure_weighted": {"policy_ce": 1.0, "value_mse": 1.0},
        "equal_input": {"policy_ce": 1.0, "value_mse": 1.0},
    }
    treatment = {
        "full_training_objective": 0.8,
        "exposure_weighted": {"policy_ce": 0.98, "value_mse": 1.001},
        "equal_input": {"policy_ce": 0.98, "value_mse": 1.001},
    }
    assert decide({"initializer": initial, "A": control, "B": treatment})[
        "classification"
    ].startswith("advance")
    treatment["equal_input"]["policy_ce"] = 1.0
    assert (
        decide({"initializer": initial, "A": control, "B": treatment})["classification"]
        == "close_optimizer_direction_branch"
    )
