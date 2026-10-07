"""Semantic regression tests for the seed436 telemetry correction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from ml.alphazero_lite.verify_seed436_direction_telemetry import (
    evidence_bindings,
    validate_semantics,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_REL = "docs/data/seed435-adam-direction-screen"


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    shutil.copytree(ROOT / DATA_REL, root / DATA_REL)
    return root


def _edit_json(root: Path, name: str, edit) -> None:
    path = root / DATA_REL / name
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_valid_evidence_passes(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    assert validate_semantics(root)["paired_updates"] == 16


@pytest.mark.parametrize("arm", ["A", "B"])
def test_incorrect_arm_cosine_fails(tmp_path: Path, arm: str) -> None:
    root = _fixture(tmp_path)
    _edit_json(
        root,
        "post-execution-telemetry.json",
        lambda data: data["arms"][arm][0].__setitem__(
            "direction_cosine_vs_paired_arm", 0.123
        ),
    )
    with pytest.raises(ValueError, match="paired_cosine_mismatch"):
        validate_semantics(root)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("cosine_archive", "cosine_archive_mismatch"),
        ("group", "group_assignment_architecture_mismatch"),
        ("overlap", "parameter_layout_architecture_mismatch"),
        ("incomplete", "parameter_layout_architecture_mismatch"),
        ("group_norm", "group_norm_mismatch"),
        ("update_norm", "update_vector_norm_mismatch"),
        ("undefined", "cosine_undefined_flag_mismatch"),
        ("nonfinite", "nonfinite_update_vector"),
    ],
)
def test_semantic_corruption_fails(tmp_path: Path, mutation: str, error: str) -> None:
    root = _fixture(tmp_path)
    data = root / DATA_REL
    if mutation == "cosine_archive":
        with np.load(data / "A-B-audit-delta-cosines.npz") as archive:
            cosines = archive["cosines"]
        cosines[0] += 0.1
        np.savez_compressed(data / "A-B-audit-delta-cosines.npz", cosines=cosines)
    elif mutation == "group":
        _edit_json(
            root,
            "post-execution-telemetry.json",
            lambda p: p["group_assignment"].__setitem__(
                next(iter(p["group_assignment"])), "value_head"
            ),
        )
    elif mutation == "overlap":
        _edit_json(
            root,
            "post-execution-telemetry.json",
            lambda p: p["parameter_layout"][1].__setitem__(
                "start", p["parameter_layout"][1]["start"] - 1
            ),
        )
    elif mutation == "incomplete":
        _edit_json(
            root, "post-execution-telemetry.json", lambda p: p["parameter_layout"].pop()
        )
    elif mutation == "group_norm":
        _edit_json(
            root,
            "post-execution-telemetry.json",
            lambda p: p["arms"]["A"][0]["group_norms"].__setitem__(
                "shared_trunk", 42.0
            ),
        )
    elif mutation == "update_norm":
        _edit_json(
            root,
            "post-execution-telemetry.json",
            lambda p: p["arms"]["A"][0].__setitem__("update_norm", 42.0),
        )
    elif mutation == "undefined":
        _edit_json(
            root,
            "post-execution-telemetry.json",
            lambda p: p["arms"]["A"][0].__setitem__(
                "cosine_undefined_zero_vector", True
            ),
        )
    else:
        with np.load(data / "A-audit-deltas.npz") as archive:
            vectors = archive["deltas"]
        vectors[0, 0] = np.inf
        np.savez_compressed(data / "A-audit-deltas.npz", deltas=vectors)
    with pytest.raises(ValueError, match=error):
        validate_semantics(root)


def test_zero_vector_cosine_fixture_and_inconsistent_undefined_evidence(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    data = root / DATA_REL
    for arm in ("A", "B"):
        with np.load(data / f"{arm}-audit-deltas.npz") as archive:
            vectors = archive["deltas"]
        vectors[0] = 0
        np.savez_compressed(data / f"{arm}-audit-deltas.npz", deltas=vectors)
    _edit_json(
        root,
        "post-execution-telemetry.json",
        lambda p: [
            row.update(
                direction_cosine_vs_paired_arm=None,
                cosine_undefined_zero_vector=True,
                update_norm=0.0,
                delta_vector_norm=0.0,
                group_norms={
                    group: 0.0
                    for group in ("shared_trunk", "policy_head", "value_head")
                },
            )
            for arm in ("A", "B")
            for row in [p["arms"][arm][0]]
        ],
    )
    with np.load(data / "A-B-audit-delta-cosines.npz") as archive:
        cosines = archive["cosines"]
    cosines[0] = np.nan
    np.savez_compressed(data / "A-B-audit-delta-cosines.npz", cosines=cosines)
    # Match the synthetic zero deltas to every original/telemetry radius record.
    for arm in ("A", "B"):
        path = data / f"{arm}-updates.json"
        updates = json.loads(path.read_text())
        updates[0]["stored_delta_norm"] = 0.0
        if arm == "A":
            updates[0]["requested_radius"] = 0.0
        else:
            updates[0]["requested_radius"] = 0.0
        path.write_text(json.dumps(updates))
    assert validate_semantics(root)["paired_updates"] == 16
    _edit_json(
        root,
        "post-execution-telemetry.json",
        lambda p: p["arms"]["B"][0].__setitem__("direction_cosine_vs_paired_arm", 0.0),
    )
    with pytest.raises(ValueError, match="zero_vector_cosine_not_undefined"):
        validate_semantics(root)


def test_zero_vector_with_positive_radius_fails(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    data = root / DATA_REL
    for arm in ("A", "B"):
        with np.load(data / f"{arm}-audit-deltas.npz") as archive:
            vectors = archive["deltas"]
        vectors[0] = 0
        np.savez_compressed(data / f"{arm}-audit-deltas.npz", deltas=vectors)
    _edit_json(
        root,
        "post-execution-telemetry.json",
        lambda p: [
            p["arms"][arm][0].update(
                direction_cosine_vs_paired_arm=None,
                cosine_undefined_zero_vector=True,
                update_norm=0.0,
                delta_vector_norm=0.0,
                group_norms={
                    group: 0.0
                    for group in ("shared_trunk", "policy_head", "value_head")
                },
            )
            for arm in ("A", "B")
        ],
    )
    with np.load(data / "A-B-audit-delta-cosines.npz") as archive:
        cosines = archive["cosines"]
    cosines[0] = np.nan
    np.savez_compressed(data / "A-B-audit-delta-cosines.npz", cosines=cosines)
    with pytest.raises(ValueError, match="original_update_norm_mismatch"):
        validate_semantics(root)


def test_relocated_cli_is_read_only_from_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    from ml.alphazero_lite.test_seed435_publication import _copy_publication

    before_original = evidence_bindings(ROOT)
    relocated = _copy_publication(tmp_path)
    receipt_relative = (
        "docs/data/seed436-direction-telemetry-verification/correction-receipt.json"
    )
    receipt_target = relocated / receipt_relative
    receipt_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / receipt_relative, receipt_target)
    before_relocated = evidence_bindings(relocated)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed436_direction_telemetry",
            "--root",
            str(relocated),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(relocated)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        json.loads(result.stdout)["semantic_verification"]["classification"]
        == "close_optimizer_direction_branch"
    )
    assert evidence_bindings(ROOT) == before_original
    assert evidence_bindings(relocated) == before_relocated
