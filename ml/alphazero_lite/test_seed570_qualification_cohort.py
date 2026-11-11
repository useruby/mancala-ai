"""Regression checks for deterministic qualification cohort filtering."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from unittest.mock import Mock, patch

import pytest

from ml.alphazero_lite.seed570_qualification_cohort import (
    ORACLE_PATH,
    QUALIFICATION_PATH,
    ROOT,
    SOURCE_SHA256,
    TRAINING_PATH,
    build,
    publish_membership,
    verify,
)


def fixture() -> tuple[dict, list[dict], list[dict]]:
    training = {"states": [{"opening_index": 1, "state": {"n": 1}}]}
    qualification = [
        {"opening_index": 1, "state": {"n": 2}},
        {"opening_index": 2, "state": {"n": 3}, "legal_absolute_actions": [0]},
    ]
    oracle = [
        {
            "state_hash": "h2",
            "legal_moves": [0],
            "action_margins": {"0": 1},
            "coverage_complete": True,
        },
        {
            "state_hash": "h3",
            "legal_moves": [0],
            "action_margins": {"0": 1},
            "coverage_complete": True,
        },
    ]
    return training, qualification, oracle


def patched_ids(canonical: dict, float32: dict) -> tuple:
    return (
        patch(
            "ml.alphazero_lite.seed570_qualification_cohort.canonical_identity",
            side_effect=lambda state: canonical[state["n"]],
        ),
        patch(
            "ml.alphazero_lite.seed570_qualification_cohort.production_float32_identity",
            side_effect=lambda state: float32[state["n"]],
        ),
        patch(
            "ml.alphazero_lite.seed570_qualification_cohort.canonical_hash",
            side_effect=lambda state: f"h{state['n']}",
        ),
    )


def mocked_rules():  # type: ignore[no-untyped-def]
    return patch(
        "ml.alphazero_lite.seed570_qualification_cohort.KalahGame.from_state",
        return_value=Mock(possible_moves=lambda: [0]),
    )


def test_shared_opening_is_rejected() -> None:
    training, qualification, oracle = fixture()
    ids = patched_ids({1: "t", 2: "a", 3: "b"}, {1: "tf", 2: "af", 3: "bf"})
    with ids[0], ids[1], ids[2], mocked_rules():
        result = build(training, qualification, oracle)
    assert result["rejections"]["shared_opening"] == 1
    assert [row["opening_index"] for row in result["roots"]] == [2]


@pytest.mark.parametrize("scheme", ["canonical", "float32"])
def test_collision_in_either_identity_scheme_excludes_root(scheme: str) -> None:
    training, qualification, oracle = fixture()
    canonical = {1: "training", 2: "candidate", 3: "candidate"}
    float32 = {1: "training-f", 2: "candidate-f", 3: "candidate-f"}
    if scheme == "canonical":
        canonical[3] = "training"
    else:
        float32[3] = "training-f"
    patches = patched_ids(canonical, float32)
    with patches[0], patches[1], patches[2], mocked_rules():
        result = build(training, qualification, oracle)
    assert result["rejections"][f"{scheme}_collision"] == 1


def test_incomplete_oracle_coverage_is_rejected() -> None:
    training, qualification, oracle = fixture()
    oracle[1]["action_margins"] = {}
    patches = patched_ids({1: "t", 2: "a", 3: "b"}, {1: "tf", 2: "af", 3: "bf"})
    with (
        patches[0],
        patches[1],
        patches[2],
        mocked_rules(),
        pytest.raises(ValueError, match="incomplete_oracle"),
    ):
        build(training, qualification, oracle)


def test_tampered_membership_is_rejected() -> None:
    membership_path = (
        ROOT
        / "docs/data/seed570-policy-prior-distillation/qualification-membership.json"
    )
    altered = deepcopy(json.loads(membership_path.read_bytes()))
    altered["roots"][0]["opening_index"] = 999
    with pytest.raises(ValueError, match="membership_mismatch"):
        verify(altered)


def test_public_verifier_rejects_matching_membership_from_tampered_source(
    tmp_path: Path,
) -> None:
    for relative_path in SOURCE_SHA256:
        source = ROOT / relative_path
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    training_path = tmp_path / TRAINING_PATH
    training = json.loads(training_path.read_bytes())
    training["source_tamper_marker"] = True
    training_path.write_text(json.dumps(training), encoding="utf-8")
    qualification = json.loads((tmp_path / QUALIFICATION_PATH).read_bytes())["states"]
    oracle = [
        json.loads(line)
        for line in (tmp_path / ORACLE_PATH).read_bytes().splitlines()
        if line
    ]
    matching_bundle = build(training, qualification, oracle)

    with pytest.raises(ValueError, match="source_binding_mismatch"):
        verify(matching_bundle, root=tmp_path)


def test_publisher_preserves_existing_destination_bytes_and_mtime(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "qualification-membership.json"
    original_bytes = b"existing frozen membership\n"
    destination.write_bytes(original_bytes)
    original_mtime_ns = 1_700_000_000_123_456_789
    os.utime(destination, ns=(original_mtime_ns, original_mtime_ns))

    with pytest.raises(FileExistsError):
        publish_membership(ROOT, destination)

    assert destination.read_bytes() == original_bytes
    assert destination.stat().st_mtime_ns == original_mtime_ns
