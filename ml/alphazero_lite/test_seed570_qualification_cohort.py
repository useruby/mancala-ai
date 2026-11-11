"""Regression checks for deterministic qualification cohort filtering."""

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from ml.alphazero_lite.seed570_qualification_cohort import build, verify


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
        verify(result, training, qualification, oracle)
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
        verify(result, training, qualification, oracle)
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
        verify({}, training, qualification, oracle)


def test_tampered_membership_is_rejected() -> None:
    training, qualification, oracle = fixture()
    patches = patched_ids({1: "t", 2: "a", 3: "b"}, {1: "tf", 2: "af", 3: "bf"})
    with patches[0], patches[1], patches[2], mocked_rules():
        bundle = build(training, qualification, oracle)
        altered = deepcopy(bundle)
        altered["roots"][0]["opening_index"] = 999
        with pytest.raises(ValueError, match="membership_mismatch"):
            verify(altered, training, qualification, oracle)
