"""Tests for the public seed398 publication verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed398_publication as verifier


def test_publication_reproduces_registered_4096_game_result() -> None:
    result = verifier.verify_publication()
    assert result["status"] == "verified"
    assert result["games"] == 4096
    assert result["primary_97_5_intervals"]["policy_FS_minus_SS"] == [
        -0.03125,
        0.016607666015625355,
    ]
    assert result["primary_97_5_intervals"]["value_SF_minus_SS"] == [
        -0.02294921875,
        0.025390625,
    ]


def test_command_rejects_changed_value_source_when_policy_is_unchanged() -> None:
    binding = json.loads((verifier.DATA / "evaluation-binding.json").read_text())
    registration = json.loads((verifier.DATA / "registration.json").read_text())
    treatment = binding["treatments"]["FS"]
    command = list(binding["reports"]["FS"]["command"])
    value_flag = command.index("--challenger-value-artifact")
    command[value_flag + 1] = binding["treatments"]["FF"]["value_artifact"]
    assert (
        command[command.index("--challenger-policy-artifact") + 1]
        == treatment["policy_artifact"]
    )
    with pytest.raises(
        ValueError, match="command_identity_mismatch:FS:--challenger-value-artifact"
    ):
        verifier.verify_command(
            "FS",
            command,
            treatment,
            registration["component_sources"]["seed455"]["artifact"],
        )


def test_registration_binding_hash_mutation_is_rejected(tmp_path: Path) -> None:
    registration_path = verifier.DATA / "registration.json"
    registration = json.loads(registration_path.read_text())
    binding = json.loads((verifier.DATA / "evaluation-binding.json").read_text())
    mutated = tmp_path / "registration.json"
    mutated.write_bytes(registration_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding_registration_mismatch"):
        verifier.verify_binding_identity(registration, binding, mutated)


@pytest.mark.parametrize(
    ("relative_path", "mutate"),
    [
        ("original-arena-reports/FF.json", "bytes"),
        ("validated-outcome-ledger.jsonl", "bytes"),
        ("four-treatment-opening-score-matrix.json", "matrix"),
        ("analysis.json", "followup"),
    ],
)
def test_publication_rejects_altered_published_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, mutate: str
) -> None:
    copy = tmp_path / "publication"
    shutil.copytree(verifier.DATA, copy)
    target = copy / relative_path
    if mutate == "bytes":
        target.write_bytes(target.read_bytes() + b" ")
    else:
        value = json.loads(target.read_text())
        if mutate == "matrix":
            value["opening_scores"][0]["FF"] = 0.75
        else:
            value["decision"]["value_follow_up"] = True
        target.write_text(json.dumps(value))
    monkeypatch.setattr(verifier, "DATA", copy)
    with pytest.raises(ValueError):
        verifier.verify_publication()
