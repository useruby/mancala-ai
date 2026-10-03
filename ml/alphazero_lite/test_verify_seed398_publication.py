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
            registration,
        )


@pytest.mark.parametrize(
    "flag", ["--challenger-policy-artifact", "--opening-prefixes-jsonl"]
)
def test_command_rejects_changed_historical_identity(flag: str) -> None:
    binding = json.loads((verifier.DATA / "evaluation-binding.json").read_text())
    registration = json.loads((verifier.DATA / "registration.json").read_text())
    command = list(binding["reports"]["FF"]["command"])
    command[command.index(flag) + 1] += ".changed"
    with pytest.raises(ValueError, match="command_identity_mismatch:FF"):
        verifier.verify_command(
            "FF",
            command,
            registration["treatments"]["FF"],
            registration["component_sources"]["seed455"]["artifact"],
            registration,
        )


@pytest.mark.parametrize("directory", ["copy-one", "nested/copy-two"])
def test_clean_publication_verifies_from_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str
) -> None:
    copy = tmp_path / directory
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(verifier.DATA, copy)
    monkeypatch.setattr(verifier, "DATA", copy)
    assert verifier.verify_publication()["status"] == "verified"


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
    monkeypatch.setattr(verifier, "DATA", copy)
    assert verifier.verify_publication()["status"] == "verified"
    target = copy / relative_path
    if mutate == "bytes":
        payload = target.read_bytes()
        if relative_path == "validated-outcome-ledger.jsonl":
            payload = payload.replace(b"\n", b" \n", 1)
        else:
            payload += b" "
        target.write_bytes(payload)
    else:
        value = json.loads(target.read_text())
        if mutate == "matrix":
            value["opening_scores"][0]["FF"] = 0.75
        else:
            value["decision"]["value_follow_up"] = True
        target.write_text(json.dumps(value))
    expected = {
        "original-arena-reports/FF.json": "report_hash_mismatch:FF",
        "validated-outcome-ledger.jsonl": "ledger_hash_mismatch",
        "four-treatment-opening-score-matrix.json": "analysis_matrix_hash_mismatch",
        "analysis.json": "published_analysis_derivative_mismatch",
    }[relative_path]
    with pytest.raises(ValueError, match=expected):
        verifier.verify_publication()


@pytest.mark.parametrize(
    "identity",
    ["registration_sha256", "evaluation_binding_sha256", "outcome_ledger_sha256"],
)
def test_matrix_identity_headers_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identity: str
) -> None:
    copy = tmp_path / "publication"
    shutil.copytree(verifier.DATA, copy)
    monkeypatch.setattr(verifier, "DATA", copy)
    assert verifier.verify_publication()["status"] == "verified"
    path = copy / "four-treatment-opening-score-matrix.json"
    matrix = json.loads(path.read_text())
    matrix[identity] += "x"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match=f"matrix_identity_mismatch:{identity}"):
        verifier.verify_publication()
