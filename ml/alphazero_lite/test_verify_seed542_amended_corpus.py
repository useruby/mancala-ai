"""Focused tamper tests for the completed-amendment verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import verify_seed542_amended_corpus as verifier


@pytest.mark.parametrize(
    ("filename", "key"),
    [
        ("amended-accounting.json", "games"),
        ("development-membership.json", "included_rows"),
        ("training-protocol.json", "schedule_sha256"),
    ],
)
def test_published_amended_artifacts_reject_mutation(
    tmp_path: Path, filename: str, key: str
) -> None:
    rebuilt = {
        "accounting.json": {"games": 4096},
        "development-membership.json": {"included_rows": 10},
        "training-protocol.json": {"schedule_sha256": "expected"},
    }
    names = {
        "accounting.json": "amended-accounting.json",
        "development-membership.json": "development-membership.json",
        "training-protocol.json": "training-protocol.json",
    }
    for source_name, published_name in names.items():
        (tmp_path / published_name).write_text(json.dumps(rebuilt[source_name]))
    verified = verifier.compare_published_artifacts(tmp_path, rebuilt)
    assert set(verified) == set(names.values())

    bad = json.loads((tmp_path / filename).read_text())
    bad[key] = 1 if key != "schedule_sha256" else "altered"
    (tmp_path / filename).write_text(json.dumps(bad))
    with pytest.raises(
        ValueError, match=f"amended_published_artifact_mismatch:{filename}"
    ):
        verifier.compare_published_artifacts(tmp_path, rebuilt)


def test_recorded_runtime_authentication_rejects_dependency_and_version_drift() -> None:
    environment = {
        "python": "3.14.7",
        "platform": "test-platform",
        "numpy": "2.5.3",
        "torch": "2.14.0+cu130",
    }
    amendment = {
        "environment": environment,
        "interpreter": {"installed_packages": ["numpy==2.5.3", "torch==2.14.0+cu130"]},
    }
    verifier.verify_recorded_environment(
        amendment, environment, amendment["interpreter"]["installed_packages"]
    )
    with pytest.raises(ValueError, match="amended_runtime_environment_mismatch"):
        verifier.verify_recorded_environment(
            amendment,
            {**environment, "numpy": "2.5.2"},
            amendment["interpreter"]["installed_packages"],
        )
    with pytest.raises(ValueError, match="amended_runtime_dependency_set_mismatch"):
        verifier.verify_recorded_environment(amendment, environment, ["numpy==2.5.2"])
