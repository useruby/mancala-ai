"""Focused tamper tests for the completed-amendment verifier."""

from __future__ import annotations

import json
import hashlib
import subprocess
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


def test_asset_entrypoint_ignores_recorded_paths_and_rejects_relocated_tablebase_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = {
        "weights": "weights.json",
        "runtime_search_policy": "search_policy.json",
        "initializer": "initializer.npz",
        "native_probe": "native_probe",
        "tablebase": "model-artifact/runtime/kalah_v1_21.kvtb",
    }
    registered = {}
    for name, relative in assets.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"relocated:{name}".encode()
        path.write_bytes(content)
        registered[name] = {
            "path": relative,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    original_source = Path(
        "/home/alex/Mancala/ai/model-artifact/runtime/kalah_v1_21.kvtb"
    )
    registered["tablebase"]["existing_local_source"] = original_source.as_posix()
    amendment = {
        "status": "frozen continuation amendment",
        "interpreter": {"path": str(verifier.runner.INTERPRETER)},
        "environment": verifier.current_environment(),
        "assets": registered,
    }
    amendment["interpreter"]["installed_packages"] = (
        verifier.current_installed_packages()
    )
    amendment_path = (
        tmp_path / "docs/data/seed542-selfplay-4096/continuation-amendment.json"
    )
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    amendment_path.write_text(json.dumps(amendment))
    amendment_hash = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    monkeypatch.setattr(verifier.runner, "AMENDMENT_SHA256", amendment_hash)

    original_read_bytes = Path.read_bytes

    def reject_original_source(path: Path) -> bytes:
        if path == original_source:
            raise AssertionError("verifier_attempted_historical_tablebase_path")
        return original_read_bytes(path)

    def reject_historical_interpreter(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("verifier_launched_recorded_interpreter")

    monkeypatch.setattr(Path, "read_bytes", reject_original_source)
    monkeypatch.setattr(subprocess, "check_output", reject_historical_interpreter)
    verifier.main(["--root", str(tmp_path), "--asset-check-only"])

    relocated_tablebase = tmp_path / assets["tablebase"]
    relocated_tablebase.write_bytes(b"altered relocated tablebase")
    with pytest.raises(
        ValueError,
        match="amended_runtime_asset_sha256_mismatch:tablebase",
    ):
        verifier.main(["--root", str(tmp_path), "--asset-check-only"])
