"""Release assembler validation tests using disposable local source assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from deploy.assemble_release import artifact_identity, assemble


NATIVE_SOURCE = (
    "#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps({'actions': {}}))\n"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(root: Path) -> tuple[Path, list[dict]]:
    files = {
        "model-artifact/current/weights.json": ("weight", "{}"),
        "model-artifact/current/metadata.json": ("metadata", '{"artifacts":{}}'),
        "model-artifact/current/search_policy.json": (
            "policy",
            json.dumps(
                {
                    "native_probe": {
                        "path": "model-artifact/runtime/probe",
                        "sha256": hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest(),
                    },
                    "tablebase": {
                        "path": "model-artifact/runtime/tablebase",
                        "sha256": hashlib.sha256(b"tablebase test object").hexdigest(),
                    },
                }
            ),
        ),
        "model-artifact/runtime/probe": ("native_probe", NATIVE_SOURCE),
        "model-artifact/runtime/tablebase": ("tablebase", "tablebase test object"),
    }
    entries = []
    for relative, (kind, contents) in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        if kind == "native_probe":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        entries.append(
            {
                "source": relative,
                "destination": relative,
                "kind": kind,
                "sha256": sha(path),
            }
        )
    artifact = root / "model-artifact/current"
    policy = artifact / "search_policy.json"
    config = {
        "asset_root": "/assets",
        "worker_count": 1,
        "startup_timeout": 60,
        "execution_timeout": 10,
        "shutdown_timeout": 10,
        "bundles": {
            "current": {
                "bundle_id": "fixture-current",
                "kind": "single",
                "artifacts": ["model-artifact/current"],
                "expected_artifact_sha256": [artifact_identity(artifact)],
                "expected_policy_sha256": [sha(policy)],
                "policy_source": "current",
                "value_source": "current",
            }
        },
        "profiles": {
            "balanced@v1": {
                "name": "balanced@v1",
                "profile_id": "balanced@v1",
                "bundle_id": "current",
                "require_runtime_search_policy": True,
                "settings": {
                    "simulations": 1,
                    "c_puct": 1.25,
                    "fpu_mode": "zero",
                    "reuse_subtree": False,
                    "normalize_values": False,
                    "root_policy_mode": "deterministic",
                    "tactical_root_bias": 0.0,
                },
            }
        },
    }
    config_path = root / "service.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    entries.append(
        {
            "source": "service.json",
            "destination": "service.json",
            "kind": "config",
            "sha256": sha(config_path),
        }
    )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return manifest, entries


def test_assemble_valid_release_and_retain_config_paths(tmp_path: Path) -> None:
    manifest, _ = build_fixture(tmp_path)
    release = tmp_path / "release-1"
    assemble(manifest, release)
    config = json.loads((release / "service.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (release / "model-artifact/current/search_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["asset_root"] == "/assets"
    assert (
        release / config["bundles"]["current"]["artifacts"][0] / "weights.json"
    ).is_file()
    assert (release / policy["native_probe"]["path"]).is_file()
    assert (release / policy["tablebase"]["path"]).is_file()
    assert not (release / "service.json").stat().st_mode & stat.S_IWUSR


@pytest.mark.parametrize("mutation", ["missing", "hash", "lfs", "escape", "native"])
def test_invalid_release_is_rejected_without_publishing(
    tmp_path: Path, mutation: str
) -> None:
    manifest, entries = build_fixture(tmp_path)
    if mutation == "missing":
        entries[0]["source"] = "does-not-exist"
    elif mutation == "hash":
        entries[0]["sha256"] = "0" * 64
    elif mutation == "lfs":
        target = tmp_path / entries[0]["source"]
        target.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:"
            + "0" * 64
            + "\nsize 100000\n",
            encoding="utf-8",
        )
        entries[0]["sha256"] = sha(target)
    elif mutation == "escape":
        entries[0]["destination"] = "../outside"
    else:
        target = tmp_path / "model-artifact/runtime/probe"
        target.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
        probe_entry = next(
            entry for entry in entries if entry["kind"] == "native_probe"
        )
        probe_entry["sha256"] = sha(target)
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    destination = tmp_path / "rejected-release"
    with pytest.raises((ValueError, OSError)):
        assemble(manifest, destination)
    assert not destination.exists()


def test_missing_authenticated_configuration_is_rejected(tmp_path: Path) -> None:
    manifest, entries = build_fixture(tmp_path)
    config = tmp_path / "service.json"
    value = json.loads(config.read_text(encoding="utf-8"))
    value["bundles"]["current"]["expected_artifact_sha256"] = ["0" * 64]
    config.write_text(json.dumps(value), encoding="utf-8")
    config_entry = next(entry for entry in entries if entry["kind"] == "config")
    config_entry["sha256"] = sha(config)
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    destination = tmp_path / "bad-config-release"
    with pytest.raises(ValueError, match="artifact identity"):
        assemble(manifest, destination)
    assert not destination.exists()


def test_implicit_search_setting_is_rejected(tmp_path: Path) -> None:
    manifest, entries = build_fixture(tmp_path)
    config = tmp_path / "service.json"
    value = json.loads(config.read_text(encoding="utf-8"))
    del value["profiles"]["balanced@v1"]["settings"]["c_puct"]
    config.write_text(json.dumps(value), encoding="utf-8")
    config_entry = next(entry for entry in entries if entry["kind"] == "config")
    config_entry["sha256"] = sha(config)
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    destination = tmp_path / "implicit-search-release"
    with pytest.raises(ValueError, match="every search setting"):
        assemble(manifest, destination)
    assert not destination.exists()


def test_actual_release_manifest_authenticates_all_profiles_and_composed_assets(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "balanced-release"
    assemble(root / "deploy/release-manifest.json", destination)
    config = json.loads((destination / "service.json").read_text(encoding="utf-8"))
    bundle = config["bundles"]["policy-value-p"]
    profile = config["profiles"]["balanced@v1"]
    assert bundle["kind"] == "composed"
    assert bundle["policy_source"] == "candidate"
    assert bundle["value_source"] == "current"
    assert profile["settings"]["simulations"] == 1200
    expected_budgets = {
        "easy@v1": 64,
        "medium@v1": 256,
        "hard@v1": 640,
        "expert@v1": 1200,
        "balanced@v1": 1200,
    }
    for name, simulations in expected_budgets.items():
        resolved = config["profiles"][name]
        assert resolved["settings"]["simulations"] == simulations
        assert resolved["bundle_id"] == "policy-value-p"
        assert resolved["settings"] == profile["settings"] | {
            "simulations": simulations
        }
    assert config["execution_timeout"] == 10
    assert (destination / "release-manifest.json").is_file()
