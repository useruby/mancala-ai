"""Focused tests for the read-only seed520 manifest contract."""

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite.seed520_composition_manifest import (
    CompositionManifestError,
    _validate_metadata,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/data/seed520-composition/manifest.json"
RUNTIME_MANIFEST = ROOT / "docs/data/seed520-composition/manifest-v2.json"


def test_published_manifest_authenticates_both_components() -> None:
    result = validate_manifest(RUNTIME_MANIFEST)
    assert set(result["components"]) == {"policy", "value"}
    assert result["components"]["policy"]["files"]["weights.json"].endswith(
        "weights.json"
    )
    assert result["components"]["policy"]["checkpoint"].endswith("seed504_epoch4.npz")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "future", "manifest_schema_unsupported"),
        ("encoding", "other", "component_incompatible"),
    ],
)
def test_manifest_rejects_incompatible_contract(tmp_path, field, value, error) -> None:
    data = json.loads(RUNTIME_MANIFEST.read_text())
    data[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CompositionManifestError, match=error):
        validate_manifest(path)


def test_manifest_rejects_component_substitution(tmp_path) -> None:
    data = json.loads(RUNTIME_MANIFEST.read_text())
    data["components"]["policy"]["source"] = "value"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CompositionManifestError, match="source_routing_mismatch"):
        validate_manifest(path)


def test_checkpoint_manifest_is_explicitly_not_runtime_consumable() -> None:
    with pytest.raises(CompositionManifestError, match="checkpoint_only_proposal"):
        validate_manifest(MANIFEST)


def test_seed512_registration_is_authenticated_before_use(tmp_path) -> None:
    registration = (
        ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    copied = (
        tmp_path / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    copied.parent.mkdir(parents=True)
    shutil.copy2(registration, copied)
    copied.write_bytes(copied.read_bytes() + b"changed")
    with pytest.raises(
        CompositionManifestError, match="seed512_registration_hash_mismatch"
    ):
        validate_manifest(
            RUNTIME_MANIFEST,
            repository_root=tmp_path,
            asset_root=ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("input_encoding", "other", "component_metadata_incompatible"),
        ("feature_count", 26, "component_metadata_incompatible"),
        ("architecture", {"model_type": "other"}, "component_metadata_incompatible"),
        ("unexpected_extension", True, "component_metadata_extensions_unsupported"),
    ],
)
def test_authenticated_metadata_contract_rejects_incompatibility(
    field, value, error
) -> None:
    metadata = json.loads(
        (
            ROOT
            / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts/E4/metadata.json"
        ).read_text()
    )
    if field == "unexpected_extension":
        metadata["artifacts"]["unexpected_extension"] = value
    else:
        metadata[field] = value
    with pytest.raises(CompositionManifestError, match=error):
        _validate_metadata(metadata)


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        (name, mutation)
        for name in ("model.npz", "weights.json", "metadata.json", "search_policy.json")
        for mutation in ("altered", "missing")
    ],
)
def test_runtime_component_files_are_byte_authenticated(
    tmp_path, filename, mutation
) -> None:
    data = json.loads(RUNTIME_MANIFEST.read_text())
    root = tmp_path / "root"
    path = root / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    source_root = ROOT / data["components"]["policy"]["artifact_directory"]
    destination = root / data["components"]["policy"]["artifact_directory"]
    shutil.copytree(source_root, destination)
    source = ROOT / data["components"]["value"]["artifact_directory"]
    shutil.copytree(source, root / data["components"]["value"]["artifact_directory"])
    registration = (
        ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    registration_target = (
        root / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    registration_target.parent.mkdir(parents=True)
    shutil.copy2(registration, registration_target)
    native = root / "model-artifact/runtime/kalah_v1_tablebase"
    native.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "model-artifact/runtime/kalah_v1_tablebase", native)
    (native.parent / "kalah_v1_21.kvtb").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7\n"
        "size 709635072\n"
    )
    component_file = destination / filename
    if mutation == "missing":
        component_file.unlink()
    else:
        component_file.write_bytes(component_file.read_bytes() + b"altered")
    with pytest.raises(
        CompositionManifestError, match=f"component_file_mismatch:policy:{filename}"
    ):
        validate_manifest(path, repository_root=root, asset_root=ROOT)


def test_golden_fixture_bundle_verifies() -> None:
    from ml.alphazero_lite.seed521_golden_fixtures import verify

    verify()


def test_temporary_lfs_pointer_and_probe_bytes_are_authenticated(tmp_path) -> None:
    data = json.loads(RUNTIME_MANIFEST.read_text())
    root = tmp_path / "repo"
    root.mkdir()
    manifest = root / "manifest-v2.json"
    manifest.write_text(json.dumps(data))
    for name in ("E4", "initializer"):
        source = (
            ROOT
            / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"
            / name
        )
        shutil.copytree(
            source,
            root
            / "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts"
            / name,
        )
    registration = (
        ROOT / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    registration_target = (
        root / "docs/data/seed512-fixed-composition-evaluation/registration.json"
    )
    registration_target.parent.mkdir(parents=True)
    shutil.copy2(registration, registration_target)
    runtime_dir = root / "model-artifact/runtime"
    runtime_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "model-artifact/runtime/kalah_v1_tablebase", runtime_dir)
    (runtime_dir / "kalah_v1_21.kvtb").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7\n"
        "size 709635072\n"
    )
    result = validate_manifest(manifest, repository_root=root, asset_root=ROOT)
    assert result["runtime"]["exact_root_tablebase"] == str(
        ROOT / data["runtime"]["exact_root_tablebase"]["materialized_path"]
    )
    tablebase = runtime_dir / "kalah_v1_21.kvtb"
    pointer = tablebase.read_bytes()
    tablebase.write_bytes(pointer + b"corrupt")
    with pytest.raises(
        CompositionManifestError, match="runtime_pointer_or_object_corrupt"
    ):
        validate_manifest(manifest, repository_root=root, asset_root=ROOT)
    tablebase.write_bytes(pointer)
    probe = runtime_dir / "kalah_v1_tablebase"
    probe.write_bytes(probe.read_bytes() + b"corrupt")
    with pytest.raises(
        CompositionManifestError, match="runtime_pointer_or_object_corrupt"
    ):
        validate_manifest(manifest, repository_root=root, asset_root=ROOT)


@pytest.mark.parametrize("mutation", ["altered", "missing", "duplicate"])
def test_golden_fixture_mutations_are_rejected(tmp_path, mutation) -> None:
    from ml.alphazero_lite.seed521_golden_fixtures import sha256, verify

    source = ROOT / "docs/data/seed522-golden-fixtures"
    fixture_dir = tmp_path / "fixtures"
    shutil.copytree(source, fixture_dir)
    fixture_path = fixture_dir / "fixtures.json"
    data = json.loads(fixture_path.read_text())
    if mutation == "altered":
        data["cases"][0]["policy_f32_hex"] = "00" * 24
    elif mutation == "missing":
        data["cases"].pop()
        data["case_count"] -= 1
    else:
        data["cases"][1]["case_id"] = data["cases"][0]["case_id"]
    fixture_path.write_text(json.dumps(data))
    index_path = fixture_dir / "index.json"
    index = json.loads(index_path.read_text())
    index["fixtures"]["sha256"] = sha256(fixture_path)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="seed522_"):
        verify(fixture_dir=fixture_dir)
