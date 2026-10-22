"""Focused tests for the read-only seed520 manifest contract."""

import json
from pathlib import Path

import pytest

from ml.alphazero_lite.seed520_composition_manifest import (
    CompositionManifestError,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/data/seed520-composition/manifest.json"


def test_published_manifest_authenticates_both_components() -> None:
    result = validate_manifest(MANIFEST)
    assert set(result["components"]) == {"policy", "value"}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "future", "manifest_schema_unsupported"),
        ("encoding", "other", "component_incompatible"),
    ],
)
def test_manifest_rejects_incompatible_contract(tmp_path, field, value, error) -> None:
    data = json.loads(MANIFEST.read_text())
    data[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CompositionManifestError, match=error):
        validate_manifest(path)


def test_manifest_rejects_component_substitution(tmp_path) -> None:
    data = json.loads(MANIFEST.read_text())
    data["components"]["policy"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(CompositionManifestError, match="component_identity_mismatch"):
        validate_manifest(path)
