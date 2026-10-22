"""Fixture-only tests for seed518 runtime byte authentication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ml.alphazero_lite.seed518_selective_head import resolve_runtime_assets


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_assets(tmp_path: Path, local_table: bytes) -> tuple[dict, Path, Path, Path]:
    table_path = tmp_path / "model/tablebase.kvtb"
    table_path.parent.mkdir()
    table_path.write_bytes(local_table)
    obj_path = tmp_path / "cache/materialized.kvtb"
    obj_path.parent.mkdir()
    obj_path.write_bytes(b"verified object")
    probe = tmp_path / "runtime/probe"
    probe.parent.mkdir()
    probe.write_bytes(b"executable probe")
    registration = {
        "runtime_assets": {
            "exact_root_tablebase": {
                "artifact": "model/tablebase.kvtb",
                "sha256": digest(obj_path),
                "source_pointer_sha256": hashlib.sha256(
                    b"pointer representation"
                ).hexdigest(),
                "runtime_materialized_path": str(obj_path),
            },
            "exact_root_native_probe": {
                "artifact": "runtime/probe",
                "sha256": digest(probe),
            },
        }
    }
    return registration, table_path, obj_path, probe


def test_resolves_pointer_and_materialized_object(tmp_path: Path) -> None:
    registration, local, obj, probe = setup_assets(tmp_path, b"pointer representation")
    result = resolve_runtime_assets(registration, tmp_path)
    assert result["exact_root_tablebase"]["path"] == str(obj.resolve())
    assert result["exact_root_native_probe"]["path"] == str(probe.resolve())
    assert result["checked_out_tablebase"]["path"] == str(local.resolve())


def test_accepts_checked_out_object_representation(tmp_path: Path) -> None:
    registration, local, obj, _probe = setup_assets(tmp_path, b"pointer representation")
    local.write_bytes(obj.read_bytes())
    assert resolve_runtime_assets(registration, tmp_path)["exact_root_tablebase"][
        "path"
    ] == str(local.resolve())


@pytest.mark.parametrize("local", [b"corrupt", None])
def test_rejects_corrupt_or_missing_checked_out_tablebase(
    tmp_path: Path, local: bytes | None
) -> None:
    registration, path, _obj, _probe = setup_assets(tmp_path, b"pointer representation")
    if local is None:
        path.unlink()
    else:
        path.write_bytes(local)
    with pytest.raises(ValueError, match="tablebase_(missing|identity)"):
        resolve_runtime_assets(registration, tmp_path)


def test_rejects_corrupted_required_materialized_object(tmp_path: Path) -> None:
    registration, _local, obj, _probe = setup_assets(
        tmp_path, b"pointer representation"
    )
    obj.write_bytes(b"wrong object")
    with pytest.raises(ValueError, match="materialized_tablebase_identity"):
        resolve_runtime_assets(registration, tmp_path)


def test_rejects_incorrect_probe_bytes(tmp_path: Path) -> None:
    registration, _local, _obj, probe = setup_assets(
        tmp_path, b"pointer representation"
    )
    probe.write_bytes(b"wrong probe")
    with pytest.raises(ValueError, match="native_probe_identity"):
        resolve_runtime_assets(registration, tmp_path)
