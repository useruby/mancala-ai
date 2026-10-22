"""Read-only validator for the proposed opt-in seed520 composition manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "azlite_composition_manifest_v1"
EXPECTED = {
    "policy": (
        "docs/data/seed504/seed504_epoch4.npz",
        "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    ),
    "value": (
        "docs/data/seed505-strength-screen-v3/initializer.npz",
        "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
    ),
}
SIDECAR = "docs/data/seed511-fixed-composition-screen-v4/runtime-artifacts/E4/search_policy.json"
SIDECAR_SHA256 = "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d"


class CompositionManifestError(ValueError):
    """Manifest or referenced immutable component failed validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate identities and contract without modifying or substituting files.

    Pointer files are accepted only as plain relative path text; content is
    resolved and authenticated at the referenced path. Symlinks are resolved
    before enforcing repository containment.
    """
    manifest_path = Path(path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompositionManifestError("manifest_invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CompositionManifestError("manifest_schema_unsupported")
    if (
        manifest.get("encoding") != "kalah_v3"
        or manifest.get("architecture") != "residual_v3"
    ):
        raise CompositionManifestError("component_incompatible")
    if manifest.get("search_policy") != {
        "simulations": 1200,
        "workers": 1,
        "c_puct": 1.25,
        "exact_root_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "solver_identity": "native_kvtb_root_action_probe_v1",
    }:
        raise CompositionManifestError("search_policy_mismatch")
    components = manifest.get("components")
    if not isinstance(components, dict) or set(components) != set(EXPECTED):
        raise CompositionManifestError("component_set_invalid")
    root = manifest_path.parents[3]
    validated: dict[str, str] = {}
    for name, (expected_path, expected_hash) in EXPECTED.items():
        component = components[name]
        if not isinstance(component, dict):
            raise CompositionManifestError("component_invalid")
        reference = component.get("path")
        if not isinstance(reference, str) or Path(reference).suffix.lower() != ".npz":
            raise CompositionManifestError("component_extension_unsupported")
        if reference != expected_path or component.get("sha256") != expected_hash:
            raise CompositionManifestError("component_identity_mismatch")
        resolved = (root / reference).resolve()
        if (
            root not in resolved.parents
            or not resolved.is_file()
            or _sha256(resolved) != expected_hash
        ):
            raise CompositionManifestError("component_bytes_mismatch")
        if component.get("source") != ("policy" if name == "policy" else "value"):
            raise CompositionManifestError("source_routing_mismatch")
        if component.get("trunk") != "owned_by_component":
            raise CompositionManifestError("trunk_ownership_invalid")
        validated[name] = str(resolved)
    runtime = manifest.get("runtime")
    expected_runtime = {
        "search_policy_path": SIDECAR,
        "native_probe_path": "model-artifact/runtime/kalah_v1_tablebase",
        "native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        "tablebase_path": "model-artifact/runtime/kalah_v1_21.kvtb",
        "tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
        "native_probe_identity": "native_kvtb_root_action_probe_v1",
        "tablebase_identity": "seed512_registered_tablebase",
    }
    if runtime != expected_runtime or not isinstance(runtime, dict):
        raise CompositionManifestError("runtime_identity_mismatch")
    sidecar = (root / SIDECAR).resolve()
    if not sidecar.is_file() or _sha256(sidecar) != SIDECAR_SHA256:
        raise CompositionManifestError("runtime_sidecar_mismatch")
    for path_key, hash_key in (
        ("native_probe_path", "native_probe_sha256"),
        ("tablebase_path", "tablebase_sha256"),
    ):
        runtime_path = (root / runtime[path_key]).resolve()
        if (
            root not in runtime_path.parents
            or not runtime_path.is_file()
            or _sha256(runtime_path) != runtime[hash_key]
        ):
            raise CompositionManifestError("runtime_artifact_mismatch")
    return {"schema": SCHEMA, "components": validated}
