"""Read-only validation for the proposed, opt-in seed520 composition contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "azlite_composition_manifest_v1"
RUNTIME_SCHEMA = "azlite_runtime_composition_manifest_v1"
REGISTRATION = "docs/data/seed512-fixed-composition-evaluation/registration.json"
REGISTRATION_SHA256 = "44d41e9d5bc78654a8713d5e9027cb3071b4484a2042308be0591d8aafac5309"
RUNTIME_FILES = ("model.npz", "weights.json", "metadata.json", "search_policy.json")
COMPONENTS = {
    "policy": ("E4", "policy"),
    "value": ("initializer", "value"),
}
EXPECTED_RUNTIME = {
    "exact_root_native_probe": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
    "exact_root_tablebase": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
}


class CompositionManifestError(ValueError):
    """Manifest or referenced immutable component failed validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompositionManifestError("json_artifact_invalid") from error
    if not isinstance(result, dict):
        raise CompositionManifestError("json_artifact_invalid")
    return result


def _resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if root not in candidate.parents:
        raise CompositionManifestError("asset_path_escapes_root")
    return candidate


def _validate_metadata(metadata: dict[str, Any]) -> None:
    architecture = metadata.get("architecture")
    if (
        metadata.get("input_encoding") != "kalah_v3"
        or metadata.get("feature_count") != 27
        or metadata.get("policy_size") != 6
        or not isinstance(architecture, dict)
        or architecture.get("model_type") != "residual_v3"
        or architecture.get("type") != "residual_policy_value"
        or architecture.get("policy_size") != 6
        or architecture.get("value_size") != 1
        or architecture.get("trunk_size") != 96
        or architecture.get("hidden_sizes") != [96]
        or architecture.get("residual_block_count") != 3
        or architecture.get("hidden_layer_count") != 9
        or architecture.get("policy_hidden_size") != 96
        or architecture.get("value_hidden_size") != 48
        or architecture.get("activation") != "relu"
    ):
        raise CompositionManifestError("component_metadata_incompatible")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) - {
        "weights_file",
        "weights_fallback_file",
        "weights_sha256",
        "weights_json_sha256",
    }:
        raise CompositionManifestError("component_metadata_extensions_unsupported")
    if (
        artifacts.get("weights_file") != "weights.json"
        or artifacts.get("weights_fallback_file") != "model.npz"
    ):
        raise CompositionManifestError("component_metadata_incompatible")


def _validate_sidecar(sidecar: dict[str, Any]) -> None:
    expected = {
        "schema": "azlite_runtime_search_policy_v1",
        "mode": "puct_exact_root_hybrid",
        "exact_root_enabled": True,
        "exact_root_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "solver_implementation_identity": "native_kvtb_root_action_probe_v1",
        "native_probe": {
            "path": "model-artifact/runtime/kalah_v1_tablebase",
            "sha256": EXPECTED_RUNTIME["exact_root_native_probe"],
        },
        "tablebase": {
            "path": "model-artifact/runtime/kalah_v1_21.kvtb",
            "sha256": EXPECTED_RUNTIME["exact_root_tablebase"],
        },
    }
    if sidecar != expected:
        raise CompositionManifestError("runtime_sidecar_incompatible")


def validate_manifest(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
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
    if not isinstance(manifest, dict) or manifest.get("schema") not in {
        SCHEMA,
        RUNTIME_SCHEMA,
    }:
        raise CompositionManifestError("manifest_schema_unsupported")
    if manifest["schema"] == SCHEMA:
        raise CompositionManifestError(
            "checkpoint_only_proposal_not_runtime_consumable"
        )
    root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
    assets = Path(asset_root or root).resolve()
    if not root.is_dir() or not assets.is_dir():
        raise CompositionManifestError("asset_root_missing")
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
    if not isinstance(components, dict) or set(components) != set(COMPONENTS):
        raise CompositionManifestError("component_set_invalid")
    registration_path = _resolve(root, REGISTRATION)
    if (
        not registration_path.is_file()
        or _sha256(registration_path) != REGISTRATION_SHA256
    ):
        raise CompositionManifestError("seed512_registration_hash_mismatch")
    registration = _read_json(registration_path)
    registered_runtime = registration.get("runtime_assets", {})
    validated: dict[str, dict[str, Any]] = {}
    for name, (registration_name, routing) in COMPONENTS.items():
        component = components[name]
        if not isinstance(component, dict):
            raise CompositionManifestError("component_invalid")
        source = registration["components"][registration_name]
        reference = source["artifact"]
        if component.get("artifact_directory") != reference:
            raise CompositionManifestError("component_identity_mismatch")
        if component.get("source") != routing:
            raise CompositionManifestError("source_routing_mismatch")
        artifact_dir = _resolve(root, reference)
        files: dict[str, str] = {}
        for filename in RUNTIME_FILES:
            expected = source["files"].get(filename)
            actual = _resolve(root, f"{reference}/{filename}")
            if (
                not isinstance(expected, str)
                or not actual.is_file()
                or _sha256(actual) != expected
            ):
                raise CompositionManifestError(
                    f"component_file_mismatch:{name}:{filename}"
                )
            files[filename] = str(actual)
        metadata = _read_json(Path(files["metadata.json"]))
        _validate_metadata(metadata)
        _validate_sidecar(_read_json(Path(files["search_policy.json"])))
        if (
            metadata["artifacts"].get("weights_json_sha256")
            != source["files"]["weights.json"]
        ):
            raise CompositionManifestError("component_weights_metadata_hash")
        if metadata["artifacts"].get("weights_sha256") != source["files"]["model.npz"]:
            raise CompositionManifestError("component_model_metadata_hash")
        if (
            component.get("checkpoint") != source["checkpoint"]
            or component.get("checkpoint_sha256") != source["checkpoint_sha256"]
        ):
            raise CompositionManifestError("checkpoint_provenance_mismatch")
        validated[name] = {
            "directory": str(artifact_dir),
            "files": files,
            "checkpoint": source["checkpoint"],
            "checkpoint_sha256": source["checkpoint_sha256"],
        }
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != set(EXPECTED_RUNTIME):
        raise CompositionManifestError("runtime_identity_mismatch")
    resolved_runtime: dict[str, str] = {}
    for key, expected_hash in EXPECTED_RUNTIME.items():
        registration_key = key
        registered_asset = registered_runtime.get(registration_key)
        item = runtime.get(key)
        expected_keys = (
            {"path", "pointer_sha256", "materialized_path", "sha256"}
            if key == "exact_root_tablebase"
            else {"path", "sha256"}
        )
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise CompositionManifestError("runtime_identity_mismatch")
        if (
            not isinstance(registered_asset, dict)
            or registered_asset.get("sha256") != expected_hash
            or not isinstance(item, dict)
            or item.get("sha256") != expected_hash
            or item.get("path") != registered_asset.get("artifact")
        ):
            raise CompositionManifestError("runtime_identity_mismatch")
        if key == "exact_root_tablebase" and item.get(
            "pointer_sha256"
        ) != registered_asset.get("source_pointer_sha256"):
            raise CompositionManifestError("runtime_pointer_identity_mismatch")
        relative = item.get("path")
        if not isinstance(relative, str) or Path(relative).suffix not in {"", ".kvtb"}:
            raise CompositionManifestError("runtime_extension_unsupported")
        local = _resolve(root, relative)
        if not local.is_file():
            raise CompositionManifestError(f"runtime_artifact_missing:{key}")
        local_digest = _sha256(local)
        if local_digest == expected_hash:
            materialized = local
        elif key == "exact_root_tablebase" and local_digest == item.get(
            "pointer_sha256"
        ):
            materialized_path = item.get("materialized_path")
            if not isinstance(materialized_path, str):
                raise CompositionManifestError("runtime_materialized_path_missing")
            materialized = (assets / materialized_path).resolve()
            if assets not in materialized.parents:
                raise CompositionManifestError("runtime_materialized_path_escapes_root")
        else:
            raise CompositionManifestError(f"runtime_pointer_or_object_corrupt:{key}")
        if not materialized.is_file() or _sha256(materialized) != expected_hash:
            raise CompositionManifestError(f"runtime_materialized_object_corrupt:{key}")
        resolved_runtime[key] = str(materialized)
    return {
        "schema": RUNTIME_SCHEMA,
        "components": validated,
        "runtime": resolved_runtime,
    }
