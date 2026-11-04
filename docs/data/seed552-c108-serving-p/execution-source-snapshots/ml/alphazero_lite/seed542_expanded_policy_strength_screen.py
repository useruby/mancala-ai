"""Prospective reused-suite D-final policy strength screen for seed542."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.seed482_fresh_confirmation import state_from_encoded

ROOT = Path(__file__).resolve().parents[2]
OUT = "docs/data/seed542-expanded-policy-strength-screen"
SEED511 = "docs/data/seed511-fixed-composition-screen-v4"
OPENING_SHA256 = "27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d"
INITIAL_STATE = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
SUITE_SOURCE_HASHES = {
    "ml/alphazero_lite/build_opening_suite.py": "5c15d541574905ff9d7c75bc8daf8813311e22f22f220563e18cb9b85fb8bd87",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
    "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
    "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
    "ml/alphazero_lite/seed482_fresh_confirmation.py": "81460b8fedd4df15c5614fdb76ce32a7bcd77864782bfe0cc0e07316b1c23952",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_identity(state: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(state))


def float32_identity(state: dict[str, Any]) -> str:
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return sha256_bytes(vector.tobytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def import_seed511_suite(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and replay only the imported suite, never historical games."""
    root = root.resolve()
    suite_dir = root / SEED511
    registration_path = suite_dir / "registration.json"
    freeze_path = suite_dir / "execution-freeze.json"
    openings_path = suite_dir / "openings.jsonl"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    registration_sha = sha256_file(registration_path)
    suite_sha = sha256_file(openings_path)
    if (
        freeze.get("status") != "frozen_before_candidate_inference_or_games"
        or registration.get("status") != "frozen_before_candidate_inference_or_games"
        or registration_sha != freeze.get("registration_sha256")
        or suite_sha != freeze.get("suite_sha256")
        or suite_sha != registration.get("suite", {}).get("sha256")
        or suite_sha != OPENING_SHA256
    ):
        raise ValueError("import_seed511_freeze_or_suite_hash_mismatch")
    if registration["suite"].get("openings") != 1024:
        raise ValueError("import_seed511_opening_count_contract")

    authenticated_sources: dict[str, str] = {}
    for relative, expected in SUITE_SOURCE_HASHES.items():
        registered = registration.get("sources_sha256", {}).get(
            relative, registration.get("runtime_identity_sha256", {}).get(relative)
        )
        if registered != expected:
            raise ValueError(f"import_seed511_registered_source_identity:{relative}")
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"import_seed511_source_hash_mismatch:{relative}")
        authenticated_sources[relative] = actual

    raw_lines = openings_path.read_bytes().splitlines(keepends=True)
    openings = [json.loads(line) for line in raw_lines]
    if len(openings) != 1024 or len(raw_lines) != 1024:
        raise ValueError("import_seed511_opening_line_count")
    canonical_roots: set[str] = set()
    float32_roots: set[str] = set()
    orientations = {0: 0, 1: 0}
    for index, opening in enumerate(openings):
        if (
            opening.get("opening_index") != index
            or opening.get("opening_contract") != "arena_player_relative_v2"
        ):
            raise ValueError(f"import_seed511_order_or_contract:{index}")
        relative = opening.get("prefix_moves")
        absolute = opening.get("source_prefix_moves_absolute")
        if (
            not isinstance(relative, list)
            or not isinstance(absolute, list)
            or len(relative) != len(absolute)
            or not 4 <= len(relative) <= 12
        ):
            raise ValueError(f"import_seed511_prefix_shape:{index}")
        game = KalahGame.from_state(INITIAL_STATE)
        for ply, (rel_move, abs_move) in enumerate(
            zip(relative, absolute, strict=True)
        ):
            if (
                game.over()
                or abs_move // 6 != game.current_player
                or abs_move - game.current_player * 6 != rel_move
                or rel_move not in game.possible_moves()
                or not game.move(abs_move)
            ):
                raise ValueError(f"import_seed511_illegal_prefix:{index}:{ply}")
        state = game.to_state()
        if game.over() or state != opening.get("state"):
            raise ValueError(f"import_seed511_root_state_mismatch:{index}")
        if canonical_identity(state) != opening.get("state_hash"):
            raise ValueError(f"import_seed511_root_state_hash_mismatch:{index}")
        canonical = canonical_identity(state)
        float32 = float32_identity(state)
        if canonical in canonical_roots or float32 in float32_roots:
            raise ValueError(f"import_seed511_duplicate_root:{index}")
        canonical_roots.add(canonical)
        float32_roots.add(float32)
        orientations[state["current_player"]] += 1
        opening["import_validation"] = {
            "eligible_nonterminal_root": True,
            "legal_relative_moves": game.possible_moves(),
            "canonical_identity": canonical,
            "production_float32_input_identity": float32,
        }

    return openings, {
        "status": "suite_import_replay_passed",
        "historical_full_seed511_publication_verifier": {
            "status": "historical_publication_check_stopped_at_native_source_mismatch",
            "registered_native_source_sha256": "49fe76e8dc0e827db3181397dee7bfbcf3b1b35234757be1ac441ea09eefab6a",
            "current_native_source_sha256": sha256_file(
                root / "ml/alphazero_lite/native_exact_root_tablebase.py"
            ),
            "historical_games_or_outcomes_reused": False,
        },
        "historical_games_or_outcomes_imported": False,
        "registration_sha256": registration_sha,
        "suite_sha256": suite_sha,
        "openings": len(openings),
        "canonical_unique": len(canonical_roots),
        "production_float32_unique": len(float32_roots),
        "root_player_orientations": {
            str(key): value for key, value in orientations.items()
        },
        "authenticated_rules_encoding_sources": authenticated_sources,
        "original_opening_order_and_bytes_preserved": True,
    }


def _training_state_identities(
    root: Path, paths: list[tuple[str, str]]
) -> tuple[set[str], set[str], int]:
    canonical: set[str] = set()
    float32: set[str] = set()
    row_count = 0
    for relative, expected_sha in paths:
        path = root / relative
        if sha256_file(path) != expected_sha:
            raise ValueError(f"overlap_training_source_hash_mismatch:{relative}")
        for row in read_jsonl(path):
            encoded = np.asarray(row["state"], dtype="<f4")
            state = state_from_encoded(encoded)
            canonical.add(canonical_identity(state))
            float32.add(sha256_bytes(encoded.tobytes()))
            row_count += 1
    return canonical, float32, row_count


def overlap_report(root: Path, openings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare every imported root against registered C and D training sources."""
    root = root.resolve()
    seed504 = json.loads((root / "docs/data/seed504/manifest.json").read_text())
    seed542 = json.loads(
        (root / "docs/data/seed542-selfplay-4096/source-manifest.json").read_text()
    )
    c_sources = [(row["path"], row["sha256"]) for row in seed504["sources"]["training"]]
    d_sources = [
        (row["path"], row["sha256"])
        for row in seed542["sources"]
        if row["role"] == "training"
    ]
    corpora = {
        "seed504_original_training": _training_state_identities(root, c_sources),
        "seed542_expanded_training": _training_state_identities(root, d_sources),
    }
    corpus_sources = {
        "seed504_manifest_sha256": sha256_file(
            root / "docs/data/seed504/manifest.json"
        ),
        "seed504_training_sources": [
            {"path": path, "sha256": digest} for path, digest in c_sources
        ],
        "seed542_source_manifest_sha256": sha256_file(
            root / "docs/data/seed542-selfplay-4096/source-manifest.json"
        ),
        "seed542_training_sources": [
            {"path": path, "sha256": digest} for path, digest in d_sources
        ],
    }
    rows = []
    for opening in openings:
        state = opening["state"]
        can = canonical_identity(state)
        f32 = float32_identity(state)
        row: dict[str, Any] = {
            "opening_index": opening["opening_index"],
            "canonical_identity": can,
            "production_float32_input_identity": f32,
        }
        for name, (canonical, float32, _) in corpora.items():
            row[f"{name}_canonical_overlap"] = can in canonical
            row[f"{name}_production_float32_overlap"] = f32 in float32
        rows.append(row)
    aggregates = {}
    for name in corpora:
        aggregates[name] = {
            "training_rows": corpora[name][2],
            "unique_canonical_identities": len(corpora[name][0]),
            "unique_production_float32_identities": len(corpora[name][1]),
            "canonical_opening_overlaps": sum(
                row[f"{name}_canonical_overlap"] for row in rows
            ),
            "production_float32_opening_overlaps": sum(
                row[f"{name}_production_float32_overlap"] for row in rows
            ),
        }
    return {
        "schema": "seed542_reused_suite_training_overlap_v1",
        "exploratory": True,
        "reused_seed511_suite": True,
        "openings_retained": len(rows),
        "historical_seed461_overlap_limitation_unresolved": True,
        "fresh_holdout_claim": False,
        "corpus_sources": corpus_sources,
        "aggregates": aggregates,
        "per_opening": rows,
    }


def serving_component_identity(directory: Path) -> str:
    """Match the production serving identity algorithm for one artifact."""
    digest = hashlib.sha256()
    digest.update(b"azlite-serving-artifact-v1\0")
    for name in ("metadata.json", "weights.json"):
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def authenticate_runtime(root: Path, output: Path, tablebase: Path) -> dict[str, Any]:
    """Bind D/E4/initializer runtime artifacts to the current serving release."""
    root, output, tablebase = root.resolve(), output.resolve(), tablebase.resolve()
    artifacts = output / "runtime-artifacts"
    registration_path = root / SEED511 / "registration.json"
    historical = json.loads(registration_path.read_text())
    release = json.loads((root / "deploy/serving-release.json").read_text())
    release_manifest = json.loads((root / "deploy/release-manifest.json").read_text())
    bundle = release["bundles"]["policy-value-p"]

    e4 = artifacts / "E4"
    initializer = artifacts / "initializer"
    d_final = artifacts / "D-final"
    expected_historical = historical["components"]
    component_records = {}
    for name, directory in (("E4", e4), ("initializer", initializer)):
        expected_files = expected_historical[name]["files"]
        actual_files = {
            relative: sha256_file(directory / relative) for relative in expected_files
        }
        if actual_files != expected_files:
            raise ValueError(f"runtime_imported_component_hash_mismatch:{name}")
        component_records[name] = {
            "checkpoint": expected_historical[name]["checkpoint"],
            "checkpoint_sha256": expected_historical[name]["checkpoint_sha256"],
            "artifact_identity": serving_component_identity(directory),
            "files": actual_files,
        }

    d_checkpoint = root / "docs/data/seed542-fixed-update-cd-v2/D-update-108.npz"
    d_metadata = json.loads((d_final / "metadata.json").read_text())
    d_files = {
        name: sha256_file(d_final / name)
        for name in ("model.npz", "weights.json", "metadata.json", "search_policy.json")
    }
    if (
        sha256_file(d_checkpoint) != d_files["model.npz"]
        or d_metadata["architecture"]["model_type"] != "residual_v3"
        or d_metadata["input_encoding"] != "kalah_v3"
        or d_metadata["training"].get("exact_root_policy_loss_weight") != 1.0
    ):
        raise ValueError("runtime_D_final_export_checkpoint_mismatch")
    policy = json.loads((d_final / "search_policy.json").read_text())
    expected_policy = json.loads(
        (root / "model-artifact/current/search_policy.json").read_text()
    )
    if policy != expected_policy:
        raise ValueError("runtime_D_final_search_sidecar_mismatch")

    identities = [
        serving_component_identity(initializer),
        serving_component_identity(e4),
    ]
    policy_hashes = [
        sha256_file(initializer / "search_policy.json"),
        sha256_file(e4 / "search_policy.json"),
    ]
    if (
        identities != bundle["expected_artifact_sha256"]
        or policy_hashes != bundle["expected_policy_sha256"]
    ):
        raise ValueError("runtime_current_P_serving_bundle_mismatch")

    manifest_rows = {row["destination"]: row for row in release_manifest["files"]}
    release_files = {}
    for component, directory in (("E4", e4), ("initializer", initializer)):
        for filename in (
            "metadata.json",
            "model.npz",
            "weights.json",
            "search_policy.json",
        ):
            destination = f"components/{component}/{filename}"
            expected = manifest_rows[destination]["sha256"]
            actual = sha256_file(directory / filename)
            if actual != expected:
                raise ValueError(f"runtime_deploy_manifest_file_mismatch:{destination}")
            release_files[destination] = actual

    probe = root / "model-artifact/runtime/kalah_v1_tablebase"
    if (
        sha256_file(probe)
        != "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48"
    ):
        raise ValueError("runtime_native_probe_hash_mismatch")
    if (
        sha256_file(tablebase)
        != "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
    ):
        raise ValueError("runtime_tablebase_hash_mismatch")
    if (
        sha256_file(root / "model-artifact/runtime/kalah_v1_21.kvtb")
        != "a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c"
    ):
        raise ValueError("runtime_tablebase_pointer_hash_mismatch")

    result = {
        "status": "runtime_components_authenticated",
        "current_serving_bundle": "policy-value-p",
        "serving_release_sha256": sha256_file(root / "deploy/serving-release.json"),
        "release_manifest_sha256": sha256_file(root / "deploy/release-manifest.json"),
        "current_P_policy_component": "historical seed504 E4 policy with E4 own trunk",
        "challenger_policy_component": "seed542 D-final update108 policy with D-final own trunk",
        "both_value_components": "identical frozen initializer value network with initializer own trunk",
        "component_assignments": {
            "challenger_policy": "D-final",
            "challenger_policy_trunk": "D-final own trunk",
            "challenger_value": "initializer",
            "challenger_value_trunk": "initializer own trunk",
            "opponent_policy": "E4",
            "opponent_policy_trunk": "E4 own trunk",
            "opponent_value": "initializer",
            "opponent_value_trunk": "initializer own trunk",
        },
        "current_P_component_artifact_identities": identities,
        "current_P_search_policy_sha256": policy_hashes,
        "components": component_records,
        "D_final": {
            "checkpoint_sha256": sha256_file(d_checkpoint),
            "files": d_files,
        },
        "runtime_assets": {
            "native_probe_path": "model-artifact/runtime/kalah_v1_tablebase",
            "native_probe_sha256": sha256_file(probe),
            "tablebase_path": "model-artifact/runtime/kalah_v1_21.kvtb",
            "materialized_tablebase_path": str(tablebase.relative_to(root)),
            "tablebase_sha256": sha256_file(tablebase),
            "tablebase_pointer_sha256": "a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c",
        },
        "deploy_manifest_file_hashes": release_files,
    }
    (output / "runtime-authentication.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def write_import_and_overlap(root: Path, output: Path) -> dict[str, Any]:
    """Write pre-arena suite import and overlap reports."""
    openings, imported = import_seed511_suite(root)
    overlap = overlap_report(root, openings)
    output.mkdir(parents=True, exist_ok=True)
    (output / "suite-import-report.json").write_text(
        json.dumps(imported, indent=2, sort_keys=True) + "\n"
    )
    summary = {key: value for key, value in overlap.items() if key != "per_opening"}
    (output / "overlap-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "overlap-per-opening.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in overlap["per_opening"]
        )
    )
    return {
        "suite_import_sha256": sha256_file(output / "suite-import-report.json"),
        "overlap_summary_sha256": sha256_file(output / "overlap-summary.json"),
        "overlap_per_opening_sha256": sha256_file(output / "overlap-per-opening.jsonl"),
        "openings": len(openings),
        "overlap_aggregates": overlap["aggregates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / OUT)
    args = parser.parse_args()
    print(
        json.dumps(
            write_import_and_overlap(args.root, args.out), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
