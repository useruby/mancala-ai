"""Descriptive B512/S615 policy comparison on the frozen Seed 636 cohort.

The functions in this module are deterministic, side-effect-free analysis
primitives. They do not regenerate states, search, solve, or assess playing
strength.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "seed639-prospective-policy-prior-classification"
POPULATION_SIZE = 512
SELECTED_INVENTORY_SHA256 = (
    "6c6b1f680b2177cb67c79a9a642216703f160131034fbf8fe86e8130910ed077"
)
SUBSET_SIZE = 64
MINIMUM_SUBSET_GAMES = 32
MAX_PER_GAME = 1
MAX_PER_OPENING = 4
STONE_BIN_WEIGHTS = {17: 0.2, 18: 0.2, 19: 0.2, 20: 0.2, 21: 0.2}
STONE_BIN_COUNTS = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
MODEL_ARTIFACTS = {
    "B512": {
        "model.npz": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
        "weights.json": "d3effd234fe657fd8be3e3896153441d75ba338e83e83b66d7541936b3c01ccd",
        "metadata.json": "510d4d5056d1b68ddd83eed89a4e8da13e9305d2fa39a46eefcebcf13334dce6",
    },
    "S615": {
        "model.npz": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
        "weights.json": "8f36140461390d317e7e5bcf604357ca65fa1b605bd1e9562bf923b840e13433",
        "metadata.json": "24e42998824856d0242e04ff8a98431b139375e0bc4540d759373ccb335983b3",
    },
}
PINNED_MODEL_SOURCES = {
    "ml/alphazero_lite/arena.py": "c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2",
    "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
    "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
}
PINNED_ENVIRONMENT = {
    "PYTHONHASHSEED": "615",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
REGISTERED_INPUT_SHA256 = {
    "selected_manifest": "d7e095f749c14a429162be91dd3bb5f67b685fab04808658f7f26ac7882c8fab",
    "selected_verification": "4c4c177d2fd4bf8b4e0b37e65584b9c6a62ca1949c6c114b6bf891fefd15c09a",
    "archive": "d6e454f889e6155d7dccf3bf81e6769989896389d401fc5bf4cbd058059c8bf1",
    "archive_manifest": "057358aad2d72b733a29019c21f7a3f0a9b7d68e37cd5ea2954660e59c366bc5",
    "population_member": "689269182824d86d55261b8c89141bbc578867339cc1adce695f57da2c7fc6fc",
}
SEED615_EVALUATION_FREEZE_SHA256 = (
    "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
)
HISTORICAL_VERIFIER_MODULE = "ml.alphazero_lite.seed639_historical_authentication"
INHERITED_P_E4_SHA256 = (
    "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
)


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize JSON values canonically for stable input identities."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def authenticate_inputs(
    *,
    seed615_root: Path,
    selected_manifest_path: Path,
    selected_verification_path: Path,
    archive_path: Path,
    archive_manifest_path: Path,
    population_member_path: Path | None = None,
    historical_verifier_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the fixed cohort bundle and historical model evaluation."""
    from ml.alphazero_lite import experiment_lifecycle

    expected_hashes = REGISTERED_INPUT_SHA256
    actual_paths = {
        "selected_manifest": selected_manifest_path,
        "selected_verification": selected_verification_path,
        "archive": archive_path,
        "archive_manifest": archive_manifest_path,
    }
    actual_hashes = {
        name: sha256_bytes(path.read_bytes()) for name, path in actual_paths.items()
    }
    expected_external_hashes = {
        key: value
        for key, value in expected_hashes.items()
        if key != "population_member"
    }
    if actual_hashes != expected_external_hashes:
        raise ValueError("seed636_registered_input_hash_mismatch")

    selected_manifest = json.loads(selected_manifest_path.read_bytes())
    selected_verification = json.loads(selected_verification_path.read_bytes())
    if (
        selected_manifest.get("schema") != "seed636-selected-state-manifest-v1"
        or selected_manifest.get("selected_count") != POPULATION_SIZE
        or selected_manifest.get("selected_inventory_sha256")
        != SELECTED_INVENTORY_SHA256
        or selected_manifest.get("exposure_classification")
        != "known-exposure-screened; inherited P-E4 exposure unresolved"
    ):
        raise ValueError("seed636_selected_manifest_contract_mismatch")
    if (
        selected_verification.get("status") != "verified"
        or selected_verification.get("selected_count") != POPULATION_SIZE
        or selected_verification.get("selected_inventory_sha256")
        != SELECTED_INVENTORY_SHA256
        or selected_verification.get("manifest_sha256")
        != expected_hashes["selected_manifest"]
        or selected_verification.get("population_member_sha256")
        != expected_hashes["population_member"]
    ):
        raise ValueError("seed636_selected_manifest_verification_mismatch")

    archive_verification = experiment_lifecycle.verify_archive(
        archive_path, archive_manifest_path
    )
    if (
        archive_verification.get("experiment_id") != "seed636-prospective-population"
        or archive_verification.get("attempt_id") != "attempt-0001"
    ):
        raise ValueError("seed636_archive_identity_mismatch")
    members = {member["path"]: member for member in archive_verification["files"]}
    population_member = members.get("population/population.json")
    if (
        population_member is None
        or population_member["sha256"] != expected_hashes["population_member"]
    ):
        raise ValueError("seed636_population_member_archive_binding_mismatch")
    if population_member_path is None:
        try:
            with tarfile.open(archive_path, mode="r:*") as archive:
                member_stream = archive.extractfile("population/population.json")
                if member_stream is None:
                    raise ValueError("seed636_population_archive_member_missing")
                population_raw = member_stream.read()
        except (tarfile.TarError, OSError) as error:
            raise ValueError("seed636_population_archive_member_unreadable") from error
    else:
        population_raw = population_member_path.read_bytes()
    if sha256_bytes(population_raw) != expected_hashes["population_member"]:
        raise ValueError("seed636_population_member_sha256_mismatch")
    population = json.loads(population_raw)
    selected_rows = population.get("selection", {}).get("states")
    if (
        not isinstance(population, dict)
        or not isinstance(selected_rows, list)
        or len(selected_rows) != POPULATION_SIZE
        or sha256_bytes(canonical_json(selected_rows)) != SELECTED_INVENTORY_SHA256
    ):
        raise ValueError("seed636_population_inventory_mismatch")
    manifest_states = selected_manifest.get("states")
    if not isinstance(manifest_states, list) or len(manifest_states) != POPULATION_SIZE:
        raise ValueError("seed636_manifest_state_inventory_invalid")
    manifest_fields = (
        "canonical_identity",
        "production_float32_identity",
        "active_pit_stones",
        "game_id",
        "opening_identity",
        "opening_index",
        "continuation_ply",
    )
    for index, (manifest_row, population_row) in enumerate(
        zip(manifest_states, selected_rows, strict=True)
    ):
        if manifest_row.get("i") != index or any(
            manifest_row.get(field) != population_row.get(field)
            for field in manifest_fields
        ):
            raise ValueError(f"seed636_manifest_population_row_mismatch:{index}")
    game_counts = Counter(row["game_id"] for row in selected_rows)
    opening_counts = Counter(row["opening_identity"] for row in selected_rows)
    if (
        len(opening_counts) != selected_manifest.get("distinct_opening_clusters")
        or max(game_counts.values(), default=0) != selected_manifest.get("max_per_game")
        or max(opening_counts.values(), default=0)
        != selected_manifest.get("max_per_opening")
        or any(count > 2 for count in game_counts.values())
        or any(count > 4 for count in opening_counts.values())
        or len(opening_counts) < 128
    ):
        raise ValueError("seed636_manifest_cluster_coverage_mismatch")
    if not all(selected_verification.get("checks", {}).values()) or set(
        selected_verification.get("checks", {})
    ) != {
        "archive_hash",
        "archive_sidecar_identity",
        "canonical_ids_unique",
        "exposure_binding",
        "float32_ids_unique",
        "game_cap",
        "opening_cap",
        "population_member_hash",
        "projection_matches_archived_rows",
        "selected_count",
        "selected_inventory_hash",
        "sidecar_hash",
        "stone_quotas",
    }:
        raise ValueError("seed636_independent_verification_checks_invalid")

    seed615_authentication = authenticate_seed615_in_subprocess(
        seed615_root=seed615_root,
        archive_path=archive_path,
        registered_runtime=historical_verifier_runtime,
    )
    if seed615_authentication.get(
        "freeze_sha256"
    ) != SEED615_EVALUATION_FREEZE_SHA256 or seed615_authentication.get(
        "checkpoint_sha256"
    ) != {
        "B512": MODEL_ARTIFACTS["B512"]["model.npz"],
        "S615": MODEL_ARTIFACTS["S615"]["model.npz"],
        "P-E4": INHERITED_P_E4_SHA256,
    }:
        raise ValueError("seed615_historical_authentication_mismatch")
    evaluation_freeze_path = (
        seed615_root
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/execution-freeze-v4.json"
    )
    evaluation_freeze = json.loads(evaluation_freeze_path.read_bytes())
    artifact_bindings = {}
    artifact_root = (
        seed615_root
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
    )
    for model, expected in MODEL_ARTIFACTS.items():
        runtime_record = evaluation_freeze["runtime_artifacts"][model]
        actual = {
            filename: sha256_bytes((artifact_root / model / filename).read_bytes())
            for filename in expected
        }
        recorded = {
            "model.npz": runtime_record["model_npz_sha256"],
            "weights.json": runtime_record["weights_json_sha256"],
            "metadata.json": runtime_record["metadata_sha256"],
        }
        if actual != expected or actual != recorded:
            raise ValueError(f"seed615_numpy_artifact_identity_mismatch:{model}")
        metadata = json.loads((artifact_root / model / "metadata.json").read_bytes())
        if (
            metadata.get("framework") != "numpy"
            or metadata.get("input_encoding") != "kalah_v3"
            or metadata.get("artifacts", {}).get("weights_sha256")
            != expected["model.npz"]
            or metadata.get("artifacts", {}).get("weights_json_sha256")
            != expected["weights.json"]
        ):
            raise ValueError(f"seed615_numpy_artifact_metadata_mismatch:{model}")
        artifact_bindings[model] = actual
    source_bindings = {
        relative: sha256_bytes(
            (Path(__file__).resolve().parents[2] / relative).read_bytes()
        )
        for relative in PINNED_MODEL_SOURCES
    }
    if source_bindings != PINNED_MODEL_SOURCES:
        raise ValueError("seed639_approved_model_source_identity_mismatch")
    return {
        "selected_inventory_sha256": SELECTED_INVENTORY_SHA256,
        "selected_count": POPULATION_SIZE,
        "opening_clusters": len(opening_counts),
        "maximum_states_per_game": max(game_counts.values()),
        "maximum_states_per_opening": max(opening_counts.values()),
        "selected_population_rows": selected_rows,
        "input_sha256": actual_hashes,
        "seed636_archive_verification": archive_verification,
        "seed615_authentication": seed615_authentication,
        "numpy_artifact_bindings": artifact_bindings,
        "approved_model_source_bindings": source_bindings,
        "population_member_sha256": expected_hashes["population_member"],
        "checkpoint_identity": {
            "B512": MODEL_ARTIFACTS["B512"]["model.npz"],
            "S615": MODEL_ARTIFACTS["S615"]["model.npz"],
            "historical_seed615_evaluation_freeze_sha256": SEED615_EVALUATION_FREEZE_SHA256,
        },
        "exposure_classification": selected_manifest["exposure_classification"],
    }


def authenticate_seed615_in_subprocess(
    *,
    seed615_root: Path,
    archive_path: Path,
    registered_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Run read-only Seed 615 authentication in the pinned historical runtime."""
    from ml.alphazero_lite.seed639_authorization import HISTORICAL_VERIFIER_RUNTIME

    if registered_runtime != HISTORICAL_VERIFIER_RUNTIME:
        raise ValueError("seed639_historical_runtime_registration_mismatch")
    command = [
        registered_runtime["interpreter"],
        "-m",
        HISTORICAL_VERIFIER_MODULE,
        "--archive",
        str(archive_path.resolve(strict=True)),
        "--seed615-root",
        str(seed615_root.resolve(strict=True)),
    ]
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "LANG", "LC_ALL")
        if key in os.environ
    } | registered_runtime["thread_environment"]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=14_400,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "seed639_historical_authentication_subprocess_failed:"
            f"exit={completed.returncode}; stderr={completed.stderr[-4000:]}"
        )
    response = json.loads(completed.stdout)
    runtime = response.get("runtime", {})
    if (
        response.get("schema") != "seed639-seed615-historical-authentication-v1"
        or runtime.get("python_executable") != registered_runtime["interpreter"]
        or runtime.get("python_version") != registered_runtime["python_version"]
        or runtime.get("numpy_version") != registered_runtime["numpy_version"]
        or runtime.get("torch_version") != registered_runtime["torch_version"]
        or runtime.get("device") != registered_runtime["device"]
        or runtime.get("cuda_available") is not registered_runtime["cuda_available"]
        or runtime.get("thread_environment") != registered_runtime["thread_environment"]
        or runtime.get("torch_threads") != registered_runtime["torch_threads"]
        or runtime.get("deterministic_algorithms")
        is not registered_runtime["deterministic_algorithms"]
        or runtime.get("runtime_lock_path") != registered_runtime["runtime_lock_path"]
        or runtime.get("runtime_lock_sha256")
        != registered_runtime["runtime_lock_sha256"]
        or runtime.get("seed615_dependency_lock_path")
        != registered_runtime["seed615_dependency_lock_path"]
        or runtime.get("seed615_dependency_lock_sha256")
        != registered_runtime["seed615_dependency_lock_sha256"]
        or runtime.get("worktree_seed615_dependency_lock_path")
        != str(
            Path(__file__).resolve().parents[2]
            / registered_runtime["worktree_dependency_lock"]
        )
        or runtime.get("worktree_seed615_dependency_lock_sha256")
        != registered_runtime["worktree_dependency_lock_sha256"]
    ):
        raise ValueError("seed639_historical_runtime_probe_mismatch")
    authentication = response.get("authentication")
    if not isinstance(authentication, dict):
        raise ValueError("seed639_historical_authentication_receipt_invalid")
    return {
        **authentication,
        "historical_runtime": runtime,
        "subprocess_command": command,
        "subprocess_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "subprocess_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
    }


def compare_legal_priors(
    policy_b512: np.ndarray | list[float],
    policy_s615: np.ndarray | list[float],
    legal_actions: list[int],
) -> dict[str, Any]:
    """Validate, mask, and compare full six-action policy outputs.

    Input vectors must be finite nonnegative probability distributions. Their
    legal components are renormalized after masking; illegal output probability
    is not retained in either returned legal distribution. Ties use NumPy's
    first-index argmax rule, matching the existing prior tie convention.
    """
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("invalid_legal_action_set")
    if any(
        type(action) is not int or action < 0 or action >= 6 for action in legal_actions
    ):
        raise ValueError("invalid_legal_action_set")

    vectors = []
    for label, raw in (("B512", policy_b512), ("S615", policy_s615)):
        vector = np.asarray(raw, dtype=np.float64)
        if (
            vector.shape != (6,)
            or not np.all(np.isfinite(vector))
            or np.any(vector < 0)
            or not math.isclose(float(vector.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError(f"invalid_raw_probability_vector:{label}")
        legal = np.zeros(6, dtype=bool)
        legal[legal_actions] = True
        masked = np.where(legal, vector, 0.0)
        mass = float(masked.sum())
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"zero_legal_probability_mass:{label}")
        masked /= mass
        if np.any(masked[~legal] != 0) or not math.isclose(
            float(masked.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"invalid_masked_probability_vector:{label}")
        vectors.append(masked)

    b512, s615 = vectors
    b_top, s_top = int(np.argmax(b512)), int(np.argmax(s615))
    return {
        "legal_actions": list(legal_actions),
        "B512_legal_probabilities": b512.tolist(),
        "S615_legal_probabilities": s615.tolist(),
        "B512_top_action": b_top,
        "S615_top_action": s_top,
        "top_action_disagreement": b_top != s_top,
        "total_variation": float(0.5 * np.abs(b512 - s615).sum()),
        "B512_entropy_bits": _entropy_bits(b512),
        "S615_entropy_bits": _entropy_bits(s615),
        "entropy_difference_S615_minus_B512_bits": _entropy_bits(s615)
        - _entropy_bits(b512),
    }


def evaluate_population(
    population_rows: list[dict[str, Any]], evaluator_b512: Any, evaluator_s615: Any
) -> list[dict[str, Any]]:
    """Evaluate exactly the frozen 512 rows and preserve source provenance."""
    validate_population(population_rows)
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    if len(population_rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    records = []
    canonical_ids: set[str] = set()
    float32_ids: set[str] = set()
    for index, source in enumerate(population_rows):
        state = source.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"population_state_missing:{index}")
        canonical_id = sha256_bytes(canonical_json(state))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
        )
        float32_id = sha256_bytes(encoded.tobytes())
        if canonical_id != source.get("canonical_identity"):
            raise ValueError(f"canonical_identity_mismatch:{index}")
        if float32_id != source.get("production_float32_identity"):
            raise ValueError(f"production_float32_identity_mismatch:{index}")
        if canonical_id in canonical_ids or float32_id in float32_ids:
            raise ValueError(f"duplicate_population_identity:{index}")
        canonical_ids.add(canonical_id)
        float32_ids.add(float32_id)

        game = KalahGame.from_state(state)
        legal = [int(action) for action in game.possible_moves()]
        if len(legal) < 2:
            raise ValueError(f"insufficient_legal_actions:{index}")
        source_legal = source.get("legal_actions")
        if not isinstance(source_legal, list) or sorted(source_legal) != sorted(legal):
            raise ValueError(f"legal_action_set_mismatch:{index}")
        policy_b512, _ = evaluator_b512.evaluate(game)
        policy_s615, _ = evaluator_s615.evaluate(game)
        comparison = compare_legal_priors(policy_b512, policy_s615, legal)
        stones = int(sum(game.pits))
        if stones != source.get("active_pit_stones"):
            raise ValueError(f"active_pit_stone_count_mismatch:{index}")
        record = {
            "population_index": index,
            "state": state,
            "canonical_identity": canonical_id,
            "production_float32_identity": float32_id,
            "active_pit_stones": stones,
            "game_id": source.get("game_id"),
            "opening_identity": source.get("opening_identity"),
            "opening_index": source.get("opening_index"),
            "continuation_ply": source.get("continuation_ply"),
            "side_to_move": int(game.current_player),
            **comparison,
        }
        if not record["game_id"] or not record["opening_identity"]:
            raise ValueError(f"population_provenance_missing:{index}")
        records.append(record)
    return records


def validate_population(population_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Authenticate the entire selected cohort before any model call."""
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    if len(population_rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    canonical_ids: set[str] = set()
    float32_ids: set[str] = set()
    by_bin: Counter[int] = Counter()
    game_opening: dict[str, str] = {}
    games_per_opening: Counter[str] = Counter()
    states_per_game: Counter[str] = Counter()
    preflight_rows = []
    for index, source in enumerate(population_rows):
        state = source.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"population_state_missing:{index}")
        canonical_id = sha256_bytes(canonical_json(state))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
        )
        float32_id = sha256_bytes(encoded.tobytes())
        if canonical_id != source.get("canonical_identity"):
            raise ValueError(f"canonical_identity_mismatch:{index}")
        if float32_id != source.get("production_float32_identity"):
            raise ValueError(f"production_float32_identity_mismatch:{index}")
        if canonical_id in canonical_ids or float32_id in float32_ids:
            raise ValueError(f"duplicate_population_identity:{index}")
        canonical_ids.add(canonical_id)
        float32_ids.add(float32_id)

        game = KalahGame.from_state(state)
        legal = [int(action) for action in game.possible_moves()]
        if len(legal) < 2:
            raise ValueError(f"insufficient_legal_actions:{index}")
        source_legal = source.get("legal_actions")
        if not isinstance(source_legal, list) or sorted(source_legal) != sorted(legal):
            raise ValueError(f"legal_action_set_mismatch:{index}")
        stones = int(sum(game.pits))
        if stones != source.get("active_pit_stones") or stones not in STONE_BIN_COUNTS:
            raise ValueError(f"active_pit_stone_count_mismatch:{index}")
        game_id = source.get("game_id")
        opening_id = source.get("opening_identity")
        opening_index = source.get("opening_index")
        ply = source.get("continuation_ply")
        if (
            not isinstance(game_id, str)
            or not game_id
            or not isinstance(opening_id, str)
            or len(opening_id) != 64
            or type(opening_index) is not int
            or type(ply) is not int
            or ply < 0
        ):
            raise ValueError(f"population_provenance_invalid:{index}")
        previous_opening = game_opening.setdefault(game_id, opening_id)
        if previous_opening != opening_id:
            raise ValueError(f"continuation_game_opening_mismatch:{index}")
        games_per_opening[opening_id] += 1
        states_per_game[game_id] += 1
        if games_per_opening[opening_id] > 4 or states_per_game[game_id] > 2:
            raise ValueError(f"seed636_primary_cluster_cap_mismatch:{index}")
        by_bin[stones] += 1
        preflight_rows.append(
            {
                "population_index": index,
                "canonical_identity": canonical_id,
                "production_float32_identity": float32_id,
                "legal_actions": legal,
                "active_pit_stones": stones,
                "game_id": game_id,
                "opening_identity": opening_id,
                "opening_index": opening_index,
                "continuation_ply": ply,
                "side_to_move": int(game.current_player),
            }
        )
    if dict(by_bin) != STONE_BIN_COUNTS:
        raise ValueError("primary_population_stone_bin_quota_mismatch")
    if len(canonical_ids) != POPULATION_SIZE or len(float32_ids) != POPULATION_SIZE:
        raise ValueError("primary_population_unique_identity_count_mismatch")
    if len(game_opening) < 32 or len(games_per_opening) < 128:
        raise ValueError("primary_population_cluster_coverage_mismatch")
    return {
        "states_authenticated": POPULATION_SIZE,
        "canonical_identity_count": len(canonical_ids),
        "production_float32_identity_count": len(float32_ids),
        "stone_bin_counts": {str(k): by_bin[k] for k in sorted(STONE_BIN_COUNTS)},
        "continuation_games": len(game_opening),
        "opening_clusters": len(set(game_opening.values())),
        "state_bindings": preflight_rows,
    }


def _entropy_bits(vector: np.ndarray) -> float:
    positive = vector[vector > 0]
    return float(-np.sum(positive * np.log2(positive)))


def select_diagnostic_subset(
    rows: list[dict[str, Any]],
    *,
    subset_size: int = SUBSET_SIZE,
    minimum_games: int = MINIMUM_SUBSET_GAMES,
) -> dict[str, Any]:
    """Select the Seed 623 high-TV subset without modifying the input cohort.

    TV is sorted descending. Exact TV ties are resolved by SHA-256 of the
    frozen canonical state identity, then by that identity as a collision
    fallback. Greedy traversal applies the registered one-per-game and
    four-per-opening caps.
    """
    if subset_size <= 0 or minimum_games <= 0:
        raise ValueError("invalid_subset_contract")
    identities: set[str] = set()
    ranked: list[tuple[float, str, str, dict[str, Any]]] = []
    for row in rows:
        state_id = row.get("canonical_identity")
        game_id = row.get("game_id")
        opening_id = row.get("opening_identity")
        tv = row.get("total_variation")
        valid_tv = type(tv) in (int, float)
        tv_value = float(tv) if isinstance(tv, (int, float)) else float("nan")
        if (
            not isinstance(state_id, str)
            or not isinstance(game_id, str)
            or not isinstance(opening_id, str)
            or not valid_tv
            or not math.isfinite(tv_value)
            or not 0.0 <= tv_value <= 1.0
        ):
            raise ValueError("invalid_subset_row")
        if state_id in identities:
            raise ValueError("duplicate_state_identity")
        identities.add(state_id)
        tie_digest = sha256_bytes(state_id.encode("utf-8"))
        ranked.append((-tv_value, tie_digest, state_id, row))

    selected: list[dict[str, Any]] = []
    games: set[str] = set()
    opening_counts: Counter[str] = Counter()
    for _, _, _, row in sorted(ranked):
        game_id = row["game_id"]
        opening_id = row["opening_identity"]
        if game_id in games or opening_counts[opening_id] >= MAX_PER_OPENING:
            continue
        selected.append(row)
        games.add(game_id)
        opening_counts[opening_id] += 1
        if len(selected) == subset_size:
            break
    available = len(selected) == subset_size and len(games) >= minimum_games
    return {
        "available": available,
        "unavailable_reason": None
        if available
        else "fewer_than_64_selectable_states_or_32_distinct_games",
        "selected_rows": selected if available else [],
        "selected_state_ids": [row["canonical_identity"] for row in selected]
        if available
        else [],
        "selected_count": len(selected) if available else 0,
        "distinct_continuation_games": len(games) if available else 0,
        "distinct_opening_clusters": len(opening_counts) if available else 0,
        "states_per_opening_cluster": dict(sorted(opening_counts.items()))
        if available
        else {},
        "selection_contract": {
            "sort": "descending_total_variation",
            "tie_break": "sha256(utf8(canonical_identity)); canonical_identity collision fallback",
            "max_per_continuation_game": MAX_PER_GAME,
            "max_per_opening_cluster": MAX_PER_OPENING,
            "subset_size": subset_size,
            "minimum_distinct_continuation_games": minimum_games,
        },
    }


def summarize_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive complete-population and fixed-bin summaries."""
    if len(rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    bins: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stones = row.get("active_pit_stones")
        if type(stones) is not int or stones not in STONE_BIN_WEIGHTS:
            raise ValueError("invalid_stone_bin")
        bins[stones].append(row)
    if any(not bins[stones] for stones in STONE_BIN_WEIGHTS):
        raise ValueError("missing_stone_bin")
    if {stones: len(bins[stones]) for stones in STONE_BIN_COUNTS} != STONE_BIN_COUNTS:
        raise ValueError("primary_population_stone_bin_quota_mismatch")

    tv = np.asarray([float(row["total_variation"]) for row in rows])
    disagreement = np.asarray(
        [bool(row["top_action_disagreement"]) for row in rows], dtype=np.float64
    )
    entropy_delta = np.asarray(
        [float(row["entropy_difference_S615_minus_B512_bits"]) for row in rows]
    )
    entropy_b512 = np.asarray([float(row["B512_entropy_bits"]) for row in rows])
    entropy_s615 = np.asarray([float(row["S615_entropy_bits"]) for row in rows])

    def metrics(group: list[dict[str, Any]]) -> dict[str, float | int]:
        group_tv = np.asarray([float(row["total_variation"]) for row in group])
        group_disagreement = np.asarray(
            [bool(row["top_action_disagreement"]) for row in group], dtype=float
        )
        group_entropy = np.asarray(
            [float(row["entropy_difference_S615_minus_B512_bits"]) for row in group]
        )
        group_entropy_b512 = np.asarray(
            [float(row["B512_entropy_bits"]) for row in group]
        )
        group_entropy_s615 = np.asarray(
            [float(row["S615_entropy_bits"]) for row in group]
        )
        return {
            "states": len(group),
            "top_action_disagreements": int(group_disagreement.sum()),
            "top_action_disagreement_rate": float(group_disagreement.mean()),
            "mean_total_variation": float(group_tv.mean()),
            "mean_B512_entropy_bits": float(group_entropy_b512.mean()),
            "mean_S615_entropy_bits": float(group_entropy_s615.mean()),
            "mean_entropy_difference_S615_minus_B512_bits": float(group_entropy.mean()),
        }

    quantiles = {
        str(q): float(np.quantile(tv, q))
        for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
    }
    by_bin = {str(stones): metrics(group) for stones, group in sorted(bins.items())}
    standardized = {
        metric: float(
            sum(
                STONE_BIN_WEIGHTS[stones] * float(by_bin[str(stones)][metric])
                for stones in STONE_BIN_WEIGHTS
            )
        )
        for metric in (
            "top_action_disagreement_rate",
            "mean_total_variation",
            "mean_entropy_difference_S615_minus_B512_bits",
        )
    }
    return {
        "population_size": len(rows),
        "overall": {
            "top_action_disagreements": int(disagreement.sum()),
            "top_action_disagreement_rate": float(disagreement.mean()),
            "total_variation_mean": float(tv.mean()),
            "total_variation_quantiles": quantiles,
            "B512_mean_entropy_bits": float(entropy_b512.mean()),
            "S615_mean_entropy_bits": float(entropy_s615.mean()),
            "entropy_difference_S615_minus_B512_mean_bits": float(entropy_delta.mean()),
        },
        "by_stone_bin": by_bin,
        "fixed_one_fifth_bin_standardized": standardized,
    }


def summarize_subset_relationship(
    population_rows: list[dict[str, Any]], subset: dict[str, Any]
) -> dict[str, Any]:
    """Describe the enriched subset's coverage and differences from its parent."""
    if len(population_rows) != POPULATION_SIZE:
        raise ValueError("population_must_contain_exactly_512_rows")
    if not subset.get("available"):
        return {
            "available": False,
            "reason": subset.get("unavailable_reason"),
            "primary_population_size": len(population_rows),
            "subset_size": 0,
        }
    selected_ids = subset.get("selected_state_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != SUBSET_SIZE:
        raise ValueError("diagnostic_subset_identity_invalid")
    primary_by_id = {row.get("canonical_identity"): row for row in population_rows}
    if len(primary_by_id) != POPULATION_SIZE or any(
        state_id not in primary_by_id for state_id in selected_ids
    ):
        raise ValueError("diagnostic_subset_not_contained_in_primary_population")
    selected = [primary_by_id[state_id] for state_id in selected_ids]

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        by_bin: Counter[int] = Counter(int(row["active_pit_stones"]) for row in group)
        return {
            "states": len(group),
            "stone_bin_counts": {str(k): by_bin.get(k, 0) for k in STONE_BIN_WEIGHTS},
            "distinct_continuation_games": len({row["game_id"] for row in group}),
            "distinct_opening_clusters": len(
                {row["opening_identity"] for row in group}
            ),
            "top_action_disagreement_rate": float(
                np.mean([bool(row["top_action_disagreement"]) for row in group])
            ),
            "mean_total_variation": float(
                np.mean([float(row["total_variation"]) for row in group])
            ),
            "mean_entropy_difference_S615_minus_B512_bits": float(
                np.mean(
                    [
                        float(row["entropy_difference_S615_minus_B512_bits"])
                        for row in group
                    ]
                )
            ),
        }

    return {
        "available": True,
        "primary_population_size": len(population_rows),
        "subset_size": len(selected),
        "subset_is_unweighted_descriptive_only": True,
        "primary_population": describe(population_rows),
        "diagnostic_subset": describe(selected),
    }


def runtime_identity() -> dict[str, Any]:
    """Enforce and report the registered deterministic NumPy runtime."""
    expected_python = "3.14.7"
    expected_numpy = "2.5.3"
    if platform.python_version() != expected_python or np.__version__ != expected_numpy:
        raise RuntimeError("seed639_runtime_version_mismatch")
    environment = {name: os.environ.get(name) for name in PINNED_ENVIRONMENT}
    if environment != PINNED_ENVIRONMENT:
        raise RuntimeError("seed639_thread_environment_mismatch")
    return {
        "python_version": platform.python_version(),
        "python_full_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "backend": "NumPy ArtifactEvaluator",
        "numpy_build_configuration": np.show_config(mode="dicts"),
        "thread_configuration": {"environment": environment, "limit_per_pool": 1},
        "environment": environment,
        "inference_batch_size": 1,
        "evaluation_cache": False,
    }


def build_authenticated_evaluators(
    seed615_root: Path, authentication: dict[str, Any]
) -> dict[str, Any]:
    """Construct only the two source-pinned NumPy artifact evaluators."""
    from ml.alphazero_lite import arena

    root = Path(__file__).resolve().parents[2]
    for relative, expected in PINNED_MODEL_SOURCES.items():
        if sha256_bytes((root / relative).read_bytes()) != expected:
            raise ValueError(f"seed639_evaluator_source_hash_mismatch:{relative}")
    if authentication.get("numpy_artifact_bindings") != MODEL_ARTIFACTS:
        raise ValueError("seed639_authenticated_model_binding_invalid")
    artifact_root = (
        seed615_root
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
    )
    evaluators: dict[str, Any] = {}
    for model in ("B512", "S615"):
        artifact_dir = artifact_root / model
        verify_numpy_artifact_files(artifact_dir, MODEL_ARTIFACTS[model], model)
        evaluator = arena.ArtifactEvaluator(artifact_dir)
        assert_evaluator_identity(evaluator, arena.ArtifactEvaluator, model)
        evaluators[model] = evaluator
    return evaluators


def verify_numpy_artifact_files(
    artifact_dir: Path, expected_hashes: dict[str, str], model: str
) -> dict[str, str]:
    """Check exact checkpoint, exported weights, and metadata digests."""
    actual = {
        filename: sha256_bytes((artifact_dir / filename).read_bytes())
        for filename in ("model.npz", "weights.json", "metadata.json")
    }
    if actual != expected_hashes:
        raise ValueError(f"seed639_numpy_artifact_identity_mismatch:{model}")
    metadata = json.loads((artifact_dir / "metadata.json").read_bytes())
    if (
        metadata.get("framework") != "numpy"
        or metadata.get("input_encoding") != "kalah_v3"
        or metadata.get("artifacts", {}).get("weights_sha256")
        != expected_hashes["model.npz"]
        or metadata.get("artifacts", {}).get("weights_json_sha256")
        != expected_hashes["weights.json"]
    ):
        raise ValueError(f"seed639_numpy_artifact_metadata_mismatch:{model}")
    return actual


def assert_evaluator_identity(
    evaluator: Any, evaluator_class: type, model: str
) -> None:
    """Reject substituted evaluator instances or incompatible runtime settings."""
    if type(evaluator) is not evaluator_class:
        raise TypeError(f"seed639_evaluator_type_mismatch:{model}")
    if getattr(evaluator, "input_encoding", None) != "kalah_v3":
        raise ValueError(f"seed639_evaluator_encoding_mismatch:{model}")
    if hasattr(evaluator, "cache"):
        raise ValueError(f"seed639_evaluator_cache_must_be_disabled:{model}")


def _create_only_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        ).encode("ascii")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _create_only_record(path: Path, value: Any) -> None:
    payload = canonical_json(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _input_path(registration: dict[str, Any], suffix: str) -> Path:
    matches = [
        Path(path) for path in registration["input_artifacts"] if path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"seed639_registered_input_path_not_unique:{suffix}")
    return matches[0]


def run_attempt_0002_preflight() -> dict[str, Any]:
    """Authenticate owner merge and all registered inputs without model inference."""
    from ml.alphazero_lite.seed639_authorization import (
        EXECUTION_AMENDMENT,
        REGISTRATION_PATH,
        authorize_registration,
    )

    root = Path(__file__).resolve().parents[2]
    registration_path = root / REGISTRATION_PATH
    registration = json.loads(registration_path.read_bytes())
    if registration.get("execution_amendment") != EXECUTION_AMENDMENT:
        raise PermissionError("seed639_attempt_0002_amendment_not_registered")
    source_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty_status:
        raise PermissionError("seed639_attempt_0002_preflight_worktree_dirty")
    authorization = authorize_registration(registration_path, source_commit)
    if authorization.get("merge_commit") != source_commit or authorization.get(
        "registration_sha256"
    ) != sha256_bytes(registration_path.read_bytes()):
        raise PermissionError("seed639_attempt_0002_owner_merge_source_mismatch")
    runtime = runtime_identity()
    seed615_freeze_path = _input_path(
        registration, "/evaluation-preparation/execution-freeze-v4.json"
    )
    seed615_root = seed615_freeze_path.parents[4]
    archive_path = _input_path(registration, "attempt-0001.tar.zst")
    authentication = authenticate_inputs(
        seed615_root=seed615_root,
        selected_manifest_path=_input_path(
            registration, "/selected-state-manifest.json"
        ),
        selected_verification_path=_input_path(
            registration, "/selected-state-manifest-verification.json"
        ),
        archive_path=archive_path,
        archive_manifest_path=_input_path(
            registration, "attempt-0001.archive-manifest.json"
        ),
        historical_verifier_runtime=EXECUTION_AMENDMENT["historical_verifier_runtime"],
    )
    archive_verification = authentication["seed636_archive_verification"]
    preflight = {
        "schema": "seed639-attempt-0002-read-only-preflight-v1",
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": "attempt-0002",
        "source_commit": source_commit,
        "authorization": authorization,
        "runtime": runtime,
        "registered_inputs_verified": True,
        "historical_authentication": {
            key: value
            for key, value in authentication.items()
            if key not in {"selected_population_rows", "seed636_archive_verification"}
        },
        "seed636_archive_inventory": {
            key: value for key, value in archive_verification.items() if key != "files"
        },
        "population_states_validated": len(authentication["selected_population_rows"]),
        "prospective_model_inference_performed": False,
    }
    return preflight


def persist_inference_records(
    population_rows: list[dict[str, Any]],
    evaluator_b512: Any,
    evaluator_s615: Any,
    record_directory: Path,
) -> list[dict[str, Any]]:
    """Write each paired model-state record immediately with create-only I/O."""
    from ml.alphazero_lite.kalah_rules import KalahGame

    if not record_directory.is_dir():
        raise ValueError("seed639_record_directory_must_be_created_exclusively")
    records = []
    for index, state_row in enumerate(population_rows):
        game = KalahGame.from_state(state_row["state"])
        b_policy, _ = evaluator_b512.evaluate(game)
        s_policy, _ = evaluator_s615.evaluate(game)
        record = {
            "population_index": index,
            "state": state_row["state"],
            "canonical_identity": state_row["canonical_identity"],
            "production_float32_identity": state_row["production_float32_identity"],
            "active_pit_stones": state_row["active_pit_stones"],
            "game_id": state_row["game_id"],
            "opening_identity": state_row["opening_identity"],
            "opening_index": state_row["opening_index"],
            "continuation_ply": state_row["continuation_ply"],
            "side_to_move": int(game.current_player),
            **compare_legal_priors(
                b_policy,
                s_policy,
                [int(action) for action in game.possible_moves()],
            ),
        }
        _create_only_record(record_directory / f"state-{index:04d}.json", record)
        records.append(record)
    return records


def write_failure_receipt(
    output: Path,
    *,
    attempt_id: str,
    source_commit: str,
    error: BaseException,
    started_at: float,
    completed_records: int,
) -> None:
    """Create a structured receipt while retaining all already-written rows."""
    record_directory = output / "state-records"
    partial_count = (
        len(list(record_directory.glob("state-*.json")))
        if record_directory.is_dir()
        else 0
    )
    _create_only_json(
        output / "failure.json",
        {
            "schema": "seed639-failure-v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": attempt_id,
            "source_commit": source_commit,
            "exception_type": type(error).__name__,
            "error": str(error),
            "completed_state_records": completed_records,
            "partial_record_count": partial_count,
            "elapsed_seconds": time.time() - started_at,
        },
    )


def run_attempt(attempt_dir: Path) -> dict[str, Any]:
    """Execute only after lifecycle and owner-merged registration verification."""
    from ml.alphazero_lite.seed639_authorization import (
        HISTORICAL_VERIFIER_RUNTIME,
        validate_attempt,
    )

    manifest, auth = validate_attempt(attempt_dir)
    attempt_dir = attempt_dir.resolve(strict=True)
    output = attempt_dir / "seed639-output"
    output.mkdir(mode=0o700)
    started = time.time()
    completed = 0
    try:
        registration = auth["registration"]
        historical_verifier_runtime = registration.get("execution_amendment", {}).get(
            "historical_verifier_runtime", HISTORICAL_VERIFIER_RUNTIME
        )
        runtime = runtime_identity()
        declared_runtime = manifest.get("runtime_versions", {})
        if (
            declared_runtime.get("numpy") != runtime["numpy_version"]
            or declared_runtime.get("python_version") != runtime["python_version"]
            or declared_runtime.get("numerical_backend") != runtime["backend"]
            or manifest.get("device", {}).get("type") != "cpu"
            or manifest.get("worker_count") != 1
            or manifest.get("configuration", {}).get("execution_concurrency") != 1
            or manifest.get("configuration", {}).get("inference_batch_size") != 1
            or manifest.get("configuration", {}).get("evaluation_cache") is not False
            or manifest.get("configuration", {}).get("numerical_backend")
            != runtime["backend"]
            or manifest.get("configuration", {}).get("thread_configuration")
            != {"environment": PINNED_ENVIRONMENT, "threads_per_pool": 1}
            or manifest.get("configuration", {}).get("execution_scope")
            != registration["execution_scope"]
            or manifest.get("configuration", {}).get("historical_verifier_runtime")
            != historical_verifier_runtime
        ):
            raise ValueError("seed639_frozen_runtime_configuration_mismatch")
        _create_only_json(output / "runtime.json", runtime)

        seed615_freeze_path = _input_path(
            registration, "/evaluation-preparation/execution-freeze-v4.json"
        )
        seed615_root = seed615_freeze_path.parents[4]
        input_auth = authenticate_inputs(
            seed615_root=seed615_root,
            selected_manifest_path=_input_path(
                registration, "/selected-state-manifest.json"
            ),
            selected_verification_path=_input_path(
                registration, "/selected-state-manifest-verification.json"
            ),
            archive_path=_input_path(registration, "attempt-0001.tar.zst"),
            archive_manifest_path=_input_path(
                registration, "attempt-0001.archive-manifest.json"
            ),
            historical_verifier_runtime=historical_verifier_runtime,
        )
        if registration.get("checkpoint_identity") != input_auth.get(
            "checkpoint_identity"
        ):
            raise ValueError("seed639_registered_checkpoint_identity_mismatch")
        population_rows = input_auth["selected_population_rows"]
        preflight = validate_population(population_rows)
        archive_verification = input_auth["seed636_archive_verification"]
        auth_receipt = {
            key: value
            for key, value in input_auth.items()
            if key != "selected_population_rows"
            and key != "seed636_archive_verification"
        }
        auth_receipt["seed636_archive_verification"] = {
            key: value for key, value in archive_verification.items() if key != "files"
        }
        _create_only_json(output / "authentication.json", auth_receipt)
        preflight_bindings = preflight.pop("state_bindings")
        preflight_path = output / "population-preflight.jsonl"
        with preflight_path.open("xb") as stream:
            for binding in preflight_bindings:
                stream.write(canonical_json(binding) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        _create_only_json(output / "population-preflight.json", preflight)

        evaluators = build_authenticated_evaluators(seed615_root, input_auth)
        record_directory = output / "state-records"
        record_directory.mkdir(mode=0o700)
        # Full-cohort preflight completed before either authenticated evaluator
        # is loaded or called. Each successful pair is durably written at once.
        records = persist_inference_records(
            population_rows,
            evaluators["B512"],
            evaluators["S615"],
            record_directory,
        )
        completed = len(records)

        subset = select_diagnostic_subset(records)
        summary = summarize_population(records)
        relationship = summarize_subset_relationship(records, subset)
        _create_only_json(output / "diagnostic-subset.json", subset)
        _create_only_json(output / "summary.json", summary)
        _create_only_json(output / "subset-relationship.json", relationship)
        result = {
            "schema": "seed639-policy-prior-classification-results-v1",
            "experiment_id": EXPERIMENT_ID,
            "attempt_id": manifest["attempt_id"],
            "source_commit": manifest["source_commit"],
            "registration_sha256": manifest["registration_sha256"],
            "selected_inventory_sha256": SELECTED_INVENTORY_SHA256,
            "states_evaluated": completed,
            "model_state_evaluations": completed * 2,
            "exposure_classification": input_auth["exposure_classification"],
            "interpretation": "descriptive policy differences only; no playing-strength claim",
            "diagnostic_subset": {
                "available": subset["available"],
                "selected_count": subset["selected_count"],
                "distinct_continuation_games": subset["distinct_continuation_games"],
                "distinct_opening_clusters": subset["distinct_opening_clusters"],
            },
            "summary": summary,
            "subset_relationship": relationship,
            "runtime": runtime,
            "historical_verifier_runtime": input_auth["seed615_authentication"][
                "historical_runtime"
            ],
            "completed_at_epoch_seconds": time.time(),
        }
        _create_only_json(output / "results.json", result)
        return result
    except BaseException as error:
        write_failure_receipt(
            output,
            attempt_id=manifest.get("attempt_id", "unknown"),
            source_commit=manifest.get("source_commit", "unknown"),
            error=error,
            started_at=started,
            completed_records=completed,
        )
        raise


def main() -> None:
    """Run through the standard lifecycle's frozen and started attempt."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--attempt-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_attempt_0002_preflight(), sort_keys=True, indent=2))
    elif args.command == "run":
        print(json.dumps(run_attempt(args.attempt_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
