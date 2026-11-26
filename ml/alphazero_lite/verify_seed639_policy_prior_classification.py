"""Independent verification of Seed 639 per-state policy evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "seed639-prospective-policy-prior-classification"
SELECTED_INVENTORY_SHA256 = (
    "6c6b1f680b2177cb67c79a9a642216703f160131034fbf8fe86e8130910ed077"
)
REGISTERED_ENVIRONMENT = {
    "PYTHONHASHSEED": "615",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
EXPECTED_MODEL_FILES = {
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
MODEL_SOURCE_IDS = {
    "ml/alphazero_lite/arena.py": "c16a668e8f90bd87b59dbc08f6a1ddb3b5ffc912e21cc0564efaceba37e03fe2",
    "ml/alphazero_lite/self_play.py": "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c",
    "ml/alphazero_lite/input_encodings.py": "0a4f56ed60f0c2ed6b93a673dbc0850a966a907912dc26edd092bfd1ac209dbe",
    "ml/alphazero_lite/kalah_rules.py": "6e5338ef9a85b690f51b6d0d184c8bc0f54c2b28ac084a4a17a88f9f552ad036",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _entropy(probabilities: list[float]) -> float:
    vector = np.asarray(probabilities, dtype=np.float64)
    positive = vector[vector > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def require_exact_probability_vector(
    model: str, index: int, expected: np.ndarray, recorded: list[float]
) -> None:
    """Require bitwise-identical binary64 values from independent inference."""
    actual = np.asarray(recorded, dtype=np.float64)
    if (
        expected.shape != (6,)
        or actual.shape != (6,)
        or not np.array_equal(expected, actual)
    ):
        raise ValueError(f"seed639_verifier_vector_mismatch:{model}:{index}")


def require_exact_summary(recorded: dict[str, Any], reproduced: dict[str, Any]) -> None:
    """Reject any aggregate not exactly reproduced from per-state evidence."""
    if recorded != reproduced:
        raise ValueError("seed639_verifier_aggregate_summary_mismatch")


def verify_policy_records(
    population_rows: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Rebuild identities, legal masks, priors, TV, and action classes."""
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    targets = {17: 103, 18: 103, 19: 102, 20: 102, 21: 102}
    if len(population_rows) != 512 or len(records) != 512:
        raise ValueError("seed639_record_coverage_not_512")
    canonical_seen: set[str] = set()
    float_seen: set[str] = set()
    by_bin: Counter[int] = Counter()
    game_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    game_opening: dict[str, str] = {}
    state_bindings = []
    for index, (source, record) in enumerate(
        zip(population_rows, records, strict=True)
    ):
        state = source.get("state")
        if not isinstance(state, dict):
            raise ValueError(f"seed639_state_missing:{index}")
        canonical_id = _sha256(_canonical(state))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
        )
        float_id = _sha256(encoded.tobytes())
        if canonical_id != source.get(
            "canonical_identity"
        ) or canonical_id != record.get("canonical_identity"):
            raise ValueError(f"seed639_canonical_identity_mismatch:{index}")
        if float_id != source.get(
            "production_float32_identity"
        ) or float_id != record.get("production_float32_identity"):
            raise ValueError(f"seed639_float32_identity_mismatch:{index}")
        if canonical_id in canonical_seen or float_id in float_seen:
            raise ValueError(f"seed639_duplicate_identity:{index}")
        canonical_seen.add(canonical_id)
        float_seen.add(float_id)

        game = KalahGame.from_state(state)
        legal = [int(action) for action in game.possible_moves()]
        if legal != record.get("legal_actions") or legal != source.get("legal_actions"):
            raise ValueError(f"seed639_legal_actions_mismatch:{index}")
        for model in ("B512", "S615"):
            vector = record.get(f"{model}_legal_probabilities")
            if (
                not isinstance(vector, list)
                or len(vector) != 6
                or any(
                    not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                    for p in vector
                )
                or any(
                    vector[action] != 0.0 for action in range(6) if action not in legal
                )
                or not math.isclose(math.fsum(vector), 1.0, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(
                    f"seed639_invalid_masked_probabilities:{model}:{index}"
                )
        b = record["B512_legal_probabilities"]
        s = record["S615_legal_probabilities"]
        recorded_tv = float(record.get("total_variation", math.nan))
        recorded_b_entropy = float(record.get("B512_entropy_bits", math.nan))
        recorded_s_entropy = float(record.get("S615_entropy_bits", math.nan))
        recorded_entropy_delta = float(
            record.get("entropy_difference_S615_minus_B512_bits", math.nan)
        )
        b_top = max(range(6), key=lambda action: (b[action], -action))
        s_top = max(range(6), key=lambda action: (s[action], -action))
        tv = float(
            0.5
            * np.abs(
                np.asarray(b, dtype=np.float64) - np.asarray(s, dtype=np.float64)
            ).sum()
        )
        if (
            b_top != record.get("B512_top_action")
            or s_top != record.get("S615_top_action")
            or (b_top != s_top) != record.get("top_action_disagreement")
            or tv != recorded_tv
            or _entropy(b) != recorded_b_entropy
            or _entropy(s) != recorded_s_entropy
            or _entropy(s) - _entropy(b) != recorded_entropy_delta
        ):
            raise ValueError(f"seed639_policy_metric_mismatch:{index}")
        for provenance_key in (
            "active_pit_stones",
            "game_id",
            "opening_identity",
            "opening_index",
            "continuation_ply",
        ):
            if record.get(provenance_key) != source.get(provenance_key):
                raise ValueError(
                    f"seed639_provenance_mismatch:{index}:{provenance_key}"
                )
        game_id = str(source["game_id"])
        opening_id = str(source["opening_identity"])
        if game_opening.setdefault(game_id, opening_id) != opening_id:
            raise ValueError(f"seed639_game_opening_conflict:{index}")
        game_counts[game_id] += 1
        opening_counts[opening_id] += 1
        if (
            record.get("population_index") != index
            or record.get("side_to_move") != game.current_player
        ):
            raise ValueError(f"seed639_population_order_or_side_mismatch:{index}")
        stones = sum(game.pits)
        if stones != source.get("active_pit_stones"):
            raise ValueError(f"seed639_stone_bin_mismatch:{index}")
        by_bin[stones] += 1
        state_bindings.append(
            {
                "population_index": index,
                "canonical_identity": canonical_id,
                "production_float32_identity": float_id,
                "legal_actions": legal,
                "active_pit_stones": stones,
                "game_id": game_id,
                "opening_identity": opening_id,
                "opening_index": source["opening_index"],
                "continuation_ply": source["continuation_ply"],
                "side_to_move": int(game.current_player),
            }
        )
    if dict(by_bin) != targets:
        raise ValueError("seed639_stone_bin_quotas_mismatch")
    if (
        max(game_counts.values(), default=0) > 2
        or max(opening_counts.values(), default=0) > 4
        or len(opening_counts) < 128
    ):
        raise ValueError("seed639_cluster_caps_or_coverage_mismatch")
    return {
        "passed": True,
        "states_verified": 512,
        "canonical_identities": len(canonical_seen),
        "production_float32_identities": len(float_seen),
        "stone_bin_counts": {str(key): by_bin[key] for key in sorted(targets)},
        "continuation_games": len(game_counts),
        "opening_clusters": len(opening_counts),
        "state_bindings": state_bindings,
    }


def verify_subset_selection(
    records: list[dict[str, Any]], selected_state_ids: list[str]
) -> dict[str, Any]:
    """Independently reproduce the registered greedy TV-tail selection."""
    if len(records) != 512:
        raise ValueError("seed639_subset_source_count_not_512")
    ranked = sorted(
        records,
        key=lambda row: (
            -float(row["total_variation"]),
            _sha256(str(row["canonical_identity"]).encode("utf-8")),
            str(row["canonical_identity"]),
        ),
    )
    chosen: list[str] = []
    games: set[str] = set()
    openings: Counter[str] = Counter()
    for row in ranked:
        game_id = str(row["game_id"])
        opening_id = str(row["opening_identity"])
        if game_id in games or openings[opening_id] >= 4:
            continue
        chosen.append(str(row["canonical_identity"]))
        games.add(game_id)
        openings[opening_id] += 1
        if len(chosen) == 64:
            break
    available = len(chosen) == 64 and len(games) >= 32
    expected_ids = chosen if available else []
    if selected_state_ids != expected_ids:
        raise ValueError("seed639_secondary_subset_mismatch")
    return {
        "passed": True,
        "available": available,
        "selected_count": len(expected_ids),
        "distinct_continuation_games": len(games) if available else 0,
        "distinct_opening_clusters": len(openings) if available else 0,
    }


def _summary_independent(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute published statistics without calling the producer summary."""
    if len(rows) != 512:
        raise ValueError("seed639_summary_input_count_not_512")
    weights = {17: 0.2, 18: 0.2, 19: 0.2, 20: 0.2, 21: 0.2}
    bins = {
        stones: [row for row in rows if row["active_pit_stones"] == stones]
        for stones in weights
    }
    if {stones: len(group) for stones, group in bins.items()} != {
        17: 103,
        18: 103,
        19: 102,
        20: 102,
        21: 102,
    }:
        raise ValueError("seed639_independent_bin_quota_mismatch")

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        tv = np.asarray([row["total_variation"] for row in group], dtype=np.float64)
        disagree = np.asarray(
            [row["top_action_disagreement"] for row in group], dtype=np.float64
        )
        hb = np.asarray([row["B512_entropy_bits"] for row in group])
        hs = np.asarray([row["S615_entropy_bits"] for row in group])
        delta = np.asarray(
            [row["entropy_difference_S615_minus_B512_bits"] for row in group]
        )
        return {
            "states": len(group),
            "top_action_disagreements": int(disagree.sum()),
            "top_action_disagreement_rate": float(disagree.mean()),
            "mean_total_variation": float(tv.mean()),
            "mean_B512_entropy_bits": float(hb.mean()),
            "mean_S615_entropy_bits": float(hs.mean()),
            "mean_entropy_difference_S615_minus_B512_bits": float(delta.mean()),
        }

    all_tv = np.asarray([row["total_variation"] for row in rows], dtype=np.float64)
    all_disagree = np.asarray(
        [row["top_action_disagreement"] for row in rows], dtype=np.float64
    )
    all_hb = np.asarray([row["B512_entropy_bits"] for row in rows])
    all_hs = np.asarray([row["S615_entropy_bits"] for row in rows])
    all_delta = np.asarray(
        [row["entropy_difference_S615_minus_B512_bits"] for row in rows]
    )
    per_bin = {str(stones): stats(group) for stones, group in bins.items()}
    standardized = {
        name: float(
            sum(weights[stones] * per_bin[str(stones)][name] for stones in weights)
        )
        for name in (
            "top_action_disagreement_rate",
            "mean_total_variation",
            "mean_entropy_difference_S615_minus_B512_bits",
        )
    }
    return {
        "population_size": len(rows),
        "overall": {
            "top_action_disagreements": int(all_disagree.sum()),
            "top_action_disagreement_rate": float(all_disagree.mean()),
            "total_variation_mean": float(all_tv.mean()),
            "total_variation_quantiles": {
                str(q): float(np.quantile(all_tv, q))
                for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
            },
            "B512_mean_entropy_bits": float(all_hb.mean()),
            "S615_mean_entropy_bits": float(all_hs.mean()),
            "entropy_difference_S615_minus_B512_mean_bits": float(all_delta.mean()),
        },
        "by_stone_bin": per_bin,
        "fixed_one_fifth_bin_standardized": standardized,
    }


def _subset_relationship_independent(
    rows: list[dict[str, Any]], selected_ids: list[str]
) -> dict[str, Any]:
    if not selected_ids:
        return {
            "available": False,
            "reason": "fewer_than_64_selectable_states_or_32_distinct_games",
            "primary_population_size": len(rows),
            "subset_size": 0,
        }
    by_id = {row["canonical_identity"]: row for row in rows}
    if (
        len(by_id) != 512
        or len(selected_ids) != 64
        or any(identity not in by_id for identity in selected_ids)
    ):
        raise ValueError("seed639_independent_subset_primary_binding_invalid")

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        bins = Counter(int(row["active_pit_stones"]) for row in group)
        return {
            "states": len(group),
            "stone_bin_counts": {
                str(stones): bins[stones] for stones in (17, 18, 19, 20, 21)
            },
            "distinct_continuation_games": len({row["game_id"] for row in group}),
            "distinct_opening_clusters": len(
                {row["opening_identity"] for row in group}
            ),
            "top_action_disagreement_rate": float(
                np.mean([row["top_action_disagreement"] for row in group])
            ),
            "mean_total_variation": float(
                np.mean([row["total_variation"] for row in group])
            ),
            "mean_entropy_difference_S615_minus_B512_bits": float(
                np.mean(
                    [row["entropy_difference_S615_minus_B512_bits"] for row in group]
                )
            ),
        }

    subset = [by_id[identity] for identity in selected_ids]
    return {
        "available": True,
        "primary_population_size": len(rows),
        "subset_size": len(subset),
        "subset_is_unweighted_descriptive_only": True,
        "primary_population": describe(rows),
        "diagnostic_subset": describe(subset),
    }


def verify_attempt(
    attempt_dir: Path,
    *,
    associated_prs: list[dict[str, Any]] | None = None,
    pr_fetcher: Any = None,
) -> dict[str, Any]:
    """Run the independent, authenticated verifier over a completed attempt."""
    from ml.alphazero_lite import arena
    from ml.alphazero_lite import experiment_lifecycle
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.seed639_authorization import validate_attempt

    manifest, authorization = validate_attempt(
        attempt_dir, associated_prs=associated_prs, pr_fetcher=pr_fetcher
    )
    attempt_dir = attempt_dir.resolve(strict=True)
    lifecycle_check = experiment_lifecycle.verify_attempt(attempt_dir)
    if (
        not lifecycle_check.get("integrity_verified")
        or lifecycle_check.get("status") != "completed"
    ):
        raise ValueError("seed639_lifecycle_attempt_verification_failed")
    output_dir = attempt_dir / "seed639-output"
    if not output_dir.is_dir() or (output_dir / "failure.json").exists():
        raise ValueError("seed639_attempt_not_completed")

    registration = authorization["registration"]
    root = Path(__file__).resolve().parents[2]
    for relative, digest in MODEL_SOURCE_IDS.items():
        if _sha256((root / relative).read_bytes()) != digest:
            raise ValueError(f"seed639_verifier_model_source_mismatch:{relative}")
    input_map = registration["input_artifacts"]

    def input_path(suffix: str) -> Path:
        paths = [Path(path) for path in input_map if path.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError(f"seed639_verifier_input_ambiguous:{suffix}")
        return paths[0]

    selected_manifest_path = input_path("/selected-state-manifest.json")
    selected_verification_path = input_path(
        "/selected-state-manifest-verification.json"
    )
    archive_path = input_path("attempt-0001.tar.zst")
    sidecar_path = input_path("attempt-0001.archive-manifest.json")
    manifest_doc = json.loads(selected_manifest_path.read_bytes())
    selected_verification = json.loads(selected_verification_path.read_bytes())
    if (
        _sha256(selected_manifest_path.read_bytes())
        != "d7e095f749c14a429162be91dd3bb5f67b685fab04808658f7f26ac7882c8fab"
        or _sha256(selected_verification_path.read_bytes())
        != "4c4c177d2fd4bf8b4e0b37e65584b9c6a62ca1949c6c114b6bf891fefd15c09a"
        or manifest_doc.get("selected_inventory_sha256") != SELECTED_INVENTORY_SHA256
        or selected_verification.get("status") != "verified"
    ):
        raise ValueError("seed639_verifier_seed636_manifest_invalid")
    archive_auth = experiment_lifecycle.verify_archive(archive_path, sidecar_path)
    if (
        archive_auth.get("experiment_id") != "seed636-prospective-population"
        or archive_auth.get("attempt_id") != "attempt-0001"
    ):
        raise ValueError("seed639_verifier_seed636_archive_identity_invalid")
    member = next(
        (
            row
            for row in archive_auth["files"]
            if row["path"] == "population/population.json"
        ),
        None,
    )
    if member is None or member.get("sha256") != selected_verification.get(
        "population_member_sha256"
    ):
        raise ValueError("seed639_verifier_population_member_binding_invalid")
    with tarfile.open(archive_path, mode="r:*") as archive:
        stream = archive.extractfile("population/population.json")
        if stream is None:
            raise ValueError("seed639_verifier_population_member_missing")
        population_raw = stream.read()
    if _sha256(population_raw) != member["sha256"]:
        raise ValueError("seed639_verifier_population_member_hash_mismatch")
    population_bundle = json.loads(population_raw)
    population = population_bundle.get("selection", {}).get("states")
    if (
        not isinstance(population, list)
        or len(population) != 512
        or _sha256(_canonical(population)) != SELECTED_INVENTORY_SHA256
    ):
        raise ValueError("seed639_verifier_selected_inventory_mismatch")
    manifest_states = manifest_doc.get("states")
    if not isinstance(manifest_states, list) or len(manifest_states) != 512:
        raise ValueError("seed639_verifier_manifest_state_count_invalid")
    fields = (
        "canonical_identity",
        "production_float32_identity",
        "active_pit_stones",
        "game_id",
        "opening_identity",
        "opening_index",
        "continuation_ply",
    )
    for index, (manifest_row, population_row) in enumerate(
        zip(manifest_states, population, strict=True)
    ):
        if manifest_row.get("i") != index or any(
            manifest_row.get(key) != population_row.get(key) for key in fields
        ):
            raise ValueError(f"seed639_verifier_manifest_population_mismatch:{index}")
    manifest_game_counts = Counter(row["game_id"] for row in population)
    manifest_opening_counts = Counter(row["opening_identity"] for row in population)
    if (
        len(manifest_opening_counts) != manifest_doc.get("distinct_opening_clusters")
        or max(manifest_game_counts.values()) != manifest_doc.get("max_per_game")
        or max(manifest_opening_counts.values()) != manifest_doc.get("max_per_opening")
    ):
        raise ValueError("seed639_verifier_manifest_cluster_summary_mismatch")

    seed615_freeze_path = input_path("/execution-preparation/execution-freeze-v4.json")
    seed615_root = seed615_freeze_path.parents[4]
    from ml.alphazero_lite import seed615_policy_divergence_diagnostic as seed615

    historical = seed615.authenticate(seed615_root)
    if (
        historical.get("freeze_sha256")
        != "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
    ):
        raise ValueError("seed639_verifier_seed615_freeze_mismatch")
    artifact_root = (
        seed615_root
        / "docs/data/seed615-source-opening-diversity/evaluation-preparation/runtime-artifacts"
    )
    evaluators = {}
    for model, expected_files in EXPECTED_MODEL_FILES.items():
        artifact_dir = artifact_root / model
        actual = {
            name: _sha256((artifact_dir / name).read_bytes()) for name in expected_files
        }
        if actual != expected_files:
            raise ValueError(f"seed639_verifier_checkpoint_identity_mismatch:{model}")
        metadata = json.loads((artifact_dir / "metadata.json").read_bytes())
        if (
            metadata.get("framework") != "numpy"
            or metadata.get("input_encoding") != "kalah_v3"
            or metadata.get("artifacts", {}).get("weights_sha256")
            != expected_files["model.npz"]
            or metadata.get("artifacts", {}).get("weights_json_sha256")
            != expected_files["weights.json"]
        ):
            raise ValueError(f"seed639_verifier_checkpoint_metadata_mismatch:{model}")
        evaluator = arena.ArtifactEvaluator(artifact_dir)
        if (
            type(evaluator) is not arena.ArtifactEvaluator
            or evaluator.input_encoding != "kalah_v3"
        ):
            raise ValueError(f"seed639_verifier_evaluator_identity_mismatch:{model}")
        evaluators[model] = evaluator

    expected_env = REGISTERED_ENVIRONMENT
    actual_env = {key: os.environ.get(key) for key in expected_env}
    if (
        actual_env != expected_env
        or platform.python_version() != "3.14.7"
        or np.__version__ != "2.5.3"
    ):
        raise ValueError("seed639_verifier_numerical_runtime_mismatch")
    observed_runtime = {
        "python_version": platform.python_version(),
        "python_full_version": __import__("sys").version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "backend": "NumPy ArtifactEvaluator",
        "numpy_build_configuration": np.show_config(mode="dicts"),
        "thread_configuration": {"environment": actual_env, "limit_per_pool": 1},
        "environment": actual_env,
        "inference_batch_size": 1,
        "evaluation_cache": False,
    }
    if json.loads((output_dir / "runtime.json").read_bytes()) != observed_runtime:
        raise ValueError("seed639_verifier_runtime_record_mismatch")

    record_dir = output_dir / "state-records"
    record_paths = sorted(record_dir.glob("state-*.json"))
    if len(record_paths) != 512 or [p.name for p in record_paths] != [
        f"state-{index:04d}.json" for index in range(512)
    ]:
        raise ValueError("seed639_verifier_record_coverage_invalid")
    records = [json.loads(path.read_bytes()) for path in record_paths]
    verifier_population = verify_policy_records(population, records)
    preflight_path = output_dir / "population-preflight.jsonl"
    preflight_rows = [
        json.loads(line)
        for line in preflight_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if preflight_rows != verifier_population["state_bindings"]:
        raise ValueError("seed639_verifier_population_preflight_evidence_mismatch")

    # Independent second forward pass: vectors must match exact binary64 values.
    for index, (source, record) in enumerate(zip(population, records, strict=True)):
        game = KalahGame.from_state(source["state"])
        legal = [int(action) for action in game.possible_moves()]
        legal_mask = np.zeros(6, dtype=bool)
        legal_mask[legal] = True
        for model in ("B512", "S615"):
            policy, _ = evaluators[model].evaluate(game)
            vector = np.asarray(policy, dtype=np.float64)
            vector[~legal_mask] = 0.0
            mass = float(vector.sum())
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"seed639_verifier_inference_invalid:{model}:{index}")
            vector /= mass
            require_exact_probability_vector(
                model, index, vector, record[f"{model}_legal_probabilities"]
            )

    subset_doc = json.loads((output_dir / "diagnostic-subset.json").read_bytes())
    selected_ids = subset_doc.get("selected_state_ids")
    subset_check = verify_subset_selection(records, selected_ids)
    selected_rows = [
        next(row for row in records if row["canonical_identity"] == identity)
        for identity in selected_ids
    ]
    opening_counts = Counter(row["opening_identity"] for row in selected_rows)
    if (
        subset_doc.get("available") != subset_check["available"]
        or subset_doc.get("selected_count") != len(selected_ids)
        or subset_doc.get("distinct_continuation_games")
        != len({row["game_id"] for row in selected_rows})
        or subset_doc.get("distinct_opening_clusters") != len(opening_counts)
        or subset_doc.get("states_per_opening_cluster")
        != dict(sorted(opening_counts.items()))
    ):
        raise ValueError("seed639_verifier_subset_composition_mismatch")
    expected_subset_rows = [
        next(row for row in records if row["canonical_identity"] == identity)
        for identity in selected_ids
    ]
    if subset_doc.get("selected_rows") != expected_subset_rows:
        raise ValueError("seed639_verifier_subset_row_or_order_mismatch")

    expected_summary = _summary_independent(records)
    recorded_summary = json.loads((output_dir / "summary.json").read_bytes())
    require_exact_summary(recorded_summary, expected_summary)
    expected_relationship = _subset_relationship_independent(records, selected_ids)
    recorded_relationship = json.loads(
        (output_dir / "subset-relationship.json").read_bytes()
    )
    if recorded_relationship != expected_relationship:
        raise ValueError("seed639_verifier_subset_relationship_mismatch")
    results = json.loads((output_dir / "results.json").read_bytes())
    if (
        results.get("states_evaluated") != 512
        or results.get("model_state_evaluations") != 1024
        or results.get("selected_inventory_sha256") != SELECTED_INVENTORY_SHA256
        or results.get("summary") != expected_summary
        or results.get("subset_relationship") != expected_relationship
        or results.get("diagnostic_subset", {}).get("available")
        != subset_check["available"]
    ):
        raise ValueError("seed639_verifier_results_summary_mismatch")
    return {
        "passed": True,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": manifest["attempt_id"],
        "states_verified": 512,
        "model_state_evaluations_reproduced": 1024,
        "selected_inventory_sha256": SELECTED_INVENTORY_SHA256,
        "population_preflight_verified": 512,
        "subset_verification": subset_check,
        "summary_verified": True,
        "exposure_classification": manifest_doc["exposure_classification"],
    }


def main() -> None:
    """Run the independent verifier against a completed lifecycle attempt."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_attempt(args.attempt_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
