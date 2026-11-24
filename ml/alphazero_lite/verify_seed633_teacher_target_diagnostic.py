"""Independent read-only verifier for Seed 633 policy-alignment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("docs/data/seed633-teacher-target-learning")
SNAPSHOTS = Path("docs/data/seed627-authenticated-membership/snapshots")
S599 = Path("docs/data/seed599-replay-coverage")
S615 = Path("docs/data/seed615-source-opening-diversity")
EVALUATION = S615 / "evaluation-preparation"
ARTIFACTS = EVALUATION / "runtime-artifacts"
EXPECTED_SNAPSHOT_MANIFEST = (
    "e7a6af69f72780ca5ca79f9fe8214b9d347b6c6c0fb7cfea87b0617e3a16d3f3"
)
EXPECTED_SNAPSHOTS = {
    "B512": "de7650a5d4f0d7a9dec11adb3d100689381a4b1539ec0c9749a0f82feda28d59",
    "S615": "8ed6f059d491fa4a6bfd90c942d6e02d05ef1b6a82db65969402ad4351a084fd",
    "protected_holdout": "4716976a654ecba4aadccfa336d0dc3c5fdbd354ac27786848a06485d6062610",
}
EXPECTED_CHECKPOINTS = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
EXPECTED_EVALUATION_FREEZE = (
    "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
)
SELF_PLAY_SOURCE = "ml/alphazero_lite/self_play.py"
EXPECTED_SELF_PLAY_SHA256 = (
    "76718efde7ccfb708f4bb0a4314f812bab746d1364cb75ce5c8b3551b635fc0c"
)
MEMBERSHIP_PATHS = {
    "B512": S599 / "treatment-membership.json",
    "S615": S615 / "candidate-membership.jsonl",
    "protected_holdout": S599 / "heldout-membership.json",
}
MODEL_METRICS = (
    "teacher_target_cross_entropy_nats",
    "teacher_to_policy_kl_nats",
    "teacher_top_action_agreement",
    "legal_action_probability_mass",
    "policy_entropy_nats",
    "model_confidence",
    "exact_optimal_action_mass",
    "exact_optimal_action_agreement",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _torch_runtime_identity() -> tuple[str, int]:
    """Read runtime provenance only; the verifier performs model math in NumPy."""
    import torch

    return str(torch.__version__), int(torch.get_num_threads())


def _members(name: str, root: Path) -> list[dict[str, Any]]:
    raw = (root / MEMBERSHIP_PATHS[name]).read_bytes()
    if name == "S615":
        return [json.loads(line) for line in raw.splitlines() if line]
    return json.loads(raw)


def _float_identity(state: dict[str, Any]) -> str:
    vector = np.asarray(encode_state(state, input_encoding="kalah_v3"), dtype="<f4")
    return digest(vector.tobytes())


def validate_mask(state: dict[str, Any], legal_moves: list[int]) -> None:
    """Reject a changed, incomplete, duplicate, or illegal action mask."""
    actual = sorted(int(move) for move in KalahGame.from_state(state).possible_moves())
    if not actual or legal_moves != actual or len(set(legal_moves)) != len(legal_moves):
        raise ValueError("seed633_legal_mask_invalid")


def validate_target(target: list[float], legal_moves: list[int]) -> None:
    """Validate normalized teacher mass and require all target mass to be legal."""
    values = np.asarray(target, dtype=np.float64)
    if (
        values.shape != (6,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError("seed633_teacher_target_invalid")
    illegal = sorted(set(range(6)) - set(legal_moves))
    if np.any(values[illegal] != 0):
        raise ValueError("seed633_teacher_target_illegal_mass")


def validate_cohort_rows(
    published: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    """Require exact frozen row identity, target, and membership reconstruction."""
    if published != expected:
        raise ValueError("seed633_cohort_membership_or_target_mismatch")


def reconstruct_cohort(root: Path) -> list[dict[str, Any]]:
    """Rebuild source rows and membership labels without using the producer."""
    root = root.resolve()
    manifest_path = root / SNAPSHOTS / "source-identity-lineage-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    if digest(manifest_raw) != EXPECTED_SNAPSHOT_MANIFEST:
        raise ValueError("seed627_manifest_identity_mismatch")
    manifest = json.loads(manifest_raw)
    snapshot_rows = {}
    for name, expected_hash in EXPECTED_SNAPSHOTS.items():
        raw = (root / SNAPSHOTS / f"{name}.json").read_bytes()
        if (
            digest(raw) != expected_hash
            or manifest["snapshot_file_sha256"].get(name) != expected_hash
        ):
            raise ValueError(f"seed627_snapshot_identity_mismatch:{name}")
        snapshot_rows[name] = json.loads(raw)
    source_rows = {name: _members(name, root) for name in MEMBERSHIP_PATHS}
    source_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for population, members in source_rows.items():
        mapping = {}
        for source in members:
            state = source.get("canonical_state")
            if not isinstance(state, dict):
                raise ValueError(
                    f"seed633_source_membership_state_invalid:{population}"
                )
            identity = digest(canonical(state))
            if identity in mapping or source.get("canonical_identity") != identity:
                raise ValueError(
                    f"seed633_source_membership_identity_invalid:{population}"
                )
            if source.get("float32_identity") != _float_identity(state):
                raise ValueError(
                    f"seed633_source_membership_float_identity_invalid:{population}"
                )
            mapping[identity] = source
        snapshot_ids = {
            row.get("canonical_identity") for row in snapshot_rows[population]
        }
        if set(mapping) != snapshot_ids:
            raise ValueError(f"seed633_snapshot_membership_join_invalid:{population}")
        source_maps[population] = mapping

    result = []
    all_ids = set().union(*(set(value) for value in source_maps.values()))
    for identity in sorted(all_ids):
        members = [name for name in ("B512", "S615") if identity in source_maps[name]]
        heldout = identity in source_maps["protected_holdout"]
        if heldout and members:
            raise ValueError("seed633_heldout_identity_overlap")
        label = (
            "shared_training"
            if len(members) == 2
            else "B512_only"
            if members == ["B512"]
            else "S615_only"
            if members == ["S615"]
            else "neither_training_membership"
        )
        sources = [source_maps[population][identity] for population in members]
        if heldout:
            sources.append(source_maps["protected_holdout"][identity])
        source = sources[0]
        invariant_fields = (
            "canonical_state",
            "float32_identity",
            "target",
            "target_sha256",
            "legal_moves",
            "phase",
            "source",
            "opening_id",
            "teacher_evidence",
        )
        for other in sources[1:]:
            if any(source.get(field) != other.get(field) for field in invariant_fields):
                raise ValueError("seed633_duplicate_state_source_conflict")
        state = source["canonical_state"]
        legal = sorted(int(move) for move in source["legal_moves"])
        validate_mask(state, legal)
        target = [float(value) for value in source["target"]]
        validate_target(target, legal)
        evidence = source.get("teacher_evidence") or {}
        teacher = evidence.get("teacher_source")
        if teacher not in {"artifact_puct", "exact_root_tablebase"}:
            raise ValueError("seed633_teacher_type_invalid")
        exact_actions = None
        if teacher == "exact_root_tablebase":
            margins = {
                int(action): int(margin)
                for action, margin in evidence["exact_action_margins"].items()
            }
            exact_actions = sorted(
                int(action) for action in evidence["exact_optimal_actions"]
            )
            winners = sorted(
                action
                for action, margin in margins.items()
                if margin == max(margins.values())
            )
            if exact_actions != winners or not set(exact_actions).issubset(legal):
                raise ValueError("seed633_exact_optimal_set_invalid")
        result.append(
            {
                "canonical_identity": identity,
                "float32_identity": _float_identity(state),
                "state": state,
                "target": target,
                "target_sha256": source["target_sha256"],
                "legal_moves": legal,
                "membership_class": label,
                "membership_populations": sorted(members),
                "authenticated_provenance_holdout": heldout,
                "source": str(source["source"]),
                "opening_id": str(source["opening_id"]),
                "source_opening_group": f"{source['source']}:{source['opening_id']}",
                "phase": str(source["phase"]),
                "teacher_type": teacher,
                "teacher_evidence": evidence,
                "exact_optimal_actions": exact_actions,
                "inherited_PE4_exposure": "unresolved",
            }
        )
    if len(result) != 4521:
        raise ValueError("seed633_reconstructed_cohort_count_mismatch")
    return result


def validate_numpy_artifact_binding(
    label: str,
    seed_record: dict[str, Any],
    evaluation_record: dict[str, Any],
    actual_directory: Path,
    root: Path,
    actual_hashes: dict[str, str],
    metadata: dict[str, Any],
    checkpoint_sha256: str,
) -> None:
    """Authenticate exports against the fixed Seed 615 evaluation freeze."""
    expected_directory = evaluation_record.get("directory")
    if (
        not isinstance(expected_directory, str)
        or seed_record.get("artifact_directory") != expected_directory
        or actual_directory.resolve() != (root / expected_directory).resolve()
    ):
        raise ValueError(f"seed633_artifact_directory_identity_mismatch:{label}")
    for field in ("model_npz_sha256", "weights_json_sha256", "metadata_sha256"):
        expected = evaluation_record.get(field)
        if seed_record.get(field) != expected or actual_hashes.get(field) != expected:
            raise ValueError(f"seed633_artifact_identity_mismatch:{label}:{field}")
    if (
        checkpoint_sha256 != EXPECTED_CHECKPOINTS[label]
        or actual_hashes.get("model_npz_sha256") != checkpoint_sha256
        or seed_record.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError(f"seed633_artifact_identity_mismatch:{label}")
    artifact_metadata = metadata.get("artifacts", {})
    if (
        metadata.get("framework") != "numpy"
        or artifact_metadata.get("weights_file") != "weights.json"
        or artifact_metadata.get("weights_sha256") != checkpoint_sha256
        or artifact_metadata.get("weights_json_sha256")
        != actual_hashes.get("weights_json_sha256")
    ):
        raise ValueError(f"seed633_artifact_metadata_binding_mismatch:{label}")


def validate_self_play_source_identity(
    actual_sha256: str, seed_freeze_sha256: str, evaluation_freeze_sha256: str
) -> None:
    """Pin the input encoder implementation to the authenticated Seed 615 freeze."""
    if (
        actual_sha256 != EXPECTED_SELF_PLAY_SHA256
        or seed_freeze_sha256 != EXPECTED_SELF_PLAY_SHA256
        or evaluation_freeze_sha256 != EXPECTED_SELF_PLAY_SHA256
    ):
        raise ValueError("seed633_self_play_source_identity_mismatch")


def _load_model(
    root: Path,
    label: str,
    freeze: dict[str, Any],
    evaluation_freeze: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = root / ARTIFACTS / label
    record = freeze["numpy_artifacts"][label]
    evaluation_record = evaluation_freeze["runtime_artifacts"][label]
    registered_model = evaluation_freeze["registration"]["models"][label]
    model_hash = digest((directory / "model.npz").read_bytes())
    weights_raw = (directory / "weights.json").read_bytes()
    metadata_raw = (directory / "metadata.json").read_bytes()
    metadata = json.loads(metadata_raw)
    checkpoint_sha256 = digest((root / registered_model["path"]).read_bytes())
    if registered_model.get("sha256") != EXPECTED_CHECKPOINTS[label]:
        raise ValueError(f"seed633_registered_checkpoint_identity_mismatch:{label}")
    actual_hashes = {
        "model_npz_sha256": model_hash,
        "weights_json_sha256": digest(weights_raw),
        "metadata_sha256": digest(metadata_raw),
    }
    validate_numpy_artifact_binding(
        label,
        record,
        evaluation_record,
        directory,
        root,
        actual_hashes,
        metadata,
        checkpoint_sha256,
    )
    return json.loads(weights_raw), metadata


def _forward(
    weights: dict[str, Any], metadata: dict[str, Any], state: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy reconstruction of the residual-v3 policy forward pass."""
    x = np.asarray(
        encode_state(state, input_encoding=metadata["input_encoding"]), dtype=np.float32
    )
    hidden = np.maximum(
        0.0,
        (x @ np.asarray(weights["w_input"], dtype=np.float32))
        + np.asarray(weights["b_input"], dtype=np.float32),
    )
    block = 1
    while f"w_residual_{block}_1" in weights:
        residual = hidden
        first = np.maximum(
            0.0,
            (hidden @ np.asarray(weights[f"w_residual_{block}_1"], dtype=np.float32))
            + np.asarray(weights[f"b_residual_{block}_1"], dtype=np.float32),
        )
        hidden = np.maximum(
            0.0,
            (first @ np.asarray(weights[f"w_residual_{block}_2"], dtype=np.float32))
            + np.asarray(weights[f"b_residual_{block}_2"], dtype=np.float32)
            + residual,
        )
        block += 1
    policy_hidden = np.maximum(
        0.0,
        (hidden @ np.asarray(weights["w_policy_hidden"], dtype=np.float32))
        + np.asarray(weights["b_policy_hidden"], dtype=np.float32),
    )
    logits = (
        policy_hidden @ np.asarray(weights["w_policy"], dtype=np.float32)
    ) + np.asarray(weights["b_policy"], dtype=np.float32)
    if "w_policy_adapter" in weights and "b_policy_adapter" in weights:
        logits = (
            logits
            + (hidden @ np.asarray(weights["w_policy_adapter"], dtype=np.float32))
            + np.asarray(weights["b_policy_adapter"], dtype=np.float32)
        )
    logits = logits - np.max(logits)
    raw = np.exp(logits)
    raw = raw / np.sum(raw)
    return raw, hidden


def _calculate_metrics(
    row: dict[str, Any], model_weights: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    state = row["state"]
    legal = row["legal_moves"]
    validate_mask(state, legal)
    target = np.asarray(row["target"], dtype=np.float64)
    validate_target(row["target"], legal)
    raw, _hidden = _forward(model_weights, metadata, state)
    mask = np.zeros(6, dtype=bool)
    mask[legal] = True
    legal_mass = float(np.asarray(raw, dtype=np.float64)[mask].sum())
    if not math.isfinite(legal_mass) or legal_mass <= 0:
        raise ValueError("seed633_nonpositive_legal_probability_mass")
    policy = np.zeros(6, dtype=np.float64)
    policy[mask] = np.asarray(raw, dtype=np.float64)[mask] / legal_mass
    top = int(np.argmax(policy))
    target_top = np.flatnonzero(target == np.max(target[legal]))
    positive_target = target > 0
    ce = float(-np.sum(target[positive_target] * np.log(policy[positive_target])))
    positive_policy = policy > 0
    entropy = float(-np.sum(policy[positive_policy] * np.log(policy[positive_policy])))
    teacher_entropy = float(
        -np.sum(target[positive_target] * np.log(target[positive_target]))
    )
    exact = row["exact_optimal_actions"]
    return {
        "raw_policy": np.asarray(raw, dtype=np.float64).tolist(),
        "legal_policy": policy.tolist(),
        "teacher_target_cross_entropy_nats": ce,
        "teacher_to_policy_kl_nats": ce - teacher_entropy,
        "teacher_top_action_agreement": bool(top in target_top),
        "teacher_target_top_actions": [int(action) for action in target_top],
        "legal_action_probability_mass": legal_mass,
        "policy_entropy_nats": entropy,
        "model_confidence": float(policy.max()),
        "exact_optimal_action_mass": None
        if exact is None
        else float(policy[exact].sum()),
        "exact_optimal_action_agreement": None if exact is None else top in exact,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute all published aggregate and cluster-bootstrap values independently."""
    fields = (
        "teacher_target_cross_entropy_nats",
        "teacher_to_policy_kl_nats",
        "teacher_top_action_agreement",
        "legal_action_probability_mass",
        "policy_entropy_nats",
        "model_confidence",
        "exact_optimal_action_mass",
        "exact_optimal_action_agreement",
    )

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"states": len(items)}
        if not items:
            return result
        for field in fields:
            for model in ("B512", "S615"):
                values = [
                    row[model][field] for row in items if row[model][field] is not None
                ]
                result[f"{model}_{field}"] = float(np.mean(values)) if values else None
            paired = [
                row
                for row in items
                if row["B512"][field] is not None and row["S615"][field] is not None
            ]
            result[f"S615_minus_B512_{field}"] = (
                float(
                    np.mean(
                        np.asarray(
                            [row["S615"][field] for row in paired], dtype=np.float64
                        )
                        - np.asarray(
                            [row["B512"][field] for row in paired], dtype=np.float64
                        )
                    )
                )
                if paired
                else None
            )
        result["source_opening_clusters"] = len(
            {row["source_opening_group"] for row in items}
        )
        result["exact_root_tablebase_states"] = sum(
            row["teacher_type"] == "exact_root_tablebase" for row in items
        )
        result["artifact_puct_states"] = sum(
            row["teacher_type"] == "artifact_puct" for row in items
        )
        return result

    dims: dict[str, dict[str, list[dict[str, Any]]]] = {
        "membership_class": defaultdict(list),
        "phase": defaultdict(list),
        "teacher_type": defaultdict(list),
        "legal_action_count": defaultdict(list),
        "teacher_entropy_quartile": defaultdict(list),
    }
    entropies = np.asarray(
        [
            -sum(float(x) * math.log(float(x)) for x in row["target"] if x > 0)
            for row in rows
        ],
        dtype=np.float64,
    )
    cuts = np.quantile(entropies, [0.25, 0.5, 0.75], method="linear")
    for idx, row in enumerate(rows):
        for dim, key in (
            ("membership_class", row["membership_class"]),
            ("phase", row["phase"]),
            ("teacher_type", row["teacher_type"]),
            ("legal_action_count", str(row["legal_action_count"])),
            (
                "teacher_entropy_quartile",
                str(
                    min(int(np.searchsorted(cuts, entropies[idx], side="right")) + 1, 4)
                ),
            ),
        ):
            dims[dim][key].append(row)
    grouping = {
        dim: {key: aggregate(value) for key, value in sorted(values.items())}
        for dim, values in dims.items()
    }
    rng = np.random.default_rng(633)
    intervals = {}
    subsets = {
        "all_states": rows,
        "authenticated_provenance_holdout": [
            row for row in rows if row["authenticated_provenance_holdout"]
        ],
    }
    subsets.update(
        {
            population: [row for row in rows if row["membership_class"] == population]
            for population in (
                "shared_training",
                "B512_only",
                "S615_only",
                "neither_training_membership",
            )
        }
    )
    for population, subset in subsets.items():
        clusters: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            clusters[row["source_opening_group"]].append(
                row["S615_minus_B512_cross_entropy_nats"]
            )
        names = sorted(clusters)
        counts = np.asarray([len(clusters[name]) for name in names], dtype=np.float64)
        sample = rng.integers(0, len(names), size=(20000, len(names)))
        totals = np.asarray([sum(clusters[name]) for name in names])[sample].sum(axis=1)
        denominators = counts[sample].sum(axis=1)
        quantiles = np.quantile(totals / denominators, [0.025, 0.975], method="linear")
        intervals[population] = {
            "clusters": len(names),
            "mean_delta_nats": float(
                np.mean([row["S615_minus_B512_cross_entropy_nats"] for row in subset])
            ),
            "interval": [float(value) for value in quantiles],
        }
    return {
        "schema": "seed633-teacher-target-learning-summary-v1",
        "rows": len(rows),
        "paired_comparison": "S615 minus B512 on identical frozen state-target rows",
        "by_membership_class": grouping["membership_class"],
        "by_phase": grouping["phase"],
        "by_teacher_type": grouping["teacher_type"],
        "by_legal_action_count": grouping["legal_action_count"],
        "by_teacher_entropy_quartile": grouping["teacher_entropy_quartile"],
        "source_opening_cluster_bootstrap_cross_entropy_delta": intervals,
        "bootstrap_protocol": {
            "resamples": 20000,
            "seed": 633,
            "quantiles": [0.025, 0.975],
            "quantile_method": "linear",
            "interpretation": "descriptive cluster stability, not independent-population inference",
        },
        "limitations": [
            "B512/S615 training changed both opening allocation and membership.",
            "P-E4 inherited training ancestry remains unresolved; source-held-out membership is not certified fully training-independent.",
            "Teacher imitation is not playing strength; artifact_puct is a search-teacher target, not exact optimality.",
            "Exact optimal-action metrics are emitted only for exact_root_tablebase rows with authenticated margins.",
        ],
    }


def verify_bundle(root: Path = ROOT, output: Path | None = None) -> dict[str, Any]:
    """Verify source identities, full frozen cohort, every metric, aggregates, and bootstrap."""
    root = root.resolve()
    output = (root / OUTPUT if output is None else output).resolve()
    freeze_raw = (output / "freeze.json").read_bytes()
    freeze = json.loads(freeze_raw)
    if (
        freeze.get("schema") != "seed633-teacher-target-cohort-freeze-v1"
        or freeze.get("status") != "frozen_before_inference"
    ):
        raise ValueError("seed633_protocol_freeze_invalid")
    evaluation_freeze_raw = (
        root / EVALUATION / "execution-freeze-v4.json"
    ).read_bytes()
    if digest(evaluation_freeze_raw) != EXPECTED_EVALUATION_FREEZE:
        raise ValueError("seed633_evaluation_freeze_hash_mismatch")
    evaluation_freeze = json.loads(evaluation_freeze_raw)
    for relative, expected in freeze["input_sha256"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError(f"seed633_input_hash_mismatch:{relative}")
    for relative, expected in freeze["source_sha256"].items():
        if digest((root / relative).read_bytes()) != expected:
            raise ValueError(f"seed633_source_hash_mismatch:{relative}")
    validate_self_play_source_identity(
        digest((root / SELF_PLAY_SOURCE).read_bytes()),
        freeze.get("source_sha256", {}).get(SELF_PLAY_SOURCE),
        evaluation_freeze.get("source_sha256", {}).get(SELF_PLAY_SOURCE),
    )
    runtime = freeze.get("runtime", {})
    torch_version, torch_threads = _torch_runtime_identity()
    observed_runtime = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch_version_identity_only": torch_version,
        "torch_thread_count_identity_only": torch_threads,
        "registered_evaluation_runtime": evaluation_freeze["runtime"],
        "platform": platform.platform(),
        "device": "cpu",
        "backend": "NumPy ArtifactEvaluator",
        "blas_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "torch_inference": False,
    }
    if runtime != observed_runtime:
        raise ValueError("seed633_inference_runtime_mismatch")
    if runtime.get("blas_thread_environment") != {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": "615",
        "CUDA_VISIBLE_DEVICES": "",
    }:
        raise ValueError("seed633_bounded_cpu_runtime_contract_mismatch")
    cohort_raw = (output / "cohort.jsonl").read_bytes()
    if digest(cohort_raw) != freeze["cohort"]["canonical_jsonl_sha256"]:
        raise ValueError("seed633_cohort_bytes_mismatch")
    published_cohort = [json.loads(line) for line in cohort_raw.splitlines() if line]
    expected_cohort = reconstruct_cohort(root)
    validate_cohort_rows(published_cohort, expected_cohort)
    models = {
        name: _load_model(root, name, freeze, evaluation_freeze)
        for name in ("B512", "S615")
    }
    metrics_raw = (output / "per-state-metrics.jsonl").read_bytes()
    metrics = [json.loads(line) for line in metrics_raw.splitlines() if line]
    if len(metrics) != len(expected_cohort):
        raise ValueError("seed633_metric_row_count_mismatch")
    rebuilt_metrics = []
    for cohort_row, metric_row in zip(expected_cohort, metrics, strict=True):
        if metric_row.get("canonical_identity") != cohort_row["canonical_identity"]:
            raise ValueError("seed633_metric_identity_order_mismatch")
        teacher_target = np.asarray(cohort_row["target"], dtype=np.float64)
        positive_target = teacher_target > 0
        teacher_entropy = float(
            -np.sum(
                teacher_target[positive_target]
                * np.log(teacher_target[positive_target])
            )
        )
        if metric_row.get("legal_action_count") != len(
            cohort_row["legal_moves"]
        ) or not math.isclose(
            metric_row.get("teacher_entropy_nats", math.nan),
            teacher_entropy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("seed633_per_state_target_derived_metric_mismatch")
        for field in (
            "float32_identity",
            "membership_class",
            "membership_populations",
            "authenticated_provenance_holdout",
            "source",
            "opening_id",
            "source_opening_group",
            "phase",
            "teacher_type",
            "target_sha256",
            "exact_optimal_actions",
            "target",
            "legal_moves",
            "teacher_evidence",
        ):
            if metric_row.get(field) != cohort_row.get(field):
                raise ValueError(f"seed633_metric_source_field_mismatch:{field}")
        rebuilt = {
            key: metric_row[key]
            for key in metric_row
            if key
            not in {
                "B512",
                "S615",
                "S615_minus_B512_cross_entropy_nats",
                "S615_minus_B512_kl_nats",
            }
        }
        for model in ("B512", "S615"):
            expected_model = _calculate_metrics(cohort_row, *models[model])
            if metric_row.get(model) != expected_model:
                raise ValueError(
                    f"seed633_per_state_model_metric_mismatch:{model}:{cohort_row['canonical_identity']}"
                )
            rebuilt[model] = expected_model
        rebuilt["S615_minus_B512_cross_entropy_nats"] = (
            rebuilt["S615"]["teacher_target_cross_entropy_nats"]
            - rebuilt["B512"]["teacher_target_cross_entropy_nats"]
        )
        rebuilt["S615_minus_B512_kl_nats"] = (
            rebuilt["S615"]["teacher_to_policy_kl_nats"]
            - rebuilt["B512"]["teacher_to_policy_kl_nats"]
        )
        rebuilt_metrics.append(rebuilt)
    if digest(metrics_raw) != load_json(output / "summary.json").get("metrics_sha256"):
        raise ValueError("seed633_metric_dataset_digest_mismatch")
    summary = load_json(output / "summary.json")
    if summary.get("freeze_sha256") != digest(freeze_raw) or summary.get(
        "cohort_sha256"
    ) != digest(cohort_raw):
        raise ValueError("seed633_summary_freeze_or_cohort_binding_mismatch")
    observed_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"freeze_sha256", "cohort_sha256", "metrics_sha256"}
    }
    if observed_summary != _summary(rebuilt_metrics):
        raise ValueError("seed633_summary_or_bootstrap_reconstruction_mismatch")
    return {
        "status": "verified",
        "cohort_rows": len(expected_cohort),
        "membership_class_counts": dict(
            sorted(Counter(row["membership_class"] for row in expected_cohort).items())
        ),
        "artifact_checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "independent_numpy_metric_reconstruction": True,
        "aggregate_and_bootstrap_reconstruction": True,
        "p_e4_ancestry": "unresolved",
        "playing_strength_claim": False,
        "seed615_evaluation_freeze_sha256": digest(evaluation_freeze_raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = verify_bundle(args.root, args.output)
    if args.write_report:
        output = (args.root / OUTPUT if args.output is None else args.output).resolve()
        path = output / "independent-verification.json"
        if path.exists():
            raise FileExistsError("seed633_verification_report_must_be_create_only")
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
