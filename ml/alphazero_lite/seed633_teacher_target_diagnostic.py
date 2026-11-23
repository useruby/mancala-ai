"""Read-only NumPy policy alignment diagnostic for authenticated Seed 633 states."""

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

from ml.alphazero_lite import arena
from ml.alphazero_lite import verify_seed627_membership_snapshots as snapshot_verifier
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed622_training_provenance_audit import float32_identity

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("docs/data/seed633-teacher-target-learning")
SNAPSHOTS = Path("docs/data/seed627-authenticated-membership/snapshots")
S599 = Path("docs/data/seed599-replay-coverage")
S615 = Path("docs/data/seed615-source-opening-diversity")
EVALUATION = S615 / "evaluation-preparation"
ARTIFACTS = EVALUATION / "runtime-artifacts"
ACTIVE_EVALUATION_FREEZE_SHA256 = (
    "ba30f5e3efadf3358db9f03c7b4eba4f69b23e2f95dcb4089723a1fb9664f32b"
)
CHECKPOINT_SHA256 = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
}
SNAPSHOT_FILE_SHA256 = {
    "B512": "de7650a5d4f0d7a9dec11adb3d100689381a4b1539ec0c9749a0f82feda28d59",
    "S615": "8ed6f059d491fa4a6bfd90c942d6e02d05ef1b6a82db65969402ad4351a084fd",
    "protected_holdout": "4716976a654ecba4aadccfa336d0dc3c5fdbd354ac27786848a06485d6062610",
}
SNAPSHOT_MANIFEST_SHA256 = (
    "e7a6af69f72780ca5ca79f9fe8214b9d347b6c6c0fb7cfea87b0617e3a16d3f3"
)
METRIC_DEFINITIONS = {
    "teacher_target_cross_entropy": "-sum(t[a] * log(p_legal[a])) over positive teacher target entries, nats",
    "teacher_to_policy_kl": "cross_entropy - entropy(teacher_target), nats",
    "teacher_top_action_agreement": "predicted legal argmax is among all legal teacher-target maxima",
    "legal_action_probability_mass": "sum of unmasked six-action model softmax on the legal action set",
    "policy_entropy": "entropy of the model policy after legal-action masking and renormalization, nats",
    "model_confidence": "maximum legal-normalized model action probability",
    "exact_optimal_action_mass": "legal-normalized probability on authenticated exact-optimal actions; null for artifact_puct",
    "exact_optimal_action_agreement": "predicted legal argmax is an authenticated exact-optimal action; null for artifact_puct",
    "delta_direction": "S615 minus B512; negative cross-entropy/KL indicates closer teacher alignment",
}
SOURCE_PATHS = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/seed615_evaluation.py",
    "ml/alphazero_lite/seed615_policy_divergence_diagnostic.py",
    "ml/alphazero_lite/seed599_training_executor.py",
    "ml/alphazero_lite/seed622_training_provenance_audit.py",
    "ml/alphazero_lite/verify_seed599_training_result.py",
    "ml/alphazero_lite/verify_seed615_evaluation.py",
    "ml/alphazero_lite/verify_seed615_opening_diversity.py",
    "ml/alphazero_lite/verify_seed615_training_result.py",
    "ml/alphazero_lite/verify_seed622_training_provenance_audit.py",
    "ml/alphazero_lite/verify_seed627_membership_snapshots.py",
    "ml/alphazero_lite/seed633_teacher_target_diagnostic.py",
    "ml/alphazero_lite/verify_seed633_teacher_target_diagnostic.py",
)
MEMBERSHIP_PATHS = {
    "B512": S599 / "treatment-membership.json",
    "S615": S615 / "candidate-membership.jsonl",
    "protected_holdout": S599 / "heldout-membership.json",
}


def sha256_bytes(raw: bytes) -> str:
    """Return a SHA-256 hex digest."""
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for evidence identity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _torch_runtime_identity() -> tuple[str, int]:
    """Read the pinned evaluation build/thread identity; never execute a tensor."""
    import torch

    return str(torch.__version__), int(torch.get_num_threads())


def read_membership(name: str, path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if name == "S615":
        return [json.loads(line) for line in raw.splitlines() if line]
    return json.loads(raw)


def _snapshot_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    manifest_path = root / SNAPSHOTS / "source-identity-lineage-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    if sha256_bytes(manifest_raw) != SNAPSHOT_MANIFEST_SHA256:
        raise ValueError("seed627_snapshot_manifest_hash_mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("status") != "authenticated":
        raise ValueError("seed627_snapshot_manifest_not_authenticated")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, expected in SNAPSHOT_FILE_SHA256.items():
        path = root / SNAPSHOTS / f"{name}.json"
        raw = path.read_bytes()
        if (
            sha256_bytes(raw) != expected
            or manifest["snapshot_file_sha256"].get(name) != expected
        ):
            raise ValueError(f"seed627_snapshot_hash_mismatch:{name}")
        rows[name] = json.loads(raw)
    return rows


def build_cohort(root: Path = ROOT) -> list[dict[str, Any]]:
    """Reconstruct all authenticated B512/S615/holdout rows and classifications."""
    root = root.resolve()
    snapshots = _snapshot_rows(root)
    reconstructed, lineage, _input_hashes = snapshot_verifier.reconstruct()
    for name, expected_rows in reconstructed.items():
        if snapshots[name] != expected_rows:
            raise ValueError(f"seed627_snapshot_reconstruction_mismatch:{name}")
    populations = {
        name: read_membership(name, root / path)
        for name, path in MEMBERSHIP_PATHS.items()
    }
    snapshot_ids = {
        name: {row["canonical_identity"] for row in rows}
        for name, rows in snapshots.items()
    }
    source_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in populations.items():
        source_maps[name] = {}
        for row in rows:
            identity = row.get("canonical_identity")
            if not isinstance(identity, str) or identity in source_maps[name]:
                raise ValueError(
                    f"seed633_source_membership_duplicate_or_missing:{name}"
                )
            source_maps[name][identity] = row
        if set(source_maps[name]) != snapshot_ids[name]:
            raise ValueError(f"seed633_snapshot_membership_join_mismatch:{name}")

    records: list[dict[str, Any]] = []
    all_ids = set().union(*snapshot_ids.values())
    for identity in sorted(all_ids):
        memberships = [
            name for name in ("B512", "S615") if identity in source_maps[name]
        ]
        holdout = identity in source_maps["protected_holdout"]
        if holdout and memberships:
            raise ValueError("seed633_holdout_training_membership_overlap")
        classification = (
            "shared_training"
            if len(memberships) == 2
            else "B512_only"
            if memberships == ["B512"]
            else "S615_only"
            if memberships == ["S615"]
            else "neither_training_membership"
        )
        candidates = [source_maps[name][identity] for name in memberships]
        if holdout:
            candidates.append(source_maps["protected_holdout"][identity])
        row = candidates[0]
        for other in candidates[1:]:
            for field in (
                "canonical_state",
                "float32_identity",
                "target",
                "target_sha256",
                "legal_moves",
                "phase",
                "source",
                "opening_id",
                "teacher_evidence",
            ):
                if row.get(field) != other.get(field):
                    raise ValueError(
                        f"seed633_shared_identity_source_disagreement:{field}"
                    )
        state = row.get("canonical_state")
        if not isinstance(state, dict) or identity != sha256_bytes(
            canonical_json(state)
        ):
            raise ValueError("seed633_canonical_identity_mismatch")
        computed_float32 = float32_identity(state)
        if computed_float32 != row.get("float32_identity"):
            raise ValueError("seed633_float32_identity_mismatch")
        target = np.asarray(row.get("target"), dtype=np.float64)
        if (
            target.shape != (6,)
            or not np.all(np.isfinite(target))
            or np.any(target < 0)
        ):
            raise ValueError("seed633_teacher_target_shape_or_value_invalid")
        if not np.isclose(float(target.sum()), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("seed633_teacher_target_not_normalized")
        game = KalahGame.from_state(state)
        legal = sorted(int(move) for move in game.possible_moves())
        if legal != sorted(row.get("legal_moves", [])) or not legal:
            raise ValueError("seed633_authenticated_legal_moves_mismatch")
        if np.any(target[[index for index in range(6) if index not in legal]] != 0):
            raise ValueError("seed633_teacher_target_has_illegal_mass")
        evidence = row.get("teacher_evidence") or {}
        teacher_type = evidence.get("teacher_source")
        if teacher_type not in {"artifact_puct", "exact_root_tablebase"}:
            raise ValueError("seed633_unknown_teacher_type")
        exact_actions = None
        if teacher_type == "exact_root_tablebase":
            exact_actions = sorted(
                int(action) for action in evidence.get("exact_optimal_actions", [])
            )
            margins = {
                int(key): int(value)
                for key, value in evidence["exact_action_margins"].items()
            }
            max_margin = max(margins.values())
            if exact_actions != sorted(
                action for action, margin in margins.items() if margin == max_margin
            ):
                raise ValueError("seed633_exact_optimal_actions_do_not_match_margins")
            if not exact_actions or not set(exact_actions).issubset(legal):
                raise ValueError("seed633_exact_optimal_actions_invalid")
        source = str(row["source"])
        opening_id = str(row["opening_id"])
        records.append(
            {
                "canonical_identity": identity,
                "float32_identity": computed_float32,
                "state": state,
                "target": [float(value) for value in target],
                "target_sha256": row["target_sha256"],
                "legal_moves": legal,
                "membership_class": classification,
                "membership_populations": sorted(memberships),
                "authenticated_provenance_holdout": holdout,
                "source": source,
                "opening_id": opening_id,
                "source_opening_group": f"{source}:{opening_id}",
                "phase": str(row["phase"]),
                "teacher_type": teacher_type,
                "teacher_evidence": evidence,
                "exact_optimal_actions": exact_actions,
                "inherited_PE4_exposure": "unresolved",
            }
        )
    if len(records) != 4521:
        raise ValueError("seed633_cohort_count_mismatch")
    return records


def artifact_identity(
    root: Path, label: str, evaluation_freeze: dict[str, Any]
) -> dict[str, Any]:
    directory = root / ARTIFACTS / label
    frozen = evaluation_freeze["runtime_artifacts"][label]
    registration = evaluation_freeze["registration"]["models"][label]
    checkpoint_path = root / registration["path"]
    result = {
        "checkpoint_sha256": sha256_bytes(checkpoint_path.read_bytes()),
        "model_npz_sha256": sha256_bytes((directory / "model.npz").read_bytes()),
        "weights_json_sha256": sha256_bytes((directory / "weights.json").read_bytes()),
        "metadata_sha256": sha256_bytes((directory / "metadata.json").read_bytes()),
    }
    expected = {
        "checkpoint_sha256": CHECKPOINT_SHA256[label],
        "model_npz_sha256": frozen["model_npz_sha256"],
        "weights_json_sha256": frozen["weights_json_sha256"],
        "metadata_sha256": frozen["metadata_sha256"],
    }
    if result != expected or result["model_npz_sha256"] != result["checkpoint_sha256"]:
        raise ValueError(f"seed633_numpy_artifact_binding_mismatch:{label}")
    metadata = read_json(directory / "metadata.json")
    if (
        metadata.get("framework") != "numpy"
        or metadata.get("artifacts", {}).get("weights_sha256")
        != CHECKPOINT_SHA256[label]
        or metadata.get("artifacts", {}).get("weights_json_sha256")
        != result["weights_json_sha256"]
    ):
        raise ValueError(f"seed633_numpy_artifact_metadata_mismatch:{label}")
    return {**result, "artifact_directory": (ARTIFACTS / label).as_posix()}


def protocol_freeze(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct the cohort and bind protocol, code, source, artifacts, and runtime."""
    root = root.resolve()
    evaluation_freeze_path = root / EVALUATION / "execution-freeze-v4.json"
    evaluation_freeze_raw = evaluation_freeze_path.read_bytes()
    if sha256_bytes(evaluation_freeze_raw) != ACTIVE_EVALUATION_FREEZE_SHA256:
        raise ValueError("seed633_seed615_evaluation_freeze_mismatch")
    evaluation_freeze = json.loads(evaluation_freeze_raw)
    torch_version, torch_threads = _torch_runtime_identity()
    if evaluation_freeze.get("evaluation_authorized") is not False:
        raise ValueError("seed633_existing_evaluation_freeze_authorization_changed")
    if not evaluation_freeze["runtime_artifacts"]:
        raise ValueError("seed633_evaluation_artifacts_not_frozen")
    registered_runtime = evaluation_freeze["runtime"]
    runtime_env = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
        )
    }
    if (
        registered_runtime.get("python") != sys.version
        or registered_runtime.get("numpy") != np.__version__
        or registered_runtime.get("torch") != torch_version
        or registered_runtime.get("device") != "cpu"
        or registered_runtime.get("torch_threads") != torch_threads
        or runtime_env
        != {
            "PYTHONHASHSEED": "615",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    ):
        raise ValueError("seed633_registered_numpy_evaluation_runtime_mismatch")
    cohort = build_cohort(root)
    cohort_raw = b"".join(canonical_json(row) + b"\n" for row in cohort)
    manifest_path = root / SNAPSHOTS / "source-identity-lineage-manifest.json"
    input_paths = [
        root / path
        for path in (
            SNAPSHOTS / "B512.json",
            SNAPSHOTS / "S615.json",
            SNAPSHOTS / "protected_holdout.json",
            SNAPSHOTS / "source-identity-lineage-manifest.json",
            S599 / "treatment-membership.json",
            S599 / "heldout-membership.json",
            S599 / "training-run-B/training-trace.jsonl",
            S599 / "training-run-B/training-receipt.json",
            S615 / "candidate-membership.jsonl",
            S615 / "training-run-S615/training-trace.jsonl",
            S615 / "training-run-S615/training-receipt.json",
            S615 / "training-run-S615/training-integrity-report.json",
            S615 / "training-execution-freeze-v3.json",
            EVALUATION / "execution-freeze-v4.json",
        )
    ]
    preparation = read_json(root / S599 / "preparation.json")
    input_paths.extend(
        root / relative
        for relative in preparation["source_verification"]["source_artifacts"]
    )
    source_hashes = {
        relative: sha256_bytes((root / relative).read_bytes())
        for relative in SOURCE_PATHS
    }
    artifacts = {
        name: artifact_identity(root, name, evaluation_freeze)
        for name in ("B512", "S615")
    }
    counts = Counter(row["membership_class"] for row in cohort)
    all_groups = {row["source_opening_group"] for row in cohort}
    heldout_groups = {
        row["source_opening_group"]
        for row in cohort
        if row["authenticated_provenance_holdout"]
    }
    training_groups = {
        row["source_opening_group"]
        for row in cohort
        if not row["authenticated_provenance_holdout"]
    }
    if heldout_groups.intersection(training_groups):
        raise ValueError("seed633_provenance_heldout_group_overlap")
    freeze = {
        "schema": "seed633-teacher-target-cohort-freeze-v1",
        "status": "frozen_before_inference",
        "original_plan_amendment": {
            "inference_backend": "existing NumPy ArtifactEvaluator and authenticated Seed 615 evaluation artifacts",
            "checkpoints": ["B512", "S615"],
            "omitted_checkpoint": "P-E4; no parent comparison is included in this diagnostic",
            "targets": "unchanged authenticated historical teacher targets",
            "execution": "deterministic CPU forward inference only; no PyTorch model execution, search, solver, game generation, or updates",
        },
        "source_sha256": source_hashes,
        "input_sha256": {
            path.relative_to(root).as_posix(): sha256_bytes(path.read_bytes())
            for path in input_paths
        },
        "snapshot_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "seed615_evaluation_freeze_sha256": sha256_bytes(evaluation_freeze_raw),
        "seed615_posthoc_diagnostic_sha256": sha256_bytes(
            (root / S615 / "posthoc-policy-diagnostic/diagnostic.json").read_bytes()
        ),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "numpy_artifacts": artifacts,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch_version_identity_only": torch_version,
            "torch_thread_count_identity_only": torch_threads,
            "registered_evaluation_runtime": registered_runtime,
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
        },
        "cohort": {
            "rows": len(cohort),
            "canonical_jsonl_sha256": sha256_bytes(cohort_raw),
            "membership_class_counts": dict(sorted(counts.items())),
            "source_opening_groups": len(all_groups),
            "provenance_heldout_source_opening_groups": len(heldout_groups),
            "selection": "all distinct states in authenticated B512, S615, and protected-holdout snapshots; canonical identity order; no performance-based filtering",
            "p_e4_ancestry": "unresolved; identity non-overlap is not full training independence",
        },
        "metrics": METRIC_DEFINITIONS,
        "bootstrap": {
            "unit": "source-opening group; paired S615 minus B512 state deltas are retained within each cluster",
            "resamples": 20000,
            "seed": 633,
            "rng": "numpy.random.default_rng(633)",
            "interval": [0.025, 0.975],
            "quantile_method": "numpy.quantile(method='linear')",
            "scope": "descriptive stability intervals; no independent-population inference is asserted",
        },
        "historical_provenance": {
            "source_opening_labels": "authenticated but non-crosswalkable to Seed 624 game_id/opening_identity",
            "inherited_p_e4_exposure": "unresolved identity-incomplete parent training ledger",
        },
    }
    return cohort, freeze


def _raw_policy(evaluator: arena.ArtifactEvaluator, game: KalahGame) -> np.ndarray:
    """Return the original six-action unmasked softmax from the NumPy artifact."""
    hidden = evaluator.extract_trunk(game)
    policy_hidden = hidden
    if evaluator.model_type in (
        "residual_v3",
        "residual_v3_parent_additive_policy_adapter",
        "residual_v4_move_factorized",
    ):
        policy_hidden = np.maximum(
            0.0, (hidden @ evaluator.w_policy_hidden) + evaluator.b_policy_hidden
        )
    if evaluator.move_factorized:
        logits = np.asarray(
            [
                float((policy_hidden @ w).reshape(-1)[0] + b.reshape(-1)[0])
                for w, b in zip(
                    evaluator.w_policy_move, evaluator.b_policy_move, strict=True
                )
            ],
            dtype=np.float32,
        )
    else:
        logits = (policy_hidden @ evaluator.w_policy) + evaluator.b_policy
    if (
        evaluator.w_policy_adapter is not None
        and evaluator.b_policy_adapter is not None
    ):
        logits = (
            logits + (hidden @ evaluator.w_policy_adapter) + evaluator.b_policy_adapter
        )
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities)


def state_metrics(
    row: dict[str, Any], evaluators: dict[str, arena.ArtifactEvaluator]
) -> dict[str, Any]:
    """Evaluate one frozen state with both models and calculate teacher alignment."""
    game = KalahGame.from_state(row["state"])
    legal = np.asarray(row["legal_moves"], dtype=np.int64)
    target = np.asarray(row["target"], dtype=np.float64)
    target_top = np.flatnonzero(target == np.max(target[legal]))
    target_entropy = float(-np.sum(target[target > 0] * np.log(target[target > 0])))
    output: dict[str, Any] = {
        key: row[key]
        for key in (
            "canonical_identity",
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
        )
    }
    output["target"] = row["target"]
    output["legal_moves"] = row["legal_moves"]
    output["teacher_evidence"] = row["teacher_evidence"]
    output["teacher_entropy_nats"] = target_entropy
    output["legal_action_count"] = int(len(legal))
    for name, evaluator in evaluators.items():
        raw = np.asarray(_raw_policy(evaluator, game), dtype=np.float64)
        predicted_policy, _value = evaluator.evaluate(game)
        mask = np.zeros(6, dtype=bool)
        mask[legal] = True
        legal_mass = float(raw[mask].sum())
        if not math.isfinite(legal_mass) or legal_mass <= 0:
            raise ValueError("seed633_model_legal_mass_invalid")
        probabilities = np.zeros(6, dtype=np.float64)
        probabilities[mask] = raw[mask] / legal_mass
        if not np.allclose(probabilities, predicted_policy, rtol=2e-6, atol=2e-7):
            raise ValueError(
                "seed633_manual_numpy_policy_disagrees_with_artifact_evaluator"
            )
        model_top = int(np.argmax(probabilities))
        positive = target > 0
        cross_entropy = float(
            -np.sum(target[positive] * np.log(probabilities[positive]))
        )
        policy_positive = probabilities > 0
        policy_entropy = float(
            -np.sum(
                probabilities[policy_positive] * np.log(probabilities[policy_positive])
            )
        )
        model_metrics: dict[str, Any] = {
            "raw_policy": raw.tolist(),
            "legal_policy": probabilities.tolist(),
            "teacher_target_cross_entropy_nats": cross_entropy,
            "teacher_to_policy_kl_nats": cross_entropy - target_entropy,
            "teacher_top_action_agreement": bool(
                model_top in set(int(x) for x in target_top)
            ),
            "teacher_target_top_actions": [int(x) for x in target_top],
            "legal_action_probability_mass": legal_mass,
            "policy_entropy_nats": policy_entropy,
            "model_confidence": float(probabilities.max()),
        }
        exact_actions = row["exact_optimal_actions"]
        if exact_actions is None:
            model_metrics["exact_optimal_action_mass"] = None
            model_metrics["exact_optimal_action_agreement"] = None
        else:
            model_metrics["exact_optimal_action_mass"] = float(
                probabilities[exact_actions].sum()
            )
            model_metrics["exact_optimal_action_agreement"] = model_top in exact_actions
        output[name] = model_metrics
    output["S615_minus_B512_cross_entropy_nats"] = (
        output["S615"]["teacher_target_cross_entropy_nats"]
        - output["B512"]["teacher_target_cross_entropy_nats"]
    )
    output["S615_minus_B512_kl_nats"] = (
        output["S615"]["teacher_to_policy_kl_nats"]
        - output["B512"]["teacher_to_policy_kl_nats"]
    )
    return output


def _bootstrap_cluster_delta(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    values = []
    for row in rows:
        delta = float(row["S615_minus_B512_cross_entropy_nats"])
        groups[row["source_opening_group"]].append(delta)
        values.append(delta)
    if len(groups) < 2:
        return {
            "clusters": len(groups),
            "mean_delta_nats": float(np.mean(values)) if values else None,
            "interval": None,
        }
    names = sorted(groups)
    group_sums = np.asarray([sum(groups[name]) for name in names], dtype=np.float64)
    group_counts = np.asarray([len(groups[name]) for name in names], dtype=np.float64)
    sampled = rng.integers(0, len(names), size=(20000, len(names)))
    sums = group_sums[sampled].sum(axis=1)
    counts = group_counts[sampled].sum(axis=1)
    distribution = sums / counts
    return {
        "clusters": len(names),
        "mean_delta_nats": float(np.mean(values)),
        "interval": [
            float(x) for x in np.quantile(distribution, [0.025, 0.975], method="linear")
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build paired descriptive summaries and source-opening-cluster intervals."""

    def metric_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"states": 0}
        metrics = {}
        for metric in (
            "teacher_target_cross_entropy_nats",
            "teacher_to_policy_kl_nats",
            "teacher_top_action_agreement",
            "legal_action_probability_mass",
            "policy_entropy_nats",
            "model_confidence",
            "exact_optimal_action_mass",
            "exact_optimal_action_agreement",
        ):
            for model in ("B512", "S615"):
                values = [
                    row[model][metric]
                    for row in items
                    if row[model][metric] is not None
                ]
                metrics[f"{model}_{metric}"] = (
                    float(np.mean(values)) if values else None
                )
            b = [
                row["B512"][metric]
                for row in items
                if row["B512"][metric] is not None and row["S615"][metric] is not None
            ]
            s = [
                row["S615"][metric]
                for row in items
                if row["B512"][metric] is not None and row["S615"][metric] is not None
            ]
            metrics[f"S615_minus_B512_{metric}"] = (
                float(
                    np.mean(
                        np.asarray(s, dtype=np.float64)
                        - np.asarray(b, dtype=np.float64)
                    )
                )
                if b
                else None
            )
        metrics["states"] = len(items)
        metrics["source_opening_clusters"] = len(
            {row["source_opening_group"] for row in items}
        )
        metrics["exact_root_tablebase_states"] = sum(
            row["teacher_type"] == "exact_root_tablebase" for row in items
        )
        metrics["artifact_puct_states"] = sum(
            row["teacher_type"] == "artifact_puct" for row in items
        )
        return metrics

    dimensions: dict[str, dict[str, list[dict[str, Any]]]] = {
        "membership_class": defaultdict(list),
        "phase": defaultdict(list),
        "teacher_type": defaultdict(list),
        "legal_action_count": defaultdict(list),
        "teacher_entropy_quartile": defaultdict(list),
    }
    entropies = np.asarray([row["teacher_entropy_nats"] for row in rows])
    cuts = np.quantile(entropies, [0.25, 0.5, 0.75], method="linear")
    for index, row in enumerate(rows):
        dimensions["membership_class"][row["membership_class"]].append(row)
        dimensions["phase"][row["phase"]].append(row)
        dimensions["teacher_type"][row["teacher_type"]].append(row)
        dimensions["legal_action_count"][str(row["legal_action_count"])].append(row)
        quartile = int(np.searchsorted(cuts, entropies[index], side="right")) + 1
        dimensions["teacher_entropy_quartile"][str(min(quartile, 4))].append(row)
    summaries = {
        dimension: {key: metric_summary(items) for key, items in sorted(groups.items())}
        for dimension, groups in dimensions.items()
    }
    rng = np.random.default_rng(633)
    intervals = {
        "all_states": _bootstrap_cluster_delta(rows, rng),
        "authenticated_provenance_holdout": _bootstrap_cluster_delta(
            [row for row in rows if row["authenticated_provenance_holdout"]], rng
        ),
    }
    intervals.update(
        {
            name: _bootstrap_cluster_delta(
                [row for row in rows if row["membership_class"] == name], rng
            )
            for name in (
                "shared_training",
                "B512_only",
                "S615_only",
                "neither_training_membership",
            )
        }
    )
    return {
        "schema": "seed633-teacher-target-learning-summary-v1",
        "rows": len(rows),
        "paired_comparison": "S615 minus B512 on identical frozen state-target rows",
        "by_membership_class": summaries["membership_class"],
        "by_phase": summaries["phase"],
        "by_teacher_type": summaries["teacher_type"],
        "by_legal_action_count": summaries["legal_action_count"],
        "by_teacher_entropy_quartile": summaries["teacher_entropy_quartile"],
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


def write_freeze(output: Path = ROOT / OUTPUT) -> dict[str, Any]:
    """Publish cohort and protocol freeze without running any model inference."""
    if output.exists():
        raise FileExistsError("seed633_output_must_be_create_only")
    cohort, freeze = protocol_freeze(ROOT)
    cohort_raw = b"".join(canonical_json(row) + b"\n" for row in cohort)
    freeze["cohort"]["canonical_jsonl_sha256"] = sha256_bytes(cohort_raw)
    output.mkdir(parents=True)
    (output / "cohort.jsonl").write_bytes(cohort_raw)
    (output / "freeze.json").write_text(
        json.dumps(freeze, sort_keys=True, indent=2) + "\n"
    )
    return freeze


def run(output: Path = ROOT / OUTPUT) -> dict[str, Any]:
    """Run deterministic bounded CPU inference over the frozen cohort."""
    freeze = read_json(output / "freeze.json")
    cohort_raw = (output / "cohort.jsonl").read_bytes()
    if sha256_bytes(cohort_raw) != freeze["cohort"]["canonical_jsonl_sha256"]:
        raise ValueError("seed633_frozen_cohort_hash_mismatch")
    _cohort, current_freeze = protocol_freeze(ROOT)
    if current_freeze != freeze:
        raise ValueError("seed633_frozen_inputs_or_protocol_changed")
    cohort = [json.loads(line) for line in cohort_raw.splitlines() if line]
    evaluators = {
        name: arena.ArtifactEvaluator(ROOT / ARTIFACTS / name)
        for name in ("B512", "S615")
    }
    result_rows = [state_metrics(row, evaluators) for row in cohort]
    metrics_raw = b"".join(canonical_json(row) + b"\n" for row in result_rows)
    summary = summarize(result_rows)
    summary["freeze_sha256"] = sha256_bytes((output / "freeze.json").read_bytes())
    summary["cohort_sha256"] = sha256_bytes(cohort_raw)
    summary["metrics_sha256"] = sha256_bytes(metrics_raw)
    if (output / "per-state-metrics.jsonl").exists() or (
        output / "summary.json"
    ).exists():
        raise FileExistsError("seed633_results_must_be_create_only")
    (output / "per-state-metrics.jsonl").write_bytes(metrics_raw)
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("select exactly one of --freeze or --run")
    payload = write_freeze(args.output) if args.freeze else run(args.output)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
