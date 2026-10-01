"""Preregister and bind fixed E2–E4 averaging candidates; this module never trains."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import seed461_order_population as population
from ml.alphazero_lite.checkpoint_average import average_checkpoints, load_npz
from ml.alphazero_lite.runtime_search_policy import (
    resolve_strength_comparison_runtime_contract,
)
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"
BASE_REG = DATA / "seed461-cosine-lr-ablation-registration.json"
TRAINING = DATA / "seed461-cosine-lr-ablation-training.json"
TRAINING_BIND = DATA / "seed461-cosine-lr-ablation-training-binding.json"
OLD_BIND = DATA / "seed461-cosine-lr-ablation-evaluation-binding.json"
REG = DATA / "seed461-e2-e4-average-registration.json"
SUITE = DATA / "seed461-e2-e4-average-openings.jsonl"
ARTIFACT_BIND = DATA / "seed461-e2-e4-average-candidate-binding.json"
WORK = ROOT / ".tmp/seed461-e2-e4-average"
OLD_WORK = ROOT / ".tmp/seed461-cosine-lr-ablation"
OPPONENT = ROOT / ".tmp/seed461-order-confirmation/opponent-artifact"
RUNTIME_POLICY = ROOT / "model-artifact/current/search_policy.json"
ORDERS = range(38611, 38616)
EXPLORATORY = DATA / "seed461-batch-order-sensitivity-exploratory-openings.jsonl"
CANONICAL = ROOT / ".tmp/canonical-reconstruction/medium_eval.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def state_keys(path: Path) -> set[str]:
    rows = suites.load_suite_jsonl(str(path))
    return population.keys(rows)


def parameter_distance(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> float:
    if set(left) != set(right):
        raise ValueError("distance_checkpoint_keys_mismatch")
    squared = 0.0
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"distance_checkpoint_incompatible:{key}")
        if np.issubdtype(a.dtype, np.floating):
            delta = a.astype(np.float64) - b.astype(np.float64)
            squared += float(np.dot(delta.ravel(), delta.ravel()))
        elif not np.array_equal(a, b):
            raise ValueError(f"distance_nonfloating_mismatch:{key}")
    return squared**0.5


def consumed_paths() -> list[Path]:
    # Frozen suites through #386, including the two #386 cosine-LR suites.
    return [
        DATA / "seed461-batch-order-sensitivity-confirmation-openings.jsonl",
        DATA / "seed461-lr-sensitivity-openings-v2.jsonl",
        DATA / "seed461-cosine-lr-ablation-openings.jsonl",
    ]


def make_registration() -> dict[str, Any]:
    base = json.loads(BASE_REG.read_text())
    train = json.loads(TRAINING.read_text())
    training_binding = json.loads(TRAINING_BIND.read_text())
    previous_binding = json.loads(OLD_BIND.read_text())
    source_hashes: dict[str, dict[str, str]] = {}
    source_paths: dict[str, dict[str, str]] = {}
    for order in ORDERS:
        run = f"order_{order}_A"
        record = train["trajectories"][run]
        checkpoint_dir = OLD_WORK / "training" / run
        source_hashes[str(order)] = {}
        source_paths[str(order)] = {}
        for epoch in ("E2", "E3", "E4"):
            checkpoint = checkpoint_dir / f"{epoch}.npz"
            expected = record["epochs"][epoch]
            if sha(checkpoint) != expected:
                raise RuntimeError(f"source_checkpoint_hash_mismatch:{run}:{epoch}")
            source_hashes[str(order)][epoch] = expected
            source_paths[str(order)][epoch] = str(checkpoint)

    replay_paths = [Path(row["path"]) for row in base["training"]["replays"]]
    for path, row in zip(replay_paths, base["training"]["replays"], strict=True):
        if sha(path) != row["sha256"]:
            raise RuntimeError(f"frozen_replay_hash_mismatch:{path}")
    proof, excluded = population.build_proof(CANONICAL, EXPLORATORY, replay_paths)
    consumed: dict[str, Any] = {}
    for path in consumed_paths():
        if not path.is_file():
            raise RuntimeError(f"consumed_suite_missing:{path}")
        identities = state_keys(path)
        consumed[path.name] = {
            "path": str(path),
            "sha256": sha(path),
            "unique_state_identities": len(identities),
        }
        excluded |= identities
    openings = population.select_holdout(excluded, seed=387)
    if len(openings) != 256:
        raise RuntimeError("suite_opening_count_mismatch")
    suite_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in openings
    ).encode()
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    suite_keys = {suites.canonical_key(row["state"]) for row in openings}
    if len(suite_keys) != 256 or suite_keys & excluded:
        raise RuntimeError("suite_state_identity_overlap")
    # Save suite after all exclusions/provenance have been computed.
    SUITE.parent.mkdir(parents=True, exist_ok=True)
    if SUITE.exists() and SUITE.read_bytes() != suite_bytes:
        raise RuntimeError("immutable_suite_conflict")
    if not SUITE.exists():
        SUITE.write_bytes(suite_bytes)
    proof["consumed_suites_through_386"] = consumed
    proof["exclusion_counts"]["consumed_suites_through_386"] = sum(
        row["unique_state_identities"] for row in consumed.values()
    )
    proof["exclusion_counts"]["union"] = len(excluded)
    proof["suite_identity_overlap"] = {
        "suite_states": len(suite_keys),
        "excluded_states": len(excluded),
        "overlap": len(suite_keys & excluded),
        "proof": "canonical state SHA-256 identities; zero intersection",
    }
    opponent = previous_binding["opponent"]
    if opponent["artifact"] != str(OPPONENT):
        raise RuntimeError("frozen_opponent_path_mismatch")
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "sidecar_sha256"),
    ):
        if sha(OPPONENT / filename) != opponent[key]:
            raise RuntimeError(f"frozen_opponent_hash_mismatch:{filename}")
    source_registration_hash = sha(BASE_REG)
    source_training_hash = sha(TRAINING)
    if training_binding["registration_sha256"] != source_registration_hash:
        raise RuntimeError("source_training_registration_mismatch")
    return {
        "schema": "seed461-e2-e4-average-v1",
        "status": "prospectively_registered_before_candidate_construction_and_games",
        "reused_training": {
            "description": "Reused PR #386 constant-LR A trajectories; no new training was performed.",
            "pr": 386,
            "registration_sha256": source_registration_hash,
            "training_record_sha256": source_training_hash,
            "training_binding_sha256": sha(TRAINING_BIND),
            "orders": list(ORDERS),
            "epochs_and_source_sha256": source_hashes,
            "source_paths": source_paths,
        },
        "arms": {
            "A": "original constant-LR E4 checkpoint",
            "B": "uniform arithmetic mean of E2, E3, and E4 floating model parameters",
        },
        "construction": {
            "algorithm": "uniform arithmetic mean; float64 accumulation; cast to source dtype",
            "parameter_scope": "all floating checkpoint arrays, including shared trunk and policy/value heads",
            "compatibility": "identical keys, shapes, dtypes; finite floats; identical nonfloating arrays",
            "selection": "all five predefined orders; no checkpoint, coefficient, or order selection using arena outcomes",
        },
        "analysis": {
            "metrics": [
                "per-order A/B score",
                "paired B-minus-A",
                "mean paired effect",
                "worst paired effect",
                "minimum score per arm",
                "between-order score range per arm",
                "parameter L2 distances from parent and among source checkpoints",
            ],
            "bootstrap": {
                "resamples": 10000,
                "seed": 387,
                "cluster": "shared opening",
                "interval": "95% percentile",
            },
            "decision_thresholds": {
                "mean_effect_at_least": 0.03,
                "lower_interval_strictly_above": 0.0,
                "nonnegative_paired_effects_at_least": 4,
                "worst_paired_effect_at_least": -0.05,
                "B_score_range_strictly_smaller": True,
                "B_minimum_score_at_least_A": True,
            },
            "parameter_distances": "descriptive only; gain alone does not establish mechanism",
            "inference_scope": "conditional on these five orders and this dataset",
            "promotion": "never; do not change production checkpoint selection",
            "cosine_rejection": "PR #386's cosine-LR rejection remains unchanged",
        },
        "evaluation": {
            "opponent": "exact frozen seed455 artifact used in PR #384/#386",
            "opponent_binding": opponent,
            "runtime_contract": base["evaluation"]["runtime_contract"],
            "suite": {
                "path": str(SUITE),
                "sha256": suite_hash,
                "seed": 387,
                "openings": 256,
                "nonterminal": True,
                "active_pit_stones_gt": 32,
                "unique_state_identities": 256,
                "identity_overlap_proof": proof,
            },
            "games": 5120,
            "games_per_candidate": 512,
            "games_per_opening": 2,
            "simulations_per_side": 384,
            "c_puct": 1.25,
            "arena_seed": 387,
            "seed_contract": "azlite_eval_seed_v2",
            "outcome_dependent_extensions": False,
        },
    }


def preregister() -> None:
    value = make_registration()
    if REG.exists() and json.loads(REG.read_text()) != value:
        raise RuntimeError("immutable_registration_conflict")
    write_json(REG, value)
    print(f"registration_sha256={sha(REG)}")
    print(f"suite_sha256={sha(SUITE)}")


def construct() -> None:
    registration = json.loads(REG.read_text())
    if (
        registration["status"]
        != "prospectively_registered_before_candidate_construction_and_games"
    ):
        raise RuntimeError("registration_status_invalid")
    source = registration["reused_training"]
    for path, expected, identity in (
        (BASE_REG, source["registration_sha256"], "source_registration"),
        (TRAINING, source["training_record_sha256"], "source_training_record"),
        (TRAINING_BIND, source["training_binding_sha256"], "source_training_binding"),
    ):
        if sha(path) != expected:
            raise RuntimeError(f"frozen_source_identity_mismatch:{identity}")
    if sha(SUITE) != registration["evaluation"]["suite"]["sha256"]:
        raise RuntimeError("suite_hash_mismatch")
    suite_rows = suites.load_suite_jsonl(str(SUITE))
    suite_identities = population.keys(suite_rows)
    if len(suite_rows) != 256 or len(suite_identities) != 256:
        raise RuntimeError("suite_count_or_identity_mismatch")
    if any(
        int(row["pit_sum"]) <= 32 or KalahGame.from_state(row["state"]).over()
        for row in suite_rows
    ):
        raise RuntimeError("suite_opening_eligibility_mismatch")
    for entry in registration["evaluation"]["suite"]["identity_overlap_proof"][
        "consumed_suites_through_386"
    ].values():
        path = Path(entry["path"])
        if (
            sha(path) != entry["sha256"]
            or len(state_keys(path)) != entry["unique_state_identities"]
        ):
            raise RuntimeError(f"consumed_suite_identity_mismatch:{path}")
    base_registration = json.loads(BASE_REG.read_text())
    for replay in base_registration["training"]["replays"]:
        if sha(Path(replay["path"])) != replay["sha256"]:
            raise RuntimeError(f"frozen_replay_hash_mismatch:{replay['name']}")
    proof = registration["evaluation"]["suite"]["identity_overlap_proof"]
    for identity in (proof["canonical"],):
        if sha(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError("canonical_population_hash_mismatch")
    opponent = registration["evaluation"]["opponent_binding"]
    if opponent["artifact"] != str(OPPONENT):
        raise RuntimeError("frozen_opponent_path_mismatch")
    for filename, key in (
        ("weights.json", "weights_sha256"),
        ("metadata.json", "metadata_sha256"),
        ("search_policy.json", "sidecar_sha256"),
    ):
        if sha(OPPONENT / filename) != opponent[key]:
            raise RuntimeError(f"frozen_opponent_hash_mismatch:{filename}")
    runtime = registration["evaluation"]["runtime_contract"]
    for path_key, digest_key in (
        ("exact_root_native_probe", "exact_root_native_probe_sha256"),
        ("exact_root_tablebase", "exact_root_tablebase_sha256"),
    ):
        if sha(Path(runtime[path_key])) != runtime[digest_key]:
            raise RuntimeError(f"frozen_runtime_file_hash_mismatch:{path_key}")
    run_binding: dict[str, Any] = {
        "schema": "seed461-e2-e4-average-candidate-binding-v1",
        "registration_sha256": sha(REG),
        "source_training_sha256": registration["reused_training"][
            "training_record_sha256"
        ],
        "suite_sha256": sha(SUITE),
        "opponent": registration["evaluation"]["opponent_binding"],
        "candidates": {},
    }
    if ARTIFACT_BIND.exists():
        prior = json.loads(ARTIFACT_BIND.read_text())
        for field in (
            "schema",
            "registration_sha256",
            "source_training_sha256",
            "suite_sha256",
            "opponent",
        ):
            if prior.get(field) != run_binding[field]:
                raise RuntimeError(f"immutable_candidate_binding_mismatch:{field}")
    else:
        prior = None
    for order in ORDERS:
        run = f"order_{order}"
        epoch_hashes = registration["reused_training"]["epochs_and_source_sha256"][
            str(order)
        ]
        sources = [
            load_npz(registration["reused_training"]["source_paths"][str(order)][epoch])
            for epoch in ("E2", "E3", "E4")
        ]
        for epoch, values in zip(("E2", "E3", "E4"), sources, strict=True):
            source_path = Path(
                registration["reused_training"]["source_paths"][str(order)][epoch]
            )
            if sha(source_path) != epoch_hashes[epoch]:
                raise RuntimeError(f"source_checkpoint_changed:{run}:{epoch}")
            if not all(
                np.isfinite(array).all()
                for array in values.values()
                if np.issubdtype(array.dtype, np.floating)
            ):
                raise RuntimeError(f"nonfinite_source_checkpoint:{run}:{epoch}")
        average = average_checkpoints(sources)
        for arm in ("A", "B"):
            checkpoint = (
                Path(registration["reused_training"]["source_paths"][str(order)]["E4"])
                if arm == "A"
                else WORK / "checkpoints" / f"{run}_B.npz"
            )
            if arm == "B":
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                if checkpoint.exists():
                    existing = load_npz(str(checkpoint))
                    if set(existing) != set(average) or any(
                        not np.array_equal(existing[k], average[k]) for k in average
                    ):
                        raise RuntimeError(f"immutable_mean_checkpoint_conflict:{run}")
                else:
                    np.savez(checkpoint, **average)
            expected_checkpoint_hash = (
                epoch_hashes["E4"] if arm == "A" else sha(checkpoint)
            )
            artifact = WORK / "artifacts" / f"{run}_{arm}"
            old_run = f"{run}_{arm}"
            existing_candidate = (
                prior.get("candidates", {}).get(old_run) if prior else None
            )
            if existing_candidate:
                artifact = Path(existing_candidate["artifact"])
                for filename, expected_hash in existing_candidate[
                    "artifact_sha256"
                ].items():
                    if sha(artifact / filename) != expected_hash:
                        raise RuntimeError(
                            f"bound_artifact_changed:{old_run}:{filename}"
                        )
            else:
                if artifact.exists() and any(artifact.iterdir()):
                    raise RuntimeError(f"unbound_artifact_exists:{old_run}")
                artifact.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "ml/alphazero_lite/export_artifact.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--out-dir",
                        str(artifact),
                        "--version",
                        f"seed461-e2-e4-{old_run}",
                        "--model-type",
                        "residual_v3",
                        "--rules-version",
                        "kalah_v1",
                        "--input-encoding",
                        "kalah_v3",
                    ],
                    cwd=ROOT,
                    check=True,
                )
                sidecar = RUNTIME_POLICY
                shutil.copy2(sidecar, artifact / "search_policy.json")
            hashes = {
                name: sha(artifact / name)
                for name in (
                    "model.npz",
                    "weights.json",
                    "metadata.json",
                    "search_policy.json",
                )
            }
            if hashes["model.npz"] != expected_checkpoint_hash:
                raise RuntimeError(f"candidate_model_checkpoint_mismatch:{old_run}")
            contract = resolve_strength_comparison_runtime_contract(
                current_artifact=OPPONENT, challenger_artifact=artifact
            )
            if contract != registration["evaluation"]["runtime_contract"]:
                raise RuntimeError(f"candidate_runtime_contract_mismatch:{old_run}")
            for path_key, digest_key in (
                ("exact_root_native_probe", "exact_root_native_probe_sha256"),
                ("exact_root_tablebase", "exact_root_tablebase_sha256"),
            ):
                if sha(Path(contract[path_key])) != contract[digest_key]:
                    raise RuntimeError(f"runtime_file_hash_mismatch:{path_key}")
            row = {
                "artifact": str(artifact),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected_checkpoint_hash,
                "artifact_sha256": hashes,
                "runtime_contract": contract,
                "arm": arm,
                "source_checkpoint_sha256": epoch_hashes,
            }
            if existing_candidate is not None and existing_candidate != row:
                raise RuntimeError(f"immutable_candidate_identity_mismatch:{old_run}")
            run_binding["candidates"][old_run] = row
    if prior is not None and prior != run_binding:
        raise RuntimeError("immutable_candidate_binding_mismatch")
    if prior is None:
        write_json(ARTIFACT_BIND, run_binding)
    write_parameter_distances(registration, run_binding)
    print(f"candidate_binding_sha256={sha(ARTIFACT_BIND)}")


def write_parameter_distances(
    registration: dict[str, Any], binding: dict[str, Any]
) -> None:
    """Publish descriptive tensor distances, hash-bound to frozen inputs."""
    source = registration["reused_training"]
    base_registration = json.loads(BASE_REG.read_text())
    parent_path = Path(base_registration["training"]["parent"])
    expected_parent = base_registration["training"]["parent_sha256"]
    if sha(parent_path) != expected_parent:
        raise RuntimeError("registered_parent_checkpoint_hash_mismatch")
    parent = load_npz(str(parent_path))
    rows: dict[str, Any] = {}
    for order in ORDERS:
        order_key = str(order)
        checkpoints = {
            epoch: load_npz(source["source_paths"][order_key][epoch])
            for epoch in ("E2", "E3", "E4")
        }
        mean = load_npz(binding["candidates"][f"order_{order}_B"]["checkpoint"])
        rows[order_key] = {
            "from_parent": {
                name: parameter_distance(parent, tensor)
                for name, tensor in [*checkpoints.items(), ("B_mean", mean)]
            },
            "among_source_checkpoints": {
                "E2_E3": parameter_distance(checkpoints["E2"], checkpoints["E3"]),
                "E2_E4": parameter_distance(checkpoints["E2"], checkpoints["E4"]),
                "E3_E4": parameter_distance(checkpoints["E3"], checkpoints["E4"]),
            },
        }
    result = {
        "schema": "seed461-e2-e4-average-parameter-distances-v1",
        "interpretation": "descriptive only; no causal mechanism is established by an arena gain",
        "parent_checkpoint_sha256": expected_parent,
        "registration_sha256": sha(REG),
        "candidate_binding_sha256": sha(ARTIFACT_BIND),
        "per_order_l2": rows,
    }
    output = DATA / "seed461-e2-e4-average-parameter-distances.json"
    if output.exists() and json.loads(output.read_text()) != result:
        raise RuntimeError("immutable_parameter_distance_conflict")
    if not output.exists():
        write_json(output, result)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"register", "construct"}:
        raise SystemExit(
            "usage: python -m ml.alphazero_lite.run_seed461_e2_e4_average {register|construct}"
        )
    preregister() if sys.argv[1] == "register" else construct()
