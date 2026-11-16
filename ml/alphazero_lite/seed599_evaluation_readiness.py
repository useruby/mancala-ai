"""Frozen Seed 599 strength-evaluation preparation and guarded execution.

Preparation authenticates published evidence and exports policy-only candidate
bundles. Game execution is impossible until a separate owner-authorization
record binds the exact execution-freeze file bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed599-replay-coverage")
PUBLISHED = DATA / "evaluation-contract.json"
OPENINGS = DATA / "evaluation-openings.json"
OUT = DATA / "evaluation-readiness"
SERVING = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4")
RUNTIME_POLICY = Path("model-artifact/current/search_policy.json")
NATIVE_PROBE = Path("model-artifact/runtime/kalah_v1_tablebase")
TABLEBASE = Path("model-artifact/runtime/kalah_v1_21.kvtb")
ARMS = ("A", "B")
REPLICATES = 20_000
SEARCH_ENVIRONMENT = {
    "PYTHONHASHSEED": "599",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
CONTRACT_SHA256 = "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4"
OPENINGS_FILE_SHA256 = (
    "3acc395a59d90df57ddf60aa1e9d99aabde220901434f72239eb48a4047bc54a"
)
TRAINING_RECEIPTS = {
    "A": "d927dfcab00549098415af1cc69c047522086f55f7c6b5fba9d9f673f4e1b16b",
    "B": "3d81b00c438a678a2e38b9d58dcadc22f4091c0cb8e3ccbb755e1afe19ca41de",
}
TRAINING_REPORTS = {
    "A": "b1e55cff5ac1432a0683f3dd13bacc7c176dda7e5af87aa7823a178785c4f0e2",
    "B": "57fd2623874a0064814b71847763f52ad890cc1833d8daffdaa222786673da46",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha(path: Path, expected: str, identity: str) -> str:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"identity_mismatch:{identity}:{expected}:{actual}")
    return actual


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_create(path: Path, payload: bytes) -> None:
    """Create a file exactly once; an interrupted or conflicting write fails closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"create_only_conflict:{path}")
        raise ValueError(f"create_only_exists:{path}")
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        raise ValueError(f"pending_write_requires_review:{pending}")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def authenticate_registered_inputs(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate frozen contract/openings, receipts, checkpoints, and serving."""
    root = root.resolve()
    contract_path, openings_path = root / PUBLISHED, root / OPENINGS
    contract = read_json(contract_path)
    openings = read_json(openings_path)
    require_sha(contract_path, CONTRACT_SHA256, "evaluation-contract.json")
    require_sha(openings_path, OPENINGS_FILE_SHA256, "evaluation-openings.json")
    expected_openings = contract["opening_manifest_sha256"]
    actual_openings = hashlib.sha256(
        json.dumps(openings, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        len(openings) != contract["opening_count"]
        or actual_openings != expected_openings
    ):
        raise ValueError("frozen_opening_identity_mismatch")
    if contract.get("status") != "frozen-evaluation-not-authorized":
        raise ValueError("evaluation_contract_status")

    arms: dict[str, Any] = {}
    for arm in ARMS:
        directory = root / DATA / f"training-run-{arm}"
        receipt_path = directory / "training-receipt.json"
        receipt = read_json(receipt_path)
        checkpoint = directory / receipt["checkpoint"]
        report = directory / "independent-verification.json"
        expected_checkpoint = receipt["checkpoint_sha256"]
        actual_checkpoint = sha(checkpoint)
        if actual_checkpoint != expected_checkpoint:
            raise ValueError(f"training_checkpoint_hash_mismatch:{arm}")
        expected_report = read_json(report)
        if (
            sha(receipt_path) != TRAINING_RECEIPTS[arm]
            or sha(report) != TRAINING_REPORTS[arm]
            or expected_report.get("passed") is not True
            or expected_report.get("arm") != arm
            or expected_report.get("checkpoint_sha256") != expected_checkpoint
        ):
            raise ValueError(f"published_verification_report_mismatch:{arm}")
        arms[arm] = {
            "checkpoint": str(checkpoint.relative_to(root)),
            "checkpoint_sha256": actual_checkpoint,
            "receipt_sha256": sha(receipt_path),
            "published_verification_sha256": sha(report),
        }

    p_e4 = contract["serving_p_e4"]
    serving_files = {
        "model.npz": "model_sha256",
        "weights.json": "weights_sha256",
        "metadata.json": "metadata_sha256",
        "search_policy.json": "search_policy_sha256",
    }
    serving: dict[str, str] = {}
    for filename, identity_key in serving_files.items():
        path = root / SERVING / filename
        actual = sha(path)
        if actual != p_e4[identity_key]:
            raise ValueError(f"serving_artifact_hash_mismatch:{filename}")
        serving[filename] = actual
    if (
        p_e4["model_sha256"]
        != "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad"
    ):
        raise ValueError("registered_serving_checkpoint_identity_mismatch")
    for path, key in ((RUNTIME_POLICY, "search_policy_sha256"),):
        if sha(root / path) != p_e4[key]:
            raise ValueError(f"serving_runtime_policy_mismatch:{path}")
    for path, key in (
        (NATIVE_PROBE, "native_probe_sha256"),
        (TABLEBASE, "tablebase_sha256"),
    ):
        if sha(root / path) != p_e4[key]:
            raise ValueError(f"serving_exact_root_asset_mismatch:{path}")
    metadata = read_json(root / SERVING / "metadata.json")
    if (
        metadata.get("artifacts", {}).get("weights_sha256") != serving["model.npz"]
        or metadata.get("artifacts", {}).get("weights_json_sha256")
        != serving["weights.json"]
        or metadata.get("version") != "seed505-E4"
    ):
        raise ValueError("serving_metadata_internal_identity_mismatch")

    freeze = root / DATA / "training-execution-freeze.json"
    if (
        sha(freeze)
        != "bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9"
    ):
        raise ValueError("training_execution_freeze_mismatch")
    return {
        "contract_sha256": sha(contract_path),
        "openings_file_sha256": sha(openings_path),
        "openings_manifest_sha256": actual_openings,
        "training_execution_freeze_sha256": sha(freeze),
        "arms": arms,
        "serving_p_e4": serving,
        "exact_root_native_probe_sha256": sha(root / NATIVE_PROBE),
        "tablebase_sha256": sha(root / TABLEBASE),
    }


def independently_verify_training(root: Path = ROOT) -> dict[str, Any]:
    """Rerun published independent verifier and clarify exactly what it checks."""
    from ml.alphazero_lite.verify_seed599_training_result import verify_result

    report: dict[str, Any] = {
        "schema": "seed599-evaluation-training-verification-v1",
        "training_execution_freeze_sha256": sha(
            root / DATA / "training-execution-freeze.json"
        ),
        "arms": {},
    }
    for arm in ARMS:
        out = root / DATA / f"training-run-{arm}"
        result = verify_result(arm, out, root=root)
        report["arms"][arm] = {
            **result,
            "frozen_parameter_integrity": {
                "independent_tensor_equality": "all registered frozen parameters compared with torch.equal against the authenticated parent checkpoint",
                "executor_runtime_assertion": "executor in-run frozen-parameter torch.equal assertion recorded as unchanged in the authenticated receipt",
                "numerical_training_replay": "not performed",
            },
        }
    return report


def write_independent_training_report(root: Path = ROOT) -> dict[str, Any]:
    report = independently_verify_training(root)
    report_path = root / OUT / "independent-training-verification-v2.json"
    atomic_create(
        report_path, json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    )
    return report


def heldout_policy_cross_entropy(root: Path = ROOT) -> dict[str, Any]:
    """Compute held-out masked policy cross-entropy for the secondary report."""
    import torch
    from ml.alphazero_lite import train

    root = root.resolve()
    authenticate_registered_inputs(root)
    rows = read_json(root / DATA / "heldout-membership.json")
    expected = read_json(root / DATA / "preparation.json")["heldout"][
        "membership_manifest"
    ]
    identity = hashlib.sha256(canonical(rows)).hexdigest()
    if len(rows) != expected["states"] or identity != expected["sha256"]:
        raise ValueError("heldout_membership_identity_mismatch")
    torch.set_num_threads(1)
    input_states = torch.from_numpy(
        np.asarray([row["state"] for row in rows], dtype=np.float32)
    )
    by_row: dict[str, dict[str, float]] = {}
    by_group: dict[str, dict[str, list[float]]] = {}
    for arm in ARMS:
        checkpoint = (
            root / DATA / f"training-run-{arm}" / f"seed599-arm-{arm}-update-512.npz"
        )
        model = train.PolicyValueNet((96, 3), "residual_v3", 27).cpu().eval()
        if train.load_checkpoint_into_model(model, checkpoint):
            raise ValueError(f"heldout_checkpoint_load_mismatch:{arm}")
        with torch.no_grad():
            logits = model(input_states)[0].cpu().numpy().astype(np.float64)
        by_row[arm] = {}
        for index, row in enumerate(rows):
            legal = [int(move) for move in row["legal_moves"]]
            target = np.asarray(row["target"], dtype=np.float64)
            if (
                not legal
                or len(target) != 6
                or not np.isclose(float(target.sum()), 1.0, rtol=0, atol=1e-7)
                or any(target[move] < 0 for move in legal)
                or any(target[move] != 0 for move in range(6) if move not in legal)
            ):
                raise ValueError(f"heldout_policy_target_invalid:{index}")
            legal_logits = logits[index, legal]
            maximum = float(legal_logits.max())
            log_partition = maximum + float(
                np.log(np.exp(legal_logits - maximum).sum())
            )
            cross_entropy = float(
                -sum(
                    target[move] * (logits[index, move] - log_partition)
                    for move in legal
                )
            )
            by_row[arm][str(row["row_id"])] = cross_entropy
            group = f"{row['source']}:{row['opening_id']}"
            by_group.setdefault(group, {}).setdefault(arm, []).append(cross_entropy)
    if set(by_row["A"]) != set(by_row["B"]):
        raise ValueError("heldout_arm_membership_mismatch")
    group_means = {
        group: {arm: float(np.mean(scores[arm])) for arm in ARMS}
        for group, scores in by_group.items()
        if set(scores) == set(ARMS)
    }
    group_differences = [value["B"] - value["A"] for value in group_means.values()]
    a_mean = float(np.mean(list(by_row["A"].values())))
    b_mean = float(np.mean(list(by_row["B"].values())))
    return {
        "schema": "seed599-heldout-policy-cross-entropy-v1",
        "membership_sha256": identity,
        "state_count": len(rows),
        "opening_group_count": len(group_means),
        "masked_policy_cross_entropy": {
            "A": a_mean,
            "B": b_mean,
            "B_minus_A": b_mean - a_mean,
            "paired_opening_group_B_minus_A_mean": float(np.mean(group_differences)),
        },
        "decision_role": "secondary diagnostic only; not a pass gate",
        "per_opening_group_mean_cross_entropy": group_means,
    }


def write_heldout_policy_cross_entropy(root: Path = ROOT) -> dict[str, Any]:
    result = heldout_policy_cross_entropy(root)
    atomic_create(
        root / OUT / "heldout-policy-cross-entropy.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def verify_policy_parity(root: Path, arm: str) -> dict[str, Any]:
    """Compare policy probabilities from checkpoint and exported NumPy evaluator."""
    import torch
    from ml.alphazero_lite import arena, train
    from ml.alphazero_lite.kalah_rules import KalahGame
    from ml.alphazero_lite.self_play import encode_state

    root = root.resolve()
    checkpoint = (
        root / DATA / f"training-run-{arm}" / f"seed599-arm-{arm}-update-512.npz"
    )
    bundle = root / OUT / "candidate-bundles" / arm
    model = train.PolicyValueNet((96, 3), "residual_v3", 27).cpu().eval()
    if train.load_checkpoint_into_model(model, checkpoint):
        raise ValueError(f"candidate_checkpoint_load_mismatch:{arm}")
    evaluator = arena.ArtifactEvaluator(bundle)
    openings = read_json(root / OPENINGS)
    indices = (0, 1, 7, 31, 127, 511, 1023)
    max_error = 0.0
    for index in indices:
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move in openings[index]["prefix_moves"]:
            if not game.move(game.pit_index(int(move))):
                raise ValueError(f"synthetic_parity_opening_invalid:{index}")
        encoded = torch.tensor(
            [encode_state(game.to_state(), input_encoding="kalah_v3")],
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = model(encoded)[0].cpu().numpy()[0].astype(np.float64)
        legal = game.possible_moves()
        logits = logits - np.max(logits[legal])
        probs = np.zeros(6, dtype=np.float64)
        probs[legal] = np.exp(logits[legal])
        probs /= probs.sum()
        observed, _ = evaluator.evaluate(game)
        error = float(np.max(np.abs(probs - np.asarray(observed, dtype=np.float64))))
        max_error = max(max_error, error)
        if error > 2e-6:
            raise ValueError(f"candidate_policy_parity_failed:{arm}:{index}:{error}")
    return {
        "arm": arm,
        "positions": list(indices),
        "max_probability_abs_error": max_error,
        "tolerance": 2e-6,
    }


def export_candidate_bundles(root: Path = ROOT) -> dict[str, Any]:
    """Export each immutable published checkpoint into a create-only runtime bundle."""
    root = root.resolve()
    evidence = authenticate_registered_inputs(root)
    exporter = root / "ml/alphazero_lite/export_artifact.py"
    bundles: dict[str, Any] = {}
    bundle_root = root / OUT / "candidate-bundles"
    for arm in ARMS:
        destination = bundle_root / arm
        if destination.exists():
            raise ValueError(f"candidate_bundle_already_exists:{arm}")
        checkpoint = root / evidence["arms"][arm]["checkpoint"]
        subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(destination),
                "--version",
                f"seed599-arm-{arm}-policy",
                "--model-type",
                "residual_v3",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=root,
            check=True,
        )
        policy = root / SERVING / "search_policy.json"
        shutil_copy_create(policy, destination / "search_policy.json")
        bundles[arm] = {
            filename: sha(destination / filename)
            for filename in (
                "model.npz",
                "weights.json",
                "metadata.json",
                "search_policy.json",
            )
        }
        if bundles[arm]["model.npz"] != evidence["arms"][arm]["checkpoint_sha256"]:
            raise ValueError(f"candidate_checkpoint_export_mismatch:{arm}")
        metadata = read_json(destination / "metadata.json")
        if (
            metadata.get("artifacts", {}).get("weights_sha256")
            != bundles[arm]["model.npz"]
            or metadata.get("artifacts", {}).get("weights_json_sha256")
            != bundles[arm]["weights.json"]
        ):
            raise ValueError(f"candidate_metadata_identity_mismatch:{arm}")
        bundles[arm]["policy_parity"] = verify_policy_parity(root, arm)
    atomic_create(
        bundle_root / "bundle-manifest.json",
        json.dumps(bundles, indent=2, sort_keys=True).encode() + b"\n",
    )
    return bundles


def preflight(root: Path = ROOT) -> dict[str, Any]:
    """Read-only readiness check, including generated candidate exports."""
    root = root.resolve()
    registered = authenticate_registered_inputs(root)
    training_memberships = training_membership_freeze_record(root)
    report_path = root / OUT / "independent-training-verification-v2.json"
    report = read_json(report_path)
    if report.get("schema") != "seed599-evaluation-training-verification-v1":
        raise ValueError("independent_training_report_schema")
    for arm in ARMS:
        if report.get("arms", {}).get(arm, {}).get("passed") is not True:
            raise ValueError(f"independent_training_verification_failed:{arm}")
        integrity = report["arms"][arm]["frozen_parameter_integrity"]
        if (
            integrity.get("independent_tensor_equality")
            != "all registered frozen parameters compared with torch.equal against the authenticated parent checkpoint"
            or integrity.get("executor_runtime_assertion")
            != "executor in-run frozen-parameter torch.equal assertion recorded as unchanged in the authenticated receipt"
            or integrity.get("numerical_training_replay") != "not performed"
        ):
            raise ValueError(f"training_replay_claim_mismatch:{arm}")
    manifest_path = root / OUT / "candidate-bundles" / "bundle-manifest.json"
    manifest = read_json(manifest_path)
    if set(manifest) != set(ARMS):
        raise ValueError("candidate_bundle_manifest_arms")
    for arm in ARMS:
        bundle = root / OUT / "candidate-bundles" / arm
        for filename in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            if sha(bundle / filename) != manifest[arm][filename]:
                raise ValueError(f"candidate_bundle_hash_mismatch:{arm}:{filename}")
        if manifest[arm]["model.npz"] != registered["arms"][arm]["checkpoint_sha256"]:
            raise ValueError(f"candidate_bundle_checkpoint_mismatch:{arm}")
        if (
            manifest[arm]["search_policy.json"]
            != registered["serving_p_e4"]["search_policy.json"]
        ):
            raise ValueError(f"candidate_bundle_search_policy_mismatch:{arm}")
        if (
            manifest[arm].get("policy_parity", {}).get("max_probability_abs_error", 1.0)
            > 2e-6
        ):
            raise ValueError(f"candidate_bundle_policy_parity_missing_or_failed:{arm}")
    heldout_path = root / OUT / "heldout-policy-cross-entropy.json"
    heldout = read_json(heldout_path)
    if (
        heldout.get("schema") != "seed599-heldout-policy-cross-entropy-v1"
        or heldout.get("decision_role") != "secondary diagnostic only; not a pass gate"
        or heldout.get("state_count") != 1353
        or heldout.get("membership_sha256")
        != read_json(root / DATA / "preparation.json")["heldout"][
            "membership_manifest"
        ]["sha256"]
        or not np.isfinite(
            list(heldout.get("masked_policy_cross_entropy", {}).values())
        ).all()
    ):
        raise ValueError("heldout_policy_cross_entropy_report_invalid")
    return {
        "status": "read_only_preflight_passed_not_authorized_for_games",
        "registered_inputs": registered,
        "training_membership_identities": training_memberships,
        "independent_training_verification_sha256": sha(report_path),
        "candidate_bundle_manifest_sha256": sha(manifest_path),
        "candidate_bundle_manifest": manifest,
        "heldout_policy_cross_entropy_sha256": sha(heldout_path),
        "heldout_policy_cross_entropy": heldout["masked_policy_cross_entropy"],
        "evaluation_games_authorized": False,
    }


def shutil_copy_create(source: Path, destination: Path) -> None:
    """Copy bytes without replacing any existing artifact."""
    atomic_create(destination, source.read_bytes())


def execution_authorized(root: Path, freeze: dict[str, Any]) -> bool:
    """Only a separate explicit owner record bound to this exact freeze opens execution."""
    freeze_path = root / OUT / "execution-freeze.json"
    authorization_path = root / OUT / "evaluation-authorization.json"
    if not freeze_path.is_file() or not authorization_path.is_file():
        return False
    try:
        freeze_bytes = freeze_path.read_bytes()
        frozen_on_disk = json.loads(freeze_bytes)
        record = read_json(authorization_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(frozen_on_disk, dict)
        or not isinstance(record, dict)
        or freeze != frozen_on_disk
        or frozen_on_disk.get("schema") != "seed599-evaluation-execution-freeze-v1"
        or frozen_on_disk.get("status")
        != "frozen-awaiting-separate-project-owner-authorization"
        or frozen_on_disk.get("execution_authorized") is not False
    ):
        return False
    exact_freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    return (
        record.get("schema") == "seed599-evaluation-owner-authorization-v1"
        and record.get("status") == "authorized-for-evaluation"
        and record.get("execution_freeze_sha256") == exact_freeze_sha256
        and record.get("authorization", {}).get("authorized_by") == "project owner"
        and record.get("authorization", {}).get("run_games") is True
    )


def training_membership_identity(root: Path = ROOT) -> dict[str, Any]:
    """Authenticate frozen A/B membership files and return their identity sets."""
    root = root.resolve()
    training_freeze = read_json(root / DATA / "training-execution-freeze.json")
    path_by_arm = {
        "A": "control-membership.json",
        "B": "treatment-membership.json",
    }
    identities: dict[str, Any] = {}
    for arm, relative in path_by_arm.items():
        config_path = root / DATA / f"training-{arm}.json"
        config = read_json(config_path)
        if sha(config_path) != training_freeze["arms"][arm]["configuration_sha256"]:
            raise ValueError(f"training_configuration_identity_mismatch:{arm}")
        membership_path = root / DATA / relative
        rows = read_json(membership_path)
        digest = hashlib.sha256(canonical(rows)).hexdigest()
        if len(rows) != config["states"] or digest != config["membership_sha256"]:
            raise ValueError(f"training_membership_identity_mismatch:{arm}")
        canonical_ids = {str(row["canonical_identity"]) for row in rows}
        float32_ids = {str(row["float32_identity"]) for row in rows}
        if (
            len(canonical_ids) != len(rows)
            or len(float32_ids) != len(rows)
            or any(len(identity) != 64 for identity in canonical_ids | float32_ids)
        ):
            raise ValueError(f"training_membership_identity_set_invalid:{arm}")
        identities[arm] = {
            "membership_file": relative,
            "membership_sha256": digest,
            "row_count": len(rows),
            "canonical_distinct_count": len(canonical_ids),
            "float32_distinct_count": len(float32_ids),
            "canonical_identities": canonical_ids,
            "float32_identities": float32_ids,
        }
    return identities


def training_membership_freeze_record(root: Path = ROOT) -> dict[str, Any]:
    """Return the compact, frozen membership identities without state lists."""
    return {
        arm: {
            key: value
            for key, value in info.items()
            if key not in {"canonical_identities", "float32_identities"}
        }
        for arm, info in training_membership_identity(root).items()
    }


def make_execution_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Bind fixed protocol and implementation; no execution authorization is emitted."""
    root = root.resolve()
    authenticated = authenticate_registered_inputs(root)
    sources = (
        "ml/alphazero_lite/seed599_evaluation_readiness.py",
        "ml/alphazero_lite/verify_seed599_evaluation.py",
        "ml/alphazero_lite/seed599_evaluation_arena.py",
        "ml/alphazero_lite/arena.py",
        "ml/alphazero_lite/report_validation.py",
        "ml/alphazero_lite/root_prior_transforms.py",
        "ml/alphazero_lite/policy_prior_localization.py",
        "ml/alphazero_lite/endgame_tablebase.py",
        "ml/alphazero_lite/opening_cache.py",
        "ml/alphazero_lite/search_ablation.py",
        "ml/alphazero_lite/shadow_root_q.py",
        "ml/alphazero_lite/value_transforms.py",
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/verify_seed599_training_result.py",
        "ml/alphazero_lite/seed599_training_executor.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/self_play.py",
        "ml/alphazero_lite/native_exact_root_tablebase.py",
        "ml/alphazero_lite/exact_root_decision.py",
        "ml/alphazero_lite/runtime_search_policy.py",
        "ml/alphazero_lite/input_encodings.py",
        "ml/alphazero_lite/evaluation_seed_contract.py",
        "ml/alphazero_lite/export_artifact.py",
        "ml/alphazero_lite/selective_head_evaluator.py",
        "ml/alphazero_lite/requirements-seed599-evaluation.txt",
        "ml/alphazero_lite/test_seed599_evaluation_readiness.py",
    )
    manifest = root / OUT / "candidate-bundles" / "bundle-manifest.json"
    if not manifest.is_file():
        raise ValueError("candidate_bundle_manifest_missing")
    verification_report = root / OUT / "independent-training-verification-v2.json"
    if not verification_report.is_file():
        raise ValueError("independent_training_verification_report_missing")
    heldout_report_path = root / OUT / "heldout-policy-cross-entropy.json"
    if not heldout_report_path.is_file():
        raise ValueError("heldout_policy_cross_entropy_report_missing")
    return {
        "schema": "seed599-evaluation-execution-freeze-v1",
        "status": "frozen-awaiting-separate-project-owner-authorization",
        "execution_authorized": False,
        "registered_contract_sha256": authenticated["contract_sha256"],
        "registered_openings_identity": authenticated["openings_manifest_sha256"],
        "training_execution_freeze_sha256": authenticated[
            "training_execution_freeze_sha256"
        ],
        "checkpoint_and_serving_identities": authenticated,
        "training_membership_identities": training_membership_freeze_record(root),
        "candidate_bundle_manifest_sha256": sha(manifest),
        "candidate_bundle_manifest": read_json(manifest),
        "independent_training_verification_sha256": sha(verification_report),
        "heldout_policy_cross_entropy_sha256": sha(heldout_report_path),
        "source_sha256": {name: sha(root / name) for name in sources},
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": _torch_version(),
            "dependency_lock_sha256": sha(
                root / "ml/alphazero_lite/requirements-seed599-evaluation.txt"
            ),
            "device": "cpu",
            "search_workers_per_game": 1,
            "pair_parallelism": 18,
            "environment": SEARCH_ENVIRONMENT,
        },
        "execution_limits": {
            "pair_parallelism": 18,
            "registered_cpus": 18,
            "registered_memory_gib": 64,
        },
        "contract": "evaluation-contract.json byte-for-byte unchanged; evaluation-openings.json canonical identity unchanged",
        "assignments": 4096,
        "games_per_opening_candidate": 2,
        "decision": "gates from frozen evaluation-contract.json; no promotion authority",
    }


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def create_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    freeze = make_execution_freeze(root)
    path = root / OUT / "execution-freeze.json"
    prior_sha256: str | None = None
    if path.exists():
        prior = path.read_bytes()
        if prior == json.dumps(freeze, indent=2, sort_keys=True).encode() + b"\n":
            return freeze
        authorization = root / OUT / "evaluation-authorization.json"
        assignments = root / OUT / "assignments"
        if authorization.exists() or (
            assignments.exists() and any(assignments.iterdir())
        ):
            raise ValueError("cannot_refreeze_after_authorization_or_assignment_start")
        prior_record = json.loads(prior)
        if prior_record.get("execution_authorized") is not False:
            raise ValueError("cannot_supersede_authorized_execution_freeze")
        archive = (
            root
            / OUT
            / (
                f"execution-freeze-superseded-{hashlib.sha256(prior).hexdigest()[:12]}.json"
            )
        )
        prior_sha256 = hashlib.sha256(prior).hexdigest()
        if archive.exists():
            if archive.read_bytes() != prior:
                raise ValueError("superseded_execution_freeze_archive_conflict")
            path.unlink()
        else:
            os.replace(path, archive)
        freeze["supersedes_previous_execution_freeze_sha256"] = prior_sha256
    payload = json.dumps(freeze, indent=2, sort_keys=True).encode() + b"\n"
    atomic_create(path, payload)
    return freeze


def run(root: Path = ROOT) -> None:
    """Execution entry point; deliberately refuses absent separate authorization."""
    root = root.resolve()
    freeze_path = root / OUT / "execution-freeze.json"
    if not freeze_path.is_file():
        raise ValueError("evaluation_execution_freeze_missing")
    freeze = read_json(freeze_path)
    if not execution_authorized(root, freeze):
        raise ValueError("separate_project_owner_evaluation_authorization_required")
    from ml.alphazero_lite.seed599_evaluation_arena import execute_registered

    execute_registered(root, freeze)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-training", action="store_true")
    parser.add_argument("--write-training-verification", action="store_true")
    parser.add_argument("--write-heldout-diagnostic", action="store_true")
    parser.add_argument("--prepare-bundles", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.root), indent=2, sort_keys=True))
    elif args.verify_training:
        print(
            json.dumps(
                independently_verify_training(args.root), indent=2, sort_keys=True
            )
        )
    elif args.write_training_verification:
        print(
            json.dumps(
                write_independent_training_report(args.root), indent=2, sort_keys=True
            )
        )
    elif args.write_heldout_diagnostic:
        print(
            json.dumps(
                write_heldout_policy_cross_entropy(args.root),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.prepare_bundles:
        print(json.dumps(export_candidate_bundles(args.root), indent=2, sort_keys=True))
    elif args.freeze:
        print(json.dumps(create_freeze(args.root), indent=2, sort_keys=True))
    elif args.run:
        run(args.root)
    else:
        parser.error(
            "select an explicit preflight, verification, preparation, freeze, or run action"
        )


if __name__ == "__main__":
    main()
