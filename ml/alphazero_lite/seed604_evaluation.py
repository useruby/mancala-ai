"""Preparation and authorization-gated execution for Seed 604 evaluation.

This module prepares an immutable assignment ledger and freeze. It never runs
games unless a separate evaluation-authorization record binds the exact freeze.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
COHORT = Path("docs/data/seed604-evaluation-opening-cohort-v5")
OUT = DATA / "evaluation-preparation"
FREEZE = OUT / "execution-freeze.json"
READINESS = OUT / "readiness-evidence.json"
ASSIGNMENTS = OUT / "assignment-manifest.json"
AUTHORIZATION = OUT / "evaluation-authorization.json"
RESULTS = OUT / "results"
ANALYSIS = OUT / "analysis.json"
CANDIDATES = {
    "B512": Path(
        "docs/data/seed599-replay-coverage/training-run-B/seed599-arm-B-update-512.npz"
    ),
    "B2048": DATA / "training-run-B2048/seed604-B2048-update-2048.npz",
}
EXPECTED = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "B2048": "e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874",
    "P-E4": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "opening_manifest": "445052b818b36f84953f73f280b20f7421c441fac6a74a0e2bb309cf907fb410",
    "native_probe": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
    "tablebase": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
}
SERVING = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4")
NATIVE = Path("model-artifact/runtime/kalah_v1_tablebase")
TABLEBASE = Path("model-artifact/runtime/kalah_v1_21.kvtb")
SEARCH = {
    "simulations": 128,
    "c_puct": 1.25,
    "workers": 1,
    "max_plies": 200,
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
    "noise": False,
    "exact_root_enabled": True,
    "exact_root_threshold": 16,
    "exact_root_objective": "final_score_margin",
    "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    "exact_leaf_solving": False,
}
SOURCES = (
    "ml/alphazero_lite/seed604_evaluation.py",
    "ml/alphazero_lite/verify_seed604_evaluation.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/requirements-seed599-evaluation.txt",
    "ml/alphazero_lite/export_artifact.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/test_seed604_evaluation.py",
    "ml/alphazero_lite/verify_seed604_training_result.py",
    "ml/alphazero_lite/seed604_training_executor.py",
)
ENVIRONMENT = {
    "PYTHONHASHSEED": "604",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"create_only_exists:{path}")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def authenticate_inputs(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    registration = json.loads((root / (DATA / "registration.json")).read_bytes())
    cohort_registration = json.loads(
        (root / (COHORT / "registration.json")).read_bytes()
    )
    if (
        registration.get("seed") != 604
        or registration.get("status") != "prepared-not-authorized"
    ):
        raise ValueError("seed604_registration_identity_mismatch")
    evaluation = registration.get("evaluation", {})
    registered_bootstrap = evaluation.get("bootstrap", {})
    if (
        evaluation.get("opening_groups") != 1024
        or evaluation.get("games") != 4096
        or evaluation.get("design")
        != "matched opening groups; seat-swapped games for each candidate; B512 and B2048 both play serving P-E4"
        or registered_bootstrap.get("resamples") != 20000
        or registered_bootstrap.get("seed") != 604
        or registered_bootstrap.get("quantile") != "numpy.quantile(method='linear')"
        or "strictly greater than 0.50" not in evaluation.get("primary_gate", "")
        or "strictly greater than zero" not in evaluation.get("secondary_gate", "")
    ):
        raise ValueError("registered_seed604_statistical_contract_mismatch")
    if sha(root / (COHORT / "opening-manifest.json")) != EXPECTED["opening_manifest"]:
        raise ValueError("registered_opening_manifest_hash_mismatch")
    if cohort_registration.get("manifest_sha256") != EXPECTED["opening_manifest"]:
        raise ValueError("registered_opening_registration_mismatch")
    seed599_contract_path = (
        root / "docs/data/seed599-replay-coverage/evaluation-contract.json"
    )
    if (
        sha(seed599_contract_path)
        != "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4"
    ):
        raise ValueError("seed599_search_contract_identity_mismatch")
    seed599_contract = json.loads(seed599_contract_path.read_bytes())
    match = seed599_contract.get("match_settings", {})
    mapped_search = {
        "simulations": match.get("simulations_per_side"),
        "c_puct": match.get("c_puct"),
        "workers": match.get("workers"),
        "max_plies": match.get("max_plies"),
        "fpu_mode": match.get("fpu_mode"),
        "reuse_subtree": match.get("reuse_subtree"),
        "normalize_values": match.get("normalize_values"),
        "root_policy_mode": match.get("root_policy_mode"),
        "root_temperature": match.get("root_temperature"),
        "tactical_root_bias": match.get("tactical_root_bias"),
        "noise": match.get("noise"),
        "exact_root_enabled": match.get("exact_root_enabled"),
        "exact_root_threshold": match.get("exact_root_threshold"),
        "exact_root_objective": match.get("exact_root_objective"),
        "exact_root_tie_rule": match.get("exact_root_tie_rule"),
        "exact_leaf_solving": match.get("exact_leaf_solving"),
    }
    if mapped_search != SEARCH:
        raise ValueError("seed604_and_seed599_search_contract_difference")
    if (
        seed599_contract.get("serving_p_e4", {}).get("tablebase_sha256")
        != EXPECTED["tablebase"]
    ):
        raise ValueError("serving_and_evaluation_tablebase_identity_difference")
    models = {}
    for candidate, relative in CANDIDATES.items():
        actual = sha(root / relative)
        if actual != EXPECTED[candidate]:
            raise ValueError(f"candidate_checkpoint_hash_mismatch:{candidate}")
        models[candidate] = {"path": str(relative), "sha256": actual}
    p4 = root / SERVING
    if sha(p4 / "model.npz") != EXPECTED["P-E4"]:
        raise ValueError("serving_checkpoint_hash_mismatch")
    serving_files = {
        "model.npz": "model_sha256",
        "metadata.json": "metadata_sha256",
        "weights.json": "weights_sha256",
        "search_policy.json": "search_policy_sha256",
    }
    serving_identity = seed599_contract.get("serving_p_e4", {})
    for filename, identity_key in serving_files.items():
        if sha(p4 / filename) != serving_identity.get(identity_key):
            raise ValueError(f"serving_registered_artifact_mismatch:{filename}")
    metadata = json.loads((p4 / "metadata.json").read_bytes())
    if metadata.get("artifacts", {}).get("weights_sha256") != sha(
        p4 / "model.npz"
    ) or metadata.get("artifacts", {}).get("weights_json_sha256") != sha(
        p4 / "weights.json"
    ):
        raise ValueError("serving_metadata_internal_identity_mismatch")
    for name, path in (("native_probe", NATIVE), ("tablebase", TABLEBASE)):
        if not (root / path).is_file() or sha(root / path) != EXPECTED[name]:
            raise ValueError(f"exact_root_asset_hash_mismatch:{name}")
    if (root / TABLEBASE).stat().st_size != 709635072:
        raise ValueError("exact_root_tablebase_size_mismatch")
    pointer_raw = subprocess.run(
        ["git", "show", f"HEAD:{TABLEBASE.as_posix()}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    pointer_text = pointer_raw.decode("ascii")
    if (
        f"oid sha256:{EXPECTED['tablebase']}" not in pointer_text
        or "size 709635072" not in pointer_text
    ):
        raise ValueError("tracked_tablebase_lfs_pointer_mismatch")
    tracking_rule = (
        "model-artifact/runtime/kalah_v1_21.kvtb filter=lfs diff=lfs merge=lfs -text"
    )
    if (
        tracking_rule
        not in (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("tablebase_lfs_tracking_rule_missing")
    serving = json.loads((p4 / "search_policy.json").read_bytes())
    if (
        serving.get("native_probe", {}).get("sha256") != EXPECTED["native_probe"]
        or serving.get("tablebase", {}).get("sha256") != EXPECTED["tablebase"]
    ):
        raise ValueError("serving_exact_root_identity_mismatch")
    return {
        "registration_sha256": sha(root / (DATA / "registration.json")),
        "seed599_evaluation_contract_sha256": sha(seed599_contract_path),
        "search_contract_equal_to_seed599": True,
        "cohort_registration_sha256": sha(root / (COHORT / "registration.json")),
        "cohort_verification_report_sha256": sha(
            root / (COHORT / "verification-report.json")
        ),
        "cohort_runtime_provenance_sha256": sha(
            root / (COHORT / "runtime-provenance.json")
        ),
        "opening_manifest_sha256": EXPECTED["opening_manifest"],
        "models": models,
        "serving_model_sha256": EXPECTED["P-E4"],
        "serving_metadata_sha256": sha(p4 / "metadata.json"),
        "serving_weights_sha256": sha(p4 / "weights.json"),
        "serving_search_policy_sha256": sha(p4 / "search_policy.json"),
        "native_probe_sha256": sha(root / NATIVE),
        "tablebase_sha256": sha(root / TABLEBASE),
        "tablebase_size": (root / TABLEBASE).stat().st_size,
        "lfs_pointer_sha256": hashlib.sha256(pointer_raw).hexdigest(),
        "lfs_pointer_oid": EXPECTED["tablebase"],
        "gitattributes_sha256": sha(root / ".gitattributes"),
    }


def deterministic_seed(opening_id: str, candidate: str, seat: int) -> int:
    payload = f"seed604\0{opening_id}\0{candidate}\0{seat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def build_assignments(root: Path = ROOT) -> list[dict[str, Any]]:
    manifest = json.loads((root / (COHORT / "opening-manifest.json")).read_bytes())
    openings = manifest.get("openings")
    if not isinstance(openings, list) or len(openings) != 1024:
        raise ValueError("registered_opening_count_mismatch")
    assignments = []
    for index, opening in enumerate(openings):
        if opening.get("opening_id") != f"seed604-opening-{index:04d}":
            raise ValueError("registered_opening_order_mismatch")
        for candidate in ("B512", "B2048"):
            for seat in (0, 1):
                game_id = f"seed604-{candidate}-{index:04d}-p{seat}"
                assignments.append(
                    {
                        "game_id": game_id,
                        "opening_index": index,
                        "opening_id": opening["opening_id"],
                        "opening_identity": opening["canonical_identity"],
                        "candidate": candidate,
                        "candidate_checkpoint_sha256": EXPECTED[candidate],
                        "opponent": "P-E4",
                        "opponent_checkpoint_sha256": EXPECTED["P-E4"],
                        "candidate_seat": seat,
                        "game_seed": deterministic_seed(
                            opening["opening_id"], candidate, seat
                        ),
                        "prefix_moves": opening["prefix_moves"],
                    }
                )
    ids = [row["game_id"] for row in assignments]
    if len(assignments) != 4096 or len(set(ids)) != 4096:
        raise ValueError("assignment_count_or_uniqueness_mismatch")
    return assignments


def prepare_runtime_artifacts(root: Path = ROOT) -> dict[str, Any]:
    """Export unchanged NPZ bytes to runtime-readable directories, create-only."""
    root = root.resolve()
    target = root / OUT / "runtime-artifacts"
    result = {}
    for candidate, source_rel in CANDIDATES.items():
        dest = target / candidate
        if dest.exists():
            raise FileExistsError(f"runtime_artifact_already_exists:{candidate}")
        command = [
            sys.executable,
            str(root / "ml/alphazero_lite/export_artifact.py"),
            "--checkpoint",
            str(root / source_rel),
            "--out-dir",
            str(dest),
            "--version",
            f"seed604-{candidate}-policy",
            "--model-type",
            "residual_v3",
            "--input-encoding",
            "kalah_v3",
        ]
        subprocess.run(command, cwd=root, check=True)
        if sha(dest / "model.npz") != EXPECTED[candidate]:
            raise ValueError(f"runtime_checkpoint_bytes_changed:{candidate}")
        parity = verify_synthetic_composition(root, dest)
        result[candidate] = {
            "model_npz_sha256": sha(dest / "model.npz"),
            "synthetic_composition": parity,
        }
    return result


def verify_synthetic_composition(
    root: Path, candidate_artifact: Path
) -> dict[str, Any]:
    """Check policy/value source selection on synthetic states and extra turns."""
    from ml.alphazero_lite import arena
    from ml.alphazero_lite.kalah_rules import KalahGame

    candidate = arena.ArtifactEvaluator(candidate_artifact)
    serving = arena.ArtifactEvaluator(root / SERVING)
    composed = arena.ComposedArtifactEvaluator(
        serving, candidate, policy_source="candidate", value_source="current"
    )
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    positions = 0
    players_seen: set[int] = set()
    extra_turns = 0
    while not game.over() and positions < 200:
        policy, _value = candidate.evaluate(game)
        _policy, serving_value = serving.evaluate(game)
        observed_policy, observed_value = composed.evaluate(game)
        if not np.array_equal(policy, observed_policy) or observed_value != float(
            serving_value
        ):
            raise ValueError("synthetic_evaluator_composition_mismatch")
        players_seen.add(game.current_player)
        player = game.current_player
        game.move(game.pit_index(game.possible_moves()[0]))
        extra_turns += int(game.current_player == player and not game.over())
        positions += 1
    if positions == 0 or players_seen != {0, 1} or extra_turns == 0:
        raise ValueError("synthetic_composition_did_not_cover_turn_mapping")
    return {
        "positions_checked": positions,
        "players_checked": sorted(players_seen),
        "extra_turns_checked": extra_turns,
        "policy_source": "candidate including candidate trunk",
        "value_source": "serving P-E4 including serving trunk",
        "registered_openings_used": False,
    }


def _freeze_document(root: Path = ROOT) -> dict[str, Any]:
    identities = authenticate_inputs(root)
    manifest_path = root / ASSIGNMENTS
    assignment_rows = build_assignments(root)
    manifest = {
        "schema": "seed604-game-assignment-manifest-v1",
        "assignments": assignment_rows,
    }
    manifest_raw = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_raw:
            raise ValueError("assignment_manifest_conflict")
    else:
        atomic_create(manifest_path, manifest_raw)
    runtime_dir = root / OUT / "runtime-artifacts"
    runtime_artifacts = {}
    for candidate in CANDIDATES:
        model = runtime_dir / candidate / "model.npz"
        if not model.is_file() or sha(model) != EXPECTED[candidate]:
            raise ValueError(f"runtime_artifact_not_prepared:{candidate}")
        runtime_artifacts[candidate] = {
            "directory": str((OUT / "runtime-artifacts" / candidate)),
            "model_npz_sha256": sha(model),
            "weights_json_sha256": sha(runtime_dir / candidate / "weights.json"),
            "metadata_sha256": sha(runtime_dir / candidate / "metadata.json"),
        }
    composition = {}
    for candidate in CANDIDATES:
        composition[candidate] = verify_synthetic_composition(
            root, runtime_dir / candidate
        )
    sources = {path: sha(root / path) for path in SOURCES}
    from ml.alphazero_lite.verify_seed604_training_result import verify_result

    training_verification = verify_result(root / DATA / "training-run-B2048", root=root)
    if training_verification.get("passed") is not True:
        raise ValueError("independent_training_result_verification_failed")
    return {
        "schema": "seed604-evaluation-execution-freeze-v1",
        "status": "frozen-awaiting-separate-evaluation-authorization",
        "evaluation_authorized": False,
        "registration": identities,
        "training_receipt_sha256": sha(
            root / (DATA / "training-run-B2048/training-receipt.json")
        ),
        "training_verification": training_verification,
        "runtime_artifacts": runtime_artifacts,
        "synthetic_composition_verification": composition,
        "assignment_manifest": {
            "path": str(ASSIGNMENTS),
            "sha256": sha(manifest_path),
            "game_count": 4096,
            "opening_count": 1024,
            "order": "opening, candidate B512 then B2048, seat 0 then 1",
            "seed_derivation": "SHA-256 of UTF-8 seed604 NUL opening_id NUL candidate NUL seat; first 8 bytes big-endian modulo 2^31",
        },
        "search": SEARCH,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependency_lock_sha256": sha(
                root / "ml/alphazero_lite/requirements-seed599-evaluation.txt"
            ),
            "device": "cpu",
            "workers_per_game": 1,
            "pair_parallelism": 18,
            "registered_cpus": 18,
            "registered_memory_gib": 64,
            "environment": ENVIRONMENT,
        },
        "statistics": {
            "unit": "opening group",
            "clusters": 1024,
            "resamples": 20000,
            "seed": 604,
            "quantile_method": "linear",
            "primary_lower_quantile": 0.05,
            "primary_strict_lower_bound_gt": 0.5,
            "secondary_quantiles": [0.025, 0.975],
            "secondary_strict_lower_endpoint_gt": 0.0,
            "interpretation": {
                "both_pass": "absolute strength and additional optimization benefit established",
                "primary_only": "absolute strength established; incremental improvement not established",
                "secondary_only": "incremental improvement established; absolute strength not established",
                "neither": "neither claim established",
            },
        },
        "output": {
            "freeze_path": str(FREEZE),
            "readiness_path": str(READINESS),
            "assignment_manifest_path": str(ASSIGNMENTS),
            "authorization_path": str(AUTHORIZATION),
            "results_directory": str(RESULTS),
            "result_schema": "seed604-game-result-v1",
            "attempt_directory_pattern": "results/<game_id>/attempts/attempt-NNNN/",
            "analysis_path": str(ANALYSIS),
            "analysis_schema": "seed604-cluster-bootstrap-analysis-v1",
            "create_only": True,
        },
        "source_sha256": sources,
        "protocol_document_sha256": sha(root / (OUT / "README.md")),
        "materialization": {
            "method": "git lfs pull origin --include=model-artifact/runtime/kalah_v1_21.kvtb --exclude=",
            "tablebase_size": identities["tablebase_size"],
            "tablebase_sha256": identities["tablebase_sha256"],
            "native_probe_sha256": identities["native_probe_sha256"],
            "lfs_pointer_sha256": identities["lfs_pointer_sha256"],
            "lfs_pointer_oid": identities["lfs_pointer_oid"],
            "gitattributes_sha256": identities["gitattributes_sha256"],
            "cohort_registration_sha256": identities["cohort_registration_sha256"],
            "git_lfs_version": subprocess.run(
                ["git", "lfs", "version"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    }


def publish_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    document = _freeze_document(root)
    freeze_raw = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
    atomic_create(root / FREEZE, freeze_raw)
    readiness = {
        "schema": "seed604-evaluation-readiness-evidence-v1",
        "execution_freeze_sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "evaluation_authorized": False,
        "games_started": False,
        "registered_openings": 1024,
        "assignment_count": 4096,
        "checkpoint_identities": document["registration"]["models"],
        "tablebase": {
            "path": str(TABLEBASE),
            "size": document["materialization"]["tablebase_size"],
            "sha256": document["materialization"]["tablebase_sha256"],
            "materialization_provenance": document["materialization"],
        },
    }
    atomic_create(
        root / READINESS,
        json.dumps(readiness, indent=2, sort_keys=True).encode() + b"\n",
    )
    return document


def score_game(candidate_seat: int, margin: int) -> float:
    if candidate_seat not in (0, 1):
        raise ValueError("invalid_candidate_seat")
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def paired_opening_bootstrap(
    b512: np.ndarray,
    b2048: np.ndarray,
    *,
    resamples: int = 20000,
    seed: int = 604,
) -> tuple[float, list[float]]:
    if (
        b512.ndim != 1
        or b2048.ndim != 1
        or b512.shape != b2048.shape
        or b512.size != 1024
        or resamples != 20000
        or seed != 604
    ):
        raise ValueError("registered_bootstrap_contract_mismatch")
    rng = np.random.default_rng(604)
    indices = rng.integers(0, 1024, size=(20000, 1024))
    primary = b2048[indices].mean(axis=1)
    paired_difference = (b2048 - b512)[indices].mean(axis=1)
    lower = float(np.quantile(primary, 0.05, method="linear"))
    interval = [
        float(value)
        for value in np.quantile(paired_difference, [0.025, 0.975], method="linear")
    ]
    return lower, interval


def interpret_gates(primary_lower: float, secondary_lower: float) -> dict[str, Any]:
    primary = primary_lower > 0.50
    secondary = secondary_lower > 0.0
    label = (
        "both_pass"
        if primary and secondary
        else "primary_only"
        if primary
        else "secondary_only"
        if secondary
        else "neither"
    )
    return {
        "primary_pass": primary,
        "secondary_pass": secondary,
        "interpretation": label,
    }


def analyze(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scores) != 4096:
        raise ValueError("incomplete_game_ledger")
    expected = {
        (i, candidate, seat)
        for i in range(1024)
        for candidate in CANDIDATES
        for seat in (0, 1)
    }
    observed = {
        (row["opening_index"], row["candidate"], row["candidate_seat"])
        for row in scores
    }
    if observed != expected or len(observed) != len(scores):
        raise ValueError("missing_or_duplicate_assignment")
    by_opening: dict[int, dict[str, list[float]]] = {
        i: {candidate: [] for candidate in CANDIDATES} for i in range(1024)
    }
    for row in scores:
        by_opening[row["opening_index"]][row["candidate"]].append(
            score_game(row["candidate_seat"], row["margin"])
        )
    b512 = np.asarray([np.mean(by_opening[i]["B512"]) for i in range(1024)])
    b2048 = np.asarray([np.mean(by_opening[i]["B2048"]) for i in range(1024)])
    primary, interval = paired_opening_bootstrap(b512, b2048)
    gates = interpret_gates(primary, interval[0])
    return {
        "B512_score": float(b512.mean()),
        "B2048_score": float(b2048.mean()),
        "B2048_primary_lower_bound": primary,
        "B2048_minus_B512_interval": interval,
        **gates,
    }


def authorized(root: Path = ROOT) -> bool:
    root = root.resolve()
    freeze_path, auth_path = root / FREEZE, root / AUTHORIZATION
    if not freeze_path.is_file() or not auth_path.is_file():
        return False
    freeze = json.loads(freeze_path.read_bytes())
    auth = json.loads(auth_path.read_bytes())
    return (
        freeze.get("evaluation_authorized") is False
        and freeze.get("schema") == "seed604-evaluation-execution-freeze-v1"
        and auth.get("schema") == "seed604-evaluation-authorization-v1"
        and auth.get("status") == "authorized-for-evaluation"
        and auth.get("evaluation_freeze_sha256") == sha(freeze_path)
        and auth.get("authorization", {}).get("type")
        == "explicit-project-owner-authorization"
        and isinstance(auth.get("authorization", {}).get("authorized_by"), str)
        and bool(auth.get("authorization", {}).get("authorized_by", "").strip())
        and auth.get("authorization", {}).get("run_games") is True
    )


def _arena_command(
    root: Path, assignment: dict[str, Any], opening_path: Path, out_dir: Path
) -> list[str]:
    candidate = assignment["candidate"]
    candidate_artifact = root / OUT / "runtime-artifacts" / candidate
    options = {
        key: SEARCH[key]
        for key in (
            "fpu_mode",
            "reuse_subtree",
            "normalize_values",
            "root_policy_mode",
            "root_temperature",
            "tactical_root_bias",
        )
    }
    return [
        sys.executable,
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(candidate_artifact),
        "--current",
        str(root / SERVING),
        "--challenger-policy-artifact",
        str(candidate_artifact),
        "--challenger-value-artifact",
        str(root / SERVING),
        "--current-policy-artifact",
        str(root / SERVING),
        "--current-value-artifact",
        str(root / SERVING),
        "--games",
        "1",
        "--games-per-opening",
        "1",
        "--workers",
        "1",
        "--challenger-simulations",
        "128",
        "--current-simulations",
        "128",
        "--c-puct",
        "1.25",
        "--seed-contract",
        "azlite_eval_seed_v2",
        "--challenger-search-options-json",
        json.dumps(options, sort_keys=True, separators=(",", ":")),
        "--current-search-options-json",
        json.dumps(options, sort_keys=True, separators=(",", ":")),
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--challenger-starts",
        str(assignment["candidate_seat"]),
        "--exact-root-solve-threshold",
        "16",
        "--exact-root-native-probe",
        str(root / NATIVE),
        "--exact-root-tablebase",
        str(root / TABLEBASE),
        "--opening-prefixes-jsonl",
        str(opening_path),
        "--suite-sha256",
        sha(opening_path),
        "--game-jsonl",
        str(out_dir / "game.jsonl"),
        "--out",
        str(out_dir / "report.json"),
        "--seed",
        str(assignment["game_seed"]),
    ]


def _execute_one(root: Path, assignment: dict[str, Any]) -> None:
    destination = root / RESULTS / assignment["game_id"]
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "result.json").exists():
        return
    attempts = destination / "attempts"
    attempts.mkdir(exist_ok=True)
    prior = sorted(path for path in attempts.iterdir() if path.is_dir())
    for path in prior:
        if (path / "game.jsonl").exists() or (path / "report.json").exists():
            raise ValueError(
                f"unreconciled_prior_attempt_preserved:{assignment['game_id']}"
            )
    attempt = attempts / f"attempt-{len(prior) + 1:04d}"
    attempt.mkdir()
    opening_path = attempt / "opening.jsonl"
    opening = {"prefix_moves": assignment["prefix_moves"]}
    atomic_create(opening_path, canonical(opening) + b"\n")
    command = _arena_command(root, assignment, opening_path, attempt)
    atomic_create(attempt / "command.json", canonical(command) + b"\n")
    env = os.environ.copy()
    env.update(ENVIRONMENT)
    subprocess.run(command, cwd=root, env=env, check=True)
    rows = [
        json.loads(line)
        for line in (attempt / "game.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) != 1:
        raise ValueError("arena_game_record_count_mismatch")
    from ml.alphazero_lite.verify_seed604_evaluation import verify_game

    verified = verify_game(assignment, rows[0])
    entry = {
        "schema": "seed604-game-result-v1",
        **assignment,
        **verified,
        "attempt_directory": str(Path("attempts") / attempt.name),
        "game_sha256": sha(attempt / "game.jsonl"),
        "report_sha256": sha(attempt / "report.json"),
        "command_sha256": sha(attempt / "command.json"),
    }
    atomic_create(destination / "result.json", canonical(entry) + b"\n")


def _execute_seat_pair(root: Path, assignments: list[dict[str, Any]]) -> None:
    if len(assignments) != 2 or {row["candidate_seat"] for row in assignments} != {
        0,
        1,
    }:
        raise ValueError("seat_pair_assignment_invalid")
    for assignment in sorted(assignments, key=lambda row: row["candidate_seat"]):
        _execute_one(root, assignment)


def run(root: Path = ROOT) -> None:
    root = root.resolve()
    if not authorized(root):
        raise ValueError("separate_seed604_evaluation_authorization_required")
    freeze = json.loads((root / FREEZE).read_bytes())
    verifier = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.verify_seed604_evaluation",
            "--preflight",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    del verifier
    manifest = json.loads((root / ASSIGNMENTS).read_bytes())
    if sha(root / ASSIGNMENTS) != freeze["assignment_manifest"]["sha256"]:
        raise ValueError("assignment_manifest_freeze_mismatch")
    assignments = manifest["assignments"]
    result_directory = root / RESULTS
    completed = (
        {path.parent.name for path in result_directory.glob("*/result.json")}
        if result_directory.is_dir()
        else set()
    )
    pairs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in assignments:
        pairs.setdefault((row["opening_index"], row["candidate"]), []).append(row)
    pending = [
        pair
        for pair in pairs.values()
        if any(row["game_id"] not in completed for row in pair)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        futures = [pool.submit(_execute_seat_pair, root, pair) for pair in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def analyze_completed(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    from ml.alphazero_lite.verify_seed604_evaluation import verify_manifest

    verification = verify_manifest(root)
    if verification["completed_games"] != 4096:
        raise ValueError("cannot_analyze_incomplete_evaluation")
    rows = [
        json.loads(path.read_bytes())
        for path in sorted((root / RESULTS).glob("*/result.json"))
    ]
    summary = {
        "schema": "seed604-cluster-bootstrap-analysis-v1",
        "games": 4096,
        "opening_clusters": 1024,
        "capped_games": verification["capped_games"],
        "bootstrap": {
            "resamples": 20000,
            "seed": 604,
            "quantile_method": "linear",
            "paired_opening_indices": True,
            "seat_outcomes_preserved": True,
        },
        **analyze(rows),
    }
    atomic_create(
        root / ANALYSIS, json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--prepare-runtime-artifacts", action="store_true")
    parser.add_argument("--publish-freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(
            json.dumps(
                {
                    "passed": True,
                    "inputs": authenticate_inputs(),
                    "assignments": len(build_assignments()),
                    "games_started": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.prepare_runtime_artifacts:
        print(json.dumps(prepare_runtime_artifacts(), indent=2, sort_keys=True))
    elif args.publish_freeze:
        print(json.dumps(publish_freeze(), indent=2, sort_keys=True))
    elif args.run:
        run()
    elif args.analyze:
        print(json.dumps(analyze_completed(), indent=2, sort_keys=True))
    else:
        parser.error("choose one explicit operation")


if __name__ == "__main__":
    main()
