"""Preparation and authorization-gated Seed 615 playing-strength evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed615-source-opening-diversity")
OUT = DATA / "evaluation-preparation"
REGISTRATION = DATA / "registration.json"
OPENINGS = DATA / "evaluation-openings.jsonl"
ASSIGNMENTS = DATA / "evaluation-assignments.jsonl"
TRAINING = DATA / "training-run-S615"
SERVING = Path("docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4")
NATIVE = Path("model-artifact/runtime/kalah_v1_tablebase")
TABLEBASE = Path("model-artifact/runtime/kalah_v1_21.kvtb")
FREEZE = OUT / "execution-freeze-v3.json"
READINESS = OUT / "readiness-evidence-v3.json"
SUPERSEDED_FREEZE = OUT / "execution-freeze-v2.json"
AUTHORIZATION = OUT / "evaluation-authorization.json"
RESULTS = OUT / "results"
ANALYSIS = OUT / "analysis.json"
EXPECTED = {
    "B512": "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
    "S615": "cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780",
    "P-E4": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
    "openings": "9bc2bbfca07ea356343c25e94593ab638b16238895fce5062f904d2cc2092cc0",
    "assignments": "8fb71d6ee30ad6870d403843dbe1a4e798f0218617cee5fce21964372a8514ca",
    "native_probe": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
    "tablebase": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
    "serving_metadata": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
    "serving_weights": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
    "serving_search_policy": "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
}
CHECKPOINTS = {
    "B512": Path(
        "docs/data/seed599-replay-coverage/training-run-B/seed599-arm-B-update-512.npz"
    ),
    "S615": TRAINING / "seed615-S615-update-512.npz",
}
SEARCH = {
    "simulations": 128,
    "c_puct": 1.25,
    "workers": 1,
    "root_temperature": 0.0,
    "root_policy_mode": "deterministic",
    "fpu_mode": "zero",
    "noise": False,
    "reuse_subtree": False,
    "normalize_values": False,
    "tactical_root_bias": 0.0,
    "exact_root_enabled": True,
    "exact_root_threshold": 16,
    "exact_root_objective": "final_score_margin",
    "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    "exact_leaf_solving": False,
    "max_plies": 200,
}
ENVIRONMENT = {
    "PYTHONHASHSEED": "615",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
SOURCE_PATHS = (
    "ml/alphazero_lite/seed615_evaluation.py",
    "ml/alphazero_lite/verify_seed615_evaluation.py",
    "ml/alphazero_lite/test_seed615_evaluation.py",
    "ml/alphazero_lite/seed604_evaluation.py",
    "ml/alphazero_lite/verify_seed604_evaluation.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/export_artifact.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/requirements-seed599-evaluation.txt",
    "ml/alphazero_lite/seed615_training_executor.py",
    "ml/alphazero_lite/verify_seed615_training_result.py",
    "ml/alphazero_lite/verify_seed615_opening_diversity.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_asset(
    path: Path, expected_digest: str, expected_size: int, label: str
) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha(path) != expected_digest
    ):
        raise ValueError(f"{label}_asset_identity_mismatch")


def validate_registered_digest(observed: str, expected: str, label: str) -> None:
    if observed != expected:
        raise ValueError(f"registered_{label}_identity_mismatch")


def validate_preflight_artifact_state(
    authorization_exists: bool, results_exists: bool, analysis_exists: bool
) -> None:
    if authorization_exists or results_exists or analysis_exists:
        raise ValueError("unexpected_registered_evaluation_artifacts")


def validate_no_uncommitted_attempt(game_directory: Path) -> None:
    if game_directory.exists():
        raise ValueError("incomplete_attempt_preserved")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest_jsonl_rows(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical(rows)).hexdigest()


def runtime_identity(root: Path = ROOT) -> dict[str, Any]:
    import torch

    expected_environment = ENVIRONMENT
    observed_environment = {key: os.environ.get(key) for key in expected_environment}
    if observed_environment != expected_environment:
        raise ValueError("evaluation_environment_variable_mismatch")
    if platform.machine() != "x86_64" or sys.version_info[:3] != (3, 14, 7):
        raise ValueError("evaluation_runtime_platform_mismatch")
    if np.__version__ != "2.5.3" or str(torch.__version__) != "2.14.0+cu130":
        raise ValueError("evaluation_runtime_dependency_mismatch")
    if torch.cuda.is_available():
        raise ValueError("evaluation_runtime_cuda_available")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1:
        raise ValueError("evaluation_torch_thread_count_mismatch")
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "workers_per_game": 1,
        "pair_parallelism": 18,
        "registered_cpus": 18,
        "registered_memory_gib": 64,
        "environment": observed_environment,
        "dependency_lock": "ml/alphazero_lite/requirements-seed599-evaluation.txt",
        "dependency_lock_sha256": sha(
            root / "ml/alphazero_lite/requirements-seed599-evaluation.txt"
        ),
    }


def expected_assignments(openings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, opening in enumerate(openings):
        opening_id = f"seed615-opening-{index:04d}"
        if (
            opening.get("opening_id") != opening_id
            or len(opening.get("prefix_moves", [])) != 12
        ):
            raise ValueError("registered_opening_order_or_prefix_mismatch")
        for candidate in ("B512", "S615"):
            for seat in (0, 1):
                digest = hashlib.sha256(
                    f"seed615\0{opening_id}\0{candidate}\0{seat}".encode()
                ).digest()
                records.append(
                    {
                        "candidate": candidate,
                        "candidate_seat": seat,
                        "game_id": f"seed615-{candidate}-{index:04d}-p{seat}",
                        "game_seed": int.from_bytes(digest[:8], "big") % (2**31),
                        "opening_id": opening_id,
                        "opening_identity": opening["canonical_identity"],
                        "opening_index": index,
                        "opponent": "P-E4",
                    }
                )
    return records


def authenticate_inputs(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    registration = json.loads((root / REGISTRATION).read_bytes())
    opening_rows = read_jsonl(root / OPENINGS)
    assignment_rows = read_jsonl(root / ASSIGNMENTS)
    validate_registered_digest(
        digest_jsonl_rows(opening_rows), EXPECTED["openings"], "opening_manifest"
    )
    if len(opening_rows) != 1024:
        raise ValueError("registered_opening_identity_mismatch")
    validate_registered_digest(
        digest_jsonl_rows(assignment_rows),
        EXPECTED["assignments"],
        "assignment_manifest",
    )
    if len(assignment_rows) != 4096:
        raise ValueError("registered_assignment_identity_mismatch")
    if assignment_rows != expected_assignments(opening_rows):
        raise ValueError("registered_assignment_seed_or_order_mismatch")
    from ml.alphazero_lite.verify_seed615_opening_diversity import (
        validate_opening_prefix,
    )
    from ml.alphazero_lite.verify_seed615_opening_diversity import (
        verify as verify_registered_openings,
    )

    opening_audit = verify_registered_openings(root)
    if opening_audit.get("passed") is not True:
        raise ValueError("independent_registered_opening_verification_failed")

    canonical_ids: set[str] = set()
    float_ids: set[str] = set()
    for row in opening_rows:
        state = validate_opening_prefix(row["prefix_moves"])
        if state != row["state"]:
            raise ValueError("registered_opening_state_mismatch")
        if (
            row["canonical_identity"] in canonical_ids
            or row["float32_identity"] in float_ids
        ):
            raise ValueError("registered_opening_duplicate")
        canonical_ids.add(row["canonical_identity"])
        float_ids.add(row["float32_identity"])
    if (
        registration.get("seed") != 615
        or registration.get("status") != "proposed-for-review-not-authorized"
        or registration.get("authorization", {}).get("currently_authorized")
        is not False
    ):
        raise ValueError("seed615_registration_status_invalid")
    for model, path in CHECKPOINTS.items():
        if sha(root / path) != EXPECTED[model]:
            raise ValueError(f"checkpoint_identity_mismatch:{model}")
    serving_dir = root / SERVING
    if sha(serving_dir / "model.npz") != EXPECTED["P-E4"]:
        raise ValueError("serving_checkpoint_identity_mismatch")
    serving_expected_files = {
        "metadata.json": EXPECTED["serving_metadata"],
        "weights.json": EXPECTED["serving_weights"],
        "search_policy.json": EXPECTED["serving_search_policy"],
    }
    for filename, digest in serving_expected_files.items():
        if sha(serving_dir / filename) != digest:
            raise ValueError(f"serving_artifact_identity_mismatch:{filename}")
    metadata = json.loads((serving_dir / "metadata.json").read_bytes())
    policy = json.loads((serving_dir / "search_policy.json").read_bytes())
    if (
        metadata.get("artifacts", {}).get("weights_sha256") != EXPECTED["P-E4"]
        or metadata.get("artifacts", {}).get("weights_json_sha256")
        != EXPECTED["serving_weights"]
        or policy.get("native_probe", {}).get("sha256") != EXPECTED["native_probe"]
        or policy.get("tablebase", {}).get("sha256") != EXPECTED["tablebase"]
    ):
        raise ValueError("serving_runtime_internal_identity_mismatch")
    evaluation = registration.get("evaluation", {})
    registered_search = evaluation.get("search", {})
    if (
        registered_search.get("simulations_per_side") != SEARCH["simulations"]
        or registered_search.get("c_puct") != SEARCH["c_puct"]
        or registered_search.get("workers_per_game") != SEARCH["workers"]
        or registered_search.get("root_temperature") != SEARCH["root_temperature"]
        or registered_search.get("root_policy_mode") != SEARCH["root_policy_mode"]
        or registered_search.get("tactical_root_bias") != SEARCH["tactical_root_bias"]
        or registered_search.get("fpu_mode") != SEARCH["fpu_mode"]
        or registered_search.get("reuse_subtree") != SEARCH["reuse_subtree"]
        or registered_search.get("normalize_values") != SEARCH["normalize_values"]
        or registered_search.get("noise") != SEARCH["noise"]
        or registered_search.get("max_plies") != SEARCH["max_plies"]
        or registered_search.get("exact_root_enabled") != SEARCH["exact_root_enabled"]
        or registered_search.get("exact_root_threshold")
        != SEARCH["exact_root_threshold"]
        or registered_search.get("exact_root_objective")
        != SEARCH["exact_root_objective"]
        or registered_search.get("exact_root_tie_rule") != SEARCH["exact_root_tie_rule"]
        or registered_search.get("exact_leaf_solving") != SEARCH["exact_leaf_solving"]
        or registered_search.get("value_evaluator")
        != "unchanged serving P-E4 for both candidate policies"
    ):
        raise ValueError("registered_search_contract_mismatch")
    probe, tablebase = root / NATIVE, root / TABLEBASE
    validate_asset(probe, EXPECTED["native_probe"], 106776, "native_probe")
    validate_asset(
        tablebase, EXPECTED["tablebase"], 709635072, "materialized_tablebase"
    )
    pointer = subprocess.run(
        ["git", "show", f"HEAD:{TABLEBASE.as_posix()}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.decode("ascii")
    if (
        f"oid sha256:{EXPECTED['tablebase']}" not in pointer
        or "size 709635072" not in pointer
    ):
        raise ValueError("tablebase_lfs_pointer_provenance_mismatch")
    training_dir = root / TRAINING
    from ml.alphazero_lite.verify_seed615_training_result import verify_result

    training = verify_result(training_dir, root=root)
    receipt = training_dir / "training-receipt.json"
    return {
        "registration_sha256": sha(root / REGISTRATION),
        "opening_manifest_sha256": EXPECTED["openings"],
        "opening_file_sha256": sha(root / OPENINGS),
        "assignment_manifest_sha256": EXPECTED["assignments"],
        "assignment_file_sha256": sha(root / ASSIGNMENTS),
        "opening_verification": opening_audit,
        "models": {
            model: {"path": str(path), "sha256": EXPECTED[model]}
            for model, path in CHECKPOINTS.items()
        },
        "serving_model_sha256": EXPECTED["P-E4"],
        "serving_artifact_hashes": {
            name: sha(root / SERVING / name)
            for name in (
                "model.npz",
                "metadata.json",
                "weights.json",
                "search_policy.json",
            )
        },
        "training_receipt_sha256": sha(receipt),
        "training_verification": training,
        "native_probe_sha256": sha(probe),
        "tablebase_sha256": sha(tablebase),
        "tablebase_size": tablebase.stat().st_size,
        "lfs_pointer_sha256": hashlib.sha256(pointer.encode("ascii")).hexdigest(),
    }


def prepare_runtime_artifacts(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    authenticate_inputs(root)
    target = root / OUT / "runtime-artifacts"
    result = {}
    for name, source in CHECKPOINTS.items():
        destination = target / name
        if destination.exists():
            raise FileExistsError(f"runtime_artifact_already_exists:{name}")
        subprocess.run(
            [
                sys.executable,
                str(root / "ml/alphazero_lite/export_artifact.py"),
                "--checkpoint",
                str(root / source),
                "--out-dir",
                str(destination),
                "--version",
                f"seed615-{name}-policy",
                "--model-type",
                "residual_v3",
                "--input-encoding",
                "kalah_v3",
            ],
            cwd=root,
            check=True,
        )
        if sha(destination / "model.npz") != EXPECTED[name]:
            raise ValueError(f"runtime_checkpoint_bytes_changed:{name}")
        from ml.alphazero_lite.seed604_evaluation import verify_synthetic_composition

        composition = verify_synthetic_composition(root, destination)
        result[name] = {
            "model_npz_sha256": sha(destination / "model.npz"),
            "weights_json_sha256": sha(destination / "weights.json"),
            "metadata_sha256": sha(destination / "metadata.json"),
            "synthetic_composition": composition,
        }
    return result


def build_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    identities = authenticate_inputs(root)
    runtime = runtime_identity(root)
    runtime_artifacts = {}
    composition_evidence = {}
    for candidate in ("B512", "S615"):
        folder = root / OUT / "runtime-artifacts" / candidate
        metadata = json.loads((folder / "metadata.json").read_bytes())
        if sha(folder / "model.npz") != EXPECTED[candidate]:
            raise ValueError(f"runtime_model_identity_mismatch:{candidate}")
        if (
            metadata.get("architecture", {}).get("model_type") != "residual_v3"
            or metadata.get("input_encoding") != "kalah_v3"
            or metadata.get("artifacts", {}).get("weights_sha256")
            != EXPECTED[candidate]
            or metadata.get("artifacts", {}).get("weights_json_sha256")
            != sha(folder / "weights.json")
        ):
            raise ValueError(f"runtime_architecture_mismatch:{candidate}")
        runtime_artifacts[candidate] = {
            "directory": str(OUT / "runtime-artifacts" / candidate),
            "model_npz_sha256": sha(folder / "model.npz"),
            "weights_json_sha256": sha(folder / "weights.json"),
            "metadata_sha256": sha(folder / "metadata.json"),
        }
        from ml.alphazero_lite.seed604_evaluation import verify_synthetic_composition

        composition_evidence[candidate] = verify_synthetic_composition(root, folder)
    assignments = read_jsonl(root / ASSIGNMENTS)
    opening_rows = read_jsonl(root / OPENINGS)
    sources = {path: sha(root / path) for path in SOURCE_PATHS}
    statistics = {
        "clusters": 1024,
        "resamples": 20000,
        "seed": 615,
        "generator": "numpy.random.default_rng(615)",
        "quantile_method": "numpy.quantile(method='linear')",
        "primary_quantiles": [0.025, 0.975],
        "primary_strict_lower_endpoint_gt": 0.0,
        "serving_lower_quantile": 0.05,
        "serving_strict_lower_bound_gt": 0.50,
        "paired_opening_indices": True,
        "preserve_both_seat_outcomes": True,
        "interpretations": {
            "both_pass": "incremental improvement and absolute strength established",
            "incremental_only": "improvement over B512 established; absolute strength not established; do not promote automatically",
            "absolute_only": "absolute strength established; incremental improvement not established",
            "neither": "neither claim established",
        },
    }
    return {
        "schema": "seed615-evaluation-execution-freeze-v1",
        "supersedes_freeze_sha256": sha(root / SUPERSEDED_FREEZE)
        if (root / SUPERSEDED_FREEZE).is_file()
        else None,
        "status": "frozen-awaiting-separate-evaluation-authorization",
        "evaluation_authorized": False,
        "games_started": False,
        "registration": identities,
        "training_receipt_sha256": identities["training_receipt_sha256"],
        "training_verification": identities["training_verification"],
        "serving_model_sha256": EXPECTED["P-E4"],
        "serving_runtime_directory": str(SERVING),
        "runtime_artifacts": runtime_artifacts,
        "synthetic_composition_verification": composition_evidence,
        "cohort": {
            "openings_path": str(OPENINGS),
            "opening_count": len(opening_rows),
            "canonical_sha256": EXPECTED["openings"],
            "file_sha256": sha(root / OPENINGS),
            "assignments_path": str(ASSIGNMENTS),
            "assignment_count": len(assignments),
            "canonical_sha256_assignments": EXPECTED["assignments"],
            "file_sha256_assignments": sha(root / ASSIGNMENTS),
            "assignment_order": "registered order; opening, B512 then S615, seat 0 then 1",
            "seed_derivation": "SHA-256 UTF-8 seed615 NUL opening_id NUL candidate NUL seat; first 8 bytes big-endian modulo 2^31",
        },
        "search": SEARCH,
        "runtime": runtime,
        "native_assets": {
            "native_probe_sha256": identities["native_probe_sha256"],
            "tablebase_sha256": identities["tablebase_sha256"],
            "tablebase_size": identities["tablebase_size"],
            "lfs_pointer_sha256": identities["lfs_pointer_sha256"],
            "gitattributes_sha256": sha(root / ".gitattributes"),
            "git_lfs_version": subprocess.run(
                ["git", "lfs", "version"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "materialization_command": "git lfs pull origin --include=model-artifact/runtime/kalah_v1_21.kvtb --exclude=",
        },
        "statistics": statistics,
        "source_sha256": sources,
        "protocol_document_sha256": sha(root / (OUT / "README.md")),
        "output": {
            "freeze_path": str(FREEZE),
            "readiness_path": str(READINESS),
            "authorization_path": str(AUTHORIZATION),
            "results_directory": str(RESULTS),
            "analysis_path": str(ANALYSIS),
            "result_schema": "seed615-game-result-v1",
            "create_only": True,
            "preserve_incomplete_attempts": True,
            "stop_on_conflict": True,
        },
    }


def publish_freeze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    validate_preflight_artifact_state(
        (root / AUTHORIZATION).exists(),
        (root / RESULTS).exists(),
        (root / ANALYSIS).exists(),
    )
    freeze = build_freeze(root)
    raw = json.dumps(freeze, indent=2, sort_keys=True).encode() + b"\n"
    atomic_create(root / FREEZE, raw)
    readiness = {
        "schema": "seed615-evaluation-readiness-evidence-v1",
        "execution_freeze_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_authorized": False,
        "games_started": False,
        "registered_openings": 1024,
        "assignment_count": 4096,
        "checkpoint_identities": identities_for_readiness(freeze),
        "tablebase": {
            "path": str(TABLEBASE),
            "size": freeze["registration"]["tablebase_size"],
            "sha256": freeze["registration"]["tablebase_sha256"],
            "lfs_pointer_sha256": freeze["registration"]["lfs_pointer_sha256"],
        },
    }
    atomic_create(
        root / READINESS,
        json.dumps(readiness, indent=2, sort_keys=True).encode() + b"\n",
    )
    return freeze


def identities_for_readiness(freeze: dict[str, Any]) -> dict[str, Any]:
    return {
        "B512": EXPECTED["B512"],
        "S615": EXPECTED["S615"],
        "P-E4": EXPECTED["P-E4"],
        "training_receipt_sha256": freeze["training_receipt_sha256"],
    }


def paired_bootstrap(
    b512: np.ndarray, s615: np.ndarray, *, resamples: int = 20000, seed: int = 615
) -> tuple[list[float], float]:
    if (
        b512.shape != (1024,)
        or s615.shape != (1024,)
        or resamples != 20000
        or seed != 615
    ):
        raise ValueError("registered_bootstrap_contract_mismatch")
    indices = np.random.default_rng(615).integers(0, 1024, size=(20000, 1024))
    difference = np.mean((s615 - b512)[indices], axis=1)
    absolute = np.mean(s615[indices], axis=1)
    return (
        [
            float(item)
            for item in np.quantile(difference, [0.025, 0.975], method="linear")
        ],
        float(np.quantile(absolute, 0.05, method="linear")),
    )


def score_game(seat: int, margin: int) -> float:
    if seat not in (0, 1):
        raise ValueError("invalid_candidate_seat")
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def analyze_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 4096:
        raise ValueError("incomplete_game_ledger")
    observed = {
        (row["opening_index"], row["candidate"], row["candidate_seat"]) for row in rows
    }
    expected = {
        (index, candidate, seat)
        for index in range(1024)
        for candidate in ("B512", "S615")
        for seat in (0, 1)
    }
    if observed != expected or len(observed) != len(rows):
        raise ValueError("missing_or_duplicate_assignment")
    means = {name: np.empty(1024, dtype=np.float64) for name in ("B512", "S615")}
    for candidate in means:
        for index in range(1024):
            group = [
                row
                for row in rows
                if row["candidate"] == candidate and row["opening_index"] == index
            ]
            if {row["candidate_seat"] for row in group} != {0, 1}:
                raise ValueError("broken_opening_cluster_pairing")
            means[candidate][index] = np.mean(
                [score_game(row["candidate_seat"], row["margin"]) for row in group]
            )
    interval, lower = paired_bootstrap(means["B512"], means["S615"])
    incremental = interval[0] > 0.0
    absolute = lower > 0.50
    interpretation = (
        "both_pass"
        if incremental and absolute
        else "incremental_only"
        if incremental
        else "absolute_only"
        if absolute
        else "neither"
    )
    return {
        "B512_score": float(means["B512"].mean()),
        "S615_score": float(means["S615"].mean()),
        "S615_minus_B512_interval": interval,
        "S615_vs_P_E4_one_sided_lower_bound": lower,
        "incremental_improvement_pass": incremental,
        "absolute_strength_pass": absolute,
        "interpretation": interpretation,
        "automatic_promotion": False,
    }


def _arena_command(
    root: Path, assignment: dict[str, Any], opening_path: Path, attempt: Path
) -> list[str]:
    candidate = root / OUT / "runtime-artifacts" / assignment["candidate"]
    serving = root / SERVING
    options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
    }
    return [
        sys.executable,
        "-m",
        "ml.alphazero_lite.arena",
        "--challenger",
        str(candidate),
        "--current",
        str(serving),
        "--challenger-policy-artifact",
        str(candidate),
        "--challenger-value-artifact",
        str(serving),
        "--current-policy-artifact",
        str(serving),
        "--current-value-artifact",
        str(serving),
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
        str(attempt / "game.jsonl"),
        "--out",
        str(attempt / "report.json"),
        "--seed",
        str(assignment["game_seed"]),
    ]


def run(root: Path = ROOT) -> None:
    """Execute only after explicit authorization bound to the exact freeze."""
    from ml.alphazero_lite.verify_seed615_evaluation import (
        verify_authorization,
        verify_one_result,
        verify_freeze,
        verify_game,
    )

    root = root.resolve()
    verify_freeze(root, allow_execution_artifacts=True)
    verify_authorization(root)
    freeze = json.loads((root / FREEZE).read_bytes())
    if freeze.get("evaluation_authorized") is not False:
        raise ValueError("freeze_must_remain_unauthorized")
    assignments = read_jsonl(root / ASSIGNMENTS)
    openings = {row["opening_id"]: row for row in read_jsonl(root / OPENINGS)}
    result_root = root / RESULTS
    result_root.mkdir(parents=True, exist_ok=True)
    pairs: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for assignment in assignments:
        pairs.setdefault(
            (assignment["opening_index"], assignment["candidate"]), []
        ).append(assignment)
    registered_ids = {row["game_id"] for row in assignments}
    for existing in result_root.iterdir():
        if existing.name not in registered_ids:
            raise ValueError(f"unexpected_game_output:{existing.name}")
        if existing.is_dir() and not (existing / "result.json").exists():
            raise ValueError(f"incomplete_attempt_preserved:{existing.name}")
    stop_requested = threading.Event()

    def execute_pair(pair: list[dict[str, Any]]) -> None:
        if stop_requested.is_set():
            return
        try:
            if len(pair) != 2 or {row["candidate_seat"] for row in pair} != {0, 1}:
                raise ValueError("seat_pair_assignment_invalid")
            for assignment in sorted(pair, key=lambda row: row["candidate_seat"]):
                if stop_requested.is_set():
                    return
                execute_one(assignment)
        except BaseException:
            stop_requested.set()
            raise

    def execute_one(assignment: dict[str, Any]) -> None:
        game_dir = result_root / assignment["game_id"]
        if (game_dir / "result.json").exists():
            verify_one_result(root, assignment)
            return
        if game_dir.exists():
            try:
                validate_no_uncommitted_attempt(game_dir)
            except ValueError as error:
                raise ValueError(
                    f"incomplete_attempt_preserved:{assignment['game_id']}"
                ) from error
        attempts = game_dir / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        prior = sorted(path for path in attempts.iterdir() if path.is_dir())
        if prior:
            raise ValueError(
                f"unreconciled_prior_attempt_preserved:{assignment['game_id']}"
            )
        attempt = attempts / f"attempt-{len(prior) + 1:04d}"
        attempt.mkdir()
        opening_path = attempt / "opening.jsonl"
        atomic_create(
            opening_path,
            canonical(
                {"prefix_moves": openings[assignment["opening_id"]]["prefix_moves"]}
            )
            + b"\n",
        )
        command = _arena_command(root, assignment, opening_path, attempt)
        atomic_create(attempt / "command.json", canonical(command) + b"\n")
        environment = os.environ.copy()
        environment.update(ENVIRONMENT)
        subprocess.run(command, cwd=root, env=environment, check=True)
        game_lines = (attempt / "game.jsonl").read_text().splitlines()
        if len(game_lines) != 1:
            raise ValueError("arena_game_record_count_mismatch")
        game_row = json.loads(game_lines[0])
        verified = verify_game(assignment, openings[assignment["opening_id"]], game_row)
        result = {
            "schema": "seed615-game-result-v1",
            **assignment,
            **verified,
            "attempt_directory": str(Path("attempts") / attempt.name),
            "game_sha256": sha(attempt / "game.jsonl"),
            "report_sha256": sha(attempt / "report.json"),
            "command_sha256": sha(attempt / "command.json"),
        }
        atomic_create(game_dir / "result.json", canonical(result) + b"\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
        futures = [pool.submit(execute_pair, pair) for pair in pairs.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def analyze_completed(root: Path = ROOT) -> dict[str, Any]:
    from ml.alphazero_lite.verify_seed615_evaluation import verify_results

    root = root.resolve()
    verification = verify_results(root)
    if verification.get("completed_games") != 4096:
        raise ValueError("cannot_analyze_incomplete_evaluation")
    rows = [
        json.loads(path.read_bytes())
        for path in sorted((root / RESULTS).glob("*/result.json"))
    ]
    summary = {
        "schema": "seed615-cluster-bootstrap-analysis-v1",
        "games": 4096,
        "opening_clusters": 1024,
        "bootstrap": {
            "resamples": 20000,
            "seed": 615,
            "quantile_method": "linear",
            "paired_opening_indices": True,
            "seat_outcomes_preserved": True,
        },
        "results": analyze_scores(rows),
    }
    atomic_create(
        root / ANALYSIS,
        json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--prepare-runtime-artifacts", action="store_true")
    group.add_argument("--publish-freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        validate_preflight_artifact_state(
            (ROOT / AUTHORIZATION).exists(),
            (ROOT / RESULTS).exists(),
            (ROOT / ANALYSIS).exists(),
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "inputs": authenticate_inputs(),
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
    else:
        print(json.dumps(analyze_completed(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
