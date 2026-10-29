"""Prepare and durably execute the preregistered seed542 paired arena."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite import seed511_fixed_composition_screen as seed511_runtime
from ml.alphazero_lite import seed542_expanded_policy_strength_screen as protocol
from ml.alphazero_lite import (
    verify_seed542_expanded_policy_strength_screen as independent,
)

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
TRAINING_TABLEBASE = Path(
    "/home/alex/Mancala/ai/.git/lfs/objects/f1/26/"
    "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
)
TABLEBASE = ROOT / ".tmp/seed542-screen-runtime/kalah_v1_21.kvtb"
SEED = 545
OPENINGS = 1024
GAMES = 2048
SIMULATIONS = 128
RESAMPLES = 20_000
SEED_CONTRACT = "azlite_eval_seed_v2"
RUNTIME_SOURCES = (
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/build_opening_suite.py",
    "ml/alphazero_lite/classic_mcts.py",
    "ml/alphazero_lite/endgame_tablebase.py",
    "ml/alphazero_lite/eval_cache.py",
    "ml/alphazero_lite/evaluation_seed_contract.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/exact_root_policy_targets.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/opening_cache.py",
    "ml/alphazero_lite/opponent_pool.py",
    "ml/alphazero_lite/policy_prior_localization.py",
    "ml/alphazero_lite/report_validation.py",
    "ml/alphazero_lite/root_prior_transforms.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/search_ablation.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/pipeline.py",
    "ml/alphazero_lite/seed482_fresh_confirmation.py",
    "ml/alphazero_lite/seed542_screen_arena_entrypoint.py",
    "ml/alphazero_lite/seed511_fixed_composition_screen.py",
    "ml/alphazero_lite/seed542_expanded_policy_strength_screen.py",
    "ml/alphazero_lite/shadow_root_q.py",
    "ml/alphazero_lite/test_seed542_expanded_policy_strength_screen.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/value_transforms.py",
    "ml/alphazero_lite/verify_seed542_expanded_policy_strength_screen.py",
    "ml/alphazero_lite/run_seed542_expanded_policy_strength_screen.py",
)
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
ASSIGNMENTS = {
    "challenger_policy": "D-final",
    "challenger_policy_trunk": "D-final own trunk",
    "challenger_value": "initializer",
    "challenger_value_trunk": "initializer own trunk",
    "opponent_policy": "E4",
    "opponent_policy_trunk": "E4 own trunk",
    "opponent_value": "initializer",
    "opponent_value_trunk": "initializer own trunk",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def runtime_settings() -> dict[str, Any]:
    packages = sorted(
        (
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        ),
        key=str.casefold,
    )
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numerical_execution": {
            "device": "cpu",
            "dtype": "float32",
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "entrypoint": "ml.alphazero_lite.seed542_screen_arena_entrypoint",
        },
        "packages": packages,
        "arena_environment": {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "requested_resource_limits": {
            "cpu_quota": "25%",
            "cpu_weight": "1",
            "memory_max": "4294967296",
            "allowed_cpus": "30",
            "nice": "19",
            "address_space": "17179869184",
            "workers": 1,
        },
    }


def _file_manifest(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    output = root / protocol.OUT
    sources = {relative: sha(root / relative) for relative in RUNTIME_SOURCES}
    training_registration_path = (
        root / "docs/data/seed542-fixed-update-cd-v2/registration.json"
    )
    training_registration = json.loads(training_registration_path.read_text())
    for relative, expected in training_registration["execution_source_sha256"].items():
        actual = sha(root / relative)
        if actual != expected:
            raise ValueError(f"prepare_seed542_training_source_mismatch:{relative}")
        sources[relative] = actual
    authenticated = {
        "docs/data/seed511-fixed-composition-screen-v4/registration.json": sha(
            root / protocol.SEED511 / "registration.json"
        ),
        "docs/data/seed511-fixed-composition-screen-v4/execution-freeze.json": sha(
            root / protocol.SEED511 / "execution-freeze.json"
        ),
        "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl": sha(
            root / protocol.SEED511 / "openings.jsonl"
        ),
        "docs/data/seed504/manifest.json": sha(
            root / "docs/data/seed504/manifest.json"
        ),
        "docs/data/seed542-selfplay-4096/source-manifest.json": sha(
            root / "docs/data/seed542-selfplay-4096/source-manifest.json"
        ),
        "docs/data/seed542-selfplay-4096/initializer.npz": sha(
            root / "docs/data/seed542-selfplay-4096/initializer.npz"
        ),
        "docs/data/seed542-fixed-update-cd-v2/registration.json": sha(
            training_registration_path
        ),
        "docs/data/seed542-fixed-update-cd-v2/run-status.json": sha(
            root / "docs/data/seed542-fixed-update-cd-v2/run-status.json"
        ),
        "docs/data/seed542-fixed-update-cd-v2/development-metrics.json": sha(
            root / "docs/data/seed542-fixed-update-cd-v2/development-metrics.json"
        ),
        "docs/data/seed542-fixed-update-cd-v2/D-update-108.npz": sha(
            root / "docs/data/seed542-fixed-update-cd-v2/D-update-108.npz"
        ),
        "deploy/serving-release.json": sha(root / "deploy/serving-release.json"),
        "deploy/release-manifest.json": sha(root / "deploy/release-manifest.json"),
        "model-artifact/runtime/kalah_v1_tablebase": sha(
            root / "model-artifact/runtime/kalah_v1_tablebase"
        ),
        "model-artifact/runtime/kalah_v1_21.kvtb": sha(
            root / "model-artifact/runtime/kalah_v1_21.kvtb"
        ),
        str(TABLEBASE.relative_to(root)): sha(TABLEBASE),
    }
    for relative, expected in training_registration["input_sha256"].items():
        actual = sha(root / relative)
        if actual != expected:
            raise ValueError(f"prepare_seed542_training_input_mismatch:{relative}")
        authenticated[relative] = actual
    for relative, digest in sources.items():
        authenticated[relative] = digest
    for subdir in ("D-final", "E4", "initializer"):
        directory = ROOT / protocol.OUT / "runtime-artifacts" / subdir
        for path in sorted(directory.iterdir()):
            if path.is_file():
                authenticated[str(path.relative_to(root))] = sha(path)
    prelaunch_failure = output / "PRELAUNCH-FAILURE-0000.json"
    if prelaunch_failure.is_file():
        authenticated[str(prelaunch_failure.relative_to(root))] = sha(prelaunch_failure)
        archive = output / "batches/prelaunch-failures/opening-0000"
        for name in ("command.json", "attempt.json", "command-log.jsonl"):
            archived = archive / name
            authenticated[str(archived.relative_to(root))] = sha(archived)
    return sources, authenticated


def parser_checked_command(root: Path, index: int) -> list[str]:
    output = root / protocol.OUT
    batch = output / "batches"
    opening_path = batch / f"opening-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    game_path = batch / f"games-{index:04d}.jsonl"
    artifact = output / "runtime-artifacts"
    runtime_auth = json.loads((output / "runtime-authentication.json").read_text())
    probe = root / runtime_auth["runtime_assets"]["native_probe_path"]
    tablebase = (root / runtime_auth["runtime_assets"]["tablebase_path"]).resolve()
    encoded_options = json.dumps(SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    command = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.seed542_screen_arena_entrypoint",
        "--challenger",
        str(artifact / "D-final"),
        "--current",
        str(artifact / "initializer"),
        "--challenger-policy-artifact",
        str(artifact / "D-final"),
        "--challenger-value-artifact",
        str(artifact / "initializer"),
        "--current-policy-artifact",
        str(artifact / "E4"),
        "--current-value-artifact",
        str(artifact / "initializer"),
        "--games",
        "2",
        "--games-per-opening",
        "2",
        "--workers",
        "1",
        "--challenger-simulations",
        "128",
        "--current-simulations",
        "128",
        "--c-puct",
        "1.25",
        "--seed-contract",
        SEED_CONTRACT,
        "--challenger-search-options-json",
        encoded_options,
        "--current-search-options-json",
        encoded_options,
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--exact-root-native-probe",
        str(probe),
        "--exact-root-tablebase",
        str(tablebase),
        "--seed",
        str(SEED + index),
        "--opening-prefixes-jsonl",
        str(opening_path),
        "--suite-sha256",
        sha(opening_path),
        "--game-jsonl",
        str(game_path),
        "--out",
        str(report_path),
    ]
    old_argv = sys.argv
    try:
        sys.argv = ["arena.py", *command[3:]]
        parsed = arena.parse_args()
    finally:
        sys.argv = old_argv
    if (
        parsed.games != 2
        or parsed.games_per_opening != 2
        or parsed.workers != 1
        or parsed.challenger_simulations != SIMULATIONS
        or parsed.current_simulations != SIMULATIONS
        or parsed.c_puct != 1.25
        or parsed.seed_contract != SEED_CONTRACT
        or parsed.random_opening_plies != 0
        or parsed.max_moves != 200
        or parsed.seed != SEED + index
        or Path(parsed.opening_prefixes_jsonl) != opening_path
        or json.loads(parsed.challenger_search_options_json) != SEARCH_OPTIONS
        or json.loads(parsed.current_search_options_json) != SEARCH_OPTIONS
    ):
        raise ValueError(f"arena_parser_contract_mismatch:{index}")
    return command


def _verify_training(root: Path, output: Path) -> dict[str, Any]:
    venv = Path(
        "/home/alex/Mancala/ai/.tmp/seed542-fixed-update-cd/.tmp/seed542-cd-runtime/bin/python"
    )
    if not venv.is_file():
        raise ValueError("registered_seed542_verifier_runtime_missing")
    tablebase = TRAINING_TABLEBASE
    evidence = {}
    for module in (
        "verify_seed542_fixed_update_cd_v2",
        "verify_seed542_fixed_update_cd_v2_supplement",
    ):
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            "CPUQuota=25%",
            "-p",
            "CPUWeight=1",
            "-p",
            "MemoryMax=4294967296",
            "-p",
            "AllowedCPUs=30",
            "--",
            "nice",
            "-n",
            "19",
            "prlimit",
            "--as=17179869184",
            "--",
            "taskset",
            "-c",
            "30",
            "env",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "BLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            str(venv),
            "-m",
            f"ml.alphazero_lite.{module}",
            "--root",
            str(root),
            "--out",
            str(root / "docs/data/seed542-fixed-update-cd-v2"),
            "--tablebase",
            str(tablebase),
        ]
        completed = subprocess.run(
            command, cwd=root, check=True, capture_output=True, text=True
        )
        payload = completed.stdout[completed.stdout.find("{") :]
        verified = json.loads(payload)
        expected = (
            "verified_complete"
            if module.endswith("_v2") and not module.endswith("supplement")
            else "supplement_verified"
        )
        if verified.get("status") != expected or not verified.get(
            "read_only_bytes_and_mtimes_unchanged", verified.get("read_only", False)
        ):
            raise ValueError(f"seed542_training_verification_failed:{module}")
        evidence[module] = {
            "command": command,
            "status": verified["status"],
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    record = {
        "status": "both_seed542_v2_verifiers_passed",
        "runtime_settings": {
            "systemd_scope": True,
            "cpu_quota": "25%",
            "cpu_weight": "1",
            "memory_max": "4294967296",
            "allowed_cpus": "30",
            "nice": 19,
            "address_space_limit": "17179869184",
            "threads_per_numerical_library": 1,
        },
        "verifiers": evidence,
    }
    _save_json(output / "training-verification.json", record)
    return record


def _runtime_policy_preflight(root: Path, output: Path) -> dict[str, Any]:
    historical = json.loads((root / protocol.SEED511 / "registration.json").read_text())
    historical["runtime_assets"]["exact_root_tablebase"][
        "runtime_materialized_path"
    ] = str(TABLEBASE)
    pointer = seed511_runtime.materialize_bound_tablebase(root, historical)
    try:
        report = arena.resolve_strength_comparison_runtime_contract(
            current_artifact=output / "runtime-artifacts/initializer",
            challenger_artifact=output / "runtime-artifacts/D-final",
            exact_root_threshold=16,
            native_probe=root / "model-artifact/runtime/kalah_v1_tablebase",
            tablebase=root / "model-artifact/runtime/kalah_v1_21.kvtb",
        )
    finally:
        seed511_runtime.restore_bound_tablebase(root, historical, pointer)
    expected = {
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "exact_root_native_probe_sha256": "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        "exact_root_tablebase_sha256": "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
        "runtime_search_policy_sha256": protocol.sha256_file(
            output / "runtime-artifacts/D-final/search_policy.json"
        ),
    }
    if report is None or any(
        report.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("runtime_policy_preflight_contract_mismatch")
    record = {
        "status": "runtime_search_policy_contract_passed_without_inference_or_games",
        "native_tablebase_temporarily_materialized_at_sidecar_path": True,
        "restored_pointer_sha256": sha(
            root / "model-artifact/runtime/kalah_v1_21.kvtb"
        ),
        "resolved_contract": report,
    }
    _save_json(output / "runtime-policy-preflight.json", record)
    return record


def _prepare(
    root: Path, output: Path, *, run_training_verifiers: bool
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (
        sha(root / "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl")
        != protocol.OPENING_SHA256
    ):
        raise ValueError("prepare_imported_suite_bytes_changed")
    openings, import_report = protocol.import_seed511_suite(root)
    overlap = protocol.overlap_report(root, openings)
    if (output / "openings.jsonl").read_bytes() != (
        root / "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl"
    ).read_bytes():
        raise ValueError("prepare_publication_suite_copy_mismatch")
    (output / "suite-import-report.json").write_text(
        json.dumps(import_report, indent=2, sort_keys=True) + "\n"
    )
    (output / "overlap-summary.json").write_text(
        json.dumps(
            {key: value for key, value in overlap.items() if key != "per_opening"},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "overlap-per-opening.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in overlap["per_opening"]
        )
    )
    runtime_auth = protocol.authenticate_runtime(root, output, TABLEBASE)
    _runtime_policy_preflight(root, output)
    for index, raw_line in enumerate(
        (root / protocol.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    ):
        path = output / "batches" / f"opening-{index:04d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
    training = (
        _verify_training(root, output)
        if run_training_verifiers
        else json.loads((output / "training-verification.json").read_text())
    )
    if training.get("status") != "both_seed542_v2_verifiers_passed":
        raise ValueError("prepare_training_verification_not_complete")

    launch_command = [
        "systemd-run",
        "--user",
        "--unit=seed542-expanded-policy-strength-screen",
        "--collect",
        f"--property=WorkingDirectory={root}",
        "--property=CPUQuota=25%",
        "--property=CPUWeight=1",
        "--property=MemoryMax=4294967296",
        "--property=AllowedCPUs=30",
        "--property=Nice=19",
        "--property=LimitAS=17179869184",
        f"--setenv=PYTHONPATH={root}",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=BLAS_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1",
        "--",
        sys.executable,
        "-m",
        "ml.alphazero_lite.run_seed542_expanded_policy_strength_screen",
        "--root",
        str(root),
        "--out",
        str(output),
        "--execute",
    ]

    sources, authenticated = _file_manifest(root)
    for rel, expected in sources.items():
        if sha(root / rel) != expected:
            raise ValueError(f"prepare_source_hash_race:{rel}")
    registration = {
        "schema": "seed542_expanded_policy_strength_screen_v1",
        "status": "prospective_frozen_before_arena_games",
        "execution_root": str(root),
        "classification": "exploratory reused seed511 opening suite; no fresh holdout",
        "historical_seed511_full_publication_verification": "stopped at frozen native-module hash mismatch; suite imported and independently replayed against authenticated current rules/encoding dependencies; no seed511 games/outcomes imported",
        "seed461_historical_overlap_limitation_unresolved": True,
        "prior_non_outcome_bearing_attempts": [
            {
                "record": "PRELAUNCH-FAILURE-0000.json",
                "sha256": sha(output / "PRELAUNCH-FAILURE-0000.json"),
                "classification": "pre-game runtime asset contract failure; no game/report evidence",
            }
        ]
        if (output / "PRELAUNCH-FAILURE-0000.json").is_file()
        else [],
        "suite": {
            "path": "openings.jsonl",
            "sha256": protocol.OPENING_SHA256,
            "openings": OPENINGS,
            "games": GAMES,
            "source_registration_sha256": import_report["registration_sha256"],
            "source_execution_freeze_sha256": sha(
                root / protocol.SEED511 / "execution-freeze.json"
            ),
            "import_report_sha256": sha(output / "suite-import-report.json"),
            "overlap_summary_sha256": sha(output / "overlap-summary.json"),
            "overlap_per_opening_sha256": sha(output / "overlap-per-opening.jsonl"),
        },
        "component_assignments": ASSIGNMENTS,
        "component_hashes": runtime_auth,
        "runtime_assets": {
            "native_probe_sha256": runtime_auth["runtime_assets"][
                "native_probe_sha256"
            ],
            "tablebase_sha256": runtime_auth["runtime_assets"]["tablebase_sha256"],
            "search_policy_sha256": protocol.sha256_file(
                output / "runtime-artifacts/D-final/search_policy.json"
            ),
        },
        "training_verification_sha256": sha(output / "training-verification.json"),
        "runtime_policy_preflight_sha256": sha(
            output / "runtime-policy-preflight.json"
        ),
        "analysis": {
            "score": "challenger win=1, draw=0.5, opponent win=0; mean over all 2048 games",
            "cluster": "mean of both seat-swapped game points within each of 1024 openings",
            "bootstrap_resamples": RESAMPLES,
            "bootstrap_seed": SEED,
            "replacement": True,
            "preserve_multiplicity": True,
            "quantile_method": "linear",
            "one_sided_lower_quantile": 0.05,
            "descriptive_quantiles": [0.025, 0.975],
            "advance_rule": "all 2048 games validate and one-sided lower bound is strictly greater than 0.50",
            "advance_scope": "propose a separate 1200-simulation evaluation only; no promotion/deployment",
        },
        "arena_protocol": {
            "games_per_opening": 2,
            "seat_assignment": [0, 1],
            "simulations_per_side": SIMULATIONS,
            "c_puct": 1.25,
            "workers": 1,
            "search_options": SEARCH_OPTIONS,
            "exact_root_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solve": "disabled",
            "random_opening_plies": 0,
            "max_moves": 200,
            "seed_contract": SEED_CONTRACT,
            "opening_seed": "545 + opening_index",
        },
        "runtime": runtime_settings(),
        "durable_execution_command": launch_command,
        "execution_source_sha256": sources,
        "authenticated_file_sha256": authenticated,
    }
    _save_json(output / "registration.json", registration)
    freeze = {
        "status": "frozen_before_real_arena_games",
        "registration_sha256": sha(output / "registration.json"),
        "suite_sha256": protocol.OPENING_SHA256,
        "overlap_summary_sha256": sha(output / "overlap-summary.json"),
        "overlap_per_opening_sha256": sha(output / "overlap-per-opening.jsonl"),
        "runtime_authentication_sha256": sha(output / "runtime-authentication.json"),
        "runtime_policy_preflight_sha256": sha(
            output / "runtime-policy-preflight.json"
        ),
        "training_verification_sha256": sha(output / "training-verification.json"),
        "execution_source_sha256": sources,
        "authenticated_file_sha256": authenticated,
        "resource_limits": registration["runtime"]["requested_resource_limits"],
        "arena_protocol": registration["arena_protocol"],
        "analysis": registration["analysis"],
        "durable_execution_command": launch_command,
    }
    _save_json(output / "execution-freeze.json", freeze)
    commands = []
    for index in range(OPENINGS):
        argv = parser_checked_command(root, index)
        commands.append(
            {
                "opening_index": index,
                "argv": argv,
                "registration_sha256": freeze["registration_sha256"],
                "suite_sha256": freeze["suite_sha256"],
                "component_assignments": ASSIGNMENTS,
                "component_hashes": runtime_auth,
                "runtime": registration["runtime"],
                "runtime_source_sha256": sources,
            }
        )
    (output / "command-preflight.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in commands
        )
    )
    freeze["command_preflight_sha256"] = sha(output / "command-preflight.jsonl")
    _save_json(output / "execution-freeze.json", freeze)
    return {
        "status": "prepared_frozen",
        "openings": OPENINGS,
        "commands_parser_validated": len(commands),
        "suite_sha256": freeze["suite_sha256"],
        "overlap_aggregates": overlap["aggregates"],
        "registration_sha256": freeze["registration_sha256"],
    }


def _assert_frozen(root: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    if registration.get("execution_root") != str(root.resolve()):
        raise ValueError("execution_root_changed")
    if (
        sha(output / "registration.json") != freeze["registration_sha256"]
        or sha(output / "openings.jsonl") != freeze["suite_sha256"]
        or sha(output / "runtime-authentication.json")
        != freeze["runtime_authentication_sha256"]
        or sha(output / "runtime-policy-preflight.json")
        != freeze["runtime_policy_preflight_sha256"]
        or sha(output / "training-verification.json")
        != freeze["training_verification_sha256"]
        or sha(output / "command-preflight.jsonl") != freeze["command_preflight_sha256"]
        or sha(output / "suite-import-report.json")
        != registration["suite"]["import_report_sha256"]
        or sha(output / "overlap-summary.json")
        != registration["suite"]["overlap_summary_sha256"]
        or sha(output / "overlap-per-opening.jsonl")
        != registration["suite"]["overlap_per_opening_sha256"]
    ):
        raise ValueError("execution_frozen_binding_mismatch")
    for relative, expected in freeze["execution_source_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"execution_source_changed:{relative}")
    for relative, expected in freeze["authenticated_file_sha256"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if sha(path) != expected:
            raise ValueError(f"execution_authenticated_input_changed:{relative}")
    return registration, freeze


def _restore_interrupted_tablebase_materialization(root: Path) -> None:
    """Restore the registered LFS pointer if interruption left its exact object linked."""
    historical = json.loads((root / protocol.SEED511 / "registration.json").read_text())
    asset = historical["runtime_assets"]["exact_root_tablebase"]
    asset["runtime_materialized_path"] = str(TABLEBASE)
    target = root / asset["artifact"]
    actual = sha(target)
    if actual == asset["source_pointer_sha256"]:
        return
    if actual != asset["sha256"]:
        raise ValueError(
            "resume_tablebase_neither_registered_pointer_nor_materialized_object"
        )
    pointer = subprocess.run(
        ["git", "show", f"HEAD:{asset['source_git_path']}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(pointer).hexdigest() != asset["source_pointer_sha256"]:
        raise ValueError("resume_tablebase_original_pointer_identity_mismatch")
    temporary = target.with_name(target.name + ".seed542-restore")
    temporary.write_bytes(pointer)
    os.replace(temporary, target)
    if sha(target) != asset["source_pointer_sha256"]:
        raise ValueError("resume_tablebase_pointer_restore_failed")


def recovery_action(output: Path, index: int, committed: set[int]) -> str:
    """Classify one pair without retrying any outcome-bearing attempt."""
    if index in committed:
        return "skip"
    batch = output / "batches"
    paths = {
        "opening": batch / f"opening-{index:04d}.jsonl",
        "games": batch / f"games-{index:04d}.jsonl",
        "report": batch / f"report-{index:04d}.json",
        "command": batch / f"command-{index:04d}.json",
        "attempt": batch / f"attempt-{index:04d}.json",
    }
    raw = paths["games"].exists() or paths["report"].exists()
    if raw:
        if not all(
            paths[name].is_file() for name in ("opening", "games", "report", "command")
        ):
            raise ValueError(f"resume_partial_raw_pair_preserved:{index}")
        return "recover"
    if paths["command"].exists() or paths["attempt"].exists():
        raise ValueError(f"resume_prior_attempt_without_complete_raw_pair:{index}")
    failure_path = output / f"PRELAUNCH-FAILURE-{index:04d}.json"
    if failure_path.is_file():
        failure = json.loads(failure_path.read_text())
        archive = batch / "prelaunch-failures" / f"opening-{index:04d}"
        archived_command = archive / "command.json"
        archived_attempt = archive / "attempt.json"
        archived_log = archive / "command-log.jsonl"
        if (
            failure.get("status") != "preserved_non_outcome_bearing_attempt"
            or failure.get("opening_index") != index
            or failure.get("games_path_exists") is not False
            or failure.get("report_path_exists") is not False
            or not archived_command.is_file()
            or not archived_attempt.is_file()
            or not archived_log.is_file()
            or sha(archived_command) != failure.get("command_sha256")
            or sha(archived_attempt) != failure.get("attempt_sha256")
            or sha(archived_log) != failure.get("command_log_sha256")
        ):
            raise ValueError(f"resume_prelaunch_failure_not_authenticated:{index}")
        return "execute"
    return "execute"


def commit_pair(root: Path, output: Path, index: int) -> list[dict[str, Any]]:
    """Validate raw evidence independently, then atomically commit both seats."""
    _assert_frozen(root, output)
    pair = independent.validate_pair(root, output, index)
    if len(pair) != 2 or {row["challenger_player"] for row in pair} != {0, 1}:
        raise ValueError(f"commit_requires_complete_seat_pair:{index}")
    ledger_path = output / "outcomes.jsonl"
    ledger = rows(ledger_path) if ledger_path.exists() else []
    if any(int(record["opening_index"]) == index for record in ledger):
        raise ValueError(f"commit_pair_already_present:{index}")
    ledger.extend(pair)
    temporary = ledger_path.with_suffix(".jsonl.pending")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in ledger
        )
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, ledger_path)
    return pair


def run(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    _restore_interrupted_tablebase_materialization(root)
    registration, freeze = _assert_frozen(root, output)
    ledger_path = output / "outcomes.jsonl"
    command_log = output / "commands.jsonl"
    ledger = rows(ledger_path) if ledger_path.exists() else []
    committed: dict[int, list[dict[str, Any]]] = {}
    for record in ledger:
        committed.setdefault(int(record["opening_index"]), []).append(record)
    if len(ledger) % 2 or any(len(pair) != 2 for pair in committed.values()):
        raise ValueError("resume_ledger_partial_pair")
    for index in sorted(committed):
        actual = independent.validate_pair(root, output, index)
        for expected, saved in zip(
            actual,
            sorted(committed[index], key=lambda row: row["game_position"]),
            strict=True,
        ):
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"resume_committed_pair_changed:{index}")

    for index in range(OPENINGS):
        action = recovery_action(output, index, set(committed))
        if action == "skip":
            continue
        batch = output / "batches"
        opening_path = batch / f"opening-{index:04d}.jsonl"
        command_path = batch / f"command-{index:04d}.json"
        attempt_path = batch / f"attempt-{index:04d}.json"
        if action == "recover":
            pair = independent.validate_pair(root, output, index)
        else:
            argv = parser_checked_command(root, index)
            receipt = {
                "opening_index": index,
                "argv": argv,
                "registration_sha256": freeze["registration_sha256"],
                "suite_sha256": freeze["suite_sha256"],
                "component_assignments": registration["component_assignments"],
                "component_hashes": registration["component_hashes"],
                "runtime": registration["runtime"],
                "runtime_source_sha256": registration["execution_source_sha256"],
                "opening_sha256": sha(opening_path),
            }
            command_path.write_bytes(canonical(receipt) + b"\n")
            _append_jsonl(command_log, receipt)
            _save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "command_sha256": sha(command_path),
                },
            )
            historical_registration = json.loads(
                (root / protocol.SEED511 / "registration.json").read_text()
            )
            historical_registration["runtime_assets"]["exact_root_tablebase"][
                "runtime_materialized_path"
            ] = str(TABLEBASE)
            pointer = seed511_runtime.materialize_bound_tablebase(
                root, historical_registration
            )
            try:
                if (
                    sha(root / "model-artifact/runtime/kalah_v1_21.kvtb")
                    != registration["runtime_assets"]["tablebase_sha256"]
                ):
                    raise ValueError("execution_materialized_tablebase_hash_mismatch")
                result = subprocess.run(argv, cwd=root, check=False)
            finally:
                seed511_runtime.restore_bound_tablebase(
                    root, historical_registration, pointer
                )
            _save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "returncode": result.returncode,
                    "command_sha256": sha(command_path),
                },
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"arena_command_failed_preserved:{index}:{result.returncode}"
                )
            pair = independent.validate_pair(root, output, index)
        # commit_pair independently revalidates before atomic two-seat append.
        pair = commit_pair(root, output, index)
        ledger = rows(ledger_path)
        committed[index] = pair
        print(
            json.dumps(
                {"opening_index": index, "committed_games": len(ledger)}, sort_keys=True
            ),
            flush=True,
        )

    if len(ledger) != GAMES:
        return {
            "status": "incomplete",
            "validated_games": len(ledger),
            "validated_pairs": len(committed),
        }
    analysis, per_opening = runner_analysis(ledger)
    _save_json(output / "results.json", analysis)
    (output / "per-opening-results.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in per_opening
        )
    )
    return {
        "status": "all_pairs_committed",
        "validated_games": len(ledger),
        "validated_pairs": len(committed),
        "results": analysis,
    }


def runner_analysis(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(ledger) != GAMES:
        raise ValueError("analysis_requires_2048_games")
    by_opening: dict[int, list[float]] = {index: [] for index in range(OPENINGS)}
    points = {"challenger": 1.0, "draw": 0.5, "opponent": 0.0}
    wins = draws = losses = 0
    for record in ledger:
        opening = int(record["opening_index"])
        point = points[record["winner"]]
        by_opening[opening].append(point)
        wins += record["winner"] == "challenger"
        draws += record["winner"] == "draw"
        losses += record["winner"] == "opponent"
    if any(len(by_opening[index]) != 2 for index in range(OPENINGS)):
        raise ValueError("analysis_pairing_mismatch")
    clusters = np.asarray([np.mean(by_opening[index]) for index in range(OPENINGS)])
    rng = np.random.default_rng(SEED)
    sampled = clusters[rng.integers(0, OPENINGS, size=(RESAMPLES, OPENINGS))]
    bootstrap = sampled.mean(axis=1)
    lower = float(np.quantile(bootstrap, 0.05, method="linear"))
    result = {
        "status": "complete",
        "verification_status": "independent_read_only_verification_pending",
        "exploratory_reused_suite": True,
        "games": GAMES,
        "openings": OPENINGS,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "score": float((wins + 0.5 * draws) / GAMES),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "bootstrap_quantile_method": "linear",
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(bootstrap, 0.025, method="linear")),
            float(np.quantile(bootstrap, 0.975, method="linear")),
        ],
        "pass": bool(lower > 0.50),
        "decision": "advance_to_propose_separate_1200_simulation_evaluation_only"
        if lower > 0.50
        else "do_not_advance",
        "deployment_or_promotion_authorized": False,
        "fresh_confirmation": False,
        "seed461_historical_overlap_limitation_unresolved": True,
        "effect_score_vs_0_50": float((wins + 0.5 * draws) / GAMES - 0.50),
        "lower_bound_effect_vs_0_50": lower - 0.50,
    }
    opening_rows = [
        {
            "opening_index": index,
            "seat0_challenger_score": by_opening[index][0],
            "seat1_challenger_score": by_opening[index][1],
            "opening_cluster_score": float(clusters[index]),
        }
        for index in range(OPENINGS)
    ]
    return result, opening_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-training-verifiers", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        result = _prepare(
            args.root, args.out, run_training_verifiers=args.run_training_verifiers
        )
    elif args.execute:
        result = run(args.root, args.out)
    else:
        raise SystemExit("choose --prepare or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
