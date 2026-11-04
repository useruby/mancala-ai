"""Freeze and durably execute the seed552 C108-versus-serving-P screen."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed551_controlled_policy_strength as base_runner
from ml.alphazero_lite import seed552_c108_serving_p as protocol
from ml.alphazero_lite.verify_seed542_expanded_policy_strength_screen import _serving_identity

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
PROBE = ROOT / protocol.NATIVE_PROBE
TABLEBASE = ROOT / protocol.TABLEBASE_POINTER
MATERIALIZED_TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
ARTIFACTS = OUT / "runtime-artifacts"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def archive_failed_preparation(root: Path, output: Path, failure_check: str) -> Path:
    """Preserve partial preflight artifacts without allowing silent replacement."""
    root, output = root.resolve(), output.resolve()
    if any((output / "batches").glob("games-*.jsonl")):
        raise ValueError("seed552_failed_preparation_contains_games")
    files = [
        {"path": str(path.relative_to(output)), "sha256": protocol.sha256(path),
         "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(output.rglob("*")) if path.is_file()
    ]
    failure = {
        "schema": "seed552_preparation_failure_v1",
        "status": "pre_game_preparation_failed_preserved",
        "failure_check": failure_check,
        "arena_games_started": False,
        "files_before_archive": files,
    }
    write_json(output / "PREPARATION-FAILURE.json", failure)
    files.append({"path": "PREPARATION-FAILURE.json", "sha256": protocol.sha256(output / "PREPARATION-FAILURE.json"),
                  "size_bytes": (output / "PREPARATION-FAILURE.json").stat().st_size,
                  "mtime_ns": (output / "PREPARATION-FAILURE.json").stat().st_mtime_ns})
    write_json(output / "preparation-attempt-manifest.json", {"schema": "seed552_preparation_attempt_manifest_v1", "files": files})
    archives = sorted(root.glob("docs/data/seed552-c108-serving-p-preparation-attempt-*"))
    attempt = len(archives) + 1
    destination = root / f"docs/data/seed552-c108-serving-p-preparation-attempt-{attempt:04d}"
    output.rename(destination)
    return destination


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(protocol.stable_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def registered_runtime() -> dict[str, Any]:
    """Return the frozen CPU/thread/resource environment used by arena workers."""
    runtime = base_runner.runtime_settings()
    runtime["torch_threads"] = 1
    runtime["torch_interop_threads"] = 1
    env = {name: "1" for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "BLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}
    runtime["environment"] = env
    runtime["resource_limits"] = {
        "cpu_quota": "25%", "cpu_weight": "1", "memory_max": "4294967296",
        "allowed_cpus": "30", "nice": 19, "address_space_limit": "17179869184",
        "workers": 1,
    }
    runtime["tablebase_path"] = str(MATERIALIZED_TABLEBASE.resolve())
    runtime["tablebase_sha256"] = protocol.TABLEBASE_OBJECT_SHA256
    runtime["tablebase_size_bytes"] = protocol.TABLEBASE_OBJECT_SIZE
    runtime["native_probe_sha256"] = protocol.sha256(PROBE)
    runtime["observed"] = {
        "cpu_affinity": [30], "nice": 19,
        "address_space_rlimit": [17_179_869_184, 17_179_869_184],
        "cgroup_limits": {"cpu.max": "25000 100000", "cpu.weight": "1",
                           "memory.max": "4294967296", "cpuset.cpus.effective": None},
        "environment": env, "torch_threads": 1, "torch_interop_threads": 1,
    }
    return runtime


def launch_command(root: Path, output: Path) -> list[str]:
    return [
        "systemd-run", "--user", "--unit=seed552-c108-serving-p", "--collect",
        f"--property=WorkingDirectory={root}", "--property=CPUQuota=25%",
        "--property=CPUWeight=1", "--property=MemoryMax=4294967296",
        "--property=AllowedCPUs=30", "--property=Nice=19",
        "--property=LimitAS=17179869184", f"--setenv=PYTHONPATH={root}",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=MKL_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1", "--setenv=BLAS_NUM_THREADS=1",
        "--setenv=NUMEXPR_NUM_THREADS=1", "taskset", "-c", "30", sys.executable, "-m",
        "ml.alphazero_lite.run_seed552_c108_serving_p", "--root", str(root),
        "--out", str(output), "--execute",
    ]


def _copy_component(root: Path, name: str, source: Path, expected: str) -> dict[str, Any]:
    destination = ARTIFACTS / name
    if destination.exists():
        raise FileExistsError(f"seed552_refuse_component_overwrite:{name}")
    if protocol.sha256(source) != expected:
        raise ValueError(f"seed552_checkpoint_hash_mismatch:{name}")
    source_dir = source.parent
    shutil.copytree(source_dir, destination)
    if protocol.sha256(destination / "model.npz") != expected:
        raise ValueError(f"seed552_component_export_hash_mismatch:{name}")
    return {
        "source": str(source.relative_to(root)),
        "checkpoint_sha256": expected,
        "files": {p.name: protocol.sha256(p) for p in sorted(destination.iterdir()) if p.is_file()},
        "artifact_dir": str(destination.relative_to(root)),
    }


def _component_source(root: Path, name: str) -> Path:
    return root / {
        "C108": "docs/data/seed551-controlled-policy-strength-screen/runtime-artifacts/C",
        "P-E4": "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/P-E4",
        "initializer": "docs/data/seed504-policy-only-ablation/screen/runtime-artifacts/initializer",
    }[name]


def prepare(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("seed552_output_must_be_empty")
    output.mkdir(parents=True, exist_ok=True)
    historical_attempts = {
        "schema": "seed552_historical_verifier_attempts_v1",
        "attempts": [
            {
                "verifier": "ml.alphazero_lite.verify_seed551_controlled_policy_strength",
                "argv": ["--root", "docs/data/seed551-controlled-policy-strength-screen"],
                "result": "failed_seed551_freeze_binding_mismatch",
                "explanation": "The evidence directory was mistakenly passed as repository root.",
            },
            {
                "verifier": "ml.alphazero_lite.verify_seed551_controlled_policy_strength",
                "argv": ["--root", str(root), "--out", str(root / "docs/data/seed551-controlled-policy-strength-screen"), "--tablebase", str(MATERIALIZED_TABLEBASE), "--allow-relocated-copy"],
                "result": "failed_seed551_authenticated_input_changed:model-artifact/runtime/kalah_v1_21.kvtb",
                "explanation": "The current checkout contains the materialized tablebase object; the frozen #551 root expected the historical Git LFS pointer representation.",
            },
            {
                "verifier": "ml.alphazero_lite.verify_seed551_controlled_policy_strength",
                "argv": ["--root", "/home/alex/Mancala/ai/.tmp/seed551-worktree", "--out", "/home/alex/Mancala/ai/.tmp/seed551-worktree/docs/data/seed551-controlled-policy-strength-screen", "--tablebase", str(MATERIALIZED_TABLEBASE), "--allow-relocated-copy"],
                "result": "verified_complete",
                "explanation": "Verified in retained historical worktree, with separate authenticated materialized tablebase; historical evidence bytes and mtimes unchanged.",
            },
        ],
    }
    write_json(output / "historical-verifier-attempts.json", historical_attempts)
    auth_source = root / ".tmp/seed552-authentication"
    auth_target = output / "historical-authentication"
    auth_target.mkdir()
    for filename in ("seed550-training-verification.json", "seed551-publication-verification.json"):
        source = auth_source / filename
        if not source.is_file():
            raise FileNotFoundError(f"seed552_historical_verification_receipt_missing:{filename}")
        shutil.copy2(source, auth_target / filename)
    training_receipt = json.loads((auth_target / "seed550-training-verification.json").read_text())
    publication_receipt = json.loads((auth_target / "seed551-publication-verification.json").read_text())
    if (
        training_receipt.get("status") != "verified"
        or training_receipt.get("training_updates_per_arm") != 108
        or not training_receipt.get("bytes_and_mtimes_unchanged")
        or publication_receipt.get("status") != "verified_complete"
        or publication_receipt.get("verified_games") != 2048
        or not publication_receipt.get("read_only_bytes_and_mtimes_unchanged")
        or publication_receipt.get("analysis", {}).get("decision") != "do_not_advance_source_evaluator_branch"
    ):
        raise ValueError("seed552_historical_verification_receipt_contract_mismatch")
    raw, openings = protocol.load_openings(root)
    if protocol.sha256(root / protocol.C_CHECKPOINT) != protocol.C_SHA256:
        raise ValueError("seed552_C108_checkpoint_identity_mismatch")
    if protocol.sha256(root / protocol.P_CHECKPOINT) != protocol.P_SHA256:
        raise ValueError("seed552_E4_checkpoint_identity_mismatch")
    if protocol.sha256(root / protocol.INITIALIZER) != protocol.INITIALIZER_SHA256:
        raise ValueError("seed552_initializer_identity_mismatch")
    serving = json.loads((root / "deploy/serving-release.json").read_text())
    bundle = serving["bundles"]["policy-value-p"]
    if bundle["expected_artifact_sha256"] != [
        "07f7fd71ef9dea5480dd010c7808f6fcc1c2821c3f54da2203d17140a8bed79c",
        "ea6151620e9af849f6aa0aa90661be7db279a316a3b748b1f904a7a29ad70aa9",
    ]:
        raise ValueError("seed552_serving_release_component_identity_mismatch")
    raw_output = output / "openings.jsonl"
    raw_output.write_bytes(raw)
    batch = output / "batches"
    batch.mkdir()
    raw_lines = raw.splitlines(keepends=True)
    for index, line in enumerate(raw_lines):
        (batch / f"opening-{index:04d}.jsonl").write_bytes(line)

    global ARTIFACTS
    ARTIFACTS = output / "runtime-artifacts"
    ARTIFACTS.mkdir()
    components = {
        "C108": _copy_component(root, "C108", _component_source(root, "C108") / "model.npz", protocol.C_SHA256),
        "P-E4": _copy_component(root, "P-E4", _component_source(root, "P-E4") / "model.npz", protocol.P_SHA256),
        "initializer": _copy_component(root, "initializer", _component_source(root, "initializer") / "model.npz", protocol.INITIALIZER_SHA256),
    }
    identities = [
        _serving_identity(ARTIFACTS / "initializer"),
        _serving_identity(ARTIFACTS / "P-E4"),
    ]
    if identities != bundle["expected_artifact_sha256"]:
        raise ValueError("seed552_serving_P_artifact_identity_mismatch")
    policy_ids = [
        protocol.sha256(ARTIFACTS / name / "search_policy.json")
        for name in ("initializer", "P-E4")
    ]
    if policy_ids != bundle["expected_policy_sha256"]:
        raise ValueError("seed552_serving_P_sidecar_identity_mismatch")
    manifest = json.loads((root / "deploy/release-manifest.json").read_text())
    for component_name, release_component in (("P-E4", "E4"), ("initializer", "initializer")):
        prefix = f"components/{release_component}/"
        expected_files = {
            entry["destination"][len(prefix):]: entry["sha256"]
            for entry in manifest["files"]
            if entry["destination"].startswith(prefix)
        }
        actual_files = {
            name: protocol.sha256(ARTIFACTS / component_name / name)
            for name in expected_files
        }
        if actual_files != expected_files:
            raise ValueError(f"seed552_release_manifest_component_file_mismatch:{component_name}")
    tablebase_stat = MATERIALIZED_TABLEBASE.stat()
    if protocol.sha256(MATERIALIZED_TABLEBASE) != protocol.TABLEBASE_OBJECT_SHA256 or tablebase_stat.st_size != protocol.TABLEBASE_OBJECT_SIZE:
        raise ValueError("seed552_materialized_tablebase_identity_mismatch")
    if protocol.sha256(TABLEBASE) != protocol.TABLEBASE_OBJECT_SHA256 or TABLEBASE.stat().st_size != protocol.TABLEBASE_OBJECT_SIZE:
        raise ValueError("seed552_repository_tablebase_object_identity_mismatch")
    historical_pointer = subprocess.run(
        ["git", "show", f"{protocol.HISTORICAL_TABLEBASE_POINTER_COMMIT}:{protocol.TABLEBASE_POINTER}"],
        cwd=root, check=True, capture_output=True,
    ).stdout
    if __import__("hashlib").sha256(historical_pointer).hexdigest() != protocol.HISTORICAL_TABLEBASE_POINTER_SHA256:
        raise ValueError("seed552_historical_tablebase_pointer_identity_mismatch")
    pointer_blob = subprocess.run(
        ["git", "rev-parse", f"{protocol.HISTORICAL_TABLEBASE_POINTER_COMMIT}:{protocol.TABLEBASE_POINTER}"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    runtime = {
        "components": components,
        "component_assignments": protocol.ASSIGNMENTS,
        "serving_release_sha256": protocol.sha256(root / "deploy/serving-release.json"),
        "serving_release_component_artifact_identities": identities,
        "serving_release_component_policy_identities": policy_ids,
        "serving_release_manifest_sha256": protocol.sha256(root / "deploy/release-manifest.json"),
        "historical_E4_registration_sha256": protocol.sha256(root / "docs/data/seed504-policy-only-ablation/registration.json"),
        "C_training_registration_sha256": protocol.sha256(root / "docs/data/seed549-controlled-training/registration.json"),
        "tablebase": {
            "repository_path": str(TABLEBASE.relative_to(root)),
            "repository_representation": "materialized Git blob in seed552 checkout",
            "repository_object_sha256": protocol.sha256(TABLEBASE),
            "repository_object_size_bytes": TABLEBASE.stat().st_size,
            "historical_seed551_representation": "Git LFS pointer",
            "historical_pointer_sha256": protocol.HISTORICAL_TABLEBASE_POINTER_SHA256,
            "historical_pointer_git_commit": protocol.HISTORICAL_TABLEBASE_POINTER_COMMIT,
            "historical_pointer_git_blob": pointer_blob,
            "materialized_object_sha256": protocol.sha256(MATERIALIZED_TABLEBASE),
            "materialized_object_size_bytes": tablebase_stat.st_size,
            "resolved_runtime_path": str(MATERIALIZED_TABLEBASE.resolve()),
        },
        "native_probe": {"path": protocol.NATIVE_PROBE, "sha256": protocol.sha256(PROBE)},
    }
    write_json(output / "runtime-authentication.json", runtime)
    overlap = protocol.opening_overlaps(root, openings)
    write_json(output / "overlap-report.json", {k: v for k, v in overlap.items() if k != "per_opening"})
    (output / "overlap-per-opening.jsonl").write_bytes(
        b"".join(protocol.stable_json(row) + b"\n" for row in overlap["per_opening"])
    )
    write_json(output / "overlap-limitations.json", {
        "known": "reused indexed seed511 suite; exploratory candidate examined after #550-551",
        "seed461_continuation_overlap_unresolved": True,
        "historical_games_or_outcomes_imported": False,
        "training_population_overlap_analysis": "canonical-state and production-float32 against C training/development and P policy training; every suite row retained",
    })
    (output / "README.md").write_text(
        "# Seed552: C108 versus frozen serving P\n\n"
        "Exploratory higher-budget evaluation of the #550 initializer-self-play control C, update 108, against serving P (historical seed504 E4). C is a newly registered exploratory candidate; #550's candidate designation and #551's `do_not_advance_source_evaluator_branch` decision are preserved. The C-versus-T result does not imply strength against P.\n\n"
        "The screen uses exactly 1,024 seat-swapped pairs (2,048 new games) from the indexed seed511 v4 suite. Every opening is replayed and retained. Canonical-state and production-float32 overlap is reported against C training/development and P policy-training populations. The suite is reused, the limitation of unresolved seed461 continuation overlap remains, and no historical game/outcome is imported.\n\n"
        "C policy and trunk are C108; P policy and trunk are historical E4; both values and value trunks are the frozen initializer through `ComposedArtifactEvaluator`. Each exported component retains its own trunk. Serving release component and sidecar identities are authenticated. The tablebase's seed551 historical LFS pointer identity and current repository/materialized object bytes are bound separately.\n\n"
        "The preregistered analysis gives C wins 1, terminal draws 0.5 and losses 0, averaging both seats within each opening. It uses 20,000 opening-cluster bootstrap resamples with replacement, seed 545, NumPy linear quantiles, one-sided 95% lower bound at 0.05, and descriptive 95% interval at 0.025/0.975. Any capped/nonterminal game blocks a valid advancement decision. No result authorizes promotion or deployment.\n"
        "Two durable-runner invocations failed the pre-game runtime gate because systemd's `AllowedCPUs=30` did not restrict process scheduler affinity. A separate no-game runtime diagnostic observed affinity 0-31; `taskset -c 30` is now included in the frozen launcher. Both failed attempts and the diagnostic are preserved in `execution-attempts.json`; no arena games started.\n"
    )
    import ml.alphazero_lite.arena as arena_module

    initializer_eval = arena_module.ArtifactEvaluator(ARTIFACTS / "initializer")
    candidate_eval = arena_module.ArtifactEvaluator(ARTIFACTS / "C108")
    p_eval = arena_module.ArtifactEvaluator(ARTIFACTS / "P-E4")
    c_composition = arena_module.ComposedArtifactEvaluator(
        initializer_eval, candidate_eval, policy_source="candidate", value_source="current"
    )
    p_composition = arena_module.ComposedArtifactEvaluator(
        initializer_eval, p_eval, policy_source="candidate", value_source="current"
    )
    if (
        c_composition.candidate_evaluator is not candidate_eval
        or c_composition.current_evaluator is not initializer_eval
        or p_composition.candidate_evaluator is not p_eval
        or p_composition.current_evaluator is not initializer_eval
    ):
        raise ValueError("seed552_composed_evaluator_component_preflight_failed")
    policy_preflight = arena.resolve_strength_comparison_runtime_contract(
        current_artifact=ARTIFACTS / "initializer",
        challenger_artifact=ARTIFACTS / "C108",
        exact_root_threshold=16,
        native_probe=PROBE,
        tablebase=MATERIALIZED_TABLEBASE,
    )
    expected_preflight = {
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_leaf_solve_mode": "disabled",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    }
    if policy_preflight is None or any(policy_preflight.get(k) != v for k, v in expected_preflight.items()):
        raise ValueError("seed552_exact_root_runtime_preflight_failed")
    write_json(output / "runtime-policy-preflight.json", policy_preflight)
    source_paths = [
        "ml/alphazero_lite/arena.py", "ml/alphazero_lite/seed552_c108_serving_p.py",
        "ml/alphazero_lite/run_seed552_c108_serving_p.py", "ml/alphazero_lite/verify_seed552_c108_serving_p.py",
        "ml/alphazero_lite/seed552_arena_entrypoint.py", "ml/alphazero_lite/seed551_controlled_policy_strength.py",
        "ml/alphazero_lite/run_seed551_controlled_policy_strength.py",
        "ml/alphazero_lite/verify_seed551_controlled_policy_strength.py", "ml/alphazero_lite/seed511_fixed_composition_screen.py",
        "ml/alphazero_lite/seed542_expanded_policy_strength_screen.py",
        "ml/alphazero_lite/verify_seed542_expanded_policy_strength_screen.py",
        "ml/alphazero_lite/kalah_rules.py",
        "ml/alphazero_lite/classic_mcts.py", "ml/alphazero_lite/native_exact_root_tablebase.py",
        "ml/alphazero_lite/test_seed552_c108_serving_p.py",
    ]
    source_hashes = {p: protocol.sha256(root / p) for p in source_paths if (root / p).is_file()}
    observed_failed_runtime = {
        "cpu_affinity": list(range(32)), "nice": 19,
        "address_space_rlimit": [17179869184, 17179869184],
        "cgroup_limits": {"cpu.max": "25000 100000", "cpu.weight": "1",
                           "cpuset.cpus.effective": None, "memory.max": "4294967296"},
        "environment": {"BLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1"},
        "torch_interop_threads": 1, "torch_threads": 1,
    }
    execution_attempts = {
        "schema": "seed552_pre_game_execution_attempts_v1",
        "diagnostic_invocation_id": "79bed5b6f3e74053a85a46d097a2c6fa",
        "diagnostic_arena_games_started": False,
        "attempts": [
            {
                "attempt": 1, "command": launch_command(root, output),
                "systemd_invocation_id": "35221521edd14e5aaf9f374ba05fb57f",
                "systemd_unit_result": "failed", "exit_status": 1,
                "failure": "seed551_registered_runtime_contract_mismatch",
                "failure_detail": "Systemd AllowedCPUs did not restrict scheduler affinity; diagnostic observed CPUs 0-31.",
                "observed_runtime": observed_failed_runtime, "arena_games_started": False,
            },
            {
                "attempt": 2, "command": launch_command(root, output),
                "systemd_invocation_id": "14e9e9a5168a4aeaae48ebd540cf948c",
                "systemd_unit_result": "failed", "exit_status": 1,
                "failure": "seed551_registered_runtime_contract_mismatch",
                "failure_detail": "Affinity remained 0-31 with AllowedCPUs alone; taskset -c 30 is now added to the frozen launcher.",
                "observed_runtime": observed_failed_runtime, "arena_games_started": False,
            },
        ],
    }
    write_json(output / "execution-attempts.json", execution_attempts)
    registration = {
        "schema": "seed552_c108_serving_p_v1",
        "status": "prospective_candidate_and_rules_frozen_before_games",
        "claim": "exploratory C108 versus frozen serving P; no inference from C versus T",
        "preserve_seed550_candidate_designation": True,
        "preserve_seed551_do_not_advance_source_evaluator_branch": True,
        "suite_path": protocol.OPENINGS_PATH,
        "suite_sha256": protocol.OPENINGS_SHA256,
        "suite_rows": len(openings),
        "new_games": protocol.GAMES,
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(output / "runtime-authentication.json"),
        "historical_verification_receipts": {
            filename: protocol.sha256(auth_target / filename)
            for filename in ("seed550-training-verification.json", "seed551-publication-verification.json")
        },
        "historical_verifier_attempts_sha256": protocol.sha256(output / "historical-verifier-attempts.json"),
        "runtime_policy_preflight_sha256": protocol.sha256(output / "runtime-policy-preflight.json"),
        "readme_sha256": protocol.sha256(output / "README.md"),
        "authenticated_file_sha256": {
            protocol.C_CHECKPOINT: protocol.sha256(root / protocol.C_CHECKPOINT),
            protocol.P_CHECKPOINT: protocol.sha256(root / protocol.P_CHECKPOINT),
            protocol.INITIALIZER: protocol.sha256(root / protocol.INITIALIZER),
            "deploy/serving-release.json": protocol.sha256(root / "deploy/serving-release.json"),
            "deploy/release-manifest.json": protocol.sha256(root / "deploy/release-manifest.json"),
            protocol.NATIVE_PROBE: protocol.sha256(root / protocol.NATIVE_PROBE),
            protocol.TABLEBASE_POINTER: protocol.sha256(root / protocol.TABLEBASE_POINTER),
            "docs/data/seed549-controlled-training/registration.json": protocol.sha256(root / "docs/data/seed549-controlled-training/registration.json"),
            "docs/data/seed504-policy-only-ablation/registration.json": protocol.sha256(root / "docs/data/seed504-policy-only-ablation/registration.json"),
        },
        "opening_overlap_note_sha256": protocol.sha256(output / "overlap-limitations.json"),
        "opening_overlap_report_sha256": protocol.sha256(output / "overlap-report.json"),
        "opening_overlap_rows_sha256": protocol.sha256(output / "overlap-per-opening.jsonl"),
        "execution_attempts_sha256": protocol.sha256(output / "execution-attempts.json"),
        "authenticated_historical_inputs": {
            "seed550_registration": {
                "path": "docs/data/seed549-controlled-training/registration.json",
                "sha256": protocol.sha256(root / "docs/data/seed549-controlled-training/registration.json"),
            },
            "seed550_checkpoint_sha256": protocol.C_SHA256,
            "seed504_registration": {
                "path": "docs/data/seed504-policy-only-ablation/registration.json",
                "sha256": protocol.sha256(root / "docs/data/seed504-policy-only-ablation/registration.json"),
            },
            "seed504_E4_sha256": protocol.P_SHA256,
            "serving_release": "deploy/serving-release.json",
        },
        "analysis": {
            "C_win": 1.0, "terminal_draw": 0.5, "loss": 0.0,
            "seat_average_per_opening": True, "bootstrap_clusters": "opening",
            "resamples": 20_000, "replacement": True, "seed": 545,
            "lower_quantile": 0.05, "descriptive_quantiles": [0.025, 0.975],
            "numpy_quantile_method": "linear",
            "capped_nonterminal_prevents_valid_advancement": True,
            "advance": "all 2048 games valid and terminal and lower bound > 0.50: advance_to_separate_1200_simulation_screen",
            "otherwise_complete": "do_not_advance_C108",
            "incomplete_or_invalid": "no_valid_strength_decision",
            "promotion_or_deployment_authorized": False,
        },
        "arena": {
            "games_per_opening": 2, "simulations_per_side": 128, "c_puct": 1.25,
            "workers": 1, "root_policy": "deterministic", "temperature": 0.0,
            "noise": False, "fpu": 0.0, "subtree_reuse": False,
            "value_normalization": False, "tactical_bias": 0.0,
            "exact_root_threshold": 16, "exact_root_objective": "final_score_margin",
            "tie_rule": "registered prior/index", "exact_leaf_solving": False,
            "random_opening_plies": 0, "max_plies": 200,
            "seed_contract": "azlite_eval_seed_v2", "seed": "545 + opening_index",
        },
        "execution_sources": source_hashes,
        "dependencies": registered_runtime(),
        "durable_execution_command": launch_command(root, output),
        "runtime": registered_runtime(),
        "execution_root": str(root),
        "materialized_tablebase_path": str(MATERIALIZED_TABLEBASE.resolve()),
    }
    write_json(output / "registration.json", registration)
    registration_hash = protocol.sha256(output / "registration.json")
    freeze = {
        "schema": "seed552_execution_freeze_v1", "registration_sha256": registration_hash,
        "openings_sha256": protocol.sha256(raw_output), "source_sha256": source_hashes,
        "runtime_authentication_sha256": registration["runtime_authentication_sha256"],
        "tablebase_object_sha256": protocol.TABLEBASE_OBJECT_SHA256,
        "tablebase_object_size_bytes": protocol.TABLEBASE_OBJECT_SIZE,
        "tablebase_pointer_sha256": runtime["tablebase"]["historical_pointer_sha256"],
        "tablebase_repository_object_sha256": runtime["tablebase"]["repository_object_sha256"],
        "tablebase_repository_object_size_bytes": runtime["tablebase"]["repository_object_size_bytes"],
        "tablebase_pointer_git_commit": runtime["tablebase"]["historical_pointer_git_commit"],
        "tablebase_pointer_git_blob": runtime["tablebase"]["historical_pointer_git_blob"],
        "commands": "complete command-preflight.jsonl; one entry per global suite index",
        "analysis": registration["analysis"], "arena": registration["arena"],
        "runtime": registration["runtime"],
        "durable_execution_command": registration["durable_execution_command"],
    }
    write_json(output / "execution-freeze.json", freeze)
    preflight = [build_command(root, output, i, freeze) for i in range(protocol.OPENINGS)]
    (output / "command-preflight.jsonl").write_bytes(b"".join(protocol.stable_json(x) + b"\n" for x in preflight))
    freeze["command_preflight_sha256"] = protocol.sha256(output / "command-preflight.jsonl")
    write_json(output / "execution-freeze.json", freeze)
    # Validate supported argument parsing and all frozen identity links before games.
    for i, record in enumerate(preflight):
        if record["opening_index"] != i or record["suite_row_sha256"] != protocol.sha256(batch / f"opening-{i:04d}.jsonl"):
            raise ValueError(f"seed552_preflight_opening_mismatch:{i}")
    return {"status": "prepared_frozen", "openings": protocol.OPENINGS, "games": protocol.GAMES, "commands": len(preflight), "registration_sha256": registration_hash}


def build_command(root: Path, output: Path, index: int, freeze: dict[str, Any]) -> dict[str, Any]:
    b = output / "batches"
    art = output / "runtime-artifacts"
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    argv = [
        sys.executable, "-m", "ml.alphazero_lite.seed552_arena_entrypoint",
        "--challenger", str(art / "C108"), "--current", str(art / "P-E4"),
        "--challenger-policy-artifact", str(art / "C108"),
        "--challenger-value-artifact", str(art / "initializer"),
        "--current-policy-artifact", str(art / "P-E4"),
        "--current-value-artifact", str(art / "initializer"),
        "--games", "2", "--games-per-opening", "2", "--workers", "1",
        "--challenger-simulations", "128", "--current-simulations", "128",
        "--c-puct", "1.25", "--seed-contract", "azlite_eval_seed_v2",
        "--challenger-search-options-json", options, "--current-search-options-json", options,
        "--random-opening-plies", "0", "--max-moves", "200",
        "--exact-root-native-probe", protocol.NATIVE_PROBE,
        "--exact-root-tablebase", str(MATERIALIZED_TABLEBASE.resolve()),
        "--seed", str(protocol.SEED + index), "--opening-prefixes-jsonl", str(b / f"opening-{index:04d}.jsonl"),
        "--suite-sha256", protocol.sha256(b / f"opening-{index:04d}.jsonl"),
        "--game-jsonl", str(b / f"games-{index:04d}.jsonl"), "--out", str(b / f"report-{index:04d}.json"),
    ]
    old_argv = sys.argv
    try:
        sys.argv = ["arena.py", *argv[3:]]
        parsed = arena.parse_args()
    finally:
        sys.argv = old_argv
    if (parsed.games, parsed.games_per_opening, parsed.workers, parsed.challenger_simulations, parsed.current_simulations, parsed.seed) != (2, 2, 1, 128, 128, protocol.SEED + index):
        raise ValueError(f"seed552_arena_argument_contract_mismatch:{index}")
    if parsed.challenger_policy_artifact != str(art / "C108") or parsed.current_policy_artifact != str(art / "P-E4"):
        raise ValueError("seed552_component_assignment_mismatch")
    return {
        "opening_index": index, "argv": argv,
        "registration_sha256": freeze["registration_sha256"],
        "suite_row_sha256": protocol.sha256(b / f"opening-{index:04d}.jsonl"),
        "runtime_identity": registered_runtime(),
        "component_assignments": protocol.ASSIGNMENTS,
        "tablebase_object_sha256": protocol.TABLEBASE_OBJECT_SHA256,
    }


def recovery_action(batch: Path, index: int, committed: set[int]) -> str:
    if index in committed:
        return "skip"
    opening = batch / f"opening-{index:04d}.jsonl"
    if not opening.is_file():
        raise ValueError(f"seed552_frozen_opening_missing:{index}")
    names = [batch / f"{prefix}-{index:04d}.{ext}" for prefix, ext in (("command", "json"), ("games", "jsonl"), ("report", "json"))]
    exists = [p.exists() for p in names]
    if any(exists) and not all(exists):
        raise ValueError(f"seed552_incomplete_prior_attempt_preserved:{index}")
    if all(exists):
        return "recover"
    attempts = list(batch.glob(f"attempt-{index:04d}-*.json"))
    if attempts:
        raise ValueError(f"seed552_prior_attempt_preserved:{index}")
    return "execute"


def execute(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    base_runner.assert_runtime_contract(registered_runtime()["observed"])
    from ml.alphazero_lite import verify_seed552_c108_serving_p as verify

    ledger_path = output / "outcomes.jsonl"
    batch = output / "batches"
    preflight = [json.loads(line) for line in (output / "command-preflight.jsonl").read_text().splitlines() if line]
    committed = set()
    committed_rows: list[dict[str, Any]] = []
    for index in range(protocol.OPENINGS):
        completion = batch / f"completion-{index:04d}.json"
        if completion.is_file():
            expected_rows = verify.validate_pair(root, output, index)
            receipt = json.loads(completion.read_text())
            if receipt.get("outcomes") != expected_rows or receipt.get("command_sha256") != protocol.sha256(batch / f"command-{index:04d}.json") or receipt.get("games_sha256") != protocol.sha256(batch / f"games-{index:04d}.jsonl") or receipt.get("report_sha256") != protocol.sha256(batch / f"report-{index:04d}.json"):
                raise ValueError(f"seed552_committed_pair_receipt_mismatch:{index}")
            committed.add(index)
            committed_rows.extend(expected_rows)
    def rebuild_ledger(rows: list[dict[str, Any]]) -> None:
        rows.sort(key=lambda row: (row["opening_index"], row["game_position"]))
        pending = ledger_path.with_name(".outcomes.pending")
        pending.write_bytes(b"".join(protocol.stable_json(row) + b"\n" for row in rows))
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, ledger_path)

    rebuild_ledger(committed_rows)
    write_json(output / "execution-progress.json", {
        "schema": "seed552_durable_execution_progress_v1",
        "status": "running",
        "committed_opening_pairs": len(committed),
        "committed_games": len(committed_rows),
        "last_committed_opening_index": max(committed) if committed else None,
    })
    for index in range(protocol.OPENINGS):
        action = recovery_action(batch, index, committed)
        if action == "skip":
            continue
        if action == "execute":
            cmd = preflight[index]
            write_json(batch / f"command-{index:04d}.json", cmd)
            result = subprocess.run(cmd["argv"], cwd=root, check=False, capture_output=True, text=True)
            write_json(batch / f"attempt-{index:04d}-00.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError(f"seed552_arena_failed_preserved:{index}")
        pair = verify.validate_pair(root, output, index)
        # This per-opening receipt is the atomic commit point for a validated pair.
        command_path = batch / f"command-{index:04d}.json"
        games_path = batch / f"games-{index:04d}.jsonl"
        report_path = batch / f"report-{index:04d}.json"
        write_json(batch / f"completion-{index:04d}.json", {
            "opening_index": index,
            "registration_sha256": protocol.sha256(output / "registration.json"),
            "suite_sha256": protocol.OPENINGS_SHA256,
            "opening_sha256": protocol.sha256(batch / f"opening-{index:04d}.jsonl"),
            "command_sha256": protocol.sha256(command_path),
            "games_sha256": protocol.sha256(games_path),
            "report_sha256": protocol.sha256(report_path),
            "outcomes": pair,
        })
        committed_rows.extend(pair)
        rebuild_ledger(committed_rows)
        write_json(output / "execution-progress.json", {
            "schema": "seed552_durable_execution_progress_v1",
            "status": "running",
            "committed_opening_pairs": len({row["opening_index"] for row in committed_rows}),
            "committed_games": len(committed_rows),
            "last_committed_opening_index": index,
        })
    if len(committed_rows) == protocol.GAMES:
        analysis, opening_scores = protocol.analyze(committed_rows)
        write_json(output / "analysis.json", analysis)
        (output / "per-opening-scores.jsonl").write_bytes(
            b"".join(protocol.stable_json(row) + b"\n" for row in opening_scores)
        )
        result = verify.verify(root, output)
        write_json(output / "verification.json", result)
        receipt_paths = (
            "registration.json", "execution-freeze.json", "command-preflight.jsonl",
            "runtime-authentication.json", "openings.jsonl", "outcomes.jsonl",
            "analysis.json", "per-opening-scores.jsonl",
        )
        write_json(output / "verification-receipt.json", {
            "schema": "seed552_read_only_verification_receipt_v1",
            "status": result["status"],
            "verified_games": result["validated_games"],
            "verified_pairs": result["validated_pairs"],
            "result_sha256": protocol.sha256(output / "verification.json"),
            "evidence_sha256": {relative: protocol.sha256(output / relative) for relative in receipt_paths},
        })
        write_json(output / "execution-progress.json", {
            "schema": "seed552_durable_execution_progress_v1",
            "status": "complete_verified",
            "committed_opening_pairs": protocol.OPENINGS,
            "committed_games": protocol.GAMES,
            "last_committed_opening_index": protocol.OPENINGS - 1,
            "verification_status": result["status"],
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out if args.out.is_absolute() else root / args.out
    if args.prepare:
        print(json.dumps(prepare(root, output), indent=2, sort_keys=True))
    elif args.execute:
        execute(root, output)
    else:
        parser.error("specify --prepare or --execute")


if __name__ == "__main__":
    main()
