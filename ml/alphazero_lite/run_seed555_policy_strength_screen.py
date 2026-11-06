"""Prepare, freeze, execute, and recover the seed555 policy screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed551_controlled_policy_strength as runtime
from ml.alphazero_lite import seed555_policy_strength_screen as protocol

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
ARTIFACTS = OUT / "runtime-artifacts"
TABLEBASE = Path("/home/alex/Mancala/ai/.tmp/seed542-screen-runtime/kalah_v1_21.kvtb")


def _read_live_cpu_max() -> tuple[str, str]:
    memberships = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next(
        line.split("::", 1)[1] for line in memberships if line.startswith("0::")
    )
    path = Path("/sys/fs/cgroup") / relative.lstrip("/") / "cpu.max"
    try:
        return str(path), path.read_text().strip()
    except OSError as error:
        raise ValueError(
            f"seed555_cpu_max_unreadable:path={path}:error={error!r}"
        ) from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(protocol.stable_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _copy_component(
    root: Path, name: str, source: str, expected: str
) -> dict[str, Any]:
    src = root / source
    target = ARTIFACTS / name
    if protocol.sha256(src) != expected:
        raise ValueError(f"seed555_checkpoint_identity_mismatch:{name}")
    if target.exists():
        raise FileExistsError(f"seed555_component_exists:{name}")
    target.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.export_artifact",
            "--checkpoint",
            str(src),
            "--out-dir",
            str(target),
            "--version",
            f"seed555-{name}-update108",
            "--self-play-games",
            "0",
            "--model-type",
            "residual_v3",
            "--rules-version",
            "kalah_v1",
            "--input-encoding",
            "kalah_v3",
        ],
        cwd=root,
        check=True,
    )
    source_policy_path = root / "model-artifact/current/search_policy.json"
    source_policy = json.loads(source_policy_path.read_text())
    if (
        source_policy.get("schema") != "azlite_runtime_search_policy_v1"
        or source_policy.get("tablebase", {}).get("sha256") != protocol.TABLEBASE_SHA256
        or source_policy.get("native_probe", {}).get("sha256")
        != protocol.sha256(root / protocol.NATIVE_PROBE)
    ):
        raise ValueError("seed555_source_search_policy_asset_identity_mismatch")
    # The published runtime uses a Git LFS pointer in the worktree. Resolve the
    # sidecar against the separately authenticated materialized object without
    # replacing or touching the tracked pointer.
    source_policy["tablebase"]["path"] = os.path.relpath(TABLEBASE, root)
    (target / "search_policy.json").write_text(
        json.dumps(source_policy, indent=2, sort_keys=True) + "\n"
    )
    if protocol.sha256(target / "model.npz") != expected:
        raise ValueError(f"seed555_exported_artifact_identity_mismatch:{name}")
    return {
        "source": source,
        "checkpoint_sha256": expected,
        "source_search_policy_sha256": protocol.sha256(
            root / "model-artifact/current/search_policy.json"
        ),
        "artifact_dir": str(target.relative_to(root)),
        "files": {
            path.name: protocol.sha256(path)
            for path in sorted(target.iterdir())
            if path.is_file()
        },
    }


def _runtime(root: Path) -> dict[str, Any]:
    # Keep each registration/authentication document independent; the base
    # helper is cached and later component bindings must not mutate the
    # prospective live-runtime contract through a shared dictionary.
    registered = dict(runtime.runtime_settings())
    registered["torch_threads"] = 1
    registered["torch_interop_threads"] = 1
    cpu_max_path, cpu_max = _read_live_cpu_max()
    cgroup = Path(cpu_max_path).parent
    relative = next(
        line.split("::", 1)[1]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    cgroup_values: dict[str, str | None] = {"cpu.max": cpu_max}
    for field in ("cpu.weight", "memory.max", "cpuset.cpus.effective"):
        try:
            cgroup_values[field] = (cgroup / field).read_text().strip()
        except OSError:
            cgroup_values[field] = None
    registered["cgroup_scope_diagnostic"] = {
        "membership": relative,
        "cpu.max_path": str(cpu_max_path),
        "limits": cgroup_values,
    }
    registered["runtime_assets"] = {
        "native_probe_path": protocol.NATIVE_PROBE,
        "native_probe_sha256": protocol.sha256(root / protocol.NATIVE_PROBE),
        "tablebase_repository_path": protocol.TABLEBASE_PATH,
        "tablebase_repository_sha256": protocol.sha256(root / protocol.TABLEBASE_PATH),
        "tablebase_repository_size_bytes": (root / protocol.TABLEBASE_PATH)
        .stat()
        .st_size,
        "tablebase_materialized_sha256": protocol.sha256(TABLEBASE),
        "tablebase_materialized_size_bytes": TABLEBASE.stat().st_size,
        "historical_lfs_pointer_sha256": protocol.TABLEBASE_POINTER_SHA256,
        "historical_lfs_pointer_commit": protocol.TABLEBASE_POINTER_COMMIT,
        "historical_lfs_pointer_size_bytes": 134,
        "tablebase_runtime_path": str(TABLEBASE.resolve()),
    }
    if (
        registered["runtime_assets"]["tablebase_repository_sha256"]
        != protocol.TABLEBASE_POINTER_SHA256
        or registered["runtime_assets"]["tablebase_repository_size_bytes"] != 134
        or registered["runtime_assets"]["tablebase_materialized_sha256"]
        != protocol.TABLEBASE_SHA256
        or registered["runtime_assets"]["tablebase_materialized_size_bytes"]
        != protocol.TABLEBASE_SIZE
    ):
        raise ValueError("seed555_materialized_tablebase_identity_mismatch")
    pointer = subprocess.run(
        [
            "git",
            "show",
            f"{protocol.TABLEBASE_POINTER_COMMIT}:{protocol.TABLEBASE_PATH}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(pointer).hexdigest() != protocol.TABLEBASE_POINTER_SHA256:
        raise ValueError("seed555_historical_tablebase_lfs_pointer_mismatch")
    pointer_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f"{protocol.TABLEBASE_POINTER_COMMIT}:{protocol.TABLEBASE_PATH}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    registered["runtime_assets"]["historical_lfs_pointer_git_blob"] = pointer_blob
    registered["runtime_assets"]["historical_lfs_pointer_size_bytes"] = len(pointer)
    return registered


def prepare(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    # This is a no-game check in the eventual host scope. Check cpu.max before
    # creating preparation files so a failed scope probe leaves no partial freeze.
    _read_live_cpu_max()
    runtime.assert_runtime_contract()
    runtime.runtime_settings.cache_clear()
    runtime_identity = _runtime(root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("seed555_output_must_be_empty")
    raw, openings = protocol.load_openings(root)
    for path, expected in (
        (protocol.T512_CHECKPOINT, protocol.T512_SHA256),
        (protocol.C128_CHECKPOINT, protocol.C128_SHA256),
        (protocol.INITIALIZER, protocol.INITIALIZER_SHA256),
    ):
        if protocol.sha256(root / path) != expected:
            raise ValueError(f"seed555_checkpoint_identity_mismatch:{path}")
    seed554 = root / protocol.SEED554
    binding = json.loads((seed554 / "training-input-binding.json").read_text())
    search_ledger = seed554 / "searches.jsonl"
    full_replay_receipt = seed554 / "derivative-verification.json"
    physical_receipt_path = seed554 / "physical-copy-verification-04.json"
    full_replay = json.loads(full_replay_receipt.read_text())
    physical_receipt = json.loads(physical_receipt_path.read_text())
    if (
        protocol.sha256(search_ledger) != binding.get("search_evidence_sha256")
        or protocol.sha256(full_replay_receipt)
        != binding.get("derivative_verification_sha256")
        or full_replay.get("rerun_searches") is not True
        or full_replay.get("paired_searches") != 11_270
        or len(search_ledger.read_text().splitlines()) != 11_270
        or physical_receipt.get("status") != "passed"
        or physical_receipt.get("exit_status") != 0
        or physical_receipt.get("copy_bytes_and_mtimes_unchanged") is not True
    ):
        raise ValueError("seed555_seed554_full_replay_receipt_binding_mismatch")
    if protocol.sha256(seed554 / "training-input-binding.json") != (
        "9c5fce89446ec18a83f8d287394e81b288d63b439d781b111cc4faaf0e8f6422"
    ):
        raise ValueError("seed555_seed554_training_input_binding_mismatch")

    output.mkdir(parents=True, exist_ok=True)
    (output / "openings.jsonl").write_bytes(raw)
    batches = output / "batches"
    batches.mkdir()
    for index, line in enumerate(raw.splitlines(keepends=True)):
        (batches / f"opening-{index:04d}.jsonl").write_bytes(line)
    global ARTIFACTS
    ARTIFACTS = output / "runtime-artifacts"
    ARTIFACTS.mkdir()
    components = {
        "T512": _copy_component(
            root, "T512", protocol.T512_CHECKPOINT, protocol.T512_SHA256
        ),
        "C128": _copy_component(
            root, "C128", protocol.C128_CHECKPOINT, protocol.C128_SHA256
        ),
        "initializer": _copy_component(
            root, "initializer", protocol.INITIALIZER, protocol.INITIALIZER_SHA256
        ),
    }
    runtime_auth = _runtime(root)
    runtime_auth["components"] = components
    runtime_auth["component_assignments"] = protocol.ASSIGNMENTS
    write_json(output / "runtime-authentication.json", runtime_auth)

    # Confirm composition routing without running an arena game.
    initializer = arena.ArtifactEvaluator(ARTIFACTS / "initializer")
    t512 = arena.ArtifactEvaluator(ARTIFACTS / "T512")
    c128 = arena.ArtifactEvaluator(ARTIFACTS / "C128")
    challenger = arena.ComposedArtifactEvaluator(
        initializer, t512, policy_source="candidate", value_source="current"
    )
    opponent = arena.ComposedArtifactEvaluator(
        initializer, c128, policy_source="candidate", value_source="current"
    )
    if (
        challenger.candidate_evaluator is not t512
        or challenger.current_evaluator is not initializer
        or opponent.candidate_evaluator is not c128
        or opponent.current_evaluator is not initializer
    ):
        raise ValueError("seed555_composed_evaluator_preflight_failed")
    preflight = arena.resolve_strength_comparison_runtime_contract(
        current_artifact=ARTIFACTS / "C128",
        challenger_artifact=ARTIFACTS / "T512",
        exact_root_threshold=16,
        native_probe=root / protocol.NATIVE_PROBE,
        tablebase=TABLEBASE,
    )
    expected_exact = {
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }
    if preflight is None or any(
        preflight.get(k) != v for k, v in expected_exact.items()
    ):
        raise ValueError("seed555_native_runtime_preflight_failed")
    preflight.update(
        {
            "native_probe_sha256": runtime_auth["runtime_assets"][
                "native_probe_sha256"
            ],
            "materialized_tablebase_sha256": runtime_auth["runtime_assets"][
                "tablebase_materialized_sha256"
            ],
            "materialized_tablebase_size_bytes": protocol.TABLEBASE_SIZE,
            "tablebase_repository_pointer_sha256": runtime_auth["runtime_assets"][
                "tablebase_repository_sha256"
            ],
            "policy_sidecar_sha256": {
                name: components[name]["files"]["search_policy.json"]
                for name in ("T512", "C128", "initializer")
            },
        }
    )
    write_json(output / "runtime-policy-preflight.json", preflight)

    overlap = protocol.overlap_report(root, openings)
    write_json(
        output / "overlap-report.json",
        {k: v for k, v in overlap.items() if k != "per_opening"},
    )
    (output / "overlap-per-opening.jsonl").write_bytes(
        b"".join(protocol.stable_json(row) + b"\n" for row in overlap["per_opening"])
    )
    write_json(
        output / "overlap-limitations.json",
        {
            "reused_suite": True,
            "seed461_continuation_overlap_unresolved": True,
            "historical_games_or_outcomes_imported": False,
            "exploratory": True,
        },
    )
    authenticated = {
        path: protocol.sha256(root / path)
        for path in (
            protocol.T512_CHECKPOINT,
            protocol.C128_CHECKPOINT,
            protocol.INITIALIZER,
            protocol.OPENINGS_PATH,
            protocol.NATIVE_PROBE,
            protocol.TABLEBASE_PATH,
            f"{protocol.SEED554}/training-input-binding.json",
            f"{protocol.SEED554}/searches.jsonl",
            f"{protocol.SEED554}/derivative-verification.json",
            f"{protocol.SEED554}/physical-copy-verification-04.json",
        )
    }
    sources = {
        path: protocol.sha256(root / path)
        for path in (
            "ml/alphazero_lite/arena.py",
            "ml/alphazero_lite/export_artifact.py",
            "ml/alphazero_lite/runtime_search_policy.py",
            "ml/alphazero_lite/input_encodings.py",
            "ml/alphazero_lite/seed555_policy_strength_screen.py",
            "ml/alphazero_lite/run_seed555_policy_strength_screen.py",
            "ml/alphazero_lite/verify_seed555_policy_strength_screen.py",
            "ml/alphazero_lite/test_seed555_policy_strength_screen.py",
            "ml/alphazero_lite/seed555_arena_entrypoint.py",
            "ml/alphazero_lite/seed551_arena_entrypoint.py",
            "ml/alphazero_lite/seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/run_seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/verify_seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/seed511_fixed_composition_screen.py",
            "ml/alphazero_lite/seed542_expanded_policy_strength_screen.py",
            "ml/alphazero_lite/kalah_rules.py",
            "ml/alphazero_lite/classic_mcts.py",
            "ml/alphazero_lite/native_exact_root_tablebase.py",
        )
    }
    analysis_rules = protocol.ANALYSIS_RULES
    arena_rules = protocol.ARENA_RULES
    runtime_settings = runtime_identity
    readme = (
        "# Seed555: T512 versus C128 policy-strength screen\n\n"
        "Exploratory comparison of the #554 update-108 T512 challenger and regenerated C128 control. "
        "Each policy uses its own exported checkpoint and trunk; both sides use the same frozen initializer value network and initializer trunk through `ComposedArtifactEvaluator`. "
        "The teacher budgets distinguish the trained policies; arena play uses the same 128-simulation budget on both sides.\n\n"
        "Arena settings: one worker, c_puct 1.25, deterministic root policy, zero temperature and noise, FPU zero, no subtree reuse or value normalization, and zero tactical root bias. "
        "Native exact-root threshold 16 uses final-score-margin objective and highest legal network prior then lowest move index for ties. Exact-leaf solving is disabled; openings use zero random plies, a 200-ply cap, and seed `545 + opening_index` under `azlite_eval_seed_v2`.\n\n"
        "The preregistered screen contains 1,024 indexed seat-swapped opening pairs (2,048 new games) from the reused seed511 v4 suite. "
        "Every indexed opening is retained and replayed. No historical game or outcome is imported. "
        "Canonical-state and production-float32 overlap is reported against both #554 training derivatives and the unchanged development membership.\n\n"
        "The unresolved seed461 continuation-overlap limitation remains. This result is exploratory and does not establish superiority to serving P or authorize deployment. "
        "A positive result permits only proposing a separate comparison against serving P.\n\n"
        "Analysis scores T512 wins as 1, terminal draws as 0.5, and losses as 0, averaging both seats within each opening. "
        "It uses 20,000 opening-cluster bootstrap resamples with replacement, seed 545, NumPy linear quantiles, a one-sided 95% lower bound at 0.05, and a descriptive interval at 0.025/0.975. "
        "Only 2,048 valid terminal games with lower bound strictly above 0.50 permit `propose_T512_vs_serving_P_screen`; a complete screen otherwise yields `do_not_advance_teacher_budget_branch`. A capped/nonterminal game blocks a valid advancement decision.\n\n"
        "The #554 published full-replay receipt, search ledger, and derivative shards are hash-bound through `training-input-binding.json`. "
        "The repository tablebase LFS pointer identity is distinct from the separately materialized tablebase object SHA256 and byte size.\n"
    )
    (output / "README.md").write_text(readme)
    registration = {
        "schema": "seed555_t512_vs_c128_v1",
        "status": "prospective_rules_frozen_before_games",
        "exploratory": True,
        "suite_path": protocol.OPENINGS_PATH,
        "suite_sha256": protocol.OPENINGS_SHA256,
        "suite_rows": protocol.OPENINGS,
        "new_games": protocol.GAMES,
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(
            output / "runtime-authentication.json"
        ),
        "runtime_policy_preflight_sha256": protocol.sha256(
            output / "runtime-policy-preflight.json"
        ),
        "readme_sha256": protocol.sha256(output / "README.md"),
        "overlap_report_sha256": protocol.sha256(output / "overlap-report.json"),
        "overlap_rows_sha256": protocol.sha256(output / "overlap-per-opening.jsonl"),
        "overlap_limitations_sha256": protocol.sha256(
            output / "overlap-limitations.json"
        ),
        "authenticated_files_sha256": authenticated,
        "source_sha256": sources,
        "runtime": runtime_settings,
        "analysis": analysis_rules,
        "arena": arena_rules,
        "execution_root": str(root),
        "execution_output": str(output),
        "durable_execution_command": launch_command(root, output),
        "seed554": {
            "training_input_binding_sha256": protocol.sha256(
                seed554 / "training-input-binding.json"
            ),
            "search_ledger_sha256": protocol.sha256(search_ledger),
            "full_replay_receipt_sha256": protocol.sha256(full_replay_receipt),
            "physical_copy_receipt_sha256": protocol.sha256(physical_receipt_path),
            "paired_searches": 11270,
            "rerun_searches_receipt": True,
        },
    }
    write_json(output / "registration.json", registration)
    freeze = {
        "schema": "seed555_execution_freeze_v1",
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "suite_sha256": protocol.OPENINGS_SHA256,
        "source_sha256": sources,
        "analysis": analysis_rules,
        "arena": arena_rules,
        "runtime": runtime_settings,
        "execution_output": registration["execution_output"],
        "runtime_authentication_sha256": registration["runtime_authentication_sha256"],
        "commands": "command-preflight.jsonl has one entry for each global opening index",
    }
    write_json(output / "execution-freeze.json", freeze)
    commands = [
        build_command(root, output, index, freeze, runtime_settings)
        for index in range(protocol.OPENINGS)
    ]
    (output / "command-preflight.jsonl").write_bytes(
        b"".join(protocol.stable_json(row) + b"\n" for row in commands)
    )
    freeze["command_preflight_sha256"] = protocol.sha256(
        output / "command-preflight.jsonl"
    )
    write_json(output / "execution-freeze.json", freeze)
    return {
        "status": "prepared_frozen",
        "openings": protocol.OPENINGS,
        "games": protocol.GAMES,
        "commands": len(commands),
    }


def build_command(
    root: Path,
    output: Path,
    index: int,
    freeze: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    batch = output / "batches"
    art = output / "runtime-artifacts"
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    argv = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.seed555_arena_entrypoint",
        "--challenger",
        str(art / "T512"),
        "--current",
        str(art / "C128"),
        "--challenger-policy-artifact",
        str(art / "T512"),
        "--challenger-value-artifact",
        str(art / "initializer"),
        "--current-policy-artifact",
        str(art / "C128"),
        "--current-value-artifact",
        str(art / "initializer"),
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
        "azlite_eval_seed_v2",
        "--challenger-search-options-json",
        options,
        "--current-search-options-json",
        options,
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--exact-root-solve-threshold",
        "16",
        "--exact-root-native-probe",
        protocol.NATIVE_PROBE,
        "--exact-root-tablebase",
        str(TABLEBASE.resolve()),
        "--seed",
        str(protocol.SEED + index),
        "--opening-prefixes-jsonl",
        str(batch / f"opening-{index:04d}.jsonl"),
        "--suite-sha256",
        protocol.sha256(batch / f"opening-{index:04d}.jsonl"),
        "--game-jsonl",
        str(batch / f"games-{index:04d}.jsonl"),
        "--out",
        str(batch / f"report-{index:04d}.json"),
    ]
    saved_argv = sys.argv
    try:
        sys.argv = ["arena.py", *argv[3:]]
        parsed = arena.parse_args()
    finally:
        sys.argv = saved_argv
    if (
        parsed.games != 2
        or parsed.games_per_opening != 2
        or parsed.workers != 1
        or parsed.challenger_simulations != protocol.SIMULATIONS
        or parsed.current_simulations != protocol.SIMULATIONS
        or parsed.c_puct != 1.25
        or parsed.seed != protocol.SEED + index
        or parsed.exact_root_solve_threshold != 16
        or parsed.random_opening_plies != 0
        or parsed.max_moves != 200
        or json.loads(parsed.challenger_search_options_json) != protocol.SEARCH_OPTIONS
        or json.loads(parsed.current_search_options_json) != protocol.SEARCH_OPTIONS
    ):
        raise ValueError(f"seed555_arena_argument_contract_mismatch:{index}")
    return {
        "opening_index": index,
        "argv": argv,
        "registration_sha256": freeze["registration_sha256"],
        "suite_row_sha256": protocol.sha256(batch / f"opening-{index:04d}.jsonl"),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_identity": runtime_identity,
    }


def launch_command(root: Path, output: Path) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--unit=seed555-policy-strength-screen",
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
        "/usr/bin/nice",
        "-n",
        "19",
        "/usr/bin/prlimit",
        "--as=17179869184",
        "--",
        "/usr/bin/taskset",
        "-c",
        "30",
        sys.executable,
        "-m",
        "ml.alphazero_lite.run_seed555_policy_strength_screen",
        "--root",
        str(root),
        "--out",
        str(output),
        "--execute",
    ]


def recovery_action(batch: Path, index: int, committed: set[int]) -> str:
    if index in committed:
        return "skip"
    present = [
        (batch / f"{prefix}-{index:04d}.{suffix}").exists()
        for prefix, suffix in (
            ("command", "json"),
            ("games", "jsonl"),
            ("report", "json"),
        )
    ]
    if any(present) and not all(present):
        raise ValueError(f"seed555_incomplete_prior_evidence_preserved:{index}")
    if all(present):
        command_path = batch / f"command-{index:04d}.json"
        for attempt_path in batch.glob(f"attempt-{index:04d}-*.json"):
            try:
                attempt = json.loads(attempt_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"seed555_prior_attempt_unreadable_preserved:{index}"
                ) from error
            if attempt.get("returncode") != 0 or attempt.get(
                "command_sha256"
            ) != protocol.sha256(command_path):
                raise ValueError(f"seed555_failed_attempt_preserved:{index}")
        return "recover"
    if list(batch.glob(f"attempt-{index:04d}-*.json")):
        raise ValueError(f"seed555_conflicting_prior_attempt_preserved:{index}")
    return "execute"


def execute(root: Path, output: Path) -> None:
    from ml.alphazero_lite import verify_seed555_policy_strength_screen as verify

    root, output = root.resolve(), output.resolve()
    contract = verify.verify_preflight(root, output)
    expected = contract["registration"]["runtime"]
    runtime.assert_runtime_contract(expected["observed"])
    actual = _runtime(root)
    actual["observed"] = runtime.observed_runtime()
    actual_scope = actual.pop("cgroup_scope_diagnostic")
    expected_scope = expected.get("cgroup_scope_diagnostic", {})
    if actual != {
        key: value
        for key, value in expected.items()
        if key != "cgroup_scope_diagnostic"
    } or actual_scope.get("limits") != expected_scope.get("limits"):
        raise ValueError("seed555_registered_runtime_mismatch")
    verify.validate_execution_runtime_checks(output, expected)
    ledger = output / "outcomes.jsonl"
    commands = [
        json.loads(line)
        for line in (output / "command-preflight.jsonl").read_text().splitlines()
    ]
    committed: set[int] = set()
    rows: list[dict[str, Any]] = []
    actions: dict[int, str] = {}
    recoverable_pairs: dict[int, list[dict[str, Any]]] = {}
    for index in range(protocol.OPENINGS):
        marker = output / "batches" / f"completion-{index:04d}.json"
        if marker.is_file():
            pair = verify.validate_pair(root, output, index)
            receipt = json.loads(marker.read_text())
            for name in ("command", "games", "report"):
                path = (
                    output
                    / "batches"
                    / f"{name}-{index:04d}.{'jsonl' if name == 'games' else 'json'}"
                )
                if receipt.get(f"{name}_sha256") != protocol.sha256(path):
                    raise ValueError(f"seed555_committed_pair_receipt_mismatch:{index}")
            if receipt.get("outcomes") != pair:
                raise ValueError(f"seed555_committed_pair_outcomes_mismatch:{index}")
            committed.add(index)
            rows.extend(pair)
            actions[index] = "skip"
        else:
            action = recovery_action(output / "batches", index, committed)
            actions[index] = action
            if action == "recover":
                recoverable_pairs[index] = verify.validate_pair(root, output, index)
    _validate_ledger_projection(output, ledger, rows)
    # Persist the live runtime receipt only after all existing evidence has been
    # checked read-only, so rejected conflicting evidence remains byte-stable.
    append_jsonl(
        output / "execution-runtime-checks.jsonl",
        {
            "registration_sha256": protocol.sha256(output / "registration.json"),
            "runtime_contract_sha256": hashlib.sha256(
                protocol.stable_json(expected)
            ).hexdigest(),
            "runtime_identity": actual,
            "cgroup_scope_diagnostic": actual_scope,
            "status": "prelaunch_runtime_contract_passed",
            "arena_games_started_by_check": False,
        },
    )
    _reconcile_ledger(output, ledger, rows, allow_recovery=True)
    for index in range(protocol.OPENINGS):
        action = actions[index]
        if action == "skip":
            continue
        command_path = output / "batches" / f"command-{index:04d}.json"
        if action == "execute":
            command = commands[index]
            write_json(command_path, command)
            result = subprocess.run(
                command["argv"], cwd=root, capture_output=True, text=True
            )
            write_json(
                output / "batches" / f"attempt-{index:04d}-00.json",
                {
                    "opening_index": index,
                    "returncode": result.returncode,
                    "command_sha256": protocol.sha256(command_path),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )
            if result.returncode:
                raise RuntimeError(f"seed555_arena_failure_preserved:{index}")
        pair = verify.validate_pair(root, output, index)
        if action == "recover":
            if pair != recoverable_pairs[index]:
                raise ValueError(
                    f"seed555_saved_raw_pair_changed_after_prevalidation:{index}"
                )
        games_path = output / "batches" / f"games-{index:04d}.jsonl"
        report_path = output / "batches" / f"report-{index:04d}.json"
        for durable_path in (command_path, games_path, report_path):
            _fsync_file(durable_path)
        _fsync_directory(command_path.parent)
        if action == "recover":
            recovery_path = output / "batches" / f"recovery-{index:04d}.json"
            recovery_record = {
                "opening_index": index,
                "status": "complete_raw_pair_recovered_after_validation",
                "validated_game_records": 2,
                "command_sha256": protocol.sha256(command_path),
                "games_sha256": protocol.sha256(
                    output / "batches" / f"games-{index:04d}.jsonl"
                ),
                "report_sha256": protocol.sha256(
                    output / "batches" / f"report-{index:04d}.json"
                ),
                "recovery_reran_games": False,
            }
            if recovery_path.exists():
                if json.loads(recovery_path.read_text()) != recovery_record:
                    raise ValueError(f"seed555_recovery_receipt_conflict:{index}")
            else:
                write_json(recovery_path, recovery_record)
        completion = {
            "opening_index": index,
            "command_sha256": protocol.sha256(command_path),
            "games_sha256": protocol.sha256(
                output / "batches" / f"games-{index:04d}.jsonl"
            ),
            "report_sha256": protocol.sha256(
                output / "batches" / f"report-{index:04d}.json"
            ),
            "outcomes": pair,
        }
        write_json(output / "batches" / f"completion-{index:04d}.json", completion)
        rows.extend(pair)
        committed.add(index)
        _reconcile_ledger(output, ledger, rows)
    if len(rows) == protocol.GAMES:
        result, scores = protocol.analyze(rows)
        write_json(output / "analysis.json", result)
        (output / "per-opening-scores.jsonl").write_bytes(
            b"".join(protocol.stable_json(row) + b"\n" for row in scores)
        )
        write_json(output / "verification.json", verify.verify(root, output))


def _validate_ledger_projection(
    output: Path, ledger_path: Path, committed_rows: list[dict[str, Any]]
) -> None:
    pending = ledger_path.with_name(".outcomes.pending")
    if pending.exists():
        raise ValueError("seed555_pending_ledger_preserved")
    if not ledger_path.exists():
        return
    try:
        current = [
            json.loads(line) for line in ledger_path.read_text().splitlines() if line
        ]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("seed555_conflicting_ledger_preserved") from error
    expected = sorted(
        committed_rows,
        key=lambda row: (int(row["opening_index"]), int(row["game_position"])),
    )
    if (
        len(current) > len(expected)
        or len(current) % 2
        or current != expected[: len(current)]
    ):
        raise ValueError("seed555_conflicting_ledger_preserved")


def _reconcile_ledger(
    output: Path,
    path: Path,
    committed: list[dict[str, Any]],
    *,
    allow_recovery: bool = False,
) -> None:
    expected = sorted(
        committed, key=lambda row: (row["opening_index"], row["game_position"])
    )
    payload = b"".join(protocol.stable_json(row) + b"\n" for row in expected)
    existing_bytes = path.read_bytes() if path.exists() else None
    temporary = path.with_name(".outcomes.pending")
    if temporary.exists():
        raise ValueError("seed555_pending_ledger_preserved")
    if existing_bytes == payload:
        return
    if existing_bytes is not None:
        try:
            present = [
                json.loads(line)
                for line in existing_bytes.decode().splitlines()
                if line
            ]
        except json.JSONDecodeError as error:
            raise ValueError("seed555_conflicting_ledger_preserved") from error
        except UnicodeDecodeError as error:
            raise ValueError("seed555_conflicting_ledger_preserved") from error
        if (
            len(present) > len(expected)
            or len(present) % 2
            or present != expected[: len(present)]
        ):
            raise ValueError("seed555_conflicting_ledger_preserved")
    if allow_recovery and expected:
        recovery_dir = output / "ledger-recovery"
        recovery_dir.mkdir(exist_ok=True)
        sequence = 1
        while (recovery_dir / f"recovery-{sequence:04d}.json").exists():
            sequence += 1
        prior_path = None
        if existing_bytes is not None:
            prior_path = recovery_dir / f"ledger-before-recovery-{sequence:04d}.jsonl"
            if prior_path.exists():
                raise ValueError("seed555_ledger_recovery_evidence_conflict")
            shutil.copy2(path, prior_path)
            if prior_path.read_bytes() != existing_bytes:
                raise ValueError("seed555_ledger_recovery_backup_mismatch")
        pairs = sorted({int(row["opening_index"]) for row in expected})
        receipts = {
            str(index): protocol.sha256(
                output / "batches" / f"completion-{index:04d}.json"
            )
            for index in pairs
        }
        write_json(
            recovery_dir / f"recovery-{sequence:04d}.json",
            {
                "schema": "seed555_ledger_recovery_v1",
                "status": "rebuilt_from_validated_complete_pairs",
                "prior_ledger_sha256": hashlib.sha256(existing_bytes).hexdigest()
                if existing_bytes is not None
                else None,
                "preserved_ledger_path": str(prior_path.relative_to(output))
                if prior_path
                else None,
                "rebuilt_ledger_sha256": hashlib.sha256(payload).hexdigest(),
                "validated_completion_receipt_sha256": receipts,
                "validated_opening_pairs": len(pairs),
                "validated_games": len(expected),
            },
        )
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime.runtime_settings.cache_clear()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else args.root / args.out
    if args.prepare:
        print(json.dumps(prepare(args.root, output), indent=2, sort_keys=True))
    elif args.execute:
        execute(args.root, output)
    else:
        parser.error("specify --prepare or --execute")


if __name__ == "__main__":
    main()
