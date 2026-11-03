"""Prepare and durably run the frozen seed551 paired policy screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.alphazero_lite import arena
from ml.alphazero_lite import seed551_controlled_policy_strength as protocol
from ml.alphazero_lite import verify_seed551_controlled_policy_strength as verifier

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
TABLEBASE = ROOT / "model-artifact/runtime/kalah_v1_21.kvtb"
TABLEBASE_SOURCE = Path(
    "/home/alex/Mancala/ai/.tmp/seed542-screen-runtime/kalah_v1_21.kvtb"
)
PROBE = "model-artifact/runtime/kalah_v1_tablebase"
SEED = protocol.BOOTSTRAP_SEED
SIMULATIONS = protocol.SIMULATIONS


def canonical(value: Any) -> bytes:
    return protocol.stable_json(value)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def archive_failed_preparation(root: Path, output: Path, failure_check: str) -> Path:
    """Preserve a pre-game preparation failure byte-for-byte in a sibling archive."""
    root, output = root.resolve(), output.resolve()
    if not output.is_dir():
        raise FileNotFoundError("seed551_failed_preparation_output_missing")
    if any((output / "batches").glob("games-*.jsonl")):
        raise ValueError("seed551_cannot_archive_preparation_after_games_started")
    registration_path = output / "registration.json"
    freeze_path = output / "execution-freeze.json"
    if not registration_path.is_file() or not freeze_path.is_file():
        raise ValueError("seed551_failed_preparation_freeze_missing")
    registration = json.loads(registration_path.read_text())
    failure = {
        "schema": "seed551_preparation_failure_v1",
        "status": "pre_game_preparation_failed_preserved",
        "failure_check": failure_check,
        "arena_games_started": False,
        "registration_sha256": protocol.sha256(registration_path),
        "freeze_sha256": protocol.sha256(freeze_path),
        "command_preflight_sha256": protocol.sha256(output / "command-preflight.jsonl"),
        "prior_execution_root": registration.get("execution_root"),
        "execution_source_sha256": registration.get("execution_source_sha256", {}),
        "authenticated_file_sha256": registration.get("authenticated_file_sha256", {}),
    }
    save_json(output / "PREPARATION-FAILURE.json", failure)
    files = [
        {
            "path": str(path.relative_to(output)),
            "sha256": protocol.sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    save_json(
        output / "preparation-attempt-manifest.json",
        {
            "schema": "seed551_preparation_attempt_manifest_v1",
            "failure_sha256": protocol.sha256(output / "PREPARATION-FAILURE.json"),
            "files": files,
        },
    )
    destination = (
        root
        / "docs/data/seed551-controlled-policy-strength-screen-preparation-attempt-0001"
    )
    if destination.exists():
        raise FileExistsError("seed551_preparation_failure_archive_already_exists")
    output.rename(destination)
    return destination


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@lru_cache(maxsize=1)
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
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "device": "cpu",
        "dtype": "float32",
        "packages": packages,
        "environment": {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "resource_limits": {
            "cpu_quota": "25%",
            "cpu_weight": "1",
            "memory_max": "4294967296",
            "allowed_cpus": "30",
            "nice": 19,
            "address_space_limit": "17179869184",
            "workers": 1,
        },
        "observed": observed_runtime(),
    }


def _cgroup_limits() -> dict[str, str | None]:
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(
            line.split("::", 1)[1] for line in memberships if line.startswith("0::")
        )
        folder = Path("/sys/fs/cgroup") / relative.lstrip("/")
    except (OSError, StopIteration):
        return {
            name: None
            for name in ("cpu.max", "cpu.weight", "memory.max", "cpuset.cpus.effective")
        }
    values: dict[str, str | None] = {}
    for name in ("cpu.max", "cpu.weight", "memory.max", "cpuset.cpus.effective"):
        try:
            values[name] = (folder / name).read_text().strip()
        except OSError:
            values[name] = None
    return values


def observed_runtime() -> dict[str, Any]:
    """Capture effective numerical and operating-system resource settings."""
    return {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "nice": os.nice(0),
        "address_space_rlimit": list(resource.getrlimit(resource.RLIMIT_AS)),
        "cgroup_limits": _cgroup_limits(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "BLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def assert_runtime_contract(expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fail before preparation or games unless the registered limits are active."""
    observed = observed_runtime()
    expected_fixed = {
        "cpu_affinity": [30],
        "nice": 19,
        "address_space_rlimit": [17_179_869_184, 17_179_869_184],
        "environment": {name: "1" for name in observed["environment"]},
        "torch_threads": 1,
        "torch_interop_threads": 1,
    }
    limits = observed["cgroup_limits"]
    if (
        any(observed.get(key) != value for key, value in expected_fixed.items())
        or limits["cpu.max"] != "25000 100000"
        or limits["cpu.weight"] != "1"
        or limits["memory.max"] != "4294967296"
        or limits["cpuset.cpus.effective"] not in {"30", None}
        or (expected is not None and observed != expected)
    ):
        raise ValueError("seed551_registered_runtime_contract_mismatch")
    return observed


def export_checkpoint(
    root: Path, label: str, checkpoint: str, destination: Path
) -> None:
    """Export one frozen checkpoint and bind the registered serving sidecar."""
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refuse_artifact_overwrite:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.export_artifact",
        "--checkpoint",
        str(root / checkpoint),
        "--out-dir",
        str(destination),
        "--version",
        f"seed551-{label}-update108",
        "--self-play-games",
        "0",
        "--model-type",
        "residual_v3",
        "--rules-version",
        "kalah_v1",
        "--input-encoding",
        "kalah_v3",
    ]
    subprocess.run(command, cwd=root, check=True)
    shutil.copy2(root / "model-artifact/current/search_policy.json", destination)
    if protocol.sha256(destination / "model.npz") != protocol.sha256(root / checkpoint):
        raise ValueError(f"export_checkpoint_identity_mismatch:{label}")


def prepare(
    root: Path, output: Path, *, training_receipts: dict[str, Any]
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    assert_runtime_contract()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("seed551_output_must_be_empty_before_prepare")
    output.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = {
        "C": (protocol.CONTROL_CHECKPOINT, protocol.CONTROL_SHA256),
        "T": (protocol.TREATMENT_CHECKPOINT, protocol.TREATMENT_SHA256),
    }
    for label, (relative, expected) in expected_checkpoints.items():
        if protocol.sha256(root / relative) != expected:
            raise ValueError(f"seed551_update108_checkpoint_mismatch:{label}")
    corpus_result = training_receipts.get("corpus", {})
    training_result = training_receipts.get("training", {})
    corpus_output = corpus_result.get("output", corpus_result)
    training_output = training_result.get("output", training_result)
    if corpus_result.get("status") != "verified" or not corpus_output.get(
        "evidence_preserved"
    ):
        raise ValueError("seed551_corpus_readonly_verifier_not_passed")
    if training_result.get("status") != "verified" or not training_output.get(
        "bytes_and_mtimes_unchanged"
    ):
        raise ValueError("seed551_training_readonly_verifier_not_passed")
    training_receipts["authenticated_registrations"] = {
        "seed549_corpus": protocol.sha256(root / protocol.CORPUS / "registration.json"),
        "seed549_training": protocol.sha256(
            root / protocol.TRAINING / "registration.json"
        ),
    }
    verified_source_paths = set()
    for registry_path in (
        root / protocol.CORPUS / "registration.json",
        root / protocol.TRAINING / "registration.json",
    ):
        registry = json.loads(registry_path.read_text())
        verified_source_paths.update(registry.get("source_sha256", {}))
    training_receipts["verified_source_sha256"] = {
        relative: protocol.sha256(root / relative)
        for relative in sorted(verified_source_paths)
    }
    training_receipts["checkpoint_sha256"] = {
        "C_update108": protocol.sha256(root / protocol.CONTROL_CHECKPOINT),
        "T_update108": protocol.sha256(root / protocol.TREATMENT_CHECKPOINT),
    }
    save_json(output / "input-verification.json", training_receipts)
    readme = """# Seed551 controlled policy-strength screen

This is an exploratory comparison of the update-108 C and T policies. Each
policy keeps its own trunk; both sides use the same frozen initializer value
network and its own trunk through `ComposedArtifactEvaluator`. The comparison
measures the policy-strength difference associated with changing the source
self-play evaluator. It does not isolate policy-target quality, establish
superiority to serving P, or authorize promotion.

The preregistered suite contains exactly 1,024 indexed opening pairs and 2,048
games. T and C swap seats in every pair. It reuses seed511's indexed suite; the
seed461 continuation-overlap limitation remains unresolved. Canonical-state
and production-float32 overlap counts against both seed549 training populations
and the shared development population are in `overlap-summary.json` and
`overlap-per-opening.jsonl`; no suite rows were removed or replaced.

Scoring, opening-cluster bootstrap, quantiles, and decision thresholds are
frozen in `registration.json` before games. Games capped at 200 plies are
explicitly marked nonterminal and reported separately from terminal outcomes;
they use the arena's recorded current store-margin result for points.

The initial read-only input verification attempts and their corrective reruns
are preserved in `input-verification.json`. See `execution-freeze.json`,
`command-preflight.jsonl`, `commands.jsonl`, per-opening batch files, and
recovery/attempt ledgers for the complete execution trace. Independent
read-only verification reconstructs trajectory legality, winner perspective,
seat pairing, runtime contract, ledger, score, bootstrap, and decision.
The pre-game preparation attempt that surfaced the nested corpus receipt check
is preserved in `docs/data/seed551-controlled-policy-strength-screen-preparation-attempt-0001/`; it contains no arena games.

This screen is not a fresh holdout and cannot authorize deployment or
promotion. Non-advancement does not prove the source evaluator is universally
inferior.
"""
    (output / "README.md").write_text(readme)

    imported, suite_report = protocol.suite.import_seed511_suite(root)
    raw_suite = (root / protocol.suite.SEED511 / "openings.jsonl").read_bytes()
    if len(raw_suite.splitlines()) != protocol.OPENINGS:
        raise ValueError("seed551_suite_line_count_mismatch")
    (output / "openings.jsonl").write_bytes(raw_suite)
    overlap = protocol.overlap_report(root, imported)
    save_json(output / "suite-import-report.json", suite_report)
    save_json(
        output / "overlap-summary.json",
        {key: value for key, value in overlap.items() if key != "per_opening"},
    )
    (output / "overlap-per-opening.jsonl").write_bytes(
        b"".join(canonical(row) + b"\n" for row in overlap["per_opening"])
    )

    artifacts = output / "runtime-artifacts"
    export_checkpoint(root, "C", protocol.CONTROL_CHECKPOINT, artifacts / "C")
    export_checkpoint(root, "T", protocol.TREATMENT_CHECKPOINT, artifacts / "T")
    export_checkpoint(
        root, "initializer", protocol.INITIALIZER, artifacts / "initializer"
    )
    runtime = {
        "status": "components_authenticated",
        "component_assignments": protocol.ASSIGNMENTS,
        "components": {},
        "runtime_assets": {
            "native_probe_path": PROBE,
            "native_probe_sha256": protocol.sha256(root / PROBE),
            "tablebase_path": "model-artifact/runtime/kalah_v1_21.kvtb",
            "materialized_tablebase_path": str(TABLEBASE_SOURCE),
            "tablebase_sha256": protocol.sha256(TABLEBASE_SOURCE),
            "tablebase_pointer_sha256": protocol.sha256(
                root / "model-artifact/runtime/kalah_v1_21.kvtb"
            ),
        },
    }
    checkpoint_paths = {
        "C": protocol.CONTROL_CHECKPOINT,
        "T": protocol.TREATMENT_CHECKPOINT,
        "initializer": protocol.INITIALIZER,
    }
    for name in ("C", "T", "initializer"):
        path = artifacts / name
        runtime["components"][name] = {
            "checkpoint_path": checkpoint_paths[name],
            "checkpoint_sha256": protocol.sha256(root / checkpoint_paths[name]),
            "files": {
                file.name: protocol.sha256(file)
                for file in sorted(path.iterdir())
                if file.is_file()
            },
        }
    save_json(output / "runtime-authentication.json", runtime)
    from ml.alphazero_lite import seed511_fixed_composition_screen as suite_runtime

    source_registration = json.loads(
        (root / protocol.suite.SEED511 / "registration.json").read_text()
    )
    source_registration["runtime_assets"]["exact_root_tablebase"][
        "runtime_materialized_path"
    ] = str(TABLEBASE_SOURCE)
    pointer = suite_runtime.materialize_bound_tablebase(root, source_registration)
    try:
        preflight = arena.resolve_strength_comparison_runtime_contract(
            current_artifact=artifacts / "initializer",
            challenger_artifact=artifacts / "T",
            exact_root_threshold=16,
            native_probe=root / PROBE,
            tablebase=TABLEBASE,
        )
    finally:
        suite_runtime.restore_bound_tablebase(root, source_registration, pointer)
    required = {
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }
    if preflight is None or any(preflight.get(k) != v for k, v in required.items()):
        raise ValueError("seed551_runtime_contract_preflight_failed")
    preflight.update(
        {
            "exact_root_native_probe_sha256": protocol.sha256(root / PROBE),
            "exact_root_tablebase_sha256": protocol.sha256(TABLEBASE_SOURCE),
            "runtime_search_policy_sha256": protocol.sha256(
                artifacts / "T" / "search_policy.json"
            ),
        }
    )
    save_json(output / "runtime-policy-preflight.json", preflight)

    openings_dir = output / "batches"
    openings_dir.mkdir(exist_ok=True)
    for index, line in enumerate(raw_suite.splitlines(keepends=True)):
        (openings_dir / f"opening-{index:04d}.jsonl").write_bytes(line)

    execution_sources = {
        path: protocol.sha256(root / path)
        for path in (
            "ml/alphazero_lite/arena.py",
            "ml/alphazero_lite/build_opening_suite.py",
            "ml/alphazero_lite/classic_mcts.py",
            "ml/alphazero_lite/endgame_tablebase.py",
            "ml/alphazero_lite/eval_cache.py",
            "ml/alphazero_lite/evaluation_seed_contract.py",
            "ml/alphazero_lite/exact_root_decision.py",
            "ml/alphazero_lite/exact_root_policy_targets.py",
            "ml/alphazero_lite/export_artifact.py",
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
            "ml/alphazero_lite/seed482_fresh_confirmation.py",
            "ml/alphazero_lite/seed511_fixed_composition_screen.py",
            "ml/alphazero_lite/seed542_expanded_policy_strength_screen.py",
            "ml/alphazero_lite/seed551_arena_entrypoint.py",
            "ml/alphazero_lite/seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/run_seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/verify_seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/test_seed551_controlled_policy_strength.py",
            "ml/alphazero_lite/shadow_root_q.py",
            "ml/alphazero_lite/train.py",
            "ml/alphazero_lite/value_transforms.py",
        )
    }
    inputs = {
        relative: protocol.sha256(root / relative)
        for relative in (
            protocol.CONTROL_CHECKPOINT,
            protocol.TREATMENT_CHECKPOINT,
            protocol.INITIALIZER,
            "docs/data/seed511-fixed-composition-screen-v4/registration.json",
            "docs/data/seed511-fixed-composition-screen-v4/execution-freeze.json",
            "docs/data/seed511-fixed-composition-screen-v4/openings.jsonl",
            "docs/data/seed549-matched-selfplay/registration.json",
            "docs/data/seed549-matched-selfplay/shared-development-membership.jsonl",
            "docs/data/seed549-controlled-training/registration.json",
            "deploy/serving-release.json",
            "deploy/release-manifest.json",
            "model-artifact/current/search_policy.json",
            PROBE,
            "model-artifact/runtime/kalah_v1_21.kvtb",
        )
    }
    inputs.update(
        {
            str(path.relative_to(root)): protocol.sha256(path)
            for folder in (artifacts / "C", artifacts / "T", artifacts / "initializer")
            for path in sorted(folder.iterdir())
            if path.is_file()
        }
    )
    corpus_reg = json.loads((root / protocol.CORPUS / "registration.json").read_text())
    training_reg = json.loads(
        (root / protocol.TRAINING / "registration.json").read_text()
    )
    for source_map in (
        corpus_reg.get("input_sha256", {}),
        training_reg.get("input_sha256", {}),
    ):
        for relative, digest in source_map.items():
            actual = protocol.sha256(root / relative)
            if actual != digest:
                raise ValueError(
                    f"seed551_authenticated_population_input_mismatch:{relative}"
                )
            inputs[relative] = actual
    for source_map in (
        corpus_reg.get("source_sha256", {}),
        training_reg.get("source_sha256", {}),
        corpus_reg.get("runtime_asset_sha256", {}),
    ):
        for relative in source_map:
            inputs[relative] = protocol.sha256(root / relative)
    for folder in (root / protocol.CORPUS, root / protocol.TRAINING):
        for pattern in (
            "execution-correction*.json",
            "execution-freeze-correction*.json",
            "post-generation-verification-correction*.json",
            "review-correction*.json",
        ):
            for path in sorted(folder.glob(pattern)):
                inputs[str(path.relative_to(root))] = protocol.sha256(path)
    preparation_attempt_dir = root / (
        "docs/data/seed551-controlled-policy-strength-screen-preparation-attempt-0001"
    )
    preparation_failures: list[dict[str, Any]] = []
    if preparation_attempt_dir.exists():
        manifest_path = preparation_attempt_dir / "preparation-attempt-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "seed551_preparation_attempt_manifest_v1":
            raise ValueError("seed551_preparation_attempt_manifest_schema_invalid")
        for entry in manifest.get("files", []):
            path = preparation_attempt_dir / entry["path"]
            if (
                protocol.sha256(path) != entry.get("sha256")
                or path.stat().st_size != entry.get("size_bytes")
                or path.stat().st_mtime_ns != entry.get("mtime_ns")
            ):
                raise ValueError(
                    f"seed551_preparation_attempt_file_changed:{entry['path']}"
                )
            inputs[str(path.relative_to(root))] = protocol.sha256(path)
        inputs[str(manifest_path.relative_to(root))] = protocol.sha256(manifest_path)
        failure_path = preparation_attempt_dir / "PREPARATION-FAILURE.json"
        failure_record = json.loads(failure_path.read_text())
        preparation_failures.append(
            {
                "path": str(preparation_attempt_dir.relative_to(root)),
                "failure_sha256": protocol.sha256(failure_path),
                "manifest_sha256": protocol.sha256(manifest_path),
                "status": failure_record["status"],
                "failure_check": failure_record["failure_check"],
                "arena_games_started": failure_record["arena_games_started"],
            }
        )
    analysis = {
        "score": "T win=1; terminal draw=0.5; C win=0; average seat results per opening",
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": protocol.BOOTSTRAP_SEED,
        "replacement": True,
        "preserve_cluster_multiplicity": True,
        "quantiles": {"method": "linear", "lower": 0.05, "descriptive": [0.025, 0.975]},
        "decision": "advance only if all 2048 validate and lower bound strictly exceeds 0.50; otherwise do_not_advance_source_evaluator_branch",
        "advance_decision": "advance_to_separate_serving_P_screen",
        "non_advance_decision": "do_not_advance_source_evaluator_branch",
        "capped_games": "adjudicate using recorded current store-margin leader (tie=0.5), preserve and report as capped/nonterminal separately from terminal win/draw/loss counts",
    }
    protocol_record = {
        "games_per_opening": 2,
        "seats": [0, 1],
        "simulations_per_side": SIMULATIONS,
        "c_puct": 1.25,
        "workers": 1,
        "search_options": protocol.SEARCH_OPTIONS,
        "exact_root_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve": "disabled",
        "random_opening_plies": 0,
        "max_moves": 200,
        "seed_contract": protocol.SEED_CONTRACT,
        "opening_seed": "545 + opening_index",
    }
    registration = {
        "schema": "seed551_controlled_policy_strength_v1",
        "status": "prospective_frozen_before_candidate_inference_or_games",
        "execution_root": str(root),
        "claim": "policy-strength difference attributable to changing source self-play evaluator under frozen initializer value; not policy-target quality alone and not superiority to serving P",
        "exploratory_reused_seed511_suite": True,
        "seed461_continuation_overlap_limitation_unresolved": True,
        "suite_sha256": hashlib.sha256(raw_suite).hexdigest(),
        "suite_import_report_sha256": protocol.sha256(
            output / "suite-import-report.json"
        ),
        "suite_openings": protocol.OPENINGS,
        "overlap_summary_sha256": protocol.sha256(output / "overlap-summary.json"),
        "overlap_per_opening_sha256": protocol.sha256(
            output / "overlap-per-opening.jsonl"
        ),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(
            output / "runtime-authentication.json"
        ),
        "runtime_policy_preflight_sha256": protocol.sha256(
            output / "runtime-policy-preflight.json"
        ),
        "input_verification_sha256": protocol.sha256(
            output / "input-verification.json"
        ),
        "readme_sha256": protocol.sha256(output / "README.md"),
        "preserved_preparation_failures": preparation_failures,
        "analysis": analysis,
        "arena_protocol": protocol_record,
        "runtime": runtime_settings(),
        "durable_execution_command": launch_command(root, output),
        "execution_source_sha256": execution_sources,
        "authenticated_file_sha256": inputs,
    }
    save_json(output / "registration.json", registration)
    freeze = {
        "status": "frozen_before_real_arena_games",
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "suite_sha256": registration["suite_sha256"],
        "analysis": analysis,
        "arena_protocol": protocol_record,
        "execution_source_sha256": execution_sources,
        "authenticated_file_sha256": inputs,
    }
    save_json(output / "execution-freeze.json", freeze)
    commands = [
        build_command(root, output, index, freeze) for index in range(protocol.OPENINGS)
    ]
    (output / "command-preflight.jsonl").write_bytes(
        b"".join(canonical(row) + b"\n" for row in commands)
    )
    freeze["command_preflight_sha256"] = protocol.sha256(
        output / "command-preflight.jsonl"
    )
    save_json(output / "execution-freeze.json", freeze)
    verifier.verify_preflight(root, output)
    return {
        "status": "prepared_frozen",
        "openings": protocol.OPENINGS,
        "games": protocol.GAMES,
        "commands_parser_validated": len(commands),
        "registration_sha256": freeze["registration_sha256"],
        "freeze_sha256": protocol.sha256(output / "execution-freeze.json"),
        "overlap_counts": overlap["overlap_opening_counts"],
    }


def build_command(
    root: Path, output: Path, index: int, freeze: dict[str, Any]
) -> dict[str, Any]:
    batch = Path(protocol.OUT) / "batches"
    batch_files = output / "batches"
    artifacts = Path(protocol.OUT) / "runtime-artifacts"
    options = json.dumps(protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":"))
    argv = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.seed551_arena_entrypoint",
        "--challenger",
        str(artifacts / "T"),
        "--current",
        str(artifacts / "C"),
        "--challenger-policy-artifact",
        str(artifacts / "T"),
        "--challenger-value-artifact",
        str(artifacts / "initializer"),
        "--current-policy-artifact",
        str(artifacts / "C"),
        "--current-value-artifact",
        str(artifacts / "initializer"),
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
        protocol.SEED_CONTRACT,
        "--challenger-search-options-json",
        options,
        "--current-search-options-json",
        options,
        "--random-opening-plies",
        "0",
        "--max-moves",
        "200",
        "--exact-root-native-probe",
        PROBE,
        "--exact-root-tablebase",
        "model-artifact/runtime/kalah_v1_21.kvtb",
        "--seed",
        str(SEED + index),
        "--opening-prefixes-jsonl",
        str(batch / f"opening-{index:04d}.jsonl"),
        "--suite-sha256",
        protocol.sha256(batch_files / f"opening-{index:04d}.jsonl"),
        "--game-jsonl",
        str(batch / f"games-{index:04d}.jsonl"),
        "--out",
        str(batch / f"report-{index:04d}.json"),
    ]
    old_argv = sys.argv
    try:
        sys.argv = ["arena.py", *argv[3:]]
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
        or parsed.seed != SEED + index
        or parsed.random_opening_plies != 0
        or parsed.max_moves != 200
        or json.loads(parsed.challenger_search_options_json) != protocol.SEARCH_OPTIONS
        or json.loads(parsed.current_search_options_json) != protocol.SEARCH_OPTIONS
    ):
        raise ValueError(f"seed551_arena_parser_contract_mismatch:{index}")
    return {
        "opening_index": index,
        "argv": argv,
        "registration_sha256": freeze["registration_sha256"],
        "suite_sha256": freeze["suite_sha256"],
        "opening_sha256": protocol.sha256(batch_files / f"opening-{index:04d}.jsonl"),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime": runtime_settings(),
        "runtime_source_sha256": freeze["execution_source_sha256"],
    }


def launch_command(root: Path, output: Path) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--unit=seed551-controlled-policy-strength",
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
        "/usr/bin/env",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "BLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        sys.executable,
        "-m",
        "ml.alphazero_lite.run_seed551_controlled_policy_strength",
        "--root",
        str(root),
        "--out",
        str(output),
        "--execute",
    ]


def _tablebase_registration(root: Path) -> dict[str, Any]:
    registration = json.loads(
        (root / protocol.suite.SEED511 / "registration.json").read_text()
    )
    registration["runtime_assets"]["exact_root_tablebase"][
        "runtime_materialized_path"
    ] = str(TABLEBASE_SOURCE)
    return registration


def restore_interrupted_tablebase(root: Path) -> None:
    """Restore only the exact frozen LFS pointer if a prior run was interrupted."""
    registration = _tablebase_registration(root)
    asset = registration["runtime_assets"]["exact_root_tablebase"]
    actual = protocol.sha256(root / asset["artifact"])
    if actual == asset["source_pointer_sha256"]:
        return
    if actual != asset["sha256"]:
        raise ValueError("seed551_tablebase_neither_frozen_pointer_nor_materialization")
    pointer = subprocess.run(
        ["git", "show", f"HEAD:{asset['source_path']}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(pointer).hexdigest() != asset["source_pointer_sha256"]:
        raise ValueError("seed551_original_tablebase_pointer_hash_mismatch")
    temporary = root / (asset["artifact"] + ".seed551-restore")
    temporary.write_bytes(pointer)
    os.replace(temporary, root / asset["artifact"])


def recovery_action(output: Path, index: int, committed: set[int]) -> str:
    """Classify a pair without retrying or discarding prior outcome evidence."""
    if index in committed:
        return "skip"
    batch = output / "batches"
    games = batch / f"games-{index:04d}.jsonl"
    report = batch / f"report-{index:04d}.json"
    command = batch / f"command-{index:04d}.json"
    attempt = batch / f"attempt-{index:04d}.json"
    raw_exists = games.exists() or report.exists()
    if raw_exists:
        if not (games.is_file() and report.is_file() and command.is_file()):
            raise ValueError(f"seed551_unauthenticated_partial_pair_preserved:{index}")
        return "recover"
    archived_attempts = batch / "attempts" / f"opening-{index:04d}"
    if command.exists() or attempt.exists() or archived_attempts.exists():
        raise ValueError(
            f"seed551_prior_attempt_preserved_without_complete_raw_pair:{index}"
        )
    return "execute"


def execute(root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    restore_interrupted_tablebase(root)
    verifier.verify_preflight(root, output)
    registration = json.loads((output / "registration.json").read_text())
    assert_runtime_contract(registration["runtime"]["observed"])
    freeze = json.loads((output / "execution-freeze.json").read_text())
    ledger_path = output / "outcomes.jsonl"
    existing = _read_jsonl(ledger_path) if ledger_path.exists() else []
    source_opening_rows = (
        (root / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    committed: dict[int, list[dict[str, Any]]] = {}
    for row in existing:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    for index, records in committed.items():
        actual = verifier.validate_pair(root, output, index, preflight_verified=True)
        if (
            len(records) != 2
            or sorted(records, key=lambda row: row["game_position"]) != actual
        ):
            raise ValueError(f"seed551_committed_pair_corrupt:{index}")
    if len(existing) % 2:
        raise ValueError("seed551_ledger_partial_pair")
    for index in range(protocol.OPENINGS):
        action = recovery_action(output, index, set(committed))
        if action == "skip":
            continue
        paths = [
            output / "batches" / f"{kind}-{index:04d}.{suffix}"
            for kind, suffix in (
                ("games", "jsonl"),
                ("report", "json"),
                ("command", "json"),
                ("attempt", "json"),
            )
        ]
        games_path, report_path, command_path, attempt_path = paths
        opening_path = output / "batches" / f"opening-{index:04d}.jsonl"
        if opening_path.read_bytes() != source_opening_rows[index]:
            raise ValueError(f"seed551_indexed_opening_row_changed:{index}")
        if action == "recover":
            pair = verifier.validate_pair(root, output, index, preflight_verified=True)
            command_record = json.loads(command_path.read_text())
            attempt_archive = output / "batches" / "attempts" / f"opening-{index:04d}"
            attempt_archive.mkdir(parents=True, exist_ok=True)
            archived_attempt = attempt_archive / "attempt-00.json"
            if not archived_attempt.exists():
                save_json(
                    archived_attempt,
                    {
                        "opening_index": index,
                        "attempt_number": 0,
                        "argv": command_record["argv"],
                        "returncode": None,
                        "command_sha256": protocol.sha256(command_path),
                        "games_path_exists": True,
                        "report_path_exists": True,
                        "observed_runtime": command_record["runtime"]["observed"],
                        "recovered_without_attempt_completion": True,
                    },
                )
            if not attempt_path.exists():
                save_json(
                    attempt_path,
                    {
                        "opening_index": index,
                        "started": True,
                        "command_sha256": protocol.sha256(command_path),
                        "observed_runtime": command_record["runtime"]["observed"],
                        "recovered_without_attempt_completion": True,
                    },
                )
            recovery_path = output / "recovery" / f"recovery-{index:04d}.json"
            recovery_record = {
                "opening_index": index,
                "status": "complete_raw_pair_recovered_after_validation",
                "validated_game_records": 2,
                "attempt_sha256": protocol.sha256(attempt_path),
                "command_sha256": protocol.sha256(command_path),
                "games_sha256": protocol.sha256(games_path),
                "report_sha256": protocol.sha256(report_path),
                "recovery_reran_games": False,
            }
            if recovery_path.exists():
                if json.loads(recovery_path.read_text()) != recovery_record:
                    raise ValueError(f"seed551_recovery_receipt_conflict:{index}")
            else:
                save_json(recovery_path, recovery_record)
        else:
            command_record = _read_jsonl(output / "command-preflight.jsonl")[index]
            if command_record["opening_index"] != index:
                raise ValueError(f"seed551_preflight_index_mismatch:{index}")
            save_json(command_path, command_record)
            append_jsonl(output / "commands.jsonl", command_record)
            attempt_archive = output / "batches" / "attempts" / f"opening-{index:04d}"
            attempt_archive.mkdir(parents=True, exist_ok=True)
            prior_attempts = sorted(attempt_archive.glob("attempt-*.json"))
            attempt_archive_path = (
                attempt_archive / f"attempt-{len(prior_attempts):02d}.json"
            )
            attempt_record = {
                "opening_index": index,
                "attempt_number": len(prior_attempts),
                "argv": command_record["argv"],
                "started": True,
                "command_sha256": protocol.sha256(command_path),
                "games_path_exists": False,
                "report_path_exists": False,
                "observed_runtime": observed_runtime(),
            }
            save_json(attempt_archive_path, attempt_record)
            save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "command_sha256": protocol.sha256(command_path),
                    "observed_runtime": observed_runtime(),
                },
            )
            from ml.alphazero_lite import (
                seed511_fixed_composition_screen as suite_runtime,
            )

            source_registration = _tablebase_registration(root)
            tablebase_pointer = suite_runtime.materialize_bound_tablebase(
                root, source_registration
            )
            try:
                if protocol.sha256(
                    root / "model-artifact/runtime/kalah_v1_21.kvtb"
                ) != protocol.sha256(TABLEBASE_SOURCE):
                    raise ValueError("seed551_materialized_tablebase_hash_mismatch")
                result = subprocess.run(command_record["argv"], cwd=root, check=False)
            finally:
                suite_runtime.restore_bound_tablebase(
                    root, source_registration, tablebase_pointer
                )
            attempt_record.update(
                {
                    "returncode": result.returncode,
                    "games_path_exists": games_path.exists(),
                    "report_path_exists": report_path.exists(),
                    "observed_runtime": observed_runtime(),
                }
            )
            save_json(attempt_archive_path, attempt_record)
            save_json(
                attempt_path,
                {
                    "opening_index": index,
                    "started": True,
                    "returncode": result.returncode,
                    "command_sha256": protocol.sha256(command_path),
                    "observed_runtime": observed_runtime(),
                },
            )
            if result.returncode:
                raise RuntimeError(
                    f"seed551_arena_attempt_preserved:{index}:{result.returncode}"
                )
            pair = verifier.validate_pair(root, output, index, preflight_verified=True)
        if len(pair) != 2:
            raise ValueError(f"seed551_pair_validation_incomplete:{index}")
        command_record = json.loads(command_path.read_text())
        command_log_path = output / "commands.jsonl"
        command_log = _read_jsonl(command_log_path) if command_log_path.exists() else []
        index_commands = [
            row for row in command_log if int(row.get("opening_index", -1)) == index
        ]
        if not index_commands:
            append_jsonl(command_log_path, command_record)
        elif len(index_commands) != 1 or index_commands[0] != command_record:
            raise ValueError(f"seed551_command_log_conflict:{index}")
        completion_path = output / "batches" / f"completion-{index:04d}.json"
        completion = {
            "opening_index": index,
            "registration_sha256": freeze["registration_sha256"],
            "suite_sha256": freeze["suite_sha256"],
            "opening_sha256": protocol.sha256(
                output / "batches" / f"opening-{index:04d}.jsonl"
            ),
            "command_sha256": protocol.sha256(command_path),
            "games_sha256": protocol.sha256(games_path),
            "report_sha256": protocol.sha256(report_path),
        }
        if (
            completion_path.exists()
            and json.loads(completion_path.read_text()) != completion
        ):
            raise ValueError(f"seed551_completion_receipt_conflict:{index}")
        if not completion_path.exists():
            save_json(completion_path, completion)
        prior = _read_jsonl(ledger_path) if ledger_path.exists() else []
        if any(int(row["opening_index"]) == index for row in prior):
            raise ValueError(f"seed551_duplicate_ledger_commit:{index}")
        temporary = ledger_path.with_suffix(".jsonl.pending")
        temporary.write_bytes(
            b"".join(canonical(row) + b"\n" for row in [*prior, *pair])
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, ledger_path)
        committed[index] = pair
        print(
            json.dumps({"opening_index": index, "committed_pairs": len(committed)}),
            flush=True,
        )
    ledger = _read_jsonl(ledger_path)
    if len(ledger) != protocol.GAMES:
        return {"status": "incomplete", "validated_games": len(ledger)}
    results, per_opening = protocol.analyze(ledger)
    save_json(output / "results.json", results)
    (output / "per-opening-results.jsonl").write_bytes(
        b"".join(canonical(row) + b"\n" for row in per_opening)
    )
    return {"status": "complete", "validated_games": len(ledger), "results": results}


def record_independent_verification(
    root: Path, output: Path, tablebase_path: Path
) -> dict[str, Any]:
    """Run the independent CLI read-only, then publish a separate verification receipt."""
    root, output, tablebase_path = (
        root.resolve(),
        output.resolve(),
        tablebase_path.resolve(),
    )
    command = [
        sys.executable,
        "-m",
        "ml.alphazero_lite.verify_seed551_controlled_policy_strength",
        "--root",
        str(root),
        "--out",
        str(output),
        "--tablebase",
        str(tablebase_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    verification = json.loads(completed.stdout)
    if (
        verification.get("status") != "verified_complete"
        or verification.get("verified_games") != protocol.GAMES
        or verification.get("read_only_bytes_and_mtimes_unchanged") is not True
    ):
        raise ValueError("seed551_independent_verification_not_complete")
    evidence_hashes = {
        str(path.relative_to(output)): protocol.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "verification-receipt.json"
    }
    receipt = {
        "schema": "seed551_independent_verification_receipt_v1",
        "status": "independent_read_only_verified_complete",
        "command": command,
        "exit_status": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "verification_result": verification,
        "verifier_source_sha256": protocol.sha256(
            root / "ml/alphazero_lite/verify_seed551_controlled_policy_strength.py"
        ),
        "verified_games": verification["verified_games"],
        "verified_pairs": verification["verified_pairs"],
        "capped_nonterminal_games": verification["capped_nonterminal_games"],
        "evidence_sha256": evidence_hashes,
    }
    save_json(output / "verification-receipt.json", receipt)
    return {
        "status": receipt["status"],
        "receipt_sha256": protocol.sha256(output / "verification-receipt.json"),
        "verified_games": receipt["verified_games"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record-verification", action="store_true")
    parser.add_argument("--archive-failed-preparation", action="store_true")
    parser.add_argument("--failure-check", default=None)
    parser.add_argument("--input-verification", type=Path)
    parser.add_argument("--tablebase", type=Path, default=TABLEBASE_SOURCE)
    args = parser.parse_args()
    if args.archive_failed_preparation:
        if not args.failure_check:
            raise SystemExit("--archive-failed-preparation requires --failure-check")
        archived = archive_failed_preparation(args.root, args.out, args.failure_check)
        print(
            json.dumps(
                {"status": "preparation_failure_archived", "path": str(archived)}
            )
        )
        return
    if args.prepare:
        if args.input_verification is None:
            raise SystemExit("--prepare requires --input-verification")
        result = prepare(
            args.root,
            args.out,
            training_receipts=json.loads(args.input_verification.read_text()),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.execute:
        result = execute(args.root, args.out)
    elif args.record_verification:
        result = record_independent_verification(args.root, args.out, args.tablebase)
    else:
        raise SystemExit("choose --prepare or --execute")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
