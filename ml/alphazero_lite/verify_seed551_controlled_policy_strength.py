"""Independent read-only verifier for seed551 suite, arena pairs, and analysis."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import seed551_controlled_policy_strength as protocol

ROOT = protocol.ROOT
OUT = ROOT / protocol.OUT
TABLEBASE_SOURCE = Path(
    "/home/alex/Mancala/ai/.tmp/seed542-screen-runtime/kalah_v1_21.kvtb"
)
OPENINGS = 1024
GAMES = 2048
BOOTSTRAP_RESAMPLES = 20_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def replay_suite(root: Path) -> list[dict[str, Any]]:
    """Replay all indexed openings from absolute moves without importing outcomes."""
    suite_path = root / protocol.suite.SEED511 / "openings.jsonl"
    digest = sha(suite_path)
    if digest != protocol.suite.OPENING_SHA256:
        raise ValueError("seed551_source_suite_hash_mismatch")
    return _replay_suite_cached(str(root.resolve()), digest)


@lru_cache(maxsize=4)
def _replay_suite_cached(root_text: str, digest: str) -> list[dict[str, Any]]:
    root = Path(root_text)
    suite_path = root / protocol.suite.SEED511 / "openings.jsonl"
    if sha(suite_path) != digest:
        raise ValueError("seed551_source_suite_changed_during_replay")
    lines = suite_path.read_bytes().splitlines()
    if len(lines) != protocol.OPENINGS:
        raise ValueError("seed551_source_suite_count_mismatch")
    openings: list[dict[str, Any]] = []
    canonical: set[str] = set()
    float32: set[str] = set()
    for index, line in enumerate(lines):
        row = json.loads(line)
        absolute = row.get("source_prefix_moves_absolute")
        relative = row.get("prefix_moves")
        if (
            row.get("opening_index") != index
            or row.get("opening_contract") != "arena_player_relative_v2"
            or not isinstance(absolute, list)
            or not isinstance(relative, list)
            or len(absolute) != len(relative)
        ):
            raise ValueError(f"seed551_source_opening_shape:{index}")
        game = KalahGame.from_state(protocol.suite.INITIAL_STATE)
        for ply, (move, rel_move) in enumerate(zip(absolute, relative, strict=True)):
            if (
                game.over()
                or int(move) // 6 != game.current_player
                or int(move) - game.current_player * 6 != int(rel_move)
                or int(rel_move) not in game.possible_moves()
                or not game.move(int(move))
            ):
                raise ValueError(f"seed551_source_opening_illegal:{index}:{ply}")
        state = game.to_state()
        if state != row.get("state") or game.over():
            raise ValueError(f"seed551_source_opening_state_mismatch:{index}")
        identity = protocol.canonical_identity(state)
        f32 = protocol.production_float32_identity(state)
        if identity != row.get("state_hash") or identity in canonical or f32 in float32:
            raise ValueError(f"seed551_source_opening_identity_mismatch:{index}")
        canonical.add(identity)
        float32.add(f32)
        openings.append(row)
    return openings


def _observed_paths(root: Path, out: Path) -> set[Path]:
    paths = {path for path in out.rglob("*") if path.is_file()}
    registration_path = out / "registration.json"
    if registration_path.is_file():
        registration = json.loads(registration_path.read_text())
        paths.update(
            root / name for name in registration.get("execution_source_sha256", {})
        )
        paths.update(
            root / name for name in registration.get("authenticated_file_sha256", {})
        )
    paths.update(
        root / relative
        for relative in (
            protocol.CONTROL_CHECKPOINT,
            protocol.TREATMENT_CHECKPOINT,
            protocol.INITIALIZER,
            "docs/data/seed549-matched-selfplay/registration.json",
            "docs/data/seed549-controlled-training/registration.json",
        )
    )
    try:
        runtime_auth = json.loads((out / "runtime-authentication.json").read_text())
        paths.add(Path(runtime_auth["runtime_assets"]["materialized_tablebase_path"]))
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return paths


def verify_preflight(
    root: Path,
    out: Path,
    *,
    tablebase_path: Path | None = None,
    allow_relocated_copy: bool = False,
) -> dict[str, Any]:
    root, out = root.resolve(), out.resolve()
    registration = json.loads((out / "registration.json").read_text())
    freeze_path = out / "execution-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    if (
        registration.get("schema") != "seed551_controlled_policy_strength_v1"
        or registration.get("status")
        != "prospective_frozen_before_candidate_inference_or_games"
        or (
            registration.get("execution_root") != str(root) and not allow_relocated_copy
        )
        or freeze.get("status") != "frozen_before_real_arena_games"
        or freeze.get("registration_sha256") != sha(out / "registration.json")
        or freeze.get("suite_sha256") != protocol.suite.OPENING_SHA256
        or registration.get("suite_sha256") != protocol.suite.OPENING_SHA256
    ):
        raise ValueError("seed551_freeze_binding_mismatch")
    bound_outputs = {
        "openings.jsonl": registration.get("suite_sha256"),
        "suite-import-report.json": registration.get("suite_import_report_sha256"),
        "overlap-summary.json": registration.get("overlap_summary_sha256"),
        "overlap-per-opening.jsonl": registration.get("overlap_per_opening_sha256"),
        "runtime-authentication.json": registration.get(
            "runtime_authentication_sha256"
        ),
        "runtime-policy-preflight.json": registration.get(
            "runtime_policy_preflight_sha256"
        ),
        "input-verification.json": registration.get("input_verification_sha256"),
        "README.md": registration.get("readme_sha256"),
    }
    for name, expected in bound_outputs.items():
        if expected is None or sha(out / name) != expected:
            raise ValueError(f"seed551_frozen_output_binding_mismatch:{name}")
    if freeze.get("analysis") != registration.get("analysis") or freeze.get(
        "arena_protocol"
    ) != registration.get("arena_protocol"):
        raise ValueError("seed551_freeze_protocol_or_analysis_mismatch")
    runtime = registration.get("runtime", {})
    observed = runtime.get("observed", {})
    resource_limits = runtime.get("resource_limits", {})
    observed_limits = observed.get("cgroup_limits", {})
    if (
        runtime.get("device") != "cpu"
        or runtime.get("dtype") != "float32"
        or runtime.get("torch_threads") != 1
        or runtime.get("torch_interop_threads") != 1
        or observed.get("cpu_affinity") != [30]
        or observed.get("nice") != 19
        or observed.get("address_space_rlimit") != [17_179_869_184, 17_179_869_184]
        or observed.get("torch_threads") != 1
        or observed.get("torch_interop_threads") != 1
        or observed.get("environment")
        != {key: "1" for key in runtime.get("environment", {})}
        or observed_limits.get("cpu.max") != "25000 100000"
        or observed_limits.get("cpu.weight") != "1"
        or observed_limits.get("memory.max") != "4294967296"
        or observed_limits.get("cpuset.cpus.effective") not in {"30", None}
        or resource_limits
        != {
            "cpu_quota": "25%",
            "cpu_weight": "1",
            "memory_max": "4294967296",
            "allowed_cpus": "30",
            "nice": 19,
            "address_space_limit": "17179869184",
            "workers": 1,
        }
    ):
        raise ValueError("seed551_registered_runtime_contract_invalid")
    launch = registration.get("durable_execution_command", [])
    required_launch_tokens = (
        "systemd-run",
        "CPUQuota=25%",
        "CPUWeight=1",
        "MemoryMax=4294967296",
        "AllowedCPUs=30",
        "Nice=19",
        "LimitAS=17179869184",
        "taskset",
        "-c",
        "30",
        "OMP_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    )
    launch_text = "\0".join(launch)
    if any(token not in launch_text for token in required_launch_tokens):
        raise ValueError("seed551_durable_launch_resource_contract_invalid")
    proof = json.loads((out / "input-verification.json").read_text())
    training_output = proof.get("training", {}).get("output", proof.get("training", {}))
    corpus_output = proof.get("corpus", {}).get("output", proof.get("corpus", {}))
    if (
        proof.get("training", {}).get("status") != "verified"
        or proof.get("corpus", {}).get("status") != "verified"
        or training_output.get("bytes_and_mtimes_unchanged") is not True
        or corpus_output.get("evidence_preserved") is not True
    ):
        raise ValueError("seed551_input_verification_chain_incomplete")
    checks = proof.get("pre_execution_checks", {})
    regressions = checks.get("arena_runtime_regressions", {})
    quality = checks.get("python_quality_gate", {})
    registered_runtime = checks.get("registered_runtime_contract", {})
    if (
        checks.get("status") != "passed"
        or checks.get("focused_tests", {}).get("status") != "passed"
        or checks.get("focused_tests", {}).get("passed") < 1
        or regressions.get("status") != "passed"
        or regressions.get("arena_tests") < 1
        or regressions.get("runtime_search_policy_tests") < 1
        or regressions.get("selfplay_evaluator_pilot_tests") < 1
        or regressions.get("seed542_screen_recovery_and_composition_tests") < 1
        or quality.get("status") != "passed"
        or registered_runtime.get("exit_status") != 0
        or registered_runtime.get("status") != "passed"
        or registered_runtime.get("cpu_quota") != "25%"
        or registered_runtime.get("cpu_weight") != "1"
        or registered_runtime.get("memory_max") != "4294967296"
        or registered_runtime.get("allowed_cpus") != "30"
        or registered_runtime.get("cpu_affinity") != [30]
        or registered_runtime.get("nice") != 19
        or registered_runtime.get("address_space_rlimit")
        != [17_179_869_184, 17_179_869_184]
        or registered_runtime.get("torch_threads") != 1
        or registered_runtime.get("torch_interop_threads") != 1
    ):
        raise ValueError("seed551_pre_execution_gate_receipt_invalid")
    regression_failures = [
        *checks.get("preserved_failures", []),
        *checks.get("historical_verifier_failures", []),
    ]
    if not any(
        "verify_native_runtime_asset_identity_mismatch" in row.get("failing_checks", [])
        for row in regression_failures
    ):
        raise ValueError("seed551_regression_materialization_failure_not_preserved")
    corpus_registration_path = root / protocol.CORPUS / "registration.json"
    training_registration_path = root / protocol.TRAINING / "registration.json"
    corpus_registration = json.loads(corpus_registration_path.read_text())
    training_registration = json.loads(training_registration_path.read_text())
    if proof.get("authenticated_registrations") != {
        "seed549_corpus": sha(corpus_registration_path),
        "seed549_training": sha(training_registration_path),
    }:
        raise ValueError("seed551_input_registration_identity_mismatch")
    if proof.get("checkpoint_sha256") != {
        "C_update108": protocol.CONTROL_SHA256,
        "T_update108": protocol.TREATMENT_SHA256,
    }:
        raise ValueError("seed551_input_checkpoint_identity_mismatch")
    for relative, expected in proof.get("verified_source_sha256", {}).items():
        if sha(root / relative) != expected:
            raise ValueError(f"seed551_corrected_verifier_source_mismatch:{relative}")
    failures = proof.get("preserved_failures", [])
    if not any(
        row.get("failing_check") == "resource_affinity_or_nice_mismatch"
        and row.get("exit_status") == 1
        for row in failures
    ):
        raise ValueError("seed551_initial_training_affinity_failure_not_preserved")
    training_proof = proof["training"]
    training_argv = training_proof.get("command", [])
    training_runtime = training_proof.get("runtime", {})
    training_command_text = "\0".join(training_argv)
    if (
        training_proof.get("verifier") != "ml.alphazero_lite.verify_seed549_training"
        or training_proof.get("exit_status") != 0
        or training_argv[training_argv.index("-m") + 1]
        != "ml.alphazero_lite.verify_seed549_training"
        or Path(training_argv[training_argv.index("--root") + 1])
        != Path(registration["execution_root"])
        or training_argv[training_argv.index("--out") + 1]
        != str(Path(registration["execution_root"]) / protocol.TRAINING)
        or training_runtime.get("python")
        != training_registration.get("runtime", {}).get("python")
        or training_runtime.get("numpy")
        != training_registration["runtime"].get("numpy")
        or training_runtime.get("torch")
        != training_registration["runtime"].get("torch")
        or training_runtime.get("interpreter")
        != training_argv[training_argv.index("-m") - 1]
        or training_runtime.get("cpu_quota") != "25%"
        or training_runtime.get("cpu_weight") != "1"
        or training_runtime.get("memory_max") != "4294967296"
        or training_runtime.get("allowed_cpus") != "30"
        or training_runtime.get("cpu_affinity") != [30]
        or training_runtime.get("nice") != 19
        or training_runtime.get("address_space_limit") != "17179869184"
        or training_runtime.get("numerical_library_threads") != 1
        or any(
            token not in training_command_text
            for token in (
                "CPUQuota=25%",
                "CPUWeight=1",
                "MemoryMax=4294967296",
                "AllowedCPUs=30",
                "nice",
                "--as=17179869184",
                "taskset",
                "OMP_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
            )
        )
    ):
        raise ValueError("seed551_corrected_training_verifier_command_mismatch")
    corpus_proof = proof["corpus"]
    corpus_argv = corpus_proof.get("command", [])
    corpus_runtime = corpus_proof.get("runtime", {})
    corpus_command_text = "\0".join(corpus_argv)
    if (
        corpus_proof.get("verifier") != "ml.alphazero_lite.seed549_corpus verify"
        or corpus_proof.get("exit_status") != 0
        or "seed549_corpus as corpus" not in corpus_command_text
        or "corpus.verify(root)" not in corpus_command_text
        or not any(
            str(Path(registration["execution_root"])) in arg for arg in corpus_argv
        )
        or corpus_runtime.get("python")
        != corpus_registration.get("runtime", {}).get("python")
        or corpus_runtime.get("numpy") != corpus_registration["runtime"].get("numpy")
        or corpus_runtime.get("torch")
        != corpus_registration.get("runtime", {}).get("torch")
        or corpus_runtime.get("interpreter")
        != corpus_registration["runtime"].get("interpreter")
        or corpus_runtime.get("cpu_quota") != "25%"
        or corpus_runtime.get("cpu_weight") != "1"
        or corpus_runtime.get("memory_max") != "4294967296"
        or corpus_runtime.get("allowed_cpus") != "30"
        or corpus_runtime.get("cpu_affinity") != [30]
        or corpus_runtime.get("nice") != 19
        or corpus_runtime.get("address_space_limit") != "17179869184"
        or corpus_runtime.get("numerical_library_threads") != 1
        or any(
            token not in corpus_command_text
            for token in (
                "CPUQuota=25%",
                "CPUWeight=1",
                "MemoryMax=4294967296",
                "AllowedCPUs=30",
                "nice",
                "--as=17179869184",
                "taskset",
                "OMP_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
            )
        )
    ):
        raise ValueError("seed551_corrected_corpus_verifier_command_mismatch")
    _verify_execution_source_corrections(root, out, freeze)
    for relative, expected in freeze["authenticated_file_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"seed551_authenticated_input_changed:{relative}")
    for failed in registration.get("preserved_preparation_failures", []):
        directory = root / failed["path"]
        failure_path = directory / "PREPARATION-FAILURE.json"
        manifest_path = directory / "preparation-attempt-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        failure = json.loads(failure_path.read_text())
        if (
            sha(failure_path) != failed.get("failure_sha256")
            or sha(manifest_path) != failed.get("manifest_sha256")
            or failure.get("status") != "pre_game_preparation_failed_preserved"
            or failure.get("arena_games_started") is not False
        ):
            raise ValueError("seed551_preserved_preparation_failure_binding_mismatch")
        for entry in manifest.get("files", []):
            path = directory / entry["path"]
            if (
                sha(path) != entry.get("sha256")
                or path.stat().st_size != entry.get("size_bytes")
                or path.stat().st_mtime_ns != entry.get("mtime_ns")
            ):
                raise ValueError(
                    f"seed551_preserved_preparation_file_mismatch:{entry['path']}"
                )
    if sha(out / "command-preflight.jsonl") != freeze.get("command_preflight_sha256"):
        raise ValueError("seed551_preflight_command_file_changed")
    assignments = registration.get("component_assignments")
    if assignments != protocol.ASSIGNMENTS:
        raise ValueError("seed551_component_assignment_registration_mismatch")
    for name, expected in freeze["authenticated_file_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError(f"seed551_input_hash_changed:{name}")
    auth = json.loads((out / "runtime-authentication.json").read_text())
    materialized_tablebase = tablebase_path or Path(
        auth["runtime_assets"]["materialized_tablebase_path"]
    )
    if sha(materialized_tablebase) != auth["runtime_assets"]["tablebase_sha256"]:
        raise ValueError("seed551_materialized_tablebase_identity_mismatch")
    if auth.get("component_assignments") != assignments:
        raise ValueError("seed551_runtime_component_assignment_mismatch")
    for artifact, expected_checkpoint in (
        ("C", protocol.CONTROL_SHA256),
        ("T", protocol.TREATMENT_SHA256),
        ("initializer", protocol.sha256(root / protocol.INITIALIZER)),
    ):
        directory = out / "runtime-artifacts" / artifact
        if sha(directory / "model.npz") != expected_checkpoint:
            raise ValueError(f"seed551_exported_model_checkpoint_mismatch:{artifact}")
        recorded = auth.get("components", {}).get(artifact, {})
        if (
            recorded.get("checkpoint_sha256") != expected_checkpoint
            or sha(root / recorded.get("checkpoint_path", "")) != expected_checkpoint
        ):
            raise ValueError(f"seed551_checkpoint_binding_mismatch:{artifact}")
        for filename in (
            "model.npz",
            "weights.json",
            "metadata.json",
            "search_policy.json",
        ):
            if recorded.get("files", {}).get(filename) != sha(directory / filename):
                raise ValueError(
                    f"seed551_runtime_artifact_binding_mismatch:{artifact}:{filename}"
                )
    serving_policy = root / "model-artifact/current/search_policy.json"
    if any(
        sha(out / "runtime-artifacts" / artifact / "search_policy.json")
        != sha(serving_policy)
        for artifact in ("C", "T", "initializer")
    ):
        raise ValueError("seed551_search_policy_sidecar_mismatch")
    preflight = json.loads((out / "runtime-policy-preflight.json").read_text())
    required_preflight = {
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "exact_root_solve_threshold": 16,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "exact_root_native_probe_sha256": sha(
            root / "model-artifact/runtime/kalah_v1_tablebase"
        ),
        "exact_root_tablebase_sha256": sha(materialized_tablebase),
        "runtime_search_policy_sha256": sha(
            out / "runtime-artifacts/T/search_policy.json"
        ),
    }
    if (
        any(preflight.get(key) != value for key, value in required_preflight.items())
        or preflight.get("challenger_runtime_policy_inherited") is not False
    ):
        raise ValueError("seed551_full_runtime_policy_contract_mismatch")
    openings = replay_suite(root)
    out_lines = (out / "openings.jsonl").read_bytes().splitlines(keepends=True)
    source_lines = (
        (root / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    if out_lines != source_lines:
        raise ValueError("seed551_copied_suite_bytes_changed")
    commands = jsonl(out / "command-preflight.jsonl")
    source_lines = (
        (root / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    if len(commands) != protocol.OPENINGS:
        raise ValueError("seed551_command_preflight_count_mismatch")
    for index, record in enumerate(commands):
        argv = record.get("argv", [])
        expected = {
            "opening_index": index,
            "registration_sha256": freeze["registration_sha256"],
            "suite_sha256": freeze["suite_sha256"],
            "opening_sha256": sha(out / "batches" / f"opening-{index:04d}.jsonl"),
            "component_assignments": protocol.ASSIGNMENTS,
            "runtime_source_sha256": freeze["execution_source_sha256"],
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"seed551_command_binding_mismatch:{index}")
        if (
            out / "batches" / f"opening-{index:04d}.jsonl"
        ).read_bytes() != source_lines[index]:
            raise ValueError(f"seed551_indexed_opening_bytes_mismatch:{index}")
        if record.get("runtime") != registration.get("runtime"):
            raise ValueError(f"seed551_command_runtime_binding_mismatch:{index}")
        if argv[:3] != [
            registration["runtime"]["executable"],
            "-m",
            "ml.alphazero_lite.seed551_arena_entrypoint",
        ]:
            raise ValueError(f"seed551_command_entrypoint_mismatch:{index}")
        pairs = {
            "--challenger": Path(protocol.OUT) / "runtime-artifacts/T",
            "--current": Path(protocol.OUT) / "runtime-artifacts/C",
            "--challenger-policy-artifact": Path(protocol.OUT) / "runtime-artifacts/T",
            "--challenger-value-artifact": Path(protocol.OUT)
            / "runtime-artifacts/initializer",
            "--current-policy-artifact": Path(protocol.OUT) / "runtime-artifacts/C",
            "--current-value-artifact": Path(protocol.OUT)
            / "runtime-artifacts/initializer",
        }
        for option, path in pairs.items():
            try:
                actual_path = Path(argv[argv.index(option) + 1])
                actual = (
                    (root / actual_path).resolve()
                    if not actual_path.is_absolute()
                    else actual_path.resolve()
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"seed551_command_missing_option:{index}:{option}"
                ) from exc
            if actual != _rooted_path(root, path):
                raise ValueError(f"seed551_command_component_mismatch:{index}:{option}")
        scalar_contract = {
            "--games": "2",
            "--games-per-opening": "2",
            "--workers": "1",
            "--challenger-simulations": "128",
            "--current-simulations": "128",
            "--c-puct": "1.25",
            "--seed-contract": protocol.SEED_CONTRACT,
            "--random-opening-plies": "0",
            "--max-moves": "200",
            "--seed": str(545 + index),
            "--opening-prefixes-jsonl": str(
                Path(protocol.OUT) / "batches" / f"opening-{index:04d}.jsonl"
            ),
            "--suite-sha256": sha(out / "batches" / f"opening-{index:04d}.jsonl"),
            "--game-jsonl": str(
                Path(protocol.OUT) / "batches" / f"games-{index:04d}.jsonl"
            ),
            "--out": str(Path(protocol.OUT) / "batches" / f"report-{index:04d}.json"),
            "--exact-root-native-probe": "model-artifact/runtime/kalah_v1_tablebase",
            "--exact-root-tablebase": "model-artifact/runtime/kalah_v1_21.kvtb",
        }
        for option, value in scalar_contract.items():
            if _option(argv, option) != value:
                raise ValueError(f"seed551_command_scalar_mismatch:{index}:{option}")
        for option in (
            "--challenger-search-options-json",
            "--current-search-options-json",
        ):
            if json.loads(_option(argv, option)) != protocol.SEARCH_OPTIONS:
                raise ValueError(f"seed551_command_search_options_mismatch:{index}")
    return {
        "status": "preflight_verified",
        "openings": len(openings),
        "commands": len(commands),
    }


def _option(argv: list[str], option: str) -> str:
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"seed551_command_option_missing:{option}") from exc


def _rooted_path(root: Path, value: str | Path) -> Path:
    """Resolve command paths against the copied checkout, not verifier CWD."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_execution_source_corrections(
    root: Path, out: Path, freeze: dict[str, Any]
) -> None:
    expected = dict(freeze["execution_source_sha256"])
    registration_sha = freeze["registration_sha256"]
    parent_correction_sha: str | None = None
    parent_freeze_sha: str | None = None
    number = 1
    while True:
        suffix = "" if number == 1 else f"-{number:02d}"
        correction_path = out / f"execution-correction{suffix}.json"
        freeze_correction_path = out / f"execution-freeze-correction{suffix}.json"
        if not correction_path.exists() and not freeze_correction_path.exists():
            break
        if not correction_path.is_file() or not freeze_correction_path.is_file():
            raise ValueError("seed551_execution_source_correction_pair_incomplete")
        correction = json.loads(correction_path.read_text())
        freeze_correction = json.loads(freeze_correction_path.read_text())
        if (
            correction.get("schema") != "seed551_execution_source_correction_v1"
            or correction.get("registration_sha256") != registration_sha
            or (
                parent_correction_sha is not None
                and correction.get("parent_correction_sha256") != parent_correction_sha
            )
            or freeze_correction.get("registration_sha256") != registration_sha
            or freeze_correction.get("correction_sha256") != sha(correction_path)
            or (
                parent_freeze_sha is not None
                and freeze_correction.get("parent_freeze_correction_sha256")
                != parent_freeze_sha
            )
        ):
            raise ValueError("seed551_execution_source_correction_binding_mismatch")
        for relative, binding in correction.get("source_corrections", {}).items():
            if (
                relative not in expected
                or binding.get("previous_sha256") != expected[relative]
            ):
                raise ValueError(
                    f"seed551_execution_source_correction_parent_mismatch:{relative}"
                )
            expected[relative] = binding.get("corrected_sha256")
        parent_correction_sha = sha(correction_path)
        parent_freeze_sha = sha(freeze_correction_path)
        number += 1
    for relative, digest in expected.items():
        if sha(root / relative) != digest:
            raise ValueError(f"seed551_frozen_source_changed:{relative}")


def _game_outcome(
    opening: dict[str, Any], record: dict[str, Any]
) -> tuple[str, int, bool]:
    seat = int(record["challenger_player"])
    if seat not in (0, 1):
        raise ValueError("seed551_game_seat_invalid")
    game = KalahGame.from_state(protocol.suite.INITIAL_STATE)
    for move, rel in zip(
        opening["source_prefix_moves_absolute"], opening["prefix_moves"], strict=True
    ):
        if (
            game.over()
            or int(move) // 6 != game.current_player
            or int(move) - game.current_player * 6 != int(rel)
            or int(rel) not in game.possible_moves()
            or not game.move(int(move))
        ):
            raise ValueError("seed551_pair_opening_prefix_invalid")
    if game.to_state() != opening["state"]:
        raise ValueError("seed551_pair_opening_root_mismatch")
    if (
        record.get("opening_prefix_moves") != opening["prefix_moves"]
        or record.get("opening_state_hash") != opening["state_hash"]
        or record.get("opening_applied_prefix_length") != len(opening["prefix_moves"])
        or record.get("opening_contract") != "arena_player_relative_v2"
    ):
        raise ValueError("seed551_pair_suite_row_binding_mismatch")
    trajectory = [
        int(move) for move in str(record.get("trajectory", "")).split(",") if move
    ]
    if len(trajectory) != int(record.get("game_length", -1)) or len(trajectory) > 200:
        raise ValueError("seed551_game_ply_count_mismatch")
    for ply, absolute in enumerate(trajectory):
        if (
            game.over()
            or absolute // 6 != game.current_player
            or absolute - game.current_player * 6 not in game.possible_moves()
            or not game.move(absolute)
        ):
            raise ValueError(f"seed551_game_illegal_action:{ply}")
    capped = not game.over()
    if capped and len(trajectory) != 200:
        raise ValueError("seed551_game_nonterminal_before_move_cap")
    margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
    if record.get("margin") != margin:
        raise ValueError("seed551_game_margin_mismatch")
    winner = "T" if margin > 0 else "C" if margin < 0 else "draw"
    expected = {"T": "challenger", "C": "current", "draw": "draw"}[winner]
    if record.get("winner") != expected:
        raise ValueError("seed551_game_winner_perspective_mismatch")
    return winner, margin, capped


def _validate_arena_local_opening_index(game_record: dict[str, Any]) -> None:
    """Single-row arena suite files are locally indexed at zero per pair."""
    if game_record.get("opening_index") != 0:
        raise ValueError("seed551_game_arena_local_opening_index_mismatch")


def validate_pair(
    root: Path,
    out: Path,
    index: int,
    *,
    preflight_verified: bool = False,
    tablebase_path: Path | None = None,
    allow_relocated_copy: bool = False,
    require_completion_receipt: bool = False,
) -> list[dict[str, Any]]:
    root, out = root.resolve(), out.resolve()
    if not preflight_verified:
        verify_preflight(
            root,
            out,
            tablebase_path=tablebase_path,
            allow_relocated_copy=allow_relocated_copy,
        )
    openings = replay_suite(root)
    game_path = out / "batches" / f"games-{index:04d}.jsonl"
    report_path = out / "batches" / f"report-{index:04d}.json"
    command_path = out / "batches" / f"command-{index:04d}.json"
    if not (game_path.is_file() and report_path.is_file() and command_path.is_file()):
        raise ValueError(f"seed551_pair_evidence_missing:{index}")
    command = json.loads(command_path.read_text())
    frozen = jsonl(out / "command-preflight.jsonl")[index]
    if command != frozen:
        raise ValueError(f"seed551_pair_command_not_frozen:{index}")
    attempt_path = out / "batches" / f"attempt-{index:04d}.json"
    if attempt_path.is_file():
        attempt = json.loads(attempt_path.read_text())
        runtime = json.loads((out / "registration.json").read_text())["runtime"]
        if (
            attempt.get("opening_index") != index
            or attempt.get("started") is not True
            or attempt.get("command_sha256") != sha(command_path)
            or attempt.get("observed_runtime") != runtime.get("observed")
        ):
            raise ValueError(f"seed551_pair_attempt_runtime_or_exit_mismatch:{index}")
    validation_failure_path = out / "batches" / f"VALIDATION-FAILURE-{index:04d}.json"
    if validation_failure_path.is_file():
        failure = json.loads(validation_failure_path.read_text())
        archived_attempt = (
            out / "batches" / "attempts" / f"opening-{index:04d}" / "attempt-00.json"
        )
        if (
            failure.get("schema") != "seed551_pair_validation_failure_v1"
            or failure.get("status") != "complete_raw_pair_preserved_validation_failed"
            or failure.get("opening_index") != index
            or failure.get("arena_command_exit_status") != 0
            or failure.get("games_started") is not True
            or failure.get("games_sha256") != sha(game_path)
            or failure.get("report_sha256") != sha(report_path)
            or failure.get("command_sha256") != sha(command_path)
            or failure.get("attempt_sha256") != sha(attempt_path)
            or failure.get("attempt_archive_sha256") != sha(archived_attempt)
        ):
            raise ValueError(f"seed551_validation_failure_record_mismatch:{index}")
    games = jsonl(game_path)
    if len(games) != 2:
        raise ValueError(f"seed551_pair_game_count:{index}")
    source_lines = (
        (root / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)
    )
    opening_path = out / "batches" / f"opening-{index:04d}.jsonl"
    if opening_path.read_bytes() != source_lines[index]:
        raise ValueError(f"seed551_pair_opening_file_mismatch:{index}")
    report = json.loads(report_path.read_text())
    outcomes: list[dict[str, Any]] = []
    for position, game_record in enumerate(games):
        _validate_arena_local_opening_index(game_record)
        if (
            game_record.get("game_index") != position
            or game_record.get("game_within_opening") != position
            or game_record.get("challenger_player") != position
        ):
            raise ValueError(f"seed551_pair_seat_order_mismatch:{index}:{position}")
        outcome, margin, capped = _game_outcome(openings[index], game_record)
        outcomes.append(
            {
                "opening_index": index,
                "game_position": position,
                "challenger_player": position,
                "outcome": outcome,
                "margin": margin,
                "capped_nonterminal": capped,
                "game_sha256": sha(game_path),
                "report_sha256": sha(report_path),
                "command_sha256": sha(command_path),
            }
        )
    if (
        report.get("schema") != "arena_v1"
        or report.get("games_played") != 2
        or report.get("wins") != sum(row["winner"] == "challenger" for row in games)
        or report.get("losses") != sum(row["winner"] == "current" for row in games)
        or report.get("draws") != sum(row["winner"] == "draw" for row in games)
    ):
        raise ValueError(f"seed551_pair_report_accounting_mismatch:{index}")
    completion_path = out / "batches" / f"completion-{index:04d}.json"
    if require_completion_receipt and not completion_path.is_file():
        raise ValueError(f"seed551_pair_completion_receipt_missing:{index}")
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text())
        freeze = json.loads((out / "execution-freeze.json").read_text())
        expected_completion = {
            "opening_index": index,
            "registration_sha256": freeze["registration_sha256"],
            "suite_sha256": freeze["suite_sha256"],
            "opening_sha256": sha(out / "batches" / f"opening-{index:04d}.jsonl"),
            "command_sha256": sha(command_path),
            "games_sha256": sha(game_path),
            "report_sha256": sha(report_path),
        }
        if completion != expected_completion:
            raise ValueError(f"seed551_pair_completion_receipt_mismatch:{index}")
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    if (
        profile.get("simulations") != 128
        or profile.get("c_puct") != 1.25
        or profile.get("challenger_simulations") != 128
        or profile.get("current_simulations") != 128
        or profile.get("exact_root_solve_enabled") is not True
        or profile.get("exact_root_solve_threshold") != 16
        or profile.get("exact_root_solver") != "native_kvtb_root_action_probe_v1"
        or profile.get("exact_root_objective") != "final_score_margin"
        or profile.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or profile.get("exact_leaf_solve_mode") != "disabled"
        or profile.get("search_options") != protocol.SEARCH_OPTIONS
        or notes.get("workers_requested") != 1
        or notes.get("workers_used") != 1
        or notes.get("random_opening_plies") != 0
        or notes.get("seed_contract") != protocol.SEED_CONTRACT
        or notes.get("base_seed") != 545 + index
        or notes.get("runtime_search_policy_sha256")
        != sha(out / "runtime-artifacts/T/search_policy.json")
    ):
        raise ValueError(f"seed551_pair_runtime_report_mismatch:{index}")
    return outcomes


def reconstruct_analysis(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independently rebuild the registered bootstrap without publisher code."""
    if len(ledger) != GAMES:
        raise ValueError("seed551_independent_analysis_requires_2048_rows")
    points = {"T": 1.0, "draw": 0.5, "C": 0.0}
    grouped: dict[int, dict[int, dict[str, Any]]] = {
        index: {} for index in range(OPENINGS)
    }
    wins = draws = losses = capped = 0
    capped_results = {"T": 0, "draw": 0, "C": 0}
    for record in ledger:
        opening = int(record["opening_index"])
        seat = int(record["game_position"])
        if opening not in grouped or seat not in (0, 1) or seat in grouped[opening]:
            raise ValueError("seed551_independent_duplicate_ledger_coordinate")
        outcome = record["outcome"]
        if outcome not in points:
            raise ValueError("seed551_independent_unknown_outcome")
        is_capped = record.get("capped_nonterminal") is True
        capped += is_capped
        if is_capped:
            capped_results[outcome] += 1
        elif outcome == "T":
            wins += 1
        elif outcome == "draw":
            draws += 1
        else:
            losses += 1
        grouped[opening][seat] = record
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("seed551_independent_incomplete_opening_cluster")
    clusters = np.asarray(
        [
            (
                points[grouped[index][0]["outcome"]]
                + points[grouped[index][1]["outcome"]]
            )
            / 2.0
            for index in range(OPENINGS)
        ],
        dtype=np.float64,
    )
    random = np.random.default_rng(545)
    indexes = random.integers(0, OPENINGS, size=(BOOTSTRAP_RESAMPLES, OPENINGS))
    bootstrap_means = clusters[indexes].mean(axis=1)
    score = float(clusters.mean())
    lower = float(np.quantile(bootstrap_means, 0.05, method="linear"))
    advance = lower > 0.50
    result = {
        "status": "complete",
        "games": GAMES,
        "opening_pairs": OPENINGS,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "capped_nonterminal_results_by_current_store_margin": capped_results,
        "capped_points_policy": "use the arena's current store-margin winner at ply 200, ties 0.5, and report capped results separately from terminal outcomes",
        "capped_nonterminal_games": int(capped),
        "score": score,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": 545,
        "replacement": True,
        "preserved_cluster_multiplicity": True,
        "quantile_method": "linear",
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(bootstrap_means, 0.025, method="linear")),
            float(np.quantile(bootstrap_means, 0.975, method="linear")),
        ],
        "decision": "advance_to_separate_serving_P_screen"
        if advance
        else "do_not_advance_source_evaluator_branch",
        "strength_decision_valid": True,
        "non_advancement_does_not_establish_universal_source_evaluator_inferiority": True,
        "promotion_or_deployment_authorized": False,
        "fresh_holdout": False,
        "seed461_continuation_overlap_unresolved": True,
    }
    per_opening = [
        {
            "opening_index": index,
            "seat0_T_score": points[grouped[index][0]["outcome"]],
            "seat1_T_score": points[grouped[index][1]["outcome"]],
            "opening_cluster_score": float(clusters[index]),
            "seat0_capped_nonterminal": bool(grouped[index][0]["capped_nonterminal"]),
            "seat1_capped_nonterminal": bool(grouped[index][1]["capped_nonterminal"]),
            "capped_nonterminal_games": sum(
                grouped[index][seat]["capped_nonterminal"] is True for seat in (0, 1)
            ),
        }
        for index in range(OPENINGS)
    ]
    return result, per_opening


def verify(
    root: Path = ROOT,
    out: Path = OUT,
    *,
    tablebase_path: Path | None = None,
    allow_relocated_copy: bool = False,
) -> dict[str, Any]:
    root, out = root.resolve(), out.resolve()
    observed = _observed_paths(root, out)
    if tablebase_path is not None:
        observed.add(tablebase_path)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
        for path in observed
        if path.is_file()
    }
    try:
        preflight = verify_preflight(
            root,
            out,
            tablebase_path=tablebase_path,
            allow_relocated_copy=allow_relocated_copy,
        )
        ledger = (
            jsonl(out / "outcomes.jsonl") if (out / "outcomes.jsonl").exists() else []
        )
        verified: list[dict[str, Any]] = []
        indexes = sorted({int(row["opening_index"]) for row in ledger})
        for index in indexes:
            pair = validate_pair(
                root,
                out,
                index,
                preflight_verified=True,
                tablebase_path=tablebase_path,
                allow_relocated_copy=allow_relocated_copy,
                require_completion_receipt=True,
            )
            saved = sorted(
                (row for row in ledger if row["opening_index"] == index),
                key=lambda r: r["game_position"],
            )
            if pair != saved:
                raise ValueError(f"seed551_ledger_pair_reconstruction_mismatch:{index}")
            verified.extend(pair)
        if verified != ledger:
            raise ValueError("seed551_ledger_order_or_accounting_mismatch")
        result: dict[str, Any] = {
            "status": "incomplete"
            if len(ledger) != protocol.GAMES
            else "verified_complete",
            "verified_games": len(ledger),
            "verified_pairs": len(indexes),
            "capped_nonterminal_games": sum(
                row["capped_nonterminal"] for row in ledger
            ),
            "preflight": preflight,
            "read_only_bytes_and_mtimes_unchanged": True,
        }
        if len(ledger) == protocol.GAMES:
            command_log = jsonl(out / "commands.jsonl")
            preflight_commands = jsonl(out / "command-preflight.jsonl")
            if command_log != preflight_commands:
                raise ValueError("seed551_command_ledger_mismatch")
            for index, expected_command in enumerate(preflight_commands):
                attempt_path = out / "batches" / f"attempt-{index:04d}.json"
                archive = out / "batches" / "attempts" / f"opening-{index:04d}"
                archived = archive / "attempt-00.json"
                if not attempt_path.is_file() or not archived.is_file():
                    raise ValueError(
                        f"seed551_execution_attempt_archive_missing:{index}"
                    )
                attempt = json.loads(attempt_path.read_text())
                archived_attempt = json.loads(archived.read_text())
                if (
                    attempt.get("command_sha256")
                    != sha(out / "batches" / f"command-{index:04d}.json")
                    or archived_attempt.get("argv") != expected_command.get("argv")
                    or archived_attempt.get("observed_runtime")
                    != json.loads((out / "registration.json").read_text())["runtime"][
                        "observed"
                    ]
                ):
                    raise ValueError(
                        f"seed551_execution_attempt_archive_mismatch:{index}"
                    )
                if attempt.get("returncode") != 0:
                    recovery_path = out / "recovery" / f"recovery-{index:04d}.json"
                    if not recovery_path.is_file():
                        raise ValueError(
                            f"seed551_failed_attempt_without_recovery_record:{index}"
                        )
                    recovery = json.loads(recovery_path.read_text())
                recovery_path = out / "recovery" / f"recovery-{index:04d}.json"
                if recovery_path.is_file():
                    recovery = json.loads(recovery_path.read_text())
                    if (
                        recovery.get("opening_index") != index
                        or recovery.get("status")
                        != "complete_raw_pair_recovered_after_validation"
                        or recovery.get("recovery_reran_games") is not False
                        or recovery.get("attempt_sha256") != sha(attempt_path)
                        or recovery.get("command_sha256")
                        != sha(out / "batches" / f"command-{index:04d}.json")
                        or recovery.get("games_sha256")
                        != sha(out / "batches" / f"games-{index:04d}.jsonl")
                        or recovery.get("report_sha256")
                        != sha(out / "batches" / f"report-{index:04d}.json")
                    ):
                        raise ValueError(f"seed551_recovery_record_mismatch:{index}")
            rebuilt, per_opening = reconstruct_analysis(ledger)
            published = json.loads((out / "results.json").read_text())
            if rebuilt != published:
                raise ValueError("seed551_analysis_reconstruction_mismatch")
            if jsonl(out / "per-opening-results.jsonl") != per_opening:
                raise ValueError("seed551_per_opening_reconstruction_mismatch")
            result["analysis"] = rebuilt
        receipt_path = out / "verification-receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            current_evidence = {
                str(path.relative_to(out)): sha(path)
                for path in sorted(out.rglob("*"))
                if path.is_file() and path != receipt_path
            }
            if (
                receipt.get("status") != "independent_read_only_verified_complete"
                or receipt.get("verification_result") != result
                or receipt.get("stdout_sha256")
                != hashlib.sha256(receipt.get("stdout", "").encode()).hexdigest()
                or json.loads(receipt.get("stdout", "{}")) != result
                or receipt.get("evidence_sha256") != current_evidence
                or receipt.get("verifier_source_sha256")
                != sha(
                    root
                    / "ml/alphazero_lite/verify_seed551_controlled_policy_strength.py"
                )
            ):
                raise ValueError("seed551_verification_receipt_reconstruction_mismatch")
        return result
    finally:
        after = {
            path: (path.stat().st_size, path.stat().st_mtime_ns, sha(path))
            for path in observed
            if path.is_file()
        }
        if before != after:
            raise ValueError("seed551_readonly_verification_changed_bytes_or_mtimes")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tablebase", type=Path, default=None)
    parser.add_argument("--allow-relocated-copy", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.root,
                args.out,
                tablebase_path=args.tablebase,
                allow_relocated_copy=args.allow_relocated_copy,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
