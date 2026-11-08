"""Read-only independent verification for seed555 T512 versus C128."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import verify_seed551_controlled_policy_strength as prior_verify
from ml.alphazero_lite import seed555_policy_strength_screen as protocol


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_command(
    command: dict[str, Any],
    output: Path,
    index: int,
    *,
    registered_output: Path | None = None,
) -> None:
    command_output = registered_output or output
    argv = command.get("argv", [])
    if (
        command.get("opening_index") != index
        or command.get("component_assignments") != protocol.ASSIGNMENTS
    ):
        raise ValueError("seed555_command_index_or_assignment_mismatch")
    if (
        not argv
        or argv[0] != command.get("runtime_identity", {}).get("executable")
        or argv[1:3] != ["-m", "ml.alphazero_lite.seed555_arena_entrypoint"]
    ):
        raise ValueError("seed555_command_entrypoint_mismatch")
    if argv[argv.index("--seed") + 1] != str(protocol.SEED + index):
        raise ValueError("seed555_command_global_index_seed_mismatch")
    for flag, name in (
        ("--challenger-policy-artifact", "T512"),
        ("--challenger-value-artifact", "initializer"),
        ("--current-policy-artifact", "C128"),
        ("--current-value-artifact", "initializer"),
    ):
        if (
            Path(argv[argv.index(flag) + 1])
            != command_output / "runtime-artifacts" / name
        ):
            raise ValueError(f"seed555_command_component_path_mismatch:{flag}")
    expected = json.dumps(
        protocol.SEARCH_OPTIONS, sort_keys=True, separators=(",", ":")
    )
    for flag in ("--challenger-search-options-json", "--current-search-options-json"):
        if argv[argv.index(flag) + 1] != expected:
            raise ValueError(f"seed555_command_search_policy_mismatch:{flag}")
    fixed = {
        "--games": "2",
        "--games-per-opening": "2",
        "--workers": "1",
        "--challenger-simulations": "128",
        "--current-simulations": "128",
        "--c-puct": "1.25",
        "--random-opening-plies": "0",
        "--max-moves": "200",
        "--exact-root-solve-threshold": "16",
        "--seed-contract": "azlite_eval_seed_v2",
    }
    for flag, value in fixed.items():
        if argv[argv.index(flag) + 1] != value:
            raise ValueError(f"seed555_command_arena_option_mismatch:{flag}")
    expected_paths = {
        "--challenger": command_output / "runtime-artifacts/T512",
        "--current": command_output / "runtime-artifacts/C128",
        "--challenger-policy-artifact": command_output / "runtime-artifacts/T512",
        "--challenger-value-artifact": command_output / "runtime-artifacts/initializer",
        "--current-policy-artifact": command_output / "runtime-artifacts/C128",
        "--current-value-artifact": command_output / "runtime-artifacts/initializer",
        "--opening-prefixes-jsonl": command_output
        / "batches"
        / f"opening-{index:04d}.jsonl",
        "--game-jsonl": command_output / "batches" / f"games-{index:04d}.jsonl",
        "--out": command_output / "batches" / f"report-{index:04d}.json",
    }
    for flag, path in expected_paths.items():
        if Path(argv[argv.index(flag) + 1]) != path:
            raise ValueError(f"seed555_command_path_mismatch:{flag}")
    if (
        argv[argv.index("--exact-root-native-probe") + 1] != protocol.NATIVE_PROBE
        or Path(argv[argv.index("--exact-root-tablebase") + 1])
        != Path(command["runtime_identity"]["runtime_assets"]["tablebase_runtime_path"])
        or argv[argv.index("--suite-sha256") + 1]
        != protocol.sha256(output / "batches" / f"opening-{index:04d}.jsonl")
    ):
        raise ValueError("seed555_command_native_asset_or_suite_binding_mismatch")


def verify_preflight(
    root: Path,
    output: Path,
    *,
    tablebase_path: Path | None = None,
    allow_relocated_copy: bool = False,
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    registration_path = output / "registration.json"
    freeze_path = output / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    registered_root = Path(registration.get("execution_root", "")).resolve()
    relocated = registered_root != root
    if relocated and not allow_relocated_copy:
        raise ValueError("seed555_relocated_copy_not_authorized")
    if relocated and tablebase_path is None:
        raise ValueError("seed555_relocated_copy_requires_materialized_tablebase")
    registered_output = Path(registration.get("execution_output", "")).resolve()
    if not relocated and registered_output != output:
        raise ValueError("seed555_execution_output_binding_mismatch")
    if (
        registration.get("schema") != "seed555_t512_vs_c128_v1"
        or registration.get("status") != "prospective_rules_frozen_before_games"
        or not registration.get("execution_root")
        or freeze.get("registration_sha256") != protocol.sha256(registration_path)
        or freeze.get("suite_sha256") != protocol.OPENINGS_SHA256
        or protocol.sha256(output / "openings.jsonl") != protocol.OPENINGS_SHA256
        or registration.get("suite_sha256") != protocol.OPENINGS_SHA256
        or registration.get("component_assignments") != protocol.ASSIGNMENTS
        or freeze.get("analysis") != registration.get("analysis")
        or freeze.get("arena") != registration.get("arena")
        or freeze.get("source_sha256") != registration.get("source_sha256")
        or freeze.get("runtime") != registration.get("runtime")
        or freeze.get("execution_output") != registration.get("execution_output")
        or protocol.sha256(output / "runtime-authentication.json")
        != registration.get("runtime_authentication_sha256")
        or protocol.sha256(output / "runtime-policy-preflight.json")
        != registration.get("runtime_policy_preflight_sha256")
        or protocol.sha256(output / "README.md") != registration.get("readme_sha256")
        or protocol.sha256(output / "overlap-report.json")
        != registration.get("overlap_report_sha256")
        or protocol.sha256(output / "overlap-per-opening.jsonl")
        != registration.get("overlap_rows_sha256")
        or protocol.sha256(output / "overlap-limitations.json")
        != registration.get("overlap_limitations_sha256")
    ):
        raise ValueError("seed555_freeze_binding_mismatch")
    if (
        freeze.get("analysis") != registration.get("analysis")
        or freeze.get("arena") != registration.get("arena")
        or freeze.get("source_sha256") != registration.get("source_sha256")
        or freeze.get("runtime") != registration.get("runtime")
        or freeze.get("execution_output") != registration.get("execution_output")
    ):
        raise ValueError("seed555_frozen_semantic_binding_mismatch")
    if (
        registration.get("analysis") != protocol.ANALYSIS_RULES
        or registration.get("arena") != protocol.ARENA_RULES
    ):
        raise ValueError("seed555_registered_protocol_semantics_mismatch")
    expected_limits = {
        "cpu_quota": "25%",
        "cpu_weight": "1",
        "memory_max": "4294967296",
        "allowed_cpus": "30",
        "nice": 19,
        "address_space_limit": "17179869184",
        "workers": 1,
    }
    runtime_binding = registration.get("runtime", {})
    observed = runtime_binding.get("observed", {})
    if (
        runtime_binding.get("device") != "cpu"
        or runtime_binding.get("dtype") != "float32"
        or runtime_binding.get("torch_threads") != 1
        or runtime_binding.get("torch_interop_threads") != 1
        or runtime_binding.get("resource_limits") != expected_limits
        or observed.get("cpu_affinity") != [30]
        or observed.get("nice") != 19
        or observed.get("address_space_rlimit") != [17_179_869_184, 17_179_869_184]
        or observed.get("torch_threads") != 1
        or observed.get("torch_interop_threads") != 1
        or observed.get("environment")
        != {key: "1" for key in runtime_binding.get("environment", {})}
        or observed.get("cgroup_limits", {}).get("cpu.max") != "25000 100000"
        or observed.get("cgroup_limits", {}).get("cpu.weight") != "1"
        or observed.get("cgroup_limits", {}).get("memory.max") != "4294967296"
        or observed.get("cgroup_limits", {}).get("cpuset.cpus.effective")
        not in {"30", None}
        or runtime_binding.get("cgroup_scope_diagnostic", {})
        .get("limits", {})
        .get("cpu.max")
        != "25000 100000"
        or not runtime_binding.get("cgroup_scope_diagnostic", {})
        .get("cpu.max_path", "")
        .endswith("/cpu.max")
    ):
        raise ValueError("seed555_registered_numerical_or_resource_contract_mismatch")
    for relative, expected in registration.get("source_sha256", {}).items():
        actual = protocol.sha256(root / relative)
        if actual == expected:
            continue
        if relative == "ml/alphazero_lite/verify_seed555_policy_strength_screen.py":
            _verify_attempt_verifier_correction(root, output, freeze, expected, actual)
            continue
        if actual != expected:
            raise ValueError(f"seed555_frozen_source_changed:{relative}")
    for relative, digest in registration["authenticated_files_sha256"].items():
        if protocol.sha256(root / relative) != digest:
            raise ValueError(f"seed555_authenticated_input_mismatch:{relative}")
    _verify_seed554_binding(root, registration)
    runtime = json.loads((output / "runtime-authentication.json").read_text())
    if runtime.get("component_assignments") != protocol.ASSIGNMENTS:
        raise ValueError("seed555_runtime_component_assignment_mismatch")
    assets = runtime.get("runtime_assets", {})
    if (
        assets.get("tablebase_repository_sha256") != protocol.TABLEBASE_POINTER_SHA256
        or assets.get("tablebase_repository_size_bytes") != 134
        or assets.get("tablebase_materialized_sha256") != protocol.TABLEBASE_SHA256
        or assets.get("tablebase_materialized_size_bytes") != protocol.TABLEBASE_SIZE
        or assets.get("historical_lfs_pointer_sha256")
        != protocol.TABLEBASE_POINTER_SHA256
        or assets.get("historical_lfs_pointer_size_bytes") != 134
        or assets.get("native_probe_sha256")
        != protocol.sha256(root / protocol.NATIVE_PROBE)
    ):
        raise ValueError("seed555_runtime_asset_representation_mismatch")
    preflight = json.loads((output / "runtime-policy-preflight.json").read_text())
    if (
        preflight.get("exact_root_solve_threshold") != 16
        or preflight.get("exact_root_objective") != "final_score_margin"
        or preflight.get("exact_leaf_solve_mode") != "disabled"
        or preflight.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or preflight.get("native_probe_sha256") != assets.get("native_probe_sha256")
        or preflight.get("materialized_tablebase_sha256")
        != assets.get("tablebase_materialized_sha256")
        or preflight.get("materialized_tablebase_size_bytes")
        != assets.get("tablebase_materialized_size_bytes")
        or preflight.get("tablebase_repository_pointer_sha256")
        != assets.get("tablebase_repository_sha256")
        or preflight.get("policy_sidecar_sha256")
        != {
            name: runtime["components"][name]["files"]["search_policy.json"]
            for name in ("T512", "C128", "initializer")
        }
    ):
        raise ValueError("seed555_runtime_policy_preflight_binding_mismatch")
    materialized = tablebase_path or Path(assets["tablebase_runtime_path"])
    if (
        not materialized.is_file()
        or materialized.stat().st_size != protocol.TABLEBASE_SIZE
        or protocol.sha256(materialized) != protocol.TABLEBASE_SHA256
    ):
        raise ValueError("seed555_materialized_tablebase_bytes_mismatch")
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
    if (
        hashlib.sha256(pointer).hexdigest() != protocol.TABLEBASE_POINTER_SHA256
        or assets.get("historical_lfs_pointer_git_blob") != pointer_blob
        or assets.get("historical_lfs_pointer_commit")
        != protocol.TABLEBASE_POINTER_COMMIT
    ):
        raise ValueError("seed555_historical_lfs_pointer_identity_mismatch")
    for name, expected in (
        ("T512", protocol.T512_SHA256),
        ("C128", protocol.C128_SHA256),
        ("initializer", protocol.INITIALIZER_SHA256),
    ):
        component = runtime["components"][name]
        if (
            component.get("checkpoint_sha256") != expected
            or component["files"].get("model.npz") != expected
        ):
            raise ValueError(f"seed555_component_identity_mismatch:{name}")
        for relative, digest in component["files"].items():
            if protocol.sha256(root / component["artifact_dir"] / relative) != digest:
                raise ValueError(f"seed555_component_file_changed:{name}:{relative}")
        if component.get("source_search_policy_sha256") != protocol.sha256(
            root / "model-artifact/current/search_policy.json"
        ):
            raise ValueError(f"seed555_source_policy_sidecar_identity_mismatch:{name}")
        sidecar = json.loads(
            (root / component["artifact_dir"] / "search_policy.json").read_text()
        )
        if (
            sidecar.get("tablebase", {}).get("sha256") != protocol.TABLEBASE_SHA256
            or (registered_root / sidecar["tablebase"]["path"]).resolve()
            != Path(assets["tablebase_runtime_path"]).resolve()
            or sidecar.get("native_probe", {}).get("sha256")
            != assets["native_probe_sha256"]
        ):
            raise ValueError(
                f"seed555_component_policy_sidecar_semantic_mismatch:{name}"
            )
    commands = _jsonl(output / "command-preflight.jsonl")
    if len(commands) != protocol.OPENINGS or protocol.sha256(
        output / "command-preflight.jsonl"
    ) != freeze.get("command_preflight_sha256"):
        raise ValueError("seed555_command_preflight_hash_or_count_mismatch")
    indexed = (root / protocol.OPENINGS_PATH).read_bytes().splitlines(keepends=True)
    raw_copy = (output / "openings.jsonl").read_bytes().splitlines(keepends=True)
    if raw_copy != indexed:
        raise ValueError("seed555_suite_copy_mismatch")
    for index, command in enumerate(commands):
        opening = output / "batches" / f"opening-{index:04d}.jsonl"
        if opening.read_bytes() != indexed[index] or command.get(
            "suite_row_sha256"
        ) != protocol.sha256(opening):
            raise ValueError(f"seed555_indexed_opening_binding_mismatch:{index}")
        validate_command(command, output, index, registered_output=registered_output)
        if command.get("runtime_identity") != registration.get("runtime"):
            raise ValueError(f"seed555_command_runtime_binding_mismatch:{index}")
        if command.get("registration_sha256") != freeze.get("registration_sha256"):
            raise ValueError(f"seed555_command_registration_binding_mismatch:{index}")
    _, openings = protocol.load_openings(root)
    overlap = protocol.overlap_report(root, openings)
    if json.loads((output / "overlap-report.json").read_text()) != {
        key: value for key, value in overlap.items() if key != "per_opening"
    }:
        raise ValueError("seed555_overlap_summary_reconstruction_mismatch")
    if _jsonl(output / "overlap-per-opening.jsonl") != overlap["per_opening"]:
        raise ValueError("seed555_overlap_rows_reconstruction_mismatch")
    validate_execution_runtime_checks(output, runtime_binding)
    return {
        "registration": registration,
        "freeze": freeze,
        "runtime": runtime,
        "commands": commands,
    }


def _verify_seed554_binding(root: Path, registration: dict[str, Any]) -> None:
    corpus = root / protocol.SEED554
    binding_path = corpus / "training-input-binding.json"
    binding = json.loads(binding_path.read_text())
    search_path = corpus / "searches.jsonl"
    replay_path = corpus / "derivative-verification.json"
    physical_path = corpus / "physical-copy-verification-04.json"
    replay = json.loads(replay_path.read_text())
    physical = json.loads(physical_path.read_text())
    listed = registration.get("seed554", {})
    if (
        protocol.sha256(binding_path)
        != "9c5fce89446ec18a83f8d287394e81b288d63b439d781b111cc4faaf0e8f6422"
        or protocol.sha256(search_path) != binding.get("search_evidence_sha256")
        or protocol.sha256(replay_path) != binding.get("derivative_verification_sha256")
        or replay.get("rerun_searches") is not True
        or replay.get("paired_searches") != 11_270
        or len(search_path.read_text().splitlines()) != 11_270
        or protocol.sha256(physical_path) != listed.get("physical_copy_receipt_sha256")
        or physical.get("status") != "passed"
        or physical.get("exit_status") != 0
        or physical.get("copy_bytes_and_mtimes_unchanged") is not True
        or listed.get("training_input_binding_sha256") != protocol.sha256(binding_path)
        or listed.get("search_ledger_sha256") != protocol.sha256(search_path)
        or listed.get("full_replay_receipt_sha256") != protocol.sha256(replay_path)
        or listed.get("paired_searches") != 11_270
        or listed.get("rerun_searches_receipt") is not True
    ):
        raise ValueError("seed555_seed554_published_replay_binding_mismatch")
    evidence_snapshot = physical.get("evidence_snapshot", {})
    for relative in (
        f"{protocol.SEED554}/training-input-binding.json",
        f"{protocol.SEED554}/searches.jsonl",
        f"{protocol.SEED554}/derivative-verification.json",
    ):
        record = evidence_snapshot.get(relative, {})
        if (
            record.get("sha256") != protocol.sha256(root / relative)
            or record.get("bytes") != (root / relative).stat().st_size
        ):
            raise ValueError(
                f"seed555_seed554_physical_receipt_identity_mismatch:{relative}"
            )
    for arm in ("control", "treatment"):
        for shard in range(6):
            rel = f"{arm}/shard-{shard:03d}.jsonl"
            if (
                protocol.sha256(corpus / rel)
                != binding["derivative_sha256"][f"{arm}/{shard:03d}"]
            ):
                raise ValueError(f"seed555_seed554_training_derivative_mismatch:{rel}")


def _verify_attempt_verifier_correction(
    root: Path,
    output: Path,
    freeze: dict[str, Any],
    previous_sha256: str,
    current_sha256: str,
) -> None:
    """Authorize only the append-only, semantically checked attempt verifier."""
    correction_path = output / "verifier-attempt-correction-01.json"
    correction = json.loads(correction_path.read_text())
    original_verifier = "ml/alphazero_lite/verify_seed555_policy_strength_screen.py"
    correction_source = "ml/alphazero_lite/verify_seed555_attempt_ledger.py"
    previous_commit = "ae349b73b87032419ed30b02b84cac63813764cd"
    previous_bytes = subprocess.run(
        ["git", "show", f"{previous_commit}:{original_verifier}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    previous_blob = subprocess.run(
        ["git", "rev-parse", f"{previous_commit}:{original_verifier}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        correction.get("schema") != "seed555_verifier_attempt_correction_v1"
        or correction.get("status") != "append_only_verifier_correction_verified"
        or correction.get("previous_verifier_sha256") != previous_sha256
        or hashlib.sha256(previous_bytes).hexdigest() != previous_sha256
        or correction.get("previous_verifier_git_commit") != previous_commit
        or correction.get("previous_verifier_git_blob") != previous_blob
        or correction.get("corrected_verifier_sha256") != current_sha256
        or correction.get("correction_source_sha256")
        != protocol.sha256(root / correction_source)
        or correction.get("correction_verifier_source_sha256")
        != protocol.sha256(root / correction_source)
        or correction.get("registration_sha256")
        != protocol.sha256(output / "registration.json")
        or correction.get("execution_freeze_sha256")
        != protocol.sha256(output / "execution-freeze.json")
        or correction.get("verification_receipt_sha256")
        != protocol.sha256(output / "verification-receipt.json")
        or correction.get("published_outcomes_sha256")
        != protocol.sha256(output / "outcomes.jsonl")
        or correction.get("published_analysis_sha256")
        != protocol.sha256(output / "analysis.json")
        or correction.get("published_verification_sha256")
        != protocol.sha256(output / "verification.json")
        or correction.get("correction")
        != "validate committed attempts against opening index, frozen command hash, and zero exit status; missing attempts require a valid saved-pair recovery receipt"
    ):
        raise ValueError("seed555_attempt_verifier_correction_binding_mismatch")
    from ml.alphazero_lite.verify_seed555_attempt_ledger import verify_attempts

    attempts = verify_attempts(output)
    if (
        correction.get("attempts_summary_sha256")
        != attempts.get("attempts_summary_sha256")
        or correction.get("committed_pairs_checked") != protocol.OPENINGS
        or correction.get("attempt_records_checked")
        != attempts.get("attempt_records_checked")
        or correction.get("failed_attempts") != 0
        or correction.get("unexplained_missing_attempts") != 0
    ):
        raise ValueError("seed555_attempt_verifier_correction_semantics_mismatch")


def validate_execution_runtime_checks(
    output: Path, expected: dict[str, Any]
) -> list[dict[str, Any]]:
    path = output / "execution-runtime-checks.jsonl"
    if not path.is_file():
        return []
    rows = _jsonl(path)
    expected_live = {
        key: value
        for key, value in expected.items()
        if key != "cgroup_scope_diagnostic"
    }
    expected_scope = expected.get("cgroup_scope_diagnostic", {})
    expected_hash = hashlib.sha256(protocol.stable_json(expected)).hexdigest()
    for index, row in enumerate(rows):
        scope = row.get("cgroup_scope_diagnostic", {})
        if (
            row.get("status") != "prelaunch_runtime_contract_passed"
            or row.get("registration_sha256")
            != protocol.sha256(output / "registration.json")
            or row.get("arena_games_started_by_check") is not False
            or row.get("runtime_contract_sha256") != expected_hash
            or row.get("runtime_identity") != expected_live
            or scope.get("limits") != expected_scope.get("limits")
            or not scope.get("cpu.max_path", "").endswith("/cpu.max")
            or scope.get("limits", {}).get("cpu.max") != "25000 100000"
            or scope.get("limits", {}).get("memory.max") != "4294967296"
            or scope.get("limits", {}).get("cpu.weight") != "1"
        ):
            raise ValueError(f"seed555_execution_runtime_receipt_mismatch:{index}")
    return rows


def validate_pair(root: Path, output: Path, index: int) -> list[dict[str, Any]]:
    root, output = root.resolve(), output.resolve()
    openings = prior_verify.replay_suite(root)
    batch = output / "batches"
    indexed = (root / protocol.OPENINGS_PATH).read_bytes().splitlines(keepends=True)
    if (batch / f"opening-{index:04d}.jsonl").read_bytes() != indexed[index]:
        raise ValueError(f"seed555_played_opening_mismatch:{index}")
    command = json.loads((batch / f"command-{index:04d}.json").read_text())
    frozen = _jsonl(output / "command-preflight.jsonl")[index]
    if command != frozen:
        raise ValueError(f"seed555_played_command_mismatch:{index}")
    games_path = batch / f"games-{index:04d}.jsonl"
    report_path = batch / f"report-{index:04d}.json"
    games = _jsonl(games_path)
    if len(games) != 2:
        raise ValueError(f"seed555_pair_count_mismatch:{index}")
    outcomes = []
    for seat, game in enumerate(games):
        prior_verify._validate_arena_local_opening_index(game)
        if (
            game.get("game_index") != seat
            or game.get("game_within_opening") != seat
            or game.get("challenger_player") != seat
        ):
            raise ValueError(f"seed555_swapped_seat_mismatch:{index}:{seat}")
        outcome, margin, capped = prior_verify._game_outcome(openings[index], game)
        # In the seed551 helper T is the challenger and C is the opponent.
        label = "T512" if outcome == "T" else "C128" if outcome == "C" else "draw"
        outcomes.append(
            {
                "opening_index": index,
                "game_position": seat,
                "challenger_player": seat,
                "outcome": label,
                "margin": margin,
                "capped_nonterminal": bool(capped),
                "game_sha256": protocol.sha256(games_path),
                "report_sha256": protocol.sha256(report_path),
                "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"),
            }
        )
    report = json.loads(report_path.read_text())
    if report.get("schema") != "arena_v1" or report.get("games_played") != 2:
        raise ValueError(f"seed555_report_shape_mismatch:{index}")
    if (
        report.get("wins") != sum(g["winner"] == "challenger" for g in games)
        or report.get("losses") != sum(g["winner"] == "current" for g in games)
        or report.get("draws") != sum(g["winner"] == "draw" for g in games)
    ):
        raise ValueError(f"seed555_report_winner_perspective_mismatch:{index}")
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    runtime_auth = json.loads((output / "runtime-authentication.json").read_text())
    expected_asset_ids = {
        "exact_root_tablebase_sha256": runtime_auth["runtime_assets"][
            "tablebase_materialized_sha256"
        ],
        "exact_root_native_probe_sha256": runtime_auth["runtime_assets"][
            "native_probe_sha256"
        ],
    }
    for container in (notes, profile):
        for name, digest in expected_asset_ids.items():
            if container.get(name) != digest:
                raise ValueError(
                    f"seed555_report_runtime_asset_identity_mismatch:{index}:{name}"
                )
    for key, value in {
        "simulations": 128,
        "challenger_simulations": 128,
        "current_simulations": 128,
        "c_puct": 1.25,
        "exact_root_solve_enabled": True,
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_leaf_solve_mode": "disabled",
        "search_options": protocol.SEARCH_OPTIONS,
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    }.items():
        if profile.get(key) != value:
            raise ValueError(f"seed555_report_search_setting_mismatch:{index}:{key}")
    if (
        notes.get("workers_requested") != 1
        or notes.get("workers_used") != 1
        or notes.get("random_opening_plies") != 0
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("base_seed") != protocol.SEED + index
    ):
        raise ValueError(f"seed555_report_runtime_setting_mismatch:{index}")
    expected_runtime_policy = {
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "runtime_search_policy_sha256": protocol.sha256(
            output / "runtime-artifacts/C128/search_policy.json"
        ),
        "challenger_runtime_policy_inherited": False,
    }
    if any(profile.get(key) != value for key, value in expected_runtime_policy.items()):
        raise ValueError(f"seed555_report_policy_sidecar_mismatch:{index}")
    command = _jsonl(output / "command-preflight.jsonl")[index]
    components = runtime_auth["components"]
    expected_identity = {
        "opening_index": index,
        "seed": protocol.SEED + index,
        "command_preflight_entry_sha256": hashlib.sha256(
            protocol.stable_json(command)
        ).hexdigest(),
        "component_assignments": protocol.ASSIGNMENTS,
        "runtime_authentication_sha256": protocol.sha256(
            output / "runtime-authentication.json"
        ),
        "component_artifact_sha256": {
            name: value["checkpoint_sha256"] for name, value in components.items()
        },
        "search_policy_sidecar_sha256": {
            name: value["files"]["search_policy.json"]
            for name, value in components.items()
        },
        "tablebase_object_sha256": runtime_auth["runtime_assets"][
            "tablebase_materialized_sha256"
        ],
        "tablebase_object_size_bytes": runtime_auth["runtime_assets"][
            "tablebase_materialized_size_bytes"
        ],
        "tablebase_resolved_runtime_path": runtime_auth["runtime_assets"][
            "tablebase_runtime_path"
        ],
        "tablebase_repository_pointer_sha256": runtime_auth["runtime_assets"][
            "tablebase_repository_sha256"
        ],
        "tablebase_repository_pointer_size_bytes": runtime_auth["runtime_assets"][
            "tablebase_repository_size_bytes"
        ],
        "historical_lfs_pointer_sha256": runtime_auth["runtime_assets"][
            "historical_lfs_pointer_sha256"
        ],
        "historical_lfs_pointer_git_blob": runtime_auth["runtime_assets"][
            "historical_lfs_pointer_git_blob"
        ],
        "native_probe_sha256": runtime_auth["runtime_assets"]["native_probe_sha256"],
        "registered_runtime": command["runtime_identity"],
    }
    for container in (notes, profile):
        identity = container.get("seed555_execution_identity", {})
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"seed555_report_execution_identity_mismatch:{index}")
        observed = identity.get("observed_runtime", {})
        if (
            observed != command["runtime_identity"].get("observed")
            or identity.get("live_cpu_max") != "25000 100000"
            or not identity.get("live_cpu_max_path", "").endswith("/cpu.max")
        ):
            raise ValueError(f"seed555_report_observed_runtime_mismatch:{index}")
    unsigned_profile = {key: value for key, value in profile.items() if key != "hash"}
    expected_profile_hash = hashlib.sha256(
        json.dumps(unsigned_profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        profile.get("hash") != expected_profile_hash
        or notes.get("search_profile_hash") != expected_profile_hash
    ):
        raise ValueError(f"seed555_report_search_profile_hash_mismatch:{index}")
    recovery_path = batch / f"recovery-{index:04d}.json"
    if recovery_path.is_file():
        expected_recovery = {
            "opening_index": index,
            "status": "complete_raw_pair_recovered_after_validation",
            "validated_game_records": 2,
            "command_sha256": protocol.sha256(batch / f"command-{index:04d}.json"),
            "games_sha256": protocol.sha256(games_path),
            "report_sha256": protocol.sha256(report_path),
            "recovery_reran_games": False,
        }
        if json.loads(recovery_path.read_text()) != expected_recovery:
            raise ValueError(f"seed555_recovery_receipt_mismatch:{index}")
    return outcomes


def verify(
    root: Path,
    output: Path,
    *,
    tablebase_path: Path | None = None,
    allow_relocated_copy: bool = False,
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    before = _evidence_snapshot(root, output, tablebase_path)
    verify_preflight(
        root,
        output,
        tablebase_path=tablebase_path,
        allow_relocated_copy=allow_relocated_copy,
    )
    ledger_path = output / "outcomes.jsonl"
    ledger = _jsonl(ledger_path) if ledger_path.exists() else []
    indexed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        indexed.setdefault(int(row["opening_index"]), []).append(row)
    validated = {}
    for index in range(protocol.OPENINGS):
        marker = output / "batches" / f"completion-{index:04d}.json"
        evidence = [
            (output / "batches" / f"{prefix}-{index:04d}.{suffix}").exists()
            for prefix, suffix in (
                ("command", "json"),
                ("games", "jsonl"),
                ("report", "json"),
            )
        ]
        if marker.exists():
            expected = validate_pair(root, output, index)
            if (
                sorted(indexed.get(index, []), key=lambda row: row["game_position"])
                != expected
            ):
                raise ValueError(f"seed555_ledger_pair_mismatch:{index}")
            receipt = json.loads(marker.read_text())
            for name, suffix in (
                ("command", "json"),
                ("games", "jsonl"),
                ("report", "json"),
            ):
                if receipt.get(f"{name}_sha256") != protocol.sha256(
                    output / "batches" / f"{name}-{index:04d}.{suffix}"
                ):
                    raise ValueError(
                        f"seed555_completion_receipt_mismatch:{index}:{name}"
                    )
            if receipt.get("outcomes") != expected:
                raise ValueError(f"seed555_completion_outcomes_mismatch:{index}")
            validated[index] = expected
        elif any(evidence):
            raise ValueError(f"seed555_uncommitted_pair_evidence_preserved:{index}")
    if len(ledger) != 2 * len(validated):
        raise ValueError("seed555_ledger_accounting_mismatch")
    runtime_checks = validate_execution_runtime_checks(
        output, json.loads((output / "registration.json").read_text())["runtime"]
    )
    if ledger and not runtime_checks:
        raise ValueError("seed555_missing_prelaunch_runtime_receipt")
    if len(ledger) != protocol.GAMES:
        return {
            "status": "incomplete",
            "validated_games": len(ledger),
            "validated_pairs": len(validated),
        }
    analysis, scores = reconstruct_analysis(ledger)
    compare_analysis(
        json.loads((output / "analysis.json").read_text()),
        analysis,
        _jsonl(output / "per-opening-scores.jsonl"),
        scores,
    )
    after = _evidence_snapshot(root, output, tablebase_path)
    unchanged = before == after
    if not unchanged:
        raise ValueError("seed555_read_only_verification_changed_evidence")
    from ml.alphazero_lite.verify_seed555_attempt_ledger import verify_attempts

    attempts = verify_attempts(output)
    return {
        "status": "verified_complete",
        "validated_games": len(ledger),
        "validated_pairs": len(validated),
        "prelaunch_runtime_checks": len(runtime_checks),
        "analysis": analysis,
        "per_opening_scores": scores,
        "attempt_verification": attempts,
        "preflight": "verified",
        "read_only_bytes_and_mtimes_unchanged": unchanged,
        "evidence_snapshot": before,
    }


def _evidence_snapshot(
    root: Path, output: Path, tablebase_path: Path | None
) -> dict[str, dict[str, Any]]:
    root, output = root.resolve(), output.resolve()
    paths = {
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"verification.json", "verification-receipt.json"}
    }
    registration_path = output / "registration.json"
    if registration_path.is_file():
        registration = json.loads(registration_path.read_text())
        paths.update(
            root / relative for relative in registration.get("source_sha256", {})
        )
        paths.update(
            root / relative
            for relative in registration.get("authenticated_files_sha256", {})
        )
        auth_path = output / "runtime-authentication.json"
        if auth_path.is_file():
            auth = json.loads(auth_path.read_text())
            for component in auth.get("components", {}).values():
                base = root / component["artifact_dir"]
                paths.update(base / relative for relative in component.get("files", {}))
            paths.add(
                tablebase_path or Path(auth["runtime_assets"]["tablebase_runtime_path"])
            )
    return {
        str(path.resolve()): {
            "sha256": protocol.sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths, key=lambda value: str(value))
        if path.is_file()
    }


def reconstruct_analysis(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independently rebuild the preregistered score and bootstrap."""
    if len(ledger) != protocol.GAMES:
        raise ValueError("seed555_verifier_analysis_game_count_mismatch")
    point_values = {"T512": 1.0, "draw": 0.5, "C128": 0.0}
    groups: dict[int, dict[int, dict[str, Any]]] = {
        index: {} for index in range(protocol.OPENINGS)
    }
    terminal_counts = {"T512": 0, "draw": 0, "C128": 0}
    capped_counts = {"T512": 0, "draw": 0, "C128": 0}
    capped_games = 0
    for row in ledger:
        opening, seat = int(row["opening_index"]), int(row["game_position"])
        if opening not in groups or seat not in (0, 1) or seat in groups[opening]:
            raise ValueError("seed555_verifier_duplicate_ledger_coordinate")
        outcome = row.get("outcome")
        if outcome not in point_values:
            raise ValueError("seed555_verifier_invalid_outcome")
        capped = row.get("capped_nonterminal")
        if type(capped) is not bool:
            raise ValueError("seed555_verifier_terminal_status_invalid")
        if capped:
            capped_games += 1
            capped_counts[outcome] += 1
        else:
            terminal_counts[outcome] += 1
        groups[opening][seat] = row
    if any(set(pair) != {0, 1} for pair in groups.values()):
        raise ValueError("seed555_verifier_missing_seat_pair")
    opening_clusters: list[float] = []
    per_opening: list[dict[str, Any]] = []
    for opening in range(protocol.OPENINGS):
        pair = groups[opening]
        cap_count = int(pair[0]["capped_nonterminal"]) + int(
            pair[1]["capped_nonterminal"]
        )
        cluster = float(
            np.mean(
                [point_values[pair[0]["outcome"]], point_values[pair[1]["outcome"]]]
            )
        )
        opening_clusters.append(cluster)
        per_opening.append(
            {
                "opening_index": opening,
                "seat0_T512_score": point_values[pair[0]["outcome"]],
                "seat1_T512_score": point_values[pair[1]["outcome"]],
                "opening_cluster_score": None if cap_count else cluster,
                "capped_nonterminal_games": cap_count,
            }
        )
    valid = capped_games == 0
    if valid:
        values = np.asarray(opening_clusters, dtype=np.float64)
        rng = np.random.default_rng(protocol.BOOTSTRAP_SEED)
        sampled = values[
            rng.integers(
                0,
                protocol.OPENINGS,
                size=(protocol.BOOTSTRAP_RESAMPLES, protocol.OPENINGS),
            )
        ]
        means = sampled.mean(axis=1)
        score = float(values.mean())
        lower = float(np.quantile(means, 0.05, method="linear"))
        interval = [
            float(np.quantile(means, quantile, method="linear"))
            for quantile in (0.025, 0.975)
        ]
    else:
        score = None
        lower = None
        interval = None
    analysis = {
        "status": "complete",
        "games": protocol.GAMES,
        "opening_pairs": protocol.OPENINGS,
        "terminal_wins": terminal_counts["T512"],
        "terminal_draws": terminal_counts["draw"],
        "terminal_losses": terminal_counts["C128"],
        "capped_nonterminal_games": capped_games,
        "capped_outcomes_by_winner": capped_counts,
        "capped_nonterminal_prevents_valid_decision": bool(capped_games),
        "score": score,
        "bootstrap_resamples": protocol.BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": protocol.BOOTSTRAP_SEED,
        "sampling": "opening clusters with replacement; duplicate selections retained",
        "quantile_method": "NumPy linear",
        "one_sided_95_lower": lower,
        "descriptive_95_interval": interval,
        "decision": "propose_T512_vs_serving_P_screen"
        if valid and lower > 0.5
        else "no_valid_advancement_decision"
        if not valid
        else "do_not_advance_teacher_budget_branch",
        "strength_decision_valid": valid,
        "promotion_or_deployment_authorized": False,
    }
    return analysis, per_opening


def compare_analysis(
    saved: dict[str, Any],
    reconstructed: dict[str, Any],
    saved_scores: list[dict[str, Any]],
    reconstructed_scores: list[dict[str, Any]],
) -> None:
    if saved != reconstructed:
        raise ValueError("seed555_analysis_reconstruction_mismatch")
    if saved_scores != reconstructed_scores:
        raise ValueError("seed555_opening_score_reconstruction_mismatch")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=protocol.ROOT)
    parser.add_argument("--out", type=Path, default=protocol.OUT)
    parser.add_argument("--tablebase", type=Path)
    parser.add_argument("--allow-relocated-copy", action="store_true")
    args = parser.parse_args()
    output = args.out if args.out.is_absolute() else args.root / args.out
    print(
        json.dumps(
            verify(
                args.root,
                output,
                tablebase_path=args.tablebase.resolve() if args.tablebase else None,
                allow_relocated_copy=args.allow_relocated_copy,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
