"""Independent read-only verifier for the seed542 paired policy screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import seed542_expanded_policy_strength_screen as imported_suite

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / imported_suite.OUT
POINTS = {"challenger": 1.0, "draw": 0.5, "opponent": 0.0}
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(paths: set[Path]) -> dict[str, tuple[int, int, str]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, sha256(path))
        for path in sorted(paths, key=str)
    }


def _observed_evidence_paths(
    root: Path, output: Path, registration: dict[str, Any]
) -> set[Path]:
    paths = {path for path in output.rglob("*") if path.is_file()}
    for relative in (
        *registration["execution_source_sha256"],
        *registration["authenticated_file_sha256"],
    ):
        path = Path(relative)
        paths.add(path if path.is_absolute() else root / path)
    return paths


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _serving_identity(directory: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"azlite-serving-artifact-v1\0")
    for name in ("metadata.json", "weights.json"):
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _authenticate_composition(
    root: Path, output: Path, registration: dict[str, Any]
) -> None:
    runtime_auth = json.loads((output / "runtime-authentication.json").read_text())
    if registration.get("component_hashes") != runtime_auth:
        raise ValueError("verify_runtime_authentication_record_mismatch")
    expected_assignments = {
        "challenger_policy": "D-final",
        "challenger_policy_trunk": "D-final own trunk",
        "challenger_value": "initializer",
        "challenger_value_trunk": "initializer own trunk",
        "opponent_policy": "E4",
        "opponent_policy_trunk": "E4 own trunk",
        "opponent_value": "initializer",
        "opponent_value_trunk": "initializer own trunk",
    }
    if registration.get("component_assignments") != expected_assignments:
        raise ValueError("verify_component_assignment_contract_mismatch")

    artifacts = output / "runtime-artifacts"
    d_final, e4, initializer = (
        artifacts / "D-final",
        artifacts / "E4",
        artifacts / "initializer",
    )
    release = json.loads((root / "deploy/serving-release.json").read_text())
    bundle = release["bundles"]["policy-value-p"]
    identities = [_serving_identity(initializer), _serving_identity(e4)]
    if identities != bundle["expected_artifact_sha256"]:
        raise ValueError("verify_current_P_release_component_identity_mismatch")
    if [
        sha256(initializer / "search_policy.json"),
        sha256(e4 / "search_policy.json"),
    ] != bundle["expected_policy_sha256"]:
        raise ValueError("verify_current_P_release_sidecar_mismatch")
    if identities != runtime_auth.get("current_P_component_artifact_identities"):
        raise ValueError("verify_runtime_P_component_binding_mismatch")

    d_checkpoint = root / "docs/data/seed542-fixed-update-cd-v2/D-update-108.npz"
    if sha256(d_checkpoint) != sha256(d_final / "model.npz"):
        raise ValueError("verify_D_final_export_checkpoint_identity_mismatch")
    d_metadata = json.loads((d_final / "metadata.json").read_text())
    if (
        d_metadata.get("input_encoding") != "kalah_v3"
        or d_metadata.get("architecture", {}).get("model_type") != "residual_v3"
        or sha256(d_final / "search_policy.json")
        != registration["runtime_assets"]["search_policy_sha256"]
    ):
        raise ValueError("verify_D_final_runtime_contract_mismatch")
    if sha256(initializer / "model.npz") != sha256(
        root / "docs/data/seed542-selfplay-4096/initializer.npz"
    ):
        raise ValueError("verify_initializer_checkpoint_identity_mismatch")

    probe = root / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase = root / runtime_auth["runtime_assets"]["materialized_tablebase_path"]
    tablebase_pointer = root / runtime_auth["runtime_assets"]["tablebase_path"]
    if (
        sha256(probe) != runtime_auth["runtime_assets"]["native_probe_sha256"]
        or sha256(tablebase) != runtime_auth["runtime_assets"]["tablebase_sha256"]
        or sha256(tablebase_pointer)
        != runtime_auth["runtime_assets"]["tablebase_pointer_sha256"]
    ):
        raise ValueError("verify_native_runtime_asset_identity_mismatch")
    sidecar = json.loads((e4 / "search_policy.json").read_text())
    if (
        sidecar["tablebase"]["path"] != runtime_auth["runtime_assets"]["tablebase_path"]
        or sidecar["tablebase"]["sha256"]
        != runtime_auth["runtime_assets"]["tablebase_sha256"]
        or sidecar["native_probe"]["path"]
        != runtime_auth["runtime_assets"]["native_probe_path"]
        or sidecar["native_probe"]["sha256"]
        != runtime_auth["runtime_assets"]["native_probe_sha256"]
    ):
        raise ValueError("verify_sidecar_runtime_asset_routing_mismatch")


def _validate_training_receipt(
    root: Path, output: Path, registration: dict[str, Any]
) -> None:
    receipt = json.loads((output / "training-verification.json").read_text())
    if receipt.get("status") != "both_seed542_v2_verifiers_passed":
        raise ValueError("verify_seed542_training_checks_not_complete")
    limits = receipt.get("runtime_settings", {})
    expected_limits = {
        "systemd_scope": True,
        "cpu_quota": "25%",
        "cpu_weight": "1",
        "memory_max": "4294967296",
        "allowed_cpus": "30",
        "nice": 19,
        "address_space_limit": "17179869184",
        "threads_per_numerical_library": 1,
    }
    if limits != expected_limits:
        raise ValueError("verify_seed542_training_resource_contract_mismatch")
    expected_modules = {
        "verify_seed542_fixed_update_cd_v2": "verified_complete",
        "verify_seed542_fixed_update_cd_v2_supplement": "supplement_verified",
    }
    execution_root = Path(registration["execution_root"])
    for module, expected_status in expected_modules.items():
        record = receipt.get("verifiers", {}).get(module, {})
        argv = record.get("command", [])
        if (
            record.get("status") != expected_status
            or record.get("stdout", "").find(f'"status": "{expected_status}"') < 0
            or "systemd-run" not in argv
            or "--scope" not in argv
            or "CPUQuota=25%" not in argv
            or "MemoryMax=4294967296" not in argv
            or "AllowedCPUs=30" not in argv
            or "nice" not in argv
            or "19" not in argv
            or "--as=17179869184" not in argv
            or "-c" not in argv
            or "30" not in argv
            or "OMP_NUM_THREADS=1" not in argv
            or "NUMEXPR_NUM_THREADS=1" not in argv
        ):
            raise ValueError(f"verify_seed542_training_command_mismatch:{module}")
        if argv[argv.index("-m") + 1] != f"ml.alphazero_lite.{module}":
            raise ValueError(f"verify_seed542_training_entrypoint_mismatch:{module}")
        if argv[argv.index("--root") + 1] != str(execution_root):
            raise ValueError(f"verify_seed542_training_root_mismatch:{module}")
        if argv[argv.index("--out") + 1] != str(
            execution_root / "docs/data/seed542-fixed-update-cd-v2"
        ):
            raise ValueError(f"verify_seed542_training_output_mismatch:{module}")


def _option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"command_option_missing:{name}") from error


def _replay(opening: dict[str, Any], game_record: dict[str, Any]) -> tuple[str, int]:
    challenger_seat = int(game_record["challenger_player"])
    if challenger_seat not in (0, 1):
        raise ValueError("game_challenger_seat_invalid")
    game = KalahGame.from_state(imported_suite.INITIAL_STATE)
    for ply, relative in enumerate(opening["prefix_moves"]):
        if game.over() or int(relative) not in game.possible_moves():
            raise ValueError(f"trajectory_opening_prefix_illegal:{ply}")
        if not game.move(game.pit_index(int(relative))):
            raise ValueError(f"trajectory_opening_prefix_rejected:{ply}")
    if game.to_state() != opening["state"]:
        raise ValueError("trajectory_opening_root_mismatch")
    prefix = [int(move) for move in opening["prefix_moves"]]
    if game_record.get("opening_prefix_moves") != prefix:
        raise ValueError("game_opening_prefix_mismatch")
    if game_record.get("opening_state_hash") != opening["state_hash"]:
        raise ValueError("game_opening_state_hash_mismatch")
    trajectory_text = game_record.get("trajectory")
    if not isinstance(trajectory_text, str):
        raise ValueError("game_trajectory_missing")
    trajectory = [int(move) for move in trajectory_text.split(",") if move != ""]
    if len(trajectory) > 200 or len(trajectory) != game_record.get("game_length"):
        raise ValueError("game_ply_count_invalid")
    for ply, absolute in enumerate(trajectory):
        if (
            game.over()
            or absolute // 6 != game.current_player
            or absolute - game.current_player * 6 not in game.possible_moves()
            or not game.move(absolute)
        ):
            raise ValueError(f"game_trajectory_illegal:{ply}")
    if not game.over():
        raise ValueError("game_capped_or_incomplete")
    expected_margin = (
        game.captured_seeds[challenger_seat] - game.captured_seeds[1 - challenger_seat]
    )
    normalized_winner = (
        "challenger"
        if expected_margin > 0
        else "opponent"
        if expected_margin < 0
        else "draw"
    )
    report_winner = {"challenger": "challenger", "opponent": "current", "draw": "draw"}[
        normalized_winner
    ]
    if game_record.get("margin") != expected_margin:
        raise ValueError("game_margin_mismatch")
    if game_record.get("winner") != report_winner:
        raise ValueError("game_winner_mismatch")
    return normalized_winner, expected_margin


def _validate_command_and_report(
    root: Path,
    output: Path,
    index: int,
    opening: dict[str, Any],
    command_record: dict[str, Any],
    report: dict[str, Any],
    runtime: dict[str, Any],
    registration: dict[str, Any],
    freeze: dict[str, Any],
) -> None:
    argv = command_record["argv"]
    if argv[0] != registration["runtime"]["executable"] or argv[1:3] != [
        "-m",
        "ml.alphazero_lite.seed542_screen_arena_entrypoint",
    ]:
        raise ValueError(f"command_executable_or_entrypoint_mismatch:{index}")
    execution_root = Path(registration["execution_root"])
    registered_output = execution_root / imported_suite.OUT
    expected_paths = {
        "--challenger": registered_output / "runtime-artifacts/D-final",
        "--current": registered_output / "runtime-artifacts/initializer",
        "--challenger-policy-artifact": registered_output / "runtime-artifacts/D-final",
        "--challenger-value-artifact": registered_output
        / "runtime-artifacts/initializer",
        "--current-policy-artifact": registered_output / "runtime-artifacts/E4",
        "--current-value-artifact": registered_output / "runtime-artifacts/initializer",
        "--exact-root-native-probe": execution_root
        / "model-artifact/runtime/kalah_v1_tablebase",
        "--exact-root-tablebase": execution_root
        / registration["runtime_assets"]["tablebase_path"],
    }
    for option, path in expected_paths.items():
        if Path(_option(argv, option)).resolve() != path.resolve():
            raise ValueError(f"component_command_assignment_mismatch:{index}:{option}")
    expected_scalars = {
        "--games": "2",
        "--games-per-opening": "2",
        "--workers": "1",
        "--challenger-simulations": "128",
        "--current-simulations": "128",
        "--c-puct": "1.25",
        "--seed-contract": "azlite_eval_seed_v2",
        "--random-opening-plies": "0",
        "--max-moves": "200",
        "--seed": str(545 + index),
        "--suite-sha256": sha256(output / "batches" / f"opening-{index:04d}.jsonl"),
    }
    for option, expected in expected_scalars.items():
        if _option(argv, option) != expected:
            raise ValueError(f"command_contract_mismatch:{index}:{option}")
    if _option(argv, "--opening-prefixes-jsonl") != str(
        registered_output / "batches" / f"opening-{index:04d}.jsonl"
    ):
        raise ValueError(f"command_opening_path_mismatch:{index}")
    if _option(argv, "--game-jsonl") != str(
        registered_output / "batches" / f"games-{index:04d}.jsonl"
    ) or _option(argv, "--out") != str(
        registered_output / "batches" / f"report-{index:04d}.json"
    ):
        raise ValueError(f"command_output_path_mismatch:{index}")
    for option in ("--challenger-search-options-json", "--current-search-options-json"):
        if json.loads(_option(argv, option)) != SEARCH_OPTIONS:
            raise ValueError(f"command_search_options_mismatch:{index}:{option}")
    if command_record.get("registration_sha256") != freeze["registration_sha256"]:
        raise ValueError(f"command_registration_binding_mismatch:{index}")
    if command_record.get("suite_sha256") != freeze["suite_sha256"]:
        raise ValueError(f"command_suite_binding_mismatch:{index}")
    if (
        command_record.get("runtime_source_sha256")
        != registration["execution_source_sha256"]
    ):
        raise ValueError(f"command_source_binding_mismatch:{index}")
    if (
        command_record.get("component_assignments")
        != registration["component_assignments"]
    ):
        raise ValueError(f"command_component_binding_mismatch:{index}")
    if command_record.get("component_hashes") != registration["component_hashes"]:
        raise ValueError(f"command_component_hash_mismatch:{index}")
    if command_record.get("runtime") != runtime:
        raise ValueError(f"command_runtime_binding_mismatch:{index}")

    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    expected_report = {
        "challenger_simulations": 128,
        "current_simulations": 128,
        "workers_used": 1,
        "workers_requested": 1,
        "base_seed": 545 + index,
        "seed_contract": "azlite_eval_seed_v2",
        "random_opening_plies": 0,
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "exact_root_native_probe_sha256": registration["runtime_assets"][
            "native_probe_sha256"
        ],
        "exact_root_tablebase_sha256": registration["runtime_assets"][
            "tablebase_sha256"
        ],
        "runtime_search_policy_sha256": registration["runtime_assets"][
            "search_policy_sha256"
        ],
        "search_options": SEARCH_OPTIONS,
        "suite_sha256": sha256(output / "batches" / f"opening-{index:04d}.jsonl"),
    }
    for key, expected in expected_report.items():
        if notes.get(key) != expected:
            raise ValueError(f"report_runtime_contract_mismatch:{index}:{key}")
    if (
        report.get("games_played") != 2
        or report.get("games") != 2
        or profile.get("simulations") != 128
        or profile.get("challenger_simulations") != 128
        or profile.get("current_simulations") != 128
        or profile.get("c_puct") != 1.25
        or profile.get("exact_root_solve_threshold") != 16
        or profile.get("exact_root_objective") != "final_score_margin"
        or profile.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or profile.get("exact_leaf_solve_mode") != "disabled"
        or profile.get("search_options") != SEARCH_OPTIONS
        or notes.get("challenger_path")
        != str(registered_output / "runtime-artifacts/D-final")
        or notes.get("current_path")
        != str(registered_output / "runtime-artifacts/initializer")
        or notes.get("exact_root_native_probe")
        != str(execution_root / "model-artifact/runtime/kalah_v1_tablebase")
        or notes.get("exact_root_tablebase")
        != str(execution_root / registration["runtime_assets"]["tablebase_path"])
    ):
        raise ValueError(f"report_full_search_or_component_contract_mismatch:{index}")


def validate_pair(root: Path, output: Path, index: int) -> list[dict[str, Any]]:
    """Read-only validation of one raw seat-swapped pair and its bindings."""
    root, output = root.resolve(), output.resolve()
    registration = json.loads((output / "registration.json").read_text())
    freeze = json.loads((output / "execution-freeze.json").read_text())
    runtime = registration["runtime"]
    opening_path = output / "batches" / f"opening-{index:04d}.jsonl"
    game_path = output / "batches" / f"games-{index:04d}.jsonl"
    report_path = output / "batches" / f"report-{index:04d}.json"
    command_path = output / "batches" / f"command-{index:04d}.json"
    if not all(
        path.is_file() for path in (opening_path, game_path, report_path, command_path)
    ):
        raise ValueError(f"raw_pair_files_incomplete:{index}")
    opening_rows = rows(opening_path)
    games = rows(game_path)
    report = json.loads(report_path.read_text())
    command_record = json.loads(command_path.read_text())
    if len(opening_rows) != 1 or opening_rows[0]["opening_index"] != index:
        raise ValueError(f"raw_pair_opening_invalid:{index}")
    opening = opening_rows[0]
    if sha256(opening_path) != command_record.get("opening_sha256"):
        raise ValueError(f"raw_pair_opening_receipt_mismatch:{index}")
    if len(games) != 2 or {int(game["challenger_player"]) for game in games} != {0, 1}:
        raise ValueError(f"raw_pair_seat_coverage_invalid:{index}")
    if sha256(game_path) != command_record.get("games_sha256") or sha256(
        report_path
    ) != command_record.get("report_sha256"):
        raise ValueError(f"raw_pair_report_receipt_mismatch:{index}")
    _validate_command_and_report(
        root,
        output,
        index,
        opening,
        command_record,
        report,
        runtime,
        registration,
        freeze,
    )
    outcomes = []
    for position, game in enumerate(games):
        if (
            int(game.get("opening_index", -1)) != index
            or int(game.get("game_within_opening", -1)) != position
        ):
            raise ValueError(f"raw_pair_game_order_mismatch:{index}:{position}")
        winner, margin = _replay(opening, game)
        outcomes.append(
            {
                "opening_index": index,
                "game_position": position,
                "challenger_player": int(game["challenger_player"]),
                "winner": winner,
                "challenger_margin": margin,
                "game_sha256": sha256(game_path),
                "report_sha256": sha256(report_path),
                "opening_sha256": sha256(opening_path),
            }
        )
    winner_counts = {
        "challenger": sum(row["winner"] == "challenger" for row in outcomes),
        "current": sum(row["winner"] == "opponent" for row in outcomes),
        "draw": sum(row["winner"] == "draw" for row in outcomes),
    }
    if (
        report.get("wins") != winner_counts["challenger"]
        or report.get("losses") != winner_counts["current"]
        or report.get("draws") != winner_counts["draw"]
        or report.get("score")
        != (winner_counts["challenger"] + 0.5 * winner_counts["draw"]) / 2
    ):
        raise ValueError(f"raw_pair_report_outcome_count_mismatch:{index}")
    return outcomes


def independent_analysis(
    ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(ledger) != 2048:
        raise ValueError("analysis_requires_2048_validated_games")
    by_opening: dict[int, list[float]] = {index: [] for index in range(1024)}
    wins = draws = losses = 0
    for row in ledger:
        index = int(row["opening_index"])
        score = POINTS[row["winner"]]
        by_opening[index].append(score)
        wins += row["winner"] == "challenger"
        draws += row["winner"] == "draw"
        losses += row["winner"] == "opponent"
    if any(len(by_opening[index]) != 2 for index in by_opening):
        raise ValueError("analysis_opening_cluster_shape")
    clusters = np.asarray(
        [np.mean(by_opening[index]) for index in range(1024)], dtype=np.float64
    )
    rng = np.random.default_rng(545)
    sampled_indices = rng.integers(0, 1024, size=(20_000, 1024))
    boot = clusters[sampled_indices].mean(axis=1)
    lower = float(np.quantile(boot, 0.05, method="linear"))
    interval = [
        float(np.quantile(boot, 0.025, method="linear")),
        float(np.quantile(boot, 0.975, method="linear")),
    ]
    score = float((wins + 0.5 * draws) / 2048)
    per_opening = [
        {
            "opening_index": index,
            "seat0_challenger_score": by_opening[index][0],
            "seat1_challenger_score": by_opening[index][1],
            "opening_cluster_score": float(clusters[index]),
        }
        for index in range(1024)
    ]
    result = {
        "status": "complete",
        "verification_status": "independent_read_only_verification_pending",
        "exploratory_reused_suite": True,
        "games": 2048,
        "openings": 1024,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "score": score,
        "bootstrap_resamples": 20_000,
        "bootstrap_seed": 545,
        "bootstrap_quantile_method": "linear",
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": interval,
        "pass": bool(lower > 0.50),
        "decision": "advance_to_propose_separate_1200_simulation_evaluation_only"
        if lower > 0.50
        else "do_not_advance",
        "deployment_or_promotion_authorized": False,
        "fresh_confirmation": False,
        "seed461_historical_overlap_limitation_unresolved": True,
        "effect_score_vs_0_50": score - 0.50,
        "lower_bound_effect_vs_0_50": lower - 0.50,
    }
    return result, per_opening


def reconcile_pair_ledger(
    index: int, raw_pair: list[dict[str, Any]], stored_pair: list[dict[str, Any]]
) -> None:
    """Require every committed row to match independently replayed raw evidence."""
    if len(raw_pair) != 2 or len(stored_pair) != 2:
        raise ValueError(f"verify_ledger_pair_count:{index}")
    ordered = sorted(stored_pair, key=lambda row: int(row["game_position"]))
    for expected, actual in zip(raw_pair, ordered, strict=True):
        if {key: actual.get(key) for key in expected} != expected:
            raise ValueError(f"verify_ledger_raw_pair_disagreement:{index}")


def verify_published_analysis(
    ledger: list[dict[str, Any]],
    published_result: dict[str, Any],
    published_openings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Independently recompute and compare every result and per-opening field."""
    result, per_opening = independent_analysis(ledger)
    if result != published_result or per_opening != published_openings:
        raise ValueError("verify_published_analysis_mismatch")
    return result


def verify(root: Path, output: Path) -> dict[str, Any]:
    """Verify the full screen without calling any runner validation/analysis."""
    root, output = root.resolve(), output.resolve()
    registration_path = output / "registration.json"
    freeze_path = output / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    observed_paths = _observed_evidence_paths(root, output, registration)
    before_snapshot = _snapshot(observed_paths)
    if sha256(registration_path) != freeze["registration_sha256"]:
        raise ValueError("verify_registration_freeze_hash_mismatch")
    if sha256(output / "openings.jsonl") != freeze["suite_sha256"]:
        raise ValueError("verify_frozen_suite_hash_mismatch")
    if (
        sha256(output / "suite-import-report.json")
        != registration["suite"]["import_report_sha256"]
        or sha256(output / "overlap-summary.json")
        != registration["suite"]["overlap_summary_sha256"]
        or sha256(output / "overlap-per-opening.jsonl")
        != registration["suite"]["overlap_per_opening_sha256"]
        or sha256(output / "runtime-authentication.json")
        != freeze["runtime_authentication_sha256"]
        or sha256(output / "runtime-policy-preflight.json")
        != freeze["runtime_policy_preflight_sha256"]
        or sha256(output / "training-verification.json")
        != freeze["training_verification_sha256"]
        or sha256(output / "command-preflight.jsonl")
        != freeze["command_preflight_sha256"]
    ):
        raise ValueError("verify_frozen_auxiliary_binding_mismatch")
    runtime_preflight = json.loads(
        (output / "runtime-policy-preflight.json").read_text()
    )
    preflight_contract = runtime_preflight.get("resolved_contract", {})
    if (
        runtime_preflight.get("status")
        != "runtime_search_policy_contract_passed_without_inference_or_games"
        or runtime_preflight.get(
            "native_tablebase_temporarily_materialized_at_sidecar_path"
        )
        is not True
        or preflight_contract.get("exact_root_solve_threshold") != 16
        or preflight_contract.get("exact_root_objective") != "final_score_margin"
        or preflight_contract.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or preflight_contract.get("exact_leaf_solve_mode") != "disabled"
        or preflight_contract.get("exact_root_tablebase_sha256")
        != registration["runtime_assets"]["tablebase_sha256"]
    ):
        raise ValueError("verify_runtime_policy_preflight_mismatch")
    openings, import_report = imported_suite.import_seed511_suite(root)
    if import_report["suite_sha256"] != freeze["suite_sha256"]:
        raise ValueError("verify_imported_suite_binding_mismatch")
    if json.loads((output / "suite-import-report.json").read_text()) != import_report:
        raise ValueError("verify_suite_import_report_recomputation_mismatch")
    expected_overlap = imported_suite.overlap_report(root, openings)
    if (
        json.loads((output / "overlap-summary.json").read_text())
        != {
            key: value
            for key, value in expected_overlap.items()
            if key != "per_opening"
        }
        or rows(output / "overlap-per-opening.jsonl") != expected_overlap["per_opening"]
    ):
        raise ValueError("verify_overlap_report_recomputation_mismatch")
    for relative, expected in registration["execution_source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"verify_executable_source_changed:{relative}")
    for relative, expected in registration["authenticated_file_sha256"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = root / path
        if sha256(path) != expected:
            raise ValueError(f"verify_authenticated_file_changed:{relative}")
    _authenticate_composition(root, output, registration)
    _validate_training_receipt(root, output, registration)

    ledger = rows(output / "outcomes.jsonl")
    committed: dict[int, list[dict[str, Any]]] = {}
    for row in ledger:
        committed.setdefault(int(row["opening_index"]), []).append(row)
    if len(ledger) != 2048 or set(committed) != set(range(1024)):
        raise ValueError("verify_ledger_coverage")
    command_records = {
        int(row["opening_index"]): row for row in rows(output / "commands.jsonl")
    }
    if set(command_records) != set(range(1024)):
        raise ValueError("verify_command_coverage")
    preflight_records = {
        int(row["opening_index"]): row
        for row in rows(output / "command-preflight.jsonl")
    }
    if set(preflight_records) != set(range(1024)):
        raise ValueError("verify_command_preflight_coverage")
    recomputed_ledger = []
    for index, opening in enumerate(openings):
        raw_command = json.loads(
            (output / "batches" / f"command-{index:04d}.json").read_text()
        )
        if command_records[index] != raw_command:
            raise ValueError(f"verify_command_receipt_ledger_disagreement:{index}")
        expected_command = preflight_records[index]
        for field in (
            "argv",
            "registration_sha256",
            "suite_sha256",
            "component_assignments",
            "component_hashes",
            "runtime",
            "runtime_source_sha256",
        ):
            if raw_command.get(field) != expected_command.get(field):
                raise ValueError(f"verify_command_preflight_mismatch:{index}:{field}")
        pair = validate_pair(root, output, index)
        reconcile_pair_ledger(index, pair, committed[index])
        recomputed_ledger.extend(pair)
    published_result = json.loads((output / "results.json").read_text())
    published_openings = rows(output / "per-opening-results.jsonl")
    result = verify_published_analysis(
        recomputed_ledger, published_result, published_openings
    )
    if before_snapshot != _snapshot(observed_paths):
        raise ValueError("verify_read_only_evidence_bytes_or_mtimes_changed")
    return {"status": "verified_complete", "result": result, "validated_games": 2048}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
