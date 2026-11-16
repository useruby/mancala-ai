"""Independent read-only verifier for Seed 599 strength-evaluation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed599_evaluation_readiness as protocol
from ml.alphazero_lite.kalah_rules import KalahGame

INITIAL = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def score(winner: str, candidate: str) -> float:
    if winner == "draw":
        return 0.5
    if winner == "challenger":
        return 1.0 if candidate == "challenger" else 0.0
    if winner == "current":
        return 0.0 if candidate == "challenger" else 1.0
    raise ValueError("unknown_winner")


def verify_game(
    root: Path,
    arm: str,
    opening: dict[str, Any],
    opening_index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    game = KalahGame.from_state(INITIAL)
    for move in opening["prefix_moves"]:
        if (
            game.over()
            or int(move) not in game.possible_moves()
            or not game.move(game.pit_index(int(move)))
        ):
            raise ValueError(f"illegal_registered_opening:{opening_index}")
    if game.to_state() != opening["state"]:
        raise ValueError(f"opening_state_mismatch:{opening_index}")
    seat = int(row.get("challenger_player", -1))
    if seat not in (0, 1):
        raise ValueError("invalid_challenger_seat")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("missing_trajectory")
    moves = [int(move) for move in trajectory.split(",") if move != ""]
    if len(moves) != int(row.get("game_length", -1)) or len(moves) > 200:
        raise ValueError("trajectory_length_mismatch")
    for ply, absolute in enumerate(moves):
        relative = absolute - 6 * game.current_player
        if (
            game.over()
            or not 0 <= absolute < 12
            or absolute // 6 != game.current_player
            or relative not in game.possible_moves()
            or not game.move(absolute)
        ):
            raise ValueError(f"illegal_trajectory:{opening_index}:{ply}")
    if not game.over() and len(moves) != 200:
        raise ValueError("game_neither_terminal_nor_at_registered_cap")
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("winner") != winner or int(row.get("margin", 2**31)) != margin:
        raise ValueError(f"trajectory_outcome_mismatch:{opening_index}")
    return {"winner": winner, "margin": margin, "game_length": len(moves)}


def _assignment_paths(root: Path, arm: str, index: int) -> tuple[Path, Path]:
    base = root / protocol.OUT / "assignments" / f"{arm}-opening-{index:04d}"
    return base, base / "assignment.json"


def validate_assignment_binding(
    assignment: dict[str, Any],
    arm: str,
    index: int,
    opening: dict[str, Any],
    games: list[dict[str, Any]],
) -> None:
    if (
        assignment.get("arm") != arm
        or assignment.get("opening_index") != index
        or assignment.get("opening_id") != opening["opening_id"]
        or assignment.get("opening_canonical_identity") != opening["canonical_identity"]
        or len(games) != 2
        or {int(game.get("challenger_player", -1)) for game in games} != {0, 1}
    ):
        raise ValueError(f"assignment_binding_or_seats:{arm}:{index}")
    if assignment.get("execution_environment") != protocol.SEARCH_ENVIRONMENT:
        raise ValueError(f"execution_environment_mismatch:{arm}:{index}")
    expected_games = [
        {
            "position": position,
            "challenger_player": int(game["challenger_player"]),
            "winner": str(game["winner"]),
            "margin": int(game["margin"]),
            "trajectory_sha256": hashlib.sha256(
                str(game["trajectory"]).encode()
            ).hexdigest(),
        }
        for position, game in enumerate(games)
    ]
    if (
        assignment.get("challenger_seats") != [0, 1]
        or assignment.get("game_count") != 2
        or assignment.get("games") != expected_games
    ):
        raise ValueError(f"assignment_outcome_ledger_mismatch:{arm}:{index}")


def validate_target_identities(
    assignment: dict[str, Any], arm: str, freeze: dict[str, Any]
) -> None:
    identities = freeze["checkpoint_and_serving_identities"]
    expected = {
        "candidate_checkpoint_sha256": identities["arms"][arm]["checkpoint_sha256"],
        "candidate_runtime_model_sha256": freeze["candidate_bundle_manifest"][arm][
            "model.npz"
        ],
        "serving_p_e4_model_sha256": identities["serving_p_e4"]["model.npz"],
        "serving_p_e4_value_evaluator_sha256": identities["serving_p_e4"]["model.npz"],
    }
    if any(assignment.get(name) != digest for name, digest in expected.items()):
        raise ValueError(f"assignment_target_model_identity:{arm}")


def validate_command(
    root: Path,
    arm: str,
    index: int,
    command: list[str],
    opening_path: Path,
) -> None:
    contract = protocol.read_json(root / protocol.PUBLISHED)
    settings = contract["match_settings"]
    options = {
        "fpu_mode": settings["fpu_mode"],
        "reuse_subtree": settings["reuse_subtree"],
        "normalize_values": settings["normalize_values"],
        "root_policy_mode": settings["root_policy_mode"],
        "tactical_root_bias": settings["tactical_root_bias"],
        "root_temperature": settings["root_temperature"],
    }

    def option(name: str) -> str:
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(f"command_option_missing:{name}") from error

    candidate = root / protocol.OUT / "candidate-bundles" / arm
    serving = root / protocol.SERVING
    expected = {
        "--challenger": str(candidate),
        "--current": str(serving),
        "--challenger-policy-artifact": str(candidate),
        "--challenger-value-artifact": str(serving),
        "--current-policy-artifact": str(serving),
        "--current-value-artifact": str(serving),
        "--games": "2",
        "--games-per-opening": "2",
        "--workers": str(settings["workers"]),
        "--challenger-simulations": str(settings["simulations_per_side"]),
        "--current-simulations": str(settings["simulations_per_side"]),
        "--c-puct": str(settings["c_puct"]),
        "--seed-contract": "azlite_eval_seed_v2",
        "--challenger-search-options-json": json.dumps(
            options, sort_keys=True, separators=(",", ":")
        ),
        "--current-search-options-json": json.dumps(
            options, sort_keys=True, separators=(",", ":")
        ),
        "--random-opening-plies": "0",
        "--max-moves": str(settings["max_plies"]),
        "--exact-root-solve-threshold": str(settings["exact_root_threshold"]),
        "--exact-root-native-probe": str((root / protocol.NATIVE_PROBE).resolve()),
        "--exact-root-tablebase": str((root / protocol.TABLEBASE).resolve()),
        "--seed": str(599 + index),
        "--opening-prefixes-jsonl": str(opening_path),
        "--suite-sha256": protocol.sha(opening_path),
        "--game-jsonl": str(opening_path.parent / "games.jsonl"),
        "--out": str(opening_path.parent / "report.json"),
    }
    for flag, expected_value in expected.items():
        if option(flag) != expected_value:
            raise ValueError(f"command_contract_mismatch:{arm}:{index}:{flag}")


def validate_assignment_coverage(
    seen: dict[tuple[str, int], list[float]],
    opening_count: int,
) -> None:
    expected = {(arm, index) for arm in protocol.ARMS for index in range(opening_count)}
    if seen.keys() != expected or any(len(values) != 2 for values in seen.values()):
        raise ValueError("missing_or_duplicate_game_assignments")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _check_runtime_report(
    root: Path, arm: str, index: int, report: dict[str, Any]
) -> None:
    contract = protocol.read_json(root / protocol.PUBLISHED)
    settings = contract["match_settings"]
    notes = report.get("notes", {})
    if (
        report.get("games_played") != 2
        or notes.get("challenger_simulations") != settings["simulations_per_side"]
        or notes.get("current_simulations") != settings["simulations_per_side"]
        or notes.get("workers_used") != settings["workers"]
        or notes.get("base_seed") != 599 + index
        or notes.get("random_opening_plies") != 0
        or notes.get("exact_root_solve_threshold") != settings["exact_root_threshold"]
        or notes.get("exact_root_objective") != settings["exact_root_objective"]
        or notes.get("exact_root_tie_rule") != settings["exact_root_tie_rule"]
        or notes.get("exact_leaf_solve_mode") != "disabled"
        or notes.get("search_profile", {}).get("c_puct") != settings["c_puct"]
        or notes.get("search_options", {}).get("fpu_mode") != settings["fpu_mode"]
        or notes.get("search_options", {}).get("reuse_subtree")
        != settings["reuse_subtree"]
        or notes.get("search_options", {}).get("normalize_values")
        != settings["normalize_values"]
        or notes.get("search_options", {}).get("root_policy_mode")
        != settings["root_policy_mode"]
        or notes.get("search_options", {}).get("root_temperature")
        != settings["root_temperature"]
        or notes.get("search_options", {}).get("tactical_root_bias")
        != settings["tactical_root_bias"]
    ):
        raise ValueError(f"arena_search_contract_mismatch:{arm}:{index}")
    serving = contract["serving_p_e4"]
    if (
        notes.get("exact_root_native_probe_sha256") != serving["native_probe_sha256"]
        or notes.get("exact_root_tablebase_sha256") != serving["tablebase_sha256"]
        or notes.get("runtime_search_policy_sha256") != serving["search_policy_sha256"]
    ):
        raise ValueError(f"arena_target_runtime_identity_mismatch:{arm}:{index}")


def _verify_freeze(root: Path, freeze: dict[str, Any]) -> None:
    authenticated = protocol.authenticate_registered_inputs(root)
    if (
        freeze.get("execution_authorized") is not False
        or freeze.get("assignments") != 4096
        or freeze.get("registered_openings_identity")
        != authenticated["openings_manifest_sha256"]
        or freeze.get("training_execution_freeze_sha256")
        != authenticated["training_execution_freeze_sha256"]
        or freeze.get("checkpoint_and_serving_identities") != authenticated
        or freeze.get("runtime", {}).get("device") != "cpu"
        or freeze.get("runtime", {}).get("search_workers_per_game") != 1
        or freeze.get("runtime", {}).get("pair_parallelism") != 18
        or freeze.get("runtime", {}).get("environment") != protocol.SEARCH_ENVIRONMENT
        or freeze.get("execution_limits")
        != {
            "pair_parallelism": 18,
            "registered_cpus": 18,
            "registered_memory_gib": 64,
        }
    ):
        raise ValueError("execution_freeze_settings_mismatch")
    manifest_path = root / protocol.OUT / "candidate-bundles" / "bundle-manifest.json"
    if protocol.sha(manifest_path) != freeze.get("candidate_bundle_manifest_sha256"):
        raise ValueError("candidate_bundle_manifest_freeze_mismatch")
    if (
        hashlib.sha256(
            protocol.canonical(freeze["candidate_bundle_manifest"])
        ).hexdigest()
        != hashlib.sha256(
            protocol.canonical(protocol.read_json(manifest_path))
        ).hexdigest()
    ):
        raise ValueError("candidate_bundle_manifest_contents_mismatch")
    verification_report = (
        root / protocol.OUT / "independent-training-verification-v2.json"
    )
    if protocol.sha(verification_report) != freeze.get(
        "independent_training_verification_sha256"
    ):
        raise ValueError("independent_training_verification_freeze_mismatch")
    heldout_report = root / protocol.OUT / "heldout-policy-cross-entropy.json"
    if protocol.sha(heldout_report) != freeze.get(
        "heldout_policy_cross_entropy_sha256"
    ):
        raise ValueError("heldout_cross_entropy_freeze_mismatch")
    for arm in protocol.ARMS:
        bundle = root / protocol.OUT / "candidate-bundles" / arm
        for name, expected in freeze["candidate_bundle_manifest"][arm].items():
            if name == "policy_parity":
                continue
            if protocol.sha(bundle / name) != expected:
                raise ValueError(f"candidate_bundle_identity_mismatch:{arm}:{name}")
        parity = freeze["candidate_bundle_manifest"][arm].get("policy_parity", {})
        if parity.get("max_probability_abs_error", 1.0) > parity.get("tolerance", 0.0):
            raise ValueError(f"candidate_policy_parity_freeze_mismatch:{arm}")
    for source, expected in freeze.get("source_sha256", {}).items():
        if protocol.sha(root / source) != expected:
            raise ValueError(f"frozen_source_mismatch:{source}")


def _metrics(
    root: Path, evidence: dict[tuple[str, int], list[float]]
) -> dict[str, Any]:
    contract = protocol.read_json(root / protocol.PUBLISHED)
    rng = np.random.default_rng(contract["bootstrap"]["seed"])
    openings = contract["opening_count"]
    b_clusters = np.asarray([np.mean(evidence[("B", i)]) for i in range(openings)])
    a_clusters = np.asarray([np.mean(evidence[("A", i)]) for i in range(openings)])
    delta = b_clusters - a_clusters
    resamples = int(contract["bootstrap"]["resamples"])
    samples = rng.integers(0, openings, size=(resamples, openings))
    b_boot = b_clusters[samples].mean(axis=1)
    delta_boot = delta[samples].mean(axis=1)
    lower = float(
        np.quantile(
            b_boot,
            contract["bootstrap"]["primary_one_sided_lower_quantile"],
            method="linear",
        )
    )
    delta_interval = [
        float(x)
        for x in np.quantile(
            delta_boot,
            contract["bootstrap"]["coverage_two_sided_quantiles"],
            method=contract["bootstrap"]["quantile_method"],
        )
    ]
    strength_pass = lower > 0.50
    coverage_pass = delta_interval[0] > 0.0
    categories = {
        "both_conditions_pass": strength_pass and coverage_pass,
        "strength_only": strength_pass and not coverage_pass,
        "coverage_only": coverage_pass and not strength_pass,
        "neither_condition_passes": not strength_pass and not coverage_pass,
    }
    return {
        "B_score": float(b_clusters.mean()),
        "B_one_sided_95_lower": lower,
        "B_strength_pass": strength_pass,
        "B_minus_A_score": float(delta.mean()),
        "B_minus_A_two_sided_95_interval": delta_interval,
        "coverage_pass": coverage_pass,
        "outcome_category": next(name for name, passed in categories.items() if passed),
        "outcome_categories": categories,
        "bootstrap": {
            "method": contract["bootstrap"]["method"],
            "resamples": resamples,
            "seed": contract["bootstrap"]["seed"],
            "quantile_method": "linear",
            "unit": "opening; both seat outcomes retained within each opening",
        },
        "held_out_policy_cross_entropy": {
            "role": "secondary diagnostic only; not a decision gate",
            "available": True,
            "evidence_sha256": protocol.sha(
                root / protocol.OUT / "heldout-policy-cross-entropy.json"
            ),
            "values": protocol.read_json(
                root / protocol.OUT / "heldout-policy-cross-entropy.json"
            )["masked_policy_cross_entropy"],
        },
    }


def verify_readonly(
    root: Path = protocol.ROOT, *, allow_incomplete: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    authenticated = protocol.authenticate_registered_inputs(root)
    freeze_path = root / protocol.OUT / "execution-freeze.json"
    freeze = protocol.read_json(freeze_path)
    if freeze.get("status") != "frozen-awaiting-separate-project-owner-authorization":
        raise ValueError("execution_freeze_status")
    if freeze.get("registered_contract_sha256") != authenticated["contract_sha256"]:
        raise ValueError("execution_freeze_contract_binding")
    if (
        freeze.get("runtime", {}).get("python") != sys.version
        or freeze.get("runtime", {}).get("numpy") != np.__version__
        or freeze.get("runtime", {}).get("dependency_lock_sha256")
        != protocol.sha(root / "ml/alphazero_lite/requirements-seed599-evaluation.txt")
    ):
        raise ValueError("evaluation_runtime_identity_mismatch")
    import torch

    if freeze.get("runtime", {}).get("torch") != str(torch.__version__):
        raise ValueError("evaluation_torch_identity_mismatch")
    _verify_freeze(root, freeze)
    openings = protocol.read_json(root / protocol.OPENINGS)
    seen: dict[tuple[str, int], list[float]] = {}
    completed = 0
    expected_assignment_directories = {
        f"{arm}-opening-{index:04d}"
        for arm in protocol.ARMS
        for index in range(len(openings))
    }
    assignments_directory = root / protocol.OUT / "assignments"
    if assignments_directory.exists():
        observed_assignment_directories = {
            path.name for path in assignments_directory.iterdir() if path.is_dir()
        }
        if not observed_assignment_directories <= expected_assignment_directories:
            raise ValueError("unexpected_assignment_directory")
    for arm in protocol.ARMS:
        for index, opening in enumerate(openings):
            directory, assignment_path = _assignment_paths(root, arm, index)
            if directory.is_dir() and any(
                path.name.endswith(".pending") for path in directory.iterdir()
            ):
                raise ValueError(f"pending_assignment_evidence:{arm}:{index}")
            if not assignment_path.is_file():
                if any(
                    p.exists()
                    for p in (
                        directory / "opening.jsonl",
                        directory / "command.json",
                        directory / "games.jsonl",
                        directory / "report.json",
                    )
                ):
                    raise ValueError(f"orphaned_or_failed_assignment:{arm}:{index}")
                continue
            assignment = protocol.read_json(assignment_path)
            validate_target_identities(assignment, arm, freeze)
            games_path, report_path, opening_path = (
                directory / "games.jsonl",
                directory / "report.json",
                directory / "opening.jsonl",
            )
            command_path = directory / "command.json"
            for path, field in (
                (games_path, "games_sha256"),
                (report_path, "report_sha256"),
                (opening_path, "opening_file_sha256"),
                (command_path, "command_sha256"),
            ):
                if not path.is_file() or protocol.sha(path) != assignment.get(field):
                    raise ValueError(
                        f"assignment_evidence_identity:{arm}:{index}:{field}"
                    )
            if opening_path.read_bytes() != protocol.canonical(opening) + b"\n":
                raise ValueError(f"assigned_opening_changed:{arm}:{index}")
            validate_command(
                root, arm, index, protocol.read_json(command_path), opening_path
            )
            games = _read_rows(games_path)
            report = protocol.read_json(report_path)
            _check_runtime_report(root, arm, index, report)
            validate_assignment_binding(assignment, arm, index, opening, games)
            points: list[float] = []
            for game in games:
                result = verify_game(root, arm, opening, index, game)
                points.append(score(result["winner"], "challenger"))
            seen[(arm, index)] = points
            completed += 2
    expected = len(protocol.ARMS) * len(openings) * 2
    if completed != expected:
        if not allow_incomplete:
            raise ValueError(f"game_assignment_count:{completed}:{expected}")
        return {
            "status": "verified_incomplete",
            "completed_games": completed,
            "expected_games": expected,
        }
    validate_assignment_coverage(seen, len(openings))
    metrics = _metrics(root, seen)
    return {
        "status": "verified_complete",
        "game_count": completed,
        "execution_accounting": {
            "pair_assignments": completed // 2,
            "games": completed,
            "games_per_pair": 2,
            "seat_swapped_pairs": completed // 2,
            "search_workers_per_game": 1,
            "pair_parallelism": 18,
        },
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=protocol.ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_readonly(args.root, allow_incomplete=args.allow_incomplete),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
