"""Independent verifier for the Seed 615 frozen evaluation and game ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import seed615_evaluation as contract

ROOT = Path(__file__).resolve().parents[2]
INITIAL = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def validate_search(search: Any) -> None:
    if search != contract.SEARCH:
        raise ValueError("frozen_search_contract_mismatch")


def validate_statistics_contract(statistics: Any) -> None:
    expected = {
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
            "absolute_only": "absolute strength established; incremental improvement not established",
            "both_pass": "incremental improvement and absolute strength established",
            "incremental_only": "improvement over B512 established; absolute strength not established; do not promote automatically",
            "neither": "neither claim established",
        },
    }
    if statistics != expected:
        raise ValueError("frozen_statistical_contract_mismatch")


def validate_identity(observed: str, expected: str, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}_identity_mismatch")


def validate_authorization_record(record: dict[str, Any], freeze_sha256: str) -> None:
    owner = record.get("authorization", {})
    if (
        record.get("schema") != "seed615-evaluation-authorization-v1"
        or record.get("status") != "authorized-for-evaluation"
        or record.get("evaluation_freeze_sha256") != freeze_sha256
        or owner.get("type") != "explicit-project-owner-authorization"
        or not isinstance(owner.get("authorized_by"), str)
        or not owner["authorized_by"].strip()
        or owner.get("run_games") is not True
    ):
        raise ValueError("evaluation_authorization_or_freeze_binding_invalid")


def validate_result_hashes(
    result: dict[str, Any], game: bytes, report: bytes, command: bytes
) -> None:
    for field, payload in (
        ("game_sha256", game),
        ("report_sha256", report),
        ("command_sha256", command),
    ):
        if result.get(field) != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"modified_or_conflicting_result_artifact:{field}")


def validate_arena_command(
    command: list[str],
    assignment: dict[str, Any],
    root: Path,
    opening_path: Path,
    attempt: Path,
) -> None:
    if command != contract._arena_command(root, assignment, opening_path, attempt):
        raise ValueError("arena_command_contract_mismatch")


def validate_assignment_rows(
    rows: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    if (
        len(rows) != 4096
        or rows != expected
        or len({row.get("game_id") for row in rows}) != 4096
    ):
        raise ValueError("assignment_manifest_missing_duplicate_or_changed")


def validate_assignment_identity(
    observed: dict[str, Any], expected: dict[str, Any]
) -> None:
    if observed != expected:
        raise ValueError("assignment_manifest_missing_duplicate_or_changed")


def verify_game(
    assignment: dict[str, Any],
    opening: dict[str, Any],
    row: dict[str, Any],
    *,
    game_factory: Any = None,
) -> dict[str, Any]:
    seat = assignment.get("candidate_seat")
    if seat not in (0, 1):
        raise ValueError("invalid_candidate_seat")
    game = (
        KalahGame.from_state(dict(INITIAL)) if game_factory is None else game_factory()
    )
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list) or len(prefix) != 12:
        raise ValueError("opening_prefix_invalid")
    for relative_move in prefix:
        if game.over() or relative_move not in game.possible_moves():
            raise ValueError("illegal_registered_opening")
        game.move(game.pit_index(int(relative_move)))
    opening_hash = hashlib.sha256(contract.canonical(game.to_state())).hexdigest()
    if game.over() or opening_hash != assignment.get("opening_identity"):
        raise ValueError("opening_state_identity_mismatch")
    expected_provenance = {
        "game_index": 0,
        "game_within_opening": 0,
        "opening_index": 0,
        "opening_state_hash": opening_hash,
        "opening_prefix_moves": prefix,
        "opening_applied_prefix_length": 12,
        "challenger_player": seat,
    }
    if any(row.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("candidate_seat_or_opening_provenance_mismatch")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("missing_game_trajectory")
    tokens = trajectory.split(",") if trajectory else []
    if len(tokens) != row.get("game_length") or len(tokens) > 200:
        raise ValueError("trajectory_length_mismatch")
    for ply, token in enumerate(tokens):
        try:
            absolute = int(token)
        except ValueError as error:
            raise ValueError("malformed_trajectory_move") from error
        if (
            game.over()
            or absolute not in range(12)
            or absolute // 6 != game.current_player
        ):
            raise ValueError(f"illegal_trajectory:{ply}")
        if absolute % 6 not in game.possible_moves() or not game.move(absolute):
            raise ValueError(f"illegal_trajectory:{ply}")
    terminal = game.over()
    if terminal:
        capped = False
    elif len(tokens) == 200:
        capped = True
    else:
        raise ValueError("nonterminal_game_before_cap")
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("margin") != margin or row.get("winner") != winner:
        raise ValueError("terminal_or_capped_margin_mismatch")
    return {
        "winner": winner,
        "margin": margin,
        "game_length": len(tokens),
        "capped": capped,
        "score": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
        "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
    }


def reference_analysis(game_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently compute registered opening-cluster bootstrap statistics."""
    if len(game_records) != 4096:
        raise ValueError("incomplete_game_ledger")
    candidates = ("B512", "S615")
    expected = {
        (opening, candidate, seat)
        for opening in range(1024)
        for candidate in candidates
        for seat in (0, 1)
    }
    observed: dict[tuple[int, str], dict[int, float]] = {}
    for row in game_records:
        opening = row.get("opening_index")
        candidate = row.get("candidate")
        seat = row.get("candidate_seat")
        key = (opening, candidate, seat)
        if key not in expected or key in observed.setdefault((opening, candidate), {}):
            raise ValueError("unknown_or_duplicate_analysis_outcome")
        margin = row.get("margin")
        if type(margin) is not int:
            raise ValueError("invalid_analysis_margin")
        score = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        if "score" in row and row["score"] != score:
            raise ValueError("analysis_score_margin_mismatch")
        observed[(opening, candidate)][seat] = score
    if len(observed) != 2048 or any(
        set(seats) != {0, 1} for seats in observed.values()
    ):
        raise ValueError("missing_seat_swapped_opening_outcomes")
    means = {
        candidate: np.asarray(
            [
                (observed[(opening, candidate)][0] + observed[(opening, candidate)][1])
                / 2.0
                for opening in range(1024)
            ],
            dtype=np.float64,
        )
        for candidate in candidates
    }
    randomizer = np.random.default_rng(615)
    selected = randomizer.integers(0, 1024, size=(20000, 1024))
    paired_delta = means["S615"] - means["B512"]
    delta_draws = np.mean(paired_delta[selected], axis=1)
    absolute_draws = np.mean(means["S615"][selected], axis=1)
    interval = [
        float(value)
        for value in np.quantile(delta_draws, [0.025, 0.975], method="linear")
    ]
    lower = float(np.quantile(absolute_draws, 0.05, method="linear"))
    improvement_pass = interval[0] > 0.0
    absolute_pass = lower > 0.50
    interpretation = (
        "both_pass"
        if improvement_pass and absolute_pass
        else "incremental_only"
        if improvement_pass
        else "absolute_only"
        if absolute_pass
        else "neither"
    )
    return {
        "B512_score": float(np.mean(means["B512"])),
        "S615_score": float(np.mean(means["S615"])),
        "S615_minus_B512_interval": interval,
        "S615_vs_P_E4_one_sided_lower_bound": lower,
        "incremental_improvement_pass": improvement_pass,
        "absolute_strength_pass": absolute_pass,
        "interpretation": interpretation,
        "automatic_promotion": False,
    }


def analysis_document(results: dict[str, Any], *, capped_games: int) -> dict[str, Any]:
    return {
        "schema": "seed615-cluster-bootstrap-analysis-v1",
        "games": 4096,
        "opening_clusters": 1024,
        "capped_games": capped_games,
        "bootstrap": {
            "resamples": 20000,
            "seed": 615,
            "random_generator": "numpy.random.default_rng(615)",
            "quantile_method": "numpy.quantile(method='linear')",
            "primary_quantiles": [0.025, 0.975],
            "serving_lower_quantile": 0.05,
            "paired_opening_indices": True,
            "seat_outcomes_preserved": True,
        },
        "results": copy.deepcopy(results),
    }


def validate_analysis_document(
    observed: dict[str, Any],
    reference_results: dict[str, Any],
    *,
    capped_games: int,
) -> None:
    expected = analysis_document(reference_results, capped_games=capped_games)
    if not isinstance(observed, dict) or set(observed) != set(expected):
        raise ValueError("analysis_schema_mismatch")
    if observed != expected:
        raise ValueError("independent_analysis_mismatch")


def verify_freeze(
    root: Path = ROOT, *, allow_execution_artifacts: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    freeze_path = root / contract.FREEZE
    readiness_path = root / contract.READINESS
    if not freeze_path.is_file() or not readiness_path.is_file():
        raise ValueError("evaluation_freeze_or_readiness_missing")
    freeze = json.loads(freeze_path.read_bytes())
    expected = contract.build_freeze(root)
    if freeze != expected:
        raise ValueError("execution_freeze_contract_mismatch")
    if (
        freeze.get("evaluation_authorized") is not False
        or freeze.get("games_started") is not False
    ):
        raise ValueError("freeze_authorization_state_invalid")
    validate_search(freeze.get("search"))
    validate_statistics_contract(freeze.get("statistics"))
    source_hashes = freeze.get("source_sha256", {})
    if set(source_hashes) != set(contract.SOURCE_PATHS):
        raise ValueError("frozen_source_set_mismatch")
    for path, digest in source_hashes.items():
        if contract.sha(root / path) != digest:
            raise ValueError(f"frozen_source_hash_mismatch:{path}")
    readiness = json.loads(readiness_path.read_bytes())
    if (
        readiness.get("schema") != "seed615-evaluation-readiness-evidence-v1"
        or readiness.get("execution_freeze_sha256") != contract.sha(freeze_path)
        or readiness.get("evaluation_authorized") is not False
        or readiness.get("games_started") is not False
        or readiness.get("registered_openings") != 1024
        or readiness.get("assignment_count") != 4096
    ):
        raise ValueError("readiness_evidence_binding_mismatch")
    auth_path, results_path, analysis_path = (
        root / contract.AUTHORIZATION,
        root / contract.RESULTS,
        root / contract.ANALYSIS,
    )
    if not allow_execution_artifacts:
        contract.validate_preflight_artifact_state(
            auth_path.exists(), results_path.exists(), analysis_path.exists()
        )
    else:
        if not auth_path.is_file():
            raise ValueError("evaluation_authorization_missing")
    return {
        "passed": True,
        "freeze_sha256": contract.sha(freeze_path),
        "evaluation_authorized": auth_path.exists(),
        "games_started_by_verifier": False,
    }


def verify_authorization(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    try:
        record = json.loads((root / contract.AUTHORIZATION).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evaluation_authorization_missing_or_malformed") from error
    validate_authorization_record(record, contract.sha(root / contract.FREEZE))
    return record


def verify_one_result(
    root: Path,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate a committed result before allowing resume to skip it."""
    root = root.resolve()
    result_path = root / contract.RESULTS / assignment["game_id"] / "result.json"
    result = json.loads(result_path.read_bytes())
    if result.get("schema") != "seed615-game-result-v1" or any(
        result.get(key) != value for key, value in assignment.items()
    ):
        raise ValueError("game_result_assignment_mismatch")
    relative = Path(result.get("attempt_directory", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid_attempt_directory")
    attempt = result_path.parent / relative
    attempts_root = result_path.parent / "attempts"
    if sorted(path.name for path in attempts_root.iterdir() if path.is_dir()) != [
        relative.name
    ]:
        raise ValueError("unexpected_or_duplicate_game_attempt")
    command = json.loads((attempt / "command.json").read_bytes())
    validate_arena_command(
        command, assignment, root, attempt / "opening.jsonl", attempt
    )
    opening_lines = (attempt / "opening.jsonl").read_text().splitlines()
    if len(opening_lines) != 1:
        raise ValueError("opening_file_count_mismatch")
    opening = next(
        row
        for row in contract.read_jsonl(root / contract.OPENINGS)
        if row["opening_id"] == assignment["opening_id"]
    )
    if json.loads(opening_lines[0]) != {"prefix_moves": opening["prefix_moves"]}:
        raise ValueError("assigned_opening_file_mismatch")
    artifact_bytes = {}
    for name, field in (
        ("game.jsonl", "game_sha256"),
        ("report.json", "report_sha256"),
        ("command.json", "command_sha256"),
    ):
        artifact_bytes[name] = (attempt / name).read_bytes()
    validate_result_hashes(
        result,
        artifact_bytes["game.jsonl"],
        artifact_bytes["report.json"],
        artifact_bytes["command.json"],
    )
    game_lines = artifact_bytes["game.jsonl"].splitlines()
    if len(game_lines) != 1:
        raise ValueError("game_record_count_mismatch")
    game_row = json.loads(game_lines[0])
    verified = verify_game(assignment, opening, game_row)
    required_fields = set(assignment) | {
        "schema",
        "winner",
        "margin",
        "game_length",
        "capped",
        "score",
        "trajectory_sha256",
        "attempt_directory",
        "game_sha256",
        "report_sha256",
        "command_sha256",
    }
    if set(result) != required_fields:
        raise ValueError("game_result_schema_mismatch")
    if any(result.get(key) != value for key, value in verified.items()):
        raise ValueError("independent_game_result_mismatch")
    report = json.loads(artifact_bytes["report.json"])
    notes = report.get("notes", {})
    profile = notes.get("search_profile", {})
    search_options = notes.get("search_options", {})
    expected_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
    }
    if (
        report.get("games_played") != 1
        or notes.get("challenger_simulations") != 128
        or notes.get("current_simulations") != 128
        or notes.get("workers_used") != 1
        or notes.get("seed") != assignment["game_seed"]
        or notes.get("base_seed") != assignment["game_seed"]
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("random_opening_plies") != 0
        or notes.get("challenger_path")
        != str(root / contract.OUT / "runtime-artifacts" / assignment["candidate"])
        or notes.get("current_path") != str(root / contract.SERVING)
        or search_options != expected_options
        or profile.get("challenger_simulations") != 128
        or profile.get("current_simulations") != 128
        or profile.get("c_puct") != 1.25
        or profile.get("exact_root_solve_enabled") is not True
        or profile.get("exact_root_solve_threshold") != 16
        or profile.get("exact_root_objective") != "final_score_margin"
        or profile.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or profile.get("exact_leaf_solve_mode") != "disabled"
        or profile.get("exact_root_native_probe_sha256")
        != contract.EXPECTED["native_probe"]
        or profile.get("exact_root_tablebase_sha256") != contract.EXPECTED["tablebase"]
    ):
        raise ValueError("arena_report_search_contract_mismatch")
    wins = int(game_row.get("winner") == "challenger")
    losses = int(game_row.get("winner") == "current")
    draws = int(game_row.get("winner") == "draw")
    if (
        report.get("wins") != wins
        or report.get("losses") != losses
        or report.get("draws") != draws
        or report.get("score") != wins + 0.5 * draws
    ):
        raise ValueError("arena_report_outcome_mismatch")
    return {**assignment, **verified}


def verify_results(root: Path = ROOT) -> dict[str, Any]:
    """Verify every create-only result against its registered assignment and replay."""
    root = root.resolve()
    verify_freeze(root, allow_execution_artifacts=True)
    verify_authorization(root)
    openings = contract.read_jsonl(root / contract.OPENINGS)
    assignments = contract.read_jsonl(root / contract.ASSIGNMENTS)
    expected = contract.expected_assignments(openings)
    validate_assignment_rows(assignments, expected)
    by_id = {row["game_id"]: row for row in assignments}
    results_root = root / contract.RESULTS
    if not results_root.is_dir():
        raise ValueError("registered_result_directory_missing")
    results = list(results_root.glob("*/result.json"))
    expected_ids = {row["game_id"] for row in assignments}
    if {path.name for path in results_root.iterdir()} != expected_ids:
        raise ValueError("unexpected_or_missing_game_output_directory")
    if len(results) != 4096:
        raise ValueError("incomplete_game_ledger")
    observed = set()
    verified_rows = []
    capped_games = 0
    for result_path in results:
        result = json.loads(result_path.read_bytes())
        game_id = result.get("game_id")
        assignment = by_id.get(game_id)
        if assignment is None or game_id in observed:
            raise ValueError("unknown_or_duplicate_result")
        verified_record = verify_one_result(root, assignment)
        observed.add(game_id)
        verified_rows.append(verified_record)
        capped_games += int(verified_record["capped"])
    for attempt in results_root.glob("*/attempts/*"):
        if (
            attempt.is_dir()
            and (
                (attempt / "game.jsonl").exists() or (attempt / "report.json").exists()
            )
            and not (attempt.parent.parent / "result.json").exists()
        ):
            raise ValueError("unreconciled_incomplete_attempt")
    if (root / contract.ANALYSIS).exists():
        analysis = json.loads((root / contract.ANALYSIS).read_bytes())
        independent = reference_analysis(verified_rows)
        validate_analysis_document(analysis, independent, capped_games=capped_games)
    return {
        "passed": True,
        "completed_games": len(observed),
        "capped_games": capped_games,
        "evaluation_authorized": True,
        "analysis_present": (root / contract.ANALYSIS).exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--post-execution", action="store_true")
    args = parser.parse_args()
    output = verify_freeze() if args.preflight else verify_results()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
