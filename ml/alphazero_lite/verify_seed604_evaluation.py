"""Independent read-only verifier for Seed 604 evaluation preparation/results."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("docs/data/seed604-replay-optimization")
COHORT = Path("docs/data/seed604-evaluation-opening-cohort-v5")
OUT = DATA / "evaluation-preparation"
FREEZE = OUT / "execution-freeze.json"
READINESS = OUT / "readiness-evidence.json"
ASSIGNMENTS = OUT / "assignment-manifest.json"
AUTHORIZATION = OUT / "evaluation-authorization.json"
RESULTS = OUT / "results"
ANALYSIS = OUT / "analysis.json"
OPENING_PATH = Path(
    "docs/data/seed604-evaluation-opening-cohort-v5/opening-manifest.json"
)
INITIAL = {
    "player_pits": [4] * 6,
    "opponent_pits": [4] * 6,
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}
SOURCE_PATHS = (
    "ml/alphazero_lite/seed604_evaluation.py",
    "ml/alphazero_lite/verify_seed604_evaluation.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/kalah_rules.py",
    "ml/alphazero_lite/native_exact_root_tablebase.py",
    "ml/alphazero_lite/exact_root_decision.py",
    "ml/alphazero_lite/runtime_search_policy.py",
    "ml/alphazero_lite/requirements-seed599-evaluation.txt",
    "ml/alphazero_lite/export_artifact.py",
    "ml/alphazero_lite/train.py",
    "ml/alphazero_lite/input_encodings.py",
    "ml/alphazero_lite/test_seed604_evaluation.py",
    "ml/alphazero_lite/verify_seed604_training_result.py",
    "ml/alphazero_lite/seed604_training_executor.py",
)
EXPECTED_SEARCH = {
    "simulations": 128,
    "c_puct": 1.25,
    "workers": 1,
    "max_plies": 200,
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "deterministic",
    "root_temperature": 0.0,
    "tactical_root_bias": 0.0,
    "noise": False,
    "exact_root_enabled": True,
    "exact_root_threshold": 16,
    "exact_root_objective": "final_score_margin",
    "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
    "exact_leaf_solving": False,
}
EXPECTED_ENVIRONMENT = {
    "PYTHONHASHSEED": "604",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def validate_asset_bytes(raw: bytes, expected_sha256: str, expected_size: int) -> None:
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("exact_root_asset_size_or_hash_mismatch")


def validate_result_hashes(
    result: dict[str, Any], game_raw: bytes, report_raw: bytes, command_raw: bytes
) -> None:
    expected = {
        "game_sha256": hashlib.sha256(game_raw).hexdigest(),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "command_sha256": hashlib.sha256(command_raw).hexdigest(),
    }
    if any(result.get(field) != digest for field, digest in expected.items()):
        raise ValueError("modified_or_conflicting_result_artifact")


def store_margin_outcome(
    stores: list[int] | tuple[int, int], seat: int
) -> tuple[int, str, float]:
    if seat not in (0, 1) or len(stores) != 2:
        raise ValueError("invalid_store_margin_inputs")
    margin = int(stores[seat] - stores[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    score = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
    return margin, winner, score


def verify_bound_inputs(root: Path, freeze: dict[str, Any]) -> None:
    identities = freeze.get("registration", {})
    expected = {
        "registration_sha256": (root / DATA / "registration.json", None),
        "cohort_registration_sha256": (root / COHORT / "registration.json", None),
        "cohort_verification_report_sha256": (
            root / COHORT / "verification-report.json",
            None,
        ),
        "cohort_runtime_provenance_sha256": (
            root / COHORT / "runtime-provenance.json",
            None,
        ),
        "seed599_evaluation_contract_sha256": (
            root / "docs/data/seed599-replay-coverage/evaluation-contract.json",
            "8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4",
        ),
        "opening_manifest_sha256": (
            root / OPENING_PATH,
            "445052b818b36f84953f73f280b20f7421c441fac6a74a0e2bb309cf907fb410",
        ),
        "serving_model_sha256": (
            root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4/model.npz",
            "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
        ),
        "native_probe_sha256": (
            root / "model-artifact/runtime/kalah_v1_tablebase",
            "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48",
        ),
        "tablebase_sha256": (
            root / "model-artifact/runtime/kalah_v1_21.kvtb",
            "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7",
        ),
    }
    for name, (path, pinned) in expected.items():
        observed = sha(path)
        if observed != identities.get(name) or (
            pinned is not None and observed != pinned
        ):
            raise ValueError(f"frozen_input_identity_mismatch:{name}")
    for key, relative, pinned in (
        (
            "serving_metadata_sha256",
            "metadata.json",
            "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
        ),
        (
            "serving_weights_sha256",
            "weights.json",
            "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
        ),
        (
            "serving_search_policy_sha256",
            "search_policy.json",
            "b13ced03deda6bd2d1737c6d339aececfe0b3730563ca800e36d12944b08fb8d",
        ),
    ):
        observed = sha(
            root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4" / relative
        )
        if observed != identities.get(key) or observed != pinned:
            raise ValueError(f"serving_artifact_identity_mismatch:{relative}")
    if (
        identities.get("registration_sha256")
        != "b2a8eb494fefa97ee454cd52760612a92f6f766b053dfb730a6c94bb962e3d71"
    ):
        raise ValueError("seed604_registration_hash_mismatch")
    if (
        identities.get("cohort_registration_sha256")
        != "538584c5fb326b28146b4a0b4ff5da9b4e7e1631db4301de1338e6bf486eb4a4"
    ):
        raise ValueError("cohort_registration_hash_mismatch")
    if identities.get("search_contract_equal_to_seed599") is not True:
        raise ValueError("registered_search_contract_comparison_missing")
    if (root / "model-artifact/runtime/kalah_v1_21.kvtb").stat().st_size != 709635072:
        raise ValueError("frozen_tablebase_size_mismatch")
    pointer_raw = subprocess.run(
        ["git", "show", "HEAD:model-artifact/runtime/kalah_v1_21.kvtb"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    pointer_sha = hashlib.sha256(pointer_raw).hexdigest()
    if (
        pointer_sha != identities.get("lfs_pointer_sha256")
        or pointer_sha
        != "a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c"
        or identities.get("lfs_pointer_oid")
        != "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
        or "oid sha256:f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
        not in pointer_raw.decode("ascii")
        or "size 709635072" not in pointer_raw.decode("ascii")
        or sha(root / ".gitattributes") != identities.get("gitattributes_sha256")
        or "model-artifact/runtime/kalah_v1_21.kvtb filter=lfs diff=lfs merge=lfs -text"
        not in (root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    ):
        raise ValueError("lfs_materialization_provenance_mismatch")
    for candidate, checkpoint_sha in (
        ("B512", "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c"),
        ("B2048", "e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874"),
    ):
        rel = (
            Path(
                "docs/data/seed599-replay-coverage/training-run-B/seed599-arm-B-update-512.npz"
            )
            if candidate == "B512"
            else DATA / "training-run-B2048/seed604-B2048-update-2048.npz"
        )
        if sha(root / rel) != checkpoint_sha:
            raise ValueError(f"frozen_checkpoint_identity_mismatch:{candidate}")
        runtime = freeze.get("runtime_artifacts", {}).get(candidate, {})
        runtime_model = root / runtime.get("directory", "") / "model.npz"
        if not runtime_model.is_file() or sha(runtime_model) != checkpoint_sha:
            raise ValueError(f"runtime_checkpoint_identity_mismatch:{candidate}")
        for filename, field in (
            ("weights.json", "weights_json_sha256"),
            ("metadata.json", "metadata_sha256"),
        ):
            artifact = runtime_model.parent / filename
            if not artifact.is_file() or sha(artifact) != runtime.get(field):
                raise ValueError(
                    f"runtime_artifact_hash_mismatch:{candidate}:{filename}"
                )
        metadata = json.loads((runtime_model.parent / "metadata.json").read_bytes())
        if (
            metadata.get("architecture", {}).get("model_type") != "residual_v3"
            or metadata.get("input_encoding") != "kalah_v3"
            or metadata.get("artifacts", {}).get("weights_sha256") != checkpoint_sha
            or metadata.get("artifacts", {}).get("weights_json_sha256")
            != runtime.get("weights_json_sha256")
        ):
            raise ValueError(f"runtime_policy_trunk_identity_mismatch:{candidate}")
    composition = freeze.get("synthetic_composition_verification", {})
    if set(composition) != {"B512", "B2048"} or any(
        report.get("registered_openings_used") is not False
        or report.get("policy_source") != "candidate including candidate trunk"
        or report.get("value_source") != "serving P-E4 including serving trunk"
        or report.get("players_checked") != [0, 1]
        or report.get("extra_turns_checked", 0) < 1
        for report in composition.values()
    ):
        raise ValueError("synthetic_composition_evidence_invalid")
    receipt = root / DATA / "training-run-B2048/training-receipt.json"
    if sha(receipt) != freeze.get("training_receipt_sha256"):
        raise ValueError("training_receipt_hash_mismatch")
    verification = freeze.get("training_verification", {})
    if (
        verification.get("passed") is not True
        or verification.get("checkpoint_sha256")
        != "e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874"
    ):
        raise ValueError("training_result_verification_missing")
    from ml.alphazero_lite.verify_seed604_training_result import verify_result

    actual_training = verify_result(root / DATA / "training-run-B2048", root=root)
    if actual_training != verification:
        raise ValueError("frozen_training_verification_mismatch")


def seed_for(opening_id: str, candidate: str, seat: int) -> int:
    payload = f"seed604\0{opening_id}\0{candidate}\0{seat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def validate_search_contract(search: Any) -> None:
    if search != EXPECTED_SEARCH:
        raise ValueError("frozen_search_contract_mismatch")


def validate_statistics_contract(statistics: Any) -> None:
    expected = {
        "unit": "opening group",
        "clusters": 1024,
        "resamples": 20000,
        "seed": 604,
        "quantile_method": "linear",
        "primary_lower_quantile": 0.05,
        "primary_strict_lower_bound_gt": 0.5,
        "secondary_quantiles": [0.025, 0.975],
        "secondary_strict_lower_endpoint_gt": 0.0,
        "interpretation": {
            "both_pass": "absolute strength and additional optimization benefit established",
            "primary_only": "absolute strength established; incremental improvement not established",
            "secondary_only": "incremental improvement established; absolute strength not established",
            "neither": "neither claim established",
        },
    }
    if statistics != expected:
        raise ValueError("frozen_statistical_contract_mismatch")


def validate_assignment(row: dict[str, Any], expected: dict[str, Any]) -> None:
    if row != expected:
        raise ValueError("assignment_identity_or_checkpoint_mismatch")


def validate_assignment_ledger(
    rows: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> None:
    ids = [row.get("game_id") for row in rows]
    if len(rows) != 4096 or len(set(ids)) != len(ids):
        raise ValueError("assignment_manifest_missing_duplicate_or_changed")
    if rows != expected:
        raise ValueError("assignment_manifest_missing_duplicate_or_changed")


def validate_candidate_value_sources(
    command: list[str],
    assignment: dict[str, Any],
    root: Path = ROOT,
    opening_path: Path | None = None,
) -> None:
    options = {
        flag: command[index + 1]
        for index, flag in enumerate(command[:-1])
        if flag.startswith("--")
    }
    command_flags = [flag for flag in command if flag.startswith("--")]
    expected_flags = {
        "--challenger",
        "--current",
        "--challenger-policy-artifact",
        "--challenger-value-artifact",
        "--current-policy-artifact",
        "--current-value-artifact",
        "--games",
        "--games-per-opening",
        "--workers",
        "--challenger-simulations",
        "--current-simulations",
        "--c-puct",
        "--seed-contract",
        "--challenger-search-options-json",
        "--current-search-options-json",
        "--random-opening-plies",
        "--max-moves",
        "--challenger-starts",
        "--exact-root-solve-threshold",
        "--exact-root-native-probe",
        "--exact-root-tablebase",
        "--opening-prefixes-jsonl",
        "--suite-sha256",
        "--game-jsonl",
        "--out",
        "--seed",
    }
    if set(options) != expected_flags or len(command_flags) != len(options):
        raise ValueError("arena_command_flag_set_mismatch")
    candidate = str(root / (OUT / "runtime-artifacts" / assignment["candidate"]))
    serving = str(root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4")
    expected = {
        "--challenger-policy-artifact": candidate,
        "--challenger-value-artifact": serving,
        "--current-policy-artifact": serving,
        "--current-value-artifact": serving,
        "--challenger-starts": str(assignment["candidate_seat"]),
        "--challenger-simulations": "128",
        "--current-simulations": "128",
        "--workers": "1",
        "--max-moves": "200",
        "--exact-root-solve-threshold": "16",
        "--exact-root-native-probe": str(
            root / "model-artifact/runtime/kalah_v1_tablebase"
        ),
        "--exact-root-tablebase": str(root / "model-artifact/runtime/kalah_v1_21.kvtb"),
    }
    if any(options.get(flag) != value for flag, value in expected.items()):
        raise ValueError("evaluator_composition_or_search_command_mismatch")
    if options["--challenger-value-artifact"] == candidate:
        raise ValueError("candidate_value_source_replaced_p_e4")
    expected_search_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
        "tactical_root_bias": 0.0,
    }
    for flag in ("--challenger-search-options-json", "--current-search-options-json"):
        try:
            observed = json.loads(options[flag])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("search_options_missing_or_invalid") from error
        if observed != expected_search_options:
            raise ValueError("search_options_or_noise_contract_mismatch")
    if options.get("--seed") != str(assignment["game_seed"]):
        raise ValueError("deterministic_game_seed_mismatch")
    if opening_path is not None:
        if options.get("--opening-prefixes-jsonl") != str(opening_path):
            raise ValueError("assigned_opening_file_path_mismatch")
        opening_lines = opening_path.read_bytes().splitlines()
        if len(opening_lines) != 1 or len(assignment["prefix_moves"]) != 12:
            raise ValueError("assigned_opening_file_count_or_prefix_length_mismatch")
        opening_document = json.loads(opening_lines[0])
        if opening_document != {"prefix_moves": assignment["prefix_moves"]}:
            raise ValueError("assigned_opening_bytes_mismatch")
        if options.get("--suite-sha256") != sha(opening_path):
            raise ValueError("assigned_opening_hash_mismatch")
        if options.get("--game-jsonl") != str(opening_path.parent / "game.jsonl"):
            raise ValueError("game_output_path_mismatch")
        if options.get("--out") != str(opening_path.parent / "report.json"):
            raise ValueError("report_output_path_mismatch")
    common = {
        "--challenger": candidate,
        "--current": serving,
        "--games": "1",
        "--games-per-opening": "1",
        "--c-puct": "1.25",
        "--seed-contract": "azlite_eval_seed_v2",
        "--random-opening-plies": "0",
    }
    if any(options.get(flag) != value for flag, value in common.items()):
        raise ValueError("arena_command_contract_mismatch")


def validate_arena_report(
    report: dict[str, Any],
    assignment: dict[str, Any],
    game_row: dict[str, Any],
    root: Path = ROOT,
) -> None:
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
        != str(root / (OUT / "runtime-artifacts" / assignment["candidate"]))
        or notes.get("current_path")
        != str(root / "docs/data/seed552-c108-serving-p/runtime-artifacts/P-E4")
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
        != "d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48"
        or profile.get("exact_root_tablebase_sha256")
        != "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
    ):
        raise ValueError("arena_report_search_contract_mismatch")
    wins = int(game_row.get("winner") == "challenger")
    losses = int(game_row.get("winner") == "current")
    draws = int(game_row.get("winner") == "draw")
    expected_score = wins + 0.5 * draws
    if (
        report.get("wins") != wins
        or report.get("losses") != losses
        or report.get("draws") != draws
        or report.get("score") != expected_score
    ):
        raise ValueError("arena_report_outcome_mismatch")


def expected_assignments(root: Path = ROOT) -> list[dict[str, Any]]:
    manifest = json.loads((root / OPENING_PATH).read_bytes())
    openings = manifest.get("openings")
    if not isinstance(openings, list) or len(openings) != 1024:
        raise ValueError("registered_opening_count_mismatch")
    records = []
    for index, opening in enumerate(openings):
        if (
            opening.get("opening_id") != f"seed604-opening-{index:04d}"
            or len(opening.get("prefix_moves", [])) != 12
        ):
            raise ValueError("registered_opening_order_mismatch")
        for candidate, candidate_sha in (
            (
                "B512",
                "bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c",
            ),
            (
                "B2048",
                "e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874",
            ),
        ):
            for seat in (0, 1):
                records.append(
                    {
                        "game_id": f"seed604-{candidate}-{index:04d}-p{seat}",
                        "opening_index": index,
                        "opening_id": opening["opening_id"],
                        "opening_identity": opening["canonical_identity"],
                        "candidate": candidate,
                        "candidate_checkpoint_sha256": candidate_sha,
                        "opponent": "P-E4",
                        "opponent_checkpoint_sha256": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
                        "candidate_seat": seat,
                        "game_seed": seed_for(opening["opening_id"], candidate, seat),
                        "prefix_moves": opening["prefix_moves"],
                    }
                )
    return records


def verify_game(assignment: dict[str, Any], game_row: dict[str, Any]) -> dict[str, Any]:
    """Replay every move independently and verify seat-relative terminal/cap score."""
    seat = assignment.get("candidate_seat")
    if seat not in (0, 1):
        raise ValueError("invalid_candidate_seat")
    game = KalahGame.from_state(dict(INITIAL))
    for relative_move in assignment["prefix_moves"]:
        if game.over() or relative_move not in game.possible_moves():
            raise ValueError("illegal_registered_opening")
        if not game.move(game.pit_index(int(relative_move))):
            raise ValueError("registered_opening_rejected")
    if game.over() or game.to_state().get("current_player") not in (0, 1):
        raise ValueError("invalid_opening_state")
    opening_identity = hashlib.sha256(canonical(game.to_state())).hexdigest()
    provenance = {
        "game_index": 0,
        "game_within_opening": 0,
        "opening_index": 0,
        "opening_state_hash": opening_identity,
        "opening_prefix_moves": assignment["prefix_moves"],
        "opening_applied_prefix_length": len(assignment["prefix_moves"]),
        "challenger_player": seat,
    }
    if opening_identity != assignment.get("opening_identity") or any(
        game_row.get(key) != value for key, value in provenance.items()
    ):
        raise ValueError("candidate_opponent_seat_inversion")
    trajectory = game_row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("missing_game_trajectory")
    tokens = trajectory.split(",") if trajectory else []
    if len(tokens) != int(game_row.get("game_length", -1)) or len(tokens) > 200:
        raise ValueError("trajectory_length_mismatch")
    absolute_moves = []
    for ply, token in enumerate(tokens):
        try:
            absolute = int(token)
        except ValueError as error:
            raise ValueError("malformed_trajectory_move") from error
        if game.over() or absolute not in range(12):
            raise ValueError(f"illegal_trajectory:{ply}")
        if absolute // 6 != game.current_player:
            raise ValueError(f"player_to_move_mismatch:{ply}")
        relative = absolute % 6
        if relative not in game.possible_moves() or not game.move(absolute):
            raise ValueError(f"illegal_trajectory:{ply}")
        absolute_moves.append(absolute)
    capped = not game.over()
    if capped and len(absolute_moves) != 200:
        raise ValueError("nonterminal_game_not_at_cap")
    if not capped and len(absolute_moves) > 200:
        raise ValueError("terminal_game_exceeds_cap")
    margin, winner, score = store_margin_outcome(game.captured_seeds, seat)
    if game_row.get("winner") != winner or int(game_row.get("margin", 2**31)) != margin:
        raise ValueError("game_outcome_or_capped_store_margin_mismatch")
    return {
        "winner": winner,
        "margin": margin,
        "game_length": len(absolute_moves),
        "capped": capped,
        "score": score,
        "trajectory_sha256": hashlib.sha256(trajectory.encode()).hexdigest(),
    }


def verify_manifest(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    freeze_path, assignment_path = root / FREEZE, root / ASSIGNMENTS
    if not freeze_path.is_file() or not assignment_path.is_file():
        raise ValueError("evaluation_freeze_or_assignment_manifest_missing")
    freeze = json.loads(freeze_path.read_bytes())
    assignment_doc = json.loads(assignment_path.read_bytes())
    if assignment_doc.get("schema") != "seed604-game-assignment-manifest-v1":
        raise ValueError("assignment_manifest_schema_mismatch")
    rows = assignment_doc.get("assignments")
    expected_rows = expected_assignments(root)
    validate_assignment_ledger(rows, expected_rows)
    expected_fields = {
        "schema": "seed604-evaluation-execution-freeze-v1",
        "status": "frozen-awaiting-separate-evaluation-authorization",
        "evaluation_authorized": False,
    }
    if any(freeze.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("evaluation_freeze_status_invalid")
    if freeze.get("assignment_manifest", {}).get("sha256") != sha(assignment_path):
        raise ValueError("assignment_manifest_freeze_hash_mismatch")
    if freeze.get("assignment_manifest", {}).get("game_count") != 4096:
        raise ValueError("assignment_count_freeze_mismatch")
    if (
        freeze.get("assignment_manifest", {}).get("opening_count") != 1024
        or freeze.get("assignment_manifest", {}).get("seed_derivation")
        != "SHA-256 of UTF-8 seed604 NUL opening_id NUL candidate NUL seat; first 8 bytes big-endian modulo 2^31"
    ):
        raise ValueError("assignment_seed_contract_mismatch")
    for source, expected in freeze.get("source_sha256", {}).items():
        if source not in SOURCE_PATHS or sha(root / source) != expected:
            raise ValueError(f"frozen_source_hash_mismatch:{source}")
    if set(freeze.get("source_sha256", {})) != set(SOURCE_PATHS):
        raise ValueError("frozen_source_set_mismatch")
    if freeze.get("protocol_document_sha256") != sha(root / (OUT / "README.md")):
        raise ValueError("protocol_document_hash_mismatch")
    if freeze.get("output") != {
        "freeze_path": str(FREEZE),
        "readiness_path": str(READINESS),
        "assignment_manifest_path": str(ASSIGNMENTS),
        "authorization_path": str(AUTHORIZATION),
        "results_directory": str(RESULTS),
        "result_schema": "seed604-game-result-v1",
        "attempt_directory_pattern": "results/<game_id>/attempts/attempt-NNNN/",
        "analysis_path": str(ANALYSIS),
        "analysis_schema": "seed604-cluster-bootstrap-analysis-v1",
        "create_only": True,
    }:
        raise ValueError("frozen_output_contract_mismatch")
    registration = freeze.get("registration", {})
    for key, relative in (
        ("native_probe_sha256", "model-artifact/runtime/kalah_v1_tablebase"),
        ("tablebase_sha256", "model-artifact/runtime/kalah_v1_21.kvtb"),
    ):
        if sha(root / relative) != registration.get(key):
            raise ValueError(f"frozen_exact_root_asset_mismatch:{key}")
    validate_search_contract(freeze.get("search"))
    validate_statistics_contract(freeze.get("statistics"))
    runtime = freeze.get("runtime", {})
    import torch

    if (
        runtime.get("python") != sys.version
        or runtime.get("numpy") != np.__version__
        or runtime.get("torch") != str(torch.__version__)
        or runtime.get("platform") != platform.platform()
        or runtime.get("machine") != platform.machine()
        or runtime.get("dependency_lock_sha256")
        != sha(root / "ml/alphazero_lite/requirements-seed599-evaluation.txt")
        or runtime.get("device") != "cpu"
        or runtime.get("workers_per_game") != 1
        or runtime.get("pair_parallelism") != 18
        or runtime.get("registered_cpus") != 18
        or runtime.get("registered_memory_gib") != 64
        or runtime.get("environment") != EXPECTED_ENVIRONMENT
    ):
        raise ValueError("evaluation_runtime_identity_mismatch")
    authorized = False
    if (root / AUTHORIZATION).exists():
        authorization = json.loads((root / AUTHORIZATION).read_bytes())
        if (
            authorization.get("schema") != "seed604-evaluation-authorization-v1"
            or authorization.get("status") != "authorized-for-evaluation"
            or authorization.get("evaluation_freeze_sha256") != sha(freeze_path)
            or authorization.get("authorization", {}).get("type")
            != "explicit-project-owner-authorization"
            or not isinstance(
                authorization.get("authorization", {}).get("authorized_by"), str
            )
            or not authorization.get("authorization", {})
            .get("authorized_by", "")
            .strip()
            or authorization.get("authorization", {}).get("run_games") is not True
        ):
            raise ValueError("evaluation_authorization_binding_invalid")
        authorized = True
    verify_bound_inputs(root, freeze)
    readiness_path = root / READINESS
    if not readiness_path.is_file():
        raise ValueError("evaluation_readiness_evidence_missing")
    readiness = json.loads(readiness_path.read_bytes())
    if (
        readiness.get("schema") != "seed604-evaluation-readiness-evidence-v1"
        or readiness.get("execution_freeze_sha256") != sha(freeze_path)
        or readiness.get("evaluation_authorized") is not False
        or readiness.get("games_started") is not False
        or readiness.get("registered_openings") != 1024
        or readiness.get("assignment_count") != 4096
        or readiness.get("tablebase", {}).get("sha256")
        != "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
        or readiness.get("tablebase", {}).get("size") != 709635072
        or readiness.get("tablebase", {}).get("materialization_provenance")
        != freeze.get("materialization")
    ):
        raise ValueError("evaluation_readiness_binding_mismatch")
    completed, capped = 0, 0
    observed_results: list[dict[str, Any]] = []
    if (root / RESULTS).exists():
        by_id = {row["game_id"]: row for row in rows}
        seen = set()
        for result_path in sorted((root / RESULTS).glob("*/result.json")):
            result = json.loads(result_path.read_bytes())
            game_id = result.get("game_id")
            if game_id not in by_id or game_id in seen:
                raise ValueError("unexpected_or_duplicate_game_result")
            if any(result.get(key) != value for key, value in by_id[game_id].items()):
                raise ValueError("game_result_assignment_mismatch")
            required_result_fields = set(by_id[game_id]) | {
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
            if (
                set(result) != required_result_fields
                or result.get("schema") != "seed604-game-result-v1"
            ):
                raise ValueError("game_result_schema_mismatch")
            attempt_relative = Path(result.get("attempt_directory", ""))
            if attempt_relative.is_absolute() or ".." in attempt_relative.parts:
                raise ValueError("invalid_attempt_directory")
            attempt_directory = result_path.parent / attempt_relative
            game_raw = (attempt_directory / "game.jsonl").read_text().splitlines()
            if len(game_raw) != 1:
                raise ValueError("game_result_artifact_count_mismatch")
            game_row = json.loads(game_raw[0])
            verified = verify_game(by_id[game_id], game_row)
            if any(result.get(key) != value for key, value in verified.items()):
                raise ValueError("independent_game_result_mismatch")
            validate_result_hashes(
                result,
                (attempt_directory / "game.jsonl").read_bytes(),
                (attempt_directory / "report.json").read_bytes(),
                (attempt_directory / "command.json").read_bytes(),
            )
            validate_candidate_value_sources(
                json.loads((attempt_directory / "command.json").read_bytes()),
                by_id[game_id],
                root,
                attempt_directory / "opening.jsonl",
            )
            if not (attempt_directory / "report.json").is_file():
                raise ValueError("game_report_missing")
            validate_arena_report(
                json.loads((attempt_directory / "report.json").read_bytes()),
                by_id[game_id],
                game_row,
                root,
            )
            seen.add(game_id)
            completed += 1
            capped += int(verified["capped"])
            observed_results.append({**by_id[game_id], **verified})
        # A result-bearing partial run is evidence, not a complete experiment.
        for attempt_root in (root / RESULTS).glob("*/attempts/*"):
            if not attempt_root.is_dir():
                continue
            if (
                (attempt_root / "game.jsonl").exists()
                or (attempt_root / "report.json").exists()
            ) and not (attempt_root.parent.parent / "result.json").exists():
                raise ValueError("unreconciled_attempt_evidence_preserved")
    if (root / ANALYSIS).exists():
        if completed != 4096:
            raise ValueError("analysis_artifact_for_incomplete_ledger")
        summary = json.loads((root / ANALYSIS).read_bytes())
        independent = analyze_independently(observed_results)
        if (
            summary.get("schema") != "seed604-cluster-bootstrap-analysis-v1"
            or summary.get("games") != 4096
            or summary.get("opening_clusters") != 1024
            or summary.get("capped_games") != capped
            or summary.get("bootstrap")
            != {
                "resamples": 20000,
                "seed": 604,
                "quantile_method": "linear",
                "paired_opening_indices": True,
                "seat_outcomes_preserved": True,
            }
            or summary.get("B512_score") != independent["B512_score"]
            or summary.get("B2048_score") != independent["B2048_score"]
            or summary.get("B2048_primary_lower_bound") != independent["primary_lower"]
            or summary.get("B2048_minus_B512_interval")
            != independent["secondary_interval"]
            or summary.get("primary_pass") is not (independent["primary_lower"] > 0.5)
            or summary.get("secondary_pass")
            is not (independent["secondary_interval"][0] > 0.0)
            or summary.get("interpretation")
            != (
                "both_pass"
                if independent["primary_lower"] > 0.5
                and independent["secondary_interval"][0] > 0.0
                else "primary_only"
                if independent["primary_lower"] > 0.5
                else "secondary_only"
                if independent["secondary_interval"][0] > 0.0
                else "neither"
            )
        ):
            raise ValueError("independent_analysis_artifact_mismatch")
    elif completed == 4096:
        # The result ledger may be frozen before analysis is separately requested.
        pass
    return {
        "passed": True,
        "freeze_sha256": sha(freeze_path),
        "assignment_count": len(rows),
        "completed_games": completed,
        "capped_games": capped,
        "evaluation_authorized": authorized,
        "evaluation_games_started_by_verifier": False,
    }


def analyze_independently(game_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent registered paired-opening bootstrap implementation."""
    if len(game_records) != 4096:
        raise ValueError("incomplete_game_ledger")
    expected = {
        (i, arm, seat)
        for i in range(1024)
        for arm in ("B512", "B2048")
        for seat in (0, 1)
    }
    grouped: dict[tuple[int, str], dict[int, float]] = {}
    for row in game_records:
        key = (row["opening_index"], row["candidate"])
        seat = row["candidate_seat"]
        if (
            key not in {(i, arm) for i, arm, _ in expected}
            or seat not in (0, 1)
            or seat in grouped.setdefault(key, {})
        ):
            raise ValueError("duplicate_or_unknown_game")
        grouped[key][seat] = float(row["score"])
    if any(set(seats) != {0, 1} for seats in grouped.values()) or len(grouped) != 2048:
        raise ValueError("missing_seat_swapped_result")
    means = {
        arm: np.asarray(
            [(grouped[(i, arm)][0] + grouped[(i, arm)][1]) / 2 for i in range(1024)]
        )
        for arm in ("B512", "B2048")
    }
    random = np.random.default_rng(604)
    draw = random.integers(0, 1024, size=(20000, 1024))
    primary = means["B2048"][draw].mean(axis=1)
    paired_delta = (means["B2048"] - means["B512"])[draw].mean(axis=1)
    return {
        "B512_score": float(means["B512"].mean()),
        "B2048_score": float(means["B2048"].mean()),
        "primary_lower": float(np.quantile(primary, 0.05, method="linear")),
        "secondary_interval": [
            float(x) for x in np.quantile(paired_delta, [0.025, 0.975], method="linear")
        ],
    }


def paired_bootstrap_reference(
    b512: np.ndarray,
    b2048: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Independent paired index implementation, parameterized for synthetic checks."""
    if b512.ndim != 1 or b2048.shape != b512.shape or b512.size < 1 or resamples < 1:
        raise ValueError("bootstrap_cluster_shape")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, b512.size, size=(resamples, b512.size))
    score_draws = np.mean(np.take(b2048, selected), axis=1)
    delta = b2048 - b512
    paired_draws = np.mean(np.take(delta, selected), axis=1)
    low = float(np.quantile(score_draws, 0.05, method="linear"))
    interval = [
        float(x) for x in np.quantile(paired_draws, [0.025, 0.975], method="linear")
    ]
    return low, interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.preflight:
        parser.error("only read-only --preflight is supported")
    print(json.dumps(verify_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
