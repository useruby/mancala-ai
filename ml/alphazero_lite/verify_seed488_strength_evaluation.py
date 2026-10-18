"""Independent read-only verification for seed488 strength outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed482_fresh_confirmation import encode_state

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("docs/data/seed488-frozen-strength-evaluation")
OPENINGS = 512
REPLICATES = 20_000
SEED = 488
EXTENDED_EXCLUSION_SHA256 = (
    "f419d615724f8f741fb4a58bb060901d4359e4f975ca64bfa184da4e892c10cd"
)
EXTENDED_UNION_SHA256 = (
    "d7c3d782181af654d56da2d02b091047df9a9cc5f78795b2cb687c9b528dfcf6"
)
SEED479_RECEIPTS = {
    "docs/data/seed479-population-ablation/receipt.json": "f7c5edd6e6bedd496b145721edcd77196ccdb32e127340c81b1546a6db816314",
    "docs/data/seed479-population-ablation/registration.json": "4e2a03af089593f1ae32abc26f8a407bb5d980be7100776c091224a7ae0b2fb4",
    "docs/data/seed479-population-ablation/evidence.json": "9056f7654b1dca824eccc783c04b7ad1a68140667e28068f665d304eea9ce2fe",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _points(winner: str) -> float:
    if winner in ("challenger", "F"):
        return 1.0
    if winner == "draw":
        return 0.5
    if winner in ("current", "initializer"):
        return 0.0
    raise ValueError("outcome_unknown")


def _replay_game(
    opening: dict[str, Any], row: dict[str, Any], seat: int
) -> tuple[str, int, int]:
    """Replay relative prefix/trajectory exactly as the bound arena writer does."""
    game = KalahGame.from_state(
        {
            "player_pits": [4] * 6,
            "opponent_pits": [4] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    prefix = opening.get("prefix_moves")
    if not isinstance(prefix, list):
        raise ValueError("opening_prefix_missing")
    for move in prefix:
        if game.over() or int(move) not in game.possible_moves():
            raise ValueError("opening_prefix_replay")
        if not game.move(game.pit_index(int(move))):
            raise ValueError("opening_prefix_replay")
    if game.to_state() != opening.get("state"):
        raise ValueError("opening_prefix_state")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("trajectory_missing")
    moves = [int(value) for value in trajectory.split(",") if value != ""]
    if len(moves) > 200 or int(row.get("game_length", -1)) != len(moves):
        raise ValueError("trajectory_length_or_cap")
    for ply, move in enumerate(moves):
        if game.over() or move not in game.possible_moves():
            raise ValueError(f"trajectory_illegal:{ply}")
        acting_player = game.current_player
        if not game.move(game.pit_index(move)):
            raise ValueError(f"trajectory_replay:{ply}")
        # Extra turns are represented by the unchanged current_player, not by
        # alternating seats in the recorded action stream.
        if game.current_player not in (acting_player, 1 - acting_player):
            raise ValueError("trajectory_turn_transition")
    if game.over() and len(moves) < 200:
        pass
    elif len(moves) != 200:
        raise ValueError("trajectory_ended_before_terminal_or_cap")
    margin = int(game.captured_seeds[seat] - game.captured_seeds[1 - seat])
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    if row.get("winner") != winner or int(row.get("margin", 10**9)) != margin:
        raise ValueError("trajectory_outcome_mismatch")
    return winner, margin, len(moves)


def _verify_artifact_against_frozen_state(
    root: Path, out: Path, arm: str, manifest: dict[str, str]
) -> None:
    """Compare exports to frozen states using read-only inputs and no file writes."""
    import torch

    from ml.alphazero_lite import train

    artifact = out / "runtime-artifacts" / arm
    if arm == "initializer":
        frozen_path = (
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, frozen_path)
        expected = train.checkpoint_from_model(model)
    else:
        trajectory = (
            root / "docs/data/seed479-population-ablation/trajectory-evidence.npz"
        )
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        parameters = list(model.parameters())
        with np.load(trajectory, allow_pickle=False) as archive:
            for index, parameter in enumerate(parameters):
                source = archive[f"F_step_16_post_parameter_{index:02d}"]
                if tuple(source.shape) != tuple(parameter.shape):
                    raise ValueError("frozen_F_parameter_shape")
                parameter.data.copy_(torch.from_numpy(source.copy()))
        expected = train.checkpoint_from_model(model)
    with np.load(artifact / "model.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f"artifact_model_keys:{arm}")
        for key, value in expected.items():
            if not np.array_equal(saved[key], value):
                raise ValueError(f"artifact_model_state:{arm}:{key}")
    weights = json.loads((artifact / "weights.json").read_text(encoding="utf-8"))
    if set(weights) != set(expected):
        raise ValueError(f"artifact_weights_keys:{arm}")
    for key, value in expected.items():
        if not np.array_equal(np.asarray(weights[key], dtype=np.float32), value):
            raise ValueError(f"artifact_weights_state:{arm}:{key}")
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    artifact_bindings = metadata.get("artifacts", {})
    if (
        metadata.get("schema_version") != "azlite_model_v1"
        or metadata.get("version") != f"seed488-{arm}"
        or metadata.get("game") != "kalah"
        or metadata.get("rules_version") != "kalah_v1"
        or metadata.get("input_encoding") != "kalah_v3"
        or metadata.get("feature_count") != 27
        or metadata.get("policy_size") != 6
        or metadata.get("architecture", {}).get("model_type") != "residual_v3"
        or artifact_bindings.get("weights_file") != "weights.json"
        or artifact_bindings.get("weights_fallback_file") != "model.npz"
        or artifact_bindings.get("weights_sha256") != _sha(artifact / "model.npz")
        or artifact_bindings.get("weights_json_sha256")
        != _sha(artifact / "weights.json")
    ):
        raise ValueError(f"artifact_metadata:{arm}")
    policy_path = root / "model-artifact/current/search_policy.json"
    if (artifact / "search_policy.json").read_bytes() != policy_path.read_bytes():
        raise ValueError(f"artifact_search_policy:{arm}")
    expected_files = {
        "model.npz",
        "weights.json",
        "metadata.json",
        "search_policy.json",
    }
    if set(manifest) != expected_files:
        raise ValueError(f"artifact_manifest_files:{arm}")


def _verify_supersession(out: Path) -> None:
    supersession = json.loads((out / "supersession.json").read_text(encoding="utf-8"))
    if (
        supersession.get("status") != "seed488_preparation_superseded_before_execution"
        or supersession.get("supersedes_pr") != 487
        or supersession.get("no_outcomes_existed_at_supersession") is not True
    ):
        raise ValueError("supersession_record")
    for original, binding in supersession.get("preserved_originals", {}).items():
        preserved = out / binding["preserved_as"]
        if (
            _sha(preserved) != binding["sha256"]
            or preserved.stat().st_mtime_ns != binding["mtime_ns"]
        ):
            raise ValueError(f"superseded_evidence_changed:{original}")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    out = root / OUT
    reg_path = out / "registration.json"
    suite_path = out / "openings.jsonl"
    receipt_path = out / "receipt.json"
    registration = json.loads(reg_path.read_text(encoding="utf-8"))
    suite_digest = _sha(suite_path)
    if registration.get("schema") != "seed488-frozen-strength-evaluation-v2":
        raise ValueError("registration_schema")
    if registration.get("status") != "frozen_before_execution":
        raise ValueError("registration_not_frozen")
    if registration.get("fresh_confirmation_authorized") is not False:
        raise ValueError("confirmation_gate_open")
    if registration.get("supersedes") != (
        "PR #487 seed488_prepared_not_executed registration and suite; this is a preparation-only correction before any outcomes"
    ):
        raise ValueError("supersession_declaration")
    runtime = registration.get("runtime", {})
    expected_search_options = {
        "fpu_mode": "zero",
        "reuse_subtree": False,
        "normalize_values": False,
        "root_policy_mode": "deterministic",
        "tactical_root_bias": 0.0,
        "root_temperature": 0.0,
        "root_prior_transform": None,
        "random_opening_plies": 0,
    }
    if (
        runtime.get("simulations_per_move") != 1200
        or runtime.get("c_puct") != 1.25
        or runtime.get("workers") != 1
        or runtime.get("max_moves") != 200
        or runtime.get("max_moves_policy")
        != "At 200 decision plies, stop without sweeping remaining pits; compare captured stores at the cap. A game ending earlier uses Kalah terminal sweeping. Recorded trajectory length must be terminal before 200 or exactly 200."
        or runtime.get("search_options") != expected_search_options
        or runtime.get("seed_contract")
        != "azlite_eval_seed_v2; per-opening base seed = 488 + opening_index; arena's deterministic seed contract binds opening state, seat, game-within-opening, and ply"
    ):
        raise ValueError("runtime_protocol_mismatch")
    suite_contract = registration.get("opening_suite", {})
    if (
        suite_contract.get("generator_seed") != SEED
        or suite_contract.get("count") != OPENINGS
        or suite_contract.get("games_per_opening") != 2
        or suite_contract.get("game_count") != OPENINGS * 2
        or suite_contract.get("eligible_against_reconstructable_exclusions") is not True
        or "non-overlap with unresolved seed461 continuation population cannot be certified"
        not in suite_contract.get("exclusions", {}).get("limitation", "")
    ):
        raise ValueError("opening_protocol_mismatch")
    if registration.get("bootstrap") != {
        "unit": "512 opening clusters; resample with replacement and carry both seat games together",
        "replicates": REPLICATES,
        "seed": SEED,
        "advance_rule": "advance only if one-sided 95% percentile lower score bound > 0.50",
        "two_sided_interval": "95% percentile interval also reported descriptively",
    }:
        raise ValueError("bootstrap_protocol_mismatch")
    if (
        registration.get("primary_score")
        != "F wins + 0.5 * draws, divided by all 1,024 games"
        or registration.get("promotion") != "No promotion from this experiment alone."
        or registration.get("seed479_and_480_classification")
        != "promising exploratory results; preserved as exploratory"
        or registration.get("interpretation")
        != "exploratory direct strength evaluation on a newly registered suite; not fresh confirmation or an independently untouched holdout"
    ):
        raise ValueError("decision_protocol_mismatch")
    if registration.get("trajectory_contract") != (
        "arena.py at the runtime source hash; opening prefixes and trajectories are player-relative v2 moves converted by KalahGame.pit_index at each ply; rules use KalahGame.move including extra turns and terminal sweep; stored margin is challenger captured store minus current captured store; winner is the sign of that margin"
    ):
        raise ValueError("trajectory_contract")
    if registration["opening_suite"].get("opening_suite_sha256") != suite_digest:
        raise ValueError("suite_digest")
    impl = root / "ml/alphazero_lite/seed488_strength_evaluation.py"
    if registration.get("implementation_sha256") != _sha(impl):
        raise ValueError("implementation_binding")
    if registration.get("independent_verifier_sha256") != _sha(
        root / "ml/alphazero_lite/verify_seed488_strength_evaluation.py"
    ):
        raise ValueError("verifier_binding")
    if registration.get("synthetic_tests_sha256") != _sha(
        root / "ml/alphazero_lite/test_seed488_strength_evaluation.py"
    ):
        raise ValueError("synthetic_test_binding")
    for relative, digest in registration.get("generator_dependency_sha256", {}).items():
        if _sha(root / relative) != digest:
            raise ValueError(f"generator_dependency_binding:{relative}")
    expected_runtime = registration["runtime"].get("identities_sha256", {})
    if not expected_runtime:
        raise ValueError("runtime_identity_inventory")
    if registration["runtime"].get("python_version") != sys.version:
        raise ValueError("python_runtime_version")
    if registration["runtime"].get("numpy_version") != np.__version__:
        raise ValueError("numpy_runtime_version")
    for relative, digest in expected_runtime.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"runtime_binding:{relative}")
    f_endpoint = registration["arms"]["F"]
    trajectory_path = root / f_endpoint["checkpoint_source"]
    if _sha(trajectory_path) != f_endpoint["trajectory_sha256"]:
        raise ValueError("f_trajectory_binding")
    if (
        _sha(
            root
            / "docs/data/seed429-canonical-policy-normalization/training-checkpoints/initializer.npz"
        )
        != registration["arms"]["initializer"]["checkpoint_sha256"]
    ):
        raise ValueError("initializer_binding")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        parameter_arrays = [
            archive[f"F_step_16_post_parameter_{i:02d}"] for i in range(22)
        ]
    flattened = np.concatenate([x.reshape(-1).astype("<f4") for x in parameter_arrays])
    if (
        hashlib.sha256(flattened.tobytes()).hexdigest()
        != f_endpoint["parameter_state_sha256"]
    ):
        raise ValueError("f_parameter_identity")
    if registration["arms"].get("seed479_receipts") != {
        "receipt_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/receipt.json"
        ],
        "registration_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/registration.json"
        ],
        "evidence_sha256": SEED479_RECEIPTS[
            "docs/data/seed479-population-ablation/evidence.json"
        ],
    }:
        raise ValueError("seed479_receipt_binding")
    for relative, digest in SEED479_RECEIPTS.items():
        if _sha(root / relative) != digest:
            raise ValueError(f"seed479_receipt_source:{relative}")
    openings = _rows(suite_path)
    if len(openings) != OPENINGS or [int(x["opening_index"]) for x in openings] != list(
        range(OPENINGS)
    ):
        raise ValueError("opening_coverage")
    if any(
        row.get("opening_contract") != "arena_player_relative_v2" for row in openings
    ):
        raise ValueError("opening_contract")
    if all(
        row.get("opening_contract") == "arena_player_relative_v2" for row in openings
    ):
        if (
            _sha(root / "docs/data/seed482-exclusion-manifest.json.gz")
            != "129391fc88118a99552029faafb1880f1daed62476b2d1eafd7e931f621dc60d"
        ):
            raise ValueError("base_exclusion_file_hash")
        if (
            _sha(root / "docs/data/seed483-historical-probe-identities.json.gz")
            != "cb11ea5912c530f4a6c01a913b9e5083c454f639d24669a4675ef6d1c2e45c9b"
        ):
            raise ValueError("reconstructed_exclusion_file_hash")
        extended_path = root / "docs/data/seed484-extended-probe-union.json.gz"
        if _sha(extended_path) != EXTENDED_EXCLUSION_SHA256:
            raise ValueError("extended_exclusion_file_hash")
        excluded_base = json.loads(
            gzip.decompress(
                (root / "docs/data/seed482-exclusion-manifest.json.gz").read_bytes()
            )
        )
        excluded_add = json.loads(
            gzip.decompress(
                (
                    root / "docs/data/seed483-historical-probe-identities.json.gz"
                ).read_bytes()
            )
        )
        excluded_canonical = {
            row["canonical_identity"] for row in excluded_base["identities"]
        }
        excluded_canonical.update(
            row["canonical_identity"] for row in excluded_add["identities"]
        )
        excluded_float32 = {
            row["float32_identity"] for row in excluded_base["identities"]
        }
        excluded_float32.update(
            row["float32_identity"] for row in excluded_add["identities"]
        )
        extended = json.loads(gzip.decompress(extended_path.read_bytes()))
        if (
            extended.get("status") != "incomplete_seed461_replay_gap"
            or extended.get("final_union_pair_count") != 149_756
            or extended.get("final_union_pairs_sha256") != EXTENDED_UNION_SHA256
        ):
            raise ValueError("extended_exclusion_union_binding")
        reconstructed_pairs = {
            (row["canonical_identity"], row["float32_identity"])
            for row in excluded_base["identities"]
        }
        reconstructed_pairs.update(
            (row["canonical_identity"], row["float32_identity"])
            for row in excluded_add["identities"]
        )
        reconstructed_pairs.update(
            (row["canonical_identity"], row["float32_identity"])
            for row in extended["seed467_additional_identities"]
        )
        pair_payload = json.dumps(
            sorted([list(pair) for pair in reconstructed_pairs]), separators=(",", ":")
        ).encode()
        if (
            len(reconstructed_pairs) != 149_756
            or hashlib.sha256(pair_payload).hexdigest() != EXTENDED_UNION_SHA256
            or suite_contract.get("exclusions", {}).get("seed484_extended_union_sha256")
            != EXTENDED_UNION_SHA256
            or suite_contract.get("exclusions", {}).get(
                "seed484_extended_union_file_sha256"
            )
            != EXTENDED_EXCLUSION_SHA256
        ):
            raise ValueError("extended_exclusion_union_reconstruction")
        excluded_canonical.update(
            pair["canonical_identity"]
            for pair in extended["seed467_additional_identities"]
        )
        excluded_float32.update(
            pair["float32_identity"]
            for pair in extended["seed467_additional_identities"]
        )
        seen_states: set[str] = set()
        seen_float32: set[str] = set()
        for index, row in enumerate(openings):
            game = KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
            absolute_prefix = row.get("source_prefix_moves_absolute")
            if not isinstance(absolute_prefix, list) or len(absolute_prefix) != len(
                row["prefix_moves"]
            ):
                raise ValueError("opening_absolute_prefix")
            for relative, absolute in zip(
                row["prefix_moves"], absolute_prefix, strict=True
            ):
                if int(absolute) != game.pit_index(int(relative)):
                    raise ValueError("opening_prefix_coordinate_mismatch")
                if relative not in game.possible_moves() or not game.move(
                    game.pit_index(int(relative))
                ):
                    raise ValueError("opening_prefix_replay")
            state = game.to_state()
            canonical = hashlib.sha256(
                json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            float_identity = hashlib.sha256(
                np.asarray(
                    encode_state(state, input_encoding="kalah_v3"), dtype="<f4"
                ).tobytes()
            ).hexdigest()
            if state != row["state"] or canonical != row["state_hash"]:
                raise ValueError(f"opening_state_mismatch:{index}")
            if canonical in excluded_canonical or float_identity in excluded_float32:
                raise ValueError(f"opening_exclusion_overlap:{index}")
            if canonical in seen_states or float_identity in seen_float32:
                raise ValueError("opening_duplicate_state")
            seen_states.add(canonical)
            seen_float32.add(float_identity)
    if not receipt_path.exists():
        return {"status": "prepared_not_executed", "opening_count": len(openings)}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") == "seed488_prepared_not_executed":
        for relative, digest in receipt.items():
            if relative != "status" and _sha(out / relative) != digest:
                raise ValueError(f"prepared_receipt_mismatch:{relative}")
        _verify_supersession(out)
        return {"status": "prepared_not_executed", "opening_count": len(openings)}
    if receipt.get("status") != "seed488_strength_evaluation_complete":
        raise ValueError("completion_receipt_status")
    for relative, digest in receipt.items():
        if relative != "status" and _sha(out / relative) != digest:
            raise ValueError(f"receipt_mismatch:{relative}")
    _verify_supersession(out)
    artifact_manifest = json.loads(
        (out / "runtime-artifacts.json").read_text(encoding="utf-8")
    )
    for arm, artifacts in artifact_manifest.items():
        for name, digest in artifacts.items():
            if _sha(out / "runtime-artifacts" / arm / name) != digest:
                raise ValueError(f"artifact_mismatch:{arm}:{name}")
        _verify_artifact_against_frozen_state(root, out, arm, artifacts)
    results_path = out / "outcomes.jsonl"
    rows = _rows(results_path)
    expected = {(opening, seat) for opening in range(OPENINGS) for seat in (0, 1)}
    observed: dict[tuple[int, int], float] = {}
    for row in rows:
        key = (int(row["opening_index"]), int(row["F_player"]))
        if key not in expected or key in observed:
            raise ValueError("duplicate_or_invalid_game")
        if row.get("opening_state_hash") != openings[key[0]]["state_hash"]:
            raise ValueError("game_opening_identity_mismatch")
        opening = key[0]
        game_log = out / "resume" / f"games-{opening:04d}.jsonl"
        report = out / "resume" / f"report-{opening:04d}.json"
        opening_file = out / "resume" / f"opening-{opening:04d}.jsonl"
        if _sha(game_log) != row.get("batch_game_log_sha256"):
            raise ValueError("game_log_receipt_mismatch")
        if _sha(report) != row.get("batch_report_sha256"):
            raise ValueError("arena_report_receipt_mismatch")
        expected_opening_bytes = (
            json.dumps(openings[opening], sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if expected_opening_bytes != opening_file.read_bytes() or _sha(
            opening_file
        ) != row.get("batch_opening_sha256"):
            raise ValueError("paired_opening_receipt_mismatch")
        batch_rows = _rows(game_log)
        position = int(row["batch_game_position"])
        if len(batch_rows) != 2 or position not in (0, 1):
            raise ValueError("arena_batch_coverage")
        challenger_seats = [
            int(item.get("challenger_player", -1)) for item in batch_rows
        ]
        if sorted(challenger_seats) != [0, 1]:
            raise ValueError("arena_batch_seat_pairing")
        if any(
            item.get("opening_state_hash") != openings[opening]["state_hash"]
            or item.get("opening_prefix_moves", openings[opening]["prefix_moves"])
            != openings[opening]["prefix_moves"]
            for item in batch_rows
        ):
            raise ValueError("arena_batch_opening_identity")
        arena_row = batch_rows[position]
        arena_winner = arena_row.get("winner")
        if arena_winner == "challenger":
            arena_winner = "F"
        elif arena_winner == "current":
            arena_winner = "initializer"
        if int(arena_row.get("challenger_player", -1)) != key[
            1
        ] or arena_winner != row.get("winner"):
            raise ValueError("arena_game_outcome_mismatch")
        _replay_game(openings[opening], arena_row, key[1])
        report_data = json.loads(report.read_text(encoding="utf-8"))
        report_notes = report_data.get("notes", {})
        batch_outcomes = [str(game.get("winner")) for game in batch_rows]
        if (
            report_data.get("games_played") != 2
            or report_data.get("wins") != batch_outcomes.count("challenger")
            or report_data.get("losses") != batch_outcomes.count("current")
            or report_data.get("draws") != batch_outcomes.count("draw")
            or report_notes.get("challenger_simulations") != 1200
            or report_notes.get("current_simulations") != 1200
            or report_notes.get("workers_used") != 1
            or report_notes.get("workers") != 1
            or report_notes.get("games") != 2
            or report_notes.get("games_per_opening") != 2
            or report_notes.get("max_moves") != 200
            or report_notes.get("random_opening_plies") != 0
            or report_notes.get("seed_contract") != "azlite_eval_seed_v2"
            or report_notes.get("base_seed") != SEED + opening
            or report_notes.get("suite_sha256")
            != registration["opening_suite"]["opening_suite_sha256"]
            or report_notes.get("search_profile", {}).get("c_puct") != 1.25
            or report_notes.get("exact_root_solve_threshold") != 16
            or report_notes.get("exact_root_native_probe_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/runtime/kalah_v1_tablebase"
            ]
            or report_notes.get("exact_root_tablebase_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/runtime/kalah_v1_21.kvtb"
            ]
            or report_notes.get("runtime_search_policy_sha256")
            != registration["runtime"]["identities_sha256"][
                "model-artifact/current/search_policy.json"
            ]
            or report_notes.get("runtime_search_policy_mode")
            != "puct_exact_root_hybrid"
            or report_notes.get("exact_root_objective") != "final_score_margin"
            or report_notes.get("exact_root_tie_rule")
            != "highest_legal_network_prior_then_lowest_move_index"
            or report_notes.get("exact_leaf_solve_mode") != "disabled"
            or report_notes.get("search_options") != expected_search_options
            or report_notes.get("seed") != SEED + opening
        ):
            raise ValueError("arena_report_runtime_contract")
        observed[key] = _points(str(row["winner"]))
    if observed.keys() != expected:
        raise ValueError("game_coverage")
    clusters = np.asarray(
        [(observed[(i, 0)] + observed[(i, 1)]) / 2 for i in range(OPENINGS)],
        dtype=np.float64,
    )
    score = float(clusters.mean())
    rng = np.random.default_rng(SEED)
    draw_index = rng.integers(0, OPENINGS, size=(REPLICATES, OPENINGS))
    bootstrap_scores = clusters[draw_index].mean(axis=1)
    lower = float(np.quantile(bootstrap_scores, 0.05))
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    expected_result = {
        "game_count": OPENINGS * 2,
        "score": score,
        "bootstrap": {
            "method": "opening-paired percentile bootstrap; resample 512 opening clusters, carrying both seats together",
            "replicates": REPLICATES,
            "seed": SEED,
            "confidence_level": 0.95,
            "one_sided_lower": lower,
            "two_sided_interval": [
                float(np.quantile(bootstrap_scores, 0.025)),
                float(np.quantile(bootstrap_scores, 0.975)),
            ],
        },
        "advance": lower > 0.5,
        "promotion": False,
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            if key == "score" and np.isclose(
                result.get(key), value, rtol=0, atol=1e-15
            ):
                continue
            raise ValueError(f"result_mismatch:{key}")
    return {"status": "verified_complete", **expected_result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
