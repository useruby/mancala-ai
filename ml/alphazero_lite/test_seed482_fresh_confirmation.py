"""Synthetic, no-search verification of the seed482 preparation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite import seed482_fresh_confirmation as seed482


def _game(index: int, first_move: int = 0) -> dict:
    game = KalahGame([4] * 12, [0, 0], 0)
    rows: list[dict] = []
    moves: list[int] = []
    while not game.over():
        legal = game.possible_moves()
        choice = first_move if not moves and first_move in legal else legal[0]
        state = game.to_state()
        policy = np.zeros(6, dtype=np.float32)
        policy[legal] = 1.0 / len(legal)
        rows.append(
            {
                "ply": len(moves),
                "state": state,
                "policy": policy.tolist(),
                "value": 0,
                "policy_target_mode": "sharpened",
                "policy_target_noise_mode": "denoised",
            }
        )
        moves.append(choice)
        assert game.move(game.pit_index(choice))
    winner = game.winner
    for row in rows:
        row["value"] = (
            0
            if winner is None
            else (1 if winner == row["state"]["current_player"] else -1)
        )
    return {
        "game_index": index,
        "seed": seed482.expected_seed(index),
        "start_state": KalahGame([4] * 12, [0, 0], 0).to_state(),
        "moves": moves,
        "winner": winner,
        "positions": rows,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _exclusions(path: Path, states: list[dict] | None = None) -> None:
    identities = [
        {
            "state": state,
            "canonical_identity": seed482.canonical_identity(state),
            "float32_identity": seed482.float32_identity(state),
        }
        for state in states or []
    ]
    manifest = {
        "schema": "seed482-exclusions-v1",
        "evidence": {"synthetic_test_fixture": seed482.sha_bytes(b"fixture")},
        "identities": identities,
    }
    manifest["manifest_sha256"] = seed482.sha_bytes(seed482._canonical_json(manifest))
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n")


def test_fixture_selection_verification_and_frozen_evaluation_gate(
    tmp_path: Path,
) -> None:
    games = [_game(i, i % 6) for i in range(4)]
    game_path = tmp_path / "games.jsonl"
    exclusions_path = tmp_path / "exclusions.json"
    output = tmp_path / "output"
    _write_jsonl(game_path, games)
    _exclusions(exclusions_path)
    receipt = seed482.run_fixture_games(game_path, exclusions_path, output)
    assert receipt["selected_count"] > 0
    assert receipt["status"] == "shortfall"
    assert (
        seed482.verify_cohort(game_path, exclusions_path, output)["status"]
        == "verified"
    )
    gate_body = {
        "schema": "seed482-frozen-cohort-receipt-v1",
        "status": "shortfall",
        "protocol_sha256": seed482.protocol_digest(),
        "selected_count": receipt["selected_count"],
        "independent_verification": "passed",
    }
    gate_body["frozen_receipt_sha256"] = seed482.sha_bytes(
        seed482._canonical_json(gate_body)
    )
    gate_path = tmp_path / "frozen.json"
    gate_path.write_text(json.dumps(gate_body))
    with pytest.raises(ValueError, match="cohort_not_complete"):
        seed482.require_frozen_cohort(
            gate_path,
            seed482.sha_bytes(gate_path.read_bytes()),
        )


def test_resume_matches_uninterrupted_and_duplicate_canonical_and_float_ids_are_skipped(
    tmp_path: Path,
) -> None:
    games = [_game(i, i % 6) for i in range(5)]
    exclusions_path = tmp_path / "exclusions.json"
    _exclusions(exclusions_path)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, games[:2])
    _write_jsonl(second, games)
    resumed, clean = tmp_path / "resumed", tmp_path / "clean"
    seed482.run_fixture_games(first, exclusions_path, resumed)
    seed482.run_fixture_games(second, exclusions_path, resumed)
    seed482.run_fixture_games(second, exclusions_path, clean)
    assert (resumed / "selected.jsonl").read_bytes() == (
        clean / "selected.jsonl"
    ).read_bytes()

    interrupted = tmp_path / "interrupted"
    seed482.run_fixture_games(first, exclusions_path, interrupted)
    checkpoint_path = interrupted / "checkpoint.json"
    committed_checkpoint = checkpoint_path.read_bytes()
    _write_jsonl(first, games[:3])
    seed482.run_fixture_games(first, exclusions_path, interrupted)
    checkpoint_path.write_bytes(committed_checkpoint)
    seed482.run_fixture_games(second, exclusions_path, interrupted)
    assert (interrupted / "selected.jsonl").read_bytes() == (
        clean / "selected.jsonl"
    ).read_bytes()


def test_illegal_replay_and_altered_seed_are_rejected(tmp_path: Path) -> None:
    record = _game(0)
    record["moves"][0] = 6
    with pytest.raises(ValueError, match="illegal_replay_move"):
        seed482._replay(record)
    record = _game(0)
    record["seed"] += 1
    with pytest.raises(ValueError, match="game_seed_mismatch"):
        seed482._replay(record)


def test_selfplay_rows_convert_to_strictly_replayable_complete_game() -> None:
    game = _game(0)
    raw_rows = []
    profile = {
        "version": "v1",
        "kind": "self_play",
        "player_mode": "puct",
        "simulations": 1200,
        "c_puct": 1.25,
        "search_options": {
            "fpu_mode": "zero",
            "reuse_subtree": True,
            "normalize_values": False,
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
        },
        "runtime_search_policy_mode": "puct_exact_root_hybrid",
        "runtime_search_policy_native_probe_sha256": seed482.PROTOCOL["behavior_model"][
            "native_probe_sha256"
        ],
        "runtime_search_policy_tablebase_sha256": seed482.PROTOCOL["behavior_model"][
            "tablebase_sha256"
        ],
        "exact_root_solve_enabled": True,
        "exact_root_solve_threshold": 16,
        "exact_root_solver": "native_kvtb_root_action_probe_v1",
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
    }
    profile_hash = seed482.sha_bytes(seed482._canonical_json(profile))
    profile["hash"] = profile_hash
    for row in game["positions"]:
        legal = KalahGame.from_state(row["state"]).possible_moves()
        visits = [1 if move in legal else 0 for move in range(6)]
        sampling_epsilon = 0.3 if row["ply"] < 12 else 0.0
        target = seed482.build_policy_target(
            np.asarray(visits, dtype=np.int64),
            legal_moves=legal,
            temperature=1.1 if row["ply"] < 12 else 0.15,
            mode="sharpened",
        )
        raw_rows.append(
            {
                "state": seed482.encode_state(row["state"], input_encoding="kalah_v3"),
                "move_index": row["ply"],
                "policy": target,
                "value": row["value"],
                "policy_target_mode": row["policy_target_mode"],
                "policy_target_noise_mode": row["policy_target_noise_mode"],
                "policy_target_actual_mode": "sharpened",
                "value_target_mode": "default",
                "teacher_source": "puct",
                "simulations": 1200,
                "dirichlet_alpha": sampling_epsilon,
                "sampling_dirichlet_epsilon": sampling_epsilon,
                "target_dirichlet_epsilon": 0.0,
                "dirichlet_epsilon_for_sampling": sampling_epsilon,
                "dirichlet_epsilon_for_target": 0.0,
                "root_visit_counts": visits,
                "search_profile": profile,
                "search_profile_hash": profile_hash,
                "game_completed": True,
                "game_moves": game["moves"],
                "game_seed": game["seed"],
                "game_start_state": game["start_state"],
                "winner": game["winner"],
            }
        )
    trajectory_hash = seed482.trajectory_hash_for_encoded_states(
        [row["state"] for row in raw_rows], winner=game["winner"]
    )
    for row in raw_rows:
        row["game_index"] = 0
        row["game_length"] = len(raw_rows)
        row["trajectory_hash"] = trajectory_hash
    converted = seed482._game_record_from_selfplay(raw_rows, 0)
    assert seed482._replay(converted) == converted["positions"]


def test_training_historical_overlap_and_duplicate_state_selection(
    tmp_path: Path,
) -> None:
    state = _game(0)["positions"][0]["state"]
    path = tmp_path / "exclusions.json"
    _exclusions(path, [state])
    _, canonical, floats = seed482.load_exclusions(path)
    game = _game(0)
    choice = seed482.select_game_position(game, canonical, floats, set(), set())
    assert choice is not None
    # The exact first state is excluded; selection advances to a later eligible state.
    assert choice["canonical_identity"] != seed482.canonical_identity(state)
    chosen_c = {choice["canonical_identity"]}
    chosen_f = {choice["float32_identity"]}
    assert (
        seed482.select_game_position(game, set(), set(), chosen_c, chosen_f) is not None
    )
    assert (
        seed482.select_game_position(game, canonical, floats, set(), set()) is not None
    )
    all_c = {seed482.canonical_identity(row["state"]) for row in game["positions"]}
    all_f = {seed482.float32_identity(row["state"]) for row in game["positions"]}
    assert seed482.select_game_position(game, all_c, all_f, set(), set()) is None


def test_altered_labels_and_receipt_rebound_corruption_are_rejected(
    tmp_path: Path,
) -> None:
    games = [_game(i) for i in range(2)]
    game_path = tmp_path / "games.jsonl"
    exclusions = tmp_path / "exclusions.json"
    output = tmp_path / "output"
    _write_jsonl(game_path, games)
    _exclusions(exclusions)
    seed482.run_fixture_games(game_path, exclusions, output)
    rows = [json.loads(line) for line in game_path.read_text().splitlines()]
    rows[0]["positions"][0]["value"] *= -1
    _write_jsonl(game_path, rows)
    with pytest.raises(ValueError, match="value_label_mismatch"):
        seed482.verify_cohort(game_path, exclusions, output)

    _write_jsonl(game_path, games)
    cohort_path = output / "cohort.json"
    cohort = json.loads(cohort_path.read_text())
    cohort["selected_count"] += 1
    cohort.pop("receipt_sha256")
    cohort["receipt_sha256"] = seed482.sha_bytes(seed482._canonical_json(cohort))
    cohort_path.write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="cohort_completion_fields_mismatch"):
        seed482.verify_cohort(game_path, exclusions, output)


def test_altered_protocol_settings_and_candidate_dependent_selection_are_rejected(
    tmp_path: Path,
) -> None:
    games = [_game(0)]
    game_path = tmp_path / "games.jsonl"
    exclusions = tmp_path / "exclusions.json"
    _write_jsonl(game_path, games)
    _exclusions(exclusions)
    altered = dict(seed482.PROTOCOL)
    altered["distribution"] = "altered"
    with pytest.raises(ValueError, match="protocol_identity_mismatch"):
        seed482.run_fixture_games(game_path, exclusions, tmp_path / "out", altered)
    original = seed482.select_game_position(games[0], set(), set(), set(), set())
    changed_game = json.loads(json.dumps(games[0]))
    for row in changed_game["positions"]:
        row["policy"] = [1, 0, 0, 0, 0, 0]
        row["value"] = 1 - row["value"]
        row["candidate_prediction"] = [0, 0, 0, 0, 0, 1]
        row["candidate_loss"] = 1e9
    changed = seed482.select_game_position(changed_game, set(), set(), set(), set())
    assert original["canonical_identity"] == changed["canonical_identity"]


def test_independent_verifier_is_read_only(tmp_path: Path) -> None:
    games = [_game(i) for i in range(2)]
    game_path = tmp_path / "games.jsonl"
    exclusions = tmp_path / "exclusions.json"
    output = tmp_path / "output"
    _write_jsonl(game_path, games)
    _exclusions(exclusions)
    seed482.run_fixture_games(game_path, exclusions, output)
    paths = [
        output / name for name in ("selected.jsonl", "checkpoint.json", "cohort.json")
    ]
    before = {path: path.read_bytes() for path in paths}
    assert seed482.verify_cohort(game_path, exclusions, output)["status"] == "verified"
    assert before == {path: path.read_bytes() for path in paths}


def test_real_generation_is_blocked_until_historical_probe_ledger_is_complete(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(
        ValueError, match="historical_probe_identity_ledger_unavailable"
    ):
        seed482.generate_frozen_games(
            repository,
            repository / "docs/data/seed482-exclusion-manifest.json.gz",
            tmp_path / "real-generation-must-not-start",
        )
    assert not (tmp_path / "real-generation-must-not-start").exists()
