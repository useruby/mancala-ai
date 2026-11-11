"""Experiment-local P-only target collection against frozen varied opponents.

This module is deliberately independent of the hash-pinned #548/#549 runners.
It reuses their public evaluator, game, search, encoding, and target primitives.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import random
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ml.alphazero_lite import arena, self_play
from ml.alphazero_lite.exact_root_policy_targets import exact_root_policy_target
from ml.alphazero_lite.kalah_rules import KalahGame

SCHEMA = "varied_opponent_pilot_game_v1"
PAIR_CONCURRENCY = 18
WORKERS_PER_GAME = 1
CPU_QUOTA = "1800%"
CPU_WEIGHT = "100"
MEMORY_MAX_BYTES = 38_654_705_664
ALLOWED_CPUS = "14-31"
NICE = 19
PIT_COUNT = 6
SIMULATIONS = 128
C_PUCT = 1.25
MAX_MOVES = 200
TEMPERATURE_THRESHOLD = 9
TEMPERATURE_EARLY = 1.0
TEMPERATURE_LATE = 0.1
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
SEARCH_OPTIONS = {
    "fpu_mode": "zero",
    "reuse_subtree": False,
    "normalize_values": False,
    "root_policy_mode": "visit_count",
    "tactical_root_bias": 0.0,
    "root_temperature": 0.0,
}
TARGET_SETTINGS = {
    "policy_target_mode": "default",
    "policy_target_noise_mode": "denoised",
    "exact_root_policy_target_mode": "selected_one_hot",
    "value_target_mode": "default",
    "value_source": "terminal_outcome",
}
SEARCH_SETTINGS = {
    "simulations": SIMULATIONS,
    "c_puct": C_PUCT,
    "search_workers": WORKERS_PER_GAME,
    "temperature_threshold_plies": TEMPERATURE_THRESHOLD,
    "temperature_early": TEMPERATURE_EARLY,
    "temperature_late": TEMPERATURE_LATE,
    "dirichlet_alpha": DIRICHLET_ALPHA,
    "dirichlet_epsilon": DIRICHLET_EPSILON,
    "search_options": SEARCH_OPTIONS,
    "max_moves": MAX_MOVES,
}
EXPECTED_COMPONENT_HASHES = {
    "P-E4": {
        "metadata.json": "a9d133043c64ca41e3d77354f4ccf3403ab5f94286608f01589387159ac0258c",
        "model.npz": "2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad",
        "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
        "weights.json": "89dead525a255cff4760867645fd9ce05cb2bbadcf1d72c5ac9089868c4a8136",
    },
    "T512": {
        "metadata.json": "b37f3aa75d828b9b9a0743930c73c8e168b159f6b37434eba1d9871e769af1b8",
        "model.npz": "8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e",
        "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
        "weights.json": "70af73e6f8ee3f4fb76981584c1f2068a4b669300be90bd71cfdd8f71b369997",
    },
    "initializer": {
        "metadata.json": "f3b9a7d665be7b1219b4a3412bf4a499e5bafd525905b50d93deac4e6ad9ccef",
        "model.npz": "7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c",
        "search_policy.json": "280c33d88266969da72a7984dabdcfea180bacb5cb690cb759af1ebdbaac65bc",
        "weights.json": "f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c",
    },
}
OPPONENTS = frozenset({"initializer", "T512"})
_PROCESS_EVALUATORS: PilotEvaluators | None = None


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_components(artifact_root: Path) -> dict[str, dict[str, str]]:
    """Authenticate the 12 published component files before evaluator loading."""
    authenticated: dict[str, dict[str, str]] = {}
    for component, expected_files in EXPECTED_COMPONENT_HASHES.items():
        directory = artifact_root / component
        if not directory.is_dir() or {
            path.name for path in directory.iterdir() if path.is_file()
        } != set(expected_files):
            raise ValueError(f"component_file_set_mismatch:{component}")
        actual_files = {
            name: sha256(directory / name)
            for name in expected_files
            if (directory / name).is_file()
        }
        if actual_files != expected_files:
            raise ValueError(f"component_hash_mismatch:{component}")
        authenticated[component] = actual_files
    return authenticated


def effective_runtime_policy(artifact_root: Path) -> dict[str, Any]:
    """Resolve the pinned sidecar policy separately from the host runtime files."""
    root = Path(__file__).resolve().parents[2]
    sidecars = [
        json.loads((artifact_root / name / "search_policy.json").read_text())
        for name in ("P-E4", "T512", "initializer")
    ]
    first = sidecars[0]
    fields = (
        "schema",
        "mode",
        "exact_root_enabled",
        "exact_root_threshold",
        "exact_root_objective",
        "exact_root_tie_rule",
        "exact_leaf_solve",
        "solver_implementation_identity",
        "native_probe",
        "tablebase",
    )
    if any(
        {key: item.get(key) for key in fields}
        != {key: first.get(key) for key in fields}
        for item in sidecars[1:]
    ):
        raise ValueError("runtime_policy_component_mismatch")
    if (
        first.get("schema") != "azlite_runtime_search_policy_v1"
        or first.get("mode") != "puct_exact_root_hybrid"
        or first.get("exact_root_enabled") is not True
        or first.get("exact_root_threshold") != 16
        or first.get("exact_root_objective") != "final_score_margin"
        or first.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or first.get("exact_leaf_solve") != "disabled"
    ):
        raise ValueError("runtime_policy_semantics_mismatch")
    assets = {
        "native_probe": root / "model-artifact/runtime/kalah_v1_tablebase",
        "tablebase": root / "model-artifact/runtime/kalah_v1_21.kvtb",
    }
    resolved: dict[str, str] = {}
    for name, path in assets.items():
        expected = first[name]["sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"runtime_asset_hash_mismatch:{name}")
        resolved[name] = str(path.resolve())
    return {
        "effective_search_policy": {
            key: first[key]
            for key in fields
            if key not in {"native_probe", "tablebase"}
        },
        "runtime_assets": {
            name: {"path": resolved[name], "sha256": first[name]["sha256"]}
            for name in assets
        },
    }


def validate_game_specs(specs: list[dict[str, Any]]) -> None:
    """Validate whole-pair assignments before any search can start."""
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    identifiers: set[str] = set()
    for spec in specs:
        required = {
            "game_id",
            "pair_id",
            "arm",
            "opening_id",
            "opening_moves",
            "seed",
            "designated_p_seat",
            "opponent",
            "search_workers",
        }
        if not required.issubset(spec):
            raise ValueError("game_spec_missing_fields")
        game_id = str(spec["game_id"])
        if not game_id or game_id in identifiers:
            raise ValueError("duplicate_game_id")
        identifiers.add(game_id)
        seat = spec["designated_p_seat"]
        if seat not in (0, 1):
            raise ValueError("invalid_designated_p_seat")
        if spec["search_workers"] != WORKERS_PER_GAME:
            raise ValueError("game_search_worker_count_mismatch")
        if spec["arm"] == "control":
            if spec["opponent"] != "P":
                raise ValueError("control_must_be_P_vs_P")
        elif spec["arm"] == "treatment":
            if spec["opponent"] not in OPPONENTS:
                raise ValueError("invalid_treatment_opponent")
        else:
            raise ValueError("invalid_arm")
        by_pair.setdefault((str(spec["arm"]), str(spec["pair_id"])), []).append(spec)

    for (arm, _pair_id), pair_specs in by_pair.items():
        if len(pair_specs) != 2 or {s["designated_p_seat"] for s in pair_specs} != {
            0,
            1,
        }:
            raise ValueError("pair_must_have_both_seat_assignments")
        if len({s["opening_id"] for s in pair_specs}) != 1:
            raise ValueError("pair_opening_mismatch")
        if len({int(s["seed"]) for s in pair_specs}) != 1:
            raise ValueError("pair_seed_mismatch")
        if arm == "treatment" and len({s["opponent"] for s in pair_specs}) != 1:
            raise ValueError("treatment_opponent_changed_across_seat_swap")


def evaluator_for_player(
    *,
    player: int,
    designated_p_seat: int,
    arm: str,
    p_evaluator: Any,
    opponent_evaluator: Any | None,
) -> Any:
    """Select evaluator using the board's actual current player."""
    if player == designated_p_seat or arm == "control":
        return p_evaluator
    if opponent_evaluator is None:
        raise ValueError("missing_treatment_opponent_evaluator")
    return opponent_evaluator


def _row_id(game_id: str, ply: int) -> str:
    return f"{game_id}:{ply:04d}"


def play_game(
    spec: dict[str, Any],
    *,
    p_evaluator: Any,
    opponent_evaluator: Any | None,
    search_turn: Callable[[KalahGame, Any, random.Random, int, bool], dict[str, Any]],
    max_moves: int = MAX_MOVES,
) -> dict[str, Any]:
    """Play one assigned-seat game; retain rows only for the designated P seat.

    ``search_turn`` returns a legal relative move and a complete target row payload
    for P turns. It is injected for deterministic scripted fixtures.
    """
    if max_moves <= 0:
        raise ValueError("max_moves_must_be_positive")
    game = KalahGame.from_state(self_play.standard_start_state())
    opening_moves = [int(move) for move in spec["opening_moves"]]
    for move in opening_moves:
        legal = game.possible_moves()
        if move not in legal or not game.move(game.pit_index(move)):
            raise ValueError("illegal_opening_move")
        if game.over():
            raise ValueError("opening_continues_after_terminal")
    start_state = game.to_state()
    rng = random.Random(int(spec["seed"]))
    trajectory: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cpu_start = time.process_time_ns()

    for ply in range(max_moves):
        if game.over():
            break
        player = game.current_player
        state = game.to_state()
        legal_moves = game.possible_moves()
        evaluator = evaluator_for_player(
            player=player,
            designated_p_seat=int(spec["designated_p_seat"]),
            arm=str(spec["arm"]),
            p_evaluator=p_evaluator,
            opponent_evaluator=opponent_evaluator,
        )
        retain = player == int(spec["designated_p_seat"])
        result = search_turn(game, evaluator, rng, ply, retain)
        move = int(result["move"])
        if move not in game.possible_moves():
            raise ValueError("search_returned_illegal_move")
        row_id = _row_id(str(spec["game_id"]), ply) if retain else None
        if retain:
            row = dict(result.get("row", {}))
            row.update(
                {
                    "row_id": row_id,
                    "game_id": str(spec["game_id"]),
                    "pair_id": str(spec["pair_id"]),
                    "arm": str(spec["arm"]),
                    "opening_id": str(spec["opening_id"]),
                    "seed": int(spec["seed"]),
                    "opponent": str(spec["opponent"]),
                    "designated_p_seat": int(spec["designated_p_seat"]),
                    "player": player,
                    "move_index": ply,
                    "state": self_play.encode_state(state, input_encoding="kalah_v3"),
                    "canonical_state": state,
                    "stone_count": sum(state["player_pits"])
                    + sum(state["opponent_pits"]),
                    "legal_moves": legal_moves,
                    "search_value": float(row.get("value", 0.0)),
                    "value_target_mode": "default",
                    "target_settings": dict(TARGET_SETTINGS),
                }
            )
            rows.append(row)
        old_player = game.current_player
        trajectory.append(
            {
                "ply": ply,
                "player": player,
                "state": state,
                "move": move,
                "row_id": row_id,
            }
        )
        if not game.move(game.pit_index(move)):
            raise ValueError("legal_move_failed")
        trajectory[-1]["next_player"] = game.current_player
        trajectory[-1]["extra_turn"] = game.current_player == old_player

    complete = game.over()
    status = (
        "complete"
        if complete
        else "capped"
        if len(trajectory) >= max_moves
        else "incomplete"
    )
    cpu_ns = time.process_time_ns() - cpu_start
    if not complete:
        rows = []
    else:
        for row in rows:
            row["value"] = self_play.derive_self_play_value_target(
                outcome_value=self_play.outcome_for_player(
                    game.winner, int(spec["designated_p_seat"])
                ),
                search_value=float(row["search_value"]),
                move_index=int(row["move_index"]),
                mode="default",
            )
            row.pop("search_value", None)
            row["winner"] = game.winner
        if len(rows) != sum(
            int(step["player"] == int(spec["designated_p_seat"])) for step in trajectory
        ):
            raise AssertionError("retained_row_count_internal_mismatch")
    return {
        "schema": SCHEMA,
        "spec": dict(spec),
        "start_state": start_state,
        "opening_moves": opening_moves,
        "trajectory": trajectory,
        "rows": rows,
        "game_status": status,
        "max_moves": max_moves,
        "target_settings": dict(TARGET_SETTINGS),
        "search_settings": {**SEARCH_SETTINGS, "max_moves": max_moves},
        "game_completed": complete,
        "game_length": len(trajectory),
        "winner": game.winner if complete else None,
        "final_stores": list(game.captured_seeds),
        "cpu_time_ns": cpu_ns,
        "accepted_rows": len(rows) if complete else 0,
    }


def production_search_turn(
    game: KalahGame,
    evaluator: Any,
    rng: random.Random,
    ply: int,
    retain: bool,
    *,
    exact_root: Callable[[KalahGame, Any], Any] | None = None,
) -> dict[str, Any]:
    """Use the frozen 128-simulation PUCT and seed549 target semantics."""
    legal = game.possible_moves()
    temperature = TEMPERATURE_EARLY if ply < TEMPERATURE_THRESHOLD else TEMPERATURE_LATE
    if exact_root is not None:
        decision = exact_root(game, evaluator)
        if decision is not None:
            target = exact_root_policy_target(
                selected_action=decision.selected_move,
                optimal_actions=decision.optimal_moves,
                mode="selected_one_hot",
            )
            return {
                "move": int(decision.selected_move),
                "row": {
                    "policy": target,
                    "teacher_source": "exact_root_tablebase",
                    "policy_target_mode": "default",
                    "policy_target_actual_mode": "exact_root_one_hot",
                    "value": 0.0,
                    "exact_selected_action": int(decision.selected_move),
                    "exact_optimal_actions": [int(a) for a in decision.optimal_moves],
                }
                if retain
                else {},
            }
    sampling = self_play.PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        rng=rng,
        **SEARCH_OPTIONS,
    )
    visits, root = sampling.run(
        game,
        dirichlet_alpha=DIRICHLET_ALPHA if ply < TEMPERATURE_THRESHOLD else None,
        dirichlet_epsilon=DIRICHLET_EPSILON if ply < TEMPERATURE_THRESHOLD else 0.0,
    )
    action_policy = self_play.policy_from_visits(
        visits, legal_moves=legal, temperature=temperature
    )
    move = self_play.sample_move(action_policy, legal_moves=legal, rng=rng)
    if not retain:
        return {"move": move, "row": {}}
    target_search = self_play.PUCT(
        evaluator=evaluator,
        simulations=SIMULATIONS,
        c_puct=C_PUCT,
        rng=random.Random(rng.randrange(2**31)),
        **SEARCH_OPTIONS,
    )
    target_visits, _target_root = target_search.run(
        game, dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    policy = self_play.build_policy_target(
        target_visits, legal_moves=legal, temperature=temperature, mode="default"
    )
    return {
        "move": move,
        "row": {
            "policy": policy,
            "teacher_source": "artifact_puct",
            "policy_target_mode": "default",
            "policy_target_actual_mode": "default",
            "value": float(root.q_value),
            "simulations": SIMULATIONS,
            "legal_moves": legal,
            "policy_target_noise_mode": "denoised",
        },
    }


def _state_after_prefix(opening_moves: list[int]) -> KalahGame:
    game = KalahGame.from_state(self_play.standard_start_state())
    for move in opening_moves:
        if move not in game.possible_moves() or not game.move(game.pit_index(move)):
            raise ValueError("illegal_opening_move")
    return game


def verify_game(game_record: dict[str, Any]) -> dict[str, int | str]:
    """Replay a saved trajectory and reconcile retained rows to all P-seat plies."""
    spec = game_record["spec"]
    if game_record.get("target_settings") != TARGET_SETTINGS:
        raise ValueError("game_target_settings_mismatch")
    expected_search_settings = {
        **SEARCH_SETTINGS,
        "max_moves": game_record.get("max_moves"),
    }
    if game_record.get("search_settings") != expected_search_settings:
        raise ValueError("game_search_settings_mismatch")
    game = _state_after_prefix([int(move) for move in game_record["opening_moves"]])
    if game.to_state() != game_record["start_state"]:
        raise ValueError("game_start_state_mismatch")
    trajectory = game_record["trajectory"]
    expected_row_ids: list[str] = []
    seen_rows: set[str] = set()
    for ply, step in enumerate(trajectory):
        if (
            game.over()
            or step.get("ply") != ply
            or step.get("state") != game.to_state()
        ):
            raise ValueError("trajectory_state_or_index_mismatch")
        player = game.current_player
        if step.get("player") != player:
            raise ValueError("trajectory_player_mismatch")
        retain = player == int(spec["designated_p_seat"])
        expected_id = _row_id(str(spec["game_id"]), ply) if retain else None
        if step.get("row_id") != expected_id:
            raise ValueError("trajectory_row_assignment_mismatch")
        if expected_id:
            expected_row_ids.append(expected_id)
        before_player = player
        move = int(step["move"])
        if move not in game.possible_moves() or not game.move(game.pit_index(move)):
            raise ValueError("trajectory_illegal_move")
        if step.get("next_player") != game.current_player:
            raise ValueError("trajectory_next_player_mismatch")
        if step.get("extra_turn") != (game.current_player == before_player):
            raise ValueError("trajectory_extra_turn_mismatch")

    complete = game.over()
    cap_limit = int(game_record.get("max_moves", MAX_MOVES))
    expected_status = (
        "complete"
        if complete
        else "capped"
        if len(trajectory) >= cap_limit
        else "incomplete"
    )
    if game_record.get("game_status") != expected_status:
        raise ValueError("game_status_mismatch")
    if complete and (
        game_record.get("winner") != game.winner
        or game_record.get("final_stores") != game.captured_seeds
    ):
        raise ValueError("terminal_outcome_mismatch")
    rows = game_record.get("rows", [])
    if not complete and rows:
        raise ValueError("nonterminal_game_rows_not_accepted")
    for row in rows:
        row_id = row.get("row_id")
        if row_id in seen_rows:
            raise ValueError("duplicate_retained_row")
        seen_rows.add(row_id)
        if row.get("player") != spec["designated_p_seat"]:
            raise ValueError("opponent_seat_row")
        if row.get("target_settings") != TARGET_SETTINGS:
            raise ValueError("row_target_settings_mismatch")
        if row.get("game_id") != spec["game_id"] or row.get("move_index") < 0:
            raise ValueError("row_provenance_mismatch")
        if any(
            row.get(field) != spec[field]
            for field in (
                "pair_id",
                "arm",
                "opening_id",
                "designated_p_seat",
                "seed",
                "opponent",
            )
        ):
            raise ValueError("row_pair_assignment_mismatch")
        matching_step = next(
            (step for step in trajectory if step.get("row_id") == row_id), None
        )
        if (
            matching_step is None
            or row.get("move_index") != matching_step.get("ply")
            or row.get("state")
            != self_play.encode_state(matching_step["state"], input_encoding="kalah_v3")
            or row.get("canonical_state") != matching_step["state"]
            or row.get("stone_count")
            != sum(matching_step["state"]["player_pits"])
            + sum(matching_step["state"]["opponent_pits"])
        ):
            raise ValueError("row_state_or_ply_mismatch")
        policy = row.get("policy")
        if not isinstance(policy, list) or len(policy) != PIT_COUNT:
            raise ValueError("row_policy_shape_invalid")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0 for value in policy
        ):
            raise ValueError("row_policy_invalid")
    if complete and [row.get("row_id") for row in rows] != expected_row_ids:
        raise ValueError("retained_row_set_mismatch")
    if complete:
        state_by_id = {step.get("row_id"): step["state"] for step in trajectory}
        for row in rows:
            state = KalahGame.from_state(state_by_id[row["row_id"]])
            legal = state.possible_moves()
            if row.get("legal_moves") != legal:
                raise ValueError("row_legal_moves_mismatch")
            if abs(sum(float(value) for value in row["policy"]) - 1.0) > 1e-5:
                raise ValueError("row_policy_not_normalized")
            if any(
                float(row["policy"][action]) != 0.0
                for action in range(PIT_COUNT)
                if action not in legal
            ):
                raise ValueError("row_policy_assigns_illegal_mass")
            expected_value = self_play.outcome_for_player(
                game.winner, int(spec["designated_p_seat"])
            )
            if row.get("winner") != game.winner or row.get("value") != expected_value:
                raise ValueError("row_terminal_target_mismatch")
    if game_record.get("accepted_rows") != (len(rows) if complete else 0):
        raise ValueError("accepted_row_count_mismatch")
    return {
        "game_id": str(spec["game_id"]),
        "status": expected_status,
        "rows": len(rows),
    }


def verify_archive(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check seat assignment invariants and replay every archived game."""
    validate_game_specs([record["spec"] for record in records])
    results = [verify_game(record) for record in records]
    return {
        "games": len(results),
        "complete": sum(result["status"] == "complete" for result in results),
        "capped": sum(result["status"] == "capped" for result in results),
        "incomplete": sum(result["status"] == "incomplete" for result in results),
        "accepted_rows": sum(int(result["rows"]) for result in results),
    }


def run_pair_specs(
    pair_jobs: list[dict[str, Any]],
    pair_runner: Callable[[dict[str, Any]], Any],
    *,
    max_workers: int = PAIR_CONCURRENCY,
) -> list[Any]:
    """Run pair jobs concurrently while preserving each pair as one fixed unit."""
    if max_workers != PAIR_CONCURRENCY:
        raise ValueError("pilot_pair_concurrency_must_be_18")
    if not 1 <= max_workers <= PAIR_CONCURRENCY:
        raise ValueError("invalid_pair_concurrency")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(pair_runner, job) for job in pair_jobs]
        return [future.result() for future in futures]


def execute_pair_jobs(
    pair_jobs: list[dict[str, Any]],
    artifact_root: Path,
    pair_runner: Callable[[dict[str, Any]], Any],
) -> list[Any]:
    """Validate every assignment and artifact before dispatching any pair."""
    all_specs = [spec for job in pair_jobs for spec in job.get("game_specs", [])]
    validate_game_specs(all_specs)
    verify_components(artifact_root)
    if any(len(job.get("game_specs", [])) != 2 for job in pair_jobs):
        raise ValueError("pair_job_must_contain_two_seat_games")
    return run_pair_specs(pair_jobs, pair_runner)


def run_validated_pairs(
    pair_jobs: list[dict[str, Any]], artifact_root: Path
) -> list[dict[str, Any]]:
    """Authenticate all inputs, then run all seat-swapped pairs in 18-way slots."""
    if any(len(job.get("game_specs", [])) != 2 for job in pair_jobs):
        raise ValueError("pair_job_must_contain_two_seat_games")
    all_specs = [spec for job in pair_jobs for spec in job.get("game_specs", [])]
    validate_game_specs(all_specs)
    verify_components(artifact_root)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=PAIR_CONCURRENCY,
        initializer=_initialize_pair_process,
        initargs=(str(artifact_root),),
    ) as pool:
        futures = [pool.submit(_play_pair_process, job) for job in pair_jobs]
        results = [future.result() for future in futures]
    verify_archive([game for result in results for game in result["games"]])
    return results


def _initialize_pair_process(artifact_root: str) -> None:
    """Authenticate and load immutable evaluators once in each pair process."""
    global _PROCESS_EVALUATORS
    import torch

    torch.set_num_threads(WORKERS_PER_GAME)
    if torch.get_num_interop_threads() != WORKERS_PER_GAME:
        torch.set_num_interop_threads(WORKERS_PER_GAME)
    _PROCESS_EVALUATORS = load_evaluators(Path(artifact_root))


def _play_pair_process(job: dict[str, Any]) -> dict[str, Any]:
    """Run both seat assignments sequentially in one isolated pair process."""
    if _PROCESS_EVALUATORS is None:
        raise RuntimeError("pair_process_evaluators_not_initialized")
    specs = job["game_specs"]
    validate_game_specs(specs)
    games = [
        play_assigned_game(
            spec,
            evaluators=_PROCESS_EVALUATORS,
            max_moves=int(job.get("max_moves", MAX_MOVES)),
        )
        for spec in specs
    ]
    return {"pair_id": str(specs[0]["pair_id"]), "games": games}


def launch_preflight(
    specs: list[dict[str, Any]], artifact_root: Path
) -> dict[str, Any]:
    """Validate experiment inputs and execution concurrency without launching games."""
    validate_game_specs(specs)
    components = verify_components(artifact_root)
    runtime = effective_runtime_policy(artifact_root)
    scope_argv = systemd_scope_command([sys.executable, "-c", "pass"])
    subprocess.run(scope_argv, check=True, capture_output=True, text=True)
    return {
        "status": "preflight_passed_no_games_launched",
        "specifications": len(specs),
        "components": components,
        "search_policy_and_runtime": runtime,
        "pair_concurrency": PAIR_CONCURRENCY,
        "workers_per_game": WORKERS_PER_GAME,
        "resource_limits": {
            "cpu_quota": CPU_QUOTA,
            "cpu_weight": CPU_WEIGHT,
            "memory_max_bytes": MEMORY_MAX_BYTES,
            "allowed_cpus": ALLOWED_CPUS,
            "nice": NICE,
        },
        "cpu_accounting": "process_cpu_time_ns_per_game_in_isolated_pair_process",
        "systemd_scope_preflight": "passed_no_game_command_run",
    }


def systemd_scope_command(command: list[str]) -> list[str]:
    """Build the registered resource-limited systemd scope command."""
    if not command:
        raise ValueError("scope_command_must_not_be_empty")
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--property=CPUQuota={CPU_QUOTA}",
        f"--property=CPUWeight={CPU_WEIGHT}",
        f"--property=MemoryMax={MEMORY_MAX_BYTES}",
        f"--property=AllowedCPUs={ALLOWED_CPUS}",
        f"--nice={NICE}",
        "--",
        *command,
    ]


@dataclass(frozen=True)
class PilotEvaluators:
    """Resolved P and treatment evaluator compositions."""

    p: Any
    initializer: Any
    t512: Any
    runtime_policy: dict[str, Any]


def load_evaluators(artifact_root: Path) -> PilotEvaluators:
    """Authenticate bundle bytes, then create the three required compositions."""
    verify_components(artifact_root)
    runtime_policy = effective_runtime_policy(artifact_root)
    initializer = arena.ArtifactEvaluator(artifact_root / "initializer")
    p_policy = arena.ArtifactEvaluator(artifact_root / "P-E4")
    t_policy = arena.ArtifactEvaluator(artifact_root / "T512")
    return PilotEvaluators(
        p=arena.ComposedArtifactEvaluator(
            initializer, p_policy, policy_source="candidate", value_source="current"
        ),
        initializer=initializer,
        t512=arena.ComposedArtifactEvaluator(
            initializer, t_policy, policy_source="candidate", value_source="current"
        ),
        runtime_policy=runtime_policy,
    )


def play_assigned_game(
    spec: dict[str, Any],
    *,
    evaluators: PilotEvaluators,
    max_moves: int = MAX_MOVES,
) -> dict[str, Any]:
    """Run one game using a validated composed evaluator and native exact root."""
    opponent = None
    if spec["arm"] == "treatment":
        opponent = (
            evaluators.initializer
            if spec["opponent"] == "initializer"
            else evaluators.t512
        )
    policy = evaluators.runtime_policy
    resolved = policy["runtime_assets"]
    threshold = int(policy["effective_search_policy"]["exact_root_threshold"])
    native_tablebase = arena.NativeExactRootTablebase(
        resolved["native_probe"]["path"],
        resolved["tablebase"]["path"],
        warm_on_start=True,
    )

    def exact_root(game: KalahGame, evaluator: Any) -> Any:
        return arena.exact_root_decision(
            game,
            evaluator,
            tablebase=native_tablebase,
            threshold=threshold,
        )

    try:
        return play_game(
            spec,
            p_evaluator=evaluators.p,
            opponent_evaluator=opponent,
            search_turn=lambda game, evaluator, rng, ply, retain: (
                production_search_turn(
                    game, evaluator, rng, ply, retain, exact_root=exact_root
                )
            ),
            max_moves=max_moves,
        )
    finally:
        native_tablebase.close()


def main() -> None:
    """Provide a read-only launch preflight; never starts games from this CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.specs.read_text())
    if not isinstance(specs, list):
        raise SystemExit("specs_file_must_contain_json_array")
    print(json.dumps(launch_preflight(specs, args.artifacts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
