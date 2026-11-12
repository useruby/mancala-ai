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
import contextlib
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml.alphazero_lite import arena, self_play
from ml.alphazero_lite.exact_root_policy_targets import (
    exact_root_policy_target,
    validate_exact_root_metadata,
)
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
EXACT_ROOT_THRESHOLD = 16
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
    "exact_root_threshold": EXACT_ROOT_THRESHOLD,
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
_PROCESS_CHILD_CPU_START_NS = 0


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reaped_child_cpu_ns() -> int:
    """Return cumulative user+system CPU time for reaped direct children."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return int(round((usage.ru_utime + usage.ru_stime) * 1_000_000_000))


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
        or first.get("exact_root_threshold") != EXACT_ROOT_THRESHOLD
        or first.get("exact_root_objective") != "final_score_margin"
        or first.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or first.get("exact_leaf_solve") != "disabled"
    ):
        raise ValueError("runtime_policy_semantics_mismatch")
    tablebase = Path(
        os.environ.get(
            "AZLITE_KALAH_V1_TABLEBASE",
            root / "model-artifact/runtime/kalah_v1_21.kvtb",
        )
    )
    assets = {
        "native_probe": root / "model-artifact/runtime/kalah_v1_tablebase",
        "tablebase": tablebase,
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
    opening_prefixes: dict[str, tuple[int, ...]] = {}
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
        prefix = tuple(int(move) for move in spec["opening_moves"])
        bound_prefix = opening_prefixes.setdefault(str(spec["opening_id"]), prefix)
        if prefix != bound_prefix:
            raise ValueError("opening_prefix_assignment_mismatch")
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
        if (
            len({tuple(int(move) for move in s["opening_moves"]) for s in pair_specs})
            != 1
        ):
            raise ValueError("pair_opening_prefix_mismatch")
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
                    "exact_action_margins": {
                        str(int(action)): int(margin)
                        for action, margin in decision.action_margins.items()
                    },
                    "exact_legal_moves": [
                        int(action) for action in decision.legal_moves
                    ],
                    "exact_network_priors": [
                        float(value) for value in decision.network_priors
                    ],
                    "exact_active_pit_stones": int(decision.active_pit_stones),
                    "exact_root_player": int(decision.root_player),
                    "exact_solver_calls": int(decision.solver_calls),
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
            "target_visits": [
                int(round(float(value))) for value in target_visits.tolist()
            ],
            "target_temperature": float(temperature),
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
    if game_record.get("opening_moves") != spec.get("opening_moves"):
        raise ValueError("archived_opening_prefix_mismatch")
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
        active_stones = sum(matching_step["state"]["player_pits"]) + sum(
            matching_step["state"]["opponent_pits"]
        )
        expected_teacher = (
            "exact_root_tablebase"
            if active_stones <= EXACT_ROOT_THRESHOLD
            else "artifact_puct"
        )
        if row.get("teacher_source") != expected_teacher:
            raise ValueError("row_search_route_mismatch")
        if (
            expected_teacher == "exact_root_tablebase"
            and row.get("exact_selected_action") != matching_step["move"]
        ):
            raise ValueError("exact_root_selected_move_mismatch")
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
            expected_policy = reconstruct_policy_target(row, legal_moves=legal)
            if not np.allclose(
                np.asarray(row["policy"], dtype=np.float64),
                np.asarray(expected_policy, dtype=np.float64),
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError("row_policy_target_mismatch")
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


def reconstruct_policy_target(
    row: dict[str, Any], *, legal_moves: list[int]
) -> list[float]:
    """Rebuild the registered policy target from archived search evidence."""
    if row.get("teacher_source") == "exact_root_tablebase":
        required = (
            "exact_selected_action",
            "exact_optimal_actions",
            "exact_action_margins",
            "exact_legal_moves",
            "exact_network_priors",
            "exact_active_pit_stones",
            "exact_root_player",
            "exact_solver_calls",
        )
        if any(key not in row for key in required):
            raise ValueError("exact_root_decision_evidence_missing")
        if row["exact_legal_moves"] != legal_moves:
            raise ValueError("exact_root_legal_moves_mismatch")
        if row["exact_active_pit_stones"] != row["stone_count"]:
            raise ValueError("exact_root_active_stones_mismatch")
        if row["exact_root_player"] != row["player"] or row["exact_solver_calls"] <= 0:
            raise ValueError("exact_root_decision_provenance_mismatch")
        selected, optimal = validate_exact_root_metadata(
            {
                "exact_selected_action": row["exact_selected_action"],
                "exact_optimal_actions": row["exact_optimal_actions"],
                "exact_action_margins": row["exact_action_margins"],
            }
        )
        margins = {
            int(action): int(value)
            for action, value in row["exact_action_margins"].items()
        }
        if set(margins) != set(legal_moves) or optimal != sorted(
            action
            for action, margin in margins.items()
            if margin == max(margins.values())
        ):
            raise ValueError("exact_root_margin_evidence_mismatch")
        priors = [float(value) for value in row["exact_network_priors"]]
        if len(priors) != PIT_COUNT or not all(
            math.isfinite(value) for value in priors
        ):
            raise ValueError("exact_root_network_prior_evidence_invalid")
        selected_by_tie_rule = max(
            optimal, key=lambda action: (priors[action], -action)
        )
        if selected != selected_by_tie_rule:
            raise ValueError("exact_root_selected_action_mismatch")
        return exact_root_policy_target(
            selected_action=selected,
            optimal_actions=optimal,
            mode=TARGET_SETTINGS["exact_root_policy_target_mode"],
        )

    if row.get("teacher_source") != "artifact_puct":
        raise ValueError("retained_row_teacher_source_invalid")
    visits = row.get("target_visits")
    temperature = row.get("target_temperature")
    if (
        not isinstance(visits, list)
        or len(visits) != PIT_COUNT
        or not isinstance(temperature, (int, float))
        or float(temperature)
        != (
            TEMPERATURE_EARLY
            if int(row["move_index"]) < TEMPERATURE_THRESHOLD
            else TEMPERATURE_LATE
        )
    ):
        raise ValueError("puct_target_visit_evidence_invalid")
    if any(not isinstance(value, int) or value < 0 for value in visits):
        raise ValueError("puct_target_visits_invalid")
    if sum(visits) != SIMULATIONS:
        raise ValueError("puct_target_simulation_count_mismatch")
    if any(
        visits[action] != 0 for action in range(PIT_COUNT) if action not in legal_moves
    ):
        raise ValueError("puct_target_visits_include_illegal_move")
    return self_play.build_policy_target(
        np.asarray(visits, dtype=np.float32),
        legal_moves=legal_moves,
        temperature=float(temperature),
        mode=TARGET_SETTINGS["policy_target_mode"],
    )


def summarize_pair_accounting(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accepted rows, incomplete games, and measured CPU by arm."""
    summary: dict[str, dict[str, int | float]] = {}
    for pair in pair_results:
        arm = str(pair["arm"])
        cpu_ns = pair.get("pair_cpu_time_ns")
        process_cpu_ns = pair.get("pair_process_cpu_time_ns")
        child_cpu_ns = pair.get("native_child_cpu_time_ns")
        if not isinstance(cpu_ns, int) or cpu_ns < 0:
            raise ValueError("pair_cpu_accounting_invalid")
        if (
            not isinstance(process_cpu_ns, int)
            or process_cpu_ns < 0
            or not isinstance(child_cpu_ns, int)
            or child_cpu_ns < 0
            or cpu_ns != process_cpu_ns + child_cpu_ns
        ):
            raise ValueError("pair_child_cpu_accounting_invalid")
        game_cpu_values = [game.get("cpu_time_ns") for game in pair["games"]]
        if any(not isinstance(value, int) or value < 0 for value in game_cpu_values):
            raise ValueError("pair_game_cpu_accounting_invalid")
        game_cpu_ns = sum(game_cpu_values)
        if process_cpu_ns < game_cpu_ns:
            raise ValueError("pair_process_cpu_below_game_cpu")
        if cpu_ns < game_cpu_ns:
            raise ValueError("pair_cpu_below_game_cpu")
        bucket = summary.setdefault(
            arm,
            {
                "pairs": 0,
                "games": 0,
                "complete_games": 0,
                "capped_games": 0,
                "incomplete_games": 0,
                "accepted_rows": 0,
                "cpu_time_ns": 0,
                "native_child_cpu_time_ns": 0,
                "cpu_hours": 0.0,
            },
        )
        bucket["pairs"] += 1
        bucket["cpu_time_ns"] += cpu_ns
        bucket["native_child_cpu_time_ns"] += child_cpu_ns
        for game in pair["games"]:
            verify_game(game)
            bucket["games"] += 1
            status_key = {
                "complete": "complete_games",
                "capped": "capped_games",
                "incomplete": "incomplete_games",
            }[game["game_status"]]
            bucket[status_key] += 1
            bucket["accepted_rows"] += int(game["accepted_rows"])
        bucket["cpu_hours"] = bucket["cpu_time_ns"] / 3_600_000_000_000
    return summary


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
        max_tasks_per_child=1,
        initializer=_initialize_pair_process,
        initargs=(str(artifact_root),),
    ) as pool:
        futures = [pool.submit(_play_pair_process, job) for job in pair_jobs]
        results = [future.result() for future in futures]
    verify_archive([game for result in results for game in result["games"]])
    summarize_pair_accounting(results)
    return results


def pair_checkpoint_path(checkpoint_dir: Path, pair_id: str) -> Path:
    """Return the stable archive path for one committed pair."""
    safe_id = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
    return Path(checkpoint_dir) / f"{safe_id}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON record without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def commit_verified_pair(
    checkpoint_dir: Path,
    job: dict[str, Any],
    result: dict[str, Any],
) -> Path:
    """Verify and atomically commit one complete pair and its assignment."""
    specs = job.get("game_specs", [])
    validate_game_specs(specs)
    if len(specs) != 2:
        raise ValueError("pair_job_must_contain_two_seat_games")
    if (
        result.get("pair_id") != specs[0]["pair_id"]
        or result.get("arm") != specs[0]["arm"]
    ):
        raise ValueError("pair_checkpoint_assignment_mismatch")
    games = result.get("games")
    if not isinstance(games, list) or len(games) != 2:
        raise ValueError("pair_checkpoint_game_count_mismatch")
    verify_pair_result(job, result)
    assignment = json.loads(json.dumps(job, sort_keys=True))
    payload = {
        "schema": "varied_opponent_pair_checkpoint_v1",
        "assignment": assignment,
        "result": result,
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    path = pair_checkpoint_path(checkpoint_dir, str(specs[0]["pair_id"]))
    if path.exists():
        existing = path.read_bytes()
        if existing != encoded:
            raise ValueError("pair_checkpoint_conflict")
        return path
    write_json_atomic(path, payload)
    return path


def verify_pair_result(job: dict[str, Any], result: dict[str, Any]) -> None:
    """Bind a pair result to both frozen seat specs and reconcile its CPU."""
    specs = job.get("game_specs", [])
    games = result.get("games")
    if len(specs) != 2 or not isinstance(games, list) or len(games) != 2:
        raise ValueError("pair_checkpoint_game_count_mismatch")
    validate_game_specs(specs)
    expected_by_id = {str(spec["game_id"]): spec for spec in specs}
    observed_by_id: dict[str, dict[str, Any]] = {}
    for game in games:
        spec = game.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("pair_checkpoint_game_spec_missing")
        game_id = str(spec.get("game_id", ""))
        if game_id in observed_by_id:
            raise ValueError("pair_checkpoint_duplicate_game")
        observed_by_id[game_id] = spec
        if game_id not in expected_by_id or json.dumps(
            spec, sort_keys=True
        ) != json.dumps(expected_by_id[game_id], sort_keys=True):
            raise ValueError("pair_checkpoint_game_assignment_mismatch")
    if observed_by_id.keys() != expected_by_id.keys():
        raise ValueError("pair_checkpoint_game_assignment_mismatch")
    if (
        result.get("pair_id") != specs[0]["pair_id"]
        or result.get("arm") != specs[0]["arm"]
    ):
        raise ValueError("pair_checkpoint_assignment_mismatch")
    verify_archive(games)
    summarize_pair_accounting([result])


def load_pair_checkpoints(
    checkpoint_dir: Path,
    pair_jobs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate existing committed evidence and return completed and pending jobs."""
    expected = {str(job["game_specs"][0]["pair_id"]): job for job in pair_jobs}
    if len(expected) != len(pair_jobs):
        raise ValueError("pair_assignment_duplicate_id")
    completed: list[dict[str, Any]] = []
    directory = Path(checkpoint_dir)
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        payload = json.loads(path.read_text())
        if payload.get("schema") != "varied_opponent_pair_checkpoint_v1":
            raise ValueError("pair_checkpoint_schema_mismatch")
        job = payload.get("assignment")
        result = payload.get("result")
        if not isinstance(job, dict) or not isinstance(result, dict):
            raise ValueError("pair_checkpoint_payload_invalid")
        pair_id = str(job.get("game_specs", [{}])[0].get("pair_id", ""))
        if pair_id not in expected or json.dumps(job, sort_keys=True) != json.dumps(
            expected[pair_id], sort_keys=True
        ):
            raise ValueError("pair_checkpoint_assignment_conflict")
        if path != pair_checkpoint_path(directory, pair_id):
            raise ValueError("pair_checkpoint_path_mismatch")
        verify_pair_result(job, result)
        completed.append(result)
    done = {str(result["pair_id"]) for result in completed}
    return completed, [job for pair_id, job in expected.items() if pair_id not in done]


def pair_attempt_path(attempt_dir: Path, pair_id: str, attempt_number: int) -> Path:
    pair_hash = hashlib.sha256(pair_id.encode()).hexdigest()
    return Path(attempt_dir) / f"{pair_hash}.attempt-{attempt_number:04d}.json"


def reconcile_pair_attempts(
    attempt_dir: Path,
    pair_jobs: list[dict[str, Any]],
    committed_results: dict[str, dict[str, Any]],
    checkpoint_dir: Path,
) -> dict[str, int]:
    """Reconcile receipts to commits; only dispatched work never started may retry."""
    jobs = {str(job["game_specs"][0]["pair_id"]): job for job in pair_jobs}
    last_number: dict[str, int] = {pair_id: 0 for pair_id in jobs}
    for path in sorted(Path(attempt_dir).glob("*.json")):
        record = json.loads(path.read_text())
        pair_id = str(record.get("pair_id", ""))
        status = record.get("status")
        if pair_id not in jobs or status not in {"dispatched", "started", "completed"}:
            raise ValueError("pilot_attempt_record_invalid")
        assignment = record.get("assignment")
        if json.dumps(assignment, sort_keys=True) != json.dumps(
            jobs[pair_id], sort_keys=True
        ):
            raise ValueError("pilot_attempt_assignment_conflict")
        stem = path.stem
        try:
            number = int(stem.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("pilot_attempt_path_invalid") from exc
        last_number[pair_id] = max(last_number[pair_id], number)
        if pair_id in committed_results:
            if status == "dispatched":
                raise ValueError("pilot_attempt_status_conflicts_with_checkpoint")
            result = committed_results[pair_id]
            checkpoint = pair_checkpoint_path(checkpoint_dir, pair_id)
            checkpoint_digest = sha256(checkpoint)
            if status == "completed" and not {
                "checkpoint_sha256",
                "pair_cpu_time_ns",
            }.issubset(record):
                raise ValueError("pilot_attempt_checkpoint_receipt_conflict")
            if (
                "checkpoint_sha256" in record
                and record["checkpoint_sha256"] != checkpoint_digest
            ) or (
                "pair_cpu_time_ns" in record
                and record["pair_cpu_time_ns"] != result.get("pair_cpu_time_ns")
            ):
                raise ValueError("pilot_attempt_checkpoint_receipt_conflict")
            continue
        if status in {"started", "completed"}:
            raise ValueError("unreconciled_interrupted_attempt_cpu")
    return {pair_id: number + 1 for pair_id, number in last_number.items()}


def run_checkpointed_pairs(
    pair_jobs: list[dict[str, Any]],
    artifact_root: Path,
    checkpoint_dir: Path,
    *,
    attempt_dir: Path,
) -> list[dict[str, Any]]:
    """Resume only after validating commits; atomically persist each finished pair."""
    all_specs = [spec for job in pair_jobs for spec in job.get("game_specs", [])]
    validate_game_specs(all_specs)
    verify_components(artifact_root)
    committed, pending = load_pair_checkpoints(checkpoint_dir, pair_jobs)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    committed_results = {str(result["pair_id"]): result for result in committed}
    attempt_numbers = reconcile_pair_attempts(
        attempt_dir, pair_jobs, committed_results, checkpoint_dir
    )
    if not pending:
        return committed
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=PAIR_CONCURRENCY,
        max_tasks_per_child=1,
    ) as pool:
        future_jobs = {}
        for job in pending:
            pair_id = str(job["game_specs"][0]["pair_id"])
            attempt_path = pair_attempt_path(
                attempt_dir, pair_id, attempt_numbers[pair_id]
            )
            write_json_atomic(
                attempt_path,
                {
                    "pair_id": pair_id,
                    "status": "dispatched",
                    "assignment": job,
                },
            )
            future_jobs[
                pool.submit(_play_pair_attempt, job, str(artifact_root), attempt_path)
            ] = (job, attempt_path)
        for future in concurrent.futures.as_completed(future_jobs):
            job, attempt_path = future_jobs[future]
            result = future.result()
            checkpoint = commit_verified_pair(checkpoint_dir, job, result)
            write_json_atomic(
                attempt_path,
                {
                    "pair_id": result["pair_id"],
                    "status": "completed",
                    "assignment": job,
                    "checkpoint_sha256": sha256(checkpoint),
                    "pair_cpu_time_ns": result["pair_cpu_time_ns"],
                },
            )
            committed.append(result)
    return committed


def _initialize_pair_process(artifact_root: str) -> None:
    """Authenticate and load immutable evaluators once in each pair process."""
    global _PROCESS_EVALUATORS, _PROCESS_SETUP_CPU_NS, _PROCESS_CHILD_CPU_START_NS
    setup_start = time.process_time_ns()
    _PROCESS_CHILD_CPU_START_NS = reaped_child_cpu_ns()
    import torch

    torch.set_num_threads(WORKERS_PER_GAME)
    if torch.get_num_interop_threads() != WORKERS_PER_GAME:
        torch.set_num_interop_threads(WORKERS_PER_GAME)
    _PROCESS_EVALUATORS = load_evaluators(Path(artifact_root))
    _PROCESS_SETUP_CPU_NS = time.process_time_ns() - setup_start


def _play_pair_attempt(
    job: dict[str, Any], artifact_root: str, attempt_path: Path
) -> dict[str, Any]:
    """Durably mark actual process start before setup CPU or game work begins."""
    write_json_atomic(
        attempt_path,
        {
            "pair_id": str(job["game_specs"][0]["pair_id"]),
            "status": "started",
            "assignment": job,
        },
    )
    _initialize_pair_process(artifact_root)
    return _play_pair_process(job)


def _play_pair_process(job: dict[str, Any]) -> dict[str, Any]:
    """Run both seat assignments sequentially in one isolated pair process."""
    if _PROCESS_EVALUATORS is None:
        raise RuntimeError("pair_process_evaluators_not_initialized")
    pair_cpu_start = time.process_time_ns()
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
    process_cpu_ns = _PROCESS_SETUP_CPU_NS + time.process_time_ns() - pair_cpu_start
    child_cpu_ns = reaped_child_cpu_ns() - _PROCESS_CHILD_CPU_START_NS
    if child_cpu_ns < 0:
        raise ValueError("reaped_child_cpu_counter_regressed")
    pair_cpu_ns = process_cpu_ns + child_cpu_ns
    return {
        "pair_id": str(specs[0]["pair_id"]),
        "arm": str(specs[0]["arm"]),
        "games": games,
        "pair_process_cpu_time_ns": process_cpu_ns,
        "native_child_cpu_time_ns": child_cpu_ns,
        "pair_cpu_time_ns": pair_cpu_ns,
    }


def launch_preflight(
    specs: list[dict[str, Any]], artifact_root: Path
) -> dict[str, Any]:
    """Validate experiment inputs and execution concurrency without launching games."""
    validate_game_specs(specs)
    components = verify_components(artifact_root)
    runtime = effective_runtime_policy(artifact_root)
    runtime_check = (
        "from ml.alphazero_lite.varied_opponent_pilot import verify_live_resource_limits; "
        "verify_live_resource_limits({"
        f"'cpu_quota': {CPU_QUOTA!r}, 'cpu_weight': {CPU_WEIGHT!r}, "
        f"'memory_max_bytes': {MEMORY_MAX_BYTES}, 'allowed_cpus': {ALLOWED_CPUS!r}, "
        f"'nice': {NICE}}})"
    )
    scope_argv = systemd_scope_command(
        ["taskset", "-c", ALLOWED_CPUS, sys.executable, "-c", runtime_check]
    )
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
        "cpu_accounting": "pair_process_cpu_plus_reaped_native_child_cpu_by_arm",
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


def _parse_cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def verify_live_resource_limits(
    frozen: dict[str, Any],
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    affinity: set[int] | None = None,
    nice: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless the current process is inside the frozen resource scope."""
    unified = next(
        (
            line.split("::", 1)[1]
            for line in proc_cgroup.read_text().splitlines()
            if "::" in line
        ),
        None,
    )
    if unified is None:
        raise ValueError("pilot_unified_cgroup_missing")
    cgroup = cgroup_root / unified.lstrip("/")
    try:
        cpu_max = (cgroup / "cpu.max").read_text().strip().split()
        cpu_weight = int((cgroup / "cpu.weight").read_text().strip())
        memory_max = int((cgroup / "memory.max").read_text().strip())
        cpuset_path = cgroup / "cpuset.cpus.effective"
        cpuset_text = cpuset_path.read_text().strip() if cpuset_path.exists() else None
    except (OSError, ValueError) as exc:
        raise ValueError("pilot_live_cgroup_limits_unavailable") from exc
    quota_percent = int(str(frozen["cpu_quota"]).rstrip("%"))
    if (
        len(cpu_max) != 2
        or cpu_max[0] == "max"
        or int(cpu_max[0]) * 100 != quota_percent * int(cpu_max[1])
        or cpu_weight != int(frozen["cpu_weight"])
        or memory_max != int(frozen["memory_max_bytes"])
    ):
        raise ValueError("pilot_live_cgroup_limit_mismatch")
    expected_cpus = _parse_cpu_set(str(frozen["allowed_cpus"]))
    live_cpuset = _parse_cpu_set(cpuset_text) if cpuset_text else None
    actual_affinity = set(os.sched_getaffinity(0)) if affinity is None else affinity
    actual_nice = os.getpriority(os.PRIO_PROCESS, 0) if nice is None else nice
    if (
        live_cpuset is not None and live_cpuset != expected_cpus
    ) or actual_affinity != expected_cpus:
        raise ValueError("pilot_live_cpu_affinity_mismatch")
    if actual_nice != int(frozen["nice"]):
        raise ValueError("pilot_live_nice_mismatch")
    return {
        "cpu_max": " ".join(cpu_max),
        "cpu_weight": cpu_weight,
        "memory_max_bytes": memory_max,
        "cpuset_cpus_effective": cpuset_text,
        "affinity": sorted(actual_affinity),
        "nice": actual_nice,
    }


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
    parser.add_argument("--tablebase", type=Path)
    args = parser.parse_args()
    if args.tablebase is not None:
        os.environ["AZLITE_KALAH_V1_TABLEBASE"] = str(args.tablebase)
    specs = json.loads(args.specs.read_text())
    if not isinstance(specs, list):
        raise SystemExit("specs_file_must_contain_json_array")
    print(json.dumps(launch_preflight(specs, args.artifacts), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
