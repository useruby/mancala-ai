"""Pure calculation of the published E4 diagnostic report."""

from __future__ import annotations

import random
from typing import Any

from ml.alphazero_lite.kalah_rules import KalahGame


def validate_outcome(row: dict[str, Any], state_row: dict[str, Any]) -> None:
    """Validate recorded game trajectory using rules only, without a model/search."""
    if (
        row["opening_index"] != state_row["opening_index"]
        or row["state_hash"] != state_row["state_hash"]
    ):
        raise ValueError("resumed_outcome_state_binding_mismatch")
    if row["action"] not in ("SS", "FF") or row["budget"] not in (1536, 384):
        raise ValueError("resumed_outcome_key_invalid")
    game = KalahGame.from_state(state_row["state"])
    root_player = game.current_player
    moves = row["trajectory"]
    expected_forced = state_row[
        "ss_action_384" if row["action"] == "SS" else "ff_action_384"
    ]
    if (
        not moves
        or not moves[0].get("forced")
        or moves[0]["action_relative"] != expected_forced
    ):
        raise ValueError("resumed_outcome_forced_action_mismatch")
    for index, move in enumerate(moves):
        if game.over() or move["actor"] != game.current_player:
            raise ValueError("resumed_outcome_trajectory_after_terminal_or_wrong_actor")
        relative = int(move["action_relative"])
        absolute = int(move["action_absolute"])
        if (
            absolute != game.pit_index(relative)
            or relative not in game.possible_moves()
        ):
            raise ValueError("resumed_outcome_illegal_or_misindexed_action")
        if not game.move(absolute) or move["state"] != game.to_state():
            raise ValueError("resumed_outcome_state_transition_mismatch")
        if index:
            before = int(move.get("active_pit_stones_before_move", -1))
            previous = KalahGame.from_state(moves[index - 1]["state"])
            if before != sum(previous.pits):
                raise ValueError("resumed_outcome_active_stone_count_mismatch")
            backend = "native_kvtb_root_action_probe_v1" if before <= 16 else None
            if move.get("exact_root_backend") != backend:
                raise ValueError("resumed_outcome_native_backend_identity_mismatch")
            if before <= 16 and move.get("exact_root_solver_calls") != 1:
                raise ValueError("resumed_outcome_native_solver_call_missing")
    if not game.over() or row["stores"] != game.captured_seeds:
        raise ValueError("resumed_outcome_not_terminal_or_store_mismatch")
    margin = game.captured_seeds[root_player] - game.captured_seeds[1 - root_player]
    score = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    if (
        row["root_player"] != root_player
        or row["store_margin_root_perspective"] != margin
        or row["score"] != score
    ):
        raise ValueError("resumed_outcome_terminal_accounting_mismatch")


def calculate_report(
    registration: dict[str, Any],
    outcomes: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    states = {state["opening_index"]: state for state in registration["states"]}
    keyed: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in outcomes:
        validate_outcome(row, states[row["opening_index"]])
        key = (row["opening_index"], row["action"], row["budget"])
        if key in keyed:
            raise ValueError("duplicate_outcome")
        keyed[key] = row
    expected = {
        (s["opening_index"], a, b)
        for s in registration["states"]
        for a in ("SS", "FF")
        for b in (1536, 384)
    }
    if len(keyed) != 256 or set(keyed) != expected:
        raise ValueError("incomplete_outcome_evidence")
    matrix = []
    for state in registration["states"]:
        item = {
            "opening_index": state["opening_index"],
            "state_hash": state["state_hash"],
        }
        for budget in (1536, 384):
            ss, ff = (
                keyed[(state["opening_index"], "SS", budget)],
                keyed[(state["opening_index"], "FF", budget)],
            )
            item[str(budget)] = {
                "ss_score": ss["score"],
                "ff_score": ff["score"],
                "delta": ff["score"] - ss["score"],
                "ss_margin": ss["store_margin_root_perspective"],
                "ff_margin": ff["store_margin_root_perspective"],
            }
        matrix.append(item)
    base_by_index = {item["opening_index"]: item for item in baseline["paired_matrix"]}
    primary = [item["1536"]["delta"] for item in matrix]
    secondary = [item["384"]["delta"] for item in matrix]
    interaction = [
        value - base_by_index[item["opening_index"]]["1536"]["delta"]
        for value, item in zip(primary, matrix)
    ]

    def bootstrap(values: list[float]) -> list[float]:
        rng = random.Random(406)
        samples = sorted(
            sum(values[rng.randrange(64)] for _ in range(64)) / 64 for _ in range(10000)
        )
        return [samples[249], samples[9749]]

    mean_p, mean_s = sum(primary) / 64, sum(secondary) / 64
    interval = bootstrap(primary)
    if mean_p <= -0.03 and interval[1] < 0 and mean_s <= 0:
        decision = "SS advantage under both tested references"
    elif mean_p >= 0.03 and interval[0] > 0 and mean_s >= 0:
        decision = "Advantage reverses with reference"
    else:
        decision = "Reference robustness unresolved"
    return {
        "schema": "seed398-e4-reference-diagnostic-analysis-v1",
        "decision": decision,
        "primary": {
            "budget": 1536,
            "mean_delta": mean_p,
            "paired_bootstrap_95_interval": interval,
        },
        "secondary": {"budget": 384, "mean_delta": mean_s},
        "interaction_E4_minus_seed455": {
            "mean_delta": sum(interaction) / 64,
            "paired_bootstrap_95_interval": bootstrap(interaction),
            "interpretation": "descriptive",
        },
        "paired_matrix": matrix,
        "interpretation": "Retrospective first-action comparison; does not establish minimax quality, overall strength, or promotion eligibility.",
    }
