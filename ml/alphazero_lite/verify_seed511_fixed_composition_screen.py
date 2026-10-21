"""Independent replay and publication verification for seed511."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import seed511_fixed_composition_screen as protocol
from ml.alphazero_lite.verify_seed488_strength_evaluation import _replay_game


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_assets(root: Path) -> dict[str, Any]:
    return json.loads((root / protocol.OUT / "registration.json").read_text())[
        "runtime_assets"
    ]


def verify_pair(
    root: Path,
    index: int,
    opening: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> None:
    out = root / protocol.OUT
    batches = out / "batches"
    game_path = batches / f"games-{index:04d}.jsonl"
    report_path = batches / f"report-{index:04d}.json"
    opening_path = batches / f"opening-{index:04d}.jsonl"
    if not all(path.is_file() for path in (game_path, report_path, opening_path)):
        raise ValueError(f"pair_shape:{index}")
    games, report = _rows(game_path), json.loads(report_path.read_text())
    if (
        opening_path.read_bytes() != protocol.canonical_json(opening) + b"\n"
        or len(games) != 2
        or report.get("games_played") != 2
    ):
        raise ValueError(f"pair_shape:{index}")
    notes = report.get("notes", {})
    composition = report.get("seed511_composition", {})
    if (
        {int(row["challenger_player"]) for row in games} != {0, 1}
        or notes.get("challenger_simulations") != 128
        or notes.get("current_simulations") != 128
        or notes.get("workers_used") != 1
        or notes.get("workers_requested") != 1
        or notes.get("base_seed") != protocol.SEED + index
        or notes.get("seed_contract") != "azlite_eval_seed_v2"
        or notes.get("search_profile", {}).get("c_puct") != 1.25
        or notes.get("search_profile", {}).get("exact_leaf_solve_mode") != "disabled"
        or notes.get("search_profile", {}).get("exact_root_objective")
        != "final_score_margin"
        or notes.get("search_profile", {}).get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("exact_root_solve_threshold") != 16
        or notes.get("exact_leaf_solve_mode") != "disabled"
        or notes.get("exact_root_objective") != "final_score_margin"
        or notes.get("exact_root_tie_rule")
        != "highest_legal_network_prior_then_lowest_move_index"
        or notes.get("exact_root_native_probe_sha256")
        != _runtime_assets(root)["exact_root_native_probe"]["sha256"]
        or notes.get("exact_root_tablebase_sha256")
        != _runtime_assets(root)["exact_root_tablebase"]["sha256"]
        or composition.get("policy_source") != "E4"
        or composition.get("value_source") != "initializer"
        or composition.get("opponent") != "complete initializer artifact"
        or composition.get("components")
        != json.loads((out / "registration.json").read_text())["components"]
        or composition.get("runtime_assets") != _runtime_assets(root)
    ):
        raise ValueError(f"pair_runtime:{index}")
    expected_options = protocol.base.arena_effective_search_options(
        json.loads((out / "registration.json").read_text())
    )
    for resolved in (
        notes.get("search_options"),
        notes.get("search_profile", {}).get("search_options"),
    ):
        if resolved != expected_options:
            raise ValueError(f"pair_search_options:{index}")
    arm_rows = [row for row in ledger if int(row["opening_index"]) == index]
    if len(arm_rows) != 2 or {int(row["challenger_player"]) for row in arm_rows} != {
        0,
        1,
    }:
        raise ValueError(f"pair_ledger:{index}")
    for row in arm_rows:
        position = int(row["batch_game_position"])
        game = games[position]
        if (
            row["batch_game_sha256"] != _sha(game_path)
            or row["batch_report_sha256"] != _sha(report_path)
            or row["batch_opening_sha256"] != _sha(opening_path)
            or int(game["challenger_player"]) != int(row["challenger_player"])
        ):
            raise ValueError(f"pair_receipt:{index}:{position}")
        winner, _, _ = _replay_game(opening, game, int(row["challenger_player"]))
        if winner != row["winner"]:
            raise ValueError(f"pair_replay:{index}:{position}")


def analyze_independently(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    scores = {i: [] for i in range(protocol.OPENINGS)}
    point = {"challenger": 1.0, "draw": 0.5, "current": 0.0, "initializer": 0.0}
    for row in ledger:
        scores[int(row["opening_index"])].append(point[row["winner"]])
    if len(ledger) != protocol.GAMES or any(len(x) != 2 for x in scores.values()):
        raise ValueError("result_pair_count")
    clusters = np.asarray([np.mean(scores[i]) for i in range(protocol.OPENINGS)])
    rng = np.random.default_rng(511)
    draws = clusters[
        rng.integers(0, protocol.OPENINGS, (20_000, protocol.OPENINGS))
    ].mean(axis=1)
    wins = sum(row["winner"] == "challenger" for row in ledger)
    ties = sum(row["winner"] == "draw" for row in ledger)
    lower = float(np.quantile(draws, 0.05))
    return {
        "games": len(ledger),
        "wins": wins,
        "draws": ties,
        "losses": len(ledger) - wins - ties,
        "score": (wins + 0.5 * ties) / protocol.GAMES,
        "bootstrap_resamples": 20_000,
        "bootstrap_seed": 511,
        "one_sided_95_lower": lower,
        "descriptive_two_sided_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "pass": lower > 0.50,
        "promotion": False,
        "fresh_confirmation": False,
        "interpretation": "exploratory fixed-composition screen only",
    }


def verify(root: Path = protocol.ROOT) -> dict[str, Any]:
    root = root.resolve()
    out = root / protocol.OUT
    registration_path = out / "registration.json"
    registration = json.loads(registration_path.read_text())
    preflight = protocol.preflight(root, require_unstarted=False)
    freeze = json.loads((out / "execution-freeze.json").read_text())
    ledger = _rows(out / "outcomes.jsonl")
    if len(ledger) != protocol.GAMES:
        raise ValueError("outcome_ledger_count")
    openings = _rows(out / "openings.jsonl")
    for i, opening in enumerate(openings):
        verify_pair(root, i, opening, ledger)
    command_rows = _rows(out / "commands.jsonl")
    if not command_rows or any(
        int(row["opening_index"]) not in range(1024) for row in command_rows
    ):
        raise ValueError("resolved_command_coverage")
    expected_options = {
        "challenger_policy": "E4",
        "challenger_value": "initializer",
        "challenger_policy_trunk": "E4 own trunk",
        "challenger_value_trunk": "initializer own trunk",
        "opponent": "complete initializer artifact",
    }
    for row in command_rows:
        index = int(row["opening_index"])
        if (
            row.get("argv") != protocol.resolved_command(root, index)
            or row.get("component_assignments") != expected_options
            or row.get("component_hashes") != registration["components"]
            or row.get("runtime") != registration["runtime"]
            or row.get("registration_sha256") != freeze["registration_sha256"]
            or row.get("suite_sha256") != freeze["suite_sha256"]
        ):
            raise ValueError(f"resolved_command_contract:{index}")
    if {int(row["opening_index"]) for row in command_rows} != set(range(1024)):
        raise ValueError("resolved_command_opening_coverage")
    result = analyze_independently(ledger)
    published = json.loads((out / "results.json").read_text())
    if result != published:
        raise ValueError("independent_analysis_mismatch")
    return {
        "status": "verified_complete",
        "preflight": preflight,
        "games": len(ledger),
        "results": result,
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
