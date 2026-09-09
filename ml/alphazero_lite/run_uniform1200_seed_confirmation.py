#!/usr/bin/env python3
"""Run the locked fresh-seed confirmation of PR #286's uniform-1200 lane.

This runner intentionally changes only the two self-play budget flags and the
matched stochastic seed contract. It never promotes or mutates ``current``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_phase_specific_selfplay_budget_ablation import self_play_step  # noqa: E402

SCHEMA = "azlite_uniform1200_seed_confirmation_v1"
EXPECTED_PARENT_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
LANES = ("control_384_192", "uniform1200")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replace_option(command: list[str], flag: str, value: int | str | None) -> list[str]:
    rendered = list(command)
    try:
        index = rendered.index(flag)
    except ValueError:
        return rendered if value is None else [*rendered, flag, str(value)]
    if value is None:
        return rendered[:index] + rendered[index + 2 :]
    rendered[index + 1] = str(value)
    return rendered


def seed_sweep(seed: int) -> str:
    """Return the fresh three-value pool used by self_play's per-game RNG."""
    return ",".join(str(value) for value in (seed - 1, seed, seed + 1))


def train_step(config: dict[str, Any]) -> dict[str, Any]:
    matches = [step for step in config["steps"] if step.get("name") == "train"]
    if len(matches) != 1:
        raise ValueError("base config must have exactly one train step")
    return matches[0]


def lane_config(
    plan: dict[str, Any], base: dict[str, Any], *, seed: int, lane: str, workdir: Path
) -> dict[str, Any]:
    if lane not in LANES:
        raise ValueError("confirmation permits only the control and uniform1200 lanes")
    config = copy.deepcopy(base)
    config["run_id"] = f"uniform1200-confirm-seed{seed}-{lane}"
    config["seed"] = seed
    config["versions_dir"] = str(workdir / "runs" / f"seed{seed}" / lane)
    config["fixed_replay_sources"] = [
        {"path": plan["selected_replay"], "weight": 1},
        {"path": plan["controls_replay"], "weight": 2},
    ]
    settings = plan["lanes"][lane]
    self_play = self_play_step(config)
    command = replace_option(self_play["command"], "--seed", seed)
    command = replace_option(command, "--seed-sweep", seed_sweep(seed))
    command = replace_option(command, "--simulations", settings["simulations"])
    command = replace_option(
        command, "--opening-min-simulations", settings["opening_min_simulations"]
    )
    self_play["command"] = replace_option(
        command,
        "--opening-min-simulations-plies",
        settings["opening_min_simulations_plies"],
    )
    train = train_step(config)
    train["command"] = replace_option(train["command"], "--seed", seed)
    return config


def non_budget_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Stable comparison view: only path/run IDs, seeds, and budget flags vary."""
    result = copy.deepcopy(config)
    for key in ("run_id", "seed", "versions_dir"):
        result.pop(key, None)
    for step in result["steps"]:
        if step.get("name") == "self_play":
            for flag in (
                "--seed",
                "--seed-sweep",
                "--simulations",
                "--opening-min-simulations",
                "--opening-min-simulations-plies",
            ):
                step["command"] = replace_option(step["command"], flag, None)
        if step.get("name") == "train":
            step["command"] = replace_option(step["command"], "--seed", None)
    return result


def preflight(plan: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA or tuple(plan.get("lanes", ())) != LANES:
        raise ValueError("plan must contain exactly control_384_192 and uniform1200")
    if plan.get("seeds") != [44, 45, 46]:
        raise ValueError(
            "confirmation seed contract must be the fresh 44,45,46 sequence"
        )
    if plan.get("promotion", {}).get("performed") is not False:
        raise ValueError("confirmation must explicitly disable promotion")
    parent = Path(plan["current_path"]) / "weights.json"
    if sha256_file(parent) != EXPECTED_PARENT_SHA256:
        raise ValueError("current parent does not match PR #286's incumbent")
    replays = {}
    for name, path_key in (
        ("selected", "selected_replay"),
        ("controls", "controls_replay"),
    ):
        path = Path(plan[path_key])
        digest = sha256_file(path)
        if digest != plan["expected_replays"][name]:
            raise ValueError(f"{name} replay SHA-256 differs from PR #286")
        replays[name] = {"path": str(path), "sha256": digest}
    configs = [
        lane_config(plan, base, seed=44, lane=lane, workdir=Path("/tmp/unused"))
        for lane in LANES
    ]
    if non_budget_identity(configs[0]) != non_budget_identity(configs[1]):
        raise ValueError("lanes differ outside the pre-registered budget schedule")
    return {"parent_weights_sha256": EXPECTED_PARENT_SHA256, "fixed_replays": replays}


def replay_diagnostics(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    starts = [
        index for index, row in enumerate(rows) if int(row.get("move_index", -1)) == 0
    ]
    game_rows = [
        rows[start : starts[index + 1] if index + 1 < len(starts) else len(rows)]
        for index, start in enumerate(starts)
    ]
    games = [game[0] for game in game_rows if game]
    lengths = Counter(len(game) for game in game_rows)
    canonical = [json.dumps(row["state"], separators=(",", ":")) for row in rows]
    outcomes = Counter(str(row.get("winner")) for row in games)
    by_first_player = {
        str(player): Counter(
            str(row.get("winner"))
            for row in games
            if int(row.get("player", -1)) == player
        )
        for player in (0, 1)
    }
    simulations = sum(int(row.get("simulations", 0)) for row in rows)
    return {
        "games_completed": len(games),
        "replay_rows": len(rows),
        "average_rows_per_game": len(rows) / len(games) if games else None,
        "game_length_distribution": dict(sorted(lengths.items())),
        "canonical_unique_states": len(set(canonical)),
        "duplicate_rate": 1 - (len(set(canonical)) / len(canonical))
        if canonical
        else 0.0,
        "outcome_distribution": dict(outcomes),
        "first_second_player_outcome_distribution": by_first_player,
        "replay_sha256": sha256_file(path),
        "total_selfplay_search_simulations": simulations,
    }


def deterministic_exact_audit_sample(
    rows: list[dict[str, Any]], *, opening: bool, count: int = 128
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if (int(row.get("move_index", 0)) < 8) == opening]
    unique = {
        json.dumps(row["state"], separators=(",", ":")): row for row in candidates
    }
    return [
        unique[key]
        for key in sorted(
            unique, key=lambda key: hashlib.sha256(key.encode()).hexdigest()
        )[:count]
    ]


def iteration_dir(workdir: Path, *, seed: int, lane: str) -> Path:
    run_id = f"uniform1200-confirm-seed{seed}-{lane}"
    return workdir / "runs" / f"seed{seed}" / lane / f"{run_id}-iter1"


def classify(
    seed_effects: list[float],
    gate_passes: int,
    *,
    credible_regression: bool,
    oracle_contradictory: bool,
) -> str:
    wins = sum(effect > 0 for effect in seed_effects)
    if (
        wins >= 2
        and sum(seed_effects) / len(seed_effects) > 0
        and not credible_regression
        and gate_passes >= 2
        and not oracle_contradictory
    ):
        return "uniform1200_confirmed"
    if wins < 2 or credible_regression:
        return "uniform1200_seed_sensitive"
    return "uniform1200_not_confirmed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("ml/alphazero_lite/configs/uniform1200_seed_confirmation.json"),
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads(Path(plan["base_config"]).read_text(encoding="utf-8"))
    integrity = preflight(plan, base)
    runs: dict[str, Any] = {}
    for seed in plan["seeds"]:
        for lane in LANES:
            config = lane_config(plan, base, seed=seed, lane=lane, workdir=args.workdir)
            config_path = args.workdir / "configs" / f"seed{seed}-{lane}.json"
            write_json(config_path, config)
            run = {
                "config": str(config_path),
                "seed_contract": {
                    "self_play_seed": seed,
                    "self_play_seed_sweep": seed_sweep(seed),
                    "training_seed": seed,
                    "evaluation_seed": plan["evaluation"]["seed"],
                },
            }
            if not args.dry_run:
                started = time.monotonic()
                replay = (
                    iteration_dir(args.workdir, seed=seed, lane=lane)
                    / "self_play.jsonl"
                )
                if not replay.is_file():
                    subprocess.run(
                        [
                            sys.executable,
                            "ml/alphazero_lite/pipeline.py",
                            "--config",
                            str(config_path),
                        ],
                        cwd=REPO_ROOT,
                        check=True,
                    )
                run["self_play_wall_seconds"] = round(time.monotonic() - started, 3)
                run["replay_diagnostics"] = replay_diagnostics(replay)
            runs[f"seed{seed}_{lane}"] = run
    write_json(
        args.out_summary,
        {
            "schema": SCHEMA,
            "status": "dry_run"
            if args.dry_run
            else "pipelines_completed_evaluation_pending",
            "promotion": plan["promotion"],
            "preflight": integrity,
            "runs": runs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
