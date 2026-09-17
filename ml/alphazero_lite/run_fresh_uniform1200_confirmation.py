#!/usr/bin/env python3
"""Pre-register and aggregate the fresh matched control-vs-uniform1200 test.

The primary path deliberately knows nothing about the historical R61 cluster.
It changes only self-play search budget, never injects exact states, and never
promotes an artifact.  Training/evaluation workers may write result JSON files
independently; this module makes their provenance and paired W/D/L aggregation
deterministic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.forensic_exact_references import (  # noqa: E402
    outcome_optimal_actions,
    outcome_regret,
    outcome_utilities,
)
from ml.alphazero_lite.run_phase_specific_selfplay_budget_ablation import (  # noqa: E402
    self_play_step,
)

SCHEMA = "azlite_fresh_uniform1200_confirmation_v1"
LANES = ("control", "uniform1200")
SEEDS = (47, 48, 49)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replace_option(command: list[str], flag: str, value: int | str | None) -> list[str]:
    result = list(command)
    try:
        index = result.index(flag)
    except ValueError:
        return result if value is None else [*result, flag, str(value)]
    if value is None:
        return result[:index] + result[index + 2 :]
    result[index + 1] = str(value)
    return result


def train_step(config: dict[str, Any]) -> dict[str, Any]:
    matches = [step for step in config["steps"] if step.get("name") == "train"]
    if len(matches) != 1:
        raise ValueError("base recipe must contain exactly one train step")
    return matches[0]


def seed_sweep(seed: int) -> str:
    return ",".join(str(value) for value in (seed - 1, seed, seed + 1))


def lane_config(
    plan: dict[str, Any], base: dict[str, Any], seed: int, lane: str, workdir: Path
) -> dict[str, Any]:
    if lane not in LANES or seed not in SEEDS:
        raise ValueError("lane or seed is outside the pre-registered experiment")
    config = copy.deepcopy(base)
    settings = plan["lanes"][lane]
    config["run_id"] = f"fresh-uniform1200-s{seed}-{lane}"
    config["seed"] = seed
    config["versions_dir"] = str(workdir / "runs" / f"seed{seed}" / lane)
    config["fixed_replay_sources"] = [
        {"path": source["path"], "weight": source["weight"]}
        for source in plan["regenerated_replay_sources"]
    ]
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
    result = copy.deepcopy(config)
    for name in ("run_id", "seed", "versions_dir"):
        result.pop(name, None)
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
        elif step.get("name") == "train":
            step["command"] = replace_option(step["command"], "--seed", None)
    return result


def preflight(plan: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA or tuple(plan.get("seeds", ())) != SEEDS:
        raise ValueError("plan must retain the pre-registered 47,48,49 seed contract")
    if (
        tuple(plan.get("lanes", ())) != LANES
        or plan.get("promotion", {}).get("performed") is not False
    ):
        raise ValueError("plan must contain exactly two non-promoting lanes")
    parent = ROOT / plan["parent_artifact"] / "weights.json"
    reference = ROOT / plan["exact_reference"]
    if not parent.is_file() or not reference.is_file():
        raise FileNotFoundError(
            "current parent or exact forensic v2 reference is unavailable"
        )
    replay_sources = plan.get("regenerated_replay_sources", [])
    if [source.get("weight") for source in replay_sources] != [4, 1, 8, 4]:
        raise ValueError("current replay source weights must remain 4,1,8,4")
    for source in replay_sources:
        path = ROOT / str(source["path"])
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise ValueError(
                f"regenerated replay source mismatch: {source.get('name')}"
            )
    configs = [lane_config(plan, base, SEEDS[0], lane, ROOT / ".tmp") for lane in LANES]
    if non_budget_identity(configs[0]) != non_budget_identity(configs[1]):
        raise ValueError(
            "lanes differ outside self-play budget and seed/path identifiers"
        )
    for forbidden in (
        "r61",
        "t61",
        "t63",
        "a0",
        "descendant",
        "rowmatch",
        "exact_teacher",
    ):
        if forbidden in json.dumps(configs, sort_keys=True).lower():
            raise ValueError(f"forbidden intervention in primary config: {forbidden}")
    return {
        "parent_weights_sha256": sha256_file(parent),
        "exact_reference_sha256": sha256_file(reference),
    }


def outcome_name(row: dict[str, Any], action: int) -> str:
    utility = outcome_utilities(row)[action]
    return {1: "win", 0: "draw", -1: "loss"}[utility]


def exact_metrics(
    reference_rows: list[dict[str, Any]],
    actions: dict[str, int],
    policies: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    rows = [row for row in reference_rows if row.get("exact_status") == "exact_solved"]
    evaluated = []
    for row in rows:
        key = str(row["canonical_state"])
        action = actions[key]
        regret = outcome_regret(row, action)
        mass = (
            None
            if policies is None
            else sum(
                float(policies[key][move]) for move in outcome_optimal_actions(row)
            )
        )
        evaluated.append((row, action, int(regret), mass))

    def summarize(
        items: list[tuple[dict[str, Any], int, int, float | None]],
    ) -> dict[str, float | int | None]:
        regrets = [item[2] for item in items]
        masses = [item[3] for item in items if item[3] is not None]
        return {
            "roots": len(items),
            "outcome_optimal_top1_accuracy": statistics.fmean(
                float(value == 0) for value in regrets
            ),
            "outcome_optimal_policy_mass": statistics.fmean(masses) if masses else None,
            "mean_outcome_regret": statistics.fmean(regrets),
            "true_outcome_blunder_rate": statistics.fmean(
                float(value > 0) for value in regrets
            ),
            "win_to_draw_errors": sum(
                outcome_name(row, action) == "draw"
                and max(outcome_utilities(row).values()) == 1
                for row, action, _, _ in items
            ),
            "win_to_loss_errors": sum(
                outcome_name(row, action) == "loss"
                and max(outcome_utilities(row).values()) == 1
                for row, action, _, _ in items
            ),
        }

    return {
        "overall": summarize(evaluated),
        **{
            bucket: summarize(
                [item for item in evaluated if bucket in item[0].get("id", "")]
            )
            for bucket in ("capture_available", "sparse_endgame")
        },
    }


def paired_transitions(
    reference_rows: list[dict[str, Any]],
    control: dict[str, int],
    uniform: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for row in reference_rows:
        if row.get("exact_status") != "exact_solved":
            continue
        key = str(row["canonical_state"])
        ca, ua = control[key], uniform[key]
        cr, ur = int(outcome_regret(row, ca)), int(outcome_regret(row, ua))
        output.append(
            {
                "seed": seed,
                "state_id": row.get("id", key),
                "canonical_state": key,
                "bucket": next(
                    (
                        bucket
                        for bucket in ("capture_available", "sparse_endgame")
                        if bucket in row.get("id", "")
                    ),
                    "other",
                ),
                "control_action": ca,
                "uniform_action": ua,
                "exact_outcome_optimal_set": outcome_optimal_actions(row),
                "control_correct": cr == 0,
                "uniform_correct": ur == 0,
                "control_outcome_regret": cr,
                "uniform_outcome_regret": ur,
                "correctness_transition": f"{'correct' if cr == 0 else 'wrong'}_to_{'correct' if ur == 0 else 'wrong'}",
                "outcome_transition": f"{outcome_name(row, ca)}_to_{outcome_name(row, ua)}",
            }
        )
    return output


def repeated_changes(
    rows: list[dict[str, Any]], *, worse: bool
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        delta = row["uniform_outcome_regret"] - row["control_outcome_regret"]
        if (delta > 0) == worse and delta != 0:
            grouped.setdefault(row["canonical_state"], []).append(row)
    return [
        {
            "state_id": items[0]["state_id"],
            "canonical_state": key,
            "bucket": items[0]["bucket"],
            "outcome_transition_by_seed": {
                str(item["seed"]): item["outcome_transition"] for item in items
            },
            "seeds_affected": sorted(item["seed"] for item in items),
            "control_actions": {
                str(item["seed"]): item["control_action"] for item in items
            },
            "uniform_actions": {
                str(item["seed"]): item["uniform_action"] for item in items
            },
            "exact_outcome_optimal_set": items[0]["exact_outcome_optimal_set"],
        }
        for key, items in sorted(grouped.items())
        if len({item["seed"] for item in items}) >= 2
    ]


def classify(
    seed_effects: list[float],
    shadow_passes: int,
    repeated_regressions: list[dict[str, Any]],
    *,
    aggregate_regression: bool,
    production_misaligned: bool,
    heterogeneous: bool,
) -> str:
    if len(seed_effects) != 3:
        return "uniform1200_fresh_experiment_inconclusive"
    robust = (
        sum(effect > 0 for effect in seed_effects) >= 2
        and statistics.fmean(seed_effects) > 0
    )
    repeated_win_loss = any(
        "win_to_loss" in row["outcome_transition_by_seed"].values()
        for row in repeated_regressions
    )
    if (
        robust
        and not aggregate_regression
        and not repeated_win_loss
        and shadow_passes >= 2
    ):
        return (
            "uniform1200_current_production_gate_misaligned"
            if production_misaligned
            else "uniform1200_fresh_strength_confirmed"
        )
    if robust and repeated_regressions:
        return "uniform1200_strength_gain_with_repeated_outcome_regression"
    if heterogeneous:
        return "uniform1200_fresh_seed_heterogeneous"
    return "uniform1200_fresh_no_strength_gain"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("ml/alphazero_lite/configs/fresh_uniform1200_confirmation.json"),
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan["base_config"]).read_text(encoding="utf-8"))
    integrity = preflight(plan, base)
    manifests = []
    for seed in SEEDS:
        for lane in LANES:
            config = lane_config(plan, base, seed, lane, args.workdir)
            path = args.workdir / "configs" / f"seed{seed}-{lane}.json"
            write_json(path, config)
            manifests.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "config": str(path),
                    "config_sha256": sha256_file(path),
                    "parent_weights_sha256": integrity["parent_weights_sha256"],
                    "self_play_seed_sweep": seed_sweep(seed),
                    "training_seed": seed,
                    "promotion": False,
                }
            )
    write_json(
        args.out,
        {
            "schema": SCHEMA,
            "status": "pre_registered_execution_pending",
            "preflight": integrity,
            "arena": plan["arena"],
            "promotion": plan["promotion"],
            "primary_excludes_r61": True,
            "manifests": manifests,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
