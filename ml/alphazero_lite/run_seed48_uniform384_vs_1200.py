#!/usr/bin/env python3
"""Run the non-promoting matched seed48 uniform384 versus uniform1200 ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.run_phase_specific_selfplay_budget_ablation import (  # noqa: E402
    self_play_step,
)
from ml.alphazero_lite.run_seed48_target_quality_audit import (  # noqa: E402
    quality,
    search,
)
from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.self_play import state_hash  # noqa: E402

SCHEMA = "azlite_seed48_uniform384_vs_1200_v1"
LANES = ("uniform384", "uniform1200")
EXPECTED_REPLAY_WEIGHTS = (4, 1, 8, 4)


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


def option_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except ValueError:
        return None


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
    if seed not in plan["training_seeds"] or lane not in LANES:
        raise ValueError("seed or lane is outside the registered ablation")
    config = copy.deepcopy(base)
    config["run_id"] = f"seed48-uniform384-vs-1200-s{seed}-{lane}"
    config["seed"] = seed
    config["versions_dir"] = str(workdir / "runs" / f"seed{seed}" / lane)
    config["preserve_config_workers"] = True
    config["current_path"] = str(ROOT / plan["parent_artifact"])
    config["parent_artifact_path"] = str(ROOT / plan["parent_artifact"])
    config["fixed_replay_sources"] = [
        {"path": str(ROOT / source["path"]), "weight": source["weight"]}
        for source in plan["fixed_replay_sources"]
    ]
    self_play = self_play_step(config)
    command = replace_option(self_play["command"], "--seed", seed)
    command = replace_option(command, "--seed-sweep", seed_sweep(seed))
    command = replace_option(
        command, "--simulations", plan["lanes"][lane]["simulations"]
    )
    command = replace_option(command, "--opening-min-simulations", None)
    command = replace_option(command, "--opening-min-simulations-plies", None)
    if "--write-game-metadata" not in command:
        command.append("--write-game-metadata")
    self_play["command"] = command
    train = train_step(config)
    train["command"] = replace_option(train["command"], "--seed", seed)
    return config


def non_budget_identity(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for name in ("run_id", "seed", "versions_dir"):
        result.pop(name, None)
    for step in result["steps"]:
        if step.get("name") == "self_play":
            for flag in ("--seed", "--seed-sweep", "--simulations"):
                step["command"] = replace_option(step["command"], flag, None)
        if step.get("name") == "train":
            step["command"] = replace_option(step["command"], "--seed", None)
    return result


def audit_hashes(plan: dict[str, Any]) -> set[str]:
    corpus = json.loads(
        (ROOT / plan["exact_corpus"]["path"]).read_text(encoding="utf-8")
    )
    if corpus.get("scope", {}).get("training_eligible") is not False:
        raise ValueError("PR #340 corpus must remain training_eligible: false")
    return {str(row["state_hash"]) for row in corpus["states"]}


def encoded_row_hash(row: dict[str, Any]) -> str:
    state = row["state"]
    decoded = {
        "player_pits": [round(value * 48) for value in state[:6]],
        "opponent_pits": [round(value * 48) for value in state[6:12]],
        "player_store": round(state[12] * 48),
        "opponent_store": round(state[13] * 48),
        "current_player": round(state[14]),
    }
    return state_hash(KalahGame.from_state(decoded).to_state())


def jsonl_audit_collisions(path: Path, audit_hash_set: set[str]) -> list[str]:
    return sorted(
        {
            encoded_row_hash(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and encoded_row_hash(json.loads(line)) in audit_hash_set
        }
    )


def preflight(plan: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA or tuple(plan.get("training_seeds", ())) != (
        401,
        407,
        413,
        419,
        425,
        431,
    ):
        raise ValueError("plan must retain all six registered training seeds")
    if (
        tuple(plan.get("lanes", ())) != LANES
        or plan.get("promotion", {}).get("performed") is not False
    ):
        raise ValueError("plan must contain exactly two non-promoting lanes")
    parent = ROOT / plan["parent_artifact"] / "weights.json"
    if (
        not parent.is_file()
        or sha256_file(parent) != plan["expected_parent_weights_sha256"]
    ):
        raise ValueError(
            "seed48 parent weights SHA does not match the registered incumbent"
        )
    sources = plan["fixed_replay_sources"]
    if tuple(source.get("weight") for source in sources) != EXPECTED_REPLAY_WEIGHTS:
        raise ValueError("fixed replay weights must remain 4,1,8,4")
    audit_hash_set = audit_hashes(plan)
    for source in sources:
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise FileNotFoundError("budget_ablation_replay_provenance_unavailable")
        collisions = jsonl_audit_collisions(path, audit_hash_set)
        if collisions:
            raise ValueError(
                "PR #340 evaluation corpus is deliberately included in fixed replay"
            )
    sweeps = [set(seed_sweep(seed).split(",")) for seed in plan["training_seeds"]]
    if any(
        left & right
        for index, left in enumerate(sweeps)
        for right in sweeps[index + 1 :]
    ):
        raise ValueError("self-play seed sweeps must not overlap")
    configs = [
        lane_config(plan, base, 401, lane, ROOT / ".tmp/seed48-budget-ablation")
        for lane in LANES
    ]
    if non_budget_identity(configs[0]) != non_budget_identity(configs[1]):
        raise ValueError(
            "lanes differ outside --simulations and registered run identifiers"
        )
    for config, lane in zip(configs, LANES):
        command = self_play_step(config)["command"]
        if option_value(command, "--simulations") != str(
            plan["lanes"][lane]["simulations"]
        ):
            raise ValueError("lane simulation budget was not rendered")
        if option_value(command, "--opening-min-simulations") is not None:
            raise ValueError("hidden opening budget is forbidden")
        if option_value(command, "--workers") != "6":
            raise ValueError("self-play workers must remain 6")
    suite = Path(plan["evaluation"]["suite"])
    if not suite.is_file() or sha256_file(suite) != plan["evaluation"]["suite_sha256"]:
        raise FileNotFoundError("canonical evaluation suite unavailable")
    return {
        "parent_weights_sha256": sha256_file(parent),
        "audit_hash_count": len(audit_hash_set),
        "replay_sources": sources,
    }


def search_work(path: Path, audit_hash_set: set[str]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("self-play emitted no rows")
    hashes = {encoded_row_hash(row) for row in rows}
    collisions = sorted(hashes & audit_hash_set)
    # Denoised targets run a separate noise-free teacher search at every root.
    searches_per_root = (
        2
        if all(row.get("policy_target_noise_mode") == "denoised" for row in rows)
        else 1
    )
    requested = sum(int(row["simulations"]) for row in rows)
    games = {int(row["game_index"]) for row in rows if "game_index" in row}
    return {
        "generated_rows": len(rows),
        "games_completed": len(games),
        "searched_roots": len(rows),
        "teacher_searches_per_root": searches_per_root,
        "total_requested_simulations_across_roots": requested * searches_per_root,
        "simulations_per_generated_row": (requested * searches_per_root) / len(rows),
        "simulations_per_game": (requested * searches_per_root) / len(games),
        "search_profile_hashes": sorted({row["search_profile_hash"] for row in rows}),
        "audit_corpus_state_collisions": collisions,
    }


def self_play_duration(run_dir: Path) -> float:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    steps = [step for step in manifest["steps"] if step.get("name") == "self_play"]
    if len(steps) != 1 or "duration_s" not in steps[0]:
        raise ValueError("completed run manifest must contain one timed self_play step")
    return float(steps[0]["duration_s"])


def completed_lane_provenance(run_dir: Path) -> dict[str, Any]:
    """Extract reproducibility fields from an already completed pipeline run."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    steps = {step["name"]: step for step in manifest["steps"]}
    self_play_log = (run_dir / "self_play.log").read_text(encoding="utf-8")
    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    cache = re.search(r"cache_hits=(\d+) cache_misses=(\d+)", self_play_log)
    losses = {
        key: float(value)
        for key, value in re.findall(
            r"^(policy_loss|value_loss|total_loss|best_val_loss)=([0-9.]+)$",
            train_log,
            flags=re.MULTILINE,
        )
    }
    if cache is None or len(losses) != 4:
        raise ValueError(f"incomplete completed-run telemetry: {run_dir}")
    return {
        "candidate_checkpoint_sha256": sha256_file(run_dir / "checkpoint.npz"),
        "exported_weights_sha256": sha256_file(run_dir / "weights.json"),
        "step_commands": {name: step["command"] for name, step in steps.items()},
        "step_durations_s": {
            name: step.get("duration_s") for name, step in steps.items()
        },
        "cache": {"hits": int(cache.group(1)), "misses": int(cache.group(2))},
        "training_losses": losses,
        "perspective_audit": json.loads(
            (run_dir / "perspective_audit.json").read_text(encoding="utf-8")
        ),
    }


def finalize_completed_result(result: dict[str, Any]) -> dict[str, Any]:
    for seed_result in result["seed_results"]:
        for lane_result in seed_result["lanes"].values():
            run_dir = Path(lane_result["artifact"])
            lane_result["completed_run_provenance"] = completed_lane_provenance(run_dir)
    result["commands"] = [
        command
        for seed_result in result["seed_results"]
        for lane_result in seed_result["lanes"].values()
        for command in lane_result["completed_run_provenance"]["step_commands"].values()
    ]
    result["report_regenerated_from_completed_artifacts"] = True
    return result


def paired_bootstrap(
    values: list[float], *, seed: int, samples: int
) -> dict[str, float]:
    if len(values) != 6:
        raise ValueError("training-seed bootstrap requires exactly six paired values")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(values[rng.randrange(6)] for _ in range(6))
        for _ in range(samples)
    )
    return {
        "mean": statistics.fmean(values),
        "ci95_lower": means[int(samples * 0.025)],
        "ci95_upper": means[int(samples * 0.975)],
    }


def classify(aggregate: dict[str, Any], *, margin: float) -> str:
    sibling = aggregate["sibling_effect"]
    if sibling["ci95_lower"] > margin and aggregate["work_ratio_384_over_1200"] <= 0.4:
        return "uniform384_training_noninferior_and_cheaper"
    if sibling["ci95_upper"] < margin:
        return "uniform1200_training_advantage_confirmed"
    if any(
        value > 0 and other < 0
        for value, other in zip(
            aggregate["sibling_effect"]["values"],
            aggregate["parent_gain_difference"]["values"],
        )
    ):
        return "training_budget_seed_heterogeneous"
    return "training_budget_seed_heterogeneous"


def evaluate_exact(artifact: Path, plan: dict[str, Any], seed: int) -> dict[str, Any]:
    corpus = json.loads(
        (ROOT / plan["exact_corpus"]["path"]).read_text(encoding="utf-8")
    )["states"]
    evaluator = ArtifactEvaluator(artifact)
    output: dict[str, list[dict[str, Any]]] = {"raw": [], "mcts_384": []}
    for index, row in enumerate(corpus):
        state, exact, legal = row["canonical_state"], row["exact"], row["legal_moves"]
        exact = {
            **exact,
            "exact_regret_by_move": {
                int(move): value
                for move, value in exact["exact_regret_by_move"].items()
            },
            "exact_optimal_moves": [int(move) for move in exact["exact_optimal_moves"]],
        }
        prior, _ = evaluator.evaluate(KalahGame.from_state(state))
        output["raw"].append(quality(prior.tolist(), exact, legal))
        outcome = search(evaluator, state, 384, seed + index)
        output["mcts_384"].append(quality(outcome["policy"], exact, legal))
    return {
        method: {
            metric: statistics.fmean(item[metric] for item in rows)
            for metric in (
                "top_move_optimal",
                "optimal_mass",
                "top_move_regret",
                "expected_regret",
            )
        }
        for method, rows in output.items()
    }


def run_arena_pair(
    *, challenger: Path, current: Path, plan: dict[str, Any], out_dir: Path, label: str
) -> dict[str, Any]:
    """Run 128 forced games per seat under one pinned equal-budget contract."""
    from ml.alphazero_lite.run_opening_suite_seat_benchmark import run_arena

    evaluation = plan["evaluation"]
    reports = []
    commands = []
    for challenger_starts in (0, 1):
        prefix = out_dir / f"{label}-starts-{challenger_starts}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        report = run_arena(
            challenger=str(challenger),
            current=str(current),
            challenger_sims=int(evaluation["simulations"]),
            current_sims=int(evaluation["simulations"]),
            games=128,
            seed=int(evaluation["seed"]),
            workers=int(evaluation["workers"]),
            out_json=str(prefix.with_suffix(".json")),
            out_jsonl=str(prefix.with_suffix(".jsonl")),
            opening_prefixes_jsonl=str(evaluation["suite"]),
            challenger_starts=challenger_starts,
            games_per_opening=int(evaluation["games_per_opening"]),
            suite_sha256=str(evaluation["suite_sha256"]),
            seed_contract=str(evaluation["seed_contract"]),
        )
        reports.append(report)
        commands.append(
            [
                "arena.py",
                "--challenger-simulations",
                str(evaluation["simulations"]),
                "--current-simulations",
                str(evaluation["simulations"]),
                "--seed",
                str(evaluation["seed"]),
                "--seed-contract",
                str(evaluation["seed_contract"]),
                "--opening-prefixes-jsonl",
                str(evaluation["suite"]),
                "--games-per-opening",
                str(evaluation["games_per_opening"]),
                "--challenger-starts",
                str(challenger_starts),
                "--root-policy-mode",
                "deterministic",
                "--root-temperature",
                "0.0",
            ]
        )
    wins = sum(int(report["wins"]) for report in reports)
    draws = sum(int(report["draws"]) for report in reports)
    losses = sum(int(report["losses"]) for report in reports)
    total = wins + draws + losses
    contract = {
        "suite_sha256": evaluation["suite_sha256"],
        "games_per_opening": evaluation["games_per_opening"],
        "simulations": evaluation["simulations"],
        "seed": evaluation["seed"],
        "seed_contract": evaluation["seed_contract"],
        "root_policy_mode": "deterministic",
        "root_temperature": 0.0,
    }
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": total,
        "score": (wins + (0.5 * draws)) / total,
        "seat_reports": reports,
        "commands": commands,
        "evaluation_config": contract,
        "evaluation_config_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def run_regressions(artifact: Path, out_path: Path) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / "script/ai/check_superhuman_regressions"),
        "--artifact",
        str(artifact),
        "--simulations",
        "384",
        "--root-policy-mode",
        "deterministic",
        "--out",
        str(out_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 1) or not out_path.is_file():
        raise RuntimeError(
            f"regression command failed without a report: {completed.returncode}"
        )
    report = json.loads(out_path.read_text(encoding="utf-8"))
    failures = [item["id"] for item in report["results"] if not item["passed"]]
    return {
        "passed": report["passed"],
        "pass_count": len(report["results"]) - len(failures),
        "failure_ids": failures,
        "selected_moves": {
            item["id"]: item["selected_move"] for item in report["results"]
        },
        "report_sha256": sha256_file(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "ml/alphazero_lite/configs/seed48_uniform384_vs_1200.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/seed48-uniform384-vs-1200"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "docs/data/alphazero-lite-seed48-uniform384-vs-1200/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        write_json(args.out, finalize_completed_result(existing))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan["base_config"]).read_text(encoding="utf-8"))
    try:
        integrity = preflight(plan, base)
    except FileNotFoundError as error:
        if str(error) != "budget_ablation_replay_provenance_unavailable":
            raise
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": str(error),
                "promotion": plan["promotion"],
            },
        )
        return 0
    manifests = []
    for seed in plan["training_seeds"]:
        for lane in LANES:
            config = lane_config(plan, base, seed, lane, args.workdir)
            path = args.workdir / "configs" / f"seed{seed}-{lane}.json"
            write_json(path, config)
            manifests.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "config_path": str(path),
                    "config_sha256": sha256_file(path),
                    "parent_weights_sha256": integrity["parent_weights_sha256"],
                    "self_play_seed_sweep": seed_sweep(seed),
                    "promotion": False,
                }
            )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "classification": "execution_pending",
        "promotion": plan["promotion"],
        "preflight": integrity,
        "manifests": manifests,
        "commands": [],
    }
    if not args.execute:
        write_json(args.out, result)
        return 0
    audit_set = audit_hashes(plan)
    seed_results = []
    for seed in plan["training_seeds"]:
        lanes: dict[str, Any] = {}
        for lane in LANES:
            config_path = args.workdir / "configs" / f"seed{seed}-{lane}.json"
            run_dir = (
                args.workdir
                / "runs"
                / f"seed{seed}"
                / lane
                / f"seed48-uniform384-vs-1200-s{seed}-{lane}-iter1"
            )
            selfplay = run_dir / "self_play.jsonl"
            artifact = run_dir
            if not (selfplay.is_file() and (artifact / "weights.json").is_file()):
                command = [
                    sys.executable,
                    "ml/alphazero_lite/pipeline.py",
                    "--config",
                    str(config_path),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
            lanes[lane] = {
                "artifact": str(artifact),
                "candidate_weights_sha256": sha256_file(artifact / "weights.json"),
                "self_play_sha256": sha256_file(selfplay),
                "self_play_wall_seconds": self_play_duration(run_dir),
                "search_work": search_work(selfplay, audit_set),
                "exact": evaluate_exact(artifact, plan, seed),
                "regression": run_regressions(
                    artifact,
                    args.workdir / "evaluations" / f"seed{seed}-{lane}-regression.json",
                ),
            }
        sibling = run_arena_pair(
            challenger=Path(lanes["uniform384"]["artifact"]),
            current=Path(lanes["uniform1200"]["artifact"]),
            plan=plan,
            out_dir=args.workdir / "evaluations",
            label=f"seed{seed}-384-vs-1200",
        )
        parent_384 = run_arena_pair(
            challenger=Path(lanes["uniform384"]["artifact"]),
            current=ROOT / plan["parent_artifact"],
            plan=plan,
            out_dir=args.workdir / "evaluations",
            label=f"seed{seed}-384-vs-parent",
        )
        parent_1200 = run_arena_pair(
            challenger=Path(lanes["uniform1200"]["artifact"]),
            current=ROOT / plan["parent_artifact"],
            plan=plan,
            out_dir=args.workdir / "evaluations",
            label=f"seed{seed}-1200-vs-parent",
        )
        seed_results.append(
            {
                "seed": seed,
                "lanes": lanes,
                "sibling_arena": sibling,
                "parent_arenas": {"uniform384": parent_384, "uniform1200": parent_1200},
            }
        )
    samples = int(plan["analysis"]["bootstrap_samples"])
    bootstrap_seed = int(plan["analysis"]["bootstrap_seed"])
    sibling_values = [row["sibling_arena"]["score"] - 0.5 for row in seed_results]
    parent_gain_values = [
        row["parent_arenas"]["uniform384"]["score"]
        - row["parent_arenas"]["uniform1200"]["score"]
        for row in seed_results
    ]
    mass_values = [
        row["lanes"]["uniform384"]["exact"]["raw"]["optimal_mass"]
        - row["lanes"]["uniform1200"]["exact"]["raw"]["optimal_mass"]
        for row in seed_results
    ]
    regret_values = [
        row["lanes"]["uniform384"]["exact"]["raw"]["expected_regret"]
        - row["lanes"]["uniform1200"]["exact"]["raw"]["expected_regret"]
        for row in seed_results
    ]
    mcts_mass_values = [
        row["lanes"]["uniform384"]["exact"]["mcts_384"]["optimal_mass"]
        - row["lanes"]["uniform1200"]["exact"]["mcts_384"]["optimal_mass"]
        for row in seed_results
    ]
    mcts_regret_values = [
        row["lanes"]["uniform384"]["exact"]["mcts_384"]["expected_regret"]
        - row["lanes"]["uniform1200"]["exact"]["mcts_384"]["expected_regret"]
        for row in seed_results
    ]
    work_384 = sum(
        row["lanes"]["uniform384"]["search_work"][
            "total_requested_simulations_across_roots"
        ]
        for row in seed_results
    )
    work_1200 = sum(
        row["lanes"]["uniform1200"]["search_work"][
            "total_requested_simulations_across_roots"
        ]
        for row in seed_results
    )
    aggregate = {
        "experimental_unit": "matched_training_seed_pair",
        "sibling_effect": {
            "values": sibling_values,
            **paired_bootstrap(sibling_values, seed=bootstrap_seed, samples=samples),
        },
        "parent_gain_difference": {
            "values": parent_gain_values,
            **paired_bootstrap(
                parent_gain_values, seed=bootstrap_seed, samples=samples
            ),
        },
        "raw_optimal_mass_difference_384_minus_1200": {
            "values": mass_values,
            **paired_bootstrap(mass_values, seed=bootstrap_seed, samples=samples),
        },
        "raw_expected_regret_difference_384_minus_1200": {
            "values": regret_values,
            **paired_bootstrap(regret_values, seed=bootstrap_seed, samples=samples),
        },
        "mcts_384_optimal_mass_difference_384_minus_1200": {
            "values": mcts_mass_values,
            **paired_bootstrap(mcts_mass_values, seed=bootstrap_seed, samples=samples),
        },
        "mcts_384_expected_regret_difference_384_minus_1200": {
            "values": mcts_regret_values,
            **paired_bootstrap(
                mcts_regret_values, seed=bootstrap_seed, samples=samples
            ),
        },
        "work_ratio_384_over_1200": work_384 / work_1200,
    }
    result.update(
        {
            "status": "completed",
            "aggregate": aggregate,
            "classification": classify(
                aggregate,
                margin=float(plan["analysis"]["sibling_noninferiority_margin"]),
            ),
            "seed_results": seed_results,
        }
    )
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
