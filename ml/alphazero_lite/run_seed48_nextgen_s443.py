#!/usr/bin/env python3
"""Run the single pre-registered seed48 generation-N+1 candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
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
from ml.alphazero_lite.run_seed48_uniform384_vs_1200 import (  # noqa: E402
    audit_hashes,
    jsonl_audit_collisions,
    search_work,
)

SCHEMA = "azlite_seed48_generation_n_plus_1_v1"
EXPECTED_REPLAY_WEIGHTS = (4, 1, 8, 4)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def replace_option(command: list[str], flag: str, value: str | int | None) -> list[str]:
    result = list(command)
    try:
        index = result.index(flag)
    except ValueError:
        return result if value is None else [*result, flag, str(value)]
    if value is None:
        return result[:index] + result[index + 2 :]
    result[index + 1] = str(value)
    return result


def option_value(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def train_step(config: dict[str, Any]) -> dict[str, Any]:
    matches = [step for step in config["steps"] if step.get("name") == "train"]
    if len(matches) != 1:
        raise ValueError("base recipe must contain exactly one train step")
    return matches[0]


def rendered_config(
    plan: dict[str, Any], base: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    seed = plan["training_seed"]
    config["run_id"] = plan["run_id"]
    config["seed"] = seed
    config["versions_dir"] = str(workdir / "runs")
    # Pipeline otherwise applies its machine-default worker count.
    config["preserve_config_workers"] = True
    config["current_path"] = str(ROOT / plan["parent_artifact"])
    config["parent_artifact_path"] = str(ROOT / plan["parent_artifact"])
    config["fixed_replay_sources"] = [
        {"path": str(ROOT / source["path"]), "weight": source["weight"]}
        for source in plan["fixed_replay_sources"]
    ]
    self_play = self_play_step(config)
    command = self_play["command"]
    settings = plan["self_play"]
    for flag, value in (
        ("--seed", seed),
        ("--seed-sweep", ",".join(map(str, plan["self_play_seed_sweep"]))),
        ("--simulations", settings["simulations"]),
    ):
        command = replace_option(command, flag, value)
    command = replace_option(command, "--opening-min-simulations", None)
    command = replace_option(command, "--opening-min-simulations-plies", None)
    if "--write-game-metadata" not in command:
        command.append("--write-game-metadata")
    self_play["command"] = command
    train = train_step(config)
    train["command"] = replace_option(train["command"], "--seed", seed)
    return config


def seed_conflict() -> bool:
    """Only registered descendant identifiers constitute comparable provenance."""
    registered_plan = ROOT / "ml/alphazero_lite/configs/seed48_nextgen_s443.json"
    for directory in (ROOT / "docs", ROOT / "ml" / "alphazero_lite" / "configs"):
        for path in directory.rglob("*"):
            if path == registered_plan:
                continue
            if not path.is_file() or path.suffix not in {".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "seed443" in text or "s443-" in text:
                return True
    return False


def preflight(
    plan: dict[str, Any], base: dict[str, Any], workdir: Path
) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA:
        raise ValueError("registered experiment schema mismatch")
    if plan.get("training_seed") != 443 or plan.get("self_play_seed_sweep") != [
        442,
        443,
        444,
    ]:
        raise ValueError("registered seed/sweep pin mismatch")
    if seed_conflict():
        raise ValueError("registered_training_seed_conflict")
    parent = ROOT / plan["parent_artifact"]
    metadata = json.loads((parent / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("version") != plan["expected_parent_version"]
        or sha256_file(parent / "weights.json")
        != plan["expected_parent_weights_sha256"]
    ):
        raise ValueError("parent_weights_sha_preflight_failed")
    if (
        tuple(source["weight"] for source in plan["fixed_replay_sources"])
        != EXPECTED_REPLAY_WEIGHTS
    ):
        raise ValueError("replay_weight_pin_failed")
    audit_set = audit_hashes(plan)
    for source in plan["fixed_replay_sources"]:
        path = ROOT / source["path"]
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise FileNotFoundError("generation_n_plus_1_replay_provenance_unavailable")
        if jsonl_audit_collisions(path, audit_set):
            raise ValueError("audit_corpus_isolation_failed")
    config = rendered_config(plan, base, workdir)
    command = self_play_step(config)["command"]
    if option_value(command, "--simulations") != "1200" or any(
        flag in command
        for flag in ("--opening-min-simulations", "--opening-min-simulations-plies")
    ):
        raise ValueError("uniform1200_no_opening_min_contract_failed")
    for name, expected in (
        ("prefilter_suite", "prefilter_suite_sha256"),
        ("hard_suite", "hard_suite_sha256"),
    ):
        path = ROOT / plan["canonical_gate"][name]
        if not path.is_file() or sha256_file(path) != plan["canonical_gate"][expected]:
            raise ValueError("frozen_suite_sha_pin_failed")
    return {
        "parent_weights_sha256": sha256_file(parent / "weights.json"),
        "audit_hash_count": len(audit_set),
    }


def completed_selfplay_contract(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    step = next(item for item in manifest["steps"] if item.get("name") == "self_play")
    command = step["command"]
    return {
        "workers": option_value(command, "--workers"),
        "games": option_value(command, "--games"),
        "simulations": option_value(command, "--simulations"),
        "opening_minimum_present": any(
            flag in command
            for flag in ("--opening-min-simulations", "--opening-min-simulations-plies")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "ml/alphazero_lite/configs/seed48_nextgen_s443.json",
    )
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / ".tmp/seed48-nextgen-s443-uniform1200"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/data/alphazero-lite-seed48-nextgen-s443/results.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan["base_config"]).read_text(encoding="utf-8"))
    if args.finalize:
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        run_dir = Path(existing["candidate_path"])
        contract = completed_selfplay_contract(run_dir)
        manifest = existing["run_manifest"]
        durations = {step["name"]: step["duration_s"] for step in manifest["steps"]}
        losses = {
            key: float(value)
            for key, value in re.findall(
                r"^(policy_loss|value_loss|total_loss|best_val_loss)=([0-9.]+)$",
                (run_dir / "train.log").read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
        }
        existing["registered_inputs"] = {
            "parent_artifact": plan["parent_artifact"],
            "parent_version": plan["expected_parent_version"],
            "training_seed": plan["training_seed"],
            "self_play_seed_sweep": plan["self_play_seed_sweep"],
            "replay_sources": plan["fixed_replay_sources"],
        }
        existing["training_losses"] = losses
        existing["durations_s"] = {
            "self_play": durations["self_play"],
            "training": durations["train"],
        }
        existing["completed_selfplay_contract"] = contract
        if contract != {
            "workers": "6",
            "games": "1600",
            "simulations": "1200",
            "opening_minimum_present": False,
        }:
            existing["classification"] = "self_play_worker_contract_failed"
            existing["gate_status"] = "not_run_invalid_training_contract"
            existing["unrun_evaluations"] = [
                "artifact_sanity",
                "superhuman_regressions",
                "pr340_exact_corpus_diagnostic",
                "local_promotion_gate",
            ]
        write_json(args.out, existing)
        return 0
    try:
        integrity = preflight(plan, base, args.workdir)
    except (FileNotFoundError, ValueError) as error:
        write_json(
            args.out,
            {
                "schema": SCHEMA,
                "classification": str(error),
                "promotion": plan["promotion"],
            },
        )
        return 0
    config = rendered_config(plan, base, args.workdir)
    config_path = args.workdir / "config.json"
    write_json(config_path, config)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "classification": "execution_pending",
        "plan_sha256": sha256_file(args.plan),
        "config_sha256": sha256_file(config_path),
        "config_path": str(config_path),
        "preflight": integrity,
        "promotion": plan["promotion"],
    }
    if not args.execute:
        write_json(args.out, result)
        return 0
    subprocess.run(
        [sys.executable, "ml/alphazero_lite/pipeline.py", "--config", str(config_path)],
        cwd=ROOT,
        check=True,
    )
    run_dir = args.workdir / "runs" / f"{plan['run_id']}-iter1"
    selfplay = run_dir / "self_play.jsonl"
    audit_set = audit_hashes(plan)
    result.update(
        {
            "status": "candidate_generated",
            "classification": "candidate_generated",
            "candidate_path": str(run_dir),
            "self_play_sha256": sha256_file(selfplay),
            "search_work": search_work(selfplay, audit_set),
            "checkpoint_sha256": sha256_file(run_dir / "checkpoint.npz"),
            "weights_json_sha256": sha256_file(run_dir / "weights.json"),
            "metadata_json_sha256": sha256_file(run_dir / "metadata.json"),
            "perspective_audit": json.loads(
                (run_dir / "perspective_audit.json").read_text(encoding="utf-8")
            ),
            "run_manifest": json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            ),
        }
    )
    if result["weights_json_sha256"] == integrity["parent_weights_sha256"]:
        result["classification"] = "candidate_matches_current"
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
