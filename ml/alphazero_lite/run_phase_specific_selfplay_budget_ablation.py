#!/usr/bin/env python3
"""Run the non-promoting phase-specific self-play search-budget ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

SCHEMA = "azlite_phase_specific_selfplay_budget_ablation_v1"
EXPECTED_PARENT_WEIGHTS_SHA256 = (
    "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
)
EXPECTED_SELECTED_ROWS = 20
EXPECTED_CONTROL_ROWS = 5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "ml/alphazero_lite/configs/phase_specific_selfplay_budget_ablation.json"
        ),
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def option_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except ValueError:
        return None


def set_option(command: list[str], flag: str, value: int | None) -> list[str]:
    rendered = list(command)
    try:
        index = rendered.index(flag)
    except ValueError:
        if value is not None:
            rendered.extend([flag, str(value)])
        return rendered
    if value is None:
        return rendered[:index] + rendered[index + 2 :]
    rendered[index + 1] = str(value)
    return rendered


def self_play_step(config: dict[str, Any]) -> dict[str, Any]:
    matches = [step for step in config["steps"] if step.get("name") == "self_play"]
    if len(matches) != 1:
        raise ValueError("base config must have exactly one self_play step")
    return matches[0]


def validate_base_config(config: dict[str, Any]) -> None:
    command = self_play_step(config)["command"]
    expected = {
        "--games": "1600",
        "--workers": "6",
        "--seed": "42",
        "--seed-sweep": "41,42,43",
        "--temperature-threshold": "12",
        "--temperature": "1.1",
        "--temperature-late": "0.15",
        "--dirichlet-alpha": "0.3",
        "--dirichlet-epsilon": "0.3",
        "--input-encoding": "kalah_v3",
        "--policy-target-mode": "sharpened",
        "--policy-target-noise-mode": "denoised",
        "--value-target-mode": "sharpened",
    }
    mismatches = [
        f"{flag}={option_value(command, flag)!r}"
        for flag, value in expected.items()
        if option_value(command, flag) != value
    ]
    if (
        "--tree-reuse-enabled" not in command
        or "--write-root-target-telemetry" not in command
    ):
        mismatches.append("missing tree reuse or root telemetry")
    if mismatches:
        raise ValueError("base self-play controls changed: " + ", ".join(mismatches))


def lane_configs(
    plan: dict[str, Any], base: dict[str, Any], workdir: Path
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, lane in plan["lanes"].items():
        config = json.loads(json.dumps(base))
        config["run_id"] = f"phase-specific-budget-{name}"
        config["versions_dir"] = str(workdir / "runs" / name)
        config["fixed_replay_sources"] = [
            {"path": plan["selected_replay"], "weight": 1},
            {"path": plan["controls_replay"], "weight": 2},
        ]
        step = self_play_step(config)
        command = set_option(step["command"], "--simulations", lane["simulations"])
        command = set_option(
            command, "--opening-min-simulations", lane["opening_min_simulations"]
        )
        step["command"] = set_option(
            command,
            "--opening-min-simulations-plies",
            lane["opening_min_simulations_plies"],
        )
        result[name] = config
    return result


def preflight(plan: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unexpected ablation plan schema")
    if plan.get("promotion", {}).get("performed") is not False:
        raise ValueError("ablation plan must explicitly disable promotion")
    validate_base_config(base)
    selected = Path(plan["selected_replay"])
    controls = Path(plan["controls_replay"])
    selected_summary = json.loads(Path(plan["selected_replay_summary"]).read_text())
    controls_summary = json.loads(Path(plan["controls_replay_summary"]).read_text())
    if selected_summary.get("row_count") != EXPECTED_SELECTED_ROWS:
        raise ValueError("regenerated selected replay row count changed")
    if controls_summary.get("row_count") != EXPECTED_CONTROL_ROWS:
        raise ValueError("regenerated control replay row count changed")
    if selected_summary.get("path") != str(selected):
        raise ValueError("selected replay summary path does not match plan")
    if controls_summary.get("path") != str(controls):
        raise ValueError("control replay summary path does not match plan")
    parent_weights = Path(plan["current_path"]) / "weights.json"
    parent_sha = sha256_file(parent_weights)
    if parent_sha != EXPECTED_PARENT_WEIGHTS_SHA256:
        raise ValueError("current parent weights do not match the PR #285 incumbent")
    return {
        "parent_weights_sha256": parent_sha,
        "selected_replay": {
            "path": str(selected),
            "rows": EXPECTED_SELECTED_ROWS,
            "sha256": sha256_file(selected),
        },
        "controls_replay": {
            "path": str(controls),
            "rows": EXPECTED_CONTROL_ROWS,
            "sha256": sha256_file(controls),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-Specific Self-Play Search-Budget Ablation",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "This is a regenerated replay baseline, not the lost 38-row historical artifact lineage. No artifact was promoted.",
        "",
        "| lane | normal simulations | opening simulations | opening plies | pipeline status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, lane in summary["lanes"].items():
        lines.append(
            f"| {name} | {lane['simulations']} | {lane['opening_min_simulations'] or '-'} | {lane['opening_min_simulations_plies'] or '-'} | {lane['status']} |"
        )
    lines.extend(
        [
            "",
            f"Selected replay: `{summary['preflight']['selected_replay']['sha256']}` ({summary['preflight']['selected_replay']['rows']} rows).",
            f"Guard controls: `{summary['preflight']['controls_replay']['sha256']}` ({summary['preflight']['controls_replay']['rows']} rows).",
            f"Parent weights: `{summary['preflight']['parent_weights_sha256']}`.",
            "",
            "The runner invokes only `pipeline.py`; it never invokes an arena promotion gate or a promotion command.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    base_path = Path(plan["base_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    integrity = preflight(plan, base)
    configs = lane_configs(plan, base, args.workdir)
    lane_summary: dict[str, Any] = {}
    for name, config in configs.items():
        config_path = args.workdir / "configs" / f"{name}.json"
        write_json(config_path, config)
        command = [
            sys.executable,
            "ml/alphazero_lite/pipeline.py",
            "--config",
            str(config_path),
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        lane = plan["lanes"][name]
        lane_summary[name] = {
            **lane,
            "config_path": str(config_path),
            "status": "dry_run" if args.dry_run else "completed",
        }
    summary = {
        "schema": SCHEMA,
        "classification": "phase_specific_budget_ablation_preflight_only"
        if args.dry_run
        else "phase_specific_budget_ablation_completed_unclassified",
        "promotion": plan["promotion"],
        "regenerated_baseline": True,
        "preflight": integrity,
        "lanes": lane_summary,
    }
    write_json(args.out_summary, summary)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "summary": str(args.out_summary),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
