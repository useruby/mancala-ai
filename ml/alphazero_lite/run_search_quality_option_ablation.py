#!/usr/bin/env python3
"""Run focused search-quality option ablations against an existing artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_CONFIG_PATH = (
    "ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json"
)
ARENA_STEP_NAME = "arena_confirm_report"
CANDIDATE_MCTS_STEP_NAME = "mcts1200_baseline_report"
CURRENT_MCTS_STEP_NAME = "current_mcts1200_baseline_report"
SEARCH_FLAGS_WITH_VALUES = (
    "--fpu-mode",
    "--root-policy-mode",
    "--tactical-root-bias",
)
SEARCH_BOOLEAN_FLAGS = (
    "--reuse-subtree",
    "--normalize-values",
)
VARIANT_SPECS = (
    {
        "name": "parent_q_only",
        "label": "parent_q only",
        "flags_enabled": [
            "--fpu-mode parent_q",
            "--root-policy-mode deterministic",
            "--tactical-root-bias 0.1",
        ],
        "search_options": {
            "fpu_mode": "parent_q",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        },
    },
    {
        "name": "normalize_values_only",
        "label": "normalize-values only",
        "flags_enabled": [
            "--normalize-values",
            "--root-policy-mode deterministic",
            "--tactical-root-bias 0.1",
        ],
        "search_options": {
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": True,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        },
    },
    {
        "name": "reuse_subtree_only",
        "label": "reuse-subtree only",
        "flags_enabled": [
            "--reuse-subtree",
            "--root-policy-mode deterministic",
            "--tactical-root-bias 0.1",
        ],
        "search_options": {
            "fpu_mode": "zero",
            "reuse_subtree": True,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        },
    },
    {
        "name": "tactical_root_bias_only",
        "label": "tactical-root-bias only",
        "flags_enabled": [
            "--root-policy-mode deterministic",
            "--tactical-root-bias 0.1",
        ],
        "search_options": {
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        },
    },
    {
        "name": "all_search_quality_flags",
        "label": "all search-quality flags",
        "flags_enabled": [
            "--fpu-mode parent_q",
            "--reuse-subtree",
            "--normalize-values",
            "--root-policy-mode deterministic",
            "--tactical-root-bias 0.1",
        ],
        "search_options": {
            "fpu_mode": "parent_q",
            "reuse_subtree": True,
            "normalize_values": True,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.1,
        },
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--current-path", default=None)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-format", choices=("json", "markdown"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def steps_by_name(config: dict) -> dict[str, dict]:
    return {
        str(step["name"]): step
        for step in config.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }


def command_with_replacements(
    command: list[str], replacements: dict[str, str]
) -> list[str]:
    rendered: list[str] = []
    for token in command:
        rendered_token = token
        for placeholder, value in replacements.items():
            rendered_token = rendered_token.replace(placeholder, value)
        rendered.append(rendered_token)
    return rendered


def strip_search_flags(command: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in SEARCH_FLAGS_WITH_VALUES:
            index += 2
            continue
        if token in SEARCH_BOOLEAN_FLAGS:
            index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def search_flags_as_cli(search_options: dict[str, str | bool | float]) -> list[str]:
    flags = [
        "--fpu-mode",
        str(search_options["fpu_mode"]),
        "--root-policy-mode",
        str(search_options["root_policy_mode"]),
        "--tactical-root-bias",
        str(search_options["tactical_root_bias"]),
    ]
    if bool(search_options["reuse_subtree"]):
        flags.append("--reuse-subtree")
    if bool(search_options["normalize_values"]):
        flags.append("--normalize-values")
    return flags


def inject_search_flags(
    command: list[str], search_options: dict[str, str | bool | float]
) -> list[str]:
    stripped = strip_search_flags(command)
    search_flags = search_flags_as_cli(search_options)
    try:
        out_index = stripped.index("--out")
    except ValueError:
        return [*stripped, *search_flags]
    return [*stripped[:out_index], *search_flags, *stripped[out_index:]]


def command_flag_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"command missing value for {flag}")
    return command[index + 1]


def command_has_flag(command: list[str], flag: str) -> bool:
    return flag in command


def replace_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag not in updated:
        return updated
    index = updated.index(flag)
    if index + 1 >= len(updated):
        raise ValueError(f"command missing value for {flag}")
    updated[index + 1] = value
    return updated


def derive_candidate_path(config: dict) -> str:
    iteration = int(config.get("start_iteration", 1))
    versions_dir = Path(str(config["versions_dir"]))
    return str(versions_dir / f"{config['run_id']}-iter{iteration}")


def infer_summary_format(args: argparse.Namespace) -> str:
    if args.summary_format is not None:
        return args.summary_format
    return "markdown" if Path(args.out).suffix.lower() == ".md" else "json"


def base_step_commands(
    config: dict, *, candidate_path: str, current_path: str
) -> dict[str, list[str]]:
    steps = steps_by_name(config)
    replacements = {
        "{iter_dir}": candidate_path,
        "{current_path}": current_path,
        "{parent_checkpoint}": str(Path(candidate_path) / "checkpoint.npz"),
        "{iteration}": "1",
        "{run_id}": str(config["run_id"]),
        "{replay_data}": "",
        "{replay_weights}": "",
    }
    return {
        ARENA_STEP_NAME: command_with_replacements(
            list(steps[ARENA_STEP_NAME]["command"]), replacements
        ),
        CANDIDATE_MCTS_STEP_NAME: command_with_replacements(
            list(steps[CANDIDATE_MCTS_STEP_NAME]["command"]), replacements
        ),
        CURRENT_MCTS_STEP_NAME: command_with_replacements(
            list(steps[CURRENT_MCTS_STEP_NAME]["command"]), replacements
        ),
    }


def build_variant_plan(
    config: dict,
    *,
    candidate_path: str,
    current_path: str,
    report_dir: Path,
) -> dict[str, dict]:
    base_commands = base_step_commands(
        config, candidate_path=candidate_path, current_path=current_path
    )
    plans: dict[str, dict] = {}
    report_dir.mkdir(parents=True, exist_ok=True)
    for spec in VARIANT_SPECS:
        variant_dir = report_dir / spec["name"]
        arena_command = inject_search_flags(
            list(base_commands[ARENA_STEP_NAME]), spec["search_options"]
        )
        arena_command = replace_flag_value(
            arena_command, "--challenger", candidate_path
        )
        arena_command = replace_flag_value(arena_command, "--current", current_path)
        candidate_mcts_command = inject_search_flags(
            list(base_commands[CANDIDATE_MCTS_STEP_NAME]), spec["search_options"]
        )
        candidate_mcts_command = replace_flag_value(
            candidate_mcts_command, "--challenger-path", candidate_path
        )
        current_mcts_command = inject_search_flags(
            list(base_commands[CURRENT_MCTS_STEP_NAME]), spec["search_options"]
        )
        current_mcts_command = replace_flag_value(
            current_mcts_command, "--challenger-path", current_path
        )
        arena_out = variant_dir / "arena_report.json"
        candidate_mcts_out = variant_dir / "mcts1200_report.json"
        current_mcts_out = variant_dir / "current_mcts1200_report.json"
        arena_command[arena_command.index("--out") + 1] = str(arena_out)
        candidate_mcts_command[candidate_mcts_command.index("--out") + 1] = str(
            candidate_mcts_out
        )
        current_mcts_command[current_mcts_command.index("--out") + 1] = str(
            current_mcts_out
        )
        plans[spec["name"]] = {
            "name": spec["name"],
            "label": spec["label"],
            "flags_enabled": list(spec["flags_enabled"]),
            "search_options": dict(spec["search_options"]),
            "candidate_path": candidate_path,
            "current_path": current_path,
            "arena_command": arena_command,
            "candidate_mcts_command": candidate_mcts_command,
            "current_mcts_command": current_mcts_command,
            "arena_report_path": str(arena_out),
            "candidate_mcts_report_path": str(candidate_mcts_out),
            "current_mcts_report_path": str(current_mcts_out),
        }
    return plans


def arena_passed(report: dict) -> bool:
    decision = report.get("promotion_decision", {})
    return bool(decision.get("passed", False))


def summarize_variant_template(plan: dict) -> dict:
    return {
        "variant": plan["name"],
        "label": plan["label"],
        "flags_enabled": plan["flags_enabled"],
        "arena_score": None,
        "mcts1200_score": None,
        "runtime_seconds": None,
        "pass": None,
        "notes": "",
    }


def execute_variant_plan(plan: dict) -> dict:
    variant_dir = Path(plan["arena_report_path"]).parent
    variant_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for command in (
        plan["arena_command"],
        plan["candidate_mcts_command"],
        plan["current_mcts_command"],
    ):
        subprocess.run(command, check=True)
    runtime_seconds = round(time.perf_counter() - started, 2)
    arena_report = load_json(plan["arena_report_path"])
    candidate_mcts_report = load_json(plan["candidate_mcts_report_path"])
    current_mcts_report = load_json(plan["current_mcts_report_path"])
    candidate_mcts_score = float(candidate_mcts_report["score"])
    current_mcts_score = float(current_mcts_report["score"])
    passed = arena_passed(arena_report) and candidate_mcts_score >= current_mcts_score
    notes: list[str] = []
    notes.append(f"candidate={plan['candidate_path']}")
    notes.append(f"current={plan['current_path']}")
    notes.append(f"current_mcts1200_score={current_mcts_score:.4f}")
    if candidate_mcts_score < current_mcts_score:
        notes.append("candidate MCTS1200 trailed current")
    return {
        "variant": plan["name"],
        "label": plan["label"],
        "flags_enabled": plan["flags_enabled"],
        "arena_score": float(arena_report["score"]),
        "mcts1200_score": candidate_mcts_score,
        "runtime_seconds": runtime_seconds,
        "pass": passed,
        "notes": "; ".join(notes),
    }


def render_markdown_summary(entries: list[dict], *, config_path: str) -> str:
    lines = [
        "# Search-Quality Option Ablation Matrix",
        "",
        f"Config: `{config_path}`",
        "",
        "| Variant | Flags enabled | Arena score | MCTS1200 score | Runtime | Pass/fail | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        arena_score = (
            "TBD" if entry["arena_score"] is None else f"{entry['arena_score']:.4f}"
        )
        mcts_score = (
            "TBD"
            if entry["mcts1200_score"] is None
            else f"{entry['mcts1200_score']:.4f}"
        )
        runtime_value = (
            "TBD"
            if entry["runtime_seconds"] is None
            else f"{float(entry['runtime_seconds']):.2f}s"
        )
        pass_value = (
            "TBD" if entry["pass"] is None else ("pass" if entry["pass"] else "fail")
        )
        notes = entry["notes"] or ""
        lines.append(
            "| {label} | {flags} | {arena} | {mcts} | {runtime} | {passed} | {notes} |".format(
                label=entry["label"],
                flags="<br>".join(entry["flags_enabled"]),
                arena=arena_score,
                mcts=mcts_score,
                runtime=runtime_value,
                passed=pass_value,
                notes=notes,
            )
        )
    return "\n".join(lines) + "\n"


def write_summary(
    entries: list[dict], *, out_path: Path, summary_format: str, config_path: str
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_format == "markdown":
        out_path.write_text(
            render_markdown_summary(entries, config_path=config_path), encoding="utf-8"
        )
        return
    payload = {
        "schema": "search_quality_option_ablation_v1",
        "config_path": config_path,
        "variants": entries,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config_path)
    candidate_path = args.candidate_path or derive_candidate_path(config)
    current_path = args.current_path or str(config["current_path"])
    report_dir = Path(args.report_dir)
    plans = build_variant_plan(
        config,
        candidate_path=candidate_path,
        current_path=current_path,
        report_dir=report_dir,
    )
    if args.dry_run:
        entries = [
            summarize_variant_template(plans[spec["name"]]) for spec in VARIANT_SPECS
        ]
    else:
        entries = [execute_variant_plan(plans[spec["name"]]) for spec in VARIANT_SPECS]
    write_summary(
        entries,
        out_path=Path(args.out),
        summary_format=infer_summary_format(args),
        config_path=args.config_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
