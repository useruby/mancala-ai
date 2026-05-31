#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    benchmark_pass,
    load_json,
    mcts_score,
    repo_root,
    run_command,
    run_local_gate,
    write_json,
)
from ml.alphazero_lite.run_search_quality_option_ablation import inject_search_flags
from ml.alphazero_lite.run_search_quality_option_ablation import (
    resolve_step_command as resolve_search_step_command,
)


DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_SEARCH_CONTROL_CONFIG = (
    "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_003_search_prior_control"
DEFAULT_CANDIDATE_PATHS = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w1/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w1-iter1,"
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
SEARCH_CONTROL_OVERRIDES = {
    "fpu_mode": "parent_q",
    "reuse_subtree": True,
    "normalize_values": True,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-paths", default=DEFAULT_CANDIDATE_PATHS)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument(
        "--search-control-config", default=DEFAULT_SEARCH_CONTROL_CONFIG
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_candidate_paths(root: Path, value: str) -> list[Path]:
    paths = [
        resolve_path(root, item.strip()) for item in value.split(",") if item.strip()
    ]
    if not paths:
        raise SystemExit("--candidate-paths must resolve to at least one candidate")
    return paths


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def steps_by_name(config: dict) -> dict[str, dict]:
    return {
        str(step["name"]): step
        for step in config.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }


def replace_placeholder_tokens(
    command: list[str], *, candidate_path: str, current_path: str
) -> list[str]:
    rendered = []
    for token in command:
        value = str(token)
        value = value.replace("{iter_dir}", candidate_path)
        value = value.replace("{current_path}", current_path)
        rendered.append(value)
    return rendered


def control_commands(
    *,
    search_control_config: dict,
    candidate_path: Path,
    current_path: Path,
    out_dir: Path,
) -> dict[str, list[str]]:
    resolved_repo_root = repo_root()
    steps = steps_by_name(search_control_config)
    base_arena = resolve_search_step_command(
        replace_placeholder_tokens(
            list(steps["arena_confirm_report"]["command"]),
            candidate_path=str(candidate_path),
            current_path=str(current_path),
        ),
        resolved_repo_root=resolved_repo_root,
    )
    base_candidate_mcts = resolve_search_step_command(
        replace_placeholder_tokens(
            list(steps["mcts1200_baseline_report"]["command"]),
            candidate_path=str(candidate_path),
            current_path=str(current_path),
        ),
        resolved_repo_root=resolved_repo_root,
    )
    base_current_mcts = resolve_search_step_command(
        replace_placeholder_tokens(
            list(steps["current_mcts1200_baseline_report"]["command"]),
            candidate_path=str(candidate_path),
            current_path=str(current_path),
        ),
        resolved_repo_root=resolved_repo_root,
    )

    arena_command = inject_search_flags(
        [
            *base_arena,
            "--challenger",
            str(candidate_path),
            "--current",
            str(current_path),
            "--games",
            "120",
            "--challenger-simulations",
            "640",
            "--current-simulations",
            "256",
            "--workers",
            "6",
            "--out",
            str(out_dir / "arena_report.json"),
            "--min-score",
            "0.55",
        ],
        SEARCH_CONTROL_OVERRIDES,
    )
    candidate_mcts_command = inject_search_flags(
        [
            *base_candidate_mcts,
            "--challenger-path",
            str(candidate_path),
            "--games",
            "30",
            "--seed",
            "42",
            "--az-base-simulations",
            "640",
            "--mcts-simulations",
            "1200",
            "--workers",
            "6",
            "--out",
            str(out_dir / "mcts1200_report.json"),
        ],
        SEARCH_CONTROL_OVERRIDES,
    )
    current_mcts_command = inject_search_flags(
        [
            *base_current_mcts,
            "--challenger-path",
            str(current_path),
            "--games",
            "30",
            "--seed",
            "42",
            "--az-base-simulations",
            "640",
            "--mcts-simulations",
            "1200",
            "--workers",
            "6",
            "--out",
            str(out_dir / "current_mcts1200_report.json"),
        ],
        SEARCH_CONTROL_OVERRIDES,
    )
    benchmark_command = [
        python_bin(repo_root()),
        str(repo_root() / "ml/alphazero_lite/benchmark.py"),
        "--mode",
        "promotion",
        "--games",
        "60",
        "--seed",
        "42",
        "--challenger-path",
        str(candidate_path),
        "--current-path",
        str(current_path),
        "--arena-report",
        str(out_dir / "arena_report.json"),
        "--min-score",
        "0.55",
        "--mcts-report",
        str(out_dir / "mcts1200_report.json"),
        "--current-baseline-mcts-report",
        str(out_dir / "current_mcts1200_report.json"),
        "--out",
        str(out_dir / "benchmark_report.json"),
        "--fpu-mode",
        str(SEARCH_CONTROL_OVERRIDES["fpu_mode"]),
        "--root-policy-mode",
        str(SEARCH_CONTROL_OVERRIDES["root_policy_mode"]),
        "--tactical-root-bias",
        str(SEARCH_CONTROL_OVERRIDES["tactical_root_bias"]),
        "--reuse-subtree",
        "--normalize-values",
    ]
    return {
        "arena": arena_command,
        "candidate_mcts": candidate_mcts_command,
        "current_mcts": current_mcts_command,
        "benchmark": benchmark_command,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    run_root = resolve_path(root, args.output_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    current_path = resolve_path(root, args.current_path)
    reference_artifact = resolve_path(root, args.reference_artifact)
    search_control_config_path = resolve_path(root, args.search_control_config)
    search_control_config = load_json(search_control_config_path)
    candidate_paths = parse_candidate_paths(root, args.candidate_paths)

    summary = {
        "schema": "azlite_capture_002_003_search_prior_control_v1",
        "run_id": args.run_id,
        "current_path": str(current_path),
        "reference_artifact": str(reference_artifact),
        "search_control_config": str(search_control_config_path),
        "search_options": dict(SEARCH_CONTROL_OVERRIDES),
        "variants": [],
        "dry_run": bool(args.dry_run),
    }

    for candidate_path in candidate_paths:
        variant_name = candidate_path.name
        variant_out_dir = run_root / variant_name
        variant_out_dir.mkdir(parents=True, exist_ok=True)

        gate_path = variant_out_dir / "capture_002_003_search_prior_control_gate.json"
        gate_payload = run_local_gate(
            current_path=current_path,
            candidate_path=candidate_path,
            reference_artifact_path=reference_artifact,
            out_path=gate_path,
            search_options_overrides=SEARCH_CONTROL_OVERRIDES,
            dry_run=args.dry_run,
        )

        variant_payload = {
            "variant": variant_name,
            "candidate_path": str(candidate_path),
            "gate_path": str(gate_path),
            "gate": gate_payload,
            "decision": gate_payload.get("decision"),
        }

        if gate_payload.get("decision") == "reject_local_gate":
            summary["variants"].append(variant_payload)
            continue

        commands = control_commands(
            search_control_config=search_control_config,
            candidate_path=candidate_path,
            current_path=current_path,
            out_dir=variant_out_dir,
        )
        for name, command in commands.items():
            run_command(
                command,
                cwd=root,
                dry_run=args.dry_run,
                log_path=variant_out_dir / f"{name}_command.json",
            )

        if not args.dry_run:
            arena_report = load_json(variant_out_dir / "arena_report.json")
            mcts_report = load_json(variant_out_dir / "mcts1200_report.json")
            benchmark_report = load_json(variant_out_dir / "benchmark_report.json")
            variant_payload["arena_report"] = {
                "score": arena_report.get("score"),
                "confidence_interval_95": arena_report.get("confidence_interval_95"),
                "unstable_decision": arena_report.get("unstable_decision"),
            }
            variant_payload["mcts1200_report"] = {
                "score": mcts_score(mcts_report),
            }
            variant_payload["benchmark_report"] = {
                "passed": benchmark_pass(benchmark_report),
            }
            variant_payload["decision"] = "evaluated"

        summary["variants"].append(variant_payload)

    summary_path = run_root / "search_prior_control_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps({"summary_path": str(summary_path), "dry_run": bool(args.dry_run)})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
