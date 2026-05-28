#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import hard_state_teacher_labeling


DEFAULT_BASE_CONFIG = (
    "ml/alphazero_lite/configs/aggressive_v3_targeted_hard_state_replay.json"
)
DEFAULT_VARIANT_WEIGHTS = (1, 2, 4)
REQUIRED_EVAL_STEPS = {
    "hard_state_validation",
    "arena_confirm_report",
    "mcts1200_baseline_report",
    "benchmark_contract",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mine-inputs", required=True)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--current-path", default=None)
    parser.add_argument("--hard-state-validation-path", default=None)
    parser.add_argument("--top-n", type=int, default=64)
    parser.add_argument("--canonical-budget", type=int, default=384)
    parser.add_argument("--stronger-budget", type=int, default=1200)
    parser.add_argument(
        "--variant-weights", default=",".join(str(v) for v in DEFAULT_VARIANT_WEIGHTS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_csv_paths(root: Path, value: str) -> list[Path]:
    return [
        resolve_path(root, item.strip()) for item in value.split(",") if item.strip()
    ]


def parse_variant_weights(value: str) -> list[int]:
    weights = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not weights:
        raise ValueError("variant weights must not be empty")
    if any(weight <= 0 for weight in weights):
        raise ValueError("variant weights must be positive integers")
    return weights


def steps_by_name(config: dict) -> dict[str, dict]:
    return {
        str(step["name"]): step
        for step in config.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    }


def ensure_base_config_contract(config: dict) -> None:
    steps = steps_by_name(config)
    missing = sorted(REQUIRED_EVAL_STEPS - set(steps))
    if missing:
        raise ValueError(
            f"base config is missing required evaluation steps: {', '.join(missing)}"
        )
    if "train" not in steps:
        raise ValueError("base config is missing train step")


def ensure_no_validation_leakage(
    *, mine_inputs: list[Path], hard_state_validation_path: Path | None
) -> None:
    if hard_state_validation_path is None:
        return

    validation_resolved = hard_state_validation_path.resolve()
    for mine_input in mine_inputs:
        mine_resolved = mine_input.resolve()
        if mine_resolved == validation_resolved:
            raise ValueError(
                "mine inputs must not include the hard-state validation holdout path"
            )
        if mine_resolved.is_dir() and validation_resolved.is_relative_to(mine_resolved):
            raise ValueError(
                "mine input directory must not contain the hard-state validation holdout path"
            )


def derived_paths(
    root: Path, run_id: str, output_root: str, seed: int
) -> dict[str, Path]:
    base_dir = resolve_path(root, output_root) / run_id
    inputs_dir = base_dir / "inputs"
    reports_dir = base_dir / "reports"
    variants_dir = base_dir / "variants"
    return {
        "base_dir": base_dir,
        "inputs_dir": inputs_dir,
        "reports_dir": reports_dir,
        "variants_dir": variants_dir,
        "mine_input_report": inputs_dir / f"mined_hard_states_train_seed{seed}.jsonl",
        "mine_summary_report": inputs_dir
        / f"mined_hard_states_train_seed{seed}_report.json",
        "dual_budget_labels": inputs_dir / f"hard_state_labels_seed{seed}.jsonl",
        "stronger_train": inputs_dir / f"hard_state_train_seed{seed}.jsonl",
        "label_comparison": reports_dir
        / f"hard_state_label_comparison_seed{seed}.json",
        "report_template": reports_dir / "hard_state_replay_experiment_report.json",
        "manifest": base_dir / "manifest.json",
    }


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict:
    if dry_run:
        return {"command": command, "cwd": str(cwd), "returncode": None}
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return {"command": command, "cwd": str(cwd), "returncode": completed.returncode}


def stronger_rows_from_dual_budget(path: Path) -> tuple[list[dict], dict[str, object]]:
    rows = hard_state_teacher_labeling.load_labeled_rows(path)
    comparison = hard_state_teacher_labeling.build_comparison_report(rows)
    stronger_rows = [row for row in rows if row["teacher_profile"] == "stronger"]
    if not stronger_rows:
        raise ValueError("teacher labeling did not produce any stronger-profile rows")
    return stronger_rows, comparison


def policy_entropy(policy: list[float]) -> float:
    entropy = 0.0
    for probability in policy:
        value = float(probability)
        if value > 0.0:
            entropy -= value * math.log(value, 2)
    return entropy


def dataset_summary(
    *, mined_rows_path: Path, stronger_rows: list[dict], comparison: dict[str, object]
) -> dict[str, object]:
    source_counter: Counter[str] = Counter()
    selection_reason_counter: Counter[str] = Counter()
    for line in mined_rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for artifact in row.get("source_artifacts", []):
            if isinstance(artifact, str) and artifact:
                source_counter[Path(artifact).name] += 1
        for reason in row.get("selection_reasons", []):
            if isinstance(reason, str) and reason:
                selection_reason_counter[reason] += 1

    entropies = [policy_entropy(row["policy"]) for row in stronger_rows]
    top_moves = Counter(
        max(range(len(row["policy"])), key=lambda move: float(row["policy"][move]))
        for row in stronger_rows
    )
    return {
        "hard_state_count": len(stronger_rows),
        "source_distribution": dict(source_counter),
        "selection_reason_distribution": dict(selection_reason_counter),
        "label_entropy": {
            "mean": round(sum(entropies) / len(entropies), 6) if entropies else 0.0,
            "min": round(min(entropies), 6) if entropies else 0.0,
            "max": round(max(entropies), 6) if entropies else 0.0,
        },
        "top_teacher_move_distribution": dict(sorted(top_moves.items())),
        "canonical_vs_stronger": comparison,
    }


def candidate_dir_for_config(config: dict) -> Path:
    start_iteration = int(config.get("start_iteration", 1))
    iterations = int(config.get("iterations", 1))
    final_iteration = start_iteration + iterations - 1
    return (
        Path(str(config["versions_dir"])) / f"{config['run_id']}-iter{final_iteration}"
    )


def build_runtime_config(
    *,
    base_config: dict,
    run_root: Path,
    stronger_train_path: Path,
    weight: int,
    current_path: str | None,
    hard_state_validation_path: str | None,
) -> tuple[dict, Path]:
    variant_root = run_root / f"w{weight}"
    config = json.loads(json.dumps(base_config))
    config["run_id"] = f"{config['run_id']}-hard-state-replay-w{weight}"
    config["versions_dir"] = str(variant_root / "versions")
    if current_path is not None:
        config["current_path"] = current_path
        config["parent_artifact_path"] = current_path
    if hard_state_validation_path is not None:
        config["hard_state_validation_path"] = hard_state_validation_path
    fixed_replay_sources = list(config.get("fixed_replay_sources", []))
    fixed_replay_sources.append({"path": str(stronger_train_path), "weight": weight})
    config["fixed_replay_sources"] = fixed_replay_sources

    runtime_config_path = variant_root / "runtime_config.json"
    write_json(runtime_config_path, config)
    return config, runtime_config_path


def variant_report_stub(
    *, weight: int, runtime_config_path: Path, config: dict
) -> dict[str, object]:
    candidate_dir = candidate_dir_for_config(config)
    return {
        "hard_state_weight": weight,
        "runtime_config_path": str(runtime_config_path),
        "candidate_dir": str(candidate_dir),
        "metrics_to_fill": {
            "hard_state_pass_rate": None,
            "arena_score": None,
            "mcts1200_score": None,
            "promotion_benchmark": None,
        },
        "artifact_paths": {
            "hard_state_validation": str(candidate_dir / "hard_state_validation.json"),
            "arena_report": str(candidate_dir / "arena_report.json"),
            "mcts1200_report": str(candidate_dir / "mcts1200_report.json"),
            "benchmark_report": str(candidate_dir / "benchmark_report.json"),
        },
    }


def build_report_template(
    *,
    args: argparse.Namespace,
    base_config_path: Path,
    hard_state_validation_path: Path | None,
    stronger_train_path: Path,
    mine_inputs: list[Path],
    shared_dataset_summary: dict[str, object] | None,
    variants: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "experiment": {
            "run_id": args.run_id,
            "base_config": str(base_config_path),
            "mine_inputs": [str(path) for path in mine_inputs],
            "hard_state_validation_path": None
            if hard_state_validation_path is None
            else str(hard_state_validation_path),
            "hard_state_train_replay_path": str(stronger_train_path),
            "top_n": int(args.top_n),
            "canonical_budget": int(args.canonical_budget),
            "stronger_budget": int(args.stronger_budget),
            "seed": int(args.seed),
        },
        "guardrails": {
            "holdout_leakage_check": "mine inputs must not equal or contain the hard-state validation holdout path",
            "train_holdout_separation": "train artifacts are written with train/seed-specific filenames and must stay separate from holdout validation inputs",
            "promotion_rule": "do not promote a model that only improves the mined hard-state set; require arena, MCTS1200, and benchmark support",
        },
        "shared_dataset": shared_dataset_summary,
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    base_config_path = resolve_path(root, args.base_config)
    base_config = load_json(base_config_path)
    ensure_base_config_contract(base_config)

    mine_inputs = parse_csv_paths(root, args.mine_inputs)
    if not mine_inputs:
        raise SystemExit("--mine-inputs must resolve to at least one path")
    variant_weights = parse_variant_weights(args.variant_weights)

    config_validation_path = base_config.get("hard_state_validation_path")
    hard_state_validation_path = None
    if args.hard_state_validation_path is not None:
        hard_state_validation_path = resolve_path(root, args.hard_state_validation_path)
    elif isinstance(config_validation_path, str) and config_validation_path:
        hard_state_validation_path = resolve_path(root, config_validation_path)

    ensure_no_validation_leakage(
        mine_inputs=mine_inputs,
        hard_state_validation_path=hard_state_validation_path,
    )

    paths = derived_paths(root, args.run_id, args.output_root, args.seed)
    paths["base_dir"].mkdir(parents=True, exist_ok=True)

    python = python_bin(root)
    commands = {
        "mine": [
            python,
            str(root / "ml/alphazero_lite/mine_hard_states.py"),
            "--inputs",
            ",".join(str(path) for path in mine_inputs),
            "--out-jsonl",
            str(paths["mine_input_report"]),
            "--out-report",
            str(paths["mine_summary_report"]),
        ],
        "label": [
            python,
            str(root / "ml/alphazero_lite/label_hard_state_subset.py"),
            "--mined-jsonl",
            str(paths["mine_input_report"]),
            "--out",
            str(paths["dual_budget_labels"]),
            "--top-n",
            str(args.top_n),
            "--canonical-budget",
            str(args.canonical_budget),
            "--stronger-budget",
            str(args.stronger_budget),
            "--seed",
            str(args.seed),
        ],
    }

    command_results = [
        run_command(commands["mine"], cwd=root, dry_run=args.dry_run),
        run_command(commands["label"], cwd=root, dry_run=args.dry_run),
    ]

    stronger_rows: list[dict] = []
    comparison: dict[str, object] | None = None
    shared_dataset_summary: dict[str, object] | None = None
    if not args.dry_run:
        stronger_rows, comparison = stronger_rows_from_dual_budget(
            paths["dual_budget_labels"]
        )
        write_jsonl(paths["stronger_train"], stronger_rows)
        write_json(paths["label_comparison"], comparison)
        shared_dataset_summary = dataset_summary(
            mined_rows_path=paths["mine_input_report"],
            stronger_rows=stronger_rows,
            comparison=comparison,
        )

    variants: list[dict[str, object]] = []
    for weight in variant_weights:
        config, runtime_config_path = build_runtime_config(
            base_config=base_config,
            run_root=paths["variants_dir"],
            stronger_train_path=paths["stronger_train"],
            weight=weight,
            current_path=args.current_path,
            hard_state_validation_path=None
            if hard_state_validation_path is None
            else str(hard_state_validation_path),
        )
        variants.append(
            variant_report_stub(
                weight=weight,
                runtime_config_path=runtime_config_path,
                config=config,
            )
        )
        if not args.dry_run:
            command_results.append(
                run_command(
                    [
                        python,
                        str(root / "ml/alphazero_lite/pipeline.py"),
                        "--config",
                        str(runtime_config_path),
                    ],
                    cwd=root,
                    dry_run=False,
                )
            )

    report_template = build_report_template(
        args=args,
        base_config_path=base_config_path,
        hard_state_validation_path=hard_state_validation_path,
        stronger_train_path=paths["stronger_train"],
        mine_inputs=mine_inputs,
        shared_dataset_summary=shared_dataset_summary,
        variants=variants,
    )
    write_json(paths["report_template"], report_template)

    manifest = {
        "run_id": args.run_id,
        "dry_run": bool(args.dry_run),
        "paths": {key: str(value) for key, value in paths.items()},
        "commands": command_results,
        "variant_weights": variant_weights,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
