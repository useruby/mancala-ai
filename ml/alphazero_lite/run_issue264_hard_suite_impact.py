#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import hard_state_teacher_labeling as labeling
from ml.alphazero_lite.worker_config import DEFAULT_WORKERS


def resolve_absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_python_executable(repo_root: Path) -> str:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return str(Path(sys.executable).resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mined-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--init-checkpoint", required=True, type=Path)
    parser.add_argument("--current-artifact", required=True, type=Path)
    parser.add_argument("--top-n", required=True, type=int)
    parser.add_argument("--canonical-budget", required=True, type=int)
    parser.add_argument("--stronger-budget", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--challenger-simulations", required=True, type=int)
    parser.add_argument("--current-simulations", required=True, type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--min-score", required=True, type=float)
    return parser.parse_args(argv)


def run_command(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or result.stdout.strip() or "command failed"
        )
    return result.stdout


def build_final_report_from_paths(
    *,
    experiment: dict[str, Any],
    label_report_path: Path,
    arena_report_path: Path,
    baseline_checkpoint: Path,
    challenger_checkpoint: Path,
    challenger_artifact_dir: Path,
    min_score: float,
) -> dict[str, Any]:
    label_report = json.loads(label_report_path.read_text(encoding="utf-8"))
    arena_report = json.loads(arena_report_path.read_text(encoding="utf-8"))
    return labeling.build_issue264_report(
        experiment=experiment,
        label_report=label_report,
        arena_report=arena_report,
        baseline_checkpoint=str(baseline_checkpoint),
        challenger_checkpoint=str(challenger_checkpoint),
        challenger_artifact_dir=str(challenger_artifact_dir),
        min_score=min_score,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2].resolve()
    python_executable = resolve_python_executable(repo_root)
    mined_jsonl = resolve_absolute_path(args.mined_jsonl)
    out_dir = resolve_absolute_path(args.out_dir)
    init_checkpoint = resolve_absolute_path(args.init_checkpoint)
    current_artifact = resolve_absolute_path(args.current_artifact)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = (out_dir / "labeled.jsonl").resolve()
    label_report_path = (out_dir / "label_report.json").resolve()
    stronger_dataset_path = (out_dir / "stronger_only.jsonl").resolve()
    challenger_checkpoint = (out_dir / "challenger.npz").resolve()
    challenger_artifact_dir = (out_dir / "challenger_artifact").resolve()
    arena_report_path = (out_dir / "arena_report.json").resolve()
    final_report_path = (out_dir / "issue264_report.json").resolve()

    train_script = str((repo_root / "ml/alphazero_lite/train.py").resolve())
    export_script = str((repo_root / "ml/alphazero_lite/export_artifact.py").resolve())
    arena_script = str((repo_root / "ml/alphazero_lite/arena.py").resolve())

    rows = labeling.load_hard_state_rows(mined_jsonl)
    selected_rows = labeling.select_top_ranked_rows(rows, top_n=args.top_n)
    labeled_rows = labeling.build_dual_budget_rows(
        selected_rows,
        canonical_budget=args.canonical_budget,
        stronger_budget=args.stronger_budget,
        teacher_mode="classic_mcts",
        input_encoding=args.input_encoding,
        seed=args.seed,
    )
    labeling.write_jsonl(labeled_path, labeled_rows)

    label_report = labeling.build_comparison_report(
        labeling.load_labeled_rows(labeled_path)
    )
    label_report_path.write_text(
        json.dumps(label_report, indent=2) + "\n", encoding="utf-8"
    )

    stronger_rows = [
        row for row in labeled_rows if row["teacher_profile"] == "stronger"
    ]
    labeling.write_jsonl(stronger_dataset_path, stronger_rows)

    run_command(
        [
            python_executable,
            train_script,
            "--data",
            str(stronger_dataset_path),
            "--out",
            str(challenger_checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--input-encoding",
            args.input_encoding,
            "--init-checkpoint",
            str(init_checkpoint),
        ],
        cwd=repo_root,
    )
    run_command(
        [
            python_executable,
            export_script,
            "--checkpoint",
            str(challenger_checkpoint),
            "--out-dir",
            str(challenger_artifact_dir),
            "--version",
            "issue-264-hard-suite-impact",
            "--input-encoding",
            args.input_encoding,
        ],
        cwd=repo_root,
    )
    run_command(
        [
            python_executable,
            arena_script,
            "--challenger",
            str(challenger_artifact_dir),
            "--current",
            str(current_artifact),
            "--games",
            str(args.games),
            "--challenger-simulations",
            str(args.challenger_simulations),
            "--current-simulations",
            str(args.current_simulations),
            "--seed",
            str(args.seed),
            "--out",
            str(arena_report_path),
            "--min-score",
            str(args.min_score),
            "--workers",
            str(args.workers),
        ],
        cwd=repo_root,
    )

    final_report = build_final_report_from_paths(
        experiment={
            "mined_jsonl": str(mined_jsonl),
            "top_n": args.top_n,
            "canonical_budget": args.canonical_budget,
            "stronger_budget": args.stronger_budget,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "games": args.games,
            "challenger_simulations": args.challenger_simulations,
            "current_simulations": args.current_simulations,
            "workers": args.workers,
        },
        label_report_path=label_report_path,
        arena_report_path=arena_report_path,
        baseline_checkpoint=init_checkpoint,
        challenger_checkpoint=challenger_checkpoint,
        challenger_artifact_dir=challenger_artifact_dir,
        min_score=args.min_score,
    )
    final_report_path.write_text(
        json.dumps(final_report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "out_report": str(final_report_path),
                "recommendation": final_report["recommendation"]["recommendation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
