#!/usr/bin/env python3
"""Export and arena-evaluate saved top-k checkpoints against current."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.corrected_guard_kill_gate import (
    DEFAULT_FALLBACK_REFERENCE_ARTIFACT,
    parse_budgets,
    run_corrected_guard_kill_gate,
    write_json as write_guard_json,
)
from ml.alphazero_lite.superhuman_runtime_config import python_executable, repo_root
from ml.alphazero_lite.worker_config import DEFAULT_WORKERS, normalize_command_workers


DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_MODEL_TYPE = "residual_v3"
DEFAULT_INPUT_ENCODING = "kalah_v3"
DEFAULT_RULES_VERSION = "kalah_v1"
DEFAULT_GAMES = 60
DEFAULT_CHALLENGER_SIMULATIONS = 640
DEFAULT_CURRENT_SIMULATIONS = 256
DEFAULT_MIN_SCORE = 0.55

TOP_CHECKPOINT_PATTERN = re.compile(r"^checkpoint\.top(\d+)\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint.npz and checkpoint.top*.npz by arena strength."
    )
    parser.add_argument("--iter-dir", required=True)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--input-encoding", default=DEFAULT_INPUT_ENCODING)
    parser.add_argument("--rules-version", default=DEFAULT_RULES_VERSION)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--guard-reference-artifact", default=None)
    parser.add_argument("--guard-fallback-reference-artifact", default=None)
    parser.add_argument("--guard-budgets", default="384,1200")
    parser.add_argument("--guard-seed", type=int, default=17)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_checkpoints(iter_dir: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    base_checkpoint = iter_dir / "checkpoint.npz"
    if base_checkpoint.exists():
        candidates.append((0, base_checkpoint))

    for path in iter_dir.glob("checkpoint.top*.npz"):
        match = TOP_CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))

    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def default_out_path(iter_dir: Path) -> Path:
    return iter_dir / "top_k_evaluation_summary.json"


def artifact_dir_for_checkpoint(iter_dir: Path, checkpoint_path: Path) -> Path:
    return iter_dir / "top_k_exports" / checkpoint_path.stem


def report_path_for_checkpoint(export_dir: Path) -> Path:
    return export_dir / "arena_report.json"


def export_version(iter_dir: Path, checkpoint_path: Path) -> str:
    return f"{iter_dir.name}-{checkpoint_path.stem}"


def build_export_command(
    *,
    python_bin: str,
    checkpoint_path: Path,
    export_dir: Path,
    iter_dir: Path,
    model_type: str,
    input_encoding: str,
    rules_version: str,
) -> list[str]:
    return [
        python_bin,
        "ml/alphazero_lite/export_artifact.py",
        "--checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(export_dir),
        "--version",
        export_version(iter_dir, checkpoint_path),
        "--model-type",
        model_type,
        "--rules-version",
        rules_version,
        "--input-encoding",
        input_encoding,
    ]


def build_arena_command(
    *,
    python_bin: str,
    challenger_path: Path,
    current_path: Path,
    report_path: Path,
    games: int,
    workers: int,
) -> list[str]:
    command = [
        python_bin,
        "ml/alphazero_lite/arena.py",
        "--challenger",
        str(challenger_path),
        "--current",
        str(current_path),
        "--games",
        str(games),
        "--challenger-simulations",
        str(DEFAULT_CHALLENGER_SIMULATIONS),
        "--current-simulations",
        str(DEFAULT_CURRENT_SIMULATIONS),
        "--out",
        str(report_path),
        "--min-score",
        str(DEFAULT_MIN_SCORE),
    ]
    return normalize_command_workers(command, workers=workers)


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=repo_root(), check=True)


def arena_score(report: dict) -> float | None:
    if "score" in report:
        return float(report["score"])
    games_played = int(report.get("games_played", 0))
    if games_played <= 0:
        return None
    wins = int(report.get("wins", 0))
    draws = int(report.get("draws", 0))
    return (wins + (0.5 * draws)) / games_played


def load_report_summary(report_path: Path) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "score": arena_score(report),
        "wins": report.get("wins"),
        "losses": report.get("losses"),
        "draws": report.get("draws"),
        "report_path": str(report_path),
    }


def evaluate_checkpoint(
    *,
    python_bin: str,
    checkpoint_path: Path,
    iter_dir: Path,
    current_path: Path,
    model_type: str,
    input_encoding: str,
    rules_version: str,
    games: int,
    workers: int,
    guard_reference_artifact: Path | None,
    guard_fallback_reference_artifact: Path | None,
    guard_budgets: tuple[int, ...],
    guard_seed: int,
    dry_run: bool,
) -> dict[str, object]:
    export_dir = artifact_dir_for_checkpoint(iter_dir, checkpoint_path)
    report_path = report_path_for_checkpoint(export_dir)
    guard_gate_path = export_dir / "corrected_guard_kill_gate.json"
    export_command = build_export_command(
        python_bin=python_bin,
        checkpoint_path=checkpoint_path,
        export_dir=export_dir,
        iter_dir=iter_dir,
        model_type=model_type,
        input_encoding=input_encoding,
        rules_version=rules_version,
    )
    arena_command = build_arena_command(
        python_bin=python_bin,
        challenger_path=export_dir,
        current_path=current_path,
        report_path=report_path,
        games=games,
        workers=workers,
    )

    summary: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "export_dir": str(export_dir),
        "report_path": str(report_path),
        "guard_gate_path": str(guard_gate_path),
        "export_command": export_command,
        "arena_command": arena_command,
        "dry_run": dry_run,
    }
    if dry_run:
        if guard_reference_artifact is not None:
            summary["guard_reference_artifact"] = str(guard_reference_artifact)
            summary["guard_budgets"] = list(guard_budgets)
        return summary

    run_command(export_command)
    if guard_reference_artifact is not None:
        guard_payload = run_corrected_guard_kill_gate(
            candidate_path=export_dir,
            reference_artifact=guard_reference_artifact,
            fallback_reference_artifact=guard_fallback_reference_artifact,
            budgets=guard_budgets,
            seed=guard_seed,
        )
        write_guard_json(guard_gate_path, guard_payload)
        summary.update(
            {
                "guard_reference_artifact": str(guard_reference_artifact),
                "guard_budgets": list(guard_budgets),
                "guard_gate_pass": bool(guard_payload["pass"]),
                "guard_gate_decision": guard_payload["decision"],
                "guard_gate_reason": guard_payload["reason"],
                "guard_gate_rows": guard_payload["rows"],
            }
        )
        if not guard_payload["pass"]:
            summary["arena_skipped_due_to_guard"] = True
            summary["decision"] = "reject_guard_regression"
            return summary

    run_command(arena_command)
    summary.update(load_report_summary(report_path))
    summary["arena_skipped_due_to_guard"] = False
    summary["decision"] = "arena_evaluated"
    return summary


def main() -> None:
    args = parse_args()
    iter_dir = Path(args.iter_dir)
    if not iter_dir.exists():
        raise SystemExit(f"iter dir does not exist: {iter_dir}")

    current_path = Path(args.current_path)
    if not args.dry_run and not current_path.exists():
        raise SystemExit(f"current path does not exist: {current_path}")

    checkpoints = discover_checkpoints(iter_dir)
    if not checkpoints:
        raise SystemExit(f"no checkpoints found in {iter_dir}")

    out_path = Path(args.out) if args.out else default_out_path(iter_dir)
    python_bin = python_executable(repo_root())
    guard_reference_artifact = (
        None
        if args.guard_reference_artifact is None
        else Path(args.guard_reference_artifact)
    )
    guard_fallback_reference_artifact = (
        Path(args.guard_fallback_reference_artifact)
        if args.guard_fallback_reference_artifact is not None
        else DEFAULT_FALLBACK_REFERENCE_ARTIFACT
    )
    guard_budgets = parse_budgets(args.guard_budgets)
    results = [
        evaluate_checkpoint(
            python_bin=python_bin,
            checkpoint_path=checkpoint_path,
            iter_dir=iter_dir,
            current_path=current_path,
            model_type=args.model_type,
            input_encoding=args.input_encoding,
            rules_version=args.rules_version,
            games=args.games,
            workers=args.workers,
            guard_reference_artifact=guard_reference_artifact,
            guard_fallback_reference_artifact=guard_fallback_reference_artifact,
            guard_budgets=guard_budgets,
            guard_seed=int(args.guard_seed),
            dry_run=bool(args.dry_run),
        )
        for checkpoint_path in checkpoints
    ]

    summary = {
        "schema": "top_k_evaluation_summary_v1",
        "iter_dir": str(iter_dir),
        "current_path": str(current_path),
        "model_type": args.model_type,
        "input_encoding": args.input_encoding,
        "rules_version": args.rules_version,
        "games": int(args.games),
        "workers": int(args.workers),
        "guard_reference_artifact": None
        if guard_reference_artifact is None
        else str(guard_reference_artifact),
        "guard_fallback_reference_artifact": None
        if guard_fallback_reference_artifact is None
        else str(guard_fallback_reference_artifact),
        "guard_budgets": list(guard_budgets),
        "guard_seed": int(args.guard_seed),
        "challenger_simulations": DEFAULT_CHALLENGER_SIMULATIONS,
        "current_simulations": DEFAULT_CURRENT_SIMULATIONS,
        "min_score": DEFAULT_MIN_SCORE,
        "dry_run": bool(args.dry_run),
        "candidates": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote top-k evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
