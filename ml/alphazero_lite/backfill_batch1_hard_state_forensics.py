#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_HOLDOUT_SUITE = Path("ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json")
DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--suite-path")
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--mcts-simulations", type=int, default=1200)
    parser.add_argument("--teacher-simulations", type=int, default=1800)
    parser.add_argument("--artifact-simulations", type=int, default=384)
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


def discover_gate_reports(gate_root: Path) -> list[Path]:
    reports = sorted(gate_root.rglob("local_promotion.json"))
    if not reports:
        raise ValueError(f"no local_promotion.json reports found under {gate_root}")
    return reports


def validate_suite_path(root: Path, suite_path: Path | None) -> tuple[Path | None, str | None]:
    if suite_path is None:
        return None, "no train-only forensic suite supplied"

    if not suite_path.exists():
        raise ValueError(f"suite path does not exist: {suite_path}")

    holdout_path = (root / DEFAULT_HOLDOUT_SUITE).resolve()
    if suite_path.resolve() == holdout_path:
        raise ValueError(
            "suite path must not be the hard-state validation holdout; provide a separate train-only forensic suite"
        )
    return suite_path, None


def build_command(
    *,
    python: str,
    root: Path,
    suite_path: Path,
    current_path: str,
    candidate_path: str,
    out_path: Path,
    mcts_simulations: int,
    teacher_simulations: int,
    artifact_simulations: int,
    seed: int,
) -> list[str]:
    return [
        python,
        str(root / "ml/alphazero_lite/run_forensic_suite.py"),
        "--suite",
        str(suite_path),
        "--current-artifact",
        current_path,
        "--challenger-artifact",
        candidate_path,
        "--mcts-simulations",
        str(mcts_simulations),
        "--teacher-simulations",
        str(teacher_simulations),
        "--artifact-simulations",
        str(artifact_simulations),
        "--seed",
        str(seed),
        "--out",
        str(out_path),
    ]


def build_plan(
    *,
    root: Path,
    gate_reports: list[Path],
    out_root: Path,
    suite_path: Path | None,
    current_path: str,
    mcts_simulations: int,
    teacher_simulations: int,
    artifact_simulations: int,
    seed: int,
) -> dict:
    python = python_bin(root)
    variants: list[dict] = []
    blocked_reason: str | None = None

    for report_path in gate_reports:
        report = load_json(report_path)
        candidate_path = str(report.get("candidate_path", "")).strip()
        if not candidate_path:
            raise ValueError(f"missing candidate_path in {report_path}")

        variant_name = report_path.parent.name
        forensic_out = out_root / variant_name / "candidate_forensic_suite.json"
        variant = {
            "variant": variant_name,
            "gate_report": str(report_path),
            "candidate_path": candidate_path,
            "forensic_out": str(forensic_out),
            "status": "planned",
        }
        if suite_path is None:
            variant["status"] = "blocked_missing_train_suite"
            blocked_reason = "no train-only forensic suite supplied"
        else:
            variant["command"] = build_command(
                python=python,
                root=root,
                suite_path=suite_path,
                current_path=current_path,
                candidate_path=candidate_path,
                out_path=forensic_out,
                mcts_simulations=mcts_simulations,
                teacher_simulations=teacher_simulations,
                artifact_simulations=artifact_simulations,
                seed=seed,
            )
        variants.append(variant)

    return {
        "schema": "azlite_batch1_hard_state_forensics_backfill_v1",
        "gate_root": str(gate_reports[0].parents[1]),
        "out_root": str(out_root),
        "suite_path": None if suite_path is None else str(suite_path),
        "current_path": current_path,
        "blocked_reason": blocked_reason,
        "variants": variants,
    }


def execute_plan(plan: dict, *, dry_run: bool) -> dict:
    results: list[dict] = []
    for variant in plan["variants"]:
        command = variant.get("command")
        if not isinstance(command, list):
            continue
        if dry_run:
            results.append({"variant": variant["variant"], "returncode": None})
            continue

        out_path = Path(variant["forensic_out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=repo_root(), check=False)
        results.append(
            {
                "variant": variant["variant"],
                "returncode": completed.returncode,
                "forensic_out": variant["forensic_out"],
            }
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"command failed with exit code {completed.returncode}: {' '.join(command)}"
            )
    return {
        "executed": not dry_run,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    gate_root = resolve_path(root, args.gate_root)
    out_root = resolve_path(root, args.out_root)
    suite_path = None
    if args.suite_path is not None:
        suite_path = resolve_path(root, args.suite_path)
    suite_path, blocked_reason = validate_suite_path(root, suite_path)
    gate_reports = discover_gate_reports(gate_root)
    plan = build_plan(
        root=root,
        gate_reports=gate_reports,
        out_root=out_root,
        suite_path=suite_path,
        current_path=args.current_path,
        mcts_simulations=args.mcts_simulations,
        teacher_simulations=args.teacher_simulations,
        artifact_simulations=args.artifact_simulations,
        seed=args.seed,
    )
    if blocked_reason is not None:
        plan["blocked_reason"] = blocked_reason

    execution = execute_plan(plan, dry_run=args.dry_run)
    manifest_path = out_root / "manifest.json"
    write_json(manifest_path, {**plan, "execution": execution})
    print(json.dumps({"manifest": str(manifest_path), "blocked_reason": plan["blocked_reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
