#!/usr/bin/env python3
"""Diagnostic audit for opening-suite duplicate-artifact determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = Path("/tmp/azlite_opening_suite_determinism_audit")
DEFAULT_REPORT_PATH = DEFAULT_WORKDIR / "determinism_audit_report.json"
DEFAULT_DOC_PATH = (
    REPO_ROOT / "docs/alphazero-lite-opening-suite-determinism-audit-results.md"
)
PRE_FIX_FINDINGS = [
    {
        "issue": "worker_shared_games_jsonl",
        "classification": "evaluator_worker_nondeterminism",
        "symptom": "workers>1 could leave merged seat metrics reading incomplete or last-writer-wins game rows",
        "first_divergence": {
            "budget_label": "standard",
            "field": "game_rows",
            "detail": {
                "reason": "row_count_mismatch",
                "left_count": 512,
                "right_count": 0,
            },
        },
    },
    {
        "issue": "deterministic_mode_ignored_cli_seed",
        "classification": "evaluation_configuration_bug",
        "symptom": "deterministic root-temperature runs always used the first --seeds entry instead of --seed",
    },
]
BUDGET_LABELS = {
    "384:256": "standard",
    "768:256": "challenger_768_vs_256",
    "768:768": "equal_768",
    "1200:1200": "equal_high",
    "256:768": "current_high_asymmetry",
}


def _python() -> str:
    candidate = REPO_ROOT / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit duplicate-artifact determinism in the opening-suite benchmark."
    )
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--source-artifact", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument("--budget-pairs", default="384:256,256:768")
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--workers-list", default="1,24")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--aliases", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_budget_pairs(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def export_source_artifact(*, checkpoint: Path, out_dir: Path) -> Path:
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        "--version",
        "azlite-determinism-audit-source",
        "--model-type",
        "residual_v3",
        "--input-encoding",
        "kalah_v3",
        "--rules-version",
        "kalah_v3",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "export_artifact.py failed")
    return out_dir


def resolve_source_artifact(args: argparse.Namespace, workdir: Path) -> Path:
    if args.source_artifact:
        return Path(args.source_artifact)
    if not args.source_checkpoint:
        raise ValueError("pass --source-artifact or --source-checkpoint")
    exported = workdir / "source_artifact_export"
    if exported.exists():
        shutil.rmtree(exported)
    return export_source_artifact(
        checkpoint=Path(args.source_checkpoint), out_dir=exported
    )


def create_alias_artifacts(
    *, workdir: Path, source_artifact: Path, aliases: int
) -> list[dict]:
    alias_root = workdir / "aliases"
    alias_root.mkdir(parents=True, exist_ok=True)
    expected_sha = sha256_file(source_artifact / "weights.json")
    alias_infos = []
    for index in range(1, aliases + 1):
        alias_dir = alias_root / f"alias_{index:02d}"
        if alias_dir.exists():
            shutil.rmtree(alias_dir)
        shutil.copytree(source_artifact, alias_dir)
        alias_sha = sha256_file(alias_dir / "weights.json")
        if alias_sha != expected_sha:
            raise RuntimeError(
                f"alias SHA mismatch for {alias_dir}: expected {expected_sha}, got {alias_sha}"
            )
        alias_infos.append(
            {
                "alias": alias_dir.name,
                "path": str(alias_dir),
                "weights_sha256": alias_sha,
            }
        )
    return alias_infos


def run_benchmark_condition(
    *,
    condition_id: str,
    order: str,
    worker_count: int,
    seed: int,
    budget_pairs: str,
    games_per_opening: int,
    timeout: int,
    suite: str,
    current: str,
    alias_infos: list[dict],
    workdir: Path,
) -> Path:
    condition_dir = workdir / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)
    ordered_aliases = list(alias_infos)
    if order == "reversed":
        ordered_aliases.reverse()
    candidates = ",".join(alias["path"] for alias in ordered_aliases)
    cmd = [
        _python(),
        str(REPO_ROOT / "ml/alphazero_lite/run_opening_suite_seat_benchmark.py"),
        "--workdir",
        str(condition_dir),
        "--suite",
        suite,
        "--current",
        current,
        "--candidates",
        candidates,
        "--budget-pairs",
        budget_pairs,
        "--games-per-opening",
        str(games_per_opening),
        "--seed",
        str(seed),
        "--workers",
        str(worker_count),
        "--root-policy-mode",
        "deterministic",
        "--timeout",
        str(timeout),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout + 300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"condition {condition_id} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return condition_dir / "temperature_benchmark_report.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_game_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "game_index": int(row["game_index"]),
                "challenger_player": int(row["challenger_player"]),
                "winner": row["winner"],
                "margin": int(row["margin"]),
                "trajectory": row["trajectory"],
                "opening_prefix_moves": [
                    int(move) for move in row.get("opening_prefix_moves", [])
                ],
            }
        )
    normalized.sort(key=lambda row: row["game_index"])
    return normalized


def normalize_per_opening(metrics: list[dict]) -> list[dict]:
    normalized = []
    for entry in metrics:
        normalized.append(
            {
                "opening_prefix": entry.get("opening_prefix"),
                "p0_score": round(float(entry.get("p0_score", 0.0)), 12),
                "p1_score": round(float(entry.get("p1_score", 0.0)), 12),
                "ds": round(float(entry.get("ds", 0.0)), 12),
                "disadvantaged_seat_score": round(
                    float(entry.get("disadvantaged_seat_score", 0.0)), 12
                ),
                "total_games": int(entry.get("total_games", 0)),
            }
        )
    normalized.sort(key=lambda entry: str(entry.get("opening_prefix")))
    return normalized


def budget_result_summary(budget_result: dict) -> dict:
    return {
        "p0_score": round(float(budget_result.get("p0_score", 0.0)), 12),
        "p1_score": round(float(budget_result.get("p1_score", 0.0)), 12),
        "ds": round(float(budget_result.get("ds", 0.0)), 12),
        "disadvantaged_seat_score": round(
            float(budget_result.get("disadvantaged_seat_score", 0.0)), 12
        ),
        "total_games": int(budget_result.get("total_games", 0)),
        "duplicate_trajectory_rate": round(
            float(budget_result.get("duplicate_trajectory_rate", 0.0)), 12
        ),
        "per_opening_metrics": normalize_per_opening(
            budget_result.get("per_opening_metrics", [])
        ),
    }


def load_condition_outputs(
    *,
    condition_id: str,
    report_path: Path,
    seed: int,
    alias_infos: list[dict],
    budget_pairs: list[str],
) -> dict:
    report = read_json(report_path)
    seed_report = report["temperature_reports"][0]["seed_reports"][0]
    candidate_reports = {
        entry["candidate"]: entry for entry in seed_report.get("candidate_reports", [])
    }
    outputs = {
        "condition_id": condition_id,
        "report_path": str(report_path),
        "seed": seed,
        "candidate_order": [
            entry["candidate"] for entry in seed_report["candidate_reports"]
        ],
        "aliases": {},
    }
    seed_dir = report_path.parent / "temp_0_0" / f"seed_{seed}"
    for alias in alias_infos:
        alias_name = alias["alias"]
        candidate_report = candidate_reports[alias_name]
        alias_entry = {
            "candidate_sha256": candidate_report["candidate_sha256"],
            "budget_results": {},
        }
        for budget_pair in budget_pairs:
            budget_label = BUDGET_LABELS.get(
                budget_pair, budget_pair.replace(":", "_vs_")
            )
            budget_result = candidate_report.get("budget_results", {}).get(
                budget_label, {}
            )
            budget_dir = seed_dir / alias_name / budget_label
            all_rows = []
            for seat in (0, 1):
                all_rows.extend(
                    read_jsonl(budget_dir / f"starts_{seat}" / "games.jsonl")
                )
            alias_entry["budget_results"][budget_label] = {
                **budget_result_summary(budget_result),
                "games": normalize_game_rows(all_rows),
            }
        outputs["aliases"][alias_name] = alias_entry
    return outputs


def find_first_difference(left: list[dict], right: list[dict]) -> dict | None:
    if len(left) != len(right):
        return {
            "reason": "row_count_mismatch",
            "left_count": len(left),
            "right_count": len(right),
        }
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if left_row != right_row:
            return {
                "reason": "row_mismatch",
                "row_index": index,
                "left": left_row,
                "right": right_row,
            }
    return None


def compare_aliases(condition: dict) -> dict:
    alias_names = sorted(condition["aliases"])
    baseline = alias_names[0]
    comparisons = []
    mismatches = []
    for alias_name in alias_names[1:]:
        alias_mismatch = []
        baseline_entry = condition["aliases"][baseline]
        alias_entry = condition["aliases"][alias_name]
        if baseline_entry["candidate_sha256"] != alias_entry["candidate_sha256"]:
            alias_mismatch.append(
                {
                    "kind": "candidate_sha256",
                    "left": baseline_entry["candidate_sha256"],
                    "right": alias_entry["candidate_sha256"],
                }
            )
        for budget_label, baseline_budget in baseline_entry["budget_results"].items():
            alias_budget = alias_entry["budget_results"][budget_label]
            for field in (
                "p0_score",
                "p1_score",
                "ds",
                "disadvantaged_seat_score",
                "total_games",
                "duplicate_trajectory_rate",
            ):
                if baseline_budget[field] != alias_budget[field]:
                    alias_mismatch.append(
                        {
                            "kind": "budget_metric",
                            "budget_label": budget_label,
                            "field": field,
                            "left": baseline_budget[field],
                            "right": alias_budget[field],
                        }
                    )
            if (
                baseline_budget["per_opening_metrics"]
                != alias_budget["per_opening_metrics"]
            ):
                alias_mismatch.append(
                    {
                        "kind": "per_opening_metrics",
                        "budget_label": budget_label,
                    }
                )
            game_diff = find_first_difference(
                baseline_budget["games"], alias_budget["games"]
            )
            if game_diff is not None:
                alias_mismatch.append(
                    {
                        "kind": "game_rows",
                        "budget_label": budget_label,
                        "detail": game_diff,
                    }
                )
        comparisons.append(
            {
                "baseline_alias": baseline,
                "alias": alias_name,
                "matches": not alias_mismatch,
                "mismatches": alias_mismatch,
            }
        )
        mismatches.extend(alias_mismatch)
    return {
        "baseline_alias": baseline,
        "all_aliases_match": not mismatches,
        "comparisons": comparisons,
        "first_mismatch": None if not mismatches else mismatches[0],
    }


def compare_conditions(*, left: dict, right: dict, label: str) -> dict:
    mismatches = []
    for alias_name in sorted(left["aliases"]):
        left_entry = left["aliases"][alias_name]
        right_entry = right["aliases"][alias_name]
        for budget_label, left_budget in left_entry["budget_results"].items():
            right_budget = right_entry["budget_results"][budget_label]
            for field in (
                "p0_score",
                "p1_score",
                "ds",
                "disadvantaged_seat_score",
                "total_games",
                "duplicate_trajectory_rate",
            ):
                if left_budget[field] != right_budget[field]:
                    mismatches.append(
                        {
                            "alias": alias_name,
                            "budget_label": budget_label,
                            "field": field,
                            "left": left_budget[field],
                            "right": right_budget[field],
                        }
                    )
            if (
                left_budget["per_opening_metrics"]
                != right_budget["per_opening_metrics"]
            ):
                mismatches.append(
                    {
                        "alias": alias_name,
                        "budget_label": budget_label,
                        "field": "per_opening_metrics",
                    }
                )
            game_diff = find_first_difference(
                left_budget["games"], right_budget["games"]
            )
            if game_diff is not None:
                mismatches.append(
                    {
                        "alias": alias_name,
                        "budget_label": budget_label,
                        "field": "game_rows",
                        "detail": game_diff,
                    }
                )
    return {
        "label": label,
        "matches": not mismatches,
        "first_mismatch": None if not mismatches else mismatches[0],
    }


def classify(report: dict) -> str:
    condition_checks = report["condition_alias_checks"]
    all_conditions_match = all(
        check["all_aliases_match"] for check in condition_checks.values()
    )
    worker_check = report["cross_condition_checks"]["workers_1_vs_24_normal_seed42"]
    order_check = report["cross_condition_checks"][
        "workers_24_normal_vs_reversed_seed42"
    ]
    seed_check = report["cross_condition_checks"]["workers_24_seed42_vs_seed43"]
    if not all_conditions_match:
        if condition_checks["workers_1_normal_seed_42"]["all_aliases_match"] and (
            not condition_checks["workers_24_normal_seed_42"]["all_aliases_match"]
            or not condition_checks["workers_24_reversed_seed_42"]["all_aliases_match"]
            or not condition_checks["workers_24_normal_seed_43"]["all_aliases_match"]
        ):
            return "evaluator_worker_nondeterminism"
        return "evaluator_duplicate_nondeterministic"
    if not order_check["matches"]:
        return "evaluator_order_nondeterminism"
    if not worker_check["matches"]:
        return "evaluator_worker_nondeterminism"
    if not seed_check["matches"]:
        return "evaluator_seed_sensitive_deterministic"
    return "evaluator_duplicate_deterministic"


def build_markdown(report: dict) -> str:
    lines = [
        "# AlphaZero-Lite Opening-Suite Determinism Audit Results",
        "",
        "**Date**: 2026-06-13",
        f"**Classification**: `{report['classification']}`",
        "**Schema**: `azlite_opening_suite_determinism_audit_v1`",
        "",
        "## Summary",
        "",
        (
            "Audited duplicate aliases of a single candidate artifact against "
            "`model-artifact/current` under deterministic root-policy opening-suite evaluation."
        ),
        (f" Source artifact weights SHA256: `{report['source']['weights_sha256']}`."),
        "",
        "## Conditions",
        "",
        "| Condition | Workers | Candidate order | Seed | Alias match |",
        "|-----------|---------|-----------------|------|-------------|",
    ]
    for condition in report["conditions"]:
        check = report["condition_alias_checks"][condition["condition_id"]]
        lines.append(
            f"| `{condition['condition_id']}` | {condition['workers']} | {condition['order']} | {condition['seed']} | {'PASS' if check['all_aliases_match'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Condition Checks",
            "",
            "| Check | Result |",
            "|------|--------|",
        ]
    )
    for label, entry in report["cross_condition_checks"].items():
        lines.append(f"| `{label}` | {'PASS' if entry['matches'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Before Fixes",
            "",
            "| Issue | Symptom |",
            "|------|---------|",
        ]
    )
    for finding in report["pre_fix_findings"]:
        lines.append(f"| `{finding['issue']}` | {finding['symptom']} |")
    lines.extend(
        [
            "",
            "## Root Cause",
            "",
            (
                "During the audit, `arena.py` was corrected so each worker writes its own "
                "temporary `games.jsonl` and the parent process merges rows by `game_index`. "
                "The previous worker-shared path allowed last-writer-wins row loss under "
                "multiprocessing, which could corrupt per-game diagnostics and seat-aware "
                "metrics derived from `games.jsonl`."
            ),
            (
                "`run_opening_suite_seat_benchmark.py` was also corrected so deterministic "
                "root-temperature runs honor `--seed` instead of silently reusing the first "
                "`--seeds` value."
            ),
        ]
    )
    mismatch = report.get("first_detected_mismatch")
    if mismatch is not None:
        lines.extend(
            [
                "",
                "## First Mismatch",
                "",
                "```json",
                json.dumps(mismatch, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Report JSON: `{report['report_path']}`",
            f"- Audit workdir: `{report['workdir']}`",
            f"- Source artifact: `{report['source']['artifact_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    budget_pairs = parse_budget_pairs(args.budget_pairs)
    workers_list = parse_int_list(args.workers_list)
    seeds = parse_int_list(args.seeds)
    if len(workers_list) < 2:
        raise ValueError("--workers-list must contain at least two entries")
    if len(seeds) < 2:
        raise ValueError("--seeds must contain at least two entries")

    source_artifact = resolve_source_artifact(args, workdir)
    alias_infos = create_alias_artifacts(
        workdir=workdir,
        source_artifact=source_artifact,
        aliases=max(2, args.aliases),
    )
    conditions = [
        {
            "condition_id": f"workers_{workers_list[0]}_normal_seed_{seeds[0]}",
            "workers": workers_list[0],
            "order": "normal",
            "seed": seeds[0],
        },
        {
            "condition_id": f"workers_{workers_list[-1]}_normal_seed_{seeds[0]}",
            "workers": workers_list[-1],
            "order": "normal",
            "seed": seeds[0],
        },
        {
            "condition_id": f"workers_{workers_list[-1]}_reversed_seed_{seeds[0]}",
            "workers": workers_list[-1],
            "order": "reversed",
            "seed": seeds[0],
        },
        {
            "condition_id": f"workers_{workers_list[-1]}_normal_seed_{seeds[1]}",
            "workers": workers_list[-1],
            "order": "normal",
            "seed": seeds[1],
        },
    ]

    condition_outputs = {}
    for condition in conditions:
        report_path = run_benchmark_condition(
            condition_id=condition["condition_id"],
            order=condition["order"],
            worker_count=condition["workers"],
            seed=condition["seed"],
            budget_pairs=args.budget_pairs,
            games_per_opening=args.games_per_opening,
            timeout=args.timeout,
            suite=args.suite,
            current=args.current,
            alias_infos=alias_infos,
            workdir=workdir,
        )
        condition_outputs[condition["condition_id"]] = load_condition_outputs(
            condition_id=condition["condition_id"],
            report_path=report_path,
            seed=condition["seed"],
            alias_infos=alias_infos,
            budget_pairs=budget_pairs,
        )

    condition_alias_checks = {
        condition_id: compare_aliases(output)
        for condition_id, output in condition_outputs.items()
    }
    cross_condition_checks = {
        "workers_1_vs_24_normal_seed42": compare_conditions(
            left=condition_outputs[conditions[0]["condition_id"]],
            right=condition_outputs[conditions[1]["condition_id"]],
            label="workers_1_vs_24_normal_seed42",
        ),
        "workers_24_normal_vs_reversed_seed42": compare_conditions(
            left=condition_outputs[conditions[1]["condition_id"]],
            right=condition_outputs[conditions[2]["condition_id"]],
            label="workers_24_normal_vs_reversed_seed42",
        ),
        "workers_24_seed42_vs_seed43": compare_conditions(
            left=condition_outputs[conditions[1]["condition_id"]],
            right=condition_outputs[conditions[3]["condition_id"]],
            label="workers_24_seed42_vs_seed43",
        ),
    }

    first_detected_mismatch = None
    for check in condition_alias_checks.values():
        if check["first_mismatch"] is not None:
            first_detected_mismatch = check["first_mismatch"]
            break
    if first_detected_mismatch is None:
        for check in cross_condition_checks.values():
            if check["first_mismatch"] is not None:
                first_detected_mismatch = check["first_mismatch"]
                break

    report = {
        "schema": "azlite_opening_suite_determinism_audit_v1",
        "workdir": str(workdir),
        "report_path": str(DEFAULT_REPORT_PATH),
        "source": {
            "artifact_path": str(source_artifact),
            "weights_sha256": sha256_file(source_artifact / "weights.json"),
            "checkpoint_path": args.source_checkpoint,
        },
        "pre_fix_findings": PRE_FIX_FINDINGS,
        "current_artifact": str(args.current),
        "suite": str(args.suite),
        "budget_pairs": budget_pairs,
        "games_per_opening": int(args.games_per_opening),
        "aliases": alias_infos,
        "conditions": conditions,
        "condition_outputs": condition_outputs,
        "condition_alias_checks": condition_alias_checks,
        "cross_condition_checks": cross_condition_checks,
        "first_detected_mismatch": first_detected_mismatch,
    }
    report["classification"] = classify(report)

    report_path = workdir / "determinism_audit_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    DEFAULT_DOC_PATH.write_text(build_markdown(report), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {DEFAULT_DOC_PATH}")
    print(f"classification={report['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
