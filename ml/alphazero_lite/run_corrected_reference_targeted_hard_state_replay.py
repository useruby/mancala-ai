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

from ml.alphazero_lite import run_hard_state_replay_experiment


DEFAULT_CORRECTED_INVENTORY = Path(
    "/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json"
)
DEFAULT_REBASELINE_REPORT = Path(
    "/tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--forensic-report", default=str(DEFAULT_REBASELINE_REPORT))
    parser.add_argument(
        "--corrected-inventory", default=str(DEFAULT_CORRECTED_INVENTORY)
    )
    parser.add_argument(
        "--base-config",
        default=run_hard_state_replay_experiment.DEFAULT_BASE_CONFIG,
    )
    parser.add_argument("--current-path", default=None)
    parser.add_argument("--hard-state-validation-path", default=None)
    parser.add_argument("--top-n", type=int, default=64)
    parser.add_argument("--canonical-budget", type=int, default=384)
    parser.add_argument("--stronger-budget", type=int, default=1200)
    parser.add_argument(
        "--variant-weights",
        default=",".join(
            str(value)
            for value in run_hard_state_replay_experiment.DEFAULT_VARIANT_WEIGHTS
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_challenger_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    challenger = dict(payload.get("systems", {})).get("challenger")
    if not isinstance(challenger, dict):
        raise ValueError("forensic artifact must contain challenger rows")
    rows = challenger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("forensic artifact challenger rows must be a list")
    return [row for row in rows if isinstance(row, dict)]


def _write_filtered_forensic_artifact(
    *,
    source_artifact_path: Path,
    selected_canonical_states: set[str],
    output_path: Path,
) -> dict[str, Any]:
    payload = load_json(source_artifact_path)
    challenger = dict(payload.get("systems", {})).get("challenger")
    if not isinstance(challenger, dict):
        raise ValueError(
            "filtered forensic artifact source must contain challenger rows"
        )
    rows = challenger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("filtered forensic artifact challenger rows must be a list")
    filtered_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("canonical_state", "")) in selected_canonical_states
    ]
    filtered_payload = {
        **payload,
        "systems": {
            **dict(payload.get("systems", {})),
            "challenger": {
                **challenger,
                "rows": filtered_rows,
            },
        },
    }
    write_json(output_path, filtered_payload)
    return filtered_payload


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _ordered_family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_priority = [
        "capture_available",
        "opening_plies_1_8",
        "incumbent_proxy_disagreement",
        "high_value_swing",
        "high_imbalance",
        "starvation_pressure",
        "sparse_endgame",
        "early_extra_turn",
    ]
    family_rank = {family: index for index, family in enumerate(family_priority)}
    return sorted(
        rows,
        key=lambda row: (
            family_rank.get(str(row.get("_family", "")), 999),
            -float(dict(row.get("metadata", {})).get("max_regret") or 0.0),
            -float(dict(row.get("metadata", {})).get("max_value_error") or 0.0),
            str(row.get("canonical_state", "")),
        ),
    )


def apply_family_quota(
    rows: list[dict[str, Any]],
    top_n: int,
    *,
    canonical_to_family: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = {
        "capture_available": int(top_n * 0.40),
        "opening_plies_1_8": int(top_n * 0.15),
        "incumbent_proxy_disagreement": int(top_n * 0.15),
        "high_value_swing": int(top_n * 0.15),
        "high_imbalance": int(top_n * 0.15),
    }
    spill_order = ["starvation_pressure", "sparse_endgame", "early_extra_turn"]
    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    rows_with_family = []
    for row in rows:
        canonical_state = str(row.get("canonical_state", ""))
        family = canonical_to_family.get(canonical_state, "")
        rows_with_family.append({**row, "_family": family})
    for row in _ordered_family_rows(rows_with_family):
        family = str(row.get("_family", ""))
        rows_by_family.setdefault(family, []).append(row)

    selected: list[dict[str, Any]] = []
    selected_canonical_states: set[str] = set()
    family_counts: dict[str, int] = {}

    for family, quota in quotas.items():
        picks = []
        for row in rows_by_family.get(family, []):
            canonical_state = str(row.get("canonical_state", ""))
            if canonical_state in selected_canonical_states:
                continue
            picks.append(row)
            selected_canonical_states.add(canonical_state)
            if len(picks) >= quota:
                break
        selected.extend(picks)
        family_counts[family] = len(picks)

    if len(selected) < top_n:
        remaining = []
        for family in spill_order:
            remaining.extend(rows_by_family.get(family, []))
        for family, family_rows in rows_by_family.items():
            if family in quotas or family in spill_order:
                continue
            remaining.extend(family_rows)
        for row in remaining:
            canonical_state = str(row.get("canonical_state", ""))
            if canonical_state in selected_canonical_states:
                continue
            selected.append(row)
            selected_canonical_states.add(canonical_state)
            if len(selected) >= top_n:
                break

    selected = _ordered_family_rows(selected)[:top_n]
    return selected, {
        "requested_top_n": top_n,
        "selected_top_n": len(selected),
        "family_quota_targets": quotas,
        "family_selected_counts": {
            family: sum(1 for row in selected if str(row.get("_family", "")) == family)
            for family in sorted({str(row.get("_family", "")) for row in selected})
        },
    }


def main() -> None:
    args = parse_args()
    root = run_hard_state_replay_experiment.repo_root()
    python = python_bin(root)
    derived = run_hard_state_replay_experiment.derived_paths(
        root, args.run_id, args.output_root, args.seed
    )
    artifact_path = derived["inputs_dir"] / "corrected_reference_forensic_suite.json"
    artifact_summary_path = (
        derived["reports_dir"] / "corrected_reference_forensic_suite_summary.json"
    )
    filtered_artifact_path = (
        derived["inputs_dir"] / "corrected_reference_forensic_suite_quota_filtered.json"
    )
    quota_summary_path = derived["reports_dir"] / "family_quota_summary.json"
    filtered_jsonl_path = (
        derived["inputs_dir"] / f"corrected_reference_mined_seed{args.seed}.jsonl"
    )
    filtered_report_path = (
        derived["reports_dir"]
        / f"corrected_reference_mined_seed{args.seed}_report.json"
    )

    build_command = [
        python,
        "-m",
        "ml.alphazero_lite.build_corrected_reference_hard_state_artifact",
        "--forensic-report",
        str(Path(args.forensic_report)),
        "--corrected-inventory",
        str(Path(args.corrected_inventory)),
        "--out-artifact",
        str(artifact_path),
        "--out-summary",
        str(artifact_summary_path),
    ]
    if not args.dry_run:
        completed = subprocess.run(build_command, cwd=root, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"command failed with exit code {completed.returncode}: {' '.join(build_command)}"
            )

    mine_command = [
        python,
        "-m",
        "ml.alphazero_lite.mine_hard_states",
        "--inputs",
        str(artifact_path),
        "--out-jsonl",
        str(filtered_jsonl_path),
        "--out-report",
        str(filtered_report_path),
    ]
    if not args.dry_run:
        completed = subprocess.run(mine_command, cwd=root, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"command failed with exit code {completed.returncode}: {' '.join(mine_command)}"
            )
        mined_rows = [
            json.loads(line)
            for line in filtered_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        canonical_to_family = {
            str(row.get("canonical_state", "")): str(
                row.get("corrected_failure_family", row.get("bucket", ""))
            )
            for row in _load_challenger_rows(artifact_path)
        }
        selected_rows, quota_summary = apply_family_quota(
            mined_rows,
            args.top_n,
            canonical_to_family=canonical_to_family,
        )
        filtered_jsonl_path.write_text(
            "".join(json.dumps(row) + "\n" for row in selected_rows), encoding="utf-8"
        )
        _write_filtered_forensic_artifact(
            source_artifact_path=artifact_path,
            selected_canonical_states={
                str(row.get("canonical_state", "")) for row in selected_rows
            },
            output_path=filtered_artifact_path,
        )
        write_json(quota_summary_path, quota_summary)

    replay_args = [
        "--run-id",
        args.run_id,
        "--output-root",
        args.output_root,
        "--mine-inputs",
        str(filtered_artifact_path if not args.dry_run else artifact_path),
        "--base-config",
        args.base_config,
        "--top-n",
        str(args.top_n),
        "--canonical-budget",
        str(args.canonical_budget),
        "--stronger-budget",
        str(args.stronger_budget),
        "--variant-weights",
        args.variant_weights,
        "--seed",
        str(args.seed),
    ]
    if args.current_path is not None:
        replay_args.extend(["--current-path", args.current_path])
    if args.hard_state_validation_path is not None:
        replay_args.extend(
            ["--hard-state-validation-path", args.hard_state_validation_path]
        )
    if args.dry_run:
        replay_args.append("--dry-run")
    run_hard_state_replay_experiment.main(replay_args)


if __name__ == "__main__":
    main()
