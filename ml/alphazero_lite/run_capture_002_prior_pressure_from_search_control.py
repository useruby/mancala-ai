#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.capture_002_003_search_policy_arbitration import (
    build_payload,
    build_row_views,
    probe_artifact_position,
    validated_diagnostic_state,
)
from ml.alphazero_lite.capture_002_003_rule_collision_diagnostic import ROW_IDS


DEFAULT_CANDIDATE_PATH = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_CURRENT_PATH = "storage/ai/alphazero_lite/current"
DEFAULT_REFERENCE_ARTIFACT = "/tmp/azlite_failure_family_diag/train_only_forensic_references.json"
DEFAULT_SEARCH_CONTROL_CONFIG = (
    "ml/alphazero_lite/configs/aggressive_v3_superhuman_search_control.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_prior_pressure_from_search_control"
SIMULATIONS = 384
SEED = 17
C_PUCT = 1.25
SEARCH_OPTIONS = {
    "fpu_mode": "parent_q",
    "reuse_subtree": True,
    "normalize_values": True,
    "root_policy_mode": "deterministic",
    "tactical_root_bias": 0.1,
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
    parser.add_argument("--candidate-path", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--current-path", default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--search-control-config", default=DEFAULT_SEARCH_CONTROL_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path, dry_run: bool, log_path: Path) -> None:
    if dry_run:
        write_json(log_path, {"command": command, "cwd": str(cwd), "dry_run": True})
        return
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    write_json(
        log_path,
        {
            "command": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def artifact_decision(path: Path) -> dict:
    payload = load_json(path)
    return {
        "schema": payload.get("schema"),
        "classification": (payload.get("classification") or {}).get("classification"),
        "decision": payload.get("decision"),
    }


def reference_rows(reference_artifact: dict) -> dict[str, dict]:
    rows = reference_artifact.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("reference artifact must contain rows list")
    return {str(row["id"]): row for row in rows}


def arbitration_probe_row(reference_row: dict) -> dict:
    return {
        "id": str(reference_row["id"]),
        "canonical_state": str(reference_row["canonical_state"]),
        "legal_moves": [int(child["move"]) for child in list(reference_row["child_stats"])],
        "reference_move": int(reference_row["reference_move"]),
        "state": dict(reference_row["state"]),
    }


def arbitration_rows(*, candidate_path: Path, reference_artifact_path: Path) -> dict[str, dict]:
    resolved_reference_rows = reference_rows(load_json(reference_artifact_path))
    rows = {}
    for row_id in ROW_IDS:
        probe_row = arbitration_probe_row(resolved_reference_rows[row_id])
        state = validated_diagnostic_state(row=probe_row)
        probe_views = {}
        for probe_key, ablation_mode in (
            ("policy_only", "policy_only"),
            ("value_only", "value_only"),
            ("full_search", "full"),
        ):
            summary = probe_artifact_position(
                artifact_path=str(candidate_path),
                state=state,
                simulations=SIMULATIONS,
                seed=SEED,
                c_puct=C_PUCT,
                search_options=dict(SEARCH_OPTIONS),
                ablation_mode=ablation_mode,
            )
            probe_views[probe_key] = build_row_views(row=probe_row, probe_summary=summary)
        rows[row_id] = {
            **probe_views["full_search"],
            "probe_views": probe_views,
        }
    return rows


def build_arbitration_artifact(
    *,
    candidate_path: Path,
    current_path: Path,
    reference_artifact_path: Path,
    search_control_config: Path,
    out_path: Path,
) -> dict:
    rows = arbitration_rows(
        candidate_path=candidate_path,
        reference_artifact_path=reference_artifact_path,
    )
    payload = build_payload(
        selected_artifact={
            "path": str(candidate_path),
            "selected_target": str(candidate_path),
            "selected_artifact": None,
            "provenance_source": "selection_manifest.selected_target",
        },
        source_artifacts={
            "current_path": str(current_path),
            "reference_artifact": str(reference_artifact_path),
            "search_control_config": str(search_control_config),
        },
        settings={
            "base_config_path": str(search_control_config),
            "row_ids": list(ROW_IDS),
            "search_settings": {"c_puct": C_PUCT, **dict(SEARCH_OPTIONS)},
            "seeds": [SEED, SEED],
            "simulation_count": SIMULATIONS,
        },
        rows=rows,
    )
    write_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    python = python_bin(root)
    candidate_path = resolve_path(root, args.candidate_path)
    current_path = resolve_path(root, args.current_path)
    reference_artifact_path = resolve_path(root, args.reference_artifact)
    search_control_config = resolve_path(root, args.search_control_config)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    arbitration_path = out_root / "capture_002_003_search_control_arbitration.json"
    if args.dry_run:
        write_json(
            arbitration_path,
            {
                "schema": "azlite_capture_002_003_search_policy_arbitration_v1",
                "dry_run": True,
            },
        )
    else:
        build_arbitration_artifact(
            candidate_path=candidate_path,
            current_path=current_path,
            reference_artifact_path=reference_artifact_path,
            search_control_config=search_control_config,
            out_path=arbitration_path,
        )

    paths = {
        "arbitration": str(arbitration_path),
        "shared_full_search_drift": str(out_root / "shared_full_search_drift.json"),
        "selection_score_trace": str(out_root / "capture_002_selection_score_trace.json"),
        "selection_score_trace_relaxed": str(out_root / "capture_002_selection_score_trace_relaxed.json"),
        "trace_capture": str(out_root / "capture_002_trace_capture.json"),
    }

    initial_commands = [
        (
            "shared_full_search_drift",
            [
                python,
                "-m",
                "ml.alphazero_lite.shared_full_search_drift_diagnostic",
                "--source-arbitration-artifact",
                paths["arbitration"],
                "--out",
                paths["shared_full_search_drift"],
            ],
        ),
        (
            "selection_score_trace_default",
            [
                python,
                "-m",
                "ml.alphazero_lite.capture_002_selection_score_trace",
                "--source-shared-drift-artifact",
                paths["shared_full_search_drift"],
                "--out",
                paths["selection_score_trace"],
            ],
        ),
        (
            "selection_score_trace_relaxed",
            [
                python,
                "-m",
                "ml.alphazero_lite.capture_002_selection_score_trace",
                "--source-shared-drift-artifact",
                paths["shared_full_search_drift"],
                "--material-visit-share-margin",
                "0.04",
                "--out",
                paths["selection_score_trace_relaxed"],
            ],
        ),
        (
            "trace_capture",
            [
                python,
                "-m",
                "ml.alphazero_lite.capture_002_trace_capture",
                "--source-shared-drift-artifact",
                paths["shared_full_search_drift"],
                "--capture-mode",
                "extract_then_rerun",
                "--out",
                paths["trace_capture"],
            ],
        ),
    ]

    for name, command in initial_commands:
        run_command(command, cwd=root, dry_run=args.dry_run, log_path=out_root / f"{name}.log.json")

    stopped_early = False
    stop_reason = None
    decisions = {} if not args.dry_run else None

    if not args.dry_run:
        decisions.update(
            {
                key: artifact_decision(Path(path))
                for key, path in paths.items()
                if key != "arbitration"
            }
        )
        if (
            decisions["selection_score_trace"]["decision"]
            != "write_002_unresolved_trace_review_spec"
        ):
            stopped_early = True
            stop_reason = "selection_score_trace_not_unresolved"
        elif (
            decisions["selection_score_trace_relaxed"]["decision"]
            != "write_002_unresolved_trace_review_spec"
        ):
            stopped_early = True
            stop_reason = "selection_score_trace_relaxed_not_unresolved"

    if not stopped_early:
        downstream_paths = {
            "trace_cadence_review": str(out_root / "capture_002_trace_cadence_review.json"),
            "nonseparable_review": str(out_root / "capture_002_nonseparable_review.json"),
            "trace_checkpoint_canonicalization": str(
                out_root / "capture_002_trace_checkpoint_canonicalization.json"
            ),
            "nonseparable_decomposition": str(
                out_root / "capture_002_nonseparable_decomposition.json"
            ),
            "metric_co_movement_audit": str(
                out_root / "capture_002_metric_co_movement_audit.json"
            ),
            "selection_score_component_audit": str(
                out_root / "capture_002_selection_score_component_audit.json"
            ),
            "prior_pressure_component_audit": str(
                out_root / "capture_002_prior_pressure_component_audit.json"
            ),
        }
        paths.update(downstream_paths)

        downstream_commands = [
            (
                "trace_cadence_review",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_trace_cadence_review",
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-trace-capture-artifact",
                    paths["trace_capture"],
                    "--out",
                    paths["trace_cadence_review"],
                ],
            ),
            (
                "nonseparable_review",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_nonseparable_review",
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-trace-cadence-review-artifact",
                    paths["trace_cadence_review"],
                    "--source-threshold-review-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--out",
                    paths["nonseparable_review"],
                ],
            ),
            (
                "trace_checkpoint_canonicalization",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_trace_checkpoint_canonicalization",
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-threshold-relaxed-selection-score-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--out",
                    paths["trace_checkpoint_canonicalization"],
                ],
            ),
            (
                "nonseparable_decomposition",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_nonseparable_decomposition",
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-threshold-review-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--source-trace-cadence-review-artifact",
                    paths["trace_cadence_review"],
                    "--source-nonseparable-review-artifact",
                    paths["nonseparable_review"],
                    "--out",
                    paths["nonseparable_decomposition"],
                ],
            ),
            (
                "metric_co_movement_audit",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_metric_co_movement_audit",
                    "--source-decomposition-artifact",
                    paths["nonseparable_decomposition"],
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-threshold-relaxed-selection-score-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--source-checkpoint-canonicalization-artifact",
                    paths["trace_checkpoint_canonicalization"],
                    "--out",
                    paths["metric_co_movement_audit"],
                ],
            ),
            (
                "selection_score_component_audit",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_selection_score_component_audit",
                    "--source-metric-audit-artifact",
                    paths["metric_co_movement_audit"],
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-threshold-relaxed-selection-score-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--source-checkpoint-canonicalization-artifact",
                    paths["trace_checkpoint_canonicalization"],
                    "--out",
                    paths["selection_score_component_audit"],
                ],
            ),
            (
                "prior_pressure_component_audit",
                [
                    python,
                    "-m",
                    "ml.alphazero_lite.capture_002_prior_pressure_component_audit",
                    "--source-selection-score-component-audit-artifact",
                    paths["selection_score_component_audit"],
                    "--source-metric-audit-artifact",
                    paths["metric_co_movement_audit"],
                    "--source-selection-score-artifact",
                    paths["selection_score_trace"],
                    "--source-threshold-relaxed-selection-score-artifact",
                    paths["selection_score_trace_relaxed"],
                    "--source-checkpoint-canonicalization-artifact",
                    paths["trace_checkpoint_canonicalization"],
                    "--out",
                    paths["prior_pressure_component_audit"],
                ],
            ),
        ]

        for name, command in downstream_commands:
            run_command(
                command, cwd=root, dry_run=args.dry_run, log_path=out_root / f"{name}.log.json"
            )

        if not args.dry_run:
            decisions.update(
                {
                    key: artifact_decision(Path(path))
                    for key, path in downstream_paths.items()
                }
            )

    summary = {
        "schema": "azlite_capture_002_prior_pressure_from_search_control_v1",
        "run_id": args.run_id,
        "candidate_path": str(candidate_path),
        "current_path": str(current_path),
        "reference_artifact": str(reference_artifact_path),
        "search_control_config": str(search_control_config),
        "search_options": dict(SEARCH_OPTIONS),
        "paths": dict(paths),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        summary["decisions"] = decisions

    summary_path = out_root / "capture_002_prior_pressure_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "dry_run": bool(args.dry_run)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
