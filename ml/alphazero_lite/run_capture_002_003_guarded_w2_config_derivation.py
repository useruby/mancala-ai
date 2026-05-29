#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_ROOT_PRIOR_SUMMARY = (
    "/tmp/azlite_capture_002_root_prior_intervention/"
    "capture-002-root-prior-intervention/root_prior_intervention_summary.json"
)
DEFAULT_GUARDED_W2_GATE = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1/"
    "capture_002_003_rule_conditioned_gate.json"
)
DEFAULT_POLICY_TARGET_GATE = (
    "/tmp/azlite_v3_policy_target_local_versions/"
    "aggressive-v3-policy-target-local-iter1/capture_002_003_local_gate.json"
)
DEFAULT_VALUE_TARGET_ALIGNED_GATE = (
    "/tmp/azlite_v3_value_target_aligned_local_versions/"
    "aggressive-v3-value-target-aligned-local-iter1/capture_002_003_local_gate.json"
)
DEFAULT_POLICY_TARGET_ARENA = (
    "/tmp/azlite_v3_policy_target_local_versions/"
    "aggressive-v3-policy-target-local-iter1/arena_report.json"
)
DEFAULT_VALUE_TARGET_ALIGNED_ARENA = (
    "/tmp/azlite_v3_value_target_aligned_local_versions/"
    "aggressive-v3-value-target-aligned-local-iter1/arena_report.json"
)
DEFAULT_GUARDED_W2_RUNTIME_CONFIG = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/runtime_config.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_003_guarded_w2_config_derivation"
DEFAULT_RUN_ID = "capture-002-003-guarded-w2-config-derivation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root-prior-summary", default=DEFAULT_ROOT_PRIOR_SUMMARY)
    parser.add_argument("--guarded-w2-gate", default=DEFAULT_GUARDED_W2_GATE)
    parser.add_argument("--policy-target-gate", default=DEFAULT_POLICY_TARGET_GATE)
    parser.add_argument(
        "--value-target-aligned-gate", default=DEFAULT_VALUE_TARGET_ALIGNED_GATE
    )
    parser.add_argument("--policy-target-arena", default=DEFAULT_POLICY_TARGET_ARENA)
    parser.add_argument(
        "--value-target-aligned-arena", default=DEFAULT_VALUE_TARGET_ALIGNED_ARENA
    )
    parser.add_argument(
        "--guarded-w2-runtime-config", default=DEFAULT_GUARDED_W2_RUNTIME_CONFIG
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_markdown(summary: dict) -> str:
    payload = summary["artifact"]
    delta = payload["recommended_runtime_config_delta"]
    lines = [
        "# Guarded w2 Prior-Calibration Config Derivation",
        "",
        "## Outcome",
        "",
        "- classification: `guarded_w2_is_best_supported_base`",
        f"- decision: `{payload['decision']}`",
        "- recommended action: keep the guarded `w2` runtime config as the base and write a narrow prior-calibration spec instead of launching another broad sharpened-target lane",
        "",
        "## Inputs",
        "",
    ]
    for key, value in summary["input_artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Comparative Evidence",
            "",
            "- guarded `w2` stayed the best tracked-row base among the compared candidates",
            f"- policy-target local arena score: `{payload['comparative_evidence']['policy_target_local']['arena_score']}`",
            f"- value-target-aligned local arena score: `{payload['comparative_evidence']['value_target_aligned_local']['arena_score']}`",
            f"- guarded `w2` row `002` reference visit share: `{payload['comparative_evidence']['guarded_w2']['row_002']['reference_move_visit_share']}`",
            f"- policy-target local row `002` reference visit share: `{payload['comparative_evidence']['policy_target_local']['row_002']['reference_move_visit_share']}`",
            f"- value-target-aligned local row `002` reference visit share: `{payload['comparative_evidence']['value_target_aligned_local']['row_002']['reference_move_visit_share']}`",
            f"- guarded `w2` row `003` selected move: `{payload['comparative_evidence']['guarded_w2']['row_003']['searched_selected_move']}`",
            f"- policy-target local row `003` selected move: `{payload['comparative_evidence']['policy_target_local']['row_003']['searched_selected_move']}`",
            f"- value-target-aligned local row `003` selected move: `{payload['comparative_evidence']['value_target_aligned_local']['row_003']['searched_selected_move']}`",
            "",
            "## Recommended Base",
            "",
            f"- base runtime config: `{delta['base_runtime_config']}`",
            "- retain:",
        ]
    )
    for item in delta["retain"]:
        lines.append(f"  - {item}")
    lines.extend(["", "## Constraints", ""])
    for item in delta["recommended_constraints"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Avoid", ""])
    for item in delta["explicit_non_recommendations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_paths = {
        "root_prior_summary": resolve_path(root, args.root_prior_summary),
        "guarded_w2_gate": resolve_path(root, args.guarded_w2_gate),
        "policy_target_gate": resolve_path(root, args.policy_target_gate),
        "value_target_aligned_gate": resolve_path(root, args.value_target_aligned_gate),
        "policy_target_arena": resolve_path(root, args.policy_target_arena),
        "value_target_aligned_arena": resolve_path(
            root, args.value_target_aligned_arena
        ),
        "guarded_w2_runtime_config": resolve_path(root, args.guarded_w2_runtime_config),
    }

    artifact_path = out_root / "capture_002_003_guarded_w2_config_derivation.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.capture_002_003_guarded_w2_config_derivation",
            "--root-prior-summary",
            str(resolved_paths["root_prior_summary"]),
            "--guarded-w2-gate",
            str(resolved_paths["guarded_w2_gate"]),
            "--policy-target-gate",
            str(resolved_paths["policy_target_gate"]),
            "--value-target-aligned-gate",
            str(resolved_paths["value_target_aligned_gate"]),
            "--policy-target-arena",
            str(resolved_paths["policy_target_arena"]),
            "--value-target-aligned-arena",
            str(resolved_paths["value_target_aligned_arena"]),
            "--guarded-w2-runtime-config",
            str(resolved_paths["guarded_w2_runtime_config"]),
            "--out",
            str(artifact_path),
        ],
        cwd=root,
        check=True,
        capture_output=False,
        text=True,
    )

    artifact = load_json(artifact_path)
    summary = {
        "run_id": args.run_id,
        "artifact_path": str(artifact_path),
        "schema": artifact.get("schema"),
        "classification": artifact.get("classification"),
        "decision": artifact.get("decision"),
        "input_artifacts": {key: str(path) for key, path in resolved_paths.items()},
        "artifact": artifact,
    }
    summary_path = (
        out_root / "capture_002_003_guarded_w2_config_derivation_summary.json"
    )
    write_json(summary_path, summary)

    report_path = (
        root / "docs/alphazero-lite-capture-002-003-guarded-w2-config-derivation.md"
    )
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "decision": artifact.get("decision"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
