#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_ARTIFACT_PATH = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_learned_policy_vs_root_corrected_prior_capture"
DEFAULT_RUN_ID = "learned-policy-vs-root-corrected-prior-capture"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--artifact-path", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


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
    artifact = summary["artifact"]
    focus = artifact["focus_row"]["interventions"]
    preservation = artifact["preservation_row"]["interventions"]
    lines = [
        "# Learned Policy vs Root-Corrected Prior Capture",
        "",
        "## Outcome",
        "",
        f"- classification: `{artifact['classification']}`",
        f"- decision: `{artifact['decision']}`",
        "",
        "## Focus Row 002",
        "",
        f"- original selected move: `{focus['original_prior']['searched_selected_move']}`",
        f"- zero-wrong selected move: `{focus['zero_wrong_extra_turn_prior']['searched_selected_move']}`",
        f"- swap selected move: `{focus['swap_reference_and_wrong']['searched_selected_move']}`",
        f"- original policy reference probability: `{focus['original_prior']['policy_reference_probability']}`",
        f"- original policy selected-minus-reference margin: `{focus['original_prior']['policy_selected_minus_reference_margin']}`",
        "",
        "## Preservation Row 003",
        "",
        f"- original selected move: `{preservation['original_prior']['searched_selected_move']}`",
        f"- zero-wrong selected move: `{preservation['zero_wrong_extra_turn_prior']['searched_selected_move']}`",
        f"- swap selected move: `{preservation['swap_reference_and_wrong']['searched_selected_move']}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    python = python_bin(root)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)
    artifact_path = out_root / "learned_policy_vs_root_corrected_prior_capture.json"

    subprocess.run(
        [
            python,
            "-m",
            "ml.alphazero_lite.learned_policy_vs_root_corrected_prior_capture",
            "--artifact-path",
            str(resolve_path(root, args.artifact_path)),
            "--reference-artifact",
            str(resolve_path(root, args.reference_artifact)),
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
        "artifact": artifact,
    }
    summary_path = (
        out_root / "learned_policy_vs_root_corrected_prior_capture_summary.json"
    )
    write_json(summary_path, summary)

    report_path = (
        root / "docs/alphazero-lite-learned-policy-vs-root-corrected-prior-capture.md"
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
