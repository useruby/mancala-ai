#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_CAPTURE_ARTIFACT = (
    "/tmp/azlite_learned_policy_vs_root_corrected_prior_capture/"
    "learned-policy-vs-root-corrected-prior-capture/"
    "learned_policy_vs_root_corrected_prior_capture.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_learned_policy_vs_root_corrected_prior_review_spec"
DEFAULT_RUN_ID = "learned-policy-vs-root-corrected-prior-review-spec"
SCHEMA = "azlite_learned_policy_vs_root_corrected_prior_review_spec_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--capture-artifact", default=DEFAULT_CAPTURE_ARTIFACT)
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


def build_payload(capture_artifact: dict, *, capture_artifact_path: str) -> dict:
    classification = capture_artifact.get("classification")
    decision = capture_artifact.get("decision")
    if (
        classification
        != "learned_prior_underweights_reference_and_overweights_wrong_extra_turn"
    ):
        raise ValueError("capture artifact must isolate the learned-prior mismatch")
    if decision != "write_learned_policy_vs_root_corrected_prior_review_spec":
        raise ValueError("capture artifact must request the learned-policy review spec")

    return {
        "schema": SCHEMA,
        "hypothesis": "learned_policy_vs_root_corrected_prior_review_spec",
        "decision": "write_learned_policy_vs_root_corrected_prior_mismatch_capture_spec",
        "input_artifacts": {
            "capture_artifact_path": capture_artifact_path,
        },
        "source_snapshot": {
            "capture_classification": classification,
            "capture_decision": decision,
        },
        "recommended_next_branch": {
            "name": "learned_policy_vs_root_corrected_prior_mismatch_capture",
            "kind": "non_training_diagnostic",
            "goal": "capture exactly how the learned guarded-w2 raw policy differs from the root-corrected prior distributions that flip row 002 while preserving row 003",
        },
        "required_focus": {
            "focus_row_id": capture_artifact.get("focus_row_id"),
            "preservation_row_id": capture_artifact.get("preservation_row_id"),
            "artifact_path": capture_artifact.get("artifact_path"),
            "interventions": [
                "original_prior",
                "zero_wrong_extra_turn_prior",
                "swap_reference_and_wrong",
            ],
        },
        "required_outputs": [
            "one artifact that compares learned raw policy probabilities against root-corrected prior distributions for every legal move on row 002",
            "one ranking comparison showing how move order changes between learned policy, root-corrected prior, early selection score, and final visit share on row 002",
            "one preservation comparison confirming the same corrections do not destabilize row 003",
            "one markdown note stating whether the learned mismatch is best described as reference underweighting, wrong extra-turn overweighting, or both",
        ],
        "constraints": [
            "Do not launch another training lane in this branch",
            "Do not change architecture, replay composition, target modes, or search defaults",
            "Keep the guarded w2 artifact explicit in every diagnostic command",
            "Use the exact root-correction interventions that already flipped row 002 under aligned search-control settings",
        ],
    }


def build_markdown(spec_payload: dict) -> str:
    branch = spec_payload["recommended_next_branch"]
    focus = spec_payload["required_focus"]
    lines = [
        "# AlphaZero-lite Learned Policy vs Root-Corrected Prior Review Spec",
        "",
        "## Goal",
        "",
        "Diagnose exactly how the learned guarded-`w2` policy on row `002` differs from the root-corrected prior distributions that flip the row while preserving row `003`.",
        "",
        "This is a non-training diagnostic branch.",
        "",
        "## Trigger",
        "",
        "The focused learned-policy capture concluded:",
        "",
        "- classification: `learned_prior_underweights_reference_and_overweights_wrong_extra_turn`",
        "- decision: `write_learned_policy_vs_root_corrected_prior_review_spec`",
        "",
        "Supporting artifact:",
        "",
        f"- `{spec_payload['input_artifacts']['capture_artifact_path']}`",
        "",
        "## Exact Next Branch",
        "",
        f"- branch name: `{branch['name']}`",
        f"- branch type: `{branch['kind']}`",
        f"- artifact path: `{focus['artifact_path']}`",
        f"- focus row: `{focus['focus_row_id']}`",
        f"- preservation row: `{focus['preservation_row_id']}`",
        f"- required interventions: `{', '.join(focus['interventions'])}`",
        "",
        "## Required Deliverables",
        "",
    ]
    for item in spec_payload["required_outputs"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Constraints",
        "",
    ])
    for item in spec_payload["constraints"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Exit Criteria",
        "",
        "This branch is complete when one diagnostic artifact and note make the learned-policy mismatch explicit enough to explain why row `002` needs root correction while row `003` does not, or explicitly state what missing telemetry remains.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    capture_artifact_path = resolve_path(root, args.capture_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        load_json(capture_artifact_path),
        capture_artifact_path=str(capture_artifact_path),
    )
    summary_path = out_root / "learned_policy_vs_root_corrected_prior_review_spec.json"
    write_json(summary_path, payload)

    report_path = root / "docs/alphazero-lite-learned-policy-vs-root-corrected-prior-review-spec.md"
    report_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
