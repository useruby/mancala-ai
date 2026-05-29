#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_REVIEW_ARTIFACT = (
    "/tmp/azlite_capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review/"
    "capture-002-003-guarded-w2-root-vs-learned-prior-persistence-review/"
    "capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_guarded_w2_root_vs_learned_prior_persistence_spec"
DEFAULT_RUN_ID = "guarded-w2-root-vs-learned-prior-persistence-spec"
SCHEMA = "azlite_guarded_w2_root_vs_learned_prior_persistence_spec_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--review-artifact", default=DEFAULT_REVIEW_ARTIFACT)
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


def build_payload(review_artifact: dict, *, review_artifact_path: str) -> dict:
    classification = (review_artifact.get("classification") or {}).get("classification")
    decision = review_artifact.get("decision")
    if classification != "root_override_effective_but_training_nonpersistent":
        raise ValueError(
            "review artifact must classify as root_override_effective_but_training_nonpersistent"
        )
    if decision != "write_guarded_w2_root_vs_learned_prior_persistence_spec":
        raise ValueError("review artifact must request the persistence spec")

    root_review = review_artifact["root_override_review"]
    learned_review = review_artifact["learned_retry_review"]
    persistent_root_interventions = root_review["persistent_root_interventions"]

    return {
        "schema": SCHEMA,
        "hypothesis": "guarded_w2_root_vs_learned_prior_persistence_spec",
        "decision": "write_guarded_w2_root_vs_learned_prior_persistence_capture_spec",
        "input_artifacts": {
            "review_artifact_path": review_artifact_path,
        },
        "source_snapshot": {
            "review_classification": review_artifact.get("classification"),
            "review_decision": review_artifact.get("decision"),
        },
        "recommended_next_branch": {
            "kind": "non_training_diagnostic",
            "name": "guarded_w2_root_vs_learned_prior_persistence_capture",
            "goal": "capture why root-only prior correction can flip row 002 while the learned guarded-w2 policy/search stack cannot retain that correction through final selection",
        },
        "required_focus": {
            "rows": ["capture_available-002", "capture_available-003"],
            "artifact_base": review_artifact["source_snapshot"][
                "guarded_w2_root_prior_artifact"
            ]["artifact_path"],
            "persistent_root_interventions": persistent_root_interventions,
        },
        "required_outputs": [
            "one side-by-side artifact that compares original learned priors, root-overridden priors, and final visit/Q evolution on row 002",
            "one paired preservation readout showing why row 003 remains stable under root overrides",
            "one markdown note stating whether the persistence gap is best explained by backup dynamics, non-root policy mismatch, or another search-stage effect",
        ],
        "constraints": [
            "Do not launch another training lane in this branch",
            "Do not change architecture, replay composition, or target modes",
            "Keep the guarded w2 artifact explicit in every evaluation command",
            "Compare at least the original prior and the persistent root interventions that held through budgets 384 and 1200",
        ],
        "learned_retry_snapshot": learned_review,
    }


def build_markdown(spec_payload: dict) -> str:
    branch = spec_payload["recommended_next_branch"]
    focus = spec_payload["required_focus"]
    lines = [
        "# AlphaZero-lite Guarded w2 Root-vs-Learned Prior Persistence Spec",
        "",
        "## Goal",
        "",
        "Diagnose why root-only prior correction can flip guarded `w2` row `002`, but the learned guarded-`w2` policy/search stack does not retain that correction through final selection.",
        "",
        "This is a non-training diagnostic branch.",
        "",
        "## Trigger",
        "",
        "The guarded `w2` root-vs-learned persistence review concluded:",
        "",
        "- classification: `root_override_effective_but_training_nonpersistent`",
        "- decision: `write_guarded_w2_root_vs_learned_prior_persistence_spec`",
        "",
        "Supporting artifact:",
        "",
        f"- `{spec_payload['input_artifacts']['review_artifact_path']}`",
        "",
        "## Problem Statement",
        "",
        "The guarded `w2` base is responsive to root-only prior correction, but the corresponding learned prior-calibration retry only lifts row `002` prior support slightly and still search-selects move `2`.",
        "",
        'That means the current gap is no longer just "does prior matter".',
        "",
        "It is now:\n",
        "why does the root-only corrected prior produce the desired selection, while the learned policy/search stack does not preserve that correction end to end?",
        "",
        "## Exact Next Branch",
        "",
        f"- branch name: `{branch['name']}`",
        f"- branch type: `{branch['kind']}`",
        f"- base artifact: `{focus['artifact_base']}`",
        f"- tracked rows: `{', '.join(focus['rows'])}`",
        f"- required root interventions: `{', '.join(focus['persistent_root_interventions'])}`",
        "",
        "## Required Deliverables",
        "",
    ]
    for item in spec_payload["required_outputs"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Concrete Questions",
            "",
            "1. On row `002`, where does the root-only corrected trajectory diverge from the learned-policy trajectory: pre-expansion prior, early visit accumulation, child Q ranking, or later backup/selection score evolution?",
            "2. Does the learned guarded-`w2` policy understate only the reference move, or does it also overstate the wrong extra-turn move in a way that root correction temporarily masks?",
            "3. Why does row `003` remain preserved under the persistent root interventions while row `002` does not persist under learned prior pressure?",
            "",
            "## Constraints",
            "",
        ]
    )
    for item in spec_payload["constraints"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Exit Criteria",
            "",
            "This branch is complete when one diagnostic artifact and note explain where the persistence gap appears between root-only corrected search and the learned guarded-`w2` policy/search stack, or explicitly state what missing telemetry is still required.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    review_artifact_path = resolve_path(root, args.review_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    payload = build_payload(
        load_json(review_artifact_path),
        review_artifact_path=str(review_artifact_path),
    )
    summary_path = out_root / "guarded_w2_root_vs_learned_prior_persistence_spec.json"
    write_json(summary_path, payload)

    report_path = (
        root
        / "docs/alphazero-lite-guarded-w2-root-vs-learned-prior-persistence-spec.md"
    )
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
