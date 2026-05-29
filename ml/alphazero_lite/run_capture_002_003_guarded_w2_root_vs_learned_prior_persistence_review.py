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
DEFAULT_GUARDED_W2_TRAIN_LOG = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1/train.log"
)
DEFAULT_PRIOR_CALIBRATION_GATE = (
    "/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1/"
    "capture_002_003_rule_conditioned_gate.json"
)
DEFAULT_PRIOR_CALIBRATION_ARENA = (
    "/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1/"
    "arena_report.json"
)
DEFAULT_PRIOR_CALIBRATION_TRAIN_LOG = (
    "/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1/"
    "train.log"
)
DEFAULT_OUTPUT_ROOT = (
    "/tmp/azlite_capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review"
)
DEFAULT_RUN_ID = "capture-002-003-guarded-w2-root-vs-learned-prior-persistence-review"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root-prior-summary", default=DEFAULT_ROOT_PRIOR_SUMMARY)
    parser.add_argument("--guarded-w2-gate", default=DEFAULT_GUARDED_W2_GATE)
    parser.add_argument("--prior-calibration-gate", default=DEFAULT_PRIOR_CALIBRATION_GATE)
    parser.add_argument("--prior-calibration-arena", default=DEFAULT_PRIOR_CALIBRATION_ARENA)
    parser.add_argument("--guarded-w2-train-log", default=DEFAULT_GUARDED_W2_TRAIN_LOG)
    parser.add_argument(
        "--prior-calibration-train-log", default=DEFAULT_PRIOR_CALIBRATION_TRAIN_LOG
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
    artifact = summary["artifact"]
    learned = artifact["learned_retry_review"]
    comparisons = learned["comparisons"]
    root_review = artifact["root_override_review"]
    lines = [
        "# Guarded w2 Root-vs-Learned Prior Persistence Review",
        "",
        "## Outcome",
        "",
        f"- classification: `{artifact['classification']['classification']}`",
        f"- decision: `{artifact['decision']}`",
        "- conclusion: root-only prior overrides were strong enough to flip guarded `w2` row `002`, but the learned prior-calibration retry did not persist that flip through final search selection and collapsed arena",
        "",
        "## Root Override Review",
        "",
        f"- persistent root interventions at required budgets `{root_review['required_budgets']}`: `{', '.join(root_review['persistent_root_interventions'])}`",
        "",
        "## Learned Retry Review",
        "",
        f"- row `002` prior improved: `{comparisons['row_002_prior_improved']}`",
        f"- row `002` visit share improved: `{comparisons['row_002_visit_share_improved']}`",
        f"- row `002` selected move still not fixed: `{comparisons['row_002_selection_still_not_fixed']}`",
        f"- row `002` Q margin worsened: `{comparisons['row_002_q_margin_worsened']}`",
        f"- arena collapsed: `{comparisons['arena_collapsed']}`",
        f"- train best-val-loss delta: `{comparisons['best_val_loss_delta']}`",
        "",
        "## Recommendation",
        "",
        "- do not retry replay-side prior reweighting on guarded `w2`",
        "- next branch should stay diagnostic and explain why root-only prior correction does not persist through the learned policy/search stack",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_paths = {
        "root_prior_summary": resolve_path(root, args.root_prior_summary),
        "guarded_w2_gate": resolve_path(root, args.guarded_w2_gate),
        "prior_calibration_gate": resolve_path(root, args.prior_calibration_gate),
        "prior_calibration_arena": resolve_path(root, args.prior_calibration_arena),
        "guarded_w2_train_log": resolve_path(root, args.guarded_w2_train_log),
        "prior_calibration_train_log": resolve_path(root, args.prior_calibration_train_log),
    }

    artifact_path = (
        out_root
        / "capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review.json"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review",
            "--root-prior-summary",
            str(resolved_paths["root_prior_summary"]),
            "--guarded-w2-gate",
            str(resolved_paths["guarded_w2_gate"]),
            "--prior-calibration-gate",
            str(resolved_paths["prior_calibration_gate"]),
            "--prior-calibration-arena",
            str(resolved_paths["prior_calibration_arena"]),
            "--guarded-w2-train-log",
            str(resolved_paths["guarded_w2_train_log"]),
            "--prior-calibration-train-log",
            str(resolved_paths["prior_calibration_train_log"]),
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
        out_root
        / "capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review_summary.json"
    )
    write_json(summary_path, summary)

    report_path = root / "docs/alphazero-lite-guarded-w2-root-vs-learned-prior-persistence-review.md"
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
