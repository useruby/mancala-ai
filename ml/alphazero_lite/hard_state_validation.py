#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def ArtifactEvaluator(*args, **kwargs):
    from ml.alphazero_lite.arena import ArtifactEvaluator as impl

    return impl(*args, **kwargs)


def build_eval_search_options(*args, **kwargs):
    from ml.alphazero_lite.arena import build_eval_search_options as impl

    return impl(*args, **kwargs)


def evaluate_artifact_position(*args, **kwargs):
    from ml.alphazero_lite.arena import evaluate_artifact_position as impl

    return impl(*args, **kwargs)


def load_suite(*args, **kwargs):
    from ml.alphazero_lite.forensic_suite import load_suite as impl

    return impl(*args, **kwargs)


def summarize_system(*args, **kwargs):
    from ml.alphazero_lite.forensic_suite import summarize_system as impl

    return impl(*args, **kwargs)


def summarize_bucket_matrix(*args, **kwargs):
    from ml.alphazero_lite.forensic_suite import summarize_bucket_matrix as impl

    return impl(*args, **kwargs)


def build_row(*args, **kwargs):
    from ml.alphazero_lite.run_forensic_suite import build_row as impl

    return impl(*args, **kwargs)


def run_reference(*args, **kwargs):
    from ml.alphazero_lite.run_forensic_suite import run_reference as impl

    return impl(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--teacher-simulations", type=int, default=1200)
    parser.add_argument("--artifact-simulations", type=int, default=384)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-prior-transform", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    validation_path = Path(args.validation_path)
    out_path = Path(args.out)

    if not validation_path.exists():
        raise SystemExit("validation path does not exist")

    suite = load_suite(validation_path)
    search_options = build_eval_search_options()
    evaluator = ArtifactEvaluator(artifact_path)

    rows: list[dict] = []
    for index, position in enumerate(suite):
        reference = run_reference(
            position.state,
            int(args.teacher_simulations),
            int(args.teacher_simulations),
            int(args.seed) + index,
            index,
        )
        system = evaluate_artifact_position(
            artifact_path=artifact_path,
            evaluator=evaluator,
            state=position.state,
            simulations=int(args.artifact_simulations),
            seed=int(args.seed) + 1000 + index,
            c_puct=float(args.c_puct),
            search_options=search_options,
            root_prior_transform=args.root_prior_transform,
        )
        rows.append(
            build_row(
                position=position,
                reference=reference,
                system=system,
            )
        )

    summary = summarize_system(rows)
    bucket_matrix = summarize_bucket_matrix({"artifact": rows})
    overall = summary["overall"]
    report = {
        "schema": "azlite_hard_state_validation_v1",
        "artifact_path": str(artifact_path),
        "validation_path": str(validation_path),
        "root_prior_transform": args.root_prior_transform,
        "position_count": len(suite),
        "policy_top1_agreement": overall["top1_agreement"],
        "average_regret": overall["average_regret"],
        "value_calibration_mae": overall["value_calibration_mae"],
        "overall": overall,
        "buckets": bucket_matrix,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote hard-state validation report to {out_path}")


if __name__ == "__main__":
    main()
