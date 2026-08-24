#!/usr/bin/env python3
# ruff: noqa: E402
"""Run held-out ordinary-PUCT evaluation for completed online-refresh checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_online_target_refresh_rerun")
    )
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    summary_path = args.workdir / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = read_jsonl(pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl")
    _suite, suite_hash = pr233._suite()
    p1 = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    p0 = REPO_ROOT / "model-artifact/current"
    frozen: dict[str, Any] = {}
    arenas: dict[str, Any] = {}
    p0_gate: dict[str, Any] = {}
    for lane, details in summary["lanes"].items():
        for step, metric in details["metrics"].items():
            if metric["fit_fraction"] < 0.25:
                continue
            key = f"{lane}:{step}"
            candidate = Path(details["artifacts"][step])
            # Both functions use ordinary PUCT only; shadow search is never
            # available on any candidate-evaluation path.
            frozen[key] = pr233._frozen_diagnostic(candidate, p1, rows)
            arenas[key] = {
                context: pr233._arena(
                    candidate,
                    p1,
                    context,
                    args.workdir / "evaluation" / key,
                    "candidate_vs_p1",
                    args.workers,
                    suite_hash,
                )
                for context in ("384:256", "1200:1200")
            }
            if all(value["safe"] for value in arenas[key].values()):
                p0_gate[key] = {
                    context: pr233._arena(
                        candidate,
                        p0,
                        context,
                        args.workdir / "evaluation" / key,
                        "candidate_vs_p0",
                        args.workers,
                        suite_hash,
                    )
                    for context in ("384:256", "1200:1200")
                }
    evaluation = {
        "schema": "azlite_online_target_refresh_evaluation_v1",
        "suite_sha256": suite_hash,
        "frozen_diagnostics": frozen,
        "arena_matrix": arenas,
        "p0_gate": p0_gate,
        "ordinary_puct_only": True,
    }
    (args.workdir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
    )
    print("evaluation_complete")


if __name__ == "__main__":
    main()
