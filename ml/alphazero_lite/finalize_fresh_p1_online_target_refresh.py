#!/usr/bin/env python3
"""Publish the completed PR #234 online-target-refresh experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))


def _checkpoint_rows(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Fit | L1 | JS | Top-1 | Movement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, details in summary["lanes"].items():
        for step in ("1", "4", "16"):
            metric = details["metrics"][step]
            rows.append(
                f"| {lane} | {step} | {metric['ce_search']:.6f} | "
                f"{metric['ce_p1_policy']:.6f} | {metric['ce_beta095']:.6f} | "
                f"{metric['fit_fraction']:.4f} | {metric['legal_policy_l1_vs_p1']:.6f} | "
                f"{metric['js_vs_p1']:.2e} | {metric['top1_disagreement']:.4f} | "
                f"{metric['movement_from_a16']:.6f} |"
            )
    return rows


def main() -> None:
    workdir = Path("/tmp/azlite_online_target_refresh_rerun")
    summary = json.loads((workdir / "summary.json").read_text())
    evaluation = json.loads((workdir / "evaluation.json").read_text())
    if not all(summary["invariants"].values()):
        raise RuntimeError("cannot publish an invariant-failed experiment")
    if any(
        row["weighted_ratio"] > 2.0
        for lane in summary["lanes"].values()
        for row in lane["gradient_telemetry"]
    ):
        raise RuntimeError("cannot publish a runtime-gradient-scale failure")

    classification = "dynamic_targets_are_effectively_static"
    follow_up = "Do not pursue selector refresh merely to force a positive result."
    combined = {
        **summary,
        **evaluation,
        "schema": "azlite_online_target_refresh_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    output = REPO_ROOT / "docs/data/alphazero-lite-online-target-refresh-summary.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")

    shadow_step16 = summary["lanes"]["online_shadow"]["gradient_telemetry"][-1]
    ordinary_step16 = summary["lanes"]["online_ordinary"]["gradient_telemetry"][-1]
    report = "\n".join(
        [
            "# Online Current-Candidate Search-Target Refresh",
            "",
            f"**Classification:** `{classification}`",
            "",
            f"**Recommended follow-up:** {follow_up}",
            "",
            "## Invariants And Calibration",
            "",
            "- Immutable P1, A16, replay, selector, and static-cache hashes matched PR #233.",
            "- In-memory A16 ordinary and shadow PUCT reproduced the cached targets.",
            "- Targets were generated from and fingerprinted against the pre-update candidate.",
            f"- Frozen global auxiliary weight: `{summary['calibration']['behavior_loss_weight']:.17g}`; maximum prospective raw ratio: `{summary['calibration']['maximum_raw_ratio']:.8f}`.",
            "- No runtime weighted auxiliary/primary ratio exceeded 2.0.",
            "",
            "## Checkpoint Metrics",
            "",
            *_checkpoint_rows(summary),
            "",
            "## Target Drift",
            "",
            f"At online-shadow step 16, drift from the PR #233 static shadow cache was mean L1 `{shadow_step16['target_drift_from_static_shadow']['l1']['mean']:.8f}` and mean JS `{shadow_step16['target_drift_from_static_shadow']['mean_js']:.3e}` with zero top-1 disagreement.",
            f"At online-ordinary step 16, drift from static shadow was mean L1 `{ordinary_step16['target_drift_from_static_shadow']['l1']['mean']:.8f}` and mean JS `{ordinary_step16['target_drift_from_static_shadow']['mean_js']:.3e}`. Online-shadow versus online-ordinary JS was `{shadow_step16['shadow_delta_from_ordinary']:.3e}`.",
            "",
            "## Ordinary-PUCT Evaluation",
            "",
            "All meaningful checkpoints were evaluated with ordinary PUCT only. At step 16, every lane had frozen rescue `0.30`, new-divergence `0.0`, and the same unsafe 1200:1200 paired effect `-0.01953125` with 95% CI `[-0.0390625, -0.00390625]`.",
            "",
            "No checkpoint was safe at both budgets; therefore no P0 gate was eligible.",
            "",
            "## Interpretation",
            "",
            "Refreshing the shadow target against the current candidate produced only extremely small shadow-target movement and did not alter ordinary-PUCT strength or frozen-root rescue relative to the static lane. The primary online-shadow comparison therefore does not support target staleness as the failure mechanism.",
            "",
        ]
    )
    (REPO_ROOT / "docs/alphazero-lite-online-target-refresh-results.md").write_text(
        report
    )


if __name__ == "__main__":
    main()
