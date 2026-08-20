#!/usr/bin/env python3
"""Fixed-schedule dynamic refresh follow-up to Gen-2 static risk masking.

The initial mask is PR #208's frozen retrospective q75 mask.  After steps 4,
16, and 32, it is replaced by the current candidate-vs-P1 top-25% replay-state
mask.  Mask construction never consumes self-play, arena, or outcome data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    CHECKPOINT_STEPS,
    _win_draw_loss,
    arena_safe,
    export_snapshot_artifacts,
)
from ml.alphazero_lite.run_gen2_hard_trust_region import (  # noqa: E402
    PR204_GEN2_REPLAY_HASH,
    PR204_UNPROJECTED_S46_STATE_HASH,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P1_EXPECTED_STATE_HASH,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_gen2_selective_risk_training import (  # noqa: E402
    build_risk_masks,
    metrics,
    train_dynamic_refresh_lane,
    train_lane,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _new_model  # noqa: E402
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import stable_hash  # noqa: E402
from ml.alphazero_lite.train import load_checkpoint_into_model  # noqa: E402

NAMESPACE = "azlite_gen2_dynamic_risk_refresh_v1"
REFRESH_AFTER_STEPS = (4, 16, 32)


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Gen-2 Dynamic Risk-Mask Refresh Results",
        "",
        f"**Primary classification:** `{summary['classification']['label']}`",
        "",
        f"**Recommended follow-up:** {summary['classification']['next_experiment']}",
        "",
        "## Fixed Contract",
        "",
        "- Initial mask: frozen PR #208 q75 retrospective mask.",
        f"- Refresh after optimizer steps: `{list(REFRESH_AFTER_STEPS)}`.",
        "- Every refresh selects the current candidate-vs-P1 top 25% of unique replay states.",
        "",
        "## Mask Refreshes",
        "",
        "| Boundary | Protected states | Threshold | Mask SHA256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for boundary, entry in summary["refreshes"].items():
        lines.append(
            f"| {boundary} | {entry['protected_unique_states']} | "
            f"{entry.get('selection_threshold', 0.0):.8f} | {entry['mask_sha256']} |"
        )
    lines += [
        "",
        "## Learning And Drift",
        "",
        "| Lane | Step | CE(search) | Fit fraction | Unprotected improvement | Current q75 L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane in ("static_q75", "dynamic_q75"):
        for step in CHECKPOINT_STEPS:
            entry = summary["metrics"][lane][str(step)]
            lines.append(
                f"| {lane} | {step} | {entry['ce_candidate_search']:.6f} | "
                f"{entry['fit_fraction']:.4f} | {entry['search_improvement_unprotected']:.6f} | "
                f"{entry['risk_migration']['current_q75']:.6f} |"
            )
    lines += [
        "",
        "## Arena Vs P1 (384:256)",
        "",
        "| Lane | Step | Effect | 95% CI | Safe |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for lane, steps in summary["arena"].items():
        for step, entry in steps.items():
            ci = entry["opening_bootstrap_ci"]
            lines.append(
                f"| {lane} | {step} | {entry['paired_candidate_effect']:+.4f} | "
                f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {entry['safe']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_gen2_dynamic_refresh")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--gen1-replay",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor/fresh_self_play.jsonl"),
    )
    parser.add_argument(
        "--gen2-replay",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl"),
    )
    parser.add_argument(
        "--p2",
        type=Path,
        default=Path(
            "/tmp/azlite_gen2_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
        ),
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-gen2-dynamic-risk-refresh-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-gen2-dynamic-risk-refresh-results.md",
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    configure_determinism(device, 43)
    p1_artifact, p1_path, _, _, p1_hash = reconstruct_and_freeze_p1(
        REPO_ROOT / "model-artifact/current", args.p1_workdir, args.gen1_replay, 24
    )
    p1, p2 = _new_model(device), _new_model(device)
    load_checkpoint_into_model(p1, p1_path)
    load_checkpoint_into_model(p2, args.p2)
    p1_state, p2_state = p1.state_dict(), p2.state_dict()
    rows = read_jsonl(args.gen2_replay)
    manifest = verify_manifest(
        Path("/tmp/azlite_gen2_selfplay_anchor/training_manifest.json")
    )
    frozen, masks = build_risk_masks(rows, p1_state, p2_state)
    configure_determinism(device, 43)
    beta095 = train_lane(manifest, args.workdir / "beta095", device, p1_path, set())
    configure_determinism(device, 43)
    static = train_lane(
        manifest, args.workdir / "static_q75", device, p1_path, masks["risk_q75"]
    )
    configure_determinism(device, 43)
    dynamic, refreshes = train_dynamic_refresh_lane(
        manifest,
        args.workdir / "dynamic_q75",
        device,
        p1_path,
        masks["risk_q75"],
        REFRESH_AFTER_STEPS,
    )
    artifacts = {
        "static_q75": export_snapshot_artifacts(static, args.workdir / "static_q75"),
        "dynamic_q75": export_snapshot_artifacts(dynamic, args.workdir / "dynamic_q75"),
    }
    baseline = metrics(
        rows,
        beta095,
        p1_state,
        p2_state,
        {
            "beta095": set(),
            "risk_q75": masks["risk_q75"],
            "risk_q90": masks["risk_q90"],
        },
        "beta095",
        0.0,
    )
    denominator = baseline["46"]["ce_candidate_search"]
    metric_masks = {
        "static_q75": masks["risk_q75"],
        "dynamic_q75": masks["risk_q75"],
        "risk_q75": masks["risk_q75"],
        "risk_q90": masks["risk_q90"],
    }
    result_metrics = {
        "static_q75": metrics(
            rows, static, p1_state, p2_state, metric_masks, "static_q75", denominator
        ),
        "dynamic_q75": metrics(
            rows, dynamic, p1_state, p2_state, metric_masks, "dynamic_q75", denominator
        ),
    }
    p2_hash = stable_hash(
        {k: v.detach().cpu().numpy().tobytes().hex() for k, v in p2_state.items()}
    )
    sanity = {
        "p1_hash": p1_hash == P1_EXPECTED_STATE_HASH,
        "p2_hash": p2_hash == PR204_UNPROJECTED_S46_STATE_HASH,
        "replay_hash": sha256_file(args.gen2_replay) == PR204_GEN2_REPLAY_HASH,
        "beta095_baseline_reproduced": stable_hash(
            {
                k: v.detach().cpu().numpy().tobytes().hex()
                for k, v in beta095[46][0].items()
            }
        )
        == PR204_UNPROJECTED_S46_STATE_HASH,
    }
    sanity["lanes_start_identical"] = all(
        torch.equal(beta095[0][0][k], dynamic[0][0][k]) for k in beta095[0][0]
    ) and all(torch.equal(static[0][0][k], dynamic[0][0][k]) for k in static[0][0])
    arena: dict[str, Any] = {"static_q75": {}, "dynamic_q75": {}}
    if args.arena:
        control = _arena_records(
            args.workdir,
            p1_artifact,
            p1_artifact,
            "384:256",
            "p1_control",
            args.arena_workers,
        )
        for lane, steps in {"static_q75": (16, 46), "dynamic_q75": (16, 46)}.items():
            for step in steps:
                records = _arena_records(
                    args.workdir / lane,
                    artifacts[lane][step],
                    p1_artifact,
                    "384:256",
                    f"{lane}_{step}",
                    args.arena_workers,
                )
                entry = paired_opening_candidate_effect(records, control)
                entry["safe"] = arena_safe(entry)
                entry["win_draw_loss"] = _win_draw_loss(records)
                arena[lane][str(step)] = entry
    dynamic46 = arena["dynamic_q75"].get("46")
    dynamic_safe = dynamic46 is not None and dynamic46["safe"]
    dynamic_fit = result_metrics["dynamic_q75"]["46"]["fit_fraction"] >= 0.25
    if not all(sanity.values()):
        classification = {
            "label": "invariant_failure",
            "next_experiment": "repair the locked-input failure",
        }
    elif dynamic_safe and dynamic_fit:
        classification = {
            "label": "dynamic_refresh_rescues_static_mask",
            "next_experiment": "evaluate less frequent refresh schedules before an online algorithm",
        }
    else:
        classification = {
            "label": "dynamic_refresh_insufficient",
            "next_experiment": "do not add a more complicated detector; reassess the shared-head constraint",
        }
    summary = {
        "schema": NAMESPACE,
        "guardrails": {
            "fresh_self_play": False,
            "arena_mask_input": False,
            "optimizer_change": False,
            "value_or_trunk_training": False,
        },
        "inputs": {
            "p1_state_hash": p1_hash,
            "p2_state_hash": p2_hash,
            "gen2_replay_sha256": sha256_file(args.gen2_replay),
            "refresh_after_steps": REFRESH_AFTER_STEPS,
            "optimizer": {"type": "Adam", "lr": 1e-5, "weight_decay": 0.0},
            "gradient_clip": 1.0,
            "batch_size": 512,
        },
        "frozen_masks": frozen,
        "refreshes": refreshes,
        "sanity": sanity,
        "metrics": result_metrics,
        "arena": arena,
        "classification": classification,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_report.write_text(render(summary), encoding="utf-8")
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()
