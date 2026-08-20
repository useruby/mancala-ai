#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluation-only arena gate for the existing PR #209 risk_q75 step-16 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    _win_draw_loss,
    arena_safe,
    puct_probe,
)
from ml.alphazero_lite.run_gen2_selective_risk_training import metrics, state_hash
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import P1_EXPECTED_STATE_HASH
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (
    _new_model,
    puct_trajectory,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import (
    ARENA_SUITE,
    _arena_records,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import stable_hash
from ml.alphazero_lite.train import load_checkpoint_into_model

P0_WEIGHTS_SHA256 = "8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a"
P1_WEIGHTS_SHA256 = "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12"
P1_CHECKPOINT_SHA256 = (
    "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
)
P2E_STATE_HASH = "77eb867796d3346292ebd18701534b52b37400325a1629e01ef1de8bd26ca45c"
P2E_WEIGHTS_SHA256 = "93f3a32f42979fbd72ae3e895e5a8749d7ae3eadf594ac310dcf5bcad0faa975"
P2E_CHECKPOINT_SHA256 = (
    "a08724caf3690c7273628c88a2478a707733e3820595cd83de615bc567ff049a"
)
P2_STATE_HASH = "336496d5fb33331240178c4b834b8faf9548e3915b45c9b5f7e4b7aad6626870"
GEN2_REPLAY_SHA256 = "2cee30547f8bc5d7cad6f02f859ee5e8644386e9b59c8a054ef74548c72ce84b"
BATCH_PLAN_SHA256 = "d9323a8df88f390b8f3a73459f1d0ef6d5939f467a7f43af5bacb40f287628bd"
TRAIN_SOURCE_SHA256 = "82d9633957ca43d3cc388c4cffdd61c90f9a8d1f0e538f5cf66c0d5389b9a971"
Q75_MASK_SHA256 = "cb36a22880c7fb5a1676c96e6e2a5f2b42d8a7562ca98083642862b6fed47d08"
Q75 = 0.008493639528751373
NONINFERIORITY_LOWER = -0.03
CONTEXTS = ("384:256", "1200:1200")


def state_digest(checkpoint: Path) -> str:
    model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(model, checkpoint)
    return stable_hash(
        {
            key: value.detach().cpu().numpy().tobytes().hex()
            for key, value in model.state_dict().items()
        }
    )


def assert_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: {actual!r} != {expected!r}")


def frozen_family_identical(*states: dict[str, torch.Tensor]) -> bool:
    prefixes = (
        "input_layer.",
        "residual_layers.",
        "value_hidden_layer.",
        "value_head.",
    )
    return all(
        all(torch.equal(states[0][name], state[name]) for state in states[1:])
        for name in states[0]
        if name.startswith(prefixes)
    )


def arena_entry(
    candidate: Path,
    opponent: Path,
    context: str,
    workdir: Path,
    role: str,
    workers: int,
) -> dict[str, Any]:
    control = _arena_records(
        workdir, opponent, opponent, context, f"{role}_control", workers
    )
    candidate_records = _arena_records(
        workdir / role, candidate, opponent, context, f"{role}_candidate", workers
    )
    effect = paired_opening_candidate_effect(candidate_records, control)
    return {
        "paired_candidate_effect": effect["paired_candidate_effect"],
        "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
        "p0_effect": effect["p0_effect"],
        "p1_effect": effect["p1_effect"],
        "win_draw_loss": _win_draw_loss(candidate_records),
        "safe": arena_safe(effect),
    }


def classification(
    arena: dict[str, dict[str, Any]], fit_fraction: float
) -> dict[str, str]:
    p2_p1_384 = arena["p2e_vs_p1"]["384:256"]
    p2_p1_1200 = arena["p2e_vs_p1"]["1200:1200"]
    p2_p0_384 = arena["p2e_vs_p0"]["384:256"]
    p2_p0_1200 = arena["p2e_vs_p0"]["1200:1200"]
    if not p2_p1_384["reproduced"]:
        return {
            "label": "invariant_failure",
            "recommended_next_experiment": "repair the failed locked-input invariant",
        }
    safe_p1_384 = bool(p2_p1_384["safe"])
    safe_p1_1200 = bool(p2_p1_1200["safe"])
    safe_p0 = bool(p2_p0_384["safe"] and p2_p0_1200["safe"])
    high_gain = p2_p1_1200["opening_bootstrap_ci"]["lower_95"] > 0.0
    if safe_p1_384 and fit_fraction >= 0.25 and high_gain and safe_p0:
        return {
            "label": "early_checkpoint_cumulative_gain",
            "recommended_next_experiment": "implement a preregistered checkpoint-selection experiment on a fresh generation, with candidate checkpoints evaluated by arena rather than a fixed 46-step endpoint",
        }
    if (
        safe_p1_384
        and safe_p1_1200
        and fit_fraction >= 0.25
        and not high_gain
        and safe_p0
    ):
        return {
            "label": "early_checkpoint_safe_gain_unproven",
            "recommended_next_experiment": "increase arena evaluation power for this checkpoint before changing training or architecture",
        }
    if safe_p1_384 and not safe_p1_1200:
        return {
            "label": "early_checkpoint_low_budget_only",
            "recommended_next_experiment": "reassess the shared policy-head parameterization as suggested by PR #210",
        }
    if safe_p1_384 and not safe_p0:
        return {
            "label": "cumulative_lineage_regression",
            "recommended_next_experiment": "make cumulative lineage benchmarks a promotion requirement",
        }
    return {
        "label": "inconclusive",
        "recommended_next_experiment": "increase arena evaluation power for this checkpoint before changing training or architecture",
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Gen-2 risk_q75 Step-16 Arena Evaluation",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        f"**Recommended next experiment:** {summary['classification']['recommended_next_experiment']}",
        "",
        "## Lineage",
        "",
    ]
    for name, value in summary["lineage"].items():
        lines.append(f"- {name}: `{value}`")
    lines += [
        "",
        "## Canonical Arena Matrix",
        "",
        "| Match | Budget | Effect | 95% CI | P0 seat | P1 seat | W/D/L | Safe |",
        "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for match, contexts in summary["arena"].items():
        for context, entry in contexts.items():
            ci, wdl = entry["opening_bootstrap_ci"], entry["win_draw_loss"]
            lines.append(
                f"| {match} | {context} | {entry['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {entry['p0_effect']:+.4f} | {entry['p1_effect']:+.4f} | {wdl['wins']}/{wdl['draws']}/{wdl['losses']} | {entry['safe']} |"
            )
    metrics = summary["training_context"]
    lines += [
        "",
        "## Training Context",
        "",
        "| Step | Fit fraction | CE(search) | CE(P1) | Policy L1 vs P1 | Protected L1 | Unprotected L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in ("16", "46"):
        value = metrics[step]
        drift = value["drift"]
        lines.append(
            f"| {step} | {value['fit_fraction']:.4f} | {value['ce_candidate_search']:.6f} | {value['ce_candidate_p1']:.6f} | {drift['mean_l1']:.6f} | {drift['protected']['mean_l1']:.6f} | {drift['unprotected']['mean_l1']:.6f} |"
        )
    lines += [
        "",
        f"- Extra fit after safe checkpoint: `{summary['descriptive_changes']['extra_fit_after_safe_checkpoint']:.4f}`",
        f"- Low-budget effect change (step46 - step16): `{summary['descriptive_changes']['low_budget_effect_change']:+.4f}`",
        "",
        "## Search Diagnostics",
        "",
        "| Comparison | Budget | Move change | Visit JS | Q-rank change | Root-value delta | Visit-margin change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison, metrics_by_context in summary["search_diagnostics"]["root"].items():
        for context, value in metrics_by_context.items():
            lines.append(
                f"| {comparison} | {context} | {value['selected_move_change_rate']:.4f} | {value['visit_js']:.6f} | {value['child_q_rank_change']:+.4f} | {value['root_value_delta']:+.6f} | {value['visit_margin']:+.4f} |"
            )
    lines += [
        "",
        "Per-depth legal-policy L1/JS telemetry is retained in the JSON summary for both P2e-vs-P1 and P2e-vs-P0 at 384:256.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_gen2_risk_q75_step16_evaluation"),
    )
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-gen2-risk-q75-step16-evaluation-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-gen2-risk-q75-step16-evaluation-results.md",
    )
    args = parser.parse_args()

    p0 = REPO_ROOT / "model-artifact/current"
    p1 = Path(
        "/tmp/azlite_fresh_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/artifact"
    )
    p1_checkpoint = p1.parent / "checkpoint.npz"
    p2e = Path(
        "/tmp/azlite_gen2_selective_risk_final/risk_q75/snapshot_artifacts/step_0016/artifact"
    )
    p2e_checkpoint = p2e.parent / "checkpoint.npz"
    p2 = Path(
        "/tmp/azlite_gen2_selfplay_anchor/beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    replay = Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl")
    manifest = Path("/tmp/azlite_gen2_selfplay_anchor/training_manifest.json")
    committed = json.loads(
        (
            REPO_ROOT
            / "docs/data/alphazero-lite-gen2-selective-risk-training-summary.json"
        ).read_text(encoding="utf-8")
    )

    assert_equal(sha256_file(p0 / "weights.json"), P0_WEIGHTS_SHA256, "P0 weights")
    assert_equal(sha256_file(p1 / "weights.json"), P1_WEIGHTS_SHA256, "P1 weights")
    assert_equal(sha256_file(p1_checkpoint), P1_CHECKPOINT_SHA256, "P1 checkpoint")
    assert_equal(state_digest(p1_checkpoint), P1_EXPECTED_STATE_HASH, "P1 state")
    assert_equal(sha256_file(p2e / "weights.json"), P2E_WEIGHTS_SHA256, "P2e weights")
    assert_equal(sha256_file(p2e_checkpoint), P2E_CHECKPOINT_SHA256, "P2e checkpoint")
    assert_equal(state_digest(p2e_checkpoint), P2E_STATE_HASH, "P2e state")
    assert_equal(state_digest(p2), P2_STATE_HASH, "PR #204 P2 state")
    assert_equal(sha256_file(replay), GEN2_REPLAY_SHA256, "Gen-2 replay")
    locked_manifest = verify_manifest(manifest)
    assert_equal(locked_manifest["seed"], 43, "training seed")
    assert_equal(locked_manifest["batch_plan"]["batch_size"], 512, "batch size")
    assert_equal(
        locked_manifest["optimizer"],
        {"type": "Adam", "lr": 1e-5, "weight_decay": 0.0},
        "optimizer",
    )
    paths = locked_manifest["artifact_paths"]
    assert_equal(
        sha256_file(Path(paths["batch_indexes"])), BATCH_PLAN_SHA256, "batch plan"
    )
    assert_equal(
        sha256_file(Path(paths["train_source_indexes"])),
        TRAIN_SOURCE_SHA256,
        "train source",
    )
    assert all(committed["sanity"].values())
    assert_equal(
        committed["frozen_masks"]["mask_sha256"]["risk_q75"],
        Q75_MASK_SHA256,
        "frozen q75 mask",
    )

    # Reuse the committed frozen state scores; do not regenerate or refresh the mask.
    frozen = committed["frozen_masks"]
    protected = {
        key
        for key, score in zip(frozen["state_hashes"], frozen["scores"], strict=True)
        if score >= Q75
    }
    assert_equal(
        hashlib.sha256("".join(sorted(protected)).encode()).hexdigest(),
        Q75_MASK_SHA256,
        "replayed q75 mask",
    )
    rows = read_jsonl(replay)
    p1_model, p2_model, p2e_model = (_new_model(torch.device("cpu")) for _ in range(3))
    load_checkpoint_into_model(p1_model, p1_checkpoint)
    load_checkpoint_into_model(p2_model, p2)
    load_checkpoint_into_model(p2e_model, p2e_checkpoint)
    if not frozen_family_identical(
        p1_model.state_dict(), p2_model.state_dict(), p2e_model.state_dict()
    ):
        raise RuntimeError("P0/P1/P2e frozen trunk/value-family invariant failed")
    reproduced = metrics(
        rows,
        {16: (p2e_model.state_dict(), {})},
        p1_model.state_dict(),
        p2_model.state_dict(),
        {"risk_q75": protected, "risk_q90": set()},
        "risk_q75",
        committed["metrics"]["beta095"]["46"]["ce_candidate_search"],
    )["16"]
    expected = committed["metrics"]["risk_q75"]["16"]
    for key in ("ce_candidate_search", "ce_candidate_p1", "fit_fraction"):
        if not np.isclose(reproduced[key], expected[key], atol=1e-12):
            raise RuntimeError(f"P2e committed metric failed to reproduce: {key}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    arena = {
        "p1_vs_p0": {
            context: arena_entry(
                p1, p0, context, args.workdir / "arena", "p1_vs_p0", args.arena_workers
            )
            for context in CONTEXTS
        },
        "p2e_vs_p1": {
            context: arena_entry(
                p2e,
                p1,
                context,
                args.workdir / "arena",
                "p2e_vs_p1",
                args.arena_workers,
            )
            for context in CONTEXTS
        },
        "p2e_vs_p0": {
            context: arena_entry(
                p2e,
                p0,
                context,
                args.workdir / "arena",
                "p2e_vs_p0",
                args.arena_workers,
            )
            for context in CONTEXTS
        },
    }
    known = committed["arena"]["risk_q75"]["16"]["384:256"]
    actual = arena["p2e_vs_p1"]["384:256"]
    actual["reproduced"] = (
        actual["paired_candidate_effect"] == known["paired_candidate_effect"]
        and actual["opening_bootstrap_ci"] == known["opening_bootstrap_ci"]
    )
    if not actual["reproduced"]:
        raise RuntimeError("P2e-vs-P1 low-budget arena did not reproduce PR #209")

    probe_rows = [
        {
            **row,
            "state": decode_state(row["state"]),
            "state_hash": state_hash(row),
            "manifest_index": index,
        }
        for index, row in enumerate(rows[:256])
    ]
    probe_hash = hashlib.sha256("".join(frozen["state_hashes"]).encode()).hexdigest()
    root = {
        "p2e_vs_p1": puct_trajectory(
            probe_rows,
            {0: p1, 16: p2e},
            args.workdir / "puct_p1",
            probe_hash,
            contexts=CONTEXTS,
        )["metrics"]["16"],
        "p2e_vs_p0": puct_trajectory(
            probe_rows,
            {0: p0, 16: p2e},
            args.workdir / "puct_p0",
            probe_hash,
            contexts=CONTEXTS,
        )["metrics"]["16"],
    }
    per_depth = {
        "p2e_vs_p1_384_256": puct_probe(
            probe_rows, p2e, p1, "384:256", modes=("incumbent_all",)
        ),
        "p2e_vs_p0_384_256": puct_probe(
            probe_rows, p2e, p0, "384:256", modes=("incumbent_all",)
        ),
    }
    training_context = {
        step: committed["metrics"]["risk_q75"][step] for step in ("16", "46")
    }
    summary = {
        "schema": "azlite_gen2_risk_q75_step16_evaluation_v1",
        "guardrails": {
            "training": False,
            "self_play": False,
            "mask_regeneration": False,
            "beta_change": False,
            "mcts_change": False,
            "architecture_change": False,
            "promotion": False,
        },
        "lineage": {
            "p0_weights_sha256": P0_WEIGHTS_SHA256,
            "p1_weights_sha256": P1_WEIGHTS_SHA256,
            "p1_state_hash": P1_EXPECTED_STATE_HASH,
            "p2e_weights_sha256": P2E_WEIGHTS_SHA256,
            "p2e_checkpoint_sha256": P2E_CHECKPOINT_SHA256,
            "p2e_state_hash": P2E_STATE_HASH,
            "p2_state_hash": P2_STATE_HASH,
            "gen2_replay_sha256": GEN2_REPLAY_SHA256,
            "batch_plan_sha256": BATCH_PLAN_SHA256,
            "train_source_sha256": TRAIN_SOURCE_SHA256,
            "frozen_q75_mask_sha256": Q75_MASK_SHA256,
        },
        "invariants": {
            "committed_sanity": committed["sanity"],
            "manifest_seed": 43,
            "batch_size": 512,
            "frozen_trunk_and_value_family_identical": True,
            "p2e_metrics_reproduced": True,
            "canonical_opening_suite_sha256": sha256_file(ARENA_SUITE),
        },
        "arena": arena,
        "candidate_vs_p0": arena["p2e_vs_p0"],
        "training_context": training_context,
        "descriptive_changes": {
            "extra_fit_after_safe_checkpoint": training_context["46"]["fit_fraction"]
            - training_context["16"]["fit_fraction"],
            "low_budget_effect_change": committed["arena"]["risk_q75"]["46"]["384:256"][
                "paired_candidate_effect"
            ]
            - known["paired_candidate_effect"],
        },
        "search_diagnostics": {"root": root, "per_depth": per_depth},
    }
    summary["classification"] = classification(
        arena, training_context["16"]["fit_fraction"]
    )
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], indent=2))


if __name__ == "__main__":
    main()
