#!/usr/bin/env python3
# ruff: noqa: E402
"""Matched multi-step causal decomposition of the frozen-trunk heads failure.

PR #199 showed that freezing the incumbent trunk does not remove the harmful
low-budget search effect: ``heads_only`` training was neutral at optimizer
steps 1 and 4, negative at step 16, and strongly negative at step 46
(approximately -0.260 at 384:256, concentrated in the P0 seat) while remaining
positive at 1200:1200. This experiment decomposes that multi-step failure into
policy-head and value-head causal components by replaying the exact PR #199
protocol with only one head family trainable at a time.

Lanes (identical replay rows, batch order, Adam lr=1e-5, batch size 512,
CE + 0.6 * Huber, seed, architecture, search settings, evaluation openings,
and frozen incumbent opponent exactly as PR #199):

- ``incumbent``: the frozen current artifact; no optimizer updates.
- ``heads_only``: both head stacks trainable (the PR #199 failure lane).
- ``policy_head``: only the policy head stack trainable.
- ``value_head``: only the value head stack trainable.
- ``all`` (optional reproduction reference): normal shared-trunk training.

Nothing about the training protocol changes between lanes except which
parameter families receive gradients. No promotion, no new self-play, no
architecture or target changes. Every produced artifact is diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import (  # noqa: E402
    checkpoint_steps,
    frozen_arena,
    group_delta,
    lane_trainable_scope,
    output_drift,
    replay_lane,
    state_hashes,
    tensors_identical,
    trunk_parameters_identical,
    with_total_loss,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    CURRENT_HASH,
    export_snapshot_artifacts,
    puct_trajectory,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    js,
    model_outputs,
    stable_hash,
)
from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states  # noqa: E402

NAMESPACE = "azlite_frozen_trunk_head_isolation_v1"
TRAINABLE_LANES = ("heads_only", "policy_head", "value_head")
REFERENCE_LANE = "all"
ARENA_CONTEXTS = ("384:256", "1200:1200")
PRIMARY_CONTEXT = "384:256"
PUCT_CONTEXTS = ("384:256",)
VALUE_STACK_PREFIXES = ("value_hidden_layer.", "value_head.")
POLICY_STACK_PREFIXES = ("policy_hidden_layer.", "policy_head.", "move_projections.")
MATERIAL_EFFECT = -0.05

# PR #199 (commit b82ee42) recorded state hashes; the heads_only lane must
# still reproduce them byte-for-byte before any causal decomposition is read.
PR199_STATE_HASHES = {
    "heads_only": {
        "0": "d265537d6b637b8433b093ebb1f9d55fef25b38e259ed7e59f7a597b35bb6f02",
        "1": "2f3eb3616c8272a3b57772f855ebd75a480cf3f44783a9934e0f7adbcff20a1f",
        "4": "9621a14339510f5151cd161ca6394987b5689bad3787360ad710781d83ba1577",
        "16": "e6b7d8852155b920ebf26e51ff79d6fbb39677671f2b027cdd26d027a9251ef4",
        "46": "1c247125b34937a25ab16aac1774d1c5de6f766124df782788383f94387b824a",
    },
    "all": {
        "0": "d265537d6b637b8433b093ebb1f9d55fef25b38e259ed7e59f7a597b35bb6f02",
        "1": "ab86110b43d6023c3a9f3456383ea58a6ec1c717ebb877c006e9c23fda01efb2",
        "4": "23716a25567fc7664f34d82bb68510ceeb6abf818ceb8a6325a3dc6871cdca25",
        "16": "b0c40d77980489cdf411761276bb527ab85226340ed285e151b552ce2a699434",
        "46": "9a15bd97cff79fa296367ed7399111e813bc00592e439f0323435bf53cc42b87",
    },
}


def group_parameters_identical(
    state: dict[str, torch.Tensor],
    incumbent: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> bool:
    """Return whether every parameter in the given families is byte-identical."""
    names = [name for name in sorted(state) if name.startswith(prefixes)]
    if not names:
        raise ValueError(f"no parameters matched prefixes: {prefixes}")
    return all(
        state[name].detach().cpu().numpy().tobytes()
        == incumbent[name].detach().cpu().numpy().tobytes()
        for name in names
    )


def probe_output_drift(
    rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
) -> dict[str, Any]:
    """Network-output drift versus the incumbent on the frozen probe.

    Reports the policy top-1 change rate, mean legal-policy L1 distance, mean
    legal-policy JS divergence (and KL, matching the existing PR #198/#199
    convention), plus the value mean absolute and signed mean output delta.
    """
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    outputs = {
        str(step): model_outputs(state, x, mask)
        for step, (state, _optimizer) in snapshots.items()
    }
    incumbent_policy, incumbent_value = outputs["0"]
    result: dict[str, Any] = {}
    for step, (policy, value) in outputs.items():
        divergence = value - incumbent_value
        result[str(step)] = {
            "policy": {
                "top1_change_from_current": float(
                    np.mean(
                        np.argmax(policy, axis=1) != np.argmax(incumbent_policy, axis=1)
                    )
                ),
                "legal_l1_from_current": float(
                    np.mean(np.sum(np.abs(policy - incumbent_policy), axis=1))
                ),
                "legal_js_from_current": float(np.mean(js(policy, incumbent_policy))),
                "legal_kl_from_current": float(
                    np.mean(
                        np.sum(
                            policy
                            * np.log(
                                np.clip(policy, 1e-12, None)
                                / np.clip(incumbent_policy, 1e-12, None)
                            ),
                            axis=1,
                        )
                    )
                ),
            },
            "value": {
                "mean_absolute_output_delta": float(np.mean(np.abs(divergence))),
                "signed_mean_output_delta": float(np.mean(divergence)),
            },
        }
    return result


def _materially_harmful(entry: dict[str, Any]) -> bool:
    """A lane reproduces the failure when the paired effect is materially
    negative and the bootstrap 95% interval excludes zero."""
    return (
        float(entry["paired_candidate_effect"]) <= MATERIAL_EFFECT
        and float(entry["opening_bootstrap_ci"]["upper_95"]) < 0.0
    )


def _final_entry(
    lane_metrics: dict[str, Any], steps: list[str]
) -> dict[str, Any] | None:
    final = str(max(int(step) for step in steps))
    return lane_metrics.get(final, {}).get(PRIMARY_CONTEXT)


def _reproduces_failure(lane_metrics: dict[str, Any], steps: list[str]) -> bool:
    entry = _final_entry(lane_metrics, steps)
    return entry is not None and _materially_harmful(entry)


NEXT_ACTIONS = {
    "value_head_accumulation": (
        "run a value-target/search-calibration experiment on the frozen trunk: first "
        "compare the existing default, sharpened, hybrid, and phase-aware value-target "
        "modes under this same 46-step replay, then test a search-calibrated value "
        "target that anchors the value head to incumbent search root values instead "
        "of raw canonical outcomes"
    ),
    "policy_head_accumulation": (
        "run a policy-prior/low-budget-search intervention under this same 46-step "
        "replay: behavior anchoring against the incumbent policy or constrained "
        "prior drift (a trust region on legal-policy divergence from the incumbent)"
    ),
    "joint_head_search_interaction": (
        "test MCTS sensitivity to independently mixed incumbent/candidate policy and "
        "value outputs (four combinations at 384:256) before changing any training"
    ),
    "both_heads_independently_harmful": (
        "stabilize each head separately (policy behavior anchoring and value "
        "calibration, verified in isolation under this same 46-step replay) rather "
        "than a dual-trunk rewrite"
    ),
    "inconclusive": (
        "the available confidence intervals cannot separate the head families; rerun "
        "with more openings or seeds before choosing an intervention"
    ),
}

FINDINGS = {
    "value_head_accumulation": (
        "Accumulated value-head drift is the primary mechanism. The `value_head` "
        "lane reproduces the PR #199 heads-only low-budget failure at 384:256 while "
        "the `policy_head` lane is substantially safer."
    ),
    "policy_head_accumulation": (
        "Accumulated policy-head drift is the primary mechanism. The `policy_head` "
        "lane reproduces the PR #199 heads-only low-budget failure at 384:256 while "
        "the `value_head` lane is substantially safer."
    ),
    "joint_head_search_interaction": (
        "Neither isolated head materially degrades low-budget arena strength, but "
        "`heads_only` does: the failure arises from combining changed policy priors "
        "and changed value estimates inside MCTS."
    ),
    "both_heads_independently_harmful": (
        "Both isolated head lanes materially degrade low-budget arena strength; the "
        "failure does not require the heads to move together."
    ),
    "inconclusive": (
        "The available confidence intervals cannot distinguish which head family "
        "causes the low-budget failure."
    ),
}


def _materially_harmful_steps(
    lane_metrics: dict[str, Any], steps: list[str]
) -> list[str]:
    result = []
    for step in steps:
        entry = lane_metrics.get(step, {}).get(PRIMARY_CONTEXT)
        if entry is not None and _materially_harmful(entry):
            result.append(step)
    return result


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the prespecified head-isolation decision rule (arena evidence only)."""
    arena = summary.get("arena") or {}
    steps = [str(step) for step in summary.get("checkpoint_steps", [])]
    lane_metrics = {lane: arena.get(lane) or {} for lane in TRAINABLE_LANES}

    if not steps or any(not metrics for metrics in lane_metrics.values()):
        return {
            "label": "inconclusive",
            "next_action": "complete the preregistered arena before classification",
            "evidence": {"arena_complete": False},
        }

    heads_reproduces = _reproduces_failure(lane_metrics["heads_only"], steps)
    value_reproduces = _reproduces_failure(lane_metrics["value_head"], steps)
    policy_reproduces = _reproduces_failure(lane_metrics["policy_head"], steps)

    final = str(max(int(step) for step in steps))
    final_entries = {
        lane: _final_entry(lane_metrics[lane], steps) for lane in TRAINABLE_LANES
    }

    if not heads_reproduces:
        label = "inconclusive"
    elif value_reproduces and policy_reproduces:
        label = "both_heads_independently_harmful"
    elif value_reproduces:
        label = "value_head_accumulation"
    elif policy_reproduces:
        label = "policy_head_accumulation"
    else:
        label = "joint_head_search_interaction"

    evidence: dict[str, Any] = {
        "material_effect_threshold": MATERIAL_EFFECT,
        "primary_context": PRIMARY_CONTEXT,
        "final_step": int(final),
        "final_paired_effects": {
            lane: (
                float(entry["paired_candidate_effect"]) if entry is not None else None
            )
            for lane, entry in final_entries.items()
        },
        "final_p0_effects": {
            lane: (float(entry.get("p0_effect", 0.0)) if entry is not None else None)
            for lane, entry in final_entries.items()
        },
        "final_p1_effects": {
            lane: (float(entry.get("p1_effect", 0.0)) if entry is not None else None)
            for lane, entry in final_entries.items()
        },
        "final_ci_upper_95": {
            lane: (
                float(entry["opening_bootstrap_ci"]["upper_95"])
                if entry is not None
                else None
            )
            for lane, entry in final_entries.items()
        },
        "heads_only_reproduces_failure": heads_reproduces,
        "value_head_reproduces_failure": value_reproduces,
        "policy_head_reproduces_failure": policy_reproduces,
        "materially_harmful_steps": {
            lane: _materially_harmful_steps(lane_metrics[lane], steps)
            for lane in TRAINABLE_LANES
        },
        "p0_failure_concentrated_in_value_head": bool(
            value_reproduces
            and final_entries["value_head"] is not None
            and float(final_entries["value_head"].get("p0_effect", 0.0))
            < 2.0 * float(final_entries["value_head"].get("p1_effect", 0.0))
        ),
    }

    return {
        "label": label,
        "next_action": NEXT_ACTIONS[label],
        "evidence": evidence,
    }


def markdown(summary: dict[str, Any]) -> str:
    """Render a compact committed record; complete detail remains in JSON."""
    classification = summary["classification"]
    label = classification["label"]
    lanes_order = [lane for lane in TRAINABLE_LANES if lane in summary["output_drift"]]
    if REFERENCE_LANE in summary["output_drift"]:
        lanes_order.append(REFERENCE_LANE)
    lines = [
        "# AlphaZero-Lite Frozen-Trunk Head-Isolation Results",
        "",
        f"**Classification:** `{label}`",
        "",
        f"- heads_only reproduces PR #199 state hashes: `{summary['deterministic_reproduction']['pr199_heads_only_state_hashes']}`",
        f"- all trainable lanes start from identical incumbent: `{summary['sanity']['lanes_start_identical']}`",
        f"- policy_head zero trunk change: `{summary['sanity']['policy_head_trunk_zero_change']}`",
        f"- policy_head zero value-stack change: `{summary['sanity']['policy_head_value_stack_zero_change']}`",
        f"- value_head zero trunk change: `{summary['sanity']['value_head_trunk_zero_change']}`",
        f"- value_head zero policy-stack change: `{summary['sanity']['value_head_policy_stack_zero_change']}`",
        f"- heads_only zero trunk change: `{summary['sanity']['heads_only_trunk_zero_change']}`",
        f"- checkpoint steps: `{summary['checkpoint_steps']}`",
        f"- current weights sha256: `{summary['inputs']['current_weights_sha256']}`",
        f"- replay sha256: `{summary['inputs']['replay_sha256']}`",
        "",
        "## Findings",
        "",
        FINDINGS[label],
        "",
    ]
    evidence = classification["evidence"]
    final = evidence.get("final_step")
    effects = evidence.get("final_paired_effects") or {}
    p0s = evidence.get("final_p0_effects") or {}
    if final and effects:
        lines.append(
            f"At the final checkpoint (step {final}) the 384:256 paired effects are: "
            + ", ".join(
                f"`{lane}` {effect:+.4f} (P0 {p0s.get(lane, 0.0):+.4f})"
                for lane, effect in effects.items()
                if effect is not None
            )
            + "."
        )
        lines.append("")
    harmful_steps = evidence.get("materially_harmful_steps") or {}
    value_steps = harmful_steps.get("value_head") or []
    if (
        label == "policy_head_accumulation"
        and value_steps
        and final is not None
        and str(final) not in value_steps
    ):
        lines.append(
            "The `value_head` lane is transiently negative at steps "
            f"{', '.join(value_steps)} before recovering to neutral or positive at "
            "the final checkpoint, so accumulated value drift alone does not cause "
            "the terminal low-budget failure; the policy-head drift does."
        )
        lines.append("")
    lines.extend(
        [
            "## Supervised objective and parameter drift (frozen validation probe)",
            "",
            "| Lane | Step | Total loss | Policy CE | Value huber | Trunk drift | Policy-head drift | Value-head drift |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in ("incumbent", *lanes_order):
        lane_steps = (
            ["0"]
            if lane == "incumbent"
            else [str(step) for step in summary["checkpoint_steps"]]
        )
        for step in lane_steps:
            if step not in summary["output_drift"].get(lane, {}):
                continue
            drift = summary["output_drift"][lane][step]
            param = summary["drift"][lane][step]
            lines.append(
                f"| {lane} | {step} | {drift['total_loss']:.4f} | "
                f"{drift['policy']['replay_teacher_cross_entropy']:.4f} | "
                f"{drift['value']['huber_loss']:.4f} | {param['trunk']:.6f} | "
                f"{param['policy_head']:.6f} | {param['value_head']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Network-output drift on the frozen probe (versus incumbent)",
            "",
            "| Lane | Step | Top-1 change | Legal L1 | Legal JS | Value mean abs delta | Value signed delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in ("incumbent", *lanes_order):
        lane_steps = (
            ["0"]
            if lane == "incumbent"
            else [str(step) for step in summary["checkpoint_steps"]]
        )
        for step in lane_steps:
            entry = summary["probe_output_drift"].get(lane, {}).get(step)
            if entry is None:
                continue
            lines.append(
                f"| {lane} | {step} | "
                f"{entry['policy']['top1_change_from_current']:.4f} | "
                f"{entry['policy']['legal_l1_from_current']:.4f} | "
                f"{entry['policy']['legal_js_from_current']:.6f} | "
                f"{entry['value']['mean_absolute_output_delta']:.4f} | "
                f"{entry['value']['signed_mean_output_delta']:+.4f} |"
            )
    lines.extend(
        [
            "",
            f"## Search diagnostics ({PRIMARY_CONTEXT} context, versus incumbent)",
            "",
            "| Lane | Step | Move change | Visit JS | Q-rank change | Root-value delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    puct = summary.get("puct") or {}
    for lane in lanes_order:
        metrics = puct.get(lane, {}).get("metrics", {})
        for step in summary["checkpoint_steps"]:
            entry = metrics.get(str(step), {}).get(PRIMARY_CONTEXT)
            if entry is None:
                continue
            lines.append(
                f"| {lane} | {step} | {entry['selected_move_change_rate']:.4f} | "
                f"{entry['visit_js']:.4f} | {entry['child_q_rank_change']:+.4f} | "
                f"{entry['root_value_delta']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Canonical arena (candidate versus frozen incumbent)",
            "",
            "| Lane | Step | Context | Paired effect | 95% CI | P0 effect | P1 effect |",
            "| --- | ---: | --- | ---: | --- | ---: | ---: |",
        ]
    )
    arena = summary.get("arena") or {}
    for lane in lanes_order:
        for step in summary["checkpoint_steps"]:
            for context in ARENA_CONTEXTS:
                entry = arena.get(lane, {}).get(str(step), {}).get(context)
                if entry is None:
                    continue
                ci = entry["opening_bootstrap_ci"]
                lines.append(
                    f"| {lane} | {step} | {context} | "
                    f"{entry['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                    f"{entry.get('p0_effect', 0.0):+.4f} | "
                    f"{entry.get('p1_effect', 0.0):+.4f} |"
                )
    lines.extend(
        [
            "",
            "## Classification evidence",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for key, value in classification["evidence"].items():
        if isinstance(value, bool):
            rendered = str(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            rendered = f"{value:+.4f}" if isinstance(value, float) else str(value)
        elif isinstance(value, dict) and all(
            isinstance(item, (int, float)) or item is None for item in value.values()
        ):
            rendered = json.dumps(
                {
                    lane: (f"{effect:+.4f}" if effect is not None else None)
                    for lane, effect in value.items()
                }
            )
        else:
            rendered = json.dumps(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(
        [
            "",
            "## Recommended next experiment (not implemented here)",
            "",
            f"`{classification['next_action']}`",
            "",
            "## Exact commands",
            "",
            "```bash",
            "python ml/alphazero_lite/run_frozen_trunk_head_isolation_ablation.py \\",
            "  --pr191-workdir /tmp/azlite_shared_trunk_learning \\",
            "  --workdir <workdir> --include-all --arena-workers 24",
            "```",
            "",
            "Full evidence: `docs/data/alphazero-lite-frozen-trunk-head-isolation-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_frozen_trunk_head_isolation")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-frozen-trunk-head-isolation-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-frozen-trunk-head-isolation-results.md",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Also replay the `all` lane as a PR #199 reproduction reference",
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena-workers", type=int, default=24)
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match the PR191 initialization")
    if manifest["architecture"]["model_type"] != "residual_v3":
        raise RuntimeError("head-isolation lanes require the residual_v3 architecture")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(manifest["seed"])
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batch_count = int(len(plan))
    steps = checkpoint_steps(batch_count)

    lanes_to_replay = [*TRAINABLE_LANES]
    if args.include_all:
        lanes_to_replay.append(REFERENCE_LANE)
    lanes: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    for lane in lanes_to_replay:
        configure_determinism(device, seed)
        lanes[lane] = replay_lane(manifest, args.workdir / lane, device, lane, steps)

    incumbent = lanes[TRAINABLE_LANES[0]][0][0]
    lanes_start_identical = all(
        tensors_identical(lanes[lane][0][0], incumbent) for lane in lanes_to_replay
    )
    if not lanes_start_identical:
        raise RuntimeError("trainable lanes do not start from identical parameters")

    invariants = {
        "heads_only_trunk_zero_change": all(
            trunk_parameters_identical(state, incumbent)
            for state, _optimizer in lanes["heads_only"].values()
        ),
        "policy_head_trunk_zero_change": all(
            trunk_parameters_identical(state, incumbent)
            for state, _optimizer in lanes["policy_head"].values()
        ),
        "policy_head_value_stack_zero_change": all(
            group_parameters_identical(state, incumbent, VALUE_STACK_PREFIXES)
            for state, _optimizer in lanes["policy_head"].values()
        ),
        "value_head_trunk_zero_change": all(
            trunk_parameters_identical(state, incumbent)
            for state, _optimizer in lanes["value_head"].values()
        ),
        "value_head_policy_stack_zero_change": all(
            group_parameters_identical(state, incumbent, POLICY_STACK_PREFIXES)
            for state, _optimizer in lanes["value_head"].values()
        ),
    }
    for name, ok in invariants.items():
        if not ok:
            raise RuntimeError(f"frozen-family invariant violated: {name}")

    heads_hashes = state_hashes(lanes["heads_only"])
    pr199_heads_reproduced = heads_hashes == PR199_STATE_HASHES["heads_only"]
    if not pr199_heads_reproduced:
        raise RuntimeError("heads_only lane no longer reproduces the PR #199 hashes")
    pr199_all_reproduced = None
    if args.include_all:
        pr199_all_reproduced = (
            state_hashes(lanes[REFERENCE_LANE]) == PR199_STATE_HASHES[REFERENCE_LANE]
        )
        if not pr199_all_reproduced:
            raise RuntimeError("all lane no longer reproduces the PR #199 hashes")

    rows = read_jsonl(Path(manifest["replay_path"]))
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(rows, validation_indexes)
    probe_manifest["validation_source_indexes_sha256"] = sha256_file(
        paths["validation_source_indexes"]
    )
    probe_manifest["replay_sha256"] = sha256_file(Path(manifest["replay_path"]))
    probe_manifest["manifest_sha256"] = stable_hash(probe_manifest)
    probe_rows = [rows[index] for index in probe_manifest["source_indexes"]]

    output: dict[str, Any] = {}
    probe_drift: dict[str, Any] = {}
    drift: dict[str, Any] = {}
    for lane in lanes_to_replay:
        output[lane] = with_total_loss(output_drift(probe_rows, lanes[lane]))
        probe_drift[lane] = probe_output_drift(probe_rows, lanes[lane])
        drift[lane] = {
            str(step): group_delta(state, incumbent)
            for step, (state, _optimizer) in lanes[lane].items()
        }
    output["incumbent"] = with_total_loss(
        output_drift(probe_rows, {0: (incumbent, {})})
    )
    probe_drift["incumbent"] = probe_output_drift(probe_rows, {0: (incumbent, {})})
    drift["incumbent"] = {"0": group_delta(incumbent, incumbent)}

    artifacts = {
        lane: export_snapshot_artifacts(lanes[lane], args.workdir / lane)
        for lane in lanes_to_replay
    }
    puct: dict[str, Any] = {}
    if args.puct:
        probe_hash = stable_hash(probe_manifest)
        for lane in lanes_to_replay:
            puct[lane] = puct_trajectory(
                probe[:256],
                artifacts[lane],
                args.workdir / lane,
                probe_hash,
                contexts=PUCT_CONTEXTS,
            )
    arena: dict[str, Any] = {}
    if args.arena:
        arena = frozen_arena(
            artifacts,
            args.current,
            args.workdir / "arena",
            args.arena_workers,
            lanes=tuple(lanes_to_replay),
        )

    summary: dict[str, Any] = {
        "schema": NAMESPACE,
        "guardrails": {
            "promotion": False,
            "new_self_play": False,
            "target_change": False,
            "lr_change": False,
            "loss_weight_change": False,
            "architecture_change": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "current_weights_sha256": CURRENT_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "training_manifest_sha256": sha256_file(
                args.pr191_workdir / "training_manifest.json"
            ),
            "initialization_checkpoint_sha256": sha256_file(
                paths["initialization_checkpoint"]
            ),
            "seed": seed,
            "optimizer": manifest["optimizer"],
            "gradient_clip": manifest["gradient_clip"],
            "value_loss": manifest["value_loss"],
            "value_loss_weight": manifest["value_loss_weight"],
            "huber_delta": manifest["huber_delta"],
            "policy_loss": manifest["policy_loss"],
            "architecture": manifest["architecture"],
            "batch_plan": manifest["batch_plan"],
            "batch_count": batch_count,
            "lane_trainable_scopes": {
                lane: lane_trainable_scope(lane) for lane in lanes_to_replay
            },
            "optimizer_step_counts": {
                lane: list(snapshots.keys()) for lane, snapshots in lanes.items()
            },
            "evaluation": {
                "opponent": "frozen incumbent (model-artifact/current)",
                "openings": "canonical 128-opening suite, 2 games per opening per seat",
                "contexts": list(ARENA_CONTEXTS),
                "puct_contexts": list(PUCT_CONTEXTS),
            },
        },
        "deterministic_reproduction": {
            "pr199_heads_only_state_hashes": pr199_heads_reproduced,
            "pr199_all_state_hashes": pr199_all_reproduced,
        },
        "sanity": {"lanes_start_identical": lanes_start_identical, **invariants},
        "checkpoint_steps": steps,
        "state_hashes": {
            lane: state_hashes(snapshots) for lane, snapshots in lanes.items()
        },
        "drift": drift,
        "output_drift": output,
        "probe_output_drift": probe_drift,
        "probe_manifest": probe_manifest,
        "puct": puct,
        "arena": arena,
    }
    summary["classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
