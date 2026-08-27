#!/usr/bin/env python3
# ruff: noqa: E402
"""Re-run PR #241 from frozen replay with independently-owned Adam state only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as distillation
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_onpolicy_shadow_replay import (
    A16_SNAPSHOT,
    A16_WEIGHTS_SHA,
    P1_CHECKPOINT,
    P1_CHECKPOINT_SHA,
    STEPS,
    exclusion_hashes,
    filter_rows,
    immutable_initial_state,
    metrics,
    optimizer_state_sha256,
    train_lane,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.train import load_checkpoint_into_model

LANES = ("ordinary_onpolicy", "shadow075_onpolicy", "shadow100_onpolicy")
ORDER_A = LANES
ORDER_B = ("shadow100_onpolicy", "ordinary_onpolicy", "shadow075_onpolicy")
EXPECTED_ROWS = {
    "ordinary_onpolicy": 27350,
    "shadow075_onpolicy": 27597,
    "shadow100_onpolicy": 26425,
}
EXPECTED_REPLAY_SHA = {
    "ordinary_onpolicy": "6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2",
    "shadow075_onpolicy": "066e52a52dfcd73186a5a98470abbc6412341c770170a76e5bd1763735bbd039",
    "shadow100_onpolicy": "38dd9e8910cd5a8a2f7eb977ddc5ab3f102f323563804902d32c92f4e49f1684",
}
SUITE_SHA = "57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04"
A16_SNAPSHOT_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
CONTEXTS = ("384:256", "1200:1200")


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def adapter_sha256(state: dict[str, torch.Tensor]) -> str:
    return state_sha256({key: state[key] for key in ADAPTER_KEYS})


def checkpoint_record(
    state: dict[str, torch.Tensor], optimizer: dict[str, Any]
) -> dict[str, str]:
    return {
        "model_state_sha256": state_sha256(state),
        "adapter_tensor_sha256": adapter_sha256(state),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer),
    }


def load_rows(
    workdir: Path, suite: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    blocked, groups = exclusion_hashes(suite)
    rows, provenance = {}, {}
    for lane in LANES:
        source = workdir / f"{lane}.jsonl"
        digest = sha256_file(source)
        if digest != EXPECTED_REPLAY_SHA[lane]:
            raise RuntimeError(f"invariant_failure: {lane} replay hash mismatch")
        eligible, exclusions = filter_rows(read_jsonl(source), blocked, groups)
        if len(eligible) != EXPECTED_ROWS[lane]:
            raise RuntimeError(f"invariant_failure: {lane} eligible row count mismatch")
        rows[lane] = eligible
        provenance[lane] = {"replay_sha256": digest, "exclusions": exclusions}
    return rows, provenance


def run_lane(
    rows: list[dict[str, Any]],
    pristine_model: dict[str, torch.Tensor],
    pristine_optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
) -> tuple[
    dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]], dict[str, str]
]:
    supplied_model = copy.deepcopy(pristine_model)
    supplied_optimizer = copy.deepcopy(pristine_optimizer)
    before = optimizer_state_sha256(pristine_optimizer)
    snapshots, optimizers = train_lane(
        rows,
        {"model": supplied_model, "optimizer": supplied_optimizer},
        parent,
        torch.device("cpu"),
    )
    after = optimizer_state_sha256(pristine_optimizer)
    if before != after:
        raise RuntimeError("invariant_failure: pristine optimizer mutation")
    if optimizer_state_sha256(supplied_optimizer) != before:
        raise RuntimeError("invariant_failure: train input optimizer mutation")
    return snapshots, optimizers, {"before": before, "after": after}


def compare_runs(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for lane in LANES:
        for step in STEPS:
            if (
                left[lane]["checkpoints"][str(step)]
                != right[lane]["checkpoints"][str(step)]
            ):
                raise RuntimeError(
                    f"optimizer_isolation_invariant_failure: {label} hashes"
                )
            if left[lane]["metrics"][str(step)] != right[lane]["metrics"][str(step)]:
                raise RuntimeError(
                    f"optimizer_isolation_invariant_failure: {label} metrics"
                )


def train_order(
    order: tuple[str, ...],
    rows_by_lane: dict[str, list[dict[str, Any]]],
    pristine_model: dict[str, torch.Tensor],
    pristine_optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]],
]:
    summary, states = {}, {}
    for lane in order:
        snapshots, optimizers, invocation = run_lane(
            rows_by_lane[lane], pristine_model, pristine_optimizer, parent
        )
        metrics_before = optimizer_state_sha256(pristine_optimizer)
        lane_metrics = metrics(
            rows_by_lane[lane],
            snapshots,
            parent,
            pristine_model,
            copy.deepcopy(pristine_optimizer),
        )
        metrics_after = optimizer_state_sha256(pristine_optimizer)
        if metrics_before != metrics_after:
            raise RuntimeError("invariant_failure: metrics mutated pristine optimizer")
        summary[lane] = {
            "optimizer_invocation": invocation,
            "checkpoints": {
                str(step): checkpoint_record(snapshots[step], optimizers[step])
                for step in STEPS
            },
            "metrics": lane_metrics,
        }
        states[lane] = (snapshots, optimizers)
    return summary, states


def repeated_lane_check(
    lane: str,
    rows: list[dict[str, Any]],
    pristine_model: dict[str, torch.Tensor],
    pristine_optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
) -> bool:
    first = run_lane(rows, pristine_model, pristine_optimizer, parent)
    second = run_lane(rows, pristine_model, pristine_optimizer, parent)
    return all(
        checkpoint_record(first[0][step], first[1][step])
        == checkpoint_record(second[0][step], second[1][step])
        for step in STEPS
    )


def historical_comparison(
    workdir: Path,
    states: dict[
        str, tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]
    ],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lane in LANES:
        result[lane] = {}
        for step in STEPS:
            historical = torch.load(
                workdir / f"{lane}_train/step_{step:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            clean_state, clean_optimizer = states[lane][0][step], states[lane][1][step]
            delta = torch.cat(
                [
                    (clean_state[key] - historical["model"][key]).reshape(-1)
                    for key in ADAPTER_KEYS
                ]
            )
            result[lane][str(step)] = {
                "model_sha_match": state_sha256(clean_state)
                == state_sha256(historical["model"]),
                "optimizer_sha_match": optimizer_state_sha256(clean_optimizer)
                == optimizer_state_sha256(historical["optimizer"]),
                "adapter_max_abs_difference": float(torch.max(torch.abs(delta))),
            }
    return result


def write_artifacts(
    workdir: Path,
    states: dict[
        str, tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]
    ],
) -> dict[str, dict[int, Path]]:
    artifacts: dict[str, dict[int, Path]] = {}
    for lane, (snapshots, optimizers) in states.items():
        artifacts[lane] = {}
        for step in STEPS:
            output = workdir / "clean_train" / lane / f"step_{step:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": snapshots[step], "optimizer": optimizers[step]},
                output.with_suffix(".pt"),
            )
            artifacts[lane][step] = export(
                snapshots[step], output, f"pr241_clean_{lane}_{step}"
            )
    return artifacts


def win_draw_loss(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {"wins": 0, "draws": 0, "losses": 0}
    for record in records:
        result[
            {"challenger": "wins", "draw": "draws", "current": "losses"}[
                record["winner"]
            ]
        ] += 1
    return result


def arena_summary(
    workdir: Path,
    artifacts: dict[str, dict[int, Path]],
    p1: Path,
    suite: Path,
    workers: int,
) -> dict[str, Any]:
    controls = {
        context: isolation.arena_records(
            workdir / "arena", p1, p1, context, "p1_control", workers, suite
        )
        for context in CONTEXTS
    }
    result: dict[str, Any] = {}
    for lane, checkpoints in artifacts.items():
        result[lane] = {}
        for step, artifact in checkpoints.items():
            result[lane][str(step)] = {}
            for context, control in controls.items():
                records = isolation.arena_records(
                    workdir / "arena",
                    artifact,
                    p1,
                    context,
                    f"{lane}_{step}",
                    workers,
                    suite,
                )
                effect = paired_opening_candidate_effect(records, control)
                ci = effect["opening_bootstrap_ci"]
                result[lane][str(step)][context] = {
                    "paired_effect": effect["paired_candidate_effect"],
                    "ci": ci,
                    "seat_effects": {
                        "p0": effect["p0_effect"],
                        "p1": effect["p1_effect"],
                    },
                    "win_draw_loss": win_draw_loss(records),
                    "safe": ci["upper_95"] >= 0.0 or ci["lower_95"] >= -0.03,
                }
    return result


def classify(arena: dict[str, Any]) -> tuple[str, str]:
    ordinary = arena["ordinary_onpolicy"]["16"]
    shadows = [arena[lane]["16"] for lane in LANES[1:]]
    ordinary_high = ordinary["1200:1200"]["paired_effect"] == 0.041015625
    shadows_improve = any(
        values["1200:1200"]["paired_effect"] > ordinary["1200:1200"]["paired_effect"]
        for values in shadows
    )
    if ordinary_high and not shadows_improve:
        return (
            "ordinary_reused_target_gain_reproduces",
            "isolate reused gameplay-tree policy targets versus fresh-tree targets",
        )
    if all(not values["1200:1200"]["safe"] for values in shadows):
        return (
            "shadow_replay_negative_result_reproduces",
            "close the shadow/target branch and return to the last fully reproduced baseline",
        )
    return (
        "inconclusive",
        "establish a further frozen-artifact reproduction before ML changes",
    )


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PR #241 Optimizer-Isolation Reproduction",
        "",
        f"**Classification:** `{result['classification']}`",
        "",
        "## Invariants",
        "",
        f"- Initial Adam SHA-256: `{result['initial_optimizer_sha256']}`",
        f"- Order invariance: `{result['invariants']['lane_order_invariant']}`",
        f"- Repeated ordinary: `{result['invariants']['repeated_ordinary']}`",
        f"- Repeated shadow075: `{result['invariants']['repeated_shadow075']}`",
        "",
        "## Step-16 Arena",
        "",
        "| Lane | 384:256 | 1200:1200 |",
        "| --- | ---: | ---: |",
    ]
    for lane in LANES:
        values = result["canonical_arena"][lane]["16"]
        lines.append(
            f"| {lane} | {values['384:256']['paired_effect']:+.8f} | {values['1200:1200']['paired_effect']:+.8f} |"
        )
    lines += ["", "## Follow-Up", "", result["recommended_follow_up"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr241_optimizer_isolation")
    )
    parser.add_argument(
        "--source-workdir",
        type=Path,
        default=Path("/tmp/azlite_onpolicy_shadow_replay"),
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if sha256_file(args.canonical_suite) != SUITE_SHA:
        raise RuntimeError("invariant_failure: canonical-suite mismatch")
    if (
        sha256_file(P1_CHECKPOINT) != P1_CHECKPOINT_SHA
        or sha256_file(A16_SNAPSHOT) != A16_SNAPSHOT_SHA
    ):
        raise RuntimeError("invariant_failure: frozen checkpoint mismatch")
    rows_by_lane, provenance = load_rows(args.source_workdir, args.canonical_suite)
    snapshot = torch.load(A16_SNAPSHOT, map_location="cpu", weights_only=False)
    pristine_model, pristine_optimizer = immutable_initial_state(snapshot)
    initial_optimizer_sha = optimizer_state_sha256(pristine_optimizer)
    p1_model = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1_model, P1_CHECKPOINT)
    parent = {
        name: value.detach().cpu().clone()
        for name, value in p1_model.state_dict().items()
    }

    order_a, states_a = train_order(
        ORDER_A, rows_by_lane, pristine_model, pristine_optimizer, parent
    )
    order_b, _states_b = train_order(
        ORDER_B, rows_by_lane, pristine_model, pristine_optimizer, parent
    )
    compare_runs(order_a, order_b, "lane order")
    repeated_ordinary = repeated_lane_check(
        "ordinary_onpolicy",
        rows_by_lane["ordinary_onpolicy"],
        pristine_model,
        pristine_optimizer,
        parent,
    )
    repeated_shadow075 = repeated_lane_check(
        "shadow075_onpolicy",
        rows_by_lane["shadow075_onpolicy"],
        pristine_model,
        pristine_optimizer,
        parent,
    )
    if not repeated_ordinary or not repeated_shadow075:
        raise RuntimeError(
            "optimizer_isolation_invariant_failure: repeated lane nonidentity"
        )
    if optimizer_state_sha256(pristine_optimizer) != initial_optimizer_sha:
        raise RuntimeError("invariant_failure: pristine optimizer changed")

    artifacts = write_artifacts(args.workdir, states_a)
    diagnostics = {
        lane: {
            str(step): distillation._frozen_diagnostic(
                artifact,
                P1_CHECKPOINT.parent / "artifact",
                read_jsonl(args.source_workdir / "p1_reference.jsonl"),
            )
            for step, artifact in checkpoints.items()
            if step == 16
        }
        for lane, checkpoints in artifacts.items()
    }
    arena = arena_summary(
        args.workdir,
        artifacts,
        P1_CHECKPOINT.parent / "artifact",
        args.canonical_suite,
        args.workers,
    )
    classification, follow_up = classify(arena)
    result = {
        "schema": "pr241_optimizer_isolation_reproduction_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
        "frozen_artifacts": {
            "a16_artifact_weights_sha256": A16_WEIGHTS_SHA,
            "a16_snapshot_sha256": A16_SNAPSHOT_SHA,
            "p1_checkpoint_sha256": P1_CHECKPOINT_SHA,
            "canonical_suite_sha256": SUITE_SHA,
        },
        "replay": provenance,
        "initial_optimizer_sha256": initial_optimizer_sha,
        "invariants": {
            "lane_order_invariant": True,
            "repeated_ordinary": repeated_ordinary,
            "repeated_shadow075": repeated_shadow075,
            "pristine_optimizer_unchanged": optimizer_state_sha256(pristine_optimizer)
            == initial_optimizer_sha,
        },
        "order_a": order_a,
        "order_b": order_b,
        "historical_checkpoint_comparison": historical_comparison(
            args.source_workdir, states_a
        ),
        "frozen_40_40_diagnostics": diagnostics,
        "canonical_arena": arena,
        "historical_opening_level_comparison": {
            "available": False,
            "reason": "PR #241 retained telemetry only, not opening-level game records.",
        },
        "p0_gate_eligible": False,
        "replay_science": {
            "ordinary_vs_p1_state_jaccard": 1.0,
            "shadow075_vs_p1_state_jaccard": 0.0787,
            "shadow100_vs_p1_state_jaccard": 0.0577,
        },
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.workdir / "report.md").write_text(markdown(result), encoding="utf-8")
    print(classification)


if __name__ == "__main__":
    main()
