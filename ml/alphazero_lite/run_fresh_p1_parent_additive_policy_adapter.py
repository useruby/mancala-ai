#!/usr/bin/env python3
"""Prospective fresh-P1 residual-logit adapter experiment; never promotes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.cumulative_lineage_gate import evaluate as cumulative_gate  # noqa: E402
from ml.alphazero_lite.policy_sublayer_graft import byte_identical, state_hash  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    build_manifest,
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
    write_fixed_npz,
)
from ml.alphazero_lite.run_fresh_p1_checkpoint_selection import (  # noqa: E402
    BETA,
    CHECKPOINT_STEPS,
    FIT_THRESHOLD,
    SEED,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P0_EXPECTED_HASH,
    P1_EXPECTED_NPZ_HASH,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _batch,
    _losses,
    _save_snapshot,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import js  # noqa: E402
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    _cross_entropy,
    _win_draw_loss,
)
from ml.alphazero_lite.train import (  # noqa: E402
    PolicyValueNet,
    apply_trainable_scope,
    checkpoint_from_state_dict,
    input_size_for_encoding,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402

MODEL_TYPE = "residual_v3_parent_additive_policy_adapter"
ADAPTER_KEYS = ("policy_adapter.weight", "policy_adapter.bias")
CONTEXTS = ("384:256", "1200:1200")


def new_model(device: torch.device) -> PolicyValueNet:
    return PolicyValueNet((96, 3), MODEL_TYPE, input_size_for_encoding("kalah_v3")).to(
        device
    )


def output(
    state: dict[str, torch.Tensor], x: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x))
    logits = logits.numpy().astype(np.float64)
    logits[~mask.astype(bool)] = -1e9
    logits -= logits.max(axis=1, keepdims=True)
    policy = np.exp(logits)
    return policy / policy.sum(axis=1, keepdims=True)


def export(state: dict[str, torch.Tensor], out_dir: Path, version: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.npz"
    write_fixed_npz(checkpoint, checkpoint_from_state_dict(state))
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "ml/alphazero_lite/export_artifact.py"),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir / "artifact"),
            "--version",
            version,
            "--model-type",
            MODEL_TYPE,
            "--input-encoding",
            "kalah_v3",
            "--rules-version",
            "kalah_v3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return out_dir / "artifact"


def train(
    manifest: dict[str, Any],
    workdir: Path,
    parent_state: dict[str, torch.Tensor],
    device: torch.device,
    pure_search: bool = False,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    model, parent = new_model(device), new_model(device)
    model.load_state_dict(parent_state)
    parent.load_state_dict(parent_state)
    apply_trainable_scope(model, "policy_adapter_only")
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.0
    )
    snapshots = {
        0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)
    }
    model.train()
    for step, indexes in enumerate(plan[:46], 1):
        batch = _batch([rows[int(i)] for i in source], indexes, device)
        target = (
            batch["p"]
            if pure_search
            else mixed_policy_target(
                batch["p"], incumbent_policy_batch(parent, batch), batch["mask"], BETA
            )
        )
        policy, value = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            snapshots[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return snapshots


def arena(
    candidate: Path,
    opponent: Path,
    context: str,
    workdir: Path,
    role: str,
    workers: int,
) -> dict[str, Any]:
    control = _arena_records(
        workdir / "control", opponent, opponent, context, f"{role}_control", workers
    )
    records = _arena_records(
        workdir / role, candidate, opponent, context, role, workers
    )
    effect = paired_opening_candidate_effect(records, control)
    ci = effect["opening_bootstrap_ci"]
    return {
        "paired_candidate_effect": effect["paired_candidate_effect"],
        "opening_bootstrap_ci": ci,
        "seat_a_effect": effect["p0_effect"],
        "seat_b_effect": effect["p1_effect"],
        "win_draw_loss": _win_draw_loss(records),
        "safe": ci["upper_95"] >= 0.0 or ci["lower_95"] >= -0.03,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_fresh_p1_parent_adapter")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-arenas", action="store_true")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-parent-additive-policy-adapter-summary.json",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    p0 = REPO_ROOT / "model-artifact/current"
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    replay = args.replay or args.workdir / "fresh_p1_self_play.jsonl"
    if (
        sha256_file(p0 / "weights.json") != P0_EXPECTED_HASH
        or sha256_file(p1_checkpoint) != P1_EXPECTED_NPZ_HASH
        or not replay.is_file()
    ):
        raise RuntimeError("immutable parent or fresh replay contract failed")
    parent = new_model(device)
    load_checkpoint_into_model(parent, p1_checkpoint)
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent.state_dict().items()
    }
    replay_audit = args.workdir / "replay_audit.json"
    replay_audit.write_text(
        json.dumps(
            {
                "schema": "azlite_fresh_p1_parent_additive_policy_adapter_v1",
                "replay_sha256": sha256_file(replay),
                "parent_checkpoint_sha256": sha256_file(p1_checkpoint),
                "policy_target_mode": "default",
                "value_target_mode": "default",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    build_manifest(
        rows=read_jsonl(replay),
        workdir=args.workdir,
        current=p0,
        replay=replay,
        seed=SEED,
        epochs=1,
        batch_size=512,
        replay_audit=replay_audit,
    )
    manifest = verify_manifest(args.workdir / "training_manifest.json")
    initial = new_model(device)
    initial.load_state_dict(parent_state)
    x = np.asarray([row["state"] for row in read_jsonl(replay)], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    zero_output_exact = np.array_equal(
        output(parent_state, x[:256], mask[:256]),
        output(initial.state_dict(), x[:256], mask[:256]),
    )
    snapshots = train(manifest, args.workdir / "beta095", parent_state, device)
    pure = train(
        manifest, args.workdir / "pure_search", parent_state, device, pure_search=True
    )
    parent_policy = output(parent_state, x, mask)
    search = np.asarray([row["policy"] for row in read_jsonl(replay)], dtype=np.float64)
    pure_ce = float(np.mean(_cross_entropy(output(pure[46][0], x, mask), search)))
    baseline_ce = float(np.mean(_cross_entropy(parent_policy, search)))
    metrics: dict[str, Any] = {}
    artifacts = {}
    inherited = {}
    for step in CHECKPOINT_STEPS:
        state = snapshots[step][0]
        candidate = output(state, x, mask)
        improvement = baseline_ce - float(np.mean(_cross_entropy(candidate, search)))
        l1 = np.abs(candidate - parent_policy).sum(axis=1)
        metrics[str(step)] = {
            "ce_candidate_search": float(np.mean(_cross_entropy(candidate, search))),
            "ce_candidate_p1": float(np.mean(_cross_entropy(candidate, parent_policy))),
            "ce_candidate_beta095": float(
                np.mean(
                    _cross_entropy(
                        candidate, (1 - BETA) * search + BETA * parent_policy
                    )
                )
            ),
            "search_target_ce_improvement_vs_p1": improvement,
            "fit_fraction": float(improvement / (baseline_ce - pure_ce)),
            "legal_policy_l1": {
                "mean": float(l1.mean()),
                "p50": float(np.percentile(l1, 50)),
                "p90": float(np.percentile(l1, 90)),
                "p95": float(np.percentile(l1, 95)),
                "p99": float(np.percentile(l1, 99)),
                "max": float(l1.max()),
            },
            "mean_js": float(js(parent_policy, candidate).mean()),
            "top1_disagreement": float(
                np.mean(
                    np.argmax(parent_policy, axis=1) != np.argmax(candidate, axis=1)
                )
            ),
        }
        inherited[str(step)] = all(
            byte_identical(state[name], parent_state[name])
            for name in state
            if name not in ADAPTER_KEYS
        )
        artifacts[step] = export(
            state, args.workdir / "artifacts" / f"step_{step:04d}", f"adapter_{step}"
        )
    eligible = [
        step
        for step in CHECKPOINT_STEPS
        if metrics[str(step)]["fit_fraction"] >= FIT_THRESHOLD
    ]
    arena_matrix: dict[str, Any] = {}
    for step in eligible if not args.skip_arenas else []:
        arena_matrix[str(step)] = {"candidate_vs_p1": {}}
        for context in CONTEXTS:
            arena_matrix[str(step)]["candidate_vs_p1"][context] = arena(
                artifacts[step],
                p1_artifact,
                context,
                args.workdir / "arena" / str(step),
                "candidate_vs_p1",
                args.workers,
            )
        if all(
            arena_matrix[str(step)]["candidate_vs_p1"][context]["safe"]
            for context in CONTEXTS
        ):
            arena_matrix[str(step)]["candidate_vs_p0"] = {
                context: arena(
                    artifacts[step],
                    p0,
                    context,
                    args.workdir / "arena" / str(step),
                    "candidate_vs_p0",
                    args.workers,
                )
                for context in CONTEXTS
            }
    invariants = {
        "zero_adapter_output_exact": zero_output_exact,
        "inherited_parameters_bit_identical": all(inherited.values()),
        "all_passed": zero_output_exact and all(inherited.values()),
    }
    summary: dict[str, Any] = {
        "schema": "azlite_fresh_p1_parent_additive_policy_adapter_v1",
        "guardrails": {
            "fresh_self_play_generated": True,
            "promotion": False,
            "beta": BETA,
            "optimizer": "Adam",
            "lr": 1e-5,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "steps": 46,
            "trainable_parameters": list(ADAPTER_KEYS),
        },
        "hashes": {
            "p0": P0_EXPECTED_HASH,
            "p1_checkpoint": P1_EXPECTED_NPZ_HASH,
            "replay": sha256_file(replay),
            "manifest": sha256_file(args.workdir / "training_manifest.json"),
            "candidates": {
                str(step): state_hash(snapshots[step][0]) for step in CHECKPOINT_STEPS
            },
        },
        "invariants": invariants,
        "training_metrics": metrics,
        "eligible_checkpoints": eligible,
        "arena_matrix": arena_matrix,
    }
    safe_steps = [
        step
        for step in eligible
        if "candidate_vs_p0" in arena_matrix.get(str(step), {})
        and all(
            arena_matrix[str(step)]["candidate_vs_p0"][context]["safe"]
            for context in CONTEXTS
        )
    ]
    summary["candidate_vs_p0"] = (
        arena_matrix[str(safe_steps[0])]["candidate_vs_p0"] if safe_steps else {}
    )
    summary["cumulative_lineage_gate"] = cumulative_gate(summary)
    summary["classification"] = (
        "adapter_safe"
        if safe_steps and invariants["all_passed"]
        else "adapter_not_safe_or_insufficient_fit"
        if invariants["all_passed"]
        else "invariant_failure"
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary["classification"])


if __name__ == "__main__":
    main()
