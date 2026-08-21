#!/usr/bin/env python3
"""Train matched PR #214 A16 target-source lanes without new self-play."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import state_round_trips_kalah_v3  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (  # noqa: E402
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    arena_records,
    state_hash,
    _suite,
)
from ml.alphazero_lite.run_fresh_p1_adapter_teacher_quality_audit import (  # noqa: E402
    canonical_hash,
    search_target,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (  # noqa: E402
    ADAPTER_KEYS,
    BETA,
    export,
    new_model,
    output,
)
from ml.alphazero_lite.run_fresh_p1_checkpoint_selection import SEED  # noqa: E402
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    _cross_entropy,
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _batch,
    _losses,
    _save_snapshot,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

LANES = ("stored384", "clean384", "clean1200")
CONTEXTS = ("384:256", "1200:1200")
STEPS = 46
LR = 1e-5


def target_field(lane: str) -> str | None:
    """Return the clean-cache field for a lane, or None for stored replay policy."""
    if lane == "stored384":
        return None
    if lane in ("clean384", "clean1200"):
        return lane
    raise ValueError(f"unknown lane: {lane}")


def cache_entry_is_valid(entry: dict[str, Any], row: dict[str, Any]) -> bool:
    """Accept only entries bound to this replay row and both exact search targets."""
    return (
        entry.get("schema") == "azlite_pr214_full_replay_clean_target_v1"
        and entry.get("state_hash") == canonical_hash(row["state"])
        and all(
            name in entry and isinstance(entry[name], dict) and "policy" in entry[name]
            for name in ("clean384", "clean1200")
        )
    )


def _target_chunk(
    artifact_path: str, chunk: list[tuple[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Build one deterministic cache chunk in an isolated P1 evaluator process."""
    evaluator = ArtifactEvaluator(Path(artifact_path))
    return [
        {
            "schema": "azlite_pr214_full_replay_clean_target_v1",
            "replay_index": index,
            "state_hash": canonical_hash(row["state"]),
            "clean384": search_target(
                evaluator, row, canonical_hash(row["state"]), 384
            ),
            "clean1200": search_target(
                evaluator, row, canonical_hash(row["state"]), 1200
            ),
        }
        for index, row in chunk
    ]


def build_clean_target_cache(
    rows: list[dict[str, Any]], artifact_path: Path, path: Path, workers: int
) -> dict[int, dict[str, Any]]:
    """Resume an append-only, full-replay P1 same-state target cache."""
    cached: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for entry in read_jsonl(path):
            index = int(entry.get("replay_index", -1))
            if 0 <= index < len(rows) and cache_entry_is_valid(entry, rows[index]):
                cached[index] = entry
    missing = [index for index in range(len(rows)) if index not in cached]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        worker_count = min(max(1, workers), len(missing))
        chunks = [
            [(index, rows[index]) for index in missing[offset::worker_count]]
            for offset in range(worker_count)
        ]
        with path.open("a", encoding="utf-8") as handle:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count
            ) as pool:
                futures = [
                    pool.submit(_target_chunk, str(artifact_path), chunk)
                    for chunk in chunks
                ]
                # Write in chunk order so a clean run produces the same cache hash.
                for completed, future in enumerate(futures, start=1):
                    for entry in future.result():
                        handle.write(json.dumps(entry, sort_keys=True) + "\n")
                        cached[int(entry["replay_index"])] = entry
                    print(
                        f"[targets] completed chunk {completed}/{worker_count}",
                        flush=True,
                    )
    if len(cached) != len(rows):
        raise RuntimeError("clean target cache does not cover the full replay")
    return cached


def lane_invariants(
    snapshots: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]],
    parent_state: dict[str, torch.Tensor],
) -> dict[str, bool]:
    """Ensure all lanes differ only through the intended adapter training targets."""
    return {
        "three_matched_lanes": tuple(snapshots) == LANES,
        "same_initial_state": all(
            state_hash(lane_snapshots[0][0]) == state_hash(parent_state)
            for lane_snapshots in snapshots.values()
        ),
        "non_adapter_parameters_bit_identical": all(
            all(
                torch.equal(state[name], parent_state[name])
                for name in state
                if name not in ADAPTER_KEYS
            )
            for lane_snapshots in snapshots.values()
            for state, _optimizer in lane_snapshots.values()
        ),
        "all_passed": False,
    }


def train_lane(
    lane: str,
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    targets: dict[int, dict[str, Any]],
    parent_state: dict[str, torch.Tensor],
    workdir: Path,
    device: torch.device,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    """Replay the frozen PR #214 batch plan with one and only one target source."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    model, parent = new_model(device), new_model(device)
    model.load_state_dict(parent_state)
    parent.load_state_dict(parent_state)
    apply_trainable_scope(model, "policy_adapter_only")
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.0
    )
    snapshots = {
        0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)
    }
    model.train()
    for step, batch_indexes in enumerate(plan[:STEPS], 1):
        batch = _batch([rows[int(index)] for index in source], batch_indexes, device)
        field = target_field(lane)
        teacher = (
            batch["p"]
            if field is None
            else torch.from_numpy(
                np.asarray(
                    [
                        targets[int(source[int(index)])][field]["policy"]
                        for index in batch_indexes
                        if int(index) >= 0
                    ],
                    dtype=np.float32,
                )
            ).to(device)
        )
        target = mixed_policy_target(
            teacher, incumbent_policy_batch(parent, batch), batch["mask"], BETA
        )
        policy, value = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step == STEPS:
            snapshots[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    return snapshots


def fit_metrics(
    state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    targets: dict[int, dict[str, Any]],
    lane: str,
) -> dict[str, float]:
    """Measure replay fit and legal policy L1 against the lane's own teacher."""
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    parent_policy, candidate = output(parent_state, x, mask), output(state, x, mask)
    field = target_field(lane)
    teacher = np.asarray(
        [
            row["policy"] if field is None else targets[index][field]["policy"]
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )
    baseline_ce = float(np.mean(_cross_entropy(parent_policy, teacher)))
    candidate_ce = float(np.mean(_cross_entropy(candidate, teacher)))
    l1 = np.abs(candidate - parent_policy).sum(axis=1)
    return {
        "teacher_cross_entropy": candidate_ce,
        "teacher_cross_entropy_improvement_vs_p1": baseline_ce - candidate_ce,
        "legal_policy_l1_mean": float(l1.mean()),
        "legal_policy_l1_p95": float(np.percentile(l1, 95)),
    }


def report(summary: dict[str, Any]) -> str:
    rows = [
        "# PR #214 Adapter Teacher Target Retrain",
        "",
        "Three matched A16 lanes use only different policy target sources. No trajectories are generated and no promotion occurs.",
        "",
        "## Invariants",
        "",
        "```json",
        json.dumps(summary["invariants"], indent=2, sort_keys=True),
        "```",
        "",
        "## Network Fit",
        "",
        "| Lane | CE improvement vs P1 | Mean legal L1 | P95 legal L1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for lane in LANES:
        metrics = summary["lanes"][lane]["network_fit"]
        rows.append(
            f"| {lane} | {metrics['teacher_cross_entropy_improvement_vs_p1']:.6f} | {metrics['legal_policy_l1_mean']:.6f} | {metrics['legal_policy_l1_p95']:.6f} |"
        )
    rows.extend(
        [
            "",
            "## Canonical Arenas",
            "",
            "| Lane | 384:256 effect | 1200:1200 effect |",
            "| --- | ---: | ---: |",
        ]
    )
    for lane in LANES:
        arena = summary["lanes"][lane]["arena"]
        rows.append(
            f"| {lane} | {arena['384:256']['paired_candidate_effect']:.4f} | {arena['1200:1200']['paired_candidate_effect']:.4f} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr214_teacher_target_retrain")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-teacher-target-retrain-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-teacher-target-retrain-results.md",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    determinism = configure_determinism(device, SEED)
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    p1_artifact = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_checkpoint = args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    manifest = verify_manifest(args.adapter_workdir / "training_manifest.json")
    rows = read_jsonl(replay)
    parent, a16 = new_model(device), new_model(device)
    load_checkpoint_into_model(parent, p1_checkpoint)
    load_checkpoint_into_model(a16, a16_checkpoint)
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    invariants = {
        "p1_checkpoint_hash": sha256_file(p1_checkpoint) == P1_CHECKPOINT_SHA,
        "a16_state_hash": state_hash(a16.state_dict()) == A16_STATE_SHA,
        "replay_hash": sha256_file(replay) == REPLAY_SHA,
        "full_replay_state_round_trip": all(
            state_round_trips_kalah_v3(row["state"]) for row in rows
        ),
        "pr214_batch_plan": (
            Path(manifest["replay_path"]).resolve() == replay.resolve()
            and len(plan) == STEPS
            and len(source) > 0
            and bool(np.all((source >= 0) & (source < len(rows))))
            and bool(np.all((plan == -1) | ((plan >= 0) & (plan < len(source)))))
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"immutable PR #214 contract failed: {invariants}")
    targets = build_clean_target_cache(
        rows, p1_artifact, args.workdir / "clean_p1_targets.jsonl", args.workers
    )
    parent_state = {
        name: value.detach().cpu().clone()
        for name, value in parent.state_dict().items()
    }
    snapshots = {
        lane: train_lane(
            lane, manifest, rows, targets, parent_state, args.workdir / lane, device
        )
        for lane in LANES
    }
    lane_checks = lane_invariants(snapshots, parent_state)
    lane_checks["all_passed"] = all(
        value for name, value in lane_checks.items() if name != "all_passed"
    )
    if not lane_checks["all_passed"]:
        raise RuntimeError(f"matched-lane invariant failed: {lane_checks}")
    _suite_rows, suite_hash = _suite()
    lanes: dict[str, Any] = {}
    for lane in LANES:
        state = snapshots[lane][STEPS][0]
        artifact = export(
            state, args.workdir / lane / "artifact", f"pr214_{lane}_step_{STEPS}"
        )
        arena = {}
        for context in CONTEXTS:
            control = arena_records(
                workdir=args.workdir / "arena_control",
                challenger=p1_artifact,
                current=p1_artifact,
                context=context,
                role="p1_control",
                workers=args.workers,
                suite_hash=suite_hash,
            )
            records = arena_records(
                workdir=args.workdir / "arena",
                challenger=artifact,
                current=p1_artifact,
                context=context,
                role=f"{lane}_vs_p1",
                workers=args.workers,
                suite_hash=suite_hash,
            )
            effect = paired_opening_candidate_effect(records, control)
            arena[context] = {
                key: effect[key]
                for key in (
                    "paired_candidate_effect",
                    "opening_bootstrap_ci",
                    "p0_effect",
                    "p1_effect",
                )
            }
        lanes[lane] = {
            "target_source": lane,
            "candidate_state_hash": state_hash(state),
            "network_fit": fit_metrics(state, parent_state, rows, targets, lane),
            "arena": arena,
        }
    summary = {
        "schema": "azlite_pr214_adapter_teacher_target_retrain_v1",
        "guardrails": {
            "fresh_self_play_generated": False,
            "trajectory_generation": False,
            "promotion": False,
            "beta": BETA,
            "optimizer": "Adam",
            "lr": LR,
            "steps": STEPS,
            "trainable_parameters": list(ADAPTER_KEYS),
        },
        "hashes": {
            "p1_checkpoint": sha256_file(p1_checkpoint),
            "a16_state": state_hash(a16.state_dict()),
            "replay": sha256_file(replay),
            "batch_manifest": sha256_file(
                args.adapter_workdir / "training_manifest.json"
            ),
            "clean_target_cache": sha256_file(args.workdir / "clean_p1_targets.jsonl"),
        },
        "search_contract": {
            "teacher": "P1",
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_noise": False,
            "seed": "sha256(pr214-teacher-audit:encoded-state-hash)",
            "temperature": "1.0 before move index 10; 0.1 afterwards",
            "target_mode": "default",
            "simulations": [384, 1200],
        },
        "determinism": determinism,
        "invariants": {**invariants, **lane_checks},
        "lanes": lanes,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(report(summary), encoding="utf-8")
    print("complete")


if __name__ == "__main__":
    main()
