#!/usr/bin/env python3
# ruff: noqa: E402
"""Execute the PR #211 prospective fresh-P1 checkpoint-selection contract."""

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

from ml.alphazero_lite.cumulative_lineage_gate import evaluate as cumulative_gate
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    build_manifest,
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C_PUCT,
    DEFAULT_GAMES,
    DEFAULT_GRAD_CLIP,
    DEFAULT_LR,
    DEFAULT_SIMULATIONS,
    PROBE_SIZE,
    _cross_entropy,
    _win_draw_loss,
    arena_safe,
    assert_legal_distribution,
    compute_dataset_diagnostics,
    generate_fresh_self_play,
    incumbent_policy_batch,
    mixed_policy_target,
    policy_drift_metrics,
)
from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import group_delta
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (
    VALUE_STACK_PREFIXES,
    group_parameters_identical,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (
    P0_EXPECTED_HASH,
    P1_EXPECTED_STATE_HASH,
    P1_EXPECTED_NPZ_HASH,
    dataset_evolution_comparison,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (
    _batch,
    _losses,
    _new_model,
    _save_snapshot,
    export_snapshot_artifacts,
    puct_trajectory,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    decoded_validation_manifest,
    model_outputs,
    stable_hash,
)
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

CHECKPOINT_STEPS = [1, 4, 16, 46]
CONTEXTS = ("384:256", "1200:1200")
BETA = 0.95
SEED = 44
TRAINABLE_SCOPE = "policy_head"
FIT_THRESHOLD = 0.25


def state_hash(state: dict[str, torch.Tensor]) -> str:
    return stable_hash(
        {
            name: value.detach().cpu().numpy().tobytes().hex()
            for name, value in state.items()
        }
    )


def train(
    manifest: dict[str, Any],
    workdir: Path,
    device: torch.device,
    parent_checkpoint: Path,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    """Train only the policy head against legal, renormalized beta targets."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    batches = [
        _batch([rows[int(i)] for i in source], indexes, device) for indexes in plan
    ]
    model, parent = _new_model(device), _new_model(device)
    load_checkpoint_into_model(model, parent_checkpoint)
    load_checkpoint_into_model(parent, parent_checkpoint)
    apply_trainable_scope(model, TRAINABLE_SCOPE)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=0.0)
    saved = {0: _save_snapshot(workdir / "snapshots/step_0000.pt", model, optimizer)}
    model.train()
    for step, batch in enumerate(batches, 1):
        p_parent = incumbent_policy_batch(parent, batch)
        target = mixed_policy_target(batch["p"], p_parent, batch["mask"], BETA)
        assert_legal_distribution(
            target.detach().cpu().numpy(), batch["mask"].detach().cpu().numpy()
        )
        policy, value = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            saved[step] = _save_snapshot(
                workdir / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    if set(saved) != {0, *CHECKPOINT_STEPS}:
        raise RuntimeError("missing prespecified checkpoint")
    return saved


def metrics(
    rows: list[dict[str, Any]],
    snapshots: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]],
    parent: dict[str, torch.Tensor],
    pure_search_ce: float,
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    search = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    parent_policy, _ = model_outputs(parent, x, mask)
    parent_policy = parent_policy.astype(np.float64)
    mixed = (1.0 - BETA) * search + BETA * parent_policy
    mixed = np.where(mask.astype(bool), mixed, 0.0)
    mixed /= mixed.sum(axis=1, keepdims=True)
    parent_search_ce = float(np.mean(_cross_entropy(parent_policy, search)))
    result = {}
    for step in CHECKPOINT_STEPS:
        candidate, _ = model_outputs(snapshots[step][0], x, mask)
        candidate = candidate.astype(np.float64)
        improvement = parent_search_ce - float(
            np.mean(_cross_entropy(candidate, search))
        )
        result[str(step)] = {
            "ce_candidate_search": float(np.mean(_cross_entropy(candidate, search))),
            "ce_candidate_p1": float(np.mean(_cross_entropy(candidate, parent_policy))),
            "ce_candidate_mixed": float(np.mean(_cross_entropy(candidate, mixed))),
            "ce_p1_search": parent_search_ce,
            "search_target_ce_improvement_vs_p1": improvement,
            "fit_fraction": float(improvement / (parent_search_ce - pure_search_ce)),
            "policy_drift_vs_p1": policy_drift_metrics(candidate, parent_policy, mask),
        }
    return result


def pure_search_step46_ce(
    manifest: dict[str, Any],
    probe_rows: list[dict[str, Any]],
    device: torch.device,
    parent_checkpoint: Path,
) -> float:
    """Compute the established fit denominator without saving another candidate."""
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(paths["train_source_indexes"], allow_pickle=False)
    plan = np.load(paths["batch_indexes"], allow_pickle=False)
    model = _new_model(device)
    load_checkpoint_into_model(model, parent_checkpoint)
    apply_trainable_scope(model, TRAINABLE_SCOPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=0.0)
    model.train()
    for indexes in plan[:46]:
        batch = _batch([rows[int(i)] for i in source], indexes, device)
        policy, value = _losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
        optimizer.step()
    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    policy, _ = model_outputs(model.state_dict(), x, mask)
    search = np.asarray([row["policy"] for row in probe_rows], dtype=np.float64)
    return float(np.mean(_cross_entropy(policy, search)))


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
    records = _arena_records(
        workdir / role, candidate, opponent, context, f"{role}_candidate", workers
    )
    effect = paired_opening_candidate_effect(records, control)
    return {
        "paired_candidate_effect": effect["paired_candidate_effect"],
        "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
        "seat_a_effect": effect["p0_effect"],
        "seat_b_effect": effect["p1_effect"],
        "win_draw_loss": _win_draw_loss(records),
        "safe": arena_safe(effect),
        "orientation": "candidate_minus_p1"
        if role == "candidate_vs_p1"
        else "candidate_minus_p0",
    }


def selection(summary: dict[str, Any]) -> dict[str, Any]:
    decisions = {}
    selected = None
    for step in CHECKPOINT_STEPS:
        key = str(step)
        cells = summary["arena_matrix"].get(key, {})
        all_safe = all(
            cells.get(compare, {}).get(context, {}).get("safe") is True
            for compare in ("candidate_vs_p1", "candidate_vs_p0")
            for context in CONTEXTS
        )
        gain = (
            cells.get("candidate_vs_p1", {})
            .get("1200:1200", {})
            .get("opening_bootstrap_ci", {})
            .get("lower_95", 0.0)
            > 0.0
        )
        decisions[key] = {
            "fit_eligible": key in summary["eligible_checkpoints"],
            "all_four_safe": all_safe,
            "high_budget_incremental_gain": gain,
            "selection_eligible": all_safe and gain,
        }
        if selected is None and decisions[key]["selection_eligible"]:
            selected = step
    return {
        "protocol": "pr211_earliest_safe_positive_high_budget",
        "checkpoints": decisions,
        "selected_checkpoint": selected,
    }


def classify(summary: dict[str, Any]) -> dict[str, str]:
    if not all(summary["invariants"].values()):
        return {
            "label": "invariant_failure",
            "next_experiment": "repair the failed contract before rerunning",
        }
    eligible = summary["eligible_checkpoints"]
    if not eligible:
        return {
            "label": "insufficient_learning",
            "next_experiment": "investigate why the fixed policy-head update did not reach the preregistered fit threshold",
        }
    decision = summary["selection_gate"]
    if (
        decision["selected_checkpoint"] is not None
        and summary["cumulative_lineage_gate"]["passed"]
    ):
        return {
            "label": "fresh_lineage_checkpoint_selected",
            "next_experiment": "generate one additional fresh self-play generation from the selected candidate inside an isolated lineage and repeat the identical gate",
        }
    high_gain = [
        step
        for step in eligible
        if decision["checkpoints"][step]["high_budget_incremental_gain"]
    ]
    if high_gain and all(
        not decision["checkpoints"][step]["all_four_safe"] for step in high_gain
    ):
        p0_unsafe = any(
            not summary["arena_matrix"][step]["candidate_vs_p0"][context]["safe"]
            for step in high_gain
            for context in CONTEXTS
        )
        if p0_unsafe:
            return {
                "label": "fresh_parent_gain_p0_blocked",
                "next_experiment": "investigate policy-head parameterization / cumulative drift rather than more checkpoint tuning",
            }
    if any(decision["checkpoints"][step]["all_four_safe"] for step in eligible):
        return {
            "label": "fresh_safe_gain_unproven",
            "next_experiment": "increase evaluation power before changing the fixed iteration rule",
        }
    return {
        "label": "no_safe_learning_checkpoint",
        "next_experiment": "investigate why the existing policy-head update is not compositionally safe",
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Fresh P1 Checkpoint-Selection Execution",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        f"**Next experiment:** {summary['classification']['next_experiment']}",
        "",
        "## Lineage And Contract",
        "",
    ]
    for key, value in summary["lineage"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Training Eligibility",
        "",
        "| Step | CE(search) | CE(P1) | CE(mixed) | Search improvement | Fit fraction | L1 mean/p99/max | Top-1 | Trunk/value/policy drift | Eligible |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |",
    ]
    for step, value in summary["training_metrics"].items():
        drift, param = value["policy_drift_vs_p1"], summary["parameter_drift"][step]
        lines.append(
            f"| {step} | {value['ce_candidate_search']:.6f} | {value['ce_candidate_p1']:.6f} | {value['ce_candidate_mixed']:.6f} | {value['search_target_ce_improvement_vs_p1']:+.6f} | {value['fit_fraction']:.4f} | {drift['legal_l1_mean']:.6f}/{drift['legal_l1_p99']:.6f}/{drift['legal_l1_max']:.6f} | {drift['top1_change_rate']:.4f} | {param['trunk']:.2e}/{param['value_head']:.2e}/{param['policy_head']:.6f} | {str(step) in summary['eligible_checkpoints']} |"
        )
    lines += [
        "",
        "## P1 Vs P0 Reference Control",
        "",
        "| Budget | Effect | 95% CI | Seat A | Seat B | W/D/L |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for context, value in summary["p1_vs_p0_reference"].items():
        ci, wdl = value["opening_bootstrap_ci"], value["win_draw_loss"]
        lines.append(
            f"| {context} | {value['paired_candidate_effect']:+.4f} | "
            f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
            f"{value['seat_a_effect']:+.4f} | {value['seat_b_effect']:+.4f} | "
            f"{wdl['wins']}/{wdl['draws']}/{wdl['losses']} |"
        )
    lines += [
        "",
        "## Arena Matrix",
        "",
        "| Step | Match | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Safe |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for step, matches in summary["arena_matrix"].items():
        for match, contexts in matches.items():
            for context, value in contexts.items():
                ci, wdl = value["opening_bootstrap_ci"], value["win_draw_loss"]
                lines.append(
                    f"| {step} | {match} | {context} | {value['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {value['seat_a_effect']:+.4f} | {value['seat_b_effect']:+.4f} | {wdl['wins']}/{wdl['draws']}/{wdl['losses']} | {value['safe']} |"
                )
    lines += [
        "",
        "## Selection And Cumulative Gate",
        "",
        f"- Selected checkpoint: `{summary['selection_gate']['selected_checkpoint']}`",
        f"- Cumulative lineage gate: `{json.dumps(summary['cumulative_lineage_gate'])}`",
        "",
        "## Search Diagnostics",
        "",
        "| Step | Reference | Budget | Move changes | Visit JS | Q-rank changes | Root-value delta | Visit-margin delta |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step, refs in summary["search_diagnostics"].items():
        for reference, contexts in refs.items():
            for context, value in contexts.items():
                lines.append(
                    f"| {step} | {reference} | {context} | "
                    f"{value['selected_move_change_rate']:.4f} | {value['visit_js']:.6f} | "
                    f"{value['child_q_rank_change']:+.4f} | {value['root_value_delta']:+.6f} | "
                    f"{value['visit_margin']:+.4f} |"
                )
    lines += [
        "",
        "## Replay Comparison",
        "",
        "Diagnostic-only comparison against the PR #204 replay is retained in the JSON summary, including entropy, L1 tails, top-1 disagreement, game length, and outcome distribution.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_checkpoint_selection"),
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--pr204-replay",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor/gen2_self_play.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-checkpoint-selection-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-checkpoint-selection-results.md",
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    p0 = REPO_ROOT / "model-artifact/current"
    p1_artifact, p1_checkpoint, p1_weights_sha, p1_checkpoint_sha, p1_state_sha = (
        reconstruct_and_freeze_p1(
            p0, args.p1_workdir, args.p1_workdir / "fresh_self_play.jsonl", args.workers
        )
    )
    if (
        p1_state_sha != P1_EXPECTED_STATE_HASH
        or p1_checkpoint_sha != P1_EXPECTED_NPZ_HASH
    ):
        raise RuntimeError("P1 is not the frozen PR #203 parent")
    p0_sha = sha256_file(p0 / "weights.json")
    if p0_sha != P0_EXPECTED_HASH:
        raise RuntimeError("model-artifact/current is not immutable P0")
    replay = args.workdir / "fresh_p1_self_play.jsonl"
    if args.resume and replay.is_file():
        existing_rows = read_jsonl(replay)
        selfplay = {
            "games_requested": DEFAULT_GAMES,
            "games_generated": len({int(row["game_index"]) for row in existing_rows}),
            "positions_generated": len(existing_rows),
            "seed": SEED,
            "simulations": DEFAULT_SIMULATIONS,
            "c_puct": DEFAULT_C_PUCT,
            "player_mode": "puct",
            "input_encoding": "kalah_v3",
            "policy_target_mode": "default",
            "value_target_mode": "default",
            "policy_target_noise_mode": "noisy",
            "tree_reuse_enabled": True,
            "replay_path": str(replay),
            "replay_sha256": sha256_file(replay),
            "checkpoint_npz_sha256": sha256_file(p1_checkpoint),
        }
    else:
        selfplay = generate_fresh_self_play(
            out_path=replay,
            checkpoint_npz=p1_checkpoint,
            games=DEFAULT_GAMES,
            seed=SEED,
            simulations=DEFAULT_SIMULATIONS,
            c_puct=DEFAULT_C_PUCT,
            workers=args.workers,
        )
    rows = read_jsonl(replay)
    if selfplay["games_generated"] != DEFAULT_GAMES:
        raise RuntimeError("fresh generation did not contain exactly 700 games")
    audit = args.workdir / "replay_audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "azlite_fresh_p1_checkpoint_selection_v1",
                "replay_sha256": selfplay["replay_sha256"],
                "parent_state_hash": p1_state_sha,
                "policy_target_mode": "default",
                "value_target_mode": "default",
            },
            indent=2,
        )
        + "\n"
    )
    build_manifest(
        rows=rows,
        workdir=args.workdir,
        current=p0,
        replay=replay,
        seed=SEED,
        epochs=1,
        batch_size=DEFAULT_BATCH_SIZE,
        replay_audit=audit,
    )
    manifest = verify_manifest(args.workdir / "training_manifest.json")
    configure_determinism(device, SEED)
    snapshots = train(manifest, args.workdir / "beta_095", device, p1_checkpoint)
    p1_model = _new_model(device)
    load_checkpoint_into_model(p1_model, p1_checkpoint)
    p1_state = p1_model.state_dict()
    invariant = {
        "p0_unchanged_before_arena": sha256_file(p0 / "weights.json") == p0_sha,
        "p1_state_verified": state_hash(p1_state) == P1_EXPECTED_STATE_HASH,
        "trunk_unchanged": all(
            group_parameters_identical(
                snapshots[step][0], p1_state, ("input_layer.", "residual_layers.")
            )
            for step in CHECKPOINT_STEPS
        ),
        "value_stack_unchanged": all(
            group_parameters_identical(
                snapshots[step][0], p1_state, VALUE_STACK_PREFIXES
            )
            for step in CHECKPOINT_STEPS
        ),
    }
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    probe, probe_manifest = decoded_validation_manifest(rows, indexes)
    probe_rows = [rows[index] for index in probe_manifest["source_indexes"]]
    pure_search_ce = pure_search_step46_ce(manifest, probe_rows, device, p1_checkpoint)
    training_metrics = metrics(probe_rows, snapshots, p1_state, pure_search_ce)
    artifacts = export_snapshot_artifacts(snapshots, args.workdir / "beta_095")
    eligible = [
        step
        for step in map(str, CHECKPOINT_STEPS)
        if training_metrics[step]["fit_fraction"] >= FIT_THRESHOLD
    ]
    arena_matrix: dict[str, Any] = {}
    for step in eligible:
        arena_matrix[step] = {"candidate_vs_p1": {}, "candidate_vs_p0": {}}
        for comparison, opponent in (
            ("candidate_vs_p1", p1_artifact),
            ("candidate_vs_p0", p0),
        ):
            for context in CONTEXTS:
                arena_matrix[step][comparison][context] = arena_entry(
                    artifacts[int(step)],
                    opponent,
                    context,
                    args.workdir / "arena" / step,
                    comparison,
                    args.arena_workers,
                )
    p1_reference = {
        context: arena_entry(
            p1_artifact,
            p0,
            context,
            args.workdir / "arena" / "p1_reference",
            "p0",
            args.arena_workers,
        )
        for context in CONTEXTS
    }
    diagnostics = {}
    probe_hash = stable_hash(
        {
            "replay": selfplay["replay_sha256"],
            "indexes": probe_manifest["source_indexes"],
        }
    )
    for step in eligible:
        diagnostics[step] = {}
        for name, opponent in (("vs_p1", p1_artifact), ("vs_p0", p0)):
            diagnostics[step][name] = puct_trajectory(
                probe[:PROBE_SIZE],
                {0: opponent, int(step): artifacts[int(step)]},
                args.workdir / "puct" / step / name,
                probe_hash,
                contexts=CONTEXTS,
            )["metrics"][step]
    candidate_vs_p0 = arena_matrix.get(
        str(
            selection({"arena_matrix": arena_matrix, "eligible_checkpoints": eligible})[
                "selected_checkpoint"
            ]
        ),
        {},
    ).get("candidate_vs_p0", {})
    summary: dict[str, Any] = {
        "schema": "azlite_fresh_p1_checkpoint_selection_v1",
        "guardrails": {
            "retrospective_q75_mask": False,
            "p2_used": False,
            "runtime_promotion": False,
            "model_artifact_current_mutated": False,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "beta": BETA,
            "trainable_scope": TRAINABLE_SCOPE,
        },
        "lineage": {
            "p0_weights_sha256": p0_sha,
            "p1_weights_sha256": p1_weights_sha,
            "p1_checkpoint_sha256": p1_checkpoint_sha,
            "p1_state_hash": p1_state_sha,
            "generation_seed": SEED,
            "fresh_replay_sha256": selfplay["replay_sha256"],
            "fresh_game_count": selfplay["games_generated"],
            "fresh_position_count": selfplay["positions_generated"],
            "selfplay_config": selfplay,
        },
        "invariants": invariant,
        "manifest_sha256": sha256_file(args.workdir / "training_manifest.json"),
        "batch_plan_sha256": sha256_file(paths["batch_indexes"]),
        "fit_fraction_denominator": {
            "kind": "in_memory_pure_search_step46_ce",
            "ce": pure_search_ce,
            "saved_checkpoint": False,
            "candidate": False,
        },
        "checkpoint_state_hashes": {
            str(step): state_hash(snapshots[step][0]) for step in CHECKPOINT_STEPS
        },
        "training_metrics": training_metrics,
        "parameter_drift": {
            str(step): group_delta(snapshots[step][0], p1_state)
            for step in CHECKPOINT_STEPS
        },
        "eligible_checkpoints": eligible,
        "p1_vs_p0_reference": p1_reference,
        "arena_matrix": arena_matrix,
        "search_diagnostics": diagnostics,
        "replay_comparison_vs_pr204": dataset_evolution_comparison(
            compute_dataset_diagnostics(rows, p1_state),
            compute_dataset_diagnostics(read_jsonl(args.pr204_replay), p1_state),
        ),
        "candidate_vs_p0": candidate_vs_p0,
    }
    summary["selection_gate"] = selection(summary)
    selected = summary["selection_gate"]["selected_checkpoint"]
    summary["candidate_vs_p0"] = (
        arena_matrix[str(selected)]["candidate_vs_p0"] if selected is not None else {}
    )
    summary["cumulative_lineage_gate"] = cumulative_gate(summary)
    invariant["p0_unchanged_after_execution"] = (
        sha256_file(p0 / "weights.json") == p0_sha
    )
    summary["classification"] = classify(summary)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(render(summary))
    print(json.dumps(summary["classification"], indent=2))


if __name__ == "__main__":
    main()
