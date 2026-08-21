#!/usr/bin/env python3
"""Retrain PR #212 with each residual_v3 policy sublayer isolated."""

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

from ml.alphazero_lite.policy_sublayer_graft import (  # noqa: E402
    HIDDEN_KEYS,
    READOUT_KEYS,
    byte_identical,
    state_hash,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_p1_checkpoint_selection import (  # noqa: E402
    BETA,
    CHECKPOINT_STEPS,
    FIT_THRESHOLD,
    SEED,
    arena_entry,
    metrics,
    pure_search_step46_ce,
)
from ml.alphazero_lite.run_fresh_p1_policy_sublayer_decomposition import (  # noqa: E402
    EXPECTED_CANDIDATE_HASHES,
    PRIMARY_CONTEXT,
    HIGH_CONTEXT,
    export_artifact,
    search_metrics,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P0_EXPECTED_HASH,
    P1_EXPECTED_NPZ_HASH,
    P1_EXPECTED_STATE_HASH,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    _batch,
    _losses,
    _new_model,
    _save_snapshot,
    export_snapshot_artifacts,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    model_outputs,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

LANES = {
    "full_policy": "policy_head",
    "hidden_only": "policy_hidden_only",
    "readout_only": "policy_readout_only",
}
POLICY_FAMILIES = {"hidden": HIDDEN_KEYS, "readout": READOUT_KEYS}
PROTECTED_PREFIXES = (
    "input_layer.",
    "residual_layers.",
    "value_hidden_layer.",
    "value_head.",
)


def _tensor_delta(
    state: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> torch.Tensor:
    return torch.cat(
        [(state[key].double() - parent[key].double()).flatten() for key in keys]
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else None


def _parameter_drift(
    state: dict[str, torch.Tensor], parent: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        family: float(torch.linalg.vector_norm(_tensor_delta(state, parent, keys)))
        for family, keys in POLICY_FAMILIES.items()
    } | {
        family: float(
            torch.linalg.vector_norm(
                torch.cat(
                    [
                        (state[key].double() - parent[key].double()).flatten()
                        for key in state
                        if key.startswith(prefixes)
                    ]
                )
            )
        )
        for family, prefixes in (
            ("trunk", PROTECTED_PREFIXES[:2]),
            ("value_stack", PROTECTED_PREFIXES[2:]),
        )
    }


def _frozen_contract(
    state: dict[str, torch.Tensor], parent: dict[str, torch.Tensor], lane: str
) -> bool:
    trainable = set(
        HIDDEN_KEYS
        if lane == "hidden_only"
        else READOUT_KEYS
        if lane == "readout_only"
        else HIDDEN_KEYS + READOUT_KEYS
    )
    return all(
        byte_identical(state[name], parent[name])
        for name in state
        if name not in trainable
    )


def gradient_parity(
    batch: dict[str, torch.Tensor], checkpoint: Path, device: torch.device
) -> dict[str, Any]:
    models = {lane: _new_model(device) for lane in LANES}
    for lane, model in models.items():
        load_checkpoint_into_model(model, checkpoint)
        apply_trainable_scope(model, LANES[lane])
        policy, value = _losses(model, batch)
        (policy + value).backward()
    result: dict[str, Any] = {
        "trunk_gradients_absent": True,
        "value_gradients_absent": True,
    }
    for lane, model in models.items():
        for name, parameter in model.named_parameters():
            if name.startswith(PROTECTED_PREFIXES):
                result["trunk_gradients_absent"] &= parameter.grad is None
                result["value_gradients_absent"] &= parameter.grad is None
    for family, lane in (("hidden", "hidden_only"), ("readout", "readout_only")):
        keys = POLICY_FAMILIES[family]
        differences = [
            float(
                torch.max(
                    torch.abs(
                        dict(models[lane].named_parameters())[key].grad
                        - dict(models["full_policy"].named_parameters())[key].grad
                    )
                )
            )
            for key in keys
        ]
        result[f"{family}_max_abs_gradient_difference"] = max(differences)
        result[f"{family}_gradient_parity"] = bool(
            all(
                torch.allclose(
                    dict(models[lane].named_parameters())[key].grad,
                    dict(models["full_policy"].named_parameters())[key].grad,
                    rtol=1e-6,
                    atol=1e-7,
                )
                for key in keys
            )
        )
    result["all_passed"] = all(
        value is True
        for key, value in result.items()
        if key.endswith(("absent", "parity"))
    )
    return result


def train_lane(
    manifest: dict[str, Any],
    workdir: Path,
    checkpoint: Path,
    lane: str,
    device: torch.device,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]:
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    rows = read_jsonl(Path(manifest["replay_path"]))
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    model, parent = _new_model(device), _new_model(device)
    load_checkpoint_into_model(model, checkpoint)
    load_checkpoint_into_model(parent, checkpoint)
    apply_trainable_scope(model, LANES[lane])
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-5,
        weight_decay=0.0,
    )
    snapshots = {
        0: _save_snapshot(workdir / lane / "snapshots/step_0000.pt", model, optimizer)
    }
    model.train()
    for step, indexes in enumerate(plan[:46], 1):
        batch = _batch([rows[int(index)] for index in source], indexes, device)
        target = mixed_policy_target(
            batch["p"], incumbent_policy_batch(parent, batch), batch["mask"], BETA
        )
        policy, value = _losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        if step in CHECKPOINT_STEPS:
            snapshots[step] = _save_snapshot(
                workdir / lane / f"snapshots/step_{step:04d}.pt", model, optimizer
            )
    if set(snapshots) != {0, *CHECKPOINT_STEPS}:
        raise RuntimeError(f"{lane} did not save the prespecified checkpoints")
    return snapshots


def trajectory_comparison(
    rows: list[dict[str, Any]],
    parent: dict[str, torch.Tensor],
    trained: dict[str, dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]]],
    grafts: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    parent_policy, _ = model_outputs(parent, x, mask)
    result: dict[str, Any] = {}
    for lane, family in (("hidden_only", "hidden"), ("readout_only", "readout")):
        result[lane] = {}
        for step in (16, 46):
            state, graft, keys = (
                trained[lane][step][0],
                grafts[step],
                POLICY_FAMILIES[family],
            )
            candidate_policy, _ = model_outputs(state, x, mask)
            graft_policy, _ = model_outputs(graft, x, mask)
            own_delta, graft_delta = (
                _tensor_delta(state, parent, keys),
                _tensor_delta(graft, parent, keys),
            )
            result[lane][str(step)] = {
                "parameter_delta_cosine": _cosine(
                    own_delta.numpy(), graft_delta.numpy()
                ),
                "parameter_norm_ratio": float(
                    torch.linalg.vector_norm(own_delta)
                    / torch.linalg.vector_norm(graft_delta)
                ),
                "output_policy_delta_cosine": _cosine(
                    (candidate_policy - parent_policy).ravel(),
                    (graft_policy - parent_policy).ravel(),
                ),
                "output_l1_independent_vs_graft": float(
                    np.abs(candidate_policy - graft_policy).sum(axis=1).mean()
                ),
            }
    return result


def classify(summary: dict[str, Any]) -> dict[str, str]:
    if not summary["invariants"]["all_passed"]:
        return {
            "label": "invariant_failure",
            "next_experiment": "repair the failed contract before rerunning",
        }
    safe = {
        lane: all(
            summary["arena"][lane][str(step)].get(context, {}).get("safe")
            for step in (16, 46)
            for context in (PRIMARY_CONTEXT, HIGH_CONTEXT)
        )
        for lane in ("hidden_only", "readout_only")
    }
    fit = {
        lane: all(
            summary["training_metrics"][lane][str(step)]["fit_fraction"]
            >= FIT_THRESHOLD
            for step in (16, 46)
        )
        for lane in safe
    }
    if all(safe[lane] and fit[lane] for lane in safe):
        return {
            "label": "both_single_sublayers_safe",
            "next_experiment": "test sequential alternating training, but do not implement it here",
        }
    if safe["hidden_only"] and fit["hidden_only"] and not safe["readout_only"]:
        return {
            "label": "hidden_only_training_viable",
            "next_experiment": "run one prospective fresh self-play generation using hidden-only training plus the committed checkpoint/promotion gate",
        }
    if safe["readout_only"] and fit["readout_only"] and not safe["hidden_only"]:
        return {
            "label": "readout_only_training_viable",
            "next_experiment": "freeze policy features and adapt only the six-action readout",
        }
    if not any(safe[lane] and fit[lane] for lane in safe):
        return {
            "label": "both_single_sublayers_unsafe",
            "next_experiment": "move to a parent-preserving additive policy adapter rather than further mutation of the existing head",
        }
    return {
        "label": "inconclusive",
        "next_experiment": "inspect the isolated trajectories without changing the fixed training recipe",
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# PR #212 Isolated Policy Sublayer Training",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        f"**Recommended follow-up:** {summary['classification']['next_experiment']}",
        "",
        "## Training Metrics",
        "",
        "| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Improvement | Fit | L1 mean/p99/max | JS | Top-1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for lane, values in summary["training_metrics"].items():
        for step, metric in values.items():
            drift = metric["policy_drift_vs_p1"]
            lines.append(
                f"| {lane} | {step} | {metric['ce_candidate_search']:.6f} | {metric['ce_candidate_p1']:.6f} | {metric['ce_candidate_mixed']:.6f} | {metric['search_target_ce_improvement_vs_p1']:+.6f} | {metric['fit_fraction']:.4f} | {drift['legal_l1_mean']:.6f}/{drift['legal_l1_p99']:.6f}/{drift['legal_l1_max']:.6f} | {drift['legal_js_mean']:.2e} | {drift['top1_change_rate']:.4f} |"
            )
    lines += [
        "",
        "## Arena Matrix",
        "",
        "| Lane | Step | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Safe | Recovery |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for lane, steps in summary["arena"].items():
        for step, contexts in steps.items():
            for context, value in contexts.items():
                ci, wdl = value["opening_bootstrap_ci"], value["win_draw_loss"]
                recovery = value.get("recovery_fraction")
                lines.append(
                    f"| {lane} | {step} | {context} | {value['paired_candidate_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {value['seat_a_effect']:+.4f} | {value['seat_b_effect']:+.4f} | {wdl['wins']}/{wdl['draws']}/{wdl['losses']} | {value['safe']} | {'-' if recovery is None else f'{recovery:.1%}'} |"
                )
    lines += [
        "",
        "## Trajectory Vs Graft",
        "",
        "| Lane | Step | Param cosine | Norm ratio | Output-delta cosine | Independent-vs-graft L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, steps in summary["trajectory_vs_graft"].items():
        for step, value in steps.items():
            lines.append(
                f"| {lane} | {step} | {value['parameter_delta_cosine']:.4f} | {value['parameter_norm_ratio']:.4f} | {value['output_policy_delta_cosine']:.4f} | {value['output_l1_independent_vs_graft']:.6f} |"
            )
    lines += [
        "",
        "## Parameter Drift",
        "",
        "| Lane | Step | Hidden | Readout | Trunk | Value stack |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, steps in summary["parameter_drift"].items():
        for step, value in steps.items():
            lines.append(
                f"| {lane} | {step} | {value['hidden']:.6e} | {value['readout']:.6e} | {value['trunk']:.2e} | {value['value_stack']:.2e} |"
            )
    lines += [
        "",
        "## Search Diagnostics",
        "",
        "| Candidate | Budget | Move changes | Visit JS | Q-rank changes | Root-value delta | Visit-margin delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, contexts in summary["search_diagnostics"].items():
        if candidate == "p1":
            continue
        for context, value in contexts.items():
            lines.append(
                f"| {candidate} | {context} | {value['selected_move_changes']:.4f} | {value['visit_js']:.6e} | {value['q_ranking_changes']:+.4f} | {value['root_value_delta']:+.6f} | {value['visit_margin_delta']:+.6f} |"
            )
    parity = summary["invariants"]["gradient_parity_detail"]
    lines += [
        "",
        "## Invariants",
        "",
        f"- Full-policy reproduction: `{summary['invariants']['full_policy_reproduced']}`",
        f"- Frozen families bit-identical: `{summary['invariants']['frozen_families_bit_identical']}`",
        f"- Hidden/readout gradient parity: `{parity['hidden_gradient_parity']}` / `{parity['readout_gradient_parity']}`",
        f"- Trunk/value gradients absent: `{parity['trunk_gradients_absent']}` / `{parity['value_gradients_absent']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_isolated_policy_sublayers"),
    )
    parser.add_argument(
        "--selection-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_checkpoint_selection"),
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-isolated-policy-sublayer-training-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-isolated-policy-sublayer-training-results.md",
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    if (
        sha256_file(REPO_ROOT / "model-artifact/current/weights.json")
        != P0_EXPECTED_HASH
        or sha256_file(checkpoint) != P1_EXPECTED_NPZ_HASH
    ):
        raise RuntimeError("P0 or exact P1 checkpoint hash mismatch")
    parent_model = _new_model(device)
    load_checkpoint_into_model(parent_model, checkpoint)
    parent = {
        name: value.detach().cpu().clone()
        for name, value in parent_model.state_dict().items()
    }
    if state_hash(parent) != P1_EXPECTED_STATE_HASH:
        raise RuntimeError("P1 state hash mismatch")
    manifest = verify_manifest(args.selection_workdir / "training_manifest.json")
    rows = read_jsonl(Path(manifest["replay_path"]))
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    first = _batch([rows[int(index)] for index in source], plan[0], device)
    first_target = mixed_policy_target(
        first["p"], incumbent_policy_batch(parent_model, first), first["mask"], BETA
    )
    parity = gradient_parity({**first, "p": first_target}, checkpoint, device)
    trained = {
        lane: train_lane(manifest, args.workdir, checkpoint, lane, device)
        for lane in LANES
    }
    full_hashes = {
        str(step): state_hash(trained["full_policy"][step][0])
        for step in CHECKPOINT_STEPS
    }
    full_reproduction = all(
        full_hashes[str(step)] == EXPECTED_CANDIDATE_HASHES[step] for step in (16, 46)
    )
    frozen = {
        lane: all(
            _frozen_contract(snapshot[0], parent, lane)
            for step, snapshot in snapshots.items()
            if step
        )
        for lane, snapshots in trained.items()
    }
    indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    _probe, probe_manifest = decoded_validation_manifest(rows, indexes)
    probe_rows = [rows[index] for index in probe_manifest["source_indexes"]]
    pure_ce = pure_search_step46_ce(manifest, probe_rows, device, checkpoint)
    training_metrics = {
        lane: metrics(probe_rows, snapshots, parent, pure_ce)
        for lane, snapshots in trained.items()
    }
    artifacts = {
        "p1": args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    }
    states = {"p1": parent}
    for lane, snapshots in trained.items():
        exported = export_snapshot_artifacts(snapshots, args.workdir / lane)
        for step in (16, 46):
            key = f"{lane}_{step}"
            artifacts[key], states[key] = exported[step], snapshots[step][0]
    grafts = {
        step: torch.load(
            args.selection_workdir / f"beta_095/snapshots/step_{step:04d}.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        for step in (16, 46)
    }
    for lane in ("hidden_only", "readout_only"):
        for step in (16, 46):
            graft_name = "hidden" if lane == "hidden_only" else "readout"
            graft_state = {
                name: value.detach().clone() for name, value in parent.items()
            }
            for key in POLICY_FAMILIES[graft_name]:
                graft_state[key] = grafts[step][key].detach().clone()
            key = f"graft_{lane}_{step}"
            states[key] = graft_state
            artifacts[key] = export_artifact(
                graft_state, args.workdir / "grafts" / key, key
            )
    trajectory = trajectory_comparison(rows, parent, trained, grafts)
    search = search_metrics(_probe[:256], artifacts, states)
    for lane in ("hidden_only", "readout_only"):
        for step in (16, 46):
            trained_search, graft_search = (
                search[f"{lane}_{step}"],
                search[f"graft_{lane}_{step}"],
            )
            trajectory[lane][str(step)]["search_diagnostic_difference_vs_graft"] = {
                context: {
                    name: float(
                        trained_search[context][name] - graft_search[context][name]
                    )
                    for name in (
                        "selected_move_changes",
                        "visit_js",
                        "q_ranking_changes",
                        "root_value_delta",
                        "visit_margin_delta",
                    )
                }
                for context in (PRIMARY_CONTEXT, HIGH_CONTEXT)
            }
    arena: dict[str, Any] = {
        lane: {str(step): {} for step in (16, 46)} for lane in LANES
    }
    for lane in LANES:
        for step in (16, 46):
            low = arena_entry(
                artifacts[f"{lane}_{step}"],
                artifacts["p1"],
                PRIMARY_CONTEXT,
                args.workdir / "arena" / lane / str(step),
                "candidate_vs_p1",
                args.workers,
            )
            arena[lane][str(step)][PRIMARY_CONTEXT] = low
            if lane != "full_policy":
                full = arena["full_policy"][str(step)][PRIMARY_CONTEXT][
                    "paired_candidate_effect"
                ]
                low["recovery_fraction"] = (
                    (low["paired_candidate_effect"] - full) / -full if full else None
                )
    for lane in LANES:
        for step in (16, 46):
            low = arena[lane][str(step)][PRIMARY_CONTEXT]
            if (
                lane == "full_policy"
                or low["safe"]
                or low.get("recovery_fraction", 0.0) >= 0.70
            ):
                arena[lane][str(step)][HIGH_CONTEXT] = arena_entry(
                    artifacts[f"{lane}_{step}"],
                    artifacts["p1"],
                    HIGH_CONTEXT,
                    args.workdir / "arena" / lane / str(step),
                    "candidate_vs_p1",
                    args.workers,
                )
    parameter_drift = {
        lane: {
            str(step): _parameter_drift(snapshots[step][0], parent)
            for step in CHECKPOINT_STEPS
        }
        for lane, snapshots in trained.items()
    }
    invariants = {
        "gradient_parity": parity["all_passed"],
        "full_policy_reproduced": full_reproduction,
        "frozen_families_bit_identical": all(frozen.values()),
        "all_passed": parity["all_passed"]
        and full_reproduction
        and all(frozen.values()),
    }
    summary: dict[str, Any] = {
        "schema": "azlite_fresh_p1_isolated_policy_sublayer_training_v1",
        "guardrails": {
            "self_play_generated": False,
            "beta": BETA,
            "optimizer": "Adam",
            "lr": 1e-5,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "batch_size": 512,
            "optimizer_steps": 46,
            "architecture_changed": False,
            "mcts_changed": False,
            "promotion": False,
        },
        "hashes": {
            "p0_weights_sha256": P0_EXPECTED_HASH,
            "p1_checkpoint_sha256": P1_EXPECTED_NPZ_HASH,
            "p1_state_hash": P1_EXPECTED_STATE_HASH,
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "manifest_sha256": sha256_file(
                args.selection_workdir / "training_manifest.json"
            ),
            "batch_plan_sha256": sha256_file(paths["batch_indexes"]),
            "checkpoint_state_hashes": {
                lane: {
                    str(step): state_hash(snapshots[step][0])
                    for step in CHECKPOINT_STEPS
                }
                for lane, snapshots in trained.items()
            },
        },
        "invariants": invariants
        | {"gradient_parity_detail": parity, "frozen_by_lane": frozen},
        "training_metrics": training_metrics,
        "parameter_drift": parameter_drift,
        "trajectory_vs_graft": trajectory,
        "search_diagnostics": search,
        "arena": arena,
    }
    summary["classification"] = classify(summary)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], indent=2))


if __name__ == "__main__":
    main()
