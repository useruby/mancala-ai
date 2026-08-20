#!/usr/bin/env python3
"""Diagnostic-only causal decomposition of PR #212's residual_v3 policy update."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.policy_sublayer_graft import (  # noqa: E402
    HIDDEN_KEYS,
    POLICY_KEYS,
    READOUT_KEYS,
    assert_candidate_contract,
    assert_graft_contract,
    graft_state,
    state_hash,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_p1_checkpoint_selection import arena_safe  # noqa: E402
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import _win_draw_loss  # noqa: E402
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P0_EXPECTED_HASH,
    P1_EXPECTED_NPZ_HASH,
    P1_EXPECTED_STATE_HASH,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    ARENA_SUITE,
    _new_model,
    write_fixed_npz,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    js,
    model_outputs,
)
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    export_checkpoint,
)
from ml.alphazero_lite.self_play import encode_state  # noqa: E402
from ml.alphazero_lite.train import (  # noqa: E402
    checkpoint_from_model,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

STEPS = (16, 46)
PRIMARY_CONTEXT = "384:256"
HIGH_CONTEXT = "1200:1200"
EXPECTED_CANDIDATE_HASHES = {
    16: "933ed483da1db3a8d17c45e9ace65ae88dbb6b13c22c18de3d13f36863fc7ca1",
    46: "58bd2ca08a4329573432313384e1a2915e61f7de6304f9e9b08e36689a0bf66f",
}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else None


def delta_metrics(
    parent: dict[str, torch.Tensor], candidates: dict[int, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family, keys in (("hidden", HIDDEN_KEYS), ("readout", READOUT_KEYS)):
        deltas = {
            step: torch.cat(
                [
                    (candidates[step][key].double() - parent[key].double()).flatten()
                    for key in keys
                ]
            )
            for step in STEPS
        }
        result[family] = {
            str(step): {
                "l2_norm": float(torch.linalg.vector_norm(deltas[step])),
                "relative_norm": float(
                    torch.linalg.vector_norm(deltas[step])
                    / torch.linalg.vector_norm(
                        torch.cat([parent[key].double().flatten() for key in keys])
                    )
                ),
                "weight_l2_norm": float(
                    torch.linalg.vector_norm(
                        candidates[step][keys[0]].double() - parent[keys[0]].double()
                    )
                ),
                "bias_l2_norm": float(
                    torch.linalg.vector_norm(
                        candidates[step][keys[1]].double() - parent[keys[1]].double()
                    )
                ),
            }
            for step in STEPS
        }
        norm46 = float(torch.linalg.vector_norm(deltas[46]))
        result[family]["step16_vs_step46"] = {
            "cosine": _cosine(deltas[16].numpy(), deltas[46].numpy()),
            "fraction_step46_delta_present_at_step16": float(
                torch.dot(deltas[16], deltas[46]) / (norm46**2)
            )
            if norm46
            else None,
        }
    return result


def export_artifact(state: dict[str, torch.Tensor], path: Path, name: str) -> Path:
    model = _new_model(torch.device("cpu"))
    model.load_state_dict(state)
    checkpoint = path / "checkpoint.npz"
    artifact = path / "artifact"
    path.mkdir(parents=True, exist_ok=True)
    write_fixed_npz(checkpoint, checkpoint_from_model(model))
    export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=artifact,
        version=name,
        policy_loss=0.0,
        value_loss=0.0,
    )
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["diagnostic_only"] = True
    metadata["promotion_forbidden"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return artifact


def policy_metrics(
    rows: list[dict[str, Any]], states: dict[str, dict[str, torch.Tensor]]
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    policies = {
        name: model_outputs(state, x, mask)[0] for name, state in states.items()
    }
    parent = policies["p1"]
    result: dict[str, Any] = {}
    for name, policy in policies.items():
        if name == "p1":
            continue
        delta = policy - parent
        l1 = np.abs(delta).sum(axis=1)
        largest = np.abs(delta).max(axis=1)
        metrics = {
            "legal_policy_l1": _summary(l1),
            "mean_js": float(np.mean(js(parent, policy))),
            "top1_disagreement": float(
                np.mean(np.argmax(parent, axis=1) != np.argmax(policy, axis=1))
            ),
            "largest_absolute_action_probability_change": float(largest.max()),
        }
        splits: dict[str, dict[str, list[int]]] = {
            "player": defaultdict(list),
            "legal_move_count": defaultdict(list),
            "ply_bucket": defaultdict(list),
        }
        for index, row in enumerate(rows):
            splits["player"][str(row.get("player", 0))].append(index)
            splits["legal_move_count"][str(int(mask[index].sum()))].append(index)
            ply = int(row.get("move_index", 0))
            splits["ply_bucket"][
                "0-9"
                if ply < 10
                else "10-19"
                if ply < 20
                else "20-29"
                if ply < 30
                else "30+"
            ].append(index)
        metrics["splits"] = {
            dimension: {
                key: {
                    "count": len(indexes),
                    "l1_mean": float(l1[indexes].mean()),
                    "js_mean": float(js(parent[indexes], policy[indexes]).mean()),
                }
                for key, indexes in buckets.items()
            }
            for dimension, buckets in splits.items()
        }
        result[name] = metrics
    # Output-space additivity: full delta compared to the sum of isolated deltas.
    for step in STEPS:
        required = {f"full_{step}", f"hidden_{step}", f"readout_{step}"}
        if not required.issubset(policies):
            continue
        full, hidden, readout = (
            policies[f"full_{step}"],
            policies[f"hidden_{step}"],
            policies[f"readout_{step}"],
        )
        full_delta, additive = full - parent, (hidden - parent) + (readout - parent)
        per_state = [_cosine(full_delta[i], additive[i]) for i in range(len(rows))]
        result[f"interaction_{step}"] = {
            "mean_full_vs_additive_delta_cosine": float(
                np.mean([value for value in per_state if value is not None])
            ),
            "relative_nonadditive_residual_l2": float(
                np.linalg.norm(full_delta - additive) / np.linalg.norm(full_delta)
            ),
        }
    return result


def search_metrics(
    probe: list[dict[str, Any]],
    artifacts: dict[str, Path],
    states: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    evaluators = {
        name: arena.ArtifactEvaluator(path) for name, path in artifacts.items()
    }
    options = arena.build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    results: dict[str, Any] = {}
    for context in (PRIMARY_CONTEXT, HIGH_CONTEXT):
        sims = int(context.split(":")[0])
        baseline = []
        for row in probe:
            baseline.append(
                arena.evaluate_artifact_position(
                    evaluator=evaluators["p1"],
                    state=row["state"],
                    simulations=sims,
                    seed=42 + row["manifest_index"],
                    c_puct=1.25,
                    search_options=options,
                )
            )
        for name, evaluator in evaluators.items():
            if name == "p1":
                continue
            changed: list[float] = []
            visit_js: list[float] = []
            root_delta: list[float] = []
            margin_delta: list[float] = []
            rank_delta: list[float] = []
            for row, base in zip(probe, baseline, strict=True):
                candidate = arena.evaluate_artifact_position(
                    evaluator=evaluator,
                    state=row["state"],
                    simulations=sims,
                    seed=42 + row["manifest_index"],
                    c_puct=1.25,
                    search_options=options,
                )
                base_stats = {int(item["move"]): item for item in base["child_stats"]}
                cand_stats = {
                    int(item["move"]): item for item in candidate["child_stats"]
                }
                moves = sorted(base_stats)
                bp = np.asarray(
                    [base_stats[move]["visits"] for move in moves], dtype=float
                )
                cp = np.asarray(
                    [cand_stats[move]["visits"] for move in moves], dtype=float
                )
                bp /= bp.sum()
                cp /= cp.sum()
                changed.append(
                    float(base["selected_move"] != candidate["selected_move"])
                )
                visit_js.append(float(js(bp[None, :], cp[None, :])[0]))
                root_delta.append(
                    float(
                        candidate.get("search_root_value", candidate["value"])
                        - base.get("search_root_value", base["value"])
                    )
                )
                base_margin = (
                    float(np.sort(bp)[-1] - np.sort(bp)[-2])
                    if len(bp) > 1
                    else float(bp[0])
                )
                candidate_margin = (
                    float(np.sort(cp)[-1] - np.sort(cp)[-2])
                    if len(cp) > 1
                    else float(cp[0])
                )
                margin_delta.append(candidate_margin - base_margin)
                rank_delta.append(
                    float(
                        sorted(
                            moves, key=lambda move: -cand_stats[move]["q_value"]
                        ).index(int(candidate["selected_move"]))
                        - sorted(
                            moves, key=lambda move: -base_stats[move]["q_value"]
                        ).index(int(base["selected_move"]))
                    )
                )
            encoded_probe = [
                {
                    "state": encode_state(row["state"], input_encoding="kalah_v3"),
                    "player": row["player"],
                    "move_index": 0,
                }
                for row in probe
            ]
            root_policy = policy_metrics(
                encoded_probe, {"p1": states["p1"], name: states[name]}
            )[name]
            results.setdefault(name, {})[context] = {
                "states": len(probe),
                "selected_move_changes": float(np.mean(changed)),
                "visit_js": float(np.mean(visit_js)),
                "q_ranking_changes": float(np.mean(rank_delta)),
                "root_value_delta": float(np.mean(root_delta)),
                "visit_margin_delta": float(np.mean(margin_delta)),
                "per_depth_legal_policy": {
                    "0": {
                        "l1_mean": root_policy["legal_policy_l1"]["mean"],
                        "js_mean": root_policy["mean_js"],
                    }
                },
            }
    return results


def arena_entry(
    candidate: Path, parent: Path, context: str, workdir: Path, role: str, workers: int
) -> dict[str, Any]:
    control = _arena_records(
        workdir / "control", parent, parent, context, f"{role}_control", workers
    )
    records = _arena_records(workdir / role, candidate, parent, context, role, workers)
    effect = paired_opening_candidate_effect(records, control)
    return {
        "paired_effect": effect["paired_candidate_effect"],
        "bootstrap_95_ci": effect["opening_bootstrap_ci"],
        "seat_a": effect["p0_effect"],
        "seat_b": effect["p1_effect"],
        "win_draw_loss": _win_draw_loss(records),
        "safe": arena_safe(effect),
    }


def classify(arena_results: dict[str, Any], invariants: bool) -> dict[str, str]:
    if not invariants:
        return {
            "label": "invariant_failure",
            "next_experiment": "repair reconstruction and graft invariants before rerunning",
        }
    labels = []
    for step in STEPS:
        full = arena_results[str(step)]["full"][PRIMARY_CONTEXT]["paired_effect"]
        recovery = {
            name: arena_results[str(step)][name][PRIMARY_CONTEXT]["recovery_fraction"]
            for name in ("hidden", "readout")
        }
        harmful = {name: recovery[name] < 0.30 for name in recovery}
        safe = {
            name: recovery[name] >= 0.70
            or arena_results[str(step)][name][PRIMARY_CONTEXT]["safe"]
            for name in recovery
        }
        if harmful["hidden"] and safe["readout"]:
            labels.append("policy_hidden_delta_causal")
        elif harmful["readout"] and safe["hidden"]:
            labels.append("policy_readout_delta_causal")
        elif (
            recovery["hidden"] >= 0.50 and recovery["readout"] >= 0.50 and full < -0.03
        ):
            labels.append("joint_policy_sublayer_interaction")
        elif harmful["hidden"] and harmful["readout"]:
            labels.append("both_policy_sublayers_independently_harmful")
        else:
            labels.append("inconclusive")
    if labels[0] != labels[1]:
        label = "step_dependent_policy_mechanism"
    else:
        label = labels[0]
    next_experiments = {
        "policy_hidden_delta_causal": "retrain the exact PR #212 replay with ONLY the final policy_head/readout trainable",
        "policy_readout_delta_causal": "test a constrained/residual action-logit adapter rather than modifying the existing readout directly",
        "joint_policy_sublayer_interaction": "train one sublayer at a time, beginning with whichever isolated graft has better teacher fit/search behavior",
        "both_policy_sublayers_independently_harmful": "move to an additive parent-preserving policy adapter architecture rather than continuing to mutate the existing policy head",
        "step_dependent_policy_mechanism": "inspect the update trajectory before selecting a new parameterization",
        "inconclusive": "expand the frozen diagnostic probe or arena evidence without changing the training recipe",
    }
    return {"label": label, "next_experiment": next_experiments[label]}


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# PR #212 Policy Sublayer Causal Decomposition",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        f"**Next experiment:** {summary['classification']['next_experiment']}",
        "",
        "## Exact Reconstruction",
        "",
        "| Model | State hash |",
        "| --- | --- |",
        f"| P1 | `{summary['hashes']['p1_state_hash']}` |",
    ]
    for step, digest in summary["hashes"]["candidates"].items():
        lines.append(f"| C{step} | `{digest}` |")
    lines += [
        "",
        "All C16/C46 trunk and value tensors are byte-identical to P1; only the four policy-hidden/readout tensors differ. Full grafts are state-identical to their source candidates.",
        "",
        "## Parameter Deltas",
        "",
        "| Family | Step | L2 | Relative | Weight L2 | Bias L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, values in summary["parameter_deltas"].items():
        for step in ("16", "46"):
            item = values[step]
            lines.append(
                f"| {family} | {step} | {item['l2_norm']:.6e} | {item['relative_norm']:.6e} | {item['weight_l2_norm']:.6e} | {item['bias_l2_norm']:.6e} |"
            )
        alignment = values["step16_vs_step46"]
        lines.append(
            f"| {family} alignment | 16 vs 46 | cosine {alignment['cosine']:.4f} | step-46 fraction {alignment['fraction_step46_delta_present_at_step16']:.4f} | | |"
        )
    lines += [
        "",
        "## Paired Arenas",
        "",
        "| Step | Graft | Budget | Effect | 95% CI | Seat A | Seat B | W/D/L | Recovery |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for step, grafts in summary["arena"].items():
        for name, contexts in grafts.items():
            for context, item in contexts.items():
                ci, wdl = item["bootstrap_95_ci"], item["win_draw_loss"]
                recovery = item.get("recovery_fraction")
                lines.append(
                    f"| {step} | {name} | {context} | {item['paired_effect']:+.4f} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | {item['seat_a']:+.4f} | {item['seat_b']:+.4f} | {wdl['wins']}/{wdl['draws']}/{wdl['losses']} | {'-' if recovery is None else f'{recovery:.1%}'} |"
                )
    lines += [
        "",
        "## Network Decomposition",
        "",
        "| Probe | Candidate | Mean L1 | P99 L1 | Mean JS | Top-1 disagreement | Max action change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for probe, metrics in summary["network_decomposition"].items():
        for name, item in metrics.items():
            if name.startswith("interaction_"):
                continue
            l1 = item["legal_policy_l1"]
            lines.append(
                f"| {probe} | {name} | {l1['mean']:.6f} | {l1['p99']:.6f} | {item['mean_js']:.6e} | {item['top1_disagreement']:.4f} | {item['largest_absolute_action_probability_change']:.6f} |"
            )
    lines += [
        "",
        "The JSON summary retains the requested player, legal-move-count, and ply-bucket splits and full-vs-additive output alignment.",
        "",
        "## Search Decomposition",
        "",
        "| Candidate | Budget | Move changes | Visit JS | Q-rank change | Root-value delta | Visit-margin delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, contexts in summary["search_decomposition"].items():
        for context, item in contexts.items():
            lines.append(
                f"| {name} | {context} | {item['selected_move_changes']:.4f} | {item['visit_js']:.6e} | {item['q_ranking_changes']:+.4f} | {item['root_value_delta']:+.6f} | {item['visit_margin_delta']:+.6f} |"
            )
    lines += [
        "",
        "The deterministic PUCT probe evaluates root states, so per-depth policy reporting is depth 0; the corresponding L1/JS is retained in JSON.",
        "",
        "## Caveat",
        "",
        "A graft is a causal parameter intervention, not proof that independently training a sublayer follows the same trajectory. No trainable scope is changed here.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_policy_sublayer_decomposition"),
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
        / "docs/data/alphazero-lite-fresh-p1-policy-sublayer-decomposition-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-policy-sublayer-decomposition-results.md",
    )
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if (
        sha256_file(REPO_ROOT / "model-artifact/current/weights.json")
        != P0_EXPECTED_HASH
    ):
        raise RuntimeError("P0 hash mismatch")
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    if (
        not p1_checkpoint.is_file()
        or sha256_file(p1_checkpoint) != P1_EXPECTED_NPZ_HASH
    ):
        raise RuntimeError("exact PR #203 P1 checkpoint is unavailable")
    p1_model = _new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1_model, p1_checkpoint)
    parent = p1_model.state_dict()
    if state_hash(parent) != P1_EXPECTED_STATE_HASH:
        raise RuntimeError("P1 state hash mismatch")
    candidates = {
        step: torch.load(
            args.selection_workdir / f"beta_095/snapshots/step_{step:04d}.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        for step in STEPS
    }
    for step, candidate in candidates.items():
        assert_candidate_contract(parent, candidate)
        if state_hash(candidate) != EXPECTED_CANDIDATE_HASHES[step]:
            raise RuntimeError(f"C{step} state hash mismatch")
    states = {"p1": parent}
    artifacts = {
        "p1": args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    }
    grafts = {}
    for step, candidate in candidates.items():
        full = graft_state(parent, candidate, POLICY_KEYS)
        hidden = graft_state(parent, candidate, HIDDEN_KEYS)
        readout = graft_state(parent, candidate, READOUT_KEYS)
        for name, state in (
            ("full", full),
            ("hidden_delta_only", hidden),
            ("readout_delta_only", readout),
        ):
            assert_graft_contract(name, parent, candidate, state)
        if state_hash(full) != state_hash(candidate):
            raise RuntimeError(f"full_{step} is not exact C{step}")
        grafts[step] = {"full": full, "hidden": hidden, "readout": readout}
        for name, state in grafts[step].items():
            key = f"{name}_{step}"
            states[key] = state
            artifacts[key] = export_artifact(
                state, args.workdir / "artifacts" / key, key
            )
    rows = read_jsonl(args.selection_workdir / "fresh_p1_self_play.jsonl")
    manifest = verify_manifest(args.selection_workdir / "training_manifest.json")
    indexes = np.load(
        Path(manifest["artifact_paths"]["validation_source_indexes"]),
        allow_pickle=False,
    )
    probe, _ = decoded_validation_manifest(rows, indexes)
    probe = probe[:256]
    canonical_rows = []
    for item in read_jsonl(ARENA_SUITE):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move in item["prefix_moves"]:
            game.move(game.pit_index(move))
        canonical_rows.append(
            {
                "state": encode_state(game.to_state(), input_encoding="kalah_v3"),
                "player": game.current_player,
                "move_index": len(item["prefix_moves"]),
            }
        )
    network = {
        "fresh_replay": policy_metrics(rows, states),
        "deterministic_puct_roots": policy_metrics(
            [
                {
                    **row,
                    "state": encode_state(row["state"], input_encoding="kalah_v3"),
                    "move_index": 0,
                }
                for row in probe
            ],
            states,
        ),
        "canonical_opening_roots": policy_metrics(canonical_rows, states),
    }
    # The existing deterministic PUCT root probe is reused verbatim; depth 0 is its only evaluated depth.
    search = search_metrics(probe, artifacts, states)
    arena_results: dict[str, Any] = {}
    for step in STEPS:
        arena_results[str(step)] = {}
        for name in ("full", "hidden", "readout"):
            key = f"{name}_{step}"
            low = arena_entry(
                artifacts[key],
                artifacts["p1"],
                PRIMARY_CONTEXT,
                args.workdir / "arena" / str(step),
                key,
                args.workers,
            )
            arena_results[str(step)][name] = {PRIMARY_CONTEXT: low}
        full_effect = arena_results[str(step)]["full"][PRIMARY_CONTEXT]["paired_effect"]
        for name in ("hidden", "readout"):
            effect = arena_results[str(step)][name][PRIMARY_CONTEXT]["paired_effect"]
            arena_results[str(step)][name][PRIMARY_CONTEXT]["recovery_fraction"] = (
                (effect - full_effect) / -full_effect if full_effect else None
            )
        high_names = ["full"] + [
            name
            for name in ("hidden", "readout")
            if arena_results[str(step)][name][PRIMARY_CONTEXT]["safe"]
            or arena_results[str(step)][name][PRIMARY_CONTEXT]["recovery_fraction"]
            >= 0.70
        ]
        for name in high_names:
            key = f"{name}_{step}"
            arena_results[str(step)][name][HIGH_CONTEXT] = arena_entry(
                artifacts[key],
                artifacts["p1"],
                HIGH_CONTEXT,
                args.workdir / "arena" / str(step),
                key,
                args.workers,
            )
    invariants = True
    summary = {
        "schema": "azlite_fresh_p1_policy_sublayer_decomposition_v1",
        "guardrails": {
            "training_run": False,
            "self_play_generated": False,
            "beta_changed": False,
            "mcts_changed": False,
            "architecture_changed": False,
            "promotion": False,
        },
        "hashes": {
            "p0_weights_sha256": P0_EXPECTED_HASH,
            "p1_checkpoint_sha256": sha256_file(p1_checkpoint),
            "p1_state_hash": state_hash(parent),
            "candidates": {str(step): state_hash(candidates[step]) for step in STEPS},
        },
        "invariants": {
            "all_passed": invariants,
            "candidate_policy_only": True,
            "full_exact_candidate": True,
            "grafts_exact_family_composition": True,
        },
        "parameter_deltas": delta_metrics(parent, candidates),
        "network_decomposition": network,
        "search_decomposition": search,
        "arena": arena_results,
    }
    summary["classification"] = classify(arena_results, invariants)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_report.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], indent=2))


if __name__ == "__main__":
    main()
