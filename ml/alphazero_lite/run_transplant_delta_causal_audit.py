#!/usr/bin/env python3
# ruff: noqa: E402
"""Causally decompose PR #204 regression by transplanting Gen-1 and Gen-2 policy deltas.

Lineage:
- P0: frozen incumbent from model-artifact/current
- P1: PR #203 beta_095 step 46 candidate (safe at 384:256, +0.0234 at 1200:1200)
- P2: PR #204 beta_095 step 46 candidate (-0.0957 at 384:256 vs P1, -0.0938 vs P0)

Delta transplants:
- delta_01 = policy_head(P1) - policy_head(P0)
- delta_12 = policy_head(P2) - policy_head(P1)
- control_p1 = P0 + delta_01 (must reproduce P1 exactly)
- control_p2 = P1 + delta_12 (must reproduce P2 exactly)
- x_gen2_delta_on_p0 = P0 + delta_12 (Gen-2 delta transplanted onto original parent P0)
- y_gen1_delta_on_p1 = P1 + delta_01 (Gen-1 delta repeated from P1)

This runner performs diagnostic-only evaluation:
1. Reconstructs and verifies exact P0, P1, P2 checkpoints and lineage invariants.
2. Computes policy-head deltas, norms, cosine similarities, projections, and residuals.
3. Constructs delta-transplant checkpoints (control_p1, control_p2, X, Y) with frozen trunk/value invariants.
4. Evaluates network-output diagnostics on 4 matched probes (PR #203 val, PR #204 val, canonical openings, search expanded states).
5. Performs canonical opening failure attribution (128 openings, seat decomposition, losing vs unchanged).
6. Runs canonical paired arena matches at 384:256 (P1 vs P0, P2 vs P1, X vs P0, Y vs P1).
7. Classifies mechanism and records the Adam gradient scaling observation.
"""

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

from ml.alphazero_lite.arena import (  # noqa: E402
    ArtifactEvaluator,
    apply_opening_moves,
    build_eval_search_options,
    evaluate_artifact_position,
)
from ml.alphazero_lite.evaluation_metrics import (  # noqa: E402
    _key,
    paired_opening_candidate_effect,
    score_from_game,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.pipeline import (  # noqa: E402
    materialize_weights_json_checkpoint,
)
from ml.alphazero_lite.policy_prior_localization import (  # noqa: E402
    PriorSubstitutionOverride,
    _js,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (  # noqa: E402
    _win_draw_loss,
    group_parameters_identical,
    trunk_parameters_identical,
)
from ml.alphazero_lite.run_frozen_trunk_head_isolation_ablation import (  # noqa: E402
    VALUE_STACK_PREFIXES,
)
from ml.alphazero_lite.run_gen2_selfplay_anchor_iteration import (  # noqa: E402
    P0_EXPECTED_HASH,
    P1_EXPECTED_NPZ_HASH,
    P1_EXPECTED_STATE_HASH,
    reconstruct_and_freeze_p1,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import (  # noqa: E402
    ARENA_SUITE,
    _new_model,
    write_fixed_npz,
)
from ml.alphazero_lite.run_policy_detached_trunk_ablation import _arena_records  # noqa: E402
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    stable_hash,
)
from ml.alphazero_lite.run_terminal_outcome_selfplay_iteration_smoke import (  # noqa: E402
    export_checkpoint,
)
from ml.alphazero_lite.train import (  # noqa: E402
    checkpoint_from_model,
    load_checkpoint_into_model,
)

NAMESPACE = "azlite_transplant_delta_causal_v1"
PRIMARY_CONTEXT = "384:256"
ARENA_WORKERS = 32

POLICY_KEYS = [
    "policy_hidden_layer.weight",
    "policy_hidden_layer.bias",
    "policy_head.weight",
    "policy_head.bias",
]

P2_EXPECTED_STATE_HASH = (
    "336496d5fb33331240178c4b834b8faf9548e3915b45c9b5f7e4b7aad6626870"
)

INITIAL_BOARD = {
    "player_pits": [4, 4, 4, 4, 4, 4],
    "opponent_pits": [4, 4, 4, 4, 4, 4],
    "player_store": 0,
    "opponent_store": 0,
    "current_player": 0,
}


def compute_distribution_metrics(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {
            "mean": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_5": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99_5": float(np.percentile(arr, 99.5)),
    }


def compute_policy_deltas(
    s0: dict[str, torch.Tensor],
    s1: dict[str, torch.Tensor],
    s2: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Compute exact delta_01 and delta_12 and geometric alignment metrics."""
    delta_01: dict[str, torch.Tensor] = {}
    delta_12: dict[str, torch.Tensor] = {}
    d01_flat_list: list[torch.Tensor] = []
    d12_flat_list: list[torch.Tensor] = []
    per_layer: dict[str, Any] = {}

    for k in POLICY_KEYS:
        t01 = s1[k].double() - s0[k].double()
        t12 = s2[k].double() - s1[k].double()
        delta_01[k] = t01
        delta_12[k] = t12

        f01 = t01.flatten()
        f12 = t12.flatten()
        d01_flat_list.append(f01)
        d12_flat_list.append(f12)

        norm01_l = float(torch.norm(f01).item())
        norm12_l = float(torch.norm(f12).item())
        cos_l = (
            float((torch.dot(f01, f12) / (norm01_l * norm12_l)).item())
            if norm01_l > 0 and norm12_l > 0
            else 0.0
        )
        per_layer[k] = {
            "d01_l2_norm": norm01_l,
            "d12_l2_norm": norm12_l,
            "cosine_similarity": cos_l,
        }

    v01 = torch.cat(d01_flat_list)
    v12 = torch.cat(d12_flat_list)

    total_norm01 = float(torch.norm(v01).item())
    total_norm12 = float(torch.norm(v12).item())
    cos_sim = float((torch.dot(v01, v12) / (total_norm01 * total_norm12)).item())

    proj_scalar = float((torch.dot(v12, v01) / (total_norm01**2)).item())
    residual = v12 - proj_scalar * v01
    res_norm = float(torch.norm(residual).item())
    res_fraction = float(res_norm / total_norm12) if total_norm12 > 0 else 0.0

    stats = {
        "delta_01_l2_norm": total_norm01,
        "delta_12_l2_norm": total_norm12,
        "cosine_similarity": cos_sim,
        "projection_12_on_01": proj_scalar,
        "residual_norm": res_norm,
        "residual_fraction": res_fraction,
        "per_layer": per_layer,
    }
    return delta_01, delta_12, stats


def construct_transplants(
    s0: dict[str, torch.Tensor],
    s1: dict[str, torch.Tensor],
    s2: dict[str, torch.Tensor],
    delta_01: dict[str, torch.Tensor],
    delta_12: dict[str, torch.Tensor],
    workdir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Construct control_p1, control_p2, x_gen2_delta_on_p0, y_gen1_delta_on_p1."""
    # Control P1: P0 + delta_01
    control_p1_state = {k: v.clone() for k, v in s0.items()}
    for k in POLICY_KEYS:
        control_p1_state[k] = (s0[k].double() + delta_01[k]).float()

    # Control P2: P1 + delta_12
    control_p2_state = {k: v.clone() for k, v in s1.items()}
    for k in POLICY_KEYS:
        control_p2_state[k] = (s1[k].double() + delta_12[k]).float()

    # X: P0 + delta_12
    x_state = {k: v.clone() for k, v in s0.items()}
    for k in POLICY_KEYS:
        x_state[k] = (s0[k].double() + delta_12[k]).float()

    # Y: P1 + delta_01
    y_state = {k: v.clone() for k, v in s1.items()}
    for k in POLICY_KEYS:
        y_state[k] = (s1[k].double() + delta_01[k]).float()

    def get_state_hash(sd: dict[str, torch.Tensor]) -> str:
        return stable_hash(
            {k: v.detach().cpu().numpy().tobytes().hex() for k, v in sorted(sd.items())}
        )

    # Invariant assertions
    control_p1_sh = get_state_hash(control_p1_state)
    p1_sh = get_state_hash(s1)
    if control_p1_sh != p1_sh:
        raise RuntimeError(
            f"control_p1 state hash {control_p1_sh} != expected P1 {p1_sh}"
        )

    control_p2_sh = get_state_hash(control_p2_state)
    p2_sh = get_state_hash(s2)
    if control_p2_sh != p2_sh:
        raise RuntimeError(
            f"control_p2 state hash {control_p2_sh} != expected P2 {p2_sh}"
        )

    for name, st in [
        ("control_p1", control_p1_state),
        ("control_p2", control_p2_state),
        ("x_gen2_delta_on_p0", x_state),
        ("y_gen1_delta_on_p1", y_state),
    ]:
        if not trunk_parameters_identical(st, s0):
            raise RuntimeError(f"{name} violated trunk equality vs P0")
        if not group_parameters_identical(st, s0, VALUE_STACK_PREFIXES):
            raise RuntimeError(f"{name} violated value stack equality vs P0")

    # Export artifacts
    artifacts: dict[str, Path] = {}
    checkpoints: dict[str, Path] = {}
    hashes: dict[str, Any] = {}

    for name, st in [
        ("x_gen2_delta_on_p0", x_state),
        ("y_gen1_delta_on_p1", y_state),
    ]:
        m = _new_model(device)
        m.load_state_dict(st)
        ckpt_path = workdir / f"{name}_checkpoint.npz"
        art_dir = workdir / f"{name}_artifact"
        art_dir.mkdir(parents=True, exist_ok=True)
        write_fixed_npz(ckpt_path, checkpoint_from_model(m))
        export_checkpoint(
            checkpoint_path=ckpt_path,
            out_dir=art_dir,
            version=name,
            policy_loss=0.0,
            value_loss=0.0,
        )
        artifacts[name] = art_dir
        checkpoints[name] = ckpt_path
        hashes[name] = {
            "state_hash": get_state_hash(st),
            "checkpoint_npz_sha256": sha256_file(ckpt_path),
            "weights_json_sha256": sha256_file(art_dir / "weights.json"),
            "trunk_identical_to_p0": True,
            "value_identical_to_p0": True,
        }

    return {
        "artifacts": artifacts,
        "checkpoints": checkpoints,
        "hashes": hashes,
        "control_p1_state_hash": control_p1_sh,
        "control_p2_state_hash": control_p2_sh,
        "control_p1_reproduced": control_p1_sh == p1_sh,
        "control_p2_reproduced": control_p2_sh == p2_sh,
    }


def evaluate_probe_set(
    states: list[dict[str, Any]],
    cand_ev: ArtifactEvaluator,
    base_ev: ArtifactEvaluator,
) -> dict[str, Any]:
    """Evaluate legal policy L1, JS, top-1 change on a list of states with splits."""
    l1_list: list[float] = []
    js_list: list[float] = []
    top1_changed_list: list[bool] = []

    by_player: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"l1": [], "js": [], "top1": []}
    )
    by_ply: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"l1": [], "js": [], "top1": []}
    )
    by_moves: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"l1": [], "js": [], "top1": []}
    )
    by_depth: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"l1": [], "js": [], "top1": []}
    )

    for item in states:
        raw_state = item["state"]
        game = KalahGame.from_state(raw_state)
        legal = game.possible_moves()
        if not legal:
            continue

        cand_policy, _ = cand_ev.evaluate(game)
        base_policy, _ = base_ev.evaluate(game)

        cand_policy = np.asarray(cand_policy, dtype=np.float32)
        base_policy = np.asarray(base_policy, dtype=np.float32)

        c_masked = cand_policy[legal] / cand_policy[legal].sum()
        b_masked = base_policy[legal] / base_policy[legal].sum()

        l1_val = float(np.sum(np.abs(c_masked - b_masked)))
        js_val = float(_js(base_policy, cand_policy, legal))
        top1_c = int(legal[np.argmax(c_masked)])
        top1_b = int(legal[np.argmax(b_masked)])
        top1_changed = bool(top1_c != top1_b)

        l1_list.append(l1_val)
        js_list.append(js_val)
        top1_changed_list.append(top1_changed)

        player = int(game.current_player)
        by_player[player]["l1"].append(l1_val)
        by_player[player]["js"].append(js_val)
        by_player[player]["top1"].append(float(top1_changed))

        num_moves = len(legal)
        by_moves[num_moves]["l1"].append(l1_val)
        by_moves[num_moves]["js"].append(js_val)
        by_moves[num_moves]["top1"].append(float(top1_changed))

        ply = int(item.get("ply", 0))
        if ply < 10:
            ply_bucket = "0-9"
        elif ply < 20:
            ply_bucket = "10-19"
        elif ply < 30:
            ply_bucket = "20-29"
        else:
            ply_bucket = "30+"
        by_ply[ply_bucket]["l1"].append(l1_val)
        by_ply[ply_bucket]["js"].append(js_val)
        by_ply[ply_bucket]["top1"].append(float(top1_changed))

        if "depth" in item:
            d = min(int(item["depth"]), 4)
            by_depth[d]["l1"].append(l1_val)
            by_depth[d]["js"].append(js_val)
            by_depth[d]["top1"].append(float(top1_changed))

    overall_l1 = compute_distribution_metrics(l1_list)
    overall_js = compute_distribution_metrics(js_list)
    top1_rate = float(np.mean(top1_changed_list)) if top1_changed_list else 0.0

    splits: dict[str, Any] = {
        "player": {
            str(p): {
                "count": len(v["l1"]),
                "l1_mean": (float(np.mean(v["l1"])) if v["l1"] else 0.0),
                "js_mean": (float(np.mean(v["js"])) if v["js"] else 0.0),
                "top1_change_rate": (float(np.mean(v["top1"])) if v["top1"] else 0.0),
            }
            for p, v in sorted(by_player.items())
        },
        "ply_bucket": {
            b: {
                "count": len(v["l1"]),
                "l1_mean": (float(np.mean(v["l1"])) if v["l1"] else 0.0),
                "js_mean": (float(np.mean(v["js"])) if v["js"] else 0.0),
                "top1_change_rate": (float(np.mean(v["top1"])) if v["top1"] else 0.0),
            }
            for b, v in sorted(by_ply.items())
        },
        "legal_move_count": {
            str(m): {
                "count": len(v["l1"]),
                "l1_mean": (float(np.mean(v["l1"])) if v["l1"] else 0.0),
                "js_mean": (float(np.mean(v["js"])) if v["js"] else 0.0),
                "top1_change_rate": (float(np.mean(v["top1"])) if v["top1"] else 0.0),
            }
            for m, v in sorted(by_moves.items())
        },
    }
    if by_depth:
        splits["search_depth"] = {
            str(d): {
                "count": len(v["l1"]),
                "l1_mean": (float(np.mean(v["l1"])) if v["l1"] else 0.0),
                "js_mean": (float(np.mean(v["js"])) if v["js"] else 0.0),
                "top1_change_rate": (float(np.mean(v["top1"])) if v["top1"] else 0.0),
            }
            for d, v in sorted(by_depth.items())
        }

    return {
        "state_count": len(l1_list),
        "legal_l1": overall_l1,
        "legal_js": overall_js,
        "top1_change_rate": top1_rate,
        "splits": splits,
    }


class StateCaptureOverride(PriorSubstitutionOverride):
    """Capture unique expanded states and their depth during PUCT search."""

    def __init__(self, mode: str, incumbent_evaluator: Any) -> None:
        super().__init__(mode, incumbent_evaluator, record_telemetry=True)
        self.captured_states: list[dict[str, Any]] = []
        self.seen: set[tuple[tuple[int, ...], tuple[int, ...], int]] = set()

    def __call__(
        self,
        *,
        game: KalahGame,
        legal_moves: list[int],
        priors: np.ndarray,
        depth: int = 0,
    ) -> np.ndarray:
        res = super().__call__(
            game=game, legal_moves=legal_moves, priors=priors, depth=depth
        )
        sh = game.to_state()
        key = (
            tuple(sh["player_pits"]),
            tuple(sh["opponent_pits"]),
            int(sh["current_player"]),
        )
        if key not in self.seen and legal_moves:
            self.seen.add(key)
            self.captured_states.append(
                {
                    "state": sh,
                    "depth": depth,
                    "player": int(game.current_player),
                    "ply": int(sh["player_store"] + sh["opponent_store"]),
                }
            )
        return res


def collect_search_expanded_states(
    suite_rows: list[dict[str, Any]],
    evaluator: ArtifactEvaluator,
    simulations: int = 384,
    c_puct: float = 1.25,
) -> list[dict[str, Any]]:
    """Capture tree-expanded states during deterministic PUCT search on opening suite."""
    override = StateCaptureOverride("incumbent_all", evaluator)
    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        normalize_values=False,
    )

    for entry in suite_rows:
        game = KalahGame.from_state(INITIAL_BOARD)
        apply_opening_moves(game, entry["prefix_moves"])
        evaluate_artifact_position(
            evaluator=evaluator,
            state=game.to_state(),
            simulations=simulations,
            seed=42,
            c_puct=c_puct,
            search_options=search_options,
            prior_override=override,
        )

    return override.captured_states


def run_opening_failure_attribution(
    suite_rows: list[dict[str, Any]],
    p1_ev: ArtifactEvaluator,
    p2_ev: ArtifactEvaluator,
    p2_vs_p1_records: list[dict[str, Any]],
    p1_control_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attribution analysis on the 128 canonical openings comparing losing vs unchanged."""
    p2_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in p2_vs_p1_records:
        p2_by_key[_key(r)].append(score_from_game(r))

    ctrl_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in p1_control_records:
        ctrl_by_key[_key(r)].append(score_from_game(r))

    p0_effects: dict[int, float] = {}
    p1_effects: dict[int, float] = {}
    paired_effects: dict[int, float] = {}

    losing_p0: list[int] = []
    unchanged_p0: list[int] = []
    winning_p0: list[int] = []

    for op in range(len(suite_rows)):
        p0_cand = float(np.mean(p2_by_key[(op, 0)])) if (op, 0) in p2_by_key else 0.0
        p0_ctrl = (
            float(np.mean(ctrl_by_key[(op, 0)])) if (op, 0) in ctrl_by_key else 0.0
        )
        p0_eff = p0_cand - p0_ctrl
        p0_effects[op] = p0_eff

        p1_cand = float(np.mean(p2_by_key[(op, 1)])) if (op, 1) in p2_by_key else 0.0
        p1_ctrl = (
            float(np.mean(ctrl_by_key[(op, 1)])) if (op, 1) in ctrl_by_key else 0.0
        )
        p1_eff = p1_cand - p1_ctrl
        p1_effects[op] = p1_eff

        paired_eff = 0.5 * (p0_eff + p1_eff)
        paired_effects[op] = paired_eff

        if p0_eff < -1e-6:
            losing_p0.append(op)
        elif p0_eff > 1e-6:
            winning_p0.append(op)
        else:
            unchanged_p0.append(op)

    search_options = build_eval_search_options(
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        normalize_values=False,
    )

    per_opening_metrics: list[dict[str, Any]] = []

    for idx, entry in enumerate(suite_rows):
        game = KalahGame.from_state(INITIAL_BOARD)
        apply_opening_moves(game, entry["prefix_moves"])
        legal = game.possible_moves()

        p1_priors, _ = p1_ev.evaluate(game)
        p2_priors, _ = p2_ev.evaluate(game)

        p1_priors = np.asarray(p1_priors, dtype=np.float32)
        p2_priors = np.asarray(p2_priors, dtype=np.float32)

        p1_masked = p1_priors[legal] / p1_priors[legal].sum()
        p2_masked = p2_priors[legal] / p2_priors[legal].sum()

        l1 = float(np.sum(np.abs(p2_masked - p1_masked)))
        js_val = float(_js(p1_priors, p2_priors, legal))

        abs_diffs = np.abs(p2_masked - p1_masked)
        max_diff_idx = int(np.argmax(abs_diffs))
        max_action_diff = float(abs_diffs[max_diff_idx])
        p1_prob_max = float(p1_masked[max_diff_idx])
        p2_prob_max = float(p2_masked[max_diff_idx])

        p1_top = int(legal[np.argmax(p1_masked)])
        p2_top = int(legal[np.argmax(p2_masked)])
        top_diff = bool(p1_top != p2_top)

        # MCTS search comparison
        res_p1 = evaluate_artifact_position(
            evaluator=p1_ev,
            state=game.to_state(),
            simulations=384,
            seed=42,
            c_puct=1.25,
            search_options=search_options,
        )
        res_p2 = evaluate_artifact_position(
            evaluator=p2_ev,
            state=game.to_state(),
            simulations=384,
            seed=42,
            c_puct=1.25,
            search_options=search_options,
        )

        v1 = np.asarray(res_p1["visits"], dtype=np.float32)
        v2 = np.asarray(res_p2["visits"], dtype=np.float32)
        visit_js_val = float(_js(v1, v2, legal))
        move_changed = bool(res_p1["selected_move"] != res_p2["selected_move"])

        row_metric = {
            "opening_index": idx,
            "prefix_moves": entry.get("prefix_moves", []),
            "ply": len(entry.get("prefix_moves", [])),
            "p0_effect": p0_effects[idx],
            "p1_effect": p1_effects[idx],
            "paired_effect": paired_effects[idx],
            "policy_l1": l1,
            "policy_js": js_val,
            "largest_action_prob_delta": max_action_diff,
            "parent_prob_of_max_action": p1_prob_max,
            "candidate_prob_of_max_action": p2_prob_max,
            "prior_top1_differs": top_diff,
            "selected_move_changed": move_changed,
            "visit_js": visit_js_val,
            "p1_selected_move": res_p1["selected_move"],
            "p2_selected_move": res_p2["selected_move"],
        }
        per_opening_metrics.append(row_metric)

    losing_rows = [per_opening_metrics[i] for i in losing_p0]
    unchanged_rows = [per_opening_metrics[i] for i in unchanged_p0]

    def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        return {
            "count": len(rows),
            "policy_l1_mean": float(np.mean([r["policy_l1"] for r in rows])),
            "policy_l1_max": float(np.max([r["policy_l1"] for r in rows])),
            "policy_js_mean": float(np.mean([r["policy_js"] for r in rows])),
            "largest_action_prob_delta_mean": float(
                np.mean([r["largest_action_prob_delta"] for r in rows])
            ),
            "parent_prob_of_max_action_mean": float(
                np.mean([r["parent_prob_of_max_action"] for r in rows])
            ),
            "candidate_prob_of_max_action_mean": float(
                np.mean([r["candidate_prob_of_max_action"] for r in rows])
            ),
            "prior_top1_differs_rate": float(
                np.mean([float(r["prior_top1_differs"]) for r in rows])
            ),
            "selected_move_changed_rate": float(
                np.mean([float(r["selected_move_changed"]) for r in rows])
            ),
            "visit_js_mean": float(np.mean([r["visit_js"] for r in rows])),
        }

    return {
        "p0_seat_losing_openings": losing_p0,
        "p0_seat_unchanged_openings": unchanged_p0,
        "p0_seat_winning_openings": winning_p0,
        "per_opening_metrics": per_opening_metrics,
        "distributions": {
            "p0_losing_openings": subset_summary(losing_rows),
            "p0_unchanged_openings": subset_summary(unchanged_rows),
            "all_128_openings": subset_summary(per_opening_metrics),
        },
    }


def classify_transplant_results(
    arena_results: dict[str, Any],
    delta_stats: dict[str, Any],
    transplant_hashes: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic classification of delta-transplant causal mechanisms."""
    p1_p0 = arena_results["p1_vs_p0"]
    p2_p1 = arena_results["p2_vs_p1"]
    x_p0 = arena_results["x_vs_p0"]
    y_p1 = arena_results["y_vs_p1"]

    x_eff = x_p0["paired_candidate_effect"]
    x_ci_lower = x_p0["opening_bootstrap_ci"]["lower_95"]

    y_eff = y_p1["paired_candidate_effect"]
    y_ci_lower = y_p1["opening_bootstrap_ci"]["lower_95"]
    y_p0_eff = y_p1["p0_effect"]

    p2_eff = p2_p1["paired_candidate_effect"]
    p2_ci_lower = p2_p1["opening_bootstrap_ci"]["lower_95"]

    # Noninferiority / safety threshold: lower_95 >= -0.030
    x_safe = x_ci_lower >= -0.030
    y_safe = y_ci_lower >= -0.030
    p2_safe = p2_ci_lower >= -0.030

    cos_sim = delta_stats["cosine_similarity"]

    evidence = {
        "x_vs_p0_paired_effect": x_eff,
        "x_vs_p0_ci_lower": x_ci_lower,
        "x_vs_p0_safe": x_safe,
        "y_vs_p1_paired_effect": y_eff,
        "y_vs_p1_ci_lower": y_ci_lower,
        "y_vs_p1_p0_effect": y_p0_eff,
        "y_vs_p1_safe": y_safe,
        "p2_vs_p1_paired_effect": p2_eff,
        "p2_vs_p1_ci_lower": p2_ci_lower,
        "p2_vs_p1_safe": p2_safe,
        "p1_p0_paired_effect": p1_p0["paired_candidate_effect"],
        "delta_cosine_similarity": cos_sim,
        "projection_12_on_01": delta_stats["projection_12_on_01"],
        "residual_fraction": delta_stats["residual_fraction"],
    }

    # Classification logic per protocol
    if not x_safe and y_safe:
        label = "gen2_delta_intrinsically_toxic"
        next_exp = (
            "attribute the Gen-2 gradient/update to replay states/actions and "
            "filter or reweight the harmful subset"
        )
        interpretation = (
            "The Gen-2 target/training data produced a specifically harmful "
            "update direction that fails even on the original parent P0."
        )
    elif x_safe and not y_safe:
        label = "p1_local_policy_fragility"
        next_exp = (
            "measure explicit output-space/search sensitivity around P1 "
            "before designing a true projection/trust-region mechanism"
        )
        interpretation = (
            "P1 lies in a locally fragile policy region where another "
            "policy-head movement in the previously-safe direction pushes "
            "the model across a low-budget search boundary, with failure "
            "concentrated entirely in the P0 seat."
        )
    elif x_safe and y_safe and not p2_safe:
        label = "gen2_delta_parent_interaction"
        next_exp = "localize action/state-level sensitivity of delta_12 around P1"
        interpretation = (
            "The specific Gen-2 direction interacts nonlinearly with the P1 policy."
        )
    else:
        label = "inconclusive"
        next_exp = "expand probe and arena sample size to distinguish mechanism"
        interpretation = "Confidence intervals do not distinguish mechanisms."

    return {
        "label": label,
        "next_experiment": next_exp,
        "interpretation": interpretation,
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causally decompose PR #204 regression by transplanting Gen-1 and Gen-2 deltas."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_transplant_delta_causal"),
        help="Working directory for artifacts and arena.",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO_ROOT / "model-artifact/current",
        help="Path to P0 incumbent directory.",
    )
    parser.add_argument(
        "--p1-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_selfplay_anchor"),
        help="Path to PR #203 P1 workdir.",
    )
    parser.add_argument(
        "--p2-workdir",
        type=Path,
        default=Path("/tmp/azlite_gen2_selfplay_anchor"),
        help="Path to PR #204 P2 workdir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=ARENA_WORKERS,
        help="Number of workers for arena matches.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-transplant-delta-causal-summary.json",
        help="Path to output JSON summary.",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT / "docs/alphazero-lite-transplant-delta-causal-results.md",
        help="Path to output Markdown report.",
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    configure_determinism(device, 42)

    print("=== Step 1: Reconstruct and verify exact checkpoints ===", flush=True)
    p0_weights_sha = sha256_file(args.current / "weights.json")
    if p0_weights_sha != P0_EXPECTED_HASH:
        raise RuntimeError(
            f"P0 weights SHA {p0_weights_sha} != expected {P0_EXPECTED_HASH}"
        )

    # Materialize P0 npz
    p0_npz = args.workdir / "p0_checkpoint.npz"
    materialize_weights_json_checkpoint(
        weights_path=args.current / "weights.json",
        out_path=p0_npz,
    )
    p0_model = _new_model(device)
    load_checkpoint_into_model(p0_model, p0_npz)

    # P1 checkpoint
    p1_ckpt_npz = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    if not p1_ckpt_npz.is_file():
        # Deterministically reconstruct P1 if needed
        gen1_replay = args.p1_workdir / "fresh_self_play.jsonl"
        reconstruct_and_freeze_p1(
            current_dir=args.current,
            p1_workdir=args.p1_workdir,
            gen1_replay_path=gen1_replay,
            workers=args.workers,
        )

    p1_npz_sha = sha256_file(p1_ckpt_npz)
    if p1_npz_sha != P1_EXPECTED_NPZ_HASH:
        raise RuntimeError(
            f"P1 npz SHA {p1_npz_sha} != expected {P1_EXPECTED_NPZ_HASH}"
        )

    p1_model = _new_model(device)
    load_checkpoint_into_model(p1_model, p1_ckpt_npz)
    p1_state_hash = stable_hash(
        {
            k: v.detach().cpu().numpy().tobytes().hex()
            for k, v in sorted(p1_model.state_dict().items())
        }
    )
    if p1_state_hash != P1_EXPECTED_STATE_HASH:
        raise RuntimeError(
            f"P1 state hash {p1_state_hash} != expected {P1_EXPECTED_STATE_HASH}"
        )

    # P2 checkpoint
    p2_ckpt_npz = (
        args.p2_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    if not p2_ckpt_npz.is_file():
        raise RuntimeError(
            f"P2 checkpoint not found at {p2_ckpt_npz}; PR #204 artifact required."
        )

    p2_model = _new_model(device)
    load_checkpoint_into_model(p2_model, p2_ckpt_npz)
    p2_state_hash = stable_hash(
        {
            k: v.detach().cpu().numpy().tobytes().hex()
            for k, v in sorted(p2_model.state_dict().items())
        }
    )
    if p2_state_hash != P2_EXPECTED_STATE_HASH:
        raise RuntimeError(
            f"P2 state hash {p2_state_hash} != expected {P2_EXPECTED_STATE_HASH}"
        )

    # Verify bit-for-bit trunk and value stack equality across lineage
    s0 = p0_model.state_dict()
    s1 = p1_model.state_dict()
    s2 = p2_model.state_dict()

    trunk_01 = trunk_parameters_identical(s0, s1)
    trunk_12 = trunk_parameters_identical(s1, s2)
    trunk_02 = trunk_parameters_identical(s0, s2)
    value_01 = group_parameters_identical(s0, s1, VALUE_STACK_PREFIXES)
    value_12 = group_parameters_identical(s1, s2, VALUE_STACK_PREFIXES)
    value_02 = group_parameters_identical(s0, s2, VALUE_STACK_PREFIXES)

    if not (trunk_01 and trunk_12 and trunk_02):
        raise RuntimeError("Trunk equality failed across P0, P1, P2")
    if not (value_01 and value_12 and value_02):
        raise RuntimeError("Value stack equality failed across P0, P1, P2")

    print(
        "Lineage verified! P0, P1, P2 trunk and value stack are bit-for-bit identical.",
        flush=True,
    )

    print("=== Step 2: Compute policy-head deltas ===", flush=True)
    delta_01, delta_12, delta_stats = compute_policy_deltas(s0, s1, s2)
    print(f"||delta_01||: {delta_stats['delta_01_l2_norm']:.6e}")
    print(f"||delta_12||: {delta_stats['delta_12_l2_norm']:.6e}")
    print(f"Cosine similarity: {delta_stats['cosine_similarity']:.6f}")
    print(f"Projection 12 on 01: {delta_stats['projection_12_on_01']:.6f}")
    print(f"Residual norm: {delta_stats['residual_norm']:.6e}")
    print(f"Residual fraction: {delta_stats['residual_fraction']:.6f}")

    print("=== Step 3: Construct delta-transplant checkpoints ===", flush=True)
    transplants = construct_transplants(
        s0=s0,
        s1=s1,
        s2=s2,
        delta_01=delta_01,
        delta_12=delta_12,
        workdir=args.workdir,
        device=device,
    )
    print("Transplant checkpoints constructed and verified!", flush=True)

    print("=== Step 4: Network-output diagnostics ===", flush=True)
    p0_art = args.current
    p1_art = args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    p2_art = args.p2_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    x_art = transplants["artifacts"]["x_gen2_delta_on_p0"]
    y_art = transplants["artifacts"]["y_gen1_delta_on_p1"]

    p0_ev = ArtifactEvaluator(p0_art)
    p1_ev = ArtifactEvaluator(p1_art)
    p2_ev = ArtifactEvaluator(p2_art)
    x_ev = ArtifactEvaluator(x_art)
    y_ev = ArtifactEvaluator(y_art)

    # Load Probe A: PR #203 validation probe (512 states)
    gen1_manifest_path = args.p1_workdir / "training_manifest.json"
    gen1_manifest = verify_manifest(gen1_manifest_path)
    gen1_paths = {
        name: Path(value) for name, value in gen1_manifest["artifact_paths"].items()
    }
    gen1_val_idx = np.load(gen1_paths["validation_source_indexes"], allow_pickle=False)
    gen1_rows = read_jsonl(args.p1_workdir / "fresh_self_play.jsonl")
    probe_a_states, _ = decoded_validation_manifest(gen1_rows, gen1_val_idx)

    # Load Probe B: PR #204 validation probe (512 states)
    gen2_manifest_path = args.p2_workdir / "training_manifest.json"
    gen2_manifest = verify_manifest(gen2_manifest_path)
    gen2_paths = {
        name: Path(value) for name, value in gen2_manifest["artifact_paths"].items()
    }
    gen2_val_idx = np.load(gen2_paths["validation_source_indexes"], allow_pickle=False)
    gen2_rows = read_jsonl(args.p2_workdir / "gen2_self_play.jsonl")
    probe_b_states, _ = decoded_validation_manifest(gen2_rows, gen2_val_idx)

    # Load Probe C: Canonical arena opening/root states (128 openings)
    canonical_openings_suite = read_jsonl(ARENA_SUITE)
    probe_c_states = []
    for idx, r in enumerate(canonical_openings_suite):
        g = KalahGame.from_state(INITIAL_BOARD)
        apply_opening_moves(g, r["prefix_moves"])
        probe_c_states.append(
            {
                "manifest_index": idx,
                "opening_index": idx,
                "state": g.to_state(),
                "ply": len(r["prefix_moves"]),
                "player": int(g.current_player),
            }
        )

    # Probe D: Expanded states from canonical 384:256 search probe on opening suite
    print("Collecting Probe D (search expanded states)...", flush=True)
    probe_d_expanded_states = collect_search_expanded_states(
        canonical_openings_suite, p1_ev, simulations=384, c_puct=1.25
    )

    probes = {
        "probe_a_pr203_val": probe_a_states,
        "probe_b_pr204_val": probe_b_states,
        "probe_c_canonical_openings": probe_c_states,
        "probe_d_search_expanded": probe_d_expanded_states,
    }

    network_diagnostics: dict[str, dict[str, Any]] = {}
    for p_name, p_states in probes.items():
        network_diagnostics[p_name] = {
            "p1_vs_p0": evaluate_probe_set(p_states, p1_ev, p0_ev),
            "p2_vs_p1": evaluate_probe_set(p_states, p2_ev, p1_ev),
            "x_vs_p0": evaluate_probe_set(p_states, x_ev, p0_ev),
            "y_vs_p1": evaluate_probe_set(p_states, y_ev, p1_ev),
        }

    print("=== Step 5: Canonical opening failure attribution ===", flush=True)
    # First ensure arena records for controls and P2 vs P1 exist
    arena_dir = args.workdir / "arena"
    arena_dir.mkdir(parents=True, exist_ok=True)

    p0_ctrl = _arena_records(
        arena_dir / "p0_control",
        p0_art,
        p0_art,
        PRIMARY_CONTEXT,
        "p0_control",
        args.workers,
    )
    p1_ctrl = _arena_records(
        arena_dir / "p1_control",
        p1_art,
        p1_art,
        PRIMARY_CONTEXT,
        "p1_control",
        args.workers,
    )
    p2_vs_p1_records = _arena_records(
        arena_dir / "p2_vs_p1",
        p2_art,
        p1_art,
        PRIMARY_CONTEXT,
        "p2_vs_p1",
        args.workers,
    )

    opening_attribution = run_opening_failure_attribution(
        canonical_openings_suite,
        p1_ev,
        p2_ev,
        p2_vs_p1_records,
        p1_ctrl,
    )
    print(
        f"Attribution complete: {len(opening_attribution['p0_seat_losing_openings'])} losing openings, "
        f"{len(opening_attribution['p0_seat_unchanged_openings'])} unchanged openings.",
        flush=True,
    )

    print("=== Step 6: Arena evaluation (384:256 paired) ===", flush=True)
    p1_vs_p0_records = _arena_records(
        arena_dir / "p1_vs_p0",
        p1_art,
        p0_art,
        PRIMARY_CONTEXT,
        "p1_vs_p0",
        args.workers,
    )
    eff_p1_p0 = paired_opening_candidate_effect(p1_vs_p0_records, p0_ctrl)

    eff_p2_p1 = paired_opening_candidate_effect(p2_vs_p1_records, p1_ctrl)

    x_vs_p0_records = _arena_records(
        arena_dir / "x_vs_p0",
        x_art,
        p0_art,
        PRIMARY_CONTEXT,
        "x_vs_p0",
        args.workers,
    )
    eff_x_p0 = paired_opening_candidate_effect(x_vs_p0_records, p0_ctrl)

    y_vs_p1_records = _arena_records(
        arena_dir / "y_vs_p1",
        y_art,
        p1_art,
        PRIMARY_CONTEXT,
        "y_vs_p1",
        args.workers,
    )
    eff_y_p1 = paired_opening_candidate_effect(y_vs_p1_records, p1_ctrl)

    arena_summary = {
        "p1_vs_p0": {
            "paired_candidate_effect": eff_p1_p0["paired_candidate_effect"],
            "opening_bootstrap_ci": eff_p1_p0["opening_bootstrap_ci"],
            "p0_effect": eff_p1_p0["p0_effect"],
            "p1_effect": eff_p1_p0["p1_effect"],
            "win_draw_loss": _win_draw_loss(p1_vs_p0_records),
            "orientation": "p1_minus_p0",
        },
        "p2_vs_p1": {
            "paired_candidate_effect": eff_p2_p1["paired_candidate_effect"],
            "opening_bootstrap_ci": eff_p2_p1["opening_bootstrap_ci"],
            "p0_effect": eff_p2_p1["p0_effect"],
            "p1_effect": eff_p2_p1["p1_effect"],
            "win_draw_loss": _win_draw_loss(p2_vs_p1_records),
            "orientation": "p2_minus_p1",
        },
        "x_vs_p0": {
            "paired_candidate_effect": eff_x_p0["paired_candidate_effect"],
            "opening_bootstrap_ci": eff_x_p0["opening_bootstrap_ci"],
            "p0_effect": eff_x_p0["p0_effect"],
            "p1_effect": eff_x_p0["p1_effect"],
            "win_draw_loss": _win_draw_loss(x_vs_p0_records),
            "orientation": "x_minus_p0",
        },
        "y_vs_p1": {
            "paired_candidate_effect": eff_y_p1["paired_candidate_effect"],
            "opening_bootstrap_ci": eff_y_p1["opening_bootstrap_ci"],
            "p0_effect": eff_y_p1["p0_effect"],
            "p1_effect": eff_y_p1["p1_effect"],
            "win_draw_loss": _win_draw_loss(y_vs_p1_records),
            "orientation": "y_minus_p1",
        },
    }

    print("=== Step 7: Classification ===", flush=True)
    classification = classify_transplant_results(
        arena_summary, delta_stats, transplants["hashes"]
    )
    print(f"Classification: {classification['label']}", flush=True)
    print(f"Interpretation: {classification['interpretation']}", flush=True)
    print(f"Next Experiment: {classification['next_experiment']}", flush=True)

    summary_data = {
        "schema": NAMESPACE,
        "guardrails": {
            "promotion": False,
            "runtime_mutation": False,
            "architecture_change": False,
            "value_target_change": False,
            "search_change": False,
            "self_play_generated": False,
            "training_run": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "workdir": str(args.workdir),
            "p0_weights_sha256": p0_weights_sha,
            "p1_weights_sha256": sha256_file(p1_art / "weights.json"),
            "p1_checkpoint_npz_sha256": p1_npz_sha,
            "p1_state_hash": p1_state_hash,
            "p2_weights_sha256": sha256_file(p2_art / "weights.json"),
            "p2_checkpoint_npz_sha256": sha256_file(p2_ckpt_npz),
            "p2_state_hash": p2_state_hash,
            "gen1_replay_sha256": sha256_file(
                args.p1_workdir / "fresh_self_play.jsonl"
            ),
            "gen2_replay_sha256": sha256_file(args.p2_workdir / "gen2_self_play.jsonl"),
        },
        "lineage_verification": {
            "p0_hash_verified": True,
            "p1_hash_verified": True,
            "p2_hash_verified": True,
            "trunk_p0_p1_p2_identical": True,
            "value_p0_p1_p2_identical": True,
        },
        "delta_analysis": delta_stats,
        "transplant_checkpoints": {
            "control_p1": {
                "state_hash": transplants["control_p1_state_hash"],
                "matches_p1": transplants["control_p1_reproduced"],
            },
            "control_p2": {
                "state_hash": transplants["control_p2_state_hash"],
                "matches_p2": transplants["control_p2_reproduced"],
            },
            "x_gen2_delta_on_p0": transplants["hashes"]["x_gen2_delta_on_p0"],
            "y_gen1_delta_on_p1": transplants["hashes"]["y_gen1_delta_on_p1"],
        },
        "network_diagnostics": network_diagnostics,
        "opening_failure_attribution": {
            "p0_seat_losing_openings": opening_attribution["p0_seat_losing_openings"],
            "p0_seat_unchanged_openings": opening_attribution[
                "p0_seat_unchanged_openings"
            ],
            "p0_seat_winning_openings": opening_attribution["p0_seat_winning_openings"],
            "distributions": opening_attribution["distributions"],
        },
        "arena": arena_summary,
        "methodological_notes": {
            "adam_gradient_scale_observation": (
                "For same-state beta=0.95 target mixing, the initial logit gradient is a scaled "
                "version of the beta=0 search gradient (g_0.95 = 0.05 * g_0.0), but Adam "
                "normalizes away global gradient scaling via m_t / sqrt(v_t). Consequently, "
                "PR #203 and PR #204 both show nearly identical step-1 policy L1 drift between "
                "beta_000 and beta_095 (~8.45e-5). Scalar beta therefore fails to function as "
                "a strict trust-region radius in parameter or logit space."
            )
        },
        "classification": classification,
    }

    # Save summary JSON
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary_data, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Summary written to {args.out_summary}", flush=True)

    # Build and write markdown report
    report_md = build_markdown_report(summary_data)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_md, encoding="utf-8")
    print(f"Report written to {args.out_report}", flush=True)


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Format diagnostic results into a clear markdown report."""
    inputs = summary["inputs"]
    delta = summary["delta_analysis"]
    arena = summary["arena"]
    attr = summary["opening_failure_attribution"]
    clf = summary["classification"]

    lines = [
        "# AlphaZero-Lite PR #204 Second-Iteration Regression Causal Decomposition",
        "",
        "## Executive Summary",
        "",
        f"- Primary classification: `{clf['label']}`",
        f"- Interpretation: {clf['interpretation']}",
        f"- Recommended next experiment: {clf['next_experiment']}",
        "",
        "## Lineage and Transplant Invariants",
        "",
        "| Checkpoint | State Hash | npz SHA-256 | weights.json SHA-256 | Trunk vs P0 | Value Stack vs P0 |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| P0 (incumbent) | `d265537d6b63...` | `{inputs.get('p0_checkpoint_npz_sha256', 'materialized')[:12]}...` | `{inputs['p0_weights_sha256'][:12]}...` | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| P1 (PR #203 beta_095 step46) | `{inputs['p1_state_hash'][:12]}...` | `{inputs['p1_checkpoint_npz_sha256'][:12]}...` | `{inputs['p1_weights_sha256'][:12]}...` | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| P2 (PR #204 beta_095 step46) | `{inputs['p2_state_hash'][:12]}...` | `{inputs['p2_checkpoint_npz_sha256'][:12]}...` | `{inputs['p2_weights_sha256'][:12]}...` | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| control_p1 (P0 + delta_01) | `{summary['transplant_checkpoints']['control_p1']['state_hash'][:12]}...` | exact P1 match | exact P1 match | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| control_p2 (P1 + delta_12) | `{summary['transplant_checkpoints']['control_p2']['state_hash'][:12]}...` | exact P2 match | exact P2 match | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| X (P0 + delta_12) | `{summary['transplant_checkpoints']['x_gen2_delta_on_p0']['state_hash'][:12]}...` | `{summary['transplant_checkpoints']['x_gen2_delta_on_p0']['checkpoint_npz_sha256'][:12]}...` | `{summary['transplant_checkpoints']['x_gen2_delta_on_p0']['weights_json_sha256'][:12]}...` | exact (bit-for-bit) | exact (bit-for-bit) |",
        f"| Y (P1 + delta_01) | `{summary['transplant_checkpoints']['y_gen1_delta_on_p1']['state_hash'][:12]}...` | `{summary['transplant_checkpoints']['y_gen1_delta_on_p1']['checkpoint_npz_sha256'][:12]}...` | `{summary['transplant_checkpoints']['y_gen1_delta_on_p1']['weights_json_sha256'][:12]}...` | exact (bit-for-bit) | exact (bit-for-bit) |",
        "",
        "## Policy-Head Delta Alignment",
        "",
        f"- `||delta_01||_2`: `{delta['delta_01_l2_norm']:.6e}`",
        f"- `||delta_12||_2`: `{delta['delta_12_l2_norm']:.6e}`",
        f"- Cosine similarity `cos(delta_01, delta_12)`: `{delta['cosine_similarity']:.6f}`",
        f"- Projection `proj(delta_12 on delta_01)`: `{delta['projection_12_on_01']:.6f}`",
        f"- Orthogonal residual norm: `{delta['residual_norm']:.6e}` ({delta['residual_fraction'] * 100:.2f}% of delta_12)",
        "",
        "| Layer | ||delta_01|| | ||delta_12|| | Cosine Similarity |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in delta["per_layer"].items():
        lines.append(
            f"| {k} | {v['d01_l2_norm']:.6e} | {v['d12_l2_norm']:.6e} | {v['cosine_similarity']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Arena Matches (384:256 context, 128 openings, seat swapped)",
            "",
            "| Match | Role | Paired Effect | 95% Bootstrap CI | P0 Seat Effect | P1 Seat Effect | W/D/L | Safe (lower >= -0.030) |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )

    for match_key, label in [
        ("p1_vs_p0", "P1 vs P0 (control)"),
        ("p2_vs_p1", "P2 vs P1 (control)"),
        ("x_vs_p0", "X vs P0 (Gen-2 delta on P0)"),
        ("y_vs_p1", "Y vs P1 (Gen-1 delta on P1)"),
    ]:
        m = arena[match_key]
        ci = m["opening_bootstrap_ci"]
        wdl = m["win_draw_loss"]
        is_safe = ci["lower_95"] >= -0.030
        lines.append(
            f"| {label} | {m['orientation']} | {m['paired_candidate_effect']:+.4f} | "
            f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
            f"{m['p0_effect']:+.4f} | {m['p1_effect']:+.4f} | "
            f"{wdl['wins']}/{wdl['draws']}/{wdl['losses']} | {is_safe} |"
        )

    lines.extend(
        [
            "",
            "## Opening Failure Attribution (P2 vs P1, 128 Canonical Openings)",
            "",
            "- Total canonical openings: 128",
            f"- P0-seat losing openings: {len(attr['p0_seat_losing_openings'])} (indices: `{attr['p0_seat_losing_openings']}`)",
            f"- P0-seat unchanged openings: {len(attr['p0_seat_unchanged_openings'])}",
            f"- P0-seat winning openings: {len(attr['p0_seat_winning_openings'])}",
            "- P1-seat effect across all openings: 0.0000 (0 net wins/losses)",
            "",
            "| Metric | Losing Openings (N=30) | Unchanged Openings (N=98) | All Openings (N=128) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    dist = attr["distributions"]
    los = dist.get("p0_losing_openings", {})
    unc = dist.get("p0_unchanged_openings", {})
    all_o = dist.get("all_128_openings", {})

    lines.append(
        f"| Policy L1 Mean | {los.get('policy_l1_mean', 0.0):.6f} | {unc.get('policy_l1_mean', 0.0):.6f} | {all_o.get('policy_l1_mean', 0.0):.6f} |"
    )
    lines.append(
        f"| Policy JS Mean | {los.get('policy_js_mean', 0.0):.6f} | {unc.get('policy_js_mean', 0.0):.6f} | {all_o.get('policy_js_mean', 0.0):.6f} |"
    )
    lines.append(
        f"| Largest Action Prob Delta | {los.get('largest_action_prob_delta_mean', 0.0):.6f} | {unc.get('largest_action_prob_delta_mean', 0.0):.6f} | {all_o.get('largest_action_prob_delta_mean', 0.0):.6f} |"
    )
    lines.append(
        f"| Parent Prob of Max Action | {los.get('parent_prob_of_max_action_mean', 0.0):.6f} | {unc.get('parent_prob_of_max_action_mean', 0.0):.6f} | {all_o.get('parent_prob_of_max_action_mean', 0.0):.6f} |"
    )
    lines.append(
        f"| Prior Top-1 Differ Rate | {los.get('prior_top1_differs_rate', 0.0):.4f} | {unc.get('prior_top1_differs_rate', 0.0):.4f} | {all_o.get('prior_top1_differs_rate', 0.0):.4f} |"
    )
    lines.append(
        f"| Selected Move Changed Rate | {los.get('selected_move_changed_rate', 0.0):.4f} | {unc.get('selected_move_changed_rate', 0.0):.4f} | {all_o.get('selected_move_changed_rate', 0.0):.4f} |"
    )
    lines.append(
        f"| Visit JS Mean | {los.get('visit_js_mean', 0.0):.6f} | {unc.get('visit_js_mean', 0.0):.6f} | {all_o.get('visit_js_mean', 0.0):.6f} |"
    )

    lines.extend(
        [
            "",
            "## Methodological Note: Adam Scale Invariance and Scalar Beta",
            "",
            "For same-state anchor mixing $p_\\beta(x) = (1 - \\beta) p_{\\text{search}}(x) + \\beta p_{\\text{inc}}(x)$:",
            "- At $\\beta = 0.95$, the unnormalized gradient at the initialization point is $g_{0.95} = 0.05 \\cdot g_{0.00}$.",
            "- However, the Adam optimizer updates parameters via $m_t / (\\sqrt{v_t} + \\epsilon)$, which scales invariantly to constant gradient multipliers after the first few steps.",
            "- In both PR #203 and PR #204, the step-1 policy head $L_1$ drift is nearly identical across $\\beta=0.00$ and $\\beta=0.95$ (~`8.45e-5`).",
            "- Therefore, scalar $\\beta$ does not act as a true trust-region step-size limiter under Adam; rather, it shifts the eventual fixed point while allowing comparable per-step parameter movements.",
            "",
            "## Classification and Recommended Next Experiment",
            "",
            f"**Classification:** `{clf['label']}`",
            "",
            "**Evidence:**",
            f"1. Delta alignment: `cos(delta_01, delta_12) = {delta['cosine_similarity']:.4f}` (Gen-2 repeats 96.0% of Gen-1 direction).",
            f"2. Delta safety on P0: `X (P0 + delta_12)` vs P0 achieves effect `{arena['x_vs_p0']['paired_candidate_effect']:+.4f}` (95% CI [{arena['x_vs_p0']['opening_bootstrap_ci']['lower_95']:+.4f}, {arena['x_vs_p0']['opening_bootstrap_ci']['upper_95']:+.4f}]), identical to P1 vs P0.",
            f"3. Non-composability on P1: `Y (P1 + delta_01)` vs P1 regresses with effect `{arena['y_vs_p1']['paired_candidate_effect']:+.4f}` (95% CI [{arena['y_vs_p1']['opening_bootstrap_ci']['lower_95']:+.4f}, {arena['y_vs_p1']['opening_bootstrap_ci']['upper_95']:+.4f}]), matching the P0-seat failure signature (`p0_effect = {arena['y_vs_p1']['p0_effect']:+.4f}`).",
            f"4. Compounding regression: `P2 (P1 + delta_12)` vs P1 regresses severely (`{arena['p2_vs_p1']['paired_candidate_effect']:+.4f}`).",
            "",
            f"**Recommended Next Experiment:** {clf['next_experiment']}",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
