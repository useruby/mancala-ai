#!/usr/bin/env python3
# ruff: noqa: E402
"""Localize the PR #200 ``policy_head`` regression inside MCTS.

PR #200 classified the frozen-trunk multi-step failure as
``policy_head_accumulation``: at optimizer step 46 the isolated ``policy_head``
lane regresses strongly at 384:256 (paired effect ~= -0.1895, CI excluding
zero) despite a tiny policy-output drift (~0.0203 legal L1, ~9.1e-5 legal JS,
~1.95% top-1 changes), while the value head is neutral. The candidate's trunk
and value stack are byte-identical to the incumbent, so only the policy prior
differs. This experiment asks whether the harmful policy prior acts primarily at
the root, in shallow tree levels, or throughout MCTS.

We reuse the canonical PUCT search (``self_play.PUCT``) extended with a single
``prior_override`` callable applied at every expanded node with the node's
tree-search depth (see ``policy_prior_localization.py``). No second MCTS is
implemented; ``c_puct``, simulations, value outputs, and model weights are
unchanged.

Interventions (depth = tree-search depth from the current move's root):

- ``candidate_all``    : candidate policy at every node (PR #200 baseline).
- ``incumbent_root``   : incumbent policy at root only.
- ``incumbent_depth1`` : incumbent policy at depths 0 and 1.
- ``incumbent_depth2`` : incumbent policy at depths 0, 1, and 2.
- ``incumbent_all``    : incumbent policy at every node (candidate value path
  unchanged; search-equivalent to the incumbent by the PR #200 frozen-family
  invariant).

Evaluation reuses the canonical paired arena (128 openings, seat swap, seed
contract, bootstrap methodology) versus the frozen incumbent at 384:256 and
1200:1200, plus the frozen PR #200 PUCT probe (256 validation states, 384:256)
reporting per-depth override telemetry and root search deltas versus
``candidate_all``. No model is trained in this PR.
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

from ml.alphazero_lite.arena import ArtifactEvaluator, evaluate_artifact_position  # noqa: E402
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect  # noqa: E402
from ml.alphazero_lite.evaluation_seed_contract import stable_hash  # noqa: E402
from ml.alphazero_lite.policy_prior_localization import (  # noqa: E402
    PRIOR_OVERRIDE_MODES,
    build_prior_substitution_override,
    summarize_override_telemetry,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_frozen_trunk_distillation_ablation import (  # noqa: E402
    ARENA_CONTEXTS,
    trunk_parameters_identical,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (  # noqa: E402
    parse_game_jsonl,
    run_arena,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import CURRENT_HASH  # noqa: E402
from ml.alphazero_lite.run_policy_detached_trunk_ablation import (  # noqa: E402
    ARENA_SUITE,
    _context_c_puct,
)
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (  # noqa: E402
    decoded_validation_manifest,
    js,
    model_outputs,
)
from ml.alphazero_lite.self_play import build_eval_search_options  # noqa: E402
from ml.alphazero_lite.train import legal_mask_matrix_for_encoded_states  # noqa: E402

NAMESPACE = "azlite_frozen_trunk_policy_prior_localization_v1"
TRAINABLE_LANES = (
    "candidate_all",
    "incumbent_root",
    "incumbent_depth1",
    "incumbent_depth2",
    "incumbent_all",
)
PRIMARY_CONTEXT = "384:256"
PUCT_CONTEXT = "384:256"
PROBE_SIZE = 256
VALUE_STACK_PREFIXES = ("value_hidden_layer.", "value_head.")
POLICY_STACK_PREFIXES = ("policy_hidden_layer.", "policy_head.", "move_projections.")

# PR #200 (commit 11f5623) recorded measurements for the step-46 policy_head lane;
# candidate_all (same artifact, no override) must reproduce them exactly under
# the deterministic contract.
PR200_CANDIDATE_STATE_HASH = (
    "ee12bcd171ee95f54fd4d400c955d9469a7eca565995eeebd566a4a56dcba4e0"
)
PR200_INCUMBENT_STATE_HASH = (
    "d265537d6b637b8433b093ebb1f9d55fef25b38e259ed7e59f7a597b35bb6f02"
)
PR200_ARENA = {
    "384:256": {
        "paired_candidate_effect": -0.189453125,
        "opening_bootstrap_ci": {"lower_95": -0.232421875, "upper_95": -0.146484375},
        "p0_effect": -0.3671875,
        "p1_effect": -0.01171875,
    },
    "1200:1200": {
        "paired_candidate_effect": -0.060546875,
        "opening_bootstrap_ci": {"lower_95": -0.080078125, "upper_95": -0.04296875},
        "p0_effect": -0.12109375,
        "p1_effect": 0.0,
    },
}
PR200_PROBE_DRIFT = {
    "policy_legal_l1": 0.020320624113082886,
    "policy_legal_js": 9.053084068000317e-05,
    "policy_top1_change": 0.01953125,
    "value_mean_abs_delta": 0.0,
}

RECOVERY_THRESHOLD = 0.70
CI_MATERIAL_CLOSER_FACTOR = 0.5
EFFECT_TOLERANCE = 1e-6

NEXT_ACTIONS = {
    "root_prior_causal": (
        "root/search-state-specific policy anchoring or target filtering: anchor the "
        "policy head to the incumbent prior on root search states (or filter policy "
        "targets to root-reached states) rather than applying global behavior anchoring "
        "across the whole replay"
    ),
    "shallow_prior_compounding": (
        "anchor/regularize priors on the states actually reached in the first few MCTS "
        "levels (depths 0-2) of the low-budget search, e.g. a depth-weighted policy "
        "trust region or same-state incumbent-prior anchoring on shallow tree nodes"
    ),
    "distributed_prior_compounding": (
        "change policy-target construction or apply same-state incumbent-prior "
        "constraints throughout the replay (a global policy trust region on legal-policy "
        "divergence from the incumbent) rather than only reducing the learning rate"
    ),
    "unexpected_nonpolicy_difference": (
        "investigate the incumbent_all equivalence failure before any training change: "
        "the PR #200 frozen-family invariants predict search equivalence, so a failure "
        "implies an implementation/provenance discrepancy in the override plumbing or "
        "artifact materialization"
    ),
    "inconclusive": (
        "the intervention differences cannot be distinguished sufficiently; rerun the "
        "arena with more openings/seeds before choosing a training intervention"
    ),
}

FINDINGS = {
    "root_prior_causal": (
        "The harmful policy prior acts primarily at the root: substituting the incumbent "
        "prior at the root alone recovers the large majority of the 384:256 paired "
        "deficit, deeper substitutions add little, and the root-only CI moves materially "
        "toward zero."
    ),
    "shallow_prior_compounding": (
        "The harmful policy prior compounds over the first few MCTS levels: root-only "
        "substitution is insufficient, but substituting through depth 1 or 2 recovers "
        "the large majority of the 384:256 paired deficit."
    ),
    "distributed_prior_compounding": (
        "The harmful policy prior is distributed throughout the tree: shallow "
        "substitutions are insufficient and only substituting the incumbent prior at "
        "every node (incumbent_all) rescues the deficit."
    ),
    "unexpected_nonpolicy_difference": (
        "incumbent_all does not reproduce incumbent-equivalent search despite the PR #200 "
        "frozen-family invariants; the result reflects a non-policy discrepancy that "
        "must be investigated before any training change."
    ),
    "inconclusive": (
        "The intervention confidence intervals cannot separate root, shallow, and "
        "distributed mechanisms."
    ),
}


def _load_state(snapshot_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    return stable_hash(
        {name: value.numpy().tobytes().hex() for name, value in state.items()}
    )


def _group_identical(
    state: dict[str, torch.Tensor],
    incumbent: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> bool:
    names = [name for name in sorted(state) if name.startswith(prefixes)]
    if not names:
        raise ValueError(f"no parameters matched prefixes: {prefixes}")
    return all(
        np.array_equal(state[name].numpy(), incumbent[name].numpy()) for name in names
    )


def _group_drift(
    state: dict[str, torch.Tensor], incumbent: dict[str, torch.Tensor]
) -> float:
    left = torch.cat(
        [
            v.reshape(-1)
            for k, v in sorted(state.items())
            if k.startswith(
                ("policy_hidden_layer.", "policy_head.", "move_projections.")
            )
        ]
    )
    right = torch.cat(
        [
            v.reshape(-1)
            for k, v in sorted(incumbent.items())
            if k.startswith(
                ("policy_hidden_layer.", "policy_head.", "move_projections.")
            )
        ]
    )
    return float(
        torch.linalg.vector_norm(left - right)
        / (torch.linalg.vector_norm(right) + 1e-20)
    )


def verify_invariants(
    candidate_snapshot: Path,
    incumbent_snapshot: Path,
    probe_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the PR #200 frozen-family invariants on the candidate artifact."""
    candidate = _load_state(candidate_snapshot)
    incumbent = _load_state(incumbent_snapshot)

    candidate_hash = _state_hash(candidate)
    incumbent_hash = _state_hash(incumbent)

    x = np.asarray([row["state"] for row in probe_rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x).astype(bool)
    cand_policy, cand_value = model_outputs(candidate, x, mask)
    inc_policy, inc_value = model_outputs(incumbent, x, mask)

    value_equal = bool(np.allclose(cand_value, inc_value, atol=1e-7, rtol=0.0))
    value_mean_abs_delta = float(np.mean(np.abs(cand_value - inc_value)))
    policy_legal_l1 = float(np.mean(np.sum(np.abs(cand_policy - inc_policy), axis=1)))
    policy_legal_js = float(np.mean(js(cand_policy, inc_policy)))
    policy_top1_change = float(
        np.mean(np.argmax(cand_policy, axis=1) != np.argmax(inc_policy, axis=1))
    )

    return {
        "candidate_state_hash": candidate_hash,
        "candidate_hash_matches_pr200": candidate_hash == PR200_CANDIDATE_STATE_HASH,
        "incumbent_state_hash": incumbent_hash,
        "incumbent_hash_matches_pr200": incumbent_hash == PR200_INCUMBENT_STATE_HASH,
        "trunk_identical_to_incumbent": trunk_parameters_identical(
            candidate, incumbent
        ),
        "value_stack_identical_to_incumbent": _group_identical(
            candidate, incumbent, VALUE_STACK_PREFIXES
        ),
        "policy_stack_changed": not _group_identical(
            candidate, incumbent, POLICY_STACK_PREFIXES
        ),
        "value_outputs_equal_on_probe": value_equal,
        "value_mean_abs_delta": value_mean_abs_delta,
        "policy_legal_l1": policy_legal_l1,
        "policy_legal_js": policy_legal_js,
        "policy_top1_change": policy_top1_change,
        "policy_head_drift": _group_drift(candidate, incumbent),
        "pr200_probe_drift_matches": (
            abs(policy_legal_l1 - PR200_PROBE_DRIFT["policy_legal_l1"]) < 1e-6
            and abs(policy_legal_js - PR200_PROBE_DRIFT["policy_legal_js"]) < 1e-7
            and abs(value_mean_abs_delta - PR200_PROBE_DRIFT["value_mean_abs_delta"])
            < 1e-9
        ),
    }


def _win_draw_loss(records: list[dict[str, Any]]) -> dict[str, int]:
    wins = losses = draws = 0
    for record in records:
        winner = record.get("winner")
        if winner == "challenger":
            wins += 1
        elif winner == "current":
            losses += 1
        else:
            draws += 1
    return {"wins": wins, "draws": draws, "losses": losses}


def _intervention_records(
    workdir: Path,
    challenger: Path,
    current: Path,
    context: str,
    role: str,
    workers: int,
    mode: str | None,
) -> list[dict[str, Any]]:
    """Run one matched-current arena comparison, caching by exact provenance."""
    challenger_hash = sha256_file(challenger / "weights.json")
    incumbent_hash = sha256_file(current / "weights.json")
    suite_hash = sha256_file(ARENA_SUITE)
    sims, current_sims = (int(value) for value in context.split(":"))
    records: list[dict[str, Any]] = []
    for seat in (0, 1):
        directory = workdir / context.replace(":", "_") / role / f"starts_{seat}"
        rec_path = directory / "arena.jsonl"
        prov_path = directory / "provenance.json"
        manifest = {
            "schema_version": 2,
            "challenger_hash": challenger_hash,
            "incumbent_hash": incumbent_hash,
            "opening_suite_hash": suite_hash,
            "search_budgets": [sims, current_sims],
            "effective_c_puct": _context_c_puct(context),
            "tactical_root_bias": 0.0,
            "root_policy_mode": "deterministic",
            "seed_contract": "canonical_v1",
            "base_seed": 42,
            "games_per_opening": 2,
            "challenger_prior_override_mode": mode
            if mode is not None
            else "candidate_all",
        }
        reusable = rec_path.is_file() and prov_path.is_file()
        if reusable:
            cached = json.loads(prov_path.read_text(encoding="utf-8"))
            reusable = all(
                cached.get(key) == value for key, value in manifest.items()
            ) and (cached.get("arena_output_hash") == sha256_file(rec_path))
        if not reusable:
            directory.mkdir(parents=True, exist_ok=True)
            run_arena(
                challenger=str(challenger),
                current=str(current),
                challenger_sims=sims,
                current_sims=current_sims,
                games=256,
                seed=42,
                workers=workers,
                out_json=str(directory / "arena.json"),
                out_jsonl=str(rec_path),
                opening_prefixes_jsonl=str(ARENA_SUITE),
                challenger_starts=seat,
                games_per_opening=2,
                root_policy_mode="deterministic",
                root_temperature=0.0,
                normalize_values=False,
                c_puct=_context_c_puct(context),
                tactical_root_bias=0.0,
                seed_ledger_output=str(directory / "seed_ledger.jsonl"),
                challenger_prior_override_mode=mode,
            )
            prov_path.write_text(
                json.dumps(
                    manifest | {"arena_output_hash": sha256_file(rec_path)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        records.extend(parse_game_jsonl(str(rec_path)))
    return records


def _trajectory_divergence(
    baseline: list[dict[str, Any]], intervention: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare intervention trajectories to the candidate_all baseline.

    Games are matched by (opening_index, game_within_opening, challenger_player).
    Reports the fraction of games whose move trajectory diverges from baseline
    and the mean first-divergence ply (counting only plies before divergence).
    """

    def _game_key(record: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(record["opening_index"]),
            int(record["game_within_opening"]),
            int(record["challenger_player"]),
        )

    base = {_game_key(r): r for r in baseline}
    games = 0
    diverged = 0
    first_div_plies: list[int] = []
    for record in intervention:
        k = _game_key(record)
        if k not in base:
            continue
        games += 1
        bt = [int(m) for m in str(base[k].get("trajectory", "")).split(",") if m != ""]
        it = [int(m) for m in str(record.get("trajectory", "")).split(",") if m != ""]
        diverge = next(
            (ply for ply, (a, b) in enumerate(zip(bt, it)) if a != b),
            None,
        )
        if diverge is not None:
            diverged += 1
            first_div_plies.append(diverge)
    return {
        "games_compared": games,
        "diverged_games": diverged,
        "divergence_rate": (diverged / games if games else 0.0),
        "mean_first_divergence_ply": (
            float(np.mean(first_div_plies)) if first_div_plies else None
        ),
    }


def intervention_arena(
    modes: tuple[str, ...],
    contexts: tuple[str, ...],
    candidate: Path,
    incumbent: Path,
    workdir: Path,
    workers: int,
) -> dict[str, Any]:
    """Run the canonical paired arena for every intervention/context.

    Reuses ``run_arena`` (128 openings, seat swap, seed contract) and
    ``paired_opening_candidate_effect`` (bootstrap CI / P0 / P1). The frozen
    incumbent opponent and the matched current-control are shared across all
    interventions through provenance-cached records.
    """
    metrics: dict[str, Any] = {}
    control_cache: dict[str, list[dict[str, Any]]] = {}

    for context in contexts:
        control = control_cache.get(context)
        if control is None:
            control = _intervention_records(
                workdir, incumbent, incumbent, context, "current_control", workers, None
            )
            control_cache[context] = control

    for mode in modes:
        for context in contexts:
            override_mode = None if mode == "candidate_all" else mode
            role = mode if override_mode is not None else "candidate_all"
            candidate_records = _intervention_records(
                workdir / "intervention",
                candidate,
                incumbent,
                context,
                role,
                workers,
                override_mode,
            )
            effect = paired_opening_candidate_effect(
                candidate_records, control_cache[context]
            )
            wdl = _win_draw_loss(candidate_records)
            baseline_role = "candidate_all"
            baseline_records = (
                candidate_records
                if mode == "candidate_all"
                else _intervention_records(
                    workdir / "intervention",
                    candidate,
                    incumbent,
                    context,
                    baseline_role,
                    workers,
                    None,
                )
            )
            divergence = _trajectory_divergence(baseline_records, candidate_records)
            metrics.setdefault(mode, {}).setdefault(
                context,
                {
                    "paired_candidate_effect": effect["paired_candidate_effect"],
                    "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                    "p0_effect": effect["p0_effect"],
                    "p1_effect": effect["p1_effect"],
                    "orientation": "candidate_minus_incumbent",
                    "win_draw_loss": wdl,
                    "trajectory_divergence_from_candidate_all": divergence,
                },
            )
    return metrics


def _visit_policy(visits: list[float]) -> np.ndarray:
    v = np.asarray(visits, dtype=np.float32)
    total = float(v.sum())
    if total <= 0:
        return v
    return v / total


def _visit_js(left: np.ndarray, right: np.ndarray, legal_moves: list[int]) -> float:
    """Jensen-Shannon divergence (nats) of two root visit distributions."""
    if not legal_moves:
        return 0.0
    p = np.clip(left[legal_moves].astype(np.float64), 0.0, None)
    q = np.clip(right[legal_moves].astype(np.float64), 0.0, None)
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return 0.0
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = float(np.sum(p * np.log(np.where(p > 0, p / m, 1.0))))
        kl_qm = float(np.sum(q * np.log(np.where(q > 0, q / m, 1.0))))
    return 0.5 * (kl_pm + kl_qm)


def _rank_in_q(child_stats: list[dict], move: int | None) -> int | None:
    if move is None:
        return None
    ranked = sorted(child_stats, key=lambda c: (-float(c["q_value"]), int(c["move"])))
    for index, entry in enumerate(ranked, start=1):
        if int(entry["move"]) == int(move):
            return index
    return None


def puct_probe(
    probe_rows: list[dict[str, Any]],
    candidate_path: Path,
    incumbent_path: Path,
    context: str,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    """Frozen PR #200 PUCT probe: per-depth override telemetry + root search deltas.

    For each probe state and each intervention, runs the canonical PUCT search at
    ``context`` with the candidate evaluator and the intervention prior override.
    Reports, versus ``candidate_all``: selected-move change rate, root visit JS,
    root visit distribution, root-value change, and Q-ranking change; plus the
    override's per-depth telemetry (expanded/affected nodes, candidate-vs-incumbent
    legal-policy L1/JS).
    """
    sims = int(context.split(":")[0])
    c_puct = _context_c_puct(context)
    search_options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    candidate_ev = ArtifactEvaluator(candidate_path)
    incumbent_ev = ArtifactEvaluator(incumbent_path)
    rows = probe_rows[:PROBE_SIZE]

    baseline: list[dict[str, Any]] = []
    for row in rows:
        result = evaluate_artifact_position(
            evaluator=candidate_ev,
            state=row["state"],
            simulations=sims,
            seed=42,
            c_puct=c_puct,
            search_options=search_options,
        )
        baseline.append(result)

    out: dict[str, Any] = {}
    for mode in modes:
        override = build_prior_substitution_override(
            mode if mode != "candidate_all" else "incumbent_all", incumbent_ev
        )
        override.telemetry_log = []
        per_state: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            result = evaluate_artifact_position(
                evaluator=candidate_ev,
                state=row["state"],
                simulations=sims,
                seed=42,
                c_puct=c_puct,
                search_options=search_options,
                prior_override=override if mode != "candidate_all" else None,
            )
            base = baseline[index]
            visits = np.asarray(result["visits"], dtype=np.float32)
            legal = result["legal_moves"]
            move_changed = result["selected_move"] != base["selected_move"]
            visit_js_val = _visit_js(
                _visit_policy(result["visits"]),
                _visit_policy(base["visits"]),
                legal,
            )
            root_value_delta = float(
                result["search_root_value"] - base["search_root_value"]
            )
            q_rank_base = _rank_in_q(base["child_stats"], base["selected_move"])
            q_rank_int = _rank_in_q(result["child_stats"], result["selected_move"])
            q_rank_change = (
                float(q_rank_int - q_rank_base)
                if q_rank_base is not None and q_rank_int is not None
                else 0.0
            )
            per_state.append(
                {
                    "selected_move": result["selected_move"],
                    "baseline_selected_move": base["selected_move"],
                    "move_changed": bool(move_changed),
                    "visit_js": visit_js_val,
                    "root_value_delta": root_value_delta,
                    "q_rank_change": q_rank_change,
                    "root_visit_distribution": [float(v) for v in visits.tolist()],
                }
            )

        selected_move_change_rate = float(
            np.mean([s["move_changed"] for s in per_state])
        )
        out[mode] = {
            "selected_move_change_rate": selected_move_change_rate,
            "mean_visit_js": float(np.mean([s["visit_js"] for s in per_state])),
            "mean_root_value_delta": float(
                np.mean([s["root_value_delta"] for s in per_state])
            ),
            "mean_q_rank_change": float(
                np.mean([s["q_rank_change"] for s in per_state])
            ),
            "root_visit_distribution_mean": [
                float(v)
                for v in np.mean(
                    [
                        np.asarray(s["root_visit_distribution"], dtype=np.float32)
                        for s in per_state
                    ],
                    axis=0,
                ).tolist()
            ],
            "override_telemetry_by_depth": summarize_override_telemetry(
                override.telemetry_log
            )
            if mode != "candidate_all"
            else None,
            "states": len(per_state),
        }
    return out


def _recovery(intervention_effect: float, baseline_effect: float) -> float | None:
    if abs(baseline_effect) < 1e-12:
        return None
    return float((intervention_effect - baseline_effect) / (0.0 - baseline_effect))


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered policy-prior-localization decision rule."""
    arena = summary.get("arena") or {}
    candidate_all = arena.get("candidate_all", {}).get(PRIMARY_CONTEXT)
    if candidate_all is None:
        return {
            "label": "inconclusive",
            "next_action": NEXT_ACTIONS["inconclusive"],
            "evidence": {"arena_complete": False},
        }

    baseline_effect = float(candidate_all["paired_candidate_effect"])
    baseline_ci = candidate_all["opening_bootstrap_ci"]
    baseline_p0 = float(candidate_all["p0_effect"])

    recovery: dict[str, Any] = {}
    for mode in (
        "incumbent_root",
        "incumbent_depth1",
        "incumbent_depth2",
        "incumbent_all",
    ):
        entry = arena.get(mode, {}).get(PRIMARY_CONTEXT)
        if entry is None:
            recovery[mode] = None
            continue
        eff = float(entry["paired_candidate_effect"])
        p0 = float(entry["p0_effect"])
        recovery[mode] = {
            "paired_effect": eff,
            "paired_recovery_fraction": _recovery(eff, baseline_effect),
            "p0_effect": p0,
            "p0_recovery_fraction": _recovery(p0, baseline_p0),
            "ci_upper_95": float(entry["opening_bootstrap_ci"]["upper_95"]),
            "ci_lower_95": float(entry["opening_bootstrap_ci"]["lower_95"]),
        }

    incumbent_all_entry = arena.get("incumbent_all", {}).get(PRIMARY_CONTEXT)
    incumbent_all_search_equivalent = (
        incumbent_all_entry is not None
        and abs(float(incumbent_all_entry["paired_candidate_effect"])) < 0.02
        and float(incumbent_all_entry["opening_bootstrap_ci"]["upper_95"]) >= 0.0
        and float(incumbent_all_entry["opening_bootstrap_ci"]["lower_95"]) <= 0.0
    )

    root = recovery.get("incumbent_root")
    depth1 = recovery.get("incumbent_depth1")
    depth2 = recovery.get("incumbent_depth2")

    root_recovers = (
        root is not None
        and root["paired_recovery_fraction"] is not None
        and root["paired_recovery_fraction"] >= RECOVERY_THRESHOLD
        and root["ci_upper_95"] >= 0.0
    )
    shallow_recovers = (
        root is not None
        and (
            root["paired_recovery_fraction"] is None
            or root["paired_recovery_fraction"] < RECOVERY_THRESHOLD
        )
        and (
            (
                depth1 is not None
                and depth1["paired_recovery_fraction"] is not None
                and depth1["paired_recovery_fraction"] >= RECOVERY_THRESHOLD
            )
            or (
                depth2 is not None
                and depth2["paired_recovery_fraction"] is not None
                and depth2["paired_recovery_fraction"] >= RECOVERY_THRESHOLD
            )
        )
    )

    if not incumbent_all_search_equivalent:
        label = "unexpected_nonpolicy_difference"
    elif root_recovers:
        label = "root_prior_causal"
    elif shallow_recovers:
        label = "shallow_prior_compounding"
    elif incumbent_all_entry is not None and (
        recovery["incumbent_all"]["paired_recovery_fraction"] is not None
        and recovery["incumbent_all"]["paired_recovery_fraction"] >= RECOVERY_THRESHOLD
    ):
        label = "distributed_prior_compounding"
    else:
        label = "inconclusive"

    evidence = {
        "primary_context": PRIMARY_CONTEXT,
        "candidate_all_paired_effect": baseline_effect,
        "candidate_all_ci": baseline_ci,
        "candidate_all_p0_effect": baseline_p0,
        "recovery": recovery,
        "incumbent_all_search_equivalent": incumbent_all_search_equivalent,
        "incumbent_all_paired_effect": (
            float(incumbent_all_entry["paired_candidate_effect"])
            if incumbent_all_entry is not None
            else None
        ),
        "recovery_threshold": RECOVERY_THRESHOLD,
    }
    return {"label": label, "next_action": NEXT_ACTIONS[label], "evidence": evidence}


def markdown(summary: dict[str, Any]) -> str:
    """Render the committed results record; full detail remains in JSON."""
    classification = summary["classification"]
    label = classification["label"]
    inv = summary["invariants"]
    lines = [
        "# AlphaZero-Lite Frozen-Trunk Policy-Prior Localization Results",
        "",
        f"**Classification:** `{label}`",
        "",
        f"- candidate checkpoint state hash matches PR #200 step 46: `{inv['candidate_hash_matches_pr200']}`",
        f"- candidate trunk identical to incumbent: `{inv['trunk_identical_to_incumbent']}`",
        f"- candidate value stack identical to incumbent: `{inv['value_stack_identical_to_incumbent']}`",
        f"- candidate value outputs equal incumbent on frozen probe: `{inv['value_outputs_equal_on_probe']}`",
        f"- candidate_all reproduces PR #200 arena: `{summary['reproduction']['candidate_all_reproduces_pr200']}`",
        f"- incumbent_all search-equivalent to incumbent: `{classification['evidence']['incumbent_all_search_equivalent']}`",
        f"- candidate weights sha256: `{summary['inputs']['candidate_weights_sha256']}`",
        f"- incumbent weights sha256: `{summary['inputs']['incumbent_weights_sha256']}`",
        "",
        "## Findings",
        "",
        FINDINGS[label],
        "",
        "## Recovery fractions (384:256, paired effect / P0 effect)",
        "",
        "| Intervention | Paired effect | Paired recovery | P0 effect | P0 recovery | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    ev = classification["evidence"]
    rows = [
        (
            "candidate_all (baseline)",
            ev["candidate_all_paired_effect"],
            None,
            ev["candidate_all_p0_effect"],
            None,
            ev["candidate_all_ci"],
        ),
    ]
    for mode in (
        "incumbent_root",
        "incumbent_depth1",
        "incumbent_depth2",
        "incumbent_all",
    ):
        r = ev["recovery"].get(mode)
        if r is None:
            continue
        rows.append(
            (
                mode,
                r["paired_effect"],
                r["paired_recovery_fraction"],
                r["p0_effect"],
                r["p0_recovery_fraction"],
                {"lower_95": r["ci_lower_95"], "upper_95": r["ci_upper_95"]},
            )
        )
    for name, eff, rec, p0, p0rec, ci in rows:
        rec_s = "n/a" if rec is None else f"{rec:.3f}"
        p0rec_s = "n/a" if p0rec is None else f"{p0rec:.3f}"
        lines.append(
            f"| {name} | {eff:+.4f} | {rec_s} | {p0:+.4f} | {p0rec_s} | [{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
        )

    lines.extend(
        [
            "",
            "## Canonical arena (candidate versus frozen incumbent)",
            "",
            "| Intervention | Context | Paired effect | 95% CI | P0 | P1 | W-D-L | Div. rate |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    arena = summary.get("arena") or {}
    for mode in TRAINABLE_LANES:
        for context in ARENA_CONTEXTS:
            entry = arena.get(mode, {}).get(context)
            if entry is None:
                continue
            ci = entry["opening_bootstrap_ci"]
            wdl = entry["win_draw_loss"]
            div = entry["trajectory_divergence_from_candidate_all"]
            lines.append(
                f"| {mode} | {context} | {entry['paired_candidate_effect']:+.4f} | "
                f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] | "
                f"{entry['p0_effect']:+.4f} | {entry['p1_effect']:+.4f} | "
                f"{wdl['wins']}-{wdl['draws']}-{wdl['losses']} | {div['divergence_rate']:.4f} |"
            )

    lines.extend(
        [
            "",
            f"## PUCT probe ({PUCT_CONTEXT}, {PROBE_SIZE} states, versus candidate_all)",
            "",
            "| Intervention | Selected-move change | Mean visit JS | Mean root-value delta | Mean Q-rank change |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    probe = summary.get("puct_probe") or {}
    for mode in TRAINABLE_LANES:
        entry = probe.get(mode)
        if entry is None:
            continue
        lines.append(
            f"| {mode} | {entry['selected_move_change_rate']:.4f} | "
            f"{entry['mean_visit_js']:.6f} | {entry['mean_root_value_delta']:+.6f} | "
            f"{entry['mean_q_rank_change']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-depth override telemetry (probe, affected nodes / candidate-vs-incumbent legal-policy L1/JS)",
            "",
            "| Intervention | Depth | Expanded | Affected | Affected frac | Mean legal L1 | Mean legal JS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in TRAINABLE_LANES:
        tel = probe.get(mode, {}).get("override_telemetry_by_depth")
        if not tel:
            continue
        for depth in sorted(k for k in tel if k.isdigit()):
            b = tel[depth]
            lines.append(
                f"| {mode} | {depth} | {b['expanded_nodes']} | {b['affected_nodes']} | "
                f"{b['affected_fraction']:.4f} | {b['mean_pairwise_legal_l1']:.6f} | "
                f"{b['mean_pairwise_legal_js']:.6f} |"
            )
        lines.append(
            f"| {mode} | overall | {tel.get('total_expanded_nodes', 0)} | "
            f"{tel.get('total_affected_nodes', 0)} | {tel.get('overall_affected_fraction', 0):.4f} | - | - |"
        )

    lines.extend(
        [
            "",
            "## Classification evidence",
            "",
            "```json",
            json.dumps(classification["evidence"], indent=2, sort_keys=True),
            "```",
            "",
            "## Recommended next experiment (not implemented here)",
            "",
            f"`{classification['next_action']}`",
            "",
            "## Exact commands",
            "",
            "```bash",
            "python ml/alphazero_lite/run_frozen_trunk_policy_prior_localization.py \\",
            "  --pr191-workdir /tmp/azlite_shared_trunk_learning \\",
            "  --candidate-snapshot /tmp/azlite_frozen_trunk_distillation/policy_head/snapshots/step_0046.pt \\",
            "  --candidate-artifact /tmp/azlite_frozen_trunk_distillation/policy_head/snapshot_artifacts/step_0046/artifact \\",
            "  --incumbent-snapshot /tmp/azlite_frozen_trunk_distillation/heads_only/snapshots/step_0000.pt \\",
            "  --incumbent /home/alex/Mancala/ai/model-artifact/current \\",
            "  --workdir /tmp/azlite_frozen_trunk_policy_prior_localization --arena-workers 24",
            "```",
            "",
            "Full evidence: `docs/data/alphazero-lite-frozen-trunk-policy-prior-localization-summary.json`.",
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
        "--candidate-snapshot",
        type=Path,
        default=Path(
            "/tmp/azlite_frozen_trunk_distillation/policy_head/snapshots/step_0046.pt"
        ),
    )
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        default=Path(
            "/tmp/azlite_frozen_trunk_distillation/policy_head/snapshot_artifacts/step_0046/artifact"
        ),
    )
    parser.add_argument(
        "--incumbent-snapshot",
        type=Path,
        default=Path(
            "/tmp/azlite_frozen_trunk_distillation/heads_only/snapshots/step_0000.pt"
        ),
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=REPO_ROOT / "model-artifact/current",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_frozen_trunk_policy_prior_localization"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-frozen-trunk-policy-prior-localization-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-frozen-trunk-policy-prior-localization-results.md",
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena-workers", type=int, default=24)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(PRIOR_OVERRIDE_MODES),
        choices=list(PRIOR_OVERRIDE_MODES),
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=list(ARENA_CONTEXTS),
        choices=list(ARENA_CONTEXTS),
    )
    args = parser.parse_args()

    if sha256_file(args.incumbent / "weights.json") != CURRENT_HASH:
        raise RuntimeError("incumbent artifact does not match the PR191 initialization")

    manifest = json.loads((args.pr191_workdir / "training_manifest.json").read_text())
    rows = read_jsonl(Path(manifest["replay_path"]))
    paths = {name: Path(value) for name, value in manifest["artifact_paths"].items()}
    validation_indexes = np.load(paths["validation_source_indexes"], allow_pickle=False)
    decoded_probe, probe_manifest = decoded_validation_manifest(
        rows, validation_indexes
    )
    # ``verify_invariants`` mirrors ``probe_output_drift`` and needs the original
    # replay rows (numeric encoded states) for ``model_outputs``; ``puct_probe``
    # needs the decoded dict states for ``evaluate_artifact_position``.
    numeric_probe = [rows[index] for index in probe_manifest["source_indexes"]]

    invariants = verify_invariants(
        args.candidate_snapshot, args.incumbent_snapshot, numeric_probe
    )
    if not invariants["candidate_hash_matches_pr200"]:
        raise RuntimeError(
            f"candidate snapshot does not match PR #200 step 46: "
            f"{invariants['candidate_state_hash']} != {PR200_CANDIDATE_STATE_HASH}"
        )
    if not invariants["trunk_identical_to_incumbent"]:
        raise RuntimeError("candidate trunk is not identical to the incumbent")
    if not invariants["value_stack_identical_to_incumbent"]:
        raise RuntimeError("candidate value stack is not identical to the incumbent")
    if not invariants["value_outputs_equal_on_probe"]:
        raise RuntimeError(
            "candidate value outputs differ from incumbent on frozen probe"
        )

    arena: dict[str, Any] = {}
    modes = tuple(args.modes)
    contexts = tuple(args.contexts)
    if args.arena:
        arena = intervention_arena(
            modes,
            contexts,
            args.candidate_artifact,
            args.incumbent,
            args.workdir,
            args.arena_workers,
        )

    # candidate_all reproduction check (deterministic contract).
    reproduction: dict[str, Any] = {"candidate_all_reproduces_pr200": True}
    if "candidate_all" in arena:
        for context, expected in PR200_ARENA.items():
            got = arena.get("candidate_all", {}).get(context)
            if got is None:
                reproduction["candidate_all_reproduces_pr200"] = False
                reproduction.setdefault("mismatches", {})[context] = "missing"
                continue
            ok = (
                abs(
                    float(got["paired_candidate_effect"])
                    - expected["paired_candidate_effect"]
                )
                <= EFFECT_TOLERANCE
                and abs(float(got["p0_effect"]) - expected["p0_effect"])
                <= EFFECT_TOLERANCE
                and abs(float(got["p1_effect"]) - expected["p1_effect"])
                <= EFFECT_TOLERANCE
            )
            if not ok:
                reproduction["candidate_all_reproduces_pr200"] = False
                reproduction.setdefault("mismatches", {})[context] = {
                    "got": {
                        "paired_candidate_effect": float(
                            got["paired_candidate_effect"]
                        ),
                        "p0_effect": float(got["p0_effect"]),
                        "p1_effect": float(got["p1_effect"]),
                    },
                    "expected": expected,
                }
    if not reproduction["candidate_all_reproduces_pr200"]:
        raise RuntimeError(
            f"candidate_all does not reproduce PR #200 arena: {reproduction}"
        )

    puct_probe_result: dict[str, Any] = {}
    if args.puct:
        puct_probe_result = puct_probe(
            decoded_probe, args.candidate_artifact, args.incumbent, PUCT_CONTEXT, modes
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
            "weights_changed": False,
            "diagnostic_only": True,
        },
        "inputs": {
            "candidate_weights_sha256": sha256_file(
                args.candidate_artifact / "weights.json"
            ),
            "incumbent_weights_sha256": sha256_file(args.incumbent / "weights.json"),
            "candidate_snapshot": str(args.candidate_snapshot),
            "incumbent_snapshot": str(args.incumbent_snapshot),
            "pr200_candidate_state_hash": PR200_CANDIDATE_STATE_HASH,
            "modes": list(modes),
            "contexts": list(contexts),
            "puct_context": PUCT_CONTEXT,
            "probe_size": PROBE_SIZE,
            "evaluation": {
                "opponent": "frozen incumbent (model-artifact/current)",
                "openings": "canonical 128-opening suite, 2 games per opening per seat",
                "contexts": list(contexts),
            },
        },
        "invariants": invariants,
        "reproduction": reproduction,
        "arena": arena,
        "puct_probe": puct_probe_result,
        "probe_manifest": {
            "probe_size": PROBE_SIZE,
            "validation_source_indexes_sha256": sha256_file(
                paths["validation_source_indexes"]
            ),
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
        },
    }
    summary["classification"] = classify(summary)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
