#!/usr/bin/env python3
# ruff: noqa: E402
"""Distill frozen PR #229 shadow-search root visits into the PR #214 A16 adapter.

This is deliberately a one-off, non-promoting experiment.  It neither creates
self-play nor permits shadow-Q outside the frozen target-cache construction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.arena import ArtifactEvaluator, canonical_game_state_hash
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    write_fixed_npz,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    _suite,
    arena_records,
    state_hash,
)
from ml.alphazero_lite.run_fresh_p1_adapter_margin_sensitivity import (
    HELD_OUT_HASHES,
)
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    FROZEN,
    MANIFEST,
    PR222,
    _control_subset,
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    BETA,
    export,
    new_model,
    output,
)
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    _cross_entropy,
    mixed_policy_target,
)
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.shadow_root_q import run_shadow_root_q_search
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SIMULATIONS = 1200
SEED = 231
TARGET_WEIGHTED_RATIO = 0.50
MAX_CONFIGURED_WEIGHT = 0.25
TARGET_CACHE_SHA = "526a1e06fda3f96575c886b3d52a1e2a06a75a572c51203b0b7e61806090cecb"
SELECTOR_MANIFEST_SHA = (
    "2bf821c1debc78538961467657ff3c5757b50ccbcd37094dcf932edda5c4b4b3"
)
STEPS = (1, 4, 16)
LANES = ("baseline_continue", "shadow_sensitive", "parent_sensitive", "shadow_random25")
P1_WORKDIR = Path("/tmp/azlite_fresh_selfplay_anchor")
A16_WORKDIR = Path("/tmp/azlite_fresh_p1_parent_adapter")
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


def calibrated_weight(max_raw_ratio: float) -> float:
    """Freeze the protocol's single global auxiliary coefficient."""
    if not np.isfinite(max_raw_ratio) or max_raw_ratio <= 0.0:
        raise RuntimeError("invalid maximum raw auxiliary gradient ratio")
    return min(MAX_CONFIGURED_WEIGHT, TARGET_WEIGHTED_RATIO / max_raw_ratio)


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    """Return only the adapter gradient in the stable contract ordering."""
    values = []
    parameters = dict(model.named_parameters())
    for key in ADAPTER_KEYS:
        gradient = parameters[key].grad
        if gradient is None:
            raise RuntimeError(f"missing adapter gradient for {key}")
        values.append(gradient.detach().reshape(-1))
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise RuntimeError("non-finite adapter gradient")
    return result


def _gradient_pair(primary: torch.Tensor, auxiliary: torch.Tensor) -> dict[str, float]:
    primary_norm = float(torch.linalg.vector_norm(primary).cpu())
    auxiliary_norm = float(torch.linalg.vector_norm(auxiliary).cpu())
    if primary_norm <= 1e-12:
        raise RuntimeError("primary adapter gradient norm is too small")
    if not np.isfinite(primary_norm) or not np.isfinite(auxiliary_norm):
        raise RuntimeError("non-finite adapter gradient norm")
    return {
        "primary_norm": primary_norm,
        "auxiliary_norm": auxiliary_norm,
        "raw_ratio": auxiliary_norm / primary_norm,
        "gradient_cosine": float(torch.dot(primary, auxiliary).cpu())
        / (primary_norm * auxiliary_norm)
        if auxiliary_norm > 0.0
        else 0.0,
    }


def _seed(state_hash_value: str) -> int:
    return int(
        hashlib.sha256(f"pr231-distill:{state_hash_value}".encode()).hexdigest()[:16],
        16,
    )


def _js(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0
    return float(
        0.5
        * np.sum(left * np.log(np.maximum(left, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5
        * np.sum(right * np.log(np.maximum(right, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.percentile(values, percentile))
        for key, percentile in (("mean", 50), ("p50", 50), ("p90", 90), ("p99", 99))
    } | {"mean": float(values.mean())}


def _ordinary(
    game: KalahGame, evaluator: ArtifactEvaluator, seed: int
) -> tuple[np.ndarray, dict]:
    search = PUCT(
        evaluator,
        SIMULATIONS,
        1.25,
        random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, _ = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return visits.astype(np.float64), search.root_summary()


def _init_worker(p1: str, a16: str) -> None:
    global _P1, _A16
    _P1, _A16 = ArtifactEvaluator(Path(p1)), ArtifactEvaluator(Path(a16))


def _cache_record(item: dict[str, Any]) -> dict[str, Any]:
    if _P1 is None or _A16 is None:
        raise RuntimeError("target-cache worker not initialized")
    game = KalahGame.from_state(item["state"])
    seed = _seed(item["state_hash"])
    ordinary_a16, a16_summary = _ordinary(game, _A16, seed)
    ordinary_p1, p1_summary = _ordinary(game, _P1, seed)
    shadow, _, shadow_meta = run_shadow_root_q_search(
        game,
        main_evaluator=_A16,
        shadow_evaluator=_P1,
        simulations=SIMULATIONS,
        c_puct=1.25,
        seed=seed,
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        shadow_q_weight=1.0,
    )
    return {
        "ordinary_a16": ordinary_a16,
        "ordinary_p1": ordinary_p1,
        "shadow_a16": shadow.astype(np.float64),
        "a16_move": a16_summary["selected_move"],
        "p1_move": p1_summary["selected_move"],
        "shadow_move": shadow_meta["main_summary"]["selected_move"],
        "a16_root_q": a16_summary["root_q_value"],
        "p1_root_q": p1_summary["root_q_value"],
        "shadow_root_q": shadow_meta["main_summary"]["root_q_value"],
    }


def _frozen_hashes() -> tuple[set[str], set[str]]:
    frozen = json.loads(FROZEN.read_text())
    prior = json.loads(PR222.read_text())
    primary = set(frozen["full_amplified_1200"])
    by_hash = {row["state_hash"]: row for row in prior["records"]}
    controls = _control_subset(
        prior["records"], [by_hash[key] for key in sorted(primary)]
    )
    return primary, {row["state_hash"] for row in controls}


def _population(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest = json.loads(MANIFEST.read_text())
    primary, controls = _frozen_hashes()
    _, suite_hash = _suite()
    # The canonical suite contributes opening roots; its complete trajectories are
    # never reconstructed or used for this training population.
    canonical = {
        canonical_game_state_hash(
            KalahGame.from_state(
                {
                    "player_pits": [4] * 6,
                    "opponent_pits": [4] * 6,
                    "player_store": 0,
                    "opponent_store": 0,
                    "current_player": 0,
                }
            )
        )
    }
    selected, rejected = (
        [],
        {
            "frozen_amplified": 0,
            "washed_controls": 0,
            "held_out": 0,
            "canonical_opening": 0,
            "duplicate": 0,
            "not_fresh_replay": 0,
        },
    )
    seen: set[str] = set()
    for meta in manifest["rows"]:
        index = int(meta["replay_index"])
        if index >= len(rows):
            rejected["not_fresh_replay"] += 1
            continue
        state = decode_kalah_v3_base_state(list(rows[index]["state"]))
        state_hash_value = canonical_game_state_hash(KalahGame.from_state(state))
        if state_hash_value != meta["state_hash"]:
            raise RuntimeError("manifest state does not match fresh-P1 replay")
        reason = (
            "frozen_amplified"
            if state_hash_value in primary
            else "washed_controls"
            if state_hash_value in controls
            else "held_out"
            if state_hash_value in HELD_OUT_HASHES
            else "canonical_opening"
            if state_hash_value in canonical
            else "duplicate"
            if state_hash_value in seen
            else None
        )
        if reason:
            rejected[reason] += 1
            continue
        seen.add(state_hash_value)
        selected.append(
            {"state_hash": state_hash_value, "replay_index": index, "state": state}
        )
    if not selected:
        raise RuntimeError("no eligible fresh-P1 roots")
    rejected["manifest_roots"] = len(manifest["rows"])
    rejected["eligible_roots"] = len(selected)
    rejected["canonical_suite_sha256"] = suite_hash  # type: ignore[assignment]
    return selected, rejected


def _save_cache(
    workdir: Path, population: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], str]:
    arrays = {
        key: np.asarray([record[key] for record in records])
        for key in ("ordinary_a16", "ordinary_p1", "shadow_a16")
    }
    arrays |= {
        key: np.asarray([record[key] for record in records], dtype=np.float64)
        for key in ("a16_root_q", "p1_root_q", "shadow_root_q")
    }
    arrays |= {
        key: np.asarray([record[key] for record in records], dtype=np.int64)
        for key in ("a16_move", "p1_move", "shadow_move")
    }
    path = workdir / "target_cache.npz"
    write_fixed_npz(path, arrays)
    return arrays, sha256_file(path)


def _selector(
    workdir: Path, population: list[dict[str, Any]], cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    a16, p1, shadow = (
        cache[key] / cache[key].sum(axis=1, keepdims=True)
        for key in ("ordinary_a16", "ordinary_p1", "shadow_a16")
    )
    sensitivity = np.asarray([_js(a16[i], shadow[i]) for i in range(len(population))])
    count = len(population) // 4
    order = sorted(
        range(len(population)),
        key=lambda i: (-sensitivity[i], population[i]["state_hash"]),
    )
    sensitive = np.asarray(order[:count], dtype=np.int64)
    random_order = sorted(
        range(len(population)),
        key=lambda i: hashlib.sha256(
            f"{SEED}:{population[i]['state_hash']}".encode()
        ).hexdigest(),
    )
    random_set = np.asarray(random_order[:count], dtype=np.int64)
    manifest = {
        "schema": "azlite_shadow_target_selector_v1",
        "seed": SEED,
        "population_count": len(population),
        "sensitive_indexes": sensitive.tolist(),
        "random_indexes": random_set.tolist(),
        "sensitive_hashes": [population[i]["state_hash"] for i in sensitive],
        "random_hashes": [population[i]["state_hash"] for i in random_set],
    }
    manifest["sha256_excluding_this_field"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = workdir / "selector_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    diagnostics = {
        "threshold": float(sensitivity[order[count - 1]]),
        "state_count": count,
        "shadow_js": _summary(sensitivity),
        "ordinary_a16_p1_move_disagreement": float(
            np.mean(cache["a16_move"] != cache["p1_move"])
        ),
        "shadow_rescue_rate": float(np.mean(cache["shadow_move"] == cache["p1_move"])),
        "shadow_change_rate": float(np.mean(cache["shadow_move"] != cache["a16_move"])),
        "policy_l1_a16_p1": _summary(np.abs(a16 - p1).sum(axis=1)),
        "selector_sha256": sha256_file(path),
    }
    return sensitive, random_set, diagnostics


def _model_metrics(
    state: dict[str, torch.Tensor],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    anchor_indexes: np.ndarray,
    anchor_replay_indexes: np.ndarray,
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    candidate, parent = output(state, x, mask), output(parent_state, x, mask)
    search = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    pure_state = None
    baseline = float(np.mean(_cross_entropy(parent, search)))
    current = float(np.mean(_cross_entropy(candidate, search)))
    # PR #214's denominator is the fixed pure-search step-46 endpoint.
    pure_path = A16_WORKDIR / "pure_search/snapshots/step_0046.pt"
    if pure_path.is_file():
        pure_state = torch.load(pure_path, map_location="cpu", weights_only=False)[
            "model"
        ]
        pure = float(np.mean(_cross_entropy(output(pure_state, x, mask), search)))
    else:
        raise RuntimeError("missing immutable PR #214 pure-search endpoint")
    l1 = np.abs(candidate - parent).sum(axis=1)
    adapter_delta = torch.cat(
        [(state[key] - parent_state[key]).reshape(-1) for key in ADAPTER_KEYS]
    )
    anchor_candidate = candidate[anchor_replay_indexes]
    shadow_target = cache["shadow_a16"][anchor_indexes] / SIMULATIONS
    parent_target = cache["ordinary_p1"][anchor_indexes] / SIMULATIONS
    ordinary_a16 = cache["ordinary_a16"][anchor_indexes] / SIMULATIONS
    return {
        "ce_search": current,
        "ce_p1_policy": float(np.mean(_cross_entropy(candidate, parent))),
        "ce_beta095": float(
            np.mean(_cross_entropy(candidate, (1 - BETA) * search + BETA * parent))
        ),
        "fit_fraction": float((baseline - current) / (baseline - pure)),
        "legal_policy_l1_vs_p1": float(l1.mean()),
        "js_vs_p1": float(
            np.mean([_js(candidate[i], parent[i]) for i in range(len(candidate))])
        ),
        "top1_disagreement": float(
            np.mean(np.argmax(candidate, axis=1) != np.argmax(parent, axis=1))
        ),
        "adapter_parameter_norm": float(
            torch.linalg.vector_norm(
                torch.cat([state[key].reshape(-1) for key in ADAPTER_KEYS])
            )
        ),
        "movement_from_a16": float(torch.linalg.vector_norm(adapter_delta)),
        "anchor": {
            "ce_shadow_target": float(
                np.mean(_cross_entropy(anchor_candidate, shadow_target))
            ),
            "ce_parent_search_target": float(
                np.mean(_cross_entropy(anchor_candidate, parent_target))
            ),
            "ordinary_search_visit_js_vs_p1": float(
                np.mean(
                    [
                        _js(ordinary_a16[i], parent_target[i])
                        for i in range(len(anchor_indexes))
                    ]
                )
            ),
            "ordinary_root_move_disagreement_vs_p1": float(
                np.mean(
                    cache["a16_move"][anchor_indexes]
                    != cache["p1_move"][anchor_indexes]
                )
            ),
        },
    }


def _losses_for_step(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    source: np.ndarray,
    indexes: np.ndarray,
    anchors: np.ndarray | None,
    target: np.ndarray | None,
    anchor_order: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [rows[int(source[i])] for i in indexes if i >= 0]
    x = torch.tensor(
        np.asarray([row["state"] for row in selected], dtype=np.float32), device=device
    )
    p = torch.tensor(
        np.asarray([row["policy"] for row in selected], dtype=np.float32), device=device
    )
    v = torch.tensor(
        np.asarray([row["value"] for row in selected], dtype=np.float32).reshape(-1, 1),
        device=device,
    )
    mask = torch.tensor(
        legal_mask_matrix_for_encoded_states(x.detach().cpu().numpy()), device=device
    )
    parent = new_model(device)
    parent.load_state_dict(parent_state)
    parent.eval()
    with torch.no_grad():
        parent_logits, _ = parent(x)
        parent_policy = torch.softmax(parent_logits.masked_fill(mask <= 0, -1e9), dim=1)
    policy, value = model(x)
    primary = (
        compute_policy_cross_entropy(
            policy.masked_fill(mask <= 0, -1e9),
            mixed_policy_target(p, parent_policy, mask, BETA),
        ).mean()
        + 0.6
        * compute_value_loss_vector(
            value, v, value_loss="huber", huber_delta=1.0
        ).mean()
    )
    auxiliary = torch.zeros((), device=device)
    if target is not None:
        assert anchors is not None
        position = ((step - 1) * len(indexes) + np.arange(len(indexes))) % len(
            anchor_order
        )
        a = anchor_order[position]
        ax = torch.tensor(
            np.asarray([rows[int(anchors[i])]["state"] for i in a], dtype=np.float32),
            device=device,
        )
        am = torch.tensor(
            legal_mask_matrix_for_encoded_states(ax.detach().cpu().numpy()),
            device=device,
        )
        at = torch.tensor(target[a], dtype=torch.float32, device=device)
        logits, _ = model(ax)
        auxiliary = compute_policy_cross_entropy(
            logits.masked_fill(am <= 0, -1e9), at
        ).mean()
    return primary, auxiliary


def _anchor_order(name: str, anchors: np.ndarray | None) -> np.ndarray:
    if anchors is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray(
        sorted(
            range(len(anchors)),
            key=lambda i: hashlib.sha256(
                f"{SEED}:{name}:{anchors[i]}".encode()
            ).hexdigest(),
        ),
        dtype=np.int64,
    )


def _measure_gradient(
    name: str,
    snapshot: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    source: np.ndarray,
    indexes: np.ndarray,
    anchors: np.ndarray,
    target: np.ndarray,
    step: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model = new_model(device)
    model.load_state_dict(snapshot["model"])
    apply_trainable_scope(model, "policy_adapter_only")
    primary, auxiliary = _losses_for_step(
        model,
        parent_state,
        rows,
        source,
        indexes,
        anchors,
        target,
        _anchor_order(name, anchors),
        step,
        device,
    )
    model.zero_grad(set_to_none=True)
    primary.backward(retain_graph=True)
    primary_gradient = _gradient_vector(model).clone()
    model.zero_grad(set_to_none=True)
    auxiliary.backward()
    auxiliary_gradient = _gradient_vector(model).clone()
    return (
        _gradient_pair(primary_gradient, auxiliary_gradient),
        primary_gradient,
        auxiliary_gradient,
    )


def _train_lane(
    name: str,
    snapshot: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    source: np.ndarray,
    plan: np.ndarray,
    anchors: np.ndarray | None,
    target: np.ndarray | None,
    device: torch.device,
    weight: float,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, float]]]:
    model = new_model(device)
    model.load_state_dict(snapshot["model"])
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.0
    )
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    anchor_order = _anchor_order(name, anchors)
    captures: dict[int, dict[str, torch.Tensor]] = {}
    telemetry: list[dict[str, float]] = []
    for step, indexes in enumerate(plan[16:32], 1):
        primary, auxiliary = _losses_for_step(
            model,
            parent_state,
            rows,
            source,
            indexes,
            anchors,
            target,
            anchor_order,
            step,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        primary.backward(retain_graph=True)
        primary_gradient = _gradient_vector(model).clone()
        if target is None:
            detail = {
                "primary_norm": float(torch.linalg.vector_norm(primary_gradient)),
                "auxiliary_norm": 0.0,
                "raw_ratio": 0.0,
                "gradient_cosine": 0.0,
            }
        else:
            optimizer.zero_grad(set_to_none=True)
            auxiliary.backward(retain_graph=True)
            detail = _gradient_pair(primary_gradient, _gradient_vector(model))
        detail["step"] = float(step)
        detail["weighted_ratio"] = weight * detail["raw_ratio"]
        if detail["weighted_ratio"] > 2.0:
            raise RuntimeError("runtime auxiliary gradient scale failure")
        telemetry.append(detail)
        optimizer.zero_grad(set_to_none=True)
        (primary + weight * auxiliary).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in STEPS:
            captures[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    return captures, telemetry


def _frozen_diagnostic(
    candidate: Path, parent: Path, rows: list[dict[str, Any]]
) -> dict[str, float]:
    """Use ordinary PUCT only on the preregistered 40 amplified and 40 control roots."""
    amplified, controls = _frozen_hashes()
    manifest = {
        row["state_hash"]: row for row in json.loads(MANIFEST.read_text())["rows"]
    }
    candidate_eval, parent_eval = (
        ArtifactEvaluator(candidate),
        ArtifactEvaluator(parent),
    )

    def move(state_hash_value: str, evaluator: ArtifactEvaluator) -> int:
        meta = manifest[state_hash_value]
        state = decode_kalah_v3_base_state(
            list(rows[int(meta["replay_index"])]["state"])
        )
        _, summary = _ordinary(
            KalahGame.from_state(state), evaluator, _seed(state_hash_value)
        )
        return int(summary["selected_move"])

    rescue = [
        move(key, candidate_eval) == move(key, parent_eval) for key in sorted(amplified)
    ]
    divergence = [
        move(key, candidate_eval) != move(key, parent_eval) for key in sorted(controls)
    ]
    return {
        "rescue_rate": float(np.mean(rescue)),
        "new_divergence_rate": float(np.mean(divergence)),
    }


def _arena(
    candidate: Path,
    opponent: Path,
    context: str,
    workdir: Path,
    role: str,
    workers: int,
    suite_hash: str,
) -> dict[str, Any]:
    control = arena_records(
        workdir=workdir,
        challenger=opponent,
        current=opponent,
        context=context,
        role=f"{role}_p1_control",
        workers=workers,
        suite_hash=suite_hash,
    )
    treatment = arena_records(
        workdir=workdir,
        challenger=candidate,
        current=opponent,
        context=context,
        role=role,
        workers=workers,
        suite_hash=suite_hash,
    )
    effect = paired_opening_candidate_effect(
        treatment, control, bootstrap_samples=10_000, bootstrap_seed=42
    )
    ci = effect["opening_bootstrap_ci"]
    return {
        "paired_candidate_effect": effect["paired_candidate_effect"],
        "opening_bootstrap_ci": ci,
        "seat_a_effect": effect["p0_effect"],
        "seat_b_effect": effect["p1_effect"],
        "win_draw_loss": {
            "wins": sum(row["winner"] == "challenger" for row in treatment),
            "draws": sum(row["winner"] == "draw" for row in treatment),
            "losses": sum(row["winner"] == "current" for row in treatment),
        },
        "safe": ci["upper_95"] >= 0.0 or ci["lower_95"] >= -0.03,
    }


def _report(summary: dict[str, Any]) -> str:
    calibration_rows = [
        "| Lane | Batch | Primary norm | Auxiliary norm | Raw ratio | Cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, values in summary["calibration"]["lanes"].items():
        for row in values:
            calibration_rows.append(
                f"| {lane} | {int(row['batch'])} | {row['primary_norm']:.9g} | {row['auxiliary_norm']:.9g} | {row['raw_ratio']:.6f} | {row['gradient_cosine']:.6f} |"
            )
    metric_rows = [
        "| Lane | Step | CE(search) | CE(P1) | CE(beta095) | Fit | L1 | JS | Top-1 | Adapter norm | Movement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, details in summary["lanes"].items():
        for step, metric in details["metrics"].items():
            metric_rows.append(
                f"| {lane} | {step} | {metric['ce_search']:.6f} | {metric['ce_p1_policy']:.6f} | {metric['ce_beta095']:.6f} | {metric['fit_fraction']:.4f} | {metric['legal_policy_l1_vs_p1']:.6f} | {metric['js_vs_p1']:.2e} | {metric['top1_disagreement']:.4f} | {metric['adapter_parameter_norm']:.6f} | {metric['movement_from_a16']:.6f} |"
            )
    return "\n".join(
        [
            "# Shadow-Target Policy Distillation",
            "",
            f"**Classification:** `{summary['classification']}`",
            "",
            f"**Recommended follow-up:** {summary['recommended_follow_up']}",
            "",
            "## Calibration",
            "",
            *calibration_rows,
            "",
            "```json",
            json.dumps(summary["calibration"]["decision"], indent=2, sort_keys=True),
            "```",
            "",
            "## Gradient Geometry",
            "",
            summary["calibration"]["geometry_interpretation"],
            "",
            "```json",
            json.dumps(
                {
                    "per_lane_cosine": summary["calibration"]["cosine_summary"],
                    "shadow_sensitive_vs_parent_sensitive": summary["calibration"][
                        "shadow_sensitive_vs_parent_sensitive_cosine"
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Checkpoint Metrics",
            "",
            *metric_rows,
            "",
            "Anchor CE and cached ordinary-search anchor metrics are in the JSON summary for every checkpoint.",
            "",
            "## Evaluation",
            "",
            "```json",
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "frozen_diagnostics",
                        "arena_matrix",
                        "p0_gate",
                        "invariants",
                    )
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_shadow_target_distillation")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-arenas", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    replay = A16_WORKDIR / "fresh_p1_self_play.jsonl"
    p1_checkpoint = P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    p1_artifact = P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_artifact = A16_WORKDIR / "artifacts/step_0016/artifact"
    snapshot_path = A16_WORKDIR / "beta095/snapshots/step_0016.pt"
    hashes = {
        "p1_checkpoint": sha256_file(p1_checkpoint),
        "a16_artifact": sha256_file(a16_artifact / "weights.json"),
        "replay": sha256_file(replay),
        "a16_snapshot_state": None,
    }
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    hashes["a16_snapshot_state"] = state_hash(snapshot["model"])
    if (
        hashes["p1_checkpoint"] != P1_CHECKPOINT_SHA
        or hashes["replay"] != REPLAY_SHA
        or hashes["a16_snapshot_state"] != A16_STATE_SHA
    ):
        raise RuntimeError("immutable PR #214 starting artifact mismatch")
    parent_model = new_model(torch.device("cpu"))
    load_checkpoint_into_model(parent_model, p1_checkpoint)
    parent_state = {
        key: value.detach().clone() for key, value in parent_model.state_dict().items()
    }
    rows = read_jsonl(replay)
    population, exclusions = _population(rows)
    cache_path = args.workdir / "target_cache.npz"
    if not cache_path.is_file():
        raise RuntimeError(
            "missing immutable PR #232 target cache; regeneration is prohibited"
        )
    loaded = np.load(cache_path, allow_pickle=False)
    cache = {key: loaded[key] for key in loaded.files}
    cache_hash = sha256_file(cache_path)
    selector_path = args.workdir / "selector_manifest.json"
    selector_hash = sha256_file(selector_path) if selector_path.is_file() else None
    if (
        cache_hash != TARGET_CACHE_SHA
        or selector_hash != SELECTOR_MANIFEST_SHA
        or cache["ordinary_a16"].shape[0] != len(population)
    ):
        raise RuntimeError("immutable PR #232 target cache or selector mismatch")
    selector_manifest = json.loads(selector_path.read_text())
    sensitive = np.asarray(selector_manifest["sensitive_indexes"], dtype=np.int64)
    random_set = np.asarray(selector_manifest["random_indexes"], dtype=np.int64)
    if len(sensitive) != 1003 or len(random_set) != 1003:
        raise RuntimeError("immutable selector count mismatch")
    selector = {
        "selector_sha256": selector_hash,
        "population_count": len(population),
        "sensitive_count": len(sensitive),
        "random_count": len(random_set),
    }
    target_identity = {
        "l1": _summary(
            np.abs(
                cache["shadow_a16"][sensitive] - cache["ordinary_p1"][sensitive]
            ).sum(axis=1)
            / SIMULATIONS
        ),
        "mean_js": float(
            np.mean(
                [
                    _js(
                        cache["shadow_a16"][i] / SIMULATIONS,
                        cache["ordinary_p1"][i] / SIMULATIONS,
                    )
                    for i in sensitive
                ]
            )
        ),
        "top1_disagreement": float(
            np.mean(
                np.argmax(cache["shadow_a16"][sensitive], axis=1)
                != np.argmax(cache["ordinary_p1"][sensitive], axis=1)
            )
        ),
        "visit_winner_disagreement": float(
            np.mean(cache["shadow_move"][sensitive] != cache["p1_move"][sensitive])
        ),
    }
    source, plan = (
        np.load(A16_WORKDIR / "train_source_indexes.npy"),
        np.load(A16_WORKDIR / "batch_indexes.npy"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    sensitive_replay = np.asarray(
        [population[i]["replay_index"] for i in sensitive], dtype=np.int64
    )
    random_replay = np.asarray(
        [population[i]["replay_index"] for i in random_set], dtype=np.int64
    )
    targets = {
        "baseline_continue": (None, None),
        "shadow_sensitive": (
            sensitive_replay,
            cache["shadow_a16"][sensitive] / SIMULATIONS,
        ),
        "parent_sensitive": (
            sensitive_replay,
            cache["ordinary_p1"][sensitive] / SIMULATIONS,
        ),
        "shadow_random25": (
            random_replay,
            cache["shadow_a16"][random_set] / SIMULATIONS,
        ),
    }
    calibration_lanes = {
        name: [] for name in ("shadow_sensitive", "parent_sensitive", "shadow_random25")
    }
    geometry_pairs = []
    for batch in range(1, 5):
        measured = {}
        for lane in calibration_lanes:
            anchors, target = targets[lane]
            assert anchors is not None and target is not None
            metric, primary_gradient, auxiliary_gradient = _measure_gradient(
                lane,
                snapshot,
                parent_state,
                rows,
                source,
                plan[16 + batch - 1],
                anchors,
                target,
                batch,
                device,
            )
            calibration_lanes[lane].append({"batch": batch, **metric})
            measured[lane] = (primary_gradient, auxiliary_gradient)
        shadow, parent = (
            measured["shadow_sensitive"][1],
            measured["parent_sensitive"][1],
        )
        geometry_pairs.append(
            float(torch.dot(shadow, parent).cpu())
            / float(
                torch.linalg.vector_norm(shadow).cpu()
                * torch.linalg.vector_norm(parent).cpu()
            )
        )
    max_raw_ratio = max(
        row["raw_ratio"] for values in calibration_lanes.values() for row in values
    )
    weight = calibrated_weight(max_raw_ratio)
    for values in calibration_lanes.values():
        for row in values:
            row["weighted_ratio"] = weight * row["raw_ratio"]
            if row["weighted_ratio"] > 0.500001:
                raise RuntimeError("calibration weighted-ratio invariant failure")
    cosine_summary = {
        lane: {
            "median": float(np.median([row["gradient_cosine"] for row in values])),
            "min": float(np.min([row["gradient_cosine"] for row in values])),
            "max": float(np.max([row["gradient_cosine"] for row in values])),
        }
        for lane, values in calibration_lanes.items()
    }
    delta = (
        cosine_summary["shadow_sensitive"]["median"]
        - cosine_summary["parent_sensitive"]["median"]
    )
    relation = (
        "more aligned"
        if delta > 0.01
        else "less aligned"
        if delta < -0.01
        else "effectively the same"
    )
    calibration = {
        "batches": [1, 2, 3, 4],
        "lanes": calibration_lanes,
        "cosine_summary": cosine_summary,
        "shadow_sensitive_vs_parent_sensitive_cosine": geometry_pairs,
        "geometry_interpretation": f"Shadow-sensitive auxiliary gradients are {relation} with the primary objective than the parent anchor by median cosine ({delta:+.6f}); shadow-vs-parent auxiliary cosine is {float(np.median(geometry_pairs)):.6f} median.",
        "decision": {
            "target_weighted_ratio": TARGET_WEIGHTED_RATIO,
            "max_configured_weight": MAX_CONFIGURED_WEIGHT,
            "max_raw_ratio": max_raw_ratio,
            "behavior_loss_weight": weight,
            "acceptance_max_weighted_ratio": max(
                row["weighted_ratio"]
                for values in calibration_lanes.values()
                for row in values
            ),
        },
    }
    lanes: dict[str, Any] = {}
    inherited = {
        key: value
        for key, value in snapshot["model"].items()
        if key not in ADAPTER_KEYS
    }
    for lane in LANES:
        anchor_indexes, anchor_target = targets[lane]
        states, telemetry = _train_lane(
            lane,
            snapshot,
            parent_state,
            rows,
            source,
            plan,
            anchor_indexes,
            anchor_target,
            device,
            weight,
        )
        lane_metrics = {}
        artifacts = {}
        for step, state in states.items():
            unchanged = all(
                torch.equal(state[key], value) for key, value in inherited.items()
            )
            if not unchanged:
                raise RuntimeError("inherited parameter drift")
            lane_metrics[str(step)] = _model_metrics(
                state, parent_state, rows, sensitive, sensitive_replay, cache
            ) | {"inherited_parameters_byte_identical": unchanged}
            artifacts[str(step)] = str(
                export(
                    state,
                    args.workdir / "artifacts" / lane / f"step_{16 + step:04d}",
                    f"{lane}_{16 + step}",
                )
            )
        lanes[lane] = {
            "gradient_telemetry": telemetry,
            "metrics": lane_metrics,
            "artifacts": artifacts,
        }
    suite, suite_hash = _suite()
    frozen_diagnostics: dict[str, Any] = {}
    arena_matrix: dict[str, Any] = {}
    p0_gate: dict[str, Any] = {}
    p0_artifact = REPO_ROOT / "model-artifact/current"
    if not args.skip_arenas:
        for lane, details in lanes.items():
            for step, metric in details["metrics"].items():
                if metric["fit_fraction"] < 0.25:
                    continue
                artifact = Path(details["artifacts"][step])
                key = f"{lane}:{step}"
                frozen_diagnostics[key] = _frozen_diagnostic(
                    artifact, p1_artifact, rows
                )
                contexts = {
                    context: _arena(
                        artifact,
                        p1_artifact,
                        context,
                        args.workdir / "evaluation" / key,
                        "candidate_vs_p1",
                        args.workers,
                        suite_hash,
                    )
                    for context in ("384:256", "1200:1200")
                }
                arena_matrix[key] = contexts
                if all(value["safe"] for value in contexts.values()):
                    p0_gate[key] = {
                        context: _arena(
                            artifact,
                            p0_artifact,
                            context,
                            args.workdir / "evaluation" / key,
                            "candidate_vs_p0",
                            args.workers,
                            suite_hash,
                        )
                        for context in ("384:256", "1200:1200")
                    }
    invariants = {
        "artifact_hashes": True,
        "target_cache_hash": cache_hash == TARGET_CACHE_SHA,
        "selector_manifest_hash": selector_hash == SELECTOR_MANIFEST_SHA,
        "inherited_parameters_byte_identical": all(
            metric["inherited_parameters_byte_identical"]
            for lane in lanes.values()
            for metric in lane["metrics"].values()
        ),
        "calibration_acceptance": calibration["decision"][
            "acceptance_max_weighted_ratio"
        ]
        <= 0.500001,
    }
    meaningful = [
        key
        for key, details in lanes.items()
        for step, metric in details["metrics"].items()
        if metric["fit_fraction"] >= 0.25
    ]
    safe = [
        key
        for key, contexts in arena_matrix.items()
        if all(context["safe"] for context in contexts.values())
    ]
    runtime_failure = any(
        row["weighted_ratio"] > 2.0
        for lane in lanes.values()
        for row in lane["gradient_telemetry"]
    )
    classification = (
        "invariant_failure"
        if not all(invariants.values())
        else "runtime_gradient_scale_failure"
        if runtime_failure
        else "calibrated_distillation_still_unsafe"
        if meaningful and arena_matrix and not safe
        else "inconclusive"
    )
    follow_up = {
        "invariant_failure": "Repair the immutable artifact or parameter invariant before interpreting this experiment.",
        "runtime_gradient_scale_failure": "Use a constrained or proximal update rather than a static loss coefficient.",
        "calibrated_distillation_still_unsafe": "Test online or iteratively refreshed search-target generation rather than a static cache.",
        "inconclusive": "Retain this frozen calibrated protocol; no training or selector change is justified by these results.",
    }[classification]
    summary = {
        "schema": "azlite_shadow_target_distillation_v1",
        "hashes": hashes
        | {
            "target_cache": cache_hash,
            "selector_manifest": selector_hash,
            "canonical_suite": suite_hash,
            "p0_artifact": sha256_file(p0_artifact / "weights.json"),
        },
        "exclusions": exclusions,
        "selector": selector,
        "target_identity": target_identity,
        "guardrails": {
            "self_play_generated": False,
            "behavior_loss_weight": weight,
            "beta": BETA,
            "trainable": list(ADAPTER_KEYS),
            "search": {
                "simulations": SIMULATIONS,
                "c_puct": 1.25,
                "fpu_mode": "zero",
                "normalize_values": False,
                "root_noise": False,
                "shadow_used_only_for_targets": True,
            },
        },
        "lanes": lanes,
        "calibration": calibration,
        "invariants": invariants,
        "frozen_diagnostics": frozen_diagnostics,
        "arena_matrix": arena_matrix,
        "p0_gate": p0_gate,
        "classification": classification,
        "recommended_follow_up": follow_up,
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (
        REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-shadow-target-distillation-summary.json"
    ).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (
        REPO_ROOT / "docs/alphazero-lite-fresh-p1-shadow-target-distillation-results.md"
    ).write_text(_report(summary))
    print(classification)


if __name__ == "__main__":
    main()
