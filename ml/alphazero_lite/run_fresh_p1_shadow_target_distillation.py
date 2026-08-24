#!/usr/bin/env python3
# ruff: noqa: E402
"""Distill frozen PR #229 shadow-search root visits into the PR #214 A16 adapter.

This is deliberately a one-off, non-promoting experiment.  It neither creates
self-play nor permits shadow-Q outside the frozen target-cache construction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
    read_jsonl,
    sha256_file,
    write_fixed_npz,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    state_hash,
    _suite,
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
WEIGHT = 0.25
STEPS = (1, 4, 16)
LANES = ("baseline_continue", "shadow_sensitive", "parent_sensitive", "shadow_random25")
P1_WORKDIR = Path("/tmp/azlite_fresh_selfplay_anchor")
A16_WORKDIR = Path("/tmp/azlite_fresh_p1_parent_adapter")
_P1: ArtifactEvaluator | None = None
_A16: ArtifactEvaluator | None = None


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
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, float]]:
    model = new_model(device)
    model.load_state_dict(snapshot["model"])
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.0
    )
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    anchor_order = np.asarray(
        []
        if anchors is None
        else sorted(
            range(len(anchors)),
            key=lambda i: hashlib.sha256(
                f"{SEED}:{name}:{anchors[i]}".encode()
            ).hexdigest(),
        ),
        dtype=np.int64,
    )
    captures: dict[int, dict[str, torch.Tensor]] = {}
    gradient: dict[str, float] = {}
    for step, indexes in enumerate(plan[16:32], 1):
        selected = [rows[int(source[i])] for i in indexes if i >= 0]
        x = torch.tensor(
            np.asarray([row["state"] for row in selected], dtype=np.float32),
            device=device,
        )
        p = torch.tensor(
            np.asarray([row["policy"] for row in selected], dtype=np.float32),
            device=device,
        )
        v = torch.tensor(
            np.asarray([row["value"] for row in selected], dtype=np.float32).reshape(
                -1, 1
            ),
            device=device,
        )
        mask = torch.tensor(
            legal_mask_matrix_for_encoded_states(x.detach().cpu().numpy()),
            device=device,
        )
        parent = new_model(device)
        parent.load_state_dict(parent_state)
        parent.eval()
        with torch.no_grad():
            parent_logits, _ = parent(x)
            parent_policy = torch.softmax(
                parent_logits.masked_fill(mask <= 0, -1e9), dim=1
            )
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
                np.asarray(
                    [rows[int(anchors[i])]["state"] for i in a], dtype=np.float32
                ),
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
        if step == 1:
            optimizer.zero_grad(set_to_none=True)
            primary.backward(retain_graph=True)
            primary_norm = float(
                torch.sqrt(
                    sum(
                        torch.sum(p.grad**2)
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )
            )
            if target is None:
                auxiliary_norm = 0.0
            else:
                optimizer.zero_grad(set_to_none=True)
                auxiliary.backward(retain_graph=True)
                auxiliary_norm = float(
                    torch.sqrt(
                        sum(
                            torch.sum(p.grad**2)
                            for p in model.parameters()
                            if p.grad is not None
                        )
                    )
                )
            gradient = {
                "primary": primary_norm,
                "auxiliary": auxiliary_norm,
                "auxiliary_to_primary": auxiliary_norm / max(primary_norm, 1e-20),
            }
            if target is not None and gradient["auxiliary_to_primary"] > 10.0:
                raise RuntimeError(
                    "invalid auxiliary gradient scale: "
                    f"primary={primary_norm:.9g} auxiliary={auxiliary_norm:.9g} "
                    f"ratio={gradient['auxiliary_to_primary']:.6f}x"
                )
        optimizer.zero_grad(set_to_none=True)
        (primary + WEIGHT * auxiliary).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        if step in STEPS:
            captures[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    return captures, gradient


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
    if cache_path.is_file():
        loaded = np.load(cache_path, allow_pickle=False)
        cache = {key: loaded[key] for key in loaded.files}
        if cache["ordinary_a16"].shape[0] != len(population):
            raise RuntimeError("target cache population size mismatch")
        cache_hash = sha256_file(cache_path)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(p1_artifact), str(a16_artifact)),
        ) as pool:
            records = list(pool.map(_cache_record, population, chunksize=1))
        cache, cache_hash = _save_cache(args.workdir, population, records)
    sensitive, random_set, selector = _selector(args.workdir, population, cache)
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
    lanes: dict[str, Any] = {}
    inherited = {
        key: value
        for key, value in snapshot["model"].items()
        if key not in ADAPTER_KEYS
    }
    for lane in LANES:
        anchor_indexes, anchor_target = targets[lane]
        states, gradients = _train_lane(
            lane,
            snapshot,
            parent_state,
            rows,
            source,
            plan,
            anchor_indexes,
            anchor_target,
            device,
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
            "first_batch_gradient_norms": gradients,
            "metrics": lane_metrics,
            "artifacts": artifacts,
        }
    summary = {
        "schema": "azlite_shadow_target_distillation_v1",
        "hashes": hashes | {"target_cache": cache_hash},
        "exclusions": exclusions,
        "selector": selector,
        "target_identity": target_identity,
        "guardrails": {
            "self_play_generated": False,
            "behavior_loss_weight": WEIGHT,
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
        "classification": "inconclusive",
        "recommended_follow_up": "Classify after the preregistered ordinary-PUCT held-out and arena gates.",
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "target caches and four training lanes complete; ordinary arena gates are intentionally pending"
        if args.skip_arenas
        else "training complete; run ordinary arena gates"
    )


if __name__ == "__main__":
    main()
