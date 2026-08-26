#!/usr/bin/env python3
# ruff: noqa: E402, E701, E702
"""Fit a detached action-Q diagnostic probe on frozen fresh-P1 root searches."""

from __future__ import annotations

import argparse
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

from ml.alphazero_lite.action_q_probe import ActionQProbe
from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_q_trust_region import parent_q_counterfactual
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
)
from ml.alphazero_lite.self_play import PUCT, encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    input_size_for_encoding,
    load_checkpoint_into_model,
)

WORKDIR = Path("/tmp/azlite_action_q_probe")
SEED = 237
SIMULATIONS = 1200


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"action-q-probe:{state_hash}".encode()).hexdigest()[:16], 16
    )


def _search(
    item: dict[str, Any],
    evaluator: ArtifactEvaluator,
    *,
    trace: bool = False,
    snapshots: bool = False,
    override: Any = None,
) -> tuple[dict, list[dict], list[dict]]:
    rows: list[dict] = []
    refs: list[dict] = []

    def hook(simulation: int, root: Any) -> None:
        refs.append(
            {
                "simulation": simulation,
                "children": {
                    int(move): {
                        "visits": int(child.visit_count),
                        "q_value": float(child.q_value),
                    }
                    for move, child in root.children.items()
                },
            }
        )

    search = PUCT(
        evaluator,
        SIMULATIONS,
        1.25,
        random.Random(_seed(item["state_hash"])),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        selection_trace=rows if trace else None,
        pre_simulation_hook=hook if snapshots else None,
        selection_q_override=override,
    )
    search.run(
        KalahGame.from_state(item["state"]), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    return search.root_summary(), rows, refs


def _root_target(item: dict[str, Any], p1: ArtifactEvaluator) -> dict[str, Any]:
    summary, _, _ = _search(item, p1)
    legal = np.zeros(6, dtype=bool)
    visits = np.zeros(6, dtype=np.int64)
    q = np.zeros(6, dtype=np.float32)
    for child in summary["child_stats"]:
        action = int(child["move"])
        legal[action] = True
        visits[action] = int(child["visits"])
        q[action] = float(child["q_value"])
    eligible = legal & (visits > 0)
    if not eligible.any():
        raise RuntimeError("P1 root has no visited legal child")
    return {
        "state_hash": item["state_hash"],
        "state": item["state"],
        "legal": legal.tolist(),
        "visits": visits.tolist(),
        "q": q.tolist(),
        "normalized_visits": (visits / visits.sum()).tolist(),
        "selected_move": int(summary["selected_move"]),
        "eligible": eligible.tolist(),
    }


def _centered(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    result[mask] = values[mask] - values[mask].mean()
    return result


def _metrics(records: list[dict], predicted: np.ndarray) -> dict[str, Any]:
    accum: dict[str, list[float]] = {
        key: []
        for key in (
            "mse",
            "l1",
            "cosine",
            "best_q_action_agreement",
            "top_two_ordering_agreement",
            "pairwise_rank_agreement",
        )
    }
    bins: dict[str, list[float]] = {"1": [], "2_7": [], "8_31": [], "32_plus": []}
    for record, raw in zip(records, predicted, strict=True):
        mask = np.asarray(record["eligible"], bool)
        target = _centered(np.asarray(record["q"]), mask)
        pred = _centered(raw, mask)
        t, p = target[mask], pred[mask]
        accum["mse"].append(float(np.mean((p - t) ** 2)))
        accum["l1"].append(float(np.mean(abs(p - t))))
        denom = np.linalg.norm(t) * np.linalg.norm(p)
        accum["cosine"].append(
            float(np.dot(t, p) / denom) if denom > 1e-12 else float(np.allclose(t, p))
        )
        actions = np.flatnonzero(mask)
        target_order = sorted(actions, key=lambda a: (-target[a], a))
        pred_order = sorted(actions, key=lambda a: (-pred[a], a))
        accum["best_q_action_agreement"].append(float(target_order[0] == pred_order[0]))
        accum["top_two_ordering_agreement"].append(
            float(target_order[:2] == pred_order[:2])
        )
        pairs = [(a, b) for i, a in enumerate(actions) for b in actions[i + 1 :]]
        accum["pairwise_rank_agreement"].append(
            float(
                np.mean(
                    [
                        (target[a] - target[b]) * (pred[a] - pred[b]) >= 0
                        for a, b in pairs
                    ]
                )
            )
            if pairs
            else 1.0
        )
        for action in actions:
            visit = record["visits"][action]
            name = (
                "1"
                if visit == 1
                else "2_7"
                if visit < 8
                else "8_31"
                if visit < 32
                else "32_plus"
            )
            bins[name].append(float(abs(pred[action] - target[action])))
    return {key: float(np.mean(value)) for key, value in accum.items()} | {
        "visit_bin_centered_q_l1": {
            key: float(np.mean(value)) if value else None for key, value in bins.items()
        }
    }


def _features(
    model: PolicyValueNet, records: list[dict], device: torch.device
) -> np.ndarray:
    x = np.asarray(
        [encode_state(row["state"], input_encoding="kalah_v3") for row in records],
        np.float32,
    )
    with torch.no_grad():
        return model.trunk_features(torch.from_numpy(x).to(device)).cpu().numpy()


def _one_step(records: list[dict], p1: ArtifactEvaluator) -> np.ndarray:
    output = np.zeros((len(records), 6), np.float32)
    for index, record in enumerate(records):
        root = KalahGame.from_state(record["state"])
        player = root.current_player
        for action in np.flatnonzero(np.asarray(record["legal"], bool)):
            child = root.clone()
            assert child.move(child.pit_index(int(action)))
            _policy, value = p1.evaluate(child)
            output[index, action] = value if child.current_player == player else -value
    return output


def _offline(
    records: list[dict],
    prediction: dict[str, np.ndarray],
    a16: ArtifactEvaluator,
    p1: ArtifactEvaluator,
) -> dict[str, Any]:
    totals = {
        name: {
            "flip": 0,
            "same": 0,
            "detected": 0,
            "nonflip": 0,
            "preserved": 0,
            "regret": 0.0,
            "captured": 0.0,
            "false": 0,
            "all": 0,
            "agree": 0,
        }
        for name in prediction
    }
    by_hash = {row["state_hash"]: index for index, row in enumerate(records)}
    for item in records:
        _a, ordinary, _ = _search(item, a16, trace=True)
        _p, _p_trace, refs = _search(item, p1, trace=True, snapshots=True)
        for candidate, ref in zip(ordinary, refs, strict=True):
            root = candidate["selection_path"][0]
            exact = parent_q_counterfactual(
                root["children"], ref["children"], root["chosen_move"]
            )
            exact_flip = bool(exact["selection_flip"])
            for name, values in prediction.items():
                index = by_hash[item["state_hash"]]
                table = {
                    int(c["move"]): {
                        "visits": 1,
                        "q_value": float(values[index, int(c["move"])]),
                    }
                    for c in root["children"]
                    if int(c["visit_count"]) > 0
                }
                probe = parent_q_counterfactual(
                    root["children"], table, root["chosen_move"]
                )
                chosen = int(probe["cf_move"])
                total = totals[name]
                total["all"] += 1
                total["agree"] += chosen == int(exact["cf_move"])
                if exact_flip:
                    total["flip"] += 1
                    total["regret"] += float(exact["selection_regret"])
                    total["detected"] += chosen != int(root["chosen_move"])
                    if chosen == int(exact["cf_move"]):
                        total["same"] += 1
                        total["captured"] += float(exact["selection_regret"])
                else:
                    total["nonflip"] += 1
                    total["preserved"] += chosen == int(root["chosen_move"])
                    total["false"] += chosen != int(root["chosen_move"])
    return {
        name: {
            "exact_flip_action_recall": d["same"] / d["flip"] if d["flip"] else None,
            "exact_flip_detection_rate": d["detected"] / d["flip"]
            if d["flip"]
            else None,
            "nonflip_preservation_rate": d["preserved"] / d["nonflip"]
            if d["nonflip"]
            else None,
            "overall_exact_parent_action_agreement": d["agree"] / d["all"],
            "exact_parent_score_regret_captured": d["captured"] / d["regret"]
            if d["regret"]
            else None,
            "false_flip_rate": d["false"] / d["nonflip"] if d["nonflip"] else None,
            "exact_parent_flip_count": d["flip"],
        }
        for name, d in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    replay = pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl"
    rows = pr233.read_jsonl(replay)
    population, exclusions = pr233._population(rows)
    manifest = {
        "schema": "azlite_action_q_probe_split_v1",
        "seed": SEED,
        "eligible_hashes": sorted(row["state_hash"] for row in population),
    }
    ordered = manifest["eligible_hashes"]
    cut = len(ordered) * 4 // 5
    manifest |= {"train_hashes": ordered[:cut], "validation_hashes": ordered[cut:]}
    manifest_path = args.workdir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    p1_path = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_path = pr233.A16_WORKDIR / "artifacts/step_0016/artifact"
    p1, a16 = ArtifactEvaluator(p1_path), ArtifactEvaluator(a16_path)
    cache_path = args.workdir / "p1_root_action_q_cache.json"
    if args.reuse_cache and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        cache = [_root_target(item, p1) for item in population]
        cache_path.write_text(json.dumps(cache, sort_keys=True) + "\n")
    cache_sha = sha256_file(cache_path)
    by_hash = {row["state_hash"]: row for row in cache}
    train_rows = [by_hash[key] for key in manifest["train_hashes"]]
    val_rows = [by_hash[key] for key in manifest["validation_hashes"]]
    model = PolicyValueNet(
        (
            int(
                np.load(
                    pr233.P1_WORKDIR
                    / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
                )["w_input"].shape[1]
            ),
            3,
        ),
        "residual_v3",
        input_size_for_encoding("kalah_v3"),
    ).to(device)
    load_checkpoint_into_model(
        model, pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_h, val_h = (
        _features(model, train_rows, device),
        _features(model, val_rows, device),
    )
    probe = ActionQProbe(train_h.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.0)
    train_q = np.asarray([r["q"] for r in train_rows], np.float32)
    train_mask = np.asarray([r["eligible"] for r in train_rows], bool)
    val_q = np.asarray([r["q"] for r in val_rows], np.float32)
    val_mask = np.asarray([r["eligible"] for r in val_rows], bool)

    def loss(h: np.ndarray, q: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        raw = probe(torch.from_numpy(h).to(device))
        m = torch.from_numpy(mask).to(device)
        target = torch.from_numpy(q).to(device)
        pred = raw - (raw * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        target = target - (target * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)
        return (((pred - target) ** 2) * m).sum() / m.sum()

    best, best_state, history = float("inf"), None, []
    rng = np.random.default_rng(SEED)
    for step in range(1, 2001):
        indexes = rng.integers(0, len(train_rows), 256)
        optimizer.zero_grad(set_to_none=True)
        current = loss(train_h[indexes], train_q[indexes], train_mask[indexes])
        current.backward()
        optimizer.step()
        if step % 50 == 0:
            with torch.no_grad():
                validation = float(loss(val_h, val_q, val_mask).cpu())
            history.append({"step": step, "validation_centered_q_mse": validation})
            if validation < best:
                best, best_state = (
                    validation,
                    {
                        k: v.detach().cpu().clone()
                        for k, v in probe.state_dict().items()
                    },
                )
    assert best_state is not None
    probe.load_state_dict(best_state)
    torch.save(
        {"state_dict": best_state, "trunk_size": train_h.shape[1]},
        args.workdir / "action_q_probe.pt",
    )
    with torch.no_grad():
        train_pred = probe(torch.from_numpy(train_h).to(device)).cpu().numpy()
        val_pred = probe(torch.from_numpy(val_h).to(device)).cpu().numpy()
    baseline_train, baseline_val = _one_step(train_rows, p1), _one_step(val_rows, p1)
    offline = _offline(
        val_rows, {"learned_probe": val_pred, "one_step_value_q": baseline_val}, a16, p1
    )["learned_probe"]
    gate = all(
        [
            offline["exact_flip_action_recall"] is not None
            and offline["exact_flip_action_recall"] >= 0.70,
            offline["nonflip_preservation_rate"] is not None
            and offline["nonflip_preservation_rate"] >= 0.90,
            offline["exact_parent_score_regret_captured"] is not None
            and offline["exact_parent_score_regret_captured"] >= 0.70,
        ]
    )
    frozen_amplified, frozen_washed = pr233._frozen_hashes()
    replay_manifest = {
        row["state_hash"]: row for row in json.loads(pr233.MANIFEST.read_text())["rows"]
    }
    population_by_hash = {row["state_hash"]: row for row in population} | {
        state_hash: {
            "state_hash": state_hash,
            "replay_index": int(meta["replay_index"]),
            "state": pr233.decode_kalah_v3_base_state(
                list(rows[int(meta["replay_index"])]["state"])
            ),
        }
        for state_hash, meta in replay_manifest.items()
        if state_hash in frozen_amplified | frozen_washed
    }
    frozen_items = {
        "amplified": [population_by_hash[key] for key in sorted(frozen_amplified)],
        "washed": [population_by_hash[key] for key in sorted(frozen_washed)],
    }
    frozen_cache_path = args.workdir / "frozen_p1_root_action_q_cache.json"
    frozen_cache = {
        name: [_root_target(item, p1) for item in items]
        for name, items in frozen_items.items()
    }
    frozen_cache_path.write_text(json.dumps(frozen_cache, sort_keys=True) + "\n")
    frozen_results: dict[str, Any] = {}
    hard_root: dict[str, Any] | None = None
    for name, records in frozen_cache.items():
        h = _features(model, records, device)
        with torch.no_grad():
            predicted = probe(torch.from_numpy(h).to(device)).cpu().numpy()
        baseline = _one_step(records, p1)
        frozen_results[name] = {
            "hashes": [row["state_hash"] for row in records],
            "probe_q_metrics": _metrics(records, predicted),
            "one_step_value_q_metrics": _metrics(records, baseline),
            "selection_consequence": _offline(
                records, {"learned_probe": predicted}, a16, p1
            )["learned_probe"],
        }
        for record, values in zip(records, predicted, strict=True):
            if (
                record["state_hash"]
                == "362958a9d30519f98e27a71256c89f1acba0a61e3d93e609adb59609bff61674"
            ):
                mask = np.asarray(record["eligible"], bool)
                hard_root = {
                    "state_hash": record["state_hash"],
                    "eligible_actions": np.flatnonzero(mask).tolist(),
                    "p1_q": record["q"],
                    "probe_q": values.tolist(),
                    "probe_best_q_action": int(
                        sorted(np.flatnonzero(mask), key=lambda a: (-values[a], a))[0]
                    ),
                    "p1_selected_move": record["selected_move"],
                    "live_qprobe_search": "skipped_validation_gate_failed",
                }
    classification = "inconclusive" if gate else "static_action_q_not_learnable"
    follow_up = (
        "integrate the frozen Q head into a versioned model/checkpoint and evaluate a root-Q-head PUCT rule in canonical arenas, including policy-utilization and compute comparisons."
        if gate
        else "test a search-conditioned Q correction model using state plus root visit/Q statistics rather than expanding the policy network again."
    )
    summary = {
        "schema": "azlite_action_q_probe_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
        "guardrails": {
            "policy_value_trunk_frozen": True,
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_noise": False,
            "arena_run": False,
            "live_qprobe_search_skipped": not gate,
        },
        "hashes": {
            "replay": sha256_file(replay),
            "p1_weights": sha256_file(p1_path / "weights.json"),
            "a16_weights": sha256_file(a16_path / "weights.json"),
            "target_cache": cache_sha,
            "frozen_target_cache": sha256_file(frozen_cache_path),
            "split_manifest": sha256_file(manifest_path),
            "probe_checkpoint": sha256_file(args.workdir / "action_q_probe.pt"),
        },
        "exclusions": exclusions,
        "optimization": {
            "seed": SEED,
            "lr": 1e-3,
            "batch_size": 256,
            "weight_decay": 0,
            "steps": 2000,
            "best_validation_centered_q_mse": best,
            "validation_history": history,
        },
        "metrics": {
            "train": {
                "probe": _metrics(train_rows, train_pred),
                "one_step_value_q": _metrics(train_rows, baseline_train),
            },
            "validation": {
                "probe": _metrics(val_rows, val_pred),
                "one_step_value_q": _metrics(val_rows, baseline_val),
            },
        },
        "validation_selection_consequence": {
            "learned_probe": offline,
            "one_step_value_q": _offline(
                val_rows, {"one_step_value_q": baseline_val}, a16, p1
            )["one_step_value_q"],
            "gate_passed": gate,
        },
        "frozen_offline": frozen_results,
        "hard_root": hard_root,
    }
    out = REPO_ROOT / "docs/data/alphazero-lite-fresh-p1-action-q-probe-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = REPO_ROOT / "docs/alphazero-lite-fresh-p1-action-q-probe-results.md"
    report.write_text(
        f"# Action-Q Probe\n\n**Classification:** `{classification}`\n\n**Recommended follow-up:** {follow_up}\n\n```json\n{json.dumps(summary, indent=2, sort_keys=True)}\n```\n"
    )
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(classification)


if __name__ == "__main__":
    main()
