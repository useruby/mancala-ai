#!/usr/bin/env python3
# ruff: noqa: E402, E701, E702
"""Test whether A16 root-search context makes parent-Q corrections learnable.

This is a non-promoting diagnostic.  The P1 trunk, policy, value, and A16
search are frozen.  P1 is used only while constructing supervised cache rows,
never by the live correction lane.
"""

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

from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.context_action_q_probe import CONTEXT_SIZE, ContextActionQProbe
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

SEED = 238
SIMULATIONS = 1200
WORKDIR = Path("/tmp/azlite_context_action_q_probe")


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"context-action-q:{state_hash}".encode()).hexdigest()[:16], 16
    )


def _root_snapshot(simulation: int, root: Any) -> None:
    return None


def _search(
    item: dict[str, Any],
    evaluator: ArtifactEvaluator,
    *,
    trace: bool = False,
    snapshots: bool = False,
    override: Any = None,
    hook: Any = None,
    root_backup_history: list[dict[str, int | float]] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    trace_rows: list[dict] = []
    reference_rows: list[dict] = []

    def capture(simulation: int, root: Any) -> None:
        reference_rows.append(
            {
                "simulation": int(simulation),
                "children": {
                    int(move): {
                        "visits": int(child.visit_count),
                        "q_value": float(child.q_value),
                    }
                    for move, child in root.children.items()
                },
            }
        )
        if hook is not None:
            hook(simulation, root)

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
        selection_trace=trace_rows if trace else None,
        pre_simulation_hook=capture if snapshots or hook is not None else None,
        selection_q_override=override,
        root_backup_history=root_backup_history,
    )
    search.run(
        KalahGame.from_state(item["state"]), dirichlet_alpha=None, dirichlet_epsilon=0.0
    )
    return search.root_summary(), trace_rows, reference_rows


def _context(child: dict, *, parent_visits: int) -> np.ndarray:
    """Fixed normalized A16 evidence available before the current simulation."""
    action = int(child["move"])
    one_hot = np.zeros(6, np.float32)
    one_hot[action] = 1.0
    return np.concatenate(
        (
            one_hot,
            np.asarray(
                (
                    float(child["q_value"]),
                    np.log1p(float(child["visit_count"])) / np.log1p(SIMULATIONS),
                    float(child["prior"]),
                    float(child["u_component"]),
                    float(parent_visits) / SIMULATIONS,
                ),
                np.float32,
            ),
        )
    )


def _make_cache(
    population: list[dict],
    a16: ArtifactEvaluator,
    p1: ArtifactEvaluator,
    cache_path: Path,
) -> None:
    """Cache aligned root tables.  Each row is evidence available before t."""
    completed: set[str] = set()
    if cache_path.exists():
        for row in _read_cache(cache_path, tolerate_partial=True):
            completed.add(str(row["state_hash"]))
    with cache_path.open("a", encoding="utf-8") as output:
        for state_index, item in enumerate(population):
            if item["state_hash"] in completed:
                continue
            _a16_summary, trace, _ = _search(item, a16, trace=True)
            _p1_summary, _unused, references = _search(item, p1, snapshots=True)
            if len(trace) != SIMULATIONS or len(references) != SIMULATIONS:
                raise RuntimeError("incomplete A16/P1 root trace")
            for candidate, reference in zip(trace, references, strict=True):
                root = candidate["selection_path"][0]
                if int(candidate["simulation_index"]) != int(reference["simulation"]):
                    raise RuntimeError("future P1 root evidence")
                rows = []
                for child in root["children"]:
                    action = int(child["move"])
                    parent = reference["children"].get(action)
                    rows.append(
                        {
                            "action": action,
                            "a16_visits": int(child["visit_count"]),
                            "a16_q": float(child["q_value"]),
                            "u": float(child["u_component"]),
                            "prior": float(child["prior"]),
                            "p1_visits": 0 if parent is None else int(parent["visits"]),
                            "p1_q": 0.0 if parent is None else float(parent["q_value"]),
                        }
                    )
                output.write(
                    json.dumps(
                        {
                            "state_index": state_index,
                            "state_hash": item["state_hash"],
                            "simulation": int(candidate["simulation_index"]),
                            "actual_move": int(root["chosen_move"]),
                            "parent_visits": int(root["parent_visit_count"]),
                            "children": rows,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def _read_cache(path: Path, *, tolerate_partial: bool = False) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not tolerate_partial:
                    raise
        return rows


def _features(
    model: PolicyValueNet, population: list[dict], device: torch.device
) -> np.ndarray:
    x = np.asarray(
        [encode_state(row["state"], input_encoding="kalah_v3") for row in population],
        np.float32,
    )
    with torch.no_grad():
        return model.trunk_features(torch.from_numpy(x).to(device)).cpu().numpy()


def _edge_arrays(
    decisions: list[dict], allowed: set[int], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, context, target, baseline = [], [], [], []
    for decision in decisions:
        index = int(decision["state_index"])
        if index not in allowed:
            continue
        for child in decision["children"]:
            if child["a16_visits"] <= 0 or child["p1_visits"] <= 0:
                continue
            h.append(features[index])
            context.append(
                _context(
                    {
                        "move": child["action"],
                        "q_value": child["a16_q"],
                        "visit_count": child["a16_visits"],
                        "prior": child["prior"],
                        "u_component": child["u"],
                    },
                    parent_visits=decision["parent_visits"],
                )
            )
            target.append(child["p1_q"] - child["a16_q"])
            baseline.append(child["a16_q"])
    return (
        np.asarray(h, np.float32),
        np.asarray(context, np.float32),
        np.asarray(target, np.float32),
        np.asarray(baseline, np.float32),
    )


def _offline(
    decisions: list[dict],
    allowed: set[int],
    features: np.ndarray,
    probe: ContextActionQProbe,
    device: torch.device,
) -> dict[str, float | int | None]:
    total = {
        "flip": 0,
        "same": 0,
        "detected": 0,
        "nonflip": 0,
        "preserved": 0,
        "false": 0,
        "regret": 0.0,
        "captured": 0.0,
        "all": 0,
        "agree": 0,
    }
    probe.eval()
    with torch.no_grad():
        for decision in decisions:
            index = int(decision["state_index"])
            if index not in allowed:
                continue
            children = [
                {
                    "move": c["action"],
                    "visit_count": c["a16_visits"],
                    "q_value": c["a16_q"],
                    "u_component": c["u"],
                }
                for c in decision["children"]
            ]
            reference = {
                int(c["action"]): {"visits": c["p1_visits"], "q_value": c["p1_q"]}
                for c in decision["children"]
            }
            exact = parent_q_counterfactual(
                children, reference, decision["actual_move"]
            )
            correction = {}
            for child in decision["children"]:
                if child["a16_visits"] <= 0:
                    continue
                context = _context(
                    {
                        "move": child["action"],
                        "q_value": child["a16_q"],
                        "visit_count": child["a16_visits"],
                        "prior": child["prior"],
                        "u_component": child["u"],
                    },
                    parent_visits=decision["parent_visits"],
                )
                value = probe(
                    torch.from_numpy(features[index : index + 1]).to(device),
                    torch.from_numpy(context[None]).to(device),
                ).item()
                correction[int(child["action"])] = {
                    "visits": 1,
                    "q_value": child["a16_q"] + value,
                }
            learned = parent_q_counterfactual(
                children, correction, decision["actual_move"]
            )
            chosen = int(learned["cf_move"])
            total["all"] += 1
            total["agree"] += chosen == int(exact["cf_move"])
            if exact["selection_flip"]:
                total["flip"] += 1
                total["same"] += chosen == int(exact["cf_move"])
                total["detected"] += chosen != int(decision["actual_move"])
                total["regret"] += float(exact["selection_regret"])
                if chosen == int(exact["cf_move"]):
                    total["captured"] += float(exact["selection_regret"])
            else:
                total["nonflip"] += 1
                total["preserved"] += chosen == int(decision["actual_move"])
                total["false"] += chosen != int(decision["actual_move"])
    return {
        "exact_flip_action_recall": total["same"] / total["flip"]
        if total["flip"]
        else None,
        "exact_flip_detection_rate": total["detected"] / total["flip"]
        if total["flip"]
        else None,
        "nonflip_preservation_rate": total["preserved"] / total["nonflip"]
        if total["nonflip"]
        else None,
        "overall_exact_parent_action_agreement": total["agree"] / total["all"],
        "exact_parent_score_regret_captured": total["captured"] / total["regret"]
        if total["regret"]
        else None,
        "false_flip_rate": total["false"] / total["nonflip"]
        if total["nonflip"]
        else None,
        "exact_parent_flip_count": total["flip"],
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
    hashes = sorted(row["state_hash"] for row in population)
    cut = len(hashes) * 4 // 5
    split = {
        "schema": "azlite_context_action_q_probe_split_v1",
        "seed": SEED,
        "train_hashes": hashes[:cut],
        "validation_hashes": hashes[cut:],
    }
    split_path = args.workdir / "split_manifest.json"
    split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    p1_path = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16_path = pr233.A16_WORKDIR / "artifacts/step_0016/artifact"
    p1, a16 = ArtifactEvaluator(p1_path), ArtifactEvaluator(a16_path)
    cache_path = args.workdir / "aligned_root_context_cache.jsonl"
    if not args.reuse_cache and cache_path.exists():
        cache_path.unlink()
    _make_cache(population, a16, p1, cache_path)
    decisions = _read_cache(cache_path)
    index_by_hash = {row["state_hash"]: i for i, row in enumerate(population)}
    train_indexes = {index_by_hash[key] for key in split["train_hashes"]}
    validation_indexes = {index_by_hash[key] for key in split["validation_hashes"]}
    checkpoint = (
        pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    trunk_size = int(np.load(checkpoint)["w_input"].shape[1])
    trunk = PolicyValueNet(
        (trunk_size, 3), "residual_v3", input_size_for_encoding("kalah_v3")
    ).to(device)
    load_checkpoint_into_model(trunk, checkpoint)
    trunk.eval()
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)
    features = _features(trunk, population, device)
    train_h, train_c, train_y, _ = _edge_arrays(decisions, train_indexes, features)
    val_h, val_c, val_y, _ = _edge_arrays(decisions, validation_indexes, features)
    probe = ContextActionQProbe(trunk_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    best = float("inf")
    best_state = None
    history = []
    for step in range(1, 2001):
        sample = rng.integers(0, len(train_y), 256)
        optimizer.zero_grad(set_to_none=True)
        predicted = probe(
            torch.from_numpy(train_h[sample]).to(device),
            torch.from_numpy(train_c[sample]).to(device),
        )
        loss = torch.mean(
            (predicted - torch.from_numpy(train_y[sample]).to(device)) ** 2
        )
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            with torch.no_grad():
                validation = float(
                    torch.mean(
                        (
                            probe(
                                torch.from_numpy(val_h).to(device),
                                torch.from_numpy(val_c).to(device),
                            )
                            - torch.from_numpy(val_y).to(device)
                        )
                        ** 2
                    ).cpu()
                )
            history.append({"step": step, "validation_delta_q_mse": validation})
            if validation < best:
                best = validation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                }
    assert best_state is not None
    probe.load_state_dict(best_state)
    checkpoint_path = args.workdir / "context_action_q_probe.pt"
    torch.save(
        {
            "state_dict": best_state,
            "trunk_size": trunk_size,
            "context_size": CONTEXT_SIZE,
        },
        checkpoint_path,
    )
    offline = _offline(decisions, validation_indexes, features, probe, device)
    gate = bool(
        offline["exact_flip_action_recall"] is not None
        and offline["exact_flip_action_recall"] >= 0.70
        and offline["nonflip_preservation_rate"] is not None
        and offline["nonflip_preservation_rate"] >= 0.90
        and offline["exact_parent_score_regret_captured"] is not None
        and offline["exact_parent_score_regret_captured"] >= 0.70
    )
    classification = "inconclusive" if gate else "static_action_q_not_learnable"
    follow_up = (
        "test a search-conditioned Q correction model with a richer root trajectory representation."
        if not gate
        else "run the preregistered frozen 40+40 live root-only correction lane."
    )
    summary = {
        "schema": "azlite_context_action_q_probe_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
        "guardrails": {
            "policy_value_trunk_frozen": True,
            "p1_runtime": False,
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_noise": False,
            "arena_run": False,
            "live_search_skipped": not gate,
        },
        "features": {
            "action_one_hot": True,
            "a16_empirical_q": True,
            "a16_visit_count": "log1p_normalized",
            "a16_prior": True,
            "a16_u_component": True,
            "a16_parent_visit_count": "normalized",
        },
        "exclusions": exclusions,
        "hashes": {
            "replay": sha256_file(replay),
            "p1_weights": sha256_file(p1_path / "weights.json"),
            "a16_weights": sha256_file(a16_path / "weights.json"),
            "split_manifest": sha256_file(split_path),
            "aligned_root_context_cache": sha256_file(cache_path),
            "probe_checkpoint": sha256_file(checkpoint_path),
        },
        "optimization": {
            "seed": SEED,
            "lr": 1e-3,
            "batch_size": 256,
            "weight_decay": 0,
            "steps": 2000,
            "best_validation_delta_q_mse": best,
            "train_edge_rows": len(train_y),
            "validation_edge_rows": len(val_y),
            "validation_history": history,
        },
        "validation_selection_consequence": {
            "learned_context_probe": offline,
            "gate_passed": gate,
        },
    }
    json_path = (
        REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-context-action-q-probe-summary.json"
    )
    report_path = (
        REPO_ROOT / "docs/alphazero-lite-fresh-p1-context-action-q-probe-results.md"
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    json_path.write_text(text)
    report_path.write_text(
        f"# Context Action-Q Probe\n\n**Classification:** `{classification}`\n\n**Recommended follow-up:** {follow_up}\n\n```json\n{text}```\n"
    )
    (args.workdir / "summary.json").write_text(text)
    print(classification)


if __name__ == "__main__":
    main()
