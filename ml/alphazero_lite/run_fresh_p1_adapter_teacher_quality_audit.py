#!/usr/bin/env python3
"""Evaluation-only same-state teacher-quality audit for PR #214 A16."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
    evaluate_artifact_position,
)
from ml.alphazero_lite.fresh_p1_adapter_teacher_audit import (  # noqa: E402
    decode_kalah_v3_base_state,
    state_round_trips_kalah_v3,
)
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import ARENA_SUITE  # noqa: E402
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (  # noqa: E402
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_budget_factorization import (  # noqa: E402
    A16_STATE_SHA,
    P1_CHECKPOINT_SHA,
    REPLAY_SHA,
    state_hash as model_state_hash,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (  # noqa: E402
    BETA,
    new_model,
    output,
)
from ml.alphazero_lite.self_play import (  # noqa: E402
    PUCT,
    build_eval_search_options,
    build_policy_target,
    encode_state,
)
from ml.alphazero_lite.train import (  # noqa: E402
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SAMPLE_SIZE = 4096
SAMPLE_SEED = 214
TEMPERATURE_THRESHOLD = 10
TEMPERATURE_EARLY = 1.0
TEMPERATURE_LATE = 0.1
PERCENTILES = (50, 75, 90, 95, 99)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def entropy(policy: np.ndarray) -> float:
    positive = policy[policy > 0]
    return float(-np.sum(positive * np.log(positive)))


def js(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0
    return float(
        0.5
        * np.sum(left * np.log(np.maximum(left, 1e-12) / np.maximum(midpoint, 1e-12)))
        + 0.5
        * np.sum(right * np.log(np.maximum(right, 1e-12) / np.maximum(midpoint, 1e-12)))
    )


def legal_distribution(policy: np.ndarray, legal: list[int]) -> np.ndarray:
    values = np.asarray(policy, dtype=float)[legal]
    total = float(values.sum())
    if total <= 0:
        return np.full(len(legal), 1.0 / len(legal))
    return values / total


def descriptive(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "max": None, **{f"p{p}": None for p in PERCENTILES}}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        **{f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES},
        "max": float(array.max()),
    }


def top_actions(policy: np.ndarray, legal: list[int]) -> tuple[int, set[int]]:
    ordered = sorted(legal, key=lambda move: (-float(policy[move]), move))
    return ordered[0], set(ordered[:2])


def state_seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"pr214-teacher-audit:{state_hash}".encode()).hexdigest()[:16],
        16,
    )


def temperature_for(row: dict[str, Any]) -> float:
    return (
        TEMPERATURE_EARLY
        if int(row["move_index"]) < TEMPERATURE_THRESHOLD
        else TEMPERATURE_LATE
    )


def move_index_bucket(move_index: int) -> str:
    if move_index < 5:
        return "00-04"
    if move_index < TEMPERATURE_THRESHOLD:
        return "05-09"
    if move_index < 20:
        return "10-19"
    if move_index < 40:
        return "20-39"
    return "40+"


def select_sample(rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    """Round-robin deterministic strata without outcome or arena covariates."""
    entropies = np.asarray(
        [entropy(np.asarray(row["policy"], dtype=float)) for row in rows]
    )
    cuts = np.quantile(entropies, [0.25, 0.5, 0.75])
    groups: dict[tuple[str, int, int, int], list[tuple[str, int]]] = defaultdict(list)
    metadata: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        state = list(row["state"])
        hash_value = canonical_hash(state)
        legal = [int(move) for move in row["legal_moves"]]
        policy_entropy = float(entropies[index])
        quartile = int(np.searchsorted(cuts, policy_entropy, side="right")) + 1
        phase = "early" if int(row["move_index"]) < TEMPERATURE_THRESHOLD else "late"
        key = (phase, int(row["player"]), min(len(legal), 6), quartile)
        score = canonical_hash(
            {"seed": SAMPLE_SEED, "index": index, "state": hash_value}
        )
        groups[key].append((score, index))
        metadata[index] = {
            "replay_index": index,
            "state_hash": hash_value,
            "move_index": int(row["move_index"]),
            "move_index_bucket": move_index_bucket(int(row["move_index"])),
            "phase": phase,
            "player": int(row["player"]),
            "legal_move_count": len(legal),
            "stored_policy_entropy": policy_entropy,
            "stored_entropy_quartile": quartile,
        }
    for entries in groups.values():
        entries.sort()
    selected: list[int] = []
    offsets = {key: 0 for key in groups}
    while len(selected) < SAMPLE_SIZE:
        added = False
        for key in sorted(groups):
            offset = offsets[key]
            if offset < len(groups[key]) and len(selected) < SAMPLE_SIZE:
                selected.append(groups[key][offset][1])
                offsets[key] += 1
                added = True
        if not added:
            break
    if len(selected) != SAMPLE_SIZE or len(set(selected)) != SAMPLE_SIZE:
        raise RuntimeError("unable to select 4096 unique replay states")
    return selected, [metadata[index] for index in selected]


def search_target(
    evaluator: ArtifactEvaluator, row: dict[str, Any], state_hash: str, simulations: int
) -> dict[str, Any]:
    state = decode_kalah_v3_base_state(list(row["state"]))
    game = KalahGame.from_state(state)
    legal = [int(move) for move in game.possible_moves()]
    search = PUCT(
        evaluator=evaluator,
        simulations=simulations,
        c_puct=1.25,
        rng=random.Random(state_seed(state_hash)),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    policy = np.asarray(
        build_policy_target(
            visits, legal_moves=legal, temperature=temperature_for(row), mode="default"
        ),
        dtype=float,
    )
    ordered = sorted(legal, key=lambda move: (-int(visits[move]), move))
    margin = float(policy[ordered[0]] - policy[ordered[1]]) if len(ordered) > 1 else 1.0
    return {
        "policy": policy.tolist(),
        "visits": [int(value) for value in visits.tolist()],
        "top1": int(ordered[0]),
        "top1_policy_margin": margin,
        "root_visits": int(root.visit_count),
    }


def cache_targets(
    *,
    rows: list[dict[str, Any]],
    indexes: list[int],
    evaluator: ArtifactEvaluator,
    path: Path,
) -> dict[int, dict[str, Any]]:
    expected = {index: canonical_hash(rows[index]["state"]) for index in indexes}
    cached: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for entry in read_jsonl(path):
            index = int(entry["replay_index"])
            if expected.get(index) == entry.get("state_hash"):
                cached[index] = entry
    missing = [index for index in indexes if index not in cached]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for index in missing:
                row, hash_value = rows[index], expected[index]
                entry = {
                    "replay_index": index,
                    "state_hash": hash_value,
                    "clean_384": search_target(evaluator, row, hash_value, 384),
                    "clean_1200": search_target(evaluator, row, hash_value, 1200),
                }
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
                cached[index] = entry
    return cached


def policy_metrics(
    left: np.ndarray, right: np.ndarray, legal: list[int]
) -> dict[str, float]:
    left, right = legal_distribution(left, legal), legal_distribution(right, legal)
    left_top, left_top2 = top_actions(left, list(range(len(legal))))
    right_top, right_top2 = top_actions(right, list(range(len(legal))))
    return {
        "l1": float(np.abs(left - right).sum()),
        "js": js(left, right),
        "top1_disagreement": float(left_top != right_top),
        "top2_set_disagreement": float(left_top2 != right_top2),
        "entropy_difference": entropy(left) - entropy(right),
        "largest_action_probability_difference": float(np.abs(left - right).max()),
    }


def grouped_metrics(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {
        group: {
            comparison: {
                metric: descriptive(
                    [float(row["comparisons"][comparison][metric]) for row in entries]
                )
                for metric in (
                    "l1",
                    "js",
                    "top1_disagreement",
                    "top2_set_disagreement",
                    "entropy_difference",
                    "largest_action_probability_difference",
                )
            }
            for comparison in (
                "stored_vs_clean384",
                "clean384_vs_clean1200",
                "stored_vs_clean1200",
            )
        }
        for group, entries in sorted(groups.items())
    }


def held_out_probe(
    evaluator: ArtifactEvaluator,
    parent: torch.nn.Module,
    a16: torch.nn.Module,
    a16_artifact: Path,
) -> dict[str, Any]:
    """Analyze PR #216's canonical suite only after replay-only calibration."""
    rows: list[dict[str, Any]] = []
    a16_evaluator = ArtifactEvaluator(a16_artifact)
    options = build_eval_search_options(
        root_policy_mode="deterministic", tactical_root_bias=0.0, normalize_values=False
    )
    for opening_index, entry in enumerate(read_jsonl(ARENA_SUITE)):
        game = KalahGame.from_state(
            {
                "player_pits": [4] * 6,
                "opponent_pits": [4] * 6,
                "player_store": 0,
                "opponent_store": 0,
                "current_player": 0,
            }
        )
        for move in entry["prefix_moves"]:
            game.move(move)
        state = game.to_state()
        encoded = encode_state(state, input_encoding="kalah_v3")
        audit_row = {"state": encoded, "move_index": len(entry["prefix_moves"])}
        hash_value = canonical_hash(encoded)
        clean384 = search_target(evaluator, audit_row, hash_value, 384)
        clean1200 = search_target(evaluator, audit_row, hash_value, 1200)
        pr216_search = {
            str(simulations): {
                name: evaluate_artifact_position(
                    evaluator=candidate,
                    state=state,
                    simulations=simulations,
                    seed=42 + opening_index,
                    c_puct=1.25,
                    search_options=options,
                )
                for name, candidate in (("p1", evaluator), ("a16", a16_evaluator))
            }
            for simulations in (384, 1200)
        }
        x = np.asarray([encoded], dtype=np.float32)
        mask = legal_mask_matrix_for_encoded_states(x)
        pp, pa = (
            output(parent.state_dict(), x, mask)[0],
            output(a16.state_dict(), x, mask)[0],
        )
        legal = game.possible_moves()
        delta = pa[legal] - pp[legal]
        d384 = np.asarray(clean384["policy"])[legal] - pp[legal]
        d1200 = np.asarray(clean1200["policy"])[legal] - pp[legal]
        rows.append(
            {
                "opening_index": opening_index,
                "state_hash": hash_value,
                "p1_network_top1": top_actions(pp[legal], list(range(len(legal))))[0],
                "a16_network_top1": top_actions(pa[legal], list(range(len(legal))))[0],
                "clean384_top1": clean384["top1"],
                "clean1200_top1": clean1200["top1"],
                "clean384_margin": clean384["top1_policy_margin"],
                "clean1200_margin": clean1200["top1_policy_margin"],
                "cosine_a16_384": cosine(delta, d384),
                "cosine_a16_1200": cosine(delta, d1200),
                "pr216_puct_selected_moves": {
                    budget: {
                        name: result["selected_move"] for name, result in values.items()
                    }
                    for budget, values in pr216_search.items()
                },
            }
        )
    highlighted = [row for row in rows if row["opening_index"] in (10, 13)]
    return {
        "suite_sha256": sha256_file(ARENA_SUITE),
        "count": len(rows),
        "a16_p1_agree_384_diverge_1200_indexes": [
            row["opening_index"]
            for row in rows
            if row["pr216_puct_selected_moves"]["384"]["a16"]
            == row["pr216_puct_selected_moves"]["384"]["p1"]
            and row["pr216_puct_selected_moves"]["1200"]["a16"]
            != row["pr216_puct_selected_moves"]["1200"]["p1"]
        ],
        "rows_10_13": highlighted,
        "all_rows": rows,
    }


def gradient_vector(
    model: torch.nn.Module, x: np.ndarray, mask: np.ndarray, target: np.ndarray
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    logits, _ = model(torch.from_numpy(x))
    masked = logits.masked_fill(torch.from_numpy(mask) <= 0, -1e9)
    loss = (
        -(torch.from_numpy(target) * torch.log_softmax(masked, dim=1)).sum(dim=1).mean()
    )
    loss.backward()
    return torch.cat(
        [
            parameter.grad.detach().reshape(-1)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    ).numpy()


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if norm <= 1e-12 else float(np.dot(left, right) / norm)


def markdown(summary: dict[str, Any]) -> str:
    classification = summary["classification"]
    hashes = summary["hashes"]
    aggregate = summary["teacher_disagreement"]["overall"]
    rows = [
        "# PR #214 Teacher-Quality Audit",
        "",
        f"**Classification:** `{classification}`",
        "",
        f"**Recommended next experiment:** {summary['recommended_next_experiment']}",
        "",
        "## Immutable Contracts",
        "",
        "```json",
        json.dumps(
            {"hashes": hashes, "invariants": summary["invariants"]},
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Teacher Disagreement",
        "",
        "| Comparison | Mean L1 | P95 L1 | Top-1 disagreement | Mean JS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in aggregate.items():
        rows.append(
            f"| {name} | {metrics['l1']['mean']:.6f} | {metrics['l1']['p95']:.6f} | {metrics['top1_disagreement']['mean']:.4f} | {metrics['js']['mean']:.6f} |"
        )
    alignment = summary["alignment"]
    gradient = summary["initial_gradient"]
    rows.extend(
        [
            "",
            "## Learned-Delta Alignment",
            "",
            "| Metric | Clean384 | Clean1200 |",
            "| --- | ---: | ---: |",
            f"| Mean cosine | {alignment['cosine_a16_384']['mean']:.6f} | {alignment['cosine_a16_1200']['mean']:.6f} |",
            f"| Mean dot product | {alignment['dot_a16_384']['mean']:.6f} | {alignment['dot_a16_1200']['mean']:.6f} |",
            f"| Mean CE improvement | {alignment['improvement_clean384']['mean']:.6f} | {alignment['improvement_clean1200']['mean']:.6f} |",
            "",
            "## Initial Gradient Cosines",
            "",
            "| Gradient pair | Cosine |",
            "| --- | ---: |",
            *[
                f"| {name} | {value:.6f} |"
                for name, value in gradient["cosines"].items()
            ],
            "",
            "## Held-Out PR #216 Probe",
            "",
            "Reproduced the deterministic 384-agree/1200-diverge opening indexes: "
            + ", ".join(
                map(
                    str,
                    summary["held_out_pr216_probe"][
                        "a16_p1_agree_384_diverge_1200_indexes"
                    ],
                )
            )
            + ".",
            "",
            "```json",
            json.dumps(
                {
                    "cross_entropy": alignment["cross_entropy"],
                    "critical_subsets": summary["critical_subsets"],
                    "early_noise_attribution": summary["early_noise_attribution"],
                    "held_out_rows_10_13": summary["held_out_pr216_probe"][
                        "rows_10_13"
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr214_teacher_quality_audit")
    )
    parser.add_argument(
        "--p1-workdir", type=Path, default=Path("/tmp/azlite_fresh_selfplay_anchor")
    )
    parser.add_argument(
        "--adapter-workdir",
        type=Path,
        default=Path("/tmp/azlite_fresh_p1_parent_adapter"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-adapter-teacher-quality-audit-summary.json",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-adapter-teacher-quality-audit-results.md",
    )
    args = parser.parse_args()
    p1_checkpoint = (
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    a16_checkpoint = args.adapter_workdir / "artifacts/step_0016/checkpoint.npz"
    replay = args.adapter_workdir / "fresh_p1_self_play.jsonl"
    rows = read_jsonl(replay)
    parent, a16 = new_model(torch.device("cpu")), new_model(torch.device("cpu"))
    load_checkpoint_into_model(parent, p1_checkpoint)
    load_checkpoint_into_model(a16, a16_checkpoint)
    x_all = np.asarray([row["state"] for row in rows], dtype=np.float32)
    masks_all = legal_mask_matrix_for_encoded_states(x_all)
    p_parent_all, p_a16_all = (
        output(parent.state_dict(), x_all, masks_all),
        output(a16.state_dict(), x_all, masks_all),
    )
    stored_all = np.asarray([row["policy"] for row in rows], dtype=float)
    l1_mean = float(np.abs(p_parent_all - p_a16_all).sum(1).mean())
    # Reproduce the committed pure-search denominator without updating any model.
    pure_state = torch.load(
        args.adapter_workdir / "pure_search/snapshots/step_0046.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    pure_policy = output(pure_state, x_all, masks_all)

    def ce(policy: np.ndarray, target: np.ndarray) -> np.ndarray:
        return -np.sum(target * np.log(np.maximum(policy, 1e-12)), axis=1)

    fit = float(
        (ce(p_parent_all, stored_all).mean() - ce(p_a16_all, stored_all).mean())
        / (ce(p_parent_all, stored_all).mean() - ce(pure_policy, stored_all).mean())
    )
    invariants = {
        "p1_checkpoint_hash": sha256_file(p1_checkpoint) == P1_CHECKPOINT_SHA,
        "a16_state_hash": model_state_hash(a16.state_dict()) == A16_STATE_SHA,
        "replay_hash": sha256_file(replay) == REPLAY_SHA,
        "mean_legal_l1_reproduced": abs(l1_mean - 0.0017535027555326227) < 1e-10,
        "fit_fraction_reproduced": abs(fit - 0.3259840837739601) < 1e-10,
    }
    indexes, manifest_rows = select_sample(rows)
    round_trip = all(
        state_round_trips_kalah_v3(list(rows[index]["state"])) for index in indexes
    )
    invariants["sample_state_round_trip"] = round_trip
    if not all(invariants.values()):
        summary = {
            "schema": "azlite_pr214_teacher_quality_audit_v1",
            "hashes": {
                "p1_checkpoint": sha256_file(p1_checkpoint),
                "a16_state": model_state_hash(a16.state_dict()),
                "replay": sha256_file(replay),
            },
            "invariants": invariants,
            "classification": "invariant_failure",
            "recommended_next_experiment": "Repair immutable artifact, replay, or state-reconstruction contracts before interpreting teacher quality.",
        }
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        args.out_report.write_text(
            markdown(
                {
                    **summary,
                    "teacher_disagreement": {"overall": {}},
                    "alignment": {},
                    "initial_gradient": {},
                    "critical_subsets": {},
                    "early_noise_attribution": {},
                    "held_out_pr216_probe": {},
                }
            )
        )
        return
    args.workdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "azlite_pr214_teacher_audit_state_manifest_v1",
        "sample_seed": SAMPLE_SEED,
        "sample_size": SAMPLE_SIZE,
        "temperature_threshold": TEMPERATURE_THRESHOLD,
        "rows": manifest_rows,
    }
    manifest_path = args.workdir / "audit_state_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    evaluator = ArtifactEvaluator(
        args.p1_workdir / "beta_095/snapshot_artifacts/step_0046/artifact"
    )
    targets = cache_targets(
        rows=rows,
        indexes=indexes,
        evaluator=evaluator,
        path=args.workdir / "same_state_p1_targets.jsonl",
    )
    records: list[dict[str, Any]] = []
    for index, meta in zip(indexes, manifest_rows, strict=True):
        row, cached = rows[index], targets[index]
        legal = [int(move) for move in row["legal_moves"]]
        stored = np.asarray(row["policy"], dtype=float)
        clean384, clean1200 = (
            np.asarray(cached["clean_384"]["policy"]),
            np.asarray(cached["clean_1200"]["policy"]),
        )
        pp, pa = p_parent_all[index], p_a16_all[index]
        d_a16, d384, d1200 = (
            pa[legal] - pp[legal],
            clean384[legal] - pp[legal],
            clean1200[legal] - pp[legal],
        )
        record = {
            **meta,
            "comparisons": {
                "stored_vs_clean384": policy_metrics(stored, clean384, legal),
                "clean384_vs_clean1200": policy_metrics(clean384, clean1200, legal),
                "stored_vs_clean1200": policy_metrics(stored, clean1200, legal),
            },
            "clean384_top1": cached["clean_384"]["top1"],
            "clean1200_top1": cached["clean_1200"]["top1"],
            "clean1200_margin": cached["clean_1200"]["top1_policy_margin"],
            "l1_a16_vs_p1": float(np.abs(pa[legal] - pp[legal]).sum()),
            "js_a16_vs_p1": js(
                legal_distribution(pa, legal), legal_distribution(pp, legal)
            ),
            "improvement_stored": float(
                ce(pp[legal][None, :], legal_distribution(stored, legal)[None, :])[0]
                - ce(pa[legal][None, :], legal_distribution(stored, legal)[None, :])[0]
            ),
            "improvement_clean384": float(
                ce(pp[legal][None, :], legal_distribution(clean384, legal)[None, :])[0]
                - ce(pa[legal][None, :], legal_distribution(clean384, legal)[None, :])[
                    0
                ]
            ),
            "improvement_clean1200": float(
                ce(pp[legal][None, :], legal_distribution(clean1200, legal)[None, :])[0]
                - ce(pa[legal][None, :], legal_distribution(clean1200, legal)[None, :])[
                    0
                ]
            ),
            "ce_p1_stored": float(
                ce(pp[legal][None, :], legal_distribution(stored, legal)[None, :])[0]
            ),
            "ce_a16_stored": float(
                ce(pa[legal][None, :], legal_distribution(stored, legal)[None, :])[0]
            ),
            "ce_p1_clean384": float(
                ce(pp[legal][None, :], legal_distribution(clean384, legal)[None, :])[0]
            ),
            "ce_a16_clean384": float(
                ce(pa[legal][None, :], legal_distribution(clean384, legal)[None, :])[0]
            ),
            "ce_p1_clean1200": float(
                ce(pp[legal][None, :], legal_distribution(clean1200, legal)[None, :])[0]
            ),
            "ce_a16_clean1200": float(
                ce(pa[legal][None, :], legal_distribution(clean1200, legal)[None, :])[0]
            ),
            "cosine_a16_384": cosine(d_a16, d384),
            "cosine_a16_1200": cosine(d_a16, d1200),
            "dot_a16_384": float(np.dot(d_a16, d384)),
            "dot_a16_1200": float(np.dot(d_a16, d1200)),
            "norm_a16": float(np.linalg.norm(d_a16)),
            "norm_384": float(np.linalg.norm(d384)),
            "norm_1200": float(np.linalg.norm(d1200)),
            "a16_moves_toward_384_top": float(
                d_a16[legal.index(cached["clean_384"]["top1"])] > 0
            ),
            "a16_moves_toward_1200_top": float(
                d_a16[legal.index(cached["clean_1200"]["top1"])] > 0
            ),
        }
        records.append(record)

    def subset_summary(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(selected),
            "fraction": len(selected) / len(records),
            "improvement_clean384": descriptive(
                [r["improvement_clean384"] for r in selected]
            ),
            "improvement_clean1200": descriptive(
                [r["improvement_clean1200"] for r in selected]
            ),
            "cosine_a16_384": descriptive(
                [
                    r["cosine_a16_384"]
                    for r in selected
                    if r["cosine_a16_384"] is not None
                ]
            ),
            "cosine_a16_1200": descriptive(
                [
                    r["cosine_a16_1200"]
                    for r in selected
                    if r["cosine_a16_1200"] is not None
                ]
            ),
            "a16_moves_toward_384_top": descriptive(
                [r["a16_moves_toward_384_top"] for r in selected]
            ),
            "a16_moves_toward_1200_top": descriptive(
                [r["a16_moves_toward_1200_top"] for r in selected]
            ),
            "l1_a16_vs_p1": descriptive([r["l1_a16_vs_p1"] for r in selected]),
            "js_a16_vs_p1": descriptive([r["js_a16_vs_p1"] for r in selected]),
        }

    disagreement = [r for r in records if r["clean384_top1"] != r["clean1200_top1"]]
    median_margin = (
        float(np.median([r["clean1200_margin"] for r in disagreement]))
        if disagreement
        else float("inf")
    )
    confident = [r for r in disagreement if r["clean1200_margin"] >= median_margin]
    aggregate = {
        name: {
            metric: descriptive(
                [float(r["comparisons"][name][metric]) for r in records]
            )
            for metric in next(iter(records))["comparisons"][name]
        }
        for name in next(iter(records))["comparisons"]
    }
    early, late = (
        [r for r in records if r["phase"] == "early"],
        [r for r in records if r["phase"] == "late"],
    )
    clean_improvements = np.asarray([r["improvement_clean384"] for r in records])
    early_improvement = float(np.sum([r["improvement_clean384"] for r in early]))
    total_improvement = float(clean_improvements.sum())
    # Exact first training batch, with target searches cached separately from calibration states.
    train_manifest = json.loads(
        (args.adapter_workdir / "training_manifest.json").read_text()
    )
    paths = {
        name: Path(value) for name, value in train_manifest["artifact_paths"].items()
    }
    source, plan = (
        np.load(paths["train_source_indexes"], allow_pickle=False),
        np.load(paths["batch_indexes"], allow_pickle=False),
    )
    batch_source = [rows[int(i)] for i in source]
    batch_rows = [batch_source[int(i)] for i in plan[0] if int(i) >= 0]
    batch_indexes = [int(source[int(i)]) for i in plan[0] if int(i) >= 0]
    batch_targets = cache_targets(
        rows=rows,
        indexes=batch_indexes,
        evaluator=evaluator,
        path=args.workdir / "first_batch_same_state_p1_targets.jsonl",
    )
    bx = np.asarray([row["state"] for row in batch_rows], dtype=np.float32)
    bm = legal_mask_matrix_for_encoded_states(bx)
    bp = output(parent.state_dict(), bx, bm)
    stored_batch = np.asarray([row["policy"] for row in batch_rows], dtype=np.float32)
    clean384_batch = np.asarray(
        [batch_targets[index]["clean_384"]["policy"] for index in batch_indexes],
        dtype=np.float32,
    )
    clean1200_batch = np.asarray(
        [batch_targets[index]["clean_1200"]["policy"] for index in batch_indexes],
        dtype=np.float32,
    )
    gradients = {}
    for name, teacher in (
        ("stored", stored_batch),
        ("clean384", clean384_batch),
        ("clean1200", clean1200_batch),
    ):
        model = new_model(torch.device("cpu"))
        model.load_state_dict(parent.state_dict())
        apply_trainable_scope(model, "policy_adapter_only")
        gradients[name] = gradient_vector(
            model, bx, bm, ((1 - BETA) * teacher + BETA * bp).astype(np.float32)
        )
    gradient_summary = {
        "norms": {
            name: float(np.linalg.norm(value)) for name, value in gradients.items()
        },
        "cosines": {
            "stored_clean384": cosine(gradients["stored"], gradients["clean384"]),
            "clean384_clean1200": cosine(gradients["clean384"], gradients["clean1200"]),
            "stored_clean1200": cosine(gradients["stored"], gradients["clean1200"]),
        },
    }
    noise_l1, budget_l1 = (
        aggregate["stored_vs_clean384"]["l1"]["mean"],
        aggregate["clean384_vs_clean1200"]["l1"]["mean"],
    )
    budget_gap = (
        subset_summary(disagreement)["improvement_clean384"]["mean"]
        - subset_summary(disagreement)["improvement_clean1200"]["mean"]
        if disagreement
        else 0.0
    )
    if (
        noise_l1 > budget_l1 * 1.5
        and aggregate["stored_vs_clean384"]["top1_disagreement"]["mean"]
        > aggregate["clean384_vs_clean1200"]["top1_disagreement"]["mean"] * 1.5
        and aggregate["stored_vs_clean384"]["l1"]["mean"] > 0
    ):
        classification, next_experiment = (
            "noisy_teacher_component_dominant",
            "Matched adapter retraining on denoised 384-simulation targets.",
        )
    elif disagreement and len(disagreement) / SAMPLE_SIZE >= 0.02 and budget_gap > 0:
        classification, next_experiment = (
            "shallow_teacher_budget_conflict",
            "Retrain the exact PR #214 adapter using cached clean1200 P1 targets, with clean384 and stored384 matched controls.",
        )
    elif noise_l1 > 0 and budget_l1 > 0:
        classification, next_experiment = (
            "teacher_noise_and_budget_both_material",
            "Run a 3-lane target-factorization retrain: stored384, clean384, clean1200.",
        )
    else:
        classification, next_experiment = (
            "teacher_targets_broadly_aligned",
            "Return to policy supervision interactions with search/value estimates rather than generating stronger targets.",
        )
    probe = held_out_probe(
        evaluator,
        parent,
        a16,
        args.adapter_workdir / "artifacts/step_0016/artifact",
    )
    summary = {
        "schema": "azlite_pr214_teacher_quality_audit_v1",
        "guardrails": {
            "training": False,
            "optimizer_steps": False,
            "self_play": False,
            "arena_state_selection": False,
            "beta": BETA,
            "c_puct": 1.25,
        },
        "hashes": {
            "p1_checkpoint": sha256_file(p1_checkpoint),
            "a16_state": model_state_hash(a16.state_dict()),
            "replay": sha256_file(replay),
            "audit_manifest": sha256_file(manifest_path),
            "target_cache": sha256_file(args.workdir / "same_state_p1_targets.jsonl"),
        },
        "invariants": invariants,
        "sample": {
            "size": len(records),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "search_contract": {
            "simulations": [384, 1200],
            "c_puct": 1.25,
            "dirichlet_epsilon": 0,
            "fpu_mode": "zero",
            "root_policy_mode": "visit_count",
            "temperature": {
                "early": TEMPERATURE_EARLY,
                "late": TEMPERATURE_LATE,
                "threshold": TEMPERATURE_THRESHOLD,
            },
            "seed": "sha256(pr214-teacher-audit:encoded-state-hash)",
        },
        "teacher_disagreement": {
            "overall": aggregate,
            "by_phase": grouped_metrics(records, "phase"),
            "by_player": grouped_metrics(records, "player"),
            "by_move_index_bucket": grouped_metrics(records, "move_index_bucket"),
            "by_legal_move_count": grouped_metrics(records, "legal_move_count"),
            "by_stored_entropy_quartile": grouped_metrics(
                records, "stored_entropy_quartile"
            ),
        },
        "alignment": {
            "cosine_a16_384": descriptive(
                [
                    r["cosine_a16_384"]
                    for r in records
                    if r["cosine_a16_384"] is not None
                ]
            ),
            "cosine_a16_1200": descriptive(
                [
                    r["cosine_a16_1200"]
                    for r in records
                    if r["cosine_a16_1200"] is not None
                ]
            ),
            "dot_a16_384": descriptive([r["dot_a16_384"] for r in records]),
            "dot_a16_1200": descriptive([r["dot_a16_1200"] for r in records]),
            "improvement_stored": descriptive(
                [r["improvement_stored"] for r in records]
            ),
            "improvement_clean384": descriptive(
                [r["improvement_clean384"] for r in records]
            ),
            "improvement_clean1200": descriptive(
                [r["improvement_clean1200"] for r in records]
            ),
            "cross_entropy": {
                name: descriptive([r[name] for r in records])
                for name in (
                    "ce_p1_stored",
                    "ce_a16_stored",
                    "ce_p1_clean384",
                    "ce_a16_clean384",
                    "ce_p1_clean1200",
                    "ce_a16_clean1200",
                )
            },
            "norm_a16": descriptive([r["norm_a16"] for r in records]),
            "norm_384": descriptive([r["norm_384"] for r in records]),
            "norm_1200": descriptive([r["norm_1200"] for r in records]),
        },
        "critical_subsets": {
            "budget_disagreement_states": subset_summary(disagreement),
            "confident_1200_disagreement": {
                **subset_summary(confident),
                "margin_cutoff_median": median_margin,
            },
            "budget_agreement_states": subset_summary(
                [r for r in records if r not in disagreement]
            ),
        },
        "early_noise_attribution": {
            "early_stored_vs_clean384": {
                metric: descriptive(
                    [r["comparisons"]["stored_vs_clean384"][metric] for r in early]
                )
                for metric in aggregate["stored_vs_clean384"]
            },
            "late_stored_vs_clean384": {
                metric: descriptive(
                    [r["comparisons"]["stored_vs_clean384"][metric] for r in late]
                )
                for metric in aggregate["stored_vs_clean384"]
            },
            "early_fraction_of_clean384_ce_improvement": None
            if abs(total_improvement) <= 1e-12
            else early_improvement / total_improvement,
        },
        "initial_gradient": gradient_summary,
        "held_out_pr216_probe": probe,
        "classification": classification,
        "recommended_next_experiment": next_experiment,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(markdown(summary), encoding="utf-8")
    print(classification)


if __name__ == "__main__":
    main()
