#!/usr/bin/env python3
# ruff: noqa: E402
"""Localize frozen PR #251 target differences to opening or post-opening rows.

This consumes only the two aligned policy replay pairs.  It never generates
self-play or target MCTS, changes a training/search setting, or promotes a
model.  The analytic adapter attribution is checked before any optimizer step.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import (
    run_pr241_optimizer_isolation_reproduction as train_contract,
)
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr250_cross_seed_adapter_gradient_audit as pr250
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.kalah_rules import move_consequence_table
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import (
    _cross_entropy,
    incumbent_policy_batch,
    mixed_policy_target,
)
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import (
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
)

SUITE_SEEDS = {"G": 7042, "H": 8042, "I": 9042}
PHASES = {
    "opening_early": lambda ply: ply <= 4,
    "opening_late": lambda ply: 5 <= ply <= 9,
    "mid_early": lambda ply: 10 <= ply <= 19,
    "mid_late": lambda ply: 20 <= ply <= 29,
    "late": lambda ply: ply >= 30,
}
LANES = ("negative", "opening_positive", "postopening_positive", "positive")
EXPECTED_MODEL_SHA = pr251.EXPECTED_FULL_MODEL_SHA


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def model_sha(state: dict[str, torch.Tensor]) -> str:
    return pr251.model_sha(state)


def cosine(left: np.ndarray | torch.Tensor, right: np.ndarray | torch.Tensor) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    if torch.linalg.vector_norm(left) == 0 or torch.linalg.vector_norm(right) == 0:
        fail("zero comparison vector")
    return float(
        torch.dot(left, right)
        / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    )


def source_pairs() -> dict[str, dict[str, Any]]:
    return {
        "seed45": {
            "positive": pr250.LANES["seed45_fixed768_positive"],
            "negative": pr250.LANES["seed45_fixed1024_negative"],
        },
        "seed46": {
            "positive": pr250.LANES["seed46_fresh1024_positive"],
            "negative": pr250.LANES["seed46_fresh768_negative"],
        },
    }


def non_policy(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "policy"}


def verify_pair(
    seed: str, pair: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = {name: item["replay"] for name, item in pair.items()}
    for name, item in pair.items():
        actual = sha256_file(item["replay"])
        if actual != item["replay_sha"]:
            fail(f"{seed} {name} replay SHA {actual}")
    positive, negative = (read_jsonl(paths[name]) for name in ("positive", "negative"))
    if len(positive) != len(negative):
        fail(f"{seed} row count")
    for index, (pos, neg) in enumerate(zip(positive, negative, strict=True)):
        if non_policy(pos) != non_policy(neg):
            fail(f"{seed} non-policy row mismatch at {index}")
        legal = set(pos["legal_moves"])
        delta = np.asarray(pos["policy"], dtype=np.float64) - np.asarray(
            neg["policy"], dtype=np.float64
        )
        if np.any(delta[[action for action in range(6) if action not in legal]] != 0.0):
            fail(f"{seed} illegal action delta at {index}")
        if not np.isclose(delta.sum(), 0.0, atol=1e-7):
            fail(f"{seed} policy delta sum at {index}")
    blocked, groups = replay.exclusion_hashes(pr250.CANONICAL_SUITE)
    pos_eligible, pos_exclusions = replay.filter_rows(positive, blocked, groups)
    neg_eligible, neg_exclusions = replay.filter_rows(negative, blocked, groups)
    if len(pos_eligible) != len(neg_eligible) or pos_exclusions != neg_exclusions:
        fail(f"{seed} exclusion mismatch")
    for index, (pos, neg) in enumerate(zip(pos_eligible, neg_eligible, strict=True)):
        if non_policy(pos) != non_policy(neg):
            fail(f"{seed} eligible row mismatch at {index}")
    return (
        pos_eligible,
        neg_eligible,
        {
            "positive_replay_sha256": sha256_file(paths["positive"]),
            "negative_replay_sha256": sha256_file(paths["negative"]),
            "raw_rows": len(positive),
            "eligible_rows": len(pos_eligible),
            "exclusions": pos_exclusions,
        },
    )


def mixed_targets(
    rows: list[dict[str, Any]], parent: dict[str, torch.Tensor]
) -> np.ndarray:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(parent)
    model.eval()
    x = torch.tensor(np.asarray([row["state"] for row in rows], dtype=np.float32))
    masks = legal_mask_matrix_for_encoded_states(x.numpy())
    with torch.no_grad():
        parent_policy = incumbent_policy_batch(
            model, {"x": x, "mask": torch.tensor(masks)}
        )
    search = torch.tensor(np.asarray([row["policy"] for row in rows], dtype=np.float32))
    return mixed_policy_target(
        search, parent_policy, torch.tensor(masks), replay.BETA
    ).numpy()


def full_gradient(
    rows: list[dict[str, Any]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> torch.Tensor:
    return pr250.full_batch_gradient(rows, a16, parent)["vector"]


def phase_name(ply: int) -> str:
    return next(name for name, includes in PHASES.items() if includes(ply))


def score_bucket(value: int) -> str:
    return (
        "<0"
        if value < 0
        else "=0"
        if value == 0
        else "1"
        if value == 1
        else "2..4"
        if value <= 4
        else ">=5"
    )


def seed_bucket(value: int) -> str:
    return (
        "1" if value == 1 else "2..3" if value <= 3 else "4..6" if value <= 6 else ">=7"
    )


def attribution(
    seed: str,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor]:
    pos_target, neg_target = (
        mixed_targets(positive, parent),
        mixed_targets(negative, parent),
    )
    delta_search = np.asarray(
        [row["policy"] for row in positive], dtype=np.float64
    ) - np.asarray([row["policy"] for row in negative], dtype=np.float64)
    delta_train = pos_target.astype(np.float64) - neg_target.astype(np.float64)
    # Targets are constructed in float32 by the frozen trainer; one float32 ULP
    # is the deterministic tolerance for the explicitly reconstructed identity.
    if not np.allclose(delta_train, 0.05 * delta_search, rtol=0, atol=3e-7):
        fail(f"{seed} beta target delta")
    model = new_model(torch.device("cpu"))
    model.load_state_dict(a16)
    model.eval()
    with torch.no_grad():
        features = model.trunk_features(
            torch.tensor(
                np.asarray([row["state"] for row in positive], dtype=np.float32)
            )
        ).double()
    weight = (
        torch.tensor(delta_train, dtype=torch.float64)[:, :, None]
        * features[:, None, :]
    )
    bias = torch.tensor(delta_train, dtype=torch.float64)
    # PR #250's "full batch" is the concatenated frozen training-batch plan;
    # held-out rows remain reportable but do not participate in an optimizer step.
    batch_indexes = np.concatenate(replay.batches(positive))
    reconstruction = torch.cat(
        (
            weight[batch_indexes].sum(dim=0).reshape(-1),
            bias[batch_indexes].sum(dim=0).reshape(-1),
        )
    ) / len(batch_indexes)
    dg = (
        full_gradient(positive, a16, parent).double()
        - full_gradient(negative, a16, parent).double()
    )
    error = torch.linalg.vector_norm(reconstruction + dg) / torch.linalg.vector_norm(dg)
    similarity = cosine(reconstruction, -dg)
    if similarity < 0.999999 or error > 1e-5:
        fail(
            f"{seed} analytic gradient reconstruction cosine={similarity} error={error}"
        )
    residual = torch.cat(
        [
            (
                torch.load(
                    source_pairs()[seed]["positive"]["checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )["model"][key]
                - torch.load(
                    source_pairs()[seed]["negative"]["checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )["model"][key]
            )
            .double()
            .reshape(-1)
            for key in ADAPTER_KEYS
        ]
    )
    direction = -dg / torch.linalg.vector_norm(dg)
    residual_direction = residual / torch.linalg.vector_norm(residual)
    # The flattening order is weight then bias, matching ADAPTER_KEYS exactly.
    score_grad = (
        torch.einsum(
            "nad,ad->na",
            weight,
            direction[: weight.shape[1] * weight.shape[2]].reshape_as(weight[0]),
        )
        + bias * direction[-6:]
    )
    score_residual = (
        torch.einsum(
            "nad,ad->na",
            weight,
            residual_direction[: weight.shape[1] * weight.shape[2]].reshape_as(
                weight[0]
            ),
        )
        + bias * residual_direction[-6:]
    )
    row_scores = score_grad.sum(dim=1).numpy()
    abs_rows = np.abs(row_scores)
    concentration = {}
    for fraction in (0.01, 0.05, 0.10, 0.20, 0.50):
        count = max(1, int(np.ceil(len(abs_rows) * fraction)))
        concentration[f"top_{int(fraction * 100)}pct"] = (
            float(np.sort(abs_rows)[-count:].sum() / abs_rows.sum())
            if abs_rows.sum()
            else 0.0
        )
    summary = {
        "gradient_reconstruction": {
            "cosine": similarity,
            "relative_l2_error": float(error),
            "rows": len(positive),
            "train_rows": len(batch_indexes),
        },
        "residual_vs_negative_gradient_cosine": cosine(residual, -dg),
        "concentration": {
            "fraction_total_absolute_by_top_rows": concentration,
            "positive_attribution_mass": float(score_grad[score_grad > 0].sum()),
            "negative_attribution_mass": float(-score_grad[score_grad < 0].sum()),
            "cancellation_ratio": float(abs(score_grad.sum()) / score_grad.abs().sum()),
            "net_attribution": float(score_grad.sum()),
        },
    }
    return (
        {
            **summary,
            "_scores": score_grad.numpy(),
            "_residual_scores": score_residual.numpy(),
            "_delta_search": delta_search,
        },
        dg,
    )


def semantic_report(rows: list[dict[str, Any]], data: dict[str, Any]) -> dict[str, Any]:
    scores, residual_scores, delta_search = (
        data["_scores"],
        data["_residual_scores"],
        data["_delta_search"],
    )
    groups: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    phase_rows: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows):
        state = decode_kalah_v3_base_state(list(row["state"]))
        encoded = np.asarray(
            encode_state(state, input_encoding="kalah_v3"), dtype=np.float32
        )
        if not np.allclose(
            encoded, np.asarray(row["state"], dtype=np.float32), atol=1e-6
        ):
            fail(f"state decode/encode mismatch at row {index}")
        phase = phase_name(int(row["move_index"]))
        row_l1 = float(np.abs(delta_search[index]).sum())
        phase_rows[phase]["rows"] += 1
        phase_rows[phase]["target_delta_l1"] += row_l1
        phase_rows[phase]["absolute_gradient_attribution"] += float(
            np.abs(scores[index]).sum()
        )
        phase_rows[phase]["signed_gradient_attribution"] += float(scores[index].sum())
        for action in row["legal_moves"]:
            consequence = move_consequence_table(state)[int(action)]
            labels = {
                "pit_index": str(action),
                "extra_turn": str(bool(consequence["gives_extra_turn"])),
                "capture": str(bool(consequence["produces_capture"])),
                "extra_turn_capture": f"extra={bool(consequence['gives_extra_turn'])},capture={bool(consequence['produces_capture'])}",
                "game_over": str(bool(consequence["game_over_after_move"])),
                "immediate_score": score_bucket(
                    int(consequence["immediate_score_delta"])
                ),
                "seed_count": seed_bucket(int(consequence["seed_count"])),
                "legal_count": str(len(row["legal_moves"])),
                "current_player": str(row["player"]),
                "phase": phase,
            }
            for family, label in labels.items():
                groups[family][label]["gradient"] += float(scores[index, action])
                groups[family][label]["residual"] += float(
                    residual_scores[index, action]
                )
                groups[family][label]["toward_mass"] += float(
                    max(delta_search[index, action], 0.0)
                )
                groups[family][label]["away_mass"] += float(
                    min(delta_search[index, action], 0.0)
                )
    for name in PHASES:
        phase_rows[name]
    total_abs = sum(
        value["absolute_gradient_attribution"] for value in phase_rows.values()
    )
    positive = sum(
        max(value["signed_gradient_attribution"], 0.0) for value in phase_rows.values()
    )
    phase = {
        key: {
            **dict(value),
            "fraction_total_absolute_attribution": value[
                "absolute_gradient_attribution"
            ]
            / total_abs
            if total_abs
            else 0.0,
            "fraction_total_positive_attribution": max(
                value["signed_gradient_attribution"], 0.0
            )
            / positive
            if positive
            else 0.0,
        }
        for key, value in phase_rows.items()
    }
    opening: dict[str, Counter[str]] = defaultdict(Counter)
    for name, value in phase.items():
        target = opening["opening" if name.startswith("opening") else "postopening"]
        for key in (
            "rows",
            "target_delta_l1",
            "absolute_gradient_attribution",
            "signed_gradient_attribution",
        ):
            target[key] += value[key]
    movement_total = sum(value["toward_mass"] for value in groups["pit_index"].values())
    semantics = {
        family: {
            label: {
                **dict(value),
                "toward_mass_normalized": value["toward_mass"] / movement_total
                if movement_total
                else 0.0,
                "away_mass_normalized": value["away_mass"] / movement_total
                if movement_total
                else 0.0,
            }
            for label, value in values.items()
        }
        for family, values in groups.items()
    }
    return {
        "phase_buckets": phase,
        "opening_vs_postopening": {key: dict(value) for key, value in opening.items()},
        "action_semantics": semantics,
    }


def semantic_alignment(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for family in (
        "phase",
        "pit_index",
        "extra_turn_capture",
        "immediate_score",
        "legal_count",
    ):
        labels = sorted(
            set(left["action_semantics"].get(family, {}))
            | set(right["action_semantics"].get(family, {}))
        )
        a = np.asarray(
            [
                left["action_semantics"]
                .get(family, {})
                .get(label, {})
                .get("gradient", 0.0)
                for label in labels
            ]
        )
        b = np.asarray(
            [
                right["action_semantics"]
                .get(family, {})
                .get(label, {})
                .get("gradient", 0.0)
                for label in labels
            ]
        )
        a /= np.abs(a).sum() or 1.0
        b /= np.abs(b).sum() or 1.0
        ranks_a = np.argsort(np.argsort(a)).astype(float)
        ranks_b = np.argsort(np.argsort(b)).astype(float)
        top_a = set(np.asarray(labels)[np.argsort(np.abs(a))[-min(3, len(labels)) :]])
        top_b = set(np.asarray(labels)[np.argsort(np.abs(b))[-min(3, len(labels)) :]])
        result[family] = {
            "bins": labels,
            "cosine": cosine(a, b),
            "spearman": cosine(ranks_a - ranks_a.mean(), ranks_b - ranks_b.mean()),
            "top3_overlap": sorted(top_a & top_b),
        }
    return result


def lane_views(
    positive: list[dict[str, Any]], negative: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    views = {name: [] for name in LANES}
    for pos, neg in zip(positive, negative, strict=True):
        opening = int(pos["move_index"]) < 10
        selected = {
            "negative": neg,
            "positive": pos,
            "opening_positive": pos if opening else neg,
            "postopening_positive": neg if opening else pos,
        }
        for name, row in selected.items():
            views[name].append(copy.deepcopy(row))
    for name, rows in views.items():
        for index, row in enumerate(rows):
            if non_policy(row) != non_policy(negative[index]):
                fail(f"{name} non-policy mutation")
    for index, (opening, post, pos, neg) in enumerate(
        zip(
            views["opening_positive"],
            views["postopening_positive"],
            positive,
            negative,
            strict=True,
        )
    ):
        if not np.array_equal(
            np.asarray(opening["policy"]) + np.asarray(post["policy"]),
            np.asarray(pos["policy"]) + np.asarray(neg["policy"]),
        ):
            fail(f"phase partition at row {index}")
    return views


def train_views(
    seed: str,
    views: dict[
        str,
        list[dict[str, Any]],
    ],
    a16: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[
    dict[str, Any], dict[str, Path], dict[str, dict[int, dict[str, torch.Tensor]]]
]:
    result, artifacts, states = {}, {}, {}
    initial_sha = replay.optimizer_state_sha256(optimizer)
    targets = {
        name: np.asarray([row["policy"] for row in rows], dtype=np.float64)
        for name, rows in views.items()
    }
    for name in LANES:
        snapshots, optimizers, invocation = train_contract.run_lane(
            views[name], a16, optimizer, parent
        )
        if name in ("negative", "positive"):
            actual = model_sha(snapshots[16])
            expected = EXPECTED_MODEL_SHA[f"{seed}_{name}"]
            if actual != expected:
                fail(f"{seed} {name} checkpoint reproduction {actual}")
        for step in replay.STEPS:
            path = workdir / "train" / seed / name / f"step_{step:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": snapshots[step], "optimizer": optimizers[step]}, path)
        artifacts[name] = export(
            snapshots[16],
            workdir / "train" / seed / name / "step_0016",
            f"pr252_{seed}_{name}",
        )
        base_metrics = replay.metrics(
            views[name], snapshots, parent, a16, copy.deepcopy(optimizer)
        )
        metrics = {}
        for step, state in snapshots.items():
            candidate = replay.policy(state, views[name])
            own = targets[name]
            metrics[str(step)] = {
                **base_metrics[str(step)],
                "ce_own_target": float(np.mean(_cross_entropy(candidate, own))),
                "ce_positive_target": float(
                    np.mean(_cross_entropy(candidate, targets["positive"]))
                ),
                "ce_negative_target": float(
                    np.mean(_cross_entropy(candidate, targets["negative"]))
                ),
                "model_sha256": model_sha(state),
                "optimizer_sha256": replay.optimizer_state_sha256(optimizers[step]),
            }
        result[name] = {"optimizer_invocation": invocation, "metrics": metrics}
        states[name] = snapshots
    if replay.optimizer_state_sha256(optimizer) != initial_sha:
        fail(f"{seed} optimizer contamination")

    def delta(name: str) -> torch.Tensor:
        return torch.cat(
            [(states[name][16][key] - a16[key]).reshape(-1) for key in ADAPTER_KEYS]
        )

    for name in ("opening_positive", "postopening_positive"):
        result[name]["adapter_delta_cosine"] = {
            "positive": cosine(delta(name), delta("positive")),
            "negative": cosine(delta(name), delta("negative")),
        }
    return result, artifacts, states


def seal_suites(workdir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    old_paths = {
        "canonical": pr249.CANONICAL_SUITE,
        **pr251.consumed_suite_paths(workdir),
    }
    old = {name: suites.load_suite_jsonl(str(path)) for name, path in old_paths.items()}
    used = set().union(*(pr249.suite_keys(entries) for entries in old.values()))
    training = set().union(*pr249.replay_states().values())
    universe = [
        entry
        for entry in pr249.all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in training
    ]
    paths, manifest = (
        {},
        {
            "consumed": {name: sha256_file(path) for name, path in old_paths.items()},
            "suites": {},
        },
    )
    for label, seed in SUITE_SEEDS.items():
        selected = suites.select_diverse(
            [
                entry
                for entry in universe
                if suites.canonical_key(entry["state"]) not in used
            ],
            128,
            seed,
        )
        keys = pr249.suite_keys(selected)
        if len(keys) != 128 or keys & used:
            fail(f"suite overlap {label}")
        path = workdir / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        manifest["suites"][label] = {
            "seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
        used |= keys
    return paths, manifest


def cached(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    return result if isinstance(result, list) and len(result) == 512 else None


def evaluate(
    artifacts: dict[str, dict[str, Path]],
    paths: dict[str, Path],
    workdir: Path,
    context: str,
) -> dict[str, Any]:
    result = {}
    for label, suite in paths.items():
        control_path = workdir / "records" / context / label / "p1_control.json"
        control = cached(control_path) or __import__(
            "ml.alphazero_lite.run_pr241_policy_target_noise_isolation",
            fromlist=["arena_records"],
        ).arena_records(
            workdir / "arena" / context / label,
            pr249.P1_ARTIFACT,
            pr249.P1_ARTIFACT,
            context,
            "p1_control",
            24,
            suite,
        )
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(
            json.dumps(control, sort_keys=True) + "\n", encoding="utf-8"
        )
        result[label] = {"candidates": {}}
        for seed, lanes in artifacts.items():
            for name, artifact in lanes.items():
                record_path = (
                    workdir / "records" / context / label / f"{seed}_{name}.json"
                )
                records = cached(record_path) or __import__(
                    "ml.alphazero_lite.run_pr241_policy_target_noise_isolation",
                    fromlist=["arena_records"],
                ).arena_records(
                    workdir / "arena" / context / label,
                    artifact,
                    pr249.P1_ARTIFACT,
                    context,
                    f"{seed}_{name}",
                    24,
                    suite,
                )
                record_path.write_text(
                    json.dumps(records, sort_keys=True) + "\n", encoding="utf-8"
                )
                result[label]["candidates"][f"{seed}_{name}"] = {
                    "effect": paired_opening_candidate_effect(records, control),
                    "records": records,
                }
    return result


def analyze(evaluation: dict[str, Any]) -> dict[str, Any]:
    contrasts = {}
    for seed in ("seed45", "seed46"):
        for name, pair in {
            "positive_minus_negative": ("positive", "negative"),
            "opening_positive_minus_negative": ("opening_positive", "negative"),
            "postopening_positive_minus_negative": ("postopening_positive", "negative"),
            "opening_positive_minus_positive": ("opening_positive", "positive"),
            "postopening_positive_minus_positive": ("postopening_positive", "positive"),
        }.items():
            suites_values, per_suite = [], {}
            for label, values in evaluation.items():
                diff = paired_effect_difference(
                    values["candidates"][f"{seed}_{pair[0]}"]["effect"],
                    values["candidates"][f"{seed}_{pair[1]}"]["effect"],
                )
                sample = np.asarray(
                    list(diff["per_opening_effect"].values()), dtype=float
                )
                suites_values.append(sample)
                per_suite[label] = float(sample.mean())
            values = np.concatenate(suites_values)
            draws = values[
                np.random.default_rng(42).integers(
                    0, len(values), (10_000, len(values))
                )
            ].mean(axis=1)
            contrasts[f"{seed}_{name}"] = {
                "per_suite": per_suite,
                "pooled": {
                    "effect": float(values.mean()),
                    "lower_95": float(np.quantile(draws, 0.025)),
                    "upper_95": float(np.quantile(draws, 0.975)),
                    "positive_suites": sum(value > 0 for value in per_suite.values()),
                },
            }
    return {"contrasts": contrasts}


def divergence(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    left_by_key = {pr249.record_key(row): row for row in left}
    right_by_key = {pr249.record_key(row): row for row in right}
    if set(left_by_key) != set(right_by_key):
        fail("arena record key mismatch")
    plies, seats, divergent, changed = Counter(), Counter(), 0, 0
    for key, row in left_by_key.items():
        other = right_by_key[key]
        moves, other_moves = (
            row["trajectory"].split(","),
            other["trajectory"].split(","),
        )
        first = next(
            (
                index
                for index, pair in enumerate(zip(moves, other_moves))
                if pair[0] != pair[1]
            ),
            None,
        )
        if first is not None or len(moves) != len(other_moves):
            divergent += 1
            plies[
                str(first if first is not None else min(len(moves), len(other_moves)))
            ] += 1
            seats[str(row["challenger_player"])] += 1
        changed += row["winner"] != other["winner"]
    return {
        "fraction_games_diverging": divergent / len(left_by_key),
        "first_divergent_ply": dict(
            sorted(plies.items(), key=lambda item: int(item[0]))
        ),
        "seat": dict(sorted(seats.items())),
        "outcome_change_fraction": changed / len(left_by_key),
    }


def telemetry(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        suite: {
            f"{seed}_{lane}_versus_negative": divergence(
                values["candidates"][f"{seed}_{lane}"]["records"],
                values["candidates"][f"{seed}_negative"]["records"],
            )
            for seed in ("seed45", "seed46")
            for lane in ("opening_positive", "postopening_positive")
        }
        for suite, values in evaluation.items()
    }


def classify(analysis: dict[str, Any], alignment: dict[str, Any]) -> str:
    c = analysis["contrasts"]

    def passed(key: str) -> bool:
        return (
            c[key]["pooled"]["lower_95"] > 0
            and c[key]["pooled"]["positive_suites"] >= 2
        )

    controls = all(
        passed(f"{seed}_positive_minus_negative") for seed in ("seed45", "seed46")
    )
    if not controls:
        return "fresh_strength_signal_fails_again"
    opening = all(
        passed(f"{seed}_opening_positive_minus_negative")
        and abs(c[f"{seed}_opening_positive_minus_positive"]["pooled"]["effect"])
        <= 0.02
        and c[f"{seed}_opening_positive_minus_positive"]["pooled"]["lower_95"]
        <= 0
        <= c[f"{seed}_opening_positive_minus_positive"]["pooled"]["upper_95"]
        for seed in ("seed45", "seed46")
    )
    post = all(
        passed(f"{seed}_postopening_positive_minus_negative")
        and abs(c[f"{seed}_postopening_positive_minus_positive"]["pooled"]["effect"])
        <= 0.02
        and c[f"{seed}_postopening_positive_minus_positive"]["pooled"]["lower_95"]
        <= 0
        <= c[f"{seed}_postopening_positive_minus_positive"]["pooled"]["upper_95"]
        for seed in ("seed45", "seed46")
    )
    if opening and not post:
        return "opening_target_delta_is_causal"
    if post and not opening:
        return "postopening_target_delta_is_causal"
    if any(
        passed(f"{seed}_opening_positive_minus_negative")
        for seed in ("seed45", "seed46")
    ) != any(
        passed(f"{seed}_postopening_positive_minus_negative")
        for seed in ("seed45", "seed46")
    ):
        return "phase_causality_is_seed_specific"
    if all(
        item["cosine"] >= 0.9 and item["spearman"] >= 0.8 for item in alignment.values()
    ):
        return "target_delta_semantics_align_without_phase_sufficiency"
    return "distributed_phase_target_delta_required"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr252_phase_target_delta_attribution"),
    )
    parser.add_argument("--freeze-suites-only", action="store_true")
    parser.add_argument("--skip-secondary", action="store_true")
    args = parser.parse_args()
    if (
        sha256_file(pr250.A16)
        != "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
    ):
        fail("A16 SHA")
    snapshot = torch.load(pr250.A16, map_location="cpu", weights_only=False)
    a16, optimizer = replay.immutable_initial_state(snapshot)
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone() for key, value in p1.state_dict().items()
    }
    args.workdir.mkdir(parents=True, exist_ok=True)
    paths, suite_manifest = seal_suites(args.workdir)
    frozen = {
        "schema": "alphazero_lite_pr252_phase_target_delta_attribution_v1",
        "suite_manifest": suite_manifest,
        "guardrails": {
            "self_play": False,
            "target_mcts": False,
            "target_budgets_changed": False,
            "beta": replay.BETA,
            "lr": 1e-5,
            "batch_size": 512,
            "grad_clip": 1.0,
            "steps": list(replay.STEPS),
            "phase_boundary": "move_index < 10",
            "primary_context": "1200:1200",
            "secondary_context": "384:256",
        },
    }
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_suites_only:
        return
    pairs, attribution_reports, semantic_reports, artifacts, training = (
        {},
        {},
        {},
        {},
        {},
    )
    for seed, pair in source_pairs().items():
        positive, negative, pair_report = verify_pair(seed, pair)
        pairs[seed] = pair_report
        attributed, _dg = attribution(seed, positive, negative, a16, parent)
        semantic = semantic_report(positive, attributed)
        attribution_reports[seed] = {
            key: value for key, value in attributed.items() if not key.startswith("_")
        }
        semantic_reports[seed] = semantic
        views = lane_views(positive, negative)
        training[seed], artifacts[seed], _states = train_views(
            seed, views, a16, optimizer, parent, args.workdir
        )
    primary = evaluate(artifacts, paths, args.workdir, "1200:1200")
    primary_analysis = analyze(primary)
    alignment = semantic_alignment(
        semantic_reports["seed45"], semantic_reports["seed46"]
    )
    result = {
        **frozen,
        "pairs": pairs,
        "attribution": attribution_reports,
        "semantic_attribution": semantic_reports,
        "cross_seed_semantic_alignment": alignment,
        "training": training,
        "primary_analysis": primary_analysis,
        "first_divergence_telemetry": telemetry(primary),
        "classification": classify(primary_analysis, alignment),
    }
    # Freeze the primary result before the preregistered secondary context runs.
    (args.workdir / "primary_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.skip_secondary:
        secondary = evaluate(artifacts, paths, args.workdir, "384:256")
        result["secondary_analysis"] = analyze(secondary)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
