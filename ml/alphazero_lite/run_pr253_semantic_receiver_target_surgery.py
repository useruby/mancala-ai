#!/usr/bin/env python3
# ruff: noqa: E402
"""Causally partition frozen positive target mass by its immediate destination.

This sealed continuation of PR #252 consumes its aligned replay pairs, trains
only the established policy adapter continuation, and evaluates only fresh
J/K/L suites. It does not generate self-play or run target MCTS.
"""

from __future__ import annotations

import argparse
import copy
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

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr250_cross_seed_adapter_gradient_audit as pr250
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite import run_pr252_phase_target_delta_attribution as pr252
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite import (
    run_pr241_optimizer_isolation_reproduction as train_contract,
)
from ml.alphazero_lite.evaluation_metrics import paired_effect_difference
from ml.alphazero_lite.kalah_rules import move_consequence_table
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file
from ml.alphazero_lite.run_fresh_p1_adapter_matched_q_feedback import (
    decode_kalah_v3_base_state,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import _cross_entropy
from ml.alphazero_lite.self_play import encode_state
from ml.alphazero_lite.train import load_checkpoint_into_model


SUITE_SEEDS = {"J": 10042, "K": 11042, "L": 12042}
LANES = ("negative", "ordinary_positive", "tactical_positive", "positive")
ZERO_TOLERANCE = 1e-12
FLOAT_TOLERANCE = 1e-7
GRADIENT_TOLERANCE = 1e-5
CONSUMED_SUITE_PATHS = {
    "A": Path("/tmp/azlite_pr249_fresh_suite_generalization/suites/suite_A.jsonl"),
    "B": Path("/tmp/azlite_pr249_fresh_suite_generalization/suites/suite_B.jsonl"),
    "C": Path("/tmp/azlite_pr249_fresh_suite_generalization/suites/suite_C.jsonl"),
    "D": Path(
        "/tmp/azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_D.jsonl"
    ),
    "E": Path(
        "/tmp/azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_E.jsonl"
    ),
    "F": Path(
        "/tmp/azlite_pr251_cross_seed_strength_residual_transfer/suites/suite_F.jsonl"
    ),
    "G": Path("/tmp/azlite_pr252_phase_target_delta_attribution/suites/suite_G.jsonl"),
    "H": Path("/tmp/azlite_pr252_phase_target_delta_attribution/suites/suite_H.jsonl"),
    "I": Path("/tmp/azlite_pr252_phase_target_delta_attribution/suites/suite_I.jsonl"),
}


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def ordinary_actions(row: dict[str, Any], index: int) -> set[int]:
    state = decode_kalah_v3_base_state(list(row["state"]))
    if not np.allclose(
        encode_state(state, input_encoding="kalah_v3"), row["state"], atol=1e-6
    ):
        fail(f"semantic decode at row {index}")
    consequences = move_consequence_table(state)
    return {
        int(action)
        for action in row["legal_moves"]
        if not consequences[int(action)]["gives_extra_turn"]
        and not consequences[int(action)]["produces_capture"]
    }


def hybrid_targets(
    positive: list[dict[str, Any]], negative: list[dict[str, Any]], seed: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    views = {"negative": copy.deepcopy(negative), "positive": copy.deepcopy(positive)}
    views["ordinary_positive"], views["tactical_positive"] = [], []
    alphas, received = [], defaultdict(float)
    for index, (pos, neg) in enumerate(zip(positive, negative, strict=True)):
        p, q = (np.asarray(row["policy"], dtype=np.float64) for row in (pos, neg))
        legal = set(pos["legal_moves"])
        d = p - q
        if not np.isclose(d.sum(), 0.0, rtol=0, atol=FLOAT_TOLERANCE):
            fail(f"{seed} delta sum at row {index}")
        if any(abs(d[action]) > ZERO_TOLERANCE for action in set(range(6)) - legal):
            fail(f"{seed} illegal delta at row {index}")
        d[np.abs(d) <= ZERO_TOLERANCE] = 0.0
        ordinary = ordinary_actions(pos, index)
        receiver = d > 0
        moved = float(d[receiver].sum())
        alpha = (
            float(d[[action for action in ordinary if d[action] > 0]].sum() / moved)
            if moved
            else 0.0
        )
        if not 0 <= alpha <= 1:
            fail(f"{seed} alpha range at row {index}")
        if moved:
            alphas.append(alpha)
        state = decode_kalah_v3_base_state(list(pos["state"]))
        consequences = move_consequence_table(state)
        for action in legal:
            if d[action] > 0:
                consequence = consequences[action]
                label = (
                    "ordinary"
                    if action in ordinary
                    else "capture_extra_turn"
                    if consequence["produces_capture"]
                    and consequence["gives_extra_turn"]
                    else "capture_only"
                    if consequence["produces_capture"]
                    else "extra_turn_only"
                )
                received[label] += float(d[action])
        ordinary_target, tactical_target = q.copy(), q.copy()
        for action in legal:
            if d[action] > 0:
                if action in ordinary:
                    ordinary_target[action] = p[action]
                else:
                    tactical_target[action] = p[action]
            elif d[action] < 0:
                ordinary_target[action] = q[action] + alpha * d[action]
                tactical_target[action] = q[action] + (1 - alpha) * d[action]
        for name, target in (
            ("ordinary", ordinary_target),
            ("tactical", tactical_target),
        ):
            if not np.isclose(target.sum(), 1.0, rtol=0, atol=FLOAT_TOLERANCE):
                fail(f"{seed} {name} normalization at row {index}")
            if np.any(target < -FLOAT_TOLERANCE) or any(
                abs(target[action]) > FLOAT_TOLERANCE
                for action in set(range(6)) - legal
            ):
                fail(f"{seed} {name} distribution at row {index}")
        if not np.allclose(
            ordinary_target + tactical_target, p + q, rtol=0, atol=FLOAT_TOLERANCE
        ):
            fail(f"{seed} decomposition at row {index}")
        if alpha == 1 and (
            not np.allclose(ordinary_target, p) or not np.allclose(tactical_target, q)
        ):
            fail(f"{seed} alpha one at row {index}")
        if alpha == 0 and (
            not np.allclose(ordinary_target, q) or not np.allclose(tactical_target, p)
        ):
            fail(f"{seed} alpha zero at row {index}")
        for name, target in (
            ("ordinary_positive", ordinary_target),
            ("tactical_positive", tactical_target),
        ):
            row = copy.deepcopy(neg)
            row["policy"] = target.tolist()
            views[name].append(row)
    total = sum(received.values())
    alpha_array = np.asarray(alphas)
    if not len(alpha_array):
        fail(f"{seed} has no moved target mass")
    return views, {
        "alpha_distribution": {
            "mean": float(alpha_array.mean()),
            **{
                f"p{pct}": float(np.percentile(alpha_array, pct))
                for pct in (10, 25, 50, 75, 90)
            },
            "fraction_alpha_zero": float(np.mean(alpha_array == 0)),
            "fraction_alpha_one": float(np.mean(alpha_array == 1)),
        },
        "receiver_mass_fraction": {
            "ordinary": received["ordinary"] / total if total else 0.0,
            "tactical": (total - received["ordinary"]) / total if total else 0.0,
            "capture_only": received["capture_only"] / total if total else 0.0,
            "extra_turn_only": received["extra_turn_only"] / total if total else 0.0,
            "capture_extra_turn": received["capture_extra_turn"] / total
            if total
            else 0.0,
        },
    }


def gradient_report(
    views: dict[str, list[dict[str, Any]]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    seed: str,
) -> dict[str, float]:
    gradients = {
        name: pr250.full_batch_gradient(rows, a16, parent)["vector"].double()
        for name, rows in views.items()
    }
    ordinary = gradients["ordinary_positive"] - gradients["negative"]
    tactical = gradients["tactical_positive"] - gradients["negative"]
    full = gradients["positive"] - gradients["negative"]
    error = torch.linalg.vector_norm(
        ordinary + tactical - full
    ) / torch.linalg.vector_norm(full)
    if error > GRADIENT_TOLERANCE:
        fail(f"{seed} gradient decomposition {error}")
    return {
        "relative_l2_error": float(error),
        "norm_fraction_ordinary_full": float(
            torch.linalg.vector_norm(ordinary) / torch.linalg.vector_norm(full)
        ),
        "norm_fraction_tactical_full": float(
            torch.linalg.vector_norm(tactical) / torch.linalg.vector_norm(full)
        ),
        "cosine_ordinary_full": pr252.cosine(ordinary, full),
        "cosine_tactical_full": pr252.cosine(tactical, full),
        "cosine_ordinary_tactical": pr252.cosine(ordinary, tactical),
    }


def train_views(
    seed: str,
    views: dict[str, list[dict[str, Any]]],
    a16: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    targets = {
        name: np.asarray([row["policy"] for row in rows], dtype=np.float64)
        for name, rows in views.items()
    }
    initial_sha, result, artifacts, states = (
        replay.optimizer_state_sha256(optimizer),
        {},
        {},
        {},
    )
    for name in LANES:
        snapshots, optimizers, invocation = train_contract.run_lane(
            views[name], a16, optimizer, parent
        )
        if name in ("negative", "positive"):
            actual = pr251.model_sha(snapshots[16])
            if actual != pr251.EXPECTED_FULL_MODEL_SHA[f"{seed}_{name}"]:
                fail(f"{seed} {name} checkpoint reproduction {actual}")
        base = replay.metrics(
            views[name], snapshots, parent, a16, copy.deepcopy(optimizer)
        )
        metrics = {}
        for step, state in snapshots.items():
            candidate = replay.policy(state, views[name])
            metrics[str(step)] = {
                **base[str(step)],
                **{
                    f"ce_{target}_target": float(
                        np.mean(_cross_entropy(candidate, values))
                    )
                    for target, values in targets.items()
                },
                "model_sha256": pr251.model_sha(state),
                "optimizer_sha256": replay.optimizer_state_sha256(optimizers[step]),
            }
        result[name] = {"optimizer_invocation": invocation, "metrics": metrics}
        states[name] = snapshots
        artifacts[name] = export(
            snapshots[16],
            workdir / "train" / seed / name / "step_0016",
            f"pr253_{seed}_{name}",
        )
    if replay.optimizer_state_sha256(optimizer) != initial_sha:
        fail(f"{seed} optimizer contamination")

    def delta(name: str) -> torch.Tensor:
        return torch.cat(
            [(states[name][16][key] - a16[key]).reshape(-1) for key in ADAPTER_KEYS]
        )

    for name in LANES:
        result[name]["adapter_delta_cosine"] = {
            "negative": pr252.cosine(delta(name), delta("negative")),
            "positive": pr252.cosine(delta(name), delta("positive")),
        }
    return result, artifacts


def seal_suites(workdir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    old_paths = {"canonical": pr249.CANONICAL_SUITE, **CONSUMED_SUITE_PATHS}
    if any(not path.is_file() for path in old_paths.values()):
        fail("missing consumed suite")
    old = {name: suites.load_suite_jsonl(str(path)) for name, path in old_paths.items()}
    used = set().union(*(pr249.suite_keys(entries) for entries in old.values()))
    training = set().union(*pr249.replay_states().values())
    universe = [
        entry
        for entry in pr249.all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in training
    ]
    paths, manifest, prefixes = (
        {},
        {
            "consumed": {name: sha256_file(path) for name, path in old_paths.items()},
            "suites": {},
        },
        set(),
    )
    old_prefixes = set().union(
        *(pr251.prefix_keys(entries) for entries in old.values())
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
        keys, current_prefixes = pr249.suite_keys(selected), pr251.prefix_keys(selected)
        if (
            len(keys) != 128
            or keys & used
            or current_prefixes & (old_prefixes | prefixes)
        ):
            fail(f"suite overlap {label}")
        path = workdir / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label], prefixes, used = path, prefixes | current_prefixes, used | keys
        manifest["suites"][label] = {
            "seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
    return paths, manifest


def analyze(evaluation: dict[str, Any]) -> dict[str, Any]:
    contrasts = {}
    for seed in ("seed45", "seed46"):
        for name, pair in {
            "positive_minus_negative": ("positive", "negative"),
            "ordinary_positive_minus_negative": ("ordinary_positive", "negative"),
            "tactical_positive_minus_negative": ("tactical_positive", "negative"),
            "ordinary_positive_minus_positive": ("ordinary_positive", "positive"),
            "tactical_positive_minus_positive": ("tactical_positive", "positive"),
        }.items():
            samples, per_suite = [], {}
            for label, values in evaluation.items():
                diff = paired_effect_difference(
                    values["candidates"][f"{seed}_{pair[0]}"]["effect"],
                    values["candidates"][f"{seed}_{pair[1]}"]["effect"],
                )
                sample = np.asarray(
                    list(diff["per_opening_effect"].values()), dtype=float
                )
                samples.append(sample)
                per_suite[label] = float(sample.mean())
            values = np.concatenate(samples)
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


def classify(analysis: dict[str, Any]) -> str:
    c = analysis["contrasts"]

    def sufficient(seed: str, lane: str) -> bool:
        item = c[f"{seed}_{lane}_minus_negative"]["pooled"]
        return item["lower_95"] > 0 and item["positive_suites"] >= 2

    def equivalent(seed: str, lane: str) -> bool:
        item = c[f"{seed}_{lane}_minus_positive"]["pooled"]
        return abs(item["effect"]) <= 0.02 and item["lower_95"] <= 0 <= item["upper_95"]

    controls = all(sufficient(seed, "positive") for seed in ("seed45", "seed46"))
    if not controls:
        return "fresh_strength_signal_fails_again"
    ordinary = [
        sufficient(seed, "ordinary_positive") and equivalent(seed, "ordinary_positive")
        for seed in ("seed45", "seed46")
    ]
    tactical = [
        sufficient(seed, "tactical_positive") and equivalent(seed, "tactical_positive")
        for seed in ("seed45", "seed46")
    ]
    if all(ordinary) and not all(tactical):
        return "ordinary_receiver_target_delta_is_causal"
    if all(tactical) and not all(ordinary):
        return "tactical_receiver_target_delta_is_causal"
    if all(ordinary) and all(tactical):
        return "both_receiver_categories_sufficient"
    if any(ordinary) != any(tactical):
        return "semantic_causality_is_seed_specific"
    if any(
        sufficient(seed, lane)
        for seed in ("seed45", "seed46")
        for lane in ("ordinary_positive", "tactical_positive")
    ):
        return "semantic_causality_is_seed_specific"
    return "receiver_categories_jointly_required"


def telemetry(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        suite: {
            f"{seed}_{lane}_versus_negative": pr252.divergence(
                values["candidates"][f"{seed}_{lane}"]["records"],
                values["candidates"][f"{seed}_negative"]["records"],
            )
            for seed in ("seed45", "seed46")
            for lane in ("ordinary_positive", "tactical_positive")
        }
        for suite, values in evaluation.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr253_semantic_receiver_target_surgery"),
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
        "schema": "alphazero_lite_pr253_semantic_receiver_target_surgery_v1",
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
            "primary_context": "1200:1200",
            "secondary_context": "384:256",
            "semantic_rule": "ordinary = not extra-turn and not capture",
        },
    }
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_suites_only:
        return
    pairs, semantics, gradients, training, artifacts = {}, {}, {}, {}, {}
    for seed, pair in pr252.source_pairs().items():
        positive, negative, pairs[seed] = pr252.verify_pair(seed, pair)
        views, semantics[seed] = hybrid_targets(positive, negative, seed)
        gradients[seed] = gradient_report(views, a16, parent, seed)
        training[seed], artifacts[seed] = train_views(
            seed, views, a16, optimizer, parent, args.workdir
        )
    primary = pr252.evaluate(artifacts, paths, args.workdir, "1200:1200")
    result = {
        **frozen,
        "pairs": pairs,
        "semantic_mass_accounting": semantics,
        "gradient_decomposition": gradients,
        "training": training,
        "primary_analysis": analyze(primary),
        "first_divergence_telemetry": telemetry(primary),
    }
    result["classification"] = classify(result["primary_analysis"])
    (args.workdir / "primary_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.skip_secondary:
        result["secondary_analysis"] = analyze(
            pr252.evaluate(artifacts, paths, args.workdir, "384:256")
        )
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
