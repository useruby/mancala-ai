#!/usr/bin/env python3
# ruff: noqa: E402
"""Sealed PR260 value-head refresh of frozen PR258 single-replay candidates.

This experiment never generates self-play and never trains a policy or trunk
tensor.  It intentionally fails closed on any replay, baseline, policy, or
artifact identity mismatch.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import arena
from ml.alphazero_lite import consumed_suite_registry as registry_module
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite import run_pr241_optimizer_isolation_reproduction as contract
from ml.alphazero_lite import run_pr258_two_replay_aggregation as pr258
from ml.alphazero_lite import run_pr259_two_replay_second_epoch as pr259
from ml.alphazero_lite.evaluation_metrics import paired_effect_difference
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import sha256_file
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch
from ml.alphazero_lite.self_play import build_eval_search_options
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
)

SOURCE = Path("/tmp/azlite_pr258_two_replay_aggregation")
SEEDS = tuple(range(53, 63))
STEPS = (1, 4, 16)
BATCH_SIZE = 512
VALUE_KEYS = (
    "value_hidden_layer.weight",
    "value_hidden_layer.bias",
    "value_head.weight",
    "value_head.bias",
)
SUITE_SEEDS = {"AB": 28042, "AC": 29042, "AD": 30042}
_DIAGNOSTIC_EVALUATORS: (
    tuple[arena.ArtifactEvaluator, arena.ArtifactEvaluator] | None
) = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def state_copy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def load_source(
    registry: dict[str, registry_module.ConsumedSuite],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Verify PR258's committed rows against the now larger consumed registry."""
    registry_module.validate(registry)
    frozen = json.loads(
        (SOURCE / "frozen_replays_and_plans.json").read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (SOURCE / "frozen_candidates.json").read_text(encoding="utf-8")
    )
    blocked, groups = replay.exclusion_hashes(pr258.pr249.CANONICAL_SUITE)
    consumed = set().union(
        *(replay.canonical_arena_hashes(spec.path) for spec in registry.values())
    )
    eligible_by_seed = {}
    for seed in SEEDS:
        path = SOURCE / f"seed{seed}/generated/ordinary_reused.jsonl"
        raw = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        eligible, exclusions = replay.filter_rows(
            raw,
            blocked | consumed,
            {**groups, "consumed_evaluation_openings": consumed},
        )
        actual = {
            "raw_rows": len(raw),
            "eligible_rows": len(eligible),
            "replay_sha256": sha256_file(path),
            "state_outcome_sha256": pr258.canonical_sha(
                [
                    {key: value for key, value in row.items() if key != "policy"}
                    for row in eligible
                ]
            ),
            "exclusions": exclusions,
        }
        if actual != frozen["replays"][str(seed)]:
            fail(
                f"PR258 replay, outcome, exclusion, or eligible-row mismatch for seed {seed}"
            )
        eligible_by_seed[seed] = eligible
    return eligible_by_seed, frozen, candidates


def value_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    named = dict(model.named_parameters())
    if tuple(name for name in named if name in VALUE_KEYS) != VALUE_KEYS:
        fail("unexpected value parameter layout")
    return [named[name] for name in VALUE_KEYS]


def value_predictions(
    state: dict[str, torch.Tensor], rows: list[dict[str, Any]]
) -> np.ndarray:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    chunks = []
    with torch.no_grad():
        for offset in range(0, len(rows), 4096):
            x = torch.tensor(
                np.asarray(
                    [r["state"] for r in rows[offset : offset + 4096]], dtype=np.float32
                )
            )
            _logits, value = model(x)
            chunks.append(value.numpy().reshape(-1))
    return np.concatenate(chunks)


def policy_logits_and_probabilities(
    state: dict[str, torch.Tensor], rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    logits_chunks, policy_chunks = [], []
    with torch.no_grad():
        for offset in range(0, len(rows), 4096):
            x_np = np.asarray(
                [r["state"] for r in rows[offset : offset + 4096]], dtype=np.float32
            )
            logits, _value = model(torch.from_numpy(x_np))
            raw = logits.numpy()
            legal = legal_mask_matrix_for_encoded_states(x_np).astype(bool)
            masked = raw.copy()
            masked[~legal] = -1e9
            masked -= masked.max(axis=1, keepdims=True)
            probability = np.exp(masked)
            policy_chunks.append(probability / probability.sum(axis=1, keepdims=True))
            logits_chunks.append(raw)
    return np.concatenate(logits_chunks), np.concatenate(policy_chunks)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        same = values == value
        result[same] = result[same].mean()
    return result


def correlation(
    left: np.ndarray, right: np.ndarray, *, rank: bool = False
) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    if rank:
        left, right = ranks(left), ranks(right)
    return float(np.corrcoef(left, right)[0, 1])


def value_metrics(
    state: dict[str, torch.Tensor], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    prediction = value_predictions(state, rows)
    target = np.asarray([row["value"] for row in rows], dtype=np.float64)
    error = prediction - target
    pairs_left, pairs_right = np.triu_indices(len(rows), 1)
    distinct = target[pairs_left] != target[pairs_right]
    return {
        "huber_loss": float(
            np.mean(np.where(np.abs(error) < 1, 0.5 * error**2, np.abs(error) - 0.5))
        ),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
        "pearson": correlation(prediction, target),
        "spearman": correlation(prediction, target, rank=True),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "pairwise_concordance": float(
            np.mean(
                np.sign(prediction[pairs_left] - prediction[pairs_right])[distinct]
                == np.sign(target[pairs_left] - target[pairs_right])[distinct]
            )
        )
        if np.any(distinct)
        else None,
    }


def reproduce_baseline(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    a16: dict[str, torch.Tensor],
    adam: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Replay the exact PR258 policy-only lane solely to verify its checkpoint."""
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(a16))
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = replay.load_isolated_optimizer(model, adam)
    parent = new_model(torch.device("cpu"))
    parent.load_state_dict(parent_state)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    model.train()
    for indexes in batches:
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        # This exactly retains the PR258 beta=.95 policy target, but only here.
        target = pr258.mixed_policy_target(
            batch["p"],
            pr258.incumbent_policy_batch(parent, batch),
            batch["mask"],
            pr258.BETA,
        )
        policy_loss, value_loss = pr258._losses(model, {**batch, "p": target})
        optimizer.zero_grad(set_to_none=True)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
    if replay.optimizer_state_sha256(adam) != pr258.ADAM_SHA:
        fail("pristine A16 Adam state mutated")
    return state_copy(model.state_dict()), copy.deepcopy(optimizer.state_dict())


def refresh_value(
    rows: list[dict[str, Any]],
    batches: list[list[int]],
    baseline: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(baseline))
    apply_trainable_scope(model, "value_head")
    parameters = value_parameters(model)
    if sum(parameter.numel() for parameter in parameters) != 4705:
        fail("value trainable parameter count is not 4705")
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0.0)
    snapshots = {0: state_copy(model.state_dict())}
    model.train()
    for step, indexes in enumerate(batches, 1):
        batch = _batch(rows, np.asarray(indexes, dtype=np.int64), torch.device("cpu"))
        _logits, prediction = model(batch["x"])
        loss = (
            0.6
            * compute_value_loss_vector(
                prediction, batch["v"], value_loss="huber", huber_delta=1.0
            ).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step in STEPS:
            snapshots[step] = state_copy(model.state_dict())
    if set(snapshots) != {0, *STEPS}:
        fail("missing value checkpoints")
    return snapshots


def combine(
    baseline: dict[str, torch.Tensor], refreshed: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = state_copy(baseline)
    for key in VALUE_KEYS:
        result[key] = refreshed[key].detach().cpu().clone()
    if any(
        not torch.equal(result[key], baseline[key])
        for key in result
        if key not in VALUE_KEYS
    ):
        fail("non-value tensor changed while combining candidate")
    return result


def policy_identity(
    baseline: dict[str, torch.Tensor],
    refreshed: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    probe: list[dict[str, Any]],
) -> dict[str, Any]:
    policy_keys = [key for key in baseline if key not in VALUE_KEYS]
    if not all(torch.equal(baseline[key], refreshed[key]) for key in policy_keys):
        fail("policy, inherited head, or trunk tensor mutated")
    max_difference = 0.0
    for group in (rows, probe):
        left_logits, left_policy = policy_logits_and_probabilities(baseline, group)
        right_logits, right_policy = policy_logits_and_probabilities(refreshed, group)
        difference = float(np.max(np.abs(left_logits - right_logits)))
        max_difference = max(max_difference, difference)
        if difference != 0.0 or not np.array_equal(left_policy, right_policy):
            fail("policy logits or legal probabilities changed")
    return {
        "policy_tensor_bit_identical": True,
        "max_policy_logit_difference": max_difference,
    }


def artifact_roundtrip(
    state: dict[str, torch.Tensor], artifact: Path, rows: list[dict[str, Any]]
) -> dict[str, float]:
    """Check the exported runtime against PyTorch on fixed replay states."""
    _logits, policy = policy_logits_and_probabilities(state, rows[:64])
    model = new_model(torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    evaluator = arena.ArtifactEvaluator(artifact)
    max_policy_difference = 0.0
    max_value_difference = 0.0
    with torch.no_grad():
        for index, row in enumerate(rows[:64]):
            game = KalahGame.from_state(decode(row["state"]))
            artifact_policy, artifact_value = evaluator.evaluate(game)
            _torch_logits, torch_value = model(
                torch.tensor(np.asarray([row["state"]], dtype=np.float32))
            )
            max_policy_difference = max(
                max_policy_difference,
                float(np.max(np.abs(policy[index] - artifact_policy))),
            )
            max_value_difference = max(
                max_value_difference,
                abs(float(torch_value.item()) - artifact_value),
            )
    if max_policy_difference > 1e-5 or max_value_difference > 1e-5:
        fail("artifact round-trip mismatch")
    return {
        "max_legal_policy_difference": max_policy_difference,
        "max_value_difference": max_value_difference,
        "states": 64,
    }


def value_drift(
    baseline: dict[str, torch.Tensor],
    refreshed: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    before, after = (
        value_predictions(baseline, rows),
        value_predictions(refreshed, rows),
    )
    shift, target = np.abs(after - before), np.asarray([row["value"] for row in rows])
    result: dict[str, Any] = {
        "mean_absolute_shift": float(shift.mean()),
        "p50": float(np.percentile(shift, 50)),
        "p90": float(np.percentile(shift, 90)),
        "p99": float(np.percentile(shift, 99)),
        "sign_flip_rate": float(np.mean(np.sign(before) != np.sign(after))),
        "correlation": correlation(before, after),
        "calibration_by_target": {},
        "move_index": {},
    }
    for value in (-1, 0, 1):
        selected = target == value
        result["calibration_by_target"][str(value)] = {
            "count": int(selected.sum()),
            "baseline_mean": float(before[selected].mean()) if selected.any() else None,
            "refreshed_mean": float(after[selected].mean()) if selected.any() else None,
        }
    for label, selected in {
        "lt10": np.asarray([r["move_index"] < 10 for r in rows]),
        "10_29": np.asarray([10 <= r["move_index"] < 30 for r in rows]),
        "ge30": np.asarray([r["move_index"] >= 30 for r in rows]),
    }.items():
        result["move_index"][label] = {
            "count": int(selected.sum()),
            "mean_absolute_shift": float(shift[selected].mean())
            if selected.any()
            else None,
        }
    return result


def decode(encoded: list[float]) -> dict[str, Any]:
    values = np.asarray(encoded[:15], dtype=np.float32)
    return {
        "player_pits": [int(round(x * 48)) for x in values[:6]],
        "opponent_pits": [int(round(x * 48)) for x in values[6:12]],
        "player_store": int(round(values[12] * 48)),
        "opponent_store": int(round(values[13] * 48)),
        "current_player": int(round(values[14])),
    }


def _init_diagnostic_workers(baseline_artifact: str, refreshed_artifact: str) -> None:
    global _DIAGNOSTIC_EVALUATORS
    _DIAGNOSTIC_EVALUATORS = (
        arena.ArtifactEvaluator(Path(baseline_artifact)),
        arena.ArtifactEvaluator(Path(refreshed_artifact)),
    )


def _diagnostic_row(
    item: tuple[int, dict[str, Any], str],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if _DIAGNOSTIC_EVALUATORS is None:
        fail("diagnostic worker evaluators are not initialized")
    index, row, context = item
    state, seed, pair = decode(row["state"]), 260_042 + index, []
    for evaluator in _DIAGNOSTIC_EVALUATORS:
        result = arena.evaluate_artifact_position(
            evaluator=evaluator,
            state=state,
            simulations=int(context),
            seed=seed,
            c_puct=1.25,
            search_options=build_eval_search_options(
                root_policy_mode="deterministic",
                tactical_root_bias=0.0,
                normalize_values=False,
            ),
        )
        children = result["child_stats"]
        visits = np.asarray([child["visits"] for child in children], dtype=np.float64)
        visits /= visits.sum()
        selected = int(result["selected_move"])
        rank = sorted(children, key=lambda child: child["q_value"], reverse=True)
        pair.append(
            (
                selected,
                visits,
                float(result.get("search_root_value", result["value"])),
                next(
                    i for i, child in enumerate(rank) if int(child["move"]) == selected
                ),
            )
        )
    return pair[0], pair[1]


def search_diagnostics(
    baseline_artifact: Path,
    refreshed_artifact: Path,
    probe: list[dict[str, Any]],
    context: str,
    workers: int,
) -> dict[str, Any]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_diagnostic_workers,
        initargs=(str(baseline_artifact), str(refreshed_artifact)),
    ) as executor:
        records = list(
            executor.map(
                _diagnostic_row,
                ((index, row, context) for index, row in enumerate(probe)),
                chunksize=1,
            )
        )
    js_values = []
    for left_row, right_row in records:
        midpoint = (left_row[1] + right_row[1]) / 2

        def kl(probability: np.ndarray) -> float:
            positive = probability > 0
            return float(
                np.sum(
                    probability[positive]
                    * np.log(probability[positive] / midpoint[positive])
                )
            )

        js_values.append(0.5 * kl(left_row[1]) + 0.5 * kl(right_row[1]))
    return {
        "states": len(records),
        "selected_move_disagreement": float(
            np.mean([a[0] != b[0] for a, b in records])
        ),
        "visit_policy_js": float(np.mean(js_values)),
        "root_value_shift": float(np.mean([b[2] - a[2] for a, b in records])),
        "selected_child_q_rank_change": float(
            np.mean([b[3] - a[3] for a, b in records])
        ),
    }


def seal_suites(
    workdir: Path,
    registry: dict[str, registry_module.ConsumedSuite],
    replays: dict[int, list[dict[str, Any]]],
    probe: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    registry_module.validate(registry)
    through_aa = ["canonical", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "AA"]
    through_ad = [*through_aa, "AB", "AC", "AD"]
    if list(registry) == through_ad:
        paths = {label: registry[label].path for label in SUITE_SEEDS}
        if any(
            not path.is_file() or sha256_file(path) != registry[label].sha256
            for label, path in paths.items()
        ):
            fail("previously consumed AB/AC/AD suite mismatch")
        manifest = {
            "consumed": registry_module.manifest(
                {
                    label: spec
                    for label, spec in registry.items()
                    if label not in SUITE_SEEDS
                }
            ),
            "newly_consumed": {
                label: {
                    "seed": registry[label].seed,
                    "sha256": registry[label].sha256,
                    "openings": 128,
                    "status": "consumed",
                }
                for label in SUITE_SEEDS
            },
        }
        return paths, manifest, {"passed": True, "reused_consumed_suites": True}
    if list(registry) != through_aa:
        fail("registry is not canonical through AA")
    return pr258.seal_suites(
        workdir, registry, replays
    )  # SUITE_SEEDS is replaced by caller.


def evaluate(
    candidates: dict[str, Path],
    suites: dict[str, Path],
    workdir: Path,
    context: str,
    workers: int,
) -> dict[str, Any]:
    previous_workers = pr258.WORKERS
    pr258.WORKERS = workers
    try:
        return pr258.evaluate(candidates, suites, workdir, context)
    finally:
        pr258.WORKERS = previous_workers


def analyze(raw: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    per_seed, bootstrap = {}, []
    for seed in SEEDS:
        suite_deltas, baseline_effects, refreshed_effects = {}, [], []
        per_suite = []
        for label in labels:
            values = raw[label]
            difference = paired_effect_difference(
                values[f"seed{seed}_refreshed"]["effect"],
                values[f"seed{seed}_baseline"]["effect"],
            )
            vector = np.asarray(
                [
                    difference["per_opening_effect"][key]
                    for key in sorted(difference["per_opening_effect"])
                ]
            )
            suite_deltas[label] = float(vector.mean())
            per_suite.append(vector)
            baseline_effects.append(
                values[f"seed{seed}_baseline"]["effect"]["paired_candidate_effect"]
            )
            refreshed_effects.append(
                values[f"seed{seed}_refreshed"]["effect"]["paired_candidate_effect"]
            )
        per_seed[str(seed)] = {
            "suite_deltas": suite_deltas,
            "delta": float(np.concatenate(per_suite).mean()),
            "baseline_absolute_p1_effect": float(np.mean(baseline_effects)),
            "refreshed_absolute_p1_effect": float(np.mean(refreshed_effects)),
        }
        bootstrap.append(per_suite)
    values = np.asarray([per_seed[str(seed)]["delta"] for seed in SEEDS])
    rng, draws = np.random.default_rng(42), []
    for _ in range(10_000):
        seed_draws = []
        for seed_index in rng.integers(0, len(SEEDS), len(SEEDS)):
            suite = bootstrap[seed_index][rng.integers(0, len(labels))]
            seed_draws.append(rng.choice(suite, len(suite), replace=True).mean())
        draws.append(float(np.mean(seed_draws)))
    return {
        "per_seed": per_seed,
        "primary": {
            "mean_delta": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seed_count": int((values > 0).sum()),
            "hierarchical_replay_seed_suite_opening_ci": {
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
            },
        },
    }


def classify(
    primary: dict[str, Any], learning_count: int, shallow: dict[str, Any] | None
) -> str:
    values, summary = (
        [v["delta"] for v in primary["per_seed"].values()],
        primary["primary"],
    )
    success = (
        summary["mean_delta"] > 0
        and summary["hierarchical_replay_seed_suite_opening_ci"]["lower_95"] > 0
        and summary["positive_seed_count"] >= 8
        and min(values) >= -0.02
        and learning_count >= 8
    )
    if learning_count < 8:
        return "value_head_does_not_learn"
    if success:
        shallow_values = (
            [v["delta"] for v in shallow["per_seed"].values()] if shallow else []
        )
        if shallow and (
            shallow["primary"]["mean_delta"] < -0.02
            or sum(v < -0.02 for v in shallow_values) >= 3
        ):
            return "value_refresh_improves_high_budget_with_shallow_harm"
        return "value_head_refresh_improves_strength"
    if summary["mean_delta"] < 0 or min(values) < -0.02:
        return "value_refresh_degrades_strength"
    if summary["mean_delta"] > 0:
        return "value_refresh_is_seed_sensitive"
    return "value_refresh_improves_value_loss_not_strength"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr260_value_head_refresh")
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    started, workdir = time.monotonic(), args.workdir
    registry = registry_module.load(workdir)
    replays, frozen_plans, source_candidates = load_source(registry)
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, adam = replay.immutable_initial_state(snapshot)
    if (
        sha256_file(replay.A16_SNAPSHOT) != pr258.A16_SHA
        or replay.optimizer_state_sha256(adam) != pr258.ADAM_SHA
    ):
        fail("A16 source artifact mismatch")
    parent_model = new_model(torch.device("cpu"))
    pr258.load_checkpoint_into_model(parent_model, replay.P1_CHECKPOINT)
    parent_state = state_copy(parent_model.state_dict())
    lanes = pr259.lanes_from_manifest(replays, frozen_plans)
    probes = []
    seen_probe_states = set()
    for row in sorted(
        (row for rows in replays.values() for row in rows),
        key=lambda row: pr258.canonical_sha(row["state"]),
    ):
        state_hash = pr258.canonical_sha(row["state"])
        if state_hash not in seen_probe_states:
            probes.append(row)
            seen_probe_states.add(state_hash)
        if len(probes) == 256:
            break
    if len(probes) != 256:
        fail("diagnostic probe hashes are not unique")
    training, artifacts, models = {}, {}, {}
    for pair_index, (seed_a, seed_b) in enumerate(pr258.PAIRS, 1):
        pair = f"pair{pair_index}"
        for seed, lane in ((seed_a, "single_a"), (seed_b, "single_b")):
            rows, batches, validation = lanes[pair][lane]
            baseline, baseline_adam = reproduce_baseline(
                rows, batches, a16, adam, parent_state
            )
            expected = source_candidates["training"][pair][lane]["checkpoints"]["16"]
            saved = torch.load(
                SOURCE / f"train/{pair}/{lane}/step_0016.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                contract.state_sha256(baseline) != expected["model_sha256"]
                or replay.optimizer_state_sha256(baseline_adam)
                != expected["optimizer_sha256"]
                or not state_equal(baseline, saved["model"])
            ):
                fail(f"PR258 baseline does not reproduce for seed {seed}")
            snapshots = refresh_value(rows, batches, baseline)
            final = combine(baseline, snapshots[16])
            identity = policy_identity(baseline, final, rows, probes)
            checkpoints = {"initial": value_metrics(baseline, rows)}
            checkpoints.update(
                {str(step): value_metrics(snapshots[step], rows) for step in STEPS}
            )
            heldout = {
                "initial": value_metrics(baseline, validation),
                "16": value_metrics(final, validation),
            }
            if not heldout["16"]["huber_loss"] < heldout["initial"]["huber_loss"]:
                heldout["learning_invariant"] = False
            else:
                heldout["learning_invariant"] = True
            directory = workdir / "train" / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": baseline, "optimizer": baseline_adam},
                directory / "adapter_only_step16.pt",
            )
            torch.save({"model": final}, directory / "adapter_plus_value_refresh.pt")
            artifacts[str(seed)] = {
                "baseline": export(
                    baseline,
                    directory / "adapter_only_step16",
                    f"pr260_seed{seed}_baseline",
                ),
                "refreshed": export(
                    final,
                    directory / "adapter_plus_value_refresh",
                    f"pr260_seed{seed}_refreshed",
                ),
            }
            roundtrip = {
                name: artifact_roundtrip(state, artifacts[str(seed)][name], rows)
                for name, state in (("baseline", baseline), ("refreshed", final))
            }
            training[str(seed)] = {
                "plan": frozen_plans["sample_plans"][pair]["plans"][lane],
                "trainable_parameters": list(VALUE_KEYS),
                "trainable_parameter_count": 4705,
                "checkpoints": checkpoints,
                "heldout": heldout,
                "value_drift": value_drift(baseline, final, validation),
                "policy_identity": identity,
                "artifact_roundtrip": roundtrip,
                "baseline_model_sha256": contract.state_sha256(baseline),
                "baseline_optimizer_sha256": replay.optimizer_state_sha256(
                    baseline_adam
                ),
                "refreshed_model_sha256": contract.state_sha256(final),
            }
            models[f"seed{seed}_baseline"] = artifacts[str(seed)]["baseline"]
            models[f"seed{seed}_refreshed"] = artifacts[str(seed)]["refreshed"]
    frozen = {
        "schema": "alphazero_lite_pr260_value_head_refresh_v1",
        "source_pr258_manifest_sha256": sha256_file(SOURCE / "frozen_manifest.json"),
        "training": training,
        "probe": {
            "count": 256,
            "state_hashes": [pr258.canonical_sha(row["state"]) for row in probes],
        },
        "guardrails": {
            "new_self_play_games": 0,
            "policy_target_changed": False,
            "trunk_trained": False,
            "optimizer_steps": 16,
            "value_loss": "huber",
            "huber_delta": 1.0,
            "value_loss_weight": 0.6,
            "lr": 1e-5,
            "weight_decay": 0.0,
            "promotion": False,
        },
    }
    frozen["candidate_model_sha256"] = pr258.canonical_sha(
        {
            seed: {
                key: training[seed][f"{key}_model_sha256"]
                for key in ("baseline", "refreshed")
            }
            for seed in training
        }
    )
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "frozen_candidates.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pr258.SUITE_SEEDS = SUITE_SEEDS
    paths, suite_manifest, preflight = seal_suites(workdir, registry, replays, probes)
    if not preflight["passed"]:
        fail("AB/AC/AD preflight failed")
    frozen["suite_manifest"], frozen["preflight"] = suite_manifest, preflight
    frozen["suite_manifest_sha256"] = pr258.canonical_sha(suite_manifest)
    frozen_path = workdir / "frozen_manifest.json"
    frozen_path.write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.freeze_only:
        return
    diagnostic = {
        str(seed): {
            context: search_diagnostics(
                artifacts[str(seed)]["baseline"],
                artifacts[str(seed)]["refreshed"],
                probes,
                context,
                args.workers,
            )
            for context in ("384", "1200")
        }
        for seed in SEEDS
    }
    primary = analyze(
        evaluate(models, paths, workdir, "1200:1200", args.workers), tuple(SUITE_SEEDS)
    )
    (workdir / "primary_1200_results.json").write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shallow = analyze(
        evaluate(models, paths, workdir, "384:256", args.workers), tuple(SUITE_SEEDS)
    )
    learned = sum(item["heldout"]["learning_invariant"] for item in training.values())
    deltas = np.asarray(
        [primary["per_seed"][str(seed)]["delta"] for seed in SEEDS], dtype=np.float64
    )
    search_causality = {
        metric: correlation(
            np.asarray(
                [diagnostic[str(seed)]["1200"][metric] for seed in SEEDS],
                dtype=np.float64,
            ),
            deltas,
            rank=True,
        )
        for metric in ("selected_move_disagreement", "visit_policy_js")
    }
    result = {
        **frozen,
        "search_diagnostics": diagnostic,
        "search_causality_correspondence": search_causality,
        "primary_evaluation": primary,
        "shallow_evaluation": shallow,
        "heldout_huber_improved_seeds": learned,
        "classification": classify(primary, learned, shallow),
        "wall_clock_seconds": time.monotonic() - started,
    }
    (workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
