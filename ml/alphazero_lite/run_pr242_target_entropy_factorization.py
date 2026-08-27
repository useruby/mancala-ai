#!/usr/bin/env python3
# ruff: noqa: E402
"""Factor PR #242's fixed noisy target effect into entropy and action ordering."""

from __future__ import annotations

import argparse
import copy
import itertools
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

from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as shadow_replay
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_onpolicy_shadow_replay import (
    A16_SNAPSHOT,
    P1_CHECKPOINT,
    exclusion_hashes,
    filter_rows,
    metrics,
    train_lane,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.train import load_checkpoint_into_model

LANES = (
    "fresh_denoised",
    "denoised_order_noisy_entropy",
    "noisy_order_denoised_entropy",
    "fresh_noisy",
)
CONTEXTS = ("384:256", "1200:1200")
ENTROPY_TOLERANCE = 1e-8
EXPECTED_FRESH_DIAGNOSTICS = {
    "mean_legal_policy_l1": 0.0759,
    "mean_js": 0.00819,
    "top1_disagreement": 0.0409,
    "target_entropy_difference": -0.0356,
}


def entropy(policy: list[float] | np.ndarray, legal_moves: list[int]) -> float:
    values = np.asarray(policy, dtype=np.float64)[legal_moves]
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def match_entropy(
    policy: list[float] | np.ndarray,
    legal_moves: list[int],
    desired_entropy: float,
) -> list[float]:
    """Adjust only concentration, preserving the deterministic source ordering."""
    source = np.asarray(policy, dtype=np.float64)
    legal = np.asarray(legal_moves, dtype=np.int64)
    source_legal = source[legal]
    current_entropy = entropy(source, legal_moves)
    if abs(current_entropy - desired_entropy) <= ENTROPY_TOLERANCE:
        return source.tolist()

    if desired_entropy > current_entropy:
        endpoint = np.full(len(legal), 1.0 / len(legal), dtype=np.float64)
    else:
        endpoint = np.zeros(len(legal), dtype=np.float64)
        endpoint[int(np.argmax(source_legal))] = 1.0
    endpoint_entropy = entropy(endpoint, list(range(len(legal))))
    if (
        not min(current_entropy, endpoint_entropy)
        <= desired_entropy
        <= max(current_entropy, endpoint_entropy)
    ):
        raise RuntimeError("requested entropy is outside the concentration path")

    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        candidate = (1.0 - midpoint) * source_legal + midpoint * endpoint
        candidate_entropy = entropy(candidate, list(range(len(legal))))
        if desired_entropy > current_entropy:
            if candidate_entropy < desired_entropy:
                lower = midpoint
            else:
                upper = midpoint
        elif candidate_entropy > desired_entropy:
            lower = midpoint
        else:
            upper = midpoint
    result = np.zeros_like(source)
    result[legal] = (1.0 - upper) * source_legal + upper * endpoint
    if abs(entropy(result, legal_moves) - desired_entropy) > ENTROPY_TOLERANCE:
        raise RuntimeError("entropy matching tolerance failure")
    if int(np.argmax(result[legal])) != int(np.argmax(source_legal)):
        raise RuntimeError("source top1 changed during entropy matching")
    return result.tolist()


def build_targets(
    rows: list[dict[str, Any]],
    fresh_noisy: list[list[float]],
    fresh_denoised: list[list[float]],
) -> dict[str, list[list[float]]]:
    denoised_order_noisy_entropy, noisy_order_denoised_entropy = [], []
    for row, noisy, denoised in zip(rows, fresh_noisy, fresh_denoised, strict=True):
        legal = [int(move) for move in row["legal_moves"]]
        if bool(row["action_sampling_noise_enabled"]):
            denoised_order_noisy_entropy.append(
                match_entropy(denoised, legal, entropy(noisy, legal))
            )
            noisy_order_denoised_entropy.append(
                match_entropy(noisy, legal, entropy(denoised, legal))
            )
        else:
            denoised_order_noisy_entropy.append(denoised)
            noisy_order_denoised_entropy.append(noisy)
    return {
        "fresh_denoised": fresh_denoised,
        "denoised_order_noisy_entropy": denoised_order_noisy_entropy,
        "noisy_order_denoised_entropy": noisy_order_denoised_entropy,
        "fresh_noisy": fresh_noisy,
    }


def target_diagnostics(
    rows: list[dict[str, Any]], targets: dict[str, list[list[float]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for left, right in itertools.combinations(LANES, 2):
        records = [
            isolation.policy_stats(a, b, row["legal_moves"])
            for row, a, b in zip(rows, targets[left], targets[right], strict=True)
        ]
        groups: dict[str, list[dict[str, float | bool]]] = defaultdict(list)
        for row, record in zip(rows, records, strict=True):
            groups[f"root_noise:{bool(row['action_sampling_noise_enabled'])}"].append(
                record
            )
            groups[f"phase:{isolation.phase(row)}"].append(record)
            groups[f"legal_move_count:{len(row['legal_moves'])}"].append(record)
        summary = isolation.summarize(records)
        summary["mean_left_entropy"] = float(
            np.mean(
                [
                    entropy(targets[left][index], row["legal_moves"])
                    for index, row in enumerate(rows)
                ]
            )
        )
        summary["mean_right_entropy"] = float(
            np.mean(
                [
                    entropy(targets[right][index], row["legal_moves"])
                    for index, row in enumerate(rows)
                ]
            )
        )
        result[f"{left}_vs_{right}"] = {
            "overall": summary,
            "by_stratum": {
                key: isolation.summarize(value) for key, value in sorted(groups.items())
            },
        }
    return result


def assert_factorization(
    rows: list[dict[str, Any]], targets: dict[str, list[list[float]]]
) -> dict[str, Any]:
    source = [
        {**row, "policy": targets["fresh_denoised"][index]}
        for index, row in enumerate(rows)
    ]
    views = {
        name: [
            {**copy.deepcopy(row), "policy": values[index]}
            for index, row in enumerate(rows)
        ]
        for name, values in targets.items()
    }
    result = isolation.assert_views(source, views)
    disagreement_rows = 0
    for index, row in enumerate(rows):
        legal = row["legal_moves"]
        noisy, denoised = (
            targets["fresh_noisy"][index],
            targets["fresh_denoised"][index],
        )
        b, c = (
            targets["denoised_order_noisy_entropy"][index],
            targets["noisy_order_denoised_entropy"][index],
        )
        if not bool(row["action_sampling_noise_enabled"]):
            if b != denoised or c != noisy:
                raise RuntimeError("noise-disabled target changed")
            continue
        if abs(entropy(b, legal) - entropy(noisy, legal)) > ENTROPY_TOLERANCE:
            raise RuntimeError("denoised-order entropy mismatch")
        if abs(entropy(c, legal) - entropy(denoised, legal)) > ENTROPY_TOLERANCE:
            raise RuntimeError("noisy-order entropy mismatch")
        if np.argmax(np.asarray(b)[legal]) != np.argmax(np.asarray(denoised)[legal]):
            raise RuntimeError("denoised source top1 changed")
        if np.argmax(np.asarray(c)[legal]) != np.argmax(np.asarray(noisy)[legal]):
            raise RuntimeError("noisy source top1 changed")
        for source_policy, transformed_policy in ((denoised, b), (noisy, c)):
            source_legal = np.asarray(source_policy, dtype=np.float64)[legal]
            transformed_legal = np.asarray(transformed_policy, dtype=np.float64)[legal]
            for left, right in itertools.combinations(range(len(legal)), 2):
                if (
                    source_legal[left] > source_legal[right]
                    and not transformed_legal[left] > transformed_legal[right]
                ):
                    raise RuntimeError("source action ordering changed")
        if np.argmax(np.asarray(noisy)[legal]) != np.argmax(
            np.asarray(denoised)[legal]
        ):
            disagreement_rows += 1
    result["top1_disagreement_rows_checked"] = disagreement_rows
    return result


def all_target_ce(
    rows: list[dict[str, Any]],
    state: dict[str, torch.Tensor],
    targets: dict[str, list[list[float]]],
) -> dict[str, float]:
    return isolation.all_target_metrics(rows, state, targets)


def win_draw_loss(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {"wins": 0, "draws": 0, "losses": 0}
    for row in records:
        result[
            {"challenger": "wins", "draw": "draws", "current": "losses"}[row["winner"]]
        ] += 1
    return result


def arena_summary(
    workdir: Path, artifacts: dict[str, Path], p1: Path, workers: int, suite: Path
) -> dict[str, Any]:
    controls = {
        context: isolation.arena_records(
            workdir / "arena", p1, p1, context, "p1_control", workers, suite
        )
        for context in CONTEXTS
    }
    result: dict[str, Any] = {lane: {} for lane in LANES}
    effects: dict[str, dict[str, dict[str, Any]]] = {lane: {} for lane in LANES}
    for lane in LANES:
        for context, control in controls.items():
            records = isolation.arena_records(
                workdir / "arena", artifacts[lane], p1, context, lane, workers, suite
            )
            effect = paired_opening_candidate_effect(records, control)
            effects[lane][context] = effect
            result[lane][context] = {
                "effect": effect["paired_candidate_effect"],
                "ci": effect["opening_bootstrap_ci"],
                "seat_effects": {"p0": effect["p0_effect"], "p1": effect["p1_effect"]},
                "win_draw_loss": win_draw_loss(records),
            }
    result["paired_lane_differences"] = {
        "denoised_order_noisy_entropy_minus_fresh_denoised": {
            context: paired_effect_difference(
                effects["denoised_order_noisy_entropy"][context],
                effects["fresh_denoised"][context],
            )
            for context in CONTEXTS
        },
        "noisy_order_denoised_entropy_minus_fresh_noisy": {
            context: paired_effect_difference(
                effects["noisy_order_denoised_entropy"][context],
                effects["fresh_noisy"][context],
            )
            for context in CONTEXTS
        },
    }
    return result


def classify(arena: dict[str, Any]) -> str:
    high = arena["paired_lane_differences"]
    add_entropy = high["denoised_order_noisy_entropy_minus_fresh_denoised"][
        "1200:1200"
    ]["paired_candidate_effect"]
    remove_entropy = high["noisy_order_denoised_entropy_minus_fresh_noisy"][
        "1200:1200"
    ]["paired_candidate_effect"]
    noisy_gain = (
        arena["fresh_noisy"]["1200:1200"]["effect"]
        - arena["fresh_denoised"]["1200:1200"]["effect"]
    )
    if noisy_gain <= 0.0:
        return "target_factorization_not_reproducible"
    threshold = abs(noisy_gain) * 0.5
    if add_entropy >= threshold and remove_entropy <= -threshold:
        return "noisy_entropy_is_sufficient"
    if add_entropy < threshold and remove_entropy >= -threshold:
        return "noisy_action_direction_is_sufficient"
    if add_entropy < threshold and remove_entropy <= -threshold:
        return "entropy_and_direction_jointly_required"
    if add_entropy >= threshold and remove_entropy >= -threshold:
        return "both_hybrids_preserve_gain"
    return "target_factorization_not_reproducible"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr242_target_entropy_factorization"),
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("/tmp/azlite_onpolicy_shadow_replay/ordinary_onpolicy.jsonl"),
    )
    parser.add_argument(
        "--canonical-suite",
        type=Path,
        default=Path("/tmp/azlite_opening_suite/medium_eval.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument(
        "--reuse-targets",
        action="store_true",
        help="Resume a completed deterministic target reconstruction from this workdir.",
    )
    args = parser.parse_args()
    if (
        sha256_file(args.replay) != isolation.SOURCE_SHA
        or sha256_file(args.canonical_suite) != isolation.SUITE_SHA
    ):
        raise RuntimeError(
            "invariant_failure: frozen replay or canonical suite hash mismatch"
        )
    if sha256_file(P1_CHECKPOINT) != shadow_replay.P1_CHECKPOINT_SHA:
        raise RuntimeError("invariant_failure: P1 checkpoint hash mismatch")
    rows, exclusions = filter_rows(
        read_jsonl(args.replay), *exclusion_hashes(args.canonical_suite)
    )
    if len(rows) != 27350:
        raise RuntimeError("invariant_failure: eligible replay row count mismatch")
    if (
        sha256_file(isolation.A16_TARGET_CHECKPOINT)
        != isolation.A16_TARGET_CHECKPOINT_SHA
    ):
        raise RuntimeError("invariant_failure: A16 target evaluator hash mismatch")
    if args.reuse_targets:
        fresh_noisy = [
            row["policy"] for row in read_jsonl(args.workdir / "fresh_noisy.jsonl")
        ]
        fresh_denoised = [
            row["policy"] for row in read_jsonl(args.workdir / "fresh_denoised.jsonl")
        ]
        if len(fresh_noisy) != len(rows) or len(fresh_denoised) != len(rows):
            raise RuntimeError("invariant_failure: cached target row count mismatch")
    else:
        fresh_noisy, fresh_denoised = isolation.parallel_target_pair(
            rows, isolation.A16_TARGET_CHECKPOINT, args.workers
        )
    baseline = isolation.target_diagnostics(
        rows,
        {
            "original_noisy": [row["policy"] for row in rows],
            "fresh_noisy": fresh_noisy,
            "fresh_denoised": fresh_denoised,
        },
    )["fresh_noisy_vs_fresh_denoised"]["overall"]
    for key, expected in EXPECTED_FRESH_DIAGNOSTICS.items():
        if not np.isclose(baseline[key], expected, atol=5e-5):
            raise RuntimeError(
                f"invariant_failure: PR #242 diagnostic mismatch for {key}"
            )
    targets = build_targets(rows, fresh_noisy, fresh_denoised)
    invariants = assert_factorization(rows, targets)
    for name, values in targets.items():
        isolation.write_jsonl(
            args.workdir / f"{name}.jsonl",
            [{**row, "policy": values[index]} for index, row in enumerate(rows)],
        )
    initial = torch.load(A16_SNAPSHOT, map_location="cpu", weights_only=False)
    p1_model = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1_model, P1_CHECKPOINT)
    parent = {
        key: value.detach().cpu().clone()
        for key, value in p1_model.state_dict().items()
    }
    artifacts, training = {}, {}
    for name in LANES:
        view = [
            {**copy.deepcopy(row), "policy": targets[name][index]}
            for index, row in enumerate(rows)
        ]
        snapshots, optimizers = train_lane(view, initial, parent, torch.device("cpu"))
        training[name] = {
            "lane_metrics": metrics(
                view, snapshots, parent, initial["model"], initial["optimizer"]
            ),
            "cross_target_ce": {},
        }
        artifacts[name] = None
        for step, state in snapshots.items():
            output = args.workdir / f"{name}_train" / f"step_{step:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "optimizer": optimizers[step]},
                output.with_suffix(".pt"),
            )
            artifact = export(state, output, f"pr242_{name}_{step}")
            training[name]["cross_target_ce"][str(step)] = all_target_ce(
                rows, state, targets
            )
            if step == 16:
                artifacts[name] = artifact
    result: dict[str, Any] = {
        "schema": "pr242_target_entropy_factorization_v1",
        "classification": "target_factorization_not_reproducible",
        "source": {
            "sha256": isolation.SOURCE_SHA,
            "row_count": 27642,
            "eligible_row_count": len(rows),
            "exclusions": exclusions,
        },
        "frozen_artifacts": {
            "a16_snapshot_sha256": sha256_file(A16_SNAPSHOT),
            "a16_target_evaluator_sha256": isolation.A16_TARGET_CHECKPOINT_SHA,
            "p1_checkpoint_sha256": shadow_replay.P1_CHECKPOINT_SHA,
            "canonical_suite_sha256": isolation.SUITE_SHA,
        },
        "seed_contract": "Inherited unchanged from PR #242: "
        + "sha256([namespace, game_index, move_index, canonical_state_hash])[:16] mod 2**31",
        "reconstructed_pr242_fresh_diagnostics": baseline,
        "invariants": invariants,
        "target_differences": target_diagnostics(rows, targets),
        "training": training,
    }
    if not args.skip_arena:
        result["ordinary_puct_arena"] = arena_summary(
            args.workdir,
            artifacts,
            P1_CHECKPOINT.parent / "artifact",
            args.workers,
            args.canonical_suite,
        )
        result["classification"] = classify(result["ordinary_puct_arena"])
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["classification"])


if __name__ == "__main__":
    main()
