#!/usr/bin/env python3
# ruff: noqa: E402
"""PR #265: separate unique AlphaZero data scale from optimizer exposure.

The freeze stage creates ten candidates and seals AN/AO/AP before an arena is
run.  The two lanes have identical presentation and batch-size sequences; only
the number of games from which those presentations are drawn differs.
"""

from __future__ import annotations

import argparse
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

from ml.alphazero_lite import consumed_suite_registry as registry_module
from ml.alphazero_lite import run_pr258_two_replay_aggregation as pr258
from ml.alphazero_lite import run_pr264_joint_alphazero_iteration as pr264
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite.evaluation_metrics import paired_effect_difference
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    canonical_manifest_hash,
    configure_determinism,
    game_split,
    read_jsonl,
    sha256_file,
    write_json,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    compute_policy_cross_entropy,
    compute_value_loss_vector,
    legal_mask_matrix_for_encoded_states,
)

SEEDS = (68, 69, 70, 71, 72)
LANES = ("repeat20_matched", "unique100_once")
SUITE_SEEDS = {"AN": 40042, "AO": 41042, "AP": 42042}
GAMES, SIMULATIONS, BATCH_SIZE, SMALL_FRACTION = 3500, 384, 512, 0.20
PLAN_SEED = 265042


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def expected_registry() -> list[str]:
    return [
        "canonical",
        *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "AA",
        "AB",
        "AC",
        "AD",
        "AE",
        "AF",
        "AG",
        "AH",
        "AI",
        "AJ",
        "AK",
        "AL",
        "AM",
    ]


def canonical_sha(value: Any) -> str:
    return pr258.canonical_sha(value)


def state_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def batches(indexes: list[int]) -> list[list[int]]:
    return [
        indexes[start : start + BATCH_SIZE]
        for start in range(0, len(indexes), BATCH_SIZE)
    ]


def build_plans(
    rows: list[dict[str, Any]], seed: int, replay_sha: str
) -> dict[str, Any]:
    """Preregister game partitions and equal-length state presentation plans."""
    train_games, validation_games = game_split(rows, seed)
    train_set, validation_set = set(train_games), set(validation_games)
    if train_set & validation_set:
        fail(f"seed {seed} game split overlaps")
    rng = np.random.default_rng(np.random.SeedSequence([PLAN_SEED, seed]))
    small_count = max(1, round(len(train_games) * SMALL_FRACTION))
    small_games = sorted(
        rng.permutation(np.asarray(train_games))[:small_count].tolist()
    )
    small_set = set(small_games)
    train_indexes = [
        i for i, row in enumerate(rows) if int(row["game_index"]) in train_set
    ]
    validation_indexes = [
        i for i, row in enumerate(rows) if int(row["game_index"]) in validation_set
    ]
    small_indexes = [
        i for i in train_indexes if int(rows[i]["game_index"]) in small_set
    ]
    if not small_indexes or set(small_indexes) & set(validation_indexes):
        fail(f"seed {seed} nested subset is invalid")
    unique_order = rng.permutation(np.asarray(train_indexes)).astype(np.int64).tolist()
    repeat_order: list[int] = []
    cycle = 0
    while len(repeat_order) < len(unique_order):
        cycle_rng = np.random.default_rng(
            np.random.SeedSequence([PLAN_SEED, seed, cycle, 20])
        )
        repeat_order.extend(
            cycle_rng.permutation(np.asarray(small_indexes)).astype(np.int64).tolist()
        )
        cycle += 1
    repeat_order = repeat_order[: len(unique_order)]
    result = {
        "schema": "azlite_pr265_unique_data_scale_plan_v1",
        "seed": seed,
        "replay_sha256": replay_sha,
        "game_split": {
            "train_games": train_games,
            "validation_games": validation_games,
        },
        "small_subset_games": small_games,
        "source_indexes": {
            "train": train_indexes,
            "validation": validation_indexes,
            "small": small_indexes,
        },
        "lanes": {
            "unique100_once": {"order": unique_order, "batches": batches(unique_order)},
            "repeat20_matched": {
                "order": repeat_order,
                "batches": batches(repeat_order),
            },
        },
        "batch_size": BATCH_SIZE,
        "recipe": {
            "optimizer": "Adam",
            "lr": 1e-6,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "value_loss": "huber",
            "value_loss_weight": 1.0,
        },
    }
    for lane in LANES:
        plan = result["lanes"][lane]
        plan["ordering_sha256"] = canonical_sha(plan["order"])
        plan["batch_plan_sha256"] = canonical_sha(plan["batches"])
        plan["presentations"] = len(plan["order"])
        plan["optimizer_steps"] = len(plan["batches"])
    if [len(batch) for batch in result["lanes"][LANES[0]]["batches"]] != [
        len(batch) for batch in result["lanes"][LANES[1]]["batches"]
    ]:
        fail(f"seed {seed} batch-size sequence differs")
    result["plan_sha256"] = canonical_manifest_hash(result)
    return result


def validate_pure_targets(
    rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Keep only completed games whose stored targets are raw pure-AZ targets."""
    by_game: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_game.setdefault(int(row["game_index"]), []).append(row)
    completed = {
        game_index
        for game_index, game_rows in by_game.items()
        if all(row.get("game_completed") is True for row in game_rows)
    }
    completion_ratio = len(completed) / max(len(by_game), 1)
    if completion_ratio < 0.9:
        fail(f"seed {seed} completed-game ratio is below 90%")
    eligible = [row for row in rows if int(row["game_index"]) in completed]
    for row in eligible:
        policy = np.asarray(row.get("policy"), dtype=np.float64)
        legal = legal_mask_matrix_for_encoded_states(
            np.asarray([row["state"]], dtype=np.float32)
        )[0]
        expected_value = (
            0.0
            if row.get("winner") is None
            else (1.0 if int(row["winner"]) == int(row["player"]) else -1.0)
        )
        if (
            row.get("value_target_mode") != "default"
            or row.get("policy_target_mode") != "default"
            or row.get("policy_target_noise_mode") != "noisy"
            or int(row.get("simulations", -1)) != SIMULATIONS
        ):
            fail(f"seed {seed} has non-default pure-AZ target metadata")
        if (
            policy.shape != (6,)
            or not np.isfinite(policy).all()
            or (policy < 0).any()
            or not np.isclose(policy.sum(), 1.0, atol=1e-6)
            or np.any(policy[~legal.astype(bool)] != 0.0)
        ):
            fail(f"seed {seed} has an invalid policy target")
        if not np.isclose(float(row["value"]), expected_value, atol=1e-7):
            fail(f"seed {seed} value target is not its terminal outcome")
    return eligible, {
        "games_total": len(by_game),
        "games_completed": len(completed),
        "games_excluded_incomplete": len(by_game) - len(completed),
        "completed_game_ratio": completion_ratio,
        "rows_total": len(rows),
        "rows_eligible": len(eligible),
        "rows_excluded_incomplete": len(rows) - len(eligible),
    }


def pure_az_telemetry(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    gradient_norm: float,
    clipped: int,
    steps: int,
) -> dict[str, Any]:
    """Report held-out pure-AZ fit without inventing a parent-policy metric."""
    probability, values = pr264.policy_probs(model.cpu(), rows)
    a16 = new_model(torch.device("cpu"))
    a16.load_state_dict(initial)
    a16_probability, _ = pr264.policy_probs(a16, rows)
    target = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    value_target = np.asarray([row["value"] for row in rows], dtype=np.float64)
    error = values - value_target
    delta: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "total",
            "input_trunk",
            "policy_hidden_readout",
            "adapter",
            "value_hidden_readout",
        )
    }
    for name, parameter in model.state_dict().items():
        vector = (parameter.cpu() - initial[name]).reshape(-1)
        delta["total"].append(vector)
        delta[pr264.family(name)].append(vector)
    return {
        "policy_ce_own_target": float(
            np.mean(pr258._cross_entropy(probability, target))
        ),
        "policy_ce_raw_a16_mcts_target": float(
            np.mean(pr258._cross_entropy(probability, target))
        ),
        "legal_policy_l1_vs_a16": float(
            np.abs(probability - a16_probability).sum(1).mean()
        ),
        "legal_policy_js_vs_a16": float(pr264.js(probability, a16_probability).mean()),
        "top1_disagreement_vs_a16": float(
            np.mean(np.argmax(probability, 1) != np.argmax(a16_probability, 1))
        ),
        "value_huber": float(
            np.mean(np.where(np.abs(error) < 1, 0.5 * error**2, np.abs(error) - 0.5))
        ),
        "value_mae": float(np.abs(error).mean()),
        "value_sign_accuracy": float(np.mean(np.sign(values) == np.sign(value_target))),
        "prediction_mean": float(values.mean()),
        "prediction_std": float(values.std()),
        "parameter_deltas": {
            key: float(torch.linalg.vector_norm(torch.cat(value)))
            for key, value in delta.items()
        },
        "gradient_norm": gradient_norm,
        "clipping_frequency": clipped / max(steps, 1),
    }


def tensor_data(
    rows: list[dict[str, Any]], device: torch.device
) -> dict[str, torch.Tensor]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    return {
        "x": torch.from_numpy(x).to(device),
        "p": torch.from_numpy(
            np.asarray([row["policy"] for row in rows], dtype=np.float32)
        ).to(device),
        "v": torch.from_numpy(
            np.asarray([row["value"] for row in rows], dtype=np.float32).reshape(-1, 1)
        ).to(device),
        "mask": torch.from_numpy(legal_mask_matrix_for_encoded_states(x)).to(device),
    }


def gradient_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clipped: int,
    steps: int,
    last: float,
) -> dict[str, Any]:
    return {
        "gradient_norm": last,
        "clipping_frequency": clipped / steps,
        "optimizer_steps": steps,
    }


def train_lane(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    lane: str,
    initial: dict[str, torch.Tensor],
    directory: Path,
) -> dict[str, Any]:
    configure_determinism(torch.device("cpu"), int(plan["seed"]))
    model = new_model(torch.device("cpu"))
    model.load_state_dict(copy.deepcopy(initial))
    apply_trainable_scope(model, "all")
    names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if len(names) != len(list(model.named_parameters())):
        fail("candidate parameter scope is not full network")
    if sum(parameter.numel() for parameter in model.parameters()) != 73_741:
        fail("candidate parameter count is not the frozen A16 architecture")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.0)
    fresh_optimizer_sha = replay.optimizer_state_sha256(optimizer.state_dict())
    data = tensor_data(rows, torch.device("cpu"))
    lane_plan = plan["lanes"][lane]
    exposure_points = {
        max(
            1, int(np.ceil(lane_plan["presentations"] * fraction))
        ): f"exposure_{int(fraction * 100)}pct"
        for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)
    }
    telemetry, clipped, presentations, grad = {}, 0, 0, 0.0
    model.train()
    for step, indexes in enumerate(lane_plan["batches"], 1):
        index = torch.tensor(indexes, dtype=torch.long)
        logits, prediction = model(data["x"][index])
        policy = compute_policy_cross_entropy(
            logits.masked_fill(data["mask"][index] <= 0, -1e9), data["p"][index]
        ).mean()
        value = compute_value_loss_vector(
            prediction, data["v"][index], value_loss="huber", huber_delta=1.0
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        (policy + value).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        clipped += grad > 1.0
        optimizer.step()
        presentations += len(indexes)
        for boundary, label in exposure_points.items():
            if label not in telemetry and presentations >= boundary:
                telemetry[label] = gradient_snapshot(
                    model, optimizer, clipped, step, grad
                )
    validation = [rows[i] for i in plan["source_indexes"]["validation"]]
    heldout = pure_az_telemetry(
        model,
        initial,
        validation,
        grad,
        clipped,
        len(lane_plan["batches"]),
    )
    state = state_copy(model.cpu())
    artifact = export(state, directory, f"pr265_seed{plan['seed']}_{lane}")
    used = [rows[i] for i in lane_plan["order"]]
    return {
        "artifact": str(artifact),
        "artifact_model_sha256": sha256_file(artifact / "model.npz"),
        "model_state_sha256": pr258.contract.state_sha256(state),
        "optimizer_initial_sha256": fresh_optimizer_sha,
        "trainable_parameter_names": names,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "unique_games": len({int(row["game_index"]) for row in used}),
        "unique_states": len({canonical_sha(row["state"]) for row in used}),
        "presentations": presentations,
        "optimizer_steps": len(lane_plan["batches"]),
        "gradient_exposure": telemetry,
        "heldout_telemetry": heldout,
        "policy_target_provenance": "raw_noisy_visit_count_pure_alphazero",
    }


def hierarchy(raw: dict[str, Any], lane: str) -> dict[str, Any]:
    nested, per_seed = [], {}
    for seed in SEEDS:
        suite_vectors = [
            np.asarray(
                [
                    raw[label][f"seed{seed}_{lane}"]["effect"]["per_opening_effect"][
                        key
                    ]
                    for key in sorted(
                        raw[label][f"seed{seed}_{lane}"]["effect"]["per_opening_effect"]
                    )
                ]
            )
            for label in SUITE_SEEDS
        ]
        nested.append(suite_vectors)
        per_seed[str(seed)] = {
            "effect": float(np.concatenate(suite_vectors).mean()),
            "suite_effects": {
                label: float(vector.mean())
                for label, vector in zip(SUITE_SEEDS, suite_vectors, strict=True)
            },
        }
    return summarize_hierarchy(nested, per_seed)


def contrast(raw: dict[str, Any]) -> dict[str, Any]:
    nested, per_seed = [], {}
    for seed in SEEDS:
        vectors = []
        for label in SUITE_SEEDS:
            left = raw[label][f"seed{seed}_unique100_once"]["effect"]
            right = raw[label][f"seed{seed}_repeat20_matched"]["effect"]
            vectors.append(
                np.asarray(
                    list(
                        paired_effect_difference(left, right)[
                            "per_opening_effect"
                        ].values()
                    )
                )
            )
        nested.append(vectors)
        per_seed[str(seed)] = {
            "effect": float(np.concatenate(vectors).mean()),
            "suite_effects": {
                label: float(vector.mean())
                for label, vector in zip(SUITE_SEEDS, vectors, strict=True)
            },
        }
    return summarize_hierarchy(nested, per_seed)


def summarize_hierarchy(
    nested: list[list[np.ndarray]], per_seed: dict[str, Any]
) -> dict[str, Any]:
    values = np.asarray([per_seed[str(seed)]["effect"] for seed in SEEDS])
    rng, draws = np.random.default_rng(265042), []
    for _ in range(10_000):
        draws.append(
            float(
                np.mean(
                    [
                        rng.choice(
                            nested[i][rng.integers(0, len(SUITE_SEEDS))],
                            128,
                            replace=True,
                        ).mean()
                        for i in rng.integers(0, len(SEEDS), len(SEEDS))
                    ]
                )
            )
        )
    return {
        "per_seed": per_seed,
        "pooled": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "positive_seed_count": int((values > 0).sum()),
            "hierarchical_ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        },
    }


def classify(primary: dict[str, Any], paired: dict[str, Any]) -> str:
    unique, repeat, contrast_result = (
        primary["unique100_once"],
        primary["repeat20_matched"],
        paired,
    )

    def screen(result: dict[str, Any]) -> bool:
        return (
            result["pooled"]["mean"] > 0
            and result["pooled"]["hierarchical_ci95"][0] > 0
            and result["pooled"]["positive_seed_count"] >= 4
            and min(item["effect"] for item in result["per_seed"].values()) >= -0.02
        )

    contrast_passes = (
        contrast_result["pooled"]["mean"] > 0
        and contrast_result["pooled"]["hierarchical_ci95"][0] > 0
    )
    if screen(unique) and contrast_passes:
        return "unique_scale_beats_incumbent_and_repetition"
    if screen(repeat) and not contrast_passes:
        return "optimizer_exposure_is_sufficient"
    if screen(unique):
        return "replacement_without_unique_data_evidence"
    if unique["pooled"]["mean"] > 0:
        return "unique_scale_improves_but_not_replacement"
    if unique["pooled"]["mean"] < 0 and repeat["pooled"]["mean"] < 0:
        return "scale_degrades_strength"
    return "scale_does_not_rescue_high_budget_strength"


def decision_for(classification: str) -> str:
    return {
        "unique_scale_beats_incumbent_and_repetition": (
            "Recommend a separately frozen consolidation candidate as the next PR."
        ),
        "optimizer_exposure_is_sufficient": (
            "Additional optimization, not unique-data scale, explains the result; do not promote."
        ),
        "replacement_without_unique_data_evidence": (
            "Record replacement evidence but no causal unique-data conclusion; do not promote."
        ),
        "unique_scale_improves_but_not_replacement": "Do not promote.",
        "scale_does_not_rescue_high_budget_strength": (
            "Close incremental A16 fitting; do not propose more epochs, target mixing, or isolated scopes."
        ),
        "scale_degrades_strength": (
            "Close incremental A16 fitting; do not propose more epochs, target mixing, or isolated scopes."
        ),
    }[classification]


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PR #265 Unique-Data Scale Results",
        "",
        f"**Classification:** `{summary['classification']}`.",
        "",
        "## Frozen Contract",
        "",
        "- Five 3,500-game A16 replays (seeds 68-72), ordinary reused-tree 384-simulation PUCT.",
        "- Pure raw noisy visit-count targets, terminal-outcome values, and full 73,741-parameter Adam updates.",
        "- `repeat20_matched` and `unique100_once` have identical presentation, step, and batch-size sequences.",
        "",
        "## Frozen Evidence",
        "",
        f"- Candidate aggregate SHA-256: `{summary['candidate_aggregate_sha256']}`.",
        f"- Suite manifest SHA-256: `{summary['suite_manifest_sha256']}`; preflight SHA-256: `{summary['preflight_sha256']}`.",
        f"- A16 hashes: `{json.dumps(summary['a16_hashes'], sort_keys=True)}`.",
        "",
        "## Results",
        "",
    ]
    for title, result in (
        ("Primary 1200:1200", summary["primary_1200"]),
        ("Secondary 384:384", summary["secondary_384"]),
    ):
        lines += [
            f"### {title}",
            "",
            "| Lane | Mean | Median | SD | Range | Positive seeds | Hierarchical CI95 |",
            "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        for lane in LANES:
            pooled = result[lane]["pooled"]
            lines.append(
                f"| {lane} | {pooled['mean']:+.6f} | {pooled['median']:+.6f} | {pooled['sd']:.6f} | [{pooled['min']:+.6f}, {pooled['max']:+.6f}] | {pooled['positive_seed_count']}/5 | [{pooled['hierarchical_ci95'][0]:+.6f}, {pooled['hierarchical_ci95'][1]:+.6f}] |"
            )
        contrast_result = result["unique_minus_repeat"]["pooled"]
        lines += [
            f"\nPaired unique-minus-repeat: {contrast_result['mean']:+.6f}; CI95 [{contrast_result['hierarchical_ci95'][0]:+.6f}, {contrast_result['hierarchical_ci95'][1]:+.6f}].",
            "",
            "Per-seed effects (AN/AO/AP):",
        ]
        for seed in SEEDS:
            unique = result["unique100_once"]["per_seed"][str(seed)]
            repeat = result["repeat20_matched"]["per_seed"][str(seed)]
            paired = result["unique_minus_repeat"]["per_seed"][str(seed)]
            lines.append(
                f"- Seed {seed}: unique {unique['effect']:+.6f} (AN/AO/AP {unique['suite_effects']}), repeat {repeat['effect']:+.6f} (AN/AO/AP {repeat['suite_effects']}), contrast {paired['effect']:+.6f} (AN/AO/AP {paired['suite_effects']})."
            )
        lines += [
            "",
            "Replay completion, exclusions, unique exposure, presentations, and optimizer steps are preserved in the compact JSON summary.",
            "",
        ]
    lines += [
        "## Next Action",
        "",
        decision_for(summary["classification"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Exclude full replay orders while retaining all frozen evidence and effects."""
    return {
        "schema": "azlite_pr265_unique_data_scale_summary_v1",
        "classification": summary["classification"],
        "decision": decision_for(summary["classification"]),
        "a16_hashes": summary["a16_hashes"],
        "candidate_aggregate_sha256": summary["candidate_aggregate_sha256"],
        "suite_manifest_sha256": summary["suite_manifest_sha256"],
        "preflight_sha256": summary["preflight_sha256"],
        "frozen_suites": summary["suite_manifest"]["newly_consumed"],
        "preflight": {
            key: summary["preflight"][key] for key in (*SUITE_SEEDS, "passed")
        },
        "seeds": {
            seed: {
                "raw_replay_sha256": info["raw_replay_sha256"],
                "filtered_replay_sha256": info["filtered_replay_sha256"],
                "exclusions": info["exclusions"],
                "completion": info["completion"],
                "plan_sha256": info["plan"]["plan_sha256"],
                "lanes": {
                    lane: {
                        key: info["lanes"][lane][key]
                        for key in (
                            "artifact_model_sha256",
                            "model_state_sha256",
                            "total_parameter_count",
                            "trainable_parameter_count",
                            "unique_games",
                            "unique_states",
                            "presentations",
                            "optimizer_steps",
                        )
                    }
                    | {
                        "ordering_sha256": info["plan"]["lanes"][lane][
                            "ordering_sha256"
                        ],
                        "batch_plan_sha256": info["plan"]["lanes"][lane][
                            "batch_plan_sha256"
                        ],
                    }
                    for lane in LANES
                },
            }
            for seed, info in summary["seeds"].items()
        },
        "primary_1200": summary["primary_1200"],
        "secondary_384": summary["secondary_384"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_pr265_unique_data_scale")
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only and args.evaluate_only:
        fail("freeze-only and evaluate-only are mutually exclusive")
    args.workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if not args.evaluate_only:
        registry = registry_module.load(args.workdir)
        if list(registry) != expected_registry():
            fail("consumed-suite registry is not authoritative through AM")
        hashes = pr264.verify_a16()
        snapshot = torch.load(
            replay.A16_SNAPSHOT, map_location="cpu", weights_only=False
        )
        initial, _unused_adam = replay.immutable_initial_state(snapshot)
        frozen, replays = (
            {
                "schema": "azlite_pr265_unique_data_scale_v1",
                "a16_hashes": hashes,
                "seeds": {},
            },
            {},
        )
        for seed in SEEDS:
            directory, generated = (
                args.workdir / f"seed{seed}",
                args.workdir / f"seed{seed}" / "generated" / "ordinary_reused.jsonl",
            )
            if generated.is_file():
                raw = read_jsonl(generated)
            else:
                # PR264's command builder is the frozen contract; only its game
                # count is prospectively scaled for this experiment.
                previous_games, pr264.GAMES = pr264.GAMES, GAMES
                try:
                    raw = pr264.generate_replay(seed, generated, args.workers)
                finally:
                    pr264.GAMES = previous_games
            if set(int(row["game_index"]) for row in raw) != set(range(GAMES)):
                fail(f"seed {seed} does not contain exactly {GAMES} games")
            completed_rows, completion = validate_pure_targets(raw, seed)
            rows, exclusions = pr264.eligible_rows(completed_rows, registry)
            if len({canonical_sha(row["state"]) for row in rows}) < 5_000:
                fail(
                    f"seed {seed} has fewer than 5,000 verified unique eligible states"
                )
            directory.mkdir(parents=True, exist_ok=True)
            replay_path = directory / "replay.jsonl"
            replay_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            plan = build_plans(rows, seed, sha256_file(replay_path))
            write_json(directory / "training_plan.json", plan)
            lanes = {
                lane: train_lane(rows, plan, lane, initial, directory / lane)
                for lane in LANES
            }
            if (
                lanes[LANES[0]]["presentations"] != lanes[LANES[1]]["presentations"]
                or lanes[LANES[0]]["optimizer_steps"]
                != lanes[LANES[1]]["optimizer_steps"]
                or lanes[LANES[0]]["optimizer_initial_sha256"]
                != lanes[LANES[1]]["optimizer_initial_sha256"]
            ):
                fail(f"seed {seed} optimizer exposure is not matched")
            replays[seed] = rows
            frozen["seeds"][str(seed)] = {
                "raw_replay_sha256": sha256_file(generated),
                "filtered_replay_sha256": sha256_file(replay_path),
                "exclusions": exclusions,
                "completion": completion,
                "plan": plan,
                "lanes": lanes,
            }
        frozen["candidate_aggregate_sha256"] = canonical_sha(
            {
                f"seed{seed}_{lane}": frozen["seeds"][str(seed)]["lanes"][lane][
                    "model_state_sha256"
                ]
                for seed in SEEDS
                for lane in LANES
            }
        )
        write_json(args.workdir / "frozen_candidates.json", frozen)
        old = pr258.SUITE_SEEDS
        pr258.SUITE_SEEDS = SUITE_SEEDS
        suites, suite_manifest, preflight = pr258.seal_suites(
            args.workdir, registry, replays
        )
        pr258.SUITE_SEEDS = old
        if not preflight["passed"]:
            fail("AN/AO/AP preflight failed")
        frozen.update(
            {
                "suite_manifest": suite_manifest,
                "suite_manifest_sha256": canonical_sha(suite_manifest),
                "preflight": preflight,
                "preflight_sha256": canonical_sha(preflight),
            }
        )
        write_json(args.workdir / "frozen_manifest.json", frozen)
        if args.freeze_only:
            return
    frozen_path = args.workdir / "frozen_manifest.json"
    if not frozen_path.is_file():
        fail("evaluation requires frozen_manifest.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        canonical_sha(frozen["suite_manifest"]) != frozen["suite_manifest_sha256"]
        or canonical_sha(frozen["preflight"]) != frozen["preflight_sha256"]
        or not frozen["preflight"]["passed"]
    ):
        fail("frozen suite evidence mismatch")
    candidates = {
        f"seed{seed}_{lane}": Path(
            frozen["seeds"][str(seed)]["lanes"][lane]["artifact"]
        )
        for seed in SEEDS
        for lane in LANES
    }
    suites = {
        label: args.workdir / "suites" / f"suite_{label}.jsonl" for label in SUITE_SEEDS
    }
    primary_raw = pr264.direct_evaluate(
        candidates, suites, args.workdir, "1200:1200", args.workers
    )
    secondary_raw = pr264.direct_evaluate(
        candidates, suites, args.workdir, "384:384", args.workers
    )
    primary = {lane: hierarchy(primary_raw, lane) for lane in LANES} | {
        "unique_minus_repeat": contrast(primary_raw)
    }
    secondary = {lane: hierarchy(secondary_raw, lane) for lane in LANES} | {
        "unique_minus_repeat": contrast(secondary_raw)
    }
    summary = frozen | {
        "primary_1200": primary,
        "secondary_384": secondary,
        "classification": classify(primary, primary["unique_minus_repeat"]),
        "wall_clock_seconds": time.monotonic() - started,
    }
    write_json(args.workdir / "summary.json", summary)
    write_report(args.workdir / "report.md", summary)
    write_json(
        REPO_ROOT / "docs/data/alphazero-lite-pr265-unique-data-scale-summary.json",
        compact_summary(summary),
    )
    write_report(
        REPO_ROOT / "docs/alphazero-lite-pr265-unique-data-scale-results.md", summary
    )
    print(summary["classification"])


if __name__ == "__main__":
    main()
