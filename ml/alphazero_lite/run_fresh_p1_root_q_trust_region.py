#!/usr/bin/env python3
# ruff: noqa: E402
"""Test a training-only ordinary-PUCT root-Q directional trust region.

This non-promoting experiment starts at immutable P1, generates no self-play,
and changes only the accepted length of the original beta=.95 Adam proposal.
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

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.root_q_trust_region import (
    LAMBDAS,
    adam_proposal_and_restore,
    apply_delta,
    choose_lambda,
    root_q_diagnostics,
    select_guard_indexes,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    ADAPTER_KEYS,
    BETA,
    export,
    new_model,
    output,
)
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.run_fresh_selfplay_anchor_iteration import _cross_entropy
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.train import (
    apply_trainable_scope,
    legal_mask_matrix_for_encoded_states,
)

SEED = 235
SIMULATIONS = 1200
STEPS = (1, 4, 16)
WORKDIR = Path("/tmp/azlite_root_q_trust_region")


def _seed(state_hash: str) -> int:
    return int(
        hashlib.sha256(f"root-q-trust-region:{state_hash}".encode()).hexdigest()[:16],
        16,
    )


def _ordinary(
    game: KalahGame, evaluator: ArtifactEvaluator, simulations: int, seed: int
) -> tuple[np.ndarray, dict]:
    search = PUCT(
        evaluator,
        simulations,
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


def _root_record(
    item: dict[str, Any], evaluator: ArtifactEvaluator, simulations: int
) -> dict:
    visits, summary = _ordinary(
        KalahGame.from_state(item["state"]),
        evaluator,
        simulations,
        _seed(item["state_hash"]),
    )
    return {
        "state_hash": item["state_hash"],
        "legal_actions": [int(value["move"]) for value in summary["child_stats"]],
        "child_visits": visits.tolist(),
        "child_q_values": [float(value["q_value"]) for value in summary["child_stats"]],
        "selected_move": int(summary["selected_move"]),
        "visit_distribution": (visits / visits.sum()).tolist(),
        "summary": summary,
    }


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def _candidate_evaluator(
    state: dict[str, torch.Tensor], workdir: Path, label: str
) -> ArtifactEvaluator:
    artifact = export(state, workdir / "trial_artifacts" / label, label)
    return ArtifactEvaluator(artifact)


def _guard_searches(
    state: dict[str, torch.Tensor],
    items: list[dict],
    reference: list[dict],
    workdir: Path,
    label: str,
    constraint_key: str,
) -> tuple[float, dict[str, float]]:
    evaluator = _candidate_evaluator(state, workdir, label)
    diagnostics = [
        root_q_diagnostics(
            _root_record(item, evaluator, SIMULATIONS)["summary"], record["summary"]
        )
        for item, record in zip(items, reference, strict=True)
    ]
    aggregate = {
        key: float(np.mean([entry[key] for entry in diagnostics]))
        for key in (
            "q_direction_error",
            "per_action_q_rank_disagreement",
            "best_q_action_disagreement",
            "top_two_q_order_disagreement",
            "centered_q_l1",
            "root_move_disagreement",
            "visit_js",
        )
    }
    return aggregate[constraint_key], aggregate


def _metrics(
    state: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    pure: dict[str, torch.Tensor],
) -> dict[str, float]:
    x = np.asarray([row["state"] for row in rows], dtype=np.float32)
    mask = legal_mask_matrix_for_encoded_states(x)
    candidate, reference = output(state, x, mask), output(parent, x, mask)
    target = np.asarray([row["policy"] for row in rows], dtype=np.float64)
    baseline = float(np.mean(_cross_entropy(reference, target)))
    current = float(np.mean(_cross_entropy(candidate, target)))
    pure_ce = float(np.mean(_cross_entropy(output(pure, x, mask), target)))
    adapter = torch.cat(
        [(state[key] - parent[key]).reshape(-1) for key in ADAPTER_KEYS]
    )
    return {
        "ce_search": current,
        "fit_fraction": float((baseline - current) / (baseline - pure_ce)),
        "policy_l1_vs_p1": float(np.abs(candidate - reference).sum(axis=1).mean()),
        "adapter_norm": float(torch.linalg.vector_norm(adapter)),
    }


def _loss(
    model: torch.nn.Module,
    parent_state: dict[str, torch.Tensor],
    rows: list[dict],
    source: np.ndarray,
    indexes: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    primary, _ = pr233._losses_for_step(
        model,
        parent_state,
        rows,
        source,
        indexes,
        None,
        None,
        np.asarray([], dtype=np.int64),
        1,
        device,
    )
    return primary


def _state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _train_baseline(
    parent_state: dict[str, torch.Tensor],
    rows: list[dict],
    source: np.ndarray,
    plan: np.ndarray,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]]:
    model = new_model(device)
    model.load_state_dict(parent_state)
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-5,
        weight_decay=0.0,
    )
    captures = {}
    for step, indexes in enumerate(plan[:16], 1):
        optimizer.zero_grad(set_to_none=True)
        _loss(model, parent_state, rows, source, indexes, device).backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        if step in STEPS:
            captures[step] = _state(model)
    return captures


def _frozen_diagnostic(
    state: dict[str, torch.Tensor],
    items: list[dict],
    references: list[dict],
    workdir: Path,
    label: str,
) -> dict[str, Any]:
    evaluator = _candidate_evaluator(state, workdir, label)
    candidate = [_root_record(item, evaluator, SIMULATIONS) for item in items]
    return {
        "records": [
            {
                "state_hash": item["state_hash"],
                "candidate_move": record["selected_move"],
                "p1_move": references[index]["selected_move"],
            }
            for index, (item, record) in enumerate(zip(items, candidate, strict=True))
        ],
        "q_diagnostics": {
            key: float(
                np.mean(
                    [
                        root_q_diagnostics(
                            record["summary"], references[index]["summary"]
                        )[key]
                        for index, record in enumerate(candidate)
                    ]
                )
            )
            for key in (
                "q_direction_error",
                "per_action_q_rank_disagreement",
                "best_q_action_disagreement",
                "top_two_q_order_disagreement",
                "centered_q_l1",
                "root_move_disagreement",
                "visit_js",
            )
        },
    }


def _classification(
    constrained: dict[str, Any],
    arena: dict[str, Any],
    frozen: dict[str, Any],
    constraint_key: str,
) -> tuple[str, str]:
    activity = constrained["constraint_activity"]
    metric = constrained["metrics"]["16"]
    safe = bool(arena) and all(value["safe"] for value in arena.values())
    if (
        not constrained["invariants"]["adam_lambda_one_parity"]
        or not constrained["invariants"]["constrained_matches_baseline_when_lambda_one"]
    ):
        return (
            "optimizer_line_search_invariant_failure",
            "Repair Adam proposal parity before interpreting this result.",
        )
    if activity["fraction_lambda_0"] > 0.5 or metric["fit_fraction"] < 0.25:
        return (
            "root_q_constraint_stalls",
            "Add a separate root action-value/Q head or increase representational degrees of freedom rather than weakening the constraint.",
        )
    if activity["fraction_lambda_1"] >= 0.95:
        return (
            "unconstrained_update_already_within_rank_budget"
            if constraint_key == "per_action_q_rank_disagreement"
            else "unconstrained_update_already_within_q_budget",
            "Change the Q-consistency statistic, not the threshold.",
        )
    if (
        safe
        and frozen["amplified"]["rescue_rate"] > 0.3
        and frozen["washed"]["new_divergence_rate"] == 0
    ):
        return (
            "root_q_trust_region_rescues_training",
            "Run a prospective fresh P1->P2 AlphaZero generation with the same fixed search-Q trust-region protocol.",
        )
    if (
        constrained["generalization"]["amplified"][constraint_key]
        > constrained["epsilon_constraint"]
    ):
        return (
            "guard_q_consistency_does_not_generalize",
            "Audit which state distribution is missing before changing the constraint.",
        )
    return (
        "q_direction_preserved_but_game_unsafe",
        "Audit which additional search-level statistic is needed beyond root-Q direction.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-arenas", action="store_true")
    parser.add_argument(
        "--statistic",
        choices=("q_direction", "per_action_q_rank"),
        default="q_direction",
    )
    args = parser.parse_args()
    constraint_key = {
        "q_direction": "q_direction_error",
        "per_action_q_rank": "per_action_q_rank_disagreement",
    }[args.statistic]
    args.workdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    replay = pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl"
    p1_checkpoint = (
        pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    p1_artifact = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    rows = read_jsonl(replay)
    population, exclusions = pr233._population(rows)
    cache_path = Path("/tmp/azlite_shadow_target_distillation/target_cache.npz")
    if sha256_file(cache_path) != pr233.TARGET_CACHE_SHA:
        raise RuntimeError("immutable PR #233 shadow-sensitivity cache mismatch")
    cache_file = np.load(cache_path, allow_pickle=False)
    cache = {key: cache_file[key] for key in cache_file.files}
    sensitivity = [
        pr233._js(
            cache["ordinary_a16"][i] / SIMULATIONS, cache["shadow_a16"][i] / SIMULATIONS
        )
        for i in range(len(population))
    ]
    primary_indexes, secondary_indexes, manifest = select_guard_indexes(
        population, sensitivity, 32, SEED
    )
    guard_manifest_sha = _write_json(args.workdir / "guard_manifest.json", manifest)
    primary = [population[int(index)] for index in primary_indexes]
    secondary = [population[int(index)] for index in secondary_indexes]
    parent_model = new_model(device)
    pr233.load_checkpoint_into_model(parent_model, p1_checkpoint)
    parent_state = _state(parent_model)
    p1 = ArtifactEvaluator(p1_artifact)
    references = {}
    for name, items in (("primary", primary), ("secondary", secondary)):
        references[name] = {
            str(budget): [_root_record(item, p1, budget) for item in items]
            for budget in (384, 1200)
        }
    reference_cache = {
        "primary": references["primary"],
        "secondary": references["secondary"],
    }
    reference_cache_sha = _write_json(
        args.workdir / "p1_root_q_reference_cache.json", reference_cache
    )
    self_errors = [
        root_q_diagnostics(low["summary"], high["summary"])[constraint_key]
        for low, high in zip(
            references["primary"]["384"], references["primary"]["1200"], strict=True
        )
    ]
    epsilon_q = float(np.mean(self_errors))
    calibration = {
        "constraint_key": constraint_key,
        "epsilon_constraint": epsilon_q,
        "median": float(np.median(self_errors)),
        "p90": float(np.percentile(self_errors, 90)),
        "max": float(np.max(self_errors)),
        "best_q_disagreement_rate": float(
            np.mean(
                [
                    root_q_diagnostics(low["summary"], high["summary"])[
                        "best_q_action_disagreement"
                    ]
                    for low, high in zip(
                        references["primary"]["384"],
                        references["primary"]["1200"],
                        strict=True,
                    )
                ]
            )
        ),
    }
    source, plan = (
        np.load(pr233.A16_WORKDIR / "train_source_indexes.npy"),
        np.load(pr233.A16_WORKDIR / "batch_indexes.npy"),
    )
    baseline = _train_baseline(parent_state, rows, source, plan, device)
    expected_a16 = torch.load(
        pr233.A16_WORKDIR / "beta095/snapshots/step_0016.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    baseline_parity = all(
        torch.equal(baseline[16][key], expected_a16[key]) for key in baseline[16]
    )
    pure = torch.load(
        pr233.A16_WORKDIR / "pure_search/snapshots/step_0046.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    model = new_model(device)
    model.load_state_dict(parent_state)
    apply_trainable_scope(model, "policy_adapter_only")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=0.0)
    captures, telemetry = {}, []
    for step, indexes in enumerate(plan[:16], 1):
        before_metrics = _metrics(_state(model), parent_state, rows, pure)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, parent_state, rows, source, indexes, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        delta = adam_proposal_and_restore(parameters, optimizer)
        current = _state(model)
        full_norm = float(
            torch.linalg.vector_norm(torch.cat([value.reshape(-1) for value in delta]))
        )
        trial_errors, trial_secondary = {}, {}
        for scale in LAMBDAS:
            apply_delta(parameters, delta, scale)
            trial_state = _state(model)
            error, diagnostics = _guard_searches(
                trial_state,
                primary,
                references["primary"]["1200"],
                args.workdir,
                f"{args.statistic}_step_{step:02d}_lambda_{scale}",
                constraint_key,
            )
            _, secondary_diagnostics = _guard_searches(
                trial_state,
                secondary,
                references["secondary"]["1200"],
                args.workdir,
                f"{args.statistic}_step_{step:02d}_secondary_{scale}",
                constraint_key,
            )
            trial_errors[scale], trial_secondary[scale] = error, secondary_diagnostics
            model.load_state_dict(current)
        scale = choose_lambda(trial_errors, epsilon_q)
        apply_delta(parameters, delta, scale)
        after_state = _state(model)
        after_metrics = _metrics(after_state, parent_state, rows, pure)
        telemetry.append(
            {
                "step": step,
                "full_adam_proposal_norm": full_norm,
                "selected_lambda": scale,
                "applied_parameter_delta_norm": full_norm * scale,
                "epsilon_constraint": epsilon_q,
                "trial_constraint_error": {
                    str(key): value for key, value in trial_errors.items()
                },
                "accepted_constraint_error": trial_errors[scale],
                "primary_ce_before": float(loss.detach()),
                "primary_ce_after": float(
                    _loss(model, parent_state, rows, source, indexes, device).detach()
                ),
                "fit_fraction_before": before_metrics["fit_fraction"],
                "fit_fraction_after": after_metrics["fit_fraction"],
                "policy_l1_vs_p1": after_metrics["policy_l1_vs_p1"],
                "adapter_norm": after_metrics["adapter_norm"],
                "secondary_trial_diagnostics": {
                    str(key): value for key, value in trial_secondary.items()
                },
            }
        )
        if step in STEPS:
            captures[step] = after_state
    constrained_metrics = {
        str(step): _metrics(state, parent_state, rows, pure)
        for step, state in captures.items()
    }
    artifacts = {
        str(step): str(
            export(
                state,
                args.workdir / "artifacts" / f"step_{step:04d}",
                f"root_q_trust_{step}",
            )
        )
        for step, state in captures.items()
    }
    frozen_primary, frozen_controls = pr233._frozen_hashes()
    frozen_manifest = {
        row["state_hash"]: row for row in json.loads(pr233.MANIFEST.read_text())["rows"]
    }
    by_hash = {
        state_hash: {
            "state_hash": state_hash,
            "replay_index": int(meta["replay_index"]),
            "state": pr233.decode_kalah_v3_base_state(
                list(rows[int(meta["replay_index"])]["state"])
            ),
        }
        for state_hash, meta in frozen_manifest.items()
    }
    amplified_items = [by_hash[key] for key in sorted(frozen_primary)]
    washed_items = [by_hash[key] for key in sorted(frozen_controls)]
    amplified_reference = [
        _root_record(item, p1, SIMULATIONS) for item in amplified_items
    ]
    washed_reference = [_root_record(item, p1, SIMULATIONS) for item in washed_items]
    frozen_raw = {
        "amplified": _frozen_diagnostic(
            captures[16],
            amplified_items,
            amplified_reference,
            args.workdir,
            "frozen_amplified",
        ),
        "washed": _frozen_diagnostic(
            captures[16], washed_items, washed_reference, args.workdir, "frozen_washed"
        ),
    }
    frozen = {
        "amplified": frozen_raw["amplified"]
        | {
            "rescue_rate": float(
                np.mean(
                    [
                        record["candidate_move"] == record["p1_move"]
                        for record in frozen_raw["amplified"]["records"]
                    ]
                )
            ),
            "changed_root_hashes": [
                record["state_hash"]
                for record in frozen_raw["amplified"]["records"]
                if record["candidate_move"] != record["p1_move"]
            ],
        },
        "washed": frozen_raw["washed"]
        | {
            "new_divergence_rate": float(
                np.mean(
                    [
                        record["candidate_move"] != record["p1_move"]
                        for record in frozen_raw["washed"]["records"]
                    ]
                )
            ),
            "changed_root_hashes": [
                record["state_hash"]
                for record in frozen_raw["washed"]["records"]
                if record["candidate_move"] != record["p1_move"]
            ],
        },
    }
    generalization = {
        "primary": _guard_searches(
            captures[16],
            primary,
            references["primary"]["1200"],
            args.workdir,
            f"{args.statistic}_generalization_primary",
            constraint_key,
        )[1],
        "secondary": _guard_searches(
            captures[16],
            secondary,
            references["secondary"]["1200"],
            args.workdir,
            f"{args.statistic}_generalization_secondary",
            constraint_key,
        )[1],
        "amplified": frozen["amplified"]["q_diagnostics"],
        "washed": frozen["washed"]["q_diagnostics"],
    }
    activity = {
        f"fraction_lambda_{str(value).rstrip('0').rstrip('.').replace('.', '_')}": float(
            np.mean([row["selected_lambda"] == value for row in telemetry])
        )
        for value in LAMBDAS
    }
    scales = [row["selected_lambda"] for row in telemetry]
    zero_runs, current_zero_run = [], 0
    for scale in scales:
        current_zero_run = current_zero_run + 1 if scale == 0 else 0
        zero_runs.append(current_zero_run)
    activity |= {
        "mean_lambda": float(np.mean(scales)),
        "median_lambda": float(np.median(scales)),
        "longest_run_lambda_0": max(zero_runs, default=0),
        "cumulative_applied_full_update_norm_ratio": float(
            sum(row["applied_parameter_delta_norm"] for row in telemetry)
            / sum(row["full_adam_proposal_norm"] for row in telemetry)
        ),
    }
    constrained_matches_baseline = all(
        torch.equal(captures[step][key], baseline[step][key])
        for step in STEPS
        for key in captures[step]
    )
    constrained = {
        "metrics": constrained_metrics,
        "telemetry": telemetry,
        "constraint_key": constraint_key,
        "epsilon_constraint": epsilon_q,
        "constraint_activity": activity,
        "invariants": {
            "adam_lambda_one_parity": baseline_parity,
            "constrained_matches_baseline_when_lambda_one": constrained_matches_baseline,
            "inherited_parameters_byte_identical": all(
                torch.equal(state[key], parent_state[key])
                for state in captures.values()
                for key in state
                if key not in ADAPTER_KEYS
            ),
        },
    }
    arena, p0_gate = {}, {}
    if not args.skip_arenas:
        suite, suite_hash = pr233._suite()
        for step, state in captures.items():
            if constrained_metrics[str(step)]["fit_fraction"] < 0.25:
                continue
            arena[str(step)] = {
                context: pr233._arena(
                    Path(artifacts[str(step)]),
                    p1_artifact,
                    context,
                    args.workdir / "arena" / str(step),
                    "candidate_vs_p1",
                    args.workers,
                    suite_hash,
                )
                for context in ("384:256", "1200:1200")
            }
            if all(value["safe"] for value in arena[str(step)].values()):
                p0_gate[str(step)] = {
                    context: pr233._arena(
                        Path(artifacts[str(step)]),
                        REPO_ROOT / "model-artifact/current",
                        context,
                        args.workdir / "arena" / str(step),
                        "candidate_vs_p0",
                        args.workers,
                        suite_hash,
                    )
                    for context in ("384:256", "1200:1200")
                }
    final_arena = arena.get("16", {})
    classification, follow_up = _classification(
        constrained, final_arena, frozen, constraint_key
    )
    summary = {
        "schema": "azlite_root_q_trust_region_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
        "constraint_statistic": constraint_key,
        "hashes": {
            "p1_checkpoint": sha256_file(p1_checkpoint),
            "replay": sha256_file(replay),
            "pr233_shadow_sensitivity_cache": sha256_file(cache_path),
            "guard_manifest": guard_manifest_sha,
            "p1_root_q_reference_cache": reference_cache_sha,
        },
        "exclusions": exclusions,
        "guard_manifest": manifest,
        "calibration": calibration,
        "guardrails": {
            "self_play_generated": False,
            "beta": BETA,
            "learning_rate": 1e-5,
            "trainable": list(ADAPTER_KEYS),
            "search": {
                "simulations": SIMULATIONS,
                "c_puct": 1.25,
                "fpu_mode": "zero",
                "normalize_values": False,
                "root_noise": False,
            },
            "lambda_grid": list(LAMBDAS),
        },
        "baseline": {
            "a16_step16_parity": baseline_parity,
            "metrics": {
                str(step): _metrics(state, parent_state, rows, pure)
                for step, state in baseline.items()
            },
        },
        "constrained": constrained | {"artifacts": artifacts},
        "frozen": frozen,
        "arena_matrix": arena,
        "p0_gate": p0_gate,
        "generalization": generalization,
    }
    _write_json(args.workdir / "summary.json", summary)
    stem = (
        "root-q-rank-trust-region"
        if args.statistic == "per_action_q_rank"
        else "root-q-trust-region"
    )
    _write_json(
        REPO_ROOT / f"docs/data/alphazero-lite-fresh-p1-{stem}-summary.json", summary
    )
    title = (
        "Per-Action Q-Rank Trust Region"
        if args.statistic == "per_action_q_rank"
        else "Root-Q Direction Trust Region"
    )
    report = f"# {title}\n\n**Classification:** `{classification}`\n\n**Recommended follow-up:** {follow_up}\n\n## Calibration\n\n```json\n{json.dumps(calibration, indent=2, sort_keys=True)}\n```\n\n## Constraint Activity\n\n```json\n{json.dumps(activity, indent=2, sort_keys=True)}\n```\n\n## Results\n\n```json\n{json.dumps({'baseline': summary['baseline'], 'constrained_metrics': constrained_metrics, 'frozen': frozen, 'arena_matrix': arena, 'p0_gate': p0_gate, 'generalization': generalization}, indent=2, sort_keys=True)}\n```\n"
    (REPO_ROOT / f"docs/alphazero-lite-fresh-p1-{stem}-results.md").write_text(report)
    print(classification)


if __name__ == "__main__":
    main()
