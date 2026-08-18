#!/usr/bin/env python3
# ruff: noqa: E402
"""Aggregate shard-gradient signal-to-noise and one-step convergence audit.

Determines whether PR #196's low (~0.37) pairwise Adam cosine reflects ordinary
512-row mini-batch stochasticity or genuinely different expected update
directions across replay shards.

No new self-play, no new replay source, no target/LR/loss-weight change, no
gradient surgery, and never more than one diagnostic optimizer step per lane.
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

from ml.alphazero_lite import arena
from ml.alphazero_lite.evaluation_metrics import paired_opening_candidate_effect
from ml.alphazero_lite.evaluation_seed_contract import (
    SEED_CONTRACT_VERSION,
    stable_seed,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
    verify_manifest,
)
from ml.alphazero_lite.run_game_shard_gradient_stability_audit import (
    ARENA_SUITE,
    CONTEXTS,
    CURRENT_HASH,
    GROUPS,
    SHARDS,
    cosine,
    deterministic_batches,
    export_artifact,
    fresh_state,
    new_model,
    pairwise_output_agreement,
    parameter_group,
    partition,
    puct_record,
    state_direction_agreement,
    state_hash,
)
from ml.alphazero_lite.run_opening_suite_seat_benchmark import (
    run_arena,
    parse_game_jsonl,
)
from ml.alphazero_lite.run_optimizer_aware_trunk_dynamics_audit import _batch, _losses
from ml.alphazero_lite.run_shared_trunk_delta_attribution import (
    _context_c_puct,
    decoded_validation_manifest,
    js,
    output_metrics,
    stable_hash,
)
from ml.alphazero_lite.train import (
    PolicyValueNet,
)

EFFECTIVE_BATCH_SIZES = (1, 2, 4, 8, 16)
AGGREGATE_REPLICATES = 8
BOOTSTRAP_REPLICATES = 1000
AGGREGATE_NAMESPACE = "azlite_aggregate_microbatches_v1"
EXPECTED_SHARDS = {
    "S0": (5835, "7ad2908e2e1902504d9909d321d0bbfbe4d38cfe6eefd972918f3dc193671fa3"),
    "S1": (5835, "b95a9459e3b14dc282c9e5d27127c040cb9a75ec45ea934e5fea14b6333ce79d"),
    "S2": (5849, "6453efea34baec757f4d0f1503c71fd2d0094b783c764888381619acdd1debf6"),
    "S3": (5847, "0aa9398933addc7b6b381a714e2460b6f00c9bf8596c4f760e433b78d745efc9"),
}


def aggregate_microbatches(
    indexes: list[int], shard: str, k: int, replicate: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed(AGGREGATE_NAMESPACE, shard, k, replicate))
    return [
        rng.choice(np.asarray(indexes), size=512, replace=len(indexes) < 512)
        for _ in range(k)
    ]


def batches_from_indexes(
    rows: list[dict[str, Any]], indexes: np.ndarray, device: torch.device
) -> list[dict[str, torch.Tensor]]:
    idx = np.asarray(indexes, dtype=np.int64)
    return [_batch(rows, idx[i : i + 512], device) for i in range(0, len(idx), 512)]


def mean_full_gradient(
    model: PolicyValueNet, batches: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Mean joint gradient over batches, accumulated without any optimizer step."""
    model.zero_grad(set_to_none=True)
    for batch in batches:
        policy, value = _losses(model, batch)
        (policy + value).backward()
    n = len(batches)
    grads: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = (
            p.grad.detach().clone() / n if p.grad is not None else torch.zeros_like(p)
        )
    return grads


def group_flat(grads: dict[str, torch.Tensor], group: str) -> torch.Tensor:
    return torch.cat(
        [grads[name].reshape(-1) for name in grads if parameter_group(name) == group]
    )


def clip_scale_of(grads: dict[str, torch.Tensor], clip: float) -> float:
    total_norm = torch.sqrt(
        torch.sum(torch.stack([(g.detach() ** 2).sum() for g in grads.values()]))
    )
    return float(torch.clamp(clip / (total_norm + 1e-6), max=1.0))


def aggregate_update(
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    grads: dict[str, torch.Tensor],
    device: torch.device,
    *,
    lr: float,
    clip: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """One fresh-state Adam step from a pre-aggregated, clipped gradient."""
    model = new_model(device)
    model.load_state_dict(state)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    for name, p in model.named_parameters():
        p.grad = grads[name].to(device).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    update = {
        group: torch.cat(
            [
                (p.detach() - before[name]).reshape(-1)
                for name, p in model.named_parameters()
                if parameter_group(name) == group
            ]
        )
        for group in GROUPS
    }
    new_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    return update, new_state


def bootstrap_ci(
    values: list[float], *, seed: int, n: int = 10_000
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def variance_decomposition(
    trunk_updates: dict[str, dict[int, list[torch.Tensor]]],
) -> dict[str, Any]:
    """Within-shard vs between-shard aggregate-update cosine, per effective batch size."""
    result: dict[str, Any] = {}
    for k in EFFECTIVE_BATCH_SIZES:
        within: list[float] = []
        between: list[float] = []
        for shard in SHARDS:
            reps = trunk_updates[shard][k]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    within.append(cosine(reps[i], reps[j]))
        for r in range(AGGREGATE_REPLICATES):
            for i in range(len(SHARDS)):
                for j in range(i + 1, len(SHARDS)):
                    between.append(
                        cosine(
                            trunk_updates[SHARDS[i]][k][r],
                            trunk_updates[SHARDS[j]][k][r],
                        )
                    )
        within_ci = bootstrap_ci(within, seed=301)
        between_ci = bootstrap_ci(between, seed=302)
        result[str(k)] = {
            "within_shard": within_ci,
            "between_shard": between_ci,
            "between_minus_within": {
                "mean": between_ci["mean"] - within_ci["mean"],
                "median": between_ci["median"] - within_ci["median"],
            },
        }
    return result


def shard_mean_direction(
    batches: dict[str, list[dict[str, torch.Tensor]]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Mean gradient over the 32 diagnostic batches per shard, then clip + Adam."""
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    grads_by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for shard in SHARDS:
        grads_by_shard[shard] = mean_full_gradient(model, batches[shard])
    grand_grads = {
        name: torch.stack([grads_by_shard[shard][name] for shard in SHARDS]).mean(0)
        for name in grads_by_shard["S0"]
    }
    grand_flat = {group: group_flat(grand_grads, group) for group in GROUPS}
    grand_clip_scale = clip_scale_of(grand_grads, clip)
    grand_update, grand_state = aggregate_update(
        state, optimizer_state, grand_grads, device, lr=lr, clip=clip
    )

    per_shard: dict[str, Any] = {}
    for shard in SHARDS:
        raw_flat = {group: group_flat(grads_by_shard[shard], group) for group in GROUPS}
        scale = clip_scale_of(grads_by_shard[shard], clip)
        clipped_flat = {group: raw_flat[group] * scale for group in GROUPS}
        update, _ = aggregate_update(
            state, optimizer_state, grads_by_shard[shard], device, lr=lr, clip=clip
        )
        per_shard[shard] = {
            "raw_trunk": raw_flat["shared_trunk"],
            "clipped_trunk": clipped_flat["shared_trunk"],
            "adam_trunk": update["shared_trunk"],
            "clip_scale": scale,
        }
    pairwise = {
        signal: {
            f"{left}<->{right}": cosine(
                per_shard[left][f"{signal}_trunk"], per_shard[right][f"{signal}_trunk"]
            )
            for oi, left in enumerate(SHARDS)
            for right in SHARDS[oi + 1 :]
        }
        for signal in ("raw", "clipped", "adam")
    }
    grand_cosines = {
        signal: {
            shard: cosine(
                per_shard[shard][f"{signal}_trunk"], grand_flat["shared_trunk"]
            )
            if signal != "adam"
            else cosine(per_shard[shard]["adam_trunk"], grand_update["shared_trunk"])
            for shard in SHARDS
        }
        for signal in ("raw", "clipped", "adam")
    }
    return {
        "pairwise_cosines": pairwise,
        "grand_mean_cosines": grand_cosines,
        "grand_mean_gradient_clip_scale": grand_clip_scale,
        "grand_mean_state": grand_state,
        "per_shard_grads": grads_by_shard,
    }


def bootstrap_shard_alignment(
    rows: list[dict[str, Any]],
    assignments: dict[str, list[int]],
    state: dict[str, torch.Tensor],
    optimizer_state: dict[str, Any],
    device: torch.device,
    manifest: dict[str, Any],
    *,
    seed: int = 203,
) -> list[float]:
    """Bootstrap whole games within each shard; median between-shard Adam cosine."""
    game_rows: dict[str, dict[str, list[int]]] = {}
    for shard, indexes in assignments.items():
        mapping: dict[str, list[int]] = defaultdict(list)
        for i in indexes:
            mapping[
                f"{rows[i]['game_index']}:{rows[i].get('trajectory_hash', '')}"
            ].append(i)
        game_rows[shard] = mapping
    rng = np.random.default_rng(seed)
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    medians: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        updates: dict[str, torch.Tensor] = {}
        for shard in SHARDS:
            mapping = game_rows[shard]
            sampled = rng.choice(list(mapping.keys()), size=len(mapping), replace=True)
            idx = [i for game in sampled for i in mapping[game]]
            grads = mean_full_gradient(
                model, batches_from_indexes(rows, np.asarray(idx), device)
            )
            update, _ = aggregate_update(
                state, optimizer_state, grads, device, lr=lr, clip=clip
            )
            updates[shard] = update["shared_trunk"]
        medians.append(
            float(
                np.median(
                    [
                        cosine(updates[left], updates[right])
                        for oi, left in enumerate(SHARDS)
                        for right in SHARDS[oi + 1 :]
                    ]
                )
            )
        )
    return medians


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr191-workdir", type=Path, default=Path("/tmp/azlite_shared_trunk_learning")
    )
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/azlite_aggregate_gradient_stability")
    )
    parser.add_argument(
        "--current", type=Path, default=REPO_ROOT / "model-artifact/current"
    )
    parser.add_argument("--puct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bootstrap", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT
        / "docs/data/alphazero-lite-aggregate-gradient-stability-summary.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "docs/alphazero-lite-aggregate-gradient-stability-results.md",
    )
    args = parser.parse_args()

    manifest = verify_manifest(args.pr191_workdir / "training_manifest.json")
    if sha256_file(args.current / "weights.json") != CURRENT_HASH:
        raise RuntimeError("current artifact does not match PR191")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, int(manifest["seed"]))

    all_rows = read_jsonl(Path(manifest["replay_path"]))
    source = np.load(
        manifest["artifact_paths"]["train_source_indexes"], allow_pickle=False
    )
    rows = [all_rows[int(index)] for index in source]
    assignments, shard_manifest = partition(rows)

    # Phase A — exact baseline reproduction.
    for shard in SHARDS:
        expected_rows, expected_hash = EXPECTED_SHARDS[shard]
        if len(assignments[shard]) != expected_rows:
            raise RuntimeError(
                f"{shard} row count {len(assignments[shard])} != expected {expected_rows}"
            )
        if shard_manifest["shards"][shard]["shard_sha256"] != expected_hash:
            raise RuntimeError(
                f"{shard} shard hash does not match the PR #196 baseline"
            )

    state, optimizer_state = fresh_state(manifest, device)
    diagnostic_batches = {
        name: [
            _batch(rows, indexes, device)
            for indexes in deterministic_batches(indexes, name)
        ]
        for name, indexes in assignments.items()
    }

    # Phases C/D — effective-batch-size curve + variance decomposition.
    trunk_updates: dict[str, dict[int, list[torch.Tensor]]] = {
        shard: {k: [] for k in EFFECTIVE_BATCH_SIZES} for shard in SHARDS
    }
    raw_trunk_grads: dict[str, dict[int, list[torch.Tensor]]] = {
        shard: {k: [] for k in EFFECTIVE_BATCH_SIZES} for shard in SHARDS
    }
    model = new_model(device)
    model.load_state_dict(state)
    model.eval()
    lr = float(manifest["optimizer"]["lr"])
    clip = float(manifest["gradient_clip"])
    for shard in SHARDS:
        indexes = assignments[shard]
        for k in EFFECTIVE_BATCH_SIZES:
            for replicate in range(AGGREGATE_REPLICATES):
                micro = aggregate_microbatches(indexes, shard, k, replicate)
                batches = [_batch(rows, idx, device) for idx in micro]
                grads = mean_full_gradient(model, batches)
                update, _ = aggregate_update(
                    state, optimizer_state, grads, device, lr=lr, clip=clip
                )
                trunk_updates[shard][k].append(update["shared_trunk"])
                raw_trunk_grads[shard][k].append(group_flat(grads, "shared_trunk"))
    variance = variance_decomposition(trunk_updates)
    raw_variance = variance_decomposition(raw_trunk_grads)
    curve = {
        str(k): {
            "between_shard_median_cosine": variance[str(k)]["between_shard"]["median"],
            "within_shard_median_cosine": variance[str(k)]["within_shard"]["median"],
        }
        for k in EFFECTIVE_BATCH_SIZES
    }

    # Phase E — shard-mean direction.
    shard_means = shard_mean_direction(
        diagnostic_batches, state, optimizer_state, device, manifest
    )

    # Phase F — whole-game bootstrap.
    if args.bootstrap:
        bootstrap_cache = args.workdir / "bootstrap_medians.json"
        if bootstrap_cache.is_file():
            bootstrap_medians = json.loads(bootstrap_cache.read_text(encoding="utf-8"))
        else:
            bootstrap_medians = bootstrap_shard_alignment(
                rows, assignments, state, optimizer_state, device, manifest
            )
            bootstrap_cache.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_cache.write_text(json.dumps(bootstrap_medians) + "\n")
    else:
        bootstrap_medians = []

    # Phase G — one-step aggregate candidates (S0, S1, S2, S3, grand mean).
    candidate_states: dict[str, dict[str, torch.Tensor]] = {}
    for shard in SHARDS:
        candidate_states[f"aggregate_{shard}"] = aggregate_update(
            state,
            optimizer_state,
            shard_means["per_shard_grads"][shard],
            device,
            lr=lr,
            clip=clip,
        )[1]
    candidate_states["aggregate_grand_mean"] = shard_means["grand_mean_state"]
    repeated: dict[str, dict[str, torch.Tensor]] = {}
    for shard in SHARDS:
        repeated[f"aggregate_{shard}"] = aggregate_update(
            state,
            optimizer_state,
            shard_means["per_shard_grads"][shard],
            device,
            lr=lr,
            clip=clip,
        )[1]
    repeated["aggregate_grand_mean"] = aggregate_update(
        state,
        optimizer_state,
        {
            name: torch.stack(
                [shard_means["per_shard_grads"][shard][name] for shard in SHARDS]
            ).mean(0)
            for name in shard_means["per_shard_grads"]["S0"]
        },
        device,
        lr=lr,
        clip=clip,
    )[1]
    for name in candidate_states:
        if state_hash(candidate_states[name]) != state_hash(repeated[name]):
            raise RuntimeError(f"{name} aggregate candidate is not deterministic")

    artifacts = {
        name: export_artifact(candidate_state, name, args.workdir)
        for name, candidate_state in candidate_states.items()
    }

    validation = np.load(
        manifest["artifact_paths"]["validation_source_indexes"], allow_pickle=False
    )
    probe, probe_manifest = decoded_validation_manifest(all_rows, validation)
    probe_rows = [all_rows[index] for index in probe_manifest["source_indexes"]]
    probe_hash = stable_hash(probe_manifest)
    all_states = {"C": state, **candidate_states}
    drift = output_metrics(probe_rows, all_states)

    summary: dict[str, Any] = {
        "schema": "azlite_aggregate_gradient_stability_v1",
        "guardrails": {
            "new_self_play": False,
            "new_replay_source": False,
            "target_change": False,
            "lr_change": False,
            "loss_weight_change": False,
            "gradient_surgery": False,
            "optimizer_steps_per_lane": 1,
            "promotion": False,
        },
        "inputs": {
            "replay_sha256": sha256_file(Path(manifest["replay_path"])),
            "current_weights_sha256": CURRENT_HASH,
            "train_rows": len(rows),
        },
        "phase_a": {
            "reproduced_exactly": True,
            "shard_row_counts": {shard: len(assignments[shard]) for shard in SHARDS},
            "shard_hashes": {
                shard: shard_manifest["shards"][shard]["shard_sha256"]
                for shard in SHARDS
            },
        },
        "phase_c_effective_batch_curve": curve,
        "phase_d_variance_decomposition": {
            "adam_update": variance,
            "raw_gradient": raw_variance,
        },
        "phase_e_shard_mean": {
            "pairwise_cosines": shard_means["pairwise_cosines"],
            "grand_mean_cosines": shard_means["grand_mean_cosines"],
        },
        "phase_f_bootstrap": (
            {
                "replicates": len(bootstrap_medians),
                "median_between_shard_cosine": bootstrap_ci(
                    bootstrap_medians, seed=304
                ),
            }
            if bootstrap_medians
            else {"replicates": 0}
        ),
        "phase_g_candidates": {
            name: {"state_sha256": state_hash(candidate_state)}
            for name, candidate_state in candidate_states.items()
        },
        "phase_h_output_drift": {
            "probe_manifest": probe_manifest,
            "metrics": drift,
            "pairwise_candidate_agreement": pairwise_output_agreement(
                probe_rows, candidate_states
            ),
        },
    }

    if args.puct:
        evaluator = {
            "C": arena.ArtifactEvaluator(args.current),
            **{name: arena.ArtifactEvaluator(path) for name, path in artifacts.items()},
        }
        cache_path = args.workdir / "puct_records.jsonl"
        cached: dict[tuple[str, str, str], dict[str, Any]] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                cached[(record["name"], record["context"], record["state_hash"])] = (
                    record["record"]
                )
        missing = []
        for context in CONTEXTS:
            for row in probe[:256]:
                for name, value in evaluator.items():
                    key = (name, context, row["state_hash"])
                    if key not in cached:
                        missing.append((key, row, value))
        if missing:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as handle:
                for key, row, value in missing:
                    record = puct_record(row, value, key[1], probe_hash)
                    cached[key] = record
                    handle.write(
                        json.dumps(
                            {
                                "name": key[0],
                                "context": key[1],
                                "state_hash": key[2],
                                "record": record,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        puct_metrics: dict[str, Any] = {}
        for context in CONTEXTS:
            puct_metrics[context] = {}
            for name in candidate_states:
                pairs = [
                    (
                        cached[("C", context, row["state_hash"])],
                        cached[(name, context, row["state_hash"])],
                    )
                    for row in probe[:256]
                ]
                puct_metrics[context][name] = {
                    "move_change_rate": float(
                        np.mean([a["move"] != b["move"] for a, b in pairs])
                    ),
                    "visit_js": float(
                        np.mean(
                            [
                                js(np.asarray([a["visit"]]), np.asarray([b["visit"]]))[
                                    0
                                ]
                                for a, b in pairs
                            ]
                        )
                    ),
                    "true_child_q_ranking_changes": float(
                        np.mean(
                            [
                                a["q_ranking_move_ids"] != b["q_ranking_move_ids"]
                                for a, b in pairs
                            ]
                        )
                    ),
                    "root_value_delta": float(
                        np.mean([b["root_value"] - a["root_value"] for a, b in pairs])
                    ),
                }
        summary["phase_h_puct"] = {
            "probe_states": 256,
            "metrics": puct_metrics,
            "state_direction_agreement": {
                context: state_direction_agreement(
                    cached, context, candidate_states, probe[:256]
                )
                for context in CONTEXTS
            },
        }

    if args.arena:
        suite_sha = sha256_file(ARENA_SUITE)
        arena_results: dict[str, Any] = {}
        for context in ("384:256", "1200:1200"):
            challenger_sims, current_sims = (int(value) for value in context.split(":"))
            arena_results[context] = {}
            for name, artifact in artifacts.items():
                candidate_records, control_records = [], []
                for role, challenger, records in (
                    ("candidate", artifact, candidate_records),
                    ("control", args.current, control_records),
                ):
                    for seat in (0, 1):
                        directory = (
                            args.workdir
                            / "arena"
                            / context.replace(":", "_")
                            / name
                            / role
                            / f"starts_{seat}"
                        )
                        directory.mkdir(parents=True, exist_ok=True)
                        records_path = directory / "games.jsonl"
                        existing = (
                            parse_game_jsonl(str(records_path))
                            if records_path.is_file()
                            else []
                        )
                        if len(existing) != 128:
                            run_arena(
                                challenger=str(challenger),
                                current=str(args.current),
                                challenger_sims=challenger_sims,
                                current_sims=current_sims,
                                games=128,
                                seed=191,
                                workers=args.workers,
                                out_json=str(directory / "arena.json"),
                                out_jsonl=str(records_path),
                                opening_prefixes_jsonl=str(ARENA_SUITE),
                                challenger_starts=seat,
                                games_per_opening=1,
                                root_policy_mode="deterministic",
                                root_temperature=0.0,
                                normalize_values=False,
                                c_puct=_context_c_puct(context),
                                tactical_root_bias=0.0,
                                seed_contract=SEED_CONTRACT_VERSION,
                                suite_sha256=suite_sha,
                                seed_ledger_output=str(directory / "seed_ledger.jsonl"),
                            )
                            existing = parse_game_jsonl(str(records_path))
                        records.extend(existing)
                effect = paired_opening_candidate_effect(
                    candidate_records,
                    control_records,
                    bootstrap_samples=10_000,
                    bootstrap_seed=191,
                )
                arena_results[context][name] = {
                    "paired_candidate_effect": effect["paired_candidate_effect"],
                    "opening_bootstrap_ci": effect["opening_bootstrap_ci"],
                }
        summary["phase_i_arena"] = {
            "arena": arena_results,
            "suite_sha256": suite_sha,
            "unique_openings": 128,
            "opening_bootstrap_samples": 10_000,
            "seed_contract": SEED_CONTRACT_VERSION,
        }

    summary["classification"] = classify(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary["classification"], sort_keys=True))
    return 0


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    curve = summary.get("phase_c_effective_batch_curve", {})
    k1 = curve["1"]["between_shard_median_cosine"]
    k16 = curve["16"]["between_shard_median_cosine"]
    cosine_rises_strongly = (k16 - k1) >= 0.25

    var = summary.get("phase_d_variance_decomposition", {}).get("adam_update", {})
    v16 = var.get("16", {})
    within16 = v16.get("within_shard", {})
    between16 = v16.get("between_shard", {})
    within_highly_stable = within16.get("median", 0.0) >= 0.75
    bw_clearly_negative = (
        between16.get("upper_95", 1.0) - within16.get("lower_95", 0.0) < 0
    )
    between_materially_worse = (
        bw_clearly_negative
        or v16.get("between_minus_within", {}).get("median", 0.0) <= -0.10
    )

    shard_mean_adam = list(
        summary["phase_e_shard_mean"]["pairwise_cosines"]["adam"].values()
    )
    shard_mean_adam_median = float(np.median(shard_mean_adam))
    highly_aligned = shard_mean_adam_median >= 0.75

    pairwise = summary.get("phase_h_output_drift", {}).get(
        "pairwise_candidate_agreement", {}
    )
    output_highly_aligned = bool(pairwise) and all(
        entry["policy_top1_agreement"] >= 0.99 for entry in pairwise.values()
    )

    arena = summary.get("phase_i_arena", {}).get("arena", {}).get("384:256", {})
    arena_effects = [item["paired_candidate_effect"] for item in arena.values()]
    arena_complete = len(arena_effects) == 5
    arena_varies = arena_complete and (max(arena_effects) - min(arena_effects)) >= 0.05
    arena_consistently_harmful = (
        arena_complete
        and all(item < 0 for item in arena_effects)
        and sum(item["opening_bootstrap_ci"]["upper_95"] < 0 for item in arena.values())
        >= 4
    )

    if cosine_rises_strongly and not between_materially_worse and highly_aligned:
        if arena_consistently_harmful:
            return {
                "label": "supervised_update_direction_harmful_and_stable",
                "next_action": "stop replay-size tuning; audit supervised objective/target semantics",
            }
        return {
            "label": "minibatch_noise_explains_low_cosine",
            "next_action": "stop replay-size tuning; move to supervised objective/target semantics",
        }

    if within_highly_stable and between_materially_worse and arena_varies:
        return {
            "label": "replay_distribution_instability_confirmed",
            "next_action": "investigate per-shard replay distribution differences",
        }

    if highly_aligned and output_highly_aligned and arena_varies:
        return {
            "label": "optimizer_step_stable_but_search_nonlinear",
            "next_action": "investigate PUCT nonlinear sensitivity",
        }

    return {
        "label": "inconclusive",
        "next_action": "no unregistered intervention",
        "evidence": {
            "cosine_rises_strongly": cosine_rises_strongly,
            "k1_between_cosine": k1,
            "k16_between_cosine": k16,
            "shard_mean_adam_median": shard_mean_adam_median,
            "within_highly_stable": within_highly_stable,
            "between_materially_worse": between_materially_worse,
            "between_minus_within_median": v16.get("between_minus_within", {}).get(
                "median", 0.0
            ),
            "arena_varies": arena_varies,
            "arena_consistently_harmful": arena_consistently_harmful,
            "output_highly_aligned": output_highly_aligned,
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# AlphaZero-Lite Aggregate Gradient Stability Audit",
        "",
        f"**Classification:** `{summary['classification']['label']}`",
        "",
        "Primary question: does the harmful 384:256 effect remain once mini-batch noise is averaged out?",
        "",
        "## Effective-batch-size curve",
        "",
        "| Effective rows | Between-shard median cosine | Within-shard median cosine |",
        "| ---: | ---: | ---: |",
    ]
    for k in EFFECTIVE_BATCH_SIZES:
        row = summary["phase_c_effective_batch_curve"][str(k)]
        lines.append(
            f"| {k * 512} | {row['between_shard_median_cosine']:.4f} | "
            f"{row['within_shard_median_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Shard-mean pairwise Adam-update cosines",
            "",
            "| Pair | Raw | Clipped | Adam |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    pc = summary["phase_e_shard_mean"]["pairwise_cosines"]
    for pair in pc["raw"]:
        lines.append(
            f"| {pair} | {pc['raw'][pair]:.4f} | {pc['clipped'][pair]:.4f} | "
            f"{pc['adam'][pair]:.4f} |"
        )
    if summary["phase_f_bootstrap"]["replicates"]:
        boot = summary["phase_f_bootstrap"]["median_between_shard_cosine"]
        lines.extend(
            [
                "",
                f"## Whole-game bootstrap (n={summary['phase_f_bootstrap']['replicates']})",
                "",
                f"Median between-shard Adam cosine: {boot['median']:.4f} "
                f"(95% CI [{boot['lower_95']:.4f}, {boot['upper_95']:.4f}])",
            ]
        )
    if "phase_i_arena" in summary:
        lines.extend(
            [
                "",
                "## Canonical arena (aggregate-step candidates)",
                "",
                "| Budget | Candidate | Paired effect | 95% CI |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for context, candidates in summary["phase_i_arena"]["arena"].items():
            for name, effect in candidates.items():
                ci = effect["opening_bootstrap_ci"]
                lines.append(
                    f"| {context} | {name} | {effect['paired_candidate_effect']:+.4f} | "
                    f"[{ci['lower_95']:+.4f}, {ci['upper_95']:+.4f}] |"
                )
    classification = summary["classification"]
    if "evidence" in classification:
        lines.extend(
            [
                "",
                "## Classification evidence",
                "",
                "| Signal | Value |",
                "| --- | ---: |",
            ]
        )
        for key, value in classification["evidence"].items():
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "Full evidence: `docs/data/alphazero-lite-aggregate-gradient-stability-summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
