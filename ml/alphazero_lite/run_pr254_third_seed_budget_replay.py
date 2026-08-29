#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the sealed third seed-47 768-versus-1024 replay experiment.

This runner deliberately has only reused, fresh768, and fresh1024 lanes.  It
never constructs semantic hybrids or selects training rows from attribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
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

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import run_fresh_p1_onpolicy_shadow_replay as replay
from ml.alphazero_lite import run_pr241_optimizer_isolation_reproduction as contract
from ml.alphazero_lite import run_pr241_policy_target_noise_isolation as isolation
from ml.alphazero_lite import run_pr242_target_entropy_factorization as pr244
from ml.alphazero_lite import run_pr249_fresh_suite_generalization as pr249
from ml.alphazero_lite import run_pr250_cross_seed_adapter_gradient_audit as pr250
from ml.alphazero_lite import run_pr251_cross_seed_strength_residual_transfer as pr251
from ml.alphazero_lite import run_pr252_phase_target_delta_attribution as pr252
from ml.alphazero_lite import run_pr253_semantic_receiver_target_surgery as pr253
from ml.alphazero_lite.evaluation_metrics import (
    paired_effect_difference,
    paired_opening_candidate_effect,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    export,
    new_model,
)
from ml.alphazero_lite.run_fresh_p1_shadow_target_distillation import _frozen_diagnostic
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    Node,
    PUCT,
    build_policy_target,
    build_search_profile,
    derive_self_play_value_target,
    encode_state,
    outcome_for_player,
    sample_move,
    standard_start_state,
    trajectory_hash_for_encoded_states,
)
from ml.alphazero_lite.train import load_checkpoint_into_model

GAMES, WORKERS, SEED, BASE, MAX_MOVES = 700, 24, 47, 384, 200
LANES = ("reused", "fresh768", "fresh1024")
SUITE_SEEDS = {"M": 13042, "N": 14042, "O": 15042}
J_TO_L_SUITE_PATHS = {
    label: Path(
        f"/tmp/azlite_pr253_semantic_receiver_target_surgery/suites/suite_{label}.jsonl"
    )
    for label in ("J", "K", "L")
}
TARGET = Path("/tmp/azlite_fresh_p1_parent_adapter/artifacts/step_0016/checkpoint.npz")
A16_SHA = "f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff"
ADAM_SHA = "61d5719e75aae87d7c2ca7ed2c5b01871ac2ea1675a34c4a6c918c783894e8c7"
P1_SHA = "e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9"
TARGET_SHA = "8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34"
_EVALUATOR: CheckpointEvaluator | None = None


def fail(message: str) -> None:
    raise RuntimeError(f"invariant_failure: {message}")


def worker_for(index: int) -> int:
    cursor = 0
    for worker, count in enumerate([30] * 4 + [29] * 20):
        if cursor <= index < cursor + count:
            return worker
        cursor += count
    fail("game index outside contract")


def game_rng(index: int) -> random.Random:
    return random.Random(SEED * 1_000_003 + index + worker_for(index) * 9_973)


def init_worker(checkpoint: str) -> None:
    global _EVALUATOR
    _EVALUATOR = CheckpointEvaluator(Path(checkpoint), input_encoding="kalah_v3")


def search(
    game: KalahGame,
    rng: random.Random,
    root: Node | None,
    reuse: bool,
    simulations: int,
    noisy: bool,
    ply: int,
) -> tuple[list[float], Node, list[float]]:
    if _EVALUATOR is None:
        fail("worker evaluator not initialized")
    engine = PUCT(
        _EVALUATOR,
        simulations,
        1.25,
        rng,
        root=root,
        fpu_mode="zero",
        reuse_subtree=reuse,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
    )
    visits, result = engine.run(
        game,
        dirichlet_alpha=0.3 if noisy else None,
        dirichlet_epsilon=0.3 if noisy else 0.0,
    )
    prior = engine.root_summary()["root_prior_telemetry"]["after"]
    if prior is None:
        fail("missing root prior")
    return (
        build_policy_target(
            visits,
            legal_moves=game.possible_moves(),
            temperature=1.0 if ply < 10 else 0.1,
        ),
        result,
        [float(x) for x in prior],
    )


def generate_game(
    index: int,
) -> tuple[int, dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    game, rng, root, records = (
        KalahGame.from_state(standard_start_state()),
        game_rng(index),
        None,
        [],
    )
    for ply in range(MAX_MOVES):
        if game.over():
            break
        legal = game.possible_moves()
        if not legal:
            break
        before = rng.getstate()
        runs = {}
        for lane, budget in (("reused", BASE), ("fresh768", 768), ("fresh1024", 1024)):
            clone = random.Random()
            clone.setstate(before)
            runs[lane] = (
                *search(
                    game,
                    clone,
                    root if lane == "reused" else None,
                    lane == "reused",
                    budget,
                    ply < 10,
                    ply,
                ),
                clone,
            )
        priors = [runs[lane][2] for lane in LANES]
        if not all(np.array_equal(priors[0], item) for item in priors[1:]):
            fail(f"root-noise mismatch {index}:{ply}")
        records.append(
            {
                "state": encode_state(game.to_state(), input_encoding="kalah_v3"),
                "player": game.current_player,
                "move_index": ply,
                "legal_moves": legal,
                "search_value": runs["reused"][1].q_value,
                "policies": {lane: runs[lane][0] for lane in LANES},
            }
        )
        # Only the authoritative search clone advances gameplay RNG.
        move = sample_move(runs["reused"][0], legal, runs["reused"][3])
        root = runs["reused"][1].child_for_action(move)
        if not game.move(game.pit_index(move)):
            fail(f"illegal sampled move {index}:{ply}")
        rng = runs["reused"][3]
    if not game.over():
        fail(f"unterminated game {index}")
    winner = game.winner
    trajectory = trajectory_hash_for_encoded_states(
        [row["state"] for row in records], winner=winner
    )
    profile = build_search_profile(
        kind="self_play",
        player_mode="puct",
        simulations=BASE,
        c_puct=1.25,
        search_options={
            "fpu_mode": "zero",
            "reuse_subtree": True,
            "normalize_values": False,
            "root_policy_mode": "visit_count",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
        },
    )
    views = {lane: [] for lane in LANES}
    telemetry = []
    for row in records:
        common = {
            "state": row["state"],
            "value": derive_self_play_value_target(
                outcome_value=outcome_for_player(winner, row["player"]),
                search_value=row["search_value"],
                move_index=row["move_index"],
            ),
            "player": row["player"],
            "move_index": row["move_index"],
            "winner": winner,
            "teacher_source": "puct",
            "policy_target_mode": "default",
            "policy_target_actual_mode": "default",
            "value_target_mode": "default",
            "search_profile": profile,
            "search_profile_hash": profile["hash"],
            "teacher_search_profile": profile,
            "teacher_search_profile_hash": profile["hash"],
            "policy_target_noise_mode": "noisy",
            "action_sampling_noise_enabled": row["move_index"] < 10,
            "target_dirichlet_epsilon": 0.3 if row["move_index"] < 10 else 0.0,
            "sampling_dirichlet_epsilon": 0.3 if row["move_index"] < 10 else 0.0,
            "simulations": BASE,
            "dirichlet_alpha": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_sampling": 0.3 if row["move_index"] < 10 else 0.0,
            "dirichlet_epsilon_for_target": 0.3 if row["move_index"] < 10 else 0.0,
            "legal_moves": row["legal_moves"],
            "game_index": index,
            "game_completed": True,
            "game_length": len(records),
            "trajectory_hash": trajectory,
        }
        for lane in LANES:
            views[lane].append({**common, "policy": row["policies"][lane]})
        telemetry.append(
            {
                "game_index": index,
                "move_index": row["move_index"],
                "policies": row["policies"],
            }
        )
    return index, views, telemetry


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def generate(
    workdir: Path, workers: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=init_worker, initargs=(str(TARGET),)
    ) as executor:
        completed = list(executor.map(generate_game, range(GAMES), chunksize=1))
    completed.sort()
    views = {
        lane: [row for _, batch, _ in completed for row in batch[lane]]
        for lane in LANES
    }
    telemetry = [row for _, _, items in completed for row in items]
    for lane, rows in views.items():
        write_jsonl(workdir / "generated" / f"{lane}.jsonl", rows)
    write_jsonl(workdir / "generated" / "telemetry.jsonl", telemetry)
    return views, telemetry


def non_policy_sha(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [{k: v for k, v in row.items() if k != "policy"} for row in rows],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def batch_plan_sha(rows: list[dict[str, Any]]) -> str:
    plan = [batch.tolist() for batch in replay.batches(rows)]
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def consumed_opening_exclusions() -> tuple[set[str], dict[str, set[str]]]:
    paths = {
        "canonical": pr249.CANONICAL_SUITE,
        **pr253.CONSUMED_SUITE_PATHS,
        **J_TO_L_SUITE_PATHS,
    }
    groups = {
        "consumed_evaluation_openings": set().union(
            *(replay.canonical_arena_hashes(path) for path in paths.values())
        )
    }
    return groups["consumed_evaluation_openings"], groups


def geometry(
    rows: list[dict[str, Any]], telemetry: list[dict[str, Any]]
) -> dict[str, Any]:
    policies = {lane: [row["policies"][lane] for row in telemetry] for lane in LANES}
    result = {}
    for left, right in (
        ("fresh768", "fresh1024"),
        ("reused", "fresh768"),
        ("reused", "fresh1024"),
    ):
        result[f"{left}_vs_{right}"] = isolation.summarize(
            [
                isolation.policy_stats(a, b, row["legal_moves"])
                for row, a, b in zip(rows, policies[left], policies[right], strict=True)
            ]
        )
    movement = defaultdict(int)
    for row, a, b in zip(
        rows, policies["fresh768"], policies["fresh1024"], strict=True
    ):
        legal = row["legal_moves"]
        movement[
            "unchanged_top1"
            if legal[int(np.argmax(np.asarray(a)[legal]))]
            == legal[int(np.argmax(np.asarray(b)[legal]))]
            else "1024_flip"
        ] += 1
        if not np.array_equal(a, b):
            movement["probability_only_movement"] += 1
    result["768_to_1024_transition"] = {
        key: value / len(rows) for key, value in movement.items()
    }
    return result


def train_lanes(
    rows: dict[str, list[dict[str, Any]]],
    a16: dict[str, torch.Tensor],
    adam: dict[str, Any],
    parent: dict[str, torch.Tensor],
    workdir: Path,
) -> tuple[
    dict[str, Any], dict[str, Path], dict[str, dict[int, dict[str, torch.Tensor]]]
]:
    initial = replay.optimizer_state_sha256(adam)
    targets = {lane: [row["policy"] for row in values] for lane, values in rows.items()}
    result = {}
    artifacts = {}
    states = {}
    for lane in LANES:
        snapshots, optimizers, invocation = contract.run_lane(
            rows[lane], a16, adam, parent
        )
        states[lane] = snapshots
        metrics = replay.metrics(
            rows[lane], snapshots, parent, a16, copy.deepcopy(adam)
        )
        for step, state in snapshots.items():
            metrics[str(step)]["ce_all_target_families"] = isolation.all_target_metrics(
                rows[lane], state, targets
            )
            metrics[str(step)]["model_sha256"] = pr252.model_sha(state)
            metrics[str(step)]["optimizer_sha256"] = replay.optimizer_state_sha256(
                optimizers[step]
            )
            output = workdir / "train" / lane / f"step_{step:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": state, "optimizer": optimizers[step]},
                output.with_suffix(".pt"),
            )
        result[lane] = {"optimizer_invocation": invocation, "metrics": metrics}
        artifacts[lane] = export(
            snapshots[16], workdir / "train" / lane / "step_0016", f"pr254_{lane}"
        )
    repeated = contract.repeated_lane_check(
        "fresh768", rows["fresh768"], a16, adam, parent
    )
    if not repeated or replay.optimizer_state_sha256(adam) != initial:
        fail("optimizer contamination or repeated identity failure")
    result["optimizer_invariants"] = {
        "initial_sha256": initial,
        "pristine_unchanged": True,
        "repeated_fresh768": repeated,
    }
    return result, artifacts, states


def seal_suites(
    workdir: Path, seed47_rows: list[dict[str, Any]]
) -> tuple[dict[str, Path], dict[str, Any]]:
    old_paths = {"canonical": pr249.CANONICAL_SUITE, **pr253.CONSUMED_SUITE_PATHS}
    old = {name: suites.load_suite_jsonl(str(path)) for name, path in old_paths.items()}
    used = set().union(*(pr249.suite_keys(entries) for entries in old.values()))
    old_prefixes = set().union(
        *(pr251.prefix_keys(entries) for entries in old.values())
    )
    replay_states = set().union(
        *pr249.replay_states().values(), {tuple(row["state"]) for row in seed47_rows}
    )
    universe = [
        entry
        for entry in pr249.all_openings()
        if tuple(encode_state(entry["state"], input_encoding="kalah_v3"))
        not in replay_states
    ]
    paths = {}
    manifest = {
        "consumed": {name: sha256_file(path) for name, path in old_paths.items()},
        "suites": {},
        "replay_state_exclusions": {"seed45": True, "seed46": True, "seed47": True},
    }
    prefixes = set()
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
        current = pr251.prefix_keys(selected)
        if len(keys) != 128 or keys & used or current & (old_prefixes | prefixes):
            fail(f"suite overlap {label}")
        path = workdir / "suites" / f"suite_{label}.jsonl"
        suites.write_suite_jsonl(selected, str(path))
        paths[label] = path
        used |= keys
        prefixes |= current
        manifest["suites"][label] = {
            "seed": seed,
            "sha256": sha256_file(path),
            "openings": 128,
            "consumed": True,
        }
    return paths, manifest


def evaluate(
    artifacts: dict[str, Path], paths: dict[str, Path], workdir: Path
) -> dict[str, Any]:
    result = {}
    for label, suite in paths.items():
        control = isolation.arena_records(
            workdir / "arena" / label,
            pr249.P1_ARTIFACT,
            pr249.P1_ARTIFACT,
            "1200:1200",
            "p1_control",
            WORKERS,
            suite,
        )
        result[label] = {"candidates": {}}
        for lane, artifact in artifacts.items():
            records = isolation.arena_records(
                workdir / "arena" / label,
                artifact,
                pr249.P1_ARTIFACT,
                "1200:1200",
                lane,
                WORKERS,
                suite,
            )
            result[label]["candidates"][lane] = {
                "effect": paired_opening_candidate_effect(records, control),
                "wdl": pr244.win_draw_loss(records),
            }
    return result


def analyze(evaluation: dict[str, Any]) -> dict[str, Any]:
    output = {"absolute": {}, "contrasts": {}}
    for lane in LANES:
        output["absolute"][lane] = {
            label: {
                "effect": item["candidates"][lane]["effect"]["paired_candidate_effect"],
                "ci": item["candidates"][lane]["effect"]["opening_bootstrap_ci"],
                "p0": item["candidates"][lane]["effect"]["p0_effect"],
                "p1": item["candidates"][lane]["effect"]["p1_effect"],
                "wdl": item["candidates"][lane]["wdl"],
            }
            for label, item in evaluation.items()
        }
    for name, pair in {
        "fresh768_minus_fresh1024": ("fresh768", "fresh1024"),
        "reused_minus_fresh768": ("reused", "fresh768"),
        "reused_minus_fresh1024": ("reused", "fresh1024"),
    }.items():
        values = []
        suites_values = []
        per_suite = {}
        for label, item in evaluation.items():
            diff = paired_effect_difference(
                item["candidates"][pair[0]]["effect"],
                item["candidates"][pair[1]]["effect"],
            )
            sample = np.asarray(list(diff["per_opening_effect"].values()))
            values.extend(sample)
            suites_values.append(sample)
            per_suite[label] = float(sample.mean())
        values = np.asarray(values)
        rng = np.random.default_rng(42)
        draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
        hierarchical = np.asarray(
            [
                rng.choice(suites_values[rng.integers(0, 3)], 128, replace=True).mean()
                for _ in range(10000)
            ]
        )
        output["contrasts"][name] = {
            "per_suite": per_suite,
            "pooled": {
                "effect": float(values.mean()),
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
                "same_sign_suites": sum(
                    np.sign(v) == np.sign(values.mean()) for v in per_suite.values()
                ),
            },
            "hierarchical_suite_opening_ci": {
                "lower_95": float(np.quantile(hierarchical, 0.025)),
                "upper_95": float(np.quantile(hierarchical, 0.975)),
            },
        }
    return output


def attribution(
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    a16: dict[str, torch.Tensor],
    parent: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor]:
    # The frozen PR252 analytic method is reused, without any surgery or training.
    pos, neg = (
        pr252.mixed_targets(positive, parent),
        pr252.mixed_targets(negative, parent),
    )
    delta_search = np.asarray([r["policy"] for r in positive]) - np.asarray(
        [r["policy"] for r in negative]
    )
    if not np.allclose(
        pos.astype(np.float64) - neg.astype(np.float64),
        0.05 * delta_search,
        atol=3e-7,
        rtol=0,
    ):
        fail("beta target delta")
    model = new_model(torch.device("cpu"))
    model.load_state_dict(a16)
    model.eval()
    with torch.no_grad():
        features = model.trunk_features(
            torch.tensor(np.asarray([r["state"] for r in positive], dtype=np.float32))
        ).double()
    weight = (
        torch.tensor(pos.astype(np.float64) - neg.astype(np.float64))[:, :, None]
        * features[:, None, :]
    )
    bias = torch.tensor(pos.astype(np.float64) - neg.astype(np.float64))
    indexes = np.concatenate(replay.batches(positive))
    reconstruction = torch.cat(
        (weight[indexes].sum(0).reshape(-1), bias[indexes].sum(0).reshape(-1))
    ) / len(indexes)
    dg = (
        pr250.full_batch_gradient(positive, a16, parent)["vector"].double()
        - pr250.full_batch_gradient(negative, a16, parent)["vector"].double()
    )
    error = float(
        torch.linalg.vector_norm(reconstruction + dg) / torch.linalg.vector_norm(dg)
    )
    similarity = pr252.cosine(reconstruction, -dg)
    if similarity < 0.999999 or error > 1e-5:
        fail(f"analytic gradient reconstruction {similarity} {error}")
    direction = -dg / torch.linalg.vector_norm(dg)
    scores = (
        torch.einsum("nad,ad->na", weight, direction[:-6].reshape_as(weight[0]))
        + bias * direction[-6:]
    ).numpy()
    absolute = np.abs(scores.sum(1))
    concentration = {
        f"top_{int(f * 100)}pct": float(
            np.sort(absolute)[-max(1, int(np.ceil(len(absolute) * f))) :].sum()
            / absolute.sum()
        )
        for f in (0.01, 0.05, 0.1, 0.2, 0.5)
    }
    return {
        "gradient_reconstruction": {"cosine": similarity, "relative_l2_error": error},
        "concentration": concentration,
        "_scores": scores,
        "_delta_search": delta_search,
    }, dg


def classify(analysis: dict[str, Any]) -> tuple[str, str | None]:
    item = analysis["contrasts"]["fresh768_minus_fresh1024"]["pooled"]
    clear = (item["lower_95"] > 0 or item["upper_95"] < 0) and item[
        "same_sign_suites"
    ] >= 2
    if not clear:
        return "third_seed_no_budget_split", None
    return (
        ("third_seed_768_stronger", "fresh768")
        if item["lower_95"] > 0
        else ("third_seed_1024_stronger", "fresh1024")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/azlite_pr254_third_seed_budget_replay"),
    )
    parser.add_argument("--freeze-suites-only", action="store_true")
    parser.add_argument(
        "--resume-generated",
        action="store_true",
        help="consume existing generated seed-47 JSONL files after an interrupted run",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    snapshot = torch.load(replay.A16_SNAPSHOT, map_location="cpu", weights_only=False)
    a16, adam = replay.immutable_initial_state(snapshot)
    if (
        args.workers != WORKERS
        or sha256_file(replay.A16_SNAPSHOT) != A16_SHA
        or replay.optimizer_state_sha256(adam) != ADAM_SHA
        or sha256_file(replay.P1_CHECKPOINT) != P1_SHA
        or sha256_file(TARGET) != TARGET_SHA
    ):
        fail("frozen artifact or worker mismatch")
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.freeze_suites_only:
        fail("seed47 replay is required before replay-disjoint suites can be frozen")
    if args.resume_generated:
        generated = {
            lane: read_jsonl(args.workdir / "generated" / f"{lane}.jsonl")
            for lane in LANES
        }
        telemetry = read_jsonl(args.workdir / "generated" / "telemetry.jsonl")
    else:
        generated, telemetry = generate(args.workdir, args.workers)
    blocked, groups = replay.exclusion_hashes(pr249.CANONICAL_SUITE)
    consumed, consumed_groups = consumed_opening_exclusions()
    blocked |= consumed
    groups = {**groups, **consumed_groups}
    eligible, exclusions = {}, {}
    for lane in LANES:
        eligible[lane], exclusions[lane] = replay.filter_rows(
            generated[lane], blocked, groups
        )
    hashes = {lane: non_policy_sha(rows) for lane, rows in eligible.items()}
    if len(set(hashes.values())) != 1 or any(
        exclusions[lane] != exclusions["reused"] for lane in LANES
    ):
        fail("replay-view mismatch or exclusions")
    isolation.assert_views(eligible["reused"], eligible)
    keys = {(r["game_index"], r["move_index"]) for r in eligible["reused"]}
    telemetry = [r for r in telemetry if (r["game_index"], r["move_index"]) in keys]
    paths, suite_manifest = seal_suites(args.workdir, eligible["reused"])
    (args.workdir / "frozen_manifest.json").write_text(
        json.dumps(
            {
                "suite_manifest": suite_manifest,
                "artifacts": {
                    "a16": A16_SHA,
                    "adam": ADAM_SHA,
                    "p1": P1_SHA,
                    "evaluator": TARGET_SHA,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    p1 = new_model(torch.device("cpu"))
    load_checkpoint_into_model(p1, replay.P1_CHECKPOINT)
    parent = {k: v.detach().cpu().clone() for k, v in p1.state_dict().items()}
    training, artifacts, states = train_lanes(eligible, a16, adam, parent, args.workdir)
    evaluation = evaluate(artifacts, paths, args.workdir)
    analysis = analyze(evaluation)
    classification, strong = classify(analysis)
    result = {
        "schema": "alphazero_lite_pr254_third_seed_budget_replay_v1",
        "classification": classification,
        "frozen_artifacts": {
            "a16_snapshot_sha256": A16_SHA,
            "initial_adam_sha256": ADAM_SHA,
            "p1_checkpoint_sha256": P1_SHA,
            "target_evaluator_sha256": TARGET_SHA,
        },
        "generation": {
            "games": GAMES,
            "workers": WORKERS,
            "seed": SEED,
            "lanes": {"reused": 384, "fresh768": 768, "fresh1024": 1024},
            "replay_sha256": {
                lane: sha256_file(args.workdir / "generated" / f"{lane}.jsonl")
                for lane in LANES
            },
            "state_outcome_sha256": hashes,
            "batch_plan_sha256": {
                lane: batch_plan_sha(rows) for lane, rows in eligible.items()
            },
        },
        "exclusions": exclusions["reused"],
        "suite_manifest": suite_manifest,
        "target_geometry": geometry(eligible["reused"], telemetry),
        "training": training,
        "frozen_40_40": {
            lane: _frozen_diagnostic(
                artifact,
                replay.P1_CHECKPOINT.parent / "artifact",
                read_jsonl(
                    Path("/tmp/azlite_onpolicy_shadow_replay/p1_reference.jsonl")
                ),
            )
            for lane, artifact in artifacts.items()
        },
        "primary_evaluation": analysis,
    }
    if strong:
        weak = "fresh1024" if strong == "fresh768" else "fresh768"
        attributed, dg = attribution(eligible[strong], eligible[weak], a16, parent)
        semantic = pr252.semantic_report(eligible[strong], attributed)
        historical = json.loads(
            Path(
                "docs/data/alphazero-lite-pr252-phase-target-delta-attribution-summary.json"
            ).read_text()
        )
        result["attribution"] = {
            k: v for k, v in attributed.items() if not k.startswith("_")
        }
        result["semantic_attribution"] = semantic
        result["three_seed_semantic_alignment"] = {
            seed: pr252.semantic_alignment(
                historical["semantic_attribution"][seed], semantic
            )
            for seed in ("seed45", "seed46")
        }
        r47 = pr250.vector(states[strong][16], a16) - pr250.vector(
            states[weak][16], a16
        )
        result["parameter_gradient_geometry"] = {
            "r47_norm": float(torch.linalg.vector_norm(r47)),
            "cosine_r47_negative_dg47": pr252.cosine(r47, -dg),
        }
    (args.workdir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(classification)


if __name__ == "__main__":
    main()
