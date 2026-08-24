#!/usr/bin/env python3
# ruff: noqa: E402
"""PR #234: refresh fixed-selector search targets before every Adam update.

This intentionally reuses PR #233's immutable population, selector, static
cache, replay plan, optimizer snapshot, and evaluation helpers. It generates
no self-play and never uses shadow search for candidate evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
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
from ml.alphazero_lite.online_target_refresh import (
    TorchEvaluator,
    model_state_digest,
    ordinary_target,
    shadow_target,
    target_seed,
)
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
    read_jsonl,
    sha256_file,
)
from ml.alphazero_lite.run_fresh_p1_parent_additive_policy_adapter import (
    BETA,
    export,
    new_model,
)
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.train import apply_trainable_scope

LANES = ("baseline_continue", "static_shadow", "online_ordinary", "online_shadow")
STEPS = (1, 4, 16)
WORKDIR = Path("/tmp/azlite_online_target_refresh")
_WORKER_MODEL: torch.nn.Module | None = None
_WORKER_P1: ArtifactEvaluator | None = None
_WORKER_LANE: str | None = None


def _init_target_worker(
    state: dict[str, torch.Tensor], p1_artifact: str, lane: str
) -> None:
    """Load one immutable pre-step candidate per CPU target-search worker."""
    global _WORKER_MODEL, _WORKER_P1, _WORKER_LANE
    torch.set_num_threads(1)
    _WORKER_MODEL = new_model(torch.device("cpu"))
    _WORKER_MODEL.load_state_dict(state)
    _WORKER_MODEL.eval()
    _WORKER_P1 = ArtifactEvaluator(Path(p1_artifact))
    _WORKER_LANE = lane


def _worker_target(item: dict[str, Any]) -> np.ndarray:
    if _WORKER_MODEL is None or _WORKER_P1 is None or _WORKER_LANE is None:
        raise RuntimeError("target worker was not initialized")
    game = KalahGame.from_state(item["state"])
    candidate = TorchEvaluator(_WORKER_MODEL, torch.device("cpu"), "kalah_v3")
    seed = target_seed(item["state_hash"], _WORKER_LANE)
    if _WORKER_LANE == "online_ordinary":
        return ordinary_target(game, candidate, seed=seed)
    return shadow_target(game, candidate, _WORKER_P1, seed=seed)


def _target_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    l1 = np.abs(left - right).sum(axis=1)
    return {
        "l1": pr233._summary(l1),
        "mean_js": float(np.mean([pr233._js(a, b) for a, b in zip(left, right)])),
        "top1_disagreement": float(
            np.mean(np.argmax(left, axis=1) != np.argmax(right, axis=1))
        ),
    }


def _anchor_positions(
    step: int, batch_size: int, anchor_order: np.ndarray
) -> np.ndarray:
    return anchor_order[
        ((step - 1) * batch_size + np.arange(batch_size)) % len(anchor_order)
    ]


def _generate_targets(
    lane: str,
    model: torch.nn.Module,
    device: torch.device,
    population: list[dict[str, Any]],
    positions: np.ndarray,
    p1: ArtifactEvaluator,
    workers: int = 1,
) -> np.ndarray:
    """Freeze targets from the exact candidate state supplied by the caller."""
    items = [population[int(position)] for position in positions]
    if workers > 1 and device.type == "cpu":
        state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_target_worker,
            initargs=(state, str(p1_artifact_path(p1)), lane),
        ) as executor:
            targets = list(executor.map(_worker_target, items))
    else:
        candidate = TorchEvaluator(model, device, "kalah_v3")
        targets = []
        for item in items:
            game = KalahGame.from_state(item["state"])
            seed = target_seed(item["state_hash"], lane)
            target = (
                ordinary_target(game, candidate, seed=seed)
                if lane == "online_ordinary"
                else shadow_target(game, candidate, p1, seed=seed)
            )
            targets.append(target)
    return np.asarray(targets, dtype=np.float64)


def p1_artifact_path(evaluator: ArtifactEvaluator) -> Path:
    """Recover the immutable parent artifact path supplied by this runner."""
    # ArtifactEvaluator intentionally does not retain its source path.
    return pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"


def _parity_gate(
    snapshot: dict[str, Any],
    device: torch.device,
    population: list[dict[str, Any]],
    cache: dict[str, np.ndarray],
    sensitive: np.ndarray,
    p1: ArtifactEvaluator,
) -> dict[str, bool]:
    """Require in-memory A16 searches to reproduce cached A16 searches exactly."""
    model = new_model(device)
    model.load_state_dict(snapshot["model"])
    candidate = TorchEvaluator(model, device, "kalah_v3")
    # Four roots cover the predetermined calibration anchor batches without
    # expanding the selector or generating any training data.
    selected = sensitive[:4]
    ordinary_ok, shadow_ok = True, True
    for position in selected:
        item = population[int(position)]
        game = KalahGame.from_state(item["state"])
        seed = pr233._seed(item["state_hash"])
        ordinary_ok &= np.array_equal(
            ordinary_target(game, candidate, seed=seed),
            cache["ordinary_a16"][position] / pr233.SIMULATIONS,
        )
        shadow_ok &= np.array_equal(
            shadow_target(game, candidate, p1, seed=seed),
            cache["shadow_a16"][position] / pr233.SIMULATIONS,
        )
    if not ordinary_ok or not shadow_ok:
        raise RuntimeError("current-candidate evaluator parity failure")
    return {
        "evaluator_parity_ordinary": ordinary_ok,
        "evaluator_parity_shadow": shadow_ok,
    }


def _train_lane(
    lane: str,
    snapshot: dict[str, Any],
    parent_state: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    source: np.ndarray,
    plan: np.ndarray,
    population: list[dict[str, Any]],
    sensitive: np.ndarray,
    cache: dict[str, np.ndarray],
    p1: ArtifactEvaluator,
    device: torch.device,
    weight: float,
    workers: int,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    model = new_model(device)
    model.load_state_dict(snapshot["model"])
    apply_trainable_scope(model, "policy_adapter_only")
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-5, weight_decay=0.0
    )
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    order = pr233._anchor_order("shadow_sensitive", sensitive)
    anchors = np.asarray(
        [population[i]["replay_index"] for i in sensitive], dtype=np.int64
    )
    static = cache["shadow_a16"][sensitive] / pr233.SIMULATIONS
    captures: dict[int, dict[str, torch.Tensor]] = {}
    telemetry: list[dict[str, Any]] = []
    for step, indexes in enumerate(plan[16:32], 1):
        positions = _anchor_positions(step, len(indexes), order)
        pre_update_digest = model_state_digest(model)
        if lane == "static_shadow":
            target = static
        elif lane.startswith("online_"):
            generated = _generate_targets(
                lane, model, device, population, sensitive[positions], p1, workers
            )
            target = np.zeros_like(static)
            target[positions] = generated
        else:
            target = None
        primary, auxiliary = pr233._losses_for_step(
            model,
            parent_state,
            rows,
            source,
            indexes,
            anchors,
            target,
            order,
            step,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        primary.backward(retain_graph=True)
        primary_gradient = pr233._gradient_vector(model).clone()
        if target is None:
            detail: dict[str, Any] = {
                "primary_norm": float(torch.linalg.vector_norm(primary_gradient)),
                "auxiliary_norm": 0.0,
                "raw_ratio": 0.0,
                "gradient_cosine": 0.0,
            }
        else:
            optimizer.zero_grad(set_to_none=True)
            auxiliary.backward(retain_graph=True)
            detail = pr233._gradient_pair(
                primary_gradient, pr233._gradient_vector(model)
            )
        detail["step"] = step
        detail["weighted_ratio"] = weight * detail["raw_ratio"]
        detail["pre_update_state"] = pre_update_digest
        if lane == "online_shadow":
            ordinary = _generate_targets(
                "online_ordinary",
                model,
                device,
                population,
                sensitive[positions],
                p1,
                workers,
            )
            detail["shadow_delta_from_ordinary"] = _target_metrics(generated, ordinary)[
                "mean_js"
            ]
        if detail["weighted_ratio"] > 2.0:
            raise RuntimeError("runtime auxiliary gradient scale failure")
        optimizer.zero_grad(set_to_none=True)
        (primary + weight * auxiliary).backward()
        optimizer.step()
        if lane.startswith("online_"):
            detail["target_drift_from_static_shadow"] = _target_metrics(
                generated, static[positions]
            )
            detail["target_drift_from_p1"] = _target_metrics(
                generated,
                cache["ordinary_p1"][sensitive[positions]] / pr233.SIMULATIONS,
            )
        telemetry.append(detail)
        if step in STEPS:
            captures[step] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    return captures, telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--skip-arenas", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    replay = pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl"
    checkpoint = (
        pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    artifact = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    snapshot = torch.load(
        pr233.A16_WORKDIR / "beta095/snapshots/step_0016.pt",
        map_location="cpu",
        weights_only=False,
    )
    rows = read_jsonl(replay)
    population, exclusions = pr233._population(rows)
    cache_path = Path("/tmp/azlite_shadow_target_distillation/target_cache.npz")
    selector_path = Path(
        "/tmp/azlite_shadow_target_distillation/selector_manifest.json"
    )
    if (
        sha256_file(cache_path) != pr233.TARGET_CACHE_SHA
        or sha256_file(selector_path) != pr233.SELECTOR_MANIFEST_SHA
    ):
        raise RuntimeError("immutable PR #233 cache or selector hash mismatch")
    cache_file = np.load(cache_path, allow_pickle=False)
    cache = {key: cache_file[key] for key in cache_file.files}
    sensitive = np.asarray(
        json.loads(selector_path.read_text())["sensitive_indexes"], dtype=np.int64
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, pr233.SEED)
    parent = new_model(device)
    pr233.load_checkpoint_into_model(parent, checkpoint)
    parent_state = {
        key: value.detach().clone() for key, value in parent.state_dict().items()
    }
    p1 = ArtifactEvaluator(artifact)
    invariants = _parity_gate(snapshot, device, population, cache, sensitive, p1)
    source, plan = (
        np.load(pr233.A16_WORKDIR / "train_source_indexes.npy"),
        np.load(pr233.A16_WORKDIR / "batch_indexes.npy"),
    )
    # Calibration is prospective: all targets are generated from unchanged A16.
    calibration: dict[str, list[dict[str, float]]] = {
        name: [] for name in ("static_shadow", "online_ordinary", "online_shadow")
    }
    order = pr233._anchor_order("shadow_sensitive", sensitive)
    anchors = np.asarray(
        [population[i]["replay_index"] for i in sensitive], dtype=np.int64
    )
    for batch in range(1, 5):
        positions = _anchor_positions(batch, len(plan[16 + batch - 1]), order)
        initial = new_model(device)
        initial.load_state_dict(snapshot["model"])
        generated = {
            "static_shadow": cache["shadow_a16"][sensitive] / pr233.SIMULATIONS,
            "online_ordinary": _generate_targets(
                "online_ordinary",
                initial,
                device,
                population,
                sensitive[positions],
                p1,
                args.workers,
            ),
            "online_shadow": _generate_targets(
                "online_shadow",
                initial,
                device,
                population,
                sensitive[positions],
                p1,
                args.workers,
            ),
        }
        for lane, values in generated.items():
            target = (
                values
                if lane == "static_shadow"
                else np.zeros((len(sensitive), 6), dtype=np.float64)
            )
            if lane != "static_shadow":
                target[positions] = values
            # Target positions above follow the immutable PR #233 sensitive
            # anchor order; calibration must use that same order.
            metric, _, _ = pr233._measure_gradient(
                "shadow_sensitive",
                snapshot,
                parent_state,
                rows,
                source,
                plan[16 + batch - 1],
                anchors,
                target,
                batch,
                device,
            )
            calibration[lane].append({"batch": batch, **metric})
    maximum = max(row["raw_ratio"] for values in calibration.values() for row in values)
    weight = pr233.calibrated_weight(maximum)
    lanes: dict[str, Any] = {}
    for lane in LANES:
        states, telemetry = _train_lane(
            lane,
            snapshot,
            parent_state,
            rows,
            source,
            plan,
            population,
            sensitive,
            cache,
            p1,
            device,
            weight,
            args.workers,
        )
        lanes[lane] = {"gradient_telemetry": telemetry, "metrics": {}, "artifacts": {}}
        for step, state in states.items():
            lanes[lane]["metrics"][str(step)] = pr233._model_metrics(
                state, parent_state, rows, sensitive, anchors, cache
            )
            lanes[lane]["artifacts"][str(step)] = str(
                export(
                    state,
                    args.workdir / "artifacts" / lane / f"step_{16 + step:04d}",
                    f"{lane}_{16 + step}",
                )
            )
    summary = {
        "schema": "azlite_online_target_refresh_v1",
        "classification": "inconclusive",
        "recommended_follow_up": "Complete ordinary-PUCT evaluation before selecting a lane.",
        "hashes": {
            "p1_checkpoint": sha256_file(checkpoint),
            "replay": sha256_file(replay),
            "target_cache": sha256_file(cache_path),
            "selector_manifest": sha256_file(selector_path),
        },
        "exclusions": exclusions,
        "calibration": {
            "lanes": calibration,
            "maximum_raw_ratio": maximum,
            "behavior_loss_weight": weight,
        },
        "guardrails": {
            "beta": BETA,
            "simulations": 1200,
            "dynamic_refresh": "every_optimizer_step",
            "shadow_used_only_for_targets": True,
        },
        "invariants": invariants
        | {
            "pre_update_targets": all(
                "pre_update_state" in row
                for value in lanes.values()
                for row in value["gradient_telemetry"]
            )
        },
        "lanes": lanes,
    }
    (args.workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(summary["classification"])


if __name__ == "__main__":
    main()
