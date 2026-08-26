#!/usr/bin/env python3
# ruff: noqa: E402
"""Test selection-CE training using the immutable PR #237 context cache."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite import run_fresh_p1_context_action_q_probe as mse
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.context_action_q_probe import CONTEXT_SIZE, ContextActionQProbe
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
)
from ml.alphazero_lite.self_play import PUCT, encode_state
from ml.alphazero_lite.train import (
    PolicyValueNet,
    input_size_for_encoding,
    load_checkpoint_into_model,
)

SEED = 238
SIMULATIONS = 1200
TEMPORAL_CONTEXT_SIZE = CONTEXT_SIZE + 4
WORKDIR = Path("/tmp/azlite_context_action_q_selection_probe")
SOURCE_WORKDIR = Path("/tmp/azlite_context_action_q_probe")
CACHE_SHA = "8cfa271f6aabcd0f812312e5243c8889c6e0c697b6aa4c1e52a53932e8377aa9"
SPLIT_SHA = "e6e9dfd9ccefea4d3ce792dbbdb583525e84c5a6bbf0f922c4328408c52ad0f3"
P1_SHA = "77969733ece5ced92d3a143a0fe9d82863ca3ec4faa477470ff5826ac22e4e12"
A16_SHA = "74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789"
REPLAY_SHA = "892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _source_artifacts() -> tuple[Path, Path, Path, Path]:
    replay = pr233.A16_WORKDIR / "fresh_p1_self_play.jsonl"
    p1 = pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/artifact"
    a16 = pr233.A16_WORKDIR / "artifacts/step_0016/artifact"
    cache = SOURCE_WORKDIR / "aligned_root_context_cache.jsonl"
    expected = {
        "aligned root context cache": (cache, CACHE_SHA),
        "P1 weights": (p1 / "weights.json", P1_SHA),
        "A16 weights": (a16 / "weights.json", A16_SHA),
        "replay": (replay, REPLAY_SHA),
        "split manifest": (SOURCE_WORKDIR / "split_manifest.json", SPLIT_SHA),
    }
    for name, (path, digest) in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"immutable PR #237 {name} mismatch")
    return replay, p1, a16, cache


def _decision_arrays(cache: Path, workdir: Path) -> dict[str, np.ndarray]:
    """Build compact, read-only decision arrays from the frozen JSONL cache."""
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / "decision_arrays.npz"
    if output.exists():
        with np.load(output) as stored:
            if {"previous_a16_q", "previous_a16_visits"} <= set(stored.files):
                return {key: stored[key] for key in stored.files}
    count = sum(1 for _ in _rows(cache))
    arrays = {
        "state_index": np.empty(count, np.uint16),
        "simulation": np.empty(count, np.uint16),
        "actual_move": np.empty(count, np.uint8),
        "legal": np.zeros((count, 6), bool),
        "a16_visits": np.zeros((count, 6), np.uint16),
        "a16_q": np.zeros((count, 6), np.float32),
        "u": np.zeros((count, 6), np.float32),
        "prior": np.zeros((count, 6), np.float32),
        "p1_visits": np.zeros((count, 6), np.uint16),
        "p1_q": np.zeros((count, 6), np.float32),
        "parent_visits": np.empty(count, np.uint16),
        "previous_a16_q": np.zeros((count, 6), np.float32),
        "previous_a16_visits": np.zeros((count, 6), np.uint16),
    }
    previous: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, row in enumerate(_rows(cache)):
        arrays["state_index"][index] = row["state_index"]
        arrays["simulation"][index] = row["simulation"]
        arrays["actual_move"][index] = row["actual_move"]
        arrays["parent_visits"][index] = row["parent_visits"]
        if int(row["state_index"]) in previous:
            arrays["previous_a16_q"][index], arrays["previous_a16_visits"][index] = (
                previous[int(row["state_index"])]
            )
        for child in row["children"]:
            action = int(child["action"])
            arrays["legal"][index, action] = True
            for key, source in (
                ("a16_visits", "a16_visits"),
                ("a16_q", "a16_q"),
                ("u", "u"),
                ("prior", "prior"),
                ("p1_visits", "p1_visits"),
                ("p1_q", "p1_q"),
            ):
                arrays[key][index, action] = child[source]
        previous[int(row["state_index"])] = (
            arrays["a16_q"][index].copy(),
            arrays["a16_visits"][index].copy(),
        )
    exact_q = np.where(
        (arrays["a16_visits"] > 0) & (arrays["p1_visits"] > 0),
        arrays["p1_q"],
        arrays["a16_q"],
    )
    scores = np.where(arrays["legal"], exact_q + arrays["u"], -np.inf)
    arrays["exact_move"] = np.argmax(scores, axis=1).astype(np.uint8)
    np.savez(output, **arrays)
    return arrays


def _features(
    model: PolicyValueNet, population: list[dict[str, Any]], device: torch.device
) -> np.ndarray:
    x = np.asarray(
        [encode_state(row["state"], input_encoding="kalah_v3") for row in population],
        np.float32,
    )
    with torch.no_grad():
        return model.trunk_features(torch.from_numpy(x).to(device)).cpu().numpy()


def _model_inputs(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    *,
    temporal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    batch = len(indexes)
    actions = np.broadcast_to(np.arange(6), (batch, 6))
    context = np.zeros(
        (batch, 6, TEMPORAL_CONTEXT_SIZE if temporal else CONTEXT_SIZE), np.float32
    )
    context[np.arange(batch)[:, None], actions, actions] = 1.0
    context[:, :, 6] = arrays["a16_q"][indexes]
    context[:, :, 7] = np.log1p(arrays["a16_visits"][indexes]) / np.log1p(SIMULATIONS)
    context[:, :, 8] = arrays["prior"][indexes]
    context[:, :, 9] = arrays["u"][indexes]
    context[:, :, 10] = arrays["parent_visits"][indexes, None] / SIMULATIONS
    if temporal:
        previous_q = arrays["previous_a16_q"][indexes]
        previous_visits = arrays["previous_a16_visits"][indexes]
        context[:, :, 11] = previous_q
        context[:, :, 12] = np.log1p(previous_visits) / np.log1p(SIMULATIONS)
        context[:, :, 13] = arrays["a16_q"][indexes] - previous_q
        context[:, :, 14] = (
            arrays["a16_visits"][indexes] - previous_visits
        ) / SIMULATIONS
    return np.repeat(
        features[arrays["state_index"][indexes]], 6, axis=0
    ), context.reshape(-1, context.shape[-1])


def _scores(
    probe: ContextActionQProbe,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    temporal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values, corrections = [], []
    probe.eval()
    with torch.no_grad():
        for start in range(0, len(indexes), 8192):
            current = indexes[start : start + 8192]
            h, context = _model_inputs(arrays, current, features, temporal=temporal)
            delta = (
                probe(
                    torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
                )
                .cpu()
                .numpy()
                .reshape(-1, 6)
            )
            delta = np.where(arrays["a16_visits"][current] > 0, delta, 0.0)
            score = arrays["a16_q"][current] + delta + arrays["u"][current]
            values.append(np.where(arrays["legal"][current], score, -np.inf))
            corrections.append(delta)
    return np.concatenate(values), np.concatenate(corrections)


def _selection_ce(
    probe: ContextActionQProbe,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    device: torch.device,
    *,
    temporal: bool = False,
) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indexes), 8192):
            current = indexes[start : start + 8192]
            h, context = _model_inputs(arrays, current, features, temporal=temporal)
            delta = probe(
                torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
            ).reshape(-1, 6)
            visited = torch.from_numpy(arrays["a16_visits"][current] > 0).to(device)
            legal = torch.from_numpy(arrays["legal"][current]).to(device)
            score = torch.from_numpy(
                arrays["a16_q"][current] + arrays["u"][current]
            ).to(device) + torch.where(visited, delta, torch.zeros_like(delta))
            score = score.masked_fill(~legal, float("-inf"))
            total += float(
                F.cross_entropy(
                    score,
                    torch.from_numpy(arrays["exact_move"][current]).to(device),
                    reduction="sum",
                ).cpu()
            )
    return total / len(indexes)


def _metrics(
    arrays: dict[str, np.ndarray], indexes: np.ndarray, scores: np.ndarray
) -> dict[str, float | int | None]:
    chosen = np.argmax(scores, axis=1)
    actual = arrays["actual_move"][indexes]
    exact = arrays["exact_move"][indexes]
    flip = exact != actual
    regret_scores = np.where(
        arrays["legal"][indexes],
        np.where(
            (arrays["a16_visits"][indexes] > 0) & (arrays["p1_visits"][indexes] > 0),
            arrays["p1_q"][indexes],
            arrays["a16_q"][indexes],
        )
        + arrays["u"][indexes],
        -np.inf,
    )
    regret = (
        regret_scores[np.arange(len(indexes)), exact]
        - regret_scores[np.arange(len(indexes)), actual]
    )
    nonflip = ~flip
    return {
        "exact_flip_action_recall": float(np.mean(chosen[flip] == exact[flip]))
        if flip.any()
        else None,
        "exact_flip_detection_rate": float(np.mean(chosen[flip] != actual[flip]))
        if flip.any()
        else None,
        "nonflip_preservation_rate": float(np.mean(chosen[nonflip] == actual[nonflip]))
        if nonflip.any()
        else None,
        "false_flip_rate": float(np.mean(chosen[nonflip] != actual[nonflip]))
        if nonflip.any()
        else None,
        "overall_exact_parent_action_agreement": float(np.mean(chosen == exact)),
        "exact_parent_score_regret_captured": float(
            regret[flip & (chosen == exact)].sum() / regret[flip].sum()
        )
        if regret[flip].sum()
        else None,
        "exact_parent_flip_count": int(flip.sum()),
    }


def _strata(
    arrays: dict[str, np.ndarray], indexes: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    actual, exact = arrays["actual_move"][indexes], arrays["exact_move"][indexes]
    exact_scores = np.where(
        arrays["legal"][indexes],
        np.where(
            (arrays["a16_visits"][indexes] > 0) & (arrays["p1_visits"][indexes] > 0),
            arrays["p1_q"][indexes],
            arrays["a16_q"][indexes],
        )
        + arrays["u"][indexes],
        -np.inf,
    )
    regret = (
        exact_scores[np.arange(len(indexes)), exact]
        - exact_scores[np.arange(len(indexes)), actual]
    )
    chosen = np.argmax(scores, axis=1)
    result: dict[str, Any] = {}
    flip = exact != actual
    edges = np.quantile(regret[flip], [0.25, 0.5, 0.75])
    for number, selected in enumerate(
        np.array_split(np.argsort(regret[flip], kind="stable"), 4), 1
    ):
        positions = np.flatnonzero(flip)[selected]
        result[f"q{number}"] = {
            "regret_range": [
                float(0 if number == 1 else edges[number - 2]),
                float(edges[number - 1] if number < 4 else regret[flip].max()),
            ],
            "exact_parent_action_recall": float(
                np.mean(chosen[positions] == exact[positions])
            ),
            "flip_detection": float(np.mean(chosen[positions] != actual[positions])),
            "regret_capture": float(
                regret[positions][chosen[positions] == exact[positions]].sum()
                / regret[positions].sum()
            ),
        }
    return result


def _scale(
    corrections: np.ndarray, arrays: dict[str, np.ndarray], indexes: np.ndarray
) -> dict[str, Any]:
    visited = arrays["a16_visits"][indexes] > 0
    synchronized = visited & (arrays["p1_visits"][indexes] > 0)
    predicted = corrections[visited]
    exact = (arrays["p1_q"][indexes] - arrays["a16_q"][indexes])[synchronized]

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max_abs": float(np.abs(values).max()),
        }

    windows = {}
    for name, mask in (
        ("1_384", arrays["simulation"][indexes] <= 384),
        ("385_1200", arrays["simulation"][indexes] >= 385),
    ):
        windows[name] = {
            "mean_centered_a16_q_std": float(
                np.mean(
                    np.nanstd(
                        np.where(
                            arrays["legal"][indexes][mask],
                            arrays["a16_q"][indexes][mask],
                            np.nan,
                        ),
                        axis=1,
                    )
                )
            ),
            "mean_centered_correction_std": float(
                np.mean(
                    np.nanstd(
                        np.where(
                            arrays["legal"][indexes][mask], corrections[mask], np.nan
                        ),
                        axis=1,
                    )
                )
            ),
        }
    return {
        "predicted_delta_q": summary(predicted),
        "exact_p1_minus_a16_delta_q": summary(exact),
        "centered_scale_by_simulation_window": windows,
    }


def _classification(
    gate: bool,
    live: dict[str, Any] | None,
    mse_metrics: dict[str, Any],
    ce_metrics: dict[str, Any],
) -> tuple[str, str]:
    if not gate:
        improved = (
            ce_metrics["exact_flip_action_recall"]
            > mse_metrics["exact_flip_action_recall"]
            and ce_metrics["exact_parent_score_regret_captured"]
            > mse_metrics["exact_parent_score_regret_captured"]
        )
        return (
            (
                "selection_loss_improves_but_insufficient",
                "add temporal root-trajectory features while KEEPING this same selection loss.",
            )
            if improved
            else (
                "instantaneous_context_insufficient",
                "test a richer root-trajectory representation containing recent Q/visit/backup history, without another policy-network expansion.",
            )
        )
    if live is None:
        return "inconclusive", "audit the gated live frozen-root evaluation."
    if live["rescue_rate"] >= 0.70 and live["new_divergence_rate"] <= 0.10:
        return (
            "selection_aligned_objective_recovers_parent_actions",
            "version the learned correction into a diagnostic root-search rule and run the canonical arena before changing model architecture.",
        )
    return (
        "selection_probe_live_unstable",
        "train on on-policy corrected-search trajectories rather than static ordinary A16 traces.",
    )


def _frozen_items(replay_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    manifest = {
        row["state_hash"]: row for row in json.loads(pr233.MANIFEST.read_text())["rows"]
    }
    amplified, washed = pr233._frozen_hashes()
    return {
        name: [
            {
                "state_hash": key,
                "state": pr233.decode_kalah_v3_base_state(
                    list(replay_rows[int(manifest[key]["replay_index"])]["state"])
                ),
            }
            for key in sorted(keys)
        ]
        for name, keys in (("amplified", amplified), ("washed", washed))
    }


def _offline_frozen(
    groups: dict[str, list[dict[str, Any]]],
    probe: ContextActionQProbe,
    trunk: PolicyValueNet,
    p1_path: Path,
    a16_path: Path,
    workdir: Path,
    device: torch.device,
    *,
    temporal: bool = False,
) -> dict[str, Any]:
    """Score frozen roots only after validation checkpoint selection is complete."""
    results = {}
    p1, a16 = ArtifactEvaluator(p1_path), ArtifactEvaluator(a16_path)
    for name, items in groups.items():
        cache = workdir / f"frozen_{name}_cache.jsonl"
        mse._make_cache(items, a16, p1, cache)
        arrays = _decision_arrays(cache, workdir / f"frozen_{name}")
        features = _features(trunk, items, device)
        indexes = np.arange(len(arrays["state_index"]))
        scores, _ = _scores(probe, arrays, indexes, features, device, temporal=temporal)
        results[name] = _metrics(arrays, indexes, scores)
    return results


def _live_frozen(
    groups: dict[str, list[dict[str, Any]]],
    probe: ContextActionQProbe,
    trunk: PolicyValueNet,
    p1_path: Path,
    a16_path: Path,
    device: torch.device,
    *,
    temporal: bool = False,
) -> dict[str, float]:
    """Apply the learned correction only at the live root's visited edges."""
    p1, a16 = ArtifactEvaluator(p1_path), ArtifactEvaluator(a16_path)
    totals = {"amplified": [0, 0], "washed": [0, 0]}
    for name, items in groups.items():
        for item in items:
            feature = _features(trunk, [item], device)
            ordinary, _unused, _ = mse._search(item, a16)
            parent_hash: str | None = None
            corrections: dict[tuple[int, int], float] = {}
            previous_q = np.zeros(6, np.float32)
            previous_visits = np.zeros(6, np.uint16)

            def capture(simulation: int, root: Any) -> None:
                nonlocal parent_hash, previous_q, previous_visits
                parent_hash = PUCT(a16, 1, 1.25, random.Random(1))._state_hash(
                    root.game
                )
                children = list(root.children.items())
                context = np.zeros(
                    (
                        len(children),
                        TEMPORAL_CONTEXT_SIZE if temporal else CONTEXT_SIZE,
                    ),
                    np.float32,
                )
                total_visits = max(1, sum(child.visit_count for _, child in children))
                for row, (move, child) in enumerate(children):
                    context[row, move] = 1.0
                    context[row, 6:11] = (
                        child.q_value,
                        np.log1p(child.visit_count) / np.log1p(SIMULATIONS),
                        child.prior,
                        1.25
                        * child.prior
                        * np.sqrt(total_visits)
                        / (1 + child.visit_count),
                        root.visit_count / SIMULATIONS,
                    )
                    if temporal:
                        context[row, 11:] = (
                            previous_q[move],
                            np.log1p(previous_visits[move]) / np.log1p(SIMULATIONS),
                            child.q_value - previous_q[move],
                            (child.visit_count - previous_visits[move]) / SIMULATIONS,
                        )
                with torch.no_grad():
                    delta = (
                        probe(
                            torch.from_numpy(
                                np.repeat(feature, len(children), axis=0)
                            ).to(device),
                            torch.from_numpy(context).to(device),
                        )
                        .cpu()
                        .numpy()
                    )
                corrections.update(
                    {
                        (simulation, move): float(value)
                        for (move, child), value in zip(children, delta, strict=True)
                        if child.visit_count > 0
                    }
                )
                previous_q = np.asarray(
                    [
                        root.children.get(move).q_value
                        if move in root.children
                        else 0.0
                        for move in range(6)
                    ],
                    np.float32,
                )
                previous_visits = np.asarray(
                    [
                        root.children.get(move).visit_count
                        if move in root.children
                        else 0
                        for move in range(6)
                    ],
                    np.uint16,
                )

            def override(
                simulation: int, state_hash: str, move: int, raw_q: float, visits: int
            ) -> float | None:
                if state_hash == parent_hash and visits > 0:
                    return raw_q + corrections.get((simulation, move), 0.0)
                return None

            corrected, _unused, _ = mse._search(
                item, a16, hook=capture, override=override
            )
            p1_summary, _unused, _ = mse._search(item, p1)
            if name == "amplified":
                totals[name][0] += (
                    corrected["selected_move"] == p1_summary["selected_move"]
                )
            else:
                totals[name][0] += (
                    corrected["selected_move"] != ordinary["selected_move"]
                )
            totals[name][1] += 1
    return {
        "rescue_rate": totals["amplified"][0] / totals["amplified"][1],
        "new_divergence_rate": totals["washed"][0] / totals["washed"][1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--temporal", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    replay, p1_path, a16_path, cache = _source_artifacts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_determinism(device, SEED)
    population, exclusions = pr233._population(pr233.read_jsonl(replay))
    split = json.loads((SOURCE_WORKDIR / "split_manifest.json").read_text())
    index_by_hash = {row["state_hash"]: index for index, row in enumerate(population)}
    train = np.asarray([index_by_hash[key] for key in split["train_hashes"]], np.uint16)
    validation_roots = set(split["validation_hashes"])
    arrays = _decision_arrays(cache, args.workdir)
    train_indexes = np.flatnonzero(np.isin(arrays["state_index"], train))
    validation_indexes = np.flatnonzero(
        np.asarray(
            [
                population[index]["state_hash"] in validation_roots
                for index in arrays["state_index"]
            ]
        )
    )
    checkpoint = (
        pr233.P1_WORKDIR / "beta_095/snapshot_artifacts/step_0046/checkpoint.npz"
    )
    trunk_size = int(np.load(checkpoint)["w_input"].shape[1])
    trunk = PolicyValueNet(
        (trunk_size, 3), "residual_v3", input_size_for_encoding("kalah_v3")
    ).to(device)
    load_checkpoint_into_model(trunk, checkpoint)
    trunk.eval()
    features = _features(trunk, population, device)
    context_size = TEMPORAL_CONTEXT_SIZE if args.temporal else CONTEXT_SIZE
    probe = ContextActionQProbe(trunk_size, context_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    best, best_state, history = float("inf"), None, []
    for step in range(1, 2001):
        sampled = train_indexes[rng.integers(0, len(train_indexes), 256)]
        h, context = _model_inputs(arrays, sampled, features, temporal=args.temporal)
        delta = probe(
            torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
        ).reshape(-1, 6)
        visited = torch.from_numpy(arrays["a16_visits"][sampled] > 0).to(device)
        legal = torch.from_numpy(arrays["legal"][sampled]).to(device)
        score = torch.from_numpy(arrays["a16_q"][sampled] + arrays["u"][sampled]).to(
            device
        ) + torch.where(visited, delta, torch.zeros_like(delta))
        loss = F.cross_entropy(
            score.masked_fill(~legal, float("-inf")),
            torch.from_numpy(arrays["exact_move"][sampled]).to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            validation_ce = _selection_ce(
                probe,
                arrays,
                validation_indexes,
                features,
                device,
                temporal=args.temporal,
            )
            history.append({"step": step, "validation_selection_ce": validation_ce})
            if validation_ce < best:
                best = validation_ce
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                }
    assert best_state is not None
    probe.load_state_dict(best_state)
    checkpoint_path = args.workdir / "context_action_q_selection_probe.pt"
    torch.save(
        {
            "state_dict": best_state,
            "trunk_size": trunk_size,
            "context_size": context_size,
        },
        checkpoint_path,
    )
    scores, corrections = _scores(
        probe, arrays, validation_indexes, features, device, temporal=args.temporal
    )
    ce_metrics = _metrics(arrays, validation_indexes, scores)
    mse_probe = ContextActionQProbe(trunk_size).to(device)
    mse_probe.load_state_dict(
        torch.load(
            SOURCE_WORKDIR / "context_action_q_probe.pt",
            map_location=device,
            weights_only=True,
        )["state_dict"]
    )
    mse_scores, _unused = _scores(
        mse_probe, arrays, validation_indexes, features, device
    )
    mse_metrics = _metrics(arrays, validation_indexes, mse_scores)
    gate = bool(
        ce_metrics["exact_flip_action_recall"] is not None
        and ce_metrics["exact_flip_action_recall"] >= 0.70
        and ce_metrics["nonflip_preservation_rate"] is not None
        and ce_metrics["nonflip_preservation_rate"] >= 0.90
        and ce_metrics["exact_parent_score_regret_captured"] is not None
        and ce_metrics["exact_parent_score_regret_captured"] >= 0.70
    )
    window = {
        name: _metrics(arrays, validation_indexes[mask], scores[mask])
        for name, mask in (
            ("1_384", arrays["simulation"][validation_indexes] <= 384),
            ("385_1200", arrays["simulation"][validation_indexes] >= 385),
        )
    }
    frozen_groups = _frozen_items(pr233.read_jsonl(replay))
    frozen_offline = _offline_frozen(
        frozen_groups,
        probe,
        trunk,
        p1_path,
        a16_path,
        args.workdir,
        device,
        temporal=args.temporal,
    )
    live = (
        _live_frozen(
            frozen_groups,
            probe,
            trunk,
            p1_path,
            a16_path,
            device,
            temporal=args.temporal,
        )
        if gate
        else None
    )
    classification, follow_up = _classification(gate, live, mse_metrics, ce_metrics)
    if args.temporal:
        follow_up = "test a longer root-trajectory representation with recent backup history while keeping the same selection loss."
    summary = {
        "schema": "azlite_context_action_q_temporal_selection_probe_v1"
        if args.temporal
        else "azlite_context_action_q_selection_probe_v1",
        "classification": classification,
        "recommended_follow_up": follow_up,
        "hashes": {
            "replay": REPLAY_SHA,
            "p1_weights": P1_SHA,
            "a16_weights": A16_SHA,
            "aligned_root_context_cache": CACHE_SHA,
            "split_manifest": SPLIT_SHA,
            "probe_checkpoint": sha256_file(checkpoint_path),
        },
        "exclusions": exclusions,
        "features_unchanged": not args.temporal,
        "temporal_features": [
            "previous_a16_q",
            "previous_log_normalized_visit_count",
            "a16_q_change",
            "visit_count_change",
        ]
        if args.temporal
        else [],
        "optimization": {
            "seed": SEED,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "steps": 2000,
            "batch_size_root_decisions": 256,
            "best_validation_selection_ce": best,
            "validation_history": history,
            "train_decisions": int(len(train_indexes)),
            "validation_decisions": int(len(validation_indexes)),
        },
        "invariants": {
            "zero_correction_ordinary_a16_winner": True,
            "exact_correction_parent_q_counterfactual": True,
            "unvisited_fpu_preserved": True,
            "puct_move_tie_order": True,
            "pre_simulation_p1_evidence_only": True,
        },
        "validation_selection_consequence": ce_metrics,
        "historical_mse_context_probe": mse_metrics,
        "selection_ce_minus_historical_mse": {
            key: float(ce_metrics[key]) - float(mse_metrics[key])
            for key in (
                "exact_flip_action_recall",
                "exact_parent_score_regret_captured",
                "nonflip_preservation_rate",
                "overall_exact_parent_action_agreement",
            )
        },
        "regret_quartiles_exact_parent_flips": _strata(
            arrays, validation_indexes, scores
        ),
        "simulation_windows": window,
        "correction_magnitude": _scale(corrections, arrays, validation_indexes),
        "validation_gate_passed": gate,
        "frozen_40_40_offline": frozen_offline,
        "live_frozen_root": live
        if live is not None
        else "not_run_validation_gate_failed",
        "guardrails": {
            "puct_changed": False,
            "p1_runtime": False,
            "arena_run": False,
            "loss": "masked_cross_entropy_only",
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    suffix = "-temporal" if args.temporal else ""
    output = (
        REPO_ROOT
        / f"docs/data/alphazero-lite-fresh-p1-context-action-q{suffix}-selection-probe-summary.json"
    )
    report = (
        REPO_ROOT
        / f"docs/alphazero-lite-fresh-p1-context-action-q{suffix}-selection-probe-results.md"
    )
    output.write_text(text)
    title = (
        "Temporal Selection-Aligned Context Action-Q Probe"
        if args.temporal
        else "Selection-Aligned Context Action-Q Probe"
    )
    report.write_text(
        f"# {title}\n\n**Classification:** `{classification}`\n\n**Recommended follow-up:** {follow_up}\n\n```json\n{text}```\n"
    )
    (args.workdir / "summary.json").write_text(text)
    print(classification)


if __name__ == "__main__":
    main()
