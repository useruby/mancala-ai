#!/usr/bin/env python3
# ruff: noqa: E402
"""Test causal 32-event A16 root backup trajectories with selection CE only."""

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
from ml.alphazero_lite import run_fresh_p1_context_action_q_selection_probe as selection
from ml.alphazero_lite import run_fresh_p1_shadow_target_distillation as pr233
from ml.alphazero_lite.arena import ArtifactEvaluator, sha256_file
from ml.alphazero_lite.context_action_q_probe import CONTEXT_SIZE, ContextActionQProbe
from ml.alphazero_lite.run_deterministic_joint_heads_iteration import (
    configure_determinism,
)
from ml.alphazero_lite.self_play import PUCT
from ml.alphazero_lite.train import (
    PolicyValueNet,
    input_size_for_encoding,
    load_checkpoint_into_model,
)

SIMULATIONS = 1200
HISTORY_LENGTH = 32
EVENT_SIZE = 7
TRAJECTORY_SIZE = HISTORY_LENGTH * EVENT_SIZE
WORKDIR = Path("/tmp/azlite_context_action_q_trajectory_probe")
SOURCE_WORKDIR = Path("/tmp/azlite_context_action_q_probe")
HISTORY_CACHE_NAME = "a16_root_backup_history.jsonl"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _history_vector(
    history: list[dict[str, int | float]], *, include_values: bool
) -> np.ndarray:
    """Encode only completed root backups, oldest first, with left zero padding."""
    if len(history) > HISTORY_LENGTH:
        history = history[-HISTORY_LENGTH:]
    result = np.zeros((HISTORY_LENGTH, EVENT_SIZE), np.float32)
    offset = HISTORY_LENGTH - len(history)
    for index, event in enumerate(history, offset):
        action = int(event["action"])
        if not 0 <= action < 6:
            raise RuntimeError("root backup action is outside the six-way action space")
        result[index, action] = 1.0
        if include_values:
            result[index, 6] = float(event["root_value"])
    return result.reshape(-1)


def _trace_matches_cache(
    trace: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> bool:
    """Check every pre-selection A16 table against the immutable PR #237 cache."""
    if len(trace) != SIMULATIONS or len(expected) != SIMULATIONS:
        return False
    for observed, cached in zip(trace, expected, strict=True):
        root = observed["selection_path"][0]
        if (
            int(observed["simulation_index"]) != int(cached["simulation"])
            or int(root["chosen_move"]) != int(cached["actual_move"])
            or int(root["parent_visit_count"]) != int(cached["parent_visits"])
        ):
            return False
        actual = {int(child["move"]): child for child in root["children"]}
        for child in cached["children"]:
            current = actual.get(int(child["action"]))
            if current is None or int(current["visit_count"]) != int(
                child["a16_visits"]
            ):
                return False
            if not np.isclose(
                float(current["q_value"]), float(child["a16_q"]), rtol=0, atol=0
            ):
                return False
            if not np.isclose(
                float(current["u_component"]), float(child["u"]), rtol=0, atol=0
            ):
                return False
            if not np.isclose(
                float(current["prior"]), float(child["prior"]), rtol=0, atol=0
            ):
                return False
    return True


def _history_cache(
    population: list[dict[str, Any]],
    a16: ArtifactEvaluator,
    path: Path,
    *,
    immutable_context: Path | None = None,
) -> str:
    """Rerun ordinary A16 and persist its completed-backup history per root."""
    completed = {row["state_hash"] for row in _rows(path)} if path.exists() else set()
    expected_rows = _rows(immutable_context) if immutable_context is not None else None
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as output:
        for state_index, item in enumerate(population):
            expected = (
                [next(expected_rows) for _ in range(SIMULATIONS)]
                if expected_rows is not None
                else []
            )
            if item["state_hash"] in completed:
                continue
            history: list[dict[str, int | float]] = []
            summary, trace, _unused = mse._search(
                item,
                a16,
                trace=immutable_context is not None,
                root_backup_history=history,
            )
            if len(history) != SIMULATIONS:
                raise RuntimeError(
                    "ordinary A16 did not produce 1200 completed root backups"
                )
            if immutable_context is not None and not _trace_matches_cache(
                trace, expected
            ):
                raise RuntimeError(
                    "ordinary A16 root decision cache reproduction failed"
                )
            visits = [int(child["visits"]) for child in summary["child_stats"]]
            q_values = [float(child["q_value"]) for child in summary["child_stats"]]
            output.write(
                json.dumps(
                    {
                        "state_index": state_index,
                        "state_hash": item["state_hash"],
                        "history": history,
                        "final_visits": visits,
                        "final_q": q_values,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
    if expected_rows is not None:
        try:
            next(expected_rows)
        except StopIteration:
            pass
        else:
            raise RuntimeError("immutable context cache population differs")
    return sha256_file(path)


def _trajectory_arrays(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Store the compact causal history; one-hot expansion is deferred to batches."""
    actions = np.full((count, SIMULATIONS, HISTORY_LENGTH), 255, np.uint8)
    values = np.zeros((count, SIMULATIONS, HISTORY_LENGTH), np.float32)
    seen = set()
    for row in _rows(path):
        state_index = int(row["state_index"])
        if (
            state_index in seen
            or not 0 <= state_index < count
            or len(row["history"]) != SIMULATIONS
        ):
            raise RuntimeError("invalid root backup history cache")
        seen.add(state_index)
        history = row["history"]
        for simulation in range(SIMULATIONS):
            # At simulation t use history[0:t], never history[t].
            recent = history[max(0, simulation - HISTORY_LENGTH) : simulation]
            offset = HISTORY_LENGTH - len(recent)
            for position, event in enumerate(recent, offset):
                actions[state_index, simulation, position] = int(event["action"])
                values[state_index, simulation, position] = float(event["root_value"])
    if len(seen) != count:
        raise RuntimeError("incomplete root backup history cache")
    return actions, values


def _model_inputs(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    trajectories: tuple[np.ndarray, np.ndarray],
    *,
    include_values: bool,
    transform: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    h, instantaneous = selection._model_inputs(arrays, indexes, features)
    actions, values = trajectories
    history_actions = actions[
        arrays["state_index"][indexes], arrays["simulation"][indexes] - 1
    ].copy()
    history_values = values[
        arrays["state_index"][indexes], arrays["simulation"][indexes] - 1
    ].copy()
    if transform == "reverse":
        history_actions = history_actions[:, ::-1]
        history_values = history_values[:, ::-1]
    elif transform == "permuted":
        for row, index in enumerate(indexes):
            rng = np.random.default_rng(
                int(arrays["state_index"][index]) * 1201
                + int(arrays["simulation"][index])
            )
            order = rng.permutation(HISTORY_LENGTH)
            history_actions[row] = history_actions[row, order]
            history_values[row] = history_values[row, order]
    elif transform == "zero":
        history_actions.fill(255)
        history_values.fill(0)
    elif transform != "normal":
        raise ValueError(f"unknown history transform: {transform}")
    history = np.zeros((len(indexes), HISTORY_LENGTH, EVENT_SIZE), np.float32)
    valid = history_actions != 255
    rows, positions = np.nonzero(valid)
    history[rows, positions, history_actions[valid]] = 1.0
    if include_values:
        history[:, :, 6] = history_values
    flattened = np.repeat(history.reshape(len(indexes), -1), 6, axis=0)
    return h, np.concatenate((instantaneous, flattened), axis=1)


def _scores(
    probe: ContextActionQProbe,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    trajectories: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    include_values: bool,
    transform: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    scores, corrections = [], []
    probe.eval()
    with torch.no_grad():
        for start in range(0, len(indexes), 8192):
            current = indexes[start : start + 8192]
            h, context = _model_inputs(
                arrays,
                current,
                features,
                trajectories,
                include_values=include_values,
                transform=transform,
            )
            delta = (
                probe(
                    torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
                )
                .cpu()
                .numpy()
                .reshape(-1, 6)
            )
            delta = np.where(arrays["a16_visits"][current] > 0, delta, 0.0)
            scores.append(
                np.where(
                    arrays["legal"][current],
                    arrays["a16_q"][current] + delta + arrays["u"][current],
                    -np.inf,
                )
            )
            corrections.append(delta)
    return np.concatenate(scores), np.concatenate(corrections)


def _selection_ce(
    probe: ContextActionQProbe,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    features: np.ndarray,
    trajectories: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    *,
    include_values: bool,
) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(indexes), 8192):
            current = indexes[start : start + 8192]
            h, context = _model_inputs(
                arrays, current, features, trajectories, include_values=include_values
            )
            delta = probe(
                torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
            ).reshape(-1, 6)
            visited = torch.from_numpy(arrays["a16_visits"][current] > 0).to(device)
            legal = torch.from_numpy(arrays["legal"][current]).to(device)
            scores = torch.from_numpy(
                arrays["a16_q"][current] + arrays["u"][current]
            ).to(device) + torch.where(visited, delta, torch.zeros_like(delta))
            total += float(
                F.cross_entropy(
                    scores.masked_fill(~legal, float("-inf")),
                    torch.from_numpy(arrays["exact_move"][current]).to(device),
                    reduction="sum",
                ).cpu()
            )
    return total / len(indexes)


def _train(
    name: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    validation: np.ndarray,
    features: np.ndarray,
    trajectories: tuple[np.ndarray, np.ndarray],
    trunk_size: int,
    device: torch.device,
    workdir: Path,
) -> tuple[ContextActionQProbe, dict[str, Any]]:
    include_values = name == "trajectory_actions_values"
    configure_determinism(device, seed)
    probe = ContextActionQProbe(trunk_size, CONTEXT_SIZE + TRAJECTORY_SIZE).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    best, best_state, history = float("inf"), None, []
    for step in range(1, 2001):
        sampled = train[rng.integers(0, len(train), 256)]
        h, context = _model_inputs(
            arrays, sampled, features, trajectories, include_values=include_values
        )
        delta = probe(
            torch.from_numpy(h).to(device), torch.from_numpy(context).to(device)
        ).reshape(-1, 6)
        visited = torch.from_numpy(arrays["a16_visits"][sampled] > 0).to(device)
        legal = torch.from_numpy(arrays["legal"][sampled]).to(device)
        scores = torch.from_numpy(arrays["a16_q"][sampled] + arrays["u"][sampled]).to(
            device
        ) + torch.where(visited, delta, torch.zeros_like(delta))
        loss = F.cross_entropy(
            scores.masked_fill(~legal, float("-inf")),
            torch.from_numpy(arrays["exact_move"][sampled]).to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            value = _selection_ce(
                probe,
                arrays,
                validation,
                features,
                trajectories,
                device,
                include_values=include_values,
            )
            history.append({"step": step, "validation_selection_ce": value})
            if value < best:
                best, best_state = (
                    value,
                    {
                        key: value.detach().cpu().clone()
                        for key, value in probe.state_dict().items()
                    },
                )
    assert best_state is not None
    probe.load_state_dict(best_state)
    checkpoint = workdir / f"{name}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "trunk_size": trunk_size,
            "context_size": CONTEXT_SIZE + TRAJECTORY_SIZE,
        },
        checkpoint,
    )
    scores, corrections = _scores(
        probe,
        arrays,
        validation,
        features,
        trajectories,
        device,
        include_values=include_values,
    )
    return probe, {
        "seed": seed,
        "best_validation_selection_ce": best,
        "validation_history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics": selection._metrics(arrays, validation, scores),
        "scores": scores,
        "corrections": corrections,
    }


def _gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["exact_flip_action_recall"] >= 0.70
        and metrics["nonflip_preservation_rate"] >= 0.90
        and metrics["exact_parent_score_regret_captured"] >= 0.70
    )


def _frozen_offline(
    probe: ContextActionQProbe,
    trunk: PolicyValueNet,
    replay_rows: list[dict[str, Any]],
    p1_path: Path,
    a16_path: Path,
    device: torch.device,
    workdir: Path,
) -> dict[str, Any]:
    """Score frozen groups after checkpoint selection, never during training."""
    results = {}
    groups = selection._frozen_items(replay_rows)
    for name, items in groups.items():
        cache = workdir / f"frozen_{name}_context.jsonl"
        mse._make_cache(
            items, ArtifactEvaluator(a16_path), ArtifactEvaluator(p1_path), cache
        )
        history = workdir / f"frozen_{name}_{HISTORY_CACHE_NAME}"
        _history_cache(items, ArtifactEvaluator(a16_path), history)
        arrays = selection._decision_arrays(cache, workdir / f"frozen_{name}_arrays")
        trajectories = _trajectory_arrays(history, len(items))
        features = selection._features(trunk, items, device)
        indexes = np.arange(len(arrays["state_index"]))
        scores, _unused = _scores(
            probe, arrays, indexes, features, trajectories, device, include_values=True
        )
        results[name] = selection._metrics(arrays, indexes, scores)
    return results


def _live_frozen(
    probe: ContextActionQProbe,
    trunk: PolicyValueNet,
    replay_rows: list[dict[str, Any]],
    p1_path: Path,
    a16_path: Path,
    device: torch.device,
) -> dict[str, float]:
    """Use only current A16 statistics and completed local backups at the root."""
    totals = {"amplified": [0, 0], "washed": [0, 0]}
    a16, p1 = ArtifactEvaluator(a16_path), ArtifactEvaluator(p1_path)
    for name, items in selection._frozen_items(replay_rows).items():
        for item in items:
            feature = selection._features(trunk, [item], device)
            ordinary, _unused, _unused = mse._search(item, a16)
            completed: list[dict[str, int | float]] = []
            corrections: dict[tuple[int, int], float] = {}
            root_hash: str | None = None

            def capture(simulation: int, root: Any) -> None:
                nonlocal root_hash
                root_hash = PUCT(a16, 1, 1.25, random.Random(1))._state_hash(root.game)
                children = list(root.children.items())
                context = np.zeros(
                    (len(children), CONTEXT_SIZE + TRAJECTORY_SIZE), np.float32
                )
                total_visits = max(1, sum(child.visit_count for _, child in children))
                history = _history_vector(completed, include_values=True)
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
                    context[row, CONTEXT_SIZE:] = history
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

            def override(
                simulation: int, state_hash: str, move: int, raw_q: float, visits: int
            ) -> float | None:
                if state_hash == root_hash and visits > 0:
                    return raw_q + corrections.get((simulation, move), 0.0)
                return None

            corrected, _unused, _unused = mse._search(
                item,
                a16,
                hook=capture,
                override=override,
                root_backup_history=completed,
            )
            parent, _unused, _unused = mse._search(item, p1)
            totals[name][0] += (
                int(corrected["selected_move"] == parent["selected_move"])
                if name == "amplified"
                else int(corrected["selected_move"] != ordinary["selected_move"])
            )
            totals[name][1] += 1
    return {
        "rescue_rate": totals["amplified"][0] / totals["amplified"][1],
        "new_divergence_rate": totals["washed"][0] / totals["washed"][1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    replay, p1_path, a16_path, context_cache = selection._source_artifacts()
    population, exclusions = selection._population(replay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_sha = _history_cache(
        population,
        ArtifactEvaluator(a16_path),
        args.workdir / HISTORY_CACHE_NAME,
        immutable_context=context_cache,
    )
    arrays = selection._decision_arrays(context_cache, args.workdir / "context")
    trajectories = _trajectory_arrays(
        args.workdir / HISTORY_CACHE_NAME, len(population)
    )
    split = json.loads((SOURCE_WORKDIR / "split_manifest.json").read_text())
    index_by_hash = {row["state_hash"]: index for index, row in enumerate(population)}
    train_roots = np.asarray(
        [index_by_hash[key] for key in split["train_hashes"]], np.uint16
    )
    validation_hashes = set(split["validation_hashes"])
    train = np.flatnonzero(np.isin(arrays["state_index"], train_roots))
    validation = np.flatnonzero(
        np.asarray(
            [
                population[index]["state_hash"] in validation_hashes
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
    features = selection._features(trunk, population, device)
    lanes = {}
    probes = {}
    for name, seed in (
        ("trajectory_actions_values", 240),
        ("trajectory_actions_only", 241),
    ):
        probe, result = _train(
            name,
            seed,
            arrays,
            train,
            validation,
            features,
            trajectories,
            trunk_size,
            device,
            args.workdir,
        )
        lanes[name], probes[name] = result, probe
    value_lane = lanes["trajectory_actions_values"]
    value_metrics = value_lane["metrics"]
    diagnostics = {}
    for transform in ("normal", "reverse", "permuted", "zero"):
        scores, _unused = _scores(
            probes["trajectory_actions_values"],
            arrays,
            validation,
            features,
            trajectories,
            device,
            include_values=True,
            transform=transform,
        )
        diagnostics[transform] = selection._metrics(arrays, validation, scores)
    values_gate = _gate(value_metrics)
    frozen_offline = _frozen_offline(
        probes["trajectory_actions_values"],
        trunk,
        pr233.read_jsonl(replay),
        p1_path,
        a16_path,
        device,
        args.workdir,
    )
    live = (
        _live_frozen(
            probes["trajectory_actions_values"],
            trunk,
            pr233.read_jsonl(replay),
            p1_path,
            a16_path,
            device,
        )
        if values_gate
        else None
    )
    static = json.loads(
        (
            REPO_ROOT
            / "docs/data/alphazero-lite-fresh-p1-context-action-q-embedded-parent-policy-selection-probe-summary.json"
        ).read_text()
    )["validation_selection_consequence"]
    actions_metrics = lanes["trajectory_actions_only"]["metrics"]
    materially_better = (
        value_metrics["exact_flip_action_recall"] > static["exact_flip_action_recall"]
        and value_metrics["exact_parent_score_regret_captured"]
        > static["exact_parent_score_regret_captured"]
    )
    comparable_actions = (
        abs(
            value_metrics["exact_flip_action_recall"]
            - actions_metrics["exact_flip_action_recall"]
        )
        < 0.03
        and abs(
            value_metrics["exact_parent_score_regret_captured"]
            - actions_metrics["exact_parent_score_regret_captured"]
        )
        < 0.03
    )
    classification = "recent_root_trajectory_not_informative"
    if materially_better and not values_gate:
        classification = "trajectory_improves_but_insufficient"
    elif values_gate:
        if (
            live is not None
            and live["rescue_rate"] >= 0.70
            and live["new_divergence_rate"] <= 0.10
        ):
            classification = (
                "selection_history_is_sufficient"
                if comparable_actions and _gate(actions_metrics)
                else "recent_backup_trajectory_recovers_parent_actions"
            )
        else:
            classification = "offline_good_live_unstable"
    summary = {
        "schema": "azlite_context_action_q_trajectory_selection_probe_v1",
        "classification": classification,
        "history": {
            "length": HISTORY_LENGTH,
            "event_size": EVENT_SIZE,
            "order": "oldest_to_newest",
            "left_padding": "seven_zeros",
            "lanes": {
                "trajectory_actions_values": "selected_action_one_hot_plus_root_value",
                "trajectory_actions_only": "selected_action_one_hot_plus_zero_value",
            },
        },
        "hashes": {
            "replay": selection.REPLAY_SHA,
            "p1_weights": selection.P1_SHA,
            "a16_weights": selection.A16_SHA,
            "split_manifest": selection.SPLIT_SHA,
            "aligned_root_context_cache": selection.CACHE_SHA,
            "root_backup_history_cache": history_sha,
        },
        "exclusions": exclusions,
        "optimization": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "steps": 2000,
            "batch_size_root_decisions": 256,
            "lanes": {
                name: {
                    key: value
                    for key, value in lane.items()
                    if key not in {"scores", "corrections"}
                }
                for name, lane in lanes.items()
            },
        },
        "historical_pr238_instantaneous_selection_ce": json.loads(
            (
                REPO_ROOT
                / "docs/data/alphazero-lite-fresh-p1-context-action-q-selection-probe-summary.json"
            ).read_text()
        )["validation_selection_consequence"],
        "historical_pr238_temporal_selection_ce": json.loads(
            (
                REPO_ROOT
                / "docs/data/alphazero-lite-fresh-p1-context-action-q-temporal-selection-probe-summary.json"
            ).read_text()
        )["validation_selection_consequence"],
        "historical_pr239_embedded_parent_policy_selection_ce": static,
        "validation_gate_passed": values_gate,
        "history_dependence_diagnostics": diagnostics,
        "frozen_40_40_offline": frozen_offline,
        "regret_quartiles_exact_parent_flips": selection._strata(
            arrays, validation, value_lane["scores"]
        ),
        "simulation_windows": {
            name: selection._metrics(
                arrays, validation[mask], value_lane["scores"][mask]
            )
            for name, mask in (
                ("1_384", arrays["simulation"][validation] <= 384),
                ("385_1200", arrays["simulation"][validation] >= 385),
            )
        },
        "correction_magnitude": selection._scale(
            value_lane["corrections"], arrays, validation
        ),
        "invariants": {
            "ordinary_a16_context_cache_reproduced": True,
            "completed_backups_only": True,
            "simulation_t_backup_excluded": True,
            "p1_runtime": False,
            "selection_ce_unchanged": True,
            "unvisited_fpu_preserved": True,
        },
        "guardrails": {
            "arena_run": False,
            "p1_runtime": False,
            "loss": "masked_cross_entropy_only",
            "history_length_tuned": False,
        },
        "live_frozen_root": live
        if live is not None
        else "not_run_validation_gate_failed",
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output = (
        REPO_ROOT
        / "docs/data/alphazero-lite-fresh-p1-context-action-q-trajectory-selection-probe-summary.json"
    )
    report = (
        REPO_ROOT
        / "docs/alphazero-lite-fresh-p1-context-action-q-trajectory-selection-probe-results.md"
    )
    output.write_text(text)
    report.write_text(
        f"# Causal Root Backup Trajectory Selection Probe\n\n**Classification:** `{classification}`\n\n```json\n{text}```\n"
    )
    (args.workdir / "summary.json").write_text(text)
    print(classification)


if __name__ == "__main__":
    main()
