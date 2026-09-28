"""Evaluate a frozen training-checkpoint trajectory without changing it.

The JSON spec names immutable replay sources, a parent artifact, checkpoint
artifacts, and a frozen state manifest.  This is deliberately diagnostic: it
does not train, generate games, or select a production checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.alphazero_lite import arena, train
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.replay_source_attribution import sha256_file


def js(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0
    left_terms = left[left > 0] * np.log2(left[left > 0] / midpoint[left > 0])
    right_terms = right[right > 0] * np.log2(right[right > 0] / midpoint[right > 0])
    return float(0.5 * (np.sum(left_terms) + np.sum(right_terms)))


def normalized_policy(
    evaluator: arena.ArtifactEvaluator, state: dict[str, Any]
) -> tuple[np.ndarray, float, list[int]]:
    game = KalahGame.from_state(state)
    legal = game.possible_moves()
    policy, value = evaluator.evaluate(game)
    result = np.zeros(6, dtype=np.float64)
    result[legal] = policy[legal] / policy[legal].sum()
    return result, float(value), legal


def aggregate(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def policy_value_metrics(
    policies: list[np.ndarray],
    values: list[float],
    targets: np.ndarray,
    target_values: np.ndarray,
) -> dict[str, float | None]:
    if not policies:
        return {
            key: None
            for key in (
                "policy_ce",
                "policy_top1_agreement",
                "policy_js",
                "value_mae",
                "value_mse",
                "value_sign_accuracy",
            )
        }
    predicted = np.asarray(policies)
    target = np.asarray(targets)
    predicted = np.clip(predicted, 1e-12, 1.0)
    ce = -np.sum(target * np.log(predicted), axis=1)
    js_values = [js(p, q) for p, q in zip(predicted, target)]
    prediction_values = np.asarray(values)
    actual = target_values.reshape(-1)
    return {
        "policy_ce": float(np.mean(ce)),
        "policy_top1_agreement": float(
            np.mean(np.argmax(predicted, axis=1) == np.argmax(target, axis=1))
        ),
        "policy_js": float(np.mean(js_values)),
        "value_mae": float(np.mean(np.abs(prediction_values - actual))),
        "value_mse": float(np.mean((prediction_values - actual) ** 2)),
        "value_sign_accuracy": float(
            np.mean(np.sign(prediction_values) == np.sign(actual))
        ),
    }


def source_rows(
    spec: dict[str, Any],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[str], np.ndarray
]:
    paths = [Path(source["path"]) for source in spec["sources"]]
    weights = [int(source["weight"]) for source in spec["sources"]]
    modes = [str(source["value_target_mode"]) for source in spec["sources"]]
    for source, path in zip(spec["sources"], paths):
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise ValueError(f"source hash mismatch for {path}: {actual}")
    x, p, v, replay_indexes = train.load_jsonl_replay(
        paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
    )
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    for source, path in zip(spec["sources"], paths):
        with path.open(encoding="utf-8") as handle:
            source_data = [json.loads(line) for line in handle if line.strip()]
        rows.extend(source_data)
        names.extend([source["name"]] * len(source_data))
    np.random.seed(int(spec["training_seed"]))
    _train_positions, validation_positions = train.split_replay_positions_by_source_row(
        replay_indexes, val_split=float(spec["validation_split"])
    )
    return x, p, v, rows, names, replay_indexes[validation_positions]


def state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row["state"]
    if isinstance(state, dict):
        return state
    return {
        "player_pits": [round(48 * value) for value in state[:6]],
        "opponent_pits": [round(48 * value) for value in state[6:12]],
        "player_store": round(48 * state[12]),
        "opponent_store": round(48 * state[13]),
        "current_player": round(state[14]),
    }


def phase_masks(
    rows: list[dict[str, Any]], indexes: np.ndarray
) -> dict[str, np.ndarray]:
    active = np.asarray(
        [sum(KalahGame.from_state(state_from_row(rows[i])).pits) for i in indexes]
    )
    ply = np.asarray(
        [int(rows[i].get("ply", rows[i].get("move_index", 10**9))) for i in indexes]
    )
    return {
        "all": np.ones(len(indexes), dtype=bool),
        "high_stone": active > 32,
        "opening": ply <= 12,
        "opening_high_stone": (ply <= 12) & (active > 32),
        "solver_owned": active <= 16,
    }


def checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def parameter_drift(
    parent: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    groups = {
        "shared_trunk": ("w_input", "b_input", "w_residual", "b_residual"),
        "policy_hidden_readout": (
            "w_policy_hidden",
            "b_policy_hidden",
            "w_policy",
            "b_policy",
        ),
        "value_hidden_readout": (
            "w_value_hidden",
            "b_value_hidden",
            "w_value",
            "b_value",
        ),
    }
    result: dict[str, Any] = {}
    for name, prefixes in groups.items():
        keys = [key for key in parent if key in candidate and key.startswith(prefixes)]
        delta = math.sqrt(
            sum(float(np.sum((candidate[key] - parent[key]) ** 2)) for key in keys)
        )
        base = math.sqrt(sum(float(np.sum(parent[key] ** 2)) for key in keys))
        result[name] = {
            "l2_delta": delta,
            "relative_l2_delta": delta / base if base else 0.0,
        }
    keys = [key for key in parent if key in candidate]
    delta = math.sqrt(
        sum(float(np.sum((candidate[key] - parent[key]) ** 2)) for key in keys)
    )
    base = math.sqrt(sum(float(np.sum(parent[key] ** 2)) for key in keys))
    result["all"] = {
        "l2_delta": delta,
        "relative_l2_delta": delta / base if base else 0.0,
    }
    return result


def puct(
    evaluator: arena.ArtifactEvaluator, state: dict[str, Any], seed: int
) -> dict[str, Any]:
    return arena.evaluate_artifact_position(
        evaluator=evaluator,
        state=state,
        simulations=384,
        seed=seed,
        c_puct=1.25,
        search_options={
            "fpu_mode": "zero",
            "reuse_subtree": False,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "tactical_root_bias": 0.0,
            "root_temperature": 0.0,
        },
        exact_root_solve_threshold=16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    x, targets, target_values, rows, names, val_indexes = source_rows(spec)
    artifacts = spec["artifacts"]
    evaluators = {
        name: arena.ArtifactEvaluator(Path(path)) for name, path in artifacts.items()
    }
    states = [state_from_row(rows[index]) for index in val_indexes]
    masks = phase_masks(rows, val_indexes)
    raw = {
        name: [normalized_policy(ev, state) for state in states]
        for name, ev in evaluators.items()
    }
    validation: dict[str, Any] = {}
    source_fit: dict[str, Any] = {}
    for name, output in raw.items():
        policies = np.asarray([item[0] for item in output])
        values = np.asarray([item[1] for item in output])
        validation[name] = {
            phase: policy_value_metrics(
                list(policies[mask]),
                list(values[mask]),
                targets[val_indexes][mask],
                target_values[val_indexes][mask],
            )
            for phase, mask in masks.items()
        }
        source_fit[name] = {}
        for source in sorted(set(names)):
            source_mask = np.asarray(
                [names[index] == source for index in val_indexes], dtype=bool
            )
            source_fit[name][source] = policy_value_metrics(
                list(policies[source_mask]),
                list(values[source_mask]),
                targets[val_indexes][source_mask],
                target_values[val_indexes][source_mask],
            )
            if source == "fresh":
                active = np.asarray(
                    [
                        sum(KalahGame.from_state(states[i]).pits)
                        for i in range(len(states))
                    ]
                )
                source_fit[name][source]["phase"] = {
                    label: policy_value_metrics(
                        list(policies[source_mask & bucket]),
                        list(values[source_mask & bucket]),
                        targets[val_indexes][source_mask & bucket],
                        target_values[val_indexes][source_mask & bucket],
                    )
                    for label, bucket in {
                        ">32": active > 32,
                        "17-32": (active >= 17) & (active <= 32),
                        "<=16": active <= 16,
                    }.items()
                }

    manifest = json.loads(Path(spec["state_manifest"]).read_text(encoding="utf-8"))
    unique: dict[str, dict[str, Any]] = {}
    for key in ("first_counterfactual_divergences", "outcome_associated_divergences"):
        for row in manifest[key]:
            if "state" in row:
                state = row["state"]
                digest = hashlib.sha256(
                    json.dumps(state, sort_keys=True).encode()
                ).hexdigest()
                unique[digest] = state
    divergence_states = list(unique.values())
    parent_raw = [
        normalized_policy(evaluators["E0"], state) for state in divergence_states
    ]
    parent_puct = [
        puct(evaluators["E0"], state, i) for i, state in enumerate(divergence_states)
    ]
    retention: dict[str, Any] = {}
    previous_selected: set[int] = set()
    for name, evaluator in evaluators.items():
        candidate_raw = [
            normalized_policy(evaluator, state) for state in divergence_states
        ]
        candidate_puct = [
            puct(evaluator, state, i) for i, state in enumerate(divergence_states)
        ]
        raw_js = [
            js(left[0], right[0]) for left, right in zip(parent_raw, candidate_raw)
        ]
        raw_agree = [
            int(np.argmax(left[0]) == np.argmax(right[0]))
            for left, right in zip(parent_raw, candidate_raw)
        ]
        parent_mass = [
            right[0][int(np.argmax(left[0]))]
            for left, right in zip(parent_raw, candidate_raw)
        ]
        value_delta = [
            right[1] - left[1] for left, right in zip(parent_raw, candidate_raw)
        ]
        selected = {
            i
            for i, (left, right) in enumerate(zip(parent_puct, candidate_puct))
            if left["selected_move"] != right["selected_move"]
        }
        visit_js = []
        shares = []
        margins = []
        q_deltas = []
        for left, right in zip(parent_puct, candidate_puct):
            lp = np.asarray(left["visits"], dtype=float)
            lp /= lp.sum()
            rp = np.asarray(right["visits"], dtype=float)
            rp /= rp.sum()
            visit_js.append(js(lp, rp))
            ordered = sorted(rp, reverse=True)
            shares.append(float(ordered[0]))
            margins.append(
                float(ordered[0] - ordered[1])
                if len(ordered) > 1
                else float(ordered[0])
            )
            lq = {item["move"]: item["q_value"] for item in left["child_stats"]}
            rq = {item["move"]: item["q_value"] for item in right["child_stats"]}
            q_deltas.append(
                float(rq[left["selected_move"]] - lq[left["selected_move"]])
            )
        retention[name] = {
            "states": len(divergence_states),
            "raw_policy_js_from_E0": aggregate(raw_js),
            "raw_argmax_agreement_E0": aggregate(raw_agree),
            "mean_probability_E0_argmax": aggregate(parent_mass),
            "value_prediction_delta_E0": aggregate(value_delta),
            "value_absolute_drift_E0": aggregate([abs(item) for item in value_delta]),
            "puct_parent_agreement": 1.0 - len(selected) / len(divergence_states),
            "puct_visit_js_from_E0": aggregate(visit_js),
            "puct_mean_top1_visit_share": aggregate(shares),
            "puct_mean_top_two_margin": aggregate(margins),
            "puct_q_difference_E0_selected_action": aggregate(q_deltas),
            "puct_changed_states": len(selected),
            "puct_changed_fraction": len(selected) / len(divergence_states),
            "new_parent_disagreement_from_previous_epoch": sorted(
                selected - previous_selected
            )
            if name != "E0"
            else [],
        }
        previous_selected = selected
    parent_arrays = checkpoint_arrays(Path(spec["checkpoints"]["E0"]))
    drift = {
        name: parameter_drift(parent_arrays, checkpoint_arrays(Path(path)))
        for name, path in spec["checkpoints"].items()
        if name != "E0"
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": "azlite_checkpoint_trajectory_diagnostic_v1",
                "spec_sha256": sha256_file(args.spec),
                "validation_index_count": len(val_indexes),
                "validation_unique_source_rows": len(set(map(int, val_indexes))),
                "validation": validation,
                "source_fit": source_fit,
                "parent_retention": retention,
                "parameter_drift": drift,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
