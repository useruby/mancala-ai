#!/usr/bin/env python3
"""Compute observational phase metrics on the production replay validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import train
from ml.alphazero_lite.checkpoint_phase_selection import (
    active_pit_stones_from_encoded_state,
    active_pit_stones_from_state,
)


def masks(rows: list[dict[str, Any]], indexes: np.ndarray) -> dict[str, np.ndarray]:
    active = np.asarray(
        [
            active_pit_stones_from_state(row["state"])
            if isinstance((row := rows[int(i)])["state"], dict)
            else active_pit_stones_from_encoded_state(row["state"])
            for i in indexes
        ]
    )
    ply = np.asarray(
        [
            int(rows[int(i)].get("ply", rows[int(i)].get("move_index", 10**9)))
            for i in indexes
        ]
    )
    return {
        "all": np.ones(len(indexes), dtype=bool),
        "high_stone": active > 32,
        "opening": ply <= 12,
        "opening_high_stone": (ply <= 12) & (active > 32),
        "solver_owned": active <= 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    paths = [Path(source["path"]) for source in spec["sources"]]
    modes = [source["value_target_mode"] for source in spec["sources"]]
    weights = [int(source["weight"]) for source in spec["sources"]]
    x, p, v, replay, policy_weights = train.load_jsonl_replay(
        paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
        include_policy_loss_weights=True,
    )
    rows = [
        json.loads(line)
        for path in paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    np.random.seed(int(spec["training_seed"]))
    _train_positions, val_positions = train.split_replay_positions_by_source_row(
        replay, val_split=float(spec["validation_split"])
    )
    indexes = replay[val_positions]
    x_val = torch.from_numpy(x[indexes])
    p_val, v_val = torch.from_numpy(p[indexes]), torch.from_numpy(v[indexes])
    weights_val = torch.from_numpy(policy_weights[indexes])
    legal = torch.from_numpy(train.legal_mask_matrix_for_encoded_states(x[indexes]))
    phase_masks = masks(rows, indexes)
    report: dict[str, Any] = {"validation_count": int(len(indexes)), "phases": {}}
    for name, checkpoint in spec["checkpoints"].items():
        model = train.PolicyValueNet((96, 3), "residual_v3", 27)
        train.load_checkpoint_into_model(model, Path(checkpoint))
        model.eval()
        with torch.no_grad():
            logits, prediction = model(x_val)
        values: dict[str, Any] = {}
        for phase, mask_array in phase_masks.items():
            mask = torch.from_numpy(mask_array)
            masked_logits = logits[mask].masked_fill(legal[mask] <= 0.0, -1e9)
            policy = train.weighted_policy_loss(
                train.compute_policy_cross_entropy(masked_logits, p_val[mask]),
                weights_val[mask],
            )
            huber = train.compute_value_loss_vector(
                prediction[mask], v_val[mask], value_loss="huber", huber_delta=1.0
            ).mean()
            values[phase] = {
                "count": int(mask.sum()),
                "policy_ce": float(policy),
                "value_huber": float(huber),
                "total_loss": float(policy + 0.3 * huber),
                "value_mae": float(torch.abs(prediction[mask] - v_val[mask]).mean()),
                "value_sign_accuracy": float(
                    (torch.sign(prediction[mask]) == torch.sign(v_val[mask]))
                    .float()
                    .mean()
                ),
            }
        report["phases"][name] = values
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
