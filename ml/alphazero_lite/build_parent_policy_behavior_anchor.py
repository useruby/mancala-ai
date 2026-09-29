"""Build a policy-only parent-behavior anchor from a frozen primary replay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.checkpoint_phase_selection import active_pit_stones_from_state
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.replay_source_attribution import sha256_file
from ml.alphazero_lite.train import (
    PolicyValueNet,
    legal_mask_matrix_for_encoded_states,
    load_checkpoint_into_model,
    load_jsonl_replay,
    set_seed,
    split_replay_positions_by_source_row,
)


def state_from_encoded(state: list[float]) -> dict[str, Any]:
    return {
        "player_pits": [round(48 * value) for value in state[:6]],
        "opponent_pits": [round(48 * value) for value in state[6:12]],
        "player_store": round(48 * state[12]),
        "opponent_store": round(48 * state[13]),
        "current_player": round(state[14]),
    }


def parse_csv(value: str, converter: Any) -> list[Any]:
    return [converter(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-files", required=True)
    parser.add_argument("--source-weights", required=True)
    parser.add_argument("--source-value-modes", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-weights-sha", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val-split", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    paths = parse_csv(args.source_files, Path)
    weights = parse_csv(args.source_weights, int)
    modes = [item for item in args.source_value_modes.split(",") if item]
    if len(paths) != len(weights) or len(paths) != len(modes):
        raise ValueError("source configuration lengths must match")
    source_shas = [sha256_file(path) for path in paths]
    x, _p, values, replay_indexes = load_jsonl_replay(
        paths,
        weights,
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=modes,
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    set_seed(args.seed)
    train_positions, validation_positions = split_replay_positions_by_source_row(
        replay_indexes, val_split=args.val_split
    )
    training_indexes = replay_indexes[train_positions]
    validation_keys = {
        canonical_state_key(state_from_encoded(rows[index]["state"]))
        for index in set(replay_indexes[validation_positions].tolist())
    }
    candidate_indexes = [
        int(index)
        for index in training_indexes.tolist()
        if active_pit_stones_from_state(state_from_encoded(rows[int(index)]["state"]))
        > 32
    ]
    if not candidate_indexes:
        raise ValueError("parent_behavior_anchor_artifact_invalid: no candidates")

    model = PolicyValueNet((96, 3), "residual_v3", 27)
    load_checkpoint_into_model(model, args.parent_checkpoint)
    model.eval()
    candidate_x = x[np.asarray(candidate_indexes, dtype=np.int64)]
    legal_masks = legal_mask_matrix_for_encoded_states(candidate_x)
    policies: list[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        for batch in np.array_split(
            candidate_x, max(1, math.ceil(len(candidate_x) / 1024))
        ):
            logits, _value = model(torch.from_numpy(batch))
            mask = torch.from_numpy(legal_masks[offset : offset + len(batch)])
            policies.append(
                torch.softmax(logits.masked_fill(mask <= 0.0, -1e9), dim=1).numpy()
            )
            offset += len(batch)
    parent_policies = np.concatenate(policies)
    anchor_rows = []
    for compact_index, policy, legal_mask in zip(
        candidate_indexes, parent_policies, legal_masks
    ):
        source = rows[compact_index]
        state = source["state"]
        active = active_pit_stones_from_state(state_from_encoded(state))
        key = canonical_state_key(state_from_encoded(state))
        # Exclude equivalent training occurrences when that state is primary validation.
        if key in validation_keys:
            continue
        if active <= 32 or not np.isfinite(policy).all():
            raise ValueError("parent_behavior_anchor_artifact_invalid")
        if float(policy[legal_mask <= 0.0].sum()) != 0.0 or not np.isclose(
            policy.sum(), 1.0
        ):
            raise ValueError("parent_behavior_anchor_artifact_invalid")
        anchor_rows.append(
            {
                "state": state,
                "policy": [float(item) for item in policy],
                "value": source["value"],
                "policy_target_mode": "sharpened",
                "value_target_mode": "default",
                "behavior_anchor_parent_weights_sha256": args.parent_weights_sha,
                "behavior_anchor_target_kind": "parent_raw_legal_policy",
                "behavior_anchor_active_pit_stones": active,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in anchor_rows
        ),
        encoding="utf-8",
    )
    anchor_policies = np.asarray([row["policy"] for row in anchor_rows])
    entropies = -np.sum(
        anchor_policies * np.log2(np.clip(anchor_policies, 1e-12, 1.0)), axis=1
    )
    keys = [
        canonical_state_key(state_from_encoded(row["state"])) for row in anchor_rows
    ]
    audit = {
        "schema": "azlite_parent_policy_behavior_anchor_v1",
        "artifact_sha256": sha256_file(args.out),
        "parent_weights_sha256": args.parent_weights_sha,
        "source_replay_sha256": source_shas,
        "primary_training_split": {"seed": args.seed, "val_split": args.val_split},
        "active_stone_predicate": ">32",
        "anchor_row_count": len(anchor_rows),
        "unique_state_count": len(set(keys)),
        "duplicate_fraction": 1.0 - len(set(keys)) / len(keys),
        "validation_overlap_count": len(set(keys) & validation_keys),
        "policy_target_kind": "parent_raw_legal_policy",
        "behavior_anchor_value_loss_weight": 0,
        "policy_entropy": {
            "min": float(entropies.min()),
            "mean": float(entropies.mean()),
            "max": float(entropies.max()),
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
