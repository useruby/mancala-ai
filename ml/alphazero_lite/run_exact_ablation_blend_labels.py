#!/usr/bin/env python3
"""Build the blended ablation lane: exact policy targets + MCTS values.

Reads the frozen exact train labels and the MCTS relabels of the IDENTICAL
states, verifies row-for-row identity (source order, canonical keys,
encoded state vectors), and writes one blended row per state:

* ``state``/``policy`` from the exact lane (uniform over optimal set);
* ``value`` from the MCTS lane (smooth win-rate-derived target).

This tests whether exact policy signal can be kept while closing the
value-MAE gap observed in the v1 ablation. No model is trained here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.alphazero_lite import exact_teacher_labeling as exact

TEACHER_IDENTITY = "exact_policy_mcts_value_blend"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-train", type=Path, required=True)
    parser.add_argument("--mcts-train", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args(argv)


def build_blend_rows(
    exact_rows: list[dict[str, Any]], mcts_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from ml.alphazero_lite.run_exact_teacher_training_ablation import (
        verify_identical_states,
    )

    verify_identical_states(exact_rows, mcts_rows)
    blended: list[dict[str, Any]] = []
    for exact_row, mcts_row in zip(exact_rows, mcts_rows):
        value = float(mcts_row["value"])
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"mcts value out of range for {mcts_row['source_id']}: {value}"
            )
        blended.append(
            {
                "state": exact_row["state"],
                "policy": exact_row["policy"],
                "value": value,
                "policy_target_mode": exact_row.get("policy_target_mode", "default"),
                "value_target_mode": mcts_row.get("value_target_mode", "default"),
                "player": exact_row["player"],
                "move_index": exact_row["move_index"],
                "legal_moves": exact_row["legal_moves"],
                "canonical_state": exact_row["canonical_state"],
                "source_id": exact_row["source_id"],
                "teacher": TEACHER_IDENTITY,
                "teacher_policy_source": exact_row.get(
                    "teacher", "native_hybrid_exact"
                ),
                "teacher_value_source": mcts_row.get("teacher", "classic_mcts_1200"),
                "teacher_value_simulations": mcts_row.get("teacher_simulations"),
                "exact_optimal_actions": exact_row.get("exact_optimal_actions"),
                "exact_root_margin": exact_row.get("exact_root_margin"),
                "exact_value_training": exact_row.get("exact_value_training"),
                "source_provenance": exact_row.get("source_provenance", {}),
            }
        )
    return blended


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exact_rows = exact.read_jsonl(args.exact_train)
    mcts_rows = exact.read_jsonl(args.mcts_train)
    blended = build_blend_rows(exact_rows, mcts_rows)
    exact.write_jsonl(args.out_train, blended)
    summary = {
        "schema": "exact_policy_mcts_value_blend_v1",
        "train_rows": len(blended),
        "train_sha256": exact.sha256_file(args.out_train),
        "exact_train_path": str(args.exact_train),
        "mcts_train_path": str(args.mcts_train),
        "exact_train_sha256": exact.sha256_file(args.exact_train),
        "mcts_train_sha256": exact.sha256_file(args.mcts_train),
        "teacher": TEACHER_IDENTITY,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"blend_train_rows={len(blended)} sha={summary['train_sha256']}")
    print(f"summary_written={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
