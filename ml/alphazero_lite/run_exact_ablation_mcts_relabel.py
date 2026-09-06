#!/usr/bin/env python3
"""Relabel the frozen exact-train states with the classic-MCTS teacher.

Reads the frozen source states plus the solved exact train labels, and
relabels the IDENTICAL 8,000 train states with 1200-simulation classic MCTS
(default policy target, temperature 1.0). Output rows carry the same
``state`` vectors, ``source_id`` values, and train-row schema, differing
only in teacher targets and provenance. Deterministic per-state seeds make
reruns byte-identical. No model is trained here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite import exact_teacher_labeling as exact
from ml.alphazero_lite.classic_mcts import MCTS as ClassicMCTS
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    build_policy_target,
    top_policy_move_for_legal_moves,
    value_from_classic_mcts_root,
    visits_from_classic_mcts_root,
)

MCTS_SIMULATIONS = 1200
MCTS_POLICY_TEMPERATURE = 1.0
MCTS_POLICY_TARGET_MODE = "default"
MCTS_VALUE_TARGET_MODE = "default"
MCTS_AUDIT_SEED = 777
TEACHER_IDENTITY = "classic_mcts_1200"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--exact-train", type=Path, required=True)
    parser.add_argument("--out-train", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=MCTS_AUDIT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = {r["source_id"]: r for r in exact.read_jsonl(args.source_states)}
    exact_rows = exact.read_jsonl(args.exact_train)
    wanted = [r["source_id"] for r in exact_rows]
    if len(wanted) != len(set(wanted)):
        raise SystemExit("exact train file contains duplicate source ids")
    missing = [sid for sid in wanted if sid not in sources]
    if missing:
        raise SystemExit(f"{len(missing)} train states missing from source cohort")
    out_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, sid in enumerate(wanted):
        source = sources[sid]
        template = exact_rows[index]
        if template["source_id"] != sid:
            raise SystemExit("exact train order drifted from source-id sequence")
        game = KalahGame.from_state(source["state"])
        legal = game.possible_moves()
        mcts = ClassicMCTS(
            game.clone(),
            simulations=args.simulations,
            seed=exact.derive_mcts_audit_seed(
                base_seed=args.seed,
                canonical_state=source["canonical_state"],
                simulations=args.simulations,
            ),
        )
        root = mcts.search_root()
        visits = visits_from_classic_mcts_root(root)
        value = value_from_classic_mcts_root(root)
        policy = build_policy_target(
            visits=np.asarray(visits, dtype=np.float64),
            legal_moves=legal,
            temperature=MCTS_POLICY_TEMPERATURE,
            mode=MCTS_POLICY_TARGET_MODE,
        )
        top = top_policy_move_for_legal_moves(policy, legal)
        total = sum(float(visits[m]) for m in legal)
        out_rows.append(
            {
                "state": template["state"],
                "policy": policy,
                "value": float(value),
                "policy_target_mode": MCTS_POLICY_TARGET_MODE,
                "value_target_mode": MCTS_VALUE_TARGET_MODE,
                "player": int(source["player"]),
                "move_index": int(source["move_index"]),
                "legal_moves": legal,
                "canonical_state": source["canonical_state"],
                "source_id": sid,
                "teacher": TEACHER_IDENTITY,
                "teacher_simulations": int(args.simulations),
                "teacher_seed": int(args.seed),
                "teacher_top_move": top,
                "teacher_top_visit_share": (
                    max(float(visits[m]) for m in legal) / total if total > 0 else 0.0
                ),
                "teacher_visits": visits,
                "source_provenance": dict(source.get("provenance", {})),
            }
        )
        if (index + 1) % 1000 == 0:
            print(f"mcts-labeled {index + 1}/{len(wanted)}", flush=True)
    exact.write_jsonl(args.out_train, out_rows)
    summary = {
        "schema": "classic_mcts_train_relabel_v1",
        "train_rows": len(out_rows),
        "train_sha256": exact.sha256_file(args.out_train),
        "exact_train_path": str(args.exact_train),
        "source_states_path": str(args.source_states),
        "source_ids_identical": [r["source_id"] for r in out_rows] == wanted,
        "teacher": TEACHER_IDENTITY,
        "simulations": int(args.simulations),
        "seed": int(args.seed),
        "policy_target_mode": MCTS_POLICY_TARGET_MODE,
        "value_target_mode": MCTS_VALUE_TARGET_MODE,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"mcts_train_rows={len(out_rows)} sha={summary['train_sha256']}")
    print(f"summary_written={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
