#!/usr/bin/env python3
"""Paired classic-MCTS audit over a frozen exact-labeled cohort.

Reads the frozen source states and solved exact labels, relabels a
deterministic subset with the currently relevant classic-MCTS teacher
configuration (1200 simulations, default policy target, temperature 1.0),
and reports MCTS top action in exact optimal set, single/multi-optimum
breakdowns, aligned value disagreement, phase/stone-bucket splits, and
MCTS visit-share vs exact correctness. Never modifies the state cohort.
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

AUDIT_SIMULATIONS = 1200
AUDIT_POLICY_TEMPERATURE = 1.0
AUDIT_POLICY_TARGET_MODE = "default"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--exact-train", type=Path, required=True)
    parser.add_argument("--exact-holdout", type=Path, required=True)
    parser.add_argument("--out-pairs", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--audit-size", type=int, default=1200)
    parser.add_argument("--audit-seed", type=int, default=777)
    parser.add_argument(
        "--simulations",
        type=int,
        default=AUDIT_SIMULATIONS,
    )
    return parser.parse_args(argv)


def load_exact_by_source(*paths: Path) -> dict[str, dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in exact.read_jsonl(path):
            by_source[str(row["source_id"])] = row
    return by_source


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = exact.read_jsonl(args.source_states)
    exact_by_source = load_exact_by_source(args.exact_train, args.exact_holdout)
    available = [s for s in sources if s["source_id"] in exact_by_source]
    if len(available) < args.audit_size:
        raise SystemExit(
            f"only {len(available)} exact-labeled states available, "
            f"need {args.audit_size}"
        )
    ordered = sorted(available, key=lambda r: str(r["canonical_state"]))
    subset = ordered[: args.audit_size]
    pairs: list[dict[str, Any]] = []
    for position, source in enumerate(subset):
        label = exact_by_source[source["source_id"]]
        game = KalahGame.from_state(source["state"])
        legal = game.possible_moves()
        started = time.monotonic()
        mcts = ClassicMCTS(
            game.clone(),
            simulations=args.simulations,
            seed=exact.derive_mcts_audit_seed(
                base_seed=args.audit_seed,
                canonical_state=source["canonical_state"],
                simulations=args.simulations,
            ),
        )
        root = mcts.search_root()
        visits = visits_from_classic_mcts_root(root)
        mcts_value = value_from_classic_mcts_root(root)
        policy = build_policy_target(
            visits=np.asarray(visits, dtype=np.float64),
            legal_moves=legal,
            temperature=AUDIT_POLICY_TEMPERATURE,
            mode=AUDIT_POLICY_TARGET_MODE,
        )
        mcts_top = top_policy_move_for_legal_moves(policy, legal)
        total = sum(float(visits[m]) for m in legal)
        top_share = max(float(visits[m]) for m in legal) / total if total > 0 else 0.0
        exact_optimal = [int(a) for a in label["exact_optimal_actions"]]
        exact_value = float(label["exact_value_training"])
        pairs.append(
            {
                "source_id": source["source_id"],
                "canonical_state": source["canonical_state"],
                "bucket": source["stones_bucket"],
                "phase": source["phase"],
                "legal_moves": legal,
                "exact_optimal_actions": exact_optimal,
                "exact_value": exact_value,
                "exact_root_margin": int(label["exact_root_margin"]),
                "mcts_top": mcts_top,
                "mcts_value": mcts_value,
                "mcts_top_visit_share": top_share,
                "mcts_policy": policy,
                "mcts_simulations": int(args.simulations),
                "mcts_audit_seconds": time.monotonic() - started,
            }
        )
        if (position + 1) % 200 == 0:
            print(f"audited {position + 1}/{len(subset)}", flush=True)
    metrics = exact.compute_audit_metrics(pairs)
    summary = {
        "schema": "native_hybrid_exact_mcts_audit_v1",
        "audit_size_requested": int(args.audit_size),
        "audit_pairs": len(pairs),
        "audit_seed": int(args.audit_seed),
        "mcts_simulations": int(args.simulations),
        "mcts_policy_target_mode": AUDIT_POLICY_TARGET_MODE,
        "mcts_policy_temperature": AUDIT_POLICY_TEMPERATURE,
        "metrics": metrics,
        "pairs_sha256": exact.sha256_rows(pairs),
        "source_states_sha256": exact.sha256_file(args.source_states),
    }
    exact.write_jsonl(args.out_pairs, pairs)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    m = metrics
    print(
        f"mcts_top_in_exact_set={m['mcts_top_in_exact_optimal_set']}/{m['n']} "
        f"({m['mcts_top_in_exact_optimal_set_rate']:.4f})"
    )
    print(
        f"single_opt_agree={m['exact_single_optimum_mcts_top1_agreement']} "
        f"multi_rate={m['exact_multi_optimum_rate']:.4f} "
        f"mean_abs_val_dis={m['mean_abs_value_disagreement']:.4f}"
    )
    print(f"audit_written={args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
