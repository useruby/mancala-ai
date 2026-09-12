#!/usr/bin/env python3
# ruff: noqa: E402
"""Read-only root-Q confidence diagnostic following root-only FPU failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite import run_root_fpu_capture_diagnostic as fpu
from ml.alphazero_lite.forensic_exact_references import exact_regret, sha256_file
from ml.alphazero_lite.forensic_suite import load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import (
    CheckpointEvaluator,
    PUCT,
    root_q_confidence_override,
)

EXACT = fpu.EXACT
SUITE = fpu.SUITE
PR298 = fpu.PR298
ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)


def lane_name(alpha: float) -> str:
    return f"root_q_alpha_{int(alpha * 100):03d}"


LANES = {lane_name(alpha): alpha for alpha in ALPHAS}


def state_hash(game: KalahGame) -> str:
    encoded = json.dumps(
        game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def searched_row(
    exact: dict[str, Any],
    evaluator: CheckpointEvaluator,
    alpha: float,
    *,
    full_trace: bool,
) -> dict[str, Any]:
    game = KalahGame.from_state(exact["state"])
    history: list[dict[str, Any]] = []
    snapshots = set(range(1, 385)) if full_trace else set(fpu.CHECKPOINTS_AT)
    search = PUCT(
        evaluator=evaluator,
        simulations=384,
        c_puct=1.25,
        rng=random.Random(42),
        fpu_mode="zero",
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        selection_q_override=root_q_confidence_override(state_hash(game), alpha),
        root_snapshot_checkpoints=snapshots,
        root_backup_history=history,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    selected = search.select_root_move(root, game.possible_moves())
    cumulative, total = {}, 0.0
    for event in history:
        total += float(event["root_value"])
        cumulative[int(event["simulation"])] = total / int(event["simulation"])
    all_snapshots = [
        fpu.snapshot_with_exact(item, exact, cumulative[int(item["simulation"])])
        for item in search.root_summary()["root_snapshots"]
    ]
    first_visits = fpu.first_visit_simulations(history, game.possible_moves())
    by_simulation = {item["simulation"]: item for item in all_snapshots}
    return {
        "selected_action": selected,
        "regret": exact_regret(exact, selected),
        "visits": [int(value) for value in visits],
        "root_q": float(root.q_value),
        "root_prior": [
            float(root.children[move].prior) if move in root.children else 0.0
            for move in range(6)
        ],
        "child_q": {
            str(move): float(child.q_value) for move, child in root.children.items()
        },
        "first_visit_simulation": first_visits,
        "q_after_first_visit": (
            {
                move: fpu.snapshot_move(by_simulation[simulation], int(move))["q_value"]
                for move, simulation in first_visits.items()
                if simulation is not None
            }
            if full_trace
            else {}
        ),
        "checkpoints": [
            item for item in all_snapshots if item["simulation"] in fpu.CHECKPOINTS_AT
        ],
    }


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Root-Q Confidence Capture Diagnostic",
        "",
        "## Inherited Result",
        "",
        "Root-only parent-Q FPU did not repair uniform seeds 44/45, so this evaluates only the existing root visited-child Q confidence override with zero FPU.",
        "",
        "## Baseline",
        "",
        "`root_q_alpha_100` reproduced PR #298 before any intervention result was interpreted.",
        "",
        "## Decision-Critical Results",
        "",
        "| Seed | Family | Alpha | Move/regret on capture_available-002 |",
        "| ---: | --- | ---: | --- |",
    ]
    for seed in fpu.SEEDS:
        for family in fpu.PRIMARY_FAMILIES:
            for lane, alpha in LANES.items():
                row = result["evaluations"][f"{seed}:{family}"][lane][
                    "capture_available-002"
                ]["search"]
                lines.append(
                    f"| {seed} | {family} | {alpha:.2f} | {row['selected_action']}/{row['regret']:.0f} |"
                )
    lines += [
        "",
        "## Global Exact Safety",
        "",
        "| Alpha | Accuracy | Mean regret | Blunder rate | Correction balance |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane, alpha in LANES.items():
        summary = result["summaries"][lane]["overall"]
        balance = result["summaries"][lane]["transitions"]["overall"].get(
            "correction_balance", 0
        )
        lines.append(
            f"| {alpha:.2f} | {summary['exact_optimal_set_accuracy']:.4f} | {summary['mean_exact_regret']:.4f} | {summary['exact_blunder_rate']:.4f} | {balance} |"
        )
    lines += [
        "",
        "## Classification",
        "",
        f"`{result['hard_classification']}`",
        "",
        result["next_experiment"],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    exact_payload = json.loads(EXACT.read_text())
    exact_rows = [
        row for row in exact_payload["rows"] if row["exact_status"] == "exact_solved"
    ]
    exact_by_id = {row["id"]: row for row in exact_rows}
    suite_buckets = {item.id: item.bucket for item in load_suite(SUITE)}
    pr298 = json.loads(PR298.read_text())
    evaluators, inputs = (
        {},
        {
            "exact_v2": {"sha256": sha256_file(EXACT)},
            "pr298": {"sha256": sha256_file(PR298)},
            "checkpoints": {},
        },
    )
    for seed in fpu.SEEDS:
        for family in fpu.CHECKPOINTS:
            path = fpu.checkpoint_path(seed, family)
            expected = fpu.EXPECTED_CHECKPOINTS[seed].get(
                family, pr298["inputs"]["checkpoints"][f"{seed}:parent"]["sha256"]
            )
            digest = sha256_file(path)
            if digest != expected:
                raise RuntimeError(f"checkpoint SHA mismatch: {seed}:{family}")
            inputs["checkpoints"][f"{seed}:{family}"] = digest
            evaluators[seed, family] = CheckpointEvaluator(
                path, input_encoding="kalah_v3"
            )
    for seed in fpu.SEEDS:
        for family in fpu.PRIMARY_FAMILIES:
            for identifier in ("capture_available-002", "capture_available-020"):
                actual = searched_row(
                    exact_by_id[identifier],
                    evaluators[seed, family],
                    1.0,
                    full_trace=False,
                )
                fpu.assert_baseline(
                    actual, fpu.baseline_expected(pr298, seed, family, identifier)
                )
    evaluations: dict[str, Any] = {}
    for seed in fpu.SEEDS:
        for family in fpu.CHECKPOINTS:
            key, evaluator = f"{seed}:{family}", evaluators[seed, family]
            evaluations[key] = {lane: {} for lane in LANES}
            for identifier, exact in exact_by_id.items():
                raw = fpu.raw_row(exact, evaluator)
                for lane, alpha in LANES.items():
                    search = searched_row(
                        exact,
                        evaluator,
                        alpha,
                        full_trace=identifier == "capture_available-002",
                    )
                    evaluations[key][lane][identifier] = {
                        "bucket": suite_buckets[identifier],
                        "raw": raw,
                        "search": search,
                    }
    summaries = {}
    for lane in LANES:
        rows = [
            evaluations[f"{seed}:{family}"][lane][identifier]
            for seed in fpu.SEEDS
            for family in fpu.CHECKPOINTS
            for identifier in exact_by_id
        ]
        summaries[lane] = fpu.summarize(rows)
    repaired = all(
        evaluations[f"{seed}:uniform1200"]["root_q_alpha_050"]["capture_available-002"][
            "search"
        ]["regret"]
        == 0
        for seed in (44, 45)
    )
    classification = (
        "root_q_confidence_repairs_capture"
        if repaired
        else "root_q_confidence_not_causal"
    )
    next_experiment = "Exactly one next experiment: inspect root-child Q error versus visit count on the failing capture traces; do not train a model."
    result = {
        "schema": "azlite_root_q_confidence_capture_diagnostic_v1",
        "read_only": {"training": False, "self_play": False, "promotion": False},
        "inputs": inputs,
        "search_configuration": {
            "simulations": 384,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "seed": 42,
            "dirichlet": None,
        },
        "alphas": list(ALPHAS),
        "baseline_reproduction": {"passed": True},
        "evaluations": evaluations,
        "summaries": summaries,
        "hard_classification": classification,
        "next_experiment": next_experiment,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
