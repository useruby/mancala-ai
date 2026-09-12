#!/usr/bin/env python3
# ruff: noqa: E402
"""Read-only ROOT-only FPU causal diagnostic inherited from PR #298."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.forensic_exact_references import exact_regret, sha256_file
from ml.alphazero_lite.forensic_suite import load_suite
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT

EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
SUITE = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
PR298 = ROOT / "docs/data/alphazero-lite-capture-family-attribution-audit.json"
CHECKPOINT_ROOT = Path("/tmp/uniform1200-seed-confirmation/runs")
EXPECTED_CHECKPOINTS = {
    44: {
        "control": "979f906a80610e2a6648d5bc2ac1a88d47efad75bda53488b045688d12b2c513",
        "uniform1200": "806d04b9452b98dbd7a844e553ea8e4e8af6094cbe37790cf16bc0a0bc90278a",
    },
    45: {
        "control": "52d3df027c0cdccaba7e27f1c2827f673091a610f0103271c77411f0dca4131d",
        "uniform1200": "97a82737efebcee541e9572def4cca08da5e0f5c12728f3a439753ed1b6594da",
    },
    46: {
        "control": "ed61f4a6e943529a5d4fe5d452b00380bad0927c423f983de8fa9a363adc4ee2",
        "uniform1200": "9baa2227602df7ffa0c16c009d16973c67181a079fca810a364b8f4ce97aa604",
    },
}
CHECKPOINTS = ("control", "uniform1200", "parent")
PRIMARY_FAMILIES = ("control", "uniform1200")
SEEDS = (44, 45, 46)
CHECKPOINTS_AT = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384)
LANES = {"root_fpu_zero": "zero", "root_fpu_parent_q": "parent_q"}


def checkpoint_path(seed: int, family: str) -> Path:
    lane = "control_384_192" if family in {"control", "parent"} else family
    filename = "parent_init_checkpoint.npz" if family == "parent" else "checkpoint.npz"
    return (
        CHECKPOINT_ROOT
        / f"seed{seed}"
        / lane
        / f"uniform1200-confirm-seed{seed}-{lane}-iter1"
        / filename
    )


def raw_row(exact: dict[str, Any], evaluator: CheckpointEvaluator) -> dict[str, Any]:
    game = KalahGame.from_state(exact["state"])
    policy, value = evaluator.evaluate(game)
    legal = game.possible_moves()
    action = max(legal, key=lambda move: (float(policy[move]), -move))
    return {
        "policy": [float(item) for item in policy],
        "value": float(value),
        "selected_action": action,
        "regret": exact_regret(exact, action),
    }


def first_visit_simulations(
    history: list[dict[str, Any]], legal: list[int]
) -> dict[str, int | None]:
    result: dict[str, int | None] = {str(move): None for move in legal}
    for event in history:
        key = str(event["action"])
        if result.get(key) is None:
            result[key] = int(event["simulation"])
    return result


def snapshot_with_exact(
    snapshot: dict[str, Any], exact: dict[str, Any], root_q: float
) -> dict[str, Any]:
    selected = snapshot["selected_move"]
    visit_leader = max(
        snapshot["moves"],
        key=lambda item: (
            item["visit_count"],
            item["q_value"],
            item["prior"],
            -item["move"],
        ),
    )["move"]
    return {
        "simulation": snapshot["simulation"],
        "root_q": root_q,
        "visit_leader": visit_leader,
        "selected_move": selected,
        "exact_optimal_actions": list(exact["exact_optimal_actions"]),
        "exact_regret": exact_regret(exact, selected),
        # [prior, visits, Q, selection-Q, Q-component, U-component, score, FPU]
        "moves": {
            str(item["move"]): [
                item["prior"],
                item["visit_count"],
                item["q_value"],
                item["selection_q_value"],
                item["q_component"],
                item["u_component"],
                item["selection_score"],
                item["used_fpu"],
            ]
            for item in snapshot["moves"]
        },
    }


def snapshot_move(snapshot: dict[str, Any], move: int) -> dict[str, Any]:
    prior, visits, q_value, selection_q, q_component, u_component, score, used_fpu = (
        snapshot["moves"][str(move)]
    )
    return {
        "prior": prior,
        "visit_count": visits,
        "q_value": q_value,
        "selection_q_value": selection_q,
        "q_component": q_component,
        "u_component": u_component,
        "selection_score": score,
        "used_fpu": used_fpu,
    }


def searched_row(
    exact: dict[str, Any],
    evaluator: CheckpointEvaluator,
    root_fpu_mode: str,
    *,
    full_trace: bool = False,
) -> dict[str, Any]:
    game = KalahGame.from_state(exact["state"])
    snapshots = set(range(1, 385)) if full_trace else set(CHECKPOINTS_AT)
    history: list[dict[str, Any]] = []
    search = PUCT(
        evaluator=evaluator,
        simulations=384,
        c_puct=1.25,
        rng=random.Random(42),
        fpu_mode="zero",
        root_fpu_mode=root_fpu_mode,
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        value_transform=None,
        root_snapshot_checkpoints=snapshots,
        root_backup_history=history,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    selected = search.select_root_move(root, game.possible_moves())
    summary = search.root_summary()
    first_visits = first_visit_simulations(history, game.possible_moves())
    cumulative_root_q: dict[int, float] = {}
    total = 0.0
    for event in history:
        total += float(event["root_value"])
        cumulative_root_q[int(event["simulation"])] = total / int(event["simulation"])
    all_snapshots = [
        snapshot_with_exact(item, exact, cumulative_root_q[int(item["simulation"])])
        for item in summary["root_snapshots"]
    ]
    checkpoints = [
        item for item in all_snapshots if item["simulation"] in CHECKPOINTS_AT
    ]
    first_q = {}
    if full_trace:
        by_simulation = {item["simulation"]: item for item in all_snapshots}
        for move, simulation in first_visits.items():
            if simulation is not None:
                first_q[move] = snapshot_move(by_simulation[simulation], int(move))[
                    "q_value"
                ]
    return {
        "selected_action": selected,
        "regret": exact_regret(exact, selected),
        "visits": [int(value) for value in visits],
        "root_q": float(root.q_value),
        "root_prior": [
            float(child.prior) if index in root.children else 0.0
            for index, child in ((move, root.children.get(move)) for move in range(6))
        ],
        "child_q": {
            str(move): float(child.q_value) for move, child in root.children.items()
        },
        "first_visit_simulation": first_visits,
        "q_after_first_visit": first_q,
        "checkpoints": checkpoints,
    }


def transition(raw_regret: float | None, search_regret: float | None) -> str:
    return f"raw_{'correct' if raw_regret == 0 else 'wrong'}_to_search_{'correct' if search_regret == 0 else 'wrong'}"


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["search"]["regret"]) for row in rows]
    total = len(rows)
    return {
        "positions": total,
        "exact_optimal_set_accuracy": sum(regret == 0 for regret in regrets) / total,
        "mean_exact_regret": sum(regrets) / total,
        "median_exact_regret": statistics.median(regrets),
        "exact_blunder_rate": sum(regret > 0 for regret in regrets) / total,
    }


def transition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[transition(row["raw"]["regret"], row["search"]["regret"])] += 1
    counts["correction_balance"] = (
        counts["raw_wrong_to_search_correct"] - counts["raw_correct_to_search_wrong"]
    )
    return dict(counts)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["bucket"]].append(row)
    return {
        "overall": metrics(rows),
        "per_bucket": {
            bucket: metrics(items) for bucket, items in sorted(buckets.items())
        },
        "transitions": {
            "overall": transition_metrics(rows),
            "per_bucket": {
                bucket: transition_metrics(items)
                for bucket, items in sorted(buckets.items())
            },
        },
    }


def metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(current[key]) - float(baseline[key])
        for key in (
            "exact_optimal_set_accuracy",
            "mean_exact_regret",
            "median_exact_regret",
            "exact_blunder_rate",
        )
    }


def baseline_expected(
    payload: dict[str, Any], seed: int, family: str, identifier: str
) -> dict[str, Any]:
    source_family = "control_384_192" if family == "control" else family
    return next(
        row
        for row in payload["evaluations"][f"{seed}:{source_family}"]
        if row["id"] == identifier
    )


def assert_baseline(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if (
        actual["selected_action"] != expected["search"]["selected_action"]
        or actual["regret"] != expected["search"]["regret"]
    ):
        raise RuntimeError(
            "PR #298 baseline selected move or exact regret did not reproduce"
        )
    if actual["visits"] != [int(value) for value in expected["search"]["visits"]]:
        raise RuntimeError("PR #298 baseline root visit counts did not reproduce")
    if not np.allclose(
        actual["root_prior"], expected["search"]["root_prior"], rtol=0, atol=1e-6
    ):
        raise RuntimeError("PR #298 baseline raw policy did not reproduce")
    for child in expected["search"]["child_q"]:
        move = str(child["move"])
        if not np.isclose(actual["child_q"][move], child["q_value"], rtol=0, atol=1e-6):
            raise RuntimeError("PR #298 baseline child Q did not reproduce")


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Root-Only FPU Capture Diagnostic",
        "",
        "## Inherited Inputs",
        "",
        f"PR #298 head: `{result['inherited_pr298']['head']}` (`capture_regression_search_induced`).",
        f"Exact-v2 SHA256: `{result['inputs']['exact_v2']['sha256']}`. Checkpoint SHAs are recorded in the machine result.",
        "",
        "## Baseline Reproduction",
        "",
        "`root_fpu_zero` reproduced PR #298 selected moves, visit counts, child Q values, exact regrets, and raw policies for both critical states and all six primary checkpoints.",
        "",
        "## Semantics",
        "",
        "`root_fpu_mode=None` preserves historical behavior. `zero` and `parent_q` apply only to unvisited root children; non-root FPU remains `zero`. Existing `fpu_mode=parent_q` and `parent_value` remain aliases at non-root nodes.",
        "",
        "## Capture-002 Paired Trajectories",
        "",
        "All legal-action telemetry is in the machine result. This compact trace shows actions 1 (raw uniform error) and 2 (exact optimum), including each action's first visit and Q immediately after that visit.",
        "",
        "| Seed | Family | Lane | Priors A1/A2 | First visit/Q A1/A2 | Sim | Root Q | A1 visits/Q | A2 visits/Q | A1/A2 visits | A1-A2 Q | Selected | Regret |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for family in PRIMARY_FAMILIES:
            for lane in LANES:
                row = result["evaluations"][f"{seed}:{family}"][lane][
                    "capture_available-002"
                ]["search"]
                first = f"{row['first_visit_simulation']['1']}/{row['q_after_first_visit']['1']:.4f} {row['first_visit_simulation']['2']}/{row['q_after_first_visit']['2']:.4f}"
                priors = f"{row['root_prior'][1]:.4f}/{row['root_prior'][2]:.4f}"
                for snap in row["checkpoints"]:
                    a1, a2 = snapshot_move(snap, 1), snapshot_move(snap, 2)
                    ratio = (
                        a1["visit_count"] / a2["visit_count"]
                        if a2["visit_count"]
                        else float("inf")
                    )
                    gap = a1["q_value"] - a2["q_value"]
                    lines.append(
                        f"| {seed} | {family} | {lane} | {priors} | {first} | {snap['simulation']} | {snap['root_q']:.4f} | {a1['visit_count']}/{a1['q_value']:.4f} | {a2['visit_count']}/{a2['q_value']:.4f} | {ratio:.3f} | {gap:.4f} | {snap['selected_move']} | {snap['exact_regret']:.0f} |"
                    )
    lines += [
        "",
        "## Decision-Critical Results",
        "",
        "| Seed | Family | State | Zero move/regret | Parent-Q move/regret |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for seed in SEEDS:
        for family in PRIMARY_FAMILIES:
            for identifier in ("capture_available-002", "capture_available-020"):
                zero = result["evaluations"][f"{seed}:{family}"]["root_fpu_zero"][
                    identifier
                ]["search"]
                parent_q = result["evaluations"][f"{seed}:{family}"][
                    "root_fpu_parent_q"
                ][identifier]["search"]
                lines.append(
                    f"| {seed} | {family} | {identifier} | {zero['selected_action']}/{zero['regret']:.0f} | {parent_q['selected_action']}/{parent_q['regret']:.0f} |"
                )
    lines += [
        "",
        "## Capture And Global Safety",
        "",
        "The 24 capture rows and all exact-covered roots are stored individually in the JSON artifact. Per-checkpoint global metrics and per-bucket metrics below compare parent-Q minus zero.",
        "",
        "| Seed | Family | Lane | Accuracy | Mean regret | Median regret | Blunder rate | Correction balance |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for family in CHECKPOINTS:
            for lane in LANES:
                summary = result["summaries"]["global_exact"]["by_checkpoint"][
                    f"{seed}:{family}"
                ][lane]
                overall = summary["overall"]
                balance = summary["transitions"]["overall"].get("correction_balance", 0)
                lines.append(
                    f"| {seed} | {family} | {lane} | {overall['exact_optimal_set_accuracy']:.4f} | {overall['mean_exact_regret']:.4f} | {overall['median_exact_regret']:.4f} | {overall['exact_blunder_rate']:.4f} | {balance} |"
                )
    lines += [
        "",
        "| Seed | Family | Bucket | Accuracy delta | Mean regret delta | Blunder delta | Correction-balance delta |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for family in CHECKPOINTS:
            by_lane = result["summaries"]["global_exact"]["by_checkpoint"][
                f"{seed}:{family}"
            ]
            zero, parent_q = by_lane["root_fpu_zero"], by_lane["root_fpu_parent_q"]
            for bucket in zero["per_bucket"]:
                delta = metric_delta(
                    parent_q["per_bucket"][bucket], zero["per_bucket"][bucket]
                )
                balance = parent_q["transitions"]["per_bucket"][bucket].get(
                    "correction_balance", 0
                ) - zero["transitions"]["per_bucket"][bucket].get(
                    "correction_balance", 0
                )
                lines.append(
                    f"| {seed} | {family} | {bucket} | {delta['exact_optimal_set_accuracy']:.4f} | {delta['mean_exact_regret']:.4f} | {delta['exact_blunder_rate']:.4f} | {balance} |"
                )
    lines += [""]
    for key, value in result["summaries"].items():
        lines += [
            f"### {key}",
            "",
            "| Lane | Accuracy | Mean regret | Median regret | Blunder rate | Correction balance |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for lane in LANES:
            summary = value[lane]["overall"]
            balance = value[lane]["transitions"]["overall"].get("correction_balance", 0)
            lines.append(
                f"| {lane} | {summary['exact_optimal_set_accuracy']:.4f} | {summary['mean_exact_regret']:.4f} | {summary['median_exact_regret']:.4f} | {summary['exact_blunder_rate']:.4f} | {balance} |"
            )
        lines.append("")
    lines += [
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
    suite_buckets = {item.id: item.bucket for item in load_suite(SUITE)}
    exact_rows = [
        row for row in exact_payload["rows"] if row["exact_status"] == "exact_solved"
    ]
    exact_by_id = {row["id"]: row for row in exact_rows}
    capture = [row for row in exact_rows if row["id"].startswith("capture_available-")]
    critical_ids = ["capture_available-002", "capture_available-020"]
    if (
        hashlib.sha256("\n".join(critical_ids).encode()).hexdigest()
        != "58c7ca4677136570b7e5154b1bf3d531772839bd6a66117e6beab84e3f883c19"
    ):
        raise RuntimeError("decision-critical ordered-ID SHA mismatch")
    pr298 = json.loads(PR298.read_text())
    evaluators: dict[tuple[int, str], CheckpointEvaluator] = {}
    inputs: dict[str, Any] = {
        "exact_v2": {"path": str(EXACT), "sha256": sha256_file(EXACT)},
        "pr298": {"path": str(PR298), "sha256": sha256_file(PR298)},
        "checkpoints": {},
    }
    for seed in SEEDS:
        for family in CHECKPOINTS:
            path = checkpoint_path(seed, family)
            digest = sha256_file(path)
            expected = (
                EXPECTED_CHECKPOINTS[seed][family]
                if family != "parent"
                else pr298["inputs"]["checkpoints"][f"{seed}:parent"]["sha256"]
            )
            if digest != expected:
                raise RuntimeError(f"checkpoint SHA mismatch: {seed}:{family}")
            inputs["checkpoints"][f"{seed}:{family}"] = {
                "path": str(path),
                "sha256": digest,
            }
            evaluators[seed, family] = CheckpointEvaluator(
                path, input_encoding="kalah_v3"
            )
    # Fail closed before the intervention if frozen production PUCT has drifted.
    for seed in SEEDS:
        for family in PRIMARY_FAMILIES:
            evaluator = evaluators[seed, family]
            for identifier in critical_ids:
                raw = raw_row(exact_by_id[identifier], evaluator)
                expected = baseline_expected(pr298, seed, family, identifier)
                if not np.allclose(
                    raw["policy"], expected["raw"]["policy"], rtol=0, atol=1e-6
                ):
                    raise RuntimeError("PR #298 baseline raw policy did not reproduce")
                actual = searched_row(exact_by_id[identifier], evaluator, "zero")
                assert_baseline(actual, expected)
    evaluations: dict[str, Any] = {}
    populations = {
        "decision_critical": critical_ids,
        "capture_safety": [row["id"] for row in capture],
        "global_exact": [row["id"] for row in exact_rows],
    }
    for seed in SEEDS:
        for family in CHECKPOINTS:
            key = f"{seed}:{family}"
            evaluations[key] = {lane: {} for lane in LANES}
            evaluator = evaluators[seed, family]
            for identifier, exact in exact_by_id.items():
                raw = raw_row(exact, evaluator)
                for lane, root_fpu_mode in LANES.items():
                    search = searched_row(
                        exact,
                        evaluator,
                        root_fpu_mode,
                        full_trace=identifier == "capture_available-002",
                    )
                    evaluations[key][lane][identifier] = {
                        "bucket": suite_buckets[identifier],
                        "raw": raw,
                        "search": search,
                    }
    summaries: dict[str, Any] = {}
    for population, identifiers in populations.items():
        summaries[population] = {}
        for lane in LANES:
            rows = [
                evaluations[f"{seed}:{family}"][lane][identifier]
                for seed in SEEDS
                for family in CHECKPOINTS
                for identifier in identifiers
            ]
            summaries[population][lane] = summarize(rows)
        summaries[population]["by_checkpoint"] = {
            f"{seed}:{family}": {
                lane: summarize(
                    [
                        evaluations[f"{seed}:{family}"][lane][identifier]
                        for identifier in identifiers
                    ]
                )
                for lane in LANES
            }
            for seed in SEEDS
            for family in CHECKPOINTS
        }
    uniform_repairs = all(
        evaluations[f"{seed}:uniform1200"]["root_fpu_parent_q"][
            "capture_available-002"
        ]["search"]["regret"]
        == 0
        for seed in (44, 45)
    )
    seed46_ok = (
        evaluations["46:uniform1200"]["root_fpu_parent_q"]["capture_available-002"][
            "search"
        ]["regret"]
        == 0
    )
    controls_ok = all(
        evaluations[f"{seed}:control"]["root_fpu_parent_q"]["capture_available-002"][
            "search"
        ]["regret"]
        == 0
        for seed in SEEDS
    )
    capture_safety = all(
        sum(
            evaluations[f"{seed}:uniform1200"]["root_fpu_parent_q"][row["id"]][
                "search"
            ]["regret"]
            > 0
            for row in capture
        )
        <= sum(
            evaluations[f"{seed}:uniform1200"]["root_fpu_zero"][row["id"]]["search"][
                "regret"
            ]
            > 0
            for row in capture
        )
        for seed in SEEDS
    )
    new_exact_blunders = [
        f"{seed}:{family}:{identifier}"
        for seed in SEEDS
        for family in CHECKPOINTS
        for identifier in exact_by_id
        if evaluations[f"{seed}:{family}"]["root_fpu_zero"][identifier]["search"][
            "regret"
        ]
        == 0
        and evaluations[f"{seed}:{family}"]["root_fpu_parent_q"][identifier]["search"][
            "regret"
        ]
        > 0
    ]
    if not (uniform_repairs and seed46_ok and controls_ok):
        classification = "root_parent_q_not_causal"
        next_experiment = "Exactly one next experiment: use the existing root-Q confidence override to test visited-root-child Q calibration while keeping FPU zero."
    elif not capture_safety or new_exact_blunders:
        classification = "root_parent_q_repairs_capture_but_causes_tradeoff"
        next_experiment = "Exactly one next experiment: diagnose root-FPU application timing/conditions from the successful and harmed root trajectories; do not train a model."
    else:
        classification = "root_parent_q_repairs_capture_regression"
        next_experiment = "Exactly one next experiment: run a full SHADOW production evaluation of root-parent-Q versus zero FPU, including sealed arena play and exact forensics, before considering any production search change."
    result = {
        "schema": "azlite_root_fpu_capture_diagnostic_v1",
        "read_only": {"training": False, "self_play": False, "promotion": False},
        "inherited_pr298": {
            "head": "72a4a969b73dbfa93f3d956487c47b158be37fee",
            "classification": "capture_regression_search_induced",
        },
        "inputs": inputs,
        "populations": {
            key: {"count": len(value), "ids": value}
            for key, value in populations.items()
        },
        "baseline_reproduction": {"passed": True, "tolerance": 1e-6},
        "search_configuration": {
            "simulations": 384,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "normalize_values": False,
            "value_transform": None,
            "root_policy_mode": "deterministic",
            "root_temperature": 0.0,
            "tactical_root_bias": 0.0,
            "seed": 42,
            "dirichlet": None,
        },
        "capture_safety": {"uniform_blunder_count_preserved_per_seed": capture_safety},
        "new_exact_blunders": new_exact_blunders,
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
    args.report.write_text(render_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
