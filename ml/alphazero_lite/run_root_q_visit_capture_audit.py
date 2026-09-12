#!/usr/bin/env python3
"""Read-only root-child Q versus visit-allocation diagnostic for PR #299."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite import run_capture_family_attribution_audit as attribution
from ml.alphazero_lite import run_root_fpu_capture_diagnostic as fpu
from ml.alphazero_lite.forensic_exact_references import exact_regret, sha256_file
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT

PR299 = ROOT / "docs/data/alphazero-lite-root-q-confidence-capture-diagnostic.json"
STABILITY_WINDOW = 64
EARLY_CHECKPOINTS = (8, 16, 32, 64, 96, 128)
VISIT_CURVE_POINTS = (1, 2, 4, 8, 16, 32, 64, 96, 128)


def leader(moves: list[dict[str, Any]], field: str) -> int:
    return int(
        max(moves, key=lambda row: (float(row[field]), -int(row["move"])))["move"]
    )


def ranks(moves: list[dict[str, Any]], field: str) -> dict[int, int]:
    ordered = sorted(moves, key=lambda row: (-float(row[field]), int(row["move"])))
    return {int(row["move"]): index for index, row in enumerate(ordered, 1)}


def exact_values(exact: dict[str, Any]) -> dict[int, float]:
    return {
        int(move): float(value) for move, value in exact["exact_action_values"].items()
    }


def pairwise_q_order_accuracy(
    moves: list[dict[str, Any]], exact: dict[str, Any]
) -> float | None:
    values = exact_values(exact)
    by_move = {int(row["move"]): row for row in moves}
    correct = total = 0
    for action, value in values.items():
        for other, other_value in values.items():
            if action == other or value <= other_value:
                continue
            total += 1
            correct += float(by_move[action]["q_value"]) > float(
                by_move[other]["q_value"]
            )
    return correct / total if total else None


def correction_times(flags: list[bool]) -> tuple[int | None, int | None]:
    first = next((index + 1 for index, flag in enumerate(flags) if flag), None)
    stable = next(
        (index + 1 for index in range(len(flags)) if all(flags[index:])), None
    )
    return first, stable


def spearman_q_exact(
    moves: list[dict[str, Any]], exact: dict[str, Any]
) -> float | None:
    """Spearman correlation with average ranks for tied Q or exact values."""
    values = exact_values(exact)

    def average_ranks(items: dict[int, float]) -> dict[int, float]:
        ordered = sorted(items, key=lambda action: (-items[action], action))
        result, index = {}, 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and items[ordered[end]] == items[ordered[index]]:
                end += 1
            rank = (index + 1 + end) / 2
            result.update({action: rank for action in ordered[index:end]})
            index = end
        return result

    q_ranks = average_ranks({int(row["move"]): float(row["q_value"]) for row in moves})
    exact_ranks = average_ranks(values)
    if len(q_ranks) < 2:
        return None
    q_mean = sum(q_ranks.values()) / len(q_ranks)
    e_mean = sum(exact_ranks.values()) / len(exact_ranks)
    numerator = sum(
        (q_ranks[key] - q_mean) * (exact_ranks[key] - e_mean) for key in q_ranks
    )
    q_scale = sum((value - q_mean) ** 2 for value in q_ranks.values())
    e_scale = sum((value - e_mean) ** 2 for value in exact_ranks.values())
    return (
        None if not q_scale or not e_scale else numerator / (q_scale * e_scale) ** 0.5
    )


def q_rank_reversals(
    trace: list[dict[str, Any]], action: int, competitor: int
) -> dict[str, Any]:
    signs = []
    for row in trace:
        moves = {int(item["move"]): item for item in row["moves"]}
        signs.append(
            (float(moves[action]["q_value"]) > float(moves[competitor]["q_value"]))
            - (float(moves[action]["q_value"]) < float(moves[competitor]["q_value"]))
        )
    reversals = [
        index + 1
        for index in range(1, len(signs))
        if signs[index] and signs[index - 1] and signs[index] != signs[index - 1]
    ]
    final = reversals[-1] if reversals else None
    final_visits = (
        None
        if final is None
        else next(
            item["visit_count"]
            for item in trace[final - 1]["moves"]
            if int(item["move"]) == action
        )
    )
    return {
        "count": len(reversals),
        "simulations": reversals,
        "final_simulation": final,
        "action_visit_at_final": final_visits,
    }


def classify(metrics: dict[str, Any]) -> str:
    if metrics["visit_top_exact_final"]:
        return "search_correct"
    if not metrics["q_top_exact_final"]:
        return "persistent_q_ranking_error"
    stable = metrics["stable_q_correct_sim"]
    if stable is not None and stable <= 384 - STABILITY_WINDOW:
        return "visit_hysteresis_after_q_recovery"
    if stable is not None:
        return "late_q_recovery_insufficient_time"
    return "mixed_or_unstable"


def audit_trace(
    exact: dict[str, Any], evaluator: CheckpointEvaluator
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    game = KalahGame.from_state(exact["state"])
    trace: list[dict[str, Any]] = []
    search = PUCT(
        evaluator,
        384,
        1.25,
        random.Random(42),
        fpu_mode="zero",
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        value_transform=None,
        root_audit_trace=trace,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return {
        "selected_action": search.select_root_move(root, game.possible_moves()),
        "regret": exact_regret(
            exact, search.select_root_move(root, game.possible_moves())
        ),
        "visits": [int(value) for value in visits],
        "root_prior": [
            float(root.children[move].prior) if move in root.children else 0.0
            for move in range(6)
        ],
        "child_q": {
            str(move): float(child.q_value) for move, child in root.children.items()
        },
    }, trace


def analyze_trace(exact: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    optimum = set(int(move) for move in exact["exact_optimal_actions"])
    values = exact_values(exact)
    q_flags, visit_flags, interval = [], [], []
    timeline = []
    for row in trace:
        moves = row["moves"]
        q_leader, visit_leader, score_leader = (
            leader(moves, "q_value"),
            leader(moves, "visit_count"),
            leader(moves, "selection_score"),
        )
        q_ok, visit_ok = q_leader in optimum, visit_leader in optimum
        q_flags.append(q_ok)
        visit_flags.append(visit_ok)
        interval.append(q_ok and not visit_ok)
        timeline.append(
            {
                "simulation": row["simulation"],
                "root_q": row["root_q"],
                "selected_action": row["selected_action"],
                "q_leader": q_leader,
                "visit_leader": visit_leader,
                "puct_score_leader": score_leader,
                "q_top_exact": q_ok,
                "visit_top_exact": visit_ok,
                "pairwise_q_order_accuracy": pairwise_q_order_accuracy(moves, exact),
                "spearman_q_exact": spearman_q_exact(moves, exact),
                "exact_best_q_rank": min(
                    ranks(moves, "q_value")[move] for move in optimum
                ),
                "exact_best_visit_rank": min(
                    ranks(moves, "visit_count")[move] for move in optimum
                ),
                "moves": moves,
            }
        )
    first_q, stable_q = correction_times(q_flags)
    first_v, stable_v = correction_times(visit_flags)
    final = timeline[-1]
    best = min(optimum)
    wrong = final["visit_leader"] if final["visit_leader"] not in optimum else None
    interval_indices = [
        row["simulation"] for row, flag in zip(timeline, interval, strict=True) if flag
    ]
    after_stable = interval[stable_q - 1 :] if stable_q else []
    curves = {}
    for action in values:
        observed = {}
        for row in timeline:
            move = next(item for item in row["moves"] if int(item["move"]) == action)
            count = int(move["visit_count"])
            if count in VISIT_CURVE_POINTS or count >= 128:
                observed.setdefault(
                    str(count),
                    {
                        "simulation": row["simulation"],
                        "q": move["q_value"],
                        "q_rank": ranks(row["moves"], "q_value")[action],
                        "exact_rank": sorted(
                            values, key=lambda item: (-values[item], item)
                        ).index(action)
                        + 1,
                    },
                )
        curves[str(action)] = observed
    metrics = {
        "q_top_exact_final": q_flags[-1],
        "visit_top_exact_final": visit_flags[-1],
        "first_q_correct_sim": first_q,
        "stable_q_correct_sim": stable_q,
        "first_visit_correct_sim": first_v,
        "stable_visit_correct_sim": stable_v,
        "q_to_visit_recovery_lag": None
        if stable_q is None or stable_v is None
        else stable_v - stable_q,
        "q_correct_visit_wrong": {
            "count": len(interval_indices),
            "first": interval_indices[0] if interval_indices else None,
            "last": interval_indices[-1] if interval_indices else None,
            "at_384": interval[-1],
            "fraction_after_stable_q": sum(after_stable) / len(after_stable)
            if after_stable
            else None,
        },
        "final": {
            "visit_action": final["visit_leader"],
            "q_action": final["q_leader"],
            "puct_action": final["puct_score_leader"],
            "visit_regret": exact_regret(exact, final["visit_leader"]),
            "q_regret": exact_regret(exact, final["q_leader"]),
            "puct_regret": exact_regret(exact, final["puct_score_leader"]),
        },
        "visit_margin_best_minus_wrong": None
        if wrong is None
        else next(
            item["visit_count"] for item in final["moves"] if int(item["move"]) == best
        )
        - next(
            item["visit_count"] for item in final["moves"] if int(item["move"]) == wrong
        ),
        "curves": curves,
    }
    if wrong is not None:
        debt = []
        for checkpoint in EARLY_CHECKPOINTS:
            row = timeline[checkpoint - 1]
            by_move = {int(item["move"]): item for item in row["moves"]}
            debt.append(
                {
                    "simulation": checkpoint,
                    "prior_ratio_wrong_to_best": by_move[wrong]["prior"]
                    / by_move[best]["prior"],
                    "visit_ratio_wrong_to_best": by_move[wrong]["visit_count"]
                    / by_move[best]["visit_count"]
                    if by_move[best]["visit_count"]
                    else None,
                    "q_gap_best_minus_wrong": by_move[best]["q_value"]
                    - by_move[wrong]["q_value"],
                    "u_gap_best_minus_wrong": by_move[best]["u_component"]
                    - by_move[wrong]["u_component"],
                    "puct_gap_best_minus_wrong": by_move[best]["selection_score"]
                    - by_move[wrong]["selection_score"],
                    "early_visit_debt": by_move[wrong]["visit_count"]
                    - by_move[best]["visit_count"],
                }
            )
        metrics["early_visit_debt"] = {
            "checkpoints": debt,
            "remaining_at_384": -metrics["visit_margin_best_minus_wrong"],
        }
    metrics["mechanism"] = classify(metrics)
    if 1 in values and 2 in values:
        gaps = [
            {
                "simulation": row["simulation"],
                "q_gap_action2_minus_action1": next(
                    item["q_value"] for item in row["moves"] if int(item["move"]) == 2
                )
                - next(
                    item["q_value"] for item in row["moves"] if int(item["move"]) == 1
                ),
            }
            for row in timeline
        ]
        metrics["action_2_minus_1_q_gap"] = {
            "values": gaps,
            "zero_crossings": [
                row["simulation"]
                for row, prior in zip(gaps[1:], gaps[:-1], strict=True)
                if row["q_gap_action2_minus_action1"]
                * prior["q_gap_action2_minus_action1"]
                < 0
            ],
            "rank_reversals": q_rank_reversals(trace, 2, 1),
        }
    return {"timeline": timeline, "metrics": metrics}


def compact_sparse(search: dict[str, Any], exact: dict[str, Any]) -> dict[str, Any]:
    snapshots = search["checkpoints"]
    final = snapshots[-1]
    moves = [
        {
            "move": int(move),
            "q_value": values[2],
            "visit_count": values[1],
            "selection_score": values[6],
        }
        for move, values in final["moves"].items()
    ]
    return {
        "q_top_exact": leader(moves, "q_value") in exact["exact_optimal_actions"],
        "visit_top_exact": leader(moves, "visit_count")
        in exact["exact_optimal_actions"],
        "visit_regret": exact_regret(exact, leader(moves, "visit_count")),
        "q_regret": exact_regret(exact, leader(moves, "q_value")),
        "mechanism": "persistent_q_ranking_error"
        if leader(moves, "q_value") not in exact["exact_optimal_actions"]
        else (
            "search_correct"
            if leader(moves, "visit_count") in exact["exact_optimal_actions"]
            else "mixed_or_unstable"
        ),
    }


def aggregate_sparse(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    total = len(values)
    return {
        "roots": total,
        "q_top_exact_accuracy": sum(row["q_top_exact"] for row in values) / total,
        "visit_top_exact_accuracy": sum(row["visit_top_exact"] for row in values)
        / total,
        "q_correct_visit_wrong_rate": sum(
            row["q_top_exact"] and not row["visit_top_exact"] for row in values
        )
        / total,
        "visit_wrong_q_exact_rate": sum(
            not row["visit_top_exact"] and row["q_regret"] == 0 for row in values
        )
        / total,
        "both_q_and_visit_wrong_rate": sum(
            not row["visit_top_exact"] and not row["q_top_exact"] for row in values
        )
        / total,
        "mean_q_minus_visit_regret": sum(
            row["q_regret"] - row["visit_regret"] for row in values
        )
        / total,
        "mechanisms": {
            name: sum(row["mechanism"] == name for row in values)
            for name in sorted({row["mechanism"] for row in values})
        },
    }


def report(result: dict[str, Any]) -> str:
    lines = [
        "# Root Q/Visit Capture Audit",
        "",
        "## Inherited Inputs",
        "",
        f"PR #299 machine artifact SHA256: `{result['inputs']['pr299']['sha256']}`. PR #298 attribution SHA256: `{result['inputs']['pr298']['sha256']}`.",
        "",
        "## Baseline Reproduction",
        "",
        "The frozen root-Q-alpha 1.00, zero-FPU production configuration reproduced selected action, regret, visits, and child Q (1e-6) for both decision-critical roots and all six primary checkpoints before telemetry was interpreted.",
        "",
        "## Decision-Critical Results",
        "",
        "| Seed | Family | State | Stable Q | Stable visits | Q-to-visit lag | Final visit/Q regret | Mechanism |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for key, rows in result["decision_critical"].items():
        seed, family = key.split(":")
        for identifier, row in rows.items():
            metric = row["metrics"]
            final = metric["final"]
            lines.append(
                f"| {seed} | {family} | {identifier} | {metric['stable_q_correct_sim']} | {metric['stable_visit_correct_sim']} | {metric['q_to_visit_recovery_lag']} | {final['visit_regret']:.0f}/{final['q_regret']:.0f} | `{metric['mechanism']}` |"
            )
    lines += [
        "",
        "## Capture-002 Timelines",
        "",
        "The machine artifact stores all 384 post-backup root-child rows, including pre-selection PUCT components, selected edge, and Q/visit leaders. Q is evaluated only as an ordering; it is never numerically compared with exact margins.",
        "",
        "| Seed | Family | Q-correct/visit-wrong interval | Q rank reversals A2/A1 | A2-A1 Q zero crossings | Early debt at 128 | Final debt |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for seed in fpu.SEEDS:
        for family in fpu.PRIMARY_FAMILIES:
            metric = result["decision_critical"][f"{seed}:{family}"][
                "capture_available-002"
            ]["metrics"]
            interval = metric["q_correct_visit_wrong"]
            gap = metric["action_2_minus_1_q_gap"]
            debt = metric.get(
                "early_visit_debt", {"checkpoints": [], "remaining_at_384": None}
            )
            debt128 = next(
                (
                    item["early_visit_debt"]
                    for item in debt["checkpoints"]
                    if item["simulation"] == 128
                ),
                None,
            )
            lines.append(
                f"| {seed} | {family} | {interval['first']}-{interval['last']} ({interval['count']}) | {gap['rank_reversals']['count']} | {gap['zero_crossings']} | {debt128} | {debt['remaining_at_384']} |"
            )
    lines += [
        "",
        "## Capture And Global Validation",
        "",
        "The artifact retains PR #299 sparse snapshots for every exact capture root and every exact forensic root. These are secondary final-snapshot checks; full-resolution tracing is restricted to the mechanically selected decision-critical set.",
        "",
    ]
    lines += [
        "| Seed | Family | Capture Q-top | Capture visit-top | Q-correct/visit-wrong | Both wrong |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, summary in result["capture_aggregates"].items():
        seed, family = key.split(":")
        lines.append(
            f"| {seed} | {family} | {summary['q_top_exact_accuracy']:.3f} | {summary['visit_top_exact_accuracy']:.3f} | {summary['q_correct_visit_wrong_rate']:.3f} | {summary['both_q_and_visit_wrong_rate']:.3f} |"
        )
    lines += [
        "",
        "Global exact final snapshots are also aggregated in the machine artifact, including Q-argmax minus visit-argmax exact regret. These use decision ordering/regret only, never a numeric Q-to-exact calibration.",
        "",
    ]
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
    pr298, pr299 = json.loads(fpu.PR298.read_text()), json.loads(PR299.read_text())
    exact_rows = [
        row
        for row in json.loads(fpu.EXACT.read_text())["rows"]
        if row["exact_status"] == "exact_solved"
    ]
    exact_by_id = {row["id"]: row for row in exact_rows}
    critical = attribution.decision_critical_ids(
        json.loads(attribution.SHADOW.read_text())
    )
    if critical != pr298["decision_critical_capture_set"]["ids"]:
        raise RuntimeError("PR #298 decision-critical population drift")
    evaluators, inputs = (
        {},
        {
            "pr298": {"sha256": sha256_file(fpu.PR298)},
            "pr299": {"sha256": sha256_file(PR299)},
            "exact_v2": {"sha256": sha256_file(fpu.EXACT)},
            "checkpoints": {},
        },
    )
    for seed in fpu.SEEDS:
        for family in fpu.CHECKPOINTS:
            path = fpu.checkpoint_path(seed, family)
            digest = sha256_file(path)
            expected = fpu.EXPECTED_CHECKPOINTS[seed].get(
                family, pr298["inputs"]["checkpoints"][f"{seed}:parent"]["sha256"]
            )
            if digest != expected:
                raise RuntimeError(f"checkpoint SHA mismatch: {seed}:{family}")
            inputs["checkpoints"][f"{seed}:{family}"] = digest
            evaluators[seed, family] = CheckpointEvaluator(
                path, input_encoding="kalah_v3"
            )
    for seed in fpu.SEEDS:
        for family in fpu.PRIMARY_FAMILIES:
            for identifier in critical:
                actual, _ = audit_trace(
                    exact_by_id[identifier], evaluators[seed, family]
                )
                fpu.assert_baseline(
                    actual, fpu.baseline_expected(pr298, seed, family, identifier)
                )
    decision = {}
    for seed in fpu.SEEDS:
        for family in fpu.CHECKPOINTS:
            decision[f"{seed}:{family}"] = {}
            for identifier in critical:
                search, trace = audit_trace(
                    exact_by_id[identifier], evaluators[seed, family]
                )
                decision[f"{seed}:{family}"][identifier] = {
                    "search": search,
                    **analyze_trace(exact_by_id[identifier], trace),
                }
    capture = [row for row in exact_rows if row["id"].startswith("capture_available-")]
    sparse = {
        f"{seed}:{family}": {
            row["id"]: compact_sparse(
                pr299["evaluations"][f"{seed}:{family}"]["root_q_alpha_100"][row["id"]][
                    "search"
                ],
                row,
            )
            for row in capture
        }
        for seed in fpu.SEEDS
        for family in fpu.PRIMARY_FAMILIES
    }
    all_sparse = {
        f"{seed}:{family}": {
            identifier: compact_sparse(item["search"], exact_by_id[identifier])
            for identifier, item in pr299["evaluations"][f"{seed}:{family}"][
                "root_q_alpha_100"
            ].items()
        }
        for seed in fpu.SEEDS
        for family in fpu.PRIMARY_FAMILIES
    }
    failing = [
        decision[f"{seed}:uniform1200"]["capture_available-002"]["metrics"]["mechanism"]
        for seed in (44, 45)
    ]
    classification = (
        "capture_failure_mixed_q_and_visit"
        if "persistent_q_ranking_error" in failing
        and "visit_hysteresis_after_q_recovery" in failing
        else (
            "capture_failure_q_estimation_primary"
            if "persistent_q_ranking_error" in failing
            else "capture_q_visit_diagnostic_inconclusive"
        )
    )
    result = {
        "schema": "azlite_root_q_visit_capture_audit_v1",
        "read_only": {"training": False, "self_play": False, "promotion": False},
        "inputs": inputs,
        "search_configuration": {
            "simulations": 384,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_q_alpha": 1.0,
            "normalize_values": False,
            "root_policy_mode": "deterministic",
            "root_temperature": 0.0,
            "tactical_root_bias": 0.0,
            "seed": 42,
            "dirichlet": None,
        },
        "populations": {
            "decision_critical": critical,
            "capture_available": [row["id"] for row in capture],
            "global_exact_count": len(exact_rows),
        },
        "baseline_reproduction": {"passed": True, "tolerance": 1e-6},
        "decision_critical": decision,
        "capture_sparse": sparse,
        "capture_aggregates": {
            key: aggregate_sparse(rows) for key, rows in sparse.items()
        },
        "global_sparse": all_sparse,
        "global_aggregates": {
            key: aggregate_sparse(rows) for key, rows in all_sparse.items()
        },
        "hard_classification": classification,
        "next_experiment": "Exactly one next experiment: inspect subtree evaluator/backup error for the persistent-Q case FIRST, because changing visit allocation cannot fix a genuinely wrong Q ranking.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
