#!/usr/bin/env python3
"""Read-only outcome-versus-margin audit of committed exact forensic artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.alphazero_lite.forensic_exact_references import (
    exact_regret,
    outcome_optimal_actions,
    outcome_regret,
    outcome_regression,
    outcome_utilities,
    same_outcome_margin_regression,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
PR298 = ROOT / "docs/data/alphazero-lite-capture-family-attribution-audit.json"
PR300 = ROOT / "docs/data/alphazero-lite-root-q-visit-capture-audit.json"
SHADOW = "docs/data/alphazero-lite-exact-forensic-shadow-seed{}.json"
ROWMATCHED = "docs/data/alphazero-lite-exact-forensic-shadow-rowmatched-seed{}.json"
PR290 = "docs/data/alphazero-lite-exact-forensic-shadow-pr290-{}-seed{}.json"
THRESHOLDS = {
    "overall": {"accuracy": -0.02, "regret": 0.02, "blunder": 0.01},
    "capture_available": {"accuracy": -0.03, "regret": 0.03, "blunder": 0.02},
    "sparse_endgame": {"accuracy": -0.03, "regret": 0.03, "blunder": 0.02},
}


def margin_optimal_actions(row: dict[str, Any]) -> list[int]:
    return sorted(int(action) for action in row["exact_optimal_actions"])


def artifact_path(template: str, *args: object) -> Path:
    return ROOT / template.format(*args)


def audit_exact_row(row: dict[str, Any]) -> dict[str, Any]:
    utilities = outcome_utilities(row)
    optimal = outcome_optimal_actions(row)
    root_value = int(row["exact_root_value"])
    if max(utilities.values()) != root_value:
        raise ValueError(f"exact-root-value perspective mismatch: {row['id']}")
    margin = margin_optimal_actions(row)
    return {
        "id": row["id"],
        "bucket": row["id"].rsplit("-", 1)[0],
        "current_player": int(row["state"]["current_player"]),
        "raw_player_zero_margins": {
            str(action): int(value)
            for action, value in row["exact_action_values"].items()
        },
        "outcome_utilities": {
            str(action): value for action, value in utilities.items()
        },
        "margin_optimal_actions": margin,
        "outcome_optimal_actions": optimal,
        "forced_root_outcome": root_value,
        "margin_equals_outcome_optimal": margin == optimal,
        "outcome_optimal_count": len(optimal),
        "additional_outcome_equivalent_actions": len(optimal) - len(margin),
    }


def oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audited = [audit_exact_row(row) for row in rows]

    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "roots": count,
            "margin_equals_outcome_fraction": sum(
                item["margin_equals_outcome_optimal"] for item in items
            )
            / count,
            "outcome_set_strictly_larger_fraction": sum(
                item["outcome_optimal_count"] > len(item["margin_optimal_actions"])
                for item in items
            )
            / count,
            "mean_additional_outcome_equivalent_actions": sum(
                item["additional_outcome_equivalent_actions"] for item in items
            )
            / count,
            "multi_action_outcome_equivalence_rate": sum(
                item["outcome_optimal_count"] > 1 for item in items
            )
            / count,
        }

    return {
        "rows": audited,
        "overall": summary(audited),
        "by_bucket": {
            bucket: summary([row for row in audited if row["bucket"] == bucket])
            for bucket in sorted({row["bucket"] for row in audited})
        },
    }


def selected_metrics(exact: dict[str, Any], selected_move: int) -> dict[str, Any]:
    margin = exact_regret(exact, selected_move)
    outcome = outcome_regret(exact, selected_move)
    return {
        "selected_move": int(selected_move),
        "margin_optimal": margin == 0,
        "margin_regret": margin,
        "outcome_optimal": outcome == 0,
        "outcome_regret": outcome,
        "same_outcome_margin_regression": same_outcome_margin_regression(
            exact, selected_move
        ),
        "outcome_regression": outcome_regression(exact, selected_move),
    }


def summarize_selected(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    count = len(rows)
    return {
        "roots": count,
        "outcome_optimal_accuracy": sum(item[1]["outcome_optimal"] for item in rows)
        / count,
        "mean_outcome_regret": sum(item[1]["outcome_regret"] for item in rows) / count,
        "outcome_blunder_rate": sum(item[1]["outcome_regression"] for item in rows)
        / count,
        "margin_optimal_accuracy": sum(item[1]["margin_optimal"] for item in rows)
        / count,
        "mean_margin_regret": sum(item[1]["margin_regret"] for item in rows) / count,
        "same_outcome_margin_regression_rate": sum(
            item[1]["same_outcome_margin_regression"] for item in rows
        )
        / count,
    }


def replay_summary(
    payload: dict[str, Any], exact_by_id: dict[str, Any], system: str
) -> dict[str, Any]:
    rows = []
    for row in payload["systems"][system]["rows"]:
        exact = exact_by_id.get(row["id"])
        if exact is not None:
            rows.append((row, selected_metrics(exact, int(row["selected_move"]))))
    result = {
        "overall": summarize_selected(rows),
        "rows": {row["id"]: metric for row, metric in rows},
    }
    result["by_bucket"] = {
        bucket: summarize_selected(
            [item for item in rows if item[0]["bucket"] == bucket]
        )
        for bucket in sorted({row["bucket"] for row, _ in rows})
    }
    return result


def delta(uniform: dict[str, Any], control: dict[str, Any]) -> dict[str, float]:
    return {
        "outcome_optimal_accuracy": uniform["outcome_optimal_accuracy"]
        - control["outcome_optimal_accuracy"],
        "mean_outcome_regret": uniform["mean_outcome_regret"]
        - control["mean_outcome_regret"],
        "outcome_blunder_rate": uniform["outcome_blunder_rate"]
        - control["outcome_blunder_rate"],
        "margin_optimal_accuracy": uniform["margin_optimal_accuracy"]
        - control["margin_optimal_accuracy"],
        "mean_margin_regret": uniform["mean_margin_regret"]
        - control["mean_margin_regret"],
        "same_outcome_margin_regression_rate": uniform[
            "same_outcome_margin_regression_rate"
        ]
        - control["same_outcome_margin_regression_rate"],
    }


def shadow_decision(deltas: dict[str, dict[str, float]]) -> str:
    for scope, thresholds in THRESHOLDS.items():
        values = deltas[scope]
        if values["outcome_optimal_accuracy"] < thresholds["accuracy"]:
            return "fail"
        if values["mean_outcome_regret"] > thresholds["regret"]:
            return "fail"
        if values["outcome_blunder_rate"] > thresholds["blunder"]:
            return "fail"
    return "pass"


def q_visit_audit(pr300: dict[str, Any], exact_by_id: dict[str, Any]) -> dict[str, Any]:
    critical = ["capture_available-002", "capture_available-020"]
    result: dict[str, Any] = {}
    for key, states in pr300["decision_critical"].items():
        if key.split(":", 1)[1] not in {"control", "uniform1200"}:
            continue
        result[key] = {}
        for identifier in critical:
            final = states[identifier]["metrics"]["final"]
            exact = exact_by_id[identifier]
            q = selected_metrics(exact, final["q_action"])
            visit = selected_metrics(exact, final["visit_action"])
            result[key][identifier] = {
                "q_leader": q,
                "visit_leader": visit,
                "classification": (
                    "fully_optimal"
                    if q["margin_optimal"] and visit["margin_optimal"]
                    else None
                ),
                "true_q_outcome_error": q["outcome_regression"],
                "margin_only_q_difference": q["same_outcome_margin_regression"],
                "true_visit_outcome_error": visit["outcome_regression"],
                "margin_only_visit_difference": visit["same_outcome_margin_regression"],
            }
    return result


def decision_flip_outcomes(
    payload: dict[str, Any], exact_by_id: dict[str, Any]
) -> dict[str, int]:
    result: Counter[str] = Counter()
    current = {row["id"]: row for row in payload["systems"]["current"]["rows"]}
    uniform = {row["id"]: row for row in payload["systems"]["challenger"]["rows"]}
    for identifier, control in current.items():
        candidate = uniform[identifier]
        if (
            candidate.get("regret") is None
            or candidate["selected_move"] == control["selected_move"]
        ):
            continue
        exact = exact_by_id.get(identifier)
        if exact is None:
            continue
        control_utility = outcome_utilities(exact)[int(control["selected_move"])]
        uniform_utility = outcome_utilities(exact)[int(candidate["selected_move"])]
        label = {-1: "forced-loss", 0: "draw", 1: "forced-win"}
        result[f"{label[control_utility]}->{label[uniform_utility]}"] += 1
        if uniform_utility > control_utility:
            result["control_worse_uniform_better"] += 1
        if control_utility > uniform_utility:
            result["control_better_uniform_worse"] += 1
    return dict(sorted(result.items()))


def render(result: dict[str, Any]) -> str:
    oracle = result["oracle_comparison"]
    lines = [
        "# Exact Forensic Outcome-vs-Margin Audit",
        "",
        "## Semantics",
        "",
        "Native exact action values are player-zero final stone margins. Exact training values and PUCT terminal values are root-player forced win/draw/loss utilities (+1/0/-1). This read-only audit preserves margin regret and separately evaluates the latter objective.",
        "",
        "## Oracle Comparison",
        "",
        f"All {oracle['overall']['roots']} exact rows passed max-outcome-utility == exact_root_value.",
        "",
        "| Scope | Margin set equals outcome set | Outcome set larger | Mean extra equivalent actions | Multi-action outcome set |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scope, values in [("overall", oracle["overall"]), *oracle["by_bucket"].items()]:
        lines.append(
            f"| {scope} | {values['margin_equals_outcome_fraction']:.3f} | {values['outcome_set_strictly_larger_fraction']:.3f} | {values['mean_additional_outcome_equivalent_actions']:.3f} | {values['multi_action_outcome_equivalence_rate']:.3f} |"
        )
    lines += ["", "## Decision-Critical Tables", ""]
    for identifier, table in result["decision_critical_tables"].items():
        lines += [
            f"### {identifier}",
            "",
            "| Action | P0 margin | Root utility | Margin-optimal | Outcome-optimal | Control/44/45/46 selected |",
            "| ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in table:
            selected = "/".join(name for name in row["selected_by"] if name) or ""
            lines.append(
                f"| {row['action']} | {row['margin']} | {row['utility']:+d} | {row['margin_optimal']} | {row['outcome_optimal']} | {selected} |"
            )
        lines.append("")
    lines += [
        "## PR #287 Outcome Replay",
        "",
        "| Seed | Outcome accuracy delta | Outcome regret delta | Outcome blunder delta | Margin accuracy delta | Margin regret delta | Same-outcome margin delta | Old / outcome shadow |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for seed, item in result["pr287"].items():
        values = item["deltas"]["overall"]
        lines.append(
            f"| {seed} | {values['outcome_optimal_accuracy']:+.3f} | {values['mean_outcome_regret']:+.3f} | {values['outcome_blunder_rate']:+.3f} | {values['margin_optimal_accuracy']:+.3f} | {values['mean_margin_regret']:+.3f} | {values['same_outcome_margin_regression_rate']:+.3f} | fail / {item['outcome_aligned_shadow_decision']} |"
        )
    lines += [
        "",
        "## Margin Gate Failures",
        "",
        "The decision-critical failures are outcome-optimal; separately listed rows below are genuine W/D/L regressions.",
        "",
        "| Seed | State | Selected | Margin regret | Outcome regret | Classification |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for seed, failures in result["capture_margin_failures"].items():
        for failure in failures:
            kind = (
                "true outcome regression"
                if failure["outcome_regression"]
                else "same-outcome margin regression"
            )
            lines.append(
                f"| {seed} | {failure['id']} | {failure['selected_move']} | {failure['margin_regret']:.0f} | {failure['outcome_regret']} | {kind} |"
            )
    lines += [
        "",
        "## PR #300 Q/Visit Reinterpretation",
        "",
        "| Seed/family | State | Q margin/outcome sufficient | Visit margin/outcome sufficient | Outcome error type |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, states in result["q_visit_reinterpretation"].items():
        for identifier, values in states.items():
            q = values["q_leader"]
            visit = values["visit_leader"]
            kinds = []
            if values["margin_only_q_difference"]:
                kinds.append("margin-only Q")
            if values["margin_only_visit_difference"]:
                kinds.append("margin-only visit")
            if values["true_q_outcome_error"]:
                kinds.append("true Q outcome")
            if values["true_visit_outcome_error"]:
                kinds.append("true visit outcome")
            lines.append(
                f"| {key} | {identifier} | {q['margin_optimal']}/{q['outcome_optimal']} | {visit['margin_optimal']}/{visit['outcome_optimal']} | {', '.join(kinds) or 'fully optimal'} |"
            )
    lines += [
        "",
        "## Classification",
        "",
        f"`{result['hard_classification']}`",
        "",
        "## One Next Experiment",
        "",
        result["next_experiment"],
    ]
    return "\n".join(lines)


def main() -> int:
    exact_rows = [
        row
        for row in json.loads(EXACT.read_text())["rows"]
        if row["exact_status"] == "exact_solved"
    ]
    exact_by_id = {row["id"]: row for row in exact_rows}
    oracle = oracle_summary(exact_rows)
    pr300 = json.loads(PR300.read_text())
    pr287: dict[str, Any] = {}
    historical: dict[str, Any] = {}
    for name, template, systems in [
        ("pr287", SHADOW, ("current", "challenger")),
        ("pr288_rowmatched", ROWMATCHED, ("current", "challenger")),
        (
            "pr290_B",
            lambda seed: artifact_path(PR290, "B", seed),
            ("current", "challenger"),
        ),
        (
            "pr290_C",
            lambda seed: artifact_path(PR290, "C", seed),
            ("current", "challenger"),
        ),
        (
            "pr290_D",
            lambda seed: artifact_path(PR290, "D", seed),
            ("current", "challenger"),
        ),
    ]:
        historical[name] = {}
        for seed in (44, 45, 46):
            path = (
                artifact_path(template, seed)
                if isinstance(template, str)
                else template(seed)
            )
            payload = json.loads(path.read_text())
            control, uniform = (
                replay_summary(payload, exact_by_id, system) for system in systems
            )
            item = {
                "control": control,
                "uniform": uniform,
                "deltas": {"overall": delta(uniform["overall"], control["overall"])},
            }
            item["deltas"].update(
                {
                    bucket: delta(
                        uniform["by_bucket"][bucket], control["by_bucket"][bucket]
                    )
                    for bucket in control["by_bucket"]
                }
            )
            item["decision_flip_outcomes"] = decision_flip_outcomes(
                payload, exact_by_id
            )
            historical[name][str(seed)] = item
            if name == "pr287":
                item["outcome_aligned_shadow_decision"] = shadow_decision(
                    item["deltas"]
                )
                pr287[str(seed)] = item
    selected: dict[str, dict[str, int]] = {}
    for identifier in ("capture_available-002", "capture_available-020"):
        selected[identifier] = {}
        for seed in (44, 45, 46):
            payload = json.loads(artifact_path(SHADOW, seed).read_text())
            for system, label in (
                ("current", "control"),
                ("challenger", f"uniform{seed}"),
            ):
                row = next(
                    row
                    for row in payload["systems"][system]["rows"]
                    if row["id"] == identifier
                )
                selected[identifier][label] = int(row["selected_move"])
    tables = {}
    for identifier in ("capture_available-002", "capture_available-020"):
        exact = exact_by_id[identifier]
        utilities = outcome_utilities(exact)
        margin = set(margin_optimal_actions(exact))
        outcome = set(outcome_optimal_actions(exact))
        tables[identifier] = [
            {
                "action": action,
                "margin": int(exact["exact_action_values"][str(action)]),
                "utility": utilities[action],
                "margin_optimal": action in margin,
                "outcome_optimal": action in outcome,
                "selected_by": [
                    name
                    for name, selected_move in selected[identifier].items()
                    if selected_move == action
                ],
            }
            for action in sorted(utilities)
        ]
    capture_failures = {}
    for seed in (44, 45):
        payload = json.loads(artifact_path(SHADOW, seed).read_text())
        control = {row["id"]: row for row in payload["systems"]["current"]["rows"]}
        uniform = {row["id"]: row for row in payload["systems"]["challenger"]["rows"]}
        capture_failures[str(seed)] = []
        for identifier in exact_by_id:
            if not identifier.startswith("capture_available-"):
                continue
            if uniform[identifier]["regret"] > control[identifier]["regret"]:
                exact = exact_by_id[identifier]
                metric = selected_metrics(exact, uniform[identifier]["selected_move"])
                capture_failures[str(seed)].append(
                    {
                        "id": identifier,
                        "current_player": exact["state"]["current_player"],
                        "margin_optimal_actions": margin_optimal_actions(exact),
                        "outcome_optimal_actions": outcome_optimal_actions(exact),
                        **metric,
                    }
                )
    outcome_capture_nonregressions = [
        pr287[str(seed)]["deltas"]["capture_available"]["outcome_blunder_rate"] <= 0
        for seed in (44, 45, 46)
    ]
    overall_positive = all(
        pr287[str(seed)]["deltas"]["overall"]["outcome_optimal_accuracy"] > 0
        for seed in (44, 45, 46)
    )
    critical_zero = all(
        item["outcome_regret"] == 0
        for failures in capture_failures.values()
        for item in failures
        if item["id"] in {"capture_available-002", "capture_available-020"}
    )
    margin_mismatch_confirmed = (
        critical_zero and sum(outcome_capture_nonregressions) >= 2 and overall_positive
    )
    true_capture_regression = all(
        pr287[str(seed)]["deltas"]["capture_available"]["outcome_blunder_rate"] > 0
        for seed in (44, 45)
    )
    classification = (
        "exact_forensic_margin_objective_mismatch_confirmed"
        if margin_mismatch_confirmed
        else (
            "uniform1200_true_outcome_regression_confirmed"
            if true_capture_regression
            else "exact_forensic_objective_mismatch_mixed"
        )
    )
    next_experiment = (
        "Exactly one next experiment: run a fresh matched control-vs-uniform1200 confirmation using the outcome-aligned exact SHADOW forensic evaluation, while leaving production gate behavior unchanged."
        if classification == "exact_forensic_margin_objective_mismatch_confirmed"
        else (
            "Exactly one next experiment: perform the subtree evaluator/backup attribution audit on ONLY the true-outcome-regression states."
            if classification == "uniform1200_true_outcome_regression_confirmed"
            else "Exactly one next experiment: isolate the smallest repeated true-outcome failure family and perform evaluator/backup attribution only there."
        )
    )
    result = {
        "schema": "azlite_exact_forensic_outcome_margin_audit_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "mcts_rerun": False,
            "production_gate_modified": False,
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (EXACT, PR298, PR300)
        },
        "oracle_comparison": oracle,
        "capture_population": [
            row for row in oracle["rows"] if row["bucket"] == "capture_available"
        ],
        "decision_critical_set": json.loads(PR298.read_text())[
            "decision_critical_capture_set"
        ],
        "decision_critical_tables": tables,
        "capture_margin_failures": capture_failures,
        "pr287": pr287,
        "historical": historical,
        "q_visit_reinterpretation": q_visit_audit(pr300, exact_by_id),
        "hard_classification": classification,
        "next_experiment": next_experiment,
    }
    out = ROOT / "docs/data/alphazero-lite-exact-forensic-outcome-margin-audit.json"
    report = ROOT / "docs/alphazero-lite-exact-forensic-outcome-margin-audit.md"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report.write_text(render(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
