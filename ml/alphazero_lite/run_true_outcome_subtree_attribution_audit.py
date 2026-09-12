#!/usr/bin/env python3
"""Read-only subtree evaluator and backup attribution for PR #301 regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite import run_root_fpu_capture_diagnostic as fpu  # noqa: E402
from ml.alphazero_lite.forensic_exact_references import (  # noqa: E402
    exact_regret,
    outcome_optimal_actions,
    outcome_regret,
    outcome_utilities,
    sha256_file,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key  # noqa: E402
from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    NativeHybridProcess,
    native_label_payload,
    root_training_value,
    validate_native_response,
)
from ml.alphazero_lite.run_uniform1200_tier21_gate import (  # noqa: E402
    TIER21_SHA256,
    validate_inputs,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator, PUCT, terminal_value  # noqa: E402

EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
PR298 = ROOT / "docs/data/alphazero-lite-capture-family-attribution-audit.json"
PR301 = ROOT / "docs/data/alphazero-lite-exact-forensic-outcome-margin-audit.json"
PRIMARY = ((44, "capture_available-025", 0), (45, "capture_available-018", 1))
NEGATIVE_SEED = 46
SIMULATIONS = 384
SEARCH_SEED = 42
TIMEOUT_SECONDS = 120.0


def state_hash(game: KalahGame) -> str:
    return hashlib.sha256(
        json.dumps(
            game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def root_perspective(value: float, leaf_player: int, root_player: int) -> float:
    """Production backup sign: player identity, never depth parity."""
    return float(value if leaf_player == root_player else -value)


def reconstruct_trace(
    root_state: dict[str, Any], trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replay every selected path from the frozen root and verify leaf identity."""
    rows = []
    for simulation in trace:
        game = KalahGame.from_state(root_state)
        nodes = []
        path: list[int] = []
        for decision in simulation["selection_path"]:
            if state_hash(game) != decision["state_hash"]:
                raise RuntimeError("selection-path reconstructed state hash mismatch")
            if game.current_player != decision["player_to_move"]:
                raise RuntimeError("selection-path reconstructed player mismatch")
            move = int(decision["chosen_move"])
            if (
                game.possible_moves() != decision["legal_moves"]
                or move not in game.possible_moves()
            ):
                raise RuntimeError("selection-path reconstructed legal moves mismatch")
            nodes.append(
                {
                    "depth": len(path),
                    "path": list(path),
                    "state": game.to_state(),
                    "chosen_action": move,
                    "decision": decision,
                }
            )
            path.append(move)
            if not game.move(game.pit_index(move)):
                raise RuntimeError("selection-path replay selected an illegal move")
        if state_hash(game) != simulation["selected_leaf_state_hash"]:
            raise RuntimeError("reconstructed leaf hash does not match trace")
        if game.current_player != simulation["selected_leaf_player_to_move"]:
            raise RuntimeError("reconstructed leaf player does not match trace")
        rows.append(
            {
                "simulation": simulation["simulation_index"],
                "path": path,
                "nodes": nodes,
                "leaf_state": game.to_state(),
                "leaf_hash": state_hash(game),
                "leaf_terminal": bool(simulation["terminal_leaf"]),
                "leaf_value": float(simulation["leaf_evaluator_value"]),
                "leaf_player": game.current_player,
                "backed_up_value": float(simulation["backed_up_value"]),
            }
        )
    return rows


def label_state(game: KalahGame, process: NativeHybridProcess) -> dict[str, Any]:
    native = process.request(native_label_payload(game.to_state()), TIMEOUT_SECONDS)
    values = {
        int(action): int(value) for action, value in native["action_values"].items()
    }
    margin = validate_native_response(
        values,
        [int(a) for a in native["optimal_actions"]],
        int(native["exact_value"]),
        game.possible_moves(),
        game.current_player,
    )
    row = {
        "exact_status": "exact_solved",
        "state": game.to_state(),
        "exact_root_value": root_training_value(margin, game.current_player),
        "exact_action_values": {str(a): v for a, v in values.items()},
    }
    row["exact_outcome_utilities"] = {
        str(a): v for a, v in outcome_utilities(row).items()
    }
    row["exact_outcome_optimal_actions"] = outcome_optimal_actions(row)
    return row


def exact_leaf_value(
    label: dict[str, Any], terminal: bool, state: dict[str, Any]
) -> float | None:
    if terminal:
        return terminal_value(KalahGame.from_state(state))
    return (
        None
        if label.get("exact_status") != "exact_solved"
        else float(label["exact_root_value"])
    )


def q_leader(values: dict[int, float]) -> int:
    return max(values, key=lambda action: (values[action], -action))


def stable_first(flags: list[bool]) -> int | None:
    return next((index + 1 for index in range(len(flags)) if all(flags[index:])), None)


def classify(
    *,
    coverage_ok: bool,
    backup_mismatches: int,
    repaired: bool,
    self_degradations: int,
    solved_terminal_fraction: float,
) -> str:
    if not coverage_ok:
        return "root_attribution_inconclusive"
    if backup_mismatches:
        return "backup_perspective_fault"
    if repaired:
        return "leaf_evaluator_error"
    if self_degradations:
        return "subtree_policy_error"
    if solved_terminal_fraction == 0:
        return "search_horizon_error"
    return "root_attribution_inconclusive"


def trace_search(
    exact: dict[str, Any], evaluator: CheckpointEvaluator
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    game = KalahGame.from_state(exact["state"])
    search = PUCT(
        evaluator,
        SIMULATIONS,
        1.25,
        random.Random(SEARCH_SEED),
        fpu_mode="zero",
        normalize_values=False,
        root_policy_mode="deterministic",
        root_temperature=0.0,
        tactical_root_bias=0.0,
        value_transform=None,
        selection_trace=trace,
    )
    visits, root = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    selected = search.select_root_move(root, game.possible_moves())
    return {
        "selected_action": selected,
        "margin_regret": exact_regret(exact, selected),
        "outcome_regret": outcome_regret(exact, selected),
        "visits": [int(v) for v in visits],
        "root_prior": [
            float(root.children[m].prior) if m in root.children else 0.0
            for m in range(6)
        ],
        "child_q": {str(m): float(c.q_value) for m, c in root.children.items()},
    }, trace


def analysis(
    root_id: str,
    root_exact: dict[str, Any],
    search: dict[str, Any],
    paths: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root_player = int(root_exact["state"]["current_player"])
    backup_mismatches = []
    by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
    internal: list[dict[str, Any]] = []
    counterfactual_sums: dict[int, float] = defaultdict(float)
    counterfactual_visits: dict[int, int] = defaultdict(int)
    cf_timeline = []
    for row in paths:
        independent = root_perspective(
            row["leaf_value"], row["leaf_player"], root_player
        )
        if abs(independent - row["backed_up_value"]) > 1e-12:
            backup_mismatches.append(row["simulation"])
        action = row["path"][0]
        label = labels[canonical_state_key(row["leaf_state"])]
        exact_value = exact_leaf_value(label, row["leaf_terminal"], row["leaf_state"])
        leaf = row | {"exact_value": exact_value}
        by_action[action].append(leaf)
        if exact_value is not None:
            counterfactual_sums[action] += root_perspective(
                exact_value, row["leaf_player"], root_player
            )
            counterfactual_visits[action] += 1
        cf_timeline.append(
            {
                "simulation": row["simulation"],
                "q": {
                    str(a): counterfactual_sums[a] / counterfactual_visits[a]
                    if counterfactual_visits[a]
                    else 0.0
                    for a in map(int, root_exact["exact_action_values"])
                },
            }
        )
        for node in row["nodes"]:
            node_label = labels[canonical_state_key(node["state"])]
            if node_label.get("exact_status") != "exact_solved":
                continue
            optimal = set(node_label["exact_outcome_optimal_actions"])
            internal.append(
                {
                    "root_action": action,
                    "depth": node["depth"],
                    "path": node["path"],
                    "player": node["state"]["current_player"],
                    "chosen": node["chosen_action"],
                    "optimal": node["chosen_action"] in optimal,
                    "alternatives": node_label["exact_outcome_utilities"],
                    "decision": node["decision"],
                }
            )
    root_utilities = outcome_utilities(root_exact)
    optimum = set(outcome_optimal_actions(root_exact))
    wrong = search["selected_action"]
    baseline_q = {int(a): float(q) for a, q in search["child_q"].items()}
    cf_final = {
        int(a): counterfactual_sums[int(a)] / counterfactual_visits[int(a)]
        if counterfactual_visits[int(a)]
        else 0.0
        for a in root_exact["exact_action_values"]
    }
    repaired_flags = [
        q_leader({int(a): float(v) for a, v in row["q"].items()}) in optimum
        and q_leader({int(a): float(v) for a, v in row["q"].items()}) != wrong
        for row in cf_timeline
    ]
    repaired = (
        q_leader(baseline_q) == wrong
        and q_leader(cf_final) in optimum
        and q_leader(cf_final) != wrong
    )
    action_rows = {}
    for action, leaves in sorted(by_action.items()):
        solved = [leaf for leaf in leaves if leaf["exact_value"] is not None]
        errors = [
            leaf["leaf_value"] - leaf["exact_value"]
            for leaf in solved
            if not leaf["leaf_terminal"]
        ]
        action_rows[str(action)] = {
            "simulations": len(leaves),
            "unique_exact_solved_leaves": len({leaf["leaf_hash"] for leaf in solved}),
            "exact_wdl_distribution": dict(
                Counter(str(int(leaf["exact_value"])) for leaf in solved)
            ),
            "mean_neural_leaf_value": statistics.fmean(
                leaf["leaf_value"] for leaf in leaves
            ),
            "mean_exact_leaf_utility": statistics.fmean(
                leaf["exact_value"] for leaf in solved
            )
            if solved
            else None,
            "mean_signed_evaluator_error": statistics.fmean(errors) if errors else None,
            "wrong_sign_rate": sum(
                (leaf["leaf_value"] > 0) != (leaf["exact_value"] > 0)
                for leaf in solved
                if not leaf["leaf_terminal"]
            )
            / len(errors)
            if errors
            else None,
            "terminal_leaf_fraction": sum(leaf["leaf_terminal"] for leaf in leaves)
            / len(leaves),
            "mean_search_depth": statistics.fmean(len(leaf["path"]) for leaf in leaves),
        }
    self_degrade = [
        item
        for item in internal
        if item["player"] == root_player and not item["optimal"]
    ]
    opponent_bad = [
        item
        for item in internal
        if item["player"] != root_player and not item["optimal"]
    ]
    unique_leaves = {
        leaf["leaf_hash"]: leaf for leaves in by_action.values() for leaf in leaves
    }
    all_nodes = {
        canonical_state_key(item["state"]): item["state"]
        for row in paths
        for item in row["nodes"]
    }
    nonterminal_leaves = [
        leaf for leaf in unique_leaves.values() if not leaf["leaf_terminal"]
    ]
    leaf_coverage = (
        sum(
            labels[canonical_state_key(leaf["leaf_state"])].get("exact_status")
            == "exact_solved"
            for leaf in nonterminal_leaves
        )
        / len(nonterminal_leaves)
        if nonterminal_leaves
        else 1.0
    )
    internal_coverage = (
        sum(labels[key].get("exact_status") == "exact_solved" for key in all_nodes)
        / len(all_nodes)
        if all_nodes
        else 1.0
    )
    coverage_ok = leaf_coverage >= 0.95 and internal_coverage >= 0.90
    mechanism = classify(
        coverage_ok=coverage_ok,
        backup_mismatches=len(backup_mismatches),
        repaired=repaired,
        self_degradations=len(self_degrade),
        solved_terminal_fraction=sum(
            leaf["leaf_terminal"] for leaf in unique_leaves.values()
        )
        / len(unique_leaves),
    )
    return {
        "search": search,
        "root_outcome_utilities": root_utilities,
        "root_outcome_optimal_actions": sorted(optimum),
        "baseline_backup_parity": {
            "simulations_audited": len(paths),
            "mismatches": len(backup_mismatches),
            "mismatch_rate": len(backup_mismatches) / len(paths),
            "first_mismatch": backup_mismatches[0] if backup_mismatches else None,
        },
        "exact_coverage": {
            "unique_nonterminal_leaves": len(nonterminal_leaves),
            "leaf_fraction": leaf_coverage,
            "internal_trace_fraction": internal_coverage,
            "by_depth": dict(Counter(str(len(row["path"])) for row in paths)),
        },
        "root_action_leaf_bias": action_rows,
        "same_path_exact_backup": {
            "baseline_q": baseline_q,
            "exact_leaf_counterfactual_q": cf_final,
            "same_path_exact_q_repair": repaired,
            "first_stable_repair_simulation": stable_first(repaired_flags),
        },
        "internal_action_quality": {
            "exact_solved_nodes": len(internal),
            "root_player_self_degradation": len(self_degrade),
            "opponent_suboptimality": len(opponent_bad),
            "self_degradation_rate": len(self_degrade)
            / sum(item["player"] == root_player for item in internal)
            if internal
            else 0.0,
            "opponent_suboptimality_rate": len(opponent_bad)
            / sum(item["player"] != root_player for item in internal)
            if internal
            else 0.0,
        },
        "mechanism": mechanism,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# True-Outcome Subtree Evaluator/Backup Attribution Audit",
        "",
        "## Inherited Classification",
        "",
        "`uniform1200_true_outcome_regression_confirmed` from PR #301. The primary population is exactly seed 44 / capture_available-025 and seed 45 / capture_available-018; capture-002 and capture-020 are excluded as margin-only regressions.",
        "",
        "## Baseline And Coverage",
        "",
        "All traced baselines reproduced their historical selected moves before post-hoc labels were requested. Checkpoint and oracle SHAs are in the machine artifact.",
        "",
        "| Case | Selected | Regret | Leaf coverage | Internal coverage | Backup mismatches | Mechanism |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, row in result["cases"].items():
        if row["role"] != "primary":
            continue
        audit = row["audit"]
        lines.append(
            f"| {key} | {audit['search']['selected_action']} | {audit['search']['outcome_regret']} | {audit['exact_coverage']['leaf_fraction']:.3f} | {audit['exact_coverage']['internal_trace_fraction']:.3f} | {audit['baseline_backup_parity']['mismatches']} | `{audit['mechanism']}` |"
        )
    lines += [
        "",
        "## Same-Path Exact Backup",
        "",
        "Exact leaf W/D/L values are applied only offline to the recorded paths; visits and selections remain historical.",
        "",
    ]
    for key, row in result["cases"].items():
        if row["role"] == "primary":
            cf = row["audit"]["same_path_exact_backup"]
            lines.append(
                f"- `{key}`: repair={cf['same_path_exact_q_repair']}, first stable repair={cf['first_stable_repair_simulation']}, baseline Q={cf['baseline_q']}, exact-leaf Q={cf['exact_leaf_counterfactual_q']}."
            )
    lines += [
        "",
        "## Controls",
        "",
        "Seed-46 control/uniform traces for both roots and available SHA-verified parents are retained in the machine artifact as negative and contextual controls.",
        "",
        "## Classification",
        "",
        f"`{result['hard_classification']}`",
        "",
        "## One Next Experiment",
        "",
        result["next_experiment"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    oracle = validate_inputs(args.native_probe, args.tablebase)
    exact_by_id = {
        row["id"]: row
        for row in json.loads(EXACT.read_text())["rows"]
        if row["exact_status"] == "exact_solved"
    }
    pr298 = json.loads(PR298.read_text())
    historical = json.loads(PR301.read_text())["capture_margin_failures"]
    cases: list[tuple[int, str, str]] = [
        (seed, family, identifier)
        for seed, identifier, _wrong in PRIMARY
        for family in ("control", "uniform1200")
    ]
    cases += [
        (NEGATIVE_SEED, family, identifier)
        for _seed, identifier, _wrong in PRIMARY
        for family in ("control", "uniform1200")
    ]
    cases += [
        (seed, "parent", identifier)
        for seed, identifier, _wrong in PRIMARY
        if fpu.checkpoint_path(seed, "parent").is_file()
    ]
    traces, searches, evaluators, checkpoint_shas = {}, {}, {}, {}
    for seed, family, identifier in cases:
        path = fpu.checkpoint_path(seed, family)
        digest = sha256_file(path)
        expected = fpu.EXPECTED_CHECKPOINTS[seed].get(
            family, pr298["inputs"]["checkpoints"][f"{seed}:parent"]["sha256"]
        )
        if digest != expected:
            raise RuntimeError(f"checkpoint SHA mismatch: {seed}:{family}")
        checkpoint_shas[f"{seed}:{family}"] = digest
        evaluators.setdefault(
            (seed, family), CheckpointEvaluator(path, input_encoding="kalah_v3")
        )
        search, trace = trace_search(exact_by_id[identifier], evaluators[seed, family])
        if family in {"control", "uniform1200"} and seed in (44, 45):
            if family == "uniform1200":
                expected_move = next(
                    item["selected_move"]
                    for item in historical[str(seed)]
                    if item["id"] == identifier
                )
            else:
                shadow = json.loads(
                    (
                        ROOT
                        / f"docs/data/alphazero-lite-exact-forensic-shadow-seed{seed}.json"
                    ).read_text()
                )
                expected_move = next(
                    item["selected_move"]
                    for item in shadow["systems"]["current"]["rows"]
                    if item["id"] == identifier
                )
            if search["selected_action"] != expected_move:
                raise RuntimeError(
                    "historical true-regression baseline did not reproduce"
                )
        searches[seed, family, identifier] = search
        traces[seed, family, identifier] = reconstruct_trace(
            exact_by_id[identifier]["state"], trace
        )
    states = {}
    for exact in exact_by_id.values():
        row = dict(exact)
        row["exact_outcome_utilities"] = {
            str(action): value for action, value in outcome_utilities(row).items()
        }
        row["exact_outcome_optimal_actions"] = outcome_optimal_actions(row)
        states[canonical_state_key(row["state"])] = row
    for trace in traces.values():
        for row in trace:
            states.setdefault(
                canonical_state_key(row["leaf_state"]),
                {"exact_status": "terminal"}
                if row["leaf_terminal"]
                else {"exact_status": "pending", "state": row["leaf_state"]},
            )
            for node in row["nodes"]:
                states.setdefault(
                    canonical_state_key(node["state"]),
                    {"exact_status": "pending", "state": node["state"]},
                )
    cache_path = args.out.with_name(f"{args.out.stem}-labels.json")
    if cache_path.exists():
        cached = json.loads(cache_path.read_text()).get("labels", {})
        for key, label in cached.items():
            if key in states and label.get("exact_status") in {
                "exact_solved",
                "unresolved",
            }:
                states[key] = label
    process = NativeHybridProcess(args.native_probe, args.tablebase)
    try:
        for key, label in states.items():
            if label["exact_status"] != "pending":
                continue
            try:
                states[key] = label_state(KalahGame.from_state(label["state"]), process)
            except (TimeoutError, Exception) as error:
                states[key] = {
                    "exact_status": "unresolved",
                    "failure_reason": str(error),
                    "state": label["state"],
                }
                process.close()
                process = NativeHybridProcess(args.native_probe, args.tablebase)
            cache_path.write_text(
                json.dumps(
                    {
                        "schema": "azlite_true_outcome_subtree_attribution_labels_v1",
                        "oracle": oracle | {"timeout_seconds": TIMEOUT_SECONDS},
                        "labels": states,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        process.close()
    result_cases = {}
    for seed, family, identifier in cases:
        key = f"{seed}:{family}:{identifier}"
        role = (
            "primary"
            if (seed, identifier)
            in {(44, "capture_available-025"), (45, "capture_available-018")}
            else "negative_control"
            if seed == 46
            else "parent_context"
        )
        result_cases[key] = {
            "role": role,
            "checkpoint": checkpoint_shas[f"{seed}:{family}"],
            "audit": analysis(
                identifier,
                exact_by_id[identifier],
                searches[seed, family, identifier],
                traces[seed, family, identifier],
                states,
            ),
        }
    primary = [
        result_cases[f"{seed}:uniform1200:{identifier}"]["audit"]
        for seed, identifier, _wrong in PRIMARY
    ]
    mechanisms = [row["mechanism"] for row in primary]
    if "backup_perspective_fault" in mechanisms:
        hard, next_experiment = (
            "true_outcome_backup_semantics_fault",
            "Exactly one next action: fix and parity-test the backup semantics before any ML experiment.",
        )
    elif mechanisms == ["leaf_evaluator_error", "leaf_evaluator_error"]:
        hard, next_experiment = (
            "true_outcome_leaf_evaluator_failure_primary",
            "Exactly one next experiment: perform a tiny exact-W/D/L value-head calibration/retention audit on these failure families before training anything.",
        )
    elif mechanisms == ["subtree_policy_error", "subtree_policy_error"]:
        hard, next_experiment = (
            "true_outcome_subtree_policy_failure_primary",
            "Exactly one next experiment: diagnose the first repeated internal outcome-degrading decision family; do not alter the root search yet.",
        )
    elif mechanisms == ["search_horizon_error", "search_horizon_error"]:
        hard, next_experiment = (
            "true_outcome_search_horizon_failure_primary",
            "Exactly one next experiment: run an extended-budget diagnostic on ONLY these two roots and measure whether true W/D/L decisions converge.",
        )
    elif "root_attribution_inconclusive" in mechanisms:
        hard, next_experiment = (
            "true_outcome_subtree_attribution_inconclusive",
            "Exactly one next experiment: obtain sufficient exact descendant coverage without changing search.",
        )
    else:
        hard, next_experiment = (
            "true_outcome_evaluator_subtree_mixed",
            "Exactly one next experiment: take the higher-outcome-regret case first - capture_available-018 - and isolate its dominant mechanism with one causal counterfactual.",
        )
    result = {
        "schema": "azlite_true_outcome_subtree_attribution_audit_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "live_exact_injection": False,
        },
        "inputs": {
            "pr301_sha256": sha256_file(PR301),
            "exact_v2_sha256": sha256_file(EXACT),
            "checkpoints": checkpoint_shas,
            "oracle": oracle
            | {"timeout_seconds": TIMEOUT_SECONDS, "tablebase_sha256": TIER21_SHA256},
            "label_manifest": str(cache_path.relative_to(ROOT)),
            "label_manifest_sha256": sha256_file(cache_path),
        },
        "search_configuration": {
            "simulations": SIMULATIONS,
            "c_puct": 1.25,
            "fpu_mode": "zero",
            "root_q_confidence": 1.0,
            "deterministic_root_policy": True,
            "dirichlet": None,
            "value_normalization": False,
            "value_transform": None,
            "evaluation_seed": SEARCH_SEED,
        },
        "cases": result_cases,
        "hard_classification": hard,
        "next_experiment": next_experiment,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
