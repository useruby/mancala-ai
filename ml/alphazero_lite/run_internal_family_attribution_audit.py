#!/usr/bin/env python3
"""Read-only first internal outcome-degradation attribution for PR #302."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite import run_root_fpu_capture_diagnostic as fpu
from ml.alphazero_lite import run_true_outcome_subtree_attribution_audit as inherited
from ml.alphazero_lite.forensic_exact_references import sha256_file
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_uniform1200_replay_distribution_audit import artifact_paths
from ml.alphazero_lite.self_play import CheckpointEvaluator, terminal_value

AUDIT = ROOT / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit.json"
LABELS = (
    ROOT / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit-labels.json"
)
EXACT = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v2.json"
PRIMARY = ((44, "capture_available-025"), (45, "capture_available-018"))


def root_utility(
    label: dict[str, Any] | int, root_player: int, state_player: int
) -> int:
    """Convert the exact value at a state to the original root player's utility."""
    value = int(label if isinstance(label, int) else label["exact_root_value"])
    return value if root_player == state_player else -value


def state_root_utility(
    label: dict[str, Any], state: dict[str, Any], root_player: int
) -> int:
    if label.get("exact_status") == "terminal":
        value = terminal_value(KalahGame.from_state(state))
        assert value is not None
        return int(value if state["current_player"] == root_player else -value)
    return root_utility(label, root_player, state["current_player"])


def tactical_type(state: dict[str, Any], action: int) -> str:
    consequence = move_consequence_for_state(state, action)
    capture, extra = consequence["produces_capture"], consequence["gives_extra_turn"]
    if capture and extra:
        return "capture_and_extra_turn"
    if capture:
        return "capture"
    if extra:
        return "extra_turn"
    return "neither"


def legal_bucket(count: int) -> str:
    return "2" if count == 2 else "3-4" if count <= 4 else "5-6"


def family_signature(event: dict[str, Any]) -> str:
    optimal = event["optimal_tactical_types"]
    return "|".join(
        (
            f"win_to_{'draw' if event['utility_after'] == 0 else 'loss'}",
            event["tactical_type"],
            "+".join(sorted(optimal)),
            legal_bucket(event["legal_action_count"]),
        )
    )


def first_degradation(
    path: dict[str, Any], labels: dict[str, dict[str, Any]], root_player: int
) -> dict[str, Any] | None:
    """Return at most one root-player win-loss transition for a trace path."""
    root_action = path["path"][0]
    root_label = labels[canonical_state_key(path["nodes"][0]["state"])]
    if int(root_label["exact_outcome_utilities"][str(root_action)]) != 1:
        return None
    for node in path["nodes"]:
        state = node["state"]
        if state["current_player"] != root_player:
            continue
        label = labels[canonical_state_key(state)]
        chosen = int(node["chosen_action"])
        game = KalahGame.from_state(state)
        game.move(game.pit_index(chosen))
        post = game.to_state()
        post_label = labels[canonical_state_key(post)]
        before = state_root_utility(label, state, root_player)
        after = state_root_utility(post_label, post, root_player)
        optimal = [int(a) for a in label["exact_outcome_optimal_actions"]]
        if before == 1 and chosen not in optimal and after <= 0:
            consequence = move_consequence_for_state(state, chosen)
            event = {
                "simulation": path["simulation"],
                "root_action": root_action,
                "depth": node["depth"],
                "canonical_pre_action_state": canonical_state_key(state),
                "canonical_post_action_state": canonical_state_key(post),
                "player_to_move": root_player,
                "selected_action": chosen,
                "utility_before": before,
                "utility_after": after,
                "outcome_regret": 1
                - int(label["exact_outcome_utilities"][str(chosen)]),
                "optimal_actions": optimal,
                "tactical_type": tactical_type(state, chosen),
                "optimal_tactical_types": sorted(
                    {tactical_type(state, a) for a in optimal}
                ),
                "captures": bool(consequence["produces_capture"]),
                "extra_turn": bool(consequence["gives_extra_turn"]),
                "legal_action_count": len(game.possible_moves()),
                "state": state,
                "selection": node["decision"],
            }
            if not (
                event["player_to_move"] == root_player and before == 1 and after <= 0
            ):
                raise RuntimeError(
                    "invalid first-degradation perspective reconstruction"
                )
            return event
    return None


def select_family(events: list[dict[str, Any]]) -> str | None:
    """Pre-registered family selection: count, both roots, rate, lexical ID."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[family_signature(event)].append(event)
    eligible = [(name, rows) for name, rows in grouped.items() if len(rows) >= 10]
    if not eligible:
        return None

    def key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, float, str]:
        name, rows = item
        roots = len({row["case"] for row in rows})
        rate = len(rows) / sum(row["opportunities"] for row in rows)
        return (len(rows), roots == 2, rate, "".join(chr(255 - ord(c)) for c in name))

    return max(eligible, key=key)[0]


def raw_policy(
    evaluator: CheckpointEvaluator, state: dict[str, Any], optimal: list[int]
) -> dict[str, Any]:
    policy, value = evaluator.evaluate(KalahGame.from_state(state))
    legal = KalahGame.from_state(state).possible_moves()
    masked = [float(policy[action]) if action in legal else 0.0 for action in range(6)]
    top = min(legal, key=lambda action: (-masked[action], action))
    return {
        "policy": masked,
        "top_action": top,
        "optimal_mass": sum(masked[a] for a in optimal),
        "top_is_optimal": top in optimal,
        "value_prediction": float(value),
    }


def stage(raw: dict[str, Any], event: dict[str, Any]) -> str:
    raw_good = bool(raw["top_is_optimal"])
    search_good = event["selected_action"] in event["optimal_actions"]
    return f"raw_prior_{'good' if raw_good else 'bad'}_search_{'good' if search_good else 'bad'}"


def entropy(policy: list[float]) -> float:
    return -sum(p * math.log2(p) for p in policy if p > 0)


def replay_lookup(seed: int, family: str, keys: set[str]) -> dict[str, dict[str, Any]]:
    path = artifact_paths(ROOT, seed)[family] / "self_play.jsonl"
    found: dict[str, list[list[float]]] = defaultdict(list)
    parents: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        # The existing audit owns replay state decoding; use its canonical representation.
        from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import (
            decode_state,
        )

        state = decode_state(row["state"])
        key = canonical_state_key(state)
        if key in keys:
            found[key].append([float(x) for x in row["policy"]])
        game = KalahGame.from_state(state)
        for action in game.possible_moves():
            child = game.clone()
            child.move(child.pit_index(action))
            if canonical_state_key(child.to_state()) in keys:
                parents[key] += 1
    result = {}
    for key in keys:
        rows = found[key]
        policy = (
            [sum(row[i] for row in rows) / len(rows) for i in range(6)]
            if rows
            else None
        )
        result[key] = {
            "dynamic_count": len(rows),
            "policy": policy,
            "parent_count": parents[key],
            "fixed_sources": "not persisted in the verified historical run",
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    inherited_result, labels = (
        json.loads(AUDIT.read_text()),
        json.loads(LABELS.read_text())["labels"],
    )
    if (
        inherited_result["hard_classification"]
        != "true_outcome_subtree_policy_failure_primary"
    ):
        raise RuntimeError("unexpected inherited classification")
    exact = {row["id"]: row for row in json.loads(EXACT.read_text())["rows"]}
    cases = [
        (seed, family, identifier)
        for seed, identifier in PRIMARY
        for family in ("control", "uniform1200", "parent")
    ]
    cases += [
        (46, family, identifier)
        for _, identifier in PRIMARY
        for family in ("control", "uniform1200")
    ]
    all_events, all_opportunities, evaluations = [], defaultdict(int), {}
    for seed, family, identifier in cases:
        checkpoint = fpu.checkpoint_path(seed, family)
        digest = sha256_file(checkpoint)
        expected = inherited_result["inputs"]["checkpoints"][f"{seed}:{family}"]
        if digest != expected:
            raise RuntimeError(f"inherited checkpoint SHA mismatch: {seed}:{family}")
        evaluator = CheckpointEvaluator(checkpoint, input_encoding="kalah_v3")
        search, trace = inherited.trace_search(exact[identifier], evaluator)
        historical = inherited_result["cases"][f"{seed}:{family}:{identifier}"][
            "audit"
        ]["search"]
        if (
            search["selected_action"] != historical["selected_action"]
            or search["visits"] != historical["visits"]
        ):
            raise RuntimeError(
                f"trace failed to reproduce PR #302 baseline: {seed}:{family}:{identifier}"
            )
        paths = inherited.reconstruct_trace(exact[identifier]["state"], trace)
        root_player = exact[identifier]["state"]["current_player"]
        case = f"{seed}:{family}:{identifier}"
        evaluations[case] = evaluator
        for path in paths:
            for node in path["nodes"]:
                if node["state"]["current_player"] == root_player:
                    all_opportunities[(case, canonical_state_key(node["state"]))] += 1
            event = first_degradation(path, labels, root_player)
            if event:
                event["case"] = case
                event["cohort"] = (
                    "failing_uniform"
                    if family == "uniform1200" and seed in (44, 45)
                    else "matched_control"
                    if family == "control" and seed in (44, 45)
                    else "negative_control"
                    if seed == 46
                    else "parent"
                )
                all_events.append(event)
    for event in all_events:
        event["opportunities"] = all_opportunities[
            (event["case"], event["canonical_pre_action_state"])
        ]
        event["family"] = family_signature(event)
    selected = select_family(
        [e for e in all_events if e["cohort"] == "failing_uniform"]
    )
    selected_events = (
        [e for e in all_events if e["family"] == selected] if selected else []
    )
    raw_by_case_state = {}
    for event in selected_events:
        key = (event["case"], event["canonical_pre_action_state"])
        raw_by_case_state.setdefault(
            key,
            raw_policy(
                evaluations[event["case"]], event["state"], event["optimal_actions"]
            ),
        )
        event["raw"] = raw_by_case_state[key]
        event["stage"] = stage(event["raw"], event)
    # Family selection occurred above, before raw/replay inspection.
    family_rows = []
    for family in sorted({e["family"] for e in all_events}):
        for cohort in (
            "failing_uniform",
            "matched_control",
            "negative_control",
            "parent",
        ):
            rows = [
                e for e in all_events if e["family"] == family and e["cohort"] == cohort
            ]
            if rows:
                family_rows.append(
                    {
                        "family": family,
                        "cohort": cohort,
                        "events": len(rows),
                        "opportunities": sum(e["opportunities"] for e in rows),
                        "rate": len(rows) / sum(e["opportunities"] for e in rows),
                        "roots": len({e["case"] for e in rows}),
                        "states": len({e["canonical_pre_action_state"] for e in rows}),
                        "median_depth": statistics.median(e["depth"] for e in rows),
                        "win_to_draw": sum(e["utility_after"] == 0 for e in rows),
                        "win_to_loss": sum(e["utility_after"] == -1 for e in rows),
                    }
                )
    state_rows = []
    for key, rows in sorted(
        (
            (k, [e for e in all_events if e["canonical_pre_action_state"] == k])
            for k in {e["canonical_pre_action_state"] for e in all_events}
        ),
        key=lambda x: -len(x[1]),
    ):
        state_rows.append(
            {
                "state": key,
                "events": len(rows),
                "actions": sorted({e["selected_action"] for e in rows}),
                "opportunities": sum(e["opportunities"] for e in rows),
                "rate": len(rows) / sum(e["opportunities"] for e in rows),
                "cases": sorted({e["case"] for e in rows}),
                "median_depth": statistics.median(e["depth"] for e in rows),
            }
        )
    replay = {}
    for seed, family in (
        (44, "control"),
        (44, "uniform1200"),
        (45, "control"),
        (45, "uniform1200"),
    ):
        keys = {
            e["canonical_pre_action_state"]
            for e in selected_events
            if e["case"].startswith(f"{seed}:{family}:")
        }
        replay[f"{seed}:{family}"] = replay_lookup(seed, family, keys) if keys else {}
    raw_bad = sum(not e["raw"]["top_is_optimal"] for e in selected_events)
    repairs = sum(e["stage"] == "raw_prior_bad_search_good" for e in selected_events)
    mechanism = (
        "internal_policy_prior_failure"
        if selected_events
        and raw_bad / len(selected_events) >= 0.6
        and repairs / raw_bad < 0.5
        else "internal_policy_failure_heterogeneous"
    )
    hard = (
        "repeated_internal_prior_failure_identified"
        if mechanism == "internal_policy_prior_failure"
        else "repeated_internal_failure_heterogeneous"
    )
    depths = [e["depth"] for e in all_events if e["cohort"] == "failing_uniform"]
    result = {
        "schema": "azlite_internal_first_degradation_family_audit_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "promotion": False,
            "search_changes": False,
        },
        "inherited": {
            "classification": inherited_result["hard_classification"],
            "audit_sha256": sha256_file(AUDIT),
            "labels_sha256": sha256_file(LABELS),
            "checkpoints": inherited_result["inputs"]["checkpoints"],
        },
        "selected_family": selected,
        "events": all_events,
        "canonical_states": state_rows,
        "family_rows": family_rows,
        "replay": replay,
        "depth": {
            "min": min(depths),
            "p25": statistics.quantiles(depths, n=4, method="inclusive")[0],
            "median": statistics.median(depths),
            "p75": statistics.quantiles(depths, n=4, method="inclusive")[2],
            "le2": sum(d <= 2 for d in depths) / len(depths),
            "le3": sum(d <= 3 for d in depths) / len(depths),
            "le4": sum(d <= 4 for d in depths) / len(depths),
            "ge5": sum(d >= 5 for d in depths) / len(depths),
        },
        "mechanism": mechanism,
        "hard_classification": hard,
        "next_experiment": "Exactly one next experiment: run a family-specific replay/target provenance audit to determine why the raw policy learned the degrading move; do not change search."
        if hard.endswith("prior_failure_identified")
        else "Exactly one next experiment: take capture_available-018 first and isolate its highest-contribution internal family.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    selected_roots = sorted(
        {
            event["case"]
            for event in selected_events
            if event["cohort"] == "failing_uniform"
        }
    )
    lines = [
        "# First Repeated Internal Outcome-Degradation Audit",
        "",
        "## Inherited Inputs",
        "",
        f"Inherited classification: `{result['inherited']['classification']}`.",
        f"PR #302 audit SHA: `{result['inherited']['audit_sha256']}`.",
        f"Label-manifest SHA: `{result['inherited']['labels_sha256']}`.",
        "",
        "## Deterministic Selection",
        "",
        f"Selected family: `{selected}`.",
        f"Primary-root occurrences: `{', '.join(selected_roots)}`.",
        "This family is root-specific: its mechanically highest count did not occur in both failing roots.",
        "",
        "## Family Table",
        "",
        "| Family | Cohort | Events | Opportunities | Rate | Roots |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| `{r['family']}` | {r['cohort']} | {r['events']} | {r['opportunities']} | {r['rate']:.3f} | {r['roots']} |"
        for r in family_rows
    ]
    lines += [
        "",
        "## First-Degradation Depth",
        "",
        json.dumps(result["depth"], sort_keys=True),
        "",
        "## Local Attribution",
        "",
        f"Selected-family events: {len(selected_events)}; raw-policy non-optimal top actions: {raw_bad}; raw-bad search repairs: {repairs}.",
        "Raw policies, masked probabilities, local PUCT child statistics, exact alternatives, and replay exact-state lookup are retained per event in the machine artifact.",
        "",
        "## Classification",
        "",
        f"Family mechanism: `{mechanism}`.",
        f"Hard classification: `{hard}`.",
        "",
        "## Next Experiment",
        "",
        result["next_experiment"],
    ]
    args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
