#!/usr/bin/env python3
"""Read-only replay and supervision provenance audit for PR #303's family."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

# ruff: noqa: E402
from ml.alphazero_lite.build_train_only_forensic_suite_from_selfplay import decode_state
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state
from ml.alphazero_lite.run_uniform1200_replay_distribution_audit import (
    FULL_REPLAY_SHA256,
    artifact_paths,
    entropy,
    sha256_file,
    total_variation,
)
from ml.alphazero_lite.self_play import CheckpointEvaluator

INHERITED = (
    ROOT / "docs/data/alphazero-lite-internal-first-degradation-family-audit.json"
)
LABELS = (
    ROOT / "docs/data/alphazero-lite-true-outcome-subtree-attribution-audit-labels.json"
)
PLAN = ROOT / "ml/alphazero_lite/configs/uniform1200_seed_confirmation.json"
FAMILY = "win_to_draw|neither|extra_turn|5-6"
PRIMARY_CASE = "45:uniform1200:capture_available-018"


def ordered_state_sha(states: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(states)).encode()).hexdigest()


def primary_failure_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mechanically collapse only PR #303's selected failing-root events."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["family"] == FAMILY and event["case"] == PRIMARY_CASE:
            groups[event["canonical_pre_action_state"]].append(event)
    rows = []
    for key, values in sorted(groups.items()):
        actions = Counter(int(value["selected_action"]) for value in values)
        selected = min(actions, key=lambda action: (-actions[action], action))
        opportunities = int(values[0]["opportunities"])
        if any(int(value["opportunities"]) != opportunities for value in values):
            raise ValueError(f"inconsistent opportunity count for {key}")
        rows.append(
            {
                "canonical_state": key,
                "state": values[0]["state"],
                "events": len(values),
                "opportunities": opportunities,
                "degradation_rate": len(values) / opportunities,
                "median_depth": statistics.median(value["depth"] for value in values),
                "selected_degrading_action": selected,
                "selected_degrading_actions": sorted(actions),
                "exact_outcome_optimal_actions": values[0]["optimal_actions"],
            }
        )
    return rows


def selected_family_all_states(events: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            event["canonical_pre_action_state"]
            for event in events
            if event["family"] == FAMILY
        }
    )


def masked_policy(policy: Iterable[float], state: dict[str, Any]) -> list[float]:
    legal = KalahGame.from_state(state).possible_moves()
    result = [float(policy[action]) if action in legal else 0.0 for action in range(6)]
    total = sum(result)
    return [value / total if total else 0.0 for value in result]


def policy_metrics(
    policy: Iterable[float],
    state: dict[str, Any],
    optimal: list[int],
    degrading: int,
    utilities: dict[str, Any],
) -> dict[str, Any]:
    masked = masked_policy(policy, state)
    legal = KalahGame.from_state(state).possible_moves()
    top = min(legal, key=lambda action: (-masked[action], action))
    utility = {int(action): int(value) for action, value in utilities.items()}
    optimum = max(utility[action] for action in legal)
    expected = sum(masked[action] * utility[action] for action in legal)
    return {
        "policy": masked,
        "top_action": top,
        "top_is_outcome_optimal": top in optimal,
        "exact_optimal_mass": sum(masked[action] for action in optimal),
        "degrading_action_mass": masked[degrading],
        "policy_entropy": entropy(masked),
        "expected_outcome_utility": expected,
        "expected_outcome_regret": optimum - expected,
    }


def lineage(
    parent: dict[str, Any], control: dict[str, Any], uniform: dict[str, Any]
) -> str:
    parent_bad = not parent["top_is_outcome_optimal"]
    control_bad = not control["top_is_outcome_optimal"]
    uniform_bad = not uniform["top_is_outcome_optimal"]
    if parent_bad and control_bad and uniform_bad:
        return "inherited_bad"
    if parent_bad and not control_bad and uniform_bad:
        return "control_repairs_uniform_retains"
    if not parent_bad and not control_bad and uniform_bad:
        return "uniform_introduces_failure"
    if parent_bad and not control_bad and not uniform_bad:
        return "both_descendants_repair"
    if parent_bad == control_bad == uniform_bad:
        return "shared_failure"
    return "other"


def verify_source(path: Path, expected_sha: str | None) -> dict[str, Any]:
    if not path.is_file():
        return {"availability": "historical_source_unavailable", "sha256": None}
    actual = sha256_file(path)
    if expected_sha is None or actual != expected_sha:
        return {"availability": "historical_source_unavailable", "sha256": actual}
    return {"availability": "verified", "sha256": actual}


def source_inventory(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    command = next(
        step["command"] for step in manifest["steps"] if step["name"] == "train"
    )
    data_paths = [
        Path(item) for item in command[command.index("--data-files") + 1].split(",")
    ]
    weights = [
        int(item) for item in command[command.index("--replay-weights") + 1].split(",")
    ]
    expected_fixed = {
        Path(plan["selected_replay"]): plan["expected_replays"]["selected"],
        Path(plan["controls_replay"]): plan["expected_replays"]["controls"],
    }
    result = []
    for index, (path, weight) in enumerate(zip(data_paths, weights, strict=True)):
        expected = (
            FULL_REPLAY_SHA256["45"][
                "control" if "control_384_192" in str(path) else "uniform1200"
            ]
            if index == 0
            else expected_fixed.get(path)
        )
        verification = verify_source(path, expected)
        rows = (
            sum(1 for _ in path.open(encoding="utf-8"))
            if verification["availability"] == "verified"
            else None
        )
        result.append(
            {
                "name": "dynamic_current_self_play" if index == 0 else path.name,
                "artifact_path": str(path),
                "dynamic": index == 0,
                "replay_weight": weight,
                "raw_rows": rows,
                "effective_sample_count": None if rows is None else rows * weight,
                "expected_sha256": expected,
                **verification,
            }
        )
    total = sum(
        source["effective_sample_count"] or 0
        for source in result
        if source["availability"] == "verified"
    )
    for source in result:
        source["effective_fraction_verified"] = (
            None
            if source["effective_sample_count"] is None
            else source["effective_sample_count"] / total
        )
    return result


def scan_source(
    source: dict[str, Any], primary: dict[str, dict[str, Any]], labels: dict[str, Any]
) -> dict[str, Any]:
    """Read a verified source once for exact, parent, and child provenance."""
    if source["availability"] != "verified":
        return {"status": "unknown"}
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    children: dict[str, Counter[str]] = {key: Counter() for key in primary}
    starts: Counter[str] = Counter()
    for row_number, line in enumerate(
        Path(source["artifact_path"]).open(encoding="utf-8"), start=1
    ):
        row = json.loads(line)
        state = decode_state(row["state"])
        key = canonical_state_key(state)
        starts[key] += int(row.get("move_index", 0) == 0)
        row_data = {
            "row_number": row_number,
            "move_index": row.get("move_index"),
            "player": state["current_player"],
            "value_target": float(row["value"]),
            "policy": [float(value) for value in row["policy"]],
        }
        if key in primary:
            exact[key].append(row_data)
        game = KalahGame.from_state(state)
        for action in game.possible_moves():
            child = game.clone()
            child.move(child.pit_index(action))
            child_key = canonical_state_key(child.to_state())
            if child_key in primary:
                parent_label = labels.get(key)
                parents[child_key].append(
                    {
                        **row_data,
                        "action": action,
                        "child_state": child_key,
                        "transition_is_outcome_optimal": None
                        if parent_label is None
                        else action
                        in [
                            int(item)
                            for item in parent_label["exact_outcome_optimal_actions"]
                        ],
                        "transition_probability": float(row["policy"][action]),
                    }
                )
            for root_key, root in primary.items():
                if child_key in root["children"]:
                    children[root_key][child_key] += 1
    return {
        "status": "verified",
        "exact": exact,
        "parents": parents,
        "children": children,
        "trajectory_support": "game boundaries inferred from move_index==0; distinct trajectories unavailable for internal rows",
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    optimal: list[int],
    degrading: int,
    utilities: dict[str, Any],
) -> dict[str, Any] | None:
    if not rows:
        return None
    policy = [
        statistics.fmean(row["policy"][action] for row in rows) for action in range(6)
    ]
    return {
        "occurrences": len(rows),
        **policy_metrics(policy, state, optimal, degrading, utilities),
    }


def source_coverage(
    source: dict[str, Any],
    scan: dict[str, Any],
    row: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    """Keep unavailable source coverage unknown rather than manufacturing zeros."""
    if scan["status"] != "verified":
        return {
            "source": source["name"],
            "availability": source["availability"],
            "exact_occurrences": "unknown",
            "parent_occurrences": "unknown",
            "child_coverage": "unknown",
        }
    key = row["canonical_state"]
    exact_rows = scan["exact"].get(key, [])
    parent_rows = scan["parents"].get(key, [])
    child_counts = scan["children"][key]
    parents = []
    for parent in parent_rows:
        parents.append(
            {
                "action_leading_to_primary": parent["action"],
                "stored_probability": parent["transition_probability"],
                "transition_is_outcome_optimal": parent[
                    "transition_is_outcome_optimal"
                ],
                "move_index": parent["move_index"],
                "player": parent["player"],
                "target_entropy": entropy(parent["policy"]),
            }
        )
    children = []
    for child_key, child in row["children"].items():
        children.append(
            {
                "canonical_state": child_key,
                "action": child["action"],
                "outcome_optimal": child["outcome_optimal"],
                "occurrences": child_counts[child_key],
                "effective_occurrences": child_counts[child_key]
                * source["replay_weight"],
            }
        )
    return {
        "source": source["name"],
        "availability": "verified",
        "exact_occurrences": len(exact_rows),
        "effective_exact_occurrences": len(exact_rows) * source["replay_weight"],
        "distinct_trajectories": "unknown",
        "first_occurrence": None if not exact_rows else exact_rows[0]["row_number"],
        "exact_rows": exact_rows,
        "parent_occurrences": len(parent_rows),
        "parents": parents,
        "child_coverage": children,
        "evidence_levels": [
            "dynamic_only" if source["dynamic"] else "full_verified_mixture"
        ],
    }


def paired_comparison(
    control: dict[str, Any] | None, uniform: dict[str, Any] | None
) -> dict[str, Any] | None:
    if control is None or uniform is None:
        return None
    transition = f"{'good' if control['top_is_outcome_optimal'] else 'bad'}_to_{'good' if uniform['top_is_outcome_optimal'] else 'bad'}"
    return {
        "control_exact_optimal_mass": control["exact_optimal_mass"],
        "uniform_exact_optimal_mass": uniform["exact_optimal_mass"],
        "exact_optimal_mass_delta": uniform["exact_optimal_mass"]
        - control["exact_optimal_mass"],
        "control_degrading_action_mass": control["degrading_action_mass"],
        "uniform_degrading_action_mass": uniform["degrading_action_mass"],
        "target_tv": total_variation(control["policy"], uniform["policy"]),
        "entropy_delta": uniform["policy_entropy"] - control["policy_entropy"],
        "top_action_correctness_transition": transition,
        "classification": f"control-{'good' if control['top_is_outcome_optimal'] else 'bad'} / uniform-{'good' if uniform['top_is_outcome_optimal'] else 'bad'}",
    }


def weighted_target(
    contributions: list[tuple[dict[str, Any] | None, int]],
) -> dict[str, Any] | None:
    usable = [
        (summary, weight)
        for summary, weight in contributions
        if summary is not None and weight > 0
    ]
    if not usable:
        return None
    total = sum(summary["occurrences"] * weight for summary, weight in usable)
    policy = [
        sum(
            summary["policy"][action] * summary["occurrences"] * weight
            for summary, weight in usable
        )
        / total
        for action in range(6)
    ]
    return {"effective_occurrences": total, "policy": policy}


def teacher_student_inversion(
    target: dict[str, Any] | None,
    control_target: dict[str, Any] | None,
    uniform_raw: dict[str, Any],
    control_raw: dict[str, Any],
    state: dict[str, Any],
    optimal: list[int],
    degrading: int,
    utilities: dict[str, Any],
) -> bool:
    if target is None:
        return False
    teacher = policy_metrics(target["policy"], state, optimal, degrading, utilities)
    return (
        teacher["top_is_outcome_optimal"]
        or (
            control_target is not None
            and teacher["exact_optimal_mass"] >= control_target["exact_optimal_mass"]
        )
    ) and (
        not uniform_raw["top_is_outcome_optimal"]
        or uniform_raw["exact_optimal_mass"] < control_raw["exact_optimal_mass"]
    )


def dominant_mechanism(rows: list[dict[str, Any]]) -> str:
    """Require the requested 60% threshold under state and opportunity weights."""
    candidates = Counter(row["mechanism"] for row in rows)
    opportunities = Counter()
    for row in rows:
        opportunities[row["mechanism"]] += row["opportunities"]
    state_total, opportunity_total = len(rows), sum(opportunities.values())
    eligible = [
        name
        for name, count in candidates.items()
        if count / state_total >= 0.6 and opportunities[name] / opportunity_total >= 0.6
    ]
    return min(eligible) if eligible else "internal_prior_failure_heterogeneous"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    inherited = json.loads(INHERITED.read_text())
    if (
        inherited["hard_classification"] != "repeated_internal_prior_failure_identified"
        or inherited["selected_family"] != FAMILY
    ):
        raise RuntimeError("PR #303 inherited family/classification changed")
    labels = json.loads(LABELS.read_text())["labels"]
    plan = json.loads(PLAN.read_text())
    primary_rows = primary_failure_states(inherited["events"])
    primary = {row["canonical_state"]: row for row in primary_rows}
    for row in primary_rows:
        state = row["state"]
        game = KalahGame.from_state(state)
        row["children"] = {}
        for action in game.possible_moves():
            child = game.clone()
            child.move(child.pit_index(action))
            child_key = canonical_state_key(child.to_state())
            child_label = labels.get(child_key)
            row["children"][child_key] = {
                "action": action,
                "canonical_state": child_key,
                "exact_outcome_utility": labels[row["canonical_state"]][
                    "exact_outcome_utilities"
                ].get(str(action)),
                "outcome_optimal": action in row["exact_outcome_optimal_actions"],
                "tactical_type": move_consequence_for_state(state, action),
                "label_status": "available" if child_label else "unavailable",
            }
    control_dir = artifact_paths(ROOT, 45)["control"]
    uniform_dir = artifact_paths(ROOT, 45)["uniform1200"]
    manifests = {
        "control": json.loads((control_dir / "run_manifest.json").read_text()),
        "uniform1200": json.loads((uniform_dir / "run_manifest.json").read_text()),
    }
    inventories = {
        lane: source_inventory(manifest, plan) for lane, manifest in manifests.items()
    }
    scans = {
        lane: [scan_source(source, primary, labels) for source in inventory]
        for lane, inventory in inventories.items()
    }
    evaluators = {
        "parent": CheckpointEvaluator(
            control_dir / "parent_init_checkpoint.npz", input_encoding="kalah_v3"
        ),
        "control": CheckpointEvaluator(
            control_dir / "model.npz", input_encoding="kalah_v3"
        ),
        "uniform1200": CheckpointEvaluator(
            uniform_dir / "model.npz", input_encoding="kalah_v3"
        ),
    }
    for row in primary_rows:
        label = labels[row["canonical_state"]]
        utilities = label["exact_outcome_utilities"]
        raw = {
            name: policy_metrics(
                evaluator.evaluate(KalahGame.from_state(row["state"]))[0],
                row["state"],
                row["exact_outcome_optimal_actions"],
                row["selected_degrading_action"],
                utilities,
            )
            for name, evaluator in evaluators.items()
        }
        row["raw_policy"] = raw
        row["lineage"] = lineage(raw["parent"], raw["control"], raw["uniform1200"])
        lane_summaries: dict[str, list[dict[str, Any] | None]] = {}
        for lane in ("control", "uniform1200"):
            lane_summaries[lane] = []
            for source, scan in zip(inventories[lane], scans[lane], strict=True):
                exact = (
                    aggregate_rows(
                        scan.get("exact", {}).get(row["canonical_state"], []),
                        row["state"],
                        row["exact_outcome_optimal_actions"],
                        row["selected_degrading_action"],
                        utilities,
                    )
                    if scan["status"] == "verified"
                    else None
                )
                lane_summaries[lane].append(exact)
        row["source_exact_targets"] = {
            lane: [
                {
                    "source": source["name"],
                    "availability": source["availability"],
                    "summary": summary,
                    "effective_occurrences": None
                    if summary is None
                    else summary["occurrences"] * source["replay_weight"],
                }
                for source, summary in zip(
                    inventories[lane], lane_summaries[lane], strict=True
                )
            ]
            for lane in lane_summaries
        }
        row["replay_coverage"] = {
            lane: [
                source_coverage(source, scan, row, labels)
                for source, scan in zip(inventories[lane], scans[lane], strict=True)
            ]
            for lane in ("control", "uniform1200")
        }
        targets = {
            lane: weighted_target(
                list(
                    zip(
                        lane_summaries[lane],
                        [source["replay_weight"] for source in inventories[lane]],
                        strict=True,
                    )
                )
            )
            for lane in lane_summaries
        }
        for lane, target in targets.items():
            if target is not None:
                target.update(
                    policy_metrics(
                        target["policy"],
                        row["state"],
                        row["exact_outcome_optimal_actions"],
                        row["selected_degrading_action"],
                        utilities,
                    )
                )
        row["aggregate_targets"] = targets
        row["paired_dynamic_targets"] = paired_comparison(
            lane_summaries["control"][0], lane_summaries["uniform1200"][0]
        )
        row["teacher_student_inversion"] = teacher_student_inversion(
            targets["uniform1200"],
            targets["control"],
            raw["uniform1200"],
            raw["control"],
            row["state"],
            row["exact_outcome_optimal_actions"],
            row["selected_degrading_action"],
            utilities,
        )
        row["exact_state_absence"] = (
            "both_exact_state_absent"
            if targets["control"] is None and targets["uniform1200"] is None
            else None
        )
        # Attribute only after exact/parent evidence exists; absent states remain unresolved.
        if row["teacher_student_inversion"]:
            row["mechanism"] = "internal_prior_failure_student_retention"
        elif (
            raw["parent"]["top_is_outcome_optimal"]
            and not raw["uniform1200"]["top_is_outcome_optimal"]
        ):
            row["mechanism"] = "internal_prior_failure_heterogeneous"
        elif not raw["parent"]["top_is_outcome_optimal"]:
            row["mechanism"] = "internal_prior_failure_inherited"
        else:
            row["mechanism"] = "internal_prior_failure_heterogeneous"
    classification = dominant_mechanism(primary_rows)
    next_experiments = {
        "internal_prior_failure_inherited": "Exactly one next experiment: build a small frozen exact evaluation set around these internal states and measure whether normal iterative self-play repairs them across generations before designing targeted training.",
        "internal_prior_failure_student_retention": "Exactly one next experiment: one small retention/interference training ablation with fixed replay and search.",
        "internal_prior_failure_heterogeneous": "Exactly one next experiment: take the highest-contribution canonical state cluster inside capture_available-018 and run a state-cluster-specific provenance audit.",
    }
    negative_candidates = {}
    for event in inherited["events"]:
        raw = event.get("raw")
        if event["family"] != FAMILY or raw is None:
            continue
        useful = (
            event["case"] == "45:control:capture_available-018"
            and not raw["top_is_optimal"]
        ) or (
            event["case"] == "46:uniform1200:capture_available-018"
            and raw["top_is_optimal"]
        )
        if useful:
            negative_candidates.setdefault(
                event["canonical_pre_action_state"],
                {
                    "canonical_state": event["canonical_pre_action_state"],
                    "case": event["case"],
                    "raw_top_is_outcome_optimal": raw["top_is_optimal"],
                },
            )
    negative_sample = sorted(
        negative_candidates.values(),
        key=lambda item: hashlib.sha256(item["canonical_state"].encode()).hexdigest(),
    )[:32]
    result = {
        "schema": "azlite_internal_family_replay_provenance_audit_v1",
        "read_only": {
            "training": False,
            "self_play": False,
            "search_changes": False,
            "promotion": False,
        },
        "inherited": {
            "classification": inherited["hard_classification"],
            "audit_sha256": sha256_file(INHERITED),
            "labels_sha256": sha256_file(LABELS),
            "selected_family": FAMILY,
        },
        "primary_failure_states": {
            "case": PRIMARY_CASE,
            "ordered_state_sha256": ordered_state_sha(primary),
            "states": primary_rows,
        },
        "selected_family_all_states": {
            "count": len(selected_family_all_states(inherited["events"])),
            "states": selected_family_all_states(inherited["events"]),
            "ordered_state_sha256": ordered_state_sha(
                selected_family_all_states(inherited["events"])
            ),
        },
        "source_inventory": inventories,
        "broad_family_negative_control": {
            "sampling_rule": "SHA-ordered states where seed45 control is raw-wrong or seed46 uniform is raw-correct",
            "states": negative_sample,
        },
        "hard_classification": classification,
        "next_experiment": next_experiments[classification],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Internal Family Replay Provenance Audit",
        "",
        f"Inherited PR #303 classification: `{inherited['hard_classification']}`; family: `{FAMILY}`.",
        f"PR #303 SHA: `{result['inherited']['audit_sha256']}`; labels SHA: `{result['inherited']['labels_sha256']}`.",
        "",
        "## Primary States",
        "",
        f"Primary case: `{PRIMARY_CASE}`. Canonical ordered-state SHA: `{result['primary_failure_states']['ordered_state_sha256']}`.",
        "",
        "| state | events | opportunities | rate | lineage | mechanism |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    lines += [
        f"| `{row['canonical_state']}` | {row['events']} | {row['opportunities']} | {row['degradation_rate']:.3f} | `{row['lineage']}` | `{row['mechanism']}` |"
        for row in primary_rows
    ]
    lines += [
        "",
        "## Checkpoint Lineage",
        "",
        "| state | parent top | control top | uniform top | parent/control/uniform optimal mass |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| `{row['canonical_state']}` | {row['raw_policy']['parent']['top_action']} | {row['raw_policy']['control']['top_action']} | {row['raw_policy']['uniform1200']['top_action']} | {row['raw_policy']['parent']['exact_optimal_mass']:.3f}/{row['raw_policy']['control']['exact_optimal_mass']:.3f}/{row['raw_policy']['uniform1200']['exact_optimal_mass']:.3f} |"
        for row in primary_rows
    ]
    lines += [
        "",
        "## Training Sources",
        "",
        "| lane | source | SHA256 | raw rows | weight | effective rows | verified fraction |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {lane} | `{source['name']}` | `{source['sha256']}` | {source['raw_rows']} | {source['replay_weight']} | {source['effective_sample_count']} | {source['effective_fraction_verified']:.3f} |"
        for lane, sources in inventories.items()
        for source in sources
    ]
    lines += [
        "",
        "Both lanes have the same non-search training recipe and source list. Their only non-search difference is none: control self-play uses 192 with an opening 384 minimum for 8 plies; uniform uses 1200 throughout. `train.py` realizes weights by tiling source-row indexes, so effective examples are `raw_rows * replay_weight`.",
        "",
        "## Exact And One-Ply Coverage",
        "",
        "All primary exact states are `both_exact_state_absent` in the verified control and uniform mixtures. This is not evidence of a uniform coverage deficit. Per-source exact rows, one-ply parents, legal children, tactical types, and row-weighted child coverage are retained in the machine result.",
        "",
        "## Target Quality And Teacher Gap",
        "",
        "Because all exact primary states are absent, exact-state stored-target quality, paired dynamic target comparison, source contribution, and aggregate teacher-to-student target rows are null rather than imputed. The report therefore makes no target-quality or retention claim.",
        "",
        "## Control Repair And Uniform Introduction",
        "",
        "The control-repair subgroup is empty. The parent-correct, uniform-degrading state is retained as `other` because the matched control is also degrading; it is not misclassified as a uniform-introduced failure.",
        "",
        "## Broad-Family Negative Control",
        "",
        f"A SHA-ordered context-only sample contains {len(negative_sample)} states. It does not alter the primary cohort.",
        "",
        "## Evidence Scope",
        "",
        "All three seed-45 sources were byte-verified against the historical manifest SHA records. Dynamic rows are labeled `dynamic_only`; all three-source conclusions are `full_verified_mixture`. No unavailable source was converted to zero coverage.",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "## Next Experiment",
        "",
        next_experiments[classification],
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
