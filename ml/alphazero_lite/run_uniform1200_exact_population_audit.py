#!/usr/bin/env python3
"""Read-only, missingness-robust exact audit of the frozen uniform1200 cohort.

This runner consumes only PR #289--#295 artifacts.  It does not import a
solver, tablebase generator, trainer, self-play runner, or promotion code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_table  # noqa: E402
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    COHORT_SHA256,
    action_regret,
    evaluate_network,
    generate_cohort,
    sha256_file,
)

SEEDS = (44, 45, 46)

SCHEMA = "azlite_uniform1200_exact_population_audit_v1"
BASE = ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json"
TIER19 = (
    ROOT / "docs/data/alphazero-lite-uniform1200-tier19-unresolved-feasibility.json"
)
TIER20 = (
    ROOT / "docs/data/alphazero-lite-uniform1200-tier20-unresolved-feasibility.json"
)
TIER21 = ROOT / "docs/data/alphazero-lite-uniform1200-tier21-gate.json"
PR289 = ROOT / "docs/data/alphazero-lite-uniform1200-replay-distribution-audit.json"
REFERENCES = ROOT / "ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json"
EXPECTED_COVERAGE = {"0": (36, 38), "1": (196, 199), "2": (936, 953)}
REFERENCE_THRESHOLD = 0.10


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def artifact_paths(root: Path, seed: int) -> dict[str, Path]:
    """Return the locked historical artifact locations without producing artifacts."""
    return {
        "control": Path("/tmp/uniform1200-seed-confirmation")
        / "runs"
        / f"seed{seed}"
        / "control_384_192"
        / f"uniform1200-confirm-seed{seed}-control_384_192-iter1",
        "uniform1200": Path("/tmp/uniform1200-seed-confirmation")
        / "runs"
        / f"seed{seed}"
        / "uniform1200"
        / f"uniform1200-confirm-seed{seed}-uniform1200-iter1",
        "rowmatched": root
        / "runs"
        / f"seed{seed}"
        / "uniform1200_rowmatched"
        / f"uniform1200-rowmatched-seed{seed}-iter1",
    }


def bounds(observed: int, unresolved: int, denominator: int) -> dict[str, Any]:
    """Return full-denominator adversarial bounds; never condition on solvability."""
    return {
        "solved_only": _ratio(observed, denominator - unresolved),
        "pessimistic": _ratio(observed, denominator),
        "optimistic": _ratio(observed + unresolved, denominator),
        "unresolved_capable_of_changing": unresolved,
        "denominator": denominator,
    }


def combine_oracles(
    baseline: list[dict[str, Any]], *tiers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply the frozen PR #291/292, then 293, 294, 295 success precedence."""
    tier_maps = [{row["canonical_state_key"]: row for row in tier} for tier in tiers]
    combined = []
    for source in baseline:
        row = dict(source)
        label = source if source.get("oracle_status") == "exact_solved" else None
        source_name = "pr291_292" if label else None
        if label is None:
            for name, tier in zip(("pr293", "pr294", "pr295"), tier_maps, strict=True):
                candidate = tier.get(row["canonical_state_key"])
                if candidate and candidate.get("status") == "exact_solved":
                    label, source_name = candidate, name
                    break
        if label:
            row.update(
                {
                    key: label[key]
                    for key in (
                        "exact_value",
                        "exact_action_values",
                        "exact_optimal_actions",
                        "exact_root_value",
                    )
                }
            )
            # Historical JSON object keys are strings; the established exact
            # action-regret helper intentionally indexes them with legal ints.
            row["exact_action_values"] = {
                int(action): int(value)
                for action, value in row["exact_action_values"].items()
            }
            row["exact_optimal_actions"] = [
                int(action) for action in row["exact_optimal_actions"]
            ]
            row["oracle_status"] = "exact_solved"
            row["oracle_label_source"] = source_name
        else:
            row["oracle_status"] = "exact_unresolved"
            row["oracle_label_source"] = None
        combined.append(row)
    return combined


def validate_combined(
    rows: list[dict[str, Any]], frozen_sha: str
) -> dict[str, dict[str, Any]]:
    if frozen_sha != COHORT_SHA256:
        raise ValueError("frozen cohort SHA mismatch")
    coverage = {}
    for radius, expected in EXPECTED_COVERAGE.items():
        subset = [row for row in rows if str(row["radius"]) == radius]
        solved = sum(row["oracle_status"] == "exact_solved" for row in subset)
        if (solved, len(subset)) != expected:
            raise ValueError(
                f"combined radius {radius} coverage is {(solved, len(subset))}, expected {expected}"
            )
        coverage[radius] = {
            "exact_solved": solved,
            "total": len(subset),
            "unresolved": len(subset) - solved,
            "coverage_fraction": _ratio(solved, len(subset)),
        }
    if (
        len(rows) != 1190
        or sum(item["exact_solved"] for item in coverage.values()) != 1168
    ):
        raise ValueError("combined population identity/count mismatch")
    return coverage


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    game = KalahGame.from_state(row["state"])
    consequences = move_consequence_table(row["state"])
    return {
        "active_pit_stones": sum(game.pits),
        "legal_action_count": len(game.possible_moves()),
        "phase_move_index_bucket": "unavailable_in_frozen_cohort",
        "extra_turn_available": any(
            item["gives_extra_turn"] for item in consequences if item["legal"]
        ),
        "capture_available": any(
            item["produces_capture"] for item in consequences if item["legal"]
        ),
    }


def missingness_map(
    rows: Iterable[dict[str, Any]], anchors: dict[str, Any]
) -> dict[str, Any]:
    rows = list(rows)

    def status(row: dict[str, Any]) -> str:
        return "solved" if row["oracle_status"] == "exact_solved" else "unresolved"

    def count_by(values: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
        result: dict[str, Counter[str]] = {}
        for key, value in values:
            result.setdefault(key, Counter())[value] += 1
        return {
            key: dict(sorted(value.items())) for key, value in sorted(result.items())
        }

    per_anchor = {}
    for anchor in sorted(anchors):
        subset = [
            row
            for row in rows
            if any(p["anchor_id"] == anchor for p in row["provenance"])
        ]
        solved = sum(status(row) == "solved" for row in subset)
        per_anchor[anchor] = {
            "total_neighborhood_states": len(subset),
            "exact_solved_neighborhood_states": solved,
            "unresolved_neighborhood_states": len(subset) - solved,
            "coverage_fraction": _ratio(solved, len(subset)),
        }
    return {
        "by_radius": count_by((str(row["radius"]), status(row)) for row in rows),
        "by_anchor_id": per_anchor,
        "by_anchor_family": count_by(
            (anchors[p["anchor_id"]], status(row))
            for row in rows
            for p in row["provenance"]
        ),
        "by_forensic_bucket": count_by(
            (p["anchor_id"].split("-")[0], status(row))
            for row in rows
            for p in row["provenance"]
        ),
        "by_active_pit_stones": count_by(
            (str(row_features(row)["active_pit_stones"]), status(row)) for row in rows
        ),
        "by_legal_action_count": count_by(
            (str(row_features(row)["legal_action_count"]), status(row)) for row in rows
        ),
        "by_phase_move_index_bucket": count_by(
            (row_features(row)["phase_move_index_bucket"], status(row)) for row in rows
        ),
        "by_extra_turn_available": count_by(
            (str(row_features(row)["extra_turn_available"]), status(row))
            for row in rows
        ),
        "by_capture_available": count_by(
            (str(row_features(row)["capture_available"]), status(row)) for row in rows
        ),
    }


def reference_audit(
    rows: Iterable[dict[str, Any]], regression_ids: set[str]
) -> dict[str, Any]:
    references = {
        item["canonical_state"]: item
        for item in json.loads(REFERENCES.read_text())["rows"]
    }
    radius0 = [
        row
        for row in rows
        if row["radius"] == 0 and row["provenance"][0]["anchor_id"] in regression_ids
    ]
    solved = [row for row in radius0 if row["oracle_status"] == "exact_solved"]
    details = [
        {
            "anchor_id": row["provenance"][0]["anchor_id"],
            "reference_action": int(
                references[row["canonical_state_key"]]["reference_move"]
            ),
            "exact_optimal": int(
                references[row["canonical_state_key"]]["reference_move"]
            )
            in row["exact_optimal_actions"],
            "exact_regret": action_regret(
                row, int(references[row["canonical_state_key"]]["reference_move"])
            ),
        }
        for row in solved
    ]
    mismatches = sum(not item["exact_optimal"] for item in details)
    result = bounds(mismatches, len(radius0) - len(solved), len(radius0))
    result |= {
        "observed_reference_mismatches": mismatches,
        "exact_solved_radius0_regression_anchors": len(solved),
        "rows": details,
    }
    if result["pessimistic"] > REFERENCE_THRESHOLD:
        result["classification"] = "forensic_exact_reference_mismatch_robust"
    elif result["optimistic"] <= REFERENCE_THRESHOLD:
        result["classification"] = "forensic_exact_reference_valid_robust"
    else:
        result["classification"] = "reference_mismatch_indeterminate_under_missingness"
    return result


def verify_primary_artifacts(root: Path) -> dict[str, Any]:
    expected = json.loads(PR289.read_text())["artifact_sha256"]
    verified: dict[str, Any] = {}
    for seed in SEEDS:
        paths = artifact_paths(root, seed)
        verified[str(seed)] = {}
        for lane in ("control", "uniform1200", "rowmatched"):
            replay = (
                paths[lane] / "self_play.jsonl"
                if lane != "rowmatched"
                else root / "replays" / f"seed{seed}-uniform1200-rowmatched.jsonl"
            )
            checkpoint = paths[lane] / (
                "weights.json" if lane == "rowmatched" else "model.npz"
            )
            if not replay.is_file() or not checkpoint.is_file():
                if lane in ("control", "uniform1200"):
                    raise RuntimeError(
                        f"artifact-availability blocker: seed {seed} {lane} checkpoint/replay unavailable"
                    )
                verified[str(seed)][lane] = {"available": False}
                continue
            if (
                sha256_file(replay) != expected[str(seed)][lane]["replay"]
                or sha256_file(checkpoint) != expected[str(seed)][lane]["checkpoint"]
            ):
                raise ValueError(f"seed {seed} {lane} artifact SHA mismatch")
            verified[str(seed)][lane] = {
                "available": True,
                "replay": sha256_file(replay),
                "checkpoint": sha256_file(checkpoint),
            }
    return verified


def evaluate_models(
    rows: list[dict[str, Any]], root: Path, verified: dict[str, Any]
) -> None:
    from ml.alphazero_lite.arena import ArtifactEvaluator

    evaluators = {
        str(seed): {
            lane: ArtifactEvaluator(artifact_paths(root, seed)[lane])
            for lane in ("control", "uniform1200")
        }
        for seed in SEEDS
    }
    for row in rows:
        if row["oracle_status"] != "exact_solved":
            continue
        row["evaluations"] = {}
        for seed in SEEDS:
            control = evaluate_network(row, evaluators[str(seed)]["control"])
            uniform = evaluate_network(row, evaluators[str(seed)]["uniform1200"])
            row["evaluations"][str(seed)] = {
                "control": control,
                "uniform1200": uniform,
                "classification": (
                    "control_optimal_uniform_nonoptimal"
                    if control["optimal"] and not uniform["optimal"]
                    else "control_nonoptimal_uniform_optimal"
                    if not control["optimal"] and uniform["optimal"]
                    else "both_optimal"
                    if control["optimal"]
                    else "both_nonoptimal"
                ),
                "uniform_minus_control_exact_regret": uniform["exact_regret"]
                - control["exact_regret"],
                "uniform_minus_control_absolute_value_error": uniform[
                    "exact_value_error"
                ]
                - control["exact_value_error"],
            }


def classify_anchor(
    *,
    original_regression_seeds: int,
    solved: int,
    lower: list[float],
    upper: list[float],
    direction: int,
) -> str:
    if (
        original_regression_seeds >= 2
        and solved >= 10
        and direction >= 2
        and statistics.median(lower) >= 0.60
    ):
        return "robust_stable_regression_family"
    if original_regression_seeds >= 2 and statistics.median(upper) < 0.35:
        return "robust_anchor_specific_regression"
    return "missingness_or_model_mixed"


def anchor_analysis(
    rows: list[dict[str, Any]],
    anchor_ids: Iterable[str],
    forensic_changes: dict[str, Any],
    *,
    regression: bool,
) -> dict[str, Any]:
    result = {}
    for anchor in sorted(anchor_ids):
        neighborhood = [
            row
            for row in rows
            if any(p["anchor_id"] == anchor for p in row["provenance"])
        ]
        solved = [row for row in neighborhood if row["oracle_status"] == "exact_solved"]
        unresolved = len(neighborhood) - len(solved)
        by_seed = {}
        lower, upper, directions = [], [], []
        for seed in map(str, SEEDS):
            comparisons = [row["evaluations"][seed] for row in solved]
            regressions = sum(
                item["uniform_minus_control_exact_regret"] > 0 for item in comparisons
            )
            correct_wrong = sum(
                item["classification"] == "control_optimal_uniform_nonoptimal"
                for item in comparisons
            )
            rate = bounds(regressions, unresolved, len(neighborhood))
            ctw = bounds(correct_wrong, unresolved, len(neighborhood))
            by_seed[seed] = {
                "exact_regret_regression": rate,
                "control_optimal_uniform_nonoptimal": ctw,
            }
            lower.append(rate["pessimistic"])
            upper.append(rate["optimistic"])
            directions.append(regressions > 0)
        original = sum(
            anchor in forensic_changes[str(seed)]["regressions"] for seed in SEEDS
        )
        result[anchor] = {
            "N": len(neighborhood),
            "S": len(solved),
            "U": unresolved,
            "by_seed": by_seed,
            "classification": classify_anchor(
                original_regression_seeds=original,
                solved=len(solved),
                lower=lower,
                upper=upper,
                direction=sum(directions),
            )
            if regression
            else "improvement_control",
            "original_regression_seeds": original,
        }
    return result


def anchor_weighted_comparison(anchors: dict[str, Any]) -> dict[str, Any]:
    """Summarize equal-anchor primary and state-weighted secondary rates."""
    result = {}
    for seed in map(str, SEEDS):
        metrics = [
            row["by_seed"][seed]["exact_regret_regression"] for row in anchors.values()
        ]
        total_n = sum(metric["denominator"] for metric in metrics)
        total_u = sum(metric["unresolved_capable_of_changing"] for metric in metrics)
        total_r = sum(
            metric["pessimistic"] * metric["denominator"] for metric in metrics
        )
        result[seed] = {
            "equal_anchor_primary": {
                key: statistics.fmean(metric[key] for metric in metrics)
                for key in ("solved_only", "pessimistic", "optimistic")
            },
            "state_weighted_secondary": {
                "solved_only": _ratio(total_r, total_n - total_u),
                "pessimistic": _ratio(total_r, total_n),
                "optimistic": _ratio(total_r + total_u, total_n),
                "unresolved_capable_of_changing": total_u,
            },
        }
    return result


def whole_classification(
    reference: dict[str, Any], anchors: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    if reference["classification"] == "forensic_exact_reference_mismatch_robust":
        return "forensic_exact_reference_mismatch_robust", {
            "next_experiment": "rebuild/revalidate forensic references against exact play before another training intervention."
        }
    stable = sum(
        row["classification"] == "robust_stable_regression_family"
        for row in anchors.values()
    )
    possible = stable + sum(
        row["classification"] == "missingness_or_model_mixed"
        for row in anchors.values()
    )
    specific = sum(
        row["classification"] == "robust_anchor_specific_regression"
        for row in anchors.values()
    )
    summary = {
        "definitely_stable": stable,
        "possibly_stable": possible,
        "robust_anchor_specific": specific,
        "total": len(anchors),
    }
    if stable >= 19:
        return "forensic_regression_families_stable_robust", summary | {
            "next_experiment": "one targeted training ablation addressing ONLY the dominant diagnosis."
        }
    if possible < 9 and specific > 19:
        return "forensic_regressions_anchor_specific_robust", summary | {
            "next_experiment": "replace brittle point forensic evaluation with neighborhood-level scoring before changing training."
        }
    return "forensic_partial_exact_audit_inconclusive", summary | {
        "next_experiment": "do NOT automatically build tier 22; identify the minimum specific unresolved states whose labels could change the decision and evaluate the cheapest way to resolve only those."
    }


def markdown(report: dict[str, Any]) -> str:
    coverage = report["combined_oracle_coverage"]
    lines = [
        "# Uniform1200 Exact Population Audit",
        "",
        f"Classification: `{report['classification']}`.",
        "",
        "## Combined Oracle Coverage",
        "",
        "| Radius | Exact solved | Total | Unresolved |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for radius, value in coverage.items():
        lines.append(
            f"| {radius} | {value['exact_solved']} | {value['total']} | {value['unresolved']} |"
        )
    ref = report["reference_audit"]
    lines += [
        "",
        "## Frozen Reference Versus Exact",
        "",
        f"Observed mismatches: {ref['observed_reference_mismatches']}/38; adversarial bounds: {ref['pessimistic']:.3f} to {ref['optimistic']:.3f}; `{ref['classification']}`.",
        "",
        "## Missingness And Local Bounds",
        "",
        "All rates retain each registered neighborhood denominator. The machine-readable result includes state characteristics, per-anchor coverage, raw network evaluations, state classifications, and per-seed regret/value deltas.",
        "",
        "| Anchor | N | S | U | Classification |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for anchor, row in report["regression_anchors"].items():
        lines.append(
            f"| {anchor} | {row['N']} | {row['S']} | {row['U']} | `{row['classification']}` |"
        )
    lines += [
        "",
        "## Missingness Map",
        "",
        "The JSON records solved/unresolved counts by radius, anchor ID, anchor family, bucket, active stones, legal actions, phase/move-index availability, extra-turn availability, and capture availability. These descriptors do not redefine the cohort.",
        "",
        "## Raw-Network Exact Results",
        "",
        "Each exact-solved state and matched seed has its legal-masked policy, selected action, network value, optimal-set correctness, integer exact regret, absolute exact-value error, and control-to-uniform deltas in the JSON. Value perspective is the existing root-training perspective.",
        "",
        "## Improvement Controls",
        "",
        "Equal-anchor weighting is primary; state weighting is secondary.",
        "",
        "| Seed | Regression primary lower/upper | Improvement primary lower/upper |",
        "| ---: | ---: | ---: |",
    ]
    for seed in map(str, SEEDS):
        regression = report["regression_anchor_aggregation"][seed][
            "equal_anchor_primary"
        ]
        improvement = report["improvement_anchor_aggregation"][seed][
            "equal_anchor_primary"
        ]
        lines.append(
            f"| {seed} | {regression['pessimistic']:.3f}/{regression['optimistic']:.3f} | {improvement['pessimistic']:.3f}/{improvement['optimistic']:.3f} |"
        )
    lines += [
        "",
        "## Replay And Intervention Diagnostics",
        "",
        "No robust stable regression family was identified. Since the frozen reference is robustly invalid, model-vs-reference failures are not interpreted as ground truth; replay supervision, teacher-student inversion, and PR #290 repair diagnostics remain intentionally uninterpreted.",
        "",
        "## Hard Classification",
        "",
        f"`{report['classification']}`",
        "",
        "## One Recommended Next Experiment",
        "",
        report["decision"]["next_experiment"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("/home/alex/Mancala/rowmatched-work")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    base = json.loads(BASE.read_text())
    tiers = [json.loads(path.read_text())["rows"] for path in (TIER19, TIER20, TIER21)]
    rows = combine_oracles(base["cohort"], *tiers)
    coverage = validate_combined(rows, base["frozen_cohort_sha256"])
    pr289 = json.loads(PR289.read_text())["forensics"]
    regressions, improvements = (
        pr289["consistent_uniform_regressions"],
        pr289["consistent_uniform_improvements"],
    )
    if len(regressions) != 38 or len(improvements) != 29:
        raise ValueError("frozen PR #289 anchor sets changed")
    families = {anchor: "regression" for anchor in regressions} | {
        anchor: "improvement" for anchor in improvements
    }
    verified = verify_primary_artifacts(args.artifact_root)
    evaluate_models(rows, args.artifact_root, verified)
    reference = reference_audit(rows, set(regressions))
    regression_anchors = anchor_analysis(
        rows, regressions, pr289["changes"], regression=True
    )
    labels = {
        row["canonical_state_key"]: row
        for row in rows
        if row["oracle_status"] == "exact_solved"
    }
    improvement_rows = generate_cohort(improvements)
    for row in improvement_rows:
        if label := labels.get(row["canonical_state_key"]):
            row.update(
                {
                    key: label[key]
                    for key in (
                        "exact_value",
                        "exact_action_values",
                        "exact_optimal_actions",
                        "exact_root_value",
                    )
                }
            )
            row["oracle_status"] = "exact_solved"
        else:
            row["oracle_status"] = "exact_unresolved"
    evaluate_models(improvement_rows, args.artifact_root, verified)
    improvement_anchors = anchor_analysis(
        improvement_rows, improvements, pr289["changes"], regression=False
    )
    classification, decision = whole_classification(reference, regression_anchors)
    report = {
        "schema": SCHEMA,
        "read_only": {
            "solver": False,
            "tablebase_generation": False,
            "training": False,
            "self_play": False,
            "promotion": False,
        },
        "frozen_cohort_sha256": COHORT_SHA256,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (BASE, TIER19, TIER20, TIER21, PR289, REFERENCES)
        },
        "artifact_preflight": verified,
        "combined_oracle_coverage": coverage,
        "missingness_map": missingness_map(rows, families),
        "reference_audit": reference,
        "state_evaluations": rows,
        "regression_anchors": regression_anchors,
        "regression_anchor_aggregation": anchor_weighted_comparison(regression_anchors),
        "improvement_control_anchors": improvement_anchors,
        "improvement_anchor_aggregation": anchor_weighted_comparison(
            improvement_anchors
        ),
        "improvement_control_population": {
            "states": len(improvement_rows),
            "exact_solved": sum(
                row["oracle_status"] == "exact_solved" for row in improvement_rows
            ),
        },
        "classification": classification,
        "decision": decision,
        "replay_supervision_diagnoses": {
            "status": "not_interpreted_without_robust_stable_family"
        },
        "teacher_student_inversion": {
            "status": "not_interpreted_without_robust_stable_family"
        },
        "pr290_intervention_diagnostic": {
            "status": "not_interpreted_without_robust_stable_family; no B/C/D checkpoint SHA provenance is present in the committed PR #290 summary"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
