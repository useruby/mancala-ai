#!/usr/bin/env python3
"""Run a deterministic robustness preflight for PR #123 weighted candidates.

Does not train, generate self-play, promote, or overwrite model-artifact/current.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(REPO_ROOT))

from ml.alphazero_lite.build_opening_suite import (  # noqa: E402
    deduplicate_openings,
    enumerate_legal_prefixes,
    load_suite_jsonl,
    select_diverse,
    stratify_openings,
    write_suite_jsonl,
)
from ml.alphazero_lite.run_control_ep2_puct_head_preflight import (  # noqa: E402
    BENCHMARK_LABEL_TO_BUDGET,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_GATE_BUDGET_PAIRS,
    benchmark_candidate_name,
    benchmark_budget_results,
    build_input_summary,
    candidate_gate_preserved,
    find_candidate_report,
    gate_budget_results,
    load_json,
    pooled_per_opening_differences,
    prefix_key,
    require_existing_file,
    run_default_gate,
    run_opening_suite_benchmark,
    sha256_file,
    suite_distribution,
    suite_prefix_keys,
    verify_expected_hash,
    write_json,
)

DEFAULT_BUDGET_PAIRS = "384:256,768:256,768:768,1200:1200,1200:256,256:768"
REPORT_PATH = (
    REPO_ROOT / "docs/alphazero-lite-pr123-weighted-candidate-preflight-results.md"
)
SOURCE_DOC = "docs/alphazero-lite-promoted-current-opening-puct-disagreement-weight-ablation-results.md"
PR123_SCHEMA = "azlite_pr123_weighted_candidate_preflight_v1"
PRIMARY_BUDGET = "384:256"
EQ_768_BUDGET = "768:768"
HIGH_EQ_BUDGET = "1200:1200"
HIGH_ASYM_BUDGET = "1200:256"

REQUIRED_CANDIDATES = {
    "promoted_current_ref": {
        "expected_checkpoint_sha256": None,
        "expected_artifact_weights_sha256": "6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece",
    },
    "disagreement_w8_policy_head_e1": {
        "expected_checkpoint_sha256": "cd9ef83902516283d680fec0d3986cea832bf467042aaabd814ecd5026ec1e0e",
        "expected_artifact_weights_sha256": "2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad",
    },
    "disagreement_w4_policy_head_e2": {
        "expected_checkpoint_sha256": "04ec8f9420459359b21b7f2d3b5ab5c6f42bc054711c3e13d9538dc8456ab616",
        "expected_artifact_weights_sha256": "65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177",
    },
    "disagreement_w16_policy_head_e2": {
        "expected_checkpoint_sha256": "c4b74cf357e67798b24b3ed49a593034b6c30ea9619ef5fd0bc3d36a7bcfac9f",
        "expected_artifact_weights_sha256": "51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be",
    },
}

OPTIONAL_CANDIDATES = {
    "disagreement_w16_policy_head_e1": {
        "expected_checkpoint_sha256": "dc0843cd6f0fc4278e9a2ae04c79e50847ac9e374d0af00ae0e69154efc49865",
        "expected_artifact_weights_sha256": "ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27",
    }
}


def bootstrap_ci(
    diffs: list[float], *, seed: int, samples: int = DEFAULT_BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    if not diffs:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": samples, "n": 0}
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(samples, len(arr)))
    bootstrap_means = arr[indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.percentile(bootstrap_means, 2.5)),
        "upper": float(np.percentile(bootstrap_means, 97.5)),
        "samples": samples,
        "n": int(len(arr)),
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def parse_csv_paths(text: str) -> list[Path]:
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def candidate_dir_hints(workdir: Path, name: str) -> list[Path]:
    lane_dir = workdir / name
    return [
        Path(name),
        lane_dir,
        lane_dir / f"artifact_{name}",
        lane_dir / "artifact",
        workdir / f"artifact_{name}",
    ]


def checkpoint_hints(workdir: Path, name: str) -> list[Path]:
    lane_dir = workdir / name
    return [
        lane_dir / "checkpoint_epoch2.npz",
        lane_dir / "checkpoint_epoch1.npz",
        lane_dir / "checkpoint.npz",
    ]


def discover_candidate_entry(
    *,
    name: str,
    workdir: Path,
    pr123_summary: dict[str, Any],
    current_artifact: Path,
    current_weights_sha: str,
    expected_checkpoint_sha256: str | None,
    expected_artifact_weights_sha256: str,
) -> dict[str, Any] | None:
    if name == "promoted_current_ref":
        require_existing_file(
            current_artifact / "weights.json", "current artifact weights"
        )
        verify_expected_hash(
            current_artifact / "weights.json",
            expected_artifact_weights_sha256,
            f"{name} artifact weights",
        )
        return {
            "name": name,
            "report_name": "current",
            "artifact_dir": str(current_artifact),
            "artifact_weights_sha256": current_weights_sha,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "parent_promoted_current_hash": current_weights_sha,
            "source_experiment_doc": SOURCE_DOC,
        }

    summary_candidates = {
        str(candidate.get("candidate")): candidate
        for candidate in pr123_summary.get("candidates", [])
        if isinstance(candidate, dict)
    }
    summary_entry = summary_candidates.get(name, {})

    artifact_candidates: list[Path] = []
    summary_artifact = summary_entry.get("artifact_dir")
    if isinstance(summary_artifact, str) and summary_artifact:
        artifact_candidates.append(Path(summary_artifact))
    artifact_candidates.extend(candidate_dir_hints(workdir, name))

    artifact_dir: Path | None = None
    for candidate_dir in artifact_candidates:
        weights_path = candidate_dir / "weights.json"
        if not weights_path.is_file():
            continue
        if sha256_file(weights_path) == expected_artifact_weights_sha256:
            artifact_dir = candidate_dir
            break
    if artifact_dir is None:
        return None

    checkpoint_candidates: list[Path] = []
    summary_checkpoint = summary_entry.get("checkpoint_path")
    if isinstance(summary_checkpoint, str) and summary_checkpoint:
        checkpoint_candidates.append(Path(summary_checkpoint))
    checkpoint_candidates.extend(checkpoint_hints(workdir, name))

    checkpoint_path: Path | None = None
    checkpoint_sha256: str | None = None
    for candidate_checkpoint in checkpoint_candidates:
        if not candidate_checkpoint.is_file():
            continue
        sha_value = sha256_file(candidate_checkpoint)
        if (
            expected_checkpoint_sha256 is None
            or sha_value == expected_checkpoint_sha256
        ):
            checkpoint_path = candidate_checkpoint
            checkpoint_sha256 = sha_value
            break
    if expected_checkpoint_sha256 is not None and checkpoint_path is None:
        raise RuntimeError(
            f"missing expected checkpoint for {name}: {expected_checkpoint_sha256}"
        )

    verify_expected_hash(
        artifact_dir / "weights.json",
        expected_artifact_weights_sha256,
        f"{name} artifact weights",
    )
    if checkpoint_path is not None:
        verify_expected_hash(
            checkpoint_path,
            expected_checkpoint_sha256,
            f"{name} checkpoint",
        )

    return {
        "name": name,
        "report_name": benchmark_candidate_name(artifact_dir),
        "artifact_dir": str(artifact_dir),
        "artifact_weights_sha256": expected_artifact_weights_sha256,
        "checkpoint_path": str(checkpoint_path)
        if checkpoint_path is not None
        else None,
        "checkpoint_sha256": checkpoint_sha256,
        "parent_promoted_current_hash": current_weights_sha,
        "source_experiment_doc": SOURCE_DOC,
    }


def build_extra_heldout_suite(
    *,
    fixed_suite_path: Path,
    exclude_suite_paths: list[Path],
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    fixed_entries = load_suite_jsonl(str(fixed_suite_path))
    target_size = len(fixed_entries)
    target_ply_counts = Counter(int(entry["ply"]) for entry in fixed_entries)
    excluded_prefixes: set[str] = set()
    for suite_path in [fixed_suite_path, *exclude_suite_paths]:
        excluded_prefixes |= suite_prefix_keys(load_suite_jsonl(str(suite_path)))

    all_prefixes: list[dict[str, Any]] = []
    for max_ply in (2, 4, 6):
        all_prefixes.extend(enumerate_legal_prefixes(max_ply))
    unique, _duplicates, _duplicate_count = deduplicate_openings(all_prefixes)
    unique = stratify_openings(unique)

    filtered: list[dict[str, Any]] = []
    for entry in unique:
        entry_prefixes = {prefix_key(entry["prefix_moves"])}
        for alternate in entry.get("alternate_prefixes", []) or []:
            entry_prefixes.add(prefix_key(alternate))
        if entry_prefixes.isdisjoint(excluded_prefixes):
            filtered.append(entry)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for ply, target_count in sorted(target_ply_counts.items()):
        ply_pool = [entry for entry in filtered if int(entry["ply"]) == ply]
        quota = min(int(target_count), len(ply_pool))
        picked = select_diverse(ply_pool, quota, seed + (ply * 1000))
        for entry in picked:
            key = prefix_key(entry["prefix_moves"])
            if key in selected_keys:
                continue
            selected.append(entry)
            selected_keys.add(key)

    if len(selected) < target_size:
        deficit = target_size - len(selected)
        remainder = [
            entry
            for entry in filtered
            if prefix_key(entry["prefix_moves"]) not in selected_keys
        ]
        selected.extend(select_diverse(remainder, deficit, seed + 99991))

    if len(selected) != target_size:
        raise RuntimeError(
            f"extra heldout suite size mismatch for seed {seed}: expected {target_size}, got {len(selected)}"
        )

    rng = np.random.default_rng(seed)
    order = list(range(len(selected)))
    rng.shuffle(order)
    selected = [selected[index] for index in order]
    write_suite_jsonl(selected, str(out_path))

    heldout_entries = load_suite_jsonl(str(out_path))
    overlap = suite_prefix_keys(heldout_entries) & excluded_prefixes
    if overlap:
        first = sorted(overlap)[0]
        raise RuntimeError(
            f"extra heldout suite overlaps excluded prefixes; first={first}"
        )
    optional_keys = {"alternate_prefixes"}
    schema_matches = (
        set(heldout_entries[0].keys()) - optional_keys
        == set(fixed_entries[0].keys()) - optional_keys
    )
    return {
        "path": str(out_path),
        "seed": seed,
        "sha256": sha256_file(out_path),
        "rows": len(heldout_entries),
        "schema_matches_fixed_large": schema_matches,
        "fixed_distribution": suite_distribution(fixed_entries),
        "heldout_distribution": suite_distribution(heldout_entries),
        "overlap_with_excluded_prefixes": 0,
        "generation": "deterministic_legal_prefix_resample_no_overlap",
    }


def suite_result_rows(
    *, report: dict[str, Any], candidates: list[dict[str, Any]], suite_name: str
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "suite_name": suite_name,
        "suite_path": report.get("suite_path"),
        "suite_sha256": report.get("suite_sha256"),
        "suite_size": report.get("suite_size"),
        "current_sha256": report.get("current_sha256"),
        "candidates": {},
    }
    for candidate in candidates:
        candidate_report = find_candidate_report(report, candidate["report_name"])
        if candidate_report is None:
            raise RuntimeError(
                f"missing candidate {candidate['report_name']} in suite {suite_name}"
            )
        rows["candidates"][candidate["name"]] = {
            "candidate_path": candidate["artifact_dir"],
            "candidate_sha256": candidate_report.get("candidate_sha256"),
            "current_sha256": candidate_report.get("current_sha256"),
            "budget_results": benchmark_budget_results(candidate_report),
        }
    return rows


def benchmark_report_complete(
    *, report: dict[str, Any], candidates: list[dict[str, Any]], budget_pairs: list[str]
) -> bool:
    for candidate in candidates:
        candidate_report = find_candidate_report(report, candidate["report_name"])
        if candidate_report is None:
            return False
        budget_results = benchmark_budget_results(candidate_report)
        if any(budget_pair not in budget_results for budget_pair in budget_pairs):
            return False
    return True


def gate_report_complete(report: dict[str, Any], budget_pairs: list[str]) -> bool:
    if report.get("classification") is None:
        return False
    report_budget_results = gate_budget_results(report)
    return all(budget_pair in report_budget_results for budget_pair in budget_pairs)


def annotate_suite_deltas(
    *, suite_rows: dict[str, Any], reference_name: str, budget_pairs: list[str]
) -> None:
    for suite_row in suite_rows.values():
        deltas: dict[str, dict[str, float]] = {}
        for candidate_name, candidate_row in suite_row["candidates"].items():
            budget_deltas: dict[str, float] = {}
            for budget_pair in budget_pairs:
                candidate_ds = float(candidate_row["budget_results"][budget_pair]["ds"])
                reference_ds = float(
                    suite_row["candidates"][reference_name]["budget_results"][
                        budget_pair
                    ]["ds"]
                )
                budget_deltas[budget_pair] = candidate_ds - reference_ds
            deltas[candidate_name] = budget_deltas
        suite_row["delta_vs_promoted_current_ref"] = deltas


def aggregate_candidate_metrics(
    *,
    suite_rows: dict[str, Any],
    reference_name: str,
    candidate_name: str,
    budget_pairs: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for budget_pair in budget_pairs:
        suite_ds: dict[str, float] = {}
        suite_delta: dict[str, float] = {}
        for suite_name, suite_row in suite_rows.items():
            ds = float(
                suite_row["candidates"][candidate_name]["budget_results"][budget_pair][
                    "ds"
                ]
            )
            ref = float(
                suite_row["candidates"][reference_name]["budget_results"][budget_pair][
                    "ds"
                ]
            )
            suite_ds[suite_name] = ds
            suite_delta[suite_name] = ds - ref
        ds_values = list(suite_ds.values())
        delta_values = list(suite_delta.values())
        result[budget_pair] = {
            "suite_ds": suite_ds,
            "suite_delta_vs_promoted_current_ref": suite_delta,
            "mean_ds": statistics.fmean(ds_values),
            "worst_suite_ds": min(ds_values),
            "stddev_ds": statistics.stdev(ds_values) if len(ds_values) > 1 else 0.0,
            "mean_delta_vs_promoted_current_ref": statistics.fmean(delta_values),
            "worst_suite_delta_vs_promoted_current_ref": min(delta_values),
            "stddev_delta_vs_promoted_current_ref": statistics.stdev(delta_values)
            if len(delta_values) > 1
            else 0.0,
        }
    return result


def opening_outcome_counts(
    *, suite_rows: dict[str, Any], budget_pair: str, candidate_a: str, candidate_b: str
) -> dict[str, int]:
    diffs = pooled_per_opening_differences(
        suite_rows=suite_rows,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        budget_pair=budget_pair,
        metric_key="ds",
    )
    counts = {candidate_a: 0, candidate_b: 0, "tie": 0}
    for diff in diffs:
        if diff > 1e-12:
            counts[candidate_a] += 1
        elif diff < -1e-12:
            counts[candidate_b] += 1
        else:
            counts["tie"] += 1
    return counts


def ci_key(candidate_name: str, budget_pair: str) -> str:
    return (
        f"{candidate_name}_minus_promoted_current_ref_{budget_pair.replace(':', '_')}"
    )


def build_bootstrap_cis(
    *,
    suite_rows: dict[str, Any],
    reference_name: str,
    candidate_names: list[str],
    include_w8_w4: bool,
    include_w8_w16: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    cis: dict[str, Any] = {}
    for candidate_name in candidate_names:
        for budget_pair in (
            PRIMARY_BUDGET,
            EQ_768_BUDGET,
            HIGH_EQ_BUDGET,
            HIGH_ASYM_BUDGET,
        ):
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a=candidate_name,
                candidate_b=reference_name,
                budget_pair=budget_pair,
                metric_key="ds",
            )
            cis[ci_key(candidate_name, budget_pair)] = bootstrap_ci(
                diffs, seed=seed + len(cis), samples=bootstrap_samples
            )
    if include_w8_w4:
        for budget_pair in (PRIMARY_BUDGET, EQ_768_BUDGET):
            diffs = pooled_per_opening_differences(
                suite_rows=suite_rows,
                candidate_a="disagreement_w8_policy_head_e1",
                candidate_b="disagreement_w4_policy_head_e2",
                budget_pair=budget_pair,
                metric_key="ds",
            )
            cis[f"w8_e1_minus_w4_e2_{budget_pair.replace(':', '_')}"] = bootstrap_ci(
                diffs, seed=seed + len(cis), samples=bootstrap_samples
            )
    if include_w8_w16:
        diffs = pooled_per_opening_differences(
            suite_rows=suite_rows,
            candidate_a="disagreement_w8_policy_head_e1",
            candidate_b="disagreement_w16_policy_head_e2",
            budget_pair=PRIMARY_BUDGET,
            metric_key="ds",
        )
        cis["w8_e1_minus_w16_e2_384_256"] = bootstrap_ci(
            diffs, seed=seed + len(cis), samples=bootstrap_samples
        )
    return cis


def p0_p1_imbalance_worse(
    *, suite_rows: dict[str, Any], candidate_name: str, reference_name: str
) -> bool:
    candidate_gaps: list[float] = []
    reference_gaps: list[float] = []
    for suite_row in suite_rows.values():
        candidate_budget = suite_row["candidates"][candidate_name]["budget_results"][
            PRIMARY_BUDGET
        ]
        reference_budget = suite_row["candidates"][reference_name]["budget_results"][
            PRIMARY_BUDGET
        ]
        candidate_gaps.append(
            abs(
                float(candidate_budget["p0_score"])
                - float(candidate_budget["p1_score"])
            )
        )
        reference_gaps.append(
            abs(
                float(reference_budget["p0_score"])
                - float(reference_budget["p1_score"])
            )
        )
    return statistics.fmean(candidate_gaps) - statistics.fmean(reference_gaps) > 0.05


def candidate_has_unacceptable_768_regression(
    *, aggregate: dict[str, Any], bootstrap_cis: dict[str, Any], candidate_name: str
) -> bool:
    mean_regression = aggregate[candidate_name][EQ_768_BUDGET][
        "mean_delta_vs_promoted_current_ref"
    ]
    lower_ci = bootstrap_cis[ci_key(candidate_name, EQ_768_BUDGET)]["lower"]
    return mean_regression < -0.15 or lower_ci < -0.25


def classify_summary(summary: dict[str, Any]) -> tuple[str, str]:
    reference_name = "promoted_current_ref"
    aggregate = summary["aggregate_robustness"]
    bootstrap_cis = summary["bootstrap_cis"]
    gates = summary["default_gate"]

    if not summary["hash_validation_passed"]:
        return (
            "no_weighted_candidate",
            "Artifact hashes did not match the PR #123 expectations.",
        )

    w8 = "disagreement_w8_policy_head_e1"
    w4 = "disagreement_w4_policy_head_e2"
    w16 = "disagreement_w16_policy_head_e2"

    w8_384 = aggregate[w8][PRIMARY_BUDGET]
    w8_768 = aggregate[w8][EQ_768_BUDGET]
    w8_1200 = aggregate[w8][HIGH_EQ_BUDGET]
    w8_1200_256 = aggregate[w8][HIGH_ASYM_BUDGET]
    w4_384 = aggregate[w4][PRIMARY_BUDGET]
    w4_768 = aggregate[w4][EQ_768_BUDGET]
    w16_384 = aggregate[w16][PRIMARY_BUDGET]
    w16_768 = aggregate[w16][EQ_768_BUDGET]
    w8_gate = candidate_gate_preserved(gates[w8])
    w4_gate = candidate_gate_preserved(gates[w4])
    w16_gate = candidate_gate_preserved(gates[w16])
    w8_reject = candidate_has_unacceptable_768_regression(
        aggregate=aggregate, bootstrap_cis=bootstrap_cis, candidate_name=w8
    )
    w4_reject = candidate_has_unacceptable_768_regression(
        aggregate=aggregate, bootstrap_cis=bootstrap_cis, candidate_name=w4
    )
    w16_reject = candidate_has_unacceptable_768_regression(
        aggregate=aggregate, bootstrap_cis=bootstrap_cis, candidate_name=w16
    )

    if (
        w8_384["mean_delta_vs_promoted_current_ref"] >= 0.15
        and bootstrap_cis[ci_key(w8, PRIMARY_BUDGET)]["lower"] > 0.08
        and w8_384["worst_suite_delta_vs_promoted_current_ref"] > 0.08
        and w8_768["mean_delta_vs_promoted_current_ref"] >= -0.15
        and bootstrap_cis[ci_key(w8, EQ_768_BUDGET)]["lower"] >= -0.25
        and w8_1200["mean_delta_vs_promoted_current_ref"] >= -0.03
        and w8_1200_256["mean_delta_vs_promoted_current_ref"] >= -0.03
        and w8_gate
    ):
        return (
            "select_w8_e1_for_promotion_followup",
            "w8_e1 cleared the 384:256 movement threshold and stayed within the equal-budget and high-budget regression bounds.",
        )

    w4_within_w8 = w8_384["mean_ds"] - w4_384["mean_ds"] <= 0.05
    w4_materially_better_768 = (
        w4_768["mean_delta_vs_promoted_current_ref"]
        > w8_768["mean_delta_vs_promoted_current_ref"] + 0.05
    )
    if (
        w4_within_w8
        and w4_materially_better_768
        and w4_384["mean_delta_vs_promoted_current_ref"] >= 0.12
        and not w4_reject
        and w4_gate
    ):
        return (
            "select_w4_e2_for_promotion_followup",
            "w4_e2 stayed close to w8_e1 on 384:256 while materially improving the 768:768 robustness profile.",
        )

    if (
        not w16_reject
        and (w8_reject and w4_reject)
        and w16_384["mean_delta_vs_promoted_current_ref"] >= 0.08
        and w16_768["mean_delta_vs_promoted_current_ref"]
        >= max(
            w8_768["mean_delta_vs_promoted_current_ref"],
            w4_768["mean_delta_vs_promoted_current_ref"],
        )
        and w16_gate
    ):
        return (
            "select_w16_e2_for_promotion_followup",
            "w16_e2 was the only candidate that preserved the robustness constraints while still moving 384:256 enough to matter.",
        )

    robust_384_improvement = any(
        aggregate[name][PRIMARY_BUDGET]["mean_delta_vs_promoted_current_ref"] > 0.08
        and bootstrap_cis[ci_key(name, PRIMARY_BUDGET)]["lower"] > 0.0
        for name in summary["evaluated_weighted_candidates"]
    )
    all_bad_768 = all(
        candidate_has_unacceptable_768_regression(
            aggregate=aggregate, bootstrap_cis=bootstrap_cis, candidate_name=name
        )
        for name in summary["evaluated_weighted_candidates"]
    )
    worse_imbalance = any(
        p0_p1_imbalance_worse(
            suite_rows=summary["suite_results"],
            candidate_name=name,
            reference_name=reference_name,
        )
        for name in summary["evaluated_weighted_candidates"]
    )
    if robust_384_improvement and (all_bad_768 or worse_imbalance):
        return (
            "needs_balanced_replay_followup",
            "Weighted replay improved the primary 384:256 target, but every viable lane still carried unacceptable equal-budget or seat-balance risk.",
        )

    reproduced = any(
        aggregate[name][PRIMARY_BUDGET]["mean_delta_vs_promoted_current_ref"] > 0.0
        for name in summary["evaluated_weighted_candidates"]
    )
    if (
        not reproduced
        or any(
            not candidate_gate_preserved(gates[name])
            for name in summary["required_gate_candidates"]
        )
        or any(
            bootstrap_cis[ci_key(name, PRIMARY_BUDGET)]["lower"] <= 0.0
            for name in summary["evaluated_weighted_candidates"]
        )
    ):
        return (
            "no_weighted_candidate",
            "The fixed-large improvement did not reproduce robustly enough across held-out suites, CIs, and gate preservation.",
        )

    return (
        "needs_balanced_replay_followup",
        "The weighted lanes showed signal, but the robustness evidence was not clean enough to advance a promotion follow-up without an equal-budget stability anchor.",
    )


def write_markdown_report(summary: dict[str, Any], out_path: Path) -> None:
    manifest_rows = []
    for candidate in summary["candidates"]:
        manifest_rows.append(
            [
                candidate["name"],
                candidate["artifact_dir"],
                candidate["artifact_weights_sha256"],
                candidate.get("checkpoint_sha256") or "n/a",
                candidate["parent_promoted_current_hash"],
                candidate["source_experiment_doc"],
            ]
        )

    suite_hash_rows = []
    for suite_name, suite_info in summary["suite_inputs"].items():
        suite_hash_rows.append(
            [suite_name, suite_info["path"], suite_info["sha256"], suite_info["rows"]]
        )

    fixed_rows = []
    heldout_rows = []
    for suite_name, suite_row in summary["suite_results"].items():
        table_rows = fixed_rows if suite_name == "fixed_large" else heldout_rows
        for candidate_name in summary["candidate_order"]:
            candidate_row = suite_row["candidates"][candidate_name]
            budget = candidate_row["budget_results"][PRIMARY_BUDGET]
            table_rows.append(
                [
                    suite_name,
                    candidate_name,
                    candidate_row["candidate_sha256"],
                    candidate_row["current_sha256"],
                    fmt(budget["ds"]),
                    f"{budget['p0_score']:.4f} / {budget['p1_score']:.4f}",
                    budget["duplicate_trajectory_count"],
                    fmt(
                        suite_row["delta_vs_promoted_current_ref"][candidate_name][
                            PRIMARY_BUDGET
                        ]
                    ),
                ]
            )

    aggregate_rows = []
    for candidate_name in summary["candidate_order"]:
        agg_384 = summary["aggregate_robustness"][candidate_name][PRIMARY_BUDGET]
        agg_768 = summary["aggregate_robustness"][candidate_name][EQ_768_BUDGET]
        agg_1200 = summary["aggregate_robustness"][candidate_name][HIGH_EQ_BUDGET]
        agg_1200_256 = summary["aggregate_robustness"][candidate_name][HIGH_ASYM_BUDGET]
        aggregate_rows.append(
            [
                candidate_name,
                fmt(agg_384["mean_ds"]),
                fmt(agg_384["mean_delta_vs_promoted_current_ref"]),
                fmt(agg_384["worst_suite_delta_vs_promoted_current_ref"]),
                f"{agg_384['stddev_ds']:.4f}",
                fmt(agg_768["mean_delta_vs_promoted_current_ref"]),
                fmt(agg_1200["mean_delta_vs_promoted_current_ref"]),
                fmt(agg_1200_256["mean_delta_vs_promoted_current_ref"]),
            ]
        )

    ci_rows = []
    for label, ci in summary["bootstrap_cis"].items():
        ci_rows.append(
            [label, fmt(ci["mean"]), fmt(ci["lower"]), fmt(ci["upper"]), ci["n"]]
        )

    gate_rows = []
    for candidate_name, report in summary["default_gate"].items():
        gate_rows.append(
            [
                candidate_name,
                report["classification"],
                "yes" if candidate_gate_preserved(report) else "no",
                fmt(
                    report["budget_results"]
                    .get("384:256", {})
                    .get("disadvantaged_seat_score")
                ),
                fmt(
                    report["budget_results"]
                    .get("1200:1200", {})
                    .get("disadvantaged_seat_score")
                ),
            ]
        )

    regression_rows = []
    for candidate_name in summary["evaluated_weighted_candidates"]:
        agg = summary["aggregate_robustness"][candidate_name][EQ_768_BUDGET]
        ci = summary["bootstrap_cis"][ci_key(candidate_name, EQ_768_BUDGET)]
        regression_rows.append(
            [
                candidate_name,
                fmt(agg["mean_ds"]),
                fmt(agg["mean_delta_vs_promoted_current_ref"]),
                fmt(agg["worst_suite_delta_vs_promoted_current_ref"]),
                fmt(ci["lower"]),
                fmt(ci["upper"]),
                "unacceptable"
                if candidate_has_unacceptable_768_regression(
                    aggregate=summary["aggregate_robustness"],
                    bootstrap_cis=summary["bootstrap_cis"],
                    candidate_name=candidate_name,
                )
                else "within guardrail",
            ]
        )

    opening_counts = summary["opening_outcome_counts_384_256_disadvantaged"]
    lines = [
        "# AlphaZero-Lite PR123 Weighted Candidate Preflight Results",
        "",
        f"**Date**: {summary['date']}",
        f"**Classification**: `{summary['classification']}`",
        f"**Schema**: `{summary['schema']}`",
        "",
        "## Candidate Manifest And Hashes",
        "",
        markdown_table(
            [
                "Candidate",
                "Artifact path",
                "Artifact weights SHA256",
                "Checkpoint SHA256",
                "Parent promoted-current hash",
                "Source experiment doc",
            ],
            manifest_rows,
        ),
        "",
        "## Suite Hashes",
        "",
        markdown_table(["Suite", "Path", "SHA256", "Rows"], suite_hash_rows),
        "",
        "## Fixed-Suite Table",
        "",
        markdown_table(
            [
                "Suite",
                "Candidate",
                "Candidate SHA256",
                "Current SHA256",
                "384:256 DS",
                "384:256 P0 / P1",
                "Duplicate trajectories",
                "Delta vs promoted_current_ref",
            ],
            fixed_rows,
        ),
        "",
        "## Held-Out-Suite Table",
        "",
        markdown_table(
            [
                "Suite",
                "Candidate",
                "Candidate SHA256",
                "Current SHA256",
                "384:256 DS",
                "384:256 P0 / P1",
                "Duplicate trajectories",
                "Delta vs promoted_current_ref",
            ],
            heldout_rows,
        ),
        "",
        "## Aggregate Robustness Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean DS 384:256",
                "Mean delta 384:256",
                "Worst-suite delta 384:256",
                "Stddev DS 384:256",
                "Mean delta 768:768",
                "Mean delta 1200:1200",
                "Mean delta 1200:256",
            ],
            aggregate_rows,
        ),
        "",
        "## Bootstrap CI Table",
        "",
        markdown_table(
            ["Comparison", "Mean", "Lower 95%", "Upper 95%", "Openings"], ci_rows
        ),
        "",
        "## Gate Classification Table",
        "",
        markdown_table(
            [
                "Candidate",
                "Classification",
                "High-search preserved",
                "384:256 disadvantaged",
                "1200:1200 disadvantaged",
            ],
            gate_rows,
        ),
        "",
        "## 768:768 Regression Analysis",
        "",
        markdown_table(
            [
                "Candidate",
                "Mean DS 768:768",
                "Mean delta vs ref",
                "Worst-suite delta vs ref",
                "Lower 95%",
                "Upper 95%",
                "Assessment",
            ],
            regression_rows,
        ),
        "",
        "## Opening-Level Win Tie Loss Counts",
        "",
        markdown_table(
            ["Comparison", "Candidate better", "Reference better", "Tie"],
            [
                [
                    "w8_e1 vs promoted_current_ref",
                    opening_counts[
                        "disagreement_w8_policy_head_e1_vs_promoted_current_ref"
                    ]["disagreement_w8_policy_head_e1"],
                    opening_counts[
                        "disagreement_w8_policy_head_e1_vs_promoted_current_ref"
                    ]["promoted_current_ref"],
                    opening_counts[
                        "disagreement_w8_policy_head_e1_vs_promoted_current_ref"
                    ]["tie"],
                ],
                [
                    "w4_e2 vs promoted_current_ref",
                    opening_counts[
                        "disagreement_w4_policy_head_e2_vs_promoted_current_ref"
                    ]["disagreement_w4_policy_head_e2"],
                    opening_counts[
                        "disagreement_w4_policy_head_e2_vs_promoted_current_ref"
                    ]["promoted_current_ref"],
                    opening_counts[
                        "disagreement_w4_policy_head_e2_vs_promoted_current_ref"
                    ]["tie"],
                ],
            ],
        ),
        "",
        "## Final Classification And Recommendation",
        "",
        f"- Classification: `{summary['classification']}`",
        f"- Recommendation: `{summary['recommendation']}`",
        f"- Rationale: {summary['decision_rationale']}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir", default="/tmp/azlite_pr123_weighted_candidate_preflight"
    )
    parser.add_argument(
        "--pr123-workdir",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation",
    )
    parser.add_argument(
        "--pr123-summary",
        default="/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/summary_metrics.json",
    )
    parser.add_argument("--current", default="model-artifact/current")
    parser.add_argument(
        "--expected-current-weights-sha256",
        default=REQUIRED_CANDIDATES["promoted_current_ref"][
            "expected_artifact_weights_sha256"
        ],
    )
    parser.add_argument(
        "--fixed-large-suite", default="/tmp/azlite_opening_suite/large_eval.jsonl"
    )
    parser.add_argument(
        "--medium-suite", default="/tmp/azlite_opening_suite/medium_eval.jsonl"
    )
    parser.add_argument(
        "--heldout-suites",
        default=(
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl,"
            "/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl"
        ),
    )
    parser.add_argument("--extra-heldout-seeds", default="46,47,48")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite-timeout", type=int, default=14400)
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--budget-pairs", default=DEFAULT_BUDGET_PAIRS)
    parser.add_argument("--gate-budget-pairs", default=DEFAULT_GATE_BUDGET_PAIRS)
    parser.add_argument("--gate-games", type=int, default=60)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--skip-optional-candidate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games_per_opening != 2:
        raise ValueError(
            "games_per_opening must be exactly 2 for this deterministic preflight"
        )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    suite_dir = workdir / "suites"
    suite_dir.mkdir(parents=True, exist_ok=True)

    pr123_workdir = Path(args.pr123_workdir)
    pr123_summary_path = Path(args.pr123_summary)
    current_artifact = Path(args.current)
    fixed_large_suite = Path(args.fixed_large_suite)
    medium_suite = Path(args.medium_suite)
    heldout_suite_paths = parse_csv_paths(args.heldout_suites)
    extra_heldout_seeds = parse_csv_ints(args.extra_heldout_seeds)

    require_existing_file(pr123_summary_path, "PR123 summary")
    require_existing_file(current_artifact / "weights.json", "current artifact weights")
    require_existing_file(fixed_large_suite, "fixed large suite")
    require_existing_file(medium_suite, "medium suite")
    for heldout_suite_path in heldout_suite_paths:
        require_existing_file(
            heldout_suite_path, f"heldout suite {heldout_suite_path.name}"
        )

    pr123_summary = load_json(pr123_summary_path)
    current_weights_sha = sha256_file(current_artifact / "weights.json")
    verify_expected_hash(
        current_artifact / "weights.json",
        args.expected_current_weights_sha256,
        "current artifact weights",
    )

    candidates: list[dict[str, Any]] = []
    hash_validation_passed = True
    for name, expected in REQUIRED_CANDIDATES.items():
        entry = discover_candidate_entry(
            name=name,
            workdir=pr123_workdir,
            pr123_summary=pr123_summary,
            current_artifact=current_artifact,
            current_weights_sha=current_weights_sha,
            expected_checkpoint_sha256=expected["expected_checkpoint_sha256"],
            expected_artifact_weights_sha256=expected[
                "expected_artifact_weights_sha256"
            ],
        )
        if entry is None:
            hash_validation_passed = False
            raise RuntimeError(f"missing required candidate artifact for {name}")
        candidates.append(entry)

    if not args.skip_optional_candidate:
        for name, expected in OPTIONAL_CANDIDATES.items():
            entry = discover_candidate_entry(
                name=name,
                workdir=pr123_workdir,
                pr123_summary=pr123_summary,
                current_artifact=current_artifact,
                current_weights_sha=current_weights_sha,
                expected_checkpoint_sha256=expected["expected_checkpoint_sha256"],
                expected_artifact_weights_sha256=expected[
                    "expected_artifact_weights_sha256"
                ],
            )
            if entry is not None:
                candidates.append(entry)

    candidate_order = [candidate["name"] for candidate in candidates]
    evaluated_weighted_candidates = [
        name for name in candidate_order if name != "promoted_current_ref"
    ]

    suite_inputs: dict[str, dict[str, Any]] = {
        "fixed_large": build_input_summary(fixed_large_suite),
        "medium": build_input_summary(medium_suite),
    }
    for heldout_suite_path in heldout_suite_paths:
        suite_inputs[heldout_suite_path.stem] = build_input_summary(heldout_suite_path)

    excluded_suite_paths = list(heldout_suite_paths)
    extra_suite_paths: list[Path] = []
    for seed_value in extra_heldout_seeds:
        out_path = suite_dir / f"heldout_seed{seed_value}_large.jsonl"
        suite_info = build_extra_heldout_suite(
            fixed_suite_path=fixed_large_suite,
            exclude_suite_paths=excluded_suite_paths + extra_suite_paths,
            seed=seed_value,
            out_path=out_path,
        )
        suite_inputs[f"heldout_seed{seed_value}_large"] = suite_info
        extra_suite_paths.append(out_path)

    candidate_paths = ",".join(candidate["artifact_dir"] for candidate in candidates)
    suite_plan = [("fixed_large", fixed_large_suite)]
    suite_plan.extend((path.stem, path) for path in heldout_suite_paths)
    suite_plan.extend((path.stem, path) for path in extra_suite_paths)
    budget_pairs = [
        item.strip() for item in args.budget_pairs.split(",") if item.strip()
    ]

    suite_results: dict[str, Any] = {}
    for suite_name, suite_path in suite_plan:
        eval_dir = workdir / f"eval_{suite_name}"
        report_path = eval_dir / "temperature_benchmark_report.json"
        if report_path.is_file():
            existing_report = load_json(report_path)
            if benchmark_report_complete(
                report=existing_report, candidates=candidates, budget_pairs=budget_pairs
            ):
                report = existing_report
            else:
                report = run_opening_suite_benchmark(
                    workdir=str(eval_dir),
                    suite=str(suite_path),
                    current=str(current_artifact),
                    candidates=candidate_paths,
                    budget_pairs=args.budget_pairs,
                    games_per_opening=args.games_per_opening,
                    seed=args.seed,
                    workers=args.workers,
                    timeout=args.suite_timeout,
                )
        else:
            report = run_opening_suite_benchmark(
                workdir=str(eval_dir),
                suite=str(suite_path),
                current=str(current_artifact),
                candidates=candidate_paths,
                budget_pairs=args.budget_pairs,
                games_per_opening=args.games_per_opening,
                seed=args.seed,
                workers=args.workers,
                timeout=args.suite_timeout,
            )
        suite_results[suite_name] = suite_result_rows(
            report=report,
            candidates=candidates,
            suite_name=suite_name,
        )
    annotate_suite_deltas(
        suite_rows=suite_results,
        reference_name="promoted_current_ref",
        budget_pairs=budget_pairs,
    )

    aggregate_robustness = {
        candidate_name: aggregate_candidate_metrics(
            suite_rows=suite_results,
            reference_name="promoted_current_ref",
            candidate_name=candidate_name,
            budget_pairs=budget_pairs,
        )
        for candidate_name in candidate_order
    }

    bootstrap_cis = build_bootstrap_cis(
        suite_rows=suite_results,
        reference_name="promoted_current_ref",
        candidate_names=evaluated_weighted_candidates,
        include_w8_w4="disagreement_w4_policy_head_e2" in candidate_order,
        include_w8_w16="disagreement_w16_policy_head_e2" in candidate_order,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    opening_outcome_counts_384_256_disadvantaged = {
        "disagreement_w8_policy_head_e1_vs_promoted_current_ref": opening_outcome_counts(
            suite_rows=suite_results,
            budget_pair=PRIMARY_BUDGET,
            candidate_a="disagreement_w8_policy_head_e1",
            candidate_b="promoted_current_ref",
        ),
        "disagreement_w4_policy_head_e2_vs_promoted_current_ref": opening_outcome_counts(
            suite_rows=suite_results,
            budget_pair=PRIMARY_BUDGET,
            candidate_a="disagreement_w4_policy_head_e2",
            candidate_b="promoted_current_ref",
        ),
    }

    gate_targets = [
        "promoted_current_ref",
        "disagreement_w8_policy_head_e1",
        "disagreement_w4_policy_head_e2",
        "disagreement_w16_policy_head_e2",
    ]
    if "disagreement_w16_policy_head_e1" in candidate_order:
        gate_targets.append("disagreement_w16_policy_head_e1")

    gate_reports: dict[str, Any] = {}
    for candidate_name in gate_targets:
        candidate_dir = next(
            candidate["artifact_dir"]
            for candidate in candidates
            if candidate["name"] == candidate_name
        )
        gate_out = workdir / "eval_gate" / f"{candidate_name}.json"
        gate_budget_pairs = [
            item.strip() for item in args.gate_budget_pairs.split(",") if item.strip()
        ]
        if gate_out.is_file():
            existing_gate_report = load_json(gate_out)
            if gate_report_complete(existing_gate_report, gate_budget_pairs):
                gate_report = existing_gate_report
            else:
                gate_report = run_default_gate(
                    candidate_path=candidate_dir,
                    current_path=str(current_artifact),
                    out=str(gate_out),
                    seed=args.seed,
                    workers=args.workers,
                    games=args.gate_games,
                    budget_pairs=args.gate_budget_pairs,
                )
        else:
            gate_report = run_default_gate(
                candidate_path=candidate_dir,
                current_path=str(current_artifact),
                out=str(gate_out),
                seed=args.seed,
                workers=args.workers,
                games=args.gate_games,
                budget_pairs=args.gate_budget_pairs,
            )
        gate_reports[candidate_name] = {
            "candidate_path": candidate_dir,
            "classification": gate_report.get("classification"),
            "budget_results": gate_budget_results(gate_report),
        }

    summary: dict[str, Any] = {
        "schema": PR123_SCHEMA,
        "date": str(date.today()),
        "workdir": str(workdir),
        "pr123_workdir": str(pr123_workdir),
        "pr123_summary": str(pr123_summary_path),
        "seed": args.seed,
        "workers": args.workers,
        "root_policy_mode": "deterministic",
        "games_per_opening": args.games_per_opening,
        "budget_pairs": budget_pairs,
        "gate_budget_pairs": [
            item.strip() for item in args.gate_budget_pairs.split(",") if item.strip()
        ],
        "hash_validation_passed": hash_validation_passed,
        "candidates": candidates,
        "candidate_order": candidate_order,
        "evaluated_weighted_candidates": evaluated_weighted_candidates,
        "required_gate_candidates": gate_targets,
        "suite_inputs": suite_inputs,
        "suite_results": suite_results,
        "aggregate_robustness": aggregate_robustness,
        "bootstrap_cis": bootstrap_cis,
        "opening_outcome_counts_384_256_disadvantaged": opening_outcome_counts_384_256_disadvantaged,
        "default_gate": gate_reports,
        "report_budget_label_map": BENCHMARK_LABEL_TO_BUDGET,
        "guardrails": {
            "training": False,
            "self_play_generation": False,
            "promotion": False,
            "overwrite_current": False,
        },
    }
    classification, rationale = classify_summary(summary)
    summary["classification"] = classification
    summary["recommendation"] = classification
    summary["decision_rationale"] = rationale

    summary_path = workdir / "summary_metrics.json"
    write_json(summary_path, summary)
    write_markdown_report(summary, REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
