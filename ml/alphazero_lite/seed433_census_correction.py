"""Retrospective, append-only mathematical correction to the seed432 census."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed432_policy_target_census import entropy

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/data/seed432-policy-target-compatibility"
DATA = ROOT / "docs/data/seed433-seed432-census-correction"
THRESHOLDS = {"mean_js_nats": 0.05, "weighted_top_action_disagreement": 0.10}


def group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    active = [row for row in rows if float(row[field]) > 0]
    mass = sum(float(row[field]) for row in active)
    if not mass:
        return {
            "policy_mass": 0.0,
            "js_nats": 0.0,
            "top_action_disagreement": 0.0,
            "within_source_js_nats": 0.0,
            "between_source_js_nats": 0.0,
        }
    weights = np.asarray([float(row[field]) for row in active])
    policies = [np.asarray(row["target"], dtype=np.float64) for row in active]
    mean = np.average(np.stack(policies), axis=0, weights=weights)
    js = max(
        0.0,
        entropy(mean)
        - float(np.average([entropy(p) for p in policies], weights=weights)),
    )
    tops = [int(np.flatnonzero(policy == policy.max())[0]) for policy in policies]
    pairs = [(i, j) for i in range(len(active)) for j in range(i + 1, len(active))]
    pair_mass = sum(weights[i] * weights[j] for i, j in pairs)
    disagreement = (
        sum(weights[i] * weights[j] * (tops[i] != tops[j]) for i, j in pairs)
        / pair_mass
        if pair_mass
        else 0.0
    )
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(active):
        by_source[row["source"]].append(i)
    within = 0.0
    source_entropy = 0.0
    for indexes in by_source.values():
        source_mass = float(weights[indexes].sum())
        source_mean = np.average(
            np.stack([policies[i] for i in indexes]), axis=0, weights=weights[indexes]
        )
        source_entropy += source_mass / mass * entropy(source_mean)
        within += (
            source_mass
            / mass
            * max(
                0.0,
                entropy(source_mean)
                - float(
                    np.average(
                        [entropy(policies[i]) for i in indexes],
                        weights=weights[indexes],
                    )
                ),
            )
        )
    return {
        "policy_mass": mass,
        "js_nats": js,
        "top_action_disagreement": disagreement,
        "within_source_js_nats": within,
        "between_source_js_nats": max(0.0, entropy(mean) - source_entropy),
    }


def summarize(
    rows: list[dict[str, Any]], weighting: str, treatment: bool = False
) -> dict[str, Any]:
    prefix = "treatment_" if treatment else ""
    field = prefix + (
        "exposure_weight" if weighting == "exposure" else "equal_input_weight"
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["partition"], row["input_sha256"])].append(row)
    entries = [(members, group_metrics(members, field)) for members in groups.values()]
    positive = [
        (members, metrics) for members, metrics in entries if metrics["policy_mass"] > 0
    ]
    outer = [
        metrics["policy_mass"] if weighting == "exposure" else 1.0
        for _, metrics in positive
    ]
    denominator = float(sum(outer))

    def aggregate(key: str) -> float:
        return (
            sum(
                metrics[key] * weight
                for (_, metrics), weight in zip(positive, outer, strict=True)
            )
            / denominator
            if denominator
            else 0.0
        )

    return {
        "weighting": weighting,
        "within_group_weighting": "replay multiplicity times policy coefficient"
        if weighting == "exposure"
        else "policy coefficient without replay-copy multiplicity",
        "between_group_weighting": "total exposure mass"
        if weighting == "exposure"
        else "one unit per positive-policy-mass exact-input group",
        "input_groups": len(groups),
        "positive_mass_groups": len(positive),
        "zero_mass_groups": len(entries) - len(positive),
        "policy_mass": sum(metrics["policy_mass"] for _, metrics in entries),
        "aggregation_denominator": denominator,
        "mean_js_nats": aggregate("js_nats"),
        "weighted_top_action_disagreement": aggregate("top_action_disagreement"),
        "entropy_decomposition": {
            key: aggregate(key)
            for key in ("within_source_js_nats", "between_source_js_nats")
        },
    }


def calculate(source: Path = SOURCE, output: Path | None = DATA) -> dict[str, Any]:
    with gzip.open(
        source / "row-accounting.jsonl.gz", "rt", encoding="utf-8"
    ) as stream:
        rows = [json.loads(line) for line in stream]
    populations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stones = row["active_stones"]
        bucket = "le16" if stones <= 16 else "17-32" if stones <= 32 else "gt32"
        populations[f"{row['partition']}/{bucket}"].append(row)
    summaries = {
        name: {kind: summarize(pop, kind) for kind in ("exposure", "equal_input")}
        | {
            f"seed429_treatment_{kind}": summarize(pop, kind, True)
            for kind in ("exposure", "equal_input")
        }
        for name, pop in populations.items()
    }
    primary = summaries["train/gt32"]
    passed = all(
        primary[k]["mean_js_nats"] >= 0.05
        and primary[k]["weighted_top_action_disagreement"] >= 0.10
        for k in ("exposure", "equal_input")
    )
    result = {
        "schema": "seed433-corrected-results-v1",
        "retrospective_correction": True,
        "source_seed432_results_sha256": sha(source / "results.json"),
        "thresholds": THRESHOLDS,
        "populations": summaries,
        "classification": "material_policy_target_disagreement"
        if passed
        else "target_disagreement_not_material_under_registered_thresholds",
        "decision_scope": "original control exposure and equal-input entries for training >32 stones only; treatment entries are secondary diagnostics",
        "cross_entropy_equivalence_limitation": "For a fixed logits vector and preserved total coefficient, target averaging preserves weighted cross-entropy and gradient in exact arithmetic; this does not establish equivalence under minibatch/optimizer averaging.",
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "corrected-results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
