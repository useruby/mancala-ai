"""Same-input policy-target compatibility census for the frozen seed461 mixture."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.seed429_policy_normalization import (
    canonical_identity_from_encoded_state,
    normalize_policy_coefficients,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data/seed432-policy-target-compatibility"
THRESHOLDS = {"mean_js_nats": 0.05, "weighted_top_action_disagreement": 0.10}


def entropy(policy: np.ndarray) -> float:
    p = np.asarray(policy, dtype=np.float64)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0]), dtype=np.float64))


def group_metrics(rows: list[dict[str, Any]], weight_field: str) -> dict[str, Any]:
    active = [r for r in rows if float(r[weight_field]) > 0]
    total = sum(float(r[weight_field]) for r in active)
    if not active or total == 0:
        return {
            "rows": len(rows),
            "policy_mass": 0.0,
            "js_nats": 0.0,
            "top_action_disagreement": 0.0,
            "top_action_tie_rows": 0,
        }
    policies = [np.asarray(r["target"], dtype=np.float64) for r in active]
    weights = np.asarray([float(r[weight_field]) for r in active], dtype=np.float64)
    mean = np.average(np.stack(policies), axis=0, weights=weights)
    js = entropy(mean) - float(
        np.average([entropy(p) for p in policies], weights=weights)
    )
    top = [int(np.flatnonzero(p == p.max())[0]) for p in policies]
    tie_rows = sum(np.count_nonzero(p == p.max()) > 1 for p in policies)
    pair_mass = 0.0
    pair_disagree = 0.0
    pair_by_source: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            pair_weight = weights[i] * weights[j]
            pair_mass += pair_weight
            disagreement = float(top[i] != top[j])
            pair_disagree += pair_weight * disagreement
            pair_key = "|".join(sorted((active[i]["source"], active[j]["source"])))
            pair_by_source[pair_key][0] += pair_weight * disagreement
            pair_by_source[pair_key][1] += pair_weight
    source_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(active):
        source_rows[row["source"]].append(index)
    within = 0.0
    source_means: dict[str, tuple[float, float]] = {}
    for source, indexes in source_rows.items():
        source_weight = float(weights[indexes].sum())
        source_mean = np.average(
            np.stack([policies[i] for i in indexes]), axis=0, weights=weights[indexes]
        )
        source_means[source] = (source_weight, entropy(source_mean))
        source_js = entropy(source_mean) - float(
            np.average(
                [entropy(policies[i]) for i in indexes], weights=weights[indexes]
            )
        )
        within += source_weight / total * max(0.0, source_js)
    between = entropy(mean) - sum(w / total * h for w, h in source_means.values())
    return {
        "rows": len(rows),
        "active_rows": len(active),
        "policy_mass": total,
        "js_nats": max(0.0, js),
        "top_action_disagreement": pair_disagree / pair_mass if pair_mass else 0.0,
        "top_action_tie_rows": int(tie_rows),
        "weighted_mean_policy": mean.tolist(),
        "within_source_js_nats": within,
        "between_source_js_nats": max(0.0, between),
        "source_pair_top_action_disagreement": {
            key: values[0] / values[1] if values[1] else 0.0
            for key, values in pair_by_source.items()
        },
    }


def summarize(
    rows: list[dict[str, Any]], weighting: str, *, treatment: bool = False
) -> dict[str, Any]:
    by_input: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_input[row["input_sha256"]].append(row)
        by_canonical[row["canonical_state"]].append(row)
    prefix = "treatment_" if treatment else ""
    field = prefix + (
        "exposure_weight" if weighting == "exposure" else "equal_input_weight"
    )
    groups = [group_metrics(group, field) for group in by_input.values()]
    group_weights = [sum(float(r[field]) for r in group) for group in by_input.values()]
    denominator = sum(group_weights)
    source_pair: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for group, group_weight in zip(by_input.values(), group_weights, strict=True):
        metrics = group_metrics(group, field)
        for key, disagreement in metrics["source_pair_top_action_disagreement"].items():
            source_pair[key][0] += disagreement * group_weight
            source_pair[key][1] += group_weight
    return {
        "weighting": weighting,
        "input_groups": len(groups),
        "duplicate_groups": sum(len(g) > 1 for g in by_input.values()),
        "policy_mass_coverage": denominator,
        "mean_js_nats": sum(g["js_nats"] * w for g, w in zip(groups, group_weights))
        / denominator
        if denominator
        else 0.0,
        "weighted_pairwise_top_action_disagreement": sum(
            g["top_action_disagreement"] * w for g, w in zip(groups, group_weights)
        )
        / denominator
        if denominator
        else 0.0,
        "js_distribution": {
            "min": min((g["js_nats"] for g in groups), default=0),
            "median": float(np.median([g["js_nats"] for g in groups])) if groups else 0,
            "max": max((g["js_nats"] for g in groups), default=0),
        },
        "canonical_group_count": len(by_canonical),
        "canonical_input_partition_disagreements": sum(
            1
            for group in by_canonical.values()
            if len({r["input_sha256"] for r in group}) > 1
        ),
        "input_groups_with_multiple_canonical_states": sum(
            len({r["canonical_state"] for r in group}) > 1
            for group in by_input.values()
        ),
        "entropy_decomposition": {
            "within_source_js_nats": sum(
                g["within_source_js_nats"] * w for g, w in zip(groups, group_weights)
            )
            / denominator
            if denominator
            else 0.0,
            "between_source_js_nats": sum(
                g["between_source_js_nats"] * w for g, w in zip(groups, group_weights)
            )
            / denominator
            if denominator
            else 0.0,
            "decomposition_sum_nats": sum(
                (g["within_source_js_nats"] + g["between_source_js_nats"]) * w
                for g, w in zip(groups, group_weights)
            )
            / denominator
            if denominator
            else 0.0,
        },
        "source_coverage": {
            source: {
                "compact_rows": sum(r["source"] == source for r in rows),
                "positive_coefficient_rows": sum(
                    r["source"] == source and r["policy_coefficient"] > 0 for r in rows
                ),
                "policy_mass": sum(
                    float(r[field]) for r in rows if r["source"] == source
                ),
            }
            for source in sorted({r["source"] for r in rows})
        },
        "source_pair_top_action_disagreement": {
            pair: value[0] / value[1] if value[1] else 0.0
            for pair, value in sorted(source_pair.items())
        },
    }


def build(root: Path = ROOT, out: Path = DATA) -> dict[str, Any]:
    """Reconstruct eligible rows using the frozen loader and publish portable evidence."""
    from ml.alphazero_lite.train import load_jsonl_replay

    reg_path = root / "docs/data/seed416-policy-target-softening/registration-v3.json"
    reg = json.loads(reg_path.read_text())
    records = reg["replays"]
    paths = [Path(r["path"]) for r in records]
    if any(
        not p.is_file() or sha256_file(p) != r["sha256"]
        for p, r in zip(paths, records, strict=True)
    ):
        raise ValueError("registered_replay_hash_mismatch")
    x, policies, _values, replay, coeff = load_jsonl_replay(
        paths,
        [r["weight"] for r in records],
        policy_target_mode="sharpened",
        value_target_mode="default",
        replay_value_target_modes=[r["value_target_mode"] for r in records],
        include_policy_loss_weights=True,
    )
    split_spec = reg["training"]["source_row_split"]
    split_path = root / split_spec["path"]
    if sha256_file(split_path) != split_spec["sha256"]:
        raise ValueError("registered_split_hash_mismatch")
    with gzip.open(split_path, "rt") as stream:
        split = json.load(stream)
    train_pos = np.asarray(split["train_positions"], dtype=np.int64)
    valid_pos = np.asarray(split["validation_positions"], dtype=np.int64)
    n = len(x)
    membership: dict[int, str] = {}
    for label, positions, digest, count in (
        (
            "train",
            train_pos,
            split_spec["train_positions_sha256"],
            split_spec["train_count"],
        ),
        (
            "validation",
            valid_pos,
            split_spec["validation_positions_sha256"],
            split_spec["validation_count"],
        ),
    ):
        if (
            len(positions) != count
            or hashlib.sha256(positions.tobytes()).hexdigest() != digest
        ):
            raise ValueError("registered_split_membership_mismatch")
        for row in set(map(int, replay[positions])):
            if row in membership:
                raise ValueError("partition_split_overlap")
            membership[row] = label
    if set(membership) != set(range(n)):
        raise ValueError("partition_split_incomplete")

    rows: list[dict[str, Any]] = []
    row_index = 0
    source_rows: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for source, record in enumerate(records):
        source_rows[record["name"]] = []
        with paths[source].open(encoding="utf-8") as stream:
            for line_no, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                raw = json.loads(text)
                source_rows[record["name"]].append((line_no, raw))
                raw_coeff = float(coeff[row_index])
                target = np.asarray(policies[row_index], dtype=np.float32)
                # Retain exact loader bytes and the target values as serialized float32.
                input_bytes = np.asarray(x[row_index], dtype="<f4").tobytes()
                stones = int(np.rint(x[row_index, :12].astype(np.float64) * 48).sum())
                mult = int(record["weight"])
                item = {
                    "source": record["name"],
                    "line": line_no,
                    "compact_row": row_index,
                    "replay_multiplicity": mult,
                    "policy_coefficient": raw_coeff,
                    "partition": membership[row_index],
                    "active_stones": stones,
                    "canonical_state": canonical_identity_from_encoded_state(
                        x[row_index].tolist()
                    ),
                    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                    "input_hex": input_bytes.hex(),
                    "target": target.tolist(),
                    "target_sum": float(target.astype(np.float64).sum()),
                    "exposure_weight": mult * raw_coeff,
                    "equal_input_weight": raw_coeff,
                }
                rows.append(item)
                row_index += 1
    if row_index != n:
        raise ValueError("loader_source_row_accounting_mismatch")
    normalized, _normalization = normalize_policy_coefficients(
        coeff,
        [row["canonical_state"] for row in rows],
        np.asarray([row["active_stones"] for row in rows]),
        [row["source"] for row in rows],
        np.asarray([row["partition"] == "train" for row in rows]),
    )
    for row, treatment_coefficient in zip(rows, normalized, strict=True):
        row["treatment_coefficient"] = float(treatment_coefficient)
        row["treatment_exposure_weight"] = row["replay_multiplicity"] * float(
            treatment_coefficient
        )
        row["treatment_equal_input_weight"] = float(treatment_coefficient)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "row-accounting.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["partition"], row["input_sha256"])].append(row)
    with gzip.open(out / "group-accounting.jsonl.gz", "wt", encoding="utf-8") as stream:
        for (partition, identity), members in sorted(groups.items()):
            group_record = {
                "partition": partition,
                "input_sha256": identity,
                "canonical_states": sorted({r["canonical_state"] for r in members}),
                "compact_rows": [r["compact_row"] for r in members],
                "source_lines": [[r["source"], r["line"]] for r in members],
                "replay_multiplicities": [r["replay_multiplicity"] for r in members],
            }
            stream.write(json.dumps(group_record, separators=(",", ":")) + "\n")
    inputs = [
        "ml/alphazero_lite/train.py",
        "ml/alphazero_lite/seed429_policy_normalization.py",
        "ml/alphazero_lite/seed432_policy_target_census.py",
        "ml/alphazero_lite/verify_seed432.py",
    ]
    manifest = {
        "schema": "seed432-census-manifest-v1",
        "protocol": "exact-input byte groups; no cross-partition merge; source target rows are loader outputs",
        "registration_sha256": sha256_file(reg_path),
        "split_sha256": sha256_file(split_path),
        "replays": [
            {
                "name": r["name"],
                "weight": r["weight"],
                "sha256": sha256_file(p),
                "value_target_mode": r["value_target_mode"],
            }
            for r, p in zip(records, paths, strict=True)
        ],
        "execution_sources": {p: sha256_file(root / p) for p in inputs},
        "policy_target_mode": "sharpened",
        "loader": "train.load_jsonl_replay; include_policy_loss_weights=True",
        "split": split_spec,
        "row_accounting_sha256": sha256_file(out / "row-accounting.jsonl.gz"),
        "group_accounting_sha256": sha256_file(out / "group-accounting.jsonl.gz"),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    populations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stone = (
            "le16"
            if row["active_stones"] <= 16
            else "17-32"
            if row["active_stones"] <= 32
            else "gt32"
        )
        populations[f"{row['partition']}/{stone}"].append(row)
    results = {
        name: {
            weighting: summarize(population, weighting)
            for weighting in ("exposure", "equal_input")
        }
        | {
            f"seed429_treatment_{weighting}": summarize(
                population, weighting, treatment=True
            )
            for weighting in ("exposure", "equal_input")
        }
        for name, population in populations.items()
    }
    primary = results["train/gt32"]
    decision = (
        "material_policy_target_disagreement"
        if all(
            primary[w]["mean_js_nats"] >= THRESHOLDS["mean_js_nats"]
            and primary[w]["weighted_pairwise_top_action_disagreement"]
            >= THRESHOLDS["weighted_top_action_disagreement"]
            for w in primary
        )
        else "target_disagreement_not_material_under_registered_thresholds"
    )
    result = {
        "schema": "seed432-census-results-v1",
        "compact_rows": n,
        "expanded_positions": len(replay),
        "training_positions": len(train_pos),
        "validation_positions": len(valid_pos),
        "target_sum_accounting": {
            "maximum_absolute_deviation_from_one": max(
                (abs(row["target_sum"] - 1.0) for row in rows), default=0.0
            ),
            "rows_outside_loader_tolerance_1e-5": sum(
                abs(row["target_sum"] - 1.0) > 1e-5 for row in rows
            ),
            "zero_coefficient_rows": sum(
                row["policy_coefficient"] == 0 for row in rows
            ),
            "genuine_duplicate_compact_rows": sum(
                max(0, len(group) - 1) for group in groups.values()
            ),
            "expanded_positions_are_replay_copies_not_new_labels": True,
        },
        "populations": results,
        "thresholds": THRESHOLDS,
        "classification": decision,
        "gradient_identity": "For each fixed logits vector z and preserved total coefficient, sum_i q_i·log_softmax(z)= (sum_i q_i)·(sum_i q_i/sum_i q_i)·log_softmax(z); hence averaging targets preserves cross-entropy and its gradient exactly in real arithmetic.",
    }
    (out / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
