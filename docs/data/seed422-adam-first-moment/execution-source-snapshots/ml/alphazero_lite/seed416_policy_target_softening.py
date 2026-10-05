"""Immutable data transformation and pure analysis for the seed416 ablation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.alphazero_lite.train import derive_legal_moves_from_encoded_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_exact_root(row: dict[str, Any]) -> bool:
    return row.get("teacher_source") == "exact_root_tablebase" or str(
        row.get("policy_target_actual_mode", "")
    ).startswith("exact_root_")


def softened_policy(policy: list[float]) -> list[float]:
    values = np.asarray(policy, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid_policy_target")
    support = values > 0
    result = np.zeros_like(values)
    if not support.any():
        raise ValueError("invalid_policy_target")
    roots = np.sqrt(values[support])
    result[support] = roots / roots.sum()
    return result.tolist()


def entropy(policy: list[float]) -> float:
    return -sum(float(p) * math.log(float(p)) for p in policy if p > 0)


def transform_row(row: dict[str, Any], lane: str) -> tuple[dict[str, Any], bool]:
    if lane not in {"A", "B"}:
        raise ValueError("lane_must_be_A_or_B")
    source = row["policy"]
    legal = derive_legal_moves_from_encoded_state(row["state"])
    if legal is not None and any(
        source[index] > 1e-7 for index in range(6) if index not in legal
    ):
        raise ValueError("source_policy_assigns_illegal_mass")
    changed = lane == "B" and not is_exact_root(row)
    result = dict(row)
    result["policy"] = softened_policy(source) if changed else list(source)
    if "stored_policy_target" in row and row["stored_policy_target"] == source:
        result["stored_policy_target"] = list(result["policy"])
    if result["policy"] != source:
        provenance = dict(row.get("policy_transform_provenance", {}))
        provenance.update(
            {
                "experiment": "seed416-policy-target-softening-v1",
                "transform": "q(a)=sqrt(p(a))/sum(sqrt(p))",
                "source_semantics": "transformed stored policy distribution; not recovered visit counts",
                "lane": lane,
            }
        )
        result["policy_transform_provenance"] = provenance
    if result["state"] != row["state"] or result["value"] != row["value"]:
        raise AssertionError("immutable_row_fields_changed")
    if legal is not None and any(
        result["policy"][i] > 1e-7 for i in range(6) if i not in legal
    ):
        raise ValueError("transformed_policy_assigns_illegal_mass")
    if not np.isclose(sum(result["policy"]), 1.0, atol=1e-6):
        raise ValueError("transformed_policy_not_normalized")
    if any((a == 0) != (b == 0) for a, b in zip(source, result["policy"], strict=True)):
        raise ValueError("policy_support_changed")
    return result, result["policy"] != source


def transform_jsonl(source: Path, destination: Path, lane: str) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"immutable_derivative_exists:{destination}")
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "changed_rows": 0,
            "source_entropy": [],
            "target_entropy": [],
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8") as src,
        destination.open("x", encoding="utf-8") as dst,
    ):
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            original = json.loads(line)
            transformed, changed = transform_row(original, lane)
            move_index = int(original.get("move_index", 0))
            phase = "early" if move_index < 10 else "mid" if move_index < 30 else "late"
            bucket = counts[f"{original.get('bucket', 'unknown')}:{phase}"]
            bucket["rows"] += 1
            bucket["changed_rows"] += int(changed)
            bucket["source_entropy"].append(entropy(original["policy"]))
            bucket["target_entropy"].append(entropy(transformed["policy"]))
            dst.write(
                json.dumps(transformed, separators=(",", ":"), ensure_ascii=True) + "\n"
            )
    summaries = {
        key: {
            "rows": value["rows"],
            "changed_rows": value["changed_rows"],
            "mean_source_entropy_nats": float(np.mean(value["source_entropy"])),
            "mean_target_entropy_nats": float(np.mean(value["target_entropy"])),
            "mean_entropy_delta_nats": float(
                np.mean(value["target_entropy"]) - np.mean(value["source_entropy"])
            ),
        }
        for key, value in sorted(counts.items())
    }
    return {
        "lane": lane,
        "source": str(source),
        "source_sha256": sha256(source),
        "derivative": str(destination),
        "derivative_sha256": sha256(destination),
        "rows": sum(item["rows"] for item in counts.values()),
        "changed_rows": sum(item["changed_rows"] for item in counts.values()),
        "by_source_and_phase": summaries,
    }


def audit_jsonl_derivative(source: Path, derivative: Path, lane: str) -> dict[str, Any]:
    """Verify an existing immutable derivative and recompute its summaries."""
    if not derivative.is_file():
        raise FileNotFoundError(derivative)
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "changed_rows": 0,
            "source_entropy": [],
            "target_entropy": [],
        }
    )
    with source.open(encoding="utf-8") as src, derivative.open(encoding="utf-8") as dst:
        for index, (source_line, derivative_line) in enumerate(
            zip(src, dst, strict=True), 1
        ):
            if not source_line.strip() or not derivative_line.strip():
                if source_line.strip() != derivative_line.strip():
                    raise ValueError(f"derivative_blank_row_mismatch:{index}")
                continue
            original, actual = json.loads(source_line), json.loads(derivative_line)
            expected, changed = transform_row(original, lane)
            if actual != expected:
                raise ValueError(f"derivative_row_mismatch:{lane}:{index}")
            move_index = int(original.get("move_index", 0))
            phase = "early" if move_index < 10 else "mid" if move_index < 30 else "late"
            bucket = counts[f"{original.get('bucket', 'unknown')}:{phase}"]
            bucket["rows"] += 1
            bucket["changed_rows"] += int(changed)
            bucket["source_entropy"].append(entropy(original["policy"]))
            bucket["target_entropy"].append(entropy(actual["policy"]))
    summaries = {
        key: {
            "rows": value["rows"],
            "changed_rows": value["changed_rows"],
            "mean_source_entropy_nats": float(np.mean(value["source_entropy"])),
            "mean_target_entropy_nats": float(np.mean(value["target_entropy"])),
            "mean_entropy_delta_nats": float(
                np.mean(value["target_entropy"]) - np.mean(value["source_entropy"])
            ),
        }
        for key, value in sorted(counts.items())
    }
    return {
        "lane": lane,
        "source": str(source),
        "source_sha256": sha256(source),
        "derivative": str(derivative),
        "derivative_sha256": sha256(derivative),
        "rows": sum(item["rows"] for item in counts.values()),
        "changed_rows": sum(item["changed_rows"] for item in counts.values()),
        "by_source_and_phase": summaries,
    }


def paired_opening_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    seats: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        key = (str(row["lane"]), str(row["opening_id"]))
        grouped[key].append(float(row["opponent_score"]))
        game = row.get("game", {})
        seat = row.get("challenger_player", game.get("challenger_player"))
        if seat is not None:
            seats[key].add(int(seat))
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (lane, opening), scores in grouped.items():
        if len(scores) != 2:
            raise ValueError(f"opening_seat_pair_incomplete:{lane}:{opening}")
        if (lane, opening) in seats and seats[(lane, opening)] != {0, 1}:
            raise ValueError(f"opening_seat_pair_invalid:{lane}:{opening}")
        result[lane][opening] = sum(scores) / 2
    return dict(result)


def resume_action(previous: dict[str, Any] | None, observed: dict[str, Any]) -> str:
    """Return run/skip for a lane while rejecting changed completed evidence."""
    if previous is None:
        return "run"
    if previous != observed:
        raise ValueError("resumed_lane_evidence_mismatch")
    return "skip"


def bootstrap_paired(
    rows: list[dict[str, Any]], *, samples: int = 10_000, seed: int = 416
) -> dict[str, Any]:
    scores = paired_opening_scores(rows)
    if set(scores) != {"A", "B"} or set(scores["A"]) != set(scores["B"]):
        raise ValueError("lane_opening_matrix_mismatch")
    opening_ids = sorted(scores["A"])
    if len(opening_ids) != 512:
        raise ValueError(f"opening_count_mismatch:{len(opening_ids)}")
    delta = np.asarray([scores["B"][key] - scores["A"][key] for key in opening_ids])
    b = np.asarray([scores["B"][key] for key in opening_ids])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(opening_ids), size=(samples, len(opening_ids)))
    delta_boot = delta[indexes].mean(axis=1)
    b_boot = b[indexes].mean(axis=1)

    def interval(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    mean_delta = float(delta.mean())
    mean_b = float(b.mean())
    delta_ci, b_ci = interval(delta_boot), interval(b_boot)
    advance = (
        mean_delta >= 0.03 and delta_ci[0] > 0 and mean_b >= 0.53 and b_ci[0] > 0.50
    )
    return {
        "schema": "seed416-policy-target-softening-analysis-v1",
        "opening_clusters": len(opening_ids),
        "samples": samples,
        "seed": seed,
        "primary_mean_B_minus_A": mean_delta,
        "primary_95_percentile_interval": delta_ci,
        "B_opponent_score_mean": mean_b,
        "B_opponent_score_95_percentile_interval": b_ci,
        "decision": "advance_to_separate_confirmation"
        if advance
        else "retain_baseline",
        "inference": "conditional on this dataset and training seed",
        "per_opening": {
            key: {
                "A": scores["A"][key],
                "B": scores["B"][key],
                "B_minus_A": scores["B"][key] - scores["A"][key],
            }
            for key in opening_ids
        },
    }
