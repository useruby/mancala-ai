#!/usr/bin/env python3
"""Read-only audit of the successful seed455 and failed seed461 fresh replays.

This intentionally consumes the frozen replay and arena-localization artifacts only.
It neither invokes self-play nor creates an evaluator game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.arena_conflict_state_localization import (
    active_bucket,
    replay_row_state,
    state_descriptor,
)
from ml.alphazero_lite.forensic_suite import canonical_state_key
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "docs/data/alphazero-lite-replay-source-attribution"
OUT_DIR = ROOT / "docs/data/seed455-vs-seed461-fresh-selfplay-distribution"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 372
DATASETS = {
    "seed455": {
        "path": ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1/self_play.jsonl",
        "sha256": "0628b7212e6a141138f1618250506d5b506e45fa0fd8c5b614917825ebffcd22",
        "seeds": (454, 455, 456),
        "parent": ROOT / ".tmp/azlite-promotion-rehearsal-current",
    },
    "seed461": {
        "path": ROOT
        / ".tmp/seed455-nextgen-s461-default-value-root16/runs/seed455-nextgen-s461-default-value-root16-iter1/self_play.jsonl",
        "sha256": "97bc0b031eb041a4582e67651b0f39d63d04e8a884c6e955e217b7bbb29dc3ab",
        "seeds": (460, 461, 462),
        "parent": ROOT
        / ".tmp/seed48-nextgen-s455-default-value/runs/seed48-nextgen-s455-default-value-iter1",
    },
}
PLY_BUCKETS = (("0-4", 0, 4), ("5-8", 5, 8), ("9-12", 9, 12), (">12", 13, 9999))
ACTIVE_BUCKETS = (">40", "33-40", "25-32", "22-24", "17-21", "<=16")


class AuditError(ValueError):
    """Raised when a frozen audit input is unavailable or malformed."""


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of an immutable replay artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy(values: Iterable[float]) -> float:
    """Return Shannon entropy in bits."""
    return -sum(value * math.log2(value) for value in values if value > 0.0)


def js(left: list[float], right: list[float]) -> float:
    """Return Jensen-Shannon divergence in bits for aligned distributions."""
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]
    return sum(
        0.5 * value * math.log2(value / mean)
        for policy, mean_policy in ((left, midpoint), (right, midpoint))
        for value, mean in zip(policy, mean_policy, strict=True)
        if value > 0.0
    )


def top_action(policy: list[float]) -> int:
    """Choose a deterministic top action."""
    return min(range(len(policy)), key=lambda action: (-policy[action], action))


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic nearest-rank-like quantile."""
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0.0


def simpson_effective(counts: Iterable[int]) -> float:
    """Return the inverse Simpson effective count."""
    values = list(counts)
    total = sum(values)
    return 0.0 if not total else 1.0 / sum((count / total) ** 2 for count in values)


def ply_bucket(ply: int) -> str:
    """Map a move index into a pre-registered ply bucket."""
    for label, low, high in PLY_BUCKETS:
        if low <= ply <= high:
            return label
    raise ValueError(ply)


def source_seed(game_index: int, seeds: tuple[int, ...]) -> int:
    """Recover the seed-pool assignment used by self_play.py."""
    return seeds[game_index % len(seeds)]


def policy_summary(policy: list[float]) -> dict[str, float]:
    """Summarize one legal-masked six-action policy."""
    ordered = sorted(policy, reverse=True)
    return {
        "entropy": entropy(policy),
        "top1": ordered[0] if ordered else 0.0,
        "top2_gap": (ordered[0] - ordered[1])
        if len(ordered) > 1
        else (ordered[0] if ordered else 0.0),
        "mass_gt_005": float(sum(value > 0.05 for value in policy)),
    }


def validate_row(row: dict[str, Any]) -> None:
    """Validate the replay fields required by this audit."""
    required = (
        "state",
        "policy",
        "value",
        "player",
        "move_index",
        "winner",
        "game_index",
    )
    if any(name not in row for name in required):
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    state = replay_row_state(row)
    game = KalahGame.from_state(state)
    policy = [float(value) for value in row["policy"]]
    if len(policy) != 6 or not -1.0 <= float(row["value"]) <= 1.0:
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    legal = set(game.possible_moves())
    if any(value < -1e-8 for value in policy) or abs(sum(policy) - 1.0) > 1e-5:
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    if any(policy[action] > 1e-8 for action in set(range(6)) - legal):
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    if int(row["player"]) != game.current_player:
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")


def load_dataset(
    name: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load, hash-check, validate, and group one complete fresh replay."""
    config = DATASETS[name]
    path = config["path"]
    if not path.is_file() or sha256_file(path) != config["sha256"]:
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    rows: list[dict[str, Any]] = []
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row)
            state = replay_row_state(row)
            row["_state"] = state
            row["_key"] = canonical_state_key(state)
            row["_active"] = sum(KalahGame.from_state(state).pits)
            row["_source_seed"] = source_seed(int(row["game_index"]), config["seeds"])
            rows.append(row)
            games[str(row["game_index"])].append(row)
    if len(games) != 1600 or any(
        not game[0].get("game_completed") for game in games.values()
    ):
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    for game in games.values():
        game.sort(key=lambda row: int(row["move_index"]))
        if [int(row["move_index"]) for row in game] != list(range(len(game))):
            raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    return rows, games


def mean_summary(values: Iterable[float]) -> dict[str, float]:
    """Summarize row-level quantities descriptively."""
    materialized = list(values)
    return {"mean": statistics.fmean(materialized) if materialized else 0.0}


def bootstrap_game_mean(
    games: dict[str, list[dict[str, Any]]], metric: str
) -> dict[str, float | int]:
    """Bootstrap a game-level mean, never resampling correlated rows."""
    values = [
        statistics.fmean(float(row[metric]) for row in game) for game in games.values()
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "lower_95": percentile(draws, 0.025),
        "upper_95": percentile(draws, 0.975),
    }


def diversity(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return canonical-state concentration metrics."""
    counts = Counter(row["_key"] for row in rows)
    total = len(rows)
    ordered = sorted(counts.values(), reverse=True)
    return {
        "rows": total,
        "unique_states": len(counts),
        "unique_state_fraction": len(counts) / total if total else 0.0,
        "shannon_entropy": entropy([count / total for count in counts.values()])
        if total
        else 0.0,
        "simpson_effective_states": simpson_effective(counts.values()),
        "largest_state_frequency": ordered[0] / total if ordered else 0.0,
        "top5_state_mass": sum(ordered[:5]) / total if total else 0.0,
        "top10_state_mass": sum(ordered[:10]) / total if total else 0.0,
    }


def coverage(reference: set[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return exact canonical-state coverage against a frozen reference set."""
    occurrences = Counter(row["_key"] for row in rows)
    found = reference & set(occurrences)
    counts = [occurrences[key] for key in found]
    return {
        "covered_states": len(found),
        "coverage_fraction": len(found) / len(reference) if reference else 0.0,
        "total_occurrences": sum(counts),
        "median_occurrences": percentile(counts, 0.5),
        "p90_occurrences": percentile(counts, 0.9),
        "keys": sorted(found),
    }


def weighted_coverage(
    reference: set[str], rows: list[dict[str, Any]], weights: dict[str, float]
) -> dict[str, float]:
    """Return exact coverage weighted only by frozen absolute arena deltas."""
    present = {row["_key"] for row in rows}
    total = sum(weights.get(key, 0.0) for key in reference)
    covered = sum(weights.get(key, 0.0) for key in reference & present)
    return {
        "covered_weight": covered,
        "total_weight": total,
        "coverage_fraction": covered / total if total else 0.0,
    }


def categorical_js(left: Counter[str], right: Counter[str]) -> float:
    """Compare categorical descriptor distributions with JS divergence."""
    keys = sorted(set(left) | set(right))
    return (
        js(
            [left[key] / sum(left.values()) for key in keys],
            [right[key] / sum(right.values()) for key in keys],
        )
        if keys and sum(left.values()) and sum(right.values())
        else 0.0
    )


def descriptor_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only the fixed, interpretable descriptor family."""
    descriptors = [
        state_descriptor(row["_state"] if "_state" in row else row["state"])
        | {"ply_bucket": ply_bucket(int(row["move_index"]))}
        for row in rows
    ]
    categorical = (
        "active_stone_bucket",
        "ply_bucket",
        "legal_action_count",
        "capture_available",
        "extra_turn_available",
        "current_player",
    )
    numeric = ("active_stones", "legal_action_count")
    return {
        "count": len(descriptors),
        "categorical": {
            name: dict(Counter(str(value[name]) for value in descriptors))
            for name in categorical
        },
        "numeric": {
            name: {
                "mean": statistics.fmean(float(value[name]) for value in descriptors)
                if descriptors
                else 0.0,
                "sd": statistics.pstdev(float(value[name]) for value in descriptors)
                if len(descriptors) > 1
                else 0.0,
            }
            for name in numeric
        },
        "cross_tabs": {
            "active_stones_x_ply": dict(
                Counter(
                    f"{value['active_stone_bucket']}|{value['ply_bucket']}"
                    for value in descriptors
                )
            ),
            "active_stones_x_legal_action_count": dict(
                Counter(
                    f"{value['active_stone_bucket']}|{value['legal_action_count']}"
                    for value in descriptors
                )
            ),
        },
    }


def descriptor_comparison(
    arena: list[dict[str, Any]], left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare both replays with frozen arena states without a learned model."""
    source = {
        "arena": descriptor_distribution(arena),
        "seed455": descriptor_distribution(left),
        "seed461": descriptor_distribution(right),
    }
    comparisons: dict[str, Any] = {}
    for name in (
        "active_stone_bucket",
        "ply_bucket",
        "legal_action_count",
        "capture_available",
        "extra_turn_available",
        "current_player",
    ):
        comparisons[name] = {
            dataset: categorical_js(
                Counter(source["arena"]["categorical"][name]),
                Counter(source[dataset]["categorical"][name]),
            )
            for dataset in ("seed455", "seed461")
        }
    for name in ("active_stones", "legal_action_count"):
        comparisons[name] = {}
        for dataset in ("seed455", "seed461"):
            arena_stats, dataset_stats = (
                source["arena"]["numeric"][name],
                source[dataset]["numeric"][name],
            )
            denominator = math.sqrt(
                (arena_stats["sd"] ** 2 + dataset_stats["sd"] ** 2) / 2.0
            )
            comparisons[name][dataset] = (
                0.0
                if not denominator
                else (dataset_stats["mean"] - arena_stats["mean"]) / denominator
            )
    return {"distributions": source, "comparisons": comparisons}


def frozen_states(filename: str) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Read a frozen divergence set without redefining it."""
    rows = json.loads((ARTIFACT_DIR / filename).read_text())
    state_rows = [row for row in rows if isinstance(row, dict) and "state" in row]
    if not state_rows:
        raise AuditError("fresh_selfplay_distribution_artifact_unavailable")
    by_key = {canonical_state_key(row["state"]): row for row in state_rows}
    return set(by_key), by_key


def raw_policy(evaluator: ArtifactEvaluator, row: dict[str, Any]) -> list[float]:
    """Evaluate and legal-mask the respective frozen parent policy."""
    policy, _value = evaluator.evaluate(KalahGame.from_state(row["_state"]))
    legal = set(KalahGame.from_state(row["_state"]).possible_moves())
    values = [float(policy[action]) if action in legal else 0.0 for action in range(6)]
    total = sum(values)
    return [value / total for value in values] if total else values


def raw_search_lift(rows: list[dict[str, Any]], parent: Path) -> dict[str, Any]:
    """Measure raw-parent to stored-target correction, caching duplicate states."""
    evaluator = ArtifactEvaluator(parent)
    cache: dict[str, list[float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["_active"] > 32:
            grouped[">32_all"].append(row)
            grouped[
                f"{ply_bucket(int(row['move_index']))}|{active_bucket(row['_active'])}"
            ].append(row)
    result: dict[str, Any] = {}
    for label, group in grouped.items():
        values: list[dict[str, float]] = []
        for row in group:
            raw = cache.setdefault(row["_key"], raw_policy(evaluator, row))
            target = [float(value) for value in row["policy"]]
            raw_top, target_top = top_action(raw), top_action(target)
            values.append(
                {
                    "js": js(raw, target),
                    "argmax_agreement": float(raw_top == target_top),
                    "target_on_raw_argmax": target[raw_top],
                    "raw_on_target_argmax": raw[target_top],
                    "entropy_delta": entropy(target) - entropy(raw),
                }
            )
        result[label] = (
            {name: statistics.fmean(row[name] for row in values) for name in values[0]}
            if values
            else {}
        )
    return result


def aggregate_targets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean duplicate targets deterministically by canonical state."""
    sums: dict[str, list[float]] = {}
    values: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for row in rows:
        if row["_active"] > 32 and int(row["move_index"]) <= 12:
            sums.setdefault(row["_key"], [0.0] * 6)
            sums[row["_key"]] = [
                a + float(b)
                for a, b in zip(sums[row["_key"]], row["policy"], strict=True)
            ]
            values[row["_key"]] += float(row["value"])
            counts[row["_key"]] += 1
    return {
        key: {
            "policy": [value / counts[key] for value in policy],
            "value": values[key] / counts[key],
            "count": counts[key],
        }
        for key, policy in sums.items()
    }


def target_drift(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare paired mean targets for shared high-stone opening states."""
    shared = sorted(set(left) & set(right))
    rows = [
        {
            "state_id": key,
            "target_js": js(left[key]["policy"], right[key]["policy"]),
            "argmax_change": top_action(left[key]["policy"])
            != top_action(right[key]["policy"]),
            "top1_mass_delta": max(right[key]["policy"]) - max(left[key]["policy"]),
            "entropy_delta": entropy(right[key]["policy"])
            - entropy(left[key]["policy"]),
            "value_delta": right[key]["value"] - left[key]["value"],
            "seed455_target": left[key]["policy"],
            "seed461_target": right[key]["policy"],
            "seed455_occurrences": left[key]["count"],
            "seed461_occurrences": right[key]["count"],
        }
        for key in shared
    ]
    summary: dict[str, Any] = {"shared_states": len(rows)}
    if rows:
        for name in ("target_js", "top1_mass_delta", "entropy_delta", "value_delta"):
            summary[name] = statistics.fmean(float(row[name]) for row in rows)
        summary["argmax_change_fraction"] = statistics.fmean(
            float(row["argmax_change"]) for row in rows
        )
    return summary, rows


def dataset_summary(
    rows: list[dict[str, Any]], games: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Build game, phase, diversity, and exact-root summaries for one replay."""
    lengths = [len(game) for game in games.values()]
    winners: Counter[str] = Counter(str(game[0]["winner"]) for game in games.values())
    by_ply = {
        str(ply): diversity([row for row in rows if int(row["move_index"]) == ply])
        for ply in range(13)
    }
    opening = {
        label: diversity([row for row in rows if low <= int(row["move_index"]) <= high])
        for label, low, high in PLY_BUCKETS[:3]
    }
    active = Counter(active_bucket(row["_active"]) for row in rows)
    ply_counts = Counter(ply_bucket(int(row["move_index"])) for row in rows)
    high = [row for row in rows if row["_active"] > 32]
    high_open = [row for row in high if int(row["move_index"]) <= 12]
    seed_summary = {}
    for seed in sorted({row["_source_seed"] for row in rows}):
        seed_rows = [row for row in rows if row["_source_seed"] == seed]
        seed_high_open = [
            row
            for row in seed_rows
            if row["_active"] > 32 and int(row["move_index"]) <= 12
        ]
        seed_summary[str(seed)] = {
            "rows": len(seed_rows),
            "highstone_row_share": sum(row["_active"] > 32 for row in seed_rows)
            / len(seed_rows),
            "opening_diversity": diversity(seed_high_open),
        }
    return {
        "rows": len(rows),
        "games": len(games),
        "game_length": {
            "mean": statistics.fmean(lengths),
            "median": percentile(lengths, 0.5),
            "p10": percentile(lengths, 0.1),
            "p90": percentile(lengths, 0.9),
        },
        "winner_distribution": dict(winners),
        "draw_rate": winners.get("None", 0) / len(games),
        "player0_win_rate": winners.get("0", 0) / len(games),
        "player1_win_rate": winners.get("1", 0) / len(games),
        "rows_by_active_bucket": dict(active),
        "rows_by_ply_bucket": dict(ply_counts),
        "opening_diversity_by_ply": by_ply,
        "opening_diversity_by_bucket": opening,
        "highstone": {
            "rows": len(high),
            "row_share": len(high) / len(rows),
            "opening": diversity(high_open),
            "mean_target_entropy": statistics.fmean(
                entropy(row["policy"]) for row in high
            ),
        },
        "per_source_seed": seed_summary,
        "exact_root": {
            "rows_at_or_below_16": sum(row["_active"] <= 16 for row in rows),
            "handoffs": sum(
                row.get("teacher_source") == "exact_root_tablebase" for row in rows
            ),
        },
        "game_level_bootstrap": {"value": bootstrap_game_mean(games, "value")},
    }


def classify(
    left: dict[str, Any],
    right: dict[str, Any],
    divergence: dict[str, Any],
    lift: dict[str, Any],
    drift: dict[str, Any],
) -> str:
    """Apply the pre-registered classification hierarchy descriptively."""
    left_open, right_open = left["highstone"]["opening"], right["highstone"]["opening"]
    coverage_drop = (
        divergence["seed461"]["coverage_fraction"]
        < divergence["seed455"]["coverage_fraction"]
    )
    diversity_drop = (
        right_open["simpson_effective_states"] < left_open["simpson_effective_states"]
        and right_open["top10_state_mass"] > left_open["top10_state_mass"]
    )
    if coverage_drop and diversity_drop:
        return "nplus2_opening_diversity_collapse"
    if drift.get("target_js", 0.0) > 0.1:
        return "nplus2_teacher_policy_drift_primary"
    return "nplus2_fresh_distribution_no_clear_regression"


def run(out_dir: Path = OUT_DIR) -> dict[str, Any]:
    """Execute the complete read-only fresh-generation distribution audit."""
    loaded = {name: load_dataset(name) for name in DATASETS}
    summaries = {name: dataset_summary(*loaded[name]) for name in DATASETS}
    first_keys, first_rows = frozen_states("first-divergences.json")
    outcome_keys, _outcome_rows = frozen_states("outcome-associated-divergences.json")
    first_coverage = {name: coverage(first_keys, loaded[name][0]) for name in DATASETS}
    outcome_coverage = {
        name: coverage(outcome_keys, loaded[name][0]) for name in DATASETS
    }
    pair_deltas = {
        (row["arm"], int(row["opening_id"])): abs(float(row["pair_score_delta"]))
        for row in json.loads(
            (ARTIFACT_DIR / "opening-pair-contributions.json").read_text()
        )
    }
    first_weights = {
        key: pair_deltas.get((str(row["arm"]), int(row["opening_id"])), 0.0)
        for key, row in first_rows.items()
    }
    weighted_first_coverage = {
        name: weighted_coverage(first_keys, loaded[name][0], first_weights)
        for name in DATASETS
    }
    targets = {name: aggregate_targets(loaded[name][0]) for name in DATASETS}
    drift, drift_rows = target_drift(targets["seed455"], targets["seed461"])
    overlap = {}
    for subset, predicate in {
        "global": lambda row: True,
        "highstone": lambda row: row["_active"] > 32,
        **{
            f"ply_{ply}": lambda row, ply=ply: int(row["move_index"]) == ply
            for ply in range(13)
        },
    }.items():
        left_keys = {row["_key"] for row in loaded["seed455"][0] if predicate(row)}
        right_keys = {row["_key"] for row in loaded["seed461"][0] if predicate(row)}
        overlap[subset] = {
            "seed455_unique": len(left_keys),
            "seed461_unique": len(right_keys),
            "intersection": len(left_keys & right_keys),
            "seed455_only": len(left_keys - right_keys),
            "seed461_only": len(right_keys - left_keys),
            "jaccard": len(left_keys & right_keys) / len(left_keys | right_keys)
            if left_keys | right_keys
            else 0.0,
        }
    lift = {
        name: raw_search_lift(loaded[name][0], DATASETS[name]["parent"])
        for name in DATASETS
    }
    descriptors = descriptor_comparison(
        [
            {"state": row["state"], "move_index": row["ply"]}
            for row in first_rows.values()
        ],
        loaded["seed455"][0],
        loaded["seed461"][0],
    )
    classification = classify(
        summaries["seed455"], summaries["seed461"], first_coverage, lift, drift
    )
    payload = {
        "schema": "azlite_seed455_vs_seed461_fresh_distribution_v1",
        "semantic_identity": "seed455-vs-seed461-fresh-selfplay-distribution",
        "arena_localization_commit": "7bf67be12f55e614ee7fc30062a7e789ae715540",
        "datasets": {
            name: {
                "sha256": config["sha256"],
                "path": str(config["path"].relative_to(ROOT)),
            }
            for name, config in DATASETS.items()
        },
        "summaries": summaries,
        "first_divergence_coverage": first_coverage,
        "outcome_associated_coverage": outcome_coverage,
        "arena_delta_weighted_first_divergence_coverage": weighted_first_coverage,
        "state_overlap": overlap,
        "shared_state_target_drift": drift,
        "raw_to_search_lift": lift,
        "descriptor_comparison": descriptors,
        "classification": classification,
        "scope": {
            "training_runs": 0,
            "self_play_games": 0,
            "arena_games": 0,
            "canonical_games": 0,
            "promotions": 0,
        },
    }
    for entry in first_coverage.values():
        entry.pop("keys")
    for entry in outcome_coverage.values():
        entry.pop("keys")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final-classification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    arena_drift_rows = [
        row
        | {
            "active_stones": first_rows[row["state_id"]]["active_pit_stones"],
            "ply": first_rows[row["state_id"]]["ply"],
        }
        for row in drift_rows
        if row["state_id"] in first_keys
    ]
    (out_dir / "shared-state-target-drift.json").write_text(
        json.dumps(arena_drift_rows, indent=2) + "\n"
    )
    compact = {
        "game-level-distribution-summary.json": summaries,
        "opening-diversity-curves.json": {
            name: summary["opening_diversity_by_ply"]
            for name, summary in summaries.items()
        },
        "arena-divergence-coverage.json": {
            "first": first_coverage,
            "outcome": outcome_coverage,
            "weighted": weighted_first_coverage,
        },
        "arena-descriptor-comparison.json": descriptors,
        "per-seed-summary.json": {
            name: summary["per_source_seed"] for name, summary in summaries.items()
        },
    }
    for filename, content in compact.items():
        (out_dir / filename).write_text(
            json.dumps(content, indent=2, sort_keys=True) + "\n"
        )
    return payload


def main() -> None:
    """Run the audit from its command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.out_dir)["classification"]))


if __name__ == "__main__":
    main()
