#!/usr/bin/env python3
"""Seat-aware arena metrics, classification, and re-ranking for AlphaZero-lite.

Does not train, promote, or overwrite any model.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson_interval(score: float, sample_size: int) -> dict[str, float]:
    z = 1.96
    if sample_size <= 0:
        return {"lower": 0.0, "upper": 0.0}
    denominator = 1.0 + ((z**2) / sample_size)
    center = score + ((z**2) / (2.0 * sample_size))
    margin = (
        z
        * (((score * (1.0 - score)) + ((z**2) / (4.0 * sample_size))) / sample_size)
        ** 0.5
    )
    return {
        "lower": max(0.0, (center - margin) / denominator),
        "upper": min(1.0, (center + margin) / denominator),
    }


def parse_game_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_seat_split_metrics(entries: list[dict]) -> dict:
    p0_wins = p0_losses = p0_draws = 0
    p1_wins = p1_losses = p1_draws = 0
    first_moves_challenger: Counter = Counter()
    first_moves_current: Counter = Counter()
    p0_margins: list[int] = []
    p1_margins: list[int] = []
    p0_lengths: list[int] = []
    p1_lengths: list[int] = []
    trajectory_hashes: list[str] = []

    for ent in entries:
        cp = ent["challenger_player"]
        winner = ent["winner"]
        margin = ent["margin"]
        if cp == 0:
            if winner == "challenger":
                p0_wins += 1
            elif winner == "current":
                p0_losses += 1
            else:
                p0_draws += 1
            p0_margins.append(margin)
            p0_lengths.append(ent["game_length"])
        else:
            if winner == "challenger":
                p1_wins += 1
            elif winner == "current":
                p1_losses += 1
            else:
                p1_draws += 1
            p1_margins.append(margin)
            p1_lengths.append(ent["game_length"])

        if ent.get("first_move_challenger") is not None:
            first_moves_challenger[str(ent["first_move_challenger"])] += 1
        if ent.get("first_move_current") is not None:
            first_moves_current[str(ent["first_move_current"])] += 1
        trajectory_hashes.append(ent["trajectory"])

    p0_total = p0_wins + p0_losses + p0_draws
    p1_total = p1_wins + p1_losses + p1_draws

    traj_counter = Counter(trajectory_hashes)
    duplicate_count = sum(c for c in traj_counter.values() if c > 1)

    p0_score = (p0_wins + 0.5 * p0_draws) / max(p0_total, 1)
    p1_score = (p1_wins + 0.5 * p1_draws) / max(p1_total, 1)

    return {
        "challenger_starts_0": {
            "games": p0_total,
            "wins": p0_wins,
            "losses": p0_losses,
            "draws": p0_draws,
            "score": p0_score,
            "ci95": _wilson_interval(p0_score, p0_total),
            "margin_mean": statistics.fmean(p0_margins) if p0_margins else 0.0,
            "margin_median": statistics.median(p0_margins) if p0_margins else 0.0,
            "game_length_mean": statistics.fmean(p0_lengths) if p0_lengths else 0.0,
            "game_length_median": statistics.median(p0_lengths) if p0_lengths else 0.0,
        },
        "challenger_starts_1": {
            "games": p1_total,
            "wins": p1_wins,
            "losses": p1_losses,
            "draws": p1_draws,
            "score": p1_score,
            "ci95": _wilson_interval(p1_score, p1_total),
            "margin_mean": statistics.fmean(p1_margins) if p1_margins else 0.0,
            "margin_median": statistics.median(p1_margins) if p1_margins else 0.0,
            "game_length_mean": statistics.fmean(p1_lengths) if p1_lengths else 0.0,
            "game_length_median": statistics.median(p1_lengths) if p1_lengths else 0.0,
        },
        "disadvantaged_seat_score": p1_score,
        "first_move_challenger_dist": dict(first_moves_challenger.most_common(6)),
        "first_move_current_dist": dict(first_moves_current.most_common(6)),
        "margin_mean": statistics.fmean(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "margin_median": statistics.median(p0_margins + p1_margins)
        if p0_margins + p1_margins
        else 0.0,
        "game_length_mean": statistics.fmean(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "game_length_median": statistics.median(p0_lengths + p1_lengths)
        if p0_lengths + p1_lengths
        else 0.0,
        "unique_trajectories": len(traj_counter),
        "duplicate_trajectory_count": duplicate_count,
        "total_games": p0_total + p1_total,
    }


STANDARD_BUDGET_PAIRS = [
    (384, 256),
    (1200, 1200),
    (1200, 256),
    (256, 768),
]

BUDGET_PAIR_LABELS = {
    (384, 256): "standard",
    (1200, 1200): "equal_high",
    (1200, 256): "challenger_high",
    (256, 768): "current_high_asymmetry",
}

CANDIDATE_CLASSIFICATIONS = [
    "seat_artifact_only",
    "high_search_breakthrough",
    "standard_budget_breakthrough",
    "regression_masked_by_seat",
    "unclassified",
]


def classify_candidate(
    budget_results: dict[str, dict],
    *,
    standard_alternating_score: float,
) -> str:
    standard = budget_results.get("standard", {})
    equal_high = budget_results.get("equal_high", {})
    challenger_high = budget_results.get("challenger_high", {})
    current_high = budget_results.get("current_high_asymmetry", {})

    std_disadvantaged = float(standard.get("disadvantaged_seat_score", 0.0))
    equal_disadvantaged = float(equal_high.get("disadvantaged_seat_score", 0.0))
    challenger_high_disadvantaged = float(
        challenger_high.get("disadvantaged_seat_score", 0.0)
    )
    current_high_disadvantaged = float(
        current_high.get("disadvantaged_seat_score", 0.0)
    )

    any_breakthrough = (
        equal_disadvantaged > 0.1
        or challenger_high_disadvantaged > 0.1
        or std_disadvantaged > 0.1
    )

    if not any_breakthrough and abs(standard_alternating_score - 0.50) < 1e-6:
        if current_high_disadvantaged <= 0.0 and std_disadvantaged <= 0.0:
            return "seat_artifact_only"

    if std_disadvantaged > 0.1:
        return "standard_budget_breakthrough"

    if equal_disadvantaged > 0.1:
        return "high_search_breakthrough"

    if challenger_high_disadvantaged > 0.1:
        return "high_search_breakthrough"

    if current_high_disadvantaged <= 0.0 and std_disadvantaged <= 0.0:
        return "regression_masked_by_seat"

    return "unclassified"


def build_seat_aware_report(
    *,
    candidate_path: str,
    current_path: str,
    arena_results: list[dict],
    move_time_mean_ms: float | None = None,
    move_time_p95_ms: float | None = None,
) -> dict:
    candidate_sha = sha256_file(Path(candidate_path) / "weights.json")
    current_sha = sha256_file(Path(current_path) / "weights.json")

    budget_results: dict[str, dict] = {}
    standard_alternating_score: float = 0.0

    for result in arena_results:
        budget_label = result.get("budget_label", "unknown")
        result_data = {}
        if "seat_metrics" in result:
            result_data = dict(result["seat_metrics"])
        result_data["arena_score"] = result.get("arena_score", 0.0)
        result_data["arena_wins"] = result.get("arena_wins", 0)
        result_data["arena_losses"] = result.get("arena_losses", 0)
        result_data["arena_draws"] = result.get("arena_draws", 0)
        result_data["move_time_mean_ms"] = result.get("move_time_mean_ms")
        result_data["move_time_p95_ms"] = result.get("move_time_p95_ms")
        budget_results[budget_label] = result_data

        if budget_label == "standard":
            standard_alternating_score = result_data.get("arena_score", 0.0)

    classification = classify_candidate(
        budget_results,
        standard_alternating_score=standard_alternating_score,
    )

    ranking_table = _build_ranking_row(
        candidate_path=candidate_path,
        candidate_sha=candidate_sha,
        current_sha=current_sha,
        budget_results=budget_results,
        classification=classification,
        move_time_mean_ms=move_time_mean_ms,
        move_time_p95_ms=move_time_p95_ms,
    )

    return {
        "schema": "azlite_seat_aware_promotion_gate_v1",
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_sha,
        "current_path": current_path,
        "current_sha256": current_sha,
        "classification": classification,
        "standard_alternating_score": standard_alternating_score,
        "budget_results": budget_results,
        "ranking_table": ranking_table,
    }


def _build_ranking_row(
    *,
    candidate_path: str,
    candidate_sha: str,
    current_sha: str,
    budget_results: dict[str, dict],
    classification: str,
    move_time_mean_ms: float | None = None,
    move_time_p95_ms: float | None = None,
) -> dict:
    def _score(budget_label: str) -> float | None:
        return budget_results.get(budget_label, {}).get("arena_score")

    def _disadvantaged(budget_label: str) -> float | None:
        return budget_results.get(budget_label, {}).get("disadvantaged_seat_score")

    margin_mean = None
    p95 = None
    for budget_result in budget_results.values():
        if "margin_mean" in budget_result:
            margin_mean = budget_result["margin_mean"]
            break
    for budget_result in budget_results.values():
        if budget_result.get("move_time_p95_ms") is not None:
            p95 = budget_result["move_time_p95_ms"]
            break

    candidate_name = str(Path(candidate_path).name)
    if "iter0_candidate" in str(candidate_path):
        candidate_name = "iter0_reference"
    elif "iter1_continue_no_new_data" in str(candidate_path):
        candidate_name = "iter1_continue_no_new_data"
    elif "iter1_candidate_random_replay" in str(candidate_path):
        candidate_name = "iter1_candidate_random_replay"

    return {
        "candidate": candidate_name,
        "artifact_sha256": candidate_sha,
        "current_sha256": current_sha,
        "standard_alternating_score": _score("standard"),
        "standard_disadvantaged_seat_score": _disadvantaged("standard"),
        "equal_high_disadvantaged_seat_score": _disadvantaged("equal_high"),
        "challenger_high_disadvantaged_seat_score": _disadvantaged("challenger_high"),
        "current_high_disadvantaged_seat_score": _disadvantaged(
            "current_high_asymmetry"
        ),
        "margin_mean": margin_mean,
        "latency_p95_ms": p95 or move_time_p95_ms,
        "classification": classification,
    }


def build_candidate_ranking(
    candidate_reports: list[dict],
) -> list[dict]:
    rows = [report.get("ranking_table", {}) for report in candidate_reports]
    rows = [r for r in rows if isinstance(r, dict) and r]

    def _rank_key(row: dict) -> tuple:
        std_ds = float(row.get("standard_disadvantaged_seat_score") or -1.0)
        eq_ds = float(row.get("equal_high_disadvantaged_seat_score") or -1.0)
        ch_ds = float(row.get("challenger_high_disadvantaged_seat_score") or -1.0)
        return (-std_ds, -eq_ds, -ch_ds)

    return sorted(rows, key=_rank_key)
