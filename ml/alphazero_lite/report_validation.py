from __future__ import annotations

from typing import Any


EXPECTED_ARENA_SCHEMA = "arena_v1"


class ArenaReportValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_arena_report(
    *,
    report: dict[str, Any],
    min_score: float,
    min_confidence_lower_bound: float | None = None,
 ) -> dict[str, Any]:
    if report.get("schema") != EXPECTED_ARENA_SCHEMA:
        raise_validation("SCHEMA", f"schema must be {EXPECTED_ARENA_SCHEMA}")

    games_played = integer_field(report, "games_played")
    wins = integer_field(report, "wins")
    losses = integer_field(report, "losses")
    draws = integer_field(report, "draws")

    if games_played <= 0:
        raise_validation("COUNTS", "games_played must be greater than 0")
    if wins < 0 or losses < 0 or draws < 0:
        raise_validation("COUNTS", "wins/losses/draws must be non-negative")
    if wins + losses + draws != games_played:
        raise_validation("COUNTS", "wins + losses + draws must equal games_played")

    score = (wins + (draws * 0.5)) / games_played
    score_passed = score >= float(min_score)
    confidence_interval = wilson_interval_95(score=score, sample_size=games_played)
    confidence_lower_bound = confidence_interval["lower"]
    confidence_passed = True
    if min_confidence_lower_bound is not None:
        confidence_passed = confidence_lower_bound >= float(min_confidence_lower_bound)
    passed = score_passed and confidence_passed
    threshold = float(min_score)

    decision = report.get("promotion_decision")
    if not isinstance(decision, dict):
        raise_validation("DECISION", "promotion_decision must be present")

    declared = decision.get("passed")
    if not isinstance(declared, bool):
        raise_validation("DECISION", "promotion_decision.passed must be boolean")
    if declared != score_passed:
        raise_validation(
            "DECISION_MISMATCH",
            "promotion_decision.passed must match score threshold outcome (not confidence gate)",
        )

    return {
        "passed": passed,
        "score": score,
        "score_passed": score_passed,
        "confidence_passed": confidence_passed,
        "confidence_lower_bound": confidence_lower_bound,
        "confidence_upper_bound": confidence_interval["upper"],
        "confidence_interval_95": confidence_interval,
        "threshold": threshold,
        "threshold_margin": score - threshold,
        "unstable_decision": (
            confidence_lower_bound <= threshold <= confidence_interval["upper"]
        ),
        "min_score": float(min_score),
        "min_confidence_lower_bound": (
            float(min_confidence_lower_bound)
            if min_confidence_lower_bound is not None
            else None
        ),
        "games_played": games_played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def integer_field(report: dict[str, Any], key: str) -> int:
    value: Any = report.get(key)
    if value is None:
        raise_validation("SCHEMA", f"{key} is missing")

    if isinstance(value, bool) or not isinstance(value, int):
        raise_validation("SCHEMA", f"{key} must be an integer")

    return value


def raise_validation(suffix: str, message: str) -> None:
    raise ArenaReportValidationError(f"ARENA_VALIDATION::{suffix}", message)


def wilson_lower_bound(*, score: float, sample_size: int, z: float = 1.96) -> float:
    return wilson_interval(score=score, sample_size=sample_size, z=z)["lower"]


def wilson_interval_95(*, score: float, sample_size: int) -> dict[str, float | str]:
    interval = wilson_interval(score=score, sample_size=sample_size, z=1.96)
    return {
        "lower": interval["lower"],
        "upper": interval["upper"],
        "method": "wilson_score",
        "caveat": "Approximation over score rate with win=1.0, draw=0.5, loss=0.0.",
    }


def wilson_interval(*, score: float, sample_size: int, z: float = 1.96) -> dict[str, float]:
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
