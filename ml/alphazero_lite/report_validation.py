from __future__ import annotations

from typing import Any


EXPECTED_ARENA_SCHEMA = "arena_v1"


class ArenaReportValidationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def validate_arena_report(*, report: dict[str, Any], min_score: float) -> dict[str, float | int | bool]:
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
    passed = score >= float(min_score)

    decision = report.get("promotion_decision")
    if not isinstance(decision, dict):
        raise_validation("DECISION", "promotion_decision must be present")
        decision = {}

    declared = decision.get("passed")
    if not isinstance(declared, bool):
        raise_validation("DECISION", "promotion_decision.passed must be boolean")
    if declared != passed:
        raise_validation("DECISION_MISMATCH", "promotion_decision.passed must match computed threshold outcome")

    return {
        "passed": passed,
        "score": score,
        "min_score": float(min_score),
        "games_played": games_played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def integer_field(report: dict[str, Any], key: str) -> int:
    value: Any = report.get(key)
    if value is None:
        raise_validation("SCHEMA", f"{key} is missing")
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        raise_validation("SCHEMA", f"{key} must be an integer")
        return 0


def raise_validation(suffix: str, message: str) -> None:
    raise ArenaReportValidationError(f"ARENA_VALIDATION::{suffix}", message)
