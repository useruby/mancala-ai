"""Validate the cumulative P0 safety evidence required for promotion."""

from __future__ import annotations

from typing import Any

REQUIRED_CONTEXTS = ("384:256", "1200:1200")
NONINFERIORITY_LOWER = -0.03


def safe(entry: dict[str, Any]) -> bool:
    """Apply the fixed noninferiority rule to one paired arena result."""
    ci = entry.get("opening_bootstrap_ci", {})
    lower = ci.get("lower_95")
    upper = ci.get("upper_95")
    return (
        isinstance(lower, (int, float))
        and isinstance(upper, (int, float))
        and (upper >= 0.0 or lower >= NONINFERIORITY_LOWER)
    )


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    """Return a promotion-ready P0 safety decision from a canonical report."""
    results = report.get("candidate_vs_p0")
    if not isinstance(results, dict):
        return {
            "passed": False,
            "failure_reasons": ["cumulative_lineage_report_malformed"],
        }

    missing = [context for context in REQUIRED_CONTEXTS if context not in results]
    if missing:
        return {
            "passed": False,
            "failure_reasons": ["cumulative_lineage_context_missing"],
            "missing_contexts": missing,
        }

    unsafe = [context for context in REQUIRED_CONTEXTS if not safe(results[context])]
    return {
        "passed": not unsafe,
        "failure_reasons": ["cumulative_lineage_p0_unsafe"] if unsafe else [],
        "unsafe_contexts": unsafe,
        "required_contexts": list(REQUIRED_CONTEXTS),
    }
