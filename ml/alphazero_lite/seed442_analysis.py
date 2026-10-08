"""Independent endpoint-metric aggregation and fixed seed442 decision rules."""

from __future__ import annotations

from typing import Any

from ml.alphazero_lite.seed442_kl_capped_adam import decide


def summarize(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered classification and all named decision clauses."""
    return decide(evidence)
