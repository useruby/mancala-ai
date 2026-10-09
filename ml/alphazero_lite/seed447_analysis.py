"""Independent seed447 metrics and decision aggregation."""

from __future__ import annotations

from typing import Any

from ml.alphazero_lite.seed447_joint_output_cap import decide


def summarize(evidence: dict[str, Any]) -> dict[str, Any]:
    """Recompute the registered fixed decision from supplied metrics and steps."""
    return decide(evidence)
