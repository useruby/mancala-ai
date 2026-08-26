"""Pure selection helpers for the context action-Q probe experiment."""

from __future__ import annotations

from typing import Iterable


def corrected_selection(
    children: Iterable[dict], corrections: dict[int, float]
) -> dict[str, object]:
    """Apply visited-edge Q corrections with the ordinary PUCT tie rule.

    Corrections for unvisited edges are deliberately ignored so their FPU
    semantics remain identical to ordinary PUCT.
    """
    rows = []
    for child in sorted(children, key=lambda entry: int(entry["move"])):
        move = int(child["move"])
        visited = int(child["visit_count"]) > 0
        correction = float(corrections.get(move, 0.0)) if visited else 0.0
        q_value = float(child["q_value"]) + correction
        rows.append(
            {
                "move": move,
                "visited": visited,
                "correction": correction,
                "corrected_q": q_value,
                "corrected_score": q_value + float(child["u_component"]),
            }
        )
    if not rows:
        raise ValueError("corrected selection requires children")
    winner = max(
        rows, key=lambda row: (float(row["corrected_score"]), -int(row["move"]))
    )
    return {"move": int(winner["move"]), "rows": rows}
