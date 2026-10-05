"""Corrected seat-weighted analysis for the immutable seed422 ledger."""

from __future__ import annotations

from typing import Any

from ml.alphazero_lite.seed422_adam_memory_analysis import (
    analyze as _analyze,
)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reanalyse the published ledger with actual per-seat denominators."""
    result = _analyze(rows)
    for lane in ("A", "B"):
        lane_rows = [row for row in rows if row["lane"] == lane]
        seat_counts = {
            seat: sum(
                int(row["game"]["challenger_player"]) == seat for row in lane_rows
            )
            for seat in (0, 1)
        }
        if seat_counts != {0: 512, 1: 512}:
            raise ValueError(f"seat_game_count_mismatch:{lane}")
        seat_scores = {
            str(seat): sum(
                float(row["opponent_score"])
                for row in lane_rows
                if int(row["game"]["challenger_player"]) == seat
            )
            / seat_counts[seat]
            for seat in (0, 1)
        }
        if sum(seat_scores.values()) / 2 != result["lanes"][lane]["score"]:
            raise ValueError(f"seat_mean_lane_score_mismatch:{lane}")
        result["lanes"][lane]["seat_scores"] = seat_scores
        result["lanes"][lane]["seat_game_counts"] = {
            str(seat): count for seat, count in seat_counts.items()
        }
    result["schema"] = "seed422-analysis-corrected-v1"
    return result
