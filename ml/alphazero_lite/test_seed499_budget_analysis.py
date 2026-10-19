"""Synthetic checks for matched-opening budget analysis."""

import pytest

from ml.alphazero_lite.seed499_budget_analysis import analyze


def _rows(winner: str) -> list[dict]:
    return [
        {
            "opening_index": opening,
            "opening_state_hash": f"state-{opening}",
            "F_player": seat,
            "winner": winner,
            "game_length": 20,
            "normal_puct_time_ms": 1.0,
            "exact_root_solve_time_ms": 0.25,
        }
        for opening in range(512)
        for seat in (0, 1)
    ]


def test_matched_budget_bootstrap_and_standalone_accounting():
    result = analyze(_rows("F"), _rows("initializer"))
    assert result["paired_budget_contrast"]["mean_difference"] == 1.0
    assert result["paired_budget_contrast"]["one_sided_95_lower"] == 1.0
    assert result["paired_budget_contrast"]["supports_budget_sensitivity"]
    assert result["standalone_128"]["score"] == 1.0
    assert result["standalone_128"]["wins"] == 1024
    assert result["standalone_128"]["recorded_search_time_ms"]["combined_total"] == 1280


def test_analysis_rejects_a_missing_seat_or_mismatched_opening():
    low = _rows("draw")
    high = _rows("draw")
    with pytest.raises(ValueError, match="128_seat_pair"):
        analyze(low[:-1], high)
    high[0]["opening_state_hash"] = "different"
    with pytest.raises(ValueError, match="historical_1200_opening_identity"):
        analyze(low, high)
