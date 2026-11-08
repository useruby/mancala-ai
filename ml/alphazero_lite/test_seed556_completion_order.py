"""Tests for the seed556 concurrent completion-order ledger correction."""

from __future__ import annotations

import pytest

from ml.alphazero_lite import seed556_exploratory_arena as protocol
from ml.alphazero_lite.verify_seed556_completion_order_correction import (
    normalize_completion_order_ledger,
)


def test_completion_order_normalizes_to_indexed_opening_seat_order() -> None:
    rows = [
        {"opening_index": opening, "game_position": seat, "value": opening * 2 + seat}
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    completion_order = rows[::2] + rows[1::2]
    assert normalize_completion_order_ledger(completion_order) == rows


def test_completion_order_rejects_receipt_rebound_duplicate_and_missing_seats() -> None:
    rows = [
        {"opening_index": opening, "game_position": seat}
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    duplicate = [*rows[:-1], rows[0]]
    with pytest.raises(ValueError, match="duplicate_ledger_opening_seat"):
        normalize_completion_order_ledger(duplicate)
    with pytest.raises(ValueError, match="coverage_mismatch"):
        normalize_completion_order_ledger(rows[:-1])
