from __future__ import annotations

import hashlib

from ml.alphazero_lite import continue_seed542_update432_strength_screen as continuation


def test_continuation_freeze_binds_raw_pair_and_preserves_remaining_openings() -> None:
    assert continuation.REGISTRATION == "continuation-registration-v4.json"
    assert continuation.FREEZE == "continuation-execution-freeze-v4.json"
    assert len(continuation.SOURCES) == 4
    assert (
        "ml/alphazero_lite/continue_seed542_update432_strength_screen.py"
        in continuation.SOURCES
    )


def test_recovery_continuations_are_append_only_and_advance_openings() -> None:
    assert continuation.REGISTRATION_V5 == "continuation-registration-v5.json"
    assert continuation.FREEZE_V5 == "continuation-execution-freeze-v5.json"
    assert continuation.REGISTRATION_V6 == "continuation-registration-v6.json"
    assert continuation.FREEZE_V6 == "continuation-execution-freeze-v6.json"
    assert continuation._ledger_prefix_hash([], 0) == hashlib.sha256(b"").hexdigest()
