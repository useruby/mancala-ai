"""Synthetic regressions for published evidence and frozen replay limits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.alphazero_lite import varied_opponent_pilot as pilot
from ml.alphazero_lite import verify_varied_opponent_pilot as verifier


def test_updated_pair_cpu_and_attempt_receipt_fail_published_digest(
    tmp_path: Path,
) -> None:
    pair_dir = tmp_path / "pairs"
    attempt_dir = tmp_path / "attempts"
    pair_dir.mkdir()
    attempt_dir.mkdir()
    pair_path = pair_dir / "pair.json"
    attempt_path = attempt_dir / "attempt.json"

    pair = {"pair_cpu_time_ns": 100}
    attempt = {"pair_cpu_time_ns": 100}
    pair_path.write_text(json.dumps(pair))
    attempt_path.write_text(json.dumps(attempt))
    published_digest = verifier._raw_evidence_manifest_digest(
        tmp_path, pair_dir, attempt_dir
    )

    pair["pair_cpu_time_ns"] = 200
    attempt["pair_cpu_time_ns"] = 200
    pair_path.write_text(json.dumps(pair))
    attempt_path.write_text(json.dumps(attempt))
    changed_digest = verifier._raw_evidence_manifest_digest(
        tmp_path, pair_dir, attempt_dir
    )
    assert changed_digest != published_digest
    assert pair["pair_cpu_time_ns"] == attempt["pair_cpu_time_ns"]

    with pytest.raises(ValueError, match="pilot_evidence_manifest_digest_mismatch"):
        verifier._require_evidence_manifest_digest(changed_digest, published_digest)


def test_frozen_move_cap_accepts_game_within_cap() -> None:
    verifier._verify_frozen_move_cap(
        [{"max_moves": pilot.MAX_MOVES, "trajectory": [{}] * 54}]
    )


def test_frozen_move_cap_rejects_terminal_trajectory_past_cap() -> None:
    game = {
        "game_status": "complete",
        "max_moves": pilot.MAX_MOVES,
        "trajectory": [{}] * (pilot.MAX_MOVES + 1),
    }

    with pytest.raises(ValueError, match="trajectory_exceeds_frozen_move_cap"):
        verifier._verify_frozen_move_cap([game])


def test_frozen_move_cap_rejects_archived_cap_drift() -> None:
    with pytest.raises(ValueError, match="frozen_game_move_cap_mismatch"):
        verifier._verify_frozen_move_cap(
            [{"max_moves": pilot.MAX_MOVES + 1, "trajectory": []}]
        )
