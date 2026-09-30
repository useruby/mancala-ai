"""Strict provenance and game-pair validation for the seed461 LR study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_training_record(
    record: dict[str, Any],
    *,
    record_path: Path,
    expected_record_sha256: str,
    expected_epochs: dict[str, Any],
    expected_selected: dict[str, str],
) -> None:
    if sha256(record_path) != expected_record_sha256:
        raise ValueError("training_record_hash_mismatch")
    trajectories = record.get("trajectories", {})
    for run, epoch_hashes in expected_epochs.items():
        actual = trajectories.get(run)
        if actual is None or actual.get("epochs") != epoch_hashes:
            raise ValueError(f"checkpoint_epoch_binding_mismatch:{run}")
        if actual.get("selected_sha256") != expected_selected.get(run):
            raise ValueError(f"selected_checkpoint_binding_mismatch:{run}")
        if actual["selected_sha256"] not in epoch_hashes.values():
            raise ValueError(f"selected_checkpoint_not_an_epoch:{run}")


def _opening_identity(row: dict[str, Any]) -> str:
    state = row.get("opening_state")
    if state is None:
        state = row.get("state")
    if state is None:
        raise ValueError("opening_state_identity_missing")
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def validate_games(
    rows: list[dict[str, Any]],
    *,
    expected_suite: str,
    expected_opponent: str,
    expected_candidate: str,
    expected_runtime: str,
    registered_openings: dict[int, str],
) -> dict[int, list[dict[str, Any]]]:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        for field, expected in (
            ("suite_sha256", expected_suite),
            ("opponent_identity", expected_opponent),
            ("candidate_identity", expected_candidate),
            ("runtime_identity", expected_runtime),
        ):
            if row.get(field) != expected:
                raise ValueError(f"report_identity_mismatch:{field}")
        index = int(row["opening_index"])
        if (
            index not in registered_openings
            or _opening_identity(row) != registered_openings[index]
        ):
            raise ValueError("opening_identity_mismatch")
        if row.get("winner") not in {"challenger", "current", "draw"}:
            raise ValueError("unknown_winner")
        by_index.setdefault(index, []).append(row)
    if set(by_index) != set(registered_openings):
        raise ValueError("missing_openings")
    for index, games in by_index.items():
        seats = [int(game["challenger_player"]) for game in games]
        if len(games) != 2 or sorted(seats) != [0, 1]:
            raise ValueError(f"seat_pairing_failure:{index}")
    return by_index
