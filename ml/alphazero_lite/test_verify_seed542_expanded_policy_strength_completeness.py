"""Tests for frozen preflight-file authentication."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ml.alphazero_lite.verify_seed542_expanded_policy_strength_completeness import (
    _verify_frozen_command_preflights,
)


PREFLIGHTS = (
    ("original", "command-preflight.jsonl"),
    ("v2", "continuation-command-preflight-v2.jsonl"),
    ("v3", "continuation-command-preflight-v3.jsonl"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(("target_label", "target_filename"), PREFLIGHTS)
def test_changed_preflight_is_rejected_when_freeze_is_unchanged(
    tmp_path: Path, target_label: str, target_filename: str
) -> None:
    freezes: dict[str, dict[str, Any]] = {}
    original_bytes: dict[str, bytes] = {}
    for label, filename in PREFLIGHTS:
        data = f"frozen {label} preflight\n".encode()
        (tmp_path / filename).write_bytes(data)
        freezes[label] = {"command_preflight_sha256": _sha256(data)}
        original_bytes[label] = data

    _verify_frozen_command_preflights(
        tmp_path, freezes["original"], freezes["v2"], freezes["v3"]
    )

    (tmp_path / target_filename).write_bytes(b"tampered preflight\n")

    with pytest.raises(
        ValueError,
        match=f"completeness_frozen_command_preflight_hash_mismatch:{target_label}",
    ):
        _verify_frozen_command_preflights(
            tmp_path, freezes["original"], freezes["v2"], freezes["v3"]
        )

    assert freezes[target_label]["command_preflight_sha256"] == _sha256(
        original_bytes[target_label]
    )
