"""Authentication tests for the seed542 continuation amendment runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml.alphazero_lite import run_seed542_amended_generation as amended


def test_amended_command_preserves_recipe_and_only_varies_plan_fields() -> None:
    ordinary = amended.command_for(542002, 64)
    supplement = amended.command_for(542064, 13)
    assert ordinary[0] == str(amended.INTERPRETER)
    assert ordinary[ordinary.index("--games") + 1] == "64"
    assert supplement[supplement.index("--games") + 1] == "13"
    assert supplement[supplement.index("--seed") + 1] == "542064"
    for command in (ordinary, supplement):
        assert command[command.index("--simulations") + 1] == "128"
        assert command[command.index("--policy-target-mode") + 1] == "default"
        assert (
            command[command.index("--exact-root-policy-target-mode") + 1]
            == "selected_one_hot"
        )
        assert command[command.index("--policy-target-noise-mode") + 1] == "denoised"
        assert command[command.index("--value-target-mode") + 1] == "default"


def test_amended_command_template_rejects_behavior_drift() -> None:
    amendment = {"command_template": amended.command_template()}
    amended.authenticate_command_template(amendment)
    changed = list(amendment["command_template"])
    changed[changed.index("128")] = "256"
    with pytest.raises(RuntimeError, match="amended_preflight_command_template_drift"):
        amended.authenticate_command_template({"command_template": changed}, changed)


def test_amended_preflight_asset_authentication_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    path.write_bytes(b"registered weights")
    assets = {
        "weights": {
            "path": "weights.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    }
    amended.authenticate_assets(tmp_path, assets)
    path.write_bytes(b"altered weights")
    with pytest.raises(
        RuntimeError, match="amended_preflight_asset_sha256_mismatch:weights.json"
    ):
        amended.authenticate_assets(tmp_path, assets)


def test_supplemental_command_differs_only_by_seed_output_and_game_count() -> None:
    ordinary = amended.command_for(542063, 64)
    supplement = amended.command_for(542064, 13)
    changed = {"--out", "--games", "--seed"}

    def unchanged_settings(command: list[str]) -> list[str]:
        result = []
        index = 0
        while index < len(command):
            if command[index] in changed:
                index += 2
            else:
                result.append(command[index])
                index += 1
        return result

    ordinary_settings = unchanged_settings(ordinary)
    supplement_settings = unchanged_settings(supplement)
    assert ordinary_settings == supplement_settings


def test_amended_state_skips_completed_and_preserves_incomplete_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(amended, "CORPUS", tmp_path)
    output = tmp_path / "shard-542002.jsonl"
    output.write_bytes(b"completed output")
    event = {
        "seed": 542002,
        "event": "finished",
        "exit_status": 0,
        "completed_games": 64,
        "capped_games": 0,
        "incomplete_or_unstarted_games": 0,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    inventory = tmp_path / "continuation-inventory.jsonl"
    inventory.write_text(json.dumps(event) + "\n")
    assert amended.amended_seed_state(542002) == "completed"
    original_bytes = output.read_bytes()

    event["completed_games"] = 63
    inventory.write_text(json.dumps(event) + "\n")
    assert amended.amended_seed_state(542002) == "preserved-unverified"
    assert output.read_bytes() == original_bytes
