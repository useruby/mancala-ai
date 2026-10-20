"""Synthetic checks for the frozen seed505 command and interruption policy."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ml.alphazero_lite import seed505_strength_screen as protocol


def test_command_uses_e4_initializer_and_registered_runtime(tmp_path: Path) -> None:
    opening = json.loads(
        (protocol.ROOT / protocol.HISTORICAL / "openings.jsonl")
        .read_text()
        .splitlines()[0]
    )
    root = tmp_path
    out = root / protocol.OUT
    (out / "runtime-artifacts").mkdir(parents=True)
    historical = protocol.ROOT / protocol.HISTORICAL / "registration.json"
    (root / protocol.HISTORICAL).mkdir(parents=True)
    (root / protocol.HISTORICAL / "registration.json").write_bytes(
        historical.read_bytes()
    )
    args = protocol.command(root, 0, opening)
    assert args[args.index("--challenger") + 1] == str(out / "runtime-artifacts/E4")
    assert args[args.index("--current") + 1] == str(
        out / "runtime-artifacts/initializer"
    )
    assert args[args.index("--challenger-simulations") + 1] == "128"
    assert args[args.index("--current-simulations") + 1] == "128"
    assert args[args.index("--c-puct") + 1] == "1.25"
    assert args[args.index("--workers") + 1] == "1"
    assert args[args.index("--seed") + 1] == "488"
    assert args[args.index("--max-moves") + 1] == "200"
    assert args[args.index("--random-opening-plies") + 1] == "0"


def test_interrupted_outcome_batch_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = protocol.ROOT
    out = tmp_path
    monkeypatch.setattr(protocol, "OUT", out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        root / ".tmp/seed500-current-checkpoint.npz", out / "initializer.npz"
    )
    registration = {"source_sha256": {}, "runtime": {"runtime_identities_sha256": {}}}
    (out / "registration.json").write_text(json.dumps(registration))
    (out / "execution-freeze.json").write_text(
        json.dumps(
            {
                "status": "frozen_before_execution",
                "registration_sha256": protocol.sha(out / "registration.json"),
            }
        )
    )
    opening = json.loads(
        (root / protocol.HISTORICAL / "openings.jsonl").read_text().splitlines()[0]
    )
    (out / "openings.jsonl").write_text(json.dumps(opening) + "\n")
    (out / "batches").mkdir()
    sentinel = out / "batches/games-0000.jsonl"
    sentinel.write_text('{"winner":"challenger"}\n')
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="interrupted_outcome_batch_preserved:0"):
        protocol.run(root, execute=True)
    assert sentinel.read_bytes() == before
