"""Launch-contract tests for the seed556 exploratory screen."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import run_seed556_exploratory_arena as runner
from ml.alphazero_lite import seed556_exploratory_arena as protocol


def test_actual_launch_command_selects_frozen_components_and_mock_launches(
    tmp_path: Path,
) -> None:
    output = tmp_path / protocol.OUT
    (output / "batches").mkdir(parents=True)
    (output / "runtime-artifacts").mkdir()
    (output / "batches/opening-0000.jsonl").write_text('{"opening_index":0}\n')
    freeze = {"registration_sha256": "registration", "component_assignments": protocol.ASSIGNMENTS}
    command = runner.build_command(tmp_path, output, 0, freeze)
    mocked = Mock(return_value=Mock(returncode=0, stdout="", stderr=""))
    result = runner.invoke(command, root=tmp_path, output=output, subprocess_run=mocked)
    assert result.returncode == 0
    argv = mocked.call_args.args[0]
    assert "ml.alphazero_lite.seed556_arena_entrypoint" in argv
    assert Path(argv[argv.index("--challenger-policy-artifact") + 1]).name == "T512"
    assert Path(argv[argv.index("--current-policy-artifact") + 1]).name == "P-E4"
    assert Path(argv[argv.index("--challenger-value-artifact") + 1]).name == "initializer"
    assert Path(argv[argv.index("--current-value-artifact") + 1]).name == "initializer"
    assert argv[argv.index("--seed") + 1] == "545"


def test_multicore_launcher_uses_eighteen_cpus_and_eighteen_pair_slots(tmp_path: Path) -> None:
    command = runner.durable_command(protocol.ROOT, tmp_path / "screen")
    assert "--unit=seed556-exploratory-arena-18way" in command
    assert any("CPUQuota=1800%" in arg for arg in command)
    assert any("AllowedCPUs=14-31" in arg for arg in command)
    assert any("MemoryMax=38654705664" in arg for arg in command)
    assert protocol.PAIR_CONCURRENCY == 18
    assert "--setenv=OMP_NUM_THREADS=1" in command


def test_c128_command_rejected_before_launch_or_evidence_write(tmp_path: Path) -> None:
    output = tmp_path / protocol.OUT
    output.mkdir(parents=True)
    before = sorted(p.relative_to(output) for p in output.rglob("*"))
    command = {
        "component_assignments": protocol.ASSIGNMENTS,
        "argv": [
            "python", "-m", "ml.alphazero_lite.seed556_arena_entrypoint",
            "--challenger-policy-artifact", str(output / "runtime-artifacts/T512"),
            "--challenger-value-artifact", str(output / "runtime-artifacts/initializer"),
            "--current-policy-artifact", str(output / "runtime-artifacts/C128"),
            "--current-value-artifact", str(output / "runtime-artifacts/initializer"),
        ],
    }
    mocked = Mock()
    with pytest.raises(ValueError, match="component_selection_mismatch"):
        runner.invoke(command, root=tmp_path, output=output, subprocess_run=mocked)
    mocked.assert_not_called()
    assert sorted(p.relative_to(output) for p in output.rglob("*")) == before


def test_interrupted_complete_pair_is_recovered_without_replay(tmp_path: Path) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    for name, suffix in (("command", "json"), ("games", "jsonl"), ("report", "json")):
        (batch / f"{name}-0004.{suffix}").write_text("saved\n")
    assert runner.recovery_action(batch, 4, set()) == "recover"
    assert runner.recovery_action(batch, 4, {4}) == "skip"
    assert runner.recovery_action(batch, 4, set()) == "recover"


def test_first_time_finalization_does_not_require_publication_receipts(
    tmp_path: Path,
) -> None:
    output = tmp_path
    rows = [
        {"opening_index": opening, "game_position": seat,
         "outcome": "T512", "capped_nonterminal": False}
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    (output / "outcomes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    verifier = Mock(return_value={"status": "verified_complete"})
    result = runner.finalize(output, verifier)
    verifier.assert_called_once_with(output)
    assert result["status"] == "verified_complete"
    assert not (output / "verification-receipt.json").exists()
