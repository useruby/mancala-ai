"""Launch-contract tests for the seed556 exploratory screen."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import run_seed556_exploratory_arena as runner
from ml.alphazero_lite import seed556_exploratory_arena as protocol


def _execution_fixture(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    output = root / "screen"
    batch = output / "batches"
    batch.mkdir(parents=True)
    raw_opening = b'{"opening_index":0}\n'
    (root / "suite.jsonl").write_bytes(raw_opening)
    frozen_source = root / "frozen_source.py"
    frozen_source.write_text("FROZEN = True\n")
    source_hashes = {"frozen_source.py": protocol.sha256(frozen_source)}
    (output / "openings.jsonl").write_bytes(raw_opening)
    (batch / "opening-0000.jsonl").write_bytes(raw_opening)
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    monkeypatch.setattr(protocol, "OPENINGS_PATH", "suite.jsonl")
    monkeypatch.setattr(
        protocol, "OPENINGS_SHA256", hashlib.sha256(raw_opening).hexdigest()
    )

    sources = {
        "T512": ("t512.npz", protocol.T512_SHA256),
        "P-E4": ("p.npz", protocol.P_SHA256),
        "initializer": ("init.npz", protocol.INITIALIZER_SHA256),
    }
    component_files = {}
    for name, (filename, digest) in sources.items():
        directory = output / "runtime-artifacts" / name
        directory.mkdir(parents=True)
        (directory / "model.npz").write_bytes(filename.encode())
        (directory / "search_policy.json").write_text("{}\n")
        component_files[name] = {
            item.name: protocol.sha256(item) for item in directory.iterdir()
        }
    for checkpoint, filename in (
        (protocol.T512_CHECKPOINT, "t512.npz"),
        (protocol.P_CHECKPOINT, "p.npz"),
        (protocol.INITIALIZER, "init.npz"),
    ):
        path = root / checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
    monkeypatch.setattr(
        protocol, "T512_SHA256", protocol.sha256(root / protocol.T512_CHECKPOINT)
    )
    monkeypatch.setattr(
        protocol, "P_SHA256", protocol.sha256(root / protocol.P_CHECKPOINT)
    )
    monkeypatch.setattr(
        protocol, "INITIALIZER_SHA256", protocol.sha256(root / protocol.INITIALIZER)
    )
    # Mirror source checkpoint bytes into the exported evaluator files.
    for name, checkpoint in (
        ("T512", protocol.T512_CHECKPOINT),
        ("P-E4", protocol.P_CHECKPOINT),
        ("initializer", protocol.INITIALIZER),
    ):
        evaluator_file = output / "runtime-artifacts" / name / "model.npz"
        evaluator_file.write_bytes((root / checkpoint).read_bytes())
        component_files[name]["model.npz"] = protocol.sha256(evaluator_file)

    monkeypatch.setattr(protocol, "TABLEBASE_PATH", "model-artifact/runtime/pointer")
    monkeypatch.setattr(protocol, "NATIVE_PROBE", "model-artifact/runtime/probe")
    assets = root / "model-artifact/runtime"
    assets.mkdir(parents=True)
    pointer = assets / "pointer"
    probe = assets / "probe"
    materialized = root / "materialized.kvtb"
    pointer.write_text("lfs pointer\n")
    probe.write_text("probe\n")
    materialized.write_text("tablebase\n")
    monkeypatch.setattr(runner, "MATERIALIZED_TABLEBASE", materialized)
    monkeypatch.setattr(protocol, "TABLEBASE_SHA256", protocol.sha256(materialized))
    monkeypatch.setattr(protocol, "TABLEBASE_SIZE", materialized.stat().st_size)
    preflight = {
        "exact_root_native_probe_sha256": protocol.sha256(probe),
        "exact_root_tablebase_sha256": protocol.sha256(materialized),
    }
    runner.write_json(output / "runtime-policy-preflight.json", preflight)
    runtime_identity = {"executable": sys.executable, "observed": {"test": True}}
    registration = {
        "schema": "seed556_t512_vs_serving_p_v1",
        "component_assignments": protocol.ASSIGNMENTS,
        "components": component_files,
        "runtime": runtime_identity,
        "source_sha256": source_hashes,
        "runtime_policy_preflight_sha256": protocol.sha256(
            output / "runtime-policy-preflight.json"
        ),
        "tablebase_lfs_pointer_sha256": protocol.sha256(pointer),
        "tablebase_materialized_sha256": protocol.sha256(materialized),
        "tablebase_materialized_size": materialized.stat().st_size,
        "execution_root": str(root.resolve()),
        "execution_output": str(output.resolve()),
    }
    runner.write_json(output / "registration.json", registration)
    freeze = {
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "component_assignments": protocol.ASSIGNMENTS,
        "source_sha256": source_hashes,
    }
    command = runner.build_command(root, output, 0, freeze)
    (output / "command-preflight.jsonl").write_bytes(
        protocol.stable_json(command) + b"\n"
    )
    freeze["command_preflight_sha256"] = protocol.sha256(
        output / "command-preflight.jsonl"
    )
    runner.write_json(output / "execution-freeze.json", freeze)
    return root, output, command


def _evidence_snapshot(path: Path):
    return {
        p.relative_to(path).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def test_actual_launch_command_selects_frozen_components_and_mock_launches(
    tmp_path: Path,
) -> None:
    output = tmp_path / protocol.OUT
    (output / "batches").mkdir(parents=True)
    (output / "runtime-artifacts").mkdir()
    (output / "batches/opening-0000.jsonl").write_text('{"opening_index":0}\n')
    freeze = {
        "registration_sha256": "registration",
        "component_assignments": protocol.ASSIGNMENTS,
    }
    command = runner.build_command(tmp_path, output, 0, freeze)
    mocked = Mock(return_value=Mock(returncode=0, stdout="", stderr=""))
    result = runner.invoke(command, root=tmp_path, output=output, subprocess_run=mocked)
    assert result.returncode == 0
    argv = mocked.call_args.args[0]
    assert "ml.alphazero_lite.seed556_arena_entrypoint" in argv
    assert Path(argv[argv.index("--challenger-policy-artifact") + 1]).name == "T512"
    assert Path(argv[argv.index("--current-policy-artifact") + 1]).name == "P-E4"
    assert (
        Path(argv[argv.index("--challenger-value-artifact") + 1]).name == "initializer"
    )
    assert Path(argv[argv.index("--current-value-artifact") + 1]).name == "initializer"
    assert argv[argv.index("--seed") + 1] == "545"


def test_multicore_launcher_uses_eighteen_cpus_and_eighteen_pair_slots(
    tmp_path: Path,
) -> None:
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
            "python",
            "-m",
            "ml.alphazero_lite.seed556_arena_entrypoint",
            "--challenger-policy-artifact",
            str(output / "runtime-artifacts/T512"),
            "--challenger-value-artifact",
            str(output / "runtime-artifacts/initializer"),
            "--current-policy-artifact",
            str(output / "runtime-artifacts/C128"),
            "--current-value-artifact",
            str(output / "runtime-artifacts/initializer"),
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
        {
            "opening_index": opening,
            "game_position": seat,
            "outcome": "T512",
            "capped_nonterminal": False,
        }
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


@pytest.mark.parametrize(
    ("relative_path", "replacement", "error"),
    [
        (
            "screen/runtime-artifacts/T512/model.npz",
            b"substituted weights",
            "component_bytes_mismatch:T512",
        ),
        (
            "screen/runtime-artifacts/T512/search_policy.json",
            b"{}\\nchanged\n",
            "component_bytes_mismatch:T512",
        ),
        ("frozen_source.py", b"FROZEN = False\n", "execution_source_mismatch"),
    ],
)
def test_execute_rehashes_frozen_inputs_before_any_write_or_launch(
    tmp_path: Path, monkeypatch, relative_path: str, replacement: bytes, error: str
) -> None:
    root, output, _ = _execution_fixture(tmp_path, monkeypatch)
    target = root / relative_path
    target.write_bytes(replacement)
    before = _evidence_snapshot(output)
    launch = Mock()
    monkeypatch.setattr(
        runner, "assert_multicore_runtime", lambda expected=None: expected
    )
    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises(ValueError, match=error):
        runner.execute(root, output)
    launch.assert_not_called()
    assert _evidence_snapshot(output) == before


def test_execute_rejects_conflicting_existing_ledger_without_writes_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    root, output, command = _execution_fixture(tmp_path, monkeypatch)
    batch = output / "batches"
    command_path = batch / "command-0000.json"
    runner.write_json(command_path, command)
    games_path = batch / "games-0000.jsonl"
    report_path = batch / "report-0000.json"
    games_path.write_text("saved games\n")
    report_path.write_text("saved report\n")
    valid_pair = [
        {"opening_index": 0, "game_position": seat, "outcome": outcome}
        for seat, outcome in enumerate(("T512", "P"))
    ]
    runner.write_json(
        batch / "completion-0000.json",
        {
            "opening_index": 0,
            "command_sha256": protocol.sha256(command_path),
            "outcomes": valid_pair,
        },
    )
    contradictory = {**valid_pair[0], "outcome": "P"}
    (output / "outcomes.jsonl").write_bytes(protocol.stable_json(contradictory) + b"\n")
    monkeypatch.setattr(
        runner, "assert_multicore_runtime", lambda expected=None: expected
    )
    monkeypatch.setattr(runner, "validate_pair", lambda *_: valid_pair)
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)
    before = _evidence_snapshot(output)
    with pytest.raises(ValueError, match="existing_ledger_conflict"):
        runner.execute(root, output)
    launch.assert_not_called()
    assert _evidence_snapshot(output) == before


def test_resume_appends_missing_rows_without_replacing_existing_ledger(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "outcomes.jsonl"
    first = {"opening_index": 0, "game_position": 0, "outcome": "T512"}
    second = {"opening_index": 0, "game_position": 1, "outcome": "P"}
    original = protocol.stable_json(first) + b"\n"
    ledger_path.write_bytes(original)
    missing = runner.missing_committed_ledger_rows(original, [first, second])
    assert missing == [second]
    runner._append_ledger_rows(ledger_path, original, missing)
    saved = ledger_path.read_bytes()
    assert saved.startswith(original)
    assert [json.loads(line) for line in saved.splitlines()] == [first, second]


def test_resume_ledger_rejects_rebound_conflicting_row() -> None:
    validated = [{"opening_index": 0, "game_position": 0, "outcome": "T512"}]
    prior = (
        protocol.stable_json({"opening_index": 0, "game_position": 0, "outcome": "P"})
        + b"\n"
    )
    with pytest.raises(ValueError, match="existing_ledger_conflict"):
        runner.missing_committed_ledger_rows(prior, validated)
