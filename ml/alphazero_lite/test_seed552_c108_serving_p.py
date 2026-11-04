"""Synthetic contract tests for seed552 composition, recovery, and analysis."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed552_c108_serving_p as runner
from ml.alphazero_lite import seed552_c108_serving_p as protocol
from ml.alphazero_lite import verify_seed551_controlled_policy_strength as old_verify
from ml.alphazero_lite import verify_seed552_c108_serving_p as verify


def test_composed_policy_and_value_components_keep_their_own_trunks() -> None:
    initializer = object()
    c = object()
    p = object()
    c108 = arena.ComposedArtifactEvaluator(
        initializer, c, policy_source="candidate", value_source="current"
    )
    serving_p = arena.ComposedArtifactEvaluator(
        initializer, p, policy_source="candidate", value_source="current"
    )
    assert c108.candidate_evaluator is c
    assert c108.current_evaluator is initializer
    assert serving_p.candidate_evaluator is p
    assert serving_p.current_evaluator is initializer


def test_command_has_c108_challenger_p_opponent_and_global_seed(tmp_path: Path) -> None:
    out = tmp_path / "seed552"
    batch = out / "batches"
    batch.mkdir(parents=True)
    line = (
        (protocol.ROOT / protocol.OPENINGS_PATH)
        .read_bytes()
        .splitlines(keepends=True)[7]
    )
    (batch / "opening-0007.jsonl").write_bytes(line)
    record = runner.build_command(
        protocol.ROOT,
        out,
        7,
        {"registration_sha256": "reg", "suite_sha256": protocol.OPENINGS_SHA256},
    )
    argv = record["argv"]
    assert argv[argv.index("--challenger-policy-artifact") + 1].endswith("/C108")
    assert argv[argv.index("--current-policy-artifact") + 1].endswith("/P-E4")
    assert argv[argv.index("--challenger-value-artifact") + 1].endswith("/initializer")
    assert argv[argv.index("--current-value-artifact") + 1].endswith("/initializer")
    assert argv[argv.index("--seed") + 1] == "552"
    assert record["opening_index"] == 7
    assert protocol.SEED + record["opening_index"] == 552


def test_durable_launcher_freezes_registered_limits(tmp_path: Path) -> None:
    command = runner.launch_command(protocol.ROOT, tmp_path / "seed552")
    assert command[0:2] == ["systemd-run", "--user"]
    assert "--unit=seed552-c108-serving-p" in command
    assert "--collect" in command
    assert "--scope" not in command
    for token in (
        "CPUQuota=25%",
        "CPUWeight=1",
        "MemoryMax=4294967296",
        "AllowedCPUs=30",
        "Nice=19",
        "LimitAS=17179869184",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "BLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "--execute",
    ):
        assert any(token in arg for arg in command)
    executable_index = command.index(runner.sys.executable)
    assert command[executable_index - 3 : executable_index] == ["taskset", "-c", "30"]


def test_local_arena_opening_index_zero_and_seat_pairing() -> None:
    opening = old_verify.replay_suite(protocol.ROOT)[7]
    state = protocol.suite.KalahGame.from_state(opening["state"])
    trajectory: list[int] = []
    extra_turn_seen = False
    while not state.over():
        relative = state.possible_moves()[0]
        absolute = state.pit_index(relative)
        mover = state.current_player
        trajectory.append(absolute)
        assert state.move(absolute)
        extra_turn_seen |= state.current_player == mover
    assert extra_turn_seen
    for seat in (0, 1):
        margin = state.captured_seeds[seat] - state.captured_seeds[1 - seat]
        row = {
            "opening_index": 0,
            "opening_prefix_moves": opening["prefix_moves"],
            "opening_state_hash": opening["state_hash"],
            "opening_applied_prefix_length": len(opening["prefix_moves"]),
            "opening_contract": "arena_player_relative_v2",
            "challenger_player": seat,
            "trajectory": ",".join(map(str, trajectory)),
            "game_length": len(trajectory),
            "margin": margin,
            "winner": "challenger"
            if margin > 0
            else "current"
            if margin < 0
            else "draw",
        }
        old_verify._validate_arena_local_opening_index(row)
        result, _, capped = verify._outcome(opening, row)
        assert result == ("C" if margin > 0 else "P" if margin < 0 else "draw")
        assert not capped


def test_analysis_uses_cluster_bootstrap_and_capped_games_block_advancement() -> None:
    rows = [
        {
            "opening_index": opening,
            "game_position": seat,
            "outcome": "C",
            "capped_nonterminal": False,
        }
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    result, per_opening = protocol.analyze(rows)
    assert result["score"] == 1.0
    assert result["one_sided_95_lower"] == 1.0
    assert result["decision"] == "advance_to_separate_1200_simulation_screen"
    assert len(per_opening) == protocol.OPENINGS
    rows[0]["outcome"] = "capped_C"
    rows[0]["capped_nonterminal"] = True
    result, _ = protocol.analyze(rows)
    assert result["capped_nonterminal_games"] == 1
    assert not result["strength_decision_valid"]
    assert result["decision"] == "no_valid_strength_decision"


def test_recovery_preserves_partial_or_conflicting_attempts(tmp_path: Path) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    (batch / "opening-0000.jsonl").write_text("opening\n")
    assert runner.recovery_action(batch, 0, set()) == "execute"
    (batch / "command-0000.json").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete_prior_attempt_preserved"):
        runner.recovery_action(batch, 0, set())
    (batch / "games-0000.jsonl").write_text("{}\n")
    (batch / "report-0000.json").write_text("{}\n")
    assert runner.recovery_action(batch, 0, set()) == "recover"
    assert runner.recovery_action(batch, 0, {0}) == "skip"


def _resume_fixture(
    tmp_path: Path, *, ledger_bytes: bytes | None
) -> tuple[Path, Path, list[dict], dict]:
    output = tmp_path / "resume-screen"
    batch = output / "batches"
    batch.mkdir(parents=True)
    runtime = {"observed": {"cpu_affinity": [30]}, "executable": runner.sys.executable}
    (output / "registration.json").write_text(json.dumps({"runtime": runtime}))
    for index in (0, 1):
        (batch / f"opening-{index:04d}.jsonl").write_text(f"opening-{index}\n")
    (batch / "command-0000.json").write_text('{"frozen":true}\n')
    (batch / "games-0000.jsonl").write_text('{"game":0}\n{"game":1}\n')
    (batch / "report-0000.json").write_text('{"report":true}\n')
    expected = [
        {
            "opening_index": 0,
            "game_position": seat,
            "outcome": "C" if seat == 0 else "P",
            "capped_nonterminal": False,
        }
        for seat in (0, 1)
    ]
    from ml.alphazero_lite.seed552_c108_serving_p import sha256

    evidence = {
        "command_sha256": sha256(batch / "command-0000.json"),
        "games_sha256": sha256(batch / "games-0000.jsonl"),
        "report_sha256": sha256(batch / "report-0000.json"),
        "opening_sha256": sha256(batch / "opening-0000.jsonl"),
        "registration_sha256": sha256(output / "registration.json"),
    }
    (batch / "completion-0000.json").write_text(
        json.dumps({**evidence, "outcomes": expected}) + "\n"
    )
    commands = [
        {"opening_index": index, "argv": ["fake-arena", str(index)]}
        for index in range(protocol.OPENINGS)
    ]
    (output / "command-preflight.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in commands)
    )
    ledger = output / "outcomes.jsonl"
    if ledger_bytes is not None:
        ledger.write_bytes(ledger_bytes)
    return output, ledger, expected, runtime


def _stub_resume_preflight(monkeypatch, runtime: dict) -> Mock:
    preflight = Mock(return_value={"registration": {"runtime": runtime}})
    monkeypatch.setattr(verify, "verify_preflight", preflight)
    monkeypatch.setattr(
        runner.base_runner, "assert_runtime_contract", lambda _expected: None
    )
    monkeypatch.setattr(runner, "registered_runtime", lambda: runtime)
    return preflight


def _file_inventory(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_execute_rejects_invalid_freeze_before_any_write_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "invalid-root"
    output = root / protocol.OUT
    output.mkdir(parents=True)
    (output / "registration.json").write_text(
        json.dumps({"execution_root": str(root)}) + "\n"
    )
    (output / "execution-freeze.json").write_text("{}\n")
    before = _file_inventory(output)
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)
    runtime_check = Mock()
    monkeypatch.setattr(runner.base_runner, "assert_runtime_contract", runtime_check)

    with pytest.raises(ValueError, match="seed552_freeze_or_suite_binding_mismatch"):
        runner.execute(root, output)

    runtime_check.assert_not_called()
    launch.assert_not_called()
    assert _file_inventory(output) == before


def test_execute_rejects_runtime_identity_before_any_write_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "runtime-root"
    output = root / protocol.OUT
    output.mkdir(parents=True)
    before = _file_inventory(output)
    runtime_expected = {"observed": {"cpu_affinity": [30]}}
    preflight = Mock(return_value={"registration": {"runtime": runtime_expected}})
    monkeypatch.setattr(verify, "verify_preflight", preflight)
    runtime_check = Mock(side_effect=ValueError("runtime_identity_invalid"))
    monkeypatch.setattr(runner.base_runner, "assert_runtime_contract", runtime_check)
    monkeypatch.setattr(runner, "registered_runtime", lambda: runtime_expected)
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)

    with pytest.raises(ValueError, match="runtime_identity_invalid"):
        runner.execute(root, output)

    preflight.assert_called_once()
    runtime_check.assert_called_once_with(runtime_expected["observed"])
    launch.assert_not_called()
    assert _file_inventory(output) == before


def test_resume_rejects_conflicting_ledger_without_overwrite_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    contradictory = (
        json.dumps({"opening_index": 0, "game_position": 0, "outcome": "P"}).encode()
        + b"\n"
    )
    output, ledger, expected, runtime = _resume_fixture(
        tmp_path, ledger_bytes=contradictory
    )
    before = _file_inventory(output)
    preflight = _stub_resume_preflight(monkeypatch, runtime)
    monkeypatch.setattr(verify, "validate_pair", lambda *_args, **_kwargs: expected)
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)

    with pytest.raises(ValueError, match="existing_ledger_conflicts_preserved"):
        runner.execute(protocol.ROOT, output)

    preflight.assert_called_once()
    launch.assert_not_called()
    assert ledger.read_bytes() == contradictory
    assert _file_inventory(output) == before
    assert not (output / "ledger-recovery").exists()


def test_resume_records_validated_pair_recovery_before_next_launch(
    tmp_path: Path, monkeypatch
) -> None:
    output, ledger, expected, runtime = _resume_fixture(tmp_path, ledger_bytes=b"")
    _stub_resume_preflight(monkeypatch, runtime)
    validate_pair = Mock(return_value=expected)
    monkeypatch.setattr(verify, "validate_pair", validate_pair)

    class StopBeforeNextArena(RuntimeError):
        pass

    launch = Mock(side_effect=StopBeforeNextArena("stop after resume gate"))
    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises(StopBeforeNextArena):
        runner.execute(protocol.ROOT, output)

    validate_pair.assert_called_once_with(protocol.ROOT.resolve(), output.resolve(), 0)
    launch.assert_called_once()
    assert ledger.read_text().splitlines() == [
        protocol.stable_json(row).decode() for row in expected
    ]
    recovery_records = list((output / "ledger-recovery").glob("recovery-*.json"))
    assert len(recovery_records) == 1
    recovery = json.loads(recovery_records[0].read_text())
    assert recovery["recovery_kind"] == "committed_pair_ledger_projection_recovery"
    assert recovery["validated_opening_pairs"] == 1
    assert recovery["validated_games"] == 2
    assert (output / recovery["preserved_ledger_path"]).read_bytes() == b""
    assert (output / "batches/command-0001.json").is_file()


def test_receipt_rebound_semantic_corruptions_are_rejected(tmp_path: Path) -> None:
    out = tmp_path / "seed552"
    (out / "batches").mkdir(parents=True)
    opening = (
        (protocol.ROOT / protocol.OPENINGS_PATH)
        .read_bytes()
        .splitlines(keepends=True)[0]
    )
    opening_path = out / "batches/opening-0000.jsonl"
    opening_path.write_bytes(opening)
    freeze = {"registration_sha256": "reg", "suite_sha256": protocol.OPENINGS_SHA256}
    command = runner.build_command(protocol.ROOT, out, 0, freeze)
    expected_runtime = command["runtime_identity"]

    def check(
        candidate: dict,
        expected_opening_sha: str = protocol.sha256(opening_path),
        runtime: dict = expected_runtime,
    ) -> None:
        verify.validate_command_semantics(
            candidate, 0, out, expected_opening_sha, runtime
        )

    bad = dict(command)
    bad["component_assignments"] = {
        **protocol.ASSIGNMENTS,
        "opponent_policy_artifact": "C108",
    }
    with pytest.raises(ValueError, match="component_assignment"):
        check(bad)

    bad = dict(command)
    bad["suite_row_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="opening_binding"):
        check(bad)

    bad = dict(command)
    bad["argv"] = list(command["argv"])
    bad["argv"][bad["argv"].index("--challenger-policy-artifact") + 1] = str(
        out / "runtime-artifacts/P-E4"
    )
    with pytest.raises(ValueError, match="component_path"):
        check(bad)

    bad = dict(command)
    bad["runtime_identity"] = {**expected_runtime, "torch": "arbitrary"}
    with pytest.raises(ValueError, match="runtime_identity"):
        check(bad)

    # Rebinding a copied opening hash cannot make a nonregistered source row valid.
    with pytest.raises(ValueError, match="opening_binding"):
        check(command, "f" * 64)


def test_receipt_rebound_outcome_ledger_and_analysis_corruption_rejected() -> None:
    opening = old_verify.replay_suite(protocol.ROOT)[0]
    synthetic = {
        "challenger_player": 0,
        "opening_prefix_moves": opening["prefix_moves"],
        "opening_state_hash": opening["state_hash"],
        "opening_applied_prefix_length": len(opening["prefix_moves"]),
        "opening_contract": "arena_player_relative_v2",
        "trajectory": "",
        "game_length": 0,
        "margin": 0,
        "winner": "draw",
    }
    with pytest.raises(ValueError, match="nonterminal_before_move_cap"):
        verify._outcome(opening, synthetic)

    ledger = [
        {
            "opening_index": i,
            "game_position": seat,
            "outcome": "C",
            "capped_nonterminal": False,
        }
        for i in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    analysis, scores = verify.reconstruct_analysis(ledger)
    mutated_ledger = [dict(row) for row in ledger]
    mutated_ledger[2]["opening_index"] = 0
    with pytest.raises(ValueError, match="duplicate_ledger"):
        verify.reconstruct_analysis(mutated_ledger)
    altered_analysis = {**analysis, "decision": "do_not_advance_C108"}
    with pytest.raises(ValueError, match="analysis_reconstruction"):
        verify.compare_analysis_binding(altered_analysis, analysis, scores, scores)
    altered_scores = [dict(scores[0], seat0_C_score=0.0), *scores[1:]]
    with pytest.raises(ValueError, match="per_opening_reconstruction"):
        verify.compare_analysis_binding(analysis, analysis, altered_scores, scores)
