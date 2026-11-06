"""Focused contracts for seed555 registration, analysis, and recovery."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import arena
from ml.alphazero_lite import run_seed555_policy_strength_screen as runner
from ml.alphazero_lite import seed555_policy_strength_screen as protocol
from ml.alphazero_lite import verify_seed551_controlled_policy_strength as prior_verify
from ml.alphazero_lite import verify_seed555_policy_strength_screen as verify


def test_seed554_full_replay_receipt_and_ledger_are_bound_to_training_input() -> None:
    corpus = protocol.ROOT / protocol.SEED554
    binding = json.loads((corpus / "training-input-binding.json").read_text())
    assert (
        protocol.sha256(corpus / "searches.jsonl") == binding["search_evidence_sha256"]
    )
    assert (
        protocol.sha256(corpus / "derivative-verification.json")
        == binding["derivative_verification_sha256"]
    )
    receipt = json.loads((corpus / "derivative-verification.json").read_text())
    assert receipt["rerun_searches"] is True
    assert receipt["paired_searches"] == 11_270


def test_indexed_suite_loader_returns_all_replayed_opening_rows() -> None:
    raw, openings = protocol.load_openings(protocol.ROOT)
    assert (
        protocol.sha256(protocol.ROOT / protocol.OPENINGS_PATH)
        == protocol.OPENINGS_SHA256
    )
    assert len(raw.splitlines()) == protocol.OPENINGS
    assert len(openings) == protocol.OPENINGS
    assert all(row.get("opening_index") == index for index, row in enumerate(openings))


def test_composed_evaluators_use_each_policy_trunk_and_shared_initializer_value() -> (
    None
):
    initializer, t512, c128 = object(), object(), object()
    challenger = arena.ComposedArtifactEvaluator(
        initializer, t512, policy_source="candidate", value_source="current"
    )
    opponent = arena.ComposedArtifactEvaluator(
        initializer, c128, policy_source="candidate", value_source="current"
    )
    assert challenger.candidate_evaluator is t512
    assert challenger.current_evaluator is initializer
    assert opponent.candidate_evaluator is c128
    assert opponent.current_evaluator is initializer


def test_runtime_authentication_cannot_mutate_cached_live_runtime_contract(
    monkeypatch,
) -> None:
    base = {"observed": {"cpu_affinity": [30]}, "resource_limits": {"workers": 1}}
    monkeypatch.setattr(runner.runtime, "runtime_settings", lambda: base)
    monkeypatch.setattr(
        runner,
        "_read_live_cpu_max",
        lambda: ("/sys/fs/cgroup/unit.service/cpu.max", "25000 100000"),
    )

    def fake_sha(path: Path) -> str:
        if path == protocol.ROOT / protocol.TABLEBASE_PATH:
            return protocol.TABLEBASE_POINTER_SHA256
        if path.resolve() == runner.TABLEBASE.resolve():
            return protocol.TABLEBASE_SHA256
        return "native-probe-sha"

    monkeypatch.setattr(protocol, "sha256", fake_sha)
    auth = runner._runtime(protocol.ROOT)
    auth["components"] = {"test": True}
    auth["component_assignments"] = {"mutated": True}

    assert "components" not in base
    assert "component_assignments" not in base
    assert "components" not in runner._runtime(protocol.ROOT)


def test_global_opening_index_drives_command_seed_but_arena_row_is_local_zero(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    line = (
        (protocol.ROOT / protocol.OPENINGS_PATH)
        .read_bytes()
        .splitlines(keepends=True)[7]
    )
    (batch / "opening-0007.jsonl").write_bytes(line)
    registration = {"registration_sha256": "registration"}
    command = runner.build_command(
        protocol.ROOT,
        tmp_path,
        7,
        registration,
        {"executable": runner.sys.executable},
    )
    argv = command["argv"]
    assert argv[1:3] == ["-m", "ml.alphazero_lite.seed555_arena_entrypoint"]
    assert argv[argv.index("--seed") + 1] == "552"
    assert argv[argv.index("--challenger-policy-artifact") + 1].endswith("/T512")
    assert argv[argv.index("--current-policy-artifact") + 1].endswith("/C128")
    assert command["opening_index"] == 7
    prior_verify._validate_arena_local_opening_index({"opening_index": 0})
    with pytest.raises(ValueError, match="arena_local_opening_index"):
        prior_verify._validate_arena_local_opening_index({"opening_index": 7})


def test_global_suite_row_zero_and_pair_seats_replay_extra_turn_and_winner_perspective() -> (
    None
):
    openings = prior_verify.replay_suite(protocol.ROOT)
    opening = openings[7]
    from ml.alphazero_lite.kalah_rules import KalahGame

    game = KalahGame.from_state(opening["state"])
    moves: list[int] = []
    extra_turn = False
    while not game.over():
        relative = game.possible_moves()[0]
        absolute = game.pit_index(relative)
        mover = game.current_player
        moves.append(absolute)
        assert game.move(absolute)
        extra_turn |= game.current_player == mover
    assert extra_turn
    for seat in (0, 1):
        margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
        row = {
            "challenger_player": seat,
            "opening_prefix_moves": opening["prefix_moves"],
            "opening_state_hash": opening["state_hash"],
            "opening_applied_prefix_length": len(opening["prefix_moves"]),
            "opening_contract": "arena_player_relative_v2",
            "trajectory": ",".join(map(str, moves)),
            "game_length": len(moves),
            "margin": margin,
            "winner": "challenger"
            if margin > 0
            else "current"
            if margin < 0
            else "draw",
        }
        result, replayed_margin, capped = prior_verify._game_outcome(opening, row)
        assert result == ("T" if margin > 0 else "C" if margin < 0 else "draw")
        assert replayed_margin == margin
        assert not capped


def test_analysis_bootstraps_opening_clusters_and_capped_games_block_decision() -> None:
    ledger = [
        {
            "opening_index": opening,
            "game_position": seat,
            "outcome": "T512",
            "capped_nonterminal": False,
        }
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    result, scores = protocol.analyze(ledger)
    assert result["score"] == 1.0
    assert result["one_sided_95_lower"] == 1.0
    assert result["decision"] == "propose_T512_vs_serving_P_screen"
    assert len(scores) == protocol.OPENINGS
    ledger[0]["outcome"] = "draw"
    ledger[0]["capped_nonterminal"] = True
    result, _ = protocol.analyze(ledger)
    assert result["capped_nonterminal_games"] == 1
    assert result["score"] is None
    assert result["one_sided_95_lower"] is None
    assert result["decision"] == "no_valid_advancement_decision"


def test_receipt_rebound_analysis_semantic_corruption_is_rejected() -> None:
    ledger = [
        {
            "opening_index": opening,
            "game_position": seat,
            "outcome": "T512",
            "capped_nonterminal": False,
        }
        for opening in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    saved, scores = protocol.analyze(ledger)
    rebuilt, rebuilt_scores = verify.reconstruct_analysis(ledger)
    verify.compare_analysis(saved, rebuilt, scores, rebuilt_scores)
    rebound = {**saved, "decision": "do_not_advance_teacher_budget_branch"}
    with pytest.raises(ValueError, match="analysis_reconstruction"):
        verify.compare_analysis(rebound, rebuilt, scores, rebuilt_scores)
    altered_scores = [dict(scores[0], seat0_T512_score=0.0), *scores[1:]]
    with pytest.raises(ValueError, match="opening_score_reconstruction"):
        verify.compare_analysis(saved, rebuilt, altered_scores, rebuilt_scores)


def test_recovery_skips_commits_recovers_complete_raw_pair_and_preserves_partial(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    (batch / "opening-0000.jsonl").write_text("opening\n")
    assert runner.recovery_action(batch, 0, {0}) == "skip"
    assert runner.recovery_action(batch, 0, set()) == "execute"
    for name, contents in (
        ("command-0000.json", "{}\n"),
        ("games-0000.jsonl", "{}\n{}\n"),
        ("report-0000.json", "{}\n"),
    ):
        (batch / name).write_text(contents)
    assert runner.recovery_action(batch, 0, set()) == "recover"
    (batch / "command-0001.json").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete_prior_evidence_preserved"):
        runner.recovery_action(batch, 1, set())


def test_command_semantics_reject_assignment_or_arena_drift(tmp_path: Path) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    (batch / "opening-0000.jsonl").write_text("{}\n")
    command = runner.build_command(
        protocol.ROOT,
        tmp_path,
        0,
        {"registration_sha256": "registration"},
        {
            "executable": runner.sys.executable,
            "runtime_assets": {
                "tablebase_runtime_path": str(runner.TABLEBASE.resolve())
            },
        },
    )
    verify.validate_command(command, tmp_path, 0)
    changed = dict(command)
    changed["component_assignments"] = {**protocol.ASSIGNMENTS, "opponent": "T512"}
    with pytest.raises(ValueError, match="assignment"):
        verify.validate_command(changed, tmp_path, 0)
    changed = dict(command)
    changed["argv"] = list(command["argv"])
    changed["argv"][changed["argv"].index("--current-simulations") + 1] = "512"
    with pytest.raises(ValueError, match="arena_option"):
        verify.validate_command(changed, tmp_path, 0)


def test_relocated_copy_command_validation_uses_frozen_execution_paths(
    tmp_path: Path,
) -> None:
    registered_output = tmp_path / "original-root" / protocol.OUT
    copied_output = tmp_path / "physical-copy" / protocol.OUT
    for output in (registered_output, copied_output):
        (output / "batches").mkdir(parents=True)
        line = (
            (protocol.ROOT / protocol.OPENINGS_PATH)
            .read_bytes()
            .splitlines(keepends=True)[0]
        )
        (output / "batches/opening-0000.jsonl").write_bytes(line)
    runtime_identity = {
        "executable": runner.sys.executable,
        "runtime_assets": {"tablebase_runtime_path": str(runner.TABLEBASE.resolve())},
    }
    command = runner.build_command(
        protocol.ROOT,
        registered_output,
        0,
        {"registration_sha256": "registration"},
        runtime_identity,
    )
    verify.validate_command(
        command, copied_output, 0, registered_output=registered_output
    )


def test_preflight_rejection_does_not_launch_or_mutate(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    output = root / protocol.OUT
    output.mkdir(parents=True)
    (output / "registration.json").write_text('{"execution_root":"wrong"}\n')
    (output / "execution-freeze.json").write_text("{}\n")
    before = {
        p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        runner.execute(root, output)
    launch.assert_not_called()
    after = {
        p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_runtime_mismatch_rejected_by_execute_before_writes_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "root"
    output = root / protocol.OUT
    output.mkdir(parents=True)
    (output / "registration.json").write_text('{"frozen":true}\n')
    before = {
        p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    expected_runtime = {
        "observed": {"cpu_affinity": [30]},
        "cgroup_scope_diagnostic": {"limits": {"cpu.max": "25000 100000"}},
    }
    monkeypatch.setattr(
        verify,
        "verify_preflight",
        lambda *_: {"registration": {"runtime": expected_runtime}},
    )
    mismatch = Mock(side_effect=ValueError("altered_runtime_binding"))
    monkeypatch.setattr(runner.runtime, "assert_runtime_contract", mismatch)
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)
    with pytest.raises(ValueError, match="altered_runtime_binding"):
        runner.execute(root, output)
    mismatch.assert_called_once_with(expected_runtime["observed"])
    launch.assert_not_called()
    after = {
        p.relative_to(output).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_execute_recovers_saved_raw_pair_after_interruption_without_rerunning(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    root = tmp_path / "root"
    output = root / protocol.OUT
    batch = output / "batches"
    batch.mkdir(parents=True)
    (output / "registration.json").write_text("{}\n")
    (output / "command-preflight.jsonl").write_text('{"opening_index":0}\n')
    (batch / "opening-0000.jsonl").write_text("opening\n")
    (batch / "command-0000.json").write_text('{"argv":["arena"]}\n')
    raw_games = b'{"game":0}\n{"game":1}\n'
    raw_report = b'{"games_played":2}\n'
    (batch / "games-0000.jsonl").write_bytes(raw_games)
    (batch / "report-0000.json").write_bytes(raw_report)
    pair = [
        {"opening_index": 0, "game_position": seat, "outcome": "T512"}
        for seat in (0, 1)
    ]
    expected_runtime = {
        "observed": {"cpu_affinity": [30]},
        "cgroup_scope_diagnostic": {"limits": {"cpu.max": "25000 100000"}},
    }
    monkeypatch.setattr(
        verify,
        "verify_preflight",
        lambda *_: {"registration": {"runtime": expected_runtime}},
    )
    monkeypatch.setattr(verify, "validate_execution_runtime_checks", lambda *_: [])
    monkeypatch.setattr(verify, "validate_pair", lambda *_: pair)
    monkeypatch.setattr(verify, "verify", lambda *_: {"status": "verified_complete"})
    monkeypatch.setattr(runner.runtime, "assert_runtime_contract", lambda *_: None)
    monkeypatch.setattr(
        runner.runtime, "observed_runtime", lambda: expected_runtime["observed"]
    )
    monkeypatch.setattr(runner, "_runtime", lambda *_: dict(expected_runtime))
    monkeypatch.setattr(protocol, "analyze", lambda _rows: ({"status": "complete"}, []))
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)

    runner.execute(root, output)

    launch.assert_not_called()
    assert (batch / "completion-0000.json").is_file()
    assert (batch / "games-0000.jsonl").read_bytes() == raw_games
    assert (batch / "report-0000.json").read_bytes() == raw_report
    assert len((output / "outcomes.jsonl").read_text().splitlines()) == 2


def test_execute_preserves_conflicting_ledger_before_runtime_receipt_or_launch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    root = tmp_path / "root"
    output = root / protocol.OUT
    batch = output / "batches"
    batch.mkdir(parents=True)
    (output / "registration.json").write_text("{}\n")
    (output / "command-preflight.jsonl").write_text('{"opening_index":0}\n')
    (batch / "opening-0000.jsonl").write_text("opening\n")
    conflicting = b'{"opening_index":0,"game_position":0,"outcome":"C128"}\n'
    ledger = output / "outcomes.jsonl"
    ledger.write_bytes(conflicting)
    runtime_contract = {
        "observed": {"cpu_affinity": [30]},
        "cgroup_scope_diagnostic": {"limits": {"cpu.max": "25000 100000"}},
    }
    monkeypatch.setattr(
        verify,
        "verify_preflight",
        lambda *_: {"registration": {"runtime": runtime_contract}},
    )
    monkeypatch.setattr(verify, "validate_execution_runtime_checks", lambda *_: [])
    monkeypatch.setattr(runner.runtime, "assert_runtime_contract", lambda *_: None)
    monkeypatch.setattr(
        runner.runtime, "observed_runtime", lambda: runtime_contract["observed"]
    )
    monkeypatch.setattr(runner, "_runtime", lambda *_: dict(runtime_contract))
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launch)
    before = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }

    with pytest.raises(ValueError, match="conflicting_ledger_preserved"):
        runner.execute(root, output)

    launch.assert_not_called()
    after = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (output / "execution-runtime-checks.jsonl").exists()


def test_ledger_recovery_receipt_and_conflicting_ledger_preservation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "screen"
    batches = output / "batches"
    batches.mkdir(parents=True)
    pair = [
        {
            "opening_index": 0,
            "game_position": seat,
            "outcome": "T512" if seat == 0 else "C128",
        }
        for seat in (0, 1)
    ]
    completion = batches / "completion-0000.json"
    completion.write_text("{}\n")
    ledger = output / "outcomes.jsonl"
    runner._reconcile_ledger(output, ledger, pair, allow_recovery=True)
    receipt = json.loads((output / "ledger-recovery/recovery-0001.json").read_text())
    assert receipt["validated_opening_pairs"] == 1
    assert receipt["validated_games"] == 2
    assert receipt["validated_completion_receipt_sha256"] == {
        "0": protocol.sha256(completion)
    }

    conflict = json.dumps({"opening_index": 1, "game_position": 0}).encode() + b"\n"
    ledger.write_bytes(conflict)
    before = (ledger.read_bytes(), ledger.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="conflicting_ledger_preserved"):
        runner._reconcile_ledger(output, ledger, pair)
    assert (ledger.read_bytes(), ledger.stat().st_mtime_ns) == before


def test_runtime_receipt_rebinding_cannot_hide_wrong_cpu_quota(tmp_path: Path) -> None:
    output = tmp_path / "screen"
    output.mkdir()
    registration = {"schema": "fixture"}
    (output / "registration.json").write_text(json.dumps(registration) + "\n")
    expected = {
        "executable": runner.sys.executable,
        "observed": {"cpu_affinity": [30]},
        "cgroup_scope_diagnostic": {
            "limits": {
                "cpu.max": "25000 100000",
                "cpu.weight": "1",
                "memory.max": "4294967296",
            }
        },
    }
    live = {
        key: value
        for key, value in expected.items()
        if key != "cgroup_scope_diagnostic"
    }
    receipt = {
        "registration_sha256": protocol.sha256(output / "registration.json"),
        "runtime_contract_sha256": hashlib.sha256(
            protocol.stable_json(expected)
        ).hexdigest(),
        "runtime_identity": live,
        "cgroup_scope_diagnostic": {
            "cpu.max_path": "/sys/fs/cgroup/unit.service/cpu.max",
            "limits": expected["cgroup_scope_diagnostic"]["limits"],
        },
        "status": "prelaunch_runtime_contract_passed",
        "arena_games_started_by_check": False,
    }
    path = output / "execution-runtime-checks.jsonl"
    path.write_bytes(protocol.stable_json(receipt) + b"\n")
    assert verify.validate_execution_runtime_checks(output, expected)
    receipt["cgroup_scope_diagnostic"]["limits"]["cpu.max"] = "50000 100000"
    path.write_bytes(protocol.stable_json(receipt) + b"\n")
    with pytest.raises(ValueError, match="runtime_receipt_mismatch"):
        verify.validate_execution_runtime_checks(output, expected)
