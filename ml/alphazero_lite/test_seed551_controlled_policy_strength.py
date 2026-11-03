"""Synthetic recovery, command, trajectory, and analysis tests for seed551."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from ml.alphazero_lite import run_seed551_controlled_policy_strength as runner
from ml.alphazero_lite import seed551_controlled_policy_strength as protocol
from ml.alphazero_lite import verify_seed551_controlled_policy_strength as verifier
from ml.alphazero_lite.kalah_rules import KalahGame

ROOT = Path(__file__).resolve().parents[2]


def _opening() -> dict[str, Any]:
    return verifier.replay_suite(ROOT)[0]


def _synthetic_game(opening: dict[str, Any], seat: int) -> dict[str, Any]:
    game = KalahGame.from_state(protocol.suite.INITIAL_STATE)
    for move in opening["source_prefix_moves_absolute"]:
        assert game.move(int(move))
    trajectory: list[int] = []
    extra_turn = False
    while not game.over():
        relative = game.possible_moves()[0]
        absolute = game.pit_index(relative)
        mover = game.current_player
        trajectory.append(absolute)
        assert game.move(absolute)
        extra_turn |= game.current_player == mover
    margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    return {
        "game_index": seat,
        "opening_index": opening["opening_index"],
        "game_within_opening": seat,
        "challenger_player": seat,
        "margin": margin,
        "game_length": len(trajectory),
        "winner": winner,
        "trajectory": ",".join(map(str, trajectory)),
        "opening_prefix_moves": opening["prefix_moves"],
        "opening_state_hash": opening["state_hash"],
        "opening_applied_prefix_length": len(opening["prefix_moves"]),
        "opening_contract": "arena_player_relative_v2",
        "has_extra_turn": extra_turn,
    }


def test_command_construction_binds_index_seed_components_and_runtime(
    tmp_path: Path,
) -> None:
    output = tmp_path / "screen"
    opening_path = output / "batches/opening-0000.jsonl"
    opening_path.parent.mkdir(parents=True, exist_ok=True)
    source_line = (
        (ROOT / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)[0]
    )
    opening_path.write_bytes(source_line)
    record = runner.build_command(
        ROOT,
        output,
        0,
        {
            "registration_sha256": "registration",
            "suite_sha256": protocol.suite.OPENING_SHA256,
            "execution_source_sha256": {"runner": "source"},
        },
    )
    argv = record["argv"]
    assert argv[:3] == [
        runner.sys.executable,
        "-m",
        "ml.alphazero_lite.seed551_arena_entrypoint",
    ]
    assert argv[argv.index("--seed") + 1] == "545"
    assert argv[argv.index("--challenger-policy-artifact") + 1].endswith(
        "/runtime-artifacts/T"
    )
    assert argv[argv.index("--current-policy-artifact") + 1].endswith(
        "/runtime-artifacts/C"
    )
    assert argv[argv.index("--challenger-value-artifact") + 1].endswith(
        "/runtime-artifacts/initializer"
    )
    assert argv[argv.index("--current-value-artifact") + 1].endswith(
        "/runtime-artifacts/initializer"
    )
    assert argv[argv.index("--max-moves") + 1] == "200"
    assert (
        json.loads(argv[argv.index("--challenger-search-options-json") + 1])
        == protocol.SEARCH_OPTIONS
    )


def test_trajectory_replay_checks_owner_convention_extra_turn_and_winner() -> None:
    opening = next(
        row
        for row in verifier.replay_suite(ROOT)
        if _synthetic_game(row, 0)["has_extra_turn"]
    )
    game0 = _synthetic_game(opening, 0)
    game1 = _synthetic_game(opening, 1)
    assert game0["has_extra_turn"]
    assert verifier._game_outcome(opening, game0) == ("T", game0["margin"], False)
    assert verifier._game_outcome(opening, game1) == ("C", game1["margin"], False)
    altered = dict(game0)
    moves = [int(move) for move in altered["trajectory"].split(",")]
    root_game = KalahGame.from_state(opening["state"])
    wrong_owner = 6 * (1 - root_game.current_player)
    moves[0] = wrong_owner
    altered["trajectory"] = ",".join(map(str, moves))
    with pytest.raises(ValueError, match="illegal_action"):
        verifier._game_outcome(opening, altered)


def test_arena_batch_opening_index_is_local_even_for_suite_row_one() -> None:
    second_suite_row = verifier.replay_suite(ROOT)[1]
    record = _synthetic_game(second_suite_row, 0)
    record["opening_index"] = 0
    verifier._validate_arena_local_opening_index(record)
    record["opening_index"] = 1
    with pytest.raises(ValueError, match="arena_local_opening_index_mismatch"):
        verifier._validate_arena_local_opening_index(record)


def test_execution_source_correction_is_append_only_and_hash_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "verifier.py"
    source.write_text("corrected verifier\n")
    original_hash = "a" * 64
    corrected_hash = verifier.sha(source)
    out = tmp_path / "output"
    out.mkdir()
    registration_sha = "registration-sha"
    freeze = {
        "registration_sha256": registration_sha,
        "execution_source_sha256": {"verifier.py": original_hash},
    }
    correction = {
        "schema": "seed551_execution_source_correction_v1",
        "registration_sha256": registration_sha,
        "source_corrections": {
            "verifier.py": {
                "previous_sha256": original_hash,
                "corrected_sha256": corrected_hash,
            }
        },
    }
    correction_path = out / "execution-correction.json"
    correction_path.write_text(json.dumps(correction))
    freeze_correction = {
        "registration_sha256": registration_sha,
        "correction_sha256": verifier.sha(correction_path),
    }
    (out / "execution-freeze-correction.json").write_text(json.dumps(freeze_correction))

    verifier._verify_execution_source_corrections(tmp_path, out, freeze)

    correction["source_corrections"]["verifier.py"]["previous_sha256"] = "b" * 64
    correction_path.write_text(json.dumps(correction))
    freeze_correction["correction_sha256"] = verifier.sha(correction_path)
    (out / "execution-freeze-correction.json").write_text(json.dumps(freeze_correction))
    with pytest.raises(ValueError, match="correction_parent_mismatch"):
        verifier._verify_execution_source_corrections(tmp_path, out, freeze)


def test_relocated_command_paths_resolve_relative_to_registered_root(
    tmp_path: Path,
) -> None:
    command_path = Path(
        "docs/data/seed551-controlled-policy-strength-screen/runtime-artifacts/T"
    )
    assert (
        verifier._rooted_path(tmp_path, command_path)
        == (tmp_path / command_path).resolve()
    )


def test_recovery_classification_recovers_complete_raw_pairs_and_rejects_partials(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batches"
    batch.mkdir()
    assert runner.recovery_action(tmp_path, 0, set()) == "execute"
    (batch / "opening-0000.jsonl").write_text("opening\n")
    command_record = {"argv": [], "runtime": {"observed": runner.observed_runtime()}}
    (batch / "command-0000.json").write_text(json.dumps(command_record) + "\n")
    (batch / "games-0000.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="partial_pair_preserved"):
        runner.recovery_action(tmp_path, 0, set())
    (batch / "report-0000.json").write_text("{}\n")
    assert runner.recovery_action(tmp_path, 0, set()) == "recover"
    assert runner.recovery_action(tmp_path, 0, {0}) == "skip"
    (batch / "games-0000.jsonl").unlink()
    (batch / "report-0000.json").unlink()
    (batch / "attempt-0000.json").write_text('{"returncode":1}\n')
    with pytest.raises(ValueError, match="prior_attempt_preserved"):
        runner.recovery_action(tmp_path, 0, set())


def _runner_fixture(
    tmp_path: Path, *, ledger: bool
) -> tuple[Path, list[dict[str, Any]]]:
    out = tmp_path / "screen"
    batch = out / "batches"
    batch.mkdir(parents=True)
    opening = _opening()
    rows = [_synthetic_game(opening, seat) for seat in (0, 1)]
    pair = [
        {
            "opening_index": 0,
            "game_position": seat,
            "challenger_player": seat,
            "outcome": "T" if rows[seat]["margin"] > 0 else "C",
            "margin": rows[seat]["margin"],
            "capped_nonterminal": False,
            "game_sha256": "games",
            "report_sha256": "report",
            "command_sha256": "command",
        }
        for seat in (0, 1)
    ]
    source_line = (
        (ROOT / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)[0]
    )
    (batch / "opening-0000.jsonl").write_bytes(source_line)
    command_record = {"argv": [], "runtime": {"observed": runner.observed_runtime()}}
    (batch / "command-0000.json").write_text(json.dumps(command_record) + "\n")
    (batch / "games-0000.jsonl").write_text("{}\n")
    (batch / "report-0000.json").write_text("{}\n")
    freeze = {
        "registration_sha256": "registration",
        "suite_sha256": protocol.suite.OPENING_SHA256,
    }
    (out / "execution-freeze.json").write_text(json.dumps(freeze))
    (out / "command-preflight.jsonl").write_text("\n")
    (out / "registration.json").write_text(
        json.dumps({"runtime": {"observed": runner.observed_runtime()}})
    )
    if ledger:
        (out / "outcomes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in pair)
        )
    return out, pair


def _pair_evidence(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    out = tmp_path / "semantic-screen"
    batch = out / "batches"
    batch.mkdir(parents=True)
    opening = _opening()
    source_line = (
        (ROOT / protocol.suite.SEED511 / "openings.jsonl")
        .read_bytes()
        .splitlines(keepends=True)[0]
    )
    (batch / "opening-0000.jsonl").write_bytes(source_line)
    freeze = {
        "registration_sha256": "registration",
        "suite_sha256": protocol.suite.OPENING_SHA256,
        "execution_source_sha256": {"runner": "identity"},
    }
    command = runner.build_command(ROOT, out, 0, freeze)
    (out / "command-preflight.jsonl").write_text(json.dumps(command) + "\n")
    (batch / "command-0000.json").write_text(json.dumps(command) + "\n")
    (out / "commands.jsonl").write_text(json.dumps(command) + "\n")
    game_rows = [_synthetic_game(opening, seat) for seat in (0, 1)]
    games_path = batch / "games-0000.jsonl"
    games_path.write_text("".join(json.dumps(row) + "\n" for row in game_rows))
    sidecar = out / "runtime-artifacts/T/search_policy.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}\n")
    native_probe_hash = protocol.sha256(
        ROOT / "model-artifact/runtime/kalah_v1_tablebase"
    )
    tablebase_hash = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"
    runtime_policy_hash = protocol.sha256(sidecar)
    runtime_policy_mode = "puct_exact_root_hybrid"
    (out / "runtime-authentication.json").write_text(
        json.dumps(
            {
                "runtime_assets": {
                    "native_probe_sha256": native_probe_hash,
                    "tablebase_sha256": tablebase_hash,
                }
            }
        )
    )
    (out / "runtime-policy-preflight.json").write_text(
        json.dumps({"runtime_search_policy_mode": runtime_policy_mode})
    )
    winners = [row["winner"] for row in game_rows]
    report = {
        "schema": "arena_v1",
        "games_played": 2,
        "wins": winners.count("challenger"),
        "losses": winners.count("current"),
        "draws": winners.count("draw"),
        "notes": {
            "seed_contract": protocol.SEED_CONTRACT,
            "base_seed": 545,
            "random_opening_plies": 0,
            "workers_requested": 1,
            "workers_used": 1,
            "search_options": protocol.SEARCH_OPTIONS,
            "exact_root_solve_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solve_mode": "disabled",
            "exact_root_native_probe_sha256": native_probe_hash,
            "exact_root_tablebase_sha256": tablebase_hash,
            "runtime_search_policy_mode": runtime_policy_mode,
            "runtime_search_policy_sha256": runtime_policy_hash,
            "challenger_runtime_policy_inherited": False,
            "search_profile": {
                "simulations": 128,
                "challenger_simulations": 128,
                "current_simulations": 128,
                "c_puct": 1.25,
                "workers_requested": 1,
                "workers_used": 1,
                "exact_root_solve_enabled": True,
                "exact_root_solve_threshold": 16,
                "exact_root_solver": "native_kvtb_root_action_probe_v1",
                "exact_root_objective": "final_score_margin",
                "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
                "exact_leaf_solve_mode": "disabled",
                "exact_root_native_probe_sha256": native_probe_hash,
                "exact_root_tablebase_sha256": tablebase_hash,
                "runtime_search_policy_mode": runtime_policy_mode,
                "runtime_search_policy_sha256": runtime_policy_hash,
                "challenger_runtime_policy_inherited": False,
                "search_options": protocol.SEARCH_OPTIONS,
            },
        },
    }
    report_path = batch / "report-0000.json"
    report_path.write_text(json.dumps(report) + "\n")
    (out / "execution-freeze.json").write_text(json.dumps(freeze))
    (out / "registration.json").write_text(json.dumps({"runtime": command["runtime"]}))
    attempt = {
        "opening_index": 0,
        "started": True,
        "returncode": 0,
        "command_sha256": protocol.sha256(batch / "command-0000.json"),
        "observed_runtime": command["runtime"]["observed"],
    }
    (batch / "attempt-0000.json").write_text(json.dumps(attempt))
    archive = batch / "attempts/opening-0000"
    archive.mkdir(parents=True)
    (archive / "attempt-00.json").write_text(
        json.dumps(
            {
                **attempt,
                "attempt_number": 0,
                "argv": command["argv"],
                "games_path_exists": True,
                "report_path_exists": True,
            }
        )
    )
    completion = {
        "opening_index": 0,
        "registration_sha256": freeze["registration_sha256"],
        "suite_sha256": freeze["suite_sha256"],
        "opening_sha256": protocol.sha256(batch / "opening-0000.jsonl"),
        "command_sha256": protocol.sha256(batch / "command-0000.json"),
        "games_sha256": protocol.sha256(games_path),
        "report_sha256": protocol.sha256(report_path),
    }
    (batch / "completion-0000.json").write_text(json.dumps(completion))
    return out, completion


def _rebind_completion(
    out: Path, completion: dict[str, Any], field: str, path: Path
) -> None:
    completion[field] = protocol.sha256(path)
    (out / "batches/completion-0000.json").write_text(json.dumps(completion))


def test_receipt_rebound_semantic_corruption_rejects_opening_command_runtime_and_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, completion = _pair_evidence(tmp_path)
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})
    assert len(verifier.validate_pair(ROOT, out, 0)) == 2

    command_path = out / "batches/command-0000.json"
    command = json.loads(command_path.read_text())
    command["argv"][command["argv"].index("--c-puct") + 1] = "1.5"
    command_path.write_text(json.dumps(command))
    _rebind_completion(out, completion, "command_sha256", command_path)
    with pytest.raises(ValueError, match="command_not_frozen"):
        verifier.validate_pair(ROOT, out, 0)

    command_path.write_text((out / "command-preflight.jsonl").read_text())
    command_path.write_text(
        json.dumps(json.loads((out / "command-preflight.jsonl").read_text()))
    )
    command_path.write_text((out / "command-preflight.jsonl").read_text().strip())
    frozen_command = json.loads((out / "command-preflight.jsonl").read_text())
    command_path.write_text(json.dumps(frozen_command))
    _rebind_completion(out, completion, "command_sha256", command_path)
    attempt_path = out / "batches/attempt-0000.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["command_sha256"] = protocol.sha256(command_path)
    attempt_path.write_text(json.dumps(attempt))
    archive_path = out / "batches/attempts/opening-0000/attempt-00.json"
    archived = json.loads(archive_path.read_text())
    archived["command_sha256"] = protocol.sha256(command_path)
    archive_path.write_text(json.dumps(archived))

    report_path = out / "batches/report-0000.json"
    report = json.loads(report_path.read_text())
    report["notes"]["search_profile"]["exact_root_objective"] = "win_loss"
    report_path.write_text(json.dumps(report))
    _rebind_completion(out, completion, "report_sha256", report_path)
    with pytest.raises(ValueError, match="runtime_report_mismatch"):
        verifier.validate_pair(ROOT, out, 0)

    # Restore the frozen report, then mutate a trajectory and rebind the receipt.
    report["notes"]["search_profile"]["exact_root_objective"] = "final_score_margin"
    report_path.write_text(json.dumps(report))
    _rebind_completion(out, completion, "report_sha256", report_path)
    games_path = out / "batches/games-0000.jsonl"
    games = [json.loads(line) for line in games_path.read_text().splitlines()]
    illegal = 6 * (1 - _opening()["state"]["current_player"])
    trajectory = [int(move) for move in games[0]["trajectory"].split(",")]
    trajectory[0] = illegal
    games[0]["trajectory"] = ",".join(map(str, trajectory))
    games_path.write_text("".join(json.dumps(row) + "\n" for row in games))
    _rebind_completion(out, completion, "games_sha256", games_path)
    with pytest.raises(ValueError, match="illegal_action"):
        verifier.validate_pair(ROOT, out, 0)

    opening_path = out / "batches/opening-0000.jsonl"
    opening_path.write_bytes(opening_path.read_bytes() + b" ")
    _rebind_completion(out, completion, "opening_sha256", opening_path)
    with pytest.raises(ValueError, match="opening_file_mismatch"):
        verifier.validate_pair(ROOT, out, 0)


def test_receipt_rebound_ledger_and_analysis_corruption_are_reconstructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, _completion = _pair_evidence(tmp_path)
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", 32)
    monkeypatch.setattr(verifier, "OPENINGS", 1)
    monkeypatch.setattr(verifier, "GAMES", 2)
    monkeypatch.setattr(verifier, "BOOTSTRAP_RESAMPLES", 32)
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})
    pair = verifier.validate_pair(ROOT, out, 0)
    ledger_path = out / "outcomes.jsonl"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in pair))
    results, per_opening = protocol.analyze(pair)
    (out / "results.json").write_text(json.dumps(results))
    (out / "per-opening-results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_opening)
    )
    assert verifier.verify(ROOT, out)["status"] == "verified_complete"

    corrupted = [dict(row) for row in pair]
    corrupted[0]["margin"] += 1
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in corrupted))
    with pytest.raises(ValueError, match="ledger_pair_reconstruction_mismatch"):
        verifier.verify(ROOT, out)

    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in pair))
    results["score"] = 0.75
    (out / "results.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="analysis_reconstruction_mismatch"):
        verifier.verify(ROOT, out)


@pytest.mark.parametrize(
    ("report_section", "field", "replacement"),
    [
        (section, field, bad)
        for section in ("notes", "search_profile")
        for field, bad in (
            ("exact_root_native_probe_sha256", "0" * 64),
            ("exact_root_tablebase_sha256", "1" * 64),
            ("runtime_search_policy_mode", "different_runtime_policy"),
        )
    ],
)
def test_full_verifier_rejects_receipt_rebound_raw_runtime_identity_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report_section: str,
    field: str,
    replacement: str,
) -> None:
    out, completion = _pair_evidence(tmp_path)
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    monkeypatch.setattr(protocol, "BOOTSTRAP_RESAMPLES", 32)
    monkeypatch.setattr(verifier, "OPENINGS", 1)
    monkeypatch.setattr(verifier, "GAMES", 2)
    monkeypatch.setattr(verifier, "BOOTSTRAP_RESAMPLES", 32)
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})

    pair = verifier.validate_pair(ROOT, out, 0)
    ledger_path = out / "outcomes.jsonl"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in pair))
    results, per_opening = protocol.analyze(pair)
    (out / "results.json").write_text(json.dumps(results))
    (out / "per-opening-results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_opening)
    )

    report_path = out / "batches/report-0000.json"
    report = json.loads(report_path.read_text())
    if report_section == "search_profile":
        report["notes"]["search_profile"][field] = replacement
    else:
        report["notes"][field] = replacement
    report_path.write_text(json.dumps(report) + "\n")
    _rebind_completion(out, completion, "report_sha256", report_path)
    rebound_ledger = [dict(row) for row in pair]
    for row in rebound_ledger:
        row["report_sha256"] = protocol.sha256(report_path)
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rebound_ledger))

    with pytest.raises(ValueError, match="runtime_report_identity_mismatch"):
        verifier.verify(ROOT, out)


def test_resume_revalidates_and_skips_committed_pairs_without_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, pair = _runner_fixture(tmp_path, ledger=True)
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    monkeypatch.setattr(runner, "restore_interrupted_tablebase", lambda _root: None)
    monkeypatch.setattr(runner, "assert_runtime_contract", lambda _expected=None: {})
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})
    validate = Mock(return_value=pair)
    monkeypatch.setattr(verifier, "validate_pair", validate)
    launched = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launched)
    monkeypatch.setattr(
        protocol, "analyze", lambda _ledger: ({"decision": "synthetic"}, [])
    )

    result = runner.execute(ROOT, out)

    assert result["status"] == "complete"
    validate.assert_called_once_with(
        ROOT.resolve(), out.resolve(), 0, preflight_verified=True
    )
    launched.assert_not_called()


def test_complete_saved_raw_pair_is_validated_and_recovered_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, pair = _runner_fixture(tmp_path, ledger=False)
    monkeypatch.setattr(protocol, "OPENINGS", 1)
    monkeypatch.setattr(protocol, "GAMES", 2)
    monkeypatch.setattr(runner, "restore_interrupted_tablebase", lambda _root: None)
    monkeypatch.setattr(runner, "assert_runtime_contract", lambda _expected=None: {})
    monkeypatch.setattr(verifier, "verify_preflight", lambda *_args, **_kwargs: {})
    validate = Mock(return_value=pair)
    monkeypatch.setattr(verifier, "validate_pair", validate)
    launched = Mock()
    monkeypatch.setattr(runner.subprocess, "run", launched)
    monkeypatch.setattr(
        protocol, "analyze", lambda _ledger: ({"decision": "synthetic"}, [])
    )

    result = runner.execute(ROOT, out)

    assert result["status"] == "complete"
    validate.assert_called_once_with(
        ROOT.resolve(), out.resolve(), 0, preflight_verified=True
    )
    launched.assert_not_called()
    assert (out / "batches/completion-0000.json").is_file()
    assert len((out / "outcomes.jsonl").read_text().splitlines()) == 2


def test_opening_cluster_bootstrap_preserves_caps_as_separate_records() -> None:
    ledger = [
        {
            "opening_index": index,
            "game_position": seat,
            "outcome": "T",
            "capped_nonterminal": index == 0 and seat == 0,
        }
        for index in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    result, clusters = protocol.analyze(ledger)
    assert len(clusters) == protocol.OPENINGS
    assert result["score"] == 1.0
    assert result["capped_nonterminal_games"] == 1
    assert result["capped_nonterminal_results_by_current_store_margin"] == {
        "T": 1,
        "draw": 0,
        "C": 0,
    }
    assert result["decision"] == "advance_to_separate_serving_P_screen"


def test_analysis_rejects_missing_or_duplicate_seat_evidence() -> None:
    ledger = [
        {"opening_index": index, "game_position": seat, "outcome": "draw"}
        for index in range(protocol.OPENINGS)
        for seat in (0, 1)
    ]
    ledger.pop()
    with pytest.raises(ValueError, match="requires_all_2048"):
        protocol.analyze(ledger)
