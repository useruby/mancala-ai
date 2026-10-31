"""Tests for the independently imported seed511 suite and screen bindings."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from ml.alphazero_lite import seed542_expanded_policy_strength_screen as screen
from ml.alphazero_lite import run_seed542_expanded_policy_strength_screen as runner
from ml.alphazero_lite import verify_seed542_expanded_policy_strength_screen as verifier
from ml.alphazero_lite.kalah_rules import KalahGame


ROOT = Path(__file__).resolve().parents[2]


def imported_root(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    suite = root / screen.SEED511
    suite.mkdir(parents=True)
    for name in ("registration.json", "execution-freeze.json", "openings.jsonl"):
        shutil.copy2(ROOT / screen.SEED511 / name, suite / name)
    for relative in screen.SUITE_SOURCE_HASHES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    native = root / "ml/alphazero_lite/native_exact_root_tablebase.py"
    shutil.copy2(ROOT / "ml/alphazero_lite/native_exact_root_tablebase.py", native)
    return root


def test_suite_import_authenticates_and_replays_all_openings(tmp_path: Path) -> None:
    root = imported_root(tmp_path)
    openings, report = screen.import_seed511_suite(root)

    assert len(openings) == 1024
    assert report["canonical_unique"] == 1024
    assert report["production_float32_unique"] == 1024
    assert report["root_player_orientations"] == {"0": 524, "1": 500}
    assert report["historical_games_or_outcomes_imported"] is False
    assert report["historical_full_seed511_publication_verifier"]["status"].endswith(
        "native_source_mismatch"
    )


def test_import_checks_registration_freeze_hash_chain(tmp_path: Path) -> None:
    root = imported_root(tmp_path)
    suite = root / screen.SEED511 / "openings.jsonl"
    suite.write_bytes(suite.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="freeze_or_suite_hash_mismatch"):
        screen.import_seed511_suite(root)


def test_import_rejects_changed_rules_or_encoding_source(tmp_path: Path) -> None:
    root = imported_root(tmp_path)
    source = root / "ml/alphazero_lite/input_encodings.py"
    source.write_bytes(source.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        screen.import_seed511_suite(root)


def test_runtime_candidate_and_serving_composition_authenticate_independently() -> None:
    output = ROOT / screen.OUT
    registration = json.loads((output / "registration.json").read_text())
    verifier._authenticate_composition(ROOT, output, registration)
    altered = json.loads(json.dumps(registration))
    altered["component_assignments"]["opponent_policy"] = "D-final"
    with pytest.raises(ValueError, match="component_assignment_contract_mismatch"):
        verifier._authenticate_composition(ROOT, output, altered)


def test_saved_opening0_raw_pair_gets_separate_completion_receipt_without_rerun(
    tmp_path: Path,
) -> None:
    source = ROOT / screen.OUT
    output = tmp_path / screen.OUT
    batch = output / "batches"
    batch.mkdir(parents=True)
    for name in (
        "registration.json",
        "execution-freeze.json",
        "runtime-authentication.json",
        "openings.jsonl",
    ):
        shutil.copy2(source / name, output / name)
    for name in (
        "opening-0000.jsonl",
        "command-0000.json",
        "games-0000.jsonl",
        "report-0000.json",
    ):
        shutil.copy2(source / "batches" / name, batch / name)
    original_command_bytes = (batch / "command-0000.json").read_bytes()
    game_hash = runner.sha(batch / "games-0000.jsonl")
    report_hash = runner.sha(batch / "report-0000.json")

    completion = runner.write_output_completion_receipt(tmp_path, output, 0)

    assert completion.is_file()
    assert (batch / "command-0000.json").read_bytes() == original_command_bytes
    receipt = json.loads(completion.read_text())
    assert receipt["games_sha256"] == game_hash
    assert receipt["report_sha256"] == report_hash
    assert len(verifier.validate_pair(tmp_path, output, 0)) == 2


def test_saved_opening1_pair_replays_arena_local_suite_index(tmp_path: Path) -> None:
    source = ROOT / screen.OUT
    output = tmp_path / screen.OUT
    batch = output / "batches"
    batch.mkdir(parents=True)
    for name in (
        "registration.json",
        "execution-freeze.json",
        "continuation-registration-v2.json",
        "continuation-execution-freeze-v2.json",
    ):
        shutil.copy2(source / name, output / name)
    for name in (
        "opening-0001.jsonl",
        "command-0001.json",
        "games-0001.jsonl",
        "report-0001.json",
    ):
        shutil.copy2(source / "batches" / name, batch / name)

    pair = verifier.validate_pair(tmp_path, output, 1, require_completion_receipt=False)

    assert len(pair) == 2
    assert {row["challenger_player"] for row in pair} == {0, 1}
    assert all(row["opening_index"] == 1 for row in pair)


def test_training_v2_receipt_requires_absolute_authenticated_paths(
    tmp_path: Path,
) -> None:
    execution_root = tmp_path.resolve()
    modules = {
        "verify_seed542_fixed_update_cd_v2": "verified_complete",
        "verify_seed542_fixed_update_cd_v2_supplement": "supplement_verified",
    }
    receipt = {
        "schema": "seed542_read_only_training_verification_receipt_v2",
        "status": "both_seed542_v2_verifiers_passed",
        "execution_root": str(execution_root),
        "runtime_settings": {
            "cpu_quota": "25%",
            "cpu_weight": "1",
            "memory_max": "4294967296",
            "allowed_cpus": "30",
            "nice": 19,
            "address_space_limit": "17179869184",
            "threads_per_numerical_library": 1,
        },
        "verifiers": {},
    }
    for module, status in modules.items():
        receipt["verifiers"][module] = {
            "command": [
                "systemd-run",
                "--scope",
                "CPUQuota=25%",
                "MemoryMax=4294967296",
                "AllowedCPUs=30",
                "nice",
                "19",
                "--as=17179869184",
                "-c",
                "30",
                "OMP_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "-m",
                f"ml.alphazero_lite.{module}",
                "--root",
                str(execution_root),
                "--out",
                str(execution_root / "docs/data/seed542-fixed-update-cd-v2"),
            ],
            "cwd": str(execution_root),
            "returncode": 0,
            "status": status,
            "stdout": f'{{"status": "{status}"}}',
        }
    output = tmp_path / screen.OUT
    output.mkdir(parents=True)
    (output / "training-verification-v2.json").write_text(json.dumps(receipt))
    registration = {"execution_root": str(execution_root)}
    verifier._validate_training_receipt(execution_root, output, registration)

    receipt["verifiers"]["verify_seed542_fixed_update_cd_v2"]["command"][
        receipt["verifiers"]["verify_seed542_fixed_update_cd_v2"]["command"].index(
            "--root"
        )
        + 1
    ] = "."
    (output / "training-verification-v2.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="training_root_mismatch"):
        verifier._validate_training_receipt(execution_root, output, registration)


def test_import_rejects_legal_prefix_state_identity_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = imported_root(tmp_path)
    suite = root / screen.SEED511 / "openings.jsonl"
    rows = screen.read_jsonl(suite)
    rows[0]["state"]["player_store"] += 1
    suite.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    registration_path = root / screen.SEED511 / "registration.json"
    freeze_path = root / screen.SEED511 / "execution-freeze.json"
    registration = json.loads(registration_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    digest = screen.sha256_file(suite)
    registration["suite"]["sha256"] = digest
    freeze["suite_sha256"] = digest
    monkeypatch.setattr(screen, "OPENING_SHA256", digest)
    registration_path.write_text(json.dumps(registration))
    freeze["registration_sha256"] = screen.sha256_file(registration_path)
    freeze_path.write_text(json.dumps(freeze))

    with pytest.raises(
        ValueError, match="root_state_mismatch|root_state_hash_mismatch"
    ):
        screen.import_seed511_suite(root)


def test_suite_prefixes_include_extra_turn_and_both_player_orientations() -> None:
    openings, _ = screen.import_seed511_suite(ROOT)
    observed_extra_turn = False
    orientations = set()
    for opening in openings:
        game = KalahGame.from_state(screen.INITIAL_STATE)
        for absolute_move in opening["source_prefix_moves_absolute"]:
            mover = game.current_player
            assert game.move(absolute_move)
            observed_extra_turn |= game.current_player == mover
        orientations.add(opening["state"]["current_player"])
    assert observed_extra_turn
    assert orientations == {0, 1}


def _synthetic_terminal_game(seat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    game = KalahGame.from_state(screen.INITIAL_STATE)
    trajectory: list[int] = []
    while not game.over():
        move = game.pit_index(game.possible_moves()[0])
        trajectory.append(move)
        assert game.move(move)
    margin = game.captured_seeds[seat] - game.captured_seeds[1 - seat]
    winner = "challenger" if margin > 0 else "current" if margin < 0 else "draw"
    opening = {
        "prefix_moves": [],
        "state": screen.INITIAL_STATE,
        "state_hash": screen.canonical_identity(screen.INITIAL_STATE),
    }
    record = {
        "challenger_player": seat,
        "opening_prefix_moves": [],
        "opening_state_hash": opening["state_hash"],
        "trajectory": ",".join(map(str, trajectory)),
        "game_length": len(trajectory),
        "margin": margin,
        "winner": winner,
    }
    return opening, record


@pytest.mark.parametrize("seat", [0, 1])
def test_independent_replay_normalizes_both_seats(seat: int) -> None:
    opening, game = _synthetic_terminal_game(seat)
    normalized, margin = verifier._replay(opening, game)
    expected = "challenger" if margin > 0 else "opponent" if margin < 0 else "draw"
    assert normalized == expected


def test_independent_replay_rejects_changed_outcome_and_incomplete_game() -> None:
    opening, game = _synthetic_terminal_game(0)
    altered = dict(game, margin=int(game["margin"]) + 1)
    with pytest.raises(ValueError, match="margin_mismatch"):
        verifier._replay(opening, altered)
    wrong_seat = dict(game, challenger_player=1)
    with pytest.raises(ValueError, match="margin_mismatch|winner_mismatch"):
        verifier._replay(opening, wrong_seat)
    truncated = dict(
        game, trajectory=game["trajectory"].split(",", 1)[0], game_length=1
    )
    with pytest.raises(ValueError, match="capped_or_incomplete"):
        verifier._replay(opening, truncated)


def test_recovery_classifies_complete_raw_pair_and_skips_committed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    batch = output / "batches"
    batch.mkdir(parents=True)
    for name, suffix in (
        ("opening", "jsonl"),
        ("games", "jsonl"),
        ("report", "json"),
        ("command", "json"),
    ):
        (batch / f"{name}-0003.{suffix}").write_text("{}\n")
    assert runner.recovery_action(output, 3, set()) == "recover"
    assert runner.recovery_action(output, 3, {3}) == "skip"


def test_recovery_preserves_and_rejects_partial_or_unreceipted_attempts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    batch = output / "batches"
    batch.mkdir(parents=True)
    (batch / "games-0004.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="partial_raw_pair_preserved"):
        runner.recovery_action(output, 4, set())
    (batch / "games-0004.jsonl").unlink()
    (batch / "command-0004.json").write_text("{}\n")
    with pytest.raises(ValueError, match="prior_attempt_without_complete_raw_pair"):
        runner.recovery_action(output, 4, set())


def test_only_receipted_pre_game_failure_can_be_retried(tmp_path: Path) -> None:
    output = tmp_path / "out"
    batch = output / "batches"
    archive = batch / "prelaunch-failures/opening-0000"
    archive.mkdir(parents=True)
    for name in ("command.json", "attempt.json", "command-log.jsonl"):
        (archive / name).write_text(name + "\n")
    failure = {
        "status": "preserved_non_outcome_bearing_attempt",
        "opening_index": 0,
        "games_path_exists": False,
        "report_path_exists": False,
        "command_sha256": runner.sha(archive / "command.json"),
        "attempt_sha256": runner.sha(archive / "attempt.json"),
        "command_log_sha256": runner.sha(archive / "command-log.jsonl"),
    }
    (output / "PRELAUNCH-FAILURE-0000.json").write_text(json.dumps(failure))
    assert runner.recovery_action(output, 0, set()) == "execute"
    (archive / "attempt.json").write_text("corrupted\n")
    with pytest.raises(ValueError, match="prelaunch_failure_not_authenticated"):
        runner.recovery_action(output, 0, set())


def test_constructed_arena_command_is_accepted_by_real_parser(tmp_path: Path) -> None:
    output = tmp_path / screen.OUT
    (output / "batches").mkdir(parents=True)
    (output / "runtime-artifacts").mkdir()
    (output / "batches/opening-0000.jsonl").write_text('{"prefix_moves": []}\n')
    probe = tmp_path / "model-artifact/runtime/kalah_v1_tablebase"
    tablebase = tmp_path / ".tmp/tablebase"
    probe.parent.mkdir(parents=True)
    probe.write_bytes(b"probe")
    tablebase.parent.mkdir(parents=True)
    tablebase.write_bytes(b"tablebase")
    (output / "runtime-authentication.json").write_text(
        json.dumps(
            {
                "runtime_assets": {
                    "native_probe_path": "model-artifact/runtime/kalah_v1_tablebase",
                    "tablebase_path": ".tmp/tablebase",
                }
            }
        )
    )
    command = runner.parser_checked_command(tmp_path, 0)
    assert command[1:3] == ["-m", "ml.alphazero_lite.seed542_screen_arena_entrypoint"]
    assert command[command.index("--games") + 1] == "2"
    assert command[command.index("--seed") + 1] == "545"


def _command_validation_fixture(tmp_path: Path):
    execution_root = tmp_path / "original-checkout"
    recorded_output = execution_root / screen.OUT
    output = tmp_path / "physical-copy" / screen.OUT
    batch = output / "batches"
    recorded_batch = recorded_output / "batches"
    batch.mkdir(parents=True)
    recorded_batch.mkdir(parents=True)
    opening_path = batch / "opening-0000.jsonl"
    opening_path.write_text('{"opening_index": 0}\n')
    shutil.copy2(opening_path, recorded_batch / opening_path.name)
    (output / "runtime-artifacts").mkdir()
    (recorded_output / "runtime-artifacts").mkdir()
    (recorded_output / "runtime-authentication.json").write_text(
        json.dumps(
            {
                "runtime_assets": {
                    "native_probe_path": "model-artifact/runtime/kalah_v1_tablebase",
                    "tablebase_path": ".tmp/tablebase",
                }
            }
        )
    )
    runtime = {
        "python": "test",
        "executable": runner.sys.executable,
        "packages": ["torch==test"],
    }
    components = {
        "D-final": "d",
        "E4": "e",
        "initializer": "i",
        "runtime_assets": {"tablebase_path": ".tmp/tablebase"},
    }
    execution_sources = {"ml/source.py": "source-hash"}
    registration = {
        "component_assignments": runner.ASSIGNMENTS,
        "component_hashes": components,
        "runtime_assets": {
            "native_probe_sha256": "probe-hash",
            "tablebase_sha256": "tablebase-hash",
            "tablebase_path": ".tmp/tablebase",
            "search_policy_sha256": "policy-hash",
        },
        "execution_source_sha256": execution_sources,
        "runtime": runtime,
        "execution_root": str(execution_root.resolve()),
    }
    freeze = {
        "registration_sha256": "registration-hash",
        "suite_sha256": screen.OPENING_SHA256,
    }
    argv = runner.parser_checked_command(execution_root, 0)
    command = {
        "argv": argv,
        "registration_sha256": freeze["registration_sha256"],
        "suite_sha256": freeze["suite_sha256"],
        "runtime_source_sha256": execution_sources,
        "component_assignments": runner.ASSIGNMENTS,
        "component_hashes": components,
        "runtime": runtime,
    }
    opening = {"opening_index": 0}
    notes = {
        "challenger_simulations": 128,
        "current_simulations": 128,
        "workers_used": 1,
        "workers_requested": 1,
        "base_seed": 545,
        "seed_contract": "azlite_eval_seed_v2",
        "random_opening_plies": 0,
        "exact_root_solve_threshold": 16,
        "exact_root_objective": "final_score_margin",
        "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
        "exact_leaf_solve_mode": "disabled",
        "exact_root_native_probe_sha256": "probe-hash",
        "exact_root_tablebase_sha256": "tablebase-hash",
        "exact_root_native_probe": str(
            execution_root / "model-artifact/runtime/kalah_v1_tablebase"
        ),
        "exact_root_tablebase": str(
            execution_root / registration["runtime_assets"]["tablebase_path"]
        ),
        "runtime_search_policy_sha256": "policy-hash",
        "search_options": runner.SEARCH_OPTIONS,
        "suite_sha256": runner.sha(opening_path),
        "challenger_path": str(recorded_output / "runtime-artifacts/D-final"),
        "current_path": str(recorded_output / "runtime-artifacts/initializer"),
        "search_profile": {
            "simulations": 128,
            "challenger_simulations": 128,
            "current_simulations": 128,
            "c_puct": 1.25,
            "exact_root_solve_threshold": 16,
            "exact_root_objective": "final_score_margin",
            "exact_root_tie_rule": "highest_legal_network_prior_then_lowest_move_index",
            "exact_leaf_solve_mode": "disabled",
            "search_options": runner.SEARCH_OPTIONS,
        },
    }
    report = {"games": 2, "games_played": 2, "notes": notes}
    return output, opening, command, report, runtime, registration, freeze


def test_independent_command_report_contract_and_semantic_mutations(
    tmp_path: Path,
) -> None:
    output, opening, command, report, runtime, registration, freeze = (
        _command_validation_fixture(tmp_path)
    )
    verifier._validate_command_and_report(
        tmp_path, output, 0, opening, command, report, runtime, registration, freeze
    )
    mutations = [
        (
            "component_assignments",
            {**runner.ASSIGNMENTS, "opponent_policy": "D-final"},
            "component_binding",
        ),
        ("component_hashes", {"D-final": "wrong"}, "component_hash"),
        ("runtime", {"python": "changed"}, "runtime_binding"),
    ]
    for field, changed, expected in mutations:
        altered = {**command, field: changed}
        with pytest.raises(ValueError, match=expected):
            verifier._validate_command_and_report(
                tmp_path,
                output,
                0,
                opening,
                altered,
                report,
                runtime,
                registration,
                freeze,
            )
    altered_argv = list(command["argv"])
    altered_argv[altered_argv.index("--c-puct") + 1] = "0.5"
    altered = {**command, "argv": altered_argv}
    with pytest.raises(ValueError, match="command_contract_mismatch"):
        verifier._validate_command_and_report(
            tmp_path, output, 0, opening, altered, report, runtime, registration, freeze
        )
    altered_report = json.loads(json.dumps(report))
    altered_report["notes"]["exact_root_solve_threshold"] = 15
    with pytest.raises(ValueError, match="report_runtime_contract_mismatch"):
        verifier._validate_command_and_report(
            tmp_path,
            output,
            0,
            opening,
            command,
            altered_report,
            runtime,
            registration,
            freeze,
        )


def test_commit_revalidates_pair_and_atomically_commits_both_seats(
    tmp_path: Path,
) -> None:
    expected = [
        {"opening_index": 8, "game_position": 0, "challenger_player": 0},
        {"opening_index": 8, "game_position": 1, "challenger_player": 1},
    ]
    with (
        patch.object(runner, "_assert_frozen"),
        patch.object(
            runner.independent, "validate_pair", return_value=expected
        ) as validate,
    ):
        committed = runner.commit_pair(tmp_path, tmp_path, 8)
    validate.assert_called_once_with(tmp_path, tmp_path, 8)
    ledger = [
        json.loads(line)
        for line in (tmp_path / "outcomes.jsonl").read_text().splitlines()
    ]
    assert ledger == committed == expected


def test_frozen_execution_rejects_changed_runtime_source_suite_and_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    output = root / "docs/data/seed542-expanded-policy-strength-screen"
    output.mkdir(parents=True)
    (root / "ml").mkdir()
    source = root / "ml/source.py"
    artifact = root / "artifact.bin"
    source.write_bytes(b"source")
    artifact.write_bytes(b"artifact")
    (output / "openings.jsonl").write_bytes(b"suite")
    for name in (
        "runtime-authentication.json",
        "runtime-policy-preflight.json",
        "training-verification.json",
        "command-preflight.jsonl",
        "suite-import-report.json",
        "overlap-summary.json",
        "overlap-per-opening.jsonl",
    ):
        (output / name).write_text("{}\n")
    registration = {
        "execution_root": str(root.resolve()),
        "execution_source_sha256": {"ml/source.py": runner.sha(source)},
        "authenticated_file_sha256": {"artifact.bin": runner.sha(artifact)},
        "suite": {
            "import_report_sha256": runner.sha(output / "suite-import-report.json"),
            "overlap_summary_sha256": runner.sha(output / "overlap-summary.json"),
            "overlap_per_opening_sha256": runner.sha(
                output / "overlap-per-opening.jsonl"
            ),
        },
    }
    registration_path = output / "registration.json"
    registration_path.write_text(json.dumps(registration))
    freeze = {
        "registration_sha256": runner.sha(registration_path),
        "suite_sha256": runner.sha(output / "openings.jsonl"),
        "runtime_authentication_sha256": runner.sha(
            output / "runtime-authentication.json"
        ),
        "runtime_policy_preflight_sha256": runner.sha(
            output / "runtime-policy-preflight.json"
        ),
        "training_verification_sha256": runner.sha(
            output / "training-verification.json"
        ),
        "command_preflight_sha256": runner.sha(output / "command-preflight.jsonl"),
        "execution_source_sha256": registration["execution_source_sha256"],
        "authenticated_file_sha256": registration["authenticated_file_sha256"],
    }
    (output / "execution-freeze.json").write_text(json.dumps(freeze))
    runner._assert_frozen(root, output)

    for path in (
        source,
        artifact,
        output / "openings.jsonl",
        output / "runtime-authentication.json",
    ):
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises(
            ValueError,
            match="execution_frozen_binding_mismatch|execution_source_changed|execution_authenticated_input_changed",
        ):
            runner._assert_frozen(root, output)
        path.write_bytes(original)


def test_bootstrap_keeps_multiplicity_and_strictly_rejects_point_five() -> None:
    tied = [
        {"opening_index": index, "winner": winner}
        for index in range(1024)
        for winner in ("challenger", "opponent")
    ]
    runner_result, _ = runner.runner_analysis(tied)
    verifier_result, _ = verifier.independent_analysis(tied)
    assert runner_result == verifier_result
    assert runner_result["score"] == 0.5
    assert runner_result["one_sided_95_lower"] == 0.5
    assert runner_result["pass"] is False
    assert runner_result["decision"] == "do_not_advance"

    sparse = [
        {"opening_index": index, "winner": "challenger" if index == 0 else "opponent"}
        for index in range(1024)
        for _ in range(2)
    ]
    boot_result, _ = runner.runner_analysis(sparse)
    # With replacement, some bootstrap replicates include opening zero more than once.
    rng = np.random.default_rng(545)
    indices = rng.integers(0, 1024, size=(20_000, 1024))
    multiplicity_boot = (indices == 0).sum(axis=1) / 1024
    assert boot_result["one_sided_95_lower"] == float(
        np.quantile(multiplicity_boot, 0.05, method="linear")
    )


def test_verifier_rejects_ledger_disagreement_after_receipts_exist() -> None:
    pair = [
        {
            "opening_index": 12,
            "game_position": 0,
            "challenger_player": 0,
            "winner": "challenger",
            "challenger_margin": 7,
            "game_sha256": "games-hash",
            "report_sha256": "report-hash",
            "opening_sha256": "opening-hash",
        },
        {
            "opening_index": 12,
            "game_position": 1,
            "challenger_player": 1,
            "winner": "opponent",
            "challenger_margin": -3,
            "game_sha256": "games-hash",
            "report_sha256": "report-hash",
            "opening_sha256": "opening-hash",
        },
    ]
    verifier.reconcile_pair_ledger(12, pair, list(pair))
    for field, value in (
        ("winner", "challenger"),
        ("challenger_player", 0),
        ("challenger_margin", 9),
    ):
        altered = [dict(row) for row in pair]
        altered[1][field] = value
        with pytest.raises(ValueError, match="ledger_raw_pair_disagreement"):
            verifier.reconcile_pair_ledger(12, pair, altered)


def test_real_analysis_verifier_rejects_score_interval_decision_and_matrix_mutations() -> (
    None
):
    ledger = [
        {"opening_index": index, "winner": winner}
        for index in range(1024)
        for winner in ("challenger", "opponent")
    ]
    result, per_opening = verifier.independent_analysis(ledger)
    mutations = [
        ("score", 0.51),
        ("one_sided_95_lower", 0.51),
        ("decision", "advance_to_propose_separate_1200_simulation_evaluation_only"),
        ("pass", True),
    ]
    for field, value in mutations:
        changed = dict(result)
        changed[field] = value
        with pytest.raises(ValueError, match="published_analysis_mismatch"):
            verifier.verify_published_analysis(ledger, changed, per_opening)
    changed_openings = [dict(row) for row in per_opening]
    changed_openings[0]["opening_cluster_score"] = 1.0
    with pytest.raises(ValueError, match="published_analysis_mismatch"):
        verifier.verify_published_analysis(ledger, result, changed_openings)
