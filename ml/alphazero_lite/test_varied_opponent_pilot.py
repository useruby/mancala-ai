"""Scripted checks for the varied-opponent target-collection worker."""

from __future__ import annotations

import json
import concurrent.futures
import shutil
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from ml.alphazero_lite import varied_opponent_pilot as pilot

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "docs/data/seed556-exploratory-arena/screen-0005/runtime-artifacts"


class _Evaluator:
    def __init__(self, name: str):
        self.name = name


def _specs(opponent: str = "initializer") -> list[dict]:
    specs = []
    for arm, selected_opponent in (("control", "P"), ("treatment", opponent)):
        pair_id = f"{arm}-opening-17"
        for seat in (0, 1):
            specs.append(
                {
                    "game_id": f"{pair_id}-P{seat}",
                    "pair_id": pair_id,
                    "arm": arm,
                    "opening_id": "opening-17",
                    "opening_moves": [],
                    "seed": 417,
                    "designated_p_seat": seat,
                    "opponent": selected_opponent,
                    "search_workers": 1,
                }
            )
    return specs


def _scripted_search(seen: list[tuple[int, str]]):
    def search(game, evaluator, _rng, ply, retain):
        seen.append((game.current_player, evaluator.name))
        legal = game.possible_moves()
        move = 2 if not seen[:-1] and game.current_player == 0 else legal[0]
        policy = [0.0] * 6
        policy[legal[0]] = 1.0
        visits = [0] * 6
        visits[legal[0]] = pilot.SIMULATIONS
        row = {
            "policy": policy,
            "legal_moves": legal,
            "value": 0.0,
            "teacher_source": "artifact_puct",
            "target_visits": visits,
            "target_temperature": (
                pilot.TEMPERATURE_EARLY
                if ply < pilot.TEMPERATURE_THRESHOLD
                else pilot.TEMPERATURE_LATE
            ),
        }
        return {"move": move, "row": row if retain else {}}

    return search


@pytest.mark.parametrize("opponent", ["initializer", "T512"])
@pytest.mark.parametrize("seat", [0, 1])
def test_play_game_routes_by_actual_player_and_retains_only_p_turns(opponent, seat):
    spec = next(
        row
        for row in _specs(opponent)
        if row["arm"] == "treatment" and row["designated_p_seat"] == seat
    )
    # Pair validation is independent of this one-game worker test.
    seen: list[tuple[int, str]] = []
    p = _Evaluator("P")
    other = _Evaluator(opponent)
    record = pilot.play_game(
        spec,
        p_evaluator=p,
        opponent_evaluator=other,
        search_turn=_scripted_search(seen),
    )

    assert record["game_status"] == "complete"
    assert pilot.verify_game(record)["rows"] == len(record["rows"])
    assert all(row["player"] == seat for row in record["rows"])
    assert all(row["designated_p_seat"] == seat for row in record["rows"])
    assert record["trajectory"][0]["move"] == 2
    assert record["trajectory"][0]["extra_turn"] is True
    assert record["trajectory"][1]["player"] == 0
    expected = p.name if seat == 0 else opponent
    assert seen[0] == (0, expected)
    assert seen[1] == (0, expected)
    assert record["cpu_time_ns"] >= 0


def test_control_routes_both_seats_to_p_but_retains_designated_seat_only():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "control" and row["designated_p_seat"] == 1
    )
    seen: list[tuple[int, str]] = []
    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=None,
        search_turn=_scripted_search(seen),
    )
    assert record["game_status"] == "complete"
    assert {evaluator for _player, evaluator in seen} == {"P"}
    assert all(row["player"] == 1 for row in record["rows"])
    assert pilot.verify_game(record)["rows"] == len(record["rows"])


def test_capped_game_is_separate_and_has_no_accepted_rows():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=_scripted_search([]),
        max_moves=2,
    )
    assert record["game_status"] == "capped"
    assert record["rows"] == []
    assert record["accepted_rows"] == 0
    assert pilot.verify_game(record)["status"] == "capped"


def test_cpu_and_row_accounting_aggregates_per_arm():
    specs = [row for row in _specs("T512") if row["arm"] == "treatment"]
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("T512"),
            search_turn=_scripted_search([]),
        )
        for spec in specs
    ]
    process_cpu_ns = sum(game["cpu_time_ns"] for game in games) + 500
    native_child_cpu_ns = 300
    summary = pilot.summarize_pair_accounting(
        [
            {
                "arm": "treatment",
                "pair_process_cpu_time_ns": process_cpu_ns,
                "native_child_cpu_time_ns": native_child_cpu_ns,
                "pair_cpu_time_ns": process_cpu_ns + native_child_cpu_ns,
                "games": games,
            }
        ]
    )["treatment"]
    assert summary["pairs"] == 1
    assert summary["games"] == 2
    assert summary["complete_games"] == 2
    assert summary["accepted_rows"] == sum(game["accepted_rows"] for game in games)
    assert summary["native_child_cpu_time_ns"] == native_child_cpu_ns
    assert summary["cpu_time_ns"] == process_cpu_ns + native_child_cpu_ns
    assert summary["cpu_hours"] == summary["cpu_time_ns"] / 3_600_000_000_000


def test_pair_process_cpu_must_cover_both_game_cpu_totals():
    specs = [row for row in _specs("T512") if row["arm"] == "treatment"]
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("T512"),
            search_turn=_scripted_search([]),
        )
        for spec in specs
    ]
    game_cpu_ns = sum(game["cpu_time_ns"] for game in games)
    with pytest.raises(ValueError, match="pair_process_cpu_below_game_cpu"):
        pilot.summarize_pair_accounting(
            [
                {
                    "arm": "treatment",
                    "games": games,
                    "pair_process_cpu_time_ns": 0,
                    "native_child_cpu_time_ns": game_cpu_ns,
                    "pair_cpu_time_ns": game_cpu_ns,
                }
            ]
        )


def test_reaped_native_probe_cpu_is_sampled(monkeypatch):
    monkeypatch_result = SimpleNamespace(ru_utime=0.125, ru_stime=0.025)
    monkeypatch.setattr(pilot.resource, "getrusage", lambda _who: monkeypatch_result)
    assert pilot.reaped_child_cpu_ns() == 150_000_000


def test_verifier_rejects_missing_duplicate_opponent_and_mutated_rows():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=_scripted_search([]),
    )
    missing = json.loads(json.dumps(record))
    missing["rows"].pop()
    missing["accepted_rows"] -= 1
    with pytest.raises(ValueError, match="retained_row_set_mismatch"):
        pilot.verify_game(missing)

    duplicate = json.loads(json.dumps(record))
    duplicate["rows"].append(dict(duplicate["rows"][0]))
    duplicate["accepted_rows"] += 1
    with pytest.raises(ValueError, match="duplicate_retained_row"):
        pilot.verify_game(duplicate)

    opponent_row = json.loads(json.dumps(record))
    opponent_row["rows"][0]["player"] = 1
    with pytest.raises(ValueError, match="opponent_seat_row"):
        pilot.verify_game(opponent_row)

    mismatched = json.loads(json.dumps(record))
    mismatched["rows"][0]["state"][0] += 0.25
    with pytest.raises(ValueError, match="row_state_or_ply_mismatch"):
        pilot.verify_game(mismatched)

    reassigned = json.loads(json.dumps(record))
    reassigned["rows"][0]["opponent"] = "T512"
    with pytest.raises(ValueError, match="row_pair_assignment_mismatch"):
        pilot.verify_game(reassigned)

    different_legal_target = json.loads(json.dumps(record))
    row = different_legal_target["rows"][0]
    alternative = next(
        action for action in row["legal_moves"] if row["policy"][action] == 0.0
    )
    row["policy"] = [float(action == alternative) for action in range(6)]
    with pytest.raises(ValueError, match="row_policy_target_mismatch"):
        pilot.verify_game(different_legal_target)


def test_exact_root_target_is_rebuilt_from_saved_decision_evidence():
    legal = list(range(6))
    row = {
        "teacher_source": "exact_root_tablebase",
        "exact_selected_action": 3,
        "exact_optimal_actions": [1, 3],
        "exact_action_margins": {"0": 1, "1": 5, "2": 4, "3": 5, "4": 0, "5": -1},
        "exact_legal_moves": legal,
        "exact_network_priors": [0.0, 0.2, 0.0, 0.7, 0.0, 0.0],
        "exact_active_pit_stones": 48,
        "exact_root_player": 0,
        "exact_solver_calls": 1,
        "stone_count": 48,
        "player": 0,
    }
    expected = pilot.reconstruct_policy_target(row, legal_moves=legal)
    assert expected[3] == 1.0
    assert sum(expected) == 1.0
    row["exact_action_margins"].pop("5")
    with pytest.raises(ValueError, match="exact_root_decision.*evidence|exact_root"):
        pilot.reconstruct_policy_target(row, legal_moves=legal)


def _exactize_row_for_fixture(row: dict, *, selected: int, legal: list[int]) -> None:
    row["teacher_source"] = "exact_root_tablebase"
    row["exact_selected_action"] = selected
    row["exact_optimal_actions"] = [selected]
    row["exact_action_margins"] = {
        str(action): 5 if action == selected else 0 for action in legal
    }
    row["exact_legal_moves"] = legal
    row["exact_network_priors"] = [
        0.9 if action == selected else 0.0 for action in range(6)
    ]
    row["exact_active_pit_stones"] = row["stone_count"]
    row["exact_root_player"] = row["player"]
    row["exact_solver_calls"] = 1
    row.pop("target_visits", None)
    row.pop("target_temperature", None)
    row["policy"] = pilot.exact_root_policy_target(
        selected_action=selected,
        optimal_actions=[selected],
        mode="selected_one_hot",
    )


def test_verifier_requires_128_visits_for_puct_target():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=_scripted_search([]),
    )
    target_row = record["rows"][0]
    target_row["target_visits"][target_row["legal_moves"][0]] = 1
    with pytest.raises(ValueError, match="puct_target_simulation_count_mismatch"):
        pilot.verify_game(record)


def test_verifier_rejects_exact_root_route_above_stone_cutoff():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=_scripted_search([]),
    )
    row = record["rows"][0]
    played = record["trajectory"][row["move_index"]]["move"]
    alternative = next(action for action in row["legal_moves"] if action != played)
    _exactize_row_for_fixture(row, selected=alternative, legal=row["legal_moves"])
    with pytest.raises(ValueError, match="row_search_route_mismatch"):
        pilot.verify_game(record)


def test_verifier_requires_exact_selection_to_match_trajectory_move():
    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    spec["opening_moves"] = [
        3,
        3,
        0,
        2,
        5,
        3,
        4,
        2,
        3,
        4,
        4,
        0,
        5,
        1,
        2,
        2,
        0,
        5,
        2,
        0,
        3,
        0,
        0,
        3,
        2,
    ]

    def choose_legal(game, evaluator, rng, ply, retain):
        result = _scripted_search([])(game, evaluator, rng, ply, retain)
        result["move"] = game.possible_moves()[0]
        if retain:
            legal = game.possible_moves()
            visits = [0] * 6
            visits[legal[0]] = pilot.SIMULATIONS
            result["row"]["legal_moves"] = legal
            result["row"]["target_visits"] = visits
            result["row"]["policy"] = [float(action == legal[0]) for action in range(6)]
        return result

    record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=choose_legal,
    )
    row = record["rows"][0]
    assert row["stone_count"] <= pilot.EXACT_ROOT_THRESHOLD
    played = record["trajectory"][row["move_index"]]["move"]
    alternative = next(action for action in row["legal_moves"] if action != played)
    _exactize_row_for_fixture(row, selected=alternative, legal=row["legal_moves"])
    with pytest.raises(ValueError, match="exact_root_selected_move_mismatch"):
        pilot.verify_game(record)


def test_pair_assignment_change_is_rejected_before_runner():
    specs = _specs("initializer")
    changed = [dict(spec) for spec in specs]
    treatment = [spec for spec in changed if spec["arm"] == "treatment"]
    treatment[1]["opponent"] = "T512"
    called = False

    def runner(_job):
        nonlocal called
        called = True

    jobs = [
        {"game_specs": [spec for spec in changed if spec["arm"] == arm]}
        for arm in ("control", "treatment")
    ]
    with pytest.raises(ValueError, match="treatment_opponent_changed_across_seat_swap"):
        pilot.execute_pair_jobs(jobs, ARTIFACTS, runner)
    assert called is False


def test_component_hash_mismatch_fails_launch_preflight(tmp_path):
    copied = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, copied)
    target = copied / "T512" / "metadata.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="component_hash_mismatch:T512"):
        pilot.launch_preflight(_specs(), copied)


def test_launch_preflight_checks_bundle_scope_and_fixed_workers(monkeypatch):
    monkeypatch.setattr(
        pilot,
        "effective_runtime_policy",
        lambda _artifacts: {"effective_search_policy": {}, "runtime_assets": {}},
    )
    monkeypatch.setattr(
        pilot.subprocess,
        "run",
        lambda argv, **_kwargs: SimpleNamespace(args=argv, returncode=0),
    )
    result = pilot.launch_preflight(_specs(), ARTIFACTS)
    assert result["status"] == "preflight_passed_no_games_launched"
    assert result["pair_concurrency"] == 18
    assert result["workers_per_game"] == 1
    assert result["systemd_scope_preflight"] == "passed_no_game_command_run"
    command = pilot.systemd_scope_command(["python", "-c", "pass"])
    assert "--property=CPUQuota=1800%" in command
    assert "--property=MemoryMax=38654705664" in command
    assert "--property=AllowedCPUs=14-31" in command


def test_runtime_asset_materialization_is_hash_checked(tmp_path, monkeypatch):
    materialized = tmp_path / "tablebase.kvtb"
    materialized.write_bytes(b"not the registered tablebase")
    monkeypatch.setenv("AZLITE_KALAH_V1_TABLEBASE", str(materialized))
    with pytest.raises(ValueError, match="runtime_asset_hash_mismatch:tablebase"):
        pilot.effective_runtime_policy(ARTIFACTS)


def test_pair_executor_uses_18_concurrent_pair_slots():
    active = 0
    maximum = 0
    lock = threading.Lock()
    barrier = threading.Barrier(pilot.PAIR_CONCURRENCY)

    def runner(_job):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        barrier.wait(timeout=5)
        time.sleep(0.005)
        with lock:
            active -= 1
        return pilot.WORKERS_PER_GAME

    results = pilot.run_pair_specs([{} for _ in range(pilot.PAIR_CONCURRENCY)], runner)
    assert maximum == pilot.PAIR_CONCURRENCY
    assert results == [pilot.WORKERS_PER_GAME] * pilot.PAIR_CONCURRENCY
    with pytest.raises(ValueError, match="pilot_pair_concurrency_must_be_18"):
        pilot.run_pair_specs([], runner, max_workers=1)


def test_game_specs_require_single_search_worker_and_swapped_seats():
    specs = _specs()
    bad_worker = [dict(spec) for spec in specs]
    bad_worker[0]["search_workers"] = 2
    with pytest.raises(ValueError, match="game_search_worker_count_mismatch"):
        pilot.validate_game_specs(bad_worker)
    bad_seat = [dict(spec) for spec in specs]
    bad_seat[1]["designated_p_seat"] = 0
    with pytest.raises(ValueError, match="pair_must_have_both_seat_assignments"):
        pilot.validate_game_specs(bad_seat)


def test_opening_prefix_is_bound_across_seats_and_arms_and_archive():
    specs = _specs("initializer")
    specs[1]["opening_moves"] = [0]
    with pytest.raises(ValueError, match="opening_prefix_assignment_mismatch"):
        pilot.validate_game_specs(specs)

    treatment = [dict(row) for row in _specs("T512") if row["arm"] == "treatment"]
    for row in treatment:
        row["opening_moves"] = [0]
    with pytest.raises(ValueError, match="opening_prefix_assignment_mismatch"):
        pilot.validate_game_specs(
            treatment + [row for row in _specs() if row["arm"] == "control"]
        )

    spec = next(
        row
        for row in _specs()
        if row["arm"] == "treatment" and row["designated_p_seat"] == 0
    )
    game_record = pilot.play_game(
        spec,
        p_evaluator=_Evaluator("P"),
        opponent_evaluator=_Evaluator("initializer"),
        search_turn=_scripted_search([]),
    )
    game_record["opening_moves"] = [0]
    with pytest.raises(ValueError, match="archived_opening_prefix_mismatch"):
        pilot.verify_game(game_record)


def test_pair_checkpoint_is_atomic_idempotent_and_resume_skips_committed(
    tmp_path, monkeypatch
):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("initializer"),
            search_turn=_scripted_search([]),
        )
        for spec in job["game_specs"]
    ]
    result = {
        "pair_id": job["game_specs"][0]["pair_id"],
        "arm": "treatment",
        "games": games,
        "pair_process_cpu_time_ns": sum(game["cpu_time_ns"] for game in games),
        "native_child_cpu_time_ns": 0,
        "pair_cpu_time_ns": sum(game["cpu_time_ns"] for game in games),
    }
    checkpoint_dir = tmp_path / "pairs"
    committed_path = pilot.commit_verified_pair(checkpoint_dir, job, result)
    assert pilot.commit_verified_pair(checkpoint_dir, job, result) == committed_path
    monkeypatch.setattr(pilot, "verify_components", lambda _root: {})
    attempt_dir = tmp_path / "attempts"
    producer_attempt = pilot.pair_attempt_path(attempt_dir, result["pair_id"], 1)
    pilot.write_json_atomic(
        producer_attempt,
        {
            "pair_id": result["pair_id"],
            "status": "started",
            "assignment": job,
        },
    )
    resumed = pilot.run_checkpointed_pairs(
        [job], ARTIFACTS, checkpoint_dir, attempt_dir=attempt_dir
    )
    assert [row["pair_id"] for row in resumed] == [result["pair_id"]]
    attempt_path = pilot.pair_attempt_path(attempt_dir, result["pair_id"], 1)
    receipt = {
        "pair_id": result["pair_id"],
        "status": "completed",
        "assignment": job,
        "checkpoint_sha256": pilot.sha256(committed_path),
        "pair_cpu_time_ns": result["pair_cpu_time_ns"],
    }
    pilot.write_json_atomic(attempt_path, receipt)
    assert (
        pilot.reconcile_pair_attempts(
            attempt_dir, [job], {result["pair_id"]: result}, checkpoint_dir
        )[result["pair_id"]]
        == 2
    )
    receipt["pair_cpu_time_ns"] += 1
    pilot.write_json_atomic(attempt_path, receipt)
    with pytest.raises(ValueError, match="pilot_attempt_checkpoint_receipt_conflict"):
        pilot.reconcile_pair_attempts(
            attempt_dir, [job], {result["pair_id"]: result}, checkpoint_dir
        )
    receipt["pair_cpu_time_ns"] = result["pair_cpu_time_ns"]
    receipt["checkpoint_sha256"] = "0" * 64
    pilot.write_json_atomic(attempt_path, receipt)
    with pytest.raises(ValueError, match="pilot_attempt_checkpoint_receipt_conflict"):
        pilot.reconcile_pair_attempts(
            attempt_dir, [job], {result["pair_id"]: result}, checkpoint_dir
        )
    completed, pending = pilot.load_pair_checkpoints(
        checkpoint_dir,
        [
            job,
            {
                "game_specs": [
                    {
                        **spec,
                        "arm": "control",
                        "opponent": "P",
                        "pair_id": "control-opening-17",
                        "game_id": f"control-opening-17-P{spec['designated_p_seat']}",
                    }
                    for spec in job["game_specs"]
                ]
            },
        ],
    )
    assert [row["pair_id"] for row in completed] == [job["game_specs"][0]["pair_id"]]
    assert len(pending) == 1
    conflicting = json.loads(committed_path.read_text())
    conflicting["result"]["pair_cpu_time_ns"] += 1
    committed_path.write_text(json.dumps(conflicting))
    with pytest.raises(ValueError, match="pair_child_cpu_accounting_invalid"):
        pilot.load_pair_checkpoints(checkpoint_dir, [job])


def test_pair_checkpoint_rejects_assignment_conflict(tmp_path):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("initializer"),
            search_turn=_scripted_search([]),
        )
        for spec in job["game_specs"]
    ]
    cpu = sum(game["cpu_time_ns"] for game in games)
    result = {
        "pair_id": job["game_specs"][0]["pair_id"],
        "arm": "treatment",
        "games": games,
        "pair_process_cpu_time_ns": cpu,
        "native_child_cpu_time_ns": 0,
        "pair_cpu_time_ns": cpu,
    }
    path = pilot.commit_verified_pair(tmp_path, job, result)
    changed = json.loads(path.read_text())
    changed["assignment"]["game_specs"][0]["seed"] += 1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="pair_checkpoint_assignment_conflict"):
        pilot.load_pair_checkpoints(tmp_path, [job])


def test_empty_pair_checkpoint_cannot_be_committed_or_skipped(tmp_path):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    result = {
        "pair_id": job["game_specs"][0]["pair_id"],
        "arm": "treatment",
        "games": [],
        "pair_process_cpu_time_ns": 0,
        "native_child_cpu_time_ns": 0,
        "pair_cpu_time_ns": 0,
    }
    with pytest.raises(ValueError, match="pair_checkpoint_game_count_mismatch"):
        pilot.commit_verified_pair(tmp_path, job, result)
    checkpoint = {
        "schema": "varied_opponent_pair_checkpoint_v1",
        "assignment": job,
        "producing_attempt": 1,
        "result": result,
    }
    path = pilot.pair_checkpoint_path(tmp_path, job["game_specs"][0]["pair_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="pair_checkpoint_game_count_mismatch"):
        pilot.load_pair_checkpoints(tmp_path, [job])


def test_interrupted_pair_attempt_is_preserved_and_blocks_resume(tmp_path):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    pair_id = job["game_specs"][0]["pair_id"]
    attempt_dir = tmp_path / "attempts"
    attempt_dir.mkdir()
    attempt = pilot.pair_attempt_path(attempt_dir, pair_id, 1)
    attempt.parent.mkdir(parents=True, exist_ok=True)
    original = {"pair_id": pair_id, "status": "started", "assignment": job}
    attempt.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="unreconciled_interrupted_attempt_cpu"):
        pilot.run_checkpointed_pairs(
            [job], ARTIFACTS, tmp_path / "pairs", attempt_dir=attempt_dir
        )
    assert json.loads(attempt.read_text()) == original


def test_dispatched_but_never_started_pair_remains_pending(tmp_path):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    pair_id = job["game_specs"][0]["pair_id"]
    attempt_dir = tmp_path / "attempts"
    path = pilot.pair_attempt_path(attempt_dir, pair_id, 1)
    pilot.write_json_atomic(
        path,
        {"pair_id": pair_id, "status": "dispatched", "assignment": job},
    )
    next_numbers = pilot.reconcile_pair_attempts(
        attempt_dir, [job], {}, tmp_path / "pairs"
    )
    completed, pending = pilot.load_pair_checkpoints(tmp_path / "pairs", [job])
    assert completed == []
    assert pending == [job]
    assert next_numbers[pair_id] == 2


def test_retry_completion_then_resume_skips_pair_and_preserves_history(
    tmp_path, monkeypatch
):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    pair_id = job["game_specs"][0]["pair_id"]
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("initializer"),
            search_turn=_scripted_search([]),
        )
        for spec in job["game_specs"]
    ]
    process_cpu = sum(game["cpu_time_ns"] for game in games) + 500
    result = {
        "pair_id": pair_id,
        "arm": "treatment",
        "games": games,
        "pair_process_cpu_time_ns": process_cpu,
        "native_child_cpu_time_ns": 0,
        "pair_cpu_time_ns": process_cpu,
    }
    attempt_dir = tmp_path / "attempts"
    attempt1 = pilot.pair_attempt_path(attempt_dir, pair_id, 1)
    pilot.write_json_atomic(
        attempt1,
        {"pair_id": pair_id, "status": "dispatched", "assignment": job},
    )
    monkeypatch.setattr(pilot, "verify_components", lambda _root: {})

    def fake_attempt(_job, _artifact_root, attempt_path):
        pilot.write_json_atomic(
            attempt_path,
            {"pair_id": pair_id, "status": "started", "assignment": job},
        )
        return result

    class SynchronousPool:
        def __init__(self, **_kwargs):
            self.submissions = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            self.submissions += 1
            future = concurrent.futures.Future()
            try:
                future.set_result(function(*args))
            except Exception as error:  # pragma: no cover - fixture propagation
                future.set_exception(error)
            return future

    pool_instances = []

    def make_pool(**kwargs):
        pool = SynchronousPool(**kwargs)
        pool_instances.append(pool)
        return pool

    monkeypatch.setattr(pilot, "_play_pair_attempt", fake_attempt)
    monkeypatch.setattr(pilot.concurrent.futures, "ProcessPoolExecutor", make_pool)
    checkpoint_dir = tmp_path / "pairs"
    first_resume = pilot.run_checkpointed_pairs(
        [job], ARTIFACTS, checkpoint_dir, attempt_dir=attempt_dir
    )
    assert [pair["pair_id"] for pair in first_resume] == [pair_id]
    assert len(pool_instances) == 1 and pool_instances[0].submissions == 1
    attempt2 = pilot.pair_attempt_path(attempt_dir, pair_id, 2)
    assert json.loads(attempt1.read_text())["status"] == "dispatched"
    assert json.loads(attempt2.read_text())["status"] == "completed"
    checkpoint = json.loads(
        pilot.pair_checkpoint_path(checkpoint_dir, pair_id).read_text()
    )
    assert checkpoint["producing_attempt"] == 2

    completed_receipt = attempt2.read_bytes()
    pilot.write_json_atomic(
        attempt2,
        {"pair_id": pair_id, "status": "started", "assignment": job},
    )
    recovery_resume = pilot.run_checkpointed_pairs(
        [job], ARTIFACTS, checkpoint_dir, attempt_dir=attempt_dir
    )
    assert [pair["pair_id"] for pair in recovery_resume] == [pair_id]
    assert len(pool_instances) == 1

    attempt2.unlink()
    with pytest.raises(ValueError, match="pair_checkpoint_producing_attempt_missing"):
        pilot.run_checkpointed_pairs(
            [job], ARTIFACTS, checkpoint_dir, attempt_dir=attempt_dir
        )
    assert json.loads(attempt1.read_text())["status"] == "dispatched"
    attempt2.write_bytes(completed_receipt)

    second_resume = pilot.run_checkpointed_pairs(
        [job], ARTIFACTS, checkpoint_dir, attempt_dir=attempt_dir
    )
    assert [pair["pair_id"] for pair in second_resume] == [pair_id]
    assert len(pool_instances) == 1
    assert json.loads(attempt1.read_text())["status"] == "dispatched"
    assert json.loads(attempt2.read_text())["status"] == "completed"


def test_pair_checkpoint_game_specs_must_match_frozen_assignment(tmp_path):
    job = {
        "game_specs": [
            spec for spec in _specs("initializer") if spec["arm"] == "treatment"
        ]
    }
    games = [
        pilot.play_game(
            spec,
            p_evaluator=_Evaluator("P"),
            opponent_evaluator=_Evaluator("initializer"),
            search_turn=_scripted_search([]),
        )
        for spec in job["game_specs"]
    ]
    cpu = sum(game["cpu_time_ns"] for game in games)
    result = {
        "pair_id": job["game_specs"][0]["pair_id"],
        "arm": "treatment",
        "games": games,
        "pair_process_cpu_time_ns": cpu,
        "native_child_cpu_time_ns": 0,
        "pair_cpu_time_ns": cpu,
    }
    games[0]["spec"]["seed"] += 1
    with pytest.raises(ValueError, match="pair_checkpoint_game_assignment_mismatch"):
        pilot.commit_verified_pair(tmp_path, job, result)
