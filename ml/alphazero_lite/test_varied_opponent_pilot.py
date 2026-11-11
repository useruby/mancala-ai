"""Scripted checks for the varied-opponent target-collection worker."""

from __future__ import annotations

import json
import shutil
import threading
import time
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
    def search(game, evaluator, _rng, _ply, retain):
        seen.append((game.current_player, evaluator.name))
        legal = game.possible_moves()
        move = 2 if not seen[:-1] and game.current_player == 0 else legal[0]
        policy = [0.0] * 6
        policy[legal[0]] = 1.0
        row = {"policy": policy, "legal_moves": legal, "value": 0.0}
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


def test_launch_preflight_checks_bundle_scope_and_fixed_workers():
    result = pilot.launch_preflight(_specs(), ARTIFACTS)
    assert result["status"] == "preflight_passed_no_games_launched"
    assert result["pair_concurrency"] == 18
    assert result["workers_per_game"] == 1
    assert result["systemd_scope_preflight"] == "passed_no_game_command_run"
    command = pilot.systemd_scope_command(["python", "-c", "pass"])
    assert "--property=CPUQuota=1800%" in command
    assert "--property=MemoryMax=38654705664" in command
    assert "--property=AllowedCPUs=14-31" in command


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
