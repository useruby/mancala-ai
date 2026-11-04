from __future__ import annotations

import copy
import hashlib
import json
import numpy as np
import pytest
from pathlib import Path
import random

from ml.alphazero_lite import self_play
from ml.alphazero_lite import endgame_tablebase, native_exact_root_tablebase
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.seed553_fixed_state_budget import (
    build_cohort,
    compare_pair,
    encoded_state,
    run_search,
    search_config,
    execute,
)
from ml.alphazero_lite.verify_seed553_fixed_state_budget import (
    _reconstruct_report,
    validate_cohort,
    validate_published_report,
    validate_search_row,
)


class CountingEvaluator(self_play.Evaluator):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        self.calls += 1
        if sum(game.pits) <= 16:
            self.low_stone_leaf_calls += 1
        return self_play.HeuristicEvaluator().evaluate(game)

    low_stone_leaf_calls = 0


def test_registered_cohort_size_buckets_and_encoding_round_trip() -> None:
    cohort = build_cohort()
    assert len(cohort) == 256
    assert sum(row["bucket"] == ">32" for row in cohort) == 128
    assert sum(row["bucket"] == "17-32" for row in cohort) == 128
    assert len({row["canonical_state_sha256"] for row in cohort}) == 256
    assert len({row["production_float32_sha256"] for row in cohort}) == 256
    for row in cohort:
        game = KalahGame.from_state(row["state"])
        assert game.possible_moves() == row["legal_moves"]
        assert (
            encoded_state(row["state"]).tobytes()
            == np.asarray(row["encoded_input_float32"], dtype=np.float32).tobytes()
        )


def test_extra_turn_preserves_current_player_and_relative_actions() -> None:
    game = KalahGame(
        pits=[0, 0, 0, 0, 0, 1, 8, 8, 8, 8, 8, 8],
        captured_seeds=[0, 0],
        current_player=0,
    )
    assert game.possible_moves() == [5]
    assert game.move(game.pit_index(5))
    assert game.current_player == 0


def test_seat_one_state_keeps_relative_legal_action_indices() -> None:
    state = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [0, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 1,
    }
    result = run_search(
        state, simulations=8, seed=553099, evaluator=CountingEvaluator()
    )
    assert result["legal_moves"] == [1, 2, 3, 4, 5]
    assert result["visits"][0] == 0
    assert result["selected_move"] in result["legal_moves"]


def test_ordinary_puct_never_loads_or_queries_tablebase(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("tablebase_dependency_invoked")

    monkeypatch.setattr(
        native_exact_root_tablebase.NativeExactRootTablebase, "__init__", forbidden
    )
    monkeypatch.setattr(endgame_tablebase.EndgameTablebase, "lookup", forbidden)
    monkeypatch.setattr(endgame_tablebase.EndgameTablebase, "lookup_cached", forbidden)
    game = KalahGame(
        pits=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 1],
        captured_seeds=[0, 0],
        current_player=0,
    )
    assert sum(game.pits) == 17
    evaluator = CountingEvaluator()
    visits, _root = self_play.PUCT(
        evaluator,
        2,
        1.25,
        random.Random(553),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="visit_count",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    ).run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    assert int(visits.sum()) == 2
    assert evaluator.low_stone_leaf_calls > 0


def test_only_simulation_budget_varies() -> None:
    small = search_config(128)
    large = search_config(512)
    assert {**small, "simulations": 512} == large


def test_search_repeats_deterministically_and_excludes_illegal_actions() -> None:
    state = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    first_eval = CountingEvaluator()
    second_eval = CountingEvaluator()
    first = run_search(state, simulations=16, seed=19, evaluator=first_eval)
    second = run_search(state, simulations=16, seed=19, evaluator=second_eval)
    assert first["visits"] == second["visits"]
    assert first["target"] == second["target"]
    assert first_eval.calls == second_eval.calls
    assert all(
        first["target"][action] == 0
        for action in range(6)
        if action not in first["legal_moves"]
    )
    assert sum(first["target"]) == pytest.approx(1.0)
    assert first["evaluator_calls"] == first_eval.calls


def test_pair_metrics_and_tie_rule() -> None:
    low = {"target": [0.5, 0.5, 0, 0, 0, 0], "selected_move": 0}
    high = {"target": [0.5, 0.5, 0, 0, 0, 0], "selected_move": 1}
    report = compare_pair(low, high)
    assert report["action_disagreement"]
    assert report["total_variation"] == 0
    assert report["bitwise_identical_target"]


def test_root_state_is_validated_before_search() -> None:
    terminal = {
        "player_pits": [0] * 6,
        "opponent_pits": [4] * 6,
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    with pytest.raises(ValueError, match="root_outside"):
        run_search(terminal, simulations=1, seed=0, evaluator=CountingEvaluator())


def test_rejected_frozen_preflight_does_not_write_or_launch_searches(
    tmp_path, monkeypatch
) -> None:
    import ml.alphazero_lite.seed553_fixed_state_budget as pilot

    out = tmp_path / pilot.OUT
    out.mkdir(parents=True)
    registration = out / "registration.json"
    freeze = out / "execution-freeze.json"
    registration.write_text('{"schema":"seed553-fixed-state-budget-v1"}\n')
    freeze.write_text('{"registration_sha256":"wrong"}\n')
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (registration, freeze)
    }
    called = False

    def search(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("search_launched_after_rejected_preflight")

    monkeypatch.setattr(pilot, "run_search", search)
    with pytest.raises(ValueError, match="registration_freeze_mismatch"):
        execute(tmp_path)
    assert not called
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (registration, freeze)
    } == before
    assert not (out / "searches.jsonl").exists()
    assert not (out / "attempts.jsonl").exists()


def _rebound_row(row: dict) -> dict:
    body = {key: value for key, value in row.items() if key != "record_sha256"}
    row["record_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return row


def _semantic_fixture() -> tuple[dict, dict, str]:
    state = {
        "player_pits": [4, 4, 4, 4, 4, 4],
        "opponent_pits": [4, 4, 4, 4, 4, 4],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    item = {
        "cohort_index": 0,
        "bucket": ">32",
        "canonical_state_sha256": "canonical-fixture",
        "production_float32_sha256": "float32-fixture",
        "legal_moves": list(range(6)),
    }
    result = run_search(
        state, simulations=128, seed=553000, evaluator=CountingEvaluator()
    )
    registration_sha = "registration-fixture"
    row = _rebound_row(
        {
            "cohort_index": 0,
            "bucket": ">32",
            "canonical_state_sha256": item["canonical_state_sha256"],
            "production_float32_sha256": item["production_float32_sha256"],
            "simulations": 128,
            "seed": 553000,
            "registration_sha256": registration_sha,
            "result": result,
        }
    )
    return row, item, registration_sha


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row.update(canonical_state_sha256="rebound-wrong"),
            "binding_mismatch",
        ),
        (
            lambda row: (
                row.update(simulations=512),
                row["result"].update(simulations=512),
            ),
            "binding_mismatch|visits_invalid",
        ),
        (
            lambda row: (row.update(seed=553001), row["result"].update(seed=553001)),
            "seed",
        ),
        (
            lambda row: row["result"]["visits"].__setitem__(
                0, row["result"]["visits"][0] + 1
            ),
            "visits_invalid|target_does_not_match",
        ),
        (
            lambda row: row["result"]["target"].__setitem__(
                0, row["result"]["target"][0] / 2
            ),
            "target_does_not_match",
        ),
        (lambda row: row["result"].update(legal_moves=[0, 1]), "legal_action_mapping"),
    ],
)
def test_verifier_rejects_receipt_rebound_semantic_mutations(mutate, message) -> None:
    row, item, registration_sha = _semantic_fixture()
    mutated = copy.deepcopy(row)
    mutate(mutated)
    _rebound_row(mutated)
    with pytest.raises(ValueError, match=message):
        validate_search_row(
            mutated,
            item,
            budget=128,
            seed=553000,
            registration_sha256=registration_sha,
        )


def test_verifier_rejects_rebound_report_aggregate_mutation() -> None:
    row, item, _registration_sha = _semantic_fixture()
    large_result = run_search(
        item_state := {
            "player_pits": [4, 4, 4, 4, 4, 4],
            "opponent_pits": [4, 4, 4, 4, 4, 4],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
        simulations=512,
        seed=553000,
        evaluator=CountingEvaluator(),
    )
    _ = item_state
    paired_row = {
        "bucket": ">32",
        "searches": {"128": row["result"], "512": large_result},
    }
    pairs = [paired_row.copy() for _ in range(128)] + [
        {**paired_row, "bucket": "17-32"} for _ in range(128)
    ]
    expected = _reconstruct_report(pairs, 13818)
    mutated = copy.deepcopy(expected)
    mutated["overall"]["highest_visit_action_disagreements"] += 1
    with pytest.raises(ValueError, match="aggregate_mismatch"):
        validate_published_report(mutated, pairs, 13818)


def test_verifier_rejects_rebound_cohort_identity_mutation() -> None:
    cohort = build_cohort()
    mutated = copy.deepcopy(cohort)
    mutated[0]["canonical_state_sha256"] = "rebound-but-not-derived-from-source"
    with pytest.raises(ValueError, match="cohort_reconstruction_mismatch"):
        validate_cohort(Path(__file__).resolve().parents[2], mutated)


def _fake_search_result(budget: int, seed: int) -> dict:
    return {
        "simulations": budget,
        "seed": seed,
        "visits": [budget, 0, 0, 0, 0, 0],
        "target": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "selected_move": 0,
        "legal_moves": [0],
        "root_visit_count": budget,
        "root_q_value": 0.0,
        "root_evaluation_value": None,
        "root_children": {"0": {"prior": 1.0, "visits": budget, "q_value": 0.0}},
        "evaluator_calls": budget + 1,
        "elapsed_seconds": 0.01,
    }


def _execution_fixture(root: Path, monkeypatch):
    import ml.alphazero_lite.seed553_fixed_state_budget as pilot

    out = root / pilot.OUT
    out.mkdir(parents=True)
    cohort = [
        {
            "cohort_index": index,
            "bucket": ">32" if index < 128 else "17-32",
            "canonical_state_sha256": f"canonical-{index}",
            "production_float32_sha256": f"encoded-{index}",
            "legal_moves": [0],
            "state": {},
        }
        for index in range(256)
    ]
    (out / "cohort.json").write_bytes(pilot._json_bytes(cohort))
    (out / "registration.json").write_bytes(pilot._json_bytes({"fixture": True}))
    monkeypatch.setattr(pilot, "preflight", lambda _root: {})
    monkeypatch.setattr(pilot, "_verify_resource_scope", lambda: {"test": True})
    monkeypatch.setattr(
        pilot.self_play, "CheckpointEvaluator", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        pilot,
        "analyze_records",
        lambda _rows: {
            "overall": {
                "target": {
                    "128": {"total_search_seconds": 1.0},
                    "512": {"total_search_seconds": 2.0},
                }
            }
        },
    )
    corpus = root / pilot.CORPUS
    corpus.mkdir(parents=True)
    (corpus / "control-source-manifest.json").write_text('{"training_rows":13818}\n')
    return pilot, out, cohort


def test_resume_preserves_completed_search_without_rerunning(
    tmp_path, monkeypatch
) -> None:
    pilot, out, cohort = _execution_fixture(tmp_path, monkeypatch)
    registration_sha = pilot.sha_bytes((out / "registration.json").read_bytes())
    result = _fake_search_result(128, 553000)
    row = {
        "cohort_index": 0,
        "bucket": cohort[0]["bucket"],
        "canonical_state_sha256": cohort[0]["canonical_state_sha256"],
        "production_float32_sha256": cohort[0]["production_float32_sha256"],
        "simulations": 128,
        "seed": 553000,
        "registration_sha256": registration_sha,
        "result": result,
    }
    row["record_sha256"] = pilot._record_hash(row)
    search_line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    (out / "searches.jsonl").write_text(search_line)
    started = {
        "event": "started",
        "cohort_index": 0,
        "simulations": 128,
        "seed": 553000,
        "attempt_number": 1,
        "registration_sha256": registration_sha,
    }
    finished = {**started, "event": "finished", "record_sha256": row["record_sha256"]}
    (out / "attempts.jsonl").write_text(
        json.dumps(started) + "\n" + json.dumps(finished) + "\n"
    )
    launched = []

    def counted_search(_state, *, simulations, seed, evaluator):
        launched.append((seed, simulations))
        return _fake_search_result(simulations, seed)

    monkeypatch.setattr(pilot, "run_search", counted_search)
    pilot.execute(tmp_path)
    assert len(launched) == 511
    assert (553000, 128) not in launched
    assert (out / "searches.jsonl").read_text().startswith(search_line)


def test_conflicting_partial_search_is_rejected_before_any_new_search(
    tmp_path, monkeypatch
) -> None:
    pilot, out, cohort = _execution_fixture(tmp_path, monkeypatch)
    registration_sha = pilot.sha_bytes((out / "registration.json").read_bytes())
    result = _fake_search_result(128, 553000)
    row = {
        "cohort_index": 0,
        "bucket": cohort[0]["bucket"],
        "canonical_state_sha256": cohort[0]["canonical_state_sha256"],
        "production_float32_sha256": cohort[0]["production_float32_sha256"],
        "simulations": 128,
        "seed": 553000,
        "registration_sha256": registration_sha,
        "result": result,
        "record_sha256": "unauthenticated",
    }
    path = out / "searches.jsonl"
    path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes(), path.stat().st_mtime_ns
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("search_launched_for_conflicting_evidence")

    monkeypatch.setattr(pilot, "run_search", forbidden)
    with pytest.raises(ValueError, match="partial_search_receipt_mismatch"):
        pilot.execute(tmp_path)
    assert not called
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_interrupted_search_is_recorded_then_retried_without_losing_evidence(
    tmp_path, monkeypatch
) -> None:
    pilot, out, _cohort = _execution_fixture(tmp_path, monkeypatch)
    registration_sha = pilot.sha_bytes((out / "registration.json").read_bytes())
    initial = {
        "event": "started",
        "cohort_index": 0,
        "simulations": 128,
        "seed": 553000,
        "attempt_number": 1,
        "registration_sha256": registration_sha,
    }
    attempts_path = out / "attempts.jsonl"
    attempts_path.write_text(json.dumps(initial) + "\n")
    monkeypatch.setattr(
        pilot,
        "run_search",
        lambda _state, *, simulations, seed, evaluator: _fake_search_result(
            simulations, seed
        ),
    )
    pilot.execute(tmp_path)
    events = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    first_search = [
        event
        for event in events
        if event["cohort_index"] == 0 and event["simulations"] == 128
    ]
    assert [event["event"] for event in first_search] == [
        "started",
        "interrupted",
        "started",
        "finished",
    ]
    assert [event["attempt_number"] for event in first_search] == [1, 1, 2, 2]
