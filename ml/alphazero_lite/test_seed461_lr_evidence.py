import hashlib
import json

import pytest

from ml.alphazero_lite.seed461_lr_evidence import (
    validate_games,
    validate_training_record,
)
from ml.alphazero_lite.analyze_seed461_lr_sensitivity import _validate_report
from ml.alphazero_lite.run_seed461_lr_sensitivity import validate_runtime


def test_training_record_binds_epoch_and_selected_checkpoint(tmp_path):
    record = {
        "trajectories": {
            "A": {
                "epochs": {"E1": "a" * 64, "E2": "b" * 64},
                "selected_sha256": "b" * 64,
            }
        }
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(record))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_training_record(
        record,
        record_path=path,
        expected_record_sha256=digest,
        expected_epochs={"A": record["trajectories"]["A"]["epochs"]},
        expected_selected={"A": "b" * 64},
    )
    with pytest.raises(ValueError, match="selected_checkpoint"):
        validate_training_record(
            record,
            record_path=path,
            expected_record_sha256=digest,
            expected_epochs={"A": record["trajectories"]["A"]["epochs"]},
            expected_selected={"A": "a" * 64},
        )


def _games():
    base = {
        "suite_sha256": "s",
        "opponent_identity": "o",
        "candidate_identity": "c",
        "runtime_identity": "r",
        "opening_index": 0,
        "opening_state": {"pits": [8]},
        "winner": "draw",
    }
    return [{**base, "challenger_player": seat} for seat in (0, 1)]


def test_game_validation_requires_identities_and_both_seats():
    openings = {0: json.dumps({"pits": [8]}, sort_keys=True, separators=(",", ":"))}
    kwargs = dict(
        expected_suite="s",
        expected_opponent="o",
        expected_candidate="c",
        expected_runtime="r",
        registered_openings=openings,
    )
    assert len(validate_games(_games(), **kwargs)[0]) == 2
    with pytest.raises(ValueError, match="unknown_winner"):
        validate_games([{**_games()[0], "winner": "other"}, _games()[1]], **kwargs)
    with pytest.raises(ValueError, match="seat_pairing"):
        validate_games([_games()[0], _games()[0]], **kwargs)
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate_games(
            [{**_games()[0], "runtime_identity": "wrong"}, _games()[1]], **kwargs
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("suite_sha256", "wrong", "suite_sha256"),
        ("challenger_path", "wrong", "challenger_path"),
        ("current_path", "wrong", "current_path"),
        ("exact_root_solve_threshold", 99, "runtime_identity"),
    ],
)
def test_report_identity_validation_rejects_mismatched_suite_candidates_and_runtime(
    field, value, expected_error
):
    report = {
        "games_played": 512,
        "score": 0.5,
        "notes": {
            "suite_sha256": "suite",
            "challenger_path": "candidate",
            "current_path": "opponent",
            "challenger_simulations": 384,
            "current_simulations": 384,
            "seed": 384,
            "search_profile_hash": "profile-hash",
            "search_profile": {
                "hash": "profile-hash",
                "c_puct": 1.25,
                "simulations": 384,
            },
            "exact_root_solve_threshold": 16,
        },
    }
    expected_runtime = {"exact_root_solve_threshold": 16}
    if field in {"suite_sha256", "challenger_path", "current_path"}:
        report["notes"][field] = value
    else:
        expected_runtime[field] = value
    with pytest.raises(ValueError, match=expected_error):
        _validate_report(
            report,
            run="test",
            candidate={"artifact": "candidate"},
            opponent={"artifact": "opponent"},
            expected_suite_hash="suite",
            expected_runtime=expected_runtime,
        )


def test_runtime_preflight_rejects_changed_registered_artifact(tmp_path, monkeypatch):
    import ml.alphazero_lite.run_seed461_lr_sensitivity as runner

    opponent = tmp_path / "opponent"
    challenger = tmp_path / "challenger"
    opponent.mkdir()
    challenger.mkdir()
    for artifact in (opponent, challenger):
        (artifact / "search_policy.json").write_text("{}")
    (opponent / "weights.json").write_text("weights")
    (opponent / "metadata.json").write_text("metadata")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    expected = {
        "runtime_contract": {"contract": "registered"},
        "opponent_artifact_identity": {
            "weights.json": hashlib.sha256(b"weights").hexdigest(),
            "metadata.json": hashlib.sha256(b"metadata").hexdigest(),
        },
    }
    monkeypatch.setattr(
        runner,
        "resolve_strength_comparison_runtime_contract",
        lambda **_: {"contract": "changed"},
    )
    with pytest.raises(RuntimeError, match="registered_runtime_contract_mismatch"):
        validate_runtime({"evaluation": expected})
