from __future__ import annotations

import json
import shutil

import pytest

import ml.alphazero_lite.seed557_retrospective as diagnostic
from ml.alphazero_lite.seed557_retrospective import (
    DATA,
    action_result,
    analyze,
    authenticate_oracle_assets,
    validate_existing_freeze,
)
from ml.alphazero_lite.verify_seed557_retrospective import verify


def test_cohort_selection_is_deterministic_and_balances_source_side() -> None:
    # Selection mechanics are exercised on small synthetic candidates via the
    # public reconstruction path in integration verification; this asserts the
    # canonical tied-optimum semantics used by the metrics.
    margins = {0: 4, 1: 4, 2: -2}
    assert action_result(margins, 0) == (0, False)
    assert action_result(margins, 1) == (0, False)
    assert action_result(margins, 2) == (6, True)


def test_action_regret_is_root_margin_regret_for_both_player_perspectives() -> None:
    # The native adapter expresses margins from the requested root perspective;
    # swapping perspectives negates every action margin and preserves regret.
    player_zero = {0: 5, 1: -1}
    player_one = {move: -margin for move, margin in player_zero.items()}
    assert action_result(player_zero, 0) == (0, False)
    assert action_result(player_zero, 1) == (6, True)
    assert action_result(player_one, 1) == (0, False)
    assert action_result(player_one, 0) == (6, True)


def _regenerate_metrics(data) -> None:
    cohort = json.loads((data / "cohort.json").read_text())["states"]
    searches = [
        json.loads(line)
        for line in (data / "search-records.jsonl").read_text().splitlines()
    ]
    oracles = [
        json.loads(line)
        for line in (data / "oracle-records.jsonl").read_text().splitlines()
    ]
    result = analyze(cohort, searches, oracles)
    root_metrics = result.pop("root_records")
    (data / "root-metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in root_metrics)
    )
    (data / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


def test_verifier_rejects_zero_visit_action_after_metrics_are_regenerated(
    tmp_path,
) -> None:
    data = tmp_path / "diagnostic"
    shutil.copytree(DATA, data)
    records_path = data / "search-records.jsonl"
    searches = [json.loads(line) for line in records_path.read_text().splitlines()]
    first = searches[0]
    original_action = first["puct_action"]
    assert first["visits"][original_action] == 83
    zero_visit = next(
        move for move in first["legal_moves"] if first["visits"][move] == 0
    )
    first["puct_action"] = zero_visit
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
    )
    _regenerate_metrics(data)
    with pytest.raises(ValueError, match="maximum-visit selection rule"):
        verify(data)


def test_verifier_rejects_search_seed_not_derived_from_frozen_state(tmp_path) -> None:
    data = tmp_path / "diagnostic"
    shutil.copytree(DATA, data)
    records_path = data / "search-records.jsonl"
    searches = [json.loads(line) for line in records_path.read_text().splitlines()]
    searches[0]["seed"] += 1
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
    )
    _regenerate_metrics(data)
    with pytest.raises(ValueError, match="search seed does not derive"):
        verify(data)


def test_verifier_checks_deterministic_tie_break_after_metrics_regeneration(
    tmp_path,
) -> None:
    data = tmp_path / "diagnostic"
    shutil.copytree(DATA, data)
    records_path = data / "search-records.jsonl"
    searches = [json.loads(line) for line in records_path.read_text().splitlines()]
    tied = next(
        row
        for row in searches
        if row["state_hash"]
        == "d1e75715982521ed8c3648011abec372abc227d80aa466b1b86158715852c433"
        and row["policy"] == "T512"
    )
    assert tied["puct_action"] == 5
    assert tied["visits"][3] == tied["visits"][5] == 64
    tied["puct_action"] = 3
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in searches)
    )
    _regenerate_metrics(data)
    with pytest.raises(ValueError, match="deterministic visit/Q/prior/index tie-break"):
        verify(data)


def test_conflicting_existing_freeze_is_rejected() -> None:
    original = {"search": {"simulations": 128}, "oracle": {"tablebase_sha256": "a"}}
    proposed = {"search": {"simulations": 128}, "oracle": {"tablebase_sha256": "b"}}
    with pytest.raises(ValueError, match="conflicting existing execution freeze"):
        validate_existing_freeze(original, proposed)


def test_offline_verifier_rejects_oracle_identity_change(tmp_path) -> None:
    data = tmp_path / "diagnostic"
    shutil.copytree(DATA, data)
    identity_path = data / "oracle-identity-verification.json"
    identity = json.loads(identity_path.read_text())
    identity["oracle_identities_in_frozen_execution"]["tablebase_size_bytes"] += 1
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="frozen oracle identities differ"):
        verify(data)


def test_prelaunch_oracle_authentication_checks_both_hashes_and_tablebase_size(
    tmp_path, monkeypatch
) -> None:
    native = tmp_path / "native-probe"
    tablebase = tmp_path / "tablebase"
    native.write_bytes(b"authenticated native probe")
    tablebase.write_bytes(b"authenticated tablebase payload")
    expected = {
        "native_probe_sha256": diagnostic.sha256(native.read_bytes()),
        "tablebase_sha256": diagnostic.sha256(tablebase.read_bytes()),
        "tablebase_size_bytes": tablebase.stat().st_size,
    }
    monkeypatch.setattr(
        diagnostic, "expected_oracle_identities", lambda: (expected, {})
    )
    assert authenticate_oracle_assets(native, tablebase) == expected

    tablebase.write_bytes(b"truncated tablebase")
    with pytest.raises(ValueError, match="do not match authenticated #556 identities"):
        authenticate_oracle_assets(native, tablebase)
