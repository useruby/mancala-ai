from ml.alphazero_lite.run_internal_family_replay_provenance_audit import (
    FAMILY,
    dominant_mechanism,
    lineage,
    ordered_state_sha,
    primary_failure_states,
    teacher_student_inversion,
    verify_source,
)


def test_primary_extraction_is_deterministic_and_keeps_selected_family() -> None:
    events = [
        {
            "family": FAMILY,
            "case": "45:uniform1200:capture_available-018",
            "canonical_pre_action_state": "b",
            "state": {},
            "selected_action": 2,
            "opportunities": 4,
            "depth": 3,
            "optimal_actions": [1],
        },
        {
            "family": FAMILY,
            "case": "45:uniform1200:capture_available-018",
            "canonical_pre_action_state": "b",
            "state": {},
            "selected_action": 2,
            "opportunities": 4,
            "depth": 1,
            "optimal_actions": [1],
        },
    ]
    rows = primary_failure_states(list(reversed(events)))
    assert rows[0]["events"] == 2 and rows[0]["degradation_rate"] == 0.5
    assert ordered_state_sha(["b"]) == ordered_state_sha(["b"])


def test_lineage_categories() -> None:
    bad, good = {"top_is_outcome_optimal": False}, {"top_is_outcome_optimal": True}
    assert lineage(bad, good, bad) == "control_repairs_uniform_retains"
    assert lineage(good, good, bad) == "uniform_introduces_failure"


def test_unavailable_source_is_not_zero(tmp_path) -> None:
    result = verify_source(tmp_path / "missing.jsonl", "abc")
    assert result["availability"] == "historical_source_unavailable"
    assert result["sha256"] is None


def test_dominance_requires_both_weightings() -> None:
    rows = [{"mechanism": "a", "opportunities": 1}] * 3 + [
        {"mechanism": "b", "opportunities": 100}
    ] * 2
    assert dominant_mechanism(rows) == "internal_prior_failure_heterogeneous"


def test_teacher_student_inversion_uses_aggregate_target() -> None:
    state = {
        "player_pits": [1, 1, 0, 0, 0, 0],
        "opponent_pits": [1, 1, 0, 0, 0, 0],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }
    raw_bad = {"top_is_outcome_optimal": False, "exact_optimal_mass": 0.1}
    raw_good = {"top_is_outcome_optimal": True, "exact_optimal_mass": 0.9}
    assert teacher_student_inversion(
        {"policy": [1, 0, 0, 0, 0, 0]},
        None,
        raw_bad,
        raw_good,
        state,
        [0],
        1,
        {"0": 1, "1": 0},
    )
