from ml.alphazero_lite.run_internal_state_cluster_provenance_audit import (
    FAMILY,
    PRIMARY_CASE,
    build_clusters,
    classify,
    cluster_signature,
)


def _state() -> dict:
    return {
        "player_pits": [1, 0, 0, 0, 0, 1],
        "opponent_pits": [1, 1, 1, 1, 1, 1],
        "player_store": 0,
        "opponent_store": 0,
        "current_player": 0,
    }


def test_cluster_signature_is_rules_only_and_deterministic() -> None:
    assert cluster_signature(_state()) == cluster_signature(_state())


def test_clusters_rank_events_then_stable_hash() -> None:
    event = {
        "case": PRIMARY_CASE,
        "family": FAMILY,
        "state": _state(),
        "canonical_pre_action_state": "a",
        "opportunities": 3,
    }
    clusters = build_clusters(
        [event, event, {**event, "canonical_pre_action_state": "b"}]
    )
    assert clusters[0]["events"] == 3
    assert clusters[0]["states"] == ["a", "b"]


def test_conservative_classification_prefers_observed_local_evidence() -> None:
    sources = [
        {
            "exact_effective_rows": 0,
            "parent_effective_rows": 0,
            "neighbor_effective_rows": 2,
        }
    ]
    states = [{"raw_policy": {"parent": {"top_is_outcome_optimal": True}}}]
    assert classify(states, sources) == "structural_neighbor_provenance_identified"
    assert (
        classify(states, [{**sources[0], "neighbor_effective_rows": 0}])
        == "no_replay_local_provenance_identified"
    )
