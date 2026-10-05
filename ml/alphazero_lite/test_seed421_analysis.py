import copy

import pytest

from ml.alphazero_lite.seed421_analysis import seed_for, validate_ledger
from ml.alphazero_lite.seed421_experiment import CountingEvaluator


def sample():
    root = {
        "state_hash": "a" * 64,
        "phase": ">32",
        "state": {
            "player_pits": [8, 0, 0, 0, 0, 0],
            "opponent_pits": [8] * 6,
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        },
    }
    output = {
        "selected_action": 0,
        "visits": [384, 0, 0, 0, 0, 0],
        "policy": [1.0, 0, 0, 0, 0, 0],
        "q_values": {"0": 0.5},
        "root_visit_count": 384,
        "root_q": 0.5,
    }
    row = {
        "root_hash": root["state_hash"],
        "phase": root["phase"],
        "budget": 384,
        "kind": "timed",
        "repetition": 0,
        "seed": seed_for(root["state_hash"], 384, 0),
        "registration_sha256": "b" * 64,
        "execution_order": ["off", "on"],
        "latency_ms": {"off": 1.0, "on": 0.9},
        "result": {"off": output, "on": copy.deepcopy(output)},
        "cache": {
            "hits": 0,
            "misses": 1,
            "requests": 1,
            "neural_calls": 1,
            "evictions": 0,
            "peak_entries": 1,
        },
        "request_counts": {"off": 1, "on": 1},
        "neural_call_counts": {"off": 1, "on": 1},
    }
    return root, row


def cohort32(root):
    roots = [copy.deepcopy(root) for _ in range(32)]
    for index, item in enumerate(roots):
        item["state_hash"] = f"{index:064x}"
        item["phase"] = ">32" if index < 16 else "17-32"
    roots[0] = root
    return roots


def test_timed_counting_wrapper_only_forwards_and_counts():
    class Evaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, game):
            self.calls += 1
            return object(), 0.25

    underlying = Evaluator()
    wrapper = CountingEvaluator(underlying)
    game = object()
    result = wrapper.evaluate(game)
    assert result[0] is not None and result[1] == 0.25
    assert wrapper.calls == underlying.calls == 1
    assert vars(wrapper) == {"evaluator": underlying, "calls": 1}


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("seed", "seed_mismatch"),
        ("action", "illegal_selected_action"),
        ("policy", "policy_visit_mismatch"),
        ("cache", "cache_request_count_mismatch"),
        ("timing", "invalid_timing"),
        ("identity", "registration_identity_mismatch"),
    ],
)
def test_rejects_corrupt_timed_pair(mutation, error):
    root, row = sample()
    if mutation == "seed":
        row["seed"] += 1
    elif mutation == "action":
        row["result"]["off"]["selected_action"] = 5
        row["result"]["on"]["selected_action"] = 5
    elif mutation == "policy":
        row["result"]["off"]["policy"][0] = 0.5
        row["result"]["on"]["policy"][0] = 0.5
    elif mutation == "cache":
        row["cache"]["hits"] = 1
    elif mutation == "timing":
        row["latency_ms"]["off"] = float("inf")
    elif mutation == "identity":
        row["registration_sha256"] = "c" * 64
    with pytest.raises(ValueError, match=error):
        validate_ledger([row], cohort32(root), "b" * 64, complete=False)


def test_resume_accepts_valid_partial_and_rejects_duplicate():
    root, row = sample()
    cohort = cohort32(root)
    validate_ledger([row], cohort, "b" * 64, complete=False)
    with pytest.raises(ValueError, match="duplicate_pair"):
        validate_ledger([row, copy.deepcopy(row)], cohort, "b" * 64, complete=False)
    with pytest.raises(ValueError, match="incomplete_evidence"):
        validate_ledger([row], cohort, "b" * 64, complete=True)


def test_resume_rejects_a_corrupted_recovered_record():
    root, row = sample()
    recovered = copy.deepcopy(row)
    recovered["request_counts"]["off"] += 1
    with pytest.raises(ValueError, match="timed_request_count_mismatch"):
        validate_ledger([recovered], cohort32(root), "b" * 64, complete=False)


def test_rejects_different_search_outputs():
    root, row = sample()
    row["result"]["on"]["root_q"] = 0.4
    with pytest.raises(ValueError, match="output_mismatch"):
        validate_ledger([row], cohort32(root), "b" * 64, complete=False)
