import copy

import pytest

from ml.alphazero_lite.seed420_analysis import analyze, validate_ledger


def example():
    cohort = [
        {
            "state_hash": f"root-{index}",
            "phase": ">32" if index < 16 else "17-32",
        }
        for index in range(32)
    ]
    output = {
        "selected_action": 1,
        "visits": [0, 2, 0, 0, 0, 0],
        "policy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "q_values": {"1": 0.25},
        "root_visit_count": 2,
        "root_q": 0.25,
    }
    trace = [{"state_sha256": "s", "policy": [1.0], "value": 0.0}]
    rows = []
    for root in cohort:
        for budget in (384, 1536):
            rows.append(
                {
                    "root_hash": root["state_hash"],
                    "phase": root["phase"],
                    "budget": budget,
                    "kind": "parity",
                    "repetition": 0,
                    "result": {
                        "off": output,
                        "on": output,
                        "evaluation_requests": 1,
                        "off_request_trace": trace,
                        "on_request_trace": trace,
                        "neural_calls_off": 1,
                        "neural_calls_on": 1,
                        "cache": {"misses": 1},
                    },
                }
            )
            for repetition in range(3):
                rows.append(
                    {
                        "root_hash": root["state_hash"],
                        "phase": root["phase"],
                        "budget": budget,
                        "kind": "timed",
                        "repetition": repetition,
                        "execution_order": ["off", "on"]
                        if repetition % 2 == 0
                        else ["on", "off"],
                        "latency_ms": {"off": 10.0, "on": 8.0},
                        "result": {"off": output, "on": output},
                        "cache": {
                            "hits": 1,
                            "misses": 1,
                            "evictions": 0,
                            "neural_calls": 1,
                            "requests": 2,
                            "peak_entries": 1,
                        },
                        "request_counts": {"off": 2, "on": 2},
                        "neural_call_counts": {"off": 2, "on": 1},
                    }
                )
    return cohort, rows


def test_complete_accounting_metrics_and_decision():
    cohort, ledger = example()
    result = analyze(ledger, cohort)
    assert result["search_count"] == 512
    assert abs(result["by_budget"]["384"]["mean_paired_relative_speedup"] - 0.2) < 1e-12


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("partial", "incomplete_pairs"),
        ("duplicate", "duplicate_pair"),
        ("substituted", "substituted_or_unregistered_pair"),
        ("mismatched", "search_output_mismatch"),
    ],
)
def test_rejects_invalid_ledger_pairs(mutation, error):
    cohort, rows = example()
    if mutation == "partial":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "substituted":
        rows[0]["root_hash"] = "unregistered"
    else:
        rows[0]["result"]["on"] = copy.deepcopy(rows[0]["result"]["on"])
        rows[0]["result"]["on"]["selected_action"] = 5
    with pytest.raises(ValueError, match=error):
        validate_ledger(rows, cohort)
