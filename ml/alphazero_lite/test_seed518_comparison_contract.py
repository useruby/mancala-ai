"""Semantic tests for the narrowly amended seed518 comparison contract."""

from __future__ import annotations

import copy

import pytest

from ml.alphazero_lite.seed518_selective_head import compare_search_outputs


@pytest.fixture
def search_output() -> dict:
    return {
        "selected_move": 2,
        "visits": [3, 0, 9, 0, 0, 0],
        "policy": [0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "value": 0.25,
        "search_root_value": 0.125,
        "child_stats": [{"move": 2, "visits": 9, "q_value": 0.5}],
        "nonterminal_leaf_count": 12,
        "root_latency_ms": 10.0,
        "exact_root_decision": {"root_latency_ms": 11.0, "optimal_moves": [2]},
    }


def test_newly_excluded_root_latency_only_passes(search_output: dict) -> None:
    changed = copy.deepcopy(search_output)
    changed["root_latency_ms"] = 999.0
    assert compare_search_outputs(search_output, changed) == []


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("selected_move",), 3),
        (("visits", 2), 8),
        (("policy", 2), 0.7),
        (("value",), 0.2),
        (("search_root_value",), 0.2),
        (("child_stats", 0, "q_value"), 0.4),
        (("nonterminal_leaf_count",), 11),
        (("exact_root_decision", "optimal_moves", 0), 3),
    ],
)
def test_semantic_output_mutations_fail(
    search_output: dict, path: tuple, replacement: object
) -> None:
    changed = copy.deepcopy(search_output)
    target = changed
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement
    assert compare_search_outputs(search_output, changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [("unregistered_extra", 1), ("latency_probe_ms", 2.0), ("timing_ms", 3.0)],
)
def test_extra_or_undeclared_timing_fields_fail(
    search_output: dict, field: str, value: object
) -> None:
    changed = copy.deepcopy(search_output)
    changed[field] = value
    assert compare_search_outputs(search_output, changed) == [field]
