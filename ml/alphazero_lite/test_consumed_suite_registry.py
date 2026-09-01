"""Regression coverage for the canonical-through-AA consumed-suite contract."""

from pathlib import Path

import pytest

from ml.alphazero_lite import build_opening_suite as suites
from ml.alphazero_lite import consumed_suite_registry as registry
from ml.alphazero_lite import run_pr254_third_seed_budget_replay as pr254


def test_registry_is_unique_deterministic_and_order_independent(tmp_path: Path) -> None:
    consumed = registry.load(tmp_path)
    assert list(consumed) == ["canonical", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "AA"]
    assert len(consumed) == len(set(consumed))
    reversed_registry = dict(reversed(list(consumed.items())))
    assert registry.final_keys(consumed) == registry.final_keys(reversed_registry)
    assert registry.prefix_keys(consumed) == registry.prefix_keys(reversed_registry)
    excluded, _ = pr254.consumed_opening_exclusions(consumed)
    assert excluded == pr254.consumed_opening_exclusions(reversed_registry)[0]


def test_j_is_a_required_overlap_exclusion_and_prefixes_preflight(
    tmp_path: Path,
) -> None:
    consumed = registry.load(tmp_path)
    j_entry = suites.load_suite_jsonl(str(consumed["J"].path))[0]
    assert suites.canonical_key(j_entry["state"]) in registry.final_keys(consumed)
    without_j = {label: spec for label, spec in consumed.items() if label != "J"}
    with pytest.raises(
        RuntimeError, match="invariant_failure: consumed-suite registry mismatch"
    ):
        pr254.seal_suites(tmp_path, [], without_j)
    suite_path = tmp_path / "suite_P.jsonl"
    suites.write_suite_jsonl([j_entry], str(suite_path))
    with pytest.raises(
        RuntimeError, match="invariant_failure: P/Q/R preflight overlap"
    ):
        pr254.preflight_suites({"P": suite_path}, consumed, [])
