from __future__ import annotations

from ml.alphazero_lite.run_pr268_gumbel_root_preflight import (
    build_corpus,
    classify,
    hierarchical_bootstrap,
)


def _row(state_hash: str, regret: float) -> dict:
    return {
        "state_hash": state_hash,
        "phase": "opening",
        "player": 0,
        "legal_action_count": 3,
        "capture_available": False,
        "extra_turn_available": False,
        "puct": {
            "regret": regret,
            "exact_best_action": True,
            "catastrophic_miss": False,
        },
        "gumbel": {
            "regret": regret / 2,
            "exact_best_action": True,
            "catastrophic_miss": False,
        },
    }


def test_corpus_is_reachable_diagnostic_only_and_deterministic() -> None:
    left = build_corpus(count=12)
    right = build_corpus(count=12)
    assert left == right
    assert all(row["diagnostic_only"] and row["not_training_eligible"] for row in left)
    assert all(1 < len(row["legal_actions"]) <= 6 for row in left)


def test_hierarchical_bootstrap_and_classification() -> None:
    rows = [_row(str(index), 0.4) for index in range(12)]
    interval = hierarchical_bootstrap(rows, seed=1)
    assert interval["upper_95"] < 0
    assert (
        classify(rows, budget_ok=True, invariants_ok=True)
        == "gumbel_root_interface_qualified"
    )


def test_budget_failure_precedes_other_gate() -> None:
    assert (
        classify([_row("a", 0.4)], budget_ok=False, invariants_ok=False)
        == "gumbel_root_budget_contract_failed"
    )
