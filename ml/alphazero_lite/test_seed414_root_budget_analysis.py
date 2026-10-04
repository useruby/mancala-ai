from __future__ import annotations

import hashlib
import json

import pytest

from ml.alphazero_lite.seed414_root_budget_analysis import calculate


def evidence(changed: bool = True):
    states = [
        {"opening_index": index, "state_hash": f"state-{index}"} for index in range(128)
    ]
    registration = {"states": states}
    probes = []
    outcomes = []
    aliases = []
    for state in states:
        index = state["opening_index"]
        action384, action1536 = 0, (1 if changed and index % 2 == 0 else 0)
        probes.append(
            {
                "opening_index": index,
                "state_hash": state["state_hash"],
                "seed": index + 414,
                "seed_context_hash": "c" * 64,
                "elapsed_wall_seconds": 1.0,
                "snapshots": {
                    "384": {"action": action384},
                    "1536": {"action": action1536},
                },
            }
        )
        for reference in ("seed455", "original_O0_E4"):
            for action in {action384, action1536}:
                outcome = {
                    "state_hash": state["state_hash"],
                    "score": 0.5 + 0.5 * (action == 1),
                    "store_margin_root_perspective": action,
                    "elapsed_wall_seconds": 2.0,
                }
                outcomes.append(
                    {
                        "opening_index": index,
                        "reference": reference,
                        "forced_action": action,
                        "outcome": outcome,
                    }
                )
            for budget, action in ((384, action384), (1536, action1536)):
                source = next(
                    row["outcome"]
                    for row in outcomes
                    if row["opening_index"] == index
                    and row["reference"] == reference
                    and row["forced_action"] == action
                )
                aliases.append(
                    {
                        "opening_index": index,
                        "reference": reference,
                        "root_budget": budget,
                        "action": action,
                        "trajectory_identity": hashlib.sha256(
                            json.dumps(source, sort_keys=True).encode()
                        ).hexdigest(),
                        "alias_of": {"root_budget": 384}
                        if budget == 1536 and action384 == action1536
                        else None,
                    }
                )
    return registration, probes, outcomes, aliases


def test_analysis_keeps_unchanged_actions_as_zero_gains() -> None:
    reg, probes, outcomes, aliases = evidence()
    report, matrix = calculate(reg, probes, outcomes, aliases)
    assert len(matrix) == 512
    assert report["primary"]["paired_units"] == 128
    assert report["state_results"][1]["zero_gain_observation"]
    assert report["state_results"][1]["paired_gain"] == 0


def test_analysis_rejects_duplicate_physical_continuation() -> None:
    reg, probes, outcomes, aliases = evidence()
    outcomes.append(outcomes[0])
    with pytest.raises(ValueError, match="duplicate_physical_trajectory"):
        calculate(reg, probes, outcomes, aliases)


def test_analysis_requires_full_logical_case_coverage() -> None:
    reg, probes, outcomes, aliases = evidence()
    aliases.pop()
    with pytest.raises(ValueError, match="logical_case_coverage_mismatch"):
        calculate(reg, probes, outcomes, aliases)
