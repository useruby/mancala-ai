from __future__ import annotations

from ml.alphazero_lite.run_true_outcome_subtree_attribution_audit import (
    PRIMARY,
    classify,
    root_perspective,
    stable_first,
)


def test_primary_population_is_exactly_the_two_pr301_outcome_regressions() -> None:
    assert PRIMARY == (
        (44, "capture_available-025", 0),
        (45, "capture_available-018", 1),
    )


def test_backup_reconstruction_uses_player_identity_for_extra_turns() -> None:
    # A repeated player is an extra-turn edge and must not flip by depth parity.
    assert root_perspective(0.4, leaf_player=0, root_player=0) == 0.4
    assert root_perspective(0.4, leaf_player=1, root_player=0) == -0.4


def test_stable_counterfactual_repair_requires_remaining_ordering() -> None:
    assert stable_first([False, True, False, True, True]) == 4


def test_mechanism_precedence_is_backup_then_evaluator_then_subtree() -> None:
    assert (
        classify(
            coverage_ok=True,
            backup_mismatches=1,
            repaired=True,
            self_degradations=3,
            solved_terminal_fraction=0.1,
        )
        == "backup_perspective_fault"
    )
    assert (
        classify(
            coverage_ok=True,
            backup_mismatches=0,
            repaired=True,
            self_degradations=3,
            solved_terminal_fraction=0.1,
        )
        == "leaf_evaluator_error"
    )
    assert (
        classify(
            coverage_ok=True,
            backup_mismatches=0,
            repaired=False,
            self_degradations=3,
            solved_terminal_fraction=0.1,
        )
        == "subtree_policy_error"
    )


def test_insufficient_exact_coverage_takes_precedence_over_attribution() -> None:
    assert (
        classify(
            coverage_ok=False,
            backup_mismatches=0,
            repaired=True,
            self_degradations=1,
            solved_terminal_fraction=0.1,
        )
        == "root_attribution_inconclusive"
    )
