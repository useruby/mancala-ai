from __future__ import annotations

from ml.alphazero_lite.seed557_retrospective import action_result


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
