"""Diagnostic-only, exact-budget Gumbel sequential-halving root search.

This module deliberately subclasses rather than edits :class:`PUCT`; production
search remains byte-for-byte independent of this PR #268 experiment.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, PUCT

EVALUATION_BUDGET = 384
CPUCT = 1.25
GUMBEL_SCALE = 1.0


class CountingEvaluator(Evaluator):
    """Count evaluator invocations, including the initial root expansion."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.calls = 0

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        self.calls += 1
        return self.evaluator.evaluate(game)


@dataclass(frozen=True)
class RootSearchResult:
    """Stable telemetry emitted by both diagnostic root-allocation lanes."""

    selected_move: int
    visits: dict[int, int]
    q_values: dict[int, float]
    evaluator_calls: int
    budget_padding_calls: int
    all_legal_actions_initially_visited: bool


class _ExactBudgetPUCT(PUCT):
    """PUCT with a neural-evaluation stop condition instead of a sim count."""

    def __init__(self, evaluator: CountingEvaluator, *, seed: int, budget: int):
        super().__init__(
            evaluator,
            simulations=budget,
            c_puct=CPUCT,
            rng=random.Random(seed),
            fpu_mode="zero",
            reuse_subtree=False,
            normalize_values=False,
            root_policy_mode="deterministic",
            tactical_root_bias=0.0,
            root_temperature=0.0,
        )
        self.budget = int(budget)
        self._diagnostic_root: Node | None = None

    def run_exact_budget(self, root_game: KalahGame) -> RootSearchResult:
        if self.budget < 1:
            raise ValueError("budget must include the root evaluation")
        root = self._root_for(root_game)
        self._diagnostic_root = root
        self._expand(
            root,
            apply_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            is_root=True,
        )
        legal_moves = sorted(root.children)
        if not legal_moves:
            raise ValueError("root must be non-terminal")
        iterations = 0
        padding_calls = 0
        stalled = 0
        # Terminal leaves do not invoke the evaluator. The guard makes a malformed
        # evaluator/tree fail rather than silently violating the budget contract.
        while self.evaluator.calls < self.budget:
            iterations += 1
            if iterations > self.budget * 128:
                raise RuntimeError("unable to consume exact neural-evaluation budget")
            self._active_simulation_index = iterations
            calls_before = self.evaluator.calls
            value = self._search(root)
            root.visit_count += 1
            root.value_sum += value
            if self.evaluator.calls == calls_before:
                stalled += 1
            else:
                stalled = 0
            if stalled >= max(16, len(legal_moves) * 4):
                # A terminal-only selected subtree cannot produce another leaf
                # evaluation. Pad only with the already-expanded root input so
                # both allocation lanes retain the exact, paired neural budget.
                self.evaluator.evaluate(root.game)
                padding_calls += 1
                stalled = 0
        self._active_simulation_index = None
        visits = {move: int(child.visit_count) for move, child in root.children.items()}
        return RootSearchResult(
            selected_move=int(self.select_root_move(root, legal_moves)),
            visits=visits,
            q_values={
                move: float(child.q_value) for move, child in root.children.items()
            },
            evaluator_calls=self.evaluator.calls,
            budget_padding_calls=padding_calls,
            all_legal_actions_initially_visited=all(
                visits[move] > 0 for move in legal_moves
            ),
        )


class _GumbelSequentialHalvingPUCT(_ExactBudgetPUCT):
    """Keep root allocations balanced within successive Gumbel-ranked rounds."""

    def __init__(self, evaluator: CountingEvaluator, *, seed: int, budget: int):
        super().__init__(evaluator, seed=seed, budget=budget)
        self._gumbel_rng = np.random.default_rng(seed)
        self._gumbels: dict[int, float] = {}
        self._active_moves: list[int] = []
        self._round = 0
        self._round_start_visits = 0
        self._round_quota = 0

    def _root_score(self, move: int) -> tuple[float, int]:
        assert self._diagnostic_root is not None
        child = self._diagnostic_root.children[move]
        # Q lies in [-1, 1]; this fixed affine mapping is the preregistered,
        # allocation-independent normalization and is never fit from results.
        normalized_q = (float(child.q_value) + 1.0) / 2.0
        score = (
            math.log(max(float(child.prior), 1e-12))
            + self._gumbels[move]
            + normalized_q
        )
        return score, -move

    def _advance_round_if_complete(self, root: Node) -> None:
        used = sum(root.children[move].visit_count for move in self._active_moves)
        if used - self._round_start_visits < self._round_quota:
            return
        if len(self._active_moves) == 1:
            return
        survivors = max(1, math.ceil(len(self._active_moves) / 2))
        self._active_moves = sorted(
            sorted(
                self._active_moves,
                key=lambda move: self._root_score(move),
                reverse=True,
            )[:survivors]
        )
        self._round += 1
        self._round_start_visits = sum(
            root.children[move].visit_count for move in self._active_moves
        )
        rounds_remaining = max(1, math.ceil(math.log2(len(self._active_moves))))
        remaining = max(
            1, self.budget - sum(child.visit_count for child in root.children.values())
        )
        self._round_quota = max(len(self._active_moves), remaining // rounds_remaining)

    def _select_child(self, node: Node) -> Node:
        if node is not self._diagnostic_root:
            return super()._select_child(node)
        if not self._active_moves:
            self._active_moves = sorted(node.children)
            self._gumbels = {
                move: float(self._gumbel_rng.gumbel(0.0, GUMBEL_SCALE))
                for move in self._active_moves
            }
            rounds = max(1, math.ceil(math.log2(len(self._active_moves))))
            self._round_quota = max(len(self._active_moves), self.budget // rounds)
        self._advance_round_if_complete(node)
        # Controlled consideration: every surviving action receives one visit
        # before any receives a second visit in each sequential-halving round.
        move = min(
            self._active_moves,
            key=lambda candidate: (
                node.children[candidate].visit_count,
                -self._root_score(candidate)[0],
                candidate,
            ),
        )
        return node.children[move]


def run_puct_root_search(
    game: KalahGame, evaluator: Evaluator, *, seed: int, budget: int = EVALUATION_BUDGET
) -> RootSearchResult:
    """Run the preregistered ordinary PUCT allocation lane."""
    counted = CountingEvaluator(evaluator)
    result = _ExactBudgetPUCT(counted, seed=seed, budget=budget).run_exact_budget(game)
    if result.evaluator_calls != budget:
        raise RuntimeError("ordinary PUCT evaluation budget contract failed")
    return result


def run_gumbel_root_search(
    game: KalahGame, evaluator: Evaluator, *, seed: int, budget: int = EVALUATION_BUDGET
) -> RootSearchResult:
    """Run the preregistered seeded-Gumbel sequential-halving allocation lane."""
    counted = CountingEvaluator(evaluator)
    result = _GumbelSequentialHalvingPUCT(
        counted, seed=seed, budget=budget
    ).run_exact_budget(game)
    if result.evaluator_calls != budget:
        raise RuntimeError("Gumbel evaluation budget contract failed")
    return result
