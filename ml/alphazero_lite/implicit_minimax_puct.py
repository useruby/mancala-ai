"""Diagnostic-only implicit-minimax PUCT; production PUCT is untouched."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ml.alphazero_lite.gumbel_root_search import CountingEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, HeuristicEvaluator, terminal_value


@dataclass
class ImplicitNode:
    game: KalahGame
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    m_value: float | None = None
    children: dict[int, "ImplicitNode"] = field(default_factory=dict)
    expanded: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


@dataclass(frozen=True)
class ImplicitMinimaxResult:
    selected_move: int
    visits: dict[int, int]
    q_values: dict[int, float]
    m_values: dict[int, float]
    priors: dict[int, float]
    exploration: dict[int, float]
    blended_selection_values: dict[int, float]
    evaluator_calls: int
    heuristic_evaluator_calls: int
    budget_padding_calls: int
    runtime_seconds: float


class ImplicitMinimaxPUCT:
    """Fixed-lambda PUCT with a separately backed-up heuristic minimax value."""

    def __init__(
        self, evaluator: Evaluator, *, seed: int, budget: int, lambda_: float = 0.25
    ):
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        self.evaluator = CountingEvaluator(evaluator)
        self.heuristic = HeuristicEvaluator()
        self.seed, self.budget, self.lambda_ = seed, budget, lambda_
        self.rng = random.Random(seed)
        self.c_puct = 1.25
        self.heuristic_calls = self.padding_calls = 0

    def _terminal(self, game: KalahGame) -> float | None:
        return terminal_value(game)

    def _heuristic(self, game: KalahGame) -> float:
        self.heuristic_calls += 1
        return float(np.clip(self.heuristic.evaluate(game)[1], -1.0, 1.0))

    def _expand(self, node: ImplicitNode) -> float:
        priors, value = self.evaluator.evaluate(node.game)
        legal = node.game.possible_moves()
        masked = np.zeros(6, dtype=np.float32)
        masked[legal] = priors[legal]
        total = float(masked.sum())
        if total <= 0:
            masked[legal] = 1.0 / len(legal)
        else:
            masked /= total
        for move in legal:
            child = node.game.clone()
            child.move(child.pit_index(move))
            node.children[move] = ImplicitNode(child, float(masked[move]))
        node.expanded = True
        node.m_value = self._heuristic(node.game)
        return float(value)

    @staticmethod
    def _transformed(parent: ImplicitNode, child: ImplicitNode, value: float) -> float:
        return (
            value if child.game.current_player == parent.game.current_player else -value
        )

    def _recompute_m(self, node: ImplicitNode) -> None:
        if not node.children:
            return
        values = []
        for child in node.children.values():
            child_m = child.m_value
            if child_m is not None:
                values.append(self._transformed(node, child, child_m))
        if values:
            node.m_value = max(values)

    def _entries(self, node: ImplicitNode):
        total = max(1, sum(child.visit_count for child in node.children.values()))
        entries = []
        for move, child in node.children.items():
            # Q is committed by _search after converting the returned child value
            # into the selecting parent's perspective, exactly like production PUCT.
            q = child.q_value if child.visit_count else 0.0  # production zero FPU
            m = child.m_value if child.m_value is not None else 0.0
            m = self._transformed(node, child, m)
            u = self.c_puct * child.prior * math.sqrt(total) / (1 + child.visit_count)
            blend = (1.0 - self.lambda_) * q + self.lambda_ * m
            entries.append((move, child, q, m, u, blend + u))
        return entries

    def _search(self, node: ImplicitNode) -> float:
        terminal = self._terminal(node.game)
        if terminal is not None:
            node.m_value = terminal
            return terminal
        if not node.expanded:
            return self._expand(node)
        move, child, *_ = max(
            self._entries(node), key=lambda item: (item[-1], -item[0])
        )
        value = self._search(child)
        if child.game.current_player != node.game.current_player:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        self._recompute_m(node)
        return value

    def run(self, game: KalahGame) -> ImplicitMinimaxResult:
        started = time.perf_counter()
        root = ImplicitNode(game.clone())
        self._expand(root)
        if not root.children:
            raise ValueError("root must be non-terminal")
        stalled = 0
        while self.evaluator.calls < self.budget:
            before = self.evaluator.calls
            value = self._search(root)
            root.visit_count += 1
            root.value_sum += value
            if self.evaluator.calls == before:
                stalled += 1
            else:
                stalled = 0
            if stalled >= max(16, len(root.children) * 4):
                self.evaluator.evaluate(root.game)
                self.padding_calls += 1
                stalled = 0
        entries = self._entries(root)
        by_move = {
            move: (child, q, m, u, score) for move, child, q, m, u, score in entries
        }
        selected = max(
            root.children,
            key=lambda move: (
                root.children[move].visit_count,
                root.children[move].q_value,
                root.children[move].prior,
                -move,
            ),
        )
        return ImplicitMinimaxResult(
            selected,
            {m: c.visit_count for m, c in root.children.items()},
            {m: c.q_value for m, c in root.children.items()},
            {m: by_move[m][2] for m in root.children},
            {m: c.prior for m, c in root.children.items()},
            {m: by_move[m][3] for m in root.children},
            {m: by_move[m][4] for m in root.children},
            self.evaluator.calls,
            self.heuristic_calls,
            self.padding_calls,
            time.perf_counter() - started,
        )


def run_implicit_minimax_puct(
    game: KalahGame,
    evaluator: Evaluator,
    *,
    seed: int,
    budget: int = 384,
    lambda_: float = 0.25,
) -> ImplicitMinimaxResult:
    result = ImplicitMinimaxPUCT(
        evaluator, seed=seed, budget=budget, lambda_=lambda_
    ).run(game)
    if result.evaluator_calls != budget:
        raise RuntimeError("implicit-minimax PUCT evaluation budget contract failed")
    return result
