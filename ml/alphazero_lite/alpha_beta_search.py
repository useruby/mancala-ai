"""Diagnostic-only, policy-free iterative-deepening alpha-beta for Kalah."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from ml.alphazero_lite.kalah_rules import KalahGame, move_consequence_for_state

BoundType = Literal["exact", "lower", "upper"]


class ScalarEvaluator(Protocol):
    """Return a scalar from the supplied game's current-player perspective."""

    lane: str

    def evaluate_value(self, game: KalahGame) -> float: ...


class ArtifactValueEvaluator:
    lane = "artifact_value"

    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator

    def evaluate_value(self, game: KalahGame) -> float:
        _policy, value = self.evaluator.evaluate(game)
        return float(value)


class HeuristicValueEvaluator:
    lane = "heuristic_value"

    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator

    def evaluate_value(self, game: KalahGame) -> float:
        _policy, value = self.evaluator.evaluate(game)
        return float(value)


@dataclass(frozen=True)
class TranspositionEntry:
    depth: int
    value: float
    bound: BoundType
    best_move: int | None
    solved: bool = False


@dataclass(frozen=True)
class AlphaBetaResult:
    selected_move: int
    root_action_values: dict[int, float]
    completed_depth: int
    principal_variation: list[int]
    leaf_evaluator_calls: int
    nodes_visited: int
    terminal_nodes: int
    alpha_beta_cutoffs: int
    transposition_table_hits: int
    transposition_table_cutoffs: int
    budget_utilization: float
    deeper_iteration_abandoned: bool
    root_solved_exactly: bool
    node_cap_reached: bool
    runtime_seconds: float
    evaluator_lane: str
    legal_moves: list[int]
    move_ordering: list[int]


class SearchBudgetExceeded(RuntimeError):
    pass


class _NodeCapExceeded(RuntimeError):
    pass


def canonical_state_key(
    game: KalahGame,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int | None, bool]:
    return (
        tuple(game.pits),
        tuple(game.captured_seeds),
        int(game.current_player),
        game.winner,
        bool(game._over),
    )


class AlphaBetaSearch:
    """Per-root alpha-beta search with an in-memory, nonpersistent TT."""

    def __init__(
        self,
        evaluator: ScalarEvaluator,
        *,
        leaf_evaluation_budget: int = 384,
        node_cap: int = 100_000,
        use_transposition_table: bool = True,
    ) -> None:
        if leaf_evaluation_budget < 1:
            raise ValueError("leaf_evaluation_budget must be >= 1")
        self.evaluator = evaluator
        self.leaf_evaluation_budget = int(leaf_evaluation_budget)
        self.node_cap = int(node_cap)
        self.use_transposition_table = bool(use_transposition_table)

    def search(self, root_game: KalahGame) -> AlphaBetaResult:
        if root_game.over():
            raise ValueError("root must be non-terminal")
        legal_moves = sorted(root_game.possible_moves())
        if not legal_moves:
            raise ValueError("root must have legal moves")
        self.root_player = root_game.current_player
        self.table: dict[
            tuple[tuple[int, ...], tuple[int, ...], int, int | None, bool],
            TranspositionEntry,
        ] = {}
        self.leaf_calls = self.nodes = self.terminals = self.cutoffs = 0
        self.tt_hits = self.tt_cutoffs = 0
        self.node_cap_reached = False
        started = time.perf_counter()
        last_values: dict[int, float] | None = None
        last_pv: list[int] = []
        completed_depth = 0
        root_solved = False
        abandoned = False
        depth = 1
        while True:
            try:
                values, pv, solved = self._search_root(root_game, depth)
            except (SearchBudgetExceeded, _NodeCapExceeded):
                abandoned = True
                break
            last_values = values
            last_pv = pv
            completed_depth = depth
            root_solved = solved
            if solved:
                break
            depth += 1
        if last_values is None:
            raise SearchBudgetExceeded("depth 1 cannot complete within the leaf budget")
        selected_move = min(last_values, key=lambda move: (-last_values[move], move))
        return AlphaBetaResult(
            selected_move=selected_move,
            root_action_values=dict(sorted(last_values.items())),
            completed_depth=completed_depth,
            principal_variation=last_pv,
            leaf_evaluator_calls=self.leaf_calls,
            nodes_visited=self.nodes,
            terminal_nodes=self.terminals,
            alpha_beta_cutoffs=self.cutoffs,
            transposition_table_hits=self.tt_hits,
            transposition_table_cutoffs=self.tt_cutoffs,
            budget_utilization=self.leaf_calls / self.leaf_evaluation_budget,
            deeper_iteration_abandoned=abandoned,
            root_solved_exactly=root_solved,
            node_cap_reached=self.node_cap_reached,
            runtime_seconds=time.perf_counter() - started,
            evaluator_lane=self.evaluator.lane,
            legal_moves=legal_moves,
            move_ordering=self._ordered_moves(root_game),
        )

    def _search_root(
        self, game: KalahGame, depth: int
    ) -> tuple[dict[int, float], list[int], bool]:
        values: dict[int, float] = {}
        solved = True
        action_pvs: dict[int, list[int]] = {}
        for move in self._ordered_moves(game):
            child = game.clone()
            child.move(child.pit_index(move))
            value, child_pv, child_solved = self._minimax(
                child, depth - 1, -float("inf"), float("inf")
            )
            values[move] = value
            action_pvs[move] = child_pv
            solved = solved and child_solved
        best_move = min(values, key=lambda candidate: (-values[candidate], candidate))
        return values, [best_move, *action_pvs[best_move]], solved

    def _minimax(
        self, game: KalahGame, depth: int, alpha: float, beta: float
    ) -> tuple[float, list[int], bool]:
        self.nodes += 1
        if self.nodes > self.node_cap:
            self.node_cap_reached = True
            raise _NodeCapExceeded("defensive node cap reached")
        if game.over():
            self.terminals += 1
            return self._terminal_value(game), [], True
        if depth <= 0:
            if self.leaf_calls >= self.leaf_evaluation_budget:
                raise SearchBudgetExceeded("leaf evaluation budget exhausted")
            self.leaf_calls += 1
            value = float(self.evaluator.evaluate_value(game))
            return (
                (value if game.current_player == self.root_player else -value),
                [],
                False,
            )
        key = canonical_state_key(game)
        entry = self.table.get(key) if self.use_transposition_table else None
        alpha_original, beta_original = alpha, beta
        if entry is not None:
            self.tt_hits += 1
            if entry.depth >= depth:
                if entry.bound == "exact":
                    self.tt_cutoffs += 1
                    return entry.value, self._pv_from_entry(game, depth), entry.solved
                if entry.bound == "lower":
                    alpha = max(alpha, entry.value)
                else:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    self.tt_cutoffs += 1
                    return entry.value, self._pv_from_entry(game, depth), False
        maximizing = game.current_player == self.root_player
        best_value = -float("inf") if maximizing else float("inf")
        best_move: int | None = None
        best_pv: list[int] = []
        all_solved = True
        for move in self._ordered_moves(
            game, entry.best_move if entry is not None else None
        ):
            child = game.clone()
            child.move(child.pit_index(move))
            value, child_pv, child_solved = self._minimax(child, depth - 1, alpha, beta)
            all_solved = all_solved and child_solved
            if (
                maximizing
                and (
                    value > best_value
                    or (value == best_value and (best_move is None or move < best_move))
                )
            ) or (
                not maximizing
                and (
                    value < best_value
                    or (value == best_value and (best_move is None or move < best_move))
                )
            ):
                best_value, best_move, best_pv = value, move, child_pv
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                self.cutoffs += 1
                all_solved = False
                break
        bound: BoundType = "exact"
        if best_value <= alpha_original:
            bound = "upper"
        elif best_value >= beta_original:
            bound = "lower"
        self._store(key, depth, best_value, bound, best_move, all_solved)
        return (
            best_value,
            ([] if best_move is None else [best_move, *best_pv]),
            all_solved,
        )

    def _ordered_moves(
        self, game: KalahGame, tt_best_move: int | None = None
    ) -> list[int]:
        state = game.to_state()
        consequences = {
            move: move_consequence_for_state(state, move)
            for move in game.possible_moves()
        }
        ordered = sorted(
            game.possible_moves(),
            key=lambda move: (
                not consequences[move]["game_over_after_move"],
                not consequences[move]["gives_extra_turn"],
                -consequences[move]["capture_count"],
                -consequences[move]["store_delta_immediate"],
                move,
            ),
        )
        if tt_best_move in ordered:
            ordered.remove(tt_best_move)
            ordered.insert(0, tt_best_move)
        return ordered

    def _store(
        self,
        key,
        depth: int,
        value: float,
        bound: BoundType,
        best_move: int | None,
        solved: bool,
    ) -> None:
        if not self.use_transposition_table:
            return
        existing = self.table.get(key)
        if existing is None or depth >= existing.depth:
            self.table[key] = TranspositionEntry(
                depth, float(value), bound, best_move, solved
            )

    def _pv_from_entry(self, game: KalahGame, depth: int) -> list[int]:
        pv: list[int] = []
        current = game.clone()
        for _ in range(depth):
            entry = self.table.get(canonical_state_key(current))
            if (
                entry is None
                or entry.best_move is None
                or entry.best_move not in current.possible_moves()
            ):
                break
            pv.append(entry.best_move)
            current.move(current.pit_index(entry.best_move))
            if current.over():
                break
        return pv

    def _terminal_value(self, game: KalahGame) -> float:
        if game.winner is None:
            return 0.0
        return 1.0 if game.winner == self.root_player else -1.0


def run_alpha_beta_search(
    game: KalahGame, evaluator: ScalarEvaluator, **kwargs
) -> AlphaBetaResult:
    return AlphaBetaSearch(evaluator, **kwargs).search(game)
