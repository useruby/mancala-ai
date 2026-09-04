"""Diagnostic-only turn-completion PUCT built on the exact-budget adapter."""

from __future__ import annotations

from dataclasses import dataclass

from ml.alphazero_lite.gumbel_root_search import (
    EVALUATION_BUDGET,
    CountingEvaluator,
    RootSearchResult,
    _ExactBudgetPUCT,
)
from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, Node, terminal_value

MAX_EXTENSION_ACTIONS = 48


@dataclass(frozen=True)
class TurnCompletionResult(RootSearchResult):
    """Exact-budget root result with turn-extension audit telemetry."""

    completed_simulations: int
    extensions_started: int
    extensions_completed: int
    incomplete_due_to_budget: int
    extension_neural_calls: int
    extra_turn_actions: int
    maximum_extension_length: int
    turn_boundary_player_changes: int
    terminal_completions: int
    repeated_state_events: int
    defensive_cap_events: int
    discarded_intermediate_values: tuple[float, ...]
    final_backed_up_values: tuple[float, ...]


class TurnCompletionPUCT(_ExactBudgetPUCT):
    """Complete a genuine extra-turn sequence before backing up its value."""

    def __init__(
        self,
        evaluator: CountingEvaluator,
        *,
        seed: int,
        budget: int,
        enabled: bool = True,
    ) -> None:
        super().__init__(evaluator, seed=seed, budget=budget)
        self.enabled = enabled
        self.extensions_started = 0
        self.extensions_completed = 0
        self.incomplete_due_to_budget = 0
        self.extension_neural_calls = 0
        self.extra_turn_actions = 0
        self.maximum_extension_length = 0
        self.turn_boundary_player_changes = 0
        self.terminal_completions = 0
        self.repeated_state_events = 0
        self.defensive_cap_events = 0
        self.discarded_intermediate_values: list[float] = []
        self.final_backed_up_values: list[float] = []
        self._incomplete_simulation = False

    def _expand_for_extension(self, node: Node) -> float | None:
        if self.evaluator.calls >= self.budget:
            self._incomplete_simulation = True
            return None
        before = self.evaluator.calls
        _, value = self._expand(
            node,
            apply_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_epsilon=0.0,
            is_root=False,
        )
        self.extension_neural_calls += self.evaluator.calls - before
        return float(value)

    def _complete_extension(self, node: Node, player: int) -> float | None:
        """Return a completed value from ``node`` in its own-player perspective."""
        self.extensions_started += 1
        actions = 0
        seen = set()
        path: list[tuple[Node, Node]] = []
        current = node
        while True:
            key = self._state_hash(current.game)
            if key in seen:
                self.repeated_state_events += 1
                raise RuntimeError("turn-completion repeated state")
            seen.add(key)
            terminal = terminal_value(current.game)
            if terminal is not None:
                self.terminal_completions += 1
                value = float(terminal)
                break
            if current.game.current_player != player:
                self.turn_boundary_player_changes += 1
                if not current.expanded:
                    value = self._expand_for_extension(current)
                    if value is None:
                        return None
                else:
                    # A previously expanded boundary continues with ordinary PUCT.
                    value = super()._search(current)
                break
            if not current.expanded:
                intermediate = self._expand_for_extension(current)
                if intermediate is None:
                    return None
                self.discarded_intermediate_values.append(intermediate)
            if not current.children:
                value = 0.0
                break
            if actions >= MAX_EXTENSION_ACTIONS:
                self.defensive_cap_events += 1
                raise RuntimeError("turn-completion defensive action cap")
            child = self._select_child(current)
            path.append((current, child))
            actions += 1
            self.extra_turn_actions += 1
            current = child
        self.maximum_extension_length = max(self.maximum_extension_length, actions)
        # The completed value is in the boundary node's perspective. Convert it
        # edge by edge using player identity, never tree depth parity.
        for parent, child in reversed(path):
            if child.game.current_player != parent.game.current_player:
                value = -value
            child.visit_count += 1
            child.value_sum += value
        self.extensions_completed += 1
        self.final_backed_up_values.append(float(value))
        return float(value)

    def _search(
        self,
        node: Node,
        depth: int = 0,
        trace_record=None,
        selected_edges=None,
    ) -> float:
        if not self.enabled:
            return super()._search(node, depth, trace_record, selected_edges)
        terminal = terminal_value(node.game)
        if terminal is not None:
            return float(terminal)
        if not node.expanded:
            _, value = self._expand(
                node,
                apply_dirichlet=False,
                dirichlet_alpha=None,
                dirichlet_epsilon=0.0,
                is_root=False,
                depth=depth,
            )
            return float(value)
        child = self._select_child(node)
        if (
            not child.expanded
            and not child.game.over()
            and child.game.current_player == node.game.current_player
        ):
            value = self._complete_extension(child, node.game.current_player)
            if value is None:
                return 0.0
        else:
            value = self._search(child, depth + 1, trace_record, selected_edges)
        if self._incomplete_simulation:
            return 0.0
        if child.game.current_player != node.game.current_player:
            value = -value
        child.visit_count += 1
        child.value_sum += value
        return float(value)

    def run_exact_budget(self, root_game: KalahGame) -> TurnCompletionResult:
        # The disabled path deliberately delegates to the existing implementation.
        if not self.enabled:
            ordinary = super().run_exact_budget(root_game)
            return TurnCompletionResult(
                **ordinary.__dict__,
                completed_simulations=sum(ordinary.visits.values()),
                extensions_started=0,
                extensions_completed=0,
                incomplete_due_to_budget=0,
                extension_neural_calls=0,
                extra_turn_actions=0,
                maximum_extension_length=0,
                turn_boundary_player_changes=0,
                terminal_completions=0,
                repeated_state_events=0,
                defensive_cap_events=0,
                discarded_intermediate_values=(),
                final_backed_up_values=(),
            )
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
        padding_calls = stalled = iterations = completed = 0
        while self.evaluator.calls < self.budget:
            iterations += 1
            if iterations > self.budget * 128:
                raise RuntimeError("unable to consume exact neural-evaluation budget")
            self._active_simulation_index = iterations
            self._incomplete_simulation = False
            before = self.evaluator.calls
            value = self._search(root)
            if not self._incomplete_simulation:
                root.visit_count += 1
                root.value_sum += value
                completed += 1
            else:
                self.incomplete_due_to_budget += 1
            stalled = stalled + 1 if self.evaluator.calls == before else 0
            if stalled >= max(16, len(legal_moves) * 4):
                self.evaluator.evaluate(root.game)
                padding_calls += 1
                stalled = 0
        self._active_simulation_index = None
        visits = {move: int(child.visit_count) for move, child in root.children.items()}
        result = TurnCompletionResult(
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
            completed_simulations=completed,
            extensions_started=self.extensions_started,
            extensions_completed=self.extensions_completed,
            incomplete_due_to_budget=self.incomplete_due_to_budget,
            extension_neural_calls=self.extension_neural_calls,
            extra_turn_actions=self.extra_turn_actions,
            maximum_extension_length=self.maximum_extension_length,
            turn_boundary_player_changes=self.turn_boundary_player_changes,
            terminal_completions=self.terminal_completions,
            repeated_state_events=self.repeated_state_events,
            defensive_cap_events=self.defensive_cap_events,
            discarded_intermediate_values=tuple(self.discarded_intermediate_values),
            final_backed_up_values=tuple(self.final_backed_up_values),
        )
        if result.evaluator_calls != self.budget:
            raise RuntimeError("turn-completion PUCT evaluation budget contract failed")
        return result


def run_turn_completion_puct(
    game: KalahGame,
    evaluator: Evaluator,
    *,
    seed: int,
    budget: int = EVALUATION_BUDGET,
    enabled: bool = True,
) -> TurnCompletionResult:
    """Run the isolated turn-completion diagnostic at an exact neural budget."""
    return TurnCompletionPUCT(
        CountingEvaluator(evaluator), seed=seed, budget=budget, enabled=enabled
    ).run_exact_budget(game)
