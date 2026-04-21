#!/usr/bin/env python3

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ml.alphazero_lite.endgame_tablebase import EndgameTablebaseContract
from ml.alphazero_lite.kalah_rules import KalahGame, NUMBER_OF_PLAYERS, PITS_PER_PLAYER


@dataclass
class Node:
    game: KalahGame
    parent: "Node | None" = None
    children: dict[int, "Node"] = field(default_factory=dict)
    visits: int = 0
    wins: float = 0.0

    def expand(self) -> None:
        if self.children:
            return

        for action in self.game.possible_moves():
            next_game = self.game.clone()
            next_game.move(next_game.pit_index(action))
            self.children[action] = Node(next_game, self)

    def select_child(self, c: float = math.sqrt(2.0)) -> "Node":
        if not self.children:
            return self

        parent_visits = self.visits if self.visits > 0 else 1
        log_parent_visits = math.log(parent_visits)
        best_child: Node | None = None
        best_score = -float("inf")

        for child in self.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.wins / float(child.visits)
                exploration = c * math.sqrt(log_parent_visits / child.visits)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        assert best_child is not None
        return best_child

    def backpropagate(self, wins: float) -> None:
        self.visits += 1
        self.wins += wins
        if self.parent is not None:
            self.parent.backpropagate(wins)


class MCTS:
    DEFAULT_SIMULATIONS = 5000
    DEPTH_OF_SIMULATION = 5
    EARLY_GAME_PLAYOUT_DEPTH = 6
    MID_GAME_PLAYOUT_DEPTH = 10
    LATE_GAME_PLAYOUT_DEPTH = 14
    EXTRA_TURN_BONUS = 3.0
    CAPTURE_WEIGHT = 10.0

    def __init__(
        self,
        game: KalahGame,
        *,
        simulations: int = DEFAULT_SIMULATIONS,
        seed: int | None = None,
        legacy_playout: bool = False,
        early_stop_enabled: bool = False,
        early_stop_min_simulations: int = 1200,
        early_stop_check_interval: int = 200,
        early_stop_top_visit_share: float = 0.85,
        early_stop_required_checks: int = 2,
        endgame_tablebase: EndgameTablebaseContract | None = None,
    ):
        self.game = game
        self.simulations = simulations
        self.player = game.current_player
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.legacy_playout = legacy_playout
        self.early_stop_enabled = early_stop_enabled
        self.early_stop_min_simulations = max(int(early_stop_min_simulations), 1)
        self.early_stop_check_interval = max(int(early_stop_check_interval), 1)
        self.early_stop_top_visit_share = float(early_stop_top_visit_share)
        self.early_stop_required_checks = max(int(early_stop_required_checks), 1)
        self.endgame_tablebase = endgame_tablebase
        self._cached_root: Node | None = None
        self._cached_root_state: dict | None = None

    def choose_move(self) -> int | None:
        possible_moves = self.game.possible_moves()
        if not possible_moves:
            return None

        root = self.search_root()
        return self._choose_move_from_root(root)

    def root_summary(self) -> dict:
        root = self.search_root()
        child_stats = []
        for action, child in sorted(root.children.items()):
            child_stats.append(
                {
                    "move": int(action),
                    "visits": int(child.visits),
                    "win_rate": float(child.wins / child.visits) if child.visits else 0.0,
                }
            )
        selected_move = self._choose_move_from_root(root)
        return {
            "selected_move": None if selected_move is None else int(selected_move),
            "child_stats": child_stats,
        }

    def choose_playout_move(self, game: KalahGame) -> int | None:
        possible_actions = game.possible_moves()
        if not possible_actions:
            return None
        if self.legacy_playout:
            return possible_actions[self.rng.randrange(len(possible_actions))]

        player = game.current_player
        before_store = game.captured_seeds[player]
        current_total, opponent_total = self.kalah_side_seed_totals(game, player)
        best_actions: list[int] = []
        best_score = -float("inf")

        for action in possible_actions:
            pit_index = game.pit_index(action)
            score = self.kalah_playout_move_score(game, pit_index, player, before_store, current_total, opponent_total)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return best_actions[self.rng.randrange(len(best_actions))]

    def search_root(self) -> Node:
        current_state = self.game.to_state()
        if self._cached_root is not None and self._cached_root_state == current_state:
            return self._cached_root

        self.player = self.game.current_player
        node = Node(self.game.clone())
        node.expand()
        if len(node.children) > 1:
            self.run_search(node, self.simulations, allow_early_stop=True)
        self._cached_root = node
        self._cached_root_state = current_state
        return node

    def _choose_move_from_root(self, root: Node) -> int | None:
        if not root.children:
            return None
        if len(root.children) == 1:
            return next(iter(root.children.keys()))

        best_action = None
        best_value = -float("inf")
        for action, child in root.children.items():
            value = 0.0 if child.wins == 0 else child.wins / float(child.visits)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def run_search(self, node: Node, simulations_to_run: int, *, allow_early_stop: bool) -> int:
        decisive_checks = 0
        simulations_run = 0

        for simulation_index in range(simulations_to_run):
            selected = node.select_child()
            while True:
                if selected.visits == 0:
                    break
                selected.expand()
                if not selected.children:
                    break
                selected = selected.select_child()

            wins = self.simulate_playout(selected.game)
            selected.backpropagate(wins)
            simulations_run = simulation_index + 1
            returned_visit_count = sum(child.visits for child in node.children.values())

            if not allow_early_stop or not self.early_stop_check(returned_visit_count):
                continue
            if self.decisive_root_policy(node):
                decisive_checks += 1
                if decisive_checks >= self.early_stop_required_checks:
                    break
            else:
                decisive_checks = 0

        return simulations_run

    def early_stop_check(self, simulations_run: int) -> bool:
        if not self.early_stop_enabled:
            return False
        if simulations_run < self.early_stop_min_simulations:
            return False
        return (simulations_run % self.early_stop_check_interval) == 0

    def decisive_root_policy(self, node: Node) -> bool:
        total = sum(child.visits for child in node.children.values())
        if total == 0:
            return False
        top = max(child.visits for child in node.children.values())
        return (top / float(total)) >= self.early_stop_top_visit_share

    def simulate_playout(self, game: KalahGame) -> float:
        if self.endgame_tablebase is not None:
            solved_value = self.endgame_tablebase.lookup(game, self.player)
            if solved_value is not None:
                return solved_value

        current_game = game.clone()
        depth = self.playout_depth_for(current_game)
        for _ in range(depth):
            if current_game.over():
                break
            action = self.choose_playout_move(current_game)
            if action is None:
                break
            current_game.move(current_game.pit_index(action))
        return self.rank(current_game)

    def rank(self, game: KalahGame) -> float:
        player_score = game.captured_seeds[self.player]
        opponent_score = game.captured_seeds[1 - self.player]
        if player_score > opponent_score:
            return 1.0
        if player_score < opponent_score:
            return 0.0
        return 0.5

    def playout_depth_for(self, game: KalahGame) -> int:
        if self.legacy_playout:
            return self.DEPTH_OF_SIMULATION

        seeds_remaining = sum(game.pits)
        if seeds_remaining <= 12:
            return self.LATE_GAME_PLAYOUT_DEPTH
        if seeds_remaining <= 24:
            return self.MID_GAME_PLAYOUT_DEPTH
        return self.EARLY_GAME_PLAYOUT_DEPTH

    def kalah_playout_move_score(
        self,
        game: KalahGame,
        pit_index: int,
        player: int,
        before_store: int,
        current_total: int | None = None,
        opponent_total: int | None = None,
    ) -> float:
        if current_total is None or opponent_total is None:
            current_total, opponent_total = self.kalah_side_seed_totals(game, player)

        seeds = game.pits[pit_index]
        if seeds == 0:
            return -float("inf")

        relative_index = pit_index - (player * PITS_PER_PLAYER)
        distance_to_store = PITS_PER_PLAYER - relative_index
        cycle_length = (PITS_PER_PLAYER * NUMBER_OF_PLAYERS) + 1
        laps, remainder = divmod(seeds, cycle_length)
        extra_turn = remainder == distance_to_store
        capture_gain = laps
        if remainder >= distance_to_store and remainder > 0:
            capture_gain += 1

        current_total_after = current_total - seeds + self.kalah_current_side_hits(laps, remainder, distance_to_store)
        opponent_total_after = opponent_total + self.kalah_opponent_side_hits(laps, remainder, distance_to_store)

        if not extra_turn:
            landing_index = self.kalah_landing_index(player, pit_index, remainder, distance_to_store)
            if landing_index is not None and game.pit_owner(landing_index) == player:
                landing_seeds = self.kalah_pit_seeds_after_sowing(game, player, pit_index, landing_index, seeds)
                opposite_index = game.opposite_pit_index(landing_index)
                opposite_seeds = self.kalah_pit_seeds_after_sowing(game, player, pit_index, opposite_index, seeds)
                if landing_seeds == 1 and opposite_seeds > 0:
                    capture_gain += landing_seeds + opposite_seeds
                    current_total_after -= landing_seeds
                    opponent_total_after -= opposite_seeds

        if self.kalah_game_over_after_move(player, extra_turn, current_total_after, opponent_total_after):
            if not extra_turn:
                capture_gain += current_total_after

        return (capture_gain * self.CAPTURE_WEIGHT) + (self.EXTRA_TURN_BONUS if extra_turn else 0.0)

    def kalah_side_seed_totals(self, game: KalahGame, player: int) -> tuple[int, int]:
        current_start = player * PITS_PER_PLAYER
        opponent_start = (1 - player) * PITS_PER_PLAYER
        current_total = sum(game.pits[current_start + offset] for offset in range(PITS_PER_PLAYER))
        opponent_total = sum(game.pits[opponent_start + offset] for offset in range(PITS_PER_PLAYER))
        return current_total, opponent_total

    def kalah_current_side_hits(self, laps: int, remainder: int, distance_to_store: int) -> int:
        hits = laps * PITS_PER_PLAYER
        hits += min(remainder, distance_to_store - 1)
        wrapped_hits = remainder - distance_to_store - PITS_PER_PLAYER
        return hits + max(wrapped_hits, 0)

    def kalah_opponent_side_hits(self, laps: int, remainder: int, distance_to_store: int) -> int:
        hits = laps * PITS_PER_PLAYER
        tail_hits = remainder - distance_to_store
        return hits + min(max(tail_hits, 0), PITS_PER_PLAYER)

    def kalah_landing_index(self, player: int, pit_index: int, remainder: int, distance_to_store: int) -> int | None:
        if remainder == 0:
            return pit_index
        if remainder < distance_to_store:
            return pit_index + remainder
        if remainder == distance_to_store:
            return None

        opponent_start = (1 - player) * PITS_PER_PLAYER
        if remainder <= distance_to_store + PITS_PER_PLAYER:
            return opponent_start + remainder - distance_to_store - 1
        return (player * PITS_PER_PLAYER) + remainder - distance_to_store - PITS_PER_PLAYER - 1

    def kalah_pit_seeds_after_sowing(self, game: KalahGame, player: int, pit_index: int, target_index: int, seeds: int) -> int:
        original_seeds = 0 if target_index == pit_index else game.pits[target_index]
        return original_seeds + self.kalah_pit_hits(seeds, self.kalah_pit_distance(player, pit_index, target_index))

    def kalah_pit_hits(self, seeds: int, distance: int) -> int:
        if seeds < distance:
            return 0
        return 1 + ((seeds - distance) // ((PITS_PER_PLAYER * NUMBER_OF_PLAYERS) + 1))

    def kalah_pit_distance(self, player: int, pit_index: int, target_index: int) -> int:
        current_start = player * PITS_PER_PLAYER
        current_end = current_start + PITS_PER_PLAYER
        opponent_start = (1 - player) * PITS_PER_PLAYER
        distance_to_store = PITS_PER_PLAYER - (pit_index - current_start)

        if current_start <= target_index < current_end and target_index > pit_index:
            return target_index - pit_index
        if opponent_start <= target_index < opponent_start + PITS_PER_PLAYER:
            return distance_to_store + target_index - opponent_start + 1
        return distance_to_store + PITS_PER_PLAYER + target_index - current_start + 1

    def kalah_game_over_after_move(self, player: int, extra_turn: bool, current_total_after: int, opponent_total_after: int) -> bool:
        next_player = player if extra_turn else 1 - player
        return current_total_after == 0 if next_player == player else opponent_total_after == 0
