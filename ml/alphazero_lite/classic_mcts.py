#!/usr/bin/env python3

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field

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

    def select_child(self, c: float = math.sqrt(2.0), value_trust_multiplier: float = 1.0) -> "Node":
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
                exploitation = (child.wins / float(child.visits)) * value_trust_multiplier
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


@dataclass
class DynamicBudgetConfig:
    enabled: bool = False
    probe_simulations: int = 0
    min_simulations: int = 0
    max_simulations: int = 0
    entropy_weight: float = 0.8
    low_margin_threshold: float = 0.2
    low_margin_weight: float = 1.5
    variance_weight: float = 1.5


@dataclass
class RootBudgetSummary:
    dynamic_budget_enabled: bool
    baseline_simulations: int
    probe_simulations: int
    chosen_simulations: int | None
    final_simulations: int
    phase_bucket: str
    entropy: float
    top_move_margin: float
    child_value_variance: float
    trigger: str
    root_latency_ms: float = 0.0


@dataclass
class ValueTrustSchedule:
    enabled: bool = False
    opening: float = 1.0
    midgame: float = 1.0
    late: float = 1.0


def _validated_value_trust_schedule(raw_schedule: dict | None) -> ValueTrustSchedule:
    if raw_schedule is None:
        return ValueTrustSchedule()

    if not isinstance(raw_schedule, dict):
        raise ValueError("value_trust_schedule must be an object")

    allowed_keys = {"enabled", "opening", "midgame", "late"}
    if set(raw_schedule.keys()) - allowed_keys:
        raise ValueError("value_trust_schedule keys must be enabled, opening, midgame, and late")

    enabled = raw_schedule.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("value_trust_schedule enabled must be a boolean")
    schedule_values = {}
    for key in ("opening", "midgame", "late"):
        raw_value = raw_schedule.get(key, 1.0)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"value_trust_schedule {key} must be numeric")
        normalized = float(raw_value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"value_trust_schedule {key} must be > 0")
        schedule_values[key] = normalized

    return ValueTrustSchedule(enabled=enabled, **schedule_values)


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
        dynamic_budget_enabled: bool = False,
        dynamic_budget_probe_simulations: int = 0,
        dynamic_budget_min_simulations: int | None = None,
        dynamic_budget_max_simulations: int | None = None,
        dynamic_budget_entropy_weight: float = 0.8,
        dynamic_budget_low_margin_threshold: float = 0.2,
        dynamic_budget_low_margin_weight: float = 1.5,
        dynamic_budget_variance_weight: float = 1.5,
        value_trust_schedule: dict | None = None,
        endgame_tablebase: EndgameTablebaseContract | None = None,
        exact_solve_enabled: bool = False,
        exact_solve_stone_threshold: int | None = None,
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
        min_simulations = simulations if dynamic_budget_min_simulations is None else int(dynamic_budget_min_simulations)
        max_simulations = simulations if dynamic_budget_max_simulations is None else int(dynamic_budget_max_simulations)
        probe_simulations = int(dynamic_budget_probe_simulations)
        if dynamic_budget_enabled:
            if min_simulations < 1:
                raise ValueError("dynamic_budget_min_simulations must be >= 1")
            if max_simulations < min_simulations:
                raise ValueError("dynamic_budget_max_simulations must be >= dynamic_budget_min_simulations")
            if probe_simulations < 1:
                raise ValueError("dynamic_budget_probe_simulations must be >= 1 when dynamic budget is enabled")
            if probe_simulations >= max_simulations:
                raise ValueError("dynamic_budget_probe_simulations must be < dynamic_budget_max_simulations")
        self.dynamic_budget_config = DynamicBudgetConfig(
            enabled=bool(dynamic_budget_enabled),
            probe_simulations=probe_simulations,
            min_simulations=min_simulations,
            max_simulations=max_simulations,
            entropy_weight=float(dynamic_budget_entropy_weight),
            low_margin_threshold=float(dynamic_budget_low_margin_threshold),
            low_margin_weight=float(dynamic_budget_low_margin_weight),
            variance_weight=float(dynamic_budget_variance_weight),
        )
        self.value_trust_schedule = _validated_value_trust_schedule(value_trust_schedule)
        self._last_root_budget_summary = RootBudgetSummary(
            dynamic_budget_enabled=bool(dynamic_budget_enabled),
            baseline_simulations=int(simulations),
            probe_simulations=int(probe_simulations) if dynamic_budget_enabled else int(simulations),
            chosen_simulations=int(simulations),
            final_simulations=int(simulations),
            phase_bucket="fixed",
            entropy=0.0,
            top_move_margin=0.0,
            child_value_variance=0.0,
            trigger="fixed_budget",
        )
        self.endgame_tablebase = endgame_tablebase
        self.exact_solve_enabled = exact_solve_enabled
        self.exact_solve_stone_threshold = exact_solve_stone_threshold
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
            "budget": asdict(self._last_root_budget_summary),
            "value_trust": self._value_trust_summary_for(root.game),
        }

    def _value_trust_summary_for(self, game: KalahGame) -> dict:
        phase_key = self._value_trust_phase_key_for(game)
        return {
            "enabled": bool(self.value_trust_schedule.enabled),
            "phase_bucket": phase_key,
            "effective_multiplier": float(self._effective_value_trust_multiplier_for(game)),
            "schedule": {
                "opening": float(self.value_trust_schedule.opening),
                "midgame": float(self.value_trust_schedule.midgame),
                "late": float(self.value_trust_schedule.late),
            },
        }

    def _value_trust_phase_key_for(self, game: KalahGame) -> str:
        return {"early": "opening", "mid": "midgame", "late": "late"}[self.phase_bucket_for(game)]

    def _effective_value_trust_multiplier_for(self, game: KalahGame) -> float:
        if not self.value_trust_schedule.enabled:
            return 1.0
        return float(getattr(self.value_trust_schedule, self._value_trust_phase_key_for(game)))

    def _select_guided_child(self, node: Node) -> Node:
        return node.select_child(value_trust_multiplier=self._effective_value_trust_multiplier_for(node.game))

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
        root = Node(self.game.clone())
        root.expand()
        started = time.perf_counter()
        probe_simulations = (
            int(self.dynamic_budget_config.probe_simulations)
            if self.dynamic_budget_config.enabled
            else int(self.simulations)
        )
        simulations_run = 0
        if len(root.children) > 1:
            if self.dynamic_budget_config.enabled:
                probe_budget = self.dynamic_budget_config.probe_simulations
                probe_simulations_run = int(self.run_search(root, probe_budget, allow_early_stop=False))
                probe_phase_bucket = self.phase_bucket_for(root.game)
                probe_entropy = self.root_entropy(root)
                probe_top_move_margin = self.top_move_margin(root)
                probe_child_value_variance = self.child_value_variance(root)
                probe_trigger = self.dynamic_budget_trigger_label(root)
                final_budget = self.choose_dynamic_budget(root)
                remaining_budget = max(final_budget - probe_budget, 0)
                second_phase_simulations_run = 0
                if remaining_budget > 0:
                    second_phase_simulations_run = int(
                        self.run_search(root, remaining_budget, allow_early_stop=True)
                    )
                simulations_run = probe_simulations_run + second_phase_simulations_run
                self._last_root_budget_summary = RootBudgetSummary(
                    dynamic_budget_enabled=True,
                    baseline_simulations=int(self.simulations),
                    probe_simulations=probe_simulations,
                    chosen_simulations=int(final_budget),
                    final_simulations=int(simulations_run),
                    phase_bucket=probe_phase_bucket,
                    entropy=probe_entropy,
                    top_move_margin=probe_top_move_margin,
                    child_value_variance=probe_child_value_variance,
                    trigger=probe_trigger,
                    root_latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            else:
                simulations_run = int(self.run_search(root, self.simulations, allow_early_stop=True))
                self._last_root_budget_summary = RootBudgetSummary(
                    dynamic_budget_enabled=False,
                    baseline_simulations=int(self.simulations),
                    probe_simulations=probe_simulations,
                    chosen_simulations=int(self.simulations),
                    final_simulations=int(simulations_run),
                    phase_bucket="fixed",
                    entropy=0.0,
                    top_move_margin=0.0,
                    child_value_variance=0.0,
                    trigger="fixed_budget",
                    root_latency_ms=(time.perf_counter() - started) * 1000.0,
                )
        else:
            self._last_root_budget_summary = RootBudgetSummary(
                dynamic_budget_enabled=bool(self.dynamic_budget_config.enabled),
                baseline_simulations=int(self.simulations),
                probe_simulations=probe_simulations,
                chosen_simulations=(
                    None
                    if self.dynamic_budget_config.enabled
                    else int(self.simulations)
                ),
                final_simulations=0,
                phase_bucket=self.phase_bucket_for(root.game) if self.dynamic_budget_config.enabled else "fixed",
                entropy=self.root_entropy(root),
                top_move_margin=self.top_move_margin(root),
                child_value_variance=self.child_value_variance(root),
                trigger=(
                    self.dynamic_budget_trigger_label(root)
                    if self.dynamic_budget_config.enabled
                    else "fixed_budget"
                ),
                root_latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        self._cached_root = root
        self._cached_root_state = current_state
        return root

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
            selected = self._select_guided_child(node)
            while True:
                if selected.visits == 0:
                    break
                selected.expand()
                if not selected.children:
                    break
                selected = self._select_guided_child(selected)

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
            if self.exact_solve_applies(game):
                solved_value = self.endgame_tablebase.lookup(game, self.player)
            else:
                solved_value = self.endgame_tablebase.lookup_cached(game, self.player)
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

    def exact_solve_applies(self, game: KalahGame) -> bool:
        if not self.exact_solve_enabled:
            return False
        if self.exact_solve_stone_threshold is None:
            return False
        if game.over():
            return False
        if not game.possible_moves():
            return False
        return sum(game.pits) <= self.exact_solve_stone_threshold

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

    def phase_bucket_for(self, game: KalahGame) -> str:
        seeds_remaining = sum(game.pits)
        if seeds_remaining <= 12:
            return "late"
        if seeds_remaining <= 24:
            return "mid"
        return "early"

    def root_entropy(self, node: Node) -> float:
        visits = [child.visits for child in node.children.values()]
        total = float(sum(visits))
        if total <= 0.0:
            return 0.0
        probabilities = [visit / total for visit in visits]
        entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
        max_entropy = math.log(len(node.children)) if len(node.children) > 1 else 1.0
        return 0.0 if max_entropy <= 0 else entropy / max_entropy

    def top_move_margin(self, node: Node) -> float:
        win_rates = sorted(
            [(child.wins / float(child.visits)) for child in node.children.values() if child.visits > 0],
            reverse=True,
        )
        if len(node.children) > 1 and len(win_rates) < 2:
            return 0.0
        if len(win_rates) < 2:
            return 1.0
        return win_rates[0] - win_rates[1]

    def child_value_variance(self, node: Node) -> float:
        win_rates = [(child.wins / float(child.visits)) for child in node.children.values() if child.visits > 0]
        if len(node.children) > 1 and len(win_rates) < 2:
            return 0.25
        if len(win_rates) < 2:
            return 0.0
        mean = sum(win_rates) / float(len(win_rates))
        return sum((value - mean) ** 2 for value in win_rates) / float(len(win_rates))

    def choose_dynamic_budget(self, node: Node) -> int:
        entropy = self.root_entropy(node)
        margin = self.top_move_margin(node)
        variance = self.child_value_variance(node)
        phase_bucket = self.phase_bucket_for(node.game)

        multiplier = 1.0
        if phase_bucket == "early":
            multiplier -= 0.15
        elif phase_bucket == "late":
            multiplier += 0.15
        multiplier += (entropy - 0.5) * self.dynamic_budget_config.entropy_weight
        multiplier += max(0.0, self.dynamic_budget_config.low_margin_threshold - margin) * self.dynamic_budget_config.low_margin_weight
        multiplier += variance * self.dynamic_budget_config.variance_weight

        unclamped = round(self.simulations * multiplier)
        return max(
            self.dynamic_budget_config.probe_simulations,
            self.dynamic_budget_config.min_simulations,
            min(self.dynamic_budget_config.max_simulations, unclamped),
        )

    def dynamic_budget_trigger_label(self, node: Node) -> str:
        labels = [self.phase_bucket_for(node.game)]
        if self.root_entropy(node) >= 0.75:
            labels.append("high_entropy")
        if self.top_move_margin(node) <= self.dynamic_budget_config.low_margin_threshold:
            labels.append("low_margin")
        if self.child_value_variance(node) >= 0.02:
            labels.append("high_variance")
        if len(labels) == 1:
            labels.append("low_uncertainty")
        return "_".join(labels)

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
