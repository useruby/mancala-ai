import numpy as np
import random

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.memoized_evaluator import MemoizedEvaluator
from ml.alphazero_lite.self_play import PUCT


class CountingEvaluator:
    input_encoding = "test-v1"

    def __init__(self):
        self.calls = 0

    def evaluate(self, game):
        self.calls += 1
        return np.arange(6, dtype=np.float32), float(game.current_player)


def state(player=0, store=0):
    return KalahGame.from_state(
        {
            "player_pits": [1, 0, 0, 0, 0, 0],
            "opponent_pits": [1, 0, 0, 0, 0, 0],
            "player_store": store,
            "opponent_store": 0,
            "current_player": player,
        }
    )


def test_copies_keys_capacity_and_reset():
    model = CountingEvaluator()
    wrapped = MemoizedEvaluator(
        model, artifact_identity="sha256:artifact-a", input_encoding="test-v1"
    )
    policy, _ = wrapped.evaluate(state())
    policy[:] = -1
    cached, _ = wrapped.evaluate(state())
    assert np.array_equal(cached, np.arange(6, dtype=np.float32))
    wrapped.evaluate(state(store=1))
    wrapped.evaluate(state(player=1))
    assert model.calls == 3
    assert wrapped.cache_stats["hits"] == 1
    assert wrapped.cache.max_entries == 4096
    wrapped.reset_telemetry()
    assert wrapped.cache_stats == {
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "neural_calls": 0,
        "requests": 0,
        "peak_entries": 0,
    }
    wrapped.evaluate(state())
    assert model.calls == 4


def test_eviction_and_artifact_encoding_separation():
    model = CountingEvaluator()
    first = MemoizedEvaluator(
        model, artifact_identity="artifact-a", input_encoding="test-v1"
    )
    second = MemoizedEvaluator(
        model, artifact_identity="artifact-b", input_encoding="test-v1"
    )
    for store in range(4097):
        first.evaluate(state(store=store))
    assert first.cache_stats["evictions"] == 1
    second.evaluate(state())
    assert model.calls == 4098


def test_puct_outputs_match_and_each_search_resets_cache():
    class DeterministicEvaluator(CountingEvaluator):
        def __init__(self):
            super().__init__()
            self.telemetry_resets = 0

        def reset_telemetry(self):
            self.telemetry_resets += 1
            self.calls = 0

        def evaluate(self, game):
            self.calls += 1
            priors = np.zeros(6, dtype=np.float32)
            for move in game.possible_moves():
                priors[move] = 1 / len(game.possible_moves())
            return priors, 0.125

    baseline_model = DeterministicEvaluator()
    cached_model = DeterministicEvaluator()
    cached = MemoizedEvaluator(
        cached_model, artifact_identity="artifact-sha256", input_encoding="test-v1"
    )
    root = KalahGame.from_state(
        {
            "current_player": 1,
            "opponent_pits": [2, 0, 0, 1, 8, 8],
            "opponent_store": 3,
            "player_pits": [0, 7, 0, 8, 0, 0],
            "player_store": 11,
        }
    )
    policies = []
    roots = []
    for evaluator in (baseline_model, cached):
        policy, tree = PUCT(
            evaluator,
            simulations=384,
            c_puct=1.25,
            rng=random.Random(420),
            fpu_mode="zero",
            reuse_subtree=False,
            normalize_values=False,
            tactical_root_bias=0,
            root_temperature=0,
        ).run(root.clone())
        policies.append(policy)
        roots.append(tree)
    off, on = roots
    assert off.visit_count == on.visit_count
    assert off.q_value == on.q_value
    assert np.array_equal(policies[0], policies[1])
    assert {m: n.visit_count for m, n in off.children.items()} == {
        m: n.visit_count for m, n in on.children.items()
    }
    assert {m: n.q_value for m, n in off.children.items()} == {
        m: n.q_value for m, n in on.children.items()
    }
    assert max(off.children, key=lambda m: off.children[m].visit_count) == max(
        on.children, key=lambda m: on.children[m].visit_count
    )
    assert cached.cache_stats["hits"] > 0
    assert cached.cache.size == cached.cache_stats["misses"]
    PUCT(
        cached,
        simulations=16,
        c_puct=1.25,
        rng=random.Random(420),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
    ).run(root.clone())
    assert cached.cache_stats["hits"] == 0
    assert cached.cache_stats["misses"] > 0
    assert cached.cache_stats["requests"] == (
        cached.cache_stats["hits"] + cached.cache_stats["misses"]
    )
    assert cached_model.telemetry_resets == 2
    assert cached_model.calls == cached.cache_stats["neural_calls"]


def test_extra_turn_and_terminal_outputs():
    extra_turn = KalahGame.from_state(
        {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [1, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    model = CountingEvaluator()
    cached = MemoizedEvaluator(
        model, artifact_identity="artifact-sha256", input_encoding="test-v1"
    )
    original = extra_turn.clone()
    assert extra_turn.move(5)
    assert extra_turn.current_player == 0
    cached.evaluate(original)
    cached.evaluate(extra_turn)
    assert model.calls == 2

    terminal = KalahGame.from_state(
        {
            "player_pits": [0] * 6,
            "opponent_pits": [0] * 6,
            "player_store": 24,
            "opponent_store": 24,
            "current_player": 0,
        }
    )
    terminal_model = CountingEvaluator()
    result = MemoizedEvaluator(
        terminal_model,
        artifact_identity="artifact-sha256",
        input_encoding="test-v1",
    )
    result.evaluate(terminal)
    result.evaluate(terminal)
    assert terminal_model.calls == 1
    assert result.cache_stats["hits"] == 1

    terminal_search_model = CountingEvaluator()
    terminal_baseline_model = CountingEvaluator()
    terminal_search = MemoizedEvaluator(
        terminal_search_model,
        artifact_identity="artifact-sha256",
        input_encoding="test-v1",
    )
    visits, root = PUCT(
        terminal_search,
        simulations=8,
        c_puct=1.25,
        rng=random.Random(7),
        fpu_mode="zero",
        root_fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
    ).run(terminal)
    baseline_visits, baseline_root = PUCT(
        terminal_baseline_model,
        simulations=8,
        c_puct=1.25,
        rng=random.Random(7),
        fpu_mode="zero",
        root_fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
    ).run(terminal)
    assert np.array_equal(visits, baseline_visits)
    assert root.visit_count == baseline_root.visit_count
    assert root.q_value == baseline_root.q_value
    assert visits.sum() == 0
    assert root.children == {}
    assert terminal_search_model.calls == 1


def test_puct_with_extra_turn_root_preserves_output():
    root = KalahGame.from_state(
        {
            "player_pits": [0, 0, 0, 0, 0, 1],
            "opponent_pits": [1, 0, 0, 0, 0, 0],
            "player_store": 0,
            "opponent_store": 0,
            "current_player": 0,
        }
    )
    off_eval = CountingEvaluator()
    on_eval = MemoizedEvaluator(
        CountingEvaluator(),
        artifact_identity="artifact-sha256",
        input_encoding="test-v1",
    )
    off = PUCT(
        off_eval,
        12,
        1.25,
        random.Random(17),
        fpu_mode="zero",
        root_fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
    )
    on = PUCT(
        on_eval,
        12,
        1.25,
        random.Random(17),
        fpu_mode="zero",
        root_fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
    )
    off_visits, off_root = off.run(root.clone())
    on_visits, on_root = on.run(root.clone())
    assert np.array_equal(off_visits, on_visits)
    assert off.root_summary()["selected_move"] == on.root_summary()["selected_move"]
    assert off_root.game.current_player == on_root.game.current_player
    assert off_root.children[5].game.current_player == 0
    assert on_root.children[5].game.current_player == 0
