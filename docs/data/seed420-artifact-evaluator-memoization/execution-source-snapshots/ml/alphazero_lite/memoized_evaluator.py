"""Opt-in, per-root memoization for deterministic policy/value evaluators."""

from __future__ import annotations

import json
import hashlib

import numpy as np

from ml.alphazero_lite.eval_cache import EvalCache


class MemoizedEvaluator:
    """Cache only neural outputs; PUCT tree statistics remain search-local.

    ``artifact_identity`` must be an immutable artifact digest, not a path or
    modification-time identity. PUCT calls ``reset_telemetry`` at each search.
    """

    CAPACITY = 4096

    def __init__(self, evaluator, *, artifact_identity: str, input_encoding: str):
        if not artifact_identity or artifact_identity == "unavailable":
            raise ValueError("immutable_artifact_identity_required")
        if not input_encoding:
            raise ValueError("input_encoding_required")
        self.evaluator = evaluator
        self.artifact_identity = artifact_identity
        self.input_encoding = input_encoding
        self.cache = EvalCache(self.CAPACITY)
        self.requests = 0
        self.neural_calls = 0
        self.peak_entries = 0
        self.evictions = 0
        self.request_trace: list[dict] | None = None

    @staticmethod
    def _key_state(game) -> str:
        if not hasattr(game, "to_state"):
            raise TypeError("memoized_evaluation_requires_canonical_state")
        return json.dumps(
            game.to_state(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def evaluate(self, game):
        self.requests += 1
        canonical_state = self._key_state(game)
        key = (self.artifact_identity, self.input_encoding, canonical_state)
        result = self.cache.get(key)
        if result is not None:
            policy, value = result
            if self.request_trace is not None:
                self.request_trace.append(
                    self._trace_record(canonical_state, policy, value, True)
                )
            return policy.copy(), value
        policy, value = self.evaluator.evaluate(game)
        stored_policy = np.asarray(policy).copy()
        value = float(value)
        self.cache.put(key, (stored_policy, value))
        self.neural_calls += 1
        if self.cache.size == self.CAPACITY and self.neural_calls > self.CAPACITY:
            self.evictions += 1
        self.peak_entries = max(self.peak_entries, self.cache.size)
        if self.request_trace is not None:
            self.request_trace.append(
                self._trace_record(canonical_state, stored_policy, value, False)
            )
        return stored_policy.copy(), value

    @staticmethod
    def _trace_record(canonical_state: str, policy, value: float, hit: bool) -> dict:
        return {
            "state_sha256": hashlib.sha256(canonical_state.encode()).hexdigest(),
            "policy": np.asarray(policy).tolist(),
            "value": float(value),
            "cache_hit": hit,
        }

    def reset_telemetry(self) -> None:
        self.cache = EvalCache(self.CAPACITY)
        self.requests = 0
        self.neural_calls = 0
        self.peak_entries = 0
        self.evictions = 0
        if self.request_trace is not None:
            self.request_trace.clear()
        reset = getattr(self.evaluator, "reset_telemetry", None)
        if callable(reset):
            reset()

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "evictions": self.evictions,
            "neural_calls": self.neural_calls,
            "requests": self.requests,
            "peak_entries": self.peak_entries,
        }
