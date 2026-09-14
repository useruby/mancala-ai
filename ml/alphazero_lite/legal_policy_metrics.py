"""Evaluation-only policy probability semantics for legal Mancala actions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

POLICY_SIZE = 6


def normalize_policy_over_legal_actions(
    policy: Sequence[float] | np.ndarray, legal_actions: Sequence[int]
) -> np.ndarray:
    """Return a finite legal-only policy without changing legal-action ranking."""
    values = np.asarray(policy, dtype=np.float64)
    if values.shape != (POLICY_SIZE,):
        raise ValueError(f"expected {POLICY_SIZE} policy entries, got {values.shape}")
    legal = tuple(sorted({int(action) for action in legal_actions}))
    if not legal or any(action < 0 or action >= POLICY_SIZE for action in legal):
        raise ValueError("legal_actions must be non-empty actions in [0, 5]")
    result = np.zeros(POLICY_SIZE, dtype=np.float64)
    legal_values = values[list(legal)]
    if not np.all(np.isfinite(legal_values)):
        raise ValueError("legal policy entries must be finite")
    # Evaluator softmax values are non-negative. Clip only round-off noise.
    legal_values = np.maximum(legal_values, 0.0)
    total = float(legal_values.sum())
    if total <= 0.0:
        result[list(legal)] = 1.0 / len(legal)
    else:
        result[list(legal)] = legal_values / total
    return result


def legal_policy_from_logits(
    logits: Sequence[float] | np.ndarray, legal_actions: Sequence[int]
) -> np.ndarray:
    """Softmax logits over legal actions only, with illegal actions exactly zero."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (POLICY_SIZE,):
        raise ValueError(f"expected {POLICY_SIZE} logits, got {values.shape}")
    legal = tuple(sorted({int(action) for action in legal_actions}))
    if not legal or any(action < 0 or action >= POLICY_SIZE for action in legal):
        raise ValueError("legal_actions must be non-empty actions in [0, 5]")
    legal_logits = values[list(legal)]
    if not np.all(np.isfinite(legal_logits)):
        raise ValueError("legal logits must be finite")
    shifted = legal_logits - np.max(legal_logits)
    probabilities = np.exp(shifted)
    result = np.zeros(POLICY_SIZE, dtype=np.float64)
    result[list(legal)] = probabilities / probabilities.sum()
    return result


def zeroed_legal_policy(
    full_policy: Sequence[float] | np.ndarray, legal_actions: Sequence[int]
) -> np.ndarray:
    """Historical metric: retain full-softmax legal mass without renormalizing."""
    values = np.asarray(full_policy, dtype=np.float64)
    if values.shape != (POLICY_SIZE,):
        raise ValueError(f"expected {POLICY_SIZE} policy entries, got {values.shape}")
    result = np.zeros(POLICY_SIZE, dtype=np.float64)
    result[list(legal_actions)] = values[list(legal_actions)]
    return result


def policy_entropy(policy: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(policy, dtype=np.float64)
    return float(-sum(p * math.log2(p) for p in values if p > 0.0))
