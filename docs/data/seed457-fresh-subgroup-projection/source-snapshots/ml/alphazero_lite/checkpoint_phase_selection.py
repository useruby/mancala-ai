"""Phase definitions and loss calculations for observational checkpoint selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


HIGH_STONE_PHASE = "high_stone"
HIGH_STONE_THRESHOLD = 32
BEST_HIGH_STONE_VALIDATION_V1 = "best_high_stone_validation_v1"
EARLIEST_EPOCH_TIE_RULE = "earliest_epoch"


def active_pit_stones_from_state(state: dict[str, Any]) -> int:
    """Return network-owned pit stones, deliberately excluding both stores."""
    try:
        pits = list(state["player_pits"]) + list(state["opponent_pits"])
    except (KeyError, TypeError) as exc:
        raise ValueError("checkpoint_phase_state_decode_failed") from exc
    if len(pits) != 12 or any(int(stones) != stones or stones < 0 for stones in pits):
        raise ValueError("checkpoint_phase_state_decode_failed")
    return sum(int(stones) for stones in pits)


def active_pit_stones_from_encoded_state(state: np.ndarray | list[float]) -> int:
    """Strictly recover active pits from the invertible base prefix of kalah_v3."""
    if len(state) < 15:
        raise ValueError("checkpoint_phase_state_decode_failed")
    pits: list[int] = []
    for index in range(12):
        value = float(state[index]) * 48.0
        rounded = round(value)
        if abs(value - rounded) > 1e-5 or rounded < 0:
            raise ValueError("checkpoint_phase_state_decode_failed")
        pits.append(int(rounded))
    return sum(pits)


def phase_mask_for_encoded_states(
    states: np.ndarray, *, phase: str = HIGH_STONE_PHASE
) -> np.ndarray:
    if phase != HIGH_STONE_PHASE:
        raise ValueError("checkpoint_phase_selection_phase_invalid")
    return np.asarray(
        [
            active_pit_stones_from_encoded_state(state) > HIGH_STONE_THRESHOLD
            for state in states
        ],
        dtype=bool,
    )


def select_epoch_by_score(scores: list[float]) -> int:
    """Return the one-indexed minimum score epoch, breaking ties by order."""
    if not scores:
        raise ValueError("checkpoint_phase_validation_empty")
    return min(range(1, len(scores) + 1), key=lambda epoch: scores[epoch - 1])


def high_stone_validation_metrics(
    *,
    logits: torch.Tensor,
    value_predictions: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    legal_mask: torch.Tensor,
    policy_loss_weights: torch.Tensor,
    phase_mask: torch.Tensor,
    value_loss_weight: float,
    value_loss: str,
    huber_delta: float,
    policy_cross_entropy: Any,
    weighted_policy_loss: Any,
    value_loss_vector: Any,
) -> dict[str, float]:
    """Evaluate the normal supervised objective on phase rows only."""
    if int(phase_mask.sum().item()) == 0:
        raise ValueError("checkpoint_phase_validation_empty")
    masked_logits = logits[phase_mask].masked_fill(legal_mask[phase_mask] <= 0.0, -1e9)
    policy_loss = weighted_policy_loss(
        policy_cross_entropy(masked_logits, policy_targets[phase_mask]),
        policy_loss_weights[phase_mask],
    )
    value_vector = value_loss_vector(
        value_predictions[phase_mask],
        value_targets[phase_mask],
        value_loss=value_loss,
        huber_delta=huber_delta,
    )
    value_component = value_vector.mean()
    score = policy_loss + (value_loss_weight * value_component)
    return {
        "phase_validation_count": int(phase_mask.sum().item()),
        "phase_validation_policy_loss": float(policy_loss.cpu().item()),
        "phase_validation_value_loss": float(value_component.cpu().item()),
        "phase_validation_value_mae": float(
            torch.abs(value_predictions[phase_mask] - value_targets[phase_mask])
            .mean()
            .cpu()
            .item()
        ),
        "phase_validation_sign_accuracy": float(
            (
                torch.sign(value_predictions[phase_mask])
                == torch.sign(value_targets[phase_mask])
            )
            .float()
            .mean()
            .cpu()
            .item()
        ),
        "phase_validation_score": float(score.cpu().item()),
    }
