"""Experiment-only selective-head evaluator for residual_v3 compositions."""

from __future__ import annotations

import numpy as np

from ml.alphazero_lite.arena import ArtifactEvaluator
from ml.alphazero_lite.kalah_rules import KalahGame, PITS_PER_PLAYER


class SelectiveHeadArtifactEvaluator:
    """Use policy-source trunk/policy head and value-source trunk/value head."""

    def __init__(
        self,
        policy_evaluator: ArtifactEvaluator,
        value_evaluator: ArtifactEvaluator,
    ) -> None:
        for label, evaluator in (
            ("policy", policy_evaluator),
            ("value", value_evaluator),
        ):
            if evaluator.model_type != "residual_v3":
                raise ValueError(
                    f"selective-head inference supports residual_v3 only; "
                    f"{label} model is {evaluator.model_type!r}"
                )
            if evaluator.move_factorized or evaluator.w_policy_adapter is not None:
                raise ValueError(
                    f"selective-head inference does not support {label} architecture extensions"
                )
        self.policy_evaluator = policy_evaluator
        self.value_evaluator = value_evaluator

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        if game.over():
            return np.zeros(PITS_PER_PLAYER, dtype=np.float32), 0.0
        policy_source = self.policy_evaluator
        value_source = self.value_evaluator
        policy_trunk = policy_source.extract_trunk(game)
        value_trunk = value_source.extract_trunk(game)

        policy_hidden = np.maximum(
            0.0,
            (policy_trunk @ policy_source.w_policy_hidden)
            + policy_source.b_policy_hidden,
        )
        logits = (policy_hidden @ policy_source.w_policy) + policy_source.b_policy
        logits = logits - np.max(logits)
        priors = np.exp(logits)
        priors = priors / np.sum(priors)
        legal_moves = game.possible_moves()
        masked = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        if legal_moves:
            masked[legal_moves] = priors[legal_moves]
            total = float(np.sum(masked))
            if total <= 0:
                masked[legal_moves] = 1.0 / len(legal_moves)
            else:
                masked /= total

        value_hidden = np.maximum(
            0.0,
            (value_trunk @ value_source.w_value_hidden) + value_source.b_value_hidden,
        )
        value = float(
            np.tanh(
                (value_hidden @ value_source.w_value + value_source.b_value).reshape(
                    -1
                )[0]
            )
        )
        return masked, value
