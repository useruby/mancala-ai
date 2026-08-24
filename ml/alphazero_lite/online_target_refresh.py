"""Deterministic, pre-update search targets for adapter-only experiments."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from ml.alphazero_lite.kalah_rules import KalahGame
from ml.alphazero_lite.self_play import Evaluator, PITS_PER_PLAYER, PUCT, encode_state
from ml.alphazero_lite.shadow_root_q import run_shadow_root_q_search

EXPERIMENT_NAMESPACE = "pr234-online-current-candidate-targets"
SIMULATIONS = 1200


class TorchEvaluator(Evaluator):
    """Evaluate the exact in-memory candidate without exporting an artifact."""

    def __init__(
        self, model: torch.nn.Module, device: torch.device, input_encoding: str
    ):
        self.model = model
        self.device = device
        self.input_encoding = input_encoding

    def evaluate(self, game: KalahGame) -> tuple[np.ndarray, float]:
        encoded = encode_state(game.to_state(), input_encoding=self.input_encoding)
        x = torch.tensor([encoded], dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            logits, value = self.model(x)
            priors = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float32)
        masked = np.zeros(PITS_PER_PLAYER, dtype=np.float32)
        legal = game.possible_moves()
        if legal:
            masked[legal] = priors[legal]
            masked /= float(masked.sum())
        return masked, float(value[0, 0].detach().cpu())


def target_seed(state_hash: str, target_type: str) -> int:
    """Use only stable state/type/namespace inputs; optimizer step is excluded."""
    digest = hashlib.sha256(
        f"{EXPERIMENT_NAMESPACE}:{target_type}:{state_hash}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def ordinary_target(game: KalahGame, evaluator: Evaluator, *, seed: int) -> np.ndarray:
    """Return the frozen deterministic ordinary-PUCT visit distribution."""
    search = PUCT(
        evaluator,
        SIMULATIONS,
        1.25,
        random.Random(seed),
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
    )
    visits, _ = search.run(game, dirichlet_alpha=None, dirichlet_epsilon=0.0)
    return visits.astype(np.float64) / SIMULATIONS


def shadow_target(
    game: KalahGame,
    main_evaluator: Evaluator,
    shadow_evaluator: Evaluator,
    *,
    seed: int,
) -> np.ndarray:
    """Return the PR #229/#233 shadow-root-Q visit distribution."""
    visits, _, _ = run_shadow_root_q_search(
        game,
        main_evaluator=main_evaluator,
        shadow_evaluator=shadow_evaluator,
        simulations=SIMULATIONS,
        c_puct=1.25,
        seed=seed,
        fpu_mode="zero",
        reuse_subtree=False,
        normalize_values=False,
        root_policy_mode="deterministic",
        tactical_root_bias=0.0,
        root_temperature=0.0,
        shadow_q_weight=1.0,
    )
    return visits.astype(np.float64) / SIMULATIONS


def model_state_digest(model: torch.nn.Module) -> str:
    """Fingerprint candidate parameters to prove target generation precedes update."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
