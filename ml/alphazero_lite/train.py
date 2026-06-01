#!/usr/bin/env python3
"""Train a policy-value MLP checkpoint from JSONL rows using PyTorch."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.input_encodings import (
    DEFAULT_INPUT_ENCODING,
    SUPPORTED_INPUT_ENCODINGS,
    feature_count_for,
)
from ml.alphazero_lite.kalah_rules import KalahGame


POLICY_SIZE = 6
MLP_MODEL_TYPES = {"mlp_v1", "mlp_deep"}
RESIDUAL_MODEL_TYPES = {"residual_v2", "residual_v3"}
SUPPORTED_MODEL_TYPES = ["mlp_v1", "mlp_deep", "residual_v2", "residual_v3"]
DEFAULT_POLICY_TARGET_MODE = "default"
SUPPORTED_POLICY_TARGET_MODES = [DEFAULT_POLICY_TARGET_MODE, "sharpened"]
DEFAULT_VALUE_TARGET_MODE = "default"
PHASE_AWARE_VALUE_TARGET_MODE = "phase_aware_sharpened"
HYBRID_VALUE_TARGET_MODE = "hybrid"
DEFAULT_LR_SCHEDULER = "cosine"
SUPPORTED_LR_SCHEDULERS = ["none", DEFAULT_LR_SCHEDULER]
SUPPORTED_VALUE_TARGET_MODES = [
    DEFAULT_VALUE_TARGET_MODE,
    "sharpened",
    PHASE_AWARE_VALUE_TARGET_MODE,
    HYBRID_VALUE_TARGET_MODE,
]
STATE_NORMALIZATION_DENOMINATOR = 48.0


def input_size_for_encoding(input_encoding: str) -> int:
    return feature_count_for(input_encoding)


def normalize_policy_target_mode(policy_target_mode: str) -> str:
    normalized = str(policy_target_mode)
    if normalized not in SUPPORTED_POLICY_TARGET_MODES:
        raise ValueError(f"unsupported policy_target_mode: {policy_target_mode}")
    return normalized


def normalize_value_target_mode(value_target_mode: str) -> str:
    normalized = str(value_target_mode)
    if normalized not in SUPPORTED_VALUE_TARGET_MODES:
        raise ValueError(f"unsupported value_target_mode: {value_target_mode}")
    return normalized


def normalize_lr_scheduler(lr_scheduler: str) -> str:
    normalized = str(lr_scheduler)
    if normalized not in SUPPORTED_LR_SCHEDULERS:
        raise ValueError(f"unsupported lr_scheduler: {lr_scheduler}")
    return normalized


def validate_input_features(x: np.ndarray, *, input_encoding: str) -> None:
    expected = input_size_for_encoding(input_encoding)
    if x.ndim != 2:
        raise ValueError("training data states must be a 2D matrix")
    if x.shape[1] != expected:
        raise ValueError(
            f"training data feature_count must be {expected} for {input_encoding}, got {x.shape[1]}"
        )


def derive_legal_moves_from_encoded_state(
    state: np.ndarray | list[float],
) -> list[int] | None:
    encoded_state = np.asarray(state, dtype=np.float32)
    if encoded_state.ndim != 1:
        return None
    if encoded_state.shape[0] not in {
        feature_count_for("kalah_v1"),
        feature_count_for("kalah_v2"),
        feature_count_for("kalah_v3"),
    }:
        return None

    base_state = encoded_state[:15]
    if not np.all(np.isfinite(base_state)):
        return None

    current_player = float(base_state[14])
    if not np.isclose(current_player, round(current_player), atol=1e-6):
        return None
    current_player_int = int(round(current_player))
    if current_player_int not in (0, 1):
        return None

    player_pits = [
        int(round(value * STATE_NORMALIZATION_DENOMINATOR)) for value in base_state[:6]
    ]
    opponent_pits = [
        int(round(value * STATE_NORMALIZATION_DENOMINATOR))
        for value in base_state[6:12]
    ]
    player_store = int(round(float(base_state[12]) * STATE_NORMALIZATION_DENOMINATOR))
    opponent_store = int(round(float(base_state[13]) * STATE_NORMALIZATION_DENOMINATOR))

    decoded_values = np.asarray(
        [
            *(value / STATE_NORMALIZATION_DENOMINATOR for value in player_pits),
            *(value / STATE_NORMALIZATION_DENOMINATOR for value in opponent_pits),
            player_store / STATE_NORMALIZATION_DENOMINATOR,
            opponent_store / STATE_NORMALIZATION_DENOMINATOR,
            float(current_player_int),
        ],
        dtype=np.float32,
    )
    if not np.allclose(decoded_values, base_state, atol=1e-6):
        return None
    if any(
        value < 0
        for value in [*player_pits, *opponent_pits, player_store, opponent_store]
    ):
        return None

    game = KalahGame.from_state(
        {
            "player_pits": player_pits,
            "opponent_pits": opponent_pits,
            "player_store": player_store,
            "opponent_store": opponent_store,
            "current_player": current_player_int,
        }
    )
    return game.possible_moves()


def validate_policy_target(
    policy: np.ndarray,
    *,
    state: list[float] | np.ndarray,
    path: Path,
    row_number: int,
    policy_target_mode: str,
    declared_mode: str | None,
) -> None:
    if policy.shape != (POLICY_SIZE,):
        raise ValueError(f"{path}:{row_number} policy must contain {POLICY_SIZE} moves")
    if not np.all(np.isfinite(policy)):
        raise ValueError(f"{path}:{row_number} policy must be finite")
    if np.any(policy < 0.0):
        raise ValueError(
            f"{path}:{row_number} policy must be a legal normalized policy target"
        )

    total = float(np.sum(policy, dtype=np.float64))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"{path}:{row_number} policy must be a legal normalized policy target"
        )

    legal_moves = derive_legal_moves_from_encoded_state(state)
    if legal_moves is not None:
        if not legal_moves:
            raise ValueError(
                f"{path}:{row_number} policy state does not expose any legal moves"
            )
        illegal_moves = [
            move
            for move in range(POLICY_SIZE)
            if move not in legal_moves and policy[move] > 1e-6
        ]
        if illegal_moves:
            raise ValueError(
                f"{path}:{row_number} policy assigns probability to illegal moves: {illegal_moves}"
            )

    if declared_mode is None:
        if policy_target_mode != DEFAULT_POLICY_TARGET_MODE:
            raise ValueError(
                f"{path}:{row_number} must declare policy_target_mode={policy_target_mode}"
            )
        return

    normalized_declared_mode = normalize_policy_target_mode(declared_mode)
    if normalized_declared_mode != policy_target_mode:
        raise ValueError(
            f"{path}:{row_number} policy_target_mode={normalized_declared_mode} does not match requested {policy_target_mode}"
        )


def declared_policy_target_mode_for_row(row: dict[str, object]) -> str | None:
    actual_mode = row.get("policy_target_actual_mode")
    if actual_mode is not None:
        return str(actual_mode)
    declared_mode = row.get("policy_target_mode")
    if declared_mode is None:
        return None
    return str(declared_mode)


def validate_value_target_mode(
    *,
    path: Path,
    row_number: int,
    value_target_mode: str,
    declared_mode: str | None,
) -> None:
    if declared_mode is None:
        if value_target_mode != DEFAULT_VALUE_TARGET_MODE:
            raise ValueError(
                f"{path}:{row_number} must declare value_target_mode={value_target_mode}"
            )
        return

    normalized_declared_mode = normalize_value_target_mode(declared_mode)
    if normalized_declared_mode != value_target_mode:
        raise ValueError(
            f"{path}:{row_number} value_target_mode={normalized_declared_mode} does not match requested {value_target_mode}"
        )


def validate_value_target(value: float, *, path: Path, row_number: int) -> float:
    normalized_value = float(value)
    if not np.isfinite(normalized_value):
        raise ValueError(f"{path}:{row_number} value must be finite")
    if normalized_value < -1.0 or normalized_value > 1.0:
        raise ValueError(f"{path}:{row_number} value must stay within [-1.0, 1.0]")
    return normalized_value


def load_jsonl(
    path: Path,
    *,
    policy_target_mode: str = DEFAULT_POLICY_TARGET_MODE,
    value_target_mode: str = DEFAULT_VALUE_TARGET_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy_target_mode = normalize_policy_target_mode(policy_target_mode)
    value_target_mode = normalize_value_target_mode(value_target_mode)
    states = []
    policies = []
    values = []

    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            policy = np.asarray(row["policy"], dtype=np.float32)
            validate_policy_target(
                policy,
                state=row["state"],
                path=path,
                row_number=row_number,
                policy_target_mode=policy_target_mode,
                declared_mode=declared_policy_target_mode_for_row(row),
            )
            validate_value_target_mode(
                path=path,
                row_number=row_number,
                value_target_mode=value_target_mode,
                declared_mode=row.get("value_target_mode"),
            )
            value = validate_value_target(
                row["value"], path=path, row_number=row_number
            )
            states.append(row["state"])
            policies.append(policy)
            values.append(value)

    x = np.asarray(states, dtype=np.float32)
    p = np.asarray(policies, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return x, p, v


def load_jsonl_replay(
    paths: list[Path],
    weights: list[int] | None = None,
    *,
    policy_target_mode: str = DEFAULT_POLICY_TARGET_MODE,
    value_target_mode: str = DEFAULT_VALUE_TARGET_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("at least one replay path is required")

    policy_target_mode = normalize_policy_target_mode(policy_target_mode)
    value_target_mode = normalize_value_target_mode(value_target_mode)

    if weights is None:
        weights = [1] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("replay weights must match replay path count")
    if any(weight <= 0 for weight in weights):
        raise ValueError("replay weights must be positive integers")

    x_chunks: list[np.ndarray] = []
    p_chunks: list[np.ndarray] = []
    v_chunks: list[np.ndarray] = []
    replay_index_chunks: list[np.ndarray] = []
    row_offset = 0

    for path, weight in zip(paths, weights):
        x, p, v = load_jsonl(
            path,
            policy_target_mode=policy_target_mode,
            value_target_mode=value_target_mode,
        )
        x_chunks.append(x)
        p_chunks.append(p)
        v_chunks.append(v)
        compact_indexes = np.arange(row_offset, row_offset + x.shape[0], dtype=np.int64)
        replay_index_chunks.append(np.tile(compact_indexes, weight))
        row_offset += x.shape[0]

    return (
        np.concatenate(x_chunks, axis=0),
        np.concatenate(p_chunks, axis=0),
        np.concatenate(v_chunks, axis=0),
        np.concatenate(replay_index_chunks, axis=0),
    )


def parse_replay_paths(text: str | None) -> list[Path]:
    if not text:
        return []
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_replay_weights(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def compute_policy_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1)


def compute_value_loss_vector(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    value_loss: str,
    huber_delta: float,
) -> torch.Tensor:
    if value_loss == "huber":
        return torch.nn.functional.smooth_l1_loss(
            predictions,
            targets,
            beta=huber_delta,
            reduction="none",
        ).reshape(-1)
    return torch.square(predictions - targets).reshape(-1)


def train_one_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    compact_x: np.ndarray,
    compact_p: np.ndarray,
    compact_v: np.ndarray,
    replay_indexes: np.ndarray,
    batch_size: int,
    device: torch.device,
    value_loss_weight: float,
    value_loss: str,
    huber_delta: float,
    grad_clip: float | None,
) -> dict[str, float | None]:
    model.train()
    x_all = torch.from_numpy(compact_x).to(device)
    p_all = torch.from_numpy(compact_p).to(device)
    v_all = torch.from_numpy(compact_v).to(device)
    replay_tensor = torch.from_numpy(replay_indexes).to(device)
    permutation = torch.randperm(replay_tensor.size(0), device=device)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    total_losses: list[float] = []
    for start in range(0, replay_tensor.size(0), batch_size):
        indexes = permutation[start : start + batch_size]
        batch_replay_indexes = replay_tensor[indexes]
        batch_x = x_all[batch_replay_indexes]
        batch_p = p_all[batch_replay_indexes]
        batch_v = v_all[batch_replay_indexes]
        logits, value_pred = model(batch_x)
        policy_loss = compute_policy_cross_entropy(logits, batch_p).mean()
        value_component = compute_value_loss_vector(
            value_pred,
            batch_v,
            value_loss=value_loss,
            huber_delta=huber_delta,
        ).mean()
        total_loss = policy_loss + (value_loss_weight * value_component)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if grad_clip is not None and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        policy_losses.append(float(policy_loss.detach().cpu().item()))
        value_losses.append(float(value_component.detach().cpu().item()))
        total_losses.append(float(total_loss.detach().cpu().item()))
    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
    }


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_sizes: tuple[int, ...], model_type: str, input_size: int):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        self.input_layer: nn.Linear | None = None
        self.policy_hidden_layer: nn.Linear | None = None
        self.value_hidden_layer: nn.Linear | None = None
        policy_head_input_size: int | None = None
        value_head_input_size: int | None = None

        if model_type in MLP_MODEL_TYPES:
            if len(hidden_sizes) < 2:
                raise ValueError("hidden_sizes must include at least two layers")

            layer_sizes = [input_size, *hidden_sizes]
            self.hidden_layers = nn.ModuleList(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            )
            policy_head_input_size = hidden_sizes[-1]
            value_head_input_size = hidden_sizes[-1]
        elif model_type in RESIDUAL_MODEL_TYPES:
            trunk_size, residual_block_count = hidden_sizes
            self.input_layer = nn.Linear(input_size, trunk_size)
            self.residual_layers = nn.ModuleList(
                nn.ModuleList(
                    [
                        nn.Linear(trunk_size, trunk_size),
                        nn.Linear(trunk_size, trunk_size),
                    ]
                )
                for _ in range(residual_block_count)
            )
            if model_type == "residual_v3":
                self.policy_hidden_layer = nn.Linear(trunk_size, trunk_size)
                self.value_hidden_layer = nn.Linear(trunk_size, max(trunk_size // 2, 8))
                policy_head_input_size = trunk_size
                value_head_input_size = max(trunk_size // 2, 8)
            else:
                policy_head_input_size = trunk_size
                value_head_input_size = trunk_size
        else:
            raise ValueError(f"unsupported model_type: {model_type}")

        assert policy_head_input_size is not None
        assert value_head_input_size is not None
        self.policy_head = nn.Linear(policy_head_input_size, POLICY_SIZE)
        self.value_head = nn.Linear(value_head_input_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_type in MLP_MODEL_TYPES:
            h = x
            for layer in self.hidden_layers:
                h = torch.relu(layer(h))
        else:
            assert self.input_layer is not None
            h = torch.relu(self.input_layer(x))
            for first_layer, second_layer in self.residual_layers:
                residual = h
                h = torch.relu(first_layer(h))
                h = torch.relu(second_layer(h) + residual)
        if self.model_type == "residual_v3":
            assert self.policy_hidden_layer is not None
            assert self.value_hidden_layer is not None
            policy_features = torch.relu(self.policy_hidden_layer(h))
            value_features = torch.relu(self.value_hidden_layer(h))
            policy_logits = self.policy_head(policy_features)
            value = torch.tanh(self.value_head(value_features))
        else:
            policy_logits = self.policy_head(h)
            value = torch.tanh(self.value_head(h))
        return policy_logits, value


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(requested)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_replay_positions_by_source_row(
    replay_indexes: np.ndarray,
    *,
    val_split: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_val_split = max(0.0, min(val_split, 0.5))
    source_rows = np.unique(replay_indexes)
    source_row_count = int(source_rows.shape[0])

    if source_row_count <= 0:
        raise ValueError("replay set is empty")

    if normalized_val_split <= 0.0 or source_row_count < 2:
        val_source_count = 0
    else:
        val_source_count = max(1, int(source_row_count * normalized_val_split))
        val_source_count = min(val_source_count, source_row_count - 1)

    train_source_count = source_rows.shape[0] - val_source_count
    if train_source_count <= 0:
        raise ValueError("training set is empty; decrease --val-split")

    shuffled_source_rows = np.random.permutation(source_rows)
    val_source_rows = shuffled_source_rows[train_source_count:]

    is_val = np.isin(replay_indexes, val_source_rows)
    train_positions = np.flatnonzero(~is_val)
    val_positions = np.flatnonzero(is_val)
    return train_positions, val_positions


def train(
    model: PolicyValueNet,
    x: np.ndarray,
    p_target: np.ndarray,
    v_target: np.ndarray,
    replay_indexes: np.ndarray | None = None,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    value_loss_weight: float,
    value_loss: str,
    huber_delta: float,
    val_split: float,
    grad_clip: float | None,
    save_top_k: int,
    lr_scheduler: str,
) -> tuple[float, float, float]:
    model.to(device)
    model.train()

    compact_size = x.shape[0]
    if replay_indexes is None:
        replay_indexes_array = np.arange(compact_size, dtype=np.int64)
    elif np.any((replay_indexes < 0) | (replay_indexes >= compact_size)):
        raise ValueError("replay indexes must point to valid samples")
    else:
        replay_indexes_array = replay_indexes.astype(np.int64, copy=False)

    train_positions, val_positions = split_replay_positions_by_source_row(
        replay_indexes_array, val_split=val_split
    )
    val_count = int(val_positions.shape[0])

    x_all = torch.from_numpy(x).to(device)
    p_all = torch.from_numpy(p_target).to(device)
    v_all = torch.from_numpy(v_target).to(device)
    train_replay_indexes = replay_indexes_array[train_positions]
    val_replay_indexes = replay_indexes_array[val_positions] if val_count > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    normalized_lr_scheduler = normalize_lr_scheduler(lr_scheduler)
    scheduler = None
    if normalized_lr_scheduler == DEFAULT_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

    policy_loss_value = 0.0
    value_loss_value = 0.0
    best_val_loss = float("inf")
    best_state = None
    top_states: list[tuple[float, dict[str, torch.Tensor]]] = []

    def maybe_record_top_state(loss_value: float):
        nonlocal top_states
        if save_top_k <= 0:
            return
        snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_states.append((loss_value, snapshot))
        top_states.sort(key=lambda item: item[0])
        top_states = top_states[:save_top_k]

    for _epoch in range(epochs):
        epoch_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            compact_x=x,
            compact_p=p_target,
            compact_v=v_target,
            replay_indexes=train_replay_indexes,
            batch_size=batch_size,
            device=device,
            value_loss_weight=value_loss_weight,
            value_loss=value_loss,
            huber_delta=huber_delta,
            grad_clip=grad_clip,
        )
        policy_loss_value = float(epoch_metrics["policy_loss"] or 0.0)
        value_loss_value = float(epoch_metrics["value_loss"] or 0.0)

        if scheduler is not None:
            scheduler.step()

        if val_count > 0:
            model.eval()
            with torch.no_grad():
                x_val = x_all[val_replay_indexes]
                p_val = p_all[val_replay_indexes]
                v_val = v_all[val_replay_indexes]
                val_logits, val_value_pred = model(x_val)
                val_policy_loss = compute_policy_cross_entropy(val_logits, p_val).mean()
                val_value_loss = compute_value_loss_vector(
                    val_value_pred,
                    v_val,
                    value_loss=value_loss,
                    huber_delta=huber_delta,
                ).mean()
                val_total = float(
                    (val_policy_loss + (value_loss_weight * val_value_loss))
                    .cpu()
                    .item()
                )
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                maybe_record_top_state(val_total)
            model.train()

        if val_count == 0:
            maybe_record_top_state(
                policy_loss_value + (value_loss_weight * value_loss_value)
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    if best_val_loss == float("inf"):
        best_val_loss = policy_loss_value + (value_loss_weight * value_loss_value)

    model.top_states = top_states

    return policy_loss_value, value_loss_value, best_val_loss


def checkpoint_from_model(model: PolicyValueNet) -> dict[str, np.ndarray]:
    state = model.state_dict()
    return checkpoint_from_state_dict(state)


def checkpoint_from_state_dict(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    checkpoint: dict[str, np.ndarray] = {
        "w_policy": state["policy_head.weight"]
        .detach()
        .cpu()
        .numpy()
        .T.astype(np.float32),
        "b_policy": state["policy_head.bias"].detach().cpu().numpy().astype(np.float32),
        "w_value": state["value_head.weight"]
        .detach()
        .cpu()
        .numpy()
        .T.astype(np.float32),
        "b_value": state["value_head.bias"].detach().cpu().numpy().astype(np.float32),
    }

    if "policy_hidden_layer.weight" in state:
        checkpoint["w_policy_hidden"] = (
            state["policy_hidden_layer.weight"]
            .detach()
            .cpu()
            .numpy()
            .T.astype(np.float32)
        )
        checkpoint["b_policy_hidden"] = (
            state["policy_hidden_layer.bias"].detach().cpu().numpy().astype(np.float32)
        )
    if "value_hidden_layer.weight" in state:
        checkpoint["w_value_hidden"] = (
            state["value_hidden_layer.weight"]
            .detach()
            .cpu()
            .numpy()
            .T.astype(np.float32)
        )
        checkpoint["b_value_hidden"] = (
            state["value_hidden_layer.bias"].detach().cpu().numpy().astype(np.float32)
        )

    if "input_layer.weight" in state:
        checkpoint["w_input"] = (
            state["input_layer.weight"].detach().cpu().numpy().T.astype(np.float32)
        )
        checkpoint["b_input"] = (
            state["input_layer.bias"].detach().cpu().numpy().astype(np.float32)
        )

        block_index = 1
        while f"residual_layers.{block_index - 1}.0.weight" in state:
            first_weight_key = f"residual_layers.{block_index - 1}.0.weight"
            first_bias_key = f"residual_layers.{block_index - 1}.0.bias"
            second_weight_key = f"residual_layers.{block_index - 1}.1.weight"
            second_bias_key = f"residual_layers.{block_index - 1}.1.bias"
            checkpoint[f"w_residual_{block_index}_1"] = (
                state[first_weight_key].detach().cpu().numpy().T.astype(np.float32)
            )
            checkpoint[f"b_residual_{block_index}_1"] = (
                state[first_bias_key].detach().cpu().numpy().astype(np.float32)
            )
            checkpoint[f"w_residual_{block_index}_2"] = (
                state[second_weight_key].detach().cpu().numpy().T.astype(np.float32)
            )
            checkpoint[f"b_residual_{block_index}_2"] = (
                state[second_bias_key].detach().cpu().numpy().astype(np.float32)
            )
            block_index += 1
        return checkpoint

    hidden_index = 1
    while f"hidden_layers.{hidden_index - 1}.weight" in state:
        weight_key = f"hidden_layers.{hidden_index - 1}.weight"
        bias_key = f"hidden_layers.{hidden_index - 1}.bias"
        checkpoint[f"w_hidden_{hidden_index}"] = (
            state[weight_key].detach().cpu().numpy().T.astype(np.float32)
        )
        checkpoint[f"b_hidden_{hidden_index}"] = (
            state[bias_key].detach().cpu().numpy().astype(np.float32)
        )
        hidden_index += 1

    # Backward-compatible aliases for legacy 2-layer Ruby evaluator paths.
    if "w_hidden_1" in checkpoint and "w_hidden_2" in checkpoint:
        checkpoint["w1"] = checkpoint["w_hidden_1"]
        checkpoint["b1"] = checkpoint["b_hidden_1"]
        checkpoint["w2"] = checkpoint["w_hidden_2"]
        checkpoint["b2"] = checkpoint["b_hidden_2"]

    return checkpoint


def load_checkpoint_into_model(model: PolicyValueNet, checkpoint_path: Path) -> None:
    checkpoint = np.load(checkpoint_path)
    state_dict = model.state_dict()

    if "w_input" in checkpoint and "b_input" in checkpoint:
        state_dict["input_layer.weight"] = torch.from_numpy(
            checkpoint["w_input"].T.copy()
        )
        state_dict["input_layer.bias"] = torch.from_numpy(checkpoint["b_input"].copy())

        block_index = 1
        while (
            f"w_residual_{block_index}_1" in checkpoint
            and f"b_residual_{block_index}_1" in checkpoint
        ):
            first_weight_key = f"residual_layers.{block_index - 1}.0.weight"
            first_bias_key = f"residual_layers.{block_index - 1}.0.bias"
            second_weight_key = f"residual_layers.{block_index - 1}.1.weight"
            second_bias_key = f"residual_layers.{block_index - 1}.1.bias"
            state_dict[first_weight_key] = torch.from_numpy(
                checkpoint[f"w_residual_{block_index}_1"].T.copy()
            )
            state_dict[first_bias_key] = torch.from_numpy(
                checkpoint[f"b_residual_{block_index}_1"].copy()
            )
            state_dict[second_weight_key] = torch.from_numpy(
                checkpoint[f"w_residual_{block_index}_2"].T.copy()
            )
            state_dict[second_bias_key] = torch.from_numpy(
                checkpoint[f"b_residual_{block_index}_2"].copy()
            )
            block_index += 1

        if (
            "policy_hidden_layer.weight" in state_dict
            and "w_policy_hidden" in checkpoint
        ):
            state_dict["policy_hidden_layer.weight"] = torch.from_numpy(
                checkpoint["w_policy_hidden"].T.copy()
            )
            state_dict["policy_hidden_layer.bias"] = torch.from_numpy(
                checkpoint["b_policy_hidden"].copy()
            )
        if "value_hidden_layer.weight" in state_dict and "w_value_hidden" in checkpoint:
            state_dict["value_hidden_layer.weight"] = torch.from_numpy(
                checkpoint["w_value_hidden"].T.copy()
            )
            state_dict["value_hidden_layer.bias"] = torch.from_numpy(
                checkpoint["b_value_hidden"].copy()
            )

        state_dict["policy_head.weight"] = torch.from_numpy(
            checkpoint["w_policy"].T.copy()
        )
        state_dict["policy_head.bias"] = torch.from_numpy(checkpoint["b_policy"].copy())
        state_dict["value_head.weight"] = torch.from_numpy(
            checkpoint["w_value"].T.copy()
        )
        state_dict["value_head.bias"] = torch.from_numpy(checkpoint["b_value"].copy())
        model.load_state_dict(state_dict)
        return

    hidden_index = 1
    while (
        f"w_hidden_{hidden_index}" in checkpoint
        and f"b_hidden_{hidden_index}" in checkpoint
    ):
        weight_key = f"hidden_layers.{hidden_index - 1}.weight"
        bias_key = f"hidden_layers.{hidden_index - 1}.bias"
        state_dict[weight_key] = torch.from_numpy(
            checkpoint[f"w_hidden_{hidden_index}"].T.copy()
        )
        state_dict[bias_key] = torch.from_numpy(
            checkpoint[f"b_hidden_{hidden_index}"].copy()
        )
        hidden_index += 1

    state_dict["policy_head.weight"] = torch.from_numpy(checkpoint["w_policy"].T.copy())
    state_dict["policy_head.bias"] = torch.from_numpy(checkpoint["b_policy"].copy())
    state_dict["value_head.weight"] = torch.from_numpy(checkpoint["w_value"].T.copy())
    state_dict["value_head.bias"] = torch.from_numpy(checkpoint["b_value"].copy())
    model.load_state_dict(state_dict)


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        raise ValueError(
            "--hidden-sizes must provide at least two comma-separated integers"
        )
    hidden_sizes = tuple(int(part) for part in parts)
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden sizes must be positive")
    return hidden_sizes


def resolve_hidden_sizes(
    model_type: str, hidden_sizes: tuple[int, ...]
) -> tuple[int, ...]:
    if model_type in RESIDUAL_MODEL_TYPES:
        if len(hidden_sizes) != 2:
            raise ValueError(
                f"--hidden-sizes for {model_type} must provide trunk_size,residual_block_count"
            )
        trunk_size, residual_block_count = hidden_sizes
        if residual_block_count <= 0:
            raise ValueError("residual_block_count must be positive")
        return trunk_size, residual_block_count
    if model_type == "mlp_deep" and len(hidden_sizes) < 3:
        return (*hidden_sizes, 64)
    return hidden_sizes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False, help="JSONL training data path")
    parser.add_argument(
        "--data-files", default=None, help="Comma-separated JSONL training data paths"
    )
    parser.add_argument(
        "--replay-weights", default=None, help="Comma-separated integer replay weights"
    )
    parser.add_argument("--out", required=True, help="Checkpoint .npz output path")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--steps", type=int, default=None, help="Deprecated alias for --epochs"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--hidden-sizes", default="64,64", help="Two comma-separated hidden layer sizes"
    )
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default="mlp_v1")
    parser.add_argument(
        "--input-encoding",
        choices=SUPPORTED_INPUT_ENCODINGS,
        default=DEFAULT_INPUT_ENCODING,
    )
    parser.add_argument(
        "--policy-target-mode",
        choices=SUPPORTED_POLICY_TARGET_MODES,
        default=DEFAULT_POLICY_TARGET_MODE,
    )
    parser.add_argument(
        "--value-target-mode",
        choices=SUPPORTED_VALUE_TARGET_MODES,
        default=DEFAULT_VALUE_TARGET_MODE,
    )
    parser.add_argument("--save-top-k", type=int, default=0)
    parser.add_argument("--top-k-dir", default=None)
    parser.add_argument(
        "--lr-scheduler",
        choices=SUPPORTED_LR_SCHEDULERS,
        default=DEFAULT_LR_SCHEDULER,
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to load before training",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    policy_target_mode = normalize_policy_target_mode(args.policy_target_mode)
    value_target_mode = normalize_value_target_mode(args.value_target_mode)

    data_path = Path(args.data) if args.data else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = select_device(args.device)
    hidden_sizes = resolve_hidden_sizes(
        args.model_type, parse_hidden_sizes(args.hidden_sizes)
    )

    replay_paths = parse_replay_paths(args.data_files)
    replay_weights = parse_replay_weights(args.replay_weights)
    replay_indexes = None

    if replay_paths:
        x, p_target, v_target, replay_indexes = load_jsonl_replay(
            replay_paths,
            replay_weights,
            policy_target_mode=policy_target_mode,
            value_target_mode=value_target_mode,
        )
    else:
        if data_path is None:
            raise SystemExit("--data is required when --data-files is not provided")
        x, p_target, v_target = load_jsonl(
            data_path,
            policy_target_mode=policy_target_mode,
            value_target_mode=value_target_mode,
        )
    validate_input_features(x, input_encoding=args.input_encoding)
    model = PolicyValueNet(
        hidden_sizes=hidden_sizes,
        model_type=args.model_type,
        input_size=input_size_for_encoding(args.input_encoding),
    )
    if args.init_checkpoint:
        load_checkpoint_into_model(model, Path(args.init_checkpoint))
    epochs = args.steps if args.steps is not None else args.epochs

    policy_loss, value_loss, best_val_loss = train(
        model,
        x,
        p_target,
        v_target,
        replay_indexes,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        value_loss_weight=args.value_loss_weight,
        value_loss=args.value_loss,
        huber_delta=args.huber_delta,
        val_split=args.val_split,
        grad_clip=args.grad_clip,
        save_top_k=args.save_top_k,
        lr_scheduler=args.lr_scheduler,
    )

    checkpoint = checkpoint_from_model(model)
    np.savez(out_path, **checkpoint)
    print(f"saved checkpoint to {out_path}")
    print(f"device={device.type}")
    print(f"input_encoding={args.input_encoding}")
    print(f"policy_target_mode={policy_target_mode}")
    print(f"value_target_mode={value_target_mode}")
    print(f"model_type={args.model_type}")
    print(f"policy_loss={policy_loss:.6f}")
    print(f"value_loss={value_loss:.6f}")
    print(f"best_val_loss={best_val_loss:.6f}")

    if args.save_top_k > 0:
        topk_dir = Path(args.top_k_dir) if args.top_k_dir else out_path.parent
        topk_dir.mkdir(parents=True, exist_ok=True)
        base = out_path.stem
        top_states = getattr(model, "top_states", [])[: args.save_top_k]
        for idx, (val_loss, state) in enumerate(top_states, start=1):
            target = topk_dir / f"{base}.top{idx}.npz"
            np.savez(target, **checkpoint_from_state_dict(state))
            print(f"saved_top_checkpoint_{idx}={target} val_loss={val_loss:.6f}")
        print(f"saved_top_k={len(top_states)}")


if __name__ == "__main__":
    main()
