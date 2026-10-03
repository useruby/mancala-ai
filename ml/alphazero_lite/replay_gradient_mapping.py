"""Compact source-row identity and cohort-gradient primitives for replay audits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ml.alphazero_lite import train


@dataclass(frozen=True)
class ReplayMapping:
    """Source identity and multiplicity indexed only by compact loaded row ID."""

    compact_source_ids: np.ndarray
    compact_source_names: tuple[str, ...]
    compact_local_row_ids: np.ndarray
    train_positions: np.ndarray
    validation_positions: np.ndarray
    train_compact_ids: np.ndarray
    validation_compact_ids: np.ndarray
    train_multiplicity: np.ndarray
    validation_multiplicity: np.ndarray


def make_replay_mapping(
    compact_source_names: list[str] | tuple[str, ...],
    replay_indexes: np.ndarray,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> ReplayMapping:
    """Map production split positions to compact rows without conflating copies."""
    replay_indexes = np.asarray(replay_indexes, dtype=np.int64)
    train_positions = np.asarray(train_positions, dtype=np.int64)
    validation_positions = np.asarray(validation_positions, dtype=np.int64)
    names = tuple(compact_source_names)
    if len(names) == 0 or not len(replay_indexes):
        raise ValueError("empty_compact_replay")
    if np.any(replay_indexes < 0) or np.any(replay_indexes >= len(names)):
        raise ValueError("replay_index_outside_compact_rows")
    expected_positions = np.arange(len(replay_indexes), dtype=np.int64)
    if not np.array_equal(
        np.sort(np.concatenate((train_positions, validation_positions))),
        expected_positions,
    ):
        raise ValueError("split_positions_must_partition_replay_positions")
    train_ids = replay_indexes[train_positions]
    validation_ids = replay_indexes[validation_positions]
    train_set, validation_set = set(train_ids.tolist()), set(validation_ids.tolist())
    if train_set & validation_set:
        raise ValueError("compact_source_row_leaks_across_split")
    # Compact IDs are assigned in source loader order. Local IDs remain unique even
    # when source rows have identical state/policy/value contents.
    local_ids = np.zeros(len(names), dtype=np.int64)
    source_ids: dict[str, int] = {}
    encoded_source_ids = np.zeros(len(names), dtype=np.int64)
    source_offsets: dict[str, int] = {}
    for compact_id, source in enumerate(names):
        encoded_source_ids[compact_id] = source_ids.setdefault(source, len(source_ids))
        local_ids[compact_id] = source_offsets.get(source, 0)
        source_offsets[source] = int(local_ids[compact_id]) + 1
    return ReplayMapping(
        compact_source_ids=encoded_source_ids,
        compact_source_names=names,
        compact_local_row_ids=local_ids,
        train_positions=train_positions,
        validation_positions=validation_positions,
        train_compact_ids=train_ids,
        validation_compact_ids=validation_ids,
        train_multiplicity=np.bincount(train_ids, minlength=len(names)),
        validation_multiplicity=np.bincount(validation_ids, minlength=len(names)),
    )


def split_compact_rows(
    replay_indexes: np.ndarray, *, seed: int, validation_split: float
) -> tuple[np.ndarray, np.ndarray]:
    """Invoke the production source-row splitter under the frozen NumPy seed."""
    np.random.seed(seed)
    train_positions, validation_positions = train.split_replay_positions_by_source_row(
        replay_indexes, val_split=validation_split
    )
    return train_positions, validation_positions


def partition_id(source: str, local_row_id: int, count: int = 4) -> int:
    """Stable, predeclared partition function; copies inherit compact-row ID."""
    if count < 2:
        raise ValueError("partition_count_must_be_at_least_two")
    payload = f"fresh-historical-gradient-v1\0{source}\0{local_row_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % count


def cohort_loss_components(
    policy_rows: torch.Tensor,
    weighted_value_rows: torch.Tensor,
    policy_weights: torch.Tensor,
    multiplicity: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-row numerator contributions over shared cohort denominators."""
    active_mult = multiplicity * selected.to(multiplicity.dtype)
    policy_mass = policy_weights * active_mult
    policy_den = policy_mass.sum()
    value_den = active_mult.sum()
    zero = policy_rows.sum() * 0.0
    policy = (
        (policy_rows * policy_mass).sum() / policy_den if policy_den.item() else zero
    )
    value = (
        (weighted_value_rows * active_mult).sum() / value_den
        if value_den.item()
        else zero
    )
    return policy, value, policy + value


def source_contribution_loss(
    policy_rows: torch.Tensor,
    weighted_value_rows: torch.Tensor,
    policy_weights: torch.Tensor,
    multiplicity: torch.Tensor,
    cohort: torch.Tensor,
    source_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source numerator divided by the whole cohort's policy/value denominators."""
    mass = multiplicity * cohort.to(multiplicity.dtype)
    policy_mass = mass * policy_weights
    pden, vden = policy_mass.sum(), mass.sum()
    zero = policy_rows.sum() * 0.0
    policy = (
        (policy_rows * policy_mass * source_mask.to(policy_mass.dtype)).sum() / pden
        if pden.item()
        else zero
    )
    value = (
        (weighted_value_rows * mass * source_mask.to(mass.dtype)).sum() / vden
        if vden.item()
        else zero
    )
    return policy, value, policy + value
