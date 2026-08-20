"""Exact residual_v3 policy-sublayer grafts for diagnostic experiments only."""

from __future__ import annotations

import hashlib
import json

import torch

HIDDEN_KEYS = ("policy_hidden_layer.weight", "policy_hidden_layer.bias")
READOUT_KEYS = ("policy_head.weight", "policy_head.bias")
POLICY_KEYS = HIDDEN_KEYS + READOUT_KEYS
TRUNK_PREFIXES = ("input_layer.", "residual_layers.")
VALUE_PREFIXES = ("value_hidden_layer.", "value_head.")


def byte_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare dtype, shape, and tensor bytes exactly."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.detach().cpu().numpy().tobytes()
        == right.detach().cpu().numpy().tobytes()
    )


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """Return a stable hash of the complete ordered state dictionary."""
    encoded = {
        name: value.detach().cpu().numpy().tobytes().hex()
        for name, value in state.items()
    }
    return hashlib.sha256(
        json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def graft_state(
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Copy exactly ``keys`` from candidate onto a cloned parent state."""
    if set(parent) != set(candidate):
        raise ValueError("parent and candidate state dictionaries differ")
    if not set(keys).issubset(parent):
        raise ValueError("graft keys are absent from the parent state")
    state = {name: value.detach().clone() for name, value in parent.items()}
    for name in keys:
        state[name] = candidate[name].detach().clone()
    return state


def assert_candidate_contract(
    parent: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> None:
    """Require that a PR #212 candidate changed only the policy stack."""
    changed = {
        name for name in parent if not byte_identical(parent[name], candidate[name])
    }
    if not changed:
        raise AssertionError("candidate contains no policy update")
    if not changed.issubset(POLICY_KEYS):
        raise AssertionError(f"candidate changed non-policy tensors: {sorted(changed)}")
    for name in parent:
        if name.startswith(TRUNK_PREFIXES + VALUE_PREFIXES) and not byte_identical(
            parent[name], candidate[name]
        ):
            raise AssertionError(f"candidate changed protected tensor: {name}")


def assert_graft_contract(
    name: str,
    parent: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    graft: dict[str, torch.Tensor],
) -> None:
    """Assert byte-exact source selection for each policy sublayer graft."""
    expected_keys = {
        "hidden_delta_only": HIDDEN_KEYS,
        "readout_delta_only": READOUT_KEYS,
        "full": POLICY_KEYS,
    }
    copied = expected_keys[name]
    for key in parent:
        source = candidate if key in copied else parent
        if not byte_identical(graft[key], source[key]):
            raise AssertionError(f"{name} has an incorrect tensor source: {key}")
    if name != "full":
        for key in TRUNK_PREFIXES + VALUE_PREFIXES:
            assert all(
                byte_identical(graft[tensor], parent[tensor])
                for tensor in parent
                if tensor.startswith(key)
            )
