"""Model-independent deterministic search seed identities for evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED_CONTRACT_VERSION = "azlite_eval_seed_v1"
SEED_IDENTITY_FIELDS = frozenset(
    {
        "contract_version",
        "base_seed",
        "suite_sha256",
        "budget_pair",
        "opening_index",
        "opening_state_hash",
        "challenger_player",
        "game_within_opening",
        "ply",
        "canonical_current_state_hash",
        "acting_role",
        "simulations",
        "effective_c_puct",
    }
)


def canonical_json(value: Any) -> bytes:
    """Serialize an identity value without depending on Python hash randomization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_hash(value: Any) -> str:
    """Return a cryptographic, stable digest for an evaluation identity."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_seed(*parts: Any) -> int:
    """Derive a PUCT-compatible deterministic seed from stable identity parts."""
    return int(stable_hash(list(parts))[:16], 16) % (2**31)


def search_seed_context(
    *,
    base_seed: int,
    suite_sha256: str,
    budget_pair: str,
    opening_index: int,
    opening_state_hash: str,
    challenger_player: int,
    game_within_opening: int,
    ply: int,
    canonical_current_state_hash: str,
    acting_role: str,
    simulations: int,
    effective_c_puct: float,
) -> dict[str, Any]:
    """Build the complete v1 seed identity, excluding model and execution details."""
    if acting_role not in {"challenger", "current"}:
        raise ValueError("acting_role must be challenger or current")
    return {
        "contract_version": SEED_CONTRACT_VERSION,
        "base_seed": int(base_seed),
        "suite_sha256": str(suite_sha256),
        "budget_pair": str(budget_pair),
        "opening_index": int(opening_index),
        "opening_state_hash": str(opening_state_hash),
        "challenger_player": int(challenger_player),
        "game_within_opening": int(game_within_opening),
        "ply": int(ply),
        "canonical_current_state_hash": str(canonical_current_state_hash),
        "acting_role": acting_role,
        "simulations": int(simulations),
        "effective_c_puct": float(effective_c_puct),
    }


def derive_search_seed(**context: Any) -> tuple[int, str]:
    """Return the search seed and context digest for a complete v1 context."""
    identity = search_seed_context(**context)
    context_hash = stable_hash(identity)
    return stable_seed(identity), context_hash


def seed_identity_ledger_record(**context: Any) -> dict[str, Any]:
    """Return the model- and execution-independent persisted seed identity."""
    identity = search_seed_context(**context)
    derived_seed, context_hash = derive_search_seed(**context)
    return {
        **identity,
        "seed_context_hash": context_hash,
        "derived_search_seed": derived_seed,
    }
