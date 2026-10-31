"""Versioned deterministic seed and provenance contracts for evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SEED_CONTRACT_VERSION_V1 = "azlite_eval_seed_v1"
SEED_CONTRACT_VERSION = "azlite_eval_seed_v2"
SUPPORTED_SEED_CONTRACT_VERSIONS = frozenset(
    {SEED_CONTRACT_VERSION_V1, SEED_CONTRACT_VERSION}
)
SEED_IDENTITY_FIELDS = frozenset(
    {
        "contract_version",
        "base_seed",
        "suite_sha256",
        "opening_index",
        "opening_state_hash",
        "challenger_player",
        "game_within_opening",
        "ply",
        "canonical_current_state_hash",
        "acting_role",
        "rng_stream_name",
    }
)
V1_SEED_IDENTITY_FIELDS = frozenset(
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
    """Serialize a value without depending on Python hash randomization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_hash(value: Any) -> str:
    """Return a cryptographic, stable digest."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def stable_seed(*parts: Any) -> int:
    """Derive a PUCT-compatible deterministic seed from stable identity parts."""
    return int(stable_hash(list(parts))[:16], 16) % (2**31)


def search_seed_context(
    *,
    base_seed: int,
    suite_sha256: str,
    opening_index: int,
    opening_state_hash: str,
    challenger_player: int,
    game_within_opening: int,
    ply: int,
    canonical_current_state_hash: str,
    acting_role: str,
    rng_stream_name: str = "puct_search",
    contract_version: str = SEED_CONTRACT_VERSION,
    # v1 fields are accepted only to read/recompute historical manifests.
    budget_pair: str | None = None,
    simulations: int | None = None,
    effective_c_puct: float | None = None,
) -> dict[str, Any]:
    """Build a versioned seed identity.

    V2 deliberately contains only exogenous game context.  Search treatment is
    recorded by :func:`search_configuration_ledger_record`, never here.
    """
    if contract_version not in SUPPORTED_SEED_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported seed contract: {contract_version}")
    if acting_role not in {"challenger", "current"}:
        raise ValueError("acting_role must be challenger or current")
    identity = {
        "contract_version": contract_version,
        "base_seed": int(base_seed),
        "suite_sha256": str(suite_sha256),
        "opening_index": int(opening_index),
        "opening_state_hash": str(opening_state_hash),
        "challenger_player": int(challenger_player),
        "game_within_opening": int(game_within_opening),
        "ply": int(ply),
        "canonical_current_state_hash": str(canonical_current_state_hash),
        "acting_role": acting_role,
    }
    if contract_version == SEED_CONTRACT_VERSION_V1:
        if budget_pair is None or simulations is None or effective_c_puct is None:
            raise ValueError(
                "v1 seed contexts require budget_pair, simulations, and c_puct"
            )
        identity.update(
            {
                "budget_pair": str(budget_pair),
                "simulations": int(simulations),
                "effective_c_puct": float(effective_c_puct),
            }
        )
    else:
        identity["rng_stream_name"] = str(rng_stream_name)
    return identity


def derive_search_seed(**context: Any) -> tuple[int, str]:
    """Return the deterministic seed and identity digest for a context."""
    identity = search_seed_context(**context)
    context_hash = stable_hash(identity)
    return stable_seed(identity), context_hash


def seed_identity_ledger_record(**context: Any) -> dict[str, Any]:
    """Return a persisted seed identity, independent of v2 treatment."""
    identity = search_seed_context(**context)
    derived_seed, context_hash = derive_search_seed(**context)
    return {
        **identity,
        "seed_context_hash": context_hash,
        "derived_search_seed": derived_seed,
    }


def search_configuration_ledger_record(
    *,
    seed_context_hash: str,
    simulations: int,
    effective_c_puct: float,
    tactical_root_bias: float,
    runtime_profile_hash: str,
    budget_pair: str,
    artifact_hash: str,
) -> dict[str, Any]:
    """Record treatment and model provenance separately from the seed identity."""
    record = {
        "seed_context_hash": str(seed_context_hash),
        "simulations": int(simulations),
        "effective_c_puct": float(effective_c_puct),
        "tactical_root_bias": float(tactical_root_bias),
        "runtime_profile_hash": str(runtime_profile_hash),
        "budget_pair": str(budget_pair),
        "artifact_hash": str(artifact_hash),
    }
    return {**record, "search_configuration_hash": stable_hash(record)}


def ledger_sha256(records: list[dict[str, Any]]) -> str:
    """Hash an already canonically ordered ledger."""
    return stable_hash(records)


def verify_provenance_ledgers(
    *,
    seed_identity_ledger: list[dict[str, Any]],
    search_configuration_ledger: list[dict[str, Any]],
    search_outcome_ledger: list[dict[str, Any]],
    seed_identity_ledger_sha256: str | None = None,
    search_configuration_ledger_sha256: str | None = None,
    search_outcome_ledger_sha256: str | None = None,
) -> dict[str, str]:
    """Recompute and validate all ledger hashes and per-record identities."""
    versions = {row.get("contract_version") for row in seed_identity_ledger}
    if len(versions) > 1:
        raise ValueError("cannot combine v1 and v2 seed manifests")
    for row in seed_identity_ledger:
        identity = {
            key: row[key]
            for key in (
                V1_SEED_IDENTITY_FIELDS
                if row["contract_version"] == SEED_CONTRACT_VERSION_V1
                else SEED_IDENTITY_FIELDS
            )
        }
        if stable_hash(identity) != row.get("seed_context_hash"):
            raise ValueError("altered seed identity ledger record")
        if stable_seed(identity) != row.get("derived_search_seed"):
            raise ValueError("altered derived search seed")
    for row in search_configuration_ledger:
        record = {
            key: value
            for key, value in row.items()
            if key != "search_configuration_hash"
        }
        if stable_hash(record) != row.get("search_configuration_hash"):
            raise ValueError("altered search configuration ledger record")
    hashes = {
        "seed_identity_ledger_sha256": ledger_sha256(seed_identity_ledger),
        "search_configuration_ledger_sha256": ledger_sha256(
            search_configuration_ledger
        ),
        "search_outcome_ledger_sha256": ledger_sha256(search_outcome_ledger),
    }
    for key, expected in (
        ("seed_identity_ledger_sha256", seed_identity_ledger_sha256),
        ("search_configuration_ledger_sha256", search_configuration_ledger_sha256),
        ("search_outcome_ledger_sha256", search_outcome_ledger_sha256),
    ):
        if expected is not None and hashes[key] != expected:
            raise ValueError(f"altered {key}")
    return hashes
