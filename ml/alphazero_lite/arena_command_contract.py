"""Normalize ``arena.py`` invocations into evaluator contracts.

The gate intentionally leaves most arena options implicit.  This module makes
those inherited defaults explicit so two commands can be compared by meaning.
"""

from __future__ import annotations

from typing import Callable

from ml.alphazero_lite.evaluation_seed_contract import SEED_CONTRACT_VERSION


OUTPUT_FLAGS = frozenset(
    {
        "--out",
        "--game-jsonl",
        "--seed-ledger-output",
        "--search-configuration-ledger-output",
        "--search-outcome-ledger-output",
    }
)
BOOLEAN_FLAGS = frozenset(
    {
        "--reuse-subtree",
        "--normalize-values",
        "--value-trust-enabled",
        "--challenger-blend-current",
    }
)
DEFAULTS: dict[str, object] = {
    "--games": 60,
    "--challenger-simulations": 384,
    "--current-simulations": 256,
    "--seed": 42,
    "--seed-contract": SEED_CONTRACT_VERSION,
    "--c-puct": 1.25,
    "--min-score": 0.55,
    "--max-moves": 200,
    "--workers": 1,
    "--random-opening-plies": 0,
    "--opening-seed": None,
    "--opening-samples": 0,
    "--opening-plies": None,
    "--opening-prefixes-jsonl": None,
    "--games-per-opening": 2,
    "--challenger-starts": None,
    "--fpu-mode": "zero",
    "--reuse-subtree": False,
    "--normalize-values": False,
    "--root-policy-mode": "deterministic",
    "--tactical-root-bias": 0.0,
    "--value-trust-enabled": False,
    "--value-trust-opening": 1.0,
    "--value-trust-midgame": 1.0,
    "--value-trust-late": 1.0,
    "--value-transform-json": None,
    "--root-prior-transform": None,
    "--challenger-root-prior-transform": None,
    "--current-root-prior-transform": None,
    "--challenger-prior-override-mode": None,
    "--challenger-value-transform-json": None,
    "--current-value-transform-json": None,
    "--challenger-search-options-json": None,
    "--current-search-options-json": None,
    "--challenger-c-puct": None,
    "--current-c-puct": None,
    "--challenger-runtime-profile-hash": None,
    "--current-runtime-profile-hash": None,
    "--challenger-shadow-artifact": None,
    "--challenger-shadow-q-weight": 1.0,
    "--challenger-blend-current": False,
    "--challenger-value-alpha": 1.0,
    "--opening-cache": None,
    "--opening-cache-training-summary": None,
}


def _value(value: str, default: object) -> object:
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def normalize_arena_command(
    command: list[str], *, artifact_identity: Callable[[str], str] | None = None
) -> dict[str, object]:
    """Return all evaluator inputs, discarding only destination paths.

    Unknown flags are retained rather than silently ignored, making additions to
    ``arena.py`` visible in equivalence checks until this contract is updated.
    """
    values = dict(DEFAULTS)
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        if token in OUTPUT_FLAGS:
            index += 2
            continue
        if token in BOOLEAN_FLAGS:
            values[token] = True
            index += 1
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise ValueError(f"arena command option lacks value: {token}")
        default = values.get(token)
        values[token] = (
            command[index + 1]
            if default is None
            else _value(command[index + 1], default)
        )
        index += 2

    for required in ("--challenger", "--current"):
        if required not in values:
            raise ValueError(f"arena command lacks required option: {required}")
    if artifact_identity is not None:
        values["--challenger"] = artifact_identity(str(values["--challenger"]))
        values["--current"] = artifact_identity(str(values["--current"]))
    return {
        key.removeprefix("--").replace("-", "_"): values[key] for key in sorted(values)
    }


def contracts_equal(left: dict[str, object], right: dict[str, object]) -> bool:
    """Compare normalized evaluator inputs without path/output exceptions."""
    return left == right
