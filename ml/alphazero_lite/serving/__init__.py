"""Executable boundary for the AI move service v1 contract."""

from .contract import (
    Error,
    ErrorBody,
    MoveRequest,
    MoveResponse,
    MoveResult,
    State,
    TerminalResult,
    canonical_state_bytes,
    from_kalah_game,
    state_sha256,
    to_kalah_game,
    validate_response,
    validate_state,
)

__all__ = [
    "Error",
    "ErrorBody",
    "MoveRequest",
    "MoveResponse",
    "MoveResult",
    "State",
    "TerminalResult",
    "canonical_state_bytes",
    "from_kalah_game",
    "state_sha256",
    "to_kalah_game",
    "validate_response",
    "validate_state",
]
