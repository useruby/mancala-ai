"""Strict API models, state conversion, canonical hashing, and response checks."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from ml.alphazero_lite.kalah_rules import KalahGame

MAX_SAFE_INTEGER = 2**53 - 1
Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@-]*$")
]
SeedCount = Annotated[StrictInt, Field(ge=0, le=48)]
Revision = Annotated[StrictInt, Field(ge=0, le=MAX_SAFE_INTEGER)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class State(StrictModel):
    player0_pits: Annotated[list[SeedCount], Field(min_length=6, max_length=6)]
    player1_pits: Annotated[list[SeedCount], Field(min_length=6, max_length=6)]
    player0_store: SeedCount
    player1_store: SeedCount
    current_player: Annotated[StrictInt, Field(ge=0, le=1)]


class MoveRequest(StrictModel):
    request_id: Identifier
    game_revision: Revision
    state: State
    profile: Identifier
    seed: Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]


class MoveResult(StrictModel):
    kind: Literal["move"]
    move: Annotated[StrictInt, Field(ge=0, le=5)]


class TerminalResult(StrictModel):
    kind: Literal["terminal"]
    move: None


class MoveResponse(StrictModel):
    request_id: Identifier
    game_revision: Revision
    state_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    result: MoveResult | TerminalResult
    model_bundle_id: Identifier
    search_profile_id: Identifier
    elapsed_ms: Revision
    completed_simulations: Revision
    termination_reason: Literal["completed", "terminal"]


class ErrorBody(StrictModel):
    code: Literal[
        "invalid_request",
        "invalid_state",
        "unknown_profile",
        "request_id_conflict",
        "capacity_exhausted",
        "workers_unavailable",
        "deadline_exceeded",
        "internal_error",
    ]
    message: str
    request_id: Identifier | None
    retryable: bool


class Error(StrictModel):
    error: ErrorBody


def validate_state(value: State | dict) -> State:
    """Parse state and enforce seed conservation and canonical terminal form."""
    # model_construct and post-construction mutation are both possible; always
    # cross the decoded-data boundary again, including all nested fields.
    state = State.model_validate(
        value.model_dump(mode="python") if isinstance(value, State) else value
    )
    total = (
        sum(state.player0_pits + state.player1_pits)
        + state.player0_store
        + state.player1_store
    )
    if total != 48:
        raise ValueError("state must conserve exactly 48 seeds")
    terminal = not any(state.player0_pits + state.player1_pits)
    if terminal:
        if state.player0_store + state.player1_store != 48:
            raise ValueError("terminal state must have settled stores")
    elif not any((state.player0_pits, state.player1_pits)[state.current_player]):
        raise ValueError("nonterminal state must have a legal pit for current player")
    return state


def canonical_state_bytes(value: State | dict) -> bytes:
    state = validate_state(value)
    obj = {
        key: getattr(state, key)
        for key in (
            "player0_pits",
            "player1_pits",
            "player0_store",
            "player1_store",
            "current_player",
        )
    }
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def state_sha256(value: State | dict) -> str:
    return hashlib.sha256(canonical_state_bytes(value)).hexdigest()


def to_kalah_game(value: State | dict) -> KalahGame:
    state = validate_state(value)
    return KalahGame(
        pits=list(state.player0_pits) + list(state.player1_pits),
        captured_seeds=[state.player0_store, state.player1_store],
        current_player=state.current_player,
    )


def from_kalah_game(game: KalahGame) -> State:
    """Convert the game's fixed ownership representation without sharing arrays."""
    return validate_state(
        {
            "player0_pits": list(game.pits[:6]),
            "player1_pits": list(game.pits[6:]),
            "player0_store": game.captured_seeds[0],
            "player1_store": game.captured_seeds[1],
            "current_player": game.current_player,
        }
    )


def validate_response(
    request: MoveRequest | dict,
    response: MoveResponse | dict,
    *,
    model_bundle_id: str | None = None,
    search_profile_id: str | None = None,
) -> MoveResponse:
    """Validate contextual binding, terminal consistency, and actor-relative legality.

    The engine supplies the resolved immutable model/profile identities for comparison.
    """
    req = MoveRequest.model_validate(
        request.model_dump(mode="python")
        if isinstance(request, MoveRequest)
        else request
    )
    validate_state(req.state)
    res = MoveResponse.model_validate(
        response.model_dump(mode="python")
        if isinstance(response, MoveResponse)
        else response
    )
    if (res.request_id, res.game_revision) != (req.request_id, req.game_revision):
        raise ValueError("response request identity does not match request")
    if res.state_sha256 != state_sha256(req.state):
        raise ValueError("response state digest does not match request")
    if model_bundle_id is not None and res.model_bundle_id != model_bundle_id:
        raise ValueError("resolved model identity mismatch")
    if search_profile_id is not None and res.search_profile_id != search_profile_id:
        raise ValueError("resolved profile identity mismatch")
    game = to_kalah_game(req.state)
    terminal = not game.possible_moves()
    if terminal:
        if (
            not isinstance(res.result, TerminalResult)
            or res.termination_reason != "terminal"
            or res.completed_simulations != 0
        ):
            raise ValueError(
                "terminal state requires terminal result and zero simulations"
            )
    elif (
        not isinstance(res.result, MoveResult)
        or res.termination_reason != "completed"
        or res.result.move not in game.possible_moves()
    ):
        raise ValueError("nonterminal state requires a legal actor-relative move")
    return res
