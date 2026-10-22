"""Synchronous HTTP adapter for the supervised move-serving engine."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ml.alphazero_lite.serving.contract import (
    MoveRequest,
    MoveResponse,
    validate_response,
    validate_state,
)
from ml.alphazero_lite.serving.engine import (
    InvalidBackendResultError,
    RuntimeBundle,
    SearchProfile,
    UnknownProfileError,
)
from ml.alphazero_lite.serving.workers import (
    CapacityExhaustedError,
    ExecutionTimeoutError,
    SearchCancelledError,
    SearchWorkers,
    StartupFailureError,
    WorkerCrashedError,
    WorkerSubmissionError,
)

LOGGER = logging.getLogger("mancala.ai.http")


@dataclass(frozen=True)
class ServiceConfig:
    """Trusted deployment configuration; never populated from requests."""

    asset_root: str
    bundles: Mapping[str, RuntimeBundle]
    profiles: Mapping[str, SearchProfile]
    worker_count: int = 2
    startup_timeout: float = 60.0
    execution_timeout: float = 10.0
    shutdown_timeout: float = 10.0
    duplicate_max_entries: int = 1024
    duplicate_retention_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.asset_root or not self.profiles:
            raise ValueError(
                "asset_root and at least one allowlisted profile are required"
            )
        if type(self.worker_count) is not int or self.worker_count < 1:
            raise ValueError("worker_count must be positive")
        if (
            min(self.startup_timeout, self.execution_timeout, self.shutdown_timeout)
            <= 0
        ):
            raise ValueError("worker deadlines must be positive")
        if self.duplicate_max_entries < 1 or self.duplicate_retention_seconds <= 0:
            raise ValueError("duplicate retention bounds must be positive")


@dataclass
class _Entry:
    fingerprint: str
    task: asyncio.Task
    callers: int
    completed_at: float | None = None
    response: dict | None = None


def create_app(
    config: ServiceConfig,
    *,
    worker_factory: Callable[..., Any] = SearchWorkers,
) -> FastAPI:
    """Create an application with injectable workers and lifespan ownership."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        workers = await asyncio.to_thread(
            worker_factory,
            config.asset_root,
            config.bundles,
            config.profiles,
            worker_count=config.worker_count,
            startup_timeout=config.startup_timeout,
            execution_timeout=config.execution_timeout,
            shutdown_timeout=config.shutdown_timeout,
        )
        app.state.workers = workers
        app.state.accepting = True
        app.state.entries = OrderedDict()
        try:
            yield
        finally:
            app.state.accepting = False
            await asyncio.to_thread(workers.shutdown, graceful=True)

    app = FastAPI(title="Mancala AI Move Service", version="1.0.0", lifespan=lifespan)

    async def error(
        status: int,
        code: str,
        message: str,
        request_id: str | None,
        retryable: bool = False,
    ):
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": request_id,
                    "retryable": retryable,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return await error(
            400,
            "invalid_request",
            "Request must be valid JSON matching the v1 schema.",
            None,
        )

    @app.get("/health/live", response_model=None)
    async def live():
        return {"status": "live"}

    @app.get(
        "/health/ready",
        response_model=None,
        responses={503: {"description": "Workers unavailable"}},
    )
    async def ready(request: Request):
        workers = getattr(request.app.state, "workers", None)
        if not request.app.state.accepting or workers is None:
            return await error(
                503,
                "workers_unavailable",
                "Search workers are unavailable.",
                None,
                True,
            )
        status = workers.status
        if status.accepting and any(
            slot.state in {"ready", "busy"} for slot in status.slots
        ):
            return {"status": "ready"}
        return await error(
            503, "workers_unavailable", "Search workers are unavailable.", None, True
        )

    async def execute(workers: Any, request: MoveRequest) -> dict:
        future = workers.submit(request)
        try:
            result = await workers.await_result(future)
        except asyncio.CancelledError:
            if asyncio.current_task() and asyncio.current_task().cancelling():
                future.cancel()
            raise
        return result

    def prune(entries: OrderedDict, now: float) -> None:
        for key, entry in list(entries.items()):
            if (
                entry.completed_at is not None
                and now - entry.completed_at > config.duplicate_retention_seconds
            ):
                del entries[key]
        completed = [
            (key, item)
            for key, item in entries.items()
            if item.completed_at is not None
        ]
        while len(completed) > config.duplicate_max_entries:
            key, _ = completed.pop(0)
            del entries[key]

    @app.post(
        "/v1/move",
        response_model=MoveResponse,
        responses={
            400: {"description": "Malformed request"},
            409: {"description": "Request ID conflict"},
            422: {"description": "Invalid state or unknown profile"},
            429: {"description": "Capacity exhausted"},
            503: {"description": "Workers unavailable"},
            504: {"description": "Deadline exceeded"},
            500: {"description": "Internal error"},
        },
    )
    async def move(request: Request):
        started = time.perf_counter()
        req_id = None
        profile = None
        outcome = "internal_error"
        try:
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    candidate = payload.get("request_id")
                    if isinstance(candidate, str) and re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}", candidate
                    ):
                        req_id = candidate
                req = MoveRequest.model_validate(payload)
            except (ValueError, ValidationError, json.JSONDecodeError):
                outcome = "invalid_request"
                return await error(
                    400,
                    "invalid_request",
                    "Request must be valid JSON matching the v1 schema.",
                    req_id,
                )
            try:
                validate_state(req.state)
            except ValueError:
                outcome = "invalid_state"
                return await error(
                    422, "invalid_state", "The game state is not valid.", req.request_id
                )
            profile = req.profile
            if profile not in config.profiles:
                outcome = "unknown_profile"
                return await error(
                    422,
                    "unknown_profile",
                    "The requested profile is not available.",
                    req.request_id,
                )
            workers = request.app.state.workers
            if not request.app.state.accepting or not workers.status.accepting:
                outcome = "workers_unavailable"
                return await error(
                    503,
                    "workers_unavailable",
                    "Search workers are stopping.",
                    req.request_id,
                    True,
                )
            fingerprint = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
            entries: OrderedDict = request.app.state.entries
            now = time.monotonic()
            prune(entries, now)
            entry = entries.get(req.request_id)
            if entry is not None and entry.fingerprint != fingerprint:
                outcome = "request_id_conflict"
                return await error(
                    409,
                    "request_id_conflict",
                    "request_id was already used with a different payload.",
                    req.request_id,
                )
            if entry is None:
                task = asyncio.create_task(execute(workers, req))
                entry = _Entry(fingerprint, task, 0)
                entries[req.request_id] = entry
            entry.callers += 1
            try:
                if entry.response is not None:
                    result = entry.response
                else:
                    result = await asyncio.shield(entry.task)
                    entry.response = result
                    entry.completed_at = time.monotonic()
                    prune(entries, entry.completed_at)
                response = validate_response(req, result, search_profile_id=None)
                outcome = "success"
                return JSONResponse(
                    status_code=200, content=response.model_dump(mode="json")
                )
            finally:
                entry.callers -= 1
                if entry.callers == 0 and not entry.task.done():
                    entry.task.cancel()
                if (
                    entry.callers == 0
                    and entry.task.done()
                    and entry.completed_at is None
                ):
                    entry.completed_at = time.monotonic()
                    prune(entries, entry.completed_at)
        except CapacityExhaustedError:
            status = request.app.state.workers.status
            if (
                any(slot.state in {"ready", "busy"} for slot in status.slots)
                and status.in_flight >= status.capacity
            ):
                outcome = "capacity_exhausted"
                return await error(
                    429,
                    "capacity_exhausted",
                    "Search capacity is temporarily exhausted.",
                    req_id,
                    True,
                )
            outcome = "workers_unavailable"
            return await error(
                503,
                "workers_unavailable",
                "No initialized search worker is available.",
                req_id,
                True,
            )
        except ExecutionTimeoutError:
            outcome = "deadline_exceeded"
            return await error(
                504, "deadline_exceeded", "Search deadline exceeded.", req_id, True
            )
        except (
            StartupFailureError,
            WorkerCrashedError,
            WorkerSubmissionError,
            SearchCancelledError,
        ):
            outcome = "workers_unavailable"
            return await error(
                503,
                "workers_unavailable",
                "Search workers are unavailable.",
                req_id,
                True,
            )
        except UnknownProfileError:
            outcome = "unknown_profile"
            return await error(
                422,
                "unknown_profile",
                "The requested profile is not available.",
                req_id,
            )
        except (InvalidBackendResultError, ValueError):
            outcome = "internal_error"
            return await error(
                500,
                "internal_error",
                "The search service returned an invalid response.",
                req_id,
            )
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                outcome = "cancelled"
                raise
            outcome = "internal_error"
            return await error(
                500,
                "internal_error",
                "An unexpected application error occurred.",
                req_id,
            )
        except BaseException:
            outcome = "internal_error"
            return await error(
                500,
                "internal_error",
                "An unexpected application error occurred.",
                req_id,
            )
        finally:
            LOGGER.info(
                "move_request",
                extra={
                    "request_id": req_id,
                    "profile": profile,
                    "outcome": outcome,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )

    # The checked-in v1 document is the authority; avoid subtly divergent
    # generated schemas for this deliberately strict decoded-JSON boundary.
    def contract_openapi() -> dict:
        path = Path(__file__).parents[3] / "docs/ai-service/api-v1.json"
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    app.openapi = contract_openapi
    return app
