"""Bounded Pebble supervision for persistent MoveEngine worker processes.

Each advertised slot owns a one-process Pebble pool. A replacement is not
usable until a generation-token handshake has run after its initializer.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, Mapping

from pebble import ProcessExpired, ProcessPool

from ml.alphazero_lite.serving.contract import MoveRequest
from ml.alphazero_lite.serving.engine import MoveEngine, RuntimeBundle, SearchProfile

_ENGINE: MoveEngine | None = None


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return _process_state(pid) != "Z"
    except (FileNotFoundError, ProcessLookupError):
        # The process can exit after kill(pid, 0) but before procfs is read.
        return False


def _process_state(pid: int) -> str:
    """Read procfs state separately so disappearance races are testable."""
    return open(f"/proc/{pid}/stat", encoding="ascii").read().split()[2]


class WorkerError(RuntimeError):
    """Base class for supervised execution failures."""


class CapacityExhaustedError(WorkerError):
    """No initialized idle worker can accept a request immediately."""


class StartupFailureError(WorkerError):
    """Worker initialization did not finish by the startup deadline."""


class ExecutionTimeoutError(WorkerError):
    """Pebble terminated a search at its per-task execution deadline."""


class WorkerCrashedError(WorkerError):
    """The worker died unexpectedly during a search."""


class WorkerSubmissionError(WorkerError):
    """Pebble rejected a task submission to an initialized worker."""


class SearchCancelledError(concurrent.futures.CancelledError):
    """A running search was terminated through Pebble cancellation."""


@dataclass(frozen=True)
class WorkerInfo:
    slot: int
    generation: int
    state: str
    pid: int | None
    error: str | None


@dataclass(frozen=True)
class WorkerStatus:
    ready: bool
    accepting: bool
    workers: int
    in_flight: int
    capacity: int
    started: bool
    slots: tuple[WorkerInfo, ...]


def _initialize_engine(
    asset_root: str,
    bundles: Mapping[str, RuntimeBundle],
    profiles: Mapping[str, SearchProfile],
) -> None:
    global _ENGINE
    _ENGINE = MoveEngine(asset_root, bundles, profiles)


def _search(snapshot: dict[str, Any]) -> dict[str, Any]:
    if _ENGINE is None:
        raise RuntimeError("worker engine was not initialized")
    return _ENGINE.select_move(snapshot).model_dump(mode="json")


def _handshake(nonce: str) -> tuple[str, int]:
    if _ENGINE is None:
        raise RuntimeError("worker engine was not initialized")
    return nonce, os.getpid()


def _invoke(function: Callable[[Any], Any], snapshot: Any) -> tuple[str, Any]:
    """Keep application exceptions separate from Pebble transport failures."""
    try:
        return "success", function(snapshot)
    except BaseException as error:
        return "application-error", error


class _SearchFuture(concurrent.futures.Future):
    """Future whose cancel operation requests worker termination via callback."""

    def __init__(
        self,
        cancel_running: Callable[[], bool],
        running: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._cancel_running = cancel_running
        self._running = running
        self.was_running = False

    def cancel(self) -> bool:
        if self.done():
            return False
        self.was_running = self._running()
        # Pebble owns cancellation semantics for running work. Its completion
        # callback will set the typed SearchCancelledError on this future.
        return self._cancel_running()


@dataclass
class _Slot:
    index: int
    generation: int = 0
    state: str = "starting"
    pid: int | None = None
    error: str | None = None
    pool: ProcessPool | None = None
    busy_future: concurrent.futures.Future | None = None
    public_future: _SearchFuture | None = None
    startup_thread: threading.Thread | None = None
    restart_thread: threading.Thread | None = None


class SearchWorkers:
    """A fixed-capacity set of initialized worker processes, without a job queue."""

    def __init__(
        self,
        asset_root: str,
        bundles: Mapping[str, RuntimeBundle],
        profiles: Mapping[str, SearchProfile],
        *,
        worker_count: int,
        start_method: str = "spawn",
        execution_timeout: float = 10.0,
        startup_timeout: float = 60.0,
        shutdown_timeout: float = 10.0,
        _initializer: Callable[..., None] | None = None,
        _initargs: tuple[Any, ...] = (),
        _task_function: Callable[..., Any] | None = None,
        _ready_function: Callable[[str], tuple[str, int]] | None = None,
    ) -> None:
        if sys.platform != "linux":
            raise RuntimeError(
                "supervised native-child cleanup requires Linux PR_SET_PDEATHSIG"
            )
        if type(worker_count) is not int or worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        if min(execution_timeout, startup_timeout, shutdown_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        self.worker_count = worker_count
        self.execution_timeout = execution_timeout
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self._context = get_context(start_method)
        self._initializer = _initializer or _initialize_engine
        self._initargs = _initargs or (str(asset_root), dict(bundles), dict(profiles))
        self._task_function = _task_function or _search
        self._ready_function = _ready_function or _handshake
        self._lock = threading.RLock()
        self._accepting = True
        self._started = True
        self._stopping = False
        self._slots = [_Slot(index) for index in range(worker_count)]
        self._startup_failed = False
        for slot in self._slots:
            self._launch(slot)
        self._monitor_thread = threading.Thread(
            target=self._monitor_idle_workers, daemon=True
        )
        self._monitor_thread.start()

    def _new_pool(self) -> ProcessPool:
        return ProcessPool(
            max_workers=1,
            initializer=self._initializer,
            initargs=self._initargs,
            context=self._context,
        )

    def _launch(self, slot: _Slot) -> None:
        with self._lock:
            if self._stopping:
                return
            slot.generation += 1
            generation = slot.generation
            token = secrets.token_hex(24)
            slot.state = "starting"
            slot.pid = None
            slot.error = None
            slot.pool = self._new_pool()
            pool = slot.pool
            ready_future = pool.schedule(
                self._ready_function, args=(token,), timeout=self.startup_timeout
            )
            thread = threading.Thread(
                target=self._confirm_ready,
                args=(slot, generation, token, pool, ready_future),
                daemon=True,
            )
            slot.startup_thread = thread
            thread.start()

    def _confirm_ready(
        self,
        slot: _Slot,
        generation: int,
        token: str,
        pool: ProcessPool,
        future: concurrent.futures.Future,
        previous_pid: int | None = None,
        require_new_pid: bool = False,
    ) -> None:
        try:
            returned_token, pid = future.result(timeout=self.startup_timeout)
            if returned_token != token or type(pid) is not int or pid <= 1:
                raise StartupFailureError("invalid worker readiness handshake")
            if require_new_pid and pid == previous_pid:
                raise StartupFailureError(
                    "failed worker PID was reused without replacement"
                )
            with self._lock:
                if (
                    self._stopping
                    or slot.generation != generation
                    or slot.pool is not pool
                ):
                    return
                slot.pid = pid
                slot.state = "ready"
                slot.error = None
        except BaseException as error:
            try:
                pool.stop()
                pool.join(timeout=self.shutdown_timeout)
            except BaseException:
                pass
            with self._lock:
                if slot.generation != generation or self._stopping:
                    return
                slot.state = "unavailable"
                slot.pid = None
                slot.error = f"{type(error).__name__}: {error}"
                self._startup_failed = True

    @property
    def status(self) -> WorkerStatus:
        with self._lock:
            slots = tuple(
                WorkerInfo(s.index, s.generation, s.state, s.pid, s.error)
                for s in self._slots
            )
            ready = sum(s.state == "ready" for s in self._slots)
            return WorkerStatus(
                ready=all(s.state in {"ready", "busy"} for s in self._slots)
                and not self._stopping,
                accepting=self._accepting and not self._stopping,
                workers=ready,
                in_flight=sum(s.state == "busy" for s in self._slots),
                capacity=self.worker_count,
                started=self._started,
                slots=slots,
            )

    def submit(
        self, request: MoveRequest | Mapping[str, Any]
    ) -> concurrent.futures.Future:
        """Validate, snapshot, and submit without waiting for a worker slot."""
        req = MoveRequest.model_validate(
            request.model_dump(mode="python")
            if isinstance(request, MoveRequest)
            else dict(request)
        )
        snapshot = req.model_dump(mode="python")
        with self._lock:
            if not self._accepting or self._stopping:
                raise CapacityExhaustedError("search workers are shutting down")
            slot = None
            for candidate in self._slots:
                if candidate.state != "ready":
                    continue
                if candidate.pid is None or _process_alive(candidate.pid):
                    slot = candidate
                    break
                generation = candidate.generation
                previous_pid = candidate.pid
                candidate.state = "unavailable"
                candidate.pid = None
                candidate.error = "idle worker exited unexpectedly"
                self._recover(candidate, generation, previous_pid, require_new_pid=True)
            if slot is None:
                if any(s.state in {"starting", "busy"} for s in self._slots):
                    raise CapacityExhaustedError(
                        "workers are initializing or processing requests"
                    )
                if self._startup_failed or any(
                    s.state == "unavailable" and s.error for s in self._slots
                ):
                    raise StartupFailureError("worker startup failed")
                raise CapacityExhaustedError("no initialized idle worker is available")
            slot.state = "busy"
            pool = slot.pool
            generation = slot.generation
            assert pool is not None
            try:
                execution = pool.schedule(
                    _invoke,
                    args=(self._task_function, snapshot),
                    timeout=self.execution_timeout,
                )
            except Exception as error:
                previous_pid = slot.pid
                slot.state = "unavailable"
                slot.pid = None
                slot.error = "submission failed"
                self._recover(slot, generation, previous_pid, require_new_pid=True)
                raise WorkerSubmissionError(
                    f"worker task submission failed: {type(error).__name__}: {error}"
                ) from error
            public = _SearchFuture(execution.cancel, execution.running)
            slot.busy_future = execution
            slot.public_future = public
            execution.add_done_callback(
                lambda completed, target=slot, gen=generation, result=public: (
                    self._completed(target, gen, completed, result)
                )
            )
            return public

    def _completed(
        self,
        slot: _Slot,
        generation: int,
        completed: concurrent.futures.Future,
        public: _SearchFuture,
    ) -> None:
        failure: BaseException | None = None
        result: Any = None
        outcome: str | None = None
        try:
            outcome, result = completed.result()
            if outcome == "application-error":
                failure = result
        except BaseException as error:
            failure = error
        # Only exceptions raised by Pebble's future represent transport. An
        # exception returned inside the _invoke envelope belongs to the app,
        # even when its type is TimeoutError or CancelledError.
        transport_failure = outcome != "application-error" and isinstance(
            failure,
            (
                concurrent.futures.TimeoutError,
                concurrent.futures.CancelledError,
                ProcessExpired,
                BrokenProcessPool,
            ),
        )
        with self._lock:
            if slot.generation != generation:
                return
            previous_pid = slot.pid
            slot.busy_future = None
            slot.public_future = None
            shutting_down = self._stopping
            if transport_failure and not shutting_down:
                slot.state = "unavailable"
                slot.pid = None
                slot.error = f"{type(failure).__name__}: {failure}"
            elif not shutting_down:
                slot.state = "ready"
        if transport_failure:
            if shutting_down:
                mapped: BaseException = SearchCancelledError(
                    "worker pool shut down during search"
                )
            elif isinstance(failure, concurrent.futures.TimeoutError):
                mapped: BaseException = ExecutionTimeoutError(
                    "search deadline exceeded"
                )
            elif isinstance(failure, concurrent.futures.CancelledError):
                mapped = SearchCancelledError("running search was cancelled")
            else:
                mapped = WorkerCrashedError("search worker exited unexpectedly")
            if not shutting_down:
                self._recover(
                    slot,
                    generation,
                    previous_pid,
                    require_new_pid=(
                        not isinstance(failure, concurrent.futures.CancelledError)
                        or public.was_running
                    ),
                )
            if not public.done():
                public.set_exception(mapped)
        elif failure is not None:
            if not public.done():
                public.set_exception(failure)
        elif not public.done():
            public.set_result(result)

    def _recover(
        self,
        slot: _Slot,
        generation: int,
        previous_pid: int | None,
        *,
        require_new_pid: bool,
    ) -> None:
        with self._lock:
            if self._stopping or slot.generation != generation:
                return
            if slot.restart_thread is not None and slot.restart_thread.is_alive():
                return
            pool = slot.pool
            if pool is None:
                slot.state = "unavailable"
                slot.error = "worker pool is missing"
                self._startup_failed = True
                return
            slot.generation += 1
            replacement_generation = slot.generation
            slot.state = "starting"
            slot.pid = None
            slot.error = None
            slot.pool = None
            thread = threading.Thread(
                target=self._replace_pool,
                args=(
                    slot,
                    replacement_generation,
                    pool,
                    previous_pid,
                    require_new_pid,
                ),
                daemon=True,
            )
            slot.restart_thread = thread
            thread.start()

    def _replace_pool(
        self,
        slot: _Slot,
        generation: int,
        old_pool: ProcessPool,
        previous_pid: int | None,
        require_new_pid: bool,
    ) -> None:
        try:
            old_pool.stop()
            old_pool.join(timeout=self.shutdown_timeout)
            with self._lock:
                if self._stopping or slot.generation != generation:
                    return
                pool = self._new_pool()
                slot.pool = pool
            token = secrets.token_hex(24)
            handshake = pool.schedule(
                self._ready_function, args=(token,), timeout=self.startup_timeout
            )
            self._confirm_ready(
                slot, generation, token, pool, handshake, previous_pid, require_new_pid
            )
        except BaseException as error:
            with self._lock:
                if slot.generation == generation and not self._stopping:
                    slot.pool = None
                    slot.state = "unavailable"
                    slot.pid = None
                    slot.error = f"replacement failed: {type(error).__name__}: {error}"
                    self._startup_failed = True

    def _monitor_idle_workers(self) -> None:
        """Detect process loss that has no task completion callback to observe."""
        interval = min(0.25, max(0.05, self.execution_timeout / 4))
        while True:
            time.sleep(interval)
            with self._lock:
                if self._stopping:
                    return
                lost = [
                    (slot, slot.generation, slot.pid)
                    for slot in self._slots
                    if slot.state == "ready"
                    and slot.pid is not None
                    and not _process_alive(slot.pid)
                ]
            for slot, generation, pid in lost:
                with self._lock:
                    if slot.generation != generation or slot.state != "ready":
                        continue
                    slot.state = "unavailable"
                    slot.pid = None
                    slot.error = "idle worker exited unexpectedly"
                self._recover(slot, generation, pid, require_new_pid=True)

    async def await_result(self, future: concurrent.futures.Future) -> dict[str, Any]:
        """Await a result without blocking an asyncio event loop."""
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def deliver(source: concurrent.futures.Future) -> None:
            try:
                outcome = (True, source.result())
            except BaseException as error:
                # Put the exception in a value. In particular, an application
                # CancelledError must not cancel the coroutine awaiting it.
                outcome = (False, error)

            def resolve() -> None:
                if not completed.done():
                    completed.set_result(outcome)

            loop.call_soon_threadsafe(resolve)

        future.add_done_callback(deliver)
        succeeded, result = await completed
        if not succeeded:
            raise result
        return result

    def shutdown(self, *, graceful: bool = True) -> None:
        """Stop admission and drain boundedly, then terminate remaining workers."""
        with self._lock:
            if self._stopping:
                return
            self._accepting = False
            self._stopping = True
            self._started = True
            pools = [slot.pool for slot in self._slots if slot.pool is not None]
            pending_futures = [
                slot.public_future
                for slot in self._slots
                if slot.public_future is not None
            ]
            threads = [slot.startup_thread for slot in self._slots]
            threads += [slot.restart_thread for slot in self._slots]
            for slot in self._slots:
                slot.state = "unavailable"
        deadline = time.monotonic() + self.shutdown_timeout
        if graceful:
            for pool in pools:
                pool.close()
            try:
                for pool in pools:
                    pool.join(timeout=max(0.0, deadline - time.monotonic()))
            except concurrent.futures.TimeoutError:
                graceful = False
        if not graceful:
            for pool in pools:
                pool.stop()
            for pool in pools:
                try:
                    pool.join(timeout=max(0.0, deadline - time.monotonic()))
                except concurrent.futures.TimeoutError:
                    pass
            for future in pending_futures:
                if future is not None and not future.done():
                    try:
                        future.set_exception(
                            SearchCancelledError("worker pool shut down during search")
                        )
                    except concurrent.futures.InvalidStateError:
                        pass
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def __enter__(self) -> SearchWorkers:
        return self

    def __exit__(self, *_args: object) -> None:
        self.shutdown()
