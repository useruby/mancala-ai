"""Process-level lifecycle tests for serving worker supervision."""

from __future__ import annotations

import os
import signal
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from ml.alphazero_lite.serving.contract import MoveRequest
from ml.alphazero_lite.serving.engine import (
    RuntimeBundle,
    SearchProfile,
    SearchSettings,
    serving_artifact_identity,
)
from ml.alphazero_lite.serving.workers import (
    CapacityExhaustedError,
    ExecutionTimeoutError,
    SearchCancelledError,
    SearchWorkers,
    StartupFailureError,
    WorkerCrashedError,
)
from ml.alphazero_lite.serving import workers as worker_module

FIXTURE = Path(__file__).parent / "fixtures/incumbent_forensic_proxy_current"
_GATE = None
_STARTED = None
_INIT_COUNT = None
_CHILD_MARKER = None
_FAIL_REPLACEMENT = None


def _test_initializer(
    gate=None,
    started=None,
    count=None,
    child_marker=None,
    fail=None,
    worker_marker=None,
):
    global _GATE, _STARTED, _INIT_COUNT, _CHILD_MARKER, _FAIL_REPLACEMENT
    _GATE, _STARTED, _INIT_COUNT = gate, started, count
    _CHILD_MARKER, _FAIL_REPLACEMENT = child_marker, fail
    if worker_marker is not None:
        Path(worker_marker).write_text(str(os.getpid()))
    ordinal = 1
    if count is not None:
        with count.get_lock():
            count.value += 1
            ordinal = count.value
        if ordinal > 1 and fail is not None and fail.value:
            raise RuntimeError("controlled replacement initialization failure")
    if gate and (count is None or ordinal > 1):
        time.sleep(gate)


def _test_ready(nonce):
    return nonce, os.getpid()


def _test_task(payload):
    if isinstance(payload, dict):
        payload = payload["request_id"]
    if _STARTED is not None:
        _STARTED.set()
    if payload == "wait":
        time.sleep(30)
    if payload == "die":
        os._exit(77)
    if payload == "raise":
        raise ValueError("controlled application exception")
    if payload == "raise-timeout":
        raise TimeoutError("application timeout")
    if payload == "raise-cancelled":
        raise __import__("concurrent.futures").futures.CancelledError(
            "application cancellation"
        )
    if payload == "native-child":
        from ml.alphazero_lite.native_exact_root_tablebase import _spawn_parent_bound

        script = (
            "import os,time; open(%r,'w').write(str(os.getpid())); time.sleep(30)"
        ) % str(_CHILD_MARKER)
        child = _spawn_parent_bound(
            [sys.executable, "-c", script],
            test_pid_file=_CHILD_MARKER,
        )
        Path(_CHILD_MARKER).write_text(str(child.pid))
        time.sleep(30)
    return {"value": payload, "pid": os.getpid()}


def _loss_during_child_start(payload):
    from ml.alphazero_lite.native_exact_root_tablebase import _spawn_parent_bound

    _spawn_parent_bound(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        test_start_delay=0.7,
        test_pid_file=str(_CHILD_MARKER) + ".early",
    )
    time.sleep(30)


def _await(predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition was not met before deadline")


def _alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.read_text().split()[2] == "Z":
            return False
    except FileNotFoundError:
        return False
    except ProcessLookupError:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _request():
    return MoveRequest.model_validate(
        {
            "request_id": "lifecycle",
            "game_revision": 1,
            "state": {
                "player0_pits": [1, 2, 3, 4, 5, 6],
                "player1_pits": [6, 5, 4, 3, 2, 1],
                "player0_store": 4,
                "player1_store": 2,
                "current_player": 0,
            },
            "profile": "default",
            "seed": 1,
        }
    )


def _action(workers, action):
    request = _request().model_dump(mode="python")
    request["request_id"] = action
    return workers.submit(request)


@pytest.fixture
def scratch():
    root = Path(__file__).parents[2] / ".tmp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="serving-lifecycle-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _fake_workers(
    ctx, *, initargs=(), task=_test_task, startup=4.0, execution=0.4, worker_count=1
):
    return SearchWorkers(
        ".",
        {},
        {},
        worker_count=worker_count,
        start_method="spawn",
        execution_timeout=execution,
        startup_timeout=startup,
        shutdown_timeout=2,
        _initializer=_test_initializer,
        _initargs=initargs,
        _task_function=task,
        _ready_function=_test_ready,
    )


def _ready(workers):
    _await(lambda: workers.status.workers == 1)
    assert workers.status.slots[0].state == "ready"
    return workers.status.slots[0].pid


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_initialization_hang_hits_deadline_and_kills_worker(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "worker.pid"
    workers = _fake_workers(
        ctx,
        initargs=(30, None, None, None, None, str(marker)),
        startup=0.5,
    )
    try:
        pid = int(_await(lambda: marker.read_text() if marker.exists() else None))
        _await(lambda: workers.status.slots[0].state == "unavailable", timeout=4)
        assert "TimeoutError" in (workers.status.slots[0].error or "")
        with pytest.raises(StartupFailureError):
            workers.submit(_request())
        _await(lambda: not _alive(pid))
        assert workers.status.workers == 0
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_timeout_keeps_slot_unavailable_until_replacement_is_initialized():
    ctx = __import__("multiprocessing").get_context("spawn")
    count = ctx.Value("i", 0)
    workers = _fake_workers(ctx, initargs=(0.8, None, count, None, None))
    try:
        original_pid = _ready(workers)
        with pytest.raises(ExecutionTimeoutError):
            _action(workers, "wait").result(timeout=4)
        assert workers.status.workers == 0
        assert workers.status.slots[0].state in {"starting", "unavailable"}
        with pytest.raises(CapacityExhaustedError):
            _action(workers, "too-early")
        replacement = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                else None
            )
        )
        assert replacement != original_pid
        assert _action(workers, "ok").result(timeout=3)["value"] == "ok"
    finally:
        workers.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_running_cancellation_kills_and_recovers_worker():
    ctx = __import__("multiprocessing").get_context("spawn")
    started = ctx.Event()
    workers = _fake_workers(ctx, initargs=(None, started, None, None, None))
    try:
        old_pid = _ready(workers)
        future = _action(workers, "wait")
        assert started.wait(3)
        assert future.cancel()
        with pytest.raises(SearchCancelledError):
            future.result(timeout=4)
        new_pid = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                else None
            )
        )
        assert new_pid != old_pid
        assert _action(workers, "next").result(timeout=3)["value"] == "next"
    finally:
        workers.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_unexpected_worker_death_is_replaced():
    ctx = __import__("multiprocessing").get_context("spawn")
    workers = _fake_workers(ctx, initargs=(None, None, None, None, None))
    try:
        old_pid = _ready(workers)
        with pytest.raises(WorkerCrashedError):
            _action(workers, "die").result(timeout=4)
        new_pid = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                else None
            )
        )
        assert new_pid != old_pid
        assert _action(workers, "recovered").result(timeout=3)["value"] == "recovered"
    finally:
        workers.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_idle_worker_death_replaces_its_pool():
    ctx = __import__("multiprocessing").get_context("spawn")
    workers = _fake_workers(ctx, initargs=(None, None, None, None, None))
    try:
        old_pid = _ready(workers)
        old_generation = workers.status.slots[0].generation
        os.kill(old_pid, signal.SIGKILL)
        _await(lambda: workers.status.slots[0].generation > old_generation)
        new_pid = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                else None
            )
        )
        assert workers.status.slots[0].generation > old_generation
        assert _action(workers, "after-idle-loss").result(timeout=3)["pid"] == new_pid
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_procfs_disappearance_race_keeps_monitoring_and_recovers(monkeypatch):
    ctx = __import__("multiprocessing").get_context("spawn")
    workers = _fake_workers(ctx, initargs=(None, None, None, None, None))
    real_kill = os.kill
    real_state = worker_module._process_state
    old_pid = None
    kill_race_pending = True
    state_race_pending = True

    def kill(pid, signal_number):
        nonlocal kill_race_pending
        if pid == old_pid and kill_race_pending:
            kill_race_pending = False
            return None
        return real_kill(pid, signal_number)

    def process_state(pid):
        nonlocal state_race_pending
        if pid == old_pid and state_race_pending:
            state_race_pending = False
            raise ProcessLookupError(pid)
        return real_state(pid)

    monkeypatch.setattr(worker_module.os, "kill", kill)
    monkeypatch.setattr(worker_module, "_process_state", process_state)
    try:
        old_pid = _ready(workers)
        old_generation = workers.status.slots[0].generation
        real_kill(old_pid, signal.SIGKILL)
        new_pid = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                and workers.status.slots[0].generation > old_generation
                else None
            )
        )
        assert not kill_race_pending
        assert not state_race_pending
        assert new_pid != old_pid
        assert _action(workers, "after-procfs-race").result(timeout=3)["pid"] == new_pid
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_idle_pool_loss_preserves_other_ready_slot():
    ctx = __import__("multiprocessing").get_context("spawn")
    workers = _fake_workers(
        ctx, initargs=(None, None, None, None, None), worker_count=2
    )
    try:
        _await(lambda: workers.status.workers == 2)
        slots = workers.status.slots
        lost = slots[0]
        healthy = slots[1]
        os.kill(lost.pid, signal.SIGKILL)
        _await(
            lambda: (
                workers.status.slots[0].generation > lost.generation
                or workers.status.slots[0].state != "ready"
            )
        )
        result = _action(workers, "healthy-slot").result(timeout=3)
        assert result["pid"] == healthy.pid
        _await(lambda: workers.status.slots[0].generation > lost.generation)
        assert workers.status.slots[1].state == "ready"
        assert workers.status.slots[1].pid == healthy.pid
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_forced_worker_death_kills_native_child_and_recovers(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "child.pid"
    workers = _fake_workers(ctx, initargs=(None, None, None, str(marker), None))
    try:
        old_pid = _ready(workers)
        future = _action(workers, "native-child")
        child_pid = int(_await(lambda: marker.read_text() if marker.exists() else None))
        os.kill(old_pid, signal.SIGKILL)
        with pytest.raises(WorkerCrashedError):
            future.result(timeout=4)
        _await(lambda: not _alive(child_pid), timeout=4)
        new_pid = _await(
            lambda: (
                workers.status.slots[0].pid
                if workers.status.slots[0].state == "ready"
                else None
            )
        )
        assert new_pid != old_pid
        assert _action(workers, "recovered").result(timeout=3)["value"] == "recovered"
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PR_SET_PDEATHSIG")
def test_native_child_stops_after_worker_timeout(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "child.pid"
    workers = _fake_workers(ctx, initargs=(None, None, None, str(marker), None))
    try:
        _ready(workers)
        with pytest.raises(ExecutionTimeoutError):
            _action(workers, "native-child").result(timeout=4)
        child_pid = int(_await(lambda: marker.read_text() if marker.exists() else None))
        _await(lambda: not _alive(child_pid), timeout=4)
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PR_SET_PDEATHSIG")
def test_parent_death_during_child_startup_does_not_orphan_child(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "child.pid"
    workers = _fake_workers(
        ctx,
        initargs=(None, None, None, str(marker), None),
        task=_loss_during_child_start,
    )
    try:
        _ready(workers)
        with pytest.raises(ExecutionTimeoutError):
            _action(workers, "loss-startup").result(timeout=4)
        child_pid = int(
            _await(
                lambda: (
                    Path(str(marker) + ".early").read_text()
                    if Path(str(marker) + ".early").exists()
                    else None
                ),
                3,
            )
        )
        _await(lambda: not _alive(child_pid), timeout=4)
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_replacement_initialization_failure_is_visible_and_not_ready():
    ctx = __import__("multiprocessing").get_context("spawn")
    count = ctx.Value("i", 0)
    fail = ctx.Value("b", False)
    workers = _fake_workers(ctx, initargs=(None, None, count, None, fail))
    try:
        _ready(workers)
        fail.value = True
        with pytest.raises(WorkerCrashedError):
            _action(workers, "die").result(timeout=4)
        _await(
            lambda: (
                workers.status.slots[0].generation > 1
                and workers.status.slots[0].state == "unavailable"
            )
        )
        status = workers.status.slots[0]
        assert status.pid is None
        assert status.error
        with pytest.raises(StartupFailureError):
            _action(workers, "later")
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_shutdown_during_execution_kills_worker_and_native_child(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "child.pid"
    workers = _fake_workers(ctx, initargs=(None, None, None, str(marker), None))
    future = None
    child_pid = None
    try:
        _ready(workers)
        future = _action(workers, "native-child")
        child_pid = int(
            _await(lambda: marker.read_text() if marker.exists() else None, 3)
        )
        workers.shutdown(graceful=False)
        _await(lambda: not _alive(child_pid), timeout=4)
        assert workers.status.workers == 0
        assert future is not None and future.done()
        with pytest.raises(SearchCancelledError):
            future.result()
    finally:
        workers.shutdown(graceful=False)
        if child_pid is not None:
            _await(lambda: not _alive(child_pid), timeout=4)
        if future is not None and not future.done():
            future.cancel()


def test_spawned_worker_searches_with_actual_move_engine(scratch):
    artifact = scratch / "model"
    artifact.mkdir()
    shutil.copy(FIXTURE / "metadata.json", artifact / "metadata.json")
    shutil.copy(FIXTURE / "weights.json", artifact / "weights.json")
    identity = serving_artifact_identity(
        artifact / "metadata.json", artifact / "weights.json"
    )
    bundle = RuntimeBundle("bundle", "single", ("model",), (identity,))
    profile = SearchProfile(
        "default", "profile", "bundle", SearchSettings(simulations=2)
    )
    workers = SearchWorkers(
        str(scratch),
        {"bundle": bundle},
        {"default": profile},
        worker_count=1,
        execution_timeout=30,
        startup_timeout=60,
    )
    try:
        _await(lambda: workers.status.ready, 60)
        request = _request()
        original_pid = workers.status.slots[0].pid
        result = workers.submit(request).result(timeout=30)
        assert result["request_id"] == request.request_id
        assert result["game_revision"] == request.game_revision
        assert result["result"]["kind"] == "move"
        assert (
            workers.submit(request).result(timeout=30)["request_id"]
            == request.request_id
        )
        assert workers.status.slots[0].pid == original_pid
    finally:
        workers.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_application_exception_propagates_without_restarting_worker():
    ctx = __import__("multiprocessing").get_context("spawn")
    workers = _fake_workers(ctx, initargs=(None, None, None, None, None))
    try:
        pid = _ready(workers)
        with pytest.raises(ValueError, match="controlled application exception"):
            _action(workers, "raise").result(timeout=4)
        assert workers.status.slots[0].state == "ready"
        assert workers.status.slots[0].pid == pid
        for action, exception, message in (
            ("raise-timeout", TimeoutError, "application timeout"),
            (
                "raise-cancelled",
                __import__("concurrent.futures").futures.CancelledError,
                "application cancellation",
            ),
        ):
            with pytest.raises(exception, match=message):
                _action(workers, action).result(timeout=4)
            assert workers.status.slots[0].state == "ready"
            assert workers.status.slots[0].pid == pid
        assert _action(workers, "still-healthy").result(timeout=3)["pid"] == pid
    finally:
        workers.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_shutdown_during_worker_initialization_kills_process(scratch):
    ctx = __import__("multiprocessing").get_context("spawn")
    marker = scratch / "starting-worker.pid"
    workers = _fake_workers(
        ctx,
        initargs=(30, None, None, None, None, str(marker)),
        startup=30,
    )
    try:
        pid = int(_await(lambda: marker.read_text() if marker.exists() else None))
        workers.shutdown(graceful=False)
        _await(lambda: not _alive(pid))
        assert workers.status.workers == 0
    finally:
        workers.shutdown(graceful=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death guarantee")
def test_multiple_workers_and_admission_are_bounded():
    ctx = __import__("multiprocessing").get_context("spawn")
    started = ctx.Event()
    workers = _fake_workers(
        ctx,
        initargs=(None, started, None, None, None),
        worker_count=2,
    )
    try:
        _await(lambda: workers.status.workers == 2)
        first = _action(workers, "wait")
        second = _action(workers, "wait")
        _await(lambda: workers.status.in_flight == 2)
        with pytest.raises(CapacityExhaustedError):
            _action(workers, "overflow")
        assert first.cancel()
        assert second.cancel()
        with pytest.raises(SearchCancelledError):
            first.result(timeout=4)
        with pytest.raises(SearchCancelledError):
            second.result(timeout=4)
    finally:
        workers.shutdown(graceful=False)


def test_unsupported_platform_fails_before_starting_workers(monkeypatch):
    monkeypatch.setattr(worker_module.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        SearchWorkers(".", {}, {}, worker_count=1)
