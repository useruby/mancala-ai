"""HTTP boundary tests with lightweight doubles and real-process integration."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections import OrderedDict
from types import SimpleNamespace
from pathlib import Path
import sys

import httpx
import pytest

from ml.alphazero_lite.serving.contract import MoveRequest, state_sha256
from ml.alphazero_lite.serving.engine import (
    RuntimeBundle,
    SearchProfile,
    SearchSettings,
    serving_artifact_identity,
)
from ml.alphazero_lite.serving.http import (
    ServiceConfig,
    _Entry,
    _finalize_entry,
    create_app,
)
from ml.alphazero_lite.serving.workers import (
    CapacityExhaustedError,
    ExecutionTimeoutError,
    SearchCancelledError,
    WorkerCrashedError,
    WorkerSubmissionError,
    _process_alive,
)

STATE = {
    "player0_pits": [4, 4, 4, 4, 4, 4],
    "player1_pits": [4, 4, 4, 4, 4, 4],
    "player0_store": 0,
    "player1_store": 0,
    "current_player": 0,
}


class Workers:
    def __init__(self, *_args, **_kwargs):
        self.status = SimpleNamespace(
            accepting=True,
            capacity=1,
            in_flight=0,
            slots=(SimpleNamespace(state="ready"),),
        )
        self.calls = 0
        self.gate = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.invalid_result = False
        self.submit_error = None
        self.await_error = None

    def submit(self, request):
        if self.submit_error is not None:
            raise self.submit_error
        self.calls += 1
        req = MoveRequest.model_validate(request)

        async def compute():
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return {
                "request_id": req.request_id,
                "game_revision": req.game_revision,
                "state_sha256": state_sha256(req.state),
                "result": {"kind": "move", "move": 99 if self.invalid_result else 0},
                "model_bundle_id": "bundle:version",
                "search_profile_id": "profile:version",
                "elapsed_ms": 1,
                "completed_simulations": 1,
                "termination_reason": "completed",
            }

        return asyncio.run_coroutine_threadsafe(compute(), asyncio.get_running_loop())

    async def await_result(self, future):
        if self.await_error is not None:
            raise self.await_error
        return await asyncio.wrap_future(future)

    def shutdown(self, **_kwargs):
        pass


@pytest.fixture
def setup_app():
    workers = []

    def factory(*args, **kwargs):
        instance = Workers(*args, **kwargs)
        workers.append(instance)
        return instance

    config = ServiceConfig(
        asset_root="unused",
        bundles={},
        profiles={
            "balanced@v1": SearchProfile(
                "balanced@v1", "balanced", "bundle", SearchSettings(simulations=1)
            )
        },
        worker_count=1,
        duplicate_max_entries=2,
        duplicate_retention_seconds=30,
    )
    return create_app(config, worker_factory=factory), workers


async def _asgi_disconnect(app, payload):
    body_sent = False
    disconnected = asyncio.Event()
    messages = []

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/move",
        "raw_path": b"/v1/move",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    while not body_sent:
        await asyncio.sleep(0)
    return task, disconnected, messages


def _payload(request_id):
    return {
        "request_id": request_id,
        "game_revision": 0,
        "state": STATE,
        "profile": "balanced@v1",
        "seed": 3,
    }


@pytest.mark.asyncio
async def test_move_and_health_stay_responsive_during_search(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            payload = {
                "request_id": "req-1",
                "game_revision": 0,
                "state": STATE,
                "profile": "balanced@v1",
                "seed": 3,
            }
            pending = asyncio.create_task(client.post("/v1/move", json=payload))
            while workers[0].calls == 0:
                await asyncio.sleep(0)
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 200
            workers[0].gate.set()
            response = await pending
            assert response.status_code == 200
            assert response.json()["request_id"] == "req-1"
            assert app.openapi()["paths"]["/v1/move"]["post"]["operationId"]


@pytest.mark.asyncio
async def test_strict_request_and_semantic_state_errors(setup_app):
    app, _workers = setup_app
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/move", content="{")
            assert response.status_code == 400
            payload = {
                "request_id": "req-2",
                "game_revision": 0,
                "state": STATE,
                "profile": "balanced@v1",
                "seed": 3,
                "extra": 1,
            }
            assert (await client.post("/v1/move", json=payload)).status_code == 400
            payload.pop("extra")
            payload["state"] = {**STATE, "player0_store": 1}
            assert (await client.post("/v1/move", json=payload)).status_code == 422


@pytest.mark.asyncio
async def test_duplicate_waiters_share_search_and_conflicting_payload_fails(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {
                "request_id": "shared-id",
                "game_revision": 0,
                "state": STATE,
                "profile": "balanced@v1",
                "seed": 3,
            }
            first = asyncio.create_task(client.post("/v1/move", json=payload))
            while workers[0].calls == 0:
                await asyncio.sleep(0)
            second = asyncio.create_task(client.post("/v1/move", json=payload))
            conflict = {**payload, "seed": 4}
            assert (await client.post("/v1/move", json=conflict)).status_code == 409
            workers[0].gate.set()
            result1, result2 = await asyncio.gather(first, second)
            assert result1.json() == result2.json()
            assert workers[0].calls == 1


@pytest.mark.asyncio
async def test_asgi_disconnect_after_body_does_not_cancel_other_duplicate(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        first, first_disconnect, _ = await _asgi_disconnect(app, _payload("disconnect"))
        while workers[0].calls == 0:
            await asyncio.sleep(0)
        second, second_disconnect, _ = await _asgi_disconnect(
            app, _payload("disconnect")
        )
        while app.state.entries["disconnect"].callers != 2:
            await asyncio.sleep(0)
        first_disconnect.set()
        await first
        assert not workers[0].cancelled.is_set()
        assert app.state.entries["disconnect"].callers == 1
        workers[0].gate.set()
        await second
        second_disconnect.set()
        assert workers[0].calls == 1


@pytest.mark.asyncio
async def test_final_disconnect_and_handler_cancellation_reclaim_entries(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        disconnected, disconnect, _ = await _asgi_disconnect(
            app, _payload("final-disconnect")
        )
        while workers[0].calls == 0:
            await asyncio.sleep(0)
        disconnect.set()
        await disconnected
        await workers[0].cancelled.wait()
        for _ in range(100):
            if "final-disconnect" not in app.state.entries:
                break
            await asyncio.sleep(0)
        assert "final-disconnect" not in app.state.entries
        workers[0].gate.set()
        retry = await _post(app, _payload("final-disconnect"))
        assert retry.status_code == 200
        assert workers[0].calls == 2

        worker = workers[0]
        worker.cancelled.clear()
        worker.gate.clear()
        request = asyncio.create_task(_post(app, _payload("handler-cancel")))
        while worker.calls < 3:
            await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await worker.cancelled.wait()
        for _ in range(100):
            if "handler-cancel" not in app.state.entries:
                break
            await asyncio.sleep(0)
        assert "handler-cancel" not in app.state.entries


@pytest.mark.asyncio
async def test_repeated_disconnections_do_not_exceed_duplicate_capacity(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        for index in range(5):
            workers[0].cancelled.clear()
            task, disconnected, _messages = await _asgi_disconnect(
                app, _payload(f"cancel-{index}")
            )
            while workers[0].calls <= index:
                await asyncio.sleep(0)
            disconnected.set()
            await task
            await workers[0].cancelled.wait()
            for _ in range(100):
                if not app.state.entries:
                    break
                await asyncio.sleep(0)
            assert not app.state.entries
            assert len(app.state.entries) <= 2


async def _post(app, payload):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post("/v1/move", json=payload)


async def _get_ready(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get("/health/ready")


@pytest.mark.asyncio
async def test_success_retention_is_bounded_and_expired_ids_are_reusable(setup_app):
    now = [100.0]
    workers = []

    def factory(*args, **kwargs):
        instance = Workers(*args, **kwargs)
        workers.append(instance)
        return instance

    app = create_app(
        ServiceConfig(
            asset_root="unused",
            bundles={},
            profiles={
                "balanced@v1": SearchProfile(
                    "balanced@v1", "balanced", "bundle", SearchSettings(simulations=1)
                )
            },
            worker_count=1,
            duplicate_max_entries=2,
            duplicate_retention_seconds=5,
        ),
        worker_factory=factory,
        clock=lambda: now[0],
    )
    async with app.router.lifespan_context(app):
        workers[0].gate.set()
        for key in ("keep-a", "keep-b", "keep-c"):
            assert (await _post(app, _payload(key))).status_code == 200
        assert len(app.state.entries) == 2
        assert "keep-a" not in app.state.entries
        assert (await _post(app, _payload("keep-b"))).status_code == 200
        now[0] += 6
        assert (await _post(app, {**_payload("keep-b"), "seed": 4})).status_code == 200


def test_late_finalization_cannot_remove_newer_same_id_entry():
    loop = asyncio.new_event_loop()
    old_task = loop.create_task(_noop())
    newer_task = loop.create_task(_noop())
    loop.run_until_complete(asyncio.gather(old_task, newer_task))
    old = _Entry("old", old_task, 0)
    newer = _Entry("new", newer_task, 1)
    entries = OrderedDict([("reused", newer)])
    _finalize_entry(entries, "reused", old, 10.0)
    assert old.finalized
    assert entries["reused"] is newer
    loop.close()


async def _noop():
    return None


@pytest.mark.asyncio
async def test_invalid_worker_result_is_not_cached(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        workers[0].gate.set()
        workers[0].invalid_result = True
        assert (await _post(app, _payload("bad-result"))).status_code == 500
        workers[0].invalid_result = False
        assert (await _post(app, _payload("bad-result"))).status_code == 200
        assert workers[0].calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ExecutionTimeoutError("supervised timeout"), 504),
        (TimeoutError("application timeout"), 500),
        (asyncio.CancelledError("application cancellation"), 500),
        (WorkerCrashedError("worker failed"), 503),
        (WorkerSubmissionError("submission failed"), 503),
        (SearchCancelledError("worker cancelled"), 503),
    ],
)
async def test_worker_failure_status_mapping(setup_app, failure, expected):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        workers[0].gate.set()
        workers[0].await_error = failure
        response = await _post(
            app, _payload(f"failure-{expected}-{type(failure).__name__}")
        )
        assert response.status_code == expected


@pytest.mark.asyncio
async def test_capacity_mapping_accounts_for_only_initialized_workers(setup_app):
    app, workers = setup_app
    async with app.router.lifespan_context(app):
        worker = workers[0]
        worker.status = SimpleNamespace(
            accepting=True,
            capacity=2,
            in_flight=1,
            slots=(SimpleNamespace(state="busy"), SimpleNamespace(state="starting")),
        )
        assert (await _get_ready(app)).status_code == 200
        worker.submit_error = CapacityExhaustedError("busy healthy capacity")
        response = await _post(app, _payload("busy-with-recovery"))
        assert response.status_code == 429

        worker.status = SimpleNamespace(
            accepting=True,
            capacity=2,
            in_flight=0,
            slots=(
                SimpleNamespace(state="starting"),
                SimpleNamespace(state="unavailable"),
            ),
        )
        response = await _post(app, _payload("no-healthy-capacity"))
        assert response.status_code == 503


@pytest.mark.asyncio
async def test_malformed_unknown_profile_and_invalid_state_statuses(setup_app):
    app, _workers = setup_app
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.post("/v1/move", content="{")).status_code == 400
            assert (
                await client.post(
                    "/v1/move",
                    json={**_payload("unknown-profile"), "profile": "missing"},
                )
            ).status_code == 422
            assert (
                await client.post(
                    "/v1/move",
                    json={
                        **_payload("invalid-state"),
                        "state": {**STATE, "player0_store": 1},
                    },
                )
            ).status_code == 422


@pytest.mark.skipif(sys.platform != "linux", reason="supervised process integration")
@pytest.mark.asyncio
async def test_real_http_search_workers_and_engine_integration(tmp_path):
    fixture = Path(__file__).parent / "fixtures/incumbent_forensic_proxy_current"
    artifact = tmp_path / "model"
    shutil.copytree(fixture, artifact)
    bundle = RuntimeBundle(
        "integration-bundle",
        "single",
        ("model",),
        (
            serving_artifact_identity(
                artifact / "metadata.json", artifact / "weights.json"
            ),
        ),
    )
    config = ServiceConfig(
        asset_root=str(tmp_path),
        bundles={"integration-bundle": bundle},
        profiles={
            "balanced@v1": SearchProfile(
                "balanced@v1",
                "integration-profile-v1",
                "integration-bundle",
                SearchSettings(simulations=2),
            )
        },
        worker_count=1,
        startup_timeout=20,
        execution_timeout=20,
        shutdown_timeout=5,
    )
    app = create_app(config)
    worker_pids = []
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            readiness = None
            for _ in range(500):
                readiness = await client.get("/health/ready")
                if readiness.status_code == 200:
                    break
                await asyncio.sleep(0.02)
            assert readiness is not None and readiness.status_code == 200
            payload = {
                "request_id": "real-engine",
                "game_revision": 7,
                "state": STATE,
                "profile": "balanced@v1",
                "seed": 13,
            }
            pending = asyncio.create_task(client.post("/v1/move", json=payload))
            assert (await client.get("/health/live")).status_code == 200
            response = await pending
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["request_id"] == payload["request_id"]
            assert body["game_revision"] == payload["game_revision"]
            assert body["state_sha256"] == state_sha256(
                MoveRequest.model_validate(payload).state
            )
            assert body["model_bundle_id"].startswith("integration-bundle:")
            assert body["search_profile_id"].startswith("balanced@v1:")
            legal = [
                index for index, count in enumerate(STATE["player0_pits"]) if count
            ]
            assert body["result"]["move"] in legal
            worker_pids = [slot.pid for slot in app.state.workers.status.slots]
    workers = app.state.workers
    assert not workers.status.accepting
    assert all(slot.state == "unavailable" for slot in workers.status.slots)
    assert all(pid is not None and not _process_alive(pid) for pid in worker_pids)
