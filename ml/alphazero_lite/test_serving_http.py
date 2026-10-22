"""HTTP boundary tests using injected in-process worker doubles."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from ml.alphazero_lite.serving.contract import MoveRequest, state_sha256
from ml.alphazero_lite.serving.engine import SearchProfile, SearchSettings
from ml.alphazero_lite.serving.http import ServiceConfig, create_app

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

    def submit(self, request):
        self.calls += 1
        req = MoveRequest.model_validate(request)

        async def compute():
            await self.gate.wait()
            return {
                "request_id": req.request_id,
                "game_revision": req.game_revision,
                "state_sha256": state_sha256(req.state),
                "result": {"kind": "move", "move": 0},
                "model_bundle_id": "bundle:version",
                "search_profile_id": "profile:version",
                "elapsed_ms": 1,
                "completed_simulations": 1,
                "termination_reason": "completed",
            }

        return asyncio.run_coroutine_threadsafe(compute(), asyncio.get_running_loop())

    async def await_result(self, future):
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
    )
    return create_app(config, worker_factory=factory), workers


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
