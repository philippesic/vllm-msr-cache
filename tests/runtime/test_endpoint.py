from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient

from tests.runtime.support import scheduler_fixture
from wait_hint_offload.contracts import Hint
from wait_hint_offload.observed import Observation
from wait_hint_offload.vllm_adapter.endpoint import WaitHintEndpoint


class UtilityBoundary:
    def __init__(self, connector: Any) -> None:
        self.core = SimpleNamespace(scheduler=SimpleNamespace(connector=connector))
        self.calls = 0

    async def call_utility_async(
        self, method: str, hint: dict[str, Any]
    ) -> dict[str, Any]:
        assert callable(self.call_utility_async)
        assert method == "publish_kv_connector_wait_hint"
        self.calls += 1
        scheduler = self.core.scheduler
        self.core.scheduler.publish_kv_connector_wait_hint = lambda raw: cast(
            Callable[[Any, dict[str, Any]], dict[str, Any]],
            getattr(Scheduler, "publish_kv_connector_wait_hint"),
        )(scheduler, raw)
        return cast(
            Callable[[Any, dict[str, Any]], dict[str, Any]],
            getattr(EngineCore, "publish_kv_connector_wait_hint"),
        )(self.core, hint)


class EngineBoundary:
    def __init__(self, connector: Any) -> None:
        self.utility = UtilityBoundary(connector)
        self.engine_core = SimpleNamespace(
            publish_kv_connector_wait_hint_async=self.rpc
        )

    async def rpc(self, raw: dict[str, Any]) -> dict[str, Any]:
        return await cast(
            Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]],
            getattr(AsyncMPClient, "publish_kv_connector_wait_hint_async"),
        )(self.utility, raw)

    async def publish_kv_connector_wait_hint(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]:
        if self.engine_core is None:
            raise RuntimeError("engine not initialized")
        return await cast(
            Callable[[Any, dict[str, Any]], Awaitable[dict[str, Any]]],
            getattr(AsyncLLM, "publish_kv_connector_wait_hint"),
        )(self, raw)


def test_authenticated_endpoint_and_actual_rpc_method_chain() -> None:
    async def scenario() -> None:
        app = FastAPI()
        plugin = WaitHintEndpoint()
        assert plugin.required_tasks == ("generate",)
        plugin.attach_router(app)
        scheduler = scheduler_fixture()
        engine = EngineBoundary(scheduler)
        await plugin.init_state(
            engine, app.state, SimpleNamespace(api_key=["test-key"])
        )
        observation = Observation(
            "epoch",
            1,
            "prefix",
            4,
            scheduler.settings.namespace,
            time.monotonic_ns(),
            (b"missing",),
        )
        assert scheduler.observed.register("client", "session", observation)
        now = time.monotonic_ns()
        hint = Hint(
            "client",
            "session",
            "epoch",
            1,
            "event",
            "prefix",
            4,
            observation.namespace,
            observation.observed_ns,
            now,
            now + 1_000_000_000,
            scheduler.clock_domain,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local"
        ) as client:
            rejected = await client.post("/v1/wait-hints", json=hint.mapping())
            assert rejected.status_code == 401
            assert engine.utility.calls == 0
            headers = {"Authorization": "Bearer test-key"}
            malformed = await client.post(
                "/v1/wait-hints",
                json={**hint.mapping(), "schema": "unknown"},
                headers=headers,
            )
            assert malformed.status_code == 400 and engine.utility.calls == 0
            okay = await client.post(
                "/v1/wait-hints", json=hint.mapping(), headers=headers
            )
            assert okay.status_code == 200
            assert okay.json()["decision"] == "no_ready_prefix"
            assert engine.utility.calls == 1
            assert okay.json()["mechanisms"]["semantic_scoring"] == "absent"
            foreign = await client.post(
                "/v1/wait-hints",
                json={**hint.mapping(), "scope": "foreign"},
                headers=headers,
            )
            assert foreign.json()["decision"] == "unobserved"
            app.state.wait_hint_engine = object()
            missing = await client.post(
                "/v1/wait-hints", json=hint.mapping(), headers=headers
            )
            assert missing.status_code == 503
        with pytest.raises(ValueError, match="API keys"):
            await plugin.init_state(engine, app.state, SimpleNamespace(api_key=None))

    asyncio.run(scenario())
