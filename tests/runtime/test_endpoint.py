from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from vllm.v1.engine.core import EngineCore

from tests.runtime.support import scheduler_fixture
from wait_hint_offload.contracts import Hint
from wait_hint_offload.observed import Observation
from wait_hint_offload.vllm_adapter import engine_rpc
from wait_hint_offload.vllm_adapter.endpoint import WaitHintEndpoint
from wait_hint_offload.vllm_adapter.engine_rpc import PUBLISH_METHOD


class UtilityBoundary:
    def __init__(self, connector: object) -> None:
        self.core = SimpleNamespace(
            scheduler=SimpleNamespace(get_kv_connector=lambda: connector)
        )
        self.calls = 0

    async def call_utility_async(
        self, method: str, hint: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        dispatch = cast(
            Callable[[Any, dict[str, Any]], dict[str, Any]],
            getattr(EngineCore, method),
        )
        return dispatch(self.core, hint)


class FailingUtility:
    async def call_utility_async(
        self, method: str, hint: dict[str, Any]
    ) -> dict[str, Any]:
        raise RuntimeError(f"no utility {method} for {sorted(hint)}")


def test_engine_method_reports_missing_connector() -> None:
    scheduler_fixture()
    assert getattr(EngineCore, PUBLISH_METHOD) is engine_rpc.publish
    engine_rpc.install()
    core = UtilityBoundary(None).core
    assert engine_rpc.publish(cast(Any, core), {})["status"] == "unsupported"


def test_authenticated_endpoint_and_actual_rpc_method_chain() -> None:
    async def scenario() -> None:
        app = FastAPI()
        plugin = WaitHintEndpoint()
        assert plugin.required_tasks == ("generate",)
        plugin.attach_router(app)
        scheduler = scheduler_fixture()
        utility = UtilityBoundary(
            SimpleNamespace(publish_wait_hint=scheduler.publish_wait_hint)
        )
        engine = SimpleNamespace(engine_core=utility)
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
            assert utility.calls == 0
            headers = {"Authorization": "Bearer test-key"}
            malformed = await client.post(
                "/v1/wait-hints",
                json={**hint.mapping(), "schema": "unknown"},
                headers=headers,
            )
            assert malformed.status_code == 400 and utility.calls == 0
            okay = await client.post(
                "/v1/wait-hints", json=hint.mapping(), headers=headers
            )
            assert okay.status_code == 200
            assert okay.json()["decision"] == "no_ready_prefix"
            assert utility.calls == 1
            assert okay.json()["mechanisms"]["semantic_scoring"] == "absent"
            foreign = await client.post(
                "/v1/wait-hints",
                json={**hint.mapping(), "scope": "foreign"},
                headers=headers,
            )
            assert foreign.json()["decision"] == "unobserved"
            app.state.wait_hint_engine = SimpleNamespace(engine_core=FailingUtility())
            failed = await client.post(
                "/v1/wait-hints", json=hint.mapping(), headers=headers
            )
            assert failed.status_code == 502
            assert PUBLISH_METHOD in failed.json()["detail"]
            app.state.wait_hint_engine = object()
            missing = await client.post(
                "/v1/wait-hints", json=hint.mapping(), headers=headers
            )
            assert missing.status_code == 503
        with pytest.raises(ValueError, match="API keys"):
            await plugin.init_state(engine, app.state, SimpleNamespace(api_key=None))

    asyncio.run(scenario())
