# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from semantic_offload.wait_hint_endpoint import SemanticWaitHintEndpointPlugin
from tests.test_wait_hint import _hint


def test_wait_hint_endpoint_plugin_is_namespaced_and_generate_only():
    app = FastAPI()
    plugin = SemanticWaitHintEndpointPlugin()

    plugin.attach_router(app)

    assert plugin.name == "semantic_wait_hint"
    assert plugin.required_tasks == ("generate",)
    assert any(
        route.path == "/plugins/semantic-offload/wait-hints"
        and "POST" in route.methods
        for route in app.routes
    )


def test_wait_hint_endpoint_plugin_binds_engine_state():
    plugin = SemanticWaitHintEndpointPlugin()
    state = SimpleNamespace()
    engine_client = object()

    asyncio.run(plugin.init_state(engine_client, state, SimpleNamespace()))

    assert state.semantic_wait_hint_engine_client is engine_client


def test_wait_hint_endpoint_validates_and_delegates_exact_payload():
    app = FastAPI()
    plugin = SemanticWaitHintEndpointPlugin()
    plugin.attach_router(app)
    observed = []

    class Engine:
        async def publish_kv_connector_wait_hint(self, payload):
            observed.append(payload)
            return {"status": "accepted", "reason": "queued"}

    app.state.semantic_wait_hint_engine_client = Engine()
    client = TestClient(app)
    payload = _hint()

    response = client.post("/plugins/semantic-offload/wait-hints", json=payload)

    assert response.status_code == 200
    assert response.json()["event_id"] == payload["event_id"]
    assert observed == [payload]

    payload["future_suffix"] = [1, 2]
    rejected = client.post("/plugins/semantic-offload/wait-hints", json=payload)
    assert rejected.status_code == 400
    assert len(observed) == 1
