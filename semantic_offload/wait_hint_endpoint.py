# SPDX-License-Identifier: Apache-2.0
"""Allowlisted HTTP endpoint for publishing scheduler-owned wait hints."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request

from semantic_offload.wait_hint import KnownPrefixWaitHint


class SemanticWaitHintEndpointPlugin:
    name = "semantic_wait_hint"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.post("/plugins/semantic-offload/wait-hints")
        async def publish_wait_hint(raw_request: Request) -> dict[str, Any]:
            engine_client = raw_request.app.state.semantic_wait_hint_engine_client
            if engine_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="semantic wait hints require an engine",
                )
            publish = getattr(
                engine_client, "publish_kv_connector_wait_hint", None
            )
            if not callable(publish):
                raise HTTPException(
                    status_code=503,
                    detail="engine lacks the reviewed KV connector wait-hint seam",
                )
            try:
                raw = await raw_request.json()
                hint = KnownPrefixWaitHint.from_mapping(raw)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            result = await publish(raw)
            if not isinstance(result, dict):
                raise HTTPException(
                    status_code=500,
                    detail="engine returned malformed wait-hint acknowledgement",
                )
            return {
                "schema": "semantic-known-prefix-wait-hint-ack-v1",
                "event_id": hint.event_id,
                **result,
            }

    async def init_state(self, engine_client, state, args) -> None:
        state.semantic_wait_hint_engine_client = engine_client
