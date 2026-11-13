from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from wait_hint_offload.contracts import Hint
from wait_hint_offload.vllm_adapter.engine_rpc import PUBLISH_METHOD


class WaitHintEndpoint:
    name = "wait_hint"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.post("/v1/wait-hints")(self.publish)

    async def init_state(self, engine_client: Any, state: Any, args: Any) -> None:
        keys = args.api_key
        if isinstance(keys, str):
            keys = [keys]
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
        ):
            raise ValueError("wait-hint endpoint requires explicit server API keys")
        state.wait_hint_keys = tuple(keys)
        state.wait_hint_engine = engine_client

    async def publish(self, request: Request) -> dict[str, Any]:
        keys = getattr(request.app.state, "wait_hint_keys", ())
        credential = request.headers.get("authorization", "").encode("utf-8")
        if not any(
            hmac.compare_digest(credential, ("Bearer " + key).encode("utf-8"))
            for key in keys
        ):
            raise HTTPException(status_code=401, detail="authentication required")
        engine = getattr(request.app.state, "wait_hint_engine", None)
        call = getattr(getattr(engine, "engine_core", None), "call_utility_async", None)
        if not callable(call):
            raise HTTPException(
                status_code=503, detail="engine utility RPC unavailable"
            )
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raise ValueError("hint must be an object")
            Hint.parse(raw)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        try:
            result = await call(PUBLISH_METHOD, raw)
        except Exception as error:
            raise HTTPException(
                status_code=502, detail=f"engine rejected hint RPC: {error}"
            ) from error
        if not isinstance(result, dict) or result.get("schema") != "wait-hint-ack-v3":
            raise HTTPException(
                status_code=502, detail="invalid engine acknowledgement"
            )
        return result
