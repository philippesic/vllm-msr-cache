from __future__ import annotations

import hmac
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from wait_hint_offload.contracts import Hint


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
        publish = getattr(engine, "publish_kv_connector_wait_hint", None)
        if not callable(publish):
            raise HTTPException(status_code=503, detail="hint RPC unavailable")
        try:
            raw = await request.json()
            if not isinstance(raw, dict):
                raise ValueError("hint must be an object")
            Hint.parse(raw)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        result = await publish(raw)
        if not isinstance(result, dict) or result.get("schema") != "wait-hint-ack-v2":
            raise HTTPException(
                status_code=502, detail="invalid engine acknowledgement"
            )
        return result
