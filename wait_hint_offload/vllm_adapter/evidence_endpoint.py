from __future__ import annotations

import hmac
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import msgspec
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.kv_offload import base

from wait_hint_offload.evidence_protocol import EvidenceRequest
from wait_hint_offload.policy_protocol import (
    PolicyAction,
    PolicyEnvelope,
    validate_policy_acknowledgement,
)


def _frame(value: dict[str, Any]) -> str:
    return (
        "data: "
        + json.dumps(
            {"schema": "pr-evidence-stream-v1"} | value,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n\n"
    )


def _client(request: Request) -> Any:
    credential = request.headers.get("authorization", "").encode()
    if not any(
        hmac.compare_digest(credential, ("Bearer " + key).encode())
        for key in request.app.state.pr_evidence_keys
    ):
        raise HTTPException(status_code=401, detail="authentication required")
    return request.app.state.pr_evidence_engine


class EvidenceEndpoint:
    name = "pr_evidence"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.post("/v1/pr-evidence/generate")(self.generate)
        app.post("/v1/pr-evidence/hints")(self.publish)
        app.post("/v1/pr-evidence/reset")(self.reset)
        app.get("/v1/pr-evidence/groups")(self.groups)

    async def init_state(self, engine_client: Any, state: Any, args: Any) -> None:
        keys = args.api_key
        if isinstance(keys, str):
            keys = [keys]
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
        ):
            raise ValueError("evidence endpoint requires explicit API keys")
        state.pr_evidence_keys = tuple(keys)
        state.pr_evidence_engine = engine_client

    async def generate(self, request: Request) -> StreamingResponse:
        engine = _client(request)
        try:
            parsed = EvidenceRequest.parse(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return StreamingResponse(
            self.stream(engine, parsed),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def stream(
        self, engine: Any, request: EvidenceRequest
    ) -> AsyncGenerator[str, None]:
        accepted_ns = time.monotonic_ns()
        envelope = None
        if request.ttl_ms is not None:
            action = PolicyAction(
                request.request_id, request.session, "hold", request.ttl_ms
            )
            envelope = msgspec.convert(
                {
                    "protocol_version": "v1",
                    "message_id": request.request_id,
                    "actions": [
                        {
                            "action_id": action.action_id,
                            "action_type": "wait_hint.request_ttl",
                            "action_version": "v1",
                            "payload": action.to_dict(),
                        }
                    ],
                },
                type=getattr(base, "KvHintsEnvelope"),
            )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=request.output_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
            skip_special_tokens=False,
            extra_args={
                "kv_transfer_params": {"wait_hint_policy": request.session.to_dict()}
            },
        )
        yield _frame(
            {
                "kind": "accepted",
                "request_id": request.request_id,
                "server_accepted_ns": accepted_ns,
            }
        )
        finished = False
        try:
            async for output in engine.generate(
                {"prompt_token_ids": list(request.prompt_token_ids)},
                sampling,
                request.request_id,
                kv_hints=envelope,
            ):
                metrics = output.metrics
                timing = (
                    None
                    if metrics is None
                    else {
                        name: getattr(metrics, name)
                        for name in (
                            "num_generation_tokens",
                            "num_preemptions",
                            "queued_ts",
                            "scheduled_ts",
                            "first_token_ts",
                            "last_token_ts",
                            "is_corrupted",
                        )
                    }
                )
                for completion in output.outputs:
                    yield _frame(
                        {
                            "kind": "tokens" if completion.token_ids else "metadata",
                            "request_id": request.request_id,
                            "server_emitted_ns": time.monotonic_ns(),
                            "token_ids": list(completion.token_ids),
                            "text": completion.text,
                            "finished": output.finished,
                            "finish_reason": completion.finish_reason,
                            "num_cached_tokens": output.num_cached_tokens,
                            "engine_timing": timing,
                        }
                    )
                finished = output.finished
            if not finished:
                raise RuntimeError("engine stream ended without a terminal output")
            yield _frame(
                {
                    "kind": "done",
                    "request_id": request.request_id,
                    "server_emitted_ns": time.monotonic_ns(),
                }
            )
        finally:
            if not finished:
                await engine.abort(request.request_id)

    async def publish(self, request: Request) -> dict[str, Any]:
        engine = _client(request)
        try:
            envelope = PolicyEnvelope.parse(await request.json())
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        result = await engine.engine_core.call_utility_async(
            "publish_kv_cache_hints", envelope.to_dict()
        )
        try:
            return validate_policy_acknowledgement(result, envelope)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def reset(self, request: Request) -> dict[str, Any]:
        engine = _client(request)
        raw = await request.json()
        if (
            not isinstance(raw, dict)
            or set(raw) != {"reset_connector"}
            or type(raw["reset_connector"]) is not bool
        ):
            raise HTTPException(status_code=400, detail="closed reset request required")
        reset = await engine.engine_core.call_utility_async(
            "reset_prefix_cache", False, raw["reset_connector"]
        )
        return {"reset": bool(reset)}

    async def groups(self, request: Request) -> dict[str, Any]:
        engine = _client(request)
        groups = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
        return {"groups": groups}
