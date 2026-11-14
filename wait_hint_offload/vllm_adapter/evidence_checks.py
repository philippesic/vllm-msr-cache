from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request as NativeRequest

from wait_hint_offload.evidence_protocol import EvidenceRequest
from wait_hint_offload.policy_protocol import PolicySession
from wait_hint_offload.vllm_adapter.evidence_endpoint import EvidenceEndpoint


class FixtureEngine:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.aborted: list[str] = []

    async def generate(
        self, prompt: dict[str, Any], sampling: Any, request_id: str, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.requests.append(
            {"prompt": prompt, "sampling": sampling, "request_id": request_id} | kwargs
        )
        first = time.monotonic()
        for ids, count, finished in (
            ([], 0, False),
            ([10], 1, False),
            ([11, 12], 3, True),
        ):
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=ids,
                        text="text" if ids else "",
                        finish_reason="length" if finished else None,
                    )
                ],
                finished=finished,
                num_cached_tokens=2,
                metrics=SimpleNamespace(
                    num_generation_tokens=count,
                    num_preemptions=0,
                    queued_ts=first,
                    scheduled_ts=first,
                    first_token_ts=first if count else 0.0,
                    last_token_ts=time.monotonic() if count else 0.0,
                    is_corrupted=False,
                ),
            )
            await asyncio.sleep(0)

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


async def observe_endpoint() -> dict[str, bool]:
    endpoint = EvidenceEndpoint()
    app = FastAPI()
    endpoint.attach_router(app)
    assert {
        getattr(route, "path", "")
        for route in app.routes
        if getattr(route, "path", "").startswith("/v1/pr-evidence/")
    } == {
        "/v1/pr-evidence/generate",
        "/v1/pr-evidence/hints",
        "/v1/pr-evidence/reset",
        "/v1/pr-evidence/groups",
    }
    engine = FixtureEngine()
    await endpoint.init_state(
        engine, app.state, SimpleNamespace(api_key=["cpu-fixture"])
    )
    request = EvidenceRequest(
        "first", (1, 2, 3), 3, PolicySession("fixture", "s", "i", 1), 12000
    )
    wire = [json.loads(frame[6:]) async for frame in endpoint.stream(engine, request)]
    assert [frame["kind"] for frame in wire] == [
        "accepted",
        "metadata",
        "tokens",
        "tokens",
        "done",
    ]
    assert [token for frame in wire for token in frame.get("token_ids", [])] == [
        10,
        11,
        12,
    ]
    captured = engine.requests[0]
    assert captured["prompt"] == {"prompt_token_ids": [1, 2, 3]}
    assert captured["sampling"].output_kind is RequestOutputKind.DELTA
    assert captured["sampling"].ignore_eos and captured["sampling"].max_tokens == 3
    assert (
        captured["sampling"].extra_args["kv_transfer_params"]["wait_hint_policy"]
        == request.session.to_dict()
    )
    assert captured["kv_hints"].actions[0].payload["ttl_ms"] == 12000
    core_request: Any = EngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=list(request.prompt_token_ids),
        mm_features=None,
        sampling_params=captured["sampling"],
        pooling_params=None,
        arrival_time=time.monotonic(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    core_request.kv_hints = captured["kv_hints"]
    getattr(InputProcessor, "assign_request_id")(core_request)
    assert core_request.external_req_id == request.request_id
    assert core_request.request_id != request.request_id
    native = NativeRequest.from_engine_core_request(core_request, None)
    assert native.kv_transfer_params == {
        "wait_hint_policy": request.session.to_dict(),
        "pr_evidence_request_id": request.request_id,
    }
    assert getattr(native, "kv_hints") == captured["kv_hints"]
    assert not engine.aborted
    stream = endpoint.stream(
        engine, EvidenceRequest("cancel", (1, 2, 3), 3, request.session, None)
    )
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    assert engine.aborted == ["cancel"]
    assert engine.requests[-1]["kv_hints"] is None
    return {
        "route_registration": True,
        "token_ids_pass_through_without_chat_rerendering": True,
        "request_ttl_and_registration_reach_engine": True,
        "delta_stream_distinguishes_metadata_tokens_and_done": True,
        "disconnect_aborts_owned_request": True,
    }
