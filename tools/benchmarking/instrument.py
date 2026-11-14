from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.resources import sample_process_tree
from tools.benchmarking.streaming import request_stream, verify_stream
from wait_hint_offload.evidence_events import read_journal


def _frame(request: str, kind: str, **fields: Any) -> bytes:
    return (
        "data: "
        + json.dumps(
            {"schema": "pr-evidence-stream-v1", "request_id": request, "kind": kind}
            | fields
        )
        + "\n\n"
    ).encode()


def _token(
    request: str,
    ids: list[int],
    count: int,
    *,
    finished: bool = False,
    clock: int | None = None,
    text: str = "",
) -> bytes:
    return _frame(
        request,
        "tokens" if ids else "metadata",
        server_emitted_ns=time.monotonic_ns() if clock is None else clock,
        token_ids=ids,
        text=text,
        finished=finished,
        finish_reason="length" if finished else None,
        num_cached_tokens=8,
        engine_timing={
            "num_generation_tokens": count,
            "num_preemptions": 0,
            "queued_ts": 1.0,
            "scheduled_ts": 1.1,
            "first_token_ts": 1.2 if count else 0.0,
            "last_token_ts": 1.5 if count else 0.0,
            "is_corrupted": False,
        },
    )


async def _sockets() -> dict[str, Any]:
    aiohttp = importlib.import_module("aiohttp")
    web = importlib.import_module("aiohttp.web")
    barrier = asyncio.Event()
    partial = asyncio.Event()
    release = asyncio.Event()
    pressure_started = asyncio.Event()
    pressure_done = asyncio.Event()
    arrived: list[str] = []

    async def handler(request: Any) -> Any:
        payload = await request.json()
        identity = payload["request_id"]
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            await response.write(
                _frame(identity, "accepted", server_accepted_ns=time.monotonic_ns())
            )
            await response.write(_token(identity, [], 0))
            if identity.startswith("parallel"):
                arrived.append(identity)
                if len(arrived) == 3:
                    barrier.set()
                await barrier.wait()
            if identity == "delay":
                await asyncio.sleep(0.04)
            if identity == "pressure":
                request.transport.set_write_buffer_limits(high=4096)
                pressure_started.set()
                for index in range(32):
                    await response.write(
                        _token(
                            identity,
                            [index],
                            index + 1,
                            finished=index == 31,
                            text="x" * 1024 * 1024,
                        )
                    )
                pressure_done.set()
            else:
                await response.write(_token(identity, [10], 1))
                if identity == "death":
                    request.transport.abort()
                    return response
                if identity in {"cancel", "timeout"}:
                    partial.set()
                    await release.wait()
                if identity == "clock":
                    await response.write(
                        _token(identity, [11, 12, 13], 4, finished=True, clock=0)
                    )
                else:
                    await response.write(
                        _token(
                            identity,
                            [11, 12, 13],
                            3 if identity == "count" else 4,
                            finished=True,
                        )
                    )
            if identity != "truncated":
                await response.write(
                    _frame(identity, "done", server_emitted_ns=time.monotonic_ns())
                )
            await response.write_eof()
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response

    app = web.Application()
    app.router.add_post("/v1/pr-evidence/generate", handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    captures = []
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0)
        ) as client:

            async def run(
                identity: str, lateness_ns: int = 0, timeout: float = 5.0
            ) -> dict[str, Any]:
                now = time.monotonic_ns()
                return await request_stream(
                    client,
                    url,
                    {"request_id": identity, "output_tokens": 4},
                    now - lateness_ns,
                    now + int(timeout * 1e9),
                )

            records = await asyncio.gather(
                *(run(f"parallel-{index}") for index in range(3))
            )
            assert all(record["outcome"] == "complete" for record in records)
            assert max(record["submitted_ns"] for record in records) < min(
                record["frames"][2]["received_ns"] for record in records
            )
            captures.extend(records)
            late = await run("late", 20_000_000)
            assert late["submission_lateness_ns"] >= 20_000_000
            assert (
                late["ttft_scheduled_ns"] - late["ttft_submitted_ns"]
                == late["submission_lateness_ns"]
            )
            assert late["coalesced_chunks"] == 1 and late["token_ids"] == [
                10,
                11,
                12,
                13,
            ]
            assert late["frames"][0]["frame"]["kind"] == "accepted"
            assert late["frames"][1]["frame"]["kind"] == "metadata"
            captures.append(late)
            delayed = await run("delay")
            assert delayed["ttft_submitted_ns"] >= 40_000_000
            captures.append(delayed)
            for identity, expected in (
                ("death", "transport_error"),
                ("clock", "capture_error"),
                ("count", "capture_error"),
                ("truncated", "stream_error"),
                ("timeout", "timeout"),
            ):
                record = await run(
                    identity, timeout=0.1 if identity == "timeout" else 5
                )
                assert record["outcome"] == expected, record
                captures.append(record)
            partial.clear()
            task = asyncio.create_task(run("cancel"))
            await partial.wait()
            task.cancel()
            cancelled = await task
            assert cancelled["outcome"] == "cancelled"
            captures.append(cancelled)
            response = await client.post(
                url + "/v1/pr-evidence/generate", json={"request_id": "pressure"}
            )
            await pressure_started.wait()
            await asyncio.sleep(0.05)
            assert not pressure_done.is_set()
            unaffected = await run("pressure-neighbor")
            assert unaffected["outcome"] == "complete" and not pressure_done.is_set()
            captures.append(unaffected)
            received = await response.read()
            assert pressure_done.is_set() and received.count(b'"kind": "tokens"') == 32
            response.release()
    finally:
        release.set()
        await runner.cleanup()
    return {
        "captures": captures,
        "parallel_barrier_requests": 3,
        "backpressure_bytes": len(received),
        "blocked_reader_did_not_block_neighbor": True,
        "injected_delay_ns": 40_000_000,
    }


async def _endpoint_http() -> dict[str, Any]:
    from types import SimpleNamespace

    from wait_hint_offload.evidence_protocol import EvidenceRequest
    from wait_hint_offload.policy_protocol import PolicySession
    from wait_hint_offload.vllm_adapter.evidence_checks import FixtureEngine
    from wait_hint_offload.vllm_adapter.evidence_endpoint import EvidenceEndpoint

    aiohttp = importlib.import_module("aiohttp")
    fastapi = importlib.import_module("fastapi")
    uvicorn = importlib.import_module("uvicorn")
    app = fastapi.FastAPI()
    endpoint = EvidenceEndpoint()
    endpoint.attach_router(app)
    engine = FixtureEngine()
    await endpoint.init_state(
        engine, app.state, SimpleNamespace(api_key=["qualification"])
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("fixture endpoint exited during startup")
                await asyncio.sleep(0.01)
        url = f"http://127.0.0.1:{port}"
        request = EvidenceRequest(
            "endpoint", (1, 2, 3), 3, PolicySession("fixture", "s", "i", 1), 12000
        )
        async with aiohttp.ClientSession(
            headers={"Authorization": "Bearer qualification"}
        ) as client:
            now = time.monotonic_ns()
            capture = await request_stream(
                client, url, request.to_dict(), now, now + 5_000_000_000
            )
            assert capture["outcome"] == "complete" and capture["token_ids"] == [
                10,
                11,
                12,
            ]
            async with client.post(
                url + "/v1/pr-evidence/generate",
                json=request.to_dict() | {"unknown": True},
            ) as response:
                assert response.status == 400
            async with client.post(
                url + "/v1/pr-evidence/generate",
                json=request.to_dict(),
                headers={"Authorization": "Bearer wrong"},
            ) as response:
                assert response.status == 401
        assert len(engine.requests) == 1 and engine.requests[0]["prompt"][
            "prompt_token_ids"
        ] == [1, 2, 3]
        assert engine.requests[0]["kv_hints"].actions[0].payload["ttl_ms"] == 12000
        return {
            "capture": capture,
            "unknown_fields_refused": True,
            "authentication_required": True,
            "native_sampling_and_hint_passthrough": True,
        }
    finally:
        setattr(server, "should_exit", True)
        await task
        listener.close()


def observe_instrument(directory: Path) -> dict[str, Any]:
    from wait_hint_offload.vllm_adapter.evidence_native_checks import observe_events

    directory.mkdir(parents=True, exist_ok=False)
    events = observe_events(directory / "events")
    sockets = asyncio.run(_sockets())
    endpoint = asyncio.run(_endpoint_http())
    resources = sample_process_tree(os.getpid())
    assert str(os.getpid()) in resources["processes"] and resources["pss_bytes"] > 0
    paths = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.jsonl"))
    }
    return {
        "schema": "pr-instrument-observation-v2",
        "scope": "CPU native manager and real loopback HTTP; no GPU transfer qualification",
        "events": events,
        "sockets": sockets,
        "endpoint": endpoint,
        "resources": resources,
        "journals": paths,
    }


def verify_instrument(observed: dict[str, Any], directory: Path) -> dict[str, Any]:
    schema = observed.get("schema")
    fields = {"schema", "scope", "events", "sockets", "resources", "journals"}
    if schema == "pr-instrument-observation-v2":
        fields.add("endpoint")
    if set(observed) != fields or schema not in {
        "pr-instrument-observation-v1",
        "pr-instrument-observation-v2",
    }:
        raise ValueError("unknown instrument observation")
    if {cell["policy"] for cell in observed["events"]["cells"]} != {
        "lru",
        "arc",
        "AdviceOffARC",
        "RequestTTLARC",
        "RequestTTLLRU",
        "PostResponseARC",
    }:
        raise ValueError("missing policy instrument qualification")
    if any(not cell["eviction_identity"] for cell in observed["events"]["cells"]):
        raise ValueError("observer changed eviction identity")
    for relative, digest in observed["journals"].items():
        path = directory / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("instrument journal identity drift")
        records = read_journal(path)
        if records[-1]["event"] != "close":
            raise ValueError("instrument qualification journal not closed")
    if schema == "pr-instrument-observation-v2":
        endpoint = observed["endpoint"]
        verify_stream(endpoint["capture"])
        if endpoint["capture"]["outcome"] != "complete" or any(
            endpoint[name] is not True
            for name in (
                "unknown_fields_refused",
                "authentication_required",
                "native_sampling_and_hint_passthrough",
            )
        ):
            raise ValueError("HTTP endpoint qualification failed")
    expected = {
        "death": "transport_error",
        "clock": "capture_error",
        "count": "capture_error",
        "truncated": "stream_error",
        "timeout": "timeout",
        "cancel": "cancelled",
    }
    captures = observed["sockets"]["captures"]
    for capture in captures:
        verify_stream(capture)
    if len(captures) != 12 or any(
        record["outcome"] != expected.get(record["request_id"], "complete")
        for record in captures
    ):
        raise ValueError("socket fault outcomes do not reconcile")
    if not observed["sockets"]["blocked_reader_did_not_block_neighbor"]:
        raise ValueError("streaming instrument serialized neighbors")
    return {"schema": "pr-instrument-verification-v1", "observations": observed}
