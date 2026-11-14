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


async def _loadgen_http(directory: Path) -> dict[str, Any]:
    from tools.benchmarking.pr_client import run_schedule
    from tools.benchmarking.pr_workload import ARMS, seeded_order

    aiohttp_web = importlib.import_module("aiohttp.web")
    returned = asyncio.Event()
    arrivals: list[dict[str, Any]] = []
    envelopes: list[dict[str, Any]] = []
    second_turns: set[str] = set()

    async def generate(request: Any) -> Any:
        payload = await request.json()
        identity = payload["request_id"]
        arrivals.append(payload)
        if identity.endswith("-t01"):
            second_turns.add(identity)
            if len(second_turns) == 14:
                returned.set()
        response = aiohttp_web.StreamResponse(
            headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(
            _frame(identity, "accepted", server_accepted_ns=time.monotonic_ns())
        )
        if identity == "s12101-0000-t00":
            request.transport.abort()
            return response
        count = 1 if identity == "s12101-0001-t00" else payload["output_tokens"]
        await response.write(
            _token(identity, list(range(10, 10 + count)), count, finished=True)
        )
        await response.write(
            _frame(identity, "done", server_emitted_ns=time.monotonic_ns())
        )
        await response.write_eof()
        return response

    async def publish(request: Any) -> Any:
        envelope = await request.json()
        envelopes.append(envelope)
        await returned.wait()
        return aiohttp_web.json_response(
            {
                "schema": "wait-hint-policy-envelope-ack-v1",
                "message_id": envelope["message_id"],
                "accepted_ns": time.monotonic_ns(),
                "actions": [
                    {
                        "action_id": action["action_id"],
                        "decision": "disabled",
                        "replayed": False,
                    }
                    for action in envelope["actions"]
                ],
            }
        )

    turn = {
        "message_index": 1,
        "prompt_tokens": [1, 2, 3],
        "recorded_output_tokens": 4,
        "requested_output_tokens": 4,
        "output_template_prefix_changed_tokens": 0,
        "previous_request_lcp_tokens": 0,
        "previous_request_lost_suffix_tokens": 0,
        "post_response_eligible": True,
        "gap_seconds": 0.005,
    }
    schedule = {
        "schema": "pr-w1-schedule-v1",
        "model": "D8",
        "partition": "pilot",
        "seed": 12101,
        "manifest_sha256": "CPU fixture, no model execution",
        "arrival_rate": 1.0,
        "one_shot_rate": 0.1,
        "horizon_seconds": 1.5,
        "request_deadline_seconds": 120.0,
        "session_deadline_seconds": 1.5,
        "ttl_ms": 12000,
        "sessions": [
            {
                "session_id": f"s12101-{index:04d}",
                "task": str(index),
                "repository": "fixture/repo",
                "trajectory": str(index),
                "source_resolved": False,
                "stratum": "0-2048",
                "ending": "context_limit",
                "arrival_seconds": 0.001 * (index + 1),
                "adopter": index < 8,
                "task_occurrence": 1,
                "turns": [
                    turn,
                    turn | {"message_index": 3, "previous_request_lcp_tokens": 3},
                ],
            }
            for index in range(16)
        ],
        "one_shots": [
            {
                "session_id": f"o12101-{index:06d}",
                "arrival_seconds": 0.01 + index * 0.01,
                "prompt_tokens": [4] * 2048,
                "requested_output_tokens": 128,
            }
            for index in range(2)
        ],
        "arm_order": sorted(
            ARMS, key=lambda arm: seeded_order(12101, "arm-order", arm)
        ),
        "stratum_weights": {"0-2048": 16},
        "task_reuses": 0,
    }
    app = aiohttp_web.Application()
    app.router.add_post("/v1/pr-evidence/generate", generate)
    app.router.add_post("/v1/pr-evidence/hints", publish)
    runner = aiohttp_web.AppRunner(app, access_log=None)
    await runner.setup()
    site = aiohttp_web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    cells = []
    try:
        for arm in ARMS:
            returned.clear()
            arrivals.clear()
            envelopes.clear()
            second_turns.clear()
            report = await run_schedule(url, schedule, arm, directory / arm)
            assert returned.is_set()
            assert report["ended_ns"] >= report["scheduled_end_ns"]
            assert len(report["sessions"]) == 18
            assert (
                sum(session["outcome"] == "failed" for session in report["sessions"])
                == 2
            )
            assert (
                sum(
                    session["not_offered_after_failure_or_deadline"]
                    for session in report["sessions"]
                )
                == 2
            )
            assert len(arrivals) == 32
            ttl_messages = sum(request["ttl_ms"] is not None for request in arrivals)
            assert ttl_messages == (14 if arm in {"A3", "A4t", "A5"} else 0)
            assert len(envelopes) == (12 if arm in {"A3p", "A4p"} else 0)
            assert all(hint["outcome"] == "accepted" for hint in report["hints"])
            _qualify_cell_metrics(schedule, arm, directory / arm, arrivals)
            cells.append(
                {
                    "arm": arm,
                    "failed_sessions": 2,
                    "offered_requests": len(arrivals),
                    "unoffered_later_turns": 2,
                    "ttl_messages": ttl_messages,
                    "post_messages": len(envelopes),
                    "fixed_horizon_reached": True,
                    "returns_did_not_wait_for_hint_ack": True,
                }
            )
    finally:
        returned.set()
        await runner.cleanup()
    return {
        "cells": cells,
        "scope": "hand-defined CPU fixture; synthetic transport and engine failures remain in session denominators",
    }


def _qualify_cell_metrics(
    schedule: dict[str, Any], arm: str, directory: Path, arrivals: list[dict[str, Any]]
) -> None:
    from tools.benchmarking.pr_metrics import measure_cell
    from tools.runtime_verify import write_record
    from wait_hint_offload.evidence_events import EventJournal

    write_record(directory / "schedule.json", schedule)
    scheduler = EventJournal(directory / "events", "scheduler")
    worker = EventJournal(directory / "events", "worker")
    scheduler.emit("cpu_config", {"num_chunks": 8, "kv_bytes_per_chunk": 32})
    scheduler.emit(
        "group_config",
        {
            "tokens_per_hash": 1,
            "partial_tail": False,
            "groups": [{"tokens_per_chunk": 1}],
        },
    )
    for payload in arrivals:
        identity = payload["request_id"]
        capture = json.loads(
            (directory / "requests" / (identity + ".json")).read_text()
        )
        if capture["outcome"] != "complete":
            continue
        scheduler.emit(
            "request_new",
            {
                "request_id": identity + "-internal",
                "external_request_id": identity,
                "prompt_tokens": len(payload["prompt_token_ids"]),
                "reachable_keys": [],
                "groups": [[]],
            },
        )
        start = 2 if identity.endswith("-t01") and arm != "A4t" else 0
        scheduler.emit(
            "scheduled",
            {
                "requests": [
                    {
                        "request_id": identity + "-internal",
                        "start": start,
                        "tokens": len(payload["prompt_token_ids"]) - start,
                    }
                ],
                "load_jobs": {},
                "store_jobs": {},
            },
        )
    scheduler.close()
    worker.close()
    metrics = measure_cell(schedule, directory, directory / "events")
    returns = [
        row
        for row in metrics["requests"]
        if row["cohort"] in {"hinted_return", "unhinted_return"}
        and row["outcome"] == "complete"
    ]
    assert sum(row["sources"]["recomputed_tokens"] for row in returns) == (
        28 if arm == "A4t" else 0
    )
    assert sum(row["sources"]["gpu_reused_tokens"] for row in returns) == (
        0 if arm == "A4t" else 28
    )
    assert metrics["session_failures"] == 2
    assert sum(row["outcome"] == "not_offered" for row in metrics["requests"]) == 2
    write_record(directory / "metrics.json", metrics)


def _server_configuration(directory: Path) -> dict[str, Any]:
    from tools.benchmarking.pr_jobs import screen_server_command
    from tools.benchmarking.pr_workload import ARMS

    cli = importlib.import_module("vllm.entrypoints.common.cli_args")
    argument_parser = importlib.import_module("vllm.utils.argparse_utils")
    parser = cli.make_arg_parser(argument_parser.FlexibleArgumentParser())
    admitted = []
    for model in ("D8", "HM", "HS"):
        for arm in ARMS:
            command = screen_server_command(
                Path("python"),
                directory,
                model,
                arm,
                8000,
                1,
                directory / "events",
                True,
            )
            args = parser.parse_args(command[3:])
            cli.validate_parsed_serve_args(args)
            assert args.dtype == "float16" and args.max_model_len == 6144
            assert args.mamba_cache_dtype == args.mamba_ssm_cache_dtype == "float16"
            assert args.num_gpu_blocks_override is None and args.max_num_seqs == 16
            admitted.append([model, arm])
    return {"native_argument_validation": admitted, "cuda_execution": False}


def observe_instrument(directory: Path) -> dict[str, Any]:
    from wait_hint_offload.vllm_adapter.evidence_native_checks import (
        observe_bytes,
        observe_events,
    )

    directory.mkdir(parents=True, exist_ok=False)
    events = observe_events(directory / "events")
    byte_probe = observe_bytes(directory / "bytes")
    server_configuration = _server_configuration(directory)
    sockets = asyncio.run(_sockets())
    endpoint = asyncio.run(_endpoint_http())
    loadgen = asyncio.run(_loadgen_http(directory / "loadgen"))
    resources = sample_process_tree(os.getpid())
    assert str(os.getpid()) in resources["processes"] and resources["pss_bytes"] > 0
    paths = {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.jsonl"))
    }
    return {
        "schema": "pr-instrument-observation-v7",
        "scope": "CPU native manager and real loopback HTTP; no GPU transfer qualification",
        "events": events,
        "byte_probe": byte_probe,
        "server_configuration": server_configuration,
        "sockets": sockets,
        "endpoint": endpoint,
        "loadgen": loadgen,
        "resources": resources,
        "journals": paths,
        "files": {
            str(path.relative_to(directory)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(directory.rglob("*.json"))
        },
    }


def verify_instrument(observed: dict[str, Any], directory: Path) -> dict[str, Any]:
    schema = observed.get("schema")
    fields = {"schema", "scope", "events", "sockets", "resources", "journals"}
    if schema in {
        "pr-instrument-observation-v2",
        "pr-instrument-observation-v3",
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
        fields.add("endpoint")
    if schema in {
        "pr-instrument-observation-v3",
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
        fields.add("loadgen")
    if schema in {
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
        fields.add("files")
    if schema in {
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
        fields.add("byte_probe")
        if observed["byte_probe"] != {
            "exact": True,
            "missing_copy": True,
            "cpu_corruption": True,
        }:
            raise ValueError("byte probe failed to detect injected corruption")
    if schema in {"pr-instrument-observation-v6", "pr-instrument-observation-v7"}:
        from tools.benchmarking.pr_workload import ARMS

        fields.add("server_configuration")
        if observed["server_configuration"] != {
            "native_argument_validation": [
                [model, arm] for model in ("D8", "HM", "HS") for arm in ARMS
            ],
            "cuda_execution": False,
        }:
            raise ValueError("native server configuration qualification failed")
    if set(observed) != fields or schema not in {
        "pr-instrument-observation-v1",
        "pr-instrument-observation-v2",
        "pr-instrument-observation-v3",
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
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
    for relative, digest in (
        observed["files"]
        if schema
        in {
            "pr-instrument-observation-v4",
            "pr-instrument-observation-v5",
            "pr-instrument-observation-v6",
            "pr-instrument-observation-v7",
        }
        else {}
    ).items():
        path = directory / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("instrument capture file identity drift")
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
    if schema in {
        "pr-instrument-observation-v2",
        "pr-instrument-observation-v3",
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
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
    if schema in {
        "pr-instrument-observation-v3",
        "pr-instrument-observation-v4",
        "pr-instrument-observation-v5",
        "pr-instrument-observation-v6",
        "pr-instrument-observation-v7",
    }:
        from tools.benchmarking.pr_workload import ARMS

        cells = observed["loadgen"]["cells"]
        expected_cells = [
            {
                "arm": arm,
                "failed_sessions": 2,
                "offered_requests": 32,
                "unoffered_later_turns": 2,
                "ttl_messages": 14 if arm in {"A3", "A4t", "A5"} else 0,
                "post_messages": 12 if arm in {"A3p", "A4p"} else 0,
                "fixed_horizon_reached": True,
                "returns_did_not_wait_for_hint_ack": True,
            }
            for arm in ARMS
        ]
        if cells != expected_cells:
            raise ValueError("load generator outcomes do not reconcile")
        for path in directory.glob("loadgen/*/requests/*.json"):
            verify_stream(json.loads(path.read_text()))
    if schema == "pr-instrument-observation-v7":
        from tools.benchmarking.pr_metrics import measure_cell

        for cell in directory.glob("loadgen/*"):
            schedule = json.loads((cell / "schedule.json").read_text())
            if measure_cell(schedule, cell, cell / "events") != json.loads(
                (cell / "metrics.json").read_text()
            ):
                raise ValueError(
                    "cell metrics disagree with retained scheduler and stream evidence"
                )
    return {"schema": "pr-instrument-verification-v1", "observations": observed}
