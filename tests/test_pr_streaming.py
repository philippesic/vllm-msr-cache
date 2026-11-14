from __future__ import annotations

from typing import Any

import pytest

from tools.benchmarking.streaming import CaptureError, StreamCapture
from wait_hint_offload.evidence_protocol import EvidenceRequest
from wait_hint_offload.policy_protocol import PolicySession


def frame(kind: str, **fields: Any) -> dict[str, Any]:
    return {"schema": "pr-evidence-stream-v1", "request_id": "r", "kind": kind} | fields


def token_frame(ids: list[int], count: int, finished: bool = False) -> dict[str, Any]:
    return frame(
        "tokens" if ids else "metadata",
        server_emitted_ns=200,
        token_ids=ids,
        text="text" if ids else "",
        finished=finished,
        finish_reason="length" if finished else None,
        num_cached_tokens=16,
        engine_timing={
            "num_generation_tokens": count,
            "num_preemptions": 0,
            "queued_ts": 0.5,
            "scheduled_ts": 0.7,
            "first_token_ts": 1.0,
            "last_token_ts": 1.6,
            "is_corrupted": False,
        },
    )


def test_stream_ttft_ignores_metadata_and_separates_token_chunk_itl() -> None:
    capture = StreamCapture("r", 4, 0, 100)
    capture.accept(frame("accepted", server_accepted_ns=150), 200)
    capture.accept(token_frame([], 0), 300)
    capture.accept(token_frame([1], 1), 400)
    capture.accept(token_frame([2, 3, 4], 4, True), 700)
    capture.accept(frame("done", server_emitted_ns=750), 800)
    result = capture.result("complete", 900)
    assert result["ttft_submitted_ns"] == 300
    assert result["ttft_scheduled_ns"] == 400
    assert result["submission_lateness_ns"] == 100
    assert result["client_mean_token_itl_ns"] == 100
    assert result["client_mean_chunk_itl_ns"] == 300
    assert result["engine_mean_token_itl_ns"] == pytest.approx(200000000)
    assert result["coalesced_chunks"] == 1
    assert result["token_ids"] == [1, 2, 3, 4]
    assert result["rejected_frame"] is None


def test_stream_retains_partial_outcome_and_rejects_clock_and_count_errors() -> None:
    capture = StreamCapture("r", 2, 0, 100)
    capture.accept(frame("accepted", server_accepted_ns=150), 200)
    capture.accept(token_frame([1], 1), 300)
    result = capture.result("cancelled", 400)
    assert result["token_ids"] == [1] and result["outcome"] == "cancelled"
    assert result["client_mean_token_itl_ns"] is None
    with pytest.raises(CaptureError, match="incomplete"):
        capture.accept(frame("done", server_emitted_ns=350), 400)
    assert capture.rejected_frame is not None
    with pytest.raises(CaptureError, match="clock"):
        capture.accept(token_frame([2], 2, True), 250)
    with pytest.raises(CaptureError, match="unknown stream frame fields"):
        capture.accept(token_frame([2], 2, True) | {"future": True}, 400)
    assert capture.tokens == [1]


def test_evidence_request_refuses_unbounded_or_boolean_parameters() -> None:
    request = EvidenceRequest(
        "request",
        (0, 1, 2),
        128,
        PolicySession("scope", "session", "incarnation", 1),
        12000,
    )
    raw = request.to_dict()
    assert EvidenceRequest.parse(raw) == request
    for changed in (
        raw | {"schema": "v2"},
        raw | {"extra": 1},
        raw | {"ttl_ms": True},
        raw | {"output_tokens": True},
        raw | {"output_tokens": 257},
        raw | {"prompt_token_ids": [True]},
        raw | {"prompt_token_ids": [-1]},
        raw | {"prompt_token_ids": [1] * 6144},
        raw | {"ttl_ms": 60001},
    ):
        with pytest.raises(ValueError):
            EvidenceRequest.parse(changed)


def test_early_engine_finish_is_treatment_failure_not_invalid_capture() -> None:
    capture = StreamCapture("r", 4, 0, 100)
    capture.accept(frame("accepted", server_accepted_ns=150), 200)
    capture.accept(token_frame([1], 1, True), 300)
    capture.accept(frame("done", server_emitted_ns=350), 400)
    observed = capture.result("complete", 500)
    assert observed["outcome"] == "engine_error"
    assert observed["token_ids"] == [1] and observed["expected_tokens"] == 4


def test_open_loop_requests_do_not_wait_for_other_responses() -> None:
    import asyncio
    import json as json_module
    import time
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from tools.benchmarking.streaming import request_stream

    async def scenario() -> list[dict[str, Any]]:
        arrived = asyncio.Event()
        starts: list[str] = []

        async def frames(request_id: str) -> AsyncIterator[bytes]:
            await arrived.wait()
            values = [
                frame("accepted", server_accepted_ns=0),
                token_frame([1, 2, 3], 3, True),
                frame("done", server_emitted_ns=300),
            ]
            for value in values:
                value["request_id"] = request_id
                yield ("data: " + json_module.dumps(value) + "\n\n").encode()

        class Client:
            @asynccontextmanager
            async def post(
                self, url: str, *, json: dict[str, Any]
            ) -> AsyncIterator[Any]:
                assert url.endswith("/v1/pr-evidence/generate")
                starts.append(json["request_id"])
                if len(starts) == 3:
                    arrived.set()
                yield SimpleNamespace(status=200, content=frames(json["request_id"]))

        start = time.monotonic_ns()
        records = await asyncio.gather(
            *(
                request_stream(
                    Client(),
                    "http://fixture",
                    {"request_id": str(index), "output_tokens": 3},
                    start,
                    start + 5_000_000_000,
                )
                for index in range(3)
            )
        )
        assert starts == ["0", "1", "2"]
        return records

    observed = asyncio.run(scenario())
    assert all(record["outcome"] == "complete" for record in observed)
    assert max(record["submitted_ns"] for record in observed) < min(
        record["frames"][0]["received_ns"] for record in observed
    )
