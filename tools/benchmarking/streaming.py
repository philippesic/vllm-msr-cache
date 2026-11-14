from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any


class CaptureError(ValueError):
    pass


@dataclass
class StreamCapture:
    request_id: str
    expected_tokens: int
    scheduled_ns: int
    submitted_ns: int
    frames: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    token_times_ns: list[int] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)
    engine_timing: dict[str, Any] | None = None
    num_cached_tokens: int | None = None
    finished: bool = False
    done: bool = False
    rejected_frame: dict[str, Any] | None = None

    def accept(self, raw: object, received_ns: int) -> None:
        self.rejected_frame = {"received_ns": received_ns, "frame": raw}
        if (
            self.done
            or not isinstance(raw, dict)
            or raw.get("schema") != "pr-evidence-stream-v1"
            or raw.get("request_id") != self.request_id
        ):
            raise CaptureError("unexpected stream identity or terminal state")
        previous_ns = (
            self.frames[-1]["received_ns"] if self.frames else self.submitted_ns
        )
        if received_ns < previous_ns or self.submitted_ns < self.scheduled_ns:
            raise CaptureError("client monotonic clock or schedule regressed")
        kind = raw.get("kind")
        expected = {"schema", "kind", "request_id"}
        if kind == "accepted":
            expected |= {"server_accepted_ns"}
        elif kind in {"tokens", "metadata"}:
            expected |= {
                "server_emitted_ns",
                "token_ids",
                "text",
                "finished",
                "finish_reason",
                "num_cached_tokens",
                "engine_timing",
            }
        elif kind == "done":
            expected |= {"server_emitted_ns"}
        else:
            raise CaptureError("unknown stream frame")
        if set(raw) != expected:
            raise CaptureError("unknown stream frame fields")
        server_ns = raw.get("server_emitted_ns", raw.get("server_accepted_ns"))
        previous = self.frames[-1]["frame"] if self.frames else {}
        previous_server = previous.get(
            "server_emitted_ns", previous.get("server_accepted_ns", 0)
        )
        if type(server_ns) is not int or server_ns < previous_server:
            raise CaptureError("server monotonic clock regressed")
        if kind == "accepted":
            if self.frames:
                raise CaptureError("acceptance must be the first frame")
        elif kind in {"tokens", "metadata"}:
            ids = raw["token_ids"]
            if (
                not self.frames
                or not isinstance(ids, list)
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise CaptureError("invalid token frame")
            if bool(ids) != (kind == "tokens") or self.finished:
                raise CaptureError("token kind or ordering mismatch")
            if (
                type(raw["finished"]) is not bool
                or not isinstance(raw["text"], str)
                or (
                    raw["finish_reason"] is not None
                    and not isinstance(raw["finish_reason"], str)
                )
            ):
                raise CaptureError("invalid completion fields")
            if raw["num_cached_tokens"] is not None and (
                type(raw["num_cached_tokens"]) is not int
                or raw["num_cached_tokens"] < 0
            ):
                raise CaptureError("invalid cached-token count")
            timing = raw["engine_timing"]
            if timing is not None:
                if not isinstance(timing, dict) or set(timing) != {
                    "num_generation_tokens",
                    "num_preemptions",
                    "queued_ts",
                    "scheduled_ts",
                    "first_token_ts",
                    "last_token_ts",
                    "is_corrupted",
                }:
                    raise CaptureError("unknown engine timing fields")
                if (
                    any(
                        type(timing[key]) is not int or timing[key] < 0
                        for key in ("num_generation_tokens", "num_preemptions")
                    )
                    or type(timing["is_corrupted"]) is not bool
                ):
                    raise CaptureError("invalid engine token counts or state")
                if any(
                    type(timing[key]) not in (int, float)
                    or not math.isfinite(timing[key])
                    or timing[key] < 0
                    for key in (
                        "queued_ts",
                        "scheduled_ts",
                        "first_token_ts",
                        "last_token_ts",
                    )
                ):
                    raise CaptureError("invalid engine timestamps")
            if ids:
                self.tokens.extend(ids)
                self.token_times_ns.append(received_ns)
                self.chunk_sizes.append(len(ids))
            self.finished = raw["finished"]
            self.engine_timing = timing
            self.num_cached_tokens = raw["num_cached_tokens"]
        elif kind == "done":
            if not self.finished:
                raise CaptureError("terminal output is incomplete")
            self.done = True
        self.frames.append({"received_ns": received_ns, "frame": raw})
        self.rejected_frame = None

    def result(self, status: str, ended_ns: int, detail: str = "") -> dict[str, Any]:
        first = self.token_times_ns[0] if self.token_times_ns else None
        last = self.token_times_ns[-1] if self.token_times_ns else None
        client_itl = (
            ((last - first) / (len(self.tokens) - 1))
            if first is not None and last is not None and len(self.tokens) > 1
            else None
        )
        chunk_itl = (
            ((last - first) / (len(self.token_times_ns) - 1))
            if first is not None and last is not None and len(self.token_times_ns) > 1
            else None
        )
        engine_itl = None
        timing = self.engine_timing
        if timing is not None and timing["num_generation_tokens"] > 1:
            if timing["last_token_ts"] < timing["first_token_ts"]:
                status, detail = "capture_error", "engine monotonic clock regressed"
            else:
                engine_itl = (
                    (timing["last_token_ts"] - timing["first_token_ts"])
                    * 1e9
                    / (timing["num_generation_tokens"] - 1)
                )
        if status == "complete" and not self.done:
            raise CaptureError("stream ended without completion")
        if status == "complete" and len(self.tokens) != self.expected_tokens:
            status, detail = (
                "engine_error",
                "engine finished outside requested output budget",
            )
        if timing is not None and timing["is_corrupted"]:
            status, detail = "engine_error", "engine reported corrupted output"
        if (
            status == "complete"
            and timing is not None
            and timing["num_generation_tokens"] != len(self.tokens)
        ):
            status, detail = "capture_error", "engine and stream token counts disagree"
        return {
            "schema": "pr-stream-capture-v1",
            "request_id": self.request_id,
            "scheduled_ns": self.scheduled_ns,
            "submitted_ns": self.submitted_ns,
            "ended_ns": ended_ns,
            "submission_lateness_ns": self.submitted_ns - self.scheduled_ns,
            "outcome": status,
            "detail": detail,
            "expected_tokens": self.expected_tokens,
            "token_ids": self.tokens,
            "ttft_submitted_ns": first - self.submitted_ns
            if first is not None
            else None,
            "ttft_scheduled_ns": first - self.scheduled_ns
            if first is not None
            else None,
            "client_mean_token_itl_ns": client_itl,
            "client_mean_chunk_itl_ns": chunk_itl,
            "engine_mean_token_itl_ns": engine_itl,
            "chunk_sizes": self.chunk_sizes,
            "coalesced_chunks": sum(size > 1 for size in self.chunk_sizes),
            "engine_timing": self.engine_timing,
            "num_cached_tokens": self.num_cached_tokens,
            "frames": self.frames,
            "rejected_frame": self.rejected_frame,
        }


async def request_stream(
    client: Any, url: str, payload: dict[str, Any], scheduled_ns: int, deadline_ns: int
) -> dict[str, Any]:
    submitted_ns = time.monotonic_ns()
    capture = StreamCapture(
        payload["request_id"], payload["output_tokens"], scheduled_ns, submitted_ns
    )
    status, detail = "complete", ""
    try:
        async with asyncio.timeout(max(0, (deadline_ns - submitted_ns) / 1e9)):
            async with client.post(
                url + "/v1/pr-evidence/generate", json=payload
            ) as response:
                if response.status != 200:
                    status, detail = "http_error", (await response.text())[:4096]
                else:
                    async for line in response.content:
                        if line.startswith(b"data: "):
                            capture.accept(json.loads(line[6:]), time.monotonic_ns())
                            if len(capture.tokens) > capture.expected_tokens:
                                status, detail = (
                                    "engine_error",
                                    "output budget exceeded",
                                )
                                break
                        elif line.strip():
                            raise CaptureError("unknown SSE line")
                    if status == "complete" and not capture.done:
                        status, detail = "stream_error", "missing terminal frame"
    except TimeoutError:
        status, detail = "timeout", "cell deadline reached"
    except asyncio.CancelledError:
        status, detail = "cancelled", "client cancellation"
    except (CaptureError, json.JSONDecodeError, KeyError, TypeError) as error:
        status, detail = "capture_error", str(error)
    except Exception as error:
        transport = isinstance(error, (ConnectionError, OSError)) or type(
            error
        ).__module__.startswith("aiohttp")
        status = "transport_error" if transport else "capture_error"
        detail = f"{type(error).__name__}: {error}"
    return capture.result(status, time.monotonic_ns(), detail)


def verify_stream(record: dict[str, Any]) -> None:
    if record.get("schema") != "pr-stream-capture-v1" or record.get("outcome") not in {
        "complete",
        "http_error",
        "timeout",
        "cancelled",
        "stream_error",
        "transport_error",
        "capture_error",
        "engine_error",
    }:
        raise CaptureError("unknown capture outcome")
    if any(
        type(record.get(name)) is not int
        for name in ("expected_tokens", "scheduled_ns", "submitted_ns", "ended_ns")
    ):
        raise CaptureError("invalid capture counts or times")
    if (
        not 1 <= record["expected_tokens"] <= 256
        or not 0
        <= record["scheduled_ns"]
        <= record["submitted_ns"]
        <= record["ended_ns"]
    ):
        raise CaptureError("capture clock or output budget outside bounds")
    replay = StreamCapture(
        record["request_id"],
        record["expected_tokens"],
        record["scheduled_ns"],
        record["submitted_ns"],
    )
    for frame in record["frames"]:
        if (
            set(frame) != {"received_ns", "frame"}
            or frame["received_ns"] > record["ended_ns"]
        ):
            raise CaptureError("unknown frame wrapper or late frame")
        replay.accept(frame["frame"], frame["received_ns"])
    rejected = record["rejected_frame"]
    if rejected is not None:
        if record["outcome"] != "capture_error" or set(rejected) != {
            "received_ns",
            "frame",
        }:
            raise CaptureError("unreconciled rejected frame")
        try:
            replay.accept(rejected["frame"], rejected["received_ns"])
        except CaptureError:
            pass
        else:
            raise CaptureError("rejected frame was valid")
    regenerated = replay.result(record["outcome"], record["ended_ns"], record["detail"])
    if regenerated != record:
        raise CaptureError("capture summary differs from raw frames")
