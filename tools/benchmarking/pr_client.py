from __future__ import annotations

import asyncio
import importlib
import json
import time
from pathlib import Path
from typing import Any

from tools.benchmarking.pr_workload import ARMS, request_payload, validate_schedule
from tools.benchmarking.streaming import request_stream, verify_stream
from tools.runtime_verify import write_record
from wait_hint_offload.evidence_events import EventJournal
from wait_hint_offload.policy_protocol import (
    PolicyAction,
    PolicyEnvelope,
    PolicySession,
    validate_policy_acknowledgement,
)


async def _publish(
    client: Any, url: str, payload: dict[str, Any], ttl_ms: int, deadline_ns: int
) -> dict[str, Any]:
    action = PolicyAction(
        payload["request_id"], PolicySession.parse(payload["session"]), "hold", ttl_ms
    )
    envelope = PolicyEnvelope(payload["request_id"], (action,)).to_dict()
    started = time.monotonic_ns()
    status, detail, acknowledgement = "accepted", "", None
    try:
        async with asyncio.timeout(max(0, min(10.0, (deadline_ns - started) / 1e9))):
            async with client.post(
                url + "/v1/pr-evidence/hints", json=envelope
            ) as response:
                body = await response.text()
                if response.status != 200:
                    status, detail = "http_error", body[:4096]
                else:
                    acknowledgement = json.loads(body)
                    validate_policy_acknowledgement(
                        acknowledgement, PolicyEnvelope.parse(envelope)
                    )
    except (ValueError, TypeError, KeyError) as error:
        status, detail = "capture_error", str(error)
    except TimeoutError:
        status, detail = "timeout", "hint delivery deadline"
    except asyncio.CancelledError:
        status, detail = "cancelled", "hint cancelled at cell end"
    except Exception as error:
        status, detail = "delivery_error", f"{type(error).__name__}: {error}"
    return {
        "request_id": payload["request_id"],
        "started_ns": started,
        "ended_ns": time.monotonic_ns(),
        "outcome": status,
        "detail": detail,
        "envelope": envelope,
        "acknowledgement": acknowledgement,
    }


async def run_schedule(
    url: str,
    schedule: dict[str, Any],
    arm: str,
    directory: Path,
    *,
    api_key: str = "local-benchmark",
) -> dict[str, Any]:
    validate_schedule(schedule)
    if arm not in ARMS or not url.startswith("http://127.0.0.1:"):
        raise ValueError("unknown schedule or arm")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "requests").mkdir()
    aiohttp = importlib.import_module("aiohttp")
    journal = EventJournal(directory, "client")
    started = time.monotonic_ns()
    deadline = started + int(schedule["horizon_seconds"] * 1e9)
    hint_tasks: list[asyncio.Task[dict[str, Any]]] = []
    sessions = []
    hints = []

    async def delay(until: int) -> None:
        await asyncio.sleep(max(0.0, (until - time.monotonic_ns()) / 1e9))

    async with aiohttp.ClientSession(
        headers={"Authorization": "Bearer " + api_key},
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=None),
    ) as client:

        async def session_run(session: dict[str, Any]) -> dict[str, Any]:
            one_shot = "turns" not in session
            turn_count = 1 if one_shot else len(session["turns"])
            arrival = started + int(session["arrival_seconds"] * 1e9)
            session_deadline = min(
                deadline, arrival + int(schedule["session_deadline_seconds"] * 1e9)
            )
            scheduled = arrival
            records = []
            for index in range(turn_count):
                await delay(min(scheduled, session_deadline))
                if time.monotonic_ns() >= session_deadline:
                    break
                payload = request_payload(schedule, session, index, arm)
                journal.emit(
                    "offered",
                    {
                        "request_id": payload["request_id"],
                        "scheduled_ns": scheduled,
                        "session_id": session["session_id"],
                        "turn": index,
                    },
                )
                capture = await request_stream(
                    client,
                    url,
                    payload,
                    scheduled,
                    min(
                        session_deadline,
                        scheduled + int(schedule["request_deadline_seconds"] * 1e9),
                    ),
                )
                write_record(
                    directory / "requests" / f"{payload['request_id']}.json", capture
                )
                verify_stream(capture)
                journal.emit(
                    "request_terminal",
                    {
                        "request_id": payload["request_id"],
                        "outcome": capture["outcome"],
                    },
                )
                records.append(
                    {
                        "request_id": payload["request_id"],
                        "outcome": capture["outcome"],
                        "ended_ns": capture["ended_ns"],
                    }
                )
                if capture["outcome"] != "complete":
                    break
                if not one_shot:
                    turn = session["turns"][index]
                    if (
                        arm in {"A3p", "A4p"}
                        and session["adopter"]
                        and turn["post_response_eligible"]
                    ):
                        hint_tasks.append(
                            asyncio.create_task(
                                _publish(
                                    client, url, payload, schedule["ttl_ms"], deadline
                                )
                            )
                        )
                    scheduled = capture["ended_ns"] + int(turn["gap_seconds"] * 1e9)
            complete = len(records) == turn_count and all(
                record["outcome"] == "complete" for record in records
            )
            result = {
                "session_id": session["session_id"],
                "one_shot": one_shot,
                "scheduled_arrival_ns": arrival,
                "ended_ns": time.monotonic_ns(),
                "outcome": "complete" if complete else "failed",
                "planned_turns": turn_count,
                "offered_turns": len(records),
                "not_offered_after_failure_or_deadline": turn_count - len(records),
                "not_offered_request_ids": [
                    f"{session['session_id']}-t{index:02d}"
                    for index in range(len(records), turn_count)
                ],
                "failure_reason": None
                if complete
                else records[-1]["outcome"]
                if records and records[-1]["outcome"] != "complete"
                else "session_deadline",
                "requests": records,
            }
            journal.emit("session_terminal", result)
            return result

        tasks = [
            asyncio.create_task(session_run(session))
            for session in schedule["sessions"] + schedule["one_shots"]
        ]
        try:
            sessions = await asyncio.gather(*tasks)
            await delay(deadline)
            if hint_tasks:
                hints = await asyncio.gather(*hint_tasks)
        finally:
            for task in tasks + hint_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, *hint_tasks, return_exceptions=True)
            journal.close()
    report = {
        "schema": "pr-client-cell-v1",
        "arm": arm,
        "started_ns": started,
        "scheduled_end_ns": deadline,
        "ended_ns": time.monotonic_ns(),
        "sessions": sessions,
        "hints": hints,
        "scheduled_sessions": len(schedule["sessions"]),
        "scheduled_one_shots": len(schedule["one_shots"]),
    }
    write_record(directory / "report.json", report)
    return report
