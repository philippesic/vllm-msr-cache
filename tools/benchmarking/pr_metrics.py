from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.benchmarking.pr_events import reconcile_events, request_sources
from tools.benchmarking.pr_workload import validate_schedule
from tools.benchmarking.streaming import verify_stream


def working_set(observed: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    ends = {
        session["session_id"]: session["ended_ns"] for session in report["sessions"]
    }
    request_sessions = {
        request["request_id"]: session["session_id"]
        for session in report["sessions"]
        for request in session["requests"]
    }
    timeline = []
    for request in observed["requests"].values():
        session = request_sessions.get(request["external_request_id"])
        if session is not None:
            timeline.append(
                (
                    request["started_ns"],
                    0,
                    session,
                    {entry["key"] for entry in request["reachable_keys"]},
                )
            )
    timeline += [(ended, 1, session, set()) for session, ended in ends.items()]
    active: dict[str, set[str]] = {}
    previous = report["started_ns"]
    end = report["scheduled_end_ns"]
    integral = peak = largest = 0
    overlap_integral = 0
    for timestamp, _, session, keys in sorted(timeline, key=lambda entry: entry[:3]):
        timestamp = min(end, max(previous, timestamp))
        union = set().union(*active.values()) if active else set()
        integral += len(union) * (timestamp - previous)
        overlap_integral += len(active) * (timestamp - previous)
        peak = max(peak, len(union))
        if keys:
            active[session] = keys
            largest = max(largest, len(keys))
        else:
            active.pop(session, None)
        previous = timestamp
    integral += len(set().union(*active.values()) if active else set()) * (
        end - previous
    )
    overlap_integral += len(active) * (end - previous)
    chunk_bytes = observed["cpu_config"]["kv_bytes_per_chunk"]
    return {
        "mean_unique_reachable_bytes": integral
        / (end - report["started_ns"])
        * chunk_bytes,
        "peak_unique_reachable_bytes": peak * chunk_bytes,
        "largest_observed_conversation_bytes": largest * chunk_bytes,
        "mean_active_prefixes": overlap_integral / (end - report["started_ns"]),
        "definition": "current submitted prompt's largest reusable native boundary, window/recurrent reachable keys, unique physical-key union while session active; no future turns, independent of cache readiness",
    }


def measure_cell(
    schedule: dict[str, Any], client: Path, events: Path
) -> dict[str, Any]:
    validate_schedule(schedule)
    report = json.loads((client / "report.json").read_text())
    observed = reconcile_events(events)
    expected_sessions = {
        session["session_id"]
        for session in schedule["sessions"] + schedule["one_shots"]
    }
    terminal_sessions = {
        session["session_id"]: session for session in report["sessions"]
    }
    if set(terminal_sessions) != expected_sessions or len(terminal_sessions) != len(
        report["sessions"]
    ):
        raise ValueError("session completion denominators do not reconcile")
    rows = []
    captures = set()
    for session in schedule["sessions"] + schedule["one_shots"]:
        one_shot = "turns" not in session
        terminal = terminal_sessions[session["session_id"]]
        turns = [session] if one_shot else session["turns"]
        offered = {request["request_id"] for request in terminal["requests"]}
        unoffered = set(terminal["not_offered_request_ids"])
        planned = {
            f"{session['session_id']}-t{index:02d}" for index in range(len(turns))
        }
        if offered & unoffered or offered | unoffered != planned:
            raise ValueError("offered and unoffered requests do not reconcile")
        for index, turn in enumerate(turns):
            request_id = f"{session['session_id']}-t{index:02d}"
            lcp = 0 if one_shot else turn["previous_request_lcp_tokens"]
            alignment = observed["eligibility_alignment_tokens"]
            eligible = min(lcp, len(turn["prompt_tokens"]) - 1) // alignment * alignment
            capture = None
            if request_id in offered:
                capture = json.loads(
                    (client / "requests" / (request_id + ".json")).read_text()
                )
                verify_stream(capture)
                captures.add(request_id + ".json")
            status = capture["outcome"] if capture else "not_offered"
            internal = observed["external_request_ids"].get(request_id)
            if status == "complete" and internal is None:
                raise ValueError("complete request lacks native event identity")
            sources = (
                request_sources(
                    observed["requests"][internal],
                    eligible,
                    status == "complete",
                    set(observed["completed_jobs"]),
                    set(observed["failed_jobs"]),
                )
                if internal is not None
                else None
            )
            rows.append(
                {
                    "request_id": request_id,
                    "session_id": session["session_id"],
                    "task": None if one_shot else session["task"],
                    "task_occurrence": 1 if one_shot else session["task_occurrence"],
                    "stratum": "one_shot" if one_shot else session["stratum"],
                    "cohort": "one_shot"
                    if one_shot
                    else "first_turn"
                    if index == 0
                    else "hinted_return"
                    if session["adopter"]
                    else "unhinted_return",
                    "outcome": status,
                    "eligible_tokens": eligible,
                    "exact_lcp_tokens": lcp,
                    "lost_alignment_tokens": lcp - eligible,
                    "sources": sources,
                    "ttft_ns": capture["ttft_scheduled_ns"] if capture else None,
                    "client_itl_ns": capture["client_mean_token_itl_ns"]
                    if capture
                    else None,
                    "engine_itl_ns": capture["engine_mean_token_itl_ns"]
                    if capture
                    else None,
                    "submission_lateness_ns": capture["submission_lateness_ns"]
                    if capture
                    else None,
                    "scheduled_ns": capture["scheduled_ns"] if capture else None,
                    "ended_ns": capture["ended_ns"] if capture else None,
                }
            )
    if captures != {path.name for path in (client / "requests").glob("*.json")}:
        raise ValueError("unexpected or missing request captures")
    metrics = {
        "schema": "pr-cell-metrics-v1",
        "model": schedule["model"],
        "seed": schedule["seed"],
        "arm": report["arm"],
        "partition": schedule["partition"],
        "stratum_weights": schedule["stratum_weights"],
        "requests": rows,
        "sessions": report["sessions"],
        "hints": report["hints"],
        "session_failures": sum(
            session["outcome"] != "complete"
            for session in report["sessions"]
            if not session["one_shot"]
        ),
        "one_shot_failures": sum(
            session["outcome"] != "complete"
            for session in report["sessions"]
            if session["one_shot"]
        ),
        "capture_failures": sum(row["outcome"] == "capture_error" for row in rows),
        "store_refusals": observed["store_refusals"],
        "store_failures": observed["store_failures"],
        "failed_transfer_jobs": observed["failed_jobs"],
        "transfer": observed["transfer"],
        "evictions": observed["evictions"],
        "occupancy": observed["occupancy"],
        "group_config": observed["group_config"],
        "cpu_config": observed["cpu_config"],
        "scope": "finite horizon; failed and unoffered requests retained, no speed ratio from censored pairs",
    }
    if all("reachable_keys" in request for request in observed["requests"].values()):
        metrics["working_set"] = working_set(observed, report)
    return metrics
