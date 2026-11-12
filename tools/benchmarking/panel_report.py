from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def cell_observations(
    output: Path, job: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    measurement = result.get("measurement", {})
    observation: dict[str, Any] = {
        "model": job["model"],
        "system": job["system"],
        "case": job["case"],
        "block": job["block"],
        "kind": job["kind"],
        "status": result["status"],
        "quality": result.get("quality", "ungraded"),
        "elapsed_seconds": measurement.get("elapsed_seconds"),
        "execution_exit": measurement.get("execution_exit"),
        "exposure": job.get("exposure"),
        "source_group": job.get("source_group"),
        "source_group_previously_exposed": job.get(
            "source_group_previously_exposed", True
        ),
    }
    if "directory" not in result:
        return observation
    directory = output / result["directory"]
    if job["kind"] == "trace":
        windows = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("resource-*.json"))
        ]
        observation["resources"] = {
            "samples": len(windows),
            "complete": bool(windows) and all(window["complete"] for window in windows),
            "rss_min_bytes": min(
                (window["sum_rss_bytes"] for window in windows), default=None
            ),
            "rss_max_bytes": max(
                (window["sum_rss_bytes"] for window in windows), default=None
            ),
            "limits": "summed RSS includes shared pages; whole-device observations include all processes",
        }
        hints = json.loads((directory / "hints.json").read_text())["events"]
        states = [
            hint["acknowledgement"]["metadata"]
            for hint in hints
            if "metadata" in hint["acknowledgement"]
        ]
        observation["metadata"] = {
            "samples": len(states),
            "last": states[-1] if states else None,
            "maximum_charged_bytes": max(
                (state["charged_bytes"] for state in states), default=None
            ),
            "within_limits": bool(states)
            and all(
                0 <= state["sessions"] <= state["max_sessions"]
                and 0
                <= state["events"]
                <= state["sessions"] * state["max_events_per_session"]
                and 0 <= state["charged_bytes"] <= state["metadata_bytes_limit"]
                for state in states
            ),
        }
        observation["operating_seconds"] = measurement.get("operating_seconds")
        observation["active_seconds"] = measurement.get("active_seconds")
        observation["quiet_seconds"] = measurement.get("quiet_seconds")
        observation["activity"] = measurement.get("activity")
        observation["cpu_pressure_ratio"] = (
            measurement.get("observed_prefix_store_bytes", 0)
            * measurement.get("competing_prefixes", 0)
            / json.loads((directory / "admission.json").read_text())["cpu_bytes"]
        )
        layout = measurement.get("cache_layout")
        observation["minimum_cpu_working_set_ratio"] = (
            layout["minimum_full_prefix_chunks"]
            * measurement.get("competing_prefixes", 0)
            / layout["cpu_chunks"]
            if layout and layout["cpu_chunks"]
            else None
        )
        observation["trace_store_bytes"] = measurement.get(
            "observed_store_bytes_during_trace"
        )
        records = [
            json.loads(path.read_text())
            for path in sorted(directory.glob("request-*.json"))
        ]
        observation["requests"] = [
            {
                "label": record["label"],
                "seconds": (record["completed_ns"] - record["started_ns"]) / 1e9,
                "cached_tokens": record["response"]["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ],
            }
            for record in records
        ]
        return observation
    requests = directory / "requests"
    admissions = list(requests.glob("request-*.admission.json"))
    captures = sorted(
        path
        for path in requests.glob("request-*.json")
        if not path.name.endswith(".admission.json")
    )
    counts: Counter[str] = Counter()
    durations: list[float] = []
    for path in captures:
        record = json.loads(path.read_text())
        if "response" not in record:
            counts["request_failures"] += 1
            continue
        usage = record["response"]["usage"]
        counts.update(
            prompt_tokens=usage["prompt_tokens"],
            generated_tokens=usage["completion_tokens"],
            cached_tokens=usage["prompt_tokens_details"]["cached_tokens"],
        )
        counts["compacted_turns"] += int(record["compaction"]["removed_messages"] > 0)
        acknowledgement = record.get("acknowledgement")
        if acknowledgement is not None:
            counts["hint:" + acknowledgement["decision"]] += 1
            counts["ready_hint_events"] += int(
                (acknowledgement.get("ready_cpu_prefix_keys") or 0) > 0
            )
        durations.append((record["forwarded_ns"] - record["start_ns"]) / 1e9)
    observation.update(
        offered_requests=len(admissions),
        captured_requests=len(captures),
        unanswered_requests=len(admissions) - len(captures),
        counts=dict(counts),
        request_seconds={
            "sum": sum(durations),
            "median": statistics.median(durations) if durations else None,
        },
    )
    if "diagnostic" in result:
        observation["diagnostic_activity"] = result["diagnostic"]["activity"]
    return observation


def paired_observations(
    observations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cell = {
        (item["kind"], item["model"], item["case"], item["block"], item["system"]): item
        for item in observations.values()
    }
    pairs = []
    for key, candidate in by_cell.items():
        if candidate["system"] != "wait-hint":
            continue
        for reference in ("native-arc", "gpu-only", "lmcache", "previous-or-base"):
            control = by_cell.get((*key[:-1], reference))
            if control is None:
                continue
            comparable = candidate["status"] == control["status"] == "complete"
            before, after = control["elapsed_seconds"], candidate["elapsed_seconds"]
            pairs.append(
                {
                    "kind": candidate["kind"],
                    "model": candidate["model"],
                    "case": candidate["case"],
                    "block": candidate["block"],
                    "reference": reference,
                    "candidate_status": candidate["status"],
                    "control_status": control["status"],
                    "candidate_quality": candidate["quality"],
                    "control_quality": control["quality"],
                    "candidate_seconds": after,
                    "control_seconds": before,
                    "maximum_request_count": max(
                        candidate.get("offered_requests", 0),
                        control.get("offered_requests", 0),
                    ),
                    "quality_regression": control["quality"] == "resolved"
                    and candidate["quality"] != "resolved",
                    "complete_pair": comparable,
                    "elapsed_ratio": after / before
                    if comparable and before and after is not None
                    else None,
                    "limits": "descriptive paired observation; no population interval or tail guarantee; incomplete pairs have no speed ratio",
                }
            )
    return pairs
