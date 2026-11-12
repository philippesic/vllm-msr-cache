from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from tools.benchmarking.probe import verify_probe


def summarize_probe(output: Path) -> dict[str, Any]:
    terminal = verify_probe(output)
    admission = json.loads((output / "admission.json").read_text())
    result: dict[str, Any] = {
        "schema": "benchmark-probe-summary-v1",
        "system": admission["system"],
        "status": terminal["status"],
        "elapsed_seconds": terminal["elapsed_seconds"],
        "claims": "feasibility only; no performance comparison or release qualification",
        "coding": None,
    }
    if admission["coding"] is None:
        return result
    admissions = sorted((output / "requests").glob("request-*.admission.json"))
    captures = sorted(
        path
        for path in (output / "requests").glob("request-*.json")
        if not path.name.endswith(".admission.json")
    )
    if not admissions or {path.name for path in captures} != {
        path.name.replace(".admission.json", ".json") for path in admissions
    }:
        raise ValueError("incomplete request capture")
    counts: Counter[str] = Counter()
    latencies: list[float] = []
    engine_first_tokens: list[float] = []
    previous_end = 0
    for index, path in enumerate(captures, 1):
        record = json.loads(path.read_text())
        original = json.loads(admissions[index - 1].read_text())
        if (
            record["schema"] != "benchmark-request-v1"
            or record["system"] != admission["system"]
            or original["generation"] != index
            or original["start_ns"] != record["start_ns"]
            or not previous_end
            <= record["start_ns"]
            <= record["received_ns"]
            <= record["forwarded_ns"]
        ):
            raise ValueError("invalid request identity or time ordering")
        previous_end = record["forwarded_ns"]
        usage = record["response"]["usage"]
        prompt, generated, cached = (
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["prompt_tokens_details"]["cached_tokens"],
        )
        if (
            any(type(value) is not int for value in (prompt, generated, cached))
            or not 0 <= cached <= prompt
            or generated < 0
            or usage["total_tokens"] != prompt + generated
        ):
            raise ValueError("invalid token accounting")
        counts.update(
            prompt_tokens=prompt, generated_tokens=generated, cached_tokens=cached
        )
        counts["compacted_turns"] += int(record["compaction"]["removed_messages"] > 0)
        acknowledgement = record["acknowledgement"]
        if acknowledgement is not None:
            if record["hint"] is None or record["hint"]["generation"] != index:
                raise ValueError("hint acknowledgement without matching publication")
            counts["hint:" + acknowledgement["decision"]] += 1
            counts["ready_hint_events"] += int(
                acknowledgement["ready_cpu_prefix_keys"] > 0
            )
        latencies.append((record["forwarded_ns"] - record["start_ns"]) / 1e9)
        first = record["response"]["metrics"]["time_to_first_token_ms"]
        if type(first) not in (int, float) or not 0 <= first < float("inf"):
            raise ValueError("invalid engine timing")
        engine_first_tokens.append(first)
    result["coding"] = {
        "instance": admission["coding"]["instance"],
        "quality": terminal["coding"]["quality"],
        "execution_exit": terminal["coding"]["execution_exit"],
        "acquisition_seconds": terminal["coding"]["elapsed_seconds"],
        "requests": len(captures),
        "counts": dict(counts),
        "request_seconds": {
            "min": min(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
            "sum": sum(latencies),
        },
        "engine_first_token_ms_median": statistics.median(engine_first_tokens),
        "limits": [
            "engine first-token timing is not client streaming TTFT",
            "cached tokens combine GPU and external reuse",
            "ready hints do not prove useful retention or avoided recomputation",
            "one exposed task does not establish representative quality",
        ],
    }
    return result
