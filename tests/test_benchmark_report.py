import json
from pathlib import Path

import pytest

from tools.benchmarking.report import summarize_probe


def test_report_requires_complete_requests_and_separates_hint_from_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = {
        "status": "restoration_observed",
        "elapsed_seconds": 30,
        "coding": {"quality": "unresolved", "execution_exit": 0, "elapsed_seconds": 20},
    }
    monkeypatch.setattr("tools.benchmarking.report.verify_probe", lambda path: terminal)
    (tmp_path / "admission.json").write_text(
        json.dumps({"system": "wait-hint", "coding": {"instance": "case"}})
    )
    requests = tmp_path / "requests"
    requests.mkdir()
    with pytest.raises(ValueError, match="incomplete request capture"):
        summarize_probe(tmp_path)
    (requests / "request-0001.admission.json").write_text(
        json.dumps({"generation": 1, "start_ns": 10})
    )
    record = {
        "schema": "benchmark-request-v1",
        "system": "wait-hint",
        "start_ns": 10,
        "received_ns": 1_000_000_010,
        "forwarded_ns": 1_100_000_010,
        "response": {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "metrics": {"time_to_first_token_ms": 500},
        },
        "compaction": {"removed_messages": 2},
        "hint": {"generation": 1},
        "acknowledgement": {"decision": "refreshed", "ready_cpu_prefix_keys": 4},
    }
    capture = requests / "request-0001.json"
    capture.write_text(json.dumps(record))
    result = summarize_probe(tmp_path)["coding"]
    assert result["quality"] == "unresolved"
    assert result["requests"] == 1
    assert result["counts"]["cached_tokens"] == 0
    assert result["counts"]["hint:refreshed"] == 1
    assert result["counts"]["compacted_turns"] == 1
    assert result["request_seconds"]["median"] == 1.1
    record["received_ns"] = 5
    capture.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="time ordering"):
        summarize_probe(tmp_path)
    (requests / "request-0002.admission.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete request capture"):
        summarize_probe(tmp_path)
