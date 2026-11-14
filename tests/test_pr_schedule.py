from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import pytest

from tools.benchmarking import pr_workload
from tools.benchmarking.pr_workload import (
    request_payload,
    validate_schedule,
    w1_schedule,
)


def corpus() -> tuple[dict[str, Any], list[Any]]:
    rows = []
    for index in range(3):
        turn = {
            "message_index": 1,
            "prompt_tokens": [1, 2, 3],
            "recorded_output_tokens": 4,
            "requested_output_tokens": 4,
            "output_template_prefix_changed_tokens": 0,
            "previous_request_lcp_tokens": 0,
            "previous_request_lost_suffix_tokens": 0,
            "post_response_eligible": True,
        }
        payload = json.dumps(
            {
                "turns": [
                    turn,
                    turn
                    | {
                        "message_index": 3,
                        "prompt_tokens": [1, 2, 3, 4],
                        "previous_request_lcp_tokens": 3,
                    },
                ],
                "ending": "context_limit",
                "stratum": "0-2048",
            }
        ).encode()
        metadata = {
            "repository": "fixture/repo",
            "trajectory": str(index),
            "resolved": index == 0,
            "episode_sha256": hashlib.sha256(payload).hexdigest(),
        }
        rows.append((f"task-{index}", json.dumps(metadata), zlib.compress(payload)))
    return {
        "manifest_sha256": "fixture",
        "admitted": {"D8": {"pilot": {"stratum_counts": {"0-2048": 3}}}},
    }, rows


def test_schedule_is_paired_open_loop_and_reuses_tasks_without_changing_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr_workload, "load_corpus", lambda *_: corpus())
    first = w1_schedule(Path("unused"), "D8", "pilot", 12101, 1.0, 0.1, 60.0)
    second = w1_schedule(Path("unused"), "D8", "pilot", 12101, 1.0, 0.1, 60.0)
    assert first == second
    validate_schedule(first)
    assert len(first["sessions"]) == 16 and first["task_reuses"] == 13
    assert sum(session["adopter"] for session in first["sessions"]) == 8
    assert {session["task"] for session in first["sessions"]} == {
        "task-0",
        "task-1",
        "task-2",
    }
    assert all(
        0 < turn["gap_seconds"] <= 300
        for session in first["sessions"]
        for turn in session["turns"]
    )
    adopter = next(session for session in first["sessions"] if session["adopter"])
    final = request_payload(first, adopter, 1, "A4t")
    assert final["ttl_ms"] == 12000
    assert request_payload(first, adopter, 1, "A3")["ttl_ms"] == 12000
    for arm in ("A1", "A2", "A3p", "A4p"):
        request = request_payload(first, adopter, 1, arm)
        assert request["prompt_token_ids"] == final["prompt_token_ids"]
        assert request["ttl_ms"] is None
    assert adopter["turns"][-1]["post_response_eligible"]
    changed = w1_schedule(Path("unused"), "D8", "pilot", 12101, 2.0, 0.1, 60.0)
    for left, right in zip(first["sessions"], changed["sessions"], strict=True):
        assert left["turns"] == right["turns"] and left["adopter"] == right["adopter"]
        assert left["arrival_seconds"] == 2 * right["arrival_seconds"]
    assert first["one_shots"] == changed["one_shots"]
    with pytest.raises(ValueError, match="confirmation"):
        w1_schedule(Path("unused"), "D8", "confirmation", 12101, 1.0, 0.1, 60.0)
    for altered in (
        first | {"seed": 13001},
        first | {"ttl_ms": 60000},
        first | {"horizon_seconds": 901},
        first | {"unknown": True},
    ):
        with pytest.raises(ValueError):
            validate_schedule(altered)
