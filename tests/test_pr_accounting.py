from pathlib import Path
from typing import Any

import pytest

from tools.benchmarking.pr_events import reconcile_events, request_sources
from wait_hint_offload.evidence_events import EventJournal


def test_actual_compute_overrides_offered_and_restored_tokens() -> None:
    request = {
        "prompt_tokens": 16,
        "scheduled": [{"start": 8, "tokens": 8}, {"start": 4, "tokens": 12}],
        "allocations": [
            {
                "cpu_tokens": 4,
                "supplied_gpu_tokens": 4,
                "load_jobs": [1],
                "load_ranges": [{"start": 4, "end": 8, "group": 0}],
            }
        ],
    }
    result = request_sources(request, 12, True, {"1"}, set())
    assert result["gpu_reused_tokens"] == 4
    assert result["cpu_reused_tokens"] == 0
    assert result["recomputed_tokens"] == 8
    assert result["repeated_prompt_compute_tokens"] == 8
    request["scheduled"] = [{"start": 8, "tokens": 8}]
    result = request_sources(request, 12, True, {"1"}, set())
    assert result["cpu_reused_tokens"] == 4
    assert result["gpu_reused_tokens"] == 4
    assert result["recomputed_tokens"] == 4
    with pytest.raises(ValueError, match="unacknowledged"):
        request_sources(request, 12, True, set(), set())
    request["scheduled"] = []
    result = request_sources(request, 12, False, {"1"}, {"1"})
    assert result["unobserved_tokens"] == 12
    assert result["successful_restored_tokens"] == 0


def _event_fixture(directory: Path, corruption: str | None) -> None:
    scheduler = EventJournal(directory, "scheduler")
    worker = EventJournal(directory, "worker")
    keys = [
        {"key": "aa", "group": 0, "position": 0},
        {"key": "bb", "group": 1, "position": 0},
    ]
    scheduler.emit("cpu_config", {"num_chunks": 2, "kv_bytes_per_chunk": 32})
    scheduler.emit(
        "group_config",
        {
            "tokens_per_hash": 4,
            "partial_tail": False,
            "groups": [{"tokens_per_chunk": 4}, {"tokens_per_chunk": 8}],
        },
    )
    scheduler.emit(
        "request_new",
        {
            "request_id": "internal",
            "external_request_id": "client",
            "prompt_tokens": 16,
            "groups": [keys[:1], keys[1:]],
        },
    )
    scheduler.emit(
        "prepare_store",
        {
            "request_id": "internal",
            "accepted": True,
            "stored": keys,
            "evicted": [],
            "cpu_block_ids": [0, 1],
        },
    )
    scheduler.emit(
        "complete_store",
        {
            "request_id": "internal",
            "keys": keys,
            "success": True,
            "ready": ["aa", "bb"] if corruption != "ready" else ["aa"],
        },
    )
    scheduler.emit(
        "prepare_load",
        {
            "request_id": "internal",
            "keys": keys,
            "cpu_block_ids": [0, 1] if corruption != "slots" else [1, 0],
        },
    )
    scheduler.emit(
        "allocated",
        {
            "request_id": "internal",
            "cpu_tokens": 8,
            "supplied_gpu_tokens": 0,
            "load_jobs": [2],
            "load_ranges": [{"start": 0, "end": 8, **key} for key in keys],
        },
    )
    scheduler.emit(
        "scheduled",
        {
            "requests": [{"request_id": "internal", "start": 8, "tokens": 8}],
            "load_jobs": {"2": "internal"},
            "store_jobs": {"1": "internal"},
        },
    )
    scheduler.emit("complete_load", {"request_id": "internal", "keys": keys})
    scheduler.emit(
        "prepare_store",
        {
            "request_id": "internal",
            "accepted": True,
            "stored": [{"key": "cc", "group": 0}],
            "evicted": keys[:1],
            "cpu_block_ids": [0],
        },
    )
    scheduler.emit(
        "complete_store",
        {
            "request_id": "internal",
            "keys": [{"key": "cc"}],
            "success": False,
            "ready": [],
        },
    )
    scheduler.emit("request_finished", {"request_id": "internal"})
    transfer: dict[str, Any] = {
        "completed": {"1": 1, "2": 1},
        "failed": [],
        "load": {"bytes": 64, "seconds": 0.001},
        "store": {"bytes": 64, "seconds": 0.002},
    }
    worker.emit("worker_transfers", transfer)
    if corruption == "duplicate":
        worker.emit("worker_transfers", transfer)
    scheduler.close()
    worker.close()


@pytest.mark.parametrize("corruption", [None, "ready", "slots", "duplicate"])
def test_native_event_reconciliation_detects_corruption(
    tmp_path: Path, corruption: str | None
) -> None:
    _event_fixture(tmp_path, corruption)
    if corruption is not None:
        with pytest.raises(ValueError):
            reconcile_events(tmp_path)
        return
    evidence = reconcile_events(tmp_path)
    assert evidence["eligibility_alignment_tokens"] == 8
    assert evidence["external_request_ids"] == {"client": "internal"}
    assert evidence["store_failures"] == 1
    assert evidence["evictions"][-1]["key"] == "aa"
    assert evidence["occupancy"][-1]["chunks"] == 1
    result = request_sources(
        evidence["requests"]["internal"],
        8,
        True,
        set(evidence["completed_jobs"]),
        set(evidence["failed_jobs"]),
    )
    assert result["cpu_reused_tokens"] == 8
    assert result["per_group_restored_tokens"] == {"0": 8, "1": 8}
