import json
from types import SimpleNamespace

import pytest

from benchmarks.research_checks.hybrid_qualification import analyze, audit_conversation_source
from semantic_offload.qualification import SchedulerTrace, create_scheduler_trace


def event(kind, ns, **kw):
    return dict(event=kind, ns=ns * 1_000_000, **kw)


def trace():
    return [
        event("start", 0, schema="kv-qualification-v1"),
        event("arrival", 1, request_id="a"),
        event("step", 2, waiting=[["a", "WAITING"]], scheduled_tokens=1, reclaimable=4, waiting_truncated=False),
        event("load_issued", 12, request_id="a", external_tokens=64),
        event("load_ready", 16, request_id="a"),
        event("admitted", 25, request_id="a"),
    ]


def test_separates_queue_restore_and_post_transfer_wait():
    result = analyze(trace())
    e = result["episodes"][0]
    assert e["async_residence_ms"] == 4
    assert e["arrival_to_issue_ms"] == 11
    assert e["ready_to_admission_ms"] == 9
    assert result["optimistic_direct_saving_ms"] == 4
    assert result["qualification"] == "inconclusive"


@pytest.mark.parametrize("change", [{"reclaimable": 3}, {"scheduled_tokens": 0}, {"waiting": [["a", "WAITING_FOR_REMOTE_KVS"]]}])
def test_no_fabricated_overlap(change):
    events = trace()
    events[2].update(change)
    assert analyze(events)["optimistic_direct_saving_ms"] == 0


def test_incomplete_transfer_and_missing_arrival_are_explicit():
    events = trace()[:4]
    result = analyze(events)
    assert result["completed_loads"] == 0
    assert result["incomplete_loads"] == ["a"]
    del events[1]
    assert analyze(events)["errors"]


def test_repeated_transfer_episodes_do_not_overwrite():
    events = trace() + [event("load_issued", 30, request_id="a", external_tokens=64), event("load_ready", 32, request_id="a")]
    result = analyze(events)
    assert result["completed_loads"] == 2
    assert result["episodes"][1]["candidate_samples"] == 0


def test_bad_trace_rejected():
    with pytest.raises(ValueError):
        analyze([])
    with pytest.raises(ValueError):
        analyze(trace()[::-1])


def test_trace_is_default_off_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_KV_QUALIFICATION_TRACE", raising=False)
    assert create_scheduler_trace() is None
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_KV_QUALIFICATION_TRACE", str(path))
    writer = create_scheduler_trace()
    with pytest.raises(FileExistsError):
        create_scheduler_trace()
    writer.close()
    assert json.loads(path.read_text())["schema"] == "kv-qualification-v1"


def test_trace_bounds_waiting_records_and_does_not_touch_cache(tmp_path):
    r = SimpleNamespace(request_id="a", status=SimpleNamespace(name="WAITING"))
    pool = SimpleNamespace(get_num_free_blocks=lambda: 4, num_gpu_blocks=8)
    scheduler = SimpleNamespace(waiting=[r] * 65, skipped_waiting=[], current_step=1,
                                running=[], max_num_running_reqs=2,
                                kv_cache_manager=SimpleNamespace(block_pool=pool))
    path = tmp_path / "trace.jsonl"
    writer = SchedulerTrace(path)
    writer.step(scheduler, {"x": 3})
    writer.close()
    row = json.loads(path.read_text().splitlines()[1])
    assert row["waiting_truncated"] and len(row["waiting"]) == 64
    assert row["scheduled_tokens"] == 3 and row["reclaimable"] == 4


def test_completion_times_cannot_become_return_labels():
    result = audit_conversation_source({"rows": [{"conversation": [
        {"role": "user", "timestamp": None}, {"role": "assistant", "timestamp": "2023-01-01"},
    ]}]})
    assert result["user_timestamp_count"] == 0
    assert result["assistant_timestamp_count"] == 1
    assert result["observed_session_end_count"] == 0
    assert not result["reusable_timing_labels"]


def test_native_arrival_receipt_and_close_hooks(tmp_path):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    path = tmp_path / "native.jsonl"
    writer = SchedulerTrace(path)
    request = SimpleNamespace(request_id="a", resumable=False, num_prompt_tokens=64,
                              status=RequestStatus.WAITING_FOR_REMOTE_KVS)
    scheduler = SimpleNamespace(
        _kv_qualification_trace=writer, requests={}, connector=None, log_stats=False,
        _enqueue_waiting_request=lambda r: None, finished_recving_kv_req_ids=set(),
        kv_event_publisher=None, ec_connector=None,
    )
    Scheduler.add_request(scheduler, request)
    Scheduler._update_from_kv_xfer_finished(scheduler, SimpleNamespace(finished_recving={"a"}, finished_sending=set()))
    assert scheduler.finished_recving_kv_req_ids == {"a"}
    Scheduler.shutdown(scheduler)
    assert scheduler._kv_qualification_trace is None
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["start", "arrival", "load_ready"]
