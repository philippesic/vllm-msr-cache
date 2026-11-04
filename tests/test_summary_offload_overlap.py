# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and ordering tests for the opt-in summary/offload overlap."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.v1.kv_offload.base import TransferResult, make_offload_key
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

import semantic_offload.worker as worker_module
from semantic_offload.connector import (
    SemanticTransferCompletion,
    SemanticWorkerMetadata,
    _new_prefetch_attribution_state,
    _transfer_event_record,
)
from semantic_offload.worker import SemanticOffloadingWorker


def _key(value: int):
    return make_offload_key(str(value).encode(), 0)


class _FakeEvent:
    def __init__(self, ready: bool = False):
        self.ready = ready
        self.synchronize_calls = 0
        self.recorded_stream = None

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.ready = True

    def record(self, stream) -> None:
        self.recorded_stream = stream


class _FakeStream:
    def __init__(self):
        self.waited_on = []

    def wait_stream(self, stream) -> None:
        self.waited_on.append(stream)


class _StreamContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        return False


def _make_worker() -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._summary_job_events = {}
    worker._summary_job_publications = {}
    worker._durable_summary_publications = {}
    worker._summary_job_generations = {}
    worker._summary_generation = 0
    worker._deferred_transfer_results = []
    worker.durable_summaries = {}
    worker._stack_cache_dirty = False
    worker._stack_cache_index = {}
    worker._stack_pending_insert = set()
    worker._stack_pending_remove = set()
    worker.summaries = {}
    worker._pending_job_keys = {}
    worker._pending_job_blocks = {}
    worker._pending_scores = {}
    worker._pending_score_head_counts = {}
    worker._failed_load_block_ids = set()
    worker._transfer_directions = {}
    worker._transfer_destination_blocks = {}
    worker._transfer_start_times = {}
    worker._transfer_completion_times = {}
    worker._stack_cache_keys = []
    worker._stack_cache = {}
    worker._allocator_peak_stats = {}
    return worker


def test_preservation_submit_records_worker_transfer_start(monkeypatch):
    worker = _make_worker()
    worker._preservation_events = True
    worker._summary_stream = None
    worker._build_summaries_for_blocks = Mock()
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: 10.5)
    src_spec = SimpleNamespace(block_ids=[])

    with patch.object(CPUOffloadingWorker, "submit_store", return_value=True):
        assert worker.submit_store(21, src_spec, SimpleNamespace()) is True
    with patch.object(CPUOffloadingWorker, "submit_load", return_value=True):
        assert worker.submit_load(22, SimpleNamespace(), SimpleNamespace()) is True

    assert worker._transfer_start_times == {21: 10.5, 22: 10.5}


def test_successful_transfer_event_contains_start_and_completion(monkeypatch):
    worker = _make_worker()
    worker._preservation_events = True
    worker._transfer_directions = {7: True}
    worker._transfer_start_times = {7: 10.0}
    result = SimpleNamespace(
        job_id=7,
        success=True,
        cancelled=False,
        transfer_size=128,
        transfer_time=0.25,
    )
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: 11.25)

    with patch.object(CPUOffloadingWorker, "get_finished", return_value=[result]):
        assert worker.get_finished() == [result]

    event = worker.pop_transfer_events()[7]
    assert event["start_time_s"] == 10.0
    assert event["completion_time_s"] == 11.25
    assert event["status"] == "success"
    assert event["success"] is True


@pytest.mark.parametrize(
    ("cancelled", "status"), [(False, "failure"), (True, "cancelled")]
)
def test_failed_or_cancelled_transfer_event_has_status_and_completion(
    monkeypatch, cancelled, status
):
    worker = _make_worker()
    worker._preservation_events = True
    worker._transfer_directions = {8: True}
    worker._transfer_destination_blocks = {8: {9}}
    worker._transfer_start_times = {8: 12.0}
    monkeypatch.setattr(worker_module.time, "perf_counter", lambda: 13.0)

    worker.record_transfer_failure(8, is_load=True, cancelled=cancelled)

    event = worker.pop_transfer_events()[8]
    assert event["status"] == status
    assert event["success"] is False
    assert event["cancelled"] is cancelled
    assert event["start_time_s"] == 12.0
    assert event["completion_time_s"] == 13.0
    assert event["bytes"] is None
    assert event["time_s"] is None


def test_multi_worker_transfer_aggregation_keeps_time_bounds_and_status():
    first = SemanticWorkerMetadata(
        transfer_events={
            31: SemanticTransferCompletion(
                is_load=True,
                bytes=100,
                time_s=0.5,
                max_time_s=0.5,
                contributors=1,
                success=True,
                status="success",
                start_time_s=20.0,
                completion_time_s=20.5,
            )
        }
    )
    second = SemanticWorkerMetadata(
        transfer_events={
            31: SemanticTransferCompletion(
                is_load=True,
                bytes=120,
                time_s=0.7,
                max_time_s=0.7,
                contributors=1,
                success=False,
                status="cancelled",
                cancelled=True,
                start_time_s=19.5,
                completion_time_s=21.0,
            )
        }
    )

    result = first.aggregate(second)
    event = result.transfer_events[31]
    assert event.contributors == 2
    assert event.start_time_s == 19.5
    assert event.completion_time_s == 21.0
    assert event.success is False
    assert event.cancelled is True
    assert event.status == "cancelled"


def test_model_runner_boundary_is_armed_only_after_preemption():
    worker = _make_worker()
    worker._first_forward_times_ns = {}
    worker._preempted_request_ids = set()
    worker._preempted_forward_times_ns = {}

    worker._on_model_runner_forward(["req-1"], 100)
    assert worker.pop_preempted_forward_times() == {}

    worker.receive_preempted_request_ids(["req-1"])
    worker._on_model_runner_forward(["req-1", "req-2"], 200)

    assert worker.pop_preempted_forward_times() == {"req-1": 200}
    assert worker._first_forward_times_ns["req-1"] == 100


def test_preempted_forward_metadata_aggregates_across_workers():
    first = SemanticWorkerMetadata(
        preempted_forward_times_ns={"req-1": 200},
    )
    second = SemanticWorkerMetadata(
        preempted_forward_times_ns={"req-1": 220, "req-2": 300},
    )

    result = first.aggregate(second)

    assert result.preempted_forward_times_ns == {
        "req-1": 200,
        "req-2": 300,
    }


def test_prefetch_transfer_record_preserves_request_job_identity_and_scope():
    info = {
        "req_id": "req-1",
        "is_load": True,
        "is_prefetch": True,
        "keys": [b"key"],
        "src_block_ids": [(0, 3)],
        "dst_block_ids": [(0, 7)],
        "expected_contributors": 1,
    }
    normal_info = {**info, "is_prefetch": False}
    prefetch = _transfer_event_record(41, info, None, status="failure")
    normal = _transfer_event_record(42, normal_info, None, status="failure")

    assert prefetch["request_id"] == normal["request_id"] == "req-1"
    assert prefetch["job_id"] == 41
    assert normal["job_id"] == 42
    assert prefetch["is_prefetch"] is True
    assert normal["is_prefetch"] is False
    assert prefetch["direction"] == normal["direction"] == "load"
    measured = _transfer_event_record(
        43,
        info,
        SemanticTransferCompletion(
            is_load=True,
            bytes=64,
            time_s=0.25,
            contributors=1,
            success=True,
            status="success",
        ),
    )
    assert measured["time_source"] == "cuda_event_elapsed_time"
    assert measured["time_s"] == 0.25
    unavailable = _new_prefetch_attribution_state()
    assert unavailable["status"] == "unavailable"
    assert unavailable["transfer_jobs"] == []
    assert "t0_preemption_timestamp_unavailable" in unavailable["unavailable_reasons"]


def test_overlap_does_not_publish_dma_completion_before_summary_event():
    worker = _make_worker()
    event = _FakeEvent()
    worker._summary_job_events[7] = event
    result = TransferResult(job_id=7, success=True)

    with patch.object(CPUOffloadingWorker, "get_finished", side_effect=[[result], []]):
        assert worker.get_finished() == []
        assert worker._deferred_transfer_results == [result]
        event.ready = True
        assert worker.get_finished() == [result]

    assert worker._deferred_transfer_results == []
    assert worker._summary_job_events == {}


def test_overlap_submits_copy_without_serialized_summary_wait(monkeypatch):
    worker = _make_worker()
    summary_stream = _FakeStream()
    current_stream = _FakeStream()
    event = _FakeEvent()
    worker._summary_stream = summary_stream
    worker._summary_offload_overlap = True
    worker._pending_job_blocks = {}
    worker._build_summaries_body = Mock()
    worker._durably_key_summaries = Mock()
    src_spec = SimpleNamespace(block_ids=[4])

    monkeypatch.setattr(
        worker_module.torch.cuda, "current_stream", lambda: current_stream
    )
    monkeypatch.setattr(
        worker_module.torch.cuda, "stream", lambda stream: _StreamContext(stream)
    )
    monkeypatch.setattr(worker_module.torch.cuda, "Event", lambda: event)
    with patch.object(CPUOffloadingWorker, "submit_store", return_value=True):
        assert worker.submit_store(21, src_spec, SimpleNamespace()) is True

    assert summary_stream.waited_on == [current_stream]
    assert current_stream.waited_on == []
    assert event.recorded_stream is summary_stream
    assert worker._summary_job_events == {21: event}


def test_raised_base_submit_rolls_back_after_fencing_summary_event(monkeypatch):
    worker = _make_worker()
    summary_stream = _FakeStream()
    current_stream = _FakeStream()
    event = _FakeEvent()
    key = _key(5)
    worker._summary_stream = summary_stream
    worker._summary_offload_overlap = True
    worker._pending_job_blocks = {}
    worker._build_summaries_body = Mock()

    def publish(job_id, _block_ids):
        worker.durable_summaries[key] = [object()]
        worker._register_summary_publications(job_id, [key])

    worker._durably_key_summaries = publish
    src_spec = SimpleNamespace(block_ids=[4])
    monkeypatch.setattr(
        worker_module.torch.cuda, "current_stream", lambda: current_stream
    )
    monkeypatch.setattr(
        worker_module.torch.cuda, "stream", lambda stream: _StreamContext(stream)
    )
    monkeypatch.setattr(worker_module.torch.cuda, "Event", lambda: event)

    with (
        patch.object(
            CPUOffloadingWorker,
            "submit_store",
            side_effect=RuntimeError("injected submit failure"),
        ),
        pytest.raises(RuntimeError, match="injected submit failure"),
    ):
        worker.submit_store(22, src_spec, SimpleNamespace())

    assert event.synchronize_calls == 1
    assert worker.durable_summaries == {}
    assert worker._summary_job_events == {}
    assert worker._summary_job_publications == {}
    assert worker._summary_job_generations == {}
    assert worker._stack_cache_dirty is True


def test_same_generation_failed_job_cannot_remove_replacement_publication():
    worker = _make_worker()
    key = _key(6)
    worker.durable_summaries[key] = [object()]
    worker._register_summary_publications(23, [key])
    old_publication = worker._summary_job_publications[23][key]
    worker._summary_job_generations[23] = 0

    worker.durable_summaries[key] = [object()]
    worker._register_summary_publications(24, [key])
    replacement = worker._durable_summary_publications[key]
    assert replacement != old_publication

    with patch.object(
        CPUOffloadingWorker,
        "get_finished",
        return_value=[TransferResult(job_id=23, success=False)],
    ):
        assert worker.get_finished() == [TransferResult(job_id=23, success=False)]

    assert key in worker.durable_summaries
    assert worker._durable_summary_publications[key] == replacement


def test_failed_store_removes_only_its_unpublished_summary():
    worker = _make_worker()
    failed_key, surviving_key = _key(1), _key(2)
    worker.durable_summaries = {failed_key: [object()], surviving_key: [object()]}
    worker._register_summary_publications(9, [failed_key])

    with patch.object(
        CPUOffloadingWorker,
        "get_finished",
        return_value=[TransferResult(job_id=9, success=False)],
    ):
        assert worker.get_finished() == [TransferResult(job_id=9, success=False)]

    assert set(worker.durable_summaries) == {surviving_key}
    assert worker._stack_cache_dirty is True


def test_abort_flush_waits_for_summary_fence_after_dma_wait(monkeypatch):
    worker = _make_worker()
    event = _FakeEvent()
    worker._summary_job_events[11] = event
    calls = []
    timings = []
    monkeypatch.setattr(worker_module, "_TIMING", True)
    monkeypatch.setattr(
        worker_module, "record_timing", lambda *args: timings.append(args)
    )

    def base_wait(_self, job_ids):
        calls.append(("dma", job_ids))

    with patch.object(CPUOffloadingWorker, "wait", base_wait):
        worker.wait({11})

    assert calls == [("dma", {11})]
    assert event.synchronize_calls == 1
    assert event.ready is True
    assert timings[0][0] == "summary_offload_fence_wait"


def test_reset_synchronizes_summary_stream_before_clearing_state():
    worker = _make_worker()
    stream = Mock()
    worker._summary_stream = stream
    worker.durable_summaries[_key(20)] = [object()]
    worker._pending_job_keys[1] = [_key(20)]
    worker._pending_job_blocks[1] = []
    worker._pending_scores = {"mean": {"req": []}}

    worker.reset_semantic_state()

    stream.synchronize.assert_called_once_with()
    assert worker.durable_summaries == {}
    assert worker._pending_job_keys == {}
    assert worker._pending_job_blocks == {}
    assert worker._pending_scores == {}
    assert worker._summary_generation == 1


def test_stale_failed_store_cannot_remove_post_reset_summary():
    worker = _make_worker()
    stream = Mock()
    key = _key(30)
    worker._summary_stream = stream
    worker.durable_summaries[key] = [object()]
    worker._register_summary_publications(9, [key])
    worker._summary_job_generations[9] = 0

    worker.reset_semantic_state()
    # The reset starts a new cache generation, which may legitimately publish
    # a summary for the same OffloadKey before the stale result arrives.
    worker.durable_summaries[key] = [object()]

    with patch.object(
        CPUOffloadingWorker,
        "get_finished",
        return_value=[TransferResult(job_id=9, success=False)],
    ):
        assert worker.get_finished() == [TransferResult(job_id=9, success=False)]

    assert key in worker.durable_summaries
    assert worker._summary_job_publications == {}
    assert worker._summary_job_generations == {}


def test_deferred_fence_emits_timing_counter(monkeypatch):
    worker = _make_worker()
    worker._summary_job_events[13] = _FakeEvent()
    calls = []
    monkeypatch.setattr(worker_module, "_TIMING", True)
    monkeypatch.setattr(worker_module, "record_count", lambda *args: calls.append(args))

    with patch.object(
        CPUOffloadingWorker,
        "get_finished",
        return_value=[TransferResult(job_id=13, success=True)],
    ):
        assert worker.get_finished() == []

    assert ("summary_offload_fence_deferred", 1) in calls


def test_summary_overlap_config_is_forwarded_as_opt_in():
    from semantic_offload import spec as spec_module

    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    spec = spec_module.SemanticOffloadingSpec.__new__(
        spec_module.SemanticOffloadingSpec
    )
    spec.extra_config = {
        "rank_ema_mode": "fast",
        "summary_offload_overlap": True,
    }
    spec.num_blocks = 4
    spec.vllm_config = object()

    with (
        patch.object(spec_module, "SemanticOffloadingWorker", FakeWorker),
        patch.object(spec_module, "spec_blocks_per_chunk", return_value=1),
    ):
        spec.create_worker(None)

    assert captured["rank_ema_mode"] == "fast"
    assert captured["summary_offload_overlap"] is True


def test_invalid_summary_overlap_config_is_rejected():
    from semantic_offload import spec as spec_module

    spec = spec_module.SemanticOffloadingSpec.__new__(
        spec_module.SemanticOffloadingSpec
    )
    spec.extra_config = {"summary_offload_overlap": "yes"}
    spec.num_blocks = 4
    spec.vllm_config = object()

    with pytest.raises(TypeError, match="summary_offload_overlap"):
        spec.create_worker(None)


@pytest.mark.parametrize("outcome", [True, False, RuntimeError("copy failed")])
def test_no_probe_store_uses_native_fences_without_summary_cuda(monkeypatch, outcome):
    worker = _make_worker()
    worker._probe_layer_name = None
    worker._summary_stream = object()
    worker._summary_offload_overlap = True
    worker._preservation_events = True
    worker._pending_job_blocks[21] = [(_key(1), [4])]
    worker._pending_job_keys[21] = [_key(1)]
    forbidden = Mock(side_effect=AssertionError("unused summary CUDA work"))
    for name in ("current_stream", "stream", "Event"):
        monkeypatch.setattr(worker_module.torch.cuda, name, forbidden)
    kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
    with patch.object(CPUOffloadingWorker, "submit_store", **kwargs) as native:
        if isinstance(outcome, Exception):
            with pytest.raises(RuntimeError, match="copy failed"):
                worker.submit_store(21, SimpleNamespace(block_ids=[4]), SimpleNamespace())
        else:
            assert worker.submit_store(21, SimpleNamespace(block_ids=[4]), SimpleNamespace()) is outcome
        native.assert_called_once()
    assert 21 in worker._transfer_start_times
    assert not worker._pending_job_blocks and not worker._pending_job_keys
    assert not worker._summary_job_events and not worker._summary_job_generations
    forbidden.assert_not_called()


def test_allocator_audit_can_be_disabled_without_losing_forward_boundaries(monkeypatch):
    worker = _make_worker()
    worker._first_forward_times_ns = {}
    worker._allocator_peak_stats = {}
    monkeypatch.setenv("SEMANTIC_ALLOCATOR_AUDIT", "0")
    forbidden = Mock(side_effect=AssertionError("allocator audit disabled"))
    monkeypatch.setattr(worker_module.torch.cuda, "memory_stats", forbidden)
    worker._on_model_runner_forward(["req"], 123)
    assert worker.pop_first_forward_times() == {"req": 123}
    assert worker._allocator_peak_stats == {}
    forbidden.assert_not_called()


def test_terminal_requests_drop_unresumed_preemption_markers():
    worker = _make_worker()
    worker._scoring_mode = "synchronous"
    worker._preempted_request_ids = {"still-live"}
    worker._preempted_forward_times_ns = {"just-finished": 456}
    for index in range(100):
        req = f"cancelled-{index}"
        worker.receive_preempted_request_ids([req])
        worker.retire_async_requests([req])
    worker.retire_async_requests(["just-finished"])
    assert worker._preempted_request_ids == {"still-live"}
    assert worker.pop_preempted_forward_times() == {"just-finished": 456}
