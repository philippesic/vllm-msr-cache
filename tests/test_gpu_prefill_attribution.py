# SPDX-License-Identifier: Apache-2.0
"""Focused checks for bounded, fail-closed GPU prefill attribution."""

from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)

from benchmarks.preservation import (
    summarize_gpu_model_runner_work,
    summarize_prefill_cuda_work,
)
from semantic_offload.connector import (
    SemanticOffloadingConnector,
    SemanticOffloadingConnectorMetadata,
    SemanticWorkerMetadata,
    _prefill_cuda_terminal_ready,
    _prefill_cuda_terminal_status,
)


def _event(request_id="internal-1", actual_prompt_tokens=10, records=None):
    return {
        "request_id": request_id,
        "external_request_id": f"external-{request_id}",
        "actual_prompt_tokens_locally_executed": actual_prompt_tokens,
        "prefill_cuda_events": records or [],
    }


def _cuda_record(
    request_id="internal-1",
    *,
    batch_id=1,
    prompt_tokens=10,
    scheduled_tokens=10,
    elapsed_s=0.002,
    transfer_categories=None,
    request_ids=None,
    attribution=None,
    execution_sequence=None,
):
    categories = list(transfer_categories or [])
    ids = list(request_ids or [request_id])
    execution = list(
        attribution
        or [
            {
                "request_id": request_id,
                "external_request_id": f"external-{request_id}",
                "scheduler_batch_id": batch_id,
                "scheduled_tokens": scheduled_tokens,
                "prompt_tokens": prompt_tokens,
            }
        ]
    )
    exact = (
        len(ids) == 1 and len(execution) == 1 and execution[0]["request_id"] == ids[0]
    )
    eligible = bool(exact and prompt_tokens > 0 and prompt_tokens == scheduled_tokens)
    if not exact:
        reason = "non_singleton_or_unmatched_batch"
    elif prompt_tokens <= 0:
        reason = "decode_only_batch"
    elif prompt_tokens != scheduled_tokens:
        reason = "mixed_prompt_decode_batch"
    elif categories:
        reason = "eligible_singleton_prompt_batch_with_transfer"
    else:
        reason = "eligible_singleton_prompt_batch"
    return {
        "schema": "semantic-bounded-prefill-cuda-v3",
        "execution_sequence": execution_sequence or batch_id,
        "request_ids": ids,
        "execution_attribution": execution,
        "started_at_ns": batch_id * 100,
        "completed_at_ns": batch_id * 100 + 50,
        "host_clock_domain": "host_boot_monotonic_ns",
        "device_time_s": elapsed_s,
        "device_time_source": "cuda_event_current_model_runner_stream_elapsed",
        "transfer_free": not categories,
        "transfer_categories": categories,
        "tensor_parallel_size": 1,
        "singleton_request": len(ids) == 1 and len(execution) == 1,
        "all_scheduled_tokens_are_prompt": bool(
            exact and prompt_tokens > 0 and prompt_tokens == scheduled_tokens
        ),
        "attribution_eligible": eligible,
        "attribution_reason": reason,
        "synchronization_overhead": "end_event_synchronize; attribution_cells_only",
    }


def test_complete_chunked_prompt_cuda_work_is_summed_once():
    records = [
        _cuda_record(batch_id=1, prompt_tokens=4, scheduled_tokens=4),
        _cuda_record(batch_id=2, prompt_tokens=6, scheduled_tokens=6),
    ]

    accounting = summarize_prefill_cuda_work([_event(records=records)])

    assert accounting["available"] is True
    assert accounting["reason"] == "complete_singleton_prompt_cuda_coverage"
    assert accounting["eligible_prompt_tokens"] == 10
    assert accounting["actual_prompt_tokens_locally_executed"] == 10
    assert accounting["device_time_s"] == pytest.approx(0.004)
    assert accounting["covered_request_count"] == 1


def test_transfer_batch_is_attributed_with_overlap_category_retained():
    records = [
        _cuda_record(batch_id=1, prompt_tokens=4, scheduled_tokens=4),
        _cuda_record(
            batch_id=2,
            prompt_tokens=6,
            scheduled_tokens=6,
            transfer_categories=["demand_load"],
        ),
    ]

    accounting = summarize_prefill_cuda_work([_event(records=records)])

    assert accounting["available"] is True
    assert accounting["device_time_s"] == pytest.approx(0.004)
    assert accounting["observed_device_time_s"] == pytest.approx(0.004)
    assert accounting["prompt_token_coverage_fraction"] == pytest.approx(1.0)
    assert (
        accounting["ineligibility_reasons"][
            "eligible_singleton_prompt_batch_with_transfer"
        ]
        == 1
    )


def test_multi_request_kernel_is_counted_once_but_not_attributed():
    attribution = [
        {
            "request_id": request_id,
            "external_request_id": f"external-{request_id}",
            "scheduler_batch_id": 4,
            "scheduled_tokens": 5,
            "prompt_tokens": 5,
        }
        for request_id in ("internal-1", "internal-2")
    ]
    record = _cuda_record(
        request_ids=["internal-1", "internal-2"],
        attribution=attribution,
    )

    accounting = summarize_prefill_cuda_work(
        [
            _event("internal-1", actual_prompt_tokens=5, records=[record]),
            _event("internal-2", actual_prompt_tokens=5, records=[record]),
        ]
    )

    assert accounting["available"] is False
    assert accounting["observed_event_count"] == 1
    assert accounting["eligible_event_count"] == 0
    assert accounting["observed_device_time_s"] is None


def test_authoritative_zero_prompt_work_is_a_real_zero_not_missing():
    accounting = summarize_prefill_cuda_work(
        [_event(actual_prompt_tokens=0, records=[])]
    )

    assert accounting["available"] is True
    assert accounting["device_time_s"] == 0.0
    assert accounting["per_request"][0]["device_time_s"] == 0.0
    assert accounting["prompt_token_coverage_fraction"] is None


def test_duplicate_eligible_cuda_event_fails_closed():
    record = _cuda_record(prompt_tokens=5, scheduled_tokens=5)
    accounting = summarize_prefill_cuda_work(
        [_event(actual_prompt_tokens=10, records=[record, record])]
    )
    assert accounting == {
        "schema": "semantic-bounded-prefill-cuda-accounting-v1",
        "available": False,
        "reason": "prefill_cuda_event_duplicated",
        "device_time_s": None,
        "observed_device_time_s": None,
        "per_request": [],
    }


def test_v2_transfer_classification_mismatch_fails_closed():
    record = _cuda_record(
        prompt_tokens=10,
        scheduled_tokens=10,
        transfer_categories=["demand_load"],
    )
    record["transfer_free"] = True

    accounting = summarize_prefill_cuda_work([_event(records=[record])])

    assert accounting["available"] is False
    assert accounting["reason"] == "prefill_cuda_timing_invalid"


def test_terminal_waits_for_exact_prompt_batch_timing_metadata():
    state = {
        "admission_episodes": [
            {
                "scheduler_batch_id": 8,
                "prompt_tokens_locally_executed": 10,
            },
            {
                "scheduler_batch_id": 9,
                "prompt_tokens_locally_executed": 0,
            },
        ],
        "prefill_cuda_events": [],
    }

    assert not _prefill_cuda_terminal_ready(
        "internal-1", state, bounded_prefill_cuda_timing=True
    )
    assert _prefill_cuda_terminal_ready(
        "internal-1", state, bounded_prefill_cuda_timing=False
    )
    state["prefill_cuda_events"] = [
        _cuda_record(batch_id=8, prompt_tokens=10, scheduled_tokens=10)
    ]
    assert _prefill_cuda_terminal_ready(
        "internal-1", state, bounded_prefill_cuda_timing=True
    )


def test_terminal_missing_timing_has_bounded_unavailable_fallback():
    state = {
        "request_finished_ns": 100,
        "admission_episodes": [
            {
                "scheduler_batch_id": 8,
                "prompt_tokens_locally_executed": 10,
            }
        ],
        "prefill_cuda_events": [],
    }

    assert (
        _prefill_cuda_terminal_status(
            "internal-1",
            state,
            bounded_prefill_cuda_timing=True,
            now_ns=999_999_999,
        )
        == "waiting"
    )
    assert (
        _prefill_cuda_terminal_status(
            "internal-1",
            state,
            bounded_prefill_cuda_timing=True,
            now_ns=1_000_000_100,
        )
        == "unavailable"
    )
    assert (
        _prefill_cuda_terminal_status(
            "internal-1",
            state,
            bounded_prefill_cuda_timing=False,
            now_ns=100,
        )
        == "ready"
    )


def test_summary_rejects_prompt_batch_id_or_token_drift():
    event = _event(records=[_cuda_record(batch_id=8)])
    event["admission_episodes"] = [
        {"scheduler_batch_id": 7, "prompt_tokens_locally_executed": 10}
    ]

    accounting = summarize_prefill_cuda_work([event])

    assert accounting["available"] is False
    assert accounting["reason"] == "prefill_cuda_timing_invalid"


def test_cell_total_counts_shared_kernel_once_without_request_split():
    attribution = [
        {
            "request_id": "internal-1",
            "external_request_id": "external-internal-1",
            "scheduler_batch_id": 1,
            "scheduled_tokens": 5,
            "prompt_tokens": 5,
        },
        {
            "request_id": "internal-2",
            "external_request_id": "external-internal-2",
            "scheduler_batch_id": 1,
            "scheduled_tokens": 3,
            "prompt_tokens": 3,
        },
    ]
    record = _cuda_record(
        request_ids=["internal-1", "internal-2"],
        attribution=attribution,
        elapsed_s=0.01,
        execution_sequence=1,
    )
    events = []
    for request_id, tokens in (("internal-1", 5), ("internal-2", 3)):
        event = _event(request_id, actual_prompt_tokens=tokens, records=[record])
        event["admission_episodes"] = [
            {
                "scheduler_batch_id": 1,
                "scheduled_tokens": tokens,
                "prompt_tokens_locally_executed": tokens,
            }
        ]
        events.append(event)

    accounting = summarize_gpu_model_runner_work(events)

    assert accounting["available"] is True
    assert accounting["execution_count"] == 1
    assert accounting["device_time_s"] == pytest.approx(0.01)
    assert accounting["scheduled_tokens"] == 8
    assert accounting["prompt_tokens"] == 8


def test_cell_total_reconciles_prompt_decode_mixed_and_zero_elapsed():
    records = [
        _cuda_record(batch_id=1, elapsed_s=0.0, prompt_tokens=10),
        _cuda_record(batch_id=2, prompt_tokens=0, scheduled_tokens=1),
        _cuda_record(batch_id=3, prompt_tokens=2, scheduled_tokens=4),
    ]
    event = _event(actual_prompt_tokens=12, records=records)
    event["admission_episodes"] = [
        {
            "scheduler_batch_id": 1,
            "scheduled_tokens": 10,
            "prompt_tokens_locally_executed": 10,
        },
        {
            "scheduler_batch_id": 2,
            "scheduled_tokens": 1,
            "prompt_tokens_locally_executed": 0,
        },
        {
            "scheduler_batch_id": 3,
            "scheduled_tokens": 4,
            "prompt_tokens_locally_executed": 2,
        },
    ]

    accounting = summarize_gpu_model_runner_work([event])

    assert accounting["available"] is True
    assert accounting["batch_kind_counts"] == {
        "prompt_only": 1,
        "decode_only": 1,
        "mixed": 1,
    }
    assert accounting["scheduled_tokens"] == 15
    assert accounting["prompt_tokens"] == 12
    assert accounting["decode_tokens"] == 3
    assert accounting["device_time_s"] == pytest.approx(0.004)


def test_cell_total_missing_or_duplicate_sequence_is_unavailable():
    event = _event(actual_prompt_tokens=5, records=[])
    event["admission_episodes"] = [
        {
            "scheduler_batch_id": 1,
            "scheduled_tokens": 5,
            "prompt_tokens_locally_executed": 5,
        }
    ]
    missing = summarize_gpu_model_runner_work([event])
    assert missing["available"] is False
    assert missing["device_time_s"] is None

    record = _cuda_record(batch_id=1, prompt_tokens=5, execution_sequence=2)
    event["prefill_cuda_events"] = [record]
    noncontiguous = summarize_gpu_model_runner_work([event])
    assert noncontiguous["available"] is False
    assert noncontiguous["device_time_s"] is None
    assert noncontiguous["observed_device_time_s"] == pytest.approx(0.002)


def test_start_load_marks_every_transfer_or_splice_category(monkeypatch):
    worker = SimpleNamespace()
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_worker = SimpleNamespace(
        worker=worker,
        _unsubmitted_store_jobs=[object()],
    )
    connector._connector_metadata = SemanticOffloadingConnectorMetadata(
        load_jobs={1: object()},
        store_jobs={2: object()},
        jobs_to_flush={2},
        prefetch_load_jobs={3: object()},
        splice_jobs={4: ([1], [2])},
        execution_attribution=[
            {
                "request_id": "internal-1",
                "scheduler_batch_id": 1,
                "scheduled_tokens": 10,
                "prompt_tokens": 10,
            }
        ],
    )
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.start_load_kv",
        lambda *_args, **_kwargs: None,
    )

    connector.start_load_kv(None)

    assert worker._bounded_prefill_no_transfer is False
    assert worker._bounded_prefill_transfer_categories == [
        "demand_load",
        "speculative_load",
        "store",
        "deferred_store",
        "store_flush",
        "gpu_splice",
    ]


def test_bounded_timer_emits_tp1_singleton_prompt_record(monkeypatch):
    class FakeCudaEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 2.5

    handle = SimpleNamespace(_on_complete=None)
    worker = SimpleNamespace(_query_capture_mode=handle)
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector._bounded_prefill_cuda_timing = True
    connector.connector_worker = SimpleNamespace(
        worker=worker, _unsubmitted_store_jobs=[]
    )
    connector._connector_metadata = SemanticOffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        execution_attribution=[
            {
                "request_id": "internal-1",
                "external_request_id": "external-internal-1",
                "scheduler_batch_id": 7,
                "scheduled_tokens": 10,
                "prompt_tokens": 10,
            }
        ],
    )
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.start_load_kv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.Event",
        lambda **_kwargs: FakeCudaEvent(),
    )

    connector._install_bounded_prefill_cuda_timer()
    connector.start_load_kv(None)
    metadata = SemanticWorkerMetadata()
    runner = SimpleNamespace(
        kv_connector_output=SimpleNamespace(kv_connector_worker_meta=metadata)
    )
    handle._on_complete(runner, None)

    record = worker._bounded_prefill_cuda_events[0]
    assert record["schema"] == "semantic-bounded-prefill-cuda-v3"
    assert record["attribution_eligible"] is True
    assert record["device_time_s"] == pytest.approx(0.0025)
    assert record["execution_attribution"][0]["scheduler_batch_id"] == 7
    handle._on_complete(runner, None)
    assert len(worker._bounded_prefill_cuda_events) == 1
    connector._connector_metadata.execution_attribution = [
        {
            "request_id": "internal-1",
            "external_request_id": "external-internal-1",
            "scheduler_batch_id": 8,
            "scheduled_tokens": 1,
            "prompt_tokens": 0,
        }
    ]
    connector.start_load_kv(None)
    handle._on_complete(runner, None)
    assert len(worker._bounded_prefill_cuda_events) == 2
    assert worker._bounded_prefill_cuda_events[1]["attribution_reason"] == (
        "decode_only_batch"
    )


def test_bounded_timer_skips_synthetic_warmup_without_scheduler_attribution(
    monkeypatch,
):
    handle = SimpleNamespace(_on_complete=None)
    worker = SimpleNamespace(_query_capture_mode=handle)
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector._bounded_prefill_cuda_timing = True
    connector.connector_worker = SimpleNamespace(
        worker=worker, _unsubmitted_store_jobs=[]
    )
    connector._connector_metadata = SemanticOffloadingConnectorMetadata(
        load_jobs={}, store_jobs={}, execution_attribution=[]
    )
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.start_load_kv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.Event",
        lambda **_kwargs: pytest.fail("warmup must not create a CUDA timing event"),
    )

    connector._install_bounded_prefill_cuda_timer()
    connector.start_load_kv(None)
    metadata = SemanticWorkerMetadata()
    handle._on_complete(SimpleNamespace(kv_connector_output=None), None)

    assert metadata.prefill_cuda_events == []


def test_bounded_timer_does_not_record_completion_before_success_callback(
    monkeypatch,
):
    class FakeCudaEvent:
        record_count = 0
        synchronize_count = 0

        def record(self):
            type(self).record_count += 1

        def synchronize(self):
            type(self).synchronize_count += 1

        def elapsed_time(self, _other):
            return 2.5

    handle = SimpleNamespace(_on_complete=None)
    worker = SimpleNamespace(_query_capture_mode=handle)
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector._bounded_prefill_cuda_timing = True
    connector.connector_worker = SimpleNamespace(
        worker=worker, _unsubmitted_store_jobs=[]
    )
    connector._connector_metadata = SemanticOffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        execution_attribution=[
            {
                "request_id": "internal-1",
                "external_request_id": "external-internal-1",
                "scheduler_batch_id": 7,
                "scheduled_tokens": 10,
                "prompt_tokens": 10,
            }
        ],
    )
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.start_load_kv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.Event",
        lambda **_kwargs: FakeCudaEvent(),
    )
    connector._install_bounded_prefill_cuda_timer()
    connector.start_load_kv(None)

    assert FakeCudaEvent.record_count == 1
    assert FakeCudaEvent.synchronize_count == 0
    assert worker._bounded_prefill_inflight_cuda_timing is not None


def test_bounded_timer_none_return_queues_event_for_post_forward_metadata(monkeypatch):
    class FakeCudaEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 2.5

    handle = SimpleNamespace(_on_complete=None)
    worker = SimpleNamespace(_query_capture_mode=handle)
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector._bounded_prefill_cuda_timing = True
    connector.connector_worker = SimpleNamespace(
        worker=worker, _unsubmitted_store_jobs=[]
    )
    connector._connector_metadata = SemanticOffloadingConnectorMetadata(
        load_jobs={},
        store_jobs={},
        execution_attribution=[
            {
                "request_id": "internal-1",
                "external_request_id": "external-internal-1",
                "scheduler_batch_id": 7,
                "scheduled_tokens": 10,
                "prompt_tokens": 10,
            }
        ],
    )
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.start_load_kv",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.is_available", lambda: True
    )
    monkeypatch.setattr(
        "semantic_offload.connector.torch.cuda.Event",
        lambda **_kwargs: FakeCudaEvent(),
    )
    connector._install_bounded_prefill_cuda_timer()
    connector.start_load_kv(None)

    handle._on_complete(SimpleNamespace(kv_connector_output=None), None)

    assert worker._bounded_prefill_inflight_cuda_timing is None
    assert len(worker._bounded_prefill_cuda_events) == 1
    assert worker._bounded_prefill_cuda_events[0]["execution_sequence"] == 1


def test_post_forward_worker_metadata_drains_cell_timing_event(monkeypatch):
    record = _cuda_record(batch_id=1, execution_sequence=1)
    worker = SimpleNamespace(
        _bounded_prefill_cuda_events=[record],
        _allocator_peak_stats={},
        _rank_ema_mode="default",
        _ema_rank_weight_floor=0.0,
        pop_pending_score_metadata=lambda: ({}, {}, None),
    )
    connector = SemanticOffloadingConnector.__new__(SemanticOffloadingConnector)
    connector.connector_worker = SimpleNamespace(worker=worker)
    connector._score_group_size = 1
    monkeypatch.setattr(
        "semantic_offload.connector.OffloadingConnector.build_connector_worker_meta",
        lambda *_args, **_kwargs: OffloadingWorkerMetadata(),
    )

    metadata = connector.build_connector_worker_meta()

    assert metadata.prefill_cuda_events == [record]
    assert worker._bounded_prefill_cuda_events == []
