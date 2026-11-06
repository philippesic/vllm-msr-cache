from types import SimpleNamespace

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)

from semantic_offload.benchmark_metrics import (
    RELEVANCE_BATCHES,
    definitions,
    record_relevance_batch,
)
from semantic_offload.spec import SemanticOffloadingSpec


def test_disabled_is_noop():
    assert definitions({}) == {}
    record_relevance_batch(SimpleNamespace())


def test_metric_is_registered_and_counted():
    assert RELEVANCE_BATCHES in SemanticOffloadingSpec.build_metric_definitions(
        {"benchmark_telemetry": True}
    )
    stats = OffloadingConnectorStats()
    scheduler = SimpleNamespace(_benchmark_telemetry=True, _connector_stats=stats)
    record_relevance_batch(scheduler)
    record_relevance_batch(scheduler)
    assert stats._values[RELEVANCE_BATCHES][()] == 2


@pytest.mark.parametrize("value", [1, "false", None, []])
def test_bad_boolean_rejected(value):
    with pytest.raises(TypeError):
        definitions({"benchmark_telemetry": value})


def test_native_abort_receipt_only_after_real_transition(monkeypatch):
    import vllm.v1.core.sched.scheduler as module
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    events = []
    monkeypatch.setenv("VLLM_BENCHMARK_ABORT_RECEIPTS", "1")
    monkeypatch.setattr(module.logger, "info", lambda *args: events.append(args))
    request = SimpleNamespace(
        request_id="internal-random-suffix",
        status=RequestStatus.WAITING,
        kv_transfer_params={"suite_probe_id": "exact-probe"},
        output_token_ids=[1],
    )
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    queue = SimpleNamespace(remove_requests=lambda requests: None)
    scheduler = SimpleNamespace(
        requests={request.request_id: request},
        running=[],
        waiting=queue,
        skipped_waiting=queue,
    )
    freed = []
    scheduler._free_request = lambda req, **kw: freed.append(req.request_id)
    result = Scheduler.finish_requests(
        scheduler, [request.request_id], RequestStatus.FINISHED_ABORTED
    )
    assert result == [request] and freed == [request.request_id] and len(events) == 1
    import json

    event = json.loads(events[0][1])
    assert event["probe_id"] == "exact-probe" and event["status"] == "FINISHED_ABORTED"
    assert (
        Scheduler.finish_requests(
            scheduler, [request.request_id], RequestStatus.FINISHED_ABORTED
        )
        == []
    )
    assert (
        len(events) == 1
    )  # Already finished or unknown IDs cannot fabricate another receipt.
    assert (
        Scheduler.finish_requests(
            scheduler, ["unknown"], RequestStatus.FINISHED_ABORTED
        )
        == []
    )


def test_native_receipt_disabled_by_default(monkeypatch):
    import vllm.v1.core.sched.scheduler as module
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    monkeypatch.delenv("VLLM_BENCHMARK_ABORT_RECEIPTS", raising=False)
    events = []
    monkeypatch.setattr(module.logger, "info", lambda *args: events.append(args))
    req = SimpleNamespace(
        request_id="x",
        status=RequestStatus.WAITING,
        kv_transfer_params={"suite_probe_id": "p"},
        output_token_ids=[],
    )
    req.is_finished = lambda: RequestStatus.is_finished(req.status)
    queue = SimpleNamespace(remove_requests=lambda requests: None)
    scheduler = SimpleNamespace(
        requests={"x": req},
        running=[],
        waiting=queue,
        skipped_waiting=queue,
        _free_request=lambda *args, **kwargs: None,
    )
    Scheduler.finish_requests(scheduler, ["x"], RequestStatus.FINISHED_ABORTED)
    assert not events



def test_allocation_observation_reports_actual_unpinned_backing():
    import torch
    from semantic_offload.benchmark_metrics import allocation_receipt
    tensor = torch.empty((4, 8), dtype=torch.int8)
    worker = SimpleNamespace(_store_handler=SimpleNamespace(dst_tensors=[tensor]),
                             _load_handler=SimpleNamespace(src_tensors=[tensor]))
    observed = allocation_receipt(worker, num_cpu_blocks=4, kv_bytes_per_chunk=8,
        requested_cpu_bytes=35, blocks_per_chunk=1,
        model_config=SimpleNamespace(dtype=torch.float16, quantization=None))
    assert observed['available'] and observed['cpu_buffers'][0]['nbytes'] == 32
    assert observed['cpu_buffers'][0]['pinned'] is False
    assert observed['cpu_buffers'][0]['load_uses_same_storage'] is True


def test_allocation_observation_detects_different_load_backing():
    import torch
    from semantic_offload.benchmark_metrics import allocation_receipt
    left, right = torch.empty((4, 8), dtype=torch.int8), torch.empty((4, 8), dtype=torch.int8)
    worker = SimpleNamespace(_store_handler=SimpleNamespace(dst_tensors=[left]),
                             _load_handler=SimpleNamespace(src_tensors=[right]))
    observed = allocation_receipt(worker, num_cpu_blocks=4, kv_bytes_per_chunk=8,
        requested_cpu_bytes=32, blocks_per_chunk=1, model_config=SimpleNamespace())
    assert observed['available']
    assert observed['cpu_buffers'][0]['load_uses_same_storage'] is False


def test_allocation_observation_missing_buffers_is_unavailable():
    from semantic_offload.benchmark_metrics import allocation_receipt
    observed = allocation_receipt(SimpleNamespace(), num_cpu_blocks=4,
        kv_bytes_per_chunk=8, requested_cpu_bytes=32, blocks_per_chunk=1,
        model_config=SimpleNamespace())
    assert observed['available'] is False and observed['error']
    assert observed['cpu_buffers'] == []
