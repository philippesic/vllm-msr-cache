from __future__ import annotations

from typing import Any, cast

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, TransferResult
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from tests.runtime.support import native_fixture
from wait_hint_offload.vllm_adapter.transfers import TransferMetadata, TransferWorker


class FakeHandler:
    def __init__(self, accepted: bool, successful: bool) -> None:
        self.accepted = accepted
        self.successful = successful
        self.pending: list[int] = []
        self.waited: set[int] = set()

    def submit_store(self, job_id: int, source: Any, destination: Any) -> bool:
        if self.accepted:
            self.pending.append(job_id)
        return self.accepted

    def submit_load(self, job_id: int, source: Any, destination: Any) -> bool:
        return self.submit_store(job_id, source, destination)

    def get_finished(self) -> list[TransferResult]:
        results = [TransferResult(i, self.successful, 64, 0.01) for i in self.pending]
        self.pending.clear()
        return results

    def wait(self, job_ids: set[int]) -> None:
        self.waited.update(job_ids)


def test_submission_and_completion_failure_never_publish_success() -> None:
    for accepted, successful in [(False, False), (True, False), (True, True)]:
        spec, _, cache = native_fixture()
        worker = TransferWorker(spec, cache)
        handler = FakeHandler(accepted, successful)
        worker.worker = cast(Any, handler)
        gpu = GPULoadStoreSpec([3], [1], [0])
        cpu = CPULoadStoreSpec([2])
        metadata = OffloadingConnectorMetadata(
            load_jobs={11: TransferJob("load", cpu, gpu)},
            store_jobs={12: TransferJob("store", gpu, cpu)},
            jobs_to_flush={12},
        )
        worker.handle_preemptions(metadata)
        worker.start_kv_transfers(metadata)
        assert worker.get_finished(set()) == (set(), {"load"})
        result = worker.build_connector_worker_meta()
        assert result is not None
        if accepted and successful:
            assert result.completed_jobs == {11: 1, 12: 1}
            assert not result.failed_jobs
            assert worker.get_block_ids_with_load_errors() == set()
        else:
            assert not result.completed_jobs
            assert result.failed_jobs == {11: 1, 12: 1}
            assert result.failed_blocks == {11: (3,)}
            assert worker.get_block_ids_with_load_errors() == {3}
            assert not worker.get_block_ids_with_load_errors()
        assert handler.waited == {12}
        assert worker.build_connector_worker_meta() is None
        assert not worker.destinations and not worker._load_jobs


def test_failure_metadata_aggregation_rejects_destination_disagreement() -> None:
    first = TransferMetadata(failed_jobs={1: 1}, failed_blocks={1: (4,)})
    second = TransferMetadata(failed_jobs={2: 1}, failed_blocks={2: (5,)})
    merged = first.aggregate(second)
    assert isinstance(merged, TransferMetadata)
    assert merged.failed_jobs == {1: 1, 2: 1}
    assert merged.failed_blocks == {1: (4,), 2: (5,)}
    with pytest.raises(ValueError, match="disagreement"):
        first.aggregate(TransferMetadata(failed_blocks={1: (6,)}))
