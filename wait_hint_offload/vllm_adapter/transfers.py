from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec

from wait_hint_offload.vllm_adapter.binding import require_runtime


@dataclass
class TransferMetadata(OffloadingWorkerMetadata):
    failed_jobs: dict[int, int] = field(default_factory=dict)
    failed_blocks: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> TransferMetadata:
        if not isinstance(other, TransferMetadata):
            raise TypeError("incompatible transfer metadata")
        completed = dict(self.completed_jobs)
        failed = dict(self.failed_jobs)
        blocks = dict(self.failed_blocks)
        for job_id, count in other.completed_jobs.items():
            completed[job_id] = completed.get(job_id, 0) + count
        for job_id, count in other.failed_jobs.items():
            failed[job_id] = failed.get(job_id, 0) + count
        for job_id, ids in other.failed_blocks.items():
            if job_id in blocks and blocks[job_id] != ids:
                raise ValueError("transfer destination disagreement")
            blocks[job_id] = ids
        return TransferMetadata(
            completed,
            self.transfer_stats.aggregate(other.transfer_stats),
            failed,
            blocks,
        )


class TransferWorker(OffloadingConnectorWorker):
    def __init__(self, spec: Any, kv_cache_config: Any) -> None:
        require_runtime()
        super().__init__(spec, kv_cache_config)
        self._connector_worker_meta = TransferMetadata()
        self.destinations: dict[int, tuple[int, ...]] = {}
        self.load_errors: set[int] = set()
        self.failed_receives: set[str] = set()

    def fail_transfer(self, job_id: int) -> None:
        meta = self._connector_worker_meta
        if not isinstance(meta, TransferMetadata):
            raise RuntimeError("transfer metadata owner")
        if job_id in meta.completed_jobs or job_id in meta.failed_jobs:
            raise RuntimeError("duplicate transfer result")
        meta.failed_jobs[job_id] = 1
        destination = self.destinations.pop(job_id, ())
        if destination:
            self.load_errors.update(destination)
            meta.failed_blocks[job_id] = destination
        request_id = self._load_jobs.pop(job_id, None)
        if request_id is not None:
            self.failed_receives.add(request_id)

    def submit_stores(self) -> None:
        if self.worker is None:
            raise RuntimeError("transfer worker is not initialized")
        for job_id, source, destination in self._unsubmitted_store_jobs:
            if not self.worker.submit_store(job_id, source, destination):
                self.fail_transfer(job_id)
        self._unsubmitted_store_jobs.clear()

    def handle_preemptions(self, metadata: OffloadingConnectorMetadata) -> None:
        if self.worker is None:
            raise RuntimeError("transfer worker is not initialized")
        for job_id in metadata.jobs_to_flush or ():
            entry = metadata.store_jobs.pop(job_id, None)
            if entry is not None:
                if not isinstance(entry.src_spec, GPULoadStoreSpec):
                    raise TypeError("store source must be GPU blocks")
                self._unsubmitted_store_jobs.append(
                    (job_id, entry.src_spec, entry.dst_spec)
                )
        self.submit_stores()
        if metadata.jobs_to_flush:
            self.worker.wait(metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata) -> None:
        if self.worker is None:
            raise RuntimeError("transfer worker is not initialized")
        self.submit_stores()
        for job_id, entry in metadata.load_jobs.items():
            if not isinstance(entry.dst_spec, GPULoadStoreSpec):
                raise TypeError("load destination must be GPU blocks")
            self._load_jobs[job_id] = entry.req_id
            self.destinations[job_id] = tuple(int(i) for i in entry.dst_spec.block_ids)
            if not self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec):
                self.fail_transfer(job_id)

    def get_finished(self, _finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        if self.worker is None:
            raise RuntimeError("transfer worker is not initialized")
        received = set(self.failed_receives)
        self.failed_receives.clear()
        for transfer in self.worker.get_finished():
            job_id = transfer.job_id
            if not transfer.success:
                self.fail_transfer(job_id)
                continue
            meta = self._connector_worker_meta
            if job_id in meta.completed_jobs:
                raise RuntimeError("duplicate transfer completion")
            stats = (
                meta.transfer_stats.load
                if job_id in self._load_jobs
                else meta.transfer_stats.store
            )
            if (
                transfer.transfer_time is not None
                and transfer.transfer_size is not None
            ):
                stats.record(transfer.transfer_size, transfer.transfer_time)
            meta.mark_completed(job_id)
            self.destinations.pop(job_id, None)
            request_id = self._load_jobs.pop(job_id, None)
            if request_id is not None:
                received.add(request_id)
        received.update(self.failed_receives)
        self.failed_receives.clear()
        return set(), received

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors = self.load_errors
        self.load_errors = set()
        return errors

    def build_connector_worker_meta(self) -> TransferMetadata | None:
        meta = self._connector_worker_meta
        if not isinstance(meta, TransferMetadata):
            raise RuntimeError("transfer metadata owner")
        if not meta.completed_jobs and not meta.failed_jobs:
            return None
        self._connector_worker_meta = TransferMetadata()
        return meta
