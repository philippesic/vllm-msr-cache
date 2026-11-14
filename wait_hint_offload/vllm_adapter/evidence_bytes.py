from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from wait_hint_offload.evidence_events import EventJournal


class AdmissionByteProbe:
    def __init__(
        self,
        native: Any,
        caches: Any,
        group_ids: tuple[int, ...],
        directory: Path,
        synchronize: Callable[[], None] = torch.cuda.synchronize,
    ):
        self.native = native
        self.caches = caches
        self.group_ids = group_ids
        self.synchronize = synchronize
        self.journal = EventJournal(directory, "byte-probe")
        self.stored: dict[tuple[int, int, int, int], str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def _regions(
        self, pointers: Any, cpu_spec: Any
    ) -> list[tuple[dict[str, int], torch.Tensor, torch.Tensor]]:
        handler = self.native._store_handler
        blocks_per_chunk = handler.blocks_per_chunk
        regions = []
        cpu_offset = pointer_offset = 0
        for group_index, count in enumerate(pointers.group_block_counts):
            count = int(count)
            skip = int(pointers.block_indices[group_index]) % blocks_per_chunk
            chunks = (
                (skip + count + blocks_per_chunk - 1) // blocks_per_chunk
                if count
                else 0
            )
            slots = cpu_spec.block_ids[cpu_offset : cpu_offset + chunks]
            refs = handler.layer_refs_per_group[group_index]
            if len(refs) != pointers.group_data_ref_counts[group_index]:
                raise ValueError("byte probe group reference count mismatch")
            for ref_index, ref in enumerate(refs):
                tensor_index = ref.tensor_idx
                cpu = handler.cpu_tensors[tensor_index]
                gpu = self.caches.tensors[tensor_index].tensor
                page = cpu.shape[1] // blocks_per_chunk
                if page * blocks_per_chunk != cpu.shape[1]:
                    raise ValueError("byte probe requires direct whole-page layout")
                storage = gpu.untyped_storage()
                for block in range(count):
                    logical = skip + block
                    slot = int(slots[logical // blocks_per_chunk])
                    subblock = logical % blocks_per_chunk
                    pointer_index = pointer_offset + ref_index * count + block
                    pointer = int(pointers.ptrs[pointer_index])
                    size = int(pointers.sizes[pointer_index])
                    if size != ref.page_size_bytes or size > page:
                        raise ValueError("byte probe page size mismatch")
                    offset = pointer - storage.data_ptr()
                    if offset < 0 or offset + size > storage.nbytes():
                        raise ValueError("byte probe pointer outside owned GPU storage")
                    target = torch.empty(0, dtype=torch.int8, device=gpu.device).set_(
                        storage, offset, (size,), (1,)
                    )
                    source = cpu[slot, subblock * page : subblock * page + size]
                    regions.append(
                        (
                            {
                                "group": self.group_ids[group_index],
                                "tensor": tensor_index,
                                "slot": slot,
                                "subblock": subblock,
                                "bytes": size,
                            },
                            target,
                            source,
                        )
                    )
            cpu_offset += chunks
            pointer_offset += count * len(refs)
        if cpu_offset != len(cpu_spec.block_ids) or pointer_offset != len(
            pointers.ptrs
        ):
            raise ValueError("byte probe transfer geometry did not reconcile")
        return regions

    def _identity(self, region: dict[str, int]) -> tuple[int, int, int, int]:
        return region["tensor"], region["slot"], region["subblock"], region["bytes"]

    def _digest(self, values: torch.Tensor) -> str:
        return hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()

    def submit_store(self, job_id: int, device_ptrs: Any, dst_spec: Any) -> bool:
        regions = self._regions(device_ptrs, dst_spec)
        slots = {region["slot"] for region, _, _ in regions}
        self.stored = {
            identity: digest
            for identity, digest in self.stored.items()
            if identity[1] not in slots
        }
        self.synchronize()
        expected = [gpu.cpu().clone() for _, gpu, _ in regions]
        accepted = self.native.submit_store(job_id, device_ptrs, dst_spec)
        records = []
        if accepted:
            self.native.wait({job_id})
            self.synchronize()
            for (region, _, cpu), before in zip(regions, expected, strict=True):
                equal = torch.equal(cpu, before)
                digest = self._digest(before)
                records.append(
                    region
                    | {
                        "equal": equal,
                        "source_sha256": digest,
                        "destination_sha256": self._digest(cpu),
                    }
                )
                if equal:
                    self.stored[self._identity(region)] = digest
        self.journal.emit(
            "store_bytes", {"job_id": job_id, "accepted": accepted, "regions": records}
        )
        if accepted and any(not record["equal"] for record in records):
            raise RuntimeError("admission byte probe found a store mismatch")
        return bool(accepted)

    def submit_load(
        self, job_id: int, src_spec: Any, device_ptrs: Any, expedited: bool = False
    ) -> bool:
        regions = self._regions(device_ptrs, src_spec)
        expected = [cpu.clone() for _, _, cpu in regions]
        provenance = [
            self.stored.get(self._identity(region)) == self._digest(before)
            for (region, _, _), before in zip(regions, expected, strict=True)
        ]
        for _, gpu, _ in regions:
            gpu.fill_(85)
        self.synchronize()
        accepted = self.native.submit_load(
            job_id, src_spec, device_ptrs, expedited=expedited
        )
        records = []
        if accepted:
            self.native.wait({job_id})
            self.synchronize()
            for (region, gpu, _), before, known in zip(
                regions, expected, provenance, strict=True
            ):
                after = gpu.cpu()
                records.append(
                    region
                    | {
                        "equal": torch.equal(after, before),
                        "matches_observed_store": known,
                        "bytes_different_from_poison": int((before != 85).sum().item()),
                        "source_sha256": self._digest(before),
                        "destination_sha256": self._digest(after),
                    }
                )
        self.journal.emit(
            "load_bytes",
            {
                "job_id": job_id,
                "accepted": accepted,
                "expedited": expedited,
                "regions": records,
            },
        )
        if accepted and any(
            not record["equal"]
            or not record["matches_observed_store"]
            or not record["bytes_different_from_poison"]
            for record in records
        ):
            raise RuntimeError(
                "admission byte probe found a restore or provenance mismatch"
            )
        return bool(accepted)

    def shutdown(self) -> None:
        try:
            self.native.shutdown()
        finally:
            self.journal.close()
