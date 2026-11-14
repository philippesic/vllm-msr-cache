from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu import manager

from wait_hint_offload.evidence_events import EventJournal, read_journal
from wait_hint_offload.vllm_adapter.evidence_connector import (
    EvidenceConnector,
    reachable_keys,
)
from wait_hint_offload.vllm_adapter.evidence_manager import (
    ObservedCPUOffloadingSpec,
    ObservedManager,
)
from wait_hint_offload.vllm_adapter.policy_checks import policy_fixture, request_fixture


def observe_events(directory: Path) -> dict[str, Any]:
    trigger = getattr(base, "StoreTrigger").ON_COMPUTE
    cells = []
    for policy in (
        "lru",
        "arc",
        "AdviceOffARC",
        "RequestTTLARC",
        "RequestTTLLRU",
        "PostResponseARC",
    ):
        sequences = []
        timings = []
        for observed in (False, True):
            native, owner = policy_fixture("AdviceOffARC", capacity=4)
            if policy in ("lru", "arc"):
                native = getattr(manager, "CPUOffloadingManager")(
                    num_chunks=4, cache_policy=policy
                )
            else:
                native, owner = policy_fixture(policy, capacity=4)
            journal = (
                EventJournal(directory / policy / str(observed), "scheduler")
                if observed
                else None
            )
            active = ObservedManager(native, journal) if journal is not None else native
            samples = []
            sequence = []
            for index in range(200):
                context, _ = request_fixture(
                    owner,
                    str(index),
                    str(index),
                    1,
                    [index.to_bytes(4, "big")],
                    ttl=True,
                )
                keys = [base.make_offload_key(index.to_bytes(4, "big"), 0)]
                assert [
                    entry["key"]
                    for entry in reachable_keys(
                        owner._req_status[str(index)].req, owner.config
                    )
                ] == [key.hex() for key in keys]
                active.on_new_request(context)
                started = time.monotonic_ns()
                result = active.prepare_store(keys, context, trigger=trigger)
                assert result is not None
                sequence.append([key.hex() for key in result.evicted_keys])
                active.complete_store(
                    result.keys_to_store,
                    context,
                    trigger=trigger,
                    success=index % 5 != 0,
                )
                if index % 5 != 0:
                    active.prepare_load(keys, context)
                    active.complete_load(keys, context)
                active.on_request_finished(context)
                samples.append(time.monotonic_ns() - started)
                owner._req_status.pop(str(index))
            active.reset_cache()
            assert not native._policy.get(keys[0])
            if journal is not None:
                journal.close()
                events = read_journal(journal.path)
                stores = [
                    record["data"]
                    for record in events
                    if record["event"] == "complete_store"
                ]
                assert len(stores) == 200
                assert sum(not record["success"] for record in stores) == 40
                assert all(
                    bool(record["ready"]) == record["success"] for record in stores
                )
                loads = [
                    record for record in events if record["event"] == "prepare_load"
                ]
                assert len(loads) == 160
                assert [
                    record["data"]["evicted"]
                    for record in events
                    if record["event"] == "prepare_store"
                ] == [
                    [{"key": key, "group": 0, "position": None} for key in keys]
                    for keys in sequence
                ]
            sequences.append(sequence)
            timings.append(
                {
                    "observed": observed,
                    "samples_ns": samples,
                    "p50_ns": sorted(samples)[99],
                    "p99_ns": sorted(samples)[197],
                }
            )
        assert sequences[0] == sequences[1]
        cells.append({"policy": policy, "eviction_identity": True, "timings": timings})
    native, _ = policy_fixture("AdviceOffARC")
    spec = object.__new__(ObservedCPUOffloadingSpec)
    setattr(spec, "_manager", native)
    setattr(spec, "extra_config", {"evidence_directory": str(directory / "factory")})
    for name, value in {
        "num_chunks": 4,
        "kv_bytes_per_chunk": 1024,
        "cpu_page_size_per_worker": 1024,
        "eviction_policy": "AdviceOffARC",
        "cache_policy_module_path": "wait_hint_offload.vllm_adapter.policy",
    }.items():
        setattr(spec, name, value)
    proxy = spec.get_manager()
    assert proxy.native is native and spec.get_manager() is proxy
    assert type(proxy.lock) is type(native.lock)
    with proxy.lock:
        assert proxy._policy is native._policy
    proxy.journal.close()
    assert EvidenceConnector.__mro__[1].__name__ == "OffloadingConnector"
    return {
        "cells": cells,
        "failures_per_trace": 40,
        "loads_per_trace": 160,
        "factory_preserves_manager_and_lock": True,
    }


def observe_bytes(directory: Path) -> dict[str, bool]:
    from types import SimpleNamespace

    import torch

    from wait_hint_offload.vllm_adapter.evidence_bytes import AdmissionByteProbe

    flags = {}
    for mode in ("exact", "missing_copy", "cpu_corruption"):
        gpu = [
            torch.arange(32, dtype=torch.int8).reshape(4, 8),
            torch.arange(16, dtype=torch.int8).reshape(4, 4),
        ]
        cpu = [
            torch.zeros((3, 16), dtype=torch.int8),
            torch.zeros((3, 8), dtype=torch.int8),
        ]
        pointers = SimpleNamespace(
            group_block_counts=(3, 2),
            group_data_ref_counts=(1, 1),
            block_indices=(1, 0),
            ptrs=[gpu[0][index].data_ptr() for index in range(3)]
            + [gpu[1][index].data_ptr() for index in range(2)],
            sizes=[8, 8, 8, 4, 4],
        )
        spec = SimpleNamespace(block_ids=[0, 1, 2])

        class Native:
            def __init__(self) -> None:
                self._store_handler = SimpleNamespace(
                    blocks_per_chunk=2,
                    cpu_tensors=cpu,
                    layer_refs_per_group=[
                        [SimpleNamespace(tensor_idx=0, page_size_bytes=8)],
                        [SimpleNamespace(tensor_idx=1, page_size_bytes=4)],
                    ],
                )

            def submit_store(
                self, job_id: int, device_ptrs: Any, dst_spec: Any
            ) -> bool:
                assert job_id == 1 and device_ptrs is pointers and dst_spec is spec
                cpu[0][0, 8:].copy_(gpu[0][0])
                cpu[0][1, :8].copy_(gpu[0][1])
                cpu[0][1, 8:].copy_(gpu[0][2])
                cpu[1][2, :4].copy_(gpu[1][0])
                cpu[1][2, 4:].copy_(gpu[1][1])
                return True

            def submit_load(
                self,
                job_id: int,
                src_spec: Any,
                device_ptrs: Any,
                expedited: bool = False,
            ) -> bool:
                assert (
                    job_id == 2
                    and src_spec is spec
                    and device_ptrs is pointers
                    and expedited
                )
                assert all(
                    torch.all(values[:count] == 85)
                    for values, count in zip(gpu, (3, 2), strict=True)
                )
                gpu[0][0].copy_(cpu[0][0, 8:])
                gpu[0][1].copy_(cpu[0][1, :8])
                gpu[0][2].copy_(cpu[0][1, 8:])
                gpu[1][0].copy_(cpu[1][2, :4])
                if mode != "missing_copy":
                    gpu[1][1].copy_(cpu[1][2, 4:])
                return True

            def wait(self, job_ids: set[int]) -> None:
                assert job_ids in ({1}, {2})

            def shutdown(self) -> None:
                return None

        probe = AdmissionByteProbe(
            Native(),
            SimpleNamespace(tensors=[SimpleNamespace(tensor=value) for value in gpu]),
            (4, 9),
            directory / mode,
            synchronize=lambda: None,
        )
        assert probe.submit_store(1, pointers, spec)
        assert torch.equal(cpu[0][1, 8:], torch.arange(16, 24, dtype=torch.int8))
        if mode == "cpu_corruption":
            cpu[0][0, 8] += 1
        detected = False
        try:
            probe.submit_load(2, spec, pointers, expedited=True)
        except RuntimeError as error:
            assert "restore or provenance mismatch" in str(error)
            detected = True
        finally:
            probe.shutdown()
        assert detected == (mode != "exact")
        records = read_journal(probe.journal.path)
        restored = next(
            record["data"]["regions"]
            for record in records
            if record["event"] == "load_bytes"
        )
        assert {region["group"] for region in restored} == {4, 9}
        assert sum(region["bytes"] for region in restored) == 32
        assert all(
            region["bytes_different_from_poison"] == region["bytes"]
            for region in restored
        )
        flags[mode] = True
    return flags
