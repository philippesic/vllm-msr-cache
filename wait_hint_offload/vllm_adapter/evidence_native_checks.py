from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu import manager

from wait_hint_offload.evidence_events import EventJournal, read_journal
from wait_hint_offload.vllm_adapter.evidence_connector import EvidenceConnector
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
