from __future__ import annotations

import gc
import math
import sys
import time
from collections.abc import Callable
from typing import Any

from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu.policies import base as policy_base

from wait_hint_offload.vllm_adapter.policy import RequestTTLARC, RequestTTLLRU
from wait_hint_offload.vllm_adapter.policy_checks import policy_fixture, request_fixture


def _owned_bytes(value: Any, seen: set[int]) -> int:
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _owned_bytes(key, seen) + _owned_bytes(item, seen)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_owned_bytes(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _owned_bytes(vars(value), seen)
    return size


def _percentile(values: list[int], quantile: float) -> int | None:
    return sorted(values)[math.ceil(quantile * len(values)) - 1] if values else None


def observe_bookkeeping(progress: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    cells = []
    for cls in (RequestTTLARC, RequestTTLLRU):
        for capacity in (10000, 100000, 1000000):
            for held_sessions in (0, 100, 1000):
                gc.collect()
                native, owner = policy_fixture(cls.__name__, capacity)
                policy = native._policy
                policy.clock = lambda: 1_000_000_000
                context = base.ReqContext(req_id="fill")
                for index in range(capacity):
                    key = base.make_offload_key(index.to_bytes(32, "big"), 0)
                    chunk = getattr(policy_base, "ChunkStatus")(index)
                    chunk.ref_cnt = 0
                    policy.insert(key, chunk, context)
                per_session = min(32, capacity // 2 // max(1, held_sessions))
                hold_times = []
                for index in range(held_sessions):
                    hashes = [
                        number.to_bytes(32, "big")
                        for number in range(
                            index * per_session, (index + 1) * per_session
                        )
                    ]
                    finished, _ = request_fixture(
                        owner, str(index), str(index), 1, hashes
                    )
                    start = time.perf_counter_ns()
                    policy.on_request_finished((), set(), set(), finished)
                    hold_times.append(time.perf_counter_ns() - start)
                    del owner._req_status[finished.req_id]
                for index in range(16):
                    request_fixture(
                        owner, f"active-{index}", f"active-{index}", 1, [], False
                    )
                assert len(policy.book.leases) == held_sessions
                held_keys = frozenset(policy.book.references)
                owned = _owned_bytes(
                    (
                        policy.book,
                        policy.observations,
                        policy.key_owners,
                        policy.outcomes,
                    ),
                    set(),
                )
                evict_times = []
                for _ in range(200):
                    start = time.perf_counter_ns()
                    evicted = policy.evict(1, set())
                    evict_times.append(time.perf_counter_ns() - start)
                    assert evicted is not None and len(evicted) == 1
                    key, chunk = evicted[0]
                    assert key not in held_keys
                    policy.insert(key, chunk, context)
                assert set(policy.book.references) == held_keys
                cell = {
                    "policy": cls.__name__,
                    "chunks": capacity,
                    "held_sessions": held_sessions,
                    "keys_per_held_session": per_session if held_sessions else 0,
                    "held_physical_keys": len(held_keys),
                    "active_requests": 16,
                    "evict_n": 1,
                    "evict_ns": evict_times,
                    "evict_p50_ns": _percentile(evict_times, 0.5),
                    "evict_p99_ns": _percentile(evict_times, 0.99),
                    "hold_finish_ns": hold_times,
                    "hold_finish_p50_ns": _percentile(hold_times, 0.5),
                    "hold_finish_p99_ns": _percentile(hold_times, 0.99),
                    "owned_metadata_bytes": owned,
                    "charged_metadata_bytes": policy.book.metadata_bytes
                    + policy.observation_bytes,
                }
                cells.append(cell)
                progress(cell)
                policy.clear()
                del native, policy, owner, context, held_keys
                gc.collect()
    return {
        "schema": "pr-bookkeeping-observation-v1",
        "scope": "CPU microbenchmark; oldest keys held, fixed logical time, 16 active request registrations; no GPU or serving throughput claim",
        "percentile": "nearest rank; 200 eviction samples per cell, no warm-up exclusion",
        "metadata": "recursive unique Python object sizes for ledger, observations, reverse key index and outcomes; excludes native cache, runtime and read-only scheduler",
        "cells": cells,
    }
