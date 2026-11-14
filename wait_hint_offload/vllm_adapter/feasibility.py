from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any, cast

from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler
from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu import manager
from vllm.v1.kv_offload.cpu.policies import arc, lru
from vllm.v1.kv_offload.cpu.policies import base as policies


class ObservedPolicy:
    native_class: Any

    def __init__(self, cache_capacity: int):
        self.cache_capacity = cache_capacity
        self.native = self.native_class(cache_capacity)
        self.calls: list[tuple[str, str | None, tuple[bytes, ...]]] = []
        self.finished_groups: list[tuple[tuple[bytes, ...], ...]] = []

    def get(self, key: bytes) -> Any | None:
        self.calls.append(("get", None, (key,)))
        return self.native.get(key)

    def insert(self, key: bytes, chunk: Any, req_context: Any) -> None:
        self.calls.append(("insert", req_context.req_id, (key,)))
        self.native.insert(key, chunk, req_context)

    def remove(self, key: bytes) -> None:
        self.calls.append(("remove", None, (key,)))
        self.native.remove(key)

    def on_store_miss(self, keys: Iterable[bytes], req_context: Any) -> None:
        keys = tuple(keys)
        self.calls.append(("store_miss", req_context.req_id, keys))
        self.native.on_store_miss(keys, req_context)

    def on_request_access(self, keys: Iterable[bytes], req_context: Any) -> None:
        keys = tuple(keys)
        self.calls.append(("access", req_context.req_id, keys))
        self.native.on_request_access(keys, req_context)

    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[bytes]],
        insertion_only_keys: set[bytes],
        reused_keys: set[bytes],
        req_context: Any,
    ) -> None:
        groups = tuple(tuple(group) for group in key_groups)
        self.finished_groups.append(groups)
        self.calls.append(("finish", req_context.req_id, tuple(sum(groups, ()))))
        self.native.on_request_finished(
            key_groups, insertion_only_keys, reused_keys, req_context
        )

    def evict(self, n: int, protected: set[bytes]) -> list[tuple[bytes, Any]] | None:
        self.calls.append(("evict", None, tuple(sorted(protected))))
        return cast(list[tuple[bytes, Any]] | None, self.native.evict(n, protected))

    def clear(self) -> None:
        self.calls.append(("clear", None, ()))
        self.native.clear()

    def mark_evictable(self, key: bytes) -> None:
        self.calls.append(("ready", None, (key,)))
        self.native.mark_evictable(key)

    def mark_non_evictable(self, key: bytes) -> None:
        self.calls.append(("busy", None, (key,)))
        self.native.mark_non_evictable(key)


ObservedLRU = type(
    "ObservedLRU",
    (ObservedPolicy, policies.CachePolicy),
    {"native_class": lru.LRUCachePolicy},
)
ObservedARC = type(
    "ObservedARC",
    (ObservedPolicy, policies.CachePolicy),
    {"native_class": arc.ARCCachePolicy},
)


def _context(request: str, session: str) -> Any:
    return getattr(base, "ReqContext")(
        req_id=request,
        kv_transfer_params={"session": session},
        kv_hints=getattr(base, "KvHintsEnvelope")(
            protocol_version="v1", message_id=request, actions=[]
        ),
    )


def _manager(name: str, capacity: int = 4) -> Any:
    return getattr(manager, "CPUOffloadingManager")(
        num_chunks=capacity,
        cache_policy=name,
        cache_policy_module_path=__name__ if name.startswith("Observed") else None,
    )


def _store(manager: Any, ctx: Any, keys: Sequence[bytes]) -> None:
    output = manager.prepare_store(
        keys, ctx, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
    )
    assert output is not None
    manager.complete_store(
        output.keys_to_store, ctx, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
    )


def _identity(name: str) -> list[list[str]]:
    sequence = []
    manager = _manager(name)
    for index, values in enumerate(((1, 2), (2, 3), (4, 5), (1, 6), (3, 7))):
        ctx = _context(str(index), str(index))
        manager.on_new_request(ctx)
        keys = [base.make_offload_key(bytes([value]), 0) for value in values]
        output = manager.prepare_store(
            keys, ctx, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert output is not None
        sequence.append([key.hex() for key in output.evicted_keys])
        manager.complete_store(
            output.keys_to_store, ctx, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        manager.on_request_finished(ctx)
    return sequence


def observe_interfaces() -> dict[str, Any]:
    config = SimpleNamespace(kv_group_configs=[SimpleNamespace(tokens_per_chunk=256)])
    request_state = SimpleNamespace(
        req=SimpleNamespace(request_id="gpu-hit"),
        offloading_context=getattr(base, "RequestOffloadingContext")(),
    )
    handler = getattr(scheduler, "OnComputeStoreHandler")(config, None)
    handler.on_load_scheduled(request_state, 512)
    skipped = handler.first_live_chunk_idx(request_state, 0)
    assert skipped == 2
    request_state.offloading_context.policy = getattr(
        base, "OffloadPolicy"
    ).REQUEST_LEVEL
    handler.reset()
    handler.on_load_scheduled(request_state, 512)
    request_level_start = handler.first_live_chunk_idx(request_state, 0)
    assert request_level_start == 0
    observations = {}
    for name, native in ((ObservedLRU.__name__, "lru"), (ObservedARC.__name__, "arc")):
        manager = _manager(name)
        policy = manager._policy
        assert isinstance(policy, ObservedPolicy)
        keys = [base.make_offload_key(bytes([value]), 0) for value in (1, 2)]
        first = _context("first", "s")
        manager.on_new_request(first)
        _store(manager, first, keys)
        manager.on_request_finished(first)
        assert policy.finished_groups[-1] == (tuple(keys),)
        before = len(policy.calls)
        returning = _context("return", "s")
        manager.on_new_request(returning)
        new_request_calls = policy.calls[before:]
        assert new_request_calls == []
        for index, key in enumerate(keys):
            returning.set_offload_key_position(key, (index + 1) * 256)
        manager.on_request_finished(returning)
        gpu_hit_groups = policy.finished_groups[-1]
        assert gpu_hit_groups == ()
        assert all(policy.native.get(key).is_ready for key in keys)
        shared = _context("shared", "other-session")
        manager.on_new_request(shared)
        for index, key in enumerate(keys):
            shared.set_offload_key_position(key, (index + 1) * 256)
        manager.on_request_finished(shared)
        assert policy.finished_groups[-1] == ()
        delayed = _context("delayed", "s2")
        key = base.make_offload_key(b"late", 0)
        manager.on_new_request(delayed)
        output = manager.prepare_store(
            [key], delayed, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert output is not None
        manager.on_request_finished(delayed)
        assert not policy.native.get(key).is_ready
        before = len(policy.calls)
        manager.complete_store(
            [key], delayed, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert ("ready", None, (key,)) in policy.calls[before:]
        assert policy.native.get(key).is_ready
        identity = _identity(name)
        stock_identity = _identity(native)
        assert identity == stock_identity
        observations[native] = {
            "callbacks_at_return": new_request_calls,
            "scheduler_gpu_hit_chunks_skipped": skipped,
            "request_level_alternative_start": request_level_start,
            "gpu_hit_finish_groups": gpu_hit_groups,
            "shared_gpu_hit_finish_groups": policy.finished_groups[-2],
            "ready_cpu_keys_present": [key.hex() for key in keys],
            "positions_available_privately": len(returning._offload_key_positions),
            "late_store_readiness_callback": True,
            "observer_identity_evictions": identity,
            "stock_identity_evictions": stock_identity,
            "callback_trace": [
                [operation, request, [key.hex() for key in called_keys]]
                for operation, request, called_keys in policy.calls
            ],
        }
    return observations


def observe_private_context() -> dict[str, Any]:
    from wait_hint_offload.vllm_adapter.policy_context import (
        prefix_from_context,
        returned_sessions,
    )

    groups = [
        SimpleNamespace(
            group_idx=index,
            tokens_per_chunk=256,
            hashes_per_chunk=1,
            sliding_window_size_in_chunks=window,
            manager_config=SimpleNamespace(params=None),
        )
        for index, window in enumerate((None, 1))
    ]
    owner = SimpleNamespace(
        config=SimpleNamespace(
            kv_group_configs=groups, supports_partial_tail=False, tokens_per_hash=256
        ),
        _req_status={},
    )
    req = SimpleNamespace(
        request_id="first",
        kv_transfer_params={"session": "s", "generation": 1},
        kv_hints=None,
        block_hashes=[b"a", b"b"],
        num_computed_tokens=513,
        num_in_flight_tokens=0,
        num_prompt_tokens=513,
        num_tokens=514,
    )
    context = getattr(scheduler, "_create_req_context")(req, owner)
    state = getattr(scheduler, "RequestOffloadState")(
        config=owner.config,
        req=req,
        req_context=context,
        offloading_context=getattr(base, "RequestOffloadingContext")(),
    )
    owner._req_status[req.request_id] = state
    state.update_offload_keys()
    snapshot = prefix_from_context(context)
    assert snapshot.complete_tokens == 513
    assert [group.group_id for group in snapshot.groups] == [0, 1]
    assert [len(group.keys) for group in snapshot.groups] == [2, 2]
    assert [group.window_chunks for group in snapshot.groups] == [None, 1]
    assert returned_sessions(owner, {"s": 1}) == set()
    newer = SimpleNamespace(req_context=_context("return", "s"))
    newer.req_context.kv_transfer_params["generation"] = 2
    owner._req_status["return"] = newer
    assert returned_sessions(owner, {"s": 1}) == {"s"}
    assert returned_sessions(owner, {"s": 2}) == set()
    assert returned_sessions(owner, {"other": 1}) == set()
    return {
        "schema": "pr-private-context-observation-v1",
        "native_context_class": type(context).__name__,
        "native_state_class": type(state).__name__,
        "full_gpu_hit_keys_per_group": [len(group.keys) for group in snapshot.groups],
        "windows": [group.window_chunks for group in snapshot.groups],
        "newer_return_detected_before_policy_callback": True,
        "older_inflight_request_does_not_release_newer_hold": True,
        "unrelated_session_unchanged": True,
        "scope": "real context/state methods with CPU scheduler/request fixtures; no serving or hold qualification",
    }
