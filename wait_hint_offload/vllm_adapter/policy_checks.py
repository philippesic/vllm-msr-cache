from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import msgspec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler
from vllm.v1.kv_offload import base
from vllm.v1.kv_offload.cpu import manager

from wait_hint_offload.policy_protocol import PolicyAction
from wait_hint_offload.vllm_adapter.policy import (
    AdviceOffARC,
    AdviceOffLRU,
    PostResponseARC,
    RequestTTLARC,
    RequestTTLLRU,
    WaitHoldPolicy,
)


def policy_fixture(name: str, capacity: int = 4) -> tuple[Any, Any]:
    native = getattr(manager, "CPUOffloadingManager")(
        num_chunks=capacity,
        cache_policy=name,
        cache_policy_module_path="wait_hint_offload.vllm_adapter.policy",
    )
    group = SimpleNamespace(
        group_idx=0,
        tokens_per_chunk=256,
        hashes_per_chunk=1,
        sliding_window_size_in_chunks=None,
        manager_config=SimpleNamespace(params=None),
    )
    owner = SimpleNamespace(
        config=getattr(scheduler, "SchedulerOffloadConfig")(
            kv_group_configs=(group,),
            blocks_per_chunk=1,
            tokens_per_hash=256,
            num_workers=1,
            offload_prompt_only=True,
            supports_partial_tail=False,
        ),
        _req_status={},
    )
    return native, owner


def request_fixture(
    owner: Any,
    request: str,
    session: str,
    generation: int,
    hashes: list[bytes],
    ttl: bool = True,
) -> tuple[Any, dict[str, Any]]:
    registration = {
        "schema": "wait-hint-policy-session-v1",
        "scope": "fixture",
        "session": session,
        "incarnation": "i",
        "generation": generation,
    }
    action = {
        "schema": "wait-hint-policy-action-v1",
        "action_id": request,
        "session": registration,
        "kind": "hold",
        "ttl_ms": 12000,
    }
    envelope = msgspec.convert(
        {
            "protocol_version": "v1",
            "message_id": request,
            "actions": [
                {
                    "action_id": request,
                    "action_type": "wait_hint.request_ttl",
                    "action_version": "v1",
                    "payload": action,
                }
            ]
            if ttl
            else [],
        },
        type=getattr(base, "KvHintsEnvelope"),
    )
    req = SimpleNamespace(
        request_id=request,
        kv_transfer_params={"wait_hint_policy": registration},
        kv_hints=envelope,
        block_hashes=hashes,
        num_computed_tokens=len(hashes) * owner.config.tokens_per_hash + 1,
        num_in_flight_tokens=0,
        num_prompt_tokens=len(hashes) * owner.config.tokens_per_hash + 1,
        num_tokens=len(hashes) * owner.config.tokens_per_hash + 2,
    )
    context = getattr(scheduler, "_create_req_context")(req, owner)
    state = getattr(scheduler, "RequestOffloadState")(
        config=owner.config,
        req=req,
        req_context=context,
        offloading_context=getattr(base, "RequestOffloadingContext")(),
    )
    owner._req_status[request] = state
    state.update_offload_keys()
    return context, action


def observe_hold_policy() -> dict[str, Any]:
    observed = {}
    for cls in (
        RequestTTLARC,
        RequestTTLLRU,
        PostResponseARC,
        AdviceOffARC,
        AdviceOffLRU,
    ):
        native, owner = policy_fixture(cls.__name__)
        policy = native._policy
        assert isinstance(policy, WaitHoldPolicy)
        now = [1_000_000_000]
        policy.clock = lambda: now[0]
        context, action = request_fixture(owner, "first", "s", 1, [b"a", b"b"])
        assert PolicyAction.parse(action).to_dict() == action
        native.on_new_request(context)
        keys = [base.make_offload_key(key, 0) for key in (b"a", b"b")]
        output = native.prepare_store(
            keys, context, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert output is not None
        assert not policy.book.references
        native.on_request_finished(context)
        assert not policy.book.references
        assert next(iter(policy.observations.values())).finished_ns == now[0]
        if not policy.request_ttl:
            assert policy.publish(action)["decision"] == "held"
        native.complete_store(
            keys, context, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert len(policy.book.references) == (2 if policy.advice_enabled else 0)
        replay = policy.publish(action)
        assert replay["replayed"]
        assert not policy.publish(action | {"ttl_ms": 13000})["replayed"]
        held = dict(policy.book.references)
        assert policy.evict(4, set()) is None
        assert policy.book.references == held
        return_ctx, _ = request_fixture(owner, "return", "s", 2, [b"a", b"b"], False)
        native.on_new_request(return_ctx)
        assert policy.book.references == held
        assert policy.evict(0, set()) == []
        assert not policy.book.references
        native.on_request_finished(return_ctx)
        assert policy.observations
        native.reset_cache()
        assert (
            not policy.book.references
            and not policy.observations
            and not policy.key_owners
        )
        observed[cls.__name__] = {
            "pending_not_held": True,
            "late_ready_hold_keys": len(held),
            "duplicate_replayed": True,
            "changed_action_refused": True,
            "impossible_eviction_preserved_holds": True,
            "return_released_before_eviction": True,
            "reset_cleared_metadata": True,
        }
    return {
        "schema": "pr-policy-observation-v3",
        "cases": observed,
        "extended": extended_hold_checks(),
        "geometry": geometry_checks(),
        "tiering": tiering_checks(),
    }


def _store_keys(native: Any, context: Any, hashes: list[bytes]) -> None:
    keys = [base.make_offload_key(key, 0) for key in hashes]
    output = native.prepare_store(
        keys, context, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
    )
    assert output is not None
    native.complete_store(
        output.keys_to_store, context, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
    )


def _identity_sequence(name: str) -> list[list[str]]:
    native, owner = policy_fixture(name)
    sequence = []
    for index, hashes in enumerate(
        ([b"a", b"b"], [b"b", b"c"], [b"d", b"e"], [b"a", b"f"], [b"c", b"g"])
    ):
        context, _ = request_fixture(owner, str(index), str(index), 1, hashes, False)
        native.on_new_request(context)
        keys = [base.make_offload_key(key, 0) for key in hashes]
        output = native.prepare_store(
            keys, context, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
        )
        assert output is not None
        sequence.append([key.hex() for key in output.evicted_keys])
        native.complete_store(
            output.keys_to_store,
            context,
            trigger=getattr(base, "StoreTrigger").ON_COMPUTE,
        )
        native.on_request_finished(context)
    return sequence


def extended_hold_checks() -> dict[str, Any]:
    identities = {}
    for disabled, stock in ((AdviceOffARC, "arc"), (AdviceOffLRU, "lru")):
        actual = _identity_sequence(disabled.__name__)
        expected = _identity_sequence(stock)
        assert actual == expected
        identities[stock] = {"disabled": actual, "stock": expected}
    for cls in (RequestTTLARC, RequestTTLLRU):
        native, owner = policy_fixture(cls.__name__)
        policy = native._policy
        now = [1_000_000_000]
        policy.clock = lambda: now[0]
        source, _ = request_fixture(owner, "source", "source", 1, [b"a", b"b"], False)
        native.on_new_request(source)
        _store_keys(native, source, [b"a", b"b"])
        native.on_request_finished(source)
        target, action = request_fixture(owner, "target", "target", 1, [b"a", b"b"])
        native.on_new_request(target)
        native.on_request_finished(target)
        assert len(policy.book.references) == 2
        shared, shared_action = request_fixture(
            owner, "shared", "shared", 1, [b"a", b"b"]
        )
        native.on_new_request(shared)
        native.on_request_finished(shared)
        assert len(policy.book.references) == 2
        assert set(policy.book.references.values()) == {2}
        cancel = action | {"action_id": "cancel", "kind": "cancel"}
        assert policy.publish(cancel)["decision"] == "cancelled"
        assert set(policy.book.references.values()) == {1}
        assert policy.publish(cancel)["replayed"]
        now[0] += 12_000_000_000
        policy.evict(0, set())
        assert not policy.book.references
        assert policy.publish(shared_action)["replayed"]
        assert not policy.book.references
        native.reset_cache()
        first, _ = request_fixture(
            owner, "first-pressure", "first-pressure", 1, [b"a", b"b"]
        )
        native.on_new_request(first)
        _store_keys(native, first, [b"a", b"b"])
        native.on_request_finished(first)
        other, _ = request_fixture(owner, "other", "other", 1, [b"c", b"d"], False)
        native.on_new_request(other)
        _store_keys(native, other, [b"c", b"d"])
        native.on_request_finished(other)
        protected = {base.make_offload_key(key, 0) for key in (b"c", b"d")}
        before = dict(policy.book.references)
        assert policy.evict(3, protected) is None
        assert policy.book.references == before
        replacement, _ = request_fixture(
            owner, "replacement", "replacement", 1, [b"e", b"f", b"g"], False
        )
        native.on_new_request(replacement)
        _store_keys(native, replacement, [b"e", b"f", b"g"])
        assert policy.book.yields == 1
        assert not policy.book.references
        native.reset_cache()
        failed, _ = request_fixture(owner, "failed", "failed", 1, [b"fail"])
        native.on_new_request(failed)
        key = base.make_offload_key(b"fail", 0)
        assert (
            native.prepare_store(
                [key], failed, trigger=getattr(base, "StoreTrigger").ON_COMPUTE
            )
            is not None
        )
        native.on_request_finished(failed)
        native.complete_store(
            [key],
            failed,
            trigger=getattr(base, "StoreTrigger").ON_COMPUTE,
            success=False,
        )
        assert policy.get(key) is None and not policy.book.references
    return {
        "identity_sequences": identities,
        "gpu_only_shared_prefix_hold": True,
        "shared_union_counted_once": True,
        "cancellation_preserves_other_owner": True,
        "expiry_and_duplicate_do_not_rehold": True,
        "failed_eviction_preserves_holds": True,
        "yield_prevents_added_store_failure": True,
        "failed_store_never_ready_or_held": True,
    }


def geometry_checks() -> dict[str, bool]:
    for cls in (RequestTTLARC, RequestTTLLRU):
        native, owner = policy_fixture(cls.__name__, 32)
        policy = native._policy
        groups = tuple(
            SimpleNamespace(
                group_idx=index,
                tokens_per_chunk=size,
                hashes_per_chunk=size // 16,
                sliding_window_size_in_chunks=window,
                manager_config=SimpleNamespace(params=None),
            )
            for index, size, window in ((0, 256, None), (1, 64, 1))
        )
        owner.config = owner.config._replace(
            kv_group_configs=groups,
            tokens_per_hash=16,
            supports_partial_tail=True,
            alignment_tokens=256,
        )
        hashes = [index.to_bytes(4, "big") for index in range(32)]
        context, _ = request_fixture(owner, "partial", "partial", 1, hashes)
        native.on_new_request(context)
        trigger = getattr(base, "StoreTrigger").ON_COMPUTE
        full = [base.make_offload_key(hashes[15], index) for index in (0, 1)]
        tail = [base.make_offload_key(hashes[23], index) for index in (0, 1)]
        output = native.prepare_store(full + tail, context, trigger=trigger)
        assert output is not None
        native.complete_store(full + tail[:1], context, trigger=trigger)
        native.on_request_finished(context)
        assert set(policy.book.references) == set(full)
        native.complete_store(tail[1:], context, trigger=trigger)
        assert set(policy.book.references) == {full[0], *tail}
        lookup_owner = SimpleNamespace(
            manager=native,
            config=owner.config,
            _chunks_being_loaded=set(),
            _events_tracker=SimpleNamespace(record_partial_lookup=lambda *args: None),
        )
        state = owner._req_status[context.req_id]
        hit, pending = getattr(
            scheduler, "OffloadingConnectorScheduler"
        )._lookup_partial_tail(lookup_owner, state, 384, False)
        assert hit and not pending
        fork_hashes = hashes[:16] + [b"fork" + key for key in hashes[16:]]
        fork, _ = request_fixture(owner, "fork", "fork", 1, fork_hashes)
        native.on_new_request(fork)
        fork_tail = [base.make_offload_key(fork_hashes[27], index) for index in (0, 1)]
        output = native.prepare_store(fork_tail, fork, trigger=trigger)
        assert output is not None
        native.complete_store(fork_tail, fork, trigger=trigger)
        native.on_request_finished(fork)
        assert set(policy.book.references) == {full[0], *tail, *fork_tail}
        assert policy.book.references[full[0]] == 2
        assert all(policy.book.references[key] == 1 for key in tail + fork_tail)
        native.reset_cache()
        owner.config = owner.config._replace(supports_partial_tail=False)
        owner.config.kv_group_configs[1].sliding_window_size_in_chunks = 2
        owner.config.kv_group_configs[1].manager_config.params = SimpleNamespace(
            sliding_window=129, extra_retained_tokens=0
        )
        window, _ = request_fixture(owner, "window", "window", 1, hashes)
        native.on_new_request(window)
        keys = [base.make_offload_key(hashes[15], 0)] + [
            base.make_offload_key(hashes[index], 1) for index in (11, 15)
        ]
        assert native.prepare_store(keys, window, trigger=trigger) is not None
        native.complete_store(keys, window, trigger=trigger)
        native.on_request_finished(window)
        assert set(policy.book.references) == set(keys)
    return {
        "partial_tail_requires_every_group_ready": True,
        "partial_boundary_matches_native_lookup": True,
        "forks_preserve_shared_anchor_and_distinct_checkpoints": True,
        "window_uses_selected_boundary_not_unaligned_finish": True,
    }


def tiering_checks() -> dict[str, bool]:
    from vllm.v1.kv_offload.tiering import manager as tiering

    for cls in (RequestTTLARC, RequestTTLLRU):
        region = SimpleNamespace(
            create_kv_memoryview=lambda: memoryview(bytearray(64)).cast(
                "B", shape=[4, 16]
            )
        )
        primary = getattr(tiering, "CPUPrimaryTierOffloadingManager")(
            num_chunks=4,
            mmap_region=region,
            cache_policy=cls.__name__,
            cache_policy_module_path="wait_hint_offload.vllm_adapter.policy",
        )
        native = getattr(tiering, "TieringOffloadingManager")(
            primary, tier_poll_interval_s=0
        )
        _, owner = policy_fixture(cls.__name__)
        context, _ = request_fixture(owner, "tier", "tier", 1, [b"a", b"b"])
        native.on_new_request(context)
        keys = [base.make_offload_key(key, 0) for key in (b"a", b"b")]
        trigger = getattr(base, "StoreTrigger").ON_COMPUTE
        assert native.prepare_store(keys, context, trigger=trigger) is not None
        native.on_request_finished(context)
        assert not primary._policy.book.references
        native.complete_store(keys, context, trigger=trigger)
        assert set(primary._policy.book.references) == set(keys)
        returned, _ = request_fixture(owner, "tier-return", "tier", 2, [b"a", b"b"])
        native.on_new_request(returned)
        assert primary._policy.evict(0, set()) == []
        assert not primary._policy.book.references
        native.on_request_finished(returned)
        native.reset_cache()
        assert not primary._policy.observations
    return {
        "primary_tier_preserves_scheduler_context": True,
        "late_store_after_finish_joins_hold": True,
        "return_releases_and_reset_clears": True,
    }
