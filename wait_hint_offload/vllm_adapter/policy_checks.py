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
        load_window_size_in_chunks=lambda boundary: None,
    )
    owner = SimpleNamespace(
        config=SimpleNamespace(kv_group_configs=(group,)), _req_status={}
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
        num_computed_tokens=len(hashes) * 256 + 1,
        num_in_flight_tokens=0,
        num_prompt_tokens=len(hashes) * 256 + 1,
        num_tokens=len(hashes) * 256 + 2,
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
        "schema": "pr-policy-observation-v2",
        "cases": observed,
        "extended": extended_hold_checks(),
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
