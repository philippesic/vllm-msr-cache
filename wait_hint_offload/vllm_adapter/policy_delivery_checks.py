from __future__ import annotations

import math
import queue
import time
from types import SimpleNamespace
from typing import Any

import msgspec
from vllm.v1.engine import core
from vllm.v1.kv_offload import base

from wait_hint_offload.policy_protocol import PolicyAction, PolicyEnvelope
from wait_hint_offload.vllm_adapter.policy_checks import policy_fixture, request_fixture


def _engine(native: Any) -> Any:
    engine: Any = object.__new__(getattr(core, "EngineCoreProc"))
    connector = SimpleNamespace(connector_scheduler=SimpleNamespace(manager=native))
    engine.scheduler = SimpleNamespace(get_kv_connector=lambda: connector)
    engine.shutdown_state = getattr(core, "EngineShutdownState").RUNNING
    assert engine.shutdown_state is getattr(core, "EngineShutdownState").RUNNING
    engine.output_queue = queue.Queue()
    return engine


def _deliver(engine: Any, envelope: dict[str, Any]) -> Any:
    request = [0, 1, "publish_kv_cache_hints", [envelope]]
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(request))
    engine._handle_client_request(
        getattr(core, "EngineCoreRequestType").UTILITY, decoded
    )
    return engine.output_queue.get_nowait()[1].utility_output


def _sequence(patched_messages: bool) -> list[list[str]]:
    native, owner = policy_fixture("AdviceOffARC")
    engine = _engine(native)
    sequence = []
    for index, hashes in enumerate(
        ([b"a", b"b"], [b"b", b"c"], [b"d", b"e"], [b"a", b"f"], [b"c", b"g"])
    ):
        context, action = request_fixture(
            owner, str(index), str(index), 1, hashes, False
        )
        native.on_new_request(context)
        keys = [base.make_offload_key(key, 0) for key in hashes]
        trigger = getattr(base, "StoreTrigger").ON_COMPUTE
        output = native.prepare_store(keys, context, trigger=trigger)
        assert output is not None
        sequence.append([key.hex() for key in output.evicted_keys])
        native.complete_store(output.keys_to_store, context, trigger=trigger)
        native.on_request_finished(context)
        if patched_messages:
            envelope = PolicyEnvelope(
                str(index), (PolicyAction.parse(action),)
            ).to_dict()
            utility = _deliver(engine, envelope)
            assert utility.failure_message is None
            assert utility.result.result["actions"][0]["decision"] == "disabled"
        del owner._req_status[context.req_id]
    return sequence


def observe_delivery() -> dict[str, Any]:
    identity = {"A3": _sequence(False), "A3p": _sequence(True)}
    assert identity["A3"] == identity["A3p"]
    native, owner = policy_fixture("PostResponseARC")
    policy = native._policy
    now = [1_000_000_000]
    policy.clock = lambda: now[0]
    engine = _engine(native)
    context, action = request_fixture(owner, "first", "session", 1, [b"a", b"b"], False)
    native.on_new_request(context)
    keys = [base.make_offload_key(key, 0) for key in (b"a", b"b")]
    trigger = getattr(base, "StoreTrigger").ON_COMPUTE
    assert native.prepare_store(keys, context, trigger=trigger) is not None
    native.on_request_finished(context)
    assert not policy.book.leases
    now[0] += 2_000_000_000
    envelope = PolicyEnvelope("delivery", (PolicyAction.parse(action),)).to_dict()
    utility = _deliver(engine, envelope)
    assert utility.failure_message is None
    ack = utility.result.result
    assert ack["accepted_ns"] == now[0]
    assert next(iter(policy.book.leases.values())).deadline_ns == 15_000_000_000
    native.complete_store(keys, context, trigger=trigger)
    assert set(policy.book.references) == set(keys)
    replay = _deliver(engine, envelope)
    assert replay.result.result["actions"][0]["replayed"]
    assert replay.result.result == policy.publish_envelope(envelope)
    before = dict(policy.book.references)
    invalid = _deliver(engine, envelope | {"future_field": True})
    assert invalid.failure_message is not None and policy.book.references == before
    cancelled = PolicyEnvelope(
        "cancel",
        (PolicyAction.parse(action | {"action_id": "cancel", "kind": "cancel"}),),
    ).to_dict()
    assert (
        _deliver(engine, cancelled).result.result["actions"][0]["decision"]
        == "cancelled"
    )
    assert not policy.book.references
    native, owner = policy_fixture("AdviceOffARC")
    native._policy.clock = lambda: 1_000_000_000
    engine = _engine(native)
    elapsed = []
    for index in range(1000):
        context, action = request_fixture(owner, str(index), "cost", index, [], False)
        native.on_new_request(context)
        native.on_request_finished(context)
        envelope = PolicyEnvelope(str(index), (PolicyAction.parse(action),)).to_dict()
        start = time.perf_counter_ns()
        utility = _deliver(engine, envelope)
        elapsed.append(time.perf_counter_ns() - start)
        assert utility.failure_message is None
        assert utility.result.result["actions"][0]["decision"] == "disabled"
        del owner._req_status[context.req_id]
    return {
        "schema": "pr-delivery-observation-v1",
        "identity_sequences": identity,
        "engine_utility_dispatch_exercised": True,
        "deadline_origin_is_acceptance": True,
        "late_ready_joined": True,
        "duplicate_replayed": True,
        "closed_schema_refused_without_mutation": True,
        "cancellation_released": True,
        "control_delivery_ns": elapsed,
        "control_delivery_p50_ns": sorted(elapsed)[math.ceil(len(elapsed) * 0.5) - 1],
        "control_delivery_p99_ns": sorted(elapsed)[math.ceil(len(elapsed) * 0.99) - 1],
        "scope": "real engine utility dispatcher and MessagePack arguments with CPU fixture scheduler; excludes interprocess/socket transport and GPU serving",
    }
