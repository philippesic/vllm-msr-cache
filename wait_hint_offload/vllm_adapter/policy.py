from __future__ import annotations

import importlib.metadata
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.kv_offload.cpu.policies import arc, lru
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy

from wait_hint_offload.policy_holds import HoldLedger
from wait_hint_offload.policy_protocol import PolicyAction, PolicySession
from wait_hint_offload.vllm_adapter.policy_context import (
    PolicyPrefix,
    prefix_from_context,
)


@dataclass
class PolicyObservation:
    session: PolicySession
    prefix: PolicyPrefix
    finished_ns: int
    expires_ns: int
    charge: int
    deadline_ns: int = 0
    actions: dict[str, tuple[PolicyAction, str]] = field(default_factory=dict)


class WaitHoldPolicy:
    native_class: Any
    advice_enabled = True
    request_ttl = True

    def __init__(self, cache_capacity: int):
        if importlib.metadata.version("vllm") != "0.34.0rc1.dev465+ge9281bd93":
            raise RuntimeError("frozen-main policy requires e9281bd93 runtime")
        self.cache_capacity = cache_capacity
        self.native = self.native_class(cache_capacity)
        self.book = HoldLedger(cache_capacity, metadata_limit=32 * 1024 * 1024)
        self.observations: OrderedDict[str, PolicyObservation] = OrderedDict()
        self.observation_bytes = 0
        self.scheduler: Any = None
        self.clock = time.monotonic_ns
        self.key_owners: dict[bytes, set[str]] = {}
        self.outcomes: dict[str, int] = {}

    def _count(self, decision: str) -> str:
        self.outcomes[decision] = self.outcomes.get(decision, 0) + 1
        return decision

    def get(self, key: bytes) -> Any:
        return self.native.get(key)

    def insert(self, key: bytes, chunk: Any, req_context: Any) -> None:
        self.native.insert(key, chunk, req_context)

    def remove(self, key: bytes) -> None:
        self.native.remove(key)
        for owner in tuple(self.key_owners.get(key, ())):
            self.book.release(owner)
            observation = self.observations.get(owner)
            if observation is not None:
                self._refresh(observation)

    def on_store_miss(self, keys: Iterable[bytes], req_context: Any) -> None:
        self.native.on_store_miss(keys, req_context)

    def on_request_access(self, keys: Iterable[bytes], req_context: Any) -> None:
        self.native.on_request_access(keys, req_context)
        self._notice_return(req_context)

    def _session(self, context: Any) -> PolicySession | None:
        params = context.kv_transfer_params
        raw = params.get("wait_hint_policy") if isinstance(params, dict) else None
        if raw is None:
            return None
        try:
            return PolicySession.parse(raw)
        except (TypeError, ValueError):
            self._count("invalid_session")
            return None

    def _notice_return(self, context: Any) -> None:
        session = self._session(context)
        if session is None:
            return
        observation = self.observations.get(session.owner)
        if (
            observation is not None
            and session.generation > observation.session.generation
        ):
            self._retire(session.owner)
            self._count("return_release")

    def _retire(self, owner: str) -> None:
        self.book.release(owner)
        observation = self.observations.pop(owner, None)
        if observation is None:
            return
        self.observation_bytes -= observation.charge
        for group in observation.prefix.groups:
            for key in group.all_keys():
                owners = self.key_owners.get(key)
                if owners is not None:
                    owners.discard(owner)
                    if not owners:
                        del self.key_owners[key]

    def _maintain(self) -> None:
        now = self.clock()
        self.book.expire(now)
        for owner, observation in tuple(self.observations.items()):
            if observation.expires_ns <= now:
                self._retire(owner)
        if self.scheduler is not None and self.observations:
            for state in tuple(self.scheduler._req_status.values()):
                self._notice_return(state.req_context)

    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[bytes]],
        insertion_only_keys: set[bytes],
        reused_keys: set[bytes],
        req_context: Any,
    ) -> None:
        self.native.on_request_finished(
            key_groups, insertion_only_keys, reused_keys, req_context
        )
        session = self._session(req_context)
        if session is None:
            return
        if not hasattr(req_context, "_scheduler"):
            self._count("missing_scheduler_context")
            return
        self.scheduler = req_context._scheduler
        self._maintain()
        previous = self.observations.get(session.owner)
        if previous is not None and session.generation <= previous.session.generation:
            self._count("stale_generation")
            return
        for state in tuple(self.scheduler._req_status.values()):
            active = self._session(state.req_context)
            if (
                active is not None
                and active.owner == session.owner
                and active.generation > session.generation
            ):
                self._count("superseded_finish")
                return
        try:
            prefix = prefix_from_context(req_context)
        except (AttributeError, TypeError, ValueError):
            self._count("unavailable_prefix")
            return
        charge = 4096 + sum(
            256 + len(key) for group in prefix.groups for key in group.all_keys()
        )
        old_charge = previous.charge if previous is not None else 0
        if self.observation_bytes - old_charge + charge > 32 * 1024 * 1024:
            self._count("observation_metadata_limit")
            return
        self._retire(session.owner)
        while len(self.observations) >= 4096:
            self._retire(next(iter(self.observations)))
            self._count("observation_retired")
        now = self.clock()
        observation = PolicyObservation(
            session, prefix, now, now + 60_000_000_000, charge
        )
        self.observations[session.owner] = observation
        self.observation_bytes += charge
        for group in prefix.groups:
            for key in group.all_keys():
                self.key_owners.setdefault(key, set()).add(session.owner)
        if self.request_ttl:
            envelope = req_context.kv_hints
            if envelope is not None:
                if envelope.protocol_version != "v1":
                    self._count("unsupported_envelope")
                    return
                if len(envelope.actions) > 256:
                    self._count("envelope_action_limit")
                    return
                for action in envelope.actions:
                    if action.action_type != "wait_hint.request_ttl":
                        continue
                    if action.action_version != "v1":
                        self._count("unsupported_action_version")
                        continue
                    try:
                        parsed = PolicyAction.parse(action.payload)
                    except (TypeError, ValueError):
                        self._count("invalid_action")
                        continue
                    if (
                        parsed.session != session
                        or parsed.action_id != action.action_id
                    ):
                        self._count("request_action_identity_mismatch")
                        continue
                    self.publish(action.payload, deadline_origin_ns=now)

    def _ready_keys(self, prefix: PolicyPrefix) -> frozenset[bytes]:
        alignment = prefix.alignment_tokens
        step = prefix.tokens_per_hash if prefix.partial_tail else alignment
        boundary = prefix.complete_tokens
        for group in prefix.groups:
            if group.window_chunks is None:
                count = 0
                for key in group.keys:
                    chunk = self.native.get(key)
                    if chunk is None or not chunk.is_ready:
                        break
                    count += 1
                extra = alignment - 1 if prefix.partial_tail else 0
                boundary = min(boundary, count * group.tokens_per_chunk + extra)
        boundary = boundary // step * step
        while boundary > 0:
            partial = boundary % alignment != 0
            keys = tuple(
                key
                for group in prefix.groups
                for key in group.required_keys(
                    boundary, partial, prefix.tokens_per_hash
                )
            )
            if all(
                (chunk := self.native.get(key)) is not None and chunk.is_ready
                for key in keys
            ):
                return frozenset(keys)
            boundary -= step
        return frozenset()

    def _refresh(self, observation: PolicyObservation) -> str:
        now = self.clock()
        if not self.advice_enabled or observation.deadline_ns <= now:
            return "inactive"
        keys = self._ready_keys(observation.prefix)
        return self.book.hold(
            observation.session.owner,
            observation.session.generation,
            observation.deadline_ns,
            keys,
            now,
        ).value

    def publish(
        self, raw: object, deadline_origin_ns: int | None = None
    ) -> dict[str, Any]:
        self._maintain()
        try:
            action = PolicyAction.parse(raw)
        except (TypeError, ValueError):
            return {"decision": self._count("invalid_action"), "replayed": False}
        observation = self.observations.get(action.session.owner)
        if observation is None or observation.session != action.session:
            return {"decision": self._count("unobserved"), "replayed": False}
        replay = observation.actions.get(action.action_id)
        if replay is not None:
            if replay[0] != action:
                return {"decision": self._count("action_conflict"), "replayed": False}
            return {"decision": replay[1], "replayed": True}
        if len(observation.actions) >= 256:
            return {"decision": self._count("action_limit"), "replayed": False}
        action_charge = 4096
        if self.observation_bytes + action_charge > 32 * 1024 * 1024:
            return {"decision": self._count("action_metadata_limit"), "replayed": False}
        if not self.advice_enabled:
            decision = "disabled"
        elif action.kind == "cancel":
            self.book.release(action.session.owner)
            observation.deadline_ns = 0
            decision = "cancelled"
        else:
            origin = self.clock() if deadline_origin_ns is None else deadline_origin_ns
            observation.deadline_ns = origin + min(action.ttl_ms, 60000) * 1_000_000
            observation.expires_ns = max(
                observation.expires_ns, observation.deadline_ns
            )
            decision = self._refresh(observation)
        observation.actions[action.action_id] = (action, decision)
        observation.charge += action_charge
        self.observation_bytes += action_charge
        self._count(decision)
        return {"decision": decision, "replayed": False}

    def mark_evictable(self, key: bytes) -> None:
        self.native.mark_evictable(key)
        self._maintain()
        for owner in tuple(self.key_owners.get(key, ())):
            observation = self.observations.get(owner)
            if observation is not None:
                self._refresh(observation)

    def mark_non_evictable(self, key: bytes) -> None:
        self.native.mark_non_evictable(key)

    def evict(self, n: int, protected: set[bytes]) -> Any:
        self._maintain()
        while self.book.references:
            result = self.native.evict(n, self.book.protection(protected))
            if result is not None:
                return result
            available = 0
            tables = (
                (self.native.chunks,)
                if hasattr(self.native, "chunks")
                else (self.native.t1, self.native.t2)
            )
            for table in tables:
                for key, chunk in table.items():
                    if chunk.ref_cnt == 0 and key not in protected:
                        available += 1
                        if available >= n:
                            break
                if available >= n:
                    break
            if available < n:
                return None
            owner = self.book.oldest_owner()
            if not self.book.yield_oldest():
                break
            observation = self.observations.get(owner) if owner is not None else None
            if observation is not None:
                observation.deadline_ns = 0
            self._count("hold_yield")
        return self.native.evict(n, protected)

    def clear(self) -> None:
        self.native.clear()
        self.book.clear()
        self.observations.clear()
        self.key_owners.clear()
        self.observation_bytes = 0
        self.scheduler = None


RequestTTLARC = type(
    "RequestTTLARC", (WaitHoldPolicy, CachePolicy), {"native_class": arc.ARCCachePolicy}
)
RequestTTLLRU = type(
    "RequestTTLLRU", (WaitHoldPolicy, CachePolicy), {"native_class": lru.LRUCachePolicy}
)
PostResponseARC = type(
    "PostResponseARC",
    (WaitHoldPolicy, CachePolicy),
    {"native_class": arc.ARCCachePolicy, "request_ttl": False},
)
AdviceOffARC = type(
    "AdviceOffARC",
    (WaitHoldPolicy, CachePolicy),
    {"native_class": arc.ARCCachePolicy, "advice_enabled": False},
)
AdviceOffLRU = type(
    "AdviceOffLRU",
    (WaitHoldPolicy, CachePolicy),
    {"native_class": lru.LRUCachePolicy, "advice_enabled": False},
)
