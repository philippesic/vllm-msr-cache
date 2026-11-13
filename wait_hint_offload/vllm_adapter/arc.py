import math
from collections.abc import Callable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

from wait_hint_offload.observed import ObservedGroup


class ARCRecency:
    def __init__(self, policy: ARCCachePolicy) -> None:
        self.policy = policy

    def ready_prefix(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        eligible = []
        for raw in keys:
            key = OffloadKey(raw)
            block = self.policy.get(key)
            if block is None or not block.is_ready:
                break
            eligible.append(key)
        return eligible

    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int:
        eligible = self.ready_prefix(keys)
        for key in reversed(eligible):
            queue = self.policy.t1 if key in self.policy.t1 else self.policy.t2
            queue.move_to_end(key)
        return len(eligible)

    def promote_prefix(self, keys: tuple[bytes, ...]) -> int:
        eligible = self.ready_prefix(keys)
        self.policy.touch(eligible, ReqContext(req_id="wait-hint"))
        return len(eligible)


class GroupARCRecency(ARCRecency):
    def __init__(
        self, policy: ARCCachePolicy, groups: tuple[ObservedGroup, ...]
    ) -> None:
        super().__init__(policy)
        if not groups or not any(group.window_chunks is None for group in groups):
            raise ValueError("full attention anchor required")
        self.groups = groups

    def ready_prefix(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        alignment = math.lcm(*(group.tokens_per_chunk for group in self.groups))
        boundary = min(
            len(group.keys) * group.tokens_per_chunk for group in self.groups
        )
        for group in self.groups:
            if group.window_chunks is None:
                ready = super().ready_prefix(group.keys)
                boundary = min(boundary, len(ready) * group.tokens_per_chunk)
        boundary = boundary // alignment * alignment
        while boundary > 0:
            required: list[OffloadKey] = []
            for group in self.groups:
                end = boundary // group.tokens_per_chunk
                start = (
                    0
                    if group.window_chunks is None
                    else max(0, end - group.window_chunks)
                )
                required.extend(OffloadKey(key) for key in group.keys[start:end])
            if all(
                (block := self.policy.get(key)) is not None and block.is_ready
                for key in required
            ):
                return list(dict.fromkeys(required))
            boundary -= alignment
        return []


HOLD_SHARE = 0.5
Owner = tuple[str, str]
Eviction = list[tuple[OffloadKey, BlockStatus]] | None


class ARCHolds:
    def __init__(
        self,
        policy: ARCCachePolicy,
        share: float,
        clock: Callable[[], int],
        current: Callable[[Owner], object | None],
    ) -> None:
        self.limit = int(policy.cache_capacity * share)
        self.clock = clock
        self.current = current
        self.holds: dict[Owner, tuple[frozenset[OffloadKey], int, object]] = {}
        self.yielded = 0
        native = policy.evict

        def evict(n: int, protected: set[OffloadKey]) -> Eviction:
            held = self.active()
            if held:
                result = native(n, protected | held)
                if result is not None:
                    return result
                self.yielded += 1
            return native(n, protected)

        setattr(policy, "evict", evict)

    def active(self) -> set[OffloadKey]:
        now = self.clock()
        for owner, (_, deadline, token) in list(self.holds.items()):
            if now >= deadline or self.current(owner) is not token:
                del self.holds[owner]
        return {key for keys, _, _ in self.holds.values() for key in keys}

    def hold(
        self, owner: Owner, keys: list[OffloadKey], deadline: int, token: object
    ) -> int:
        self.holds.pop(owner, None)
        if not keys or len(self.active()) + len(keys) > self.limit:
            return 0
        self.holds[owner] = (frozenset(keys), deadline, token)
        return len(keys)

    def release(self, owner: Owner) -> None:
        self.holds.pop(owner, None)

    def clear(self) -> None:
        self.holds.clear()


class HoldTarget:
    def __init__(
        self,
        recency: ARCRecency,
        holds: ARCHolds,
        owner: Owner,
        deadline: int,
        token: object,
    ) -> None:
        self.recency = recency
        self.holds = holds
        self.owner = owner
        self.deadline = deadline
        self.token = token
        self.held = 0

    def ready_prefix(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        return self.recency.ready_prefix(keys)

    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int:
        promoted = self.recency.promote_prefix(keys)
        self.held = self.holds.hold(
            self.owner, self.ready_prefix(keys), self.deadline, self.token
        )
        return promoted
