import math
from collections import OrderedDict
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

    def unreachable(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        return []


class GroupARCRecency(ARCRecency):
    def __init__(
        self, policy: ARCCachePolicy, groups: tuple[ObservedGroup, ...]
    ) -> None:
        super().__init__(policy)
        if not groups or not any(group.window_chunks is None for group in groups):
            raise ValueError("full attention anchor required")
        self.groups = groups

    def window(self, boundary: int, group: ObservedGroup) -> tuple[int, int]:
        end = boundary // group.tokens_per_chunk
        if group.window_chunks is None:
            return 0, end
        return max(0, end - group.window_chunks), end

    def ready_boundary(self) -> int:
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
            if all(
                (block := self.policy.get(OffloadKey(key))) is not None
                and block.is_ready
                for group in self.groups
                for key in group.keys[slice(*self.window(boundary, group))]
            ):
                return boundary
            boundary -= alignment
        return 0

    def ready_prefix(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        boundary = self.ready_boundary()
        required = [
            OffloadKey(key)
            for group in self.groups
            for key in group.keys[slice(*self.window(boundary, group))]
        ]
        return list(dict.fromkeys(required))

    def unreachable(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        boundary = self.ready_boundary()
        if not boundary:
            return []
        required = set(self.ready_prefix(keys))
        dead = [
            OffloadKey(key)
            for group in self.groups
            if group.window_chunks is not None
            for key in group.keys[: self.window(boundary, group)[0]]
        ]
        return [key for key in dict.fromkeys(dead) if key not in required]


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
        partial: bool = False,
    ) -> None:
        self.policy = policy
        self.limit = int(policy.cache_capacity * share)
        self.clock = clock
        self.current = current
        self.holds: dict[Owner, tuple[frozenset[OffloadKey], int, object]] = {}
        self.dead: OrderedDict[OffloadKey, None] = OrderedDict()
        self.yielded = 0
        self.dead_evicted = 0
        native = policy.evict

        def evict(n: int, protected: set[OffloadKey]) -> Eviction:
            while held := self.active():
                result = self.evict_dead_first(native, n, protected | held)
                if result is not None:
                    return result
                self.yielded += 1
                if not partial:
                    break
                del self.holds[min(self.holds, key=lambda o: self.holds[o][1])]
            return self.evict_dead_first(native, n, protected)

        setattr(policy, "evict", evict)

    def evict_dead_first(
        self,
        native: Callable[[int, set[OffloadKey]], Eviction],
        n: int,
        protected: set[OffloadKey],
    ) -> Eviction:
        policy = self.policy
        chosen: list[tuple[OffloadKey, BlockStatus, bool]] = []
        for key in list(self.dead):
            if len(chosen) == n:
                break
            block = policy.t1.get(key)
            recent = block is not None
            if block is None:
                block = policy.t2.get(key)
            if block is None:
                del self.dead[key]
            elif block.ref_cnt == 0 and key not in protected:
                chosen.append((key, block, recent))
        rest = native(n - len(chosen), protected | {key for key, _, _ in chosen})
        if rest is None:
            return None
        for key, _, recent in chosen:
            del self.dead[key]
            if recent:
                del policy.t1[key]
                policy.b1[key] = None
            else:
                del policy.t2[key]
                policy.b2[key] = None
        for ghost in (policy.b1, policy.b2):
            for _ in range(len(ghost) - policy.cache_capacity):
                ghost.popitem(last=False)
        self.dead_evicted += len(chosen)
        return [(key, block) for key, block, _ in chosen] + rest

    def mark_dead(self, keys: list[OffloadKey]) -> int:
        for key in keys:
            self.dead[key] = None
            self.dead.move_to_end(key)
        while len(self.dead) > self.policy.cache_capacity:
            self.dead.popitem(last=False)
        return len(keys)

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
        self.dead.clear()


class HoldTarget:
    def __init__(
        self,
        recency: ARCRecency,
        holds: ARCHolds,
        owner: Owner,
        deadline: int,
        token: object,
        evict_unreachable: bool = False,
    ) -> None:
        self.recency = recency
        self.holds = holds
        self.owner = owner
        self.deadline = deadline
        self.token = token
        self.evict_unreachable = evict_unreachable
        self.held = 0
        self.unreachable = 0

    def ready_prefix(self, keys: tuple[bytes, ...]) -> list[OffloadKey]:
        return self.recency.ready_prefix(keys)

    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int:
        promoted = self.recency.promote_prefix(keys)
        self.held = self.holds.hold(
            self.owner, self.ready_prefix(keys), self.deadline, self.token
        )
        if self.evict_unreachable:
            self.unreachable = self.holds.mark_dead(self.recency.unreachable(keys))
        return promoted
