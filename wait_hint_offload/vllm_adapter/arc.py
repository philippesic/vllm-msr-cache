import math

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

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
