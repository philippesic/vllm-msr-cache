from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy


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
