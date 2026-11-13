from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from pathlib import Path

from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies import arc as native_arc

from wait_hint_offload.observed import ObservedGroup
from wait_hint_offload.vllm_adapter.arc import (
    ARCHolds,
    ARCRecency,
    GroupARCRecency,
    HoldTarget,
    Owner,
)

NANOSECONDS = 1_000_000_000
UNBOUNDED_NS = 2**62


class NativeCache:
    def __init__(self, cpu_keys: int, hold_share: float) -> None:
        self.manager = CPUOffloadingManager(cpu_keys, cache_policy="arc")
        policy = self.manager._policy
        if not isinstance(policy, native_arc.ARCCachePolicy):
            raise ValueError("native ARC required")
        self.policy = policy
        self.context = ReqContext(req_id="replay")
        self.now = 0.0
        self.tokens: dict[Owner, object] = {}
        self.holds = ARCHolds(
            policy, hold_share, lambda: int(self.now * NANOSECONDS), self.tokens.get
        )
        self.partial = ARCHolds(
            policy,
            hold_share,
            lambda: int(self.now * NANOSECONDS),
            self.tokens.get,
            partial=True,
        )
        self.oracle = ARCHolds(policy, 1.0, lambda: 0, self.tokens.get)
        self.unreachable = 0

    def ready(self, key: bytes) -> bool:
        return self.manager.lookup(OffloadKey(key), self.context) is LookupResult.HIT

    def touch(self, keys: Sequence[bytes]) -> None:
        self.manager.touch([OffloadKey(key) for key in keys], self.context)

    def store(self, keys: Sequence[bytes]) -> tuple[list[bytes], int] | None:
        output = self.manager.prepare_store(
            [OffloadKey(key) for key in keys], self.context
        )
        if output is None:
            return None
        return [bytes(key) for key in output.keys_to_store], len(output.evicted_keys)

    def complete(self, keys: Sequence[bytes]) -> None:
        self.manager.complete_store([OffloadKey(key) for key in keys], self.context)

    def advise(
        self,
        arm: str,
        groups: Sequence[ObservedGroup],
        owner: str,
        deadline: float,
    ) -> tuple[int, int]:
        full = groups[0]
        recency = (
            GroupARCRecency(self.policy, tuple(groups))
            if len(groups) > 1
            else ARCRecency(self.policy)
        )
        anchor = set(full.keys)
        ready = sum(key in anchor for key in recency.ready_prefix(full.keys))
        identity = ("replay", owner)
        token = object()
        self.tokens[identity] = token
        if arm == "recency":
            applied = recency.refresh_prefix(full.keys)
        elif arm == "promote":
            applied = recency.promote_prefix(full.keys)
        elif arm in ("promote-hold", "reachable-hold"):
            reachable = arm == "reachable-hold"
            target = HoldTarget(
                recency,
                self.partial if reachable else self.holds,
                identity,
                int(deadline * NANOSECONDS),
                token,
                evict_unreachable=reachable,
            )
            target.refresh_prefix(full.keys)
            applied = target.held
            self.unreachable += target.unreachable
        elif arm in ("hold", "oracle"):
            holds = self.oracle if arm == "oracle" else self.holds
            limit = UNBOUNDED_NS if arm == "oracle" else int(deadline * NANOSECONDS)
            applied = holds.hold(
                identity, recency.ready_prefix(full.keys), limit, token
            )
        else:
            raise ValueError("unknown advice arm")
        return ready * full.tokens_per_chunk, applied

    def release(self, owner: str) -> None:
        self.tokens.pop(("replay", owner), None)
        self.holds.release(("replay", owner))
        self.partial.release(("replay", owner))
        self.oracle.release(("replay", owner))

    def counters(self) -> dict[str, int]:
        return {
            "hold_yields": self.holds.yielded
            + self.partial.yielded
            + self.oracle.yielded,
            "unreachable_marked": self.unreachable,
            "unreachable_evicted": self.partial.dead_evicted,
        }


def native_identity() -> dict[str, str]:
    files = {
        "arc_policy_sha256": Path(inspect.getfile(native_arc)),
        "cpu_manager_sha256": Path(inspect.getfile(CPUOffloadingManager)),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()
    }
