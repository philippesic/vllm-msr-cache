from __future__ import annotations

import heapq
from collections.abc import Iterator, Set
from dataclasses import dataclass
from enum import StrEnum


class HoldDecision(StrEnum):
    HELD = "held"
    EXPIRED = "expired"
    STALE = "stale"
    PHYSICAL_LIMIT = "physical_limit"
    METADATA_LIMIT = "metadata_limit"
    SESSION_LIMIT = "session_limit"


@dataclass(frozen=True)
class Lease:
    generation: int
    deadline_ns: int
    keys: frozenset[bytes]
    charge: int


class ProtectedUnion(Set[bytes]):
    def __init__(self, transfer: Set[bytes], held: dict[bytes, int]) -> None:
        self.transfer = transfer
        self.held = held

    def __contains__(self, key: object) -> bool:
        return key in self.transfer or key in self.held

    def __iter__(self) -> Iterator[bytes]:
        yield from self.transfer
        yield from (key for key in self.held if key not in self.transfer)

    def __len__(self) -> int:
        return len(self.transfer) + sum(key not in self.transfer for key in self.held)


class HoldLedger:
    def __init__(
        self,
        capacity: int,
        share: float = 0.5,
        max_sessions: int = 4096,
        metadata_limit: int = 64 * 1024 * 1024,
    ) -> None:
        if type(capacity) is not int or capacity < 0 or not 0 <= share <= 0.5:
            raise ValueError("invalid physical hold limit")
        if max_sessions <= 0 or metadata_limit <= 0:
            raise ValueError("invalid metadata hold limit")
        self.limit = int(capacity * share)
        self.max_sessions = max_sessions
        self.metadata_limit = metadata_limit
        self.leases: dict[str, Lease] = {}
        self.references: dict[bytes, int] = {}
        self.deadlines: list[tuple[int, str, int]] = []
        self.metadata_bytes = 0
        self.yields = 0

    def release(self, owner: str) -> None:
        lease = self.leases.pop(owner, None)
        if lease is None:
            return
        self.metadata_bytes -= lease.charge
        for key in lease.keys:
            count = self.references[key] - 1
            if count:
                self.references[key] = count
            else:
                del self.references[key]
        self._compact_deadlines()

    def _compact_deadlines(self) -> None:
        if len(self.deadlines) > max(64, 2 * len(self.leases)):
            self.deadlines = [
                (lease.deadline_ns, owner, lease.generation)
                for owner, lease in self.leases.items()
            ]
            heapq.heapify(self.deadlines)

    def oldest_owner(self) -> str | None:
        while self.deadlines:
            deadline, owner, generation = self.deadlines[0]
            lease = self.leases.get(owner)
            if lease is not None and (lease.deadline_ns, lease.generation) == (
                deadline,
                generation,
            ):
                return owner
            heapq.heappop(self.deadlines)
        return None

    def expire(self, now_ns: int) -> None:
        while (owner := self.oldest_owner()) is not None:
            if self.leases[owner].deadline_ns > now_ns:
                break
            self.release(owner)

    def yield_oldest(self) -> bool:
        owner = self.oldest_owner()
        if owner is None:
            return False
        self.release(owner)
        self.yields += 1
        return True

    def hold(
        self,
        owner: str,
        generation: int,
        deadline_ns: int,
        keys: frozenset[bytes],
        now_ns: int,
    ) -> HoldDecision:
        if not owner or len(owner) > 512 or generation < 0:
            raise ValueError("invalid hold owner")
        if any(not key or len(key) > 256 for key in keys):
            raise ValueError("invalid hold key")
        self.expire(now_ns)
        if deadline_ns <= now_ns:
            return HoldDecision.EXPIRED
        old = self.leases.get(owner)
        if old is not None and generation < old.generation:
            return HoldDecision.STALE
        if old is None and len(self.leases) >= self.max_sessions:
            return HoldDecision.SESSION_LIMIT
        charge = 2048 + len(owner.encode()) * 4 + sum(512 + len(key) for key in keys)
        old_charge = old.charge if old is not None else 0
        if self.metadata_bytes - old_charge + charge > self.metadata_limit:
            return HoldDecision.METADATA_LIMIT
        added = sum(key not in self.references for key in keys)
        removed = (
            sum(key not in keys and self.references[key] == 1 for key in old.keys)
            if old is not None
            else 0
        )
        if len(self.references) + added - removed > self.limit:
            return HoldDecision.PHYSICAL_LIMIT
        self.release(owner)
        self.leases[owner] = Lease(generation, deadline_ns, keys, charge)
        self.metadata_bytes += charge
        for key in keys:
            self.references[key] = self.references.get(key, 0) + 1
        heapq.heappush(self.deadlines, (deadline_ns, owner, generation))
        self._compact_deadlines()
        return HoldDecision.HELD

    def protection(self, transfer: Set[bytes]) -> ProtectedUnion:
        return ProtectedUnion(transfer, self.references)

    def clear(self) -> None:
        self.leases.clear()
        self.references.clear()
        self.deadlines.clear()
        self.metadata_bytes = 0
