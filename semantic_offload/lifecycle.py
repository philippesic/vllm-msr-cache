# SPDX-License-Identifier: Apache-2.0
"""Typed, opt-in producer seam for lifecycle release experiments.

The adapter intentionally does not infer lifecycle facts from request
completion. An application or higher-level scheduler must register ownership,
advance the owner generation, and submit an explicit hint. Physical removal
and ref-count safety remain manager-owned.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext

from semantic_offload.manager import LifecycleReleaseResult


@dataclass(frozen=True)
class LifecycleReleaseHint:
    event_id: str
    owner_id: str
    generation: int
    keys: tuple[OffloadKey, ...]


@dataclass(frozen=True)
class LifecycleHintResult:
    """Producer-visible status for one absent, suppressed, or applied hint."""

    status: str
    event_id: str = ""
    owner_id: str = ""
    generation: int = 0
    reason: str = ""
    manager_result: LifecycleReleaseResult | None = None


class LifecycleReleaseController:
    """Validate explicit hints before delegating to a lifecycle-aware manager."""

    def __init__(self, manager, *, enabled: bool = False) -> None:
        self.manager = manager
        self.enabled = enabled
        # Event IDs are idempotent within an owner generation. A new
        # generation may legitimately reuse an application-local event ID,
        # while a delayed event from the old generation still reaches the
        # manager and is rejected by its generation check.
        self._seen_event_ids: set[tuple[str, int, str]] = set()
        self._ledger: list[LifecycleHintResult] = []

    def register_owner(
        self,
        owner_id: str,
        generation: int,
        keys: Collection[OffloadKey],
        *,
        state: str = "active",
    ) -> None:
        self.manager.register_lifecycle_owner(owner_id, generation, keys, state=state)

    def set_owner_state(self, owner_id: str, generation: int, state: str) -> None:
        self.manager.set_lifecycle_owner_state(owner_id, generation, state)

    def retire_owner(self, owner_id: str, generation: int) -> None:
        """Release detailed owner bookkeeping after a safe terminal state."""
        self.manager.retire_lifecycle_owner(owner_id, generation)
        self._seen_event_ids = {
            identity
            for identity in self._seen_event_ids
            if identity[:2] != (owner_id, generation)
        }

    def bookkeeping_stats(self) -> dict[str, int]:
        """Return controller and manager lifecycle-state counts."""
        manager_stats = self.manager.lifecycle_bookkeeping_stats()
        return {
            **manager_stats,
            "seen_event_identities": len(self._seen_event_ids),
            "pending_controller_ledger_entries": len(self._ledger),
        }

    def submit(
        self,
        hint: LifecycleReleaseHint | None,
        req_context: ReqContext,
    ) -> LifecycleHintResult:
        if hint is None:
            result = LifecycleHintResult("absent", reason="no_hint")
            self._ledger.append(result)
            return result
        if not self.enabled:
            result = LifecycleHintResult(
                "suppressed",
                event_id=hint.event_id,
                owner_id=hint.owner_id,
                generation=hint.generation,
                reason="controller_disabled",
            )
            self._ledger.append(result)
            return result
        event_identity = (hint.owner_id, hint.generation, hint.event_id)
        if event_identity in self._seen_event_ids:
            result = LifecycleHintResult(
                "noop",
                event_id=hint.event_id,
                owner_id=hint.owner_id,
                generation=hint.generation,
                reason="duplicate_event",
            )
            self._ledger.append(result)
            return result
        self._seen_event_ids.add(event_identity)
        if (
            not hint.event_id
            or not hint.owner_id
            or hint.generation < 0
            or not hint.keys
            or len(set(hint.keys)) != len(hint.keys)
        ):
            result = LifecycleHintResult(
                "rejected",
                event_id=hint.event_id,
                owner_id=hint.owner_id,
                generation=hint.generation,
                reason="invalid_hint",
            )
            self._ledger.append(result)
            return result
        manager_result = self.manager.release_lifecycle_owner(
            hint.owner_id,
            hint.generation,
            hint.keys,
            req_context,
        )
        result = LifecycleHintResult(
            manager_result.outcome,
            event_id=hint.event_id,
            owner_id=hint.owner_id,
            generation=hint.generation,
            reason=manager_result.reason,
            manager_result=manager_result,
        )
        self._ledger.append(result)
        return result

    def drain_ledger(self) -> list[LifecycleHintResult]:
        records = self._ledger
        self._ledger = []
        return records
