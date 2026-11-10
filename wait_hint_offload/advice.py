from __future__ import annotations

from typing import Protocol

from wait_hint_offload.contracts import Hint
from wait_hint_offload.observed import ObservedStore


class RecencyTarget(Protocol):
    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int: ...


def advise(
    store: ObservedStore,
    target: RecencyTarget,
    hint: Hint,
    now_ns: int,
    clock_domain: str,
    enabled: bool,
) -> tuple[str, int]:
    if (
        hint.clock_domain != clock_domain
        or not hint.published_ns <= now_ns < hint.deadline_ns
    ):
        return "clock_or_deadline", 0
    session = store.sessions.get((hint.scope, hint.session))
    if session is None:
        return "unobserved", 0
    source = session.observation
    if (
        hint.incarnation,
        hint.generation,
        hint.prefix,
        hint.boundary,
        hint.namespace,
    ) != (
        source.incarnation,
        source.generation,
        source.prefix,
        source.boundary,
        source.namespace,
    ) or source.observed_ns != hint.observed_ns:
        return "identity_mismatch", 0
    if not enabled:
        return "disabled", 0
    previous = session.events.get(hint.event)
    if previous is not None:
        if previous[0] != hint:
            return "event_conflict", 0
        return previous[1], previous[2]
    if len(session.events) >= store.max_events:
        return "event_capacity", 0
    count = target.refresh_prefix(source.keys)
    decision = "refreshed" if count else "no_ready_prefix"
    session.events[hint.event] = (hint, decision, count)
    store.sessions.move_to_end((hint.scope, hint.session))
    return decision, count
