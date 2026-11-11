from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum

from wait_hint_offload.config import (
    DEFAULT_METADATA_BYTES,
    MAX_EVENTS,
    MAX_METADATA_BYTES,
    MAX_SESSIONS,
)
from wait_hint_offload.contracts import (
    MAX_BOUNDARY,
    MAX_KEY_BYTES,
    MAX_TIMESTAMP,
    Hint,
    Namespace,
    bounded_name,
    generation_number,
    positive_integer,
)

ENTRY_OVERHEAD_BYTES = 1024


class RegistrationDecision(StrEnum):
    ACCEPTED = "accepted"
    GENERATION_NOT_NEW = "generation_not_new"
    MEMORY_CAPACITY = "memory_capacity"


@dataclass(frozen=True)
class RegistrationOutcome:
    decision: RegistrationDecision

    @property
    def accepted(self) -> bool:
        return self.decision is RegistrationDecision.ACCEPTED


class AdviceDecision(StrEnum):
    CLOCK_OR_DEADLINE = "clock_or_deadline"
    UNOBSERVED = "unobserved"
    IDENTITY_MISMATCH = "identity_mismatch"
    DISABLED = "disabled"
    EVENT_CONFLICT = "event_conflict"
    EVENT_CAPACITY = "event_capacity"
    MEMORY_CAPACITY = "memory_capacity"
    REFRESHED = "refreshed"
    NO_READY_PREFIX = "no_ready_prefix"


@dataclass(frozen=True)
class AdviceOutcome:
    decision: AdviceDecision
    refreshed_keys: int = 0
    ready_cpu_prefix_keys: int | None = None
    replayed: bool = False


def retained_bytes(value: object) -> int:
    size = sys.getsizeof(value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            size
            + ENTRY_OVERHEAD_BYTES
            + sum(retained_bytes(getattr(value, f.name)) for f in fields(value))
        )
    if isinstance(value, tuple):
        return size + sum(retained_bytes(item) for item in value)
    return size


@dataclass(frozen=True)
class Observation:
    incarnation: str
    generation: int
    prefix: str
    boundary: int
    namespace: Namespace
    observed_ns: int
    keys: tuple[bytes, ...]

    def __post_init__(self) -> None:
        bounded_name(self.incarnation, "incarnation")
        bounded_name(self.prefix, "prefix")
        generation_number(self.generation)
        positive_integer(self.boundary, "boundary", MAX_BOUNDARY)
        positive_integer(self.observed_ns, "observed_ns", MAX_TIMESTAMP)
        if not isinstance(self.namespace, Namespace):
            raise ValueError("namespace")
        if (
            type(self.keys) is not tuple
            or not 0 < len(self.keys) <= self.boundary
            or any(
                type(key) is not bytes or not 0 < len(key) <= MAX_KEY_BYTES
                for key in self.keys
            )
        ):
            raise ValueError("observation keys")


@dataclass
class Session:
    observation: Observation
    charged_bytes: int
    events: dict[str, tuple[Hint, AdviceOutcome]] = field(default_factory=dict)


class ObservedStore:
    def __init__(
        self,
        max_sessions: int,
        max_events: int,
        metadata_bytes: int = DEFAULT_METADATA_BYTES,
    ) -> None:
        positive_integer(max_sessions, "store bounds", MAX_SESSIONS)
        positive_integer(max_events, "store bounds", MAX_EVENTS)
        positive_integer(metadata_bytes, "metadata_bytes", MAX_METADATA_BYTES)
        self._max_sessions = max_sessions
        self._max_events = max_events
        self._budget = metadata_bytes
        self._sessions: OrderedDict[tuple[str, str], Session] = OrderedDict()
        self._charged_bytes = 0
        self._retired = 0

    @property
    def retired(self) -> int:
        return self._retired

    @property
    def charged_bytes(self) -> int:
        return self._charged_bytes

    def observation(self, scope: str, session: str) -> Observation | None:
        current = self._sessions.get((scope, session))
        return current.observation if current is not None else None

    def register(
        self, scope: str, session: str, observation: Observation
    ) -> RegistrationOutcome:
        bounded_name(scope, "scope")
        bounded_name(session, "session")
        identity = (scope, session)
        current = self._sessions.get(identity)
        if current is not None:
            old = current.observation
            if (
                old.incarnation == observation.incarnation
                and observation.generation <= old.generation
            ):
                return RegistrationOutcome(RegistrationDecision.GENERATION_NOT_NEW)
        victim = current
        if current is None and len(self._sessions) >= self._max_sessions:
            victim = next(iter(self._sessions.values()))
        charge = ENTRY_OVERHEAD_BYTES + retained_bytes((identity, observation))
        total = self._charged_bytes - (victim.charged_bytes if victim else 0) + charge
        if total > self._budget:
            return RegistrationOutcome(RegistrationDecision.MEMORY_CAPACITY)
        if current is None and victim is not None:
            self._sessions.popitem(last=False)
            self._retired += 1
        self._sessions[identity] = Session(observation, charge)
        self._sessions.move_to_end(identity)
        self._charged_bytes = total
        return RegistrationOutcome(RegistrationDecision.ACCEPTED)

    def apply(
        self,
        hint: Hint,
        now_ns: int,
        clock_domain: str,
        enabled: bool,
        refresh: Callable[[tuple[bytes, ...]], int],
        ready: Callable[[tuple[bytes, ...]], int],
    ) -> AdviceOutcome:
        session = self._sessions.get((hint.scope, hint.session))
        if hint.clock_domain != clock_domain:
            return AdviceOutcome(AdviceDecision.CLOCK_OR_DEADLINE)
        if session is None:
            return AdviceOutcome(AdviceDecision.UNOBSERVED)
        source = session.observation
        if (
            hint.incarnation,
            hint.generation,
            hint.prefix,
            hint.boundary,
            hint.namespace,
            hint.observed_ns,
        ) != (
            source.incarnation,
            source.generation,
            source.prefix,
            source.boundary,
            source.namespace,
            source.observed_ns,
        ):
            return AdviceOutcome(AdviceDecision.IDENTITY_MISMATCH)
        previous = session.events.get(hint.event)
        if previous is not None:
            if previous[0] != hint:
                return AdviceOutcome(AdviceDecision.EVENT_CONFLICT)
            return replace(previous[1], replayed=True)
        if not hint.published_ns <= now_ns < hint.deadline_ns:
            return AdviceOutcome(AdviceDecision.CLOCK_OR_DEADLINE)
        if len(session.events) >= self._max_events:
            return AdviceOutcome(AdviceDecision.EVENT_CAPACITY)
        charge = ENTRY_OVERHEAD_BYTES + retained_bytes(
            (hint.event, hint, AdviceOutcome(AdviceDecision.DISABLED, 0, 0))
        )
        if self._charged_bytes + charge > self._budget:
            return AdviceOutcome(AdviceDecision.MEMORY_CAPACITY)
        available = ready(source.keys)
        count = refresh(source.keys) if enabled else 0
        decision = (
            AdviceDecision.DISABLED
            if not enabled
            else AdviceDecision.REFRESHED
            if count
            else AdviceDecision.NO_READY_PREFIX
        )
        result = AdviceOutcome(decision, count, available)
        session.events[hint.event] = (hint, result)
        session.charged_bytes += charge
        self._charged_bytes += charge
        self._sessions.move_to_end((hint.scope, hint.session))
        return result

    def clear(self) -> None:
        self._sessions.clear()
        self._charged_bytes = 0
