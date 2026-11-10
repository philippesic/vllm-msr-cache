from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from wait_hint_offload.contracts import Hint, Namespace


@dataclass(frozen=True)
class Observation:
    incarnation: str
    generation: int
    prefix: str
    boundary: int
    namespace: Namespace
    observed_ns: int
    keys: tuple[bytes, ...]


@dataclass
class Session:
    observation: Observation
    events: OrderedDict[str, tuple[Hint, str, int]] = field(default_factory=OrderedDict)


class ObservedStore:
    def __init__(self, max_sessions: int, max_events: int) -> None:
        if max_sessions <= 0 or max_events <= 0:
            raise ValueError("store bounds")
        self.max_sessions = max_sessions
        self.max_events = max_events
        self.sessions: OrderedDict[tuple[str, str], Session] = OrderedDict()
        self.retired = 0

    def register(self, scope: str, session: str, observation: Observation) -> bool:
        identity = (scope, session)
        current = self.sessions.get(identity)
        if current is not None:
            old = current.observation
            if (
                old.incarnation != observation.incarnation
                or observation.generation < old.generation
            ):
                return False
            if observation.generation == old.generation:
                return old == observation
        if current is None and len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)
            self.retired += 1
        self.sessions[identity] = Session(observation)
        self.sessions.move_to_end(identity)
        return True

    def clear(self) -> None:
        self.sessions.clear()
