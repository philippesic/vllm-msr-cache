from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from wait_hint_offload.contracts import Hint
from wait_hint_offload.observed import AdviceOutcome, ObservedStore


class RecencyTarget(Protocol):
    def refresh_prefix(self, keys: tuple[bytes, ...]) -> int: ...

    def ready_prefix(self, keys: tuple[bytes, ...]) -> Sequence[bytes]: ...


def advise(
    store: ObservedStore,
    target: RecencyTarget,
    hint: Hint,
    now_ns: int,
    clock_domain: str,
    enabled: bool,
) -> AdviceOutcome:
    return store.apply(
        hint,
        now_ns,
        clock_domain,
        enabled,
        target.refresh_prefix,
        lambda keys: len(target.ready_prefix(keys)),
    )
