from __future__ import annotations

from dataclasses import dataclass


class RetentionScore(float):
    pass


@dataclass(frozen=True)
class ExactKVIdentity:
    prefix_tokens: tuple[int, ...]
    model: str
    adapter: str | None
    positions: tuple[int, ...]
    layout: str


@dataclass(frozen=True)
class RetentionCandidate:
    identity: ExactKVIdentity
    score: RetentionScore


def same_kv_identity(stored: ExactKVIdentity, requested: ExactKVIdentity) -> bool:
    return stored == requested
