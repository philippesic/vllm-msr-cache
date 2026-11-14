from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from wait_hint_offload.contracts import (
    bounded_name,
    generation_number,
    positive_integer,
)


@dataclass(frozen=True)
class PolicySession:
    scope: str
    session: str
    incarnation: str
    generation: int

    @property
    def owner(self) -> str:
        return json.dumps(
            [self.scope, self.session, self.incarnation],
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def parse(cls, raw: object) -> PolicySession:
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "scope",
            "session",
            "incarnation",
            "generation",
        }:
            raise ValueError("closed policy session required")
        if raw["schema"] != "wait-hint-policy-session-v1":
            raise ValueError("unsupported policy session schema")
        return cls(
            bounded_name(raw["scope"], "scope"),
            bounded_name(raw["session"], "session"),
            bounded_name(raw["incarnation"], "incarnation"),
            generation_number(raw["generation"]),
        )


@dataclass(frozen=True)
class PolicyAction:
    action_id: str
    session: PolicySession
    kind: str
    ttl_ms: int

    @classmethod
    def parse(cls, raw: object) -> PolicyAction:
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "action_id",
            "session",
            "kind",
            "ttl_ms",
        }:
            raise ValueError("closed policy action required")
        if raw["schema"] != "wait-hint-policy-action-v1":
            raise ValueError("unsupported policy action schema")
        if raw["kind"] not in ("hold", "cancel"):
            raise ValueError("unsupported policy action")
        ttl = positive_integer(raw["ttl_ms"], "ttl_ms", 86400000)
        return cls(
            bounded_name(raw["action_id"], "action_id"),
            PolicySession.parse(raw["session"]),
            raw["kind"],
            ttl,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "wait-hint-policy-action-v1",
            "action_id": self.action_id,
            "session": {
                "schema": "wait-hint-policy-session-v1",
                "scope": self.session.scope,
                "session": self.session.session,
                "incarnation": self.session.incarnation,
                "generation": self.session.generation,
            },
            "kind": self.kind,
            "ttl_ms": self.ttl_ms,
        }
