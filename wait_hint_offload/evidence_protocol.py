from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wait_hint_offload.contracts import bounded_name, positive_integer
from wait_hint_offload.policy_protocol import PolicySession


@dataclass(frozen=True)
class EvidenceRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_tokens: int
    session: PolicySession
    ttl_ms: int | None

    @classmethod
    def parse(cls, raw: object) -> EvidenceRequest:
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "request_id",
            "prompt_token_ids",
            "output_tokens",
            "session",
            "ttl_ms",
        }:
            raise ValueError("closed evidence request required")
        if raw["schema"] != "pr-evidence-request-v1":
            raise ValueError("unsupported evidence request schema")
        tokens = raw["prompt_token_ids"]
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= 6144
            or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens)
        ):
            raise ValueError("invalid prompt token IDs")
        output = positive_integer(raw["output_tokens"], "output_tokens", 256)
        if len(tokens) + output > 6144:
            raise ValueError("context limit exceeded")
        ttl = raw["ttl_ms"]
        if ttl is not None:
            ttl = positive_integer(ttl, "ttl_ms", 60000)
        return cls(
            bounded_name(raw["request_id"], "request_id"),
            tuple(tokens),
            output,
            PolicySession.parse(raw["session"]),
            ttl,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pr-evidence-request-v1",
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "output_tokens": self.output_tokens,
            "session": self.session.to_dict(),
            "ttl_ms": self.ttl_ms,
        }
