from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from wait_hint_offload.contracts import Namespace, positive_integer

DEFAULT_MAX_SESSIONS = 4096
DEFAULT_MAX_EVENTS = 256
MAX_SESSIONS = 100_000
MAX_EVENTS = 100_000
DEFAULT_METADATA_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 256 * 1024 * 1024
MAX_CPU_BYTES = 2**50
MECHANISMS = ("recency", "promote-hold")


@dataclass(frozen=True)
class Settings:
    namespace: Namespace
    cpu_bytes_to_use: int
    advice_enabled: bool
    max_sessions: int
    max_events: int
    metadata_bytes: int = DEFAULT_METADATA_BYTES
    advice_mechanism: str = "recency"

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, Namespace):
            raise ValueError("namespace")
        if type(self.advice_enabled) is not bool:
            raise ValueError("advice_enabled must be boolean")
        if self.advice_mechanism not in MECHANISMS:
            raise ValueError("unknown advice_mechanism")
        for name, maximum in (
            ("cpu_bytes_to_use", MAX_CPU_BYTES),
            ("max_sessions", MAX_SESSIONS),
            ("max_events", MAX_EVENTS),
            ("metadata_bytes", MAX_METADATA_BYTES),
        ):
            positive_integer(getattr(self, name), name, maximum)

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Settings:
        allowed = {
            "namespace",
            "cpu_bytes_to_use",
            "advice_enabled",
            "max_sessions",
            "max_events",
            "metadata_bytes",
            "advice_mechanism",
        }
        if set(raw) - allowed:
            raise ValueError(f"unknown settings: {sorted(set(raw) - allowed)}")
        mechanism = raw.get("advice_mechanism", "recency")
        if not isinstance(mechanism, str) or mechanism not in MECHANISMS:
            raise ValueError("unknown advice_mechanism")
        enabled = raw.get("advice_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("advice_enabled must be boolean")
        return cls(
            Namespace.parse(raw.get("namespace")),
            positive_integer(
                raw.get("cpu_bytes_to_use"), "cpu_bytes_to_use", MAX_CPU_BYTES
            ),
            enabled,
            positive_integer(
                raw.get("max_sessions", DEFAULT_MAX_SESSIONS),
                "max_sessions",
                MAX_SESSIONS,
            ),
            positive_integer(
                raw.get("max_events", DEFAULT_MAX_EVENTS), "max_events", MAX_EVENTS
            ),
            positive_integer(
                raw.get("metadata_bytes", DEFAULT_METADATA_BYTES),
                "metadata_bytes",
                MAX_METADATA_BYTES,
            ),
            mechanism,
        )

    def mechanisms(self) -> dict[str, object]:
        return {
            "schema": "wait-hint-mechanisms-v2",
            "advice": self.advice_mechanism if self.advice_enabled else "disabled",
        }
