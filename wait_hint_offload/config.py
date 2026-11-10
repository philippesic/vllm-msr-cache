from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from wait_hint_offload.contracts import Namespace, positive_integer


@dataclass(frozen=True)
class Settings:
    namespace: Namespace
    cpu_bytes_to_use: int
    advice_enabled: bool
    max_sessions: int
    max_events: int
    store_threshold: int

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Settings:
        allowed = {
            "namespace",
            "cpu_bytes_to_use",
            "advice_enabled",
            "max_sessions",
            "max_events",
            "store_threshold",
            "eviction_policy",
        }
        if set(raw) - allowed:
            raise ValueError(f"unknown settings: {sorted(set(raw) - allowed)}")
        if raw.get("eviction_policy", "arc") != "arc":
            raise ValueError("ARC required")
        enabled = raw.get("advice_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("advice_enabled must be boolean")
        threshold = raw.get("store_threshold", 0)
        if type(threshold) is not int or threshold != 0:
            raise ValueError("store_threshold")
        return cls(
            Namespace.parse(raw.get("namespace")),
            positive_integer(raw.get("cpu_bytes_to_use"), "cpu_bytes_to_use", 2**50),
            enabled,
            positive_integer(raw.get("max_sessions", 4096), "max_sessions", 100_000),
            positive_integer(raw.get("max_events", 256), "max_events", 100_000),
            threshold,
        )

    def mechanisms(self) -> dict[str, object]:
        return {
            "schema": "wait-hint-mechanisms-v1",
            "cpu_arc_recency_advice": self.advice_enabled,
            "semantic_scoring": "absent",
            "gpu_prefetch": "absent",
        }
