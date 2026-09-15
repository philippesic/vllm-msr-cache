# SPDX-License-Identifier: Apache-2.0
"""Typed, causal contracts for known-prefix wait prefetch hints."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

WAIT_HINT_SCHEMA = "semantic-known-prefix-wait-hint-v1"
WAIT_HINT_PROVENANCE = "driver_declared_tool_wait_budget"
WAIT_HINT_GAPS_MS = frozenset({0, 50, 250, 750})
_HEX_256 = re.compile(r"[0-9a-f]{64}")
_OPAQUE_ID = re.compile(r"[A-Za-z0-9_.:|-]{1,192}")
_REQUIRED_NAMESPACE_FIELDS = frozenset(
    {
        "model_revision",
        "tokenizer_revision",
        "adapter_namespace",
        "layout_namespace",
    }
)
_HINT_FIELDS = frozenset(
    {
        "schema",
        "event_id",
        "owner_id",
        "run_id",
        "cell_id",
        "session_id",
        "generation",
        "observed_prefix_id",
        "prefix_boundary_tokens",
        "observation_available_ns",
        "input_publication_ns",
        "declared_wait_ms",
        "wait_provenance",
        "namespace",
    }
)
_FORBIDDEN_HINT_FIELDS = frozenset(
    {
        "future_suffix",
        "answer",
        "eventual_demand",
        "future_reuse",
        "actual_arrival_ns",
        "actual_resume_ns",
        "split",
        "schedule_role",
        "phase",
        "output",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass(frozen=True, slots=True)
class KnownPrefixWaitHint:
    event_id: str
    owner_id: str
    run_id: str
    cell_id: str
    session_id: str
    generation: int
    observed_prefix_id: str
    prefix_boundary_tokens: int
    observation_available_ns: int
    input_publication_ns: int
    declared_wait_ms: int | None
    wait_provenance: str | None
    namespace: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> KnownPrefixWaitHint:
        _require(raw.get("schema") == WAIT_HINT_SCHEMA, "wait hint schema")
        fields = set(raw)
        _require(
            not fields.intersection(_FORBIDDEN_HINT_FIELDS),
            "wait hint contains forbidden future or diagnostic fields",
        )
        _require(fields == _HINT_FIELDS, "wait hint fields must be exact")
        for field in ("event_id", "observed_prefix_id"):
            value = raw.get(field)
            _require(
                isinstance(value, str) and _HEX_256.fullmatch(value) is not None,
                f"wait hint {field} must be a 256-bit hex digest",
            )
        for field in ("owner_id", "run_id", "cell_id", "session_id"):
            value = raw.get(field)
            _require(
                isinstance(value, str) and _OPAQUE_ID.fullmatch(value) is not None,
                f"wait hint {field} must be opaque",
            )
        generation = raw.get("generation")
        boundary = raw.get("prefix_boundary_tokens")
        available_ns = raw.get("observation_available_ns")
        publication_ns = raw.get("input_publication_ns")
        _require(_nonnegative_int(generation), "wait hint generation")
        _require(
            _nonnegative_int(boundary) and boundary > 0,
            "wait hint prefix boundary must be positive",
        )
        _require(
            _nonnegative_int(available_ns)
            and _nonnegative_int(publication_ns)
            and available_ns <= publication_ns,
            "wait hint causal clocks are invalid",
        )
        declared_wait_ms = raw.get("declared_wait_ms")
        wait_provenance = raw.get("wait_provenance")
        if declared_wait_ms is None:
            _require(
                wait_provenance is None,
                "wait provenance requires a declared estimate",
            )
        else:
            _require(
                _nonnegative_int(declared_wait_ms)
                and declared_wait_ms in WAIT_HINT_GAPS_MS,
                "wait hint declared estimate is outside the frozen screen",
            )
            _require(
                wait_provenance == WAIT_HINT_PROVENANCE,
                "wait hint estimate provenance mismatch",
            )
        namespace = raw.get("namespace")
        _require(
            isinstance(namespace, Mapping)
            and set(namespace) == _REQUIRED_NAMESPACE_FIELDS
            and all(
                isinstance(value, str) and value
                for value in namespace.values()
            ),
            "wait hint namespace must be exact and complete",
        )
        return cls(
            event_id=raw["event_id"],
            owner_id=raw["owner_id"],
            run_id=raw["run_id"],
            cell_id=raw["cell_id"],
            session_id=raw["session_id"],
            generation=generation,
            observed_prefix_id=raw["observed_prefix_id"],
            prefix_boundary_tokens=boundary,
            observation_available_ns=available_ns,
            input_publication_ns=publication_ns,
            declared_wait_ms=declared_wait_ms,
            wait_provenance=wait_provenance,
            namespace=tuple(sorted(namespace.items())),
        )

    def namespace_dict(self) -> dict[str, str]:
        return dict(self.namespace)

    def complete_page_boundary(self, tokens_per_chunk: int) -> int:
        _require(
            _nonnegative_int(tokens_per_chunk) and tokens_per_chunk > 0,
            "tokens_per_chunk must be positive",
        )
        return self.prefix_boundary_tokens // tokens_per_chunk * tokens_per_chunk

    def remaining_wait_ns(self, now_ns: int) -> int | None:
        _require(_nonnegative_int(now_ns), "wait hint current clock")
        if self.declared_wait_ms is None:
            return None
        elapsed_ns = max(0, now_ns - self.input_publication_ns)
        return max(0, self.declared_wait_ms * 1_000_000 - elapsed_ns)


@dataclass(slots=True)
class WaitHintLease:
    hint: KnownPrefixWaitHint
    nomination_deadline_ns: int
    expires_at_ns: int
    state: str = "published"

    def __post_init__(self) -> None:
        _require(
            _nonnegative_int(self.nomination_deadline_ns)
            and self.nomination_deadline_ns >= self.hint.input_publication_ns,
            "wait hint nomination deadline",
        )
        _require(
            _nonnegative_int(self.expires_at_ns)
            and self.expires_at_ns >= self.nomination_deadline_ns,
            "wait hint lease expiry",
        )

    def active_at(self, now_ns: int) -> bool:
        _require(_nonnegative_int(now_ns), "wait hint current clock")
        return self.state in {"published", "reserved", "in_flight", "ready"} and (
            now_ns <= self.expires_at_ns
        )

    def expire(self) -> None:
        self.state = "expired"

    def cancel(self) -> None:
        self.state = "cancelled"

    def resume(self) -> None:
        self.state = "resumed"


@dataclass(frozen=True, slots=True)
class UsefulOnTimeSelection:
    selected_keys: tuple[object, ...]
    status: str
    remaining_wait_ns: int | None
    estimated_transfer_ns_per_byte: float | None
    estimated_splice_ns_per_byte: float | None
    bytes_per_key: int | None
    estimated_selected_ns: int | None
    fallback: str | None


def select_useful_on_time(
    candidate_keys: list[object],
    budget: int,
    *,
    remaining_wait_ns: int | None,
    bytes_per_key: int | None,
    estimated_transfer_ns_per_byte: float | None,
    estimated_splice_ns_per_byte: float | None,
) -> UsefulOnTimeSelection:
    """Select the longest exact chain prefix estimated ready by resume.

    Formula: per-key cost is ``ceil(bytes_per_key * (transfer_ns_per_byte +
    splice_ns_per_byte))`` and the selected prefix is the largest ``n <=
    budget`` whose cumulative cost is at most the declared wait budget
    remaining at decision time. Chain order is the deterministic tie-break.
    Missing cost or wait estimates use the explicit request-order fallback
    over the same eligible keys and budget.
    """
    _require(_nonnegative_int(budget), "useful-on-time budget")
    if budget == 0 or not candidate_keys:
        return UsefulOnTimeSelection(
            (),
            "empty_budget_or_candidates",
            remaining_wait_ns,
            None,
            None,
            None,
            0,
            None,
        )
    estimates_valid = (
        _nonnegative_int(remaining_wait_ns)
        and isinstance(bytes_per_key, int)
        and not isinstance(bytes_per_key, bool)
        and bytes_per_key > 0
        and isinstance(estimated_transfer_ns_per_byte, (int, float))
        and not isinstance(estimated_transfer_ns_per_byte, bool)
        and math.isfinite(float(estimated_transfer_ns_per_byte))
        and float(estimated_transfer_ns_per_byte) > 0.0
        and isinstance(estimated_splice_ns_per_byte, (int, float))
        and not isinstance(estimated_splice_ns_per_byte, bool)
        and math.isfinite(float(estimated_splice_ns_per_byte))
        and float(estimated_splice_ns_per_byte) > 0.0
    )
    if not estimates_valid:
        selected = tuple(candidate_keys[:budget])
        return UsefulOnTimeSelection(
            selected,
            "fallback_request_order",
            remaining_wait_ns,
            (
                float(estimated_transfer_ns_per_byte)
                if isinstance(estimated_transfer_ns_per_byte, (int, float))
                and not isinstance(estimated_transfer_ns_per_byte, bool)
                else None
            ),
            (
                float(estimated_splice_ns_per_byte)
                if isinstance(estimated_splice_ns_per_byte, (int, float))
                and not isinstance(estimated_splice_ns_per_byte, bool)
                else None
            ),
            bytes_per_key,
            None,
            "missing_wait_or_cost_estimate",
        )
    per_key_ns = max(
        1,
        math.ceil(
            bytes_per_key
            * (
                float(estimated_transfer_ns_per_byte)
                + float(estimated_splice_ns_per_byte)
            )
        ),
    )
    selected_count = min(
        budget,
        len(candidate_keys),
        int(remaining_wait_ns) // per_key_ns,
    )
    selected = tuple(candidate_keys[:selected_count])
    return UsefulOnTimeSelection(
        selected,
        "selected_on_time_prefix" if selected else "no_prefix_ready_by_deadline",
        int(remaining_wait_ns),
        float(estimated_transfer_ns_per_byte),
        float(estimated_splice_ns_per_byte),
        bytes_per_key,
        selected_count * per_key_ns,
        None,
    )
