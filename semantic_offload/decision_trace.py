# SPDX-License-Identifier: Apache-2.0
"""Leakage-safe, decision-time semantic policy feature traces.

The trace is deliberately a separate stream from the preservation event.  A
preservation event describes a request's eventual reuse outcome; this stream
describes only the candidates and state visible when an eviction decision was
made.  Keeping the two records separate makes it possible to audit that a
future-reuse label was not accidentally fed back into the same decision.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence

TRACE_SCHEMA = "semantic-decision-feature-v1"
TRACE_EVENT = "eviction_decision"
TRACE_MARKER = "SEMANTIC_DECISION_FEATURE "

_EXTERNAL_REQUEST_ID_RE = re.compile(r"^sc\|([^|]+)\|([^|]+)\|([^|]+)\|sample-(\d+)$")
_INTERNAL_REQUEST_ID_RE = re.compile(
    r"^cmpl-(sc\|[^|]+\|[^|]+\|[^|]+\|sample-\d+)-\d+-([0-9a-f]{8})$"
)
_HEX_RE = re.compile(r"[0-9a-f]+")
_FORBIDDEN_FIELDS = frozenset(
    {
        "outcome",
        "reason",
        "terminal_status",
        "terminal_reason",
        "reused_tokens",
        "loaded_tokens",
        "recomputed_tokens",
        "preservation_outcome",
        "future_reuse",
        "future_reuse_label",
        "label",
    }
)
_ALLOWED_FIELDS = frozenset(
    {
        "schema",
        "event",
        "event_instance",
        "expected_event_count",
        "request_id",
        "external_request_id",
        "run_id",
        "cell_id",
        "sample_index",
        "sequence_ordinal",
        "candidate_ordinal",
        "candidate_count",
        "selected",
        "offload_key",
        "group_idx",
        "cpu_block_id",
        "semantic_relevance_score",
        "semantic_relevance_score_status",
        "relevance_ema",
        "relevance_ema_status",
        "rank",
        "recency_norm",
        "frequency",
        "frequency_status",
        "decision_timestamp_ns",
        "clock_domain",
        "scoring_cpu_time_ns",
        "policy_cpu_time_ns",
        "critical_path_wall_time_ns",
        "synchronization_wait_ns",
        "synchronization_status",
        "policy",
        "policy_state",
        "candidate_policy_segment",
        "instrumentation_cpu_time_ns",
        "instrumentation_wall_time_ns",
    }
)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _request_provenance(
    req_id: object, params: Mapping[str, object] | None
) -> dict[str, object]:
    """Derive canonical request/run/cell identity without inventing IDs."""
    params = params or {}
    raw_req_id = req_id if isinstance(req_id, str) else None
    external = params.get("external_request_id", params.get("semantic_request_id"))
    if not isinstance(external, str):
        external = (
            raw_req_id
            if raw_req_id and _EXTERNAL_REQUEST_ID_RE.fullmatch(raw_req_id)
            else None
        )
    if external is None and raw_req_id:
        match = _INTERNAL_REQUEST_ID_RE.fullmatch(raw_req_id)
        external = match.group(1) if match is not None else None
    match = (
        _EXTERNAL_REQUEST_ID_RE.fullmatch(external)
        if isinstance(external, str)
        else None
    )
    return {
        "request_id": raw_req_id,
        "external_request_id": external,
        "run_id": params.get("run_id", params.get("semantic_run_id"))
        or (match.group(1) if match else None),
        "cell_id": params.get("cell_id", params.get("semantic_cell_id"))
        or (match.group(2) if match else None),
        "sample_index": (
            params.get("sample_index", params.get("semantic_sample_index"))
            if params.get("sample_index", params.get("semantic_sample_index"))
            is not None
            else (int(match.group(4)) if match else None)
        ),
    }


def build_decision_feature_records(
    *,
    context: Mapping[str, object],
    decision_sequence: int,
    candidates: Sequence[tuple[object, object]],
    ranked_keys: Sequence[object],
    selected_keys: set[object],
    relevance: Mapping[object, object],
    latest_relevance: Mapping[object, object],
    sink: Callable[[list[dict[str, object]]], None],
    decision_timestamp_ns: int | None = None,
    scoring_cpu_time_ns: int | None = None,
    policy_cpu_time_ns: int | None = None,
    critical_path_wall_time_ns: int | None = None,
    synchronization_wait_ns: int = 0,
    synchronization_status: str = "no_explicit_synchronization",
    policy: str = "semantic",
    policy_state: Mapping[str, object] | None = None,
    candidate_policy_segments: Mapping[object, str] | None = None,
) -> list[dict[str, object]]:
    """Build one feature record per candidate and publish the whole decision."""
    if not isinstance(decision_sequence, int) or decision_sequence < 0:
        raise ValueError("decision feature sequence must be a non-negative integer")
    if len(ranked_keys) != len(candidates) or len(set(ranked_keys)) != len(ranked_keys):
        raise ValueError("decision feature ranking has wrong candidate cardinality")
    candidate_by_key = {key: block for key, block in candidates}
    if len(candidate_by_key) != len(candidates) or set(candidate_by_key) != set(
        ranked_keys
    ):
        raise ValueError("decision feature candidates and ranking disagree")
    total = len(candidates)
    recency_position = {key: index for index, (key, _) in enumerate(candidates)}
    records: list[dict[str, object]] = []
    external_id = context.get("external_request_id")
    run_id = context.get("run_id")
    cell_id = context.get("cell_id")
    request_id = context.get("request_id")
    for rank, key in enumerate(ranked_keys):
        if (
            not isinstance(key, bytes)
            or len(key) < 4
            or _HEX_RE.fullmatch(key.hex()) is None
        ):
            raise ValueError("decision feature OffloadKey is not canonical bytes")
        block = candidate_by_key[key]
        relevance_value = relevance.get(key)
        latest_value = latest_relevance.get(key)
        if relevance_value is not None and not _finite_number(relevance_value):
            raise ValueError("decision feature relevance_ema is not finite")
        if latest_value is not None and not _finite_number(latest_value):
            raise ValueError("decision feature relevance score is not finite")
        record = {
            "schema": TRACE_SCHEMA,
            "event": TRACE_EVENT,
            "event_instance": 1,
            "expected_event_count": 1,
            "request_id": request_id,
            "external_request_id": external_id,
            "run_id": run_id,
            "cell_id": cell_id,
            "sample_index": context.get("sample_index"),
            "sequence_ordinal": decision_sequence,
            "candidate_ordinal": recency_position[key],
            "candidate_count": total,
            "selected": key in selected_keys,
            "offload_key": key.hex(),
            "group_idx": int.from_bytes(key[-4:], "big", signed=False),
            "cpu_block_id": int(block.block_id),
            "semantic_relevance_score": (
                float(latest_value) if latest_value is not None else None
            ),
            "semantic_relevance_score_status": (
                "available" if latest_value is not None else "unavailable"
            ),
            "relevance_ema": (
                float(relevance_value) if relevance_value is not None else None
            ),
            "relevance_ema_status": (
                "available" if relevance_value is not None else "unavailable"
            ),
            "rank": rank,
            "recency_norm": (recency_position[key] / (total - 1) if total > 1 else 1.0),
            # CPUOffloadingManager does not expose an authoritative access
            # frequency. None is intentional and must stay unavailable in
            # offline fitting rather than being treated as zero.
            "frequency": None,
            "frequency_status": "unavailable",
            # These timestamps/timers describe only causal decision work.
            # Event serialization happens after the decision and is excluded.
            "decision_timestamp_ns": decision_timestamp_ns,
            "clock_domain": (
                "host_monotonic" if decision_timestamp_ns is not None else None
            ),
            "scoring_cpu_time_ns": scoring_cpu_time_ns,
            "policy_cpu_time_ns": policy_cpu_time_ns,
            "critical_path_wall_time_ns": critical_path_wall_time_ns,
            "synchronization_wait_ns": synchronization_wait_ns,
            "synchronization_status": synchronization_status,
            "policy": policy,
            "policy_state": dict(policy_state or {}),
            "candidate_policy_segment": (
                candidate_policy_segments.get(key)
                if candidate_policy_segments is not None
                else None
            ),
            "instrumentation_cpu_time_ns": None,
            "instrumentation_wall_time_ns": None,
        }
        records.append(record)
    sink(records)
    return records


def validate_decision_feature_record(
    record: Mapping[str, object], *, run_id: object = None, cell_id: object = None
) -> None:
    """Reject malformed, misbound, or outcome-leaking feature records."""
    if not isinstance(record, Mapping):
        raise ValueError(  # noqa: TRY004 - malformed records use one public error type
            "decision feature record is not an object"
        )
    if any(field in record for field in _FORBIDDEN_FIELDS):
        raise ValueError("decision feature record contains an outcome/leakage field")
    if set(record) - _ALLOWED_FIELDS:
        raise ValueError("decision feature record has unknown fields")
    if record.get("schema") != TRACE_SCHEMA or record.get("event") != TRACE_EVENT:
        raise ValueError("decision feature record has the wrong schema")
    if record.get("event_instance") != 1 or record.get("expected_event_count") != 1:
        raise ValueError("decision feature record has invalid cardinality marker")
    external = record.get("external_request_id")
    request_id = record.get("request_id")
    match = (
        _EXTERNAL_REQUEST_ID_RE.fullmatch(external)
        if isinstance(external, str)
        else None
    )
    internal = (
        _INTERNAL_REQUEST_ID_RE.fullmatch(request_id)
        if isinstance(request_id, str)
        else None
    )
    if match is None or internal is None or internal.group(1) != external:
        raise ValueError("decision feature record has non-canonical request identity")
    if record.get("run_id") != match.group(1) or record.get("cell_id") != match.group(
        2
    ):
        raise ValueError(
            "decision feature record run/cell conflicts with request identity"
        )
    if run_id is not None and record.get("run_id") != run_id:
        raise ValueError("decision feature record run_id conflicts with cell")
    if cell_id is not None and record.get("cell_id") != cell_id:
        raise ValueError("decision feature record cell_id conflicts with cell")
    if record.get("sample_index") != int(match.group(4)):
        raise ValueError(
            "decision feature record sample index conflicts with request identity"
        )
    for field in (
        "sequence_ordinal",
        "candidate_ordinal",
        "candidate_count",
        "rank",
        "group_idx",
        "cpu_block_id",
    ):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"decision feature record has invalid {field}")
    if (
        record["candidate_count"] == 0
        or record["candidate_ordinal"] >= record["candidate_count"]
    ):
        raise ValueError("decision feature candidate cardinality is invalid")
    if record["rank"] >= record["candidate_count"]:
        raise ValueError("decision feature rank is invalid")
    if not isinstance(record.get("selected"), bool):
        raise ValueError(  # noqa: TRY004 - malformed records use one public error type
            "decision feature selected flag is invalid"
        )
    key = record.get("offload_key")
    if (
        not isinstance(key, str)
        or len(key) < 8
        or len(key) % 2
        or _HEX_RE.fullmatch(key) is None
    ):
        raise ValueError("decision feature OffloadKey is not canonical lowercase hex")
    if (
        int.from_bytes(bytes.fromhex(key)[-4:], "big", signed=False)
        != record["group_idx"]
    ):
        raise ValueError("decision feature group identity conflicts with OffloadKey")
    for field in ("semantic_relevance_score", "relevance_ema"):
        value = record.get(field)
        if value is not None and not _finite_number(value):
            raise ValueError(f"decision feature {field} is unavailable or non-finite")
        status = record.get(f"{field}_status")
        expected_status = "available" if value is not None else "unavailable"
        if status != expected_status:
            raise ValueError(f"decision feature {field} availability conflicts")
    if (
        not _finite_number(record.get("recency_norm"))
        or not 0.0 <= float(record["recency_norm"]) <= 1.0
    ):
        raise ValueError("decision feature recency_norm is invalid")
    frequency = record.get("frequency")
    if frequency is None:
        if record.get("frequency_status") != "unavailable":
            raise ValueError(
                "unavailable decision feature frequency lacks explicit status"
            )
    else:
        if not _finite_number(frequency):
            raise ValueError("decision feature frequency is invalid")
        if record.get("frequency_status") != "available":
            raise ValueError("finite decision feature frequency has invalid status")
    decision_timestamp_ns = record.get("decision_timestamp_ns")
    if decision_timestamp_ns is None:
        if record.get("clock_domain") is not None:
            raise ValueError("unavailable decision timestamp has a clock domain")
    elif (
        not isinstance(decision_timestamp_ns, int)
        or isinstance(decision_timestamp_ns, bool)
        or decision_timestamp_ns < 0
        or record.get("clock_domain") != "host_monotonic"
    ):
        raise ValueError("decision timestamp is invalid")
    for field in (
        "scoring_cpu_time_ns",
        "policy_cpu_time_ns",
        "critical_path_wall_time_ns",
        "instrumentation_cpu_time_ns",
        "instrumentation_wall_time_ns",
    ):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"decision feature has invalid {field}")
    sync_wait = record.get("synchronization_wait_ns")
    if sync_wait is not None and (
        not isinstance(sync_wait, int) or isinstance(sync_wait, bool) or sync_wait < 0
    ):
        raise ValueError("decision synchronization wait is invalid")
    if record.get("synchronization_status") not in (
        None,
        "no_explicit_synchronization",
        "measured",
        "unavailable",
    ):
        raise ValueError("decision synchronization status is invalid")
    if "policy" in record and (
        not isinstance(record.get("policy"), str) or not record["policy"]
    ):
        raise ValueError("decision policy identity is invalid")
    if "policy_state" in record and not isinstance(record.get("policy_state"), Mapping):
        raise ValueError("decision policy state is invalid")
    segment = record.get("candidate_policy_segment")
    if segment is not None and not isinstance(segment, str):
        raise ValueError("decision candidate policy segment is invalid")


def validate_decision_feature_records(
    records: Sequence[Mapping[str, object]],
    *,
    expected_request_ids: Sequence[str],
    run_id: object,
    cell_id: object,
) -> list[dict[str, object]]:
    """Validate a trace stream and return canonical records in sequence order."""
    expected = set(expected_request_ids)
    if len(expected) != len(expected_request_ids):
        raise ValueError("decision feature expected request IDs are duplicated")
    validated: list[dict[str, object]] = []
    for record in records:
        validate_decision_feature_record(record, run_id=run_id, cell_id=cell_id)
        if record["external_request_id"] not in expected:
            raise ValueError(
                "decision feature record is foreign to the preservation row"
            )
        validated.append(dict(record))
    by_decision: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for record in validated:
        by_decision.setdefault(
            (str(record["external_request_id"]), int(record["sequence_ordinal"])), []
        ).append(record)
    for decision_records in by_decision.values():
        declared = int(decision_records[0]["candidate_count"])
        if len(decision_records) != declared:
            raise ValueError("decision feature candidate cardinality is incomplete")
        if {int(item["candidate_ordinal"]) for item in decision_records} != set(
            range(declared)
        ):
            raise ValueError(
                "decision feature candidate ordinals are duplicated or missing"
            )
        if {int(item["rank"]) for item in decision_records} != set(range(declared)):
            raise ValueError("decision feature ranks are duplicated or missing")
        if len({item["offload_key"] for item in decision_records}) != declared:
            raise ValueError("decision feature physical identities are duplicated")
        if any(int(item["candidate_count"]) != declared for item in decision_records):
            raise ValueError("decision feature candidate counts conflict")
        if not any(item["selected"] for item in decision_records):
            raise ValueError("decision feature decision has no selected candidate")
    request_order = {
        request_id: index for index, request_id in enumerate(expected_request_ids)
    }
    return sorted(
        validated,
        key=lambda item: (
            request_order[str(item["external_request_id"])],
            int(item["sequence_ordinal"]),
            int(item["rank"]),
        ),
    )
