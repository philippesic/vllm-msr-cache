# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validation for latency and request-attribution result fields."""

import math

PERCENTILE_GROUPS = (
    ("TTFT", ("ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms")),
    ("ITL", ("itl_p50_ms", "itl_p90_ms", "itl_p99_ms")),
)
OPTIONAL_TAIL_GROUPS = (
    ("TTFT tail", ("ttft_p95_ms", "ttft_max_ms")),
    ("ITL tail", ("itl_p95_ms", "itl_max_ms")),
)
REQUEST_ATTRIBUTION_FIELDS = (
    "request_e2e_time_sum_s",
    "request_e2e_time_count",
    "request_queue_time_sum_s",
    "request_queue_time_count",
    "request_prefill_time_sum_s",
    "request_prefill_time_count",
    "request_inference_time_sum_s",
    "request_inference_time_count",
)
OPTIONAL_METRIC_FIELDS = (
    "load_bytes_delta",
    "store_bytes_delta",
    "preemptions_delta",
    *REQUEST_ATTRIBUTION_FIELDS,
)


def _empty(value) -> bool:
    return value is None or str(value).strip() == ""


def finite_nonnegative(value) -> float | None:
    """Parse a metric without turning unavailable or impossible data into zero."""
    if _empty(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def is_needle_row(row: dict) -> bool:
    return row.get("workload") in {"needle", "needle-v2"} or (
        row.get("workload") == "mixed" and row.get("sub_workload") == "needle"
    )


def validate_latency_row(
    row: dict, *, require_complete: bool = False, require_percentiles: bool = False
) -> list[str]:
    """Return latency/attribution contract violations for one result row.

    Percentiles are optional for bespoke needle rows, but if any percentile in
    a group is present the whole group must be present and ordered. A complete
    serving row must contain both percentile groups and every request timing
    sum/count field. ``require_percentiles`` applies that source-side
    requirement before request attribution is attached. Offload/preemption
    fields remain optional: absent means unavailable, while an explicit zero
    remains a valid measured value.
    """
    errors: list[str] = []
    for label, fields in PERCENTILE_GROUPS:
        values = [row.get(field) for field in fields]
        present = [not _empty(value) for value in values]
        if require_complete or require_percentiles or any(present):
            for field, value in zip(fields, values):
                if _empty(value):
                    errors.append(f"missing required percentile field {field}")
                elif finite_nonnegative(value) is None:
                    errors.append(f"invalid percentile field {field}")
            parsed = [finite_nonnegative(value) for value in values]
            if all(value is not None for value in parsed) and not (
                parsed[0] <= parsed[1] <= parsed[2]
            ):
                errors.append(
                    f"{label} percentiles are not monotonic: "
                    f"p50={parsed[0]}, p90={parsed[1]}, p99={parsed[2]}"
                )

    for label, fields in OPTIONAL_TAIL_GROUPS:
        values = [row.get(field) for field in fields]
        present = [not _empty(value) for value in values]
        if any(present):
            for field, value in zip(fields, values):
                if _empty(value):
                    errors.append(f"missing required tail field {field}")
                elif finite_nonnegative(value) is None:
                    errors.append(f"invalid tail field {field}")
            parsed = [finite_nonnegative(value) for value in values]
            p99_field = "ttft_p99_ms" if label.startswith("TTFT") else "itl_p99_ms"
            p99 = finite_nonnegative(row.get(p99_field))
            p90_field = "ttft_p90_ms" if label.startswith("TTFT") else "itl_p90_ms"
            p90 = finite_nonnegative(row.get(p90_field))
            if all(value is not None for value in parsed) and (
                not parsed[0] <= parsed[1]
                or (p99 is not None and not parsed[0] <= p99 <= parsed[1])
                or (p90 is not None and not p90 <= parsed[0])
            ):
                errors.append(
                    f"{label} values are not monotonic: "
                    f"p90={p90}, p95={parsed[0]}, p99={p99}, max={parsed[1]}"
                )

    attribution_present = [
        not _empty(row.get(field)) for field in REQUEST_ATTRIBUTION_FIELDS
    ]
    if require_complete:
        for field in REQUEST_ATTRIBUTION_FIELDS:
            if _empty(row.get(field)):
                errors.append(f"missing required attribution field {field}")
    elif any(attribution_present) and not all(attribution_present):
        errors.append("incomplete request attribution fields")

    for field in OPTIONAL_METRIC_FIELDS:
        value = row.get(field)
        if not _empty(value) and finite_nonnegative(value) is None:
            errors.append(f"invalid optional metric field {field}")
    return errors
