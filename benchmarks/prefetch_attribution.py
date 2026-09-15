"""Fail-closed request-level attribution for exact KV prefetch experiments.

This module is deliberately independent of vLLM and CUDA. It validates the
event contract emitted as an opt-in nested record by the semantic connector;
the current producer reports scheduler/worker facts and explicitly marks the
episode unavailable until a first-forward runtime seam is available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isclose, isfinite
from typing import Any

PREFETCH_ATTRIBUTION_SCHEMA = "semantic-prefetch-attribution-v1"

_STATUSES = frozenset(
    {
        "success",
        "cancelled",
        "stale",
        "failed",
        "timeout",
        "unavailable",
        "not_pressured",
        "invalid",
        "hint_consumed",
        "hint_failed",
        "hint_no_action",
    }
)
_TIMESTAMP_FIELDS = (
    "t0_preemption",
    "t1_reservation_submit",
    "t2_prefetch_start",
    "t3_prefetch_complete",
    "t3_fence_complete",
    "t4_readmission_allocation",
    "t6_first_forward",
)
_NONNEGATIVE_INT_FIELDS = (
    "prefetch_h2d_bytes",
    "blocking_h2d_bytes",
    "gpu_splice_bytes",
    "extra_preemptions",
    "allocation_failures",
    "filler_failures",
)
_EVENT_PHASES = (
    "preemption",
    "reservation_submit",
    "prefetch_start",
    "prefetch_complete",
    "fence_complete",
    "readmission_allocation",
    "blocking_load_start",
    "blocking_load_end",
    "first_forward",
)
_REQUIRED_EVENT_PHASES = frozenset(
    {
        "preemption",
        "reservation_submit",
        "prefetch_start",
        "prefetch_complete",
        "fence_complete",
        "readmission_allocation",
        "first_forward",
    }
)
_PHASE_TO_SCALAR = {
    "preemption": "t0_preemption",
    "reservation_submit": "t1_reservation_submit",
    "prefetch_start": "t2_prefetch_start",
    "prefetch_complete": "t3_prefetch_complete",
    "fence_complete": "t3_fence_complete",
    "readmission_allocation": "t4_readmission_allocation",
    "blocking_load_start": "t5_blocking_load_start",
    "blocking_load_end": "t5_blocking_load_end",
    "first_forward": "t6_first_forward",
}


class PrefetchAttributionError(ValueError):
    """Raised when an attribution episode cannot support a safe claim."""


@dataclass(frozen=True)
class PrefetchAttribution:
    run_id: str
    cell_id: str
    seed: int
    episode_id: str
    request_id: str
    status: str
    required_keys: tuple[str, ...]
    prefetched_keys: tuple[str, ...]
    intersection_keys: tuple[str, ...]
    remainder_keys: tuple[str, ...]
    reservation_id: str | None
    reservation_block_ids: tuple[str, ...]
    queue_window_s: float | None
    usable_overlap_s: float | None
    preemption_to_first_forward_s: float | None
    host_preemption_to_first_forward_s: float | None
    prefetch_time_s: float | None
    blocking_time_s: float | None
    total_transfer_time_s: float | None
    prefetch_h2d_bytes: int | None
    blocking_h2d_bytes: int | None
    total_h2d_bytes: int | None
    gpu_splice_bytes: int | None
    extra_preemptions: int | None
    allocation_failures: int | None
    filler_failures: int | None
    hidden_transfer: bool | None
    fence_before_first_forward: bool | None
    splice_ack_id: str | None
    request_first_forward_ns: int | None = None
    request_finished_ns: int | None = None
    request_finished_source: str | None = None
    staging_ready_ns: int | None = None
    destination_install_submitted_ns: int | None = None
    destination_installed_ns: int | None = None
    request_usable_ns: int | None = None
    request_usable_source: str | None = None
    transfer_hidden_before_demand_s: float | None = None
    staging_ready_lead_time_s: float | None = None
    installation_time_s: float | None = None
    exposed_blocking_stall_s: float | None = None


def _error(message: str) -> PrefetchAttributionError:
    return PrefetchAttributionError(message)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"{field} must be a non-empty string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(f"{field} must be a JSON string array")
    result = tuple(_required_string(item, f"{field}[]") for item in value)
    if len(set(result)) != len(result):
        raise _error(f"{field} contains duplicate keys")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(f"{field} must be a non-negative integer")
    return value


def _nonnegative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{field} must be a non-negative number")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise _error(f"{field} must be a non-negative finite number")
    return result


def _timestamp(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{field} must be a finite timestamp")
    result = float(value)
    if not isfinite(result):
        raise _error(f"{field} must be a finite timestamp")
    return result


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field)


def _optional_prefetch_usability(event: Mapping[str, Any]) -> dict[str, Any]:
    integer_fields = (
        "staging_ready_ns",
        "destination_install_submitted_ns",
        "destination_installed_ns",
        "request_usable_ns",
    )
    values = {
        field: _optional_nonnegative_int(event.get(field), field)
        for field in integer_fields
    }
    source = event.get("request_usable_source")
    if source is not None:
        source = _required_string(source, "request_usable_source")
    complete = event.get("request_usable_complete")
    if complete is not None and not isinstance(complete, bool):
        raise _error("request_usable_complete must be boolean")
    if complete:
        if (
            values["destination_installed_ns"] is None
            or values["request_usable_ns"] is None
            or source is None
        ):
            raise _error("complete request usability requires timestamp and source")
        if values["destination_installed_ns"] != values["request_usable_ns"]:
            raise _error("destination installation must equal request usability")
    raw_demand = event.get("demand_boundary_ns")
    raw_demand_source = event.get("demand_boundary_source")
    raw_demand_domain = event.get("demand_boundary_clock_domain")
    if raw_demand is None:
        if raw_demand_source is not None or raw_demand_domain is not None:
            raise _error("demand boundary provenance requires a timestamp")
    else:
        _nonnegative_int(raw_demand, "demand_boundary_ns")
        if raw_demand_source not in {
            "scheduler_update_state_after_alloc_entry",
            "readmission_allocation",
            "hinted_request_connector_observation",
        }:
            raise _error("unsupported demand boundary source")
        if raw_demand_domain != "host_monotonic":
            raise _error("demand boundary must use host_monotonic")
        event_t4 = event.get("t4_readmission_allocation")
        if (
            isinstance(event_t4, int)
            and not isinstance(event_t4, bool)
            and raw_demand != event_t4
        ):
            raise _error("demand boundary disagrees with t4 alias")
    ordered = [
        value
        for value in (
            values["staging_ready_ns"],
            values["destination_install_submitted_ns"],
            values["destination_installed_ns"],
        )
        if value is not None
    ]
    if any(left > right for left, right in pairwise(ordered)):
        raise _error("staging/install/usability timestamps are out of order")
    for field in (
        "transfer_hidden_before_demand_s",
        "staging_ready_lead_time_s",
        "installation_time_s",
        "exposed_blocking_stall_s",
    ):
        raw = event.get(field)
        values[field] = (
            _nonnegative_float(raw, field) if raw is not None else None
        )
    t2 = event.get("t2_prefetch_start")
    demand = event.get("t4_readmission_allocation")
    staging_ready = values["staging_ready_ns"]
    install_submit = values["destination_install_submitted_ns"]
    usable = values["request_usable_ns"]
    if all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (t2, demand, staging_ready)
    ):
        expected_hidden = max(0, min(demand, staging_ready) - t2) / 1_000_000_000
        expected_lead = max(0, demand - staging_ready) / 1_000_000_000
        if values["transfer_hidden_before_demand_s"] is not None and not _same_float(
            values["transfer_hidden_before_demand_s"], expected_hidden
        ):
            raise _error("transfer_hidden_before_demand_s is inconsistent")
        if values["staging_ready_lead_time_s"] is not None and not _same_float(
            values["staging_ready_lead_time_s"], expected_lead
        ):
            raise _error("staging_ready_lead_time_s is inconsistent")
    if install_submit is not None and usable is not None:
        expected_install = (usable - install_submit) / 1_000_000_000
        if values["installation_time_s"] is not None and not _same_float(
            values["installation_time_s"], expected_install
        ):
            raise _error("installation_time_s is inconsistent")
    if isinstance(demand, int) and not isinstance(demand, bool) and usable is not None:
        expected_stall = max(0, usable - demand) / 1_000_000_000
        if values["exposed_blocking_stall_s"] is not None and not _same_float(
            values["exposed_blocking_stall_s"], expected_stall
        ):
            raise _error("exposed_blocking_stall_s is inconsistent")
    values["request_usable_source"] = source
    return values


def _same_float(actual: float, expected: float) -> bool:
    return isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def _mapping_items(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be a JSON object")
    result = {
        _required_string(key, f"{field} key"): _required_string(
            item, f"{field}[{key!r}]"
        )
        for key, item in value.items()
    }
    if len(set(result.values())) != len(result):
        raise _error(f"{field} contains duplicate physical block IDs")
    return result


def _validate_block_pool_snapshots(event: Mapping[str, Any]) -> None:
    """Validate optional scheduler-owned BlockPool accounting snapshots."""
    raw_snapshots = event.get("block_pool_snapshots")
    if raw_snapshots is None:
        return
    if not isinstance(raw_snapshots, Sequence) or isinstance(
        raw_snapshots, (str, bytes)
    ):
        raise _error("block_pool_snapshots must be a JSON object array")
    declared_count = event.get("block_pool_snapshot_count")
    if declared_count is not None:
        declared_count = _nonnegative_int(declared_count, "block_pool_snapshot_count")
        if declared_count < len(raw_snapshots):
            raise _error("block_pool_snapshot_count is smaller than snapshots")
    truncated = event.get("block_pool_snapshots_truncated", False)
    if not isinstance(truncated, bool):
        raise _error("block_pool_snapshots_truncated must be boolean")
    if (
        truncated
        and declared_count is not None
        and declared_count <= len(raw_snapshots)
    ):
        raise _error("truncated snapshots must declare omitted observations")
    for index, snapshot in enumerate(raw_snapshots):
        if not isinstance(snapshot, Mapping):
            raise _error(f"block_pool_snapshots[{index}] must be an object")
        if snapshot.get("schema") != "semantic-block-pool-snapshot-v1":
            raise _error(f"block_pool_snapshots[{index}] has an unsupported schema")
        _required_string(snapshot.get("phase"), f"block_pool_snapshots[{index}].phase")
        _required_string(
            snapshot.get("source"), f"block_pool_snapshots[{index}].source"
        )
        _timestamp(
            snapshot.get("observed_at_ns"),
            f"block_pool_snapshots[{index}].observed_at_ns",
        )
        _nonnegative_int(
            snapshot.get("episode_id"),
            f"block_pool_snapshots[{index}].episode_id",
        )
        _nonnegative_int(
            snapshot.get("preemption_index"),
            f"block_pool_snapshots[{index}].preemption_index",
        )
        if snapshot.get("job_id") is not None:
            _nonnegative_int(
                snapshot["job_id"], f"block_pool_snapshots[{index}].job_id"
            )
        available = snapshot.get("available")
        if not isinstance(available, bool):
            raise _error(f"block_pool_snapshots[{index}].available must be boolean")
        if not available:
            _required_string(
                snapshot.get("reason"), f"block_pool_snapshots[{index}].reason"
            )
            continue
        prefix = f"block_pool_snapshots[{index}]"
        total = _nonnegative_int(snapshot.get("total_blocks"), f"{prefix}.total_blocks")
        null_count = _nonnegative_int(
            snapshot.get("null_block_count"), f"{prefix}.null_block_count"
        )
        capacity = _nonnegative_int(
            snapshot.get("schedulable_capacity_blocks"),
            f"{prefix}.schedulable_capacity_blocks",
        )
        free = _nonnegative_int(
            snapshot.get("free_block_queue_blocks"),
            f"{prefix}.free_block_queue_blocks",
        )
        reserved = _nonnegative_int(
            snapshot.get("scheduler_prefetch_reserved_blocks_total"),
            f"{prefix}.scheduler_prefetch_reserved_blocks_total",
        )
        referenced = _nonnegative_int(
            snapshot.get("estimated_referenced_blocks"),
            f"{prefix}.estimated_referenced_blocks",
        )
        underflow = snapshot.get("accounting_underflow_blocks")
        if (
            isinstance(underflow, bool)
            or not isinstance(underflow, int)
            or underflow > 0
        ):
            raise _error(f"{prefix}.accounting_underflow_blocks must be non-positive")
        consistent = snapshot.get("accounting_consistent")
        if not isinstance(consistent, bool):
            raise _error(f"{prefix}.accounting_consistent must be boolean")
        configured = _nonnegative_int(
            snapshot.get("configured_prefetch_headroom_blocks"),
            f"{prefix}.configured_prefetch_headroom_blocks",
        )
        headroom = _nonnegative_int(
            snapshot.get("scheduler_free_headroom_blocks"),
            f"{prefix}.scheduler_free_headroom_blocks",
        )
        if null_count != 1 or capacity != max(0, total - null_count):
            raise _error(f"{prefix} has inconsistent pool capacity")
        raw_referenced = capacity - free - reserved
        if referenced != max(0, raw_referenced):
            raise _error(f"{prefix}.estimated_referenced_blocks is inconsistent")
        if underflow != min(0, raw_referenced):
            raise _error(f"{prefix}.accounting_underflow_blocks is inconsistent")
        if consistent != (raw_referenced >= 0):
            raise _error(f"{prefix}.accounting_consistent is inconsistent")
        if headroom != max(0, free - configured):
            raise _error(f"{prefix}.scheduler_free_headroom_blocks is inconsistent")
        if "physical_block_bytes" in snapshot:
            block_bytes = _nonnegative_int(
                snapshot.get("physical_block_bytes"), f"{prefix}.physical_block_bytes"
            )
            staging_ids = snapshot.get("scheduler_prefetch_reserved_block_ids")
            usable_records = snapshot.get("gpu_usable_context_blocks")
            usable_ids = snapshot.get("gpu_usable_context_block_ids")
            if not isinstance(staging_ids, list) or not isinstance(usable_ids, list):
                raise _error(f"{prefix} identity fields must be arrays")
            if not isinstance(usable_records, list):
                raise _error(f"{prefix}.gpu_usable_context_blocks must be an array")
            if any(
                _nonnegative_int(value, f"{prefix}.block_id") != value
                for value in (*staging_ids, *usable_ids)
            ):
                raise _error(f"{prefix} block IDs must be non-negative integers")
            if len(set(staging_ids)) != len(staging_ids) or len(set(usable_ids)) != len(
                usable_ids
            ):
                raise _error(f"{prefix} block IDs must be unique")
            if set(staging_ids) & set(usable_ids):
                raise _error(f"{prefix} staging and usable blocks overlap")
            identity_complete = snapshot.get("staging_identity_complete")
            if not isinstance(identity_complete, bool):
                raise _error(f"{prefix}.staging_identity_complete must be boolean")
            if identity_complete != (len(staging_ids) == reserved):
                raise _error(f"{prefix} staging identity completeness is inconsistent")
            record_ids = []
            for record in usable_records:
                if not isinstance(record, Mapping):
                    raise _error(f"{prefix} usable block record must be an object")
                record_ids.append(
                    _nonnegative_int(record.get("block_id"), f"{prefix}.block_id")
                )
                _nonnegative_int(record.get("ref_cnt"), f"{prefix}.ref_cnt")
                if not isinstance(record.get("reusable_prefix"), bool):
                    raise _error(f"{prefix}.reusable_prefix must be boolean")
            if record_ids != usable_ids:
                raise _error(f"{prefix} usable block identity arrays disagree")
            if _nonnegative_int(
                snapshot.get("staging_reserved_bytes"),
                f"{prefix}.staging_reserved_bytes",
            ) != len(staging_ids) * block_bytes:
                raise _error(f"{prefix} staging bytes are inconsistent")
            if _nonnegative_int(
                snapshot.get("gpu_usable_context_bytes"),
                f"{prefix}.gpu_usable_context_bytes",
            ) != len(usable_ids) * block_bytes:
                raise _error(f"{prefix} usable bytes are inconsistent")
            if _nonnegative_int(
                snapshot.get("gpu_schedulable_capacity_bytes"),
                f"{prefix}.gpu_schedulable_capacity_bytes",
            ) != capacity * block_bytes:
                raise _error(f"{prefix} GPU capacity bytes are inconsistent")


def _validate_cpu_cache_snapshots(event: Mapping[str, Any]) -> None:
    """Validate optional scheduler-owned CPU-tier occupancy snapshots."""
    raw_snapshots = event.get("cpu_cache_snapshots")
    if raw_snapshots is None:
        return
    if not isinstance(raw_snapshots, Sequence) or isinstance(
        raw_snapshots, (str, bytes)
    ):
        raise _error("cpu_cache_snapshots must be a JSON object array")
    declared_count = event.get("cpu_cache_snapshot_count")
    if declared_count is not None:
        declared_count = _nonnegative_int(declared_count, "cpu_cache_snapshot_count")
        if declared_count < len(raw_snapshots):
            raise _error("cpu_cache_snapshot_count is smaller than snapshots")
    truncated = event.get("cpu_cache_snapshots_truncated", False)
    if not isinstance(truncated, bool):
        raise _error("cpu_cache_snapshots_truncated must be boolean")
    if (
        truncated
        and declared_count is not None
        and declared_count <= len(raw_snapshots)
    ):
        raise _error("truncated CPU snapshots must declare omitted observations")
    for index, snapshot in enumerate(raw_snapshots):
        prefix = f"cpu_cache_snapshots[{index}]"
        if not isinstance(snapshot, Mapping):
            raise _error(f"{prefix} must be an object")
        if snapshot.get("schema") != "semantic-cpu-cache-snapshot-v1":
            raise _error(f"{prefix} has an unsupported schema")
        _required_string(snapshot.get("phase"), f"{prefix}.phase")
        _required_string(snapshot.get("source"), f"{prefix}.source")
        _timestamp(snapshot.get("observed_at_ns"), f"{prefix}.observed_at_ns")
        _nonnegative_int(snapshot.get("episode_id"), f"{prefix}.episode_id")
        _nonnegative_int(snapshot.get("preemption_index"), f"{prefix}.preemption_index")
        available = snapshot.get("available")
        if not isinstance(available, bool):
            raise _error(f"{prefix}.available must be boolean")
        if not available:
            _required_string(snapshot.get("reason"), f"{prefix}.reason")
            continue
        total = _nonnegative_int(snapshot.get("total_blocks"), f"{prefix}.total_blocks")
        allocated = _nonnegative_int(
            snapshot.get("allocated_blocks"), f"{prefix}.allocated_blocks"
        )
        free_list = _nonnegative_int(
            snapshot.get("free_list_blocks"), f"{prefix}.free_list_blocks"
        )
        free_capacity = _nonnegative_int(
            snapshot.get("free_capacity_blocks"), f"{prefix}.free_capacity_blocks"
        )
        evictable = _nonnegative_int(
            snapshot.get("evictable_blocks"), f"{prefix}.evictable_blocks"
        )
        write_pending = _nonnegative_int(
            snapshot.get("write_pending_blocks"), f"{prefix}.write_pending_blocks"
        )
        referenced = _nonnegative_int(
            snapshot.get("estimated_non_evictable_blocks"),
            f"{prefix}.estimated_non_evictable_blocks",
        )
        underflow = snapshot.get("accounting_underflow_blocks")
        if (
            isinstance(underflow, bool)
            or not isinstance(underflow, int)
            or underflow > 0
        ):
            raise _error(f"{prefix}.accounting_underflow_blocks must be non-positive")
        consistent = snapshot.get("accounting_consistent")
        if not isinstance(consistent, bool):
            raise _error(f"{prefix}.accounting_consistent must be boolean")
        resident_entries = _nonnegative_int(
            snapshot.get("policy_resident_entries"),
            f"{prefix}.policy_resident_entries",
        )
        _nonnegative_int(
            snapshot.get("policy_ghost_entries"), f"{prefix}.policy_ghost_entries"
        )
        _required_string(snapshot.get("policy_class"), f"{prefix}.policy_class")
        if allocated > total or free_list > allocated:
            raise _error(f"{prefix} has inconsistent allocation bounds")
        if free_capacity != total - allocated + free_list:
            raise _error(f"{prefix}.free_capacity_blocks is inconsistent")
        raw_referenced = allocated - free_list - evictable
        if referenced != max(0, raw_referenced):
            raise _error(f"{prefix}.estimated_non_evictable_blocks is inconsistent")
        if underflow != min(0, raw_referenced):
            raise _error(f"{prefix}.accounting_underflow_blocks is inconsistent")
        if consistent != (raw_referenced >= 0):
            raise _error(f"{prefix}.accounting_consistent is inconsistent")
        if write_pending > allocated or evictable > allocated:
            raise _error(f"{prefix} has inconsistent ownership counters")
        if "physical_block_bytes" in snapshot:
            block_bytes = _nonnegative_int(
                snapshot.get("physical_block_bytes"), f"{prefix}.physical_block_bytes"
            )
            records = snapshot.get("resident_blocks")
            if not isinstance(records, list):
                raise _error(f"{prefix}.resident_blocks must be an array")
            keys = []
            ready_count = 0
            for record in records:
                if not isinstance(record, Mapping):
                    raise _error(f"{prefix} resident block record must be an object")
                keys.append(_required_string(record.get("key"), f"{prefix}.key"))
                _nonnegative_int(record.get("cpu_block_id"), f"{prefix}.cpu_block_id")
                if not isinstance(record.get("ready"), bool):
                    raise _error(f"{prefix}.ready must be boolean")
                ready_count += int(record["ready"])
                ref_cnt = record.get("ref_cnt")
                if (
                    isinstance(ref_cnt, bool)
                    or not isinstance(ref_cnt, int)
                    or ref_cnt < -1
                ):
                    raise _error(f"{prefix}.ref_cnt is invalid")
            if len(keys) != resident_entries or len(set(keys)) != len(keys):
                raise _error(f"{prefix} resident identities are inconsistent")
            expected_bytes = resident_entries * block_bytes
            if _nonnegative_int(
                snapshot.get("resident_bytes"), f"{prefix}.resident_bytes"
            ) != expected_bytes:
                raise _error(f"{prefix} resident bytes are inconsistent")
            if _nonnegative_int(
                snapshot.get("ready_resident_bytes"),
                f"{prefix}.ready_resident_bytes",
            ) != ready_count * block_bytes:
                raise _error(f"{prefix} ready resident bytes are inconsistent")
            if _nonnegative_int(
                snapshot.get("write_pending_bytes"),
                f"{prefix}.write_pending_bytes",
            ) != write_pending * block_bytes:
                raise _error(f"{prefix} pending bytes are inconsistent")
            cpu_capacity_bytes = total * block_bytes
            ready_bytes = ready_count * block_bytes
            if _nonnegative_int(
                snapshot.get("cpu_capacity_bytes"), f"{prefix}.cpu_capacity_bytes"
            ) != cpu_capacity_bytes:
                raise _error(f"{prefix} CPU capacity bytes are inconsistent")
            if _nonnegative_int(
                snapshot.get("actual_offloaded_fraction_numerator_bytes"),
                f"{prefix}.actual_offloaded_fraction_numerator_bytes",
            ) != ready_bytes or _nonnegative_int(
                snapshot.get("actual_offloaded_fraction_denominator_bytes"),
                f"{prefix}.actual_offloaded_fraction_denominator_bytes",
            ) != cpu_capacity_bytes:
                raise _error(f"{prefix} offloaded-fraction operands are inconsistent")
            fraction = snapshot.get("actual_offloaded_fraction")
            if total:
                if fraction is None or not _same_float(
                    _nonnegative_float(fraction, f"{prefix}.actual_offloaded_fraction"),
                    ready_count / total,
                ):
                    raise _error(f"{prefix} actual offloaded fraction is inconsistent")
            elif fraction is not None:
                raise _error(f"{prefix} empty CPU capacity must have unavailable fraction")


def _validate_timestamp_clock_domains(event: Mapping[str, Any], status: str) -> None:
    """Ensure timestamp provenance cannot be mistaken for one shared clock."""
    raw_domains = event.get("timestamp_clock_domains")
    if raw_domains is None:
        return
    if not isinstance(raw_domains, Mapping):
        raise _error("timestamp_clock_domains must be a JSON object")
    present_fields = [
        field
        for field in (
            *_TIMESTAMP_FIELDS,
            "t5_blocking_load_start",
            "t5_blocking_load_end",
            "destination_install_submitted",
            "destination_installed",
            "request_usable",
        )
        if event.get(field) is not None
    ]
    domains: set[str] = set()
    for field in present_fields:
        domain = _required_string(
            raw_domains.get(field), f"timestamp_clock_domains.{field}"
        )
        domains.add(domain)
    if status == "success" and len(domains) > 1:
        raise _error("success attribution cannot mix timestamp clock domains")


def _validate_speculative_transfer_accounting(event: Mapping[str, Any]) -> None:
    """Validate optional request-level speculative lifecycle totals."""
    accounting = event.get("speculative_transfer")
    if accounting is None:
        return
    if not isinstance(accounting, Mapping):
        raise _error("speculative_transfer must be a JSON object")
    legacy_integer_fields = (
        "job_count",
        "successful_job_count",
        "failed_job_count",
        "submitted_bytes",
        "completed_bytes",
        "consumed_bytes",
        "discarded_bytes",
        "released_reservation_bytes",
        "uncompleted_submitted_bytes",
        "unconsumed_completed_bytes",
        "actual_contributors",
        "expected_contributors",
    )
    values = {
        field: _nonnegative_int(accounting.get(field), f"speculative_transfer.{field}")
        for field in legacy_integer_fields
    }
    if values["job_count"] != (
        values["successful_job_count"] + values["failed_job_count"]
    ):
        raise _error("speculative transfer job outcomes do not sum to job_count")
    if values["consumed_bytes"] + values["discarded_bytes"] > values["completed_bytes"]:
        raise _error("consumed and discarded bytes exceed completed bytes")
    expected_unconsumed = (
        values["completed_bytes"]
        - values["consumed_bytes"]
        - values["discarded_bytes"]
    )
    if values["unconsumed_completed_bytes"] != expected_unconsumed:
        raise _error("unconsumed speculative bytes are inconsistent")
    if values["uncompleted_submitted_bytes"] != max(
        0, values["submitted_bytes"] - values["completed_bytes"]
    ):
        raise _error("uncompleted speculative bytes are inconsistent")
    lifecycle_integer_fields = (
        "cancelled_job_count",
        "attempted_blocks",
        "attempted_bytes",
        "submitted_jobs",
        "submitted_blocks",
        "consumption_events",
    )
    lifecycle_presence = [field in accounting for field in lifecycle_integer_fields]
    if not any(lifecycle_presence):
        # Retained v8 and earlier v1 events predate the separated reservation,
        # submission, and lateness fields. Their physical totals remain valid.
        return
    if not all(lifecycle_presence):
        raise _error("speculative lifecycle counters must be jointly available")
    lifecycle = {
        field: _nonnegative_int(accounting.get(field), f"speculative_transfer.{field}")
        for field in lifecycle_integer_fields
    }
    values.update(lifecycle)
    cost_fields = (
        "duplicated_job_count",
        "duplicated_bytes",
        "waste_completed_bytes",
    )
    cost_presence = [field in accounting for field in cost_fields]
    if any(cost_presence) and not all(cost_presence):
        raise _error("speculative duplicate/waste counters must be jointly available")
    if all(cost_presence):
        costs = {
            field: _nonnegative_int(
                accounting.get(field), f"speculative_transfer.{field}"
            )
            for field in cost_fields
        }
        if costs["duplicated_job_count"] > values["successful_job_count"]:
            raise _error("duplicated speculative jobs exceed successful jobs")
        if costs["duplicated_bytes"] > values["completed_bytes"]:
            raise _error("duplicated speculative bytes exceed completed bytes")
        if costs["waste_completed_bytes"] != (
            values["completed_bytes"] - values["consumed_bytes"]
        ):
            raise _error("speculative completed waste bytes are inconsistent")
    if values["cancelled_job_count"] > values["failed_job_count"]:
        raise _error("cancelled speculative jobs cannot exceed failed jobs")
    if values["successful_job_count"] > values["submitted_jobs"]:
        raise _error("successful speculative jobs cannot exceed submitted jobs")
    if values["submitted_blocks"] > values["attempted_blocks"]:
        raise _error("submitted speculative blocks cannot exceed attempted blocks")
    if values["consumption_events"] not in (0, 1):
        raise _error("speculative consumption_events must be zero or one")
    if values["consumption_events"] != int(values["consumed_bytes"] > 0):
        raise _error("speculative consumption event disagrees with consumed bytes")
    late_jobs = accounting.get("late_job_count")
    late_bytes = accounting.get("late_bytes")
    on_time_jobs = accounting.get("on_time_job_count")
    on_time_bytes = accounting.get("on_time_bytes")
    boundary_ns = accounting.get("demand_boundary_ns")
    boundary_source = accounting.get("demand_boundary_source")
    timing_fields = (late_jobs, late_bytes, on_time_jobs, on_time_bytes)
    if all(value is None for value in timing_fields):
        if boundary_ns is not None or boundary_source is not None:
            raise _error("speculative demand boundary lacks timing classification")
    elif any(value is None for value in timing_fields):
        raise _error("speculative on-time/late fields must be jointly available")
    else:
        late_jobs = _nonnegative_int(late_jobs, "speculative_transfer.late_job_count")
        late_bytes = _nonnegative_int(late_bytes, "speculative_transfer.late_bytes")
        on_time_jobs = _nonnegative_int(
            on_time_jobs, "speculative_transfer.on_time_job_count"
        )
        on_time_bytes = _nonnegative_int(
            on_time_bytes, "speculative_transfer.on_time_bytes"
        )
        _nonnegative_int(boundary_ns, "speculative_transfer.demand_boundary_ns")
        _required_string(
            boundary_source, "speculative_transfer.demand_boundary_source"
        )
        if late_jobs + on_time_jobs != values["successful_job_count"]:
            raise _error("speculative on-time/late jobs do not sum to successes")
        if late_bytes + on_time_bytes != values["completed_bytes"]:
            raise _error("speculative on-time/late bytes do not sum to completed bytes")

    device_time = accounting.get("device_time_s")
    if device_time is not None:
        _nonnegative_float(device_time, "speculative_transfer.device_time_s")


def _validate_speculative_transfer_costs(event: Mapping[str, Any]) -> None:
    costs = event.get("speculative_transfer_costs")
    if costs is None:
        return
    accounting = event.get("speculative_transfer")
    if not isinstance(costs, Mapping) or not isinstance(accounting, Mapping):
        raise _error("speculative transfer costs require transfer accounting")
    values = {
        field: _nonnegative_int(
            costs.get(field), f"speculative_transfer_costs.{field}"
        )
        for field in (
            "duplicated_job_count",
            "duplicated_bytes",
            "waste_completed_bytes",
        )
    }
    successful_jobs = _nonnegative_int(
        accounting.get("successful_job_count"),
        "speculative_transfer.successful_job_count",
    )
    completed_bytes = _nonnegative_int(
        accounting.get("completed_bytes"), "speculative_transfer.completed_bytes"
    )
    consumed_bytes = _nonnegative_int(
        accounting.get("consumed_bytes"), "speculative_transfer.consumed_bytes"
    )
    if values["duplicated_job_count"] > successful_jobs:
        raise _error("duplicated speculative jobs exceed successful jobs")
    if values["duplicated_bytes"] > completed_bytes:
        raise _error("duplicated speculative bytes exceed completed bytes")
    if values["waste_completed_bytes"] != completed_bytes - consumed_bytes:
        raise _error("speculative completed waste bytes are inconsistent")


def _host_preemption_to_first_forward_s(event: Mapping[str, Any]) -> float | None:
    """Return the producer-derived host execution-entry interval."""
    explicit = event.get("preemption_to_model_runner_entry_s")
    if explicit is not None:
        return _nonnegative_float(explicit, "preemption_to_model_runner_entry_s")
    # Legacy scalar fixtures predate the explicit producer-derived field and
    # express timestamps in the validator's native seconds unit.
    t0 = event.get("t0_preemption")
    t6 = event.get("t6_first_forward")
    if t0 is None or t6 is None:
        return None
    t0_value = _timestamp(t0, "t0_preemption")
    t6_value = _timestamp(t6, "t6_first_forward")
    domains = event.get("timestamp_clock_domains")
    if isinstance(domains, Mapping) and domains.get("t0_preemption") != domains.get(
        "t6_first_forward"
    ):
        return None
    if t6_value < t0_value:
        return None
    return t6_value - t0_value


def validate_prefetch_episode(event: Mapping[str, Any]) -> PrefetchAttribution:
    """Validate and derive metrics from one exact-prefetch episode."""
    if not isinstance(event, Mapping):
        raise _error("prefetch episode must be a JSON object")

    run_id = _required_string(event.get("run_id"), "run_id")
    cell_id = _required_string(event.get("cell_id"), "cell_id")
    episode_id = _required_string(event.get("episode_id"), "episode_id")
    request_id = _required_string(event.get("request_id"), "request_id")
    seed = event.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise _error("seed must be an integer")

    status = _required_string(event.get("status"), "status")
    if status not in _STATUSES:
        raise _error(f"unsupported status={status!r}")
    _validate_block_pool_snapshots(event)
    _validate_cpu_cache_snapshots(event)
    _validate_timestamp_clock_domains(event, status)
    _validate_speculative_transfer_accounting(event)
    _validate_speculative_transfer_costs(event)
    usability = _optional_prefetch_usability(event)
    request_first_forward_ns = _optional_nonnegative_int(
        event.get("request_first_forward_ns"), "request_first_forward_ns"
    )
    request_finished_ns = _optional_nonnegative_int(
        event.get("request_finished_ns"), "request_finished_ns"
    )
    if (
        request_first_forward_ns is not None
        and request_finished_ns is not None
        and request_finished_ns < request_first_forward_ns
    ):
        raise _error("request_finished_ns cannot precede request_first_forward_ns")
    request_finished_source = event.get("request_finished_source")
    if request_finished_source is not None:
        request_finished_source = _required_string(
            request_finished_source, "request_finished_source"
        )
    if (request_finished_ns is None) != (request_finished_source is None):
        raise _error(
            "request_finished_ns and request_finished_source must be jointly available"
        )
    host_preemption_to_first_forward_s = _host_preemption_to_first_forward_s(event)
    if status != "success":
        _required_string(event.get("status_reason"), "status_reason")
        return PrefetchAttribution(
            run_id=run_id,
            cell_id=cell_id,
            seed=seed,
            episode_id=episode_id,
            request_id=request_id,
            status=status,
            required_keys=(),
            prefetched_keys=(),
            intersection_keys=(),
            remainder_keys=(),
            reservation_id=None,
            reservation_block_ids=(),
            queue_window_s=None,
            usable_overlap_s=None,
            preemption_to_first_forward_s=None,
            host_preemption_to_first_forward_s=host_preemption_to_first_forward_s,
            prefetch_time_s=None,
            blocking_time_s=None,
            total_transfer_time_s=None,
            prefetch_h2d_bytes=None,
            blocking_h2d_bytes=None,
            total_h2d_bytes=None,
            gpu_splice_bytes=None,
            extra_preemptions=None,
            allocation_failures=None,
            filler_failures=None,
            hidden_transfer=None,
            fence_before_first_forward=None,
            splice_ack_id=None,
            staging_ready_ns=usability["staging_ready_ns"],
            destination_install_submitted_ns=usability[
                "destination_install_submitted_ns"
            ],
            destination_installed_ns=usability["destination_installed_ns"],
            request_usable_ns=usability["request_usable_ns"],
            request_usable_source=usability["request_usable_source"],
            transfer_hidden_before_demand_s=usability[
                "transfer_hidden_before_demand_s"
            ],
            staging_ready_lead_time_s=usability["staging_ready_lead_time_s"],
            installation_time_s=usability["installation_time_s"],
            exposed_blocking_stall_s=usability["exposed_blocking_stall_s"],
        )

    required_keys = _string_tuple(event.get("required_keys"), "required_keys")
    if not required_keys:
        raise _error("required_keys cannot be empty")
    prefetched_keys = _string_tuple(event.get("prefetched_keys"), "prefetched_keys")
    intersection_keys = _string_tuple(
        event.get("intersection_keys"), "intersection_keys"
    )
    resident_keys = set(
        _string_tuple(event.get("cpu_resident_keys"), "cpu_resident_keys")
    )
    ready_keys = set(_string_tuple(event.get("cpu_ready_keys"), "cpu_ready_keys"))
    residency_proof = _mapping_items(
        event.get("cpu_residency_proof"), "cpu_residency_proof"
    )
    required_set = set(required_keys)
    prefetched_set = set(prefetched_keys)
    if not prefetched_set <= required_set:
        raise _error("prefetched_keys must be a subset of required_keys")
    if not ready_keys <= resident_keys:
        raise _error("CPU-ready keys must be CPU-resident")
    if set(residency_proof) != prefetched_set:
        raise _error("residency proof keys must equal prefetched_keys")
    if not prefetched_set <= ready_keys:
        raise _error("every prefetched key needs CPU-resident-ready proof")
    if set(intersection_keys) != required_set & prefetched_set:
        raise _error("intersection_keys must equal required/prefetched intersection")
    if intersection_keys != tuple(
        key for key in prefetched_keys if key in required_set
    ):
        raise _error("intersection_keys must preserve prefetched key order")

    remainder_keys = _string_tuple(event.get("remainder_keys"), "remainder_keys")
    if set(remainder_keys) != required_set - prefetched_set:
        raise _error("remainder_keys must be the exact required-key complement")

    source_map = _mapping_items(
        event.get("source_key_to_gpu_block"), "source_key_to_gpu_block"
    )
    destination_map = _mapping_items(
        event.get("destination_key_to_gpu_block"), "destination_key_to_gpu_block"
    )
    if set(source_map) != prefetched_set:
        raise _error("source mapping keys must equal prefetched_keys")
    if set(destination_map) != prefetched_set:
        raise _error("destination mapping keys must equal prefetched_keys")
    if set(source_map.values()) & set(destination_map.values()):
        raise _error("source and destination physical blocks must be distinct")

    reservation_id = _required_string(event.get("reservation_id"), "reservation_id")
    reservation_block_ids = _string_tuple(
        event.get("reservation_block_ids"), "reservation_block_ids"
    )
    if not reservation_block_ids:
        raise _error("reservation_block_ids cannot be empty")
    reservation_history = event.get("reservation_history")
    if (
        not isinstance(reservation_history, Sequence)
        or isinstance(reservation_history, (str, bytes))
        or not reservation_history
    ):
        raise _error("reservation_history must be a non-empty JSON object array")
    for history in reservation_history:
        if not isinstance(history, Mapping):
            raise _error("reservation_history entries must be JSON objects")
        _required_string(history.get("phase"), "reservation_history.phase")
        _nonnegative_int(
            history.get("reserved_blocks"), "reservation_history.reserved_blocks"
        )
        _nonnegative_int(history.get("free_blocks"), "reservation_history.free_blocks")

    timestamps = {
        field: _timestamp(event.get(field), field) for field in _TIMESTAMP_FIELDS
    }
    ordered = (
        "t0_preemption",
        "t1_reservation_submit",
        "t2_prefetch_start",
        "t3_prefetch_complete",
        "t3_fence_complete",
        "t4_readmission_allocation",
        "t6_first_forward",
    )
    if any(timestamps[left] > timestamps[right] for left, right in pairwise(ordered)):
        raise _error("prefetch lifecycle timestamps are out of order")

    blocking_start = event.get("t5_blocking_load_start")
    blocking_end = event.get("t5_blocking_load_end")
    if (blocking_start is None) != (blocking_end is None):
        raise _error("blocking load start/end must be both present or both absent")
    if blocking_start is not None:
        blocking_start = _timestamp(blocking_start, "t5_blocking_load_start")
        blocking_end = _timestamp(blocking_end, "t5_blocking_load_end")
        if blocking_start < timestamps["t4_readmission_allocation"]:
            raise _error("blocking load cannot precede readmission")
        if blocking_start > blocking_end:
            raise _error("blocking load end cannot precede its start")
        if blocking_end > timestamps["t6_first_forward"]:
            raise _error("first forward cannot precede blocking load completion")
    elif remainder_keys:
        raise _error("remainder_keys require blocking load timestamps")
    elif event.get("t5_blocking_load_start") is not None:
        raise _error("empty remainder cannot have a blocking load")

    integers = {
        field: _nonnegative_int(event.get(field), field)
        for field in _NONNEGATIVE_INT_FIELDS
    }
    expected_total = integers["prefetch_h2d_bytes"] + integers["blocking_h2d_bytes"]
    if (
        _nonnegative_int(event.get("total_h2d_bytes"), "total_h2d_bytes")
        != expected_total
    ):
        raise _error("total_h2d_bytes must equal prefetch plus blocking H2D bytes")
    prefetch_time_s = _nonnegative_float(
        event.get("prefetch_time_s"), "prefetch_time_s"
    )
    blocking_time_s = _nonnegative_float(
        event.get("blocking_time_s"), "blocking_time_s"
    )
    total_transfer_time_s = _nonnegative_float(
        event.get("total_transfer_time_s"), "total_transfer_time_s"
    )
    if not _same_float(
        prefetch_time_s,
        timestamps["t3_prefetch_complete"] - timestamps["t2_prefetch_start"],
    ):
        raise _error("prefetch_time_s must match prefetch timestamps")
    if blocking_start is None:
        if blocking_time_s != 0:
            raise _error("blocking_time_s must be zero without a blocking load")
        expected_total_time = prefetch_time_s
    else:
        expected_blocking_time = blocking_end - blocking_start
        if not _same_float(blocking_time_s, expected_blocking_time):
            raise _error("blocking_time_s must match blocking timestamps")
        expected_total_time = (
            max(timestamps["t3_prefetch_complete"], blocking_end)
            - timestamps["t2_prefetch_start"]
        )
    if not _same_float(total_transfer_time_s, expected_total_time):
        raise _error("total_transfer_time_s must match transfer span")

    cleanup = event.get("terminal_cleanup")
    if not isinstance(cleanup, Mapping):
        raise _error("terminal_cleanup must be a JSON object")
    for field in (
        "reservation_released",
        "source_reuse_fenced",
        "request_state_cleared",
    ):
        if cleanup.get(field) is not True:
            raise _error(f"terminal_cleanup.{field} must be true")
    if event.get("source_reuse_before_fence") is not False:
        raise _error("source_reuse_before_fence must be false")
    if event.get("fence_before_first_forward") is not True:
        raise _error("fence_before_first_forward must be true")
    if event.get("splice_acknowledged") is not True:
        raise _error("splice_acknowledged must be true")
    splice_ack_id = _required_string(event.get("splice_ack_id"), "splice_ack_id")

    t0 = timestamps["t0_preemption"]
    t1 = timestamps["t1_reservation_submit"]
    t3_fence = timestamps["t3_fence_complete"]
    t4 = timestamps["t4_readmission_allocation"]
    t6 = timestamps["t6_first_forward"]
    return PrefetchAttribution(
        run_id=run_id,
        cell_id=cell_id,
        seed=seed,
        request_id=request_id,
        episode_id=episode_id,
        status=status,
        required_keys=required_keys,
        prefetched_keys=prefetched_keys,
        intersection_keys=intersection_keys,
        remainder_keys=remainder_keys,
        reservation_id=reservation_id,
        reservation_block_ids=reservation_block_ids,
        queue_window_s=t4 - t0,
        usable_overlap_s=t4 - t1,
        preemption_to_first_forward_s=(
            host_preemption_to_first_forward_s
            if host_preemption_to_first_forward_s is not None
            else t6 - t0
        ),
        host_preemption_to_first_forward_s=host_preemption_to_first_forward_s,
        prefetch_time_s=prefetch_time_s,
        blocking_time_s=blocking_time_s,
        total_transfer_time_s=total_transfer_time_s,
        prefetch_h2d_bytes=integers["prefetch_h2d_bytes"],
        blocking_h2d_bytes=integers["blocking_h2d_bytes"],
        total_h2d_bytes=expected_total,
        gpu_splice_bytes=integers["gpu_splice_bytes"],
        extra_preemptions=integers["extra_preemptions"],
        allocation_failures=integers["allocation_failures"],
        filler_failures=integers["filler_failures"],
        hidden_transfer=t3_fence <= t4,
        fence_before_first_forward=bool(event["fence_before_first_forward"]),
        splice_ack_id=splice_ack_id,
        request_first_forward_ns=request_first_forward_ns,
        request_finished_ns=request_finished_ns,
        request_finished_source=request_finished_source,
        staging_ready_ns=usability["staging_ready_ns"],
        destination_install_submitted_ns=usability[
            "destination_install_submitted_ns"
        ],
        destination_installed_ns=usability["destination_installed_ns"],
        request_usable_ns=usability["request_usable_ns"],
        request_usable_source=usability["request_usable_source"],
        transfer_hidden_before_demand_s=usability[
            "transfer_hidden_before_demand_s"
        ],
        staging_ready_lead_time_s=usability["staging_ready_lead_time_s"],
        installation_time_s=usability["installation_time_s"],
        exposed_blocking_stall_s=usability["exposed_blocking_stall_s"],
    )


def validate_prefetch_event_stream(
    events: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Validate an event-level stream for one request.

    Producers may emit these phase records and assemble the scalar episode
    record later.  This separate check catches duplicate, foreign, and
    out-of-order records before aggregation can hide them.
    """

    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise _error("prefetch event stream must be a JSON object array")
    if not events:
        raise _error("prefetch event stream cannot be empty")
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    seen_phases: set[str] = set()
    stream_identity: dict[str, Any] | None = None
    request_id: str | None = None
    previous_timestamp: float | None = None
    previous_index = -1
    for event in events:
        if not isinstance(event, Mapping):
            raise _error("prefetch stream entries must be JSON objects")
        event_id = _required_string(event.get("event_id"), "event_id")
        if event_id in seen_ids:
            raise _error("prefetch event stream contains duplicate event_id")
        seen_ids.add(event_id)
        ordered_ids.append(event_id)
        current_request_id = _required_string(event.get("request_id"), "request_id")
        if request_id is None:
            request_id = current_request_id
        elif current_request_id != request_id:
            raise _error("prefetch event stream contains a foreign request")
        current_identity = {
            "run_id": _required_string(event.get("run_id"), "run_id"),
            "cell_id": _required_string(event.get("cell_id"), "cell_id"),
            "episode_id": _required_string(event.get("episode_id"), "episode_id"),
            "seed": event.get("seed"),
            "request_id": current_request_id,
            "status": _required_string(event.get("status"), "status"),
        }
        if isinstance(current_identity["seed"], bool) or not isinstance(
            current_identity["seed"], int
        ):
            raise _error("stream seed must be an integer")
        if current_identity["status"] not in _STATUSES:
            raise _error("stream contains an unsupported status")
        if current_identity["status"] != "success":
            _required_string(event.get("status_reason"), "status_reason")
        if stream_identity is None:
            stream_identity = current_identity
        elif current_identity != stream_identity:
            raise _error("prefetch event stream identity does not match")
        phase = _required_string(event.get("phase"), "phase")
        try:
            phase_index = _EVENT_PHASES.index(phase)
        except ValueError as exc:
            raise _error(f"unsupported prefetch phase={phase!r}") from exc
        if phase in seen_phases:
            raise _error("prefetch event stream contains duplicate phase")
        if phase_index < previous_index:
            raise _error("prefetch event stream phases are out of order")
        seen_phases.add(phase)
        timestamp = _timestamp(event.get("timestamp"), "timestamp")
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise _error("prefetch event stream timestamps are out of order")
        previous_timestamp = timestamp
        previous_index = phase_index
    status = stream_identity["status"]
    if status == "success":
        missing = _REQUIRED_EVENT_PHASES - seen_phases
        if missing:
            raise _error(f"prefetch event stream is missing phases: {sorted(missing)}")
        if ("blocking_load_start" in seen_phases) != (
            "blocking_load_end" in seen_phases
        ):
            raise _error("blocking load phases must be emitted as a pair")
    return tuple(ordered_ids)


def aggregate_prefetch_event_streams(
    streams: Sequence[Sequence[Mapping[str, Any]]],
    *,
    expected_request_ids: Sequence[str],
    run_id: str,
    cell_id: str,
    seed: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """Validate and join one complete stream per expected request.

    Stream validation alone cannot detect an event ID reused by a second
    request or an episode duplicated across streams.  This aggregator keeps
    those checks at the harness boundary and returns streams in the caller's
    immutable request order.
    """
    expected = tuple(expected_request_ids)
    if len(expected) != len(set(expected)):
        raise _error("expected request IDs contain duplicates")
    expected_set = set(expected)
    by_request: dict[str, tuple[Mapping[str, Any], ...]] = {}
    event_ids: set[str] = set()
    episode_ids: set[str] = set()
    for stream in streams:
        validate_prefetch_event_stream(stream)
        request_id = stream[0]["request_id"]
        if request_id not in expected_set:
            raise _error("prefetch event stream contains a foreign request")
        if request_id in by_request:
            raise _error("duplicate prefetch request episode")
        for event in stream:
            if (
                event.get("run_id") != run_id
                or event.get("cell_id") != cell_id
                or event.get("seed") != seed
            ):
                raise _error("prefetch event stream identity does not match cell")
            event_id = event["event_id"]
            if event_id in event_ids:
                raise _error("duplicate prefetch event ID across streams")
            event_ids.add(event_id)
        episode_id = stream[0]["episode_id"]
        if episode_id in episode_ids:
            raise _error("duplicate prefetch episode ID across streams")
        episode_ids.add(episode_id)
        by_request[request_id] = tuple(stream)
    missing = expected_set - set(by_request)
    if missing:
        raise _error(f"missing prefetch request episodes: {sorted(missing)}")
    return tuple(by_request[request_id] for request_id in expected)


def validate_prefetch_episode_with_stream(
    episode: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> PrefetchAttribution:
    """Validate a scalar episode and prove that its phase stream agrees."""

    result = validate_prefetch_episode(episode)
    validate_prefetch_event_stream(events)
    if result.status != "success":
        return result
    for event in events:
        phase = event["phase"]
        scalar_field = _PHASE_TO_SCALAR[phase]
        scalar_value = episode.get(scalar_field)
        if scalar_value is None:
            raise _error(f"stream phase {phase!r} is absent from scalar episode")
        if not _same_float(
            _timestamp(event["timestamp"], f"{phase}.timestamp"),
            _timestamp(scalar_value, scalar_field),
        ):
            raise _error(f"stream phase {phase!r} disagrees with scalar episode")
        for field in (
            "run_id",
            "cell_id",
            "episode_id",
            "request_id",
            "seed",
            "status",
        ):
            if event.get(field) != episode.get(field):
                raise _error(f"stream/scalar identity mismatch for {field}")
    return result


def validate_prefetch_batch(
    episodes: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str | None = None,
    expected_cell_id: str | None = None,
    expected_seed: int | None = None,
    expected_episode_ids: Sequence[str] | None = None,
    streams: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[PrefetchAttribution, ...]:
    """Validate a homogeneous run/cell/seed batch without hiding joins.

    The per-episode validator cannot detect a duplicated episode emitted by a
    producer or a missing expected request. This aggregator requires every
    episode to belong to one run/cell/seed, rejects duplicate episode IDs, and
    optionally requires an exact expected episode set and a matching scalar /
    phase-stream record for every episode.
    """
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise _error("prefetch episode batch must be a JSON object array")
    if not episodes:
        raise _error("prefetch episode batch cannot be empty")
    if expected_run_id is not None:
        expected_run_id = _required_string(expected_run_id, "expected_run_id")
    if expected_cell_id is not None:
        expected_cell_id = _required_string(expected_cell_id, "expected_cell_id")
    if expected_seed is not None and (
        isinstance(expected_seed, bool) or not isinstance(expected_seed, int)
    ):
        raise _error("expected_seed must be an integer")
    expected_ids = (
        set(_string_tuple(expected_episode_ids, "expected_episode_ids"))
        if expected_episode_ids is not None
        else None
    )
    if streams is not None and not isinstance(streams, Mapping):
        raise _error("streams must be a JSON object keyed by episode_id")

    results: list[PrefetchAttribution] = []
    seen_episode_ids: set[str] = set()
    batch_identity: tuple[str, str, int] | None = None
    for episode in episodes:
        result = validate_prefetch_episode(episode)
        identity = (result.run_id, result.cell_id, result.seed)
        if batch_identity is None:
            batch_identity = identity
        elif identity != batch_identity:
            raise _error("prefetch batch contains a foreign run/cell/seed")
        if expected_run_id is not None and result.run_id != expected_run_id:
            raise _error("prefetch batch run_id does not match expectation")
        if expected_cell_id is not None and result.cell_id != expected_cell_id:
            raise _error("prefetch batch cell_id does not match expectation")
        if expected_seed is not None and result.seed != expected_seed:
            raise _error("prefetch batch seed does not match expectation")
        if result.episode_id in seen_episode_ids:
            raise _error("prefetch batch contains duplicate episode_id")
        seen_episode_ids.add(result.episode_id)
        if streams is not None:
            stream = streams.get(result.episode_id)
            if stream is None:
                raise _error(
                    f"prefetch batch is missing stream for {result.episode_id!r}"
                )
            validate_prefetch_episode_with_stream(episode, stream)
        results.append(result)

    if expected_ids is not None and seen_episode_ids != expected_ids:
        raise _error("prefetch batch episode IDs do not match expectation")
    if streams is not None and set(streams) != seen_episode_ids:
        raise _error("prefetch batch streams do not match episode IDs")
    return tuple(results)
