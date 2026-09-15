from __future__ import annotations

import json
from copy import deepcopy

import pytest

from benchmarks.prefetch_attribution import (
    PrefetchAttributionError,
    aggregate_prefetch_event_streams,
    validate_prefetch_episode,
    validate_prefetch_episode_with_stream,
    validate_prefetch_event_stream,
)
from benchmarks.run_latency_suite import _join_prefetch_attribution
from semantic_offload.connector import (
    _speculative_transfer_accounting as connector_speculative_accounting,
)
from semantic_offload.connector import _speculative_transfer_cost_accounting


def valid_event() -> dict:
    return {
        "run_id": "run-1",
        "cell_id": "rag-8",
        "seed": 3,
        "episode_id": "episode-1",
        "request_id": "req-1",
        "status": "success",
        "required_keys": ["k0", "k1", "k2"],
        "cpu_resident_ready_keys": ["k0", "k1"],
        "cpu_resident_keys": ["k0", "k1"],
        "cpu_ready_keys": ["k0", "k1"],
        "cpu_residency_proof": {"k0": "lookup-0", "k1": "lookup-1"},
        "prefetched_keys": ["k1", "k0"],
        "intersection_keys": ["k1", "k0"],
        "remainder_keys": ["k2"],
        "reservation_id": "reservation-1",
        "reservation_block_ids": ["scratch-0", "scratch-1"],
        "reservation_history": [
            {"phase": "reserved", "reserved_blocks": 2, "free_blocks": 8},
            {"phase": "released", "reserved_blocks": 0, "free_blocks": 10},
        ],
        "source_key_to_gpu_block": {"k0": "src-0", "k1": "src-1"},
        "destination_key_to_gpu_block": {"k0": "dst-0", "k1": "dst-1"},
        "t0_preemption": 10.0,
        "t1_reservation_submit": 10.1,
        "t2_prefetch_start": 10.2,
        "t3_prefetch_complete": 10.5,
        "t3_fence_complete": 10.6,
        "t4_readmission_allocation": 11.0,
        "t5_blocking_load_start": 11.1,
        "t5_blocking_load_end": 11.3,
        "t6_first_forward": 11.5,
        "prefetch_time_s": 0.3,
        "blocking_time_s": 0.2,
        "total_transfer_time_s": 1.1,
        "prefetch_h2d_bytes": 200,
        "blocking_h2d_bytes": 100,
        "total_h2d_bytes": 300,
        "gpu_splice_bytes": 400,
        "extra_preemptions": 0,
        "allocation_failures": 0,
        "filler_failures": 0,
        "source_reuse_before_fence": False,
        "fence_before_first_forward": True,
        "splice_acknowledged": True,
        "splice_ack_id": "splice-ack-1",
        "terminal_cleanup": {
            "reservation_released": True,
            "source_reuse_fenced": True,
            "request_state_cleared": True,
        },
    }


def test_valid_episode_derives_overlap_and_separate_bytes():
    result = validate_prefetch_episode(valid_event())

    assert result.queue_window_s == pytest.approx(1.0)
    assert result.usable_overlap_s == pytest.approx(0.9)
    assert result.preemption_to_first_forward_s == pytest.approx(1.5)
    assert result.total_h2d_bytes == 300
    assert result.hidden_transfer is True
    assert result.host_preemption_to_first_forward_s == pytest.approx(1.5)


def test_request_terminal_timing_is_optional_but_ordered_when_present():
    event = valid_event()
    event.update(
        request_first_forward_ns=100,
        request_finished_ns=120,
        request_finished_source="connector_request_finished",
    )
    result = validate_prefetch_episode(event)
    assert result.request_first_forward_ns == 100
    assert result.request_finished_ns == 120
    assert result.request_finished_source == "connector_request_finished"

    event["request_finished_ns"] = 99
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)

    event = valid_event()
    event["request_finished_ns"] = 120
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def _speculative_transfer_accounting() -> dict[str, object]:
    return {
        "job_count": 2,
        "successful_job_count": 1,
        "failed_job_count": 1,
        "cancelled_job_count": 1,
        "attempted_blocks": 3,
        "attempted_bytes": 300,
        "submitted_jobs": 2,
        "submitted_blocks": 2,
        "submitted_bytes": 200,
        "completed_bytes": 100,
        "consumed_bytes": 75,
        "consumption_events": 1,
        "discarded_bytes": 25,
        "duplicated_job_count": 0,
        "duplicated_bytes": 0,
        "waste_completed_bytes": 25,
        "late_job_count": 0,
        "late_bytes": 0,
        "on_time_job_count": 1,
        "on_time_bytes": 100,
        "demand_boundary_ns": 120,
        "demand_boundary_source": "readmission_allocation",
        "released_reservation_bytes": 25,
        "uncompleted_submitted_bytes": 100,
        "unconsumed_completed_bytes": 0,
        "actual_contributors": 2,
        "expected_contributors": 2,
        "device_time_s": 0.002,
    }


def test_speculative_transfer_lifecycle_totals_are_validated():
    event = valid_event()
    event["speculative_transfer"] = _speculative_transfer_accounting()
    event["speculative_transfer_costs"] = {
        "duplicated_job_count": 0,
        "duplicated_bytes": 0,
        "waste_completed_bytes": 25,
    }
    assert validate_prefetch_episode(event).status == "success"

    event["speculative_transfer"]["late_job_count"] = 1
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_duplicate_cost_counts_repeated_physical_load_of_same_key():
    state = {
        "prefetch_attempted_blocks": 1,
        "prefetch_submitted_blocks": 1,
        "prefetch_submitted_jobs": 1,
        "prefetch_consumed_blocks": 0,
        "prefetch_discarded_blocks": 1,
        "speculative_transfer_job_statuses": [
            {
                "status": "success",
                "success": True,
                "cancelled": False,
                "keys": ["same-key"],
                "bytes": 128,
                "contributors": 1,
                "expected_contributors": 1,
            }
        ],
        "transfer_job_statuses": [
            {
                "direction": "load",
                "is_prefetch": False,
                "success": True,
                "keys": ["same-key"],
            }
        ],
    }
    accounting = connector_speculative_accounting(state, 128)

    assert _speculative_transfer_cost_accounting(state, 128, accounting) == {
        "duplicated_job_count": 1,
        "duplicated_bytes": 128,
        "waste_completed_bytes": 128,
    }

    event = valid_event()
    event["speculative_transfer"] = _speculative_transfer_accounting()
    event["speculative_transfer_costs"] = {
        "duplicated_job_count": 0,
        "duplicated_bytes": 0,
        "waste_completed_bytes": 24,
    }
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_retained_v8_speculative_totals_remain_readable():
    event = valid_event()
    accounting = _speculative_transfer_accounting()
    for field in (
        "cancelled_job_count",
        "attempted_blocks",
        "attempted_bytes",
        "submitted_jobs",
        "submitted_blocks",
        "consumption_events",
        "late_job_count",
        "late_bytes",
        "on_time_job_count",
        "on_time_bytes",
        "demand_boundary_ns",
        "demand_boundary_source",
    ):
        accounting.pop(field)
    event["speculative_transfer"] = accounting
    assert validate_prefetch_episode(event).status == "success"


def test_hint_terminal_statuses_preserve_partial_evidence():
    for status in ("hint_consumed", "hint_failed", "hint_no_action"):
        event = valid_event()
        event["status"] = status
        event["status_reason"] = "bounded_hint_result"
        event["speculative_transfer"] = _speculative_transfer_accounting()
        assert validate_prefetch_episode(event).status == status


def test_hint_event_retains_installed_usable_and_exposed_stall_timing():
    event = valid_event()
    event.update(
        status="hint_consumed",
        status_reason="known_prefix_staging_spliced",
        t2_prefetch_start=100,
        t3_prefetch_complete=160,
        t4_readmission_allocation=200,
        staging_ready_ns=160,
        destination_install_submitted_ns=210,
        destination_installed_ns=250,
        request_usable_ns=250,
        request_usable_source="all_worker_splice_cuda_fences_acknowledged",
        request_usable_complete=True,
        transfer_hidden_before_demand_s=60 / 1_000_000_000,
        staging_ready_lead_time_s=40 / 1_000_000_000,
        installation_time_s=40 / 1_000_000_000,
        exposed_blocking_stall_s=50 / 1_000_000_000,
    )

    result = validate_prefetch_episode(event)

    assert result.staging_ready_ns == 160
    assert result.destination_installed_ns == 250
    assert result.request_usable_ns == 250
    assert result.exposed_blocking_stall_s == pytest.approx(50 / 1_000_000_000)

    event["destination_installed_ns"] = 249
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_promoted_demand_boundary_requires_exact_source_clock_and_t4_alias():
    event = valid_event()
    event.update(
        status="hint_consumed",
        status_reason="known_prefix_staging_spliced",
        t4_readmission_allocation=200,
        demand_boundary_ns=200,
        demand_boundary_source="hinted_request_connector_observation",
        demand_boundary_clock_domain="host_monotonic",
    )
    assert validate_prefetch_episode(event).status == "hint_consumed"

    event["demand_boundary_ns"] = 201
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)

    event["demand_boundary_ns"] = 200
    event["demand_boundary_source"] = "guessed_client_send"
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_host_preemption_interval_is_unavailable_for_bad_or_mixed_timestamps():
    event = valid_event()
    event["t6_first_forward"] = 9.0
    event["status"] = "unavailable"
    event["status_reason"] = "host_interval_invalid"
    assert validate_prefetch_episode(event).host_preemption_to_first_forward_s is None

    event = valid_event()
    event["status"] = "unavailable"
    event["status_reason"] = "mixed_clock_domains"
    event["timestamp_clock_domains"] = {
        field: "host_monotonic"
        for field in (
            "t0_preemption",
            "t1_reservation_submit",
            "t2_prefetch_start",
            "t3_prefetch_complete",
            "t3_fence_complete",
            "t4_readmission_allocation",
            "t5_blocking_load_start",
            "t5_blocking_load_end",
            "t6_first_forward",
        )
    }
    event["timestamp_clock_domains"]["t6_first_forward"] = "other_clock"
    assert validate_prefetch_episode(event).host_preemption_to_first_forward_s is None


def _valid_block_pool_snapshot() -> dict[str, object]:
    return {
        "schema": "semantic-block-pool-snapshot-v1",
        "phase": "t1_reservation_submit",
        "source": "prefetch_reservation",
        "observed_at_ns": 100,
        "episode_id": 1,
        "preemption_index": 1,
        "available": True,
        "total_blocks": 101,
        "null_block_count": 1,
        "schedulable_capacity_blocks": 100,
        "free_block_queue_blocks": 60,
        "scheduler_prefetch_reserved_blocks_total": 5,
        "estimated_referenced_blocks": 35,
        "accounting_underflow_blocks": 0,
        "accounting_consistent": True,
        "configured_prefetch_headroom_blocks": 2,
        "scheduler_free_headroom_blocks": 58,
    }


def test_block_pool_snapshot_is_validated_as_scheduler_accounting():
    event = valid_event()
    event["block_pool_snapshots"] = [_valid_block_pool_snapshot()]
    event["block_pool_snapshot_count"] = 1
    event["block_pool_snapshots_truncated"] = False

    assert validate_prefetch_episode(event).status == "success"


def test_identity_bearing_gpu_occupancy_snapshot_validates_bytes_and_disjointness():
    event = valid_event()
    snapshot = _valid_block_pool_snapshot()
    snapshot.update(
        physical_block_bytes=128,
        scheduler_prefetch_reserved_block_ids=[1, 2, 3, 4, 5],
        staging_identity_complete=True,
        staging_reserved_bytes=640,
        gpu_usable_context_blocks=[
            {
                "block_id": 8,
                "ref_cnt": 1,
                "block_hash_digest": None,
                "reusable_prefix": False,
            }
        ],
        gpu_usable_context_block_ids=[8],
        gpu_usable_context_bytes=128,
        gpu_schedulable_capacity_bytes=12_800,
    )
    event["block_pool_snapshots"] = [snapshot]
    assert validate_prefetch_episode(event).status == "success"

    snapshot["gpu_usable_context_block_ids"] = [1]
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


@pytest.mark.parametrize(
    "field, value",
    [
        ("estimated_referenced_blocks", 34),
        ("accounting_underflow_blocks", -1),
        ("scheduler_free_headroom_blocks", 60),
    ],
)
def test_block_pool_snapshot_rejects_inconsistent_accounting(field, value):
    event = valid_event()
    snapshot = _valid_block_pool_snapshot()
    snapshot[field] = value
    event["block_pool_snapshots"] = [snapshot]

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_block_pool_snapshot_allows_explicitly_truncated_history():
    event = valid_event()
    event["block_pool_snapshots"] = [_valid_block_pool_snapshot()]
    event["block_pool_snapshot_count"] = 65
    event["block_pool_snapshots_truncated"] = True

    assert validate_prefetch_episode(event).status == "success"


def test_block_pool_snapshot_requires_reason_when_unavailable():
    event = valid_event()
    snapshot = _valid_block_pool_snapshot()
    snapshot["available"] = False
    snapshot.pop("reason", None)
    event["block_pool_snapshots"] = [snapshot]

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def _valid_cpu_cache_snapshot() -> dict[str, object]:
    return {
        "schema": "semantic-cpu-cache-snapshot-v1",
        "phase": "t0_preemption",
        "source": "build_connector_meta_preempted_ids",
        "observed_at_ns": 100,
        "episode_id": 0,
        "preemption_index": 1,
        "available": True,
        "total_blocks": 128,
        "allocated_blocks": 100,
        "free_list_blocks": 10,
        "free_capacity_blocks": 38,
        "evictable_blocks": 80,
        "write_pending_blocks": 2,
        "estimated_non_evictable_blocks": 10,
        "accounting_underflow_blocks": 0,
        "accounting_consistent": True,
        "policy_resident_entries": 100,
        "policy_ghost_entries": 16,
        "policy_class": "SemanticPolicy",
    }


def test_cpu_cache_snapshot_is_validated_as_occupancy_accounting():
    event = valid_event()
    event["cpu_cache_snapshots"] = [_valid_cpu_cache_snapshot()]
    event["cpu_cache_snapshot_count"] = 1
    event["cpu_cache_snapshots_truncated"] = False

    assert validate_prefetch_episode(event).status == "success"

    event["cpu_cache_snapshots"][0]["free_capacity_blocks"] = 37
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_identity_bearing_dram_snapshot_validates_shared_blocks_once():
    event = valid_event()
    snapshot = _valid_cpu_cache_snapshot()
    snapshot.update(
        policy_resident_entries=2,
        physical_block_bytes=128,
        resident_blocks=[
            {"key": "k0", "cpu_block_id": 3, "ready": True, "ref_cnt": 0},
            {"key": "k1", "cpu_block_id": 4, "ready": False, "ref_cnt": -1},
        ],
        resident_bytes=256,
        ready_resident_bytes=128,
        cpu_capacity_bytes=16_384,
        actual_offloaded_fraction_numerator_bytes=128,
        actual_offloaded_fraction_denominator_bytes=16_384,
        actual_offloaded_fraction=1 / 128,
        write_pending_bytes=256,
    )
    event["cpu_cache_snapshots"] = [snapshot]
    assert validate_prefetch_episode(event).status == "success"

    snapshot["resident_blocks"][1]["key"] = "k0"
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_timestamp_clock_domains_are_optional_but_mixed_success_is_rejected():
    event = valid_event()
    event["timestamp_clock_domains"] = {
        field: "shared_monotonic"
        for field in (
            "t0_preemption",
            "t1_reservation_submit",
            "t2_prefetch_start",
            "t3_prefetch_complete",
            "t3_fence_complete",
            "t4_readmission_allocation",
            "t5_blocking_load_start",
            "t5_blocking_load_end",
            "t6_first_forward",
        )
    }
    assert validate_prefetch_episode(event).status == "success"

    event["timestamp_clock_domains"]["t6_first_forward"] = "worker_monotonic"
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_keys", ["k0", "k0"]),
        ("prefetched_keys", ["unknown"]),
        ("cpu_ready_keys", []),
        ("source_key_to_gpu_block", {"k0": "src-0"}),
        ("destination_key_to_gpu_block", {"k0": "dst-0"}),
    ],
)
def test_identity_and_residency_fail_closed(field, value):
    event = valid_event()
    event[field] = value

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("t1_reservation_submit", 9.0),
        ("t2_prefetch_start", 9.0),
        ("t3_fence_complete", 12.0),
        ("t4_readmission_allocation", 10.4),
        ("t5_blocking_load_end", 11.0),
        ("t6_first_forward", 11.2),
    ],
)
def test_timestamp_order_fail_closed(field, value):
    event = valid_event()
    event[field] = value

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_total_transfer_bytes_cannot_hide_staging_traffic():
    event = valid_event()
    event["total_h2d_bytes"] = 100

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_reuse_before_fence", True),
        (
            "terminal_cleanup",
            {
                "reservation_released": False,
                "source_reuse_fenced": True,
                "request_state_cleared": True,
            },
        ),
    ],
)
def test_cleanup_and_source_fence_are_mandatory(field, value):
    event = valid_event()
    event[field] = value

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode(event)


def test_unavailable_episode_requires_reason_but_remains_classified():
    event = deepcopy(valid_event())
    event["status"] = "unavailable"
    event["status_reason"] = "lookup_metadata_missing"

    result = validate_prefetch_episode(event)

    assert result.status == "unavailable"


def test_latency_join_binds_cell_seed_and_keeps_missing_unavailable():
    candidate = deepcopy(valid_event())
    candidate["request_id"] = "req-1"
    candidate["seed"] = None
    joined = _join_prefetch_attribution(
        [{"prefetch_attribution": candidate}],
        ["req-1"],
        run_id="run-1",
        cell_id="rag-8",
        seed=3,
    )

    assert joined["prefetch_attribution_join_status"] == "complete"
    records = json.loads(joined["prefetch_attribution_json"])
    assert records[0]["seed"] == 3
    assert records[0]["seed_source"] == "harness_cell"

    missing = _join_prefetch_attribution(
        [{}], ["req-1"], run_id="run-1", cell_id="rag-8", seed=3
    )
    assert missing["prefetch_attribution_join_status"] == "unavailable"
    assert (
        json.loads(missing["prefetch_attribution_json"])[0]["status"] == "unavailable"
    )


def test_latency_join_rejects_foreign_nested_attribution():
    candidate = deepcopy(valid_event())
    candidate["request_id"] = "foreign"
    joined = _join_prefetch_attribution(
        [{"prefetch_attribution": candidate}],
        ["req-1"],
        run_id="run-1",
        cell_id="rag-8",
        seed=3,
    )

    assert joined["prefetch_attribution_join_status"] == "invalid"
    record = json.loads(joined["prefetch_attribution_json"])[0]
    assert record["status"] == "invalid"
    assert record["status_reason"] == "prefetch_attribution_identity_mismatch"


def valid_stream() -> list[dict]:
    phases = [
        ("e0", "preemption", 10.0),
        ("e1", "reservation_submit", 10.1),
        ("e2", "prefetch_start", 10.2),
        ("e3", "prefetch_complete", 10.5),
        ("e4", "fence_complete", 10.6),
        ("e5", "readmission_allocation", 11.0),
        ("e6", "blocking_load_start", 11.1),
        ("e7", "blocking_load_end", 11.3),
        ("e8", "first_forward", 11.5),
    ]
    return [
        {
            "event_id": event_id,
            "run_id": "run-1",
            "cell_id": "rag-8",
            "episode_id": "episode-1",
            "seed": 3,
            "request_id": "req-1",
            "status": "success",
            "phase": phase,
            "timestamp": timestamp,
        }
        for event_id, phase, timestamp in phases
    ]


def test_event_stream_accepts_one_request_in_phase_order():
    event_ids = validate_prefetch_event_stream(valid_stream())

    assert event_ids == tuple(event["event_id"] for event in valid_stream())


def test_scalar_episode_and_stream_must_agree():
    result = validate_prefetch_episode_with_stream(valid_event(), valid_stream())

    assert result.splice_ack_id == "splice-ack-1"


def test_scalar_episode_and_stream_mismatch_fails_closed():
    stream = valid_stream()
    stream[-1] = dict(stream[-1], timestamp=11.6)

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_episode_with_stream(valid_event(), stream)


def test_event_stream_requires_all_nonoptional_phases():
    stream = valid_stream()[:-1]

    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_event_stream(stream)


def test_aggregator_rejects_duplicate_ids_and_missing_requests():
    first = valid_stream()
    second = [dict(event) for event in valid_stream()]
    for event in second:
        event["request_id"] = "req-2"
        event["episode_id"] = "episode-2"
    with pytest.raises(PrefetchAttributionError, match="event ID"):
        aggregate_prefetch_event_streams(
            [first, second],
            expected_request_ids=["req-1", "req-2"],
            run_id="run-1",
            cell_id="rag-8",
            seed=3,
        )

    with pytest.raises(PrefetchAttributionError, match="missing"):
        aggregate_prefetch_event_streams(
            [first],
            expected_request_ids=["req-1", "req-2"],
            run_id="run-1",
            cell_id="rag-8",
            seed=3,
        )


def test_aggregator_rejects_duplicate_episode_and_cross_cell_streams():
    first = valid_stream()
    second = [dict(event) for event in valid_stream()]
    for event in second:
        event["request_id"] = "req-2"
        event["event_id"] = f"other-{event['event_id']}"
    with pytest.raises(PrefetchAttributionError, match="episode ID"):
        aggregate_prefetch_event_streams(
            [first, second],
            expected_request_ids=["req-1", "req-2"],
            run_id="run-1",
            cell_id="rag-8",
            seed=3,
        )

    foreign = [dict(event, cell_id="other-cell") for event in valid_stream()]
    with pytest.raises(PrefetchAttributionError, match="cell"):
        aggregate_prefetch_event_streams(
            [foreign],
            expected_request_ids=["req-1"],
            run_id="run-1",
            cell_id="rag-8",
            seed=3,
        )


def test_non_success_streams_may_be_partial_without_fabricated_phases():
    stream = [dict(valid_stream()[0], status="unavailable", status_reason="t6_missing")]

    assert validate_prefetch_event_stream(stream) == ("e0",)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda stream: stream + [deepcopy(stream[0])],
        lambda stream: [dict(stream[0], request_id="foreign"), *stream[1:]],
        lambda stream: [stream[1], stream[0], *stream[2:]],
        lambda stream: [dict(stream[1], timestamp=9.0), *stream[1:]],
    ],
)
def test_event_stream_rejects_duplicate_foreign_or_out_of_order_records(mutator):
    with pytest.raises(PrefetchAttributionError):
        validate_prefetch_event_stream(mutator(valid_stream()))
