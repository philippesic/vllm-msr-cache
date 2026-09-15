# SPDX-License-Identifier: Apache-2.0
"""Adversarial CPU tests for authoritative Chat/RAG preservation mapping."""

import hashlib
import json

import pytest

from benchmarks.preservation import (
    AUTHORITATIVE_REQUEST_EVENTS_BASIS,
    MIXED_REQUEST_OUTCOMES_REASON,
    REQUEST_EVENT_ACCOUNTING_REASON,
    REQUEST_EVENT_DUPLICATE_REASON,
    REQUEST_EVENT_FOREIGN_REASON,
    REQUEST_EVENT_MISSING_REASON,
    REQUEST_EVENT_OUTCOME_REASON,
    REQUEST_EVENT_PHYSICAL_REASON,
    REQUEST_EVENT_TP_REASON,
    REQUEST_EVENT_TRANSFER_FAILED_REASON,
    REQUEST_EVENT_TRANSFER_REASON,
    map_preservation_events,
    validate_preservation_row,
)
from benchmarks.run_latency_suite import join_preservation_events


def _event(
    sample_index=0,
    *,
    mode="hit",
    run_id="run-1",
    cell_id="cell-1",
    lookup_count=1,
    admission_count=1,
    preempted=False,
):
    request_id = f"sc|{run_id}|{cell_id}|chat|sample-{sample_index}"
    if mode == "hit":
        required_tokens, reused_tokens, loaded_tokens, recomputed_tokens = 8, 8, 8, 0
        local_blocks, external_blocks, recomputed_blocks = 1, 1, 0
        pressure, outcome, reason = True, "hit", "authoritative_offload_events"
    elif mode == "partial":
        required_tokens, reused_tokens, loaded_tokens, recomputed_tokens = 8, 4, 4, 4
        local_blocks, external_blocks, recomputed_blocks = 1, 1, 1
        pressure, outcome, reason = True, "partial", "authoritative_offload_events"
    elif mode == "miss":
        required_tokens, reused_tokens, loaded_tokens, recomputed_tokens = 8, 0, 0, 8
        local_blocks, external_blocks, recomputed_blocks = 0, 0, 2
        pressure, outcome, reason = True, "miss", "authoritative_offload_events"
    elif mode == "cold":
        required_tokens, reused_tokens, loaded_tokens, recomputed_tokens = 8, 0, 0, 8
        local_blocks, external_blocks, recomputed_blocks = 0, 0, 2
        pressure, outcome, reason = False, "not_pressured", "cold_or_not_eligible"
    elif mode == "zero":
        required_tokens, reused_tokens, loaded_tokens, recomputed_tokens = 0, 0, 0, 0
        local_blocks, external_blocks, recomputed_blocks = 0, 0, 0
        pressure, outcome, reason = False, "not_pressured", "cold_or_not_eligible"
    else:
        raise AssertionError(mode)

    local_ids = [[0, 0, 1 + index] for index in range(local_blocks)]
    external_ids = [[0, 0, 100 + index] for index in range(external_blocks)]
    recomputed_ids = [[0, 0, 200 + index] for index in range(recomputed_blocks)]
    required_ids = local_ids + external_ids
    scheduled_ids = recomputed_ids
    admission_episodes = [
        {
            "episode_id": index,
            "lookup_id": min(index, max(0, lookup_count - 1)) if lookup_count else None,
            "scheduled_tokens": 8 if required_tokens else 0,
            "prompt_tokens_locally_executed": recomputed_tokens,
        }
        for index in range(admission_count)
    ]
    has_load = external_blocks > 0
    lookup_key = [[0, 0, 1]]
    if mode in {"hit", "partial"}:
        lookup_hits = lookup_key
        lookup_missing = []
    elif mode in {"miss", "cold"}:
        lookup_hits = []
        lookup_missing = lookup_key
    else:
        lookup_hits = []
        lookup_missing = []
    lookup_groups = (
        {
            "0": {
                "eligible": lookup_key,
                "queried": lookup_key,
                "hits": lookup_hits,
                "missing": lookup_missing,
                "pending": [],
                "retry": [],
                "observed": True,
                "no_eligible": False,
            }
        }
        if required_tokens
        else {
            "0": {
                "eligible": [],
                "queried": [],
                "hits": [],
                "missing": [],
                "pending": [],
                "retry": [],
                "observed": True,
                "no_eligible": True,
            }
        }
    )
    lookup_episodes = [
        {"lookup_id": index, "groups": lookup_groups} for index in range(lookup_count)
    ]
    prior_stored_keys = lookup_key if mode in {"hit", "partial"} else []
    prior_evicted_keys = lookup_key if mode == "miss" else []
    cold_keys = lookup_key if mode == "cold" else []
    transfer_statuses = (
        [
            {
                "job_id": 10,
                "direction": "load",
                "status": "success",
                "success": True,
                "cancelled": False,
                "contributors": 1,
                "expected_contributors": 1,
            }
        ]
        if has_load
        else []
    )
    loaded_ids = [[10, 0, 100 + index] for index in range(external_blocks)]
    return {
        "schema": "semantic-preservation-event-v1",
        "event": "request_preservation",
        "event_instance": 1,
        "expected_event_count": 1,
        "request_id": f"cmpl-{request_id}-0-deadbeef",
        "request_id_kind": "vllm_internal",
        "external_request_id": request_id,
        "request_identity_valid": True,
        "run_id": run_id,
        "cell_id": cell_id,
        "sample_id": request_id,
        "sample_index": sample_index,
        "terminal_status": "finished_length_capped",
        "terminal_reason": "completed",
        "outcome": outcome,
        "reason": reason,
        "authoritative": True,
        "event_integrity": "complete",
        "lookup_metadata_complete": True,
        "lookup_observed": True,
        "lookup_no_eligible": not bool(required_tokens),
        "transfer_metadata_complete": True,
        "admission_metadata_complete": True,
        "physical_block_identity_complete": True,
        "tp_contributors_complete": True,
        "transfer_job_statuses": transfer_statuses,
        "transfer_job_counts": {"load": 1 if has_load else 0, "store": 0},
        "lookup_keys": lookup_key if required_tokens else [],
        "prior_stored_keys": prior_stored_keys,
        "prior_evicted_keys": prior_evicted_keys,
        "cold_keys": cold_keys,
        "lookup_episodes": lookup_episodes,
        "admission_episodes": admission_episodes,
        "required_tokens": required_tokens,
        "reused_tokens": reused_tokens,
        "loaded_tokens": loaded_tokens,
        "recomputed_tokens": recomputed_tokens,
        "actual_prompt_tokens_locally_executed": recomputed_tokens,
        "cumulative_prompt_tokens_locally_executed": recomputed_tokens,
        "prompt_tokens_scheduled": 8 if required_tokens else 0,
        "eligible_kv_blocks": {
            "required": len(required_ids),
            "local": len(local_ids),
            "external": len(external_ids),
            "recomputed": len(recomputed_ids),
        },
        "required_kv_blocks": len(required_ids),
        "physical_block_ids": {
            "required": required_ids,
            "local": local_ids,
            "external": external_ids,
            "recomputed": recomputed_ids,
            "scheduled": scheduled_ids,
            "loaded": loaded_ids,
            "stored": [],
        },
        "loaded_kv_blocks": len(external_ids),
        "stored_kv_blocks": 0,
        "load_bytes": 1024 if has_load else 0,
        "store_bytes": 0,
        "admission_count": admission_count,
        "recompute_admission_count": 1 if mode == "miss" else 0,
        "tp_transfer_contributors": {
            "load": 1 if has_load else 0,
            "store": 0,
            "expected": {"load": 1 if has_load else 0, "store": 0},
        },
        "tp_transfer_contributor_totals": {
            "actual": {"load": 1 if has_load else 0, "store": 0},
            "expected": {"load": 1 if has_load else 0, "store": 0},
        },
        "tp_world_size": 1,
        "tp_rank_count": 1,
        "pressure": pressure,
        "preempted": preempted,
        "preemption_count": 1 if preempted else 0,
    }


def _mapped_row(mapping, events):
    row = {
        "sample_index": "0",
        "workload": "chat",
        "sub_workload": "",
        "preservation_outcome": mapping["preservation_outcome"],
        "preservation_reason": mapping["preservation_reason"],
        "preservation_request_id": "",
        "preservation_request_id_supplied": "false",
        "preservation_join_status": "complete",
        "preservation_event_count": str(len(events)),
        "run_id": "run-1",
        "cell_id": "cell-1",
        "request_id_prefix": "sc|run-1|cell-1|chat|sample-",
        "needle_outcome": "",
        "recall_load_bytes": "",
        "recall_store_bytes": "",
        "preservation_events_json": json.dumps(events),
        "request_ids_json": mapping["preservation_request_ids_json"],
    }
    row.update(mapping)
    return row


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("hit", "hit"),
        ("partial", "partial"),
        ("miss", "miss"),
        ("cold", "not_pressured"),
        ("zero", "not_pressured"),
    ],
)
def test_request_event_outcomes_are_based_on_token_accounting(mode, expected):
    event = _event(mode=mode)
    mapping = map_preservation_events(
        [event],
        expected_request_ids=[event["external_request_id"]],
        run_id="run-1",
        cell_id="cell-1",
    )

    assert mapping["preservation_outcome"] == expected
    assert mapping["preservation_outcome_basis"] == AUTHORITATIVE_REQUEST_EVENTS_BASIS
    assert (
        json.loads(mapping["preservation_request_outcome_details_json"])[0]["outcome"]
        == expected
    )
    assert mapping["preservation_required_tokens"] == event["required_tokens"]
    assert (
        validate_preservation_row(_mapped_row(mapping, [event]), require_fields=True)
        == []
    )


def test_mixed_request_outcomes_are_explicitly_unavailable_and_totals_are_safe():
    events = [_event(0, mode="hit"), _event(1, mode="miss")]
    mapping = map_preservation_events(
        events,
        expected_request_ids=[event["external_request_id"] for event in events],
        run_id="run-1",
        cell_id="cell-1",
    )

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == MIXED_REQUEST_OUTCOMES_REASON
    assert mapping["preservation_authoritative_request_count"] == 2
    assert mapping["preservation_required_tokens"] == 16
    assert (
        validate_preservation_row(_mapped_row(mapping, events), require_fields=True)
        == []
    )


def test_top_level_totals_alone_cannot_label_a_hit():
    event = _event(mode="hit")
    event["lookup_episodes"] = []
    event["lookup_keys"] = []

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"
    assert mapping["preservation_required_tokens"] is None


def test_live_shaped_empty_lookup_keys_and_repeated_scheduler_steps_are_valid():
    event = _event(mode="miss")
    first_step = dict(event["admission_episodes"][0])
    scheduler_steps = [dict(first_step)]
    for _ in range(16):
        scheduler_steps.append(
            {
                "episode_id": first_step["episode_id"],
                "lookup_id": first_step["lookup_id"],
                "scheduled_tokens": 1,
                "prompt_tokens_locally_executed": 0,
            }
        )
    event["admission_episodes"] = scheduler_steps
    event["admission_count"] = 1
    event["prompt_tokens_scheduled"] = 24
    event["lookup_keys"] = []

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"
    assert mapping["preservation_required_tokens"] == 8


def test_explicit_complete_no_eligible_lookup_maps_cold_request_not_pressured():
    event = _event(mode="cold")
    empty_group = {
        "eligible": [],
        "queried": [],
        "hits": [],
        "missing": [],
        "pending": [],
        "retry": [],
        "observed": True,
        "no_eligible": True,
    }
    event.update(
        {
            "lookup_keys": [],
            "lookup_episodes": [{"lookup_id": 0, "groups": {"0": empty_group}}],
            "lookup_observed": True,
            "lookup_no_eligible": True,
            "lookup_metadata_complete": True,
            "prior_stored_keys": [],
            "prior_evicted_keys": [],
            "cold_keys": [],
            "required_tokens": 93,
            "reused_tokens": 80,
            "loaded_tokens": 0,
            "recomputed_tokens": 13,
            "actual_prompt_tokens_locally_executed": 13,
            "cumulative_prompt_tokens_locally_executed": 13,
            "prompt_tokens_scheduled": 29,
            "admission_episodes": [
                {
                    "episode_id": 0,
                    "lookup_id": 0,
                    "scheduled_tokens": 13,
                    "prompt_tokens_locally_executed": 13,
                },
                *[
                    {
                        "episode_id": 0,
                        "lookup_id": 0,
                        "scheduled_tokens": 1,
                        "prompt_tokens_locally_executed": 0,
                    }
                    for _ in range(16)
                ],
            ],
            "admission_count": 1,
            "recompute_admission_count": 0,
            "physical_block_ids": {
                "required": [[0, 0, index] for index in range(1, 6)],
                "local": [[0, 0, index] for index in range(1, 6)],
                "external": [],
                "recomputed": [[0, 0, 99]],
                "scheduled": [[0, 0, 99]],
                "loaded": [],
                "stored": [],
            },
            "eligible_kv_blocks": {
                "required": 5,
                "local": 5,
                "external": 0,
                "recomputed": 1,
            },
            "required_kv_blocks": 5,
            "loaded_kv_blocks": 0,
            "stored_kv_blocks": 0,
        }
    )

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "not_pressured"
    assert mapping["preservation_reason"] == AUTHORITATIVE_REQUEST_EVENTS_BASIS


def test_no_eligible_flag_rejects_nonempty_eligible_metadata():
    event = _event(mode="cold")
    event["lookup_no_eligible"] = True
    event["lookup_keys"] = []
    group = event["lookup_episodes"][0]["groups"]["0"]
    group.update(
        {
            "queried": [],
            "hits": [],
            "missing": [],
            "pending": [],
            "retry": [],
        }
    )

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"


@pytest.mark.parametrize("no_eligible", [False, True])
def test_all_empty_lookup_episode_cannot_mix_with_queried_episode(no_eligible):
    event = _event(mode="hit", lookup_count=2)
    event["lookup_no_eligible"] = no_eligible
    event["lookup_episodes"][1]["groups"] = {
        "0": {
            "eligible": [],
            "queried": [],
            "hits": [],
            "missing": [],
            "pending": [],
            "retry": [],
            "observed": True,
            "no_eligible": True,
        }
    }

    mapping = map_preservation_events([event])

    if no_eligible:
        assert mapping["preservation_outcome"] == "unavailable"
        assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"
    else:
        assert mapping["preservation_outcome"] == "hit"


@pytest.mark.parametrize("no_eligible", [False, True])
def test_same_lookup_episode_empty_and_queried_groups_fail_closed(no_eligible):
    event = _event(mode="hit")
    queried_group = event["lookup_episodes"][0]["groups"]["0"]
    event["lookup_no_eligible"] = no_eligible
    event["lookup_episodes"][0]["groups"] = {
        "0": {
            "eligible": [],
            "queried": [],
            "hits": [],
            "missing": [],
            "pending": [],
            "retry": [],
            "observed": True,
            "no_eligible": True,
        },
        "1": queried_group,
    }

    mapping = map_preservation_events([event])

    if no_eligible:
        assert mapping["preservation_outcome"] == "unavailable"
        assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"
    else:
        assert mapping["preservation_outcome"] == "hit"


def test_recomputed_execution_blocks_are_separate_from_eligible_blocks():
    event = _event(mode="miss")
    assert event["physical_block_ids"]["recomputed"]
    assert not set(map(tuple, event["physical_block_ids"]["recomputed"])).intersection(
        map(tuple, event["physical_block_ids"]["required"])
    )
    assert map_preservation_events([event])["preservation_outcome"] == "miss"

    event["physical_block_ids"]["required"].append(
        event["physical_block_ids"]["recomputed"][0]
    )
    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def test_pressure_only_fabricated_miss_with_unclassified_lookup_is_unavailable():
    event = _event(mode="miss")
    event["pressure"] = True
    event["prior_evicted_keys"] = []
    event["lookup_episodes"][0]["groups"]["0"]["missing"] = []
    event["cold_keys"] = []
    event["prior_stored_keys"] = []

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"


def test_pressure_flag_cannot_authorize_a_no_pressure_miss():
    event = _event(mode="miss")
    event["pressure"] = False

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_OUTCOME_REASON


@pytest.mark.parametrize(
    "mutator",
    [
        lambda event: event.update({"actual_prompt_tokens_locally_executed": 0}),
        lambda event: event.update({"prompt_tokens_scheduled": 0}),
        lambda event: event["admission_episodes"][0].update(
            {"prompt_tokens_locally_executed": 0}
        ),
    ],
)
def test_contradictory_admission_prompt_accounting_is_unavailable(mutator):
    event = _event(mode="miss")
    mutator(event)

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_ACCOUNTING_REASON


def test_per_record_prompt_execution_cannot_be_offset_by_another_step():
    event = _event(mode="miss")
    event["admission_episodes"] = [
        {
            "episode_id": 0,
            "lookup_id": 0,
            "scheduled_tokens": 0,
            "prompt_tokens_locally_executed": 8,
        },
        {
            "episode_id": 0,
            "lookup_id": 0,
            "scheduled_tokens": 8,
            "prompt_tokens_locally_executed": 0,
        },
    ]

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_ACCOUNTING_REASON


def test_cumulative_prompt_replay_is_separate_from_logical_recomputed_tokens():
    event = _event(mode="miss")
    event["admission_episodes"] = [
        {
            "episode_id": 0,
            "lookup_id": 0,
            "scheduled_tokens": 8,
            "prompt_tokens_locally_executed": 8,
        },
        {
            "episode_id": 0,
            "lookup_id": 0,
            "scheduled_tokens": 8,
            "prompt_tokens_locally_executed": 8,
        },
    ]
    event["prompt_tokens_scheduled"] = 16
    event["actual_prompt_tokens_locally_executed"] = 16
    event["cumulative_prompt_tokens_locally_executed"] = 16

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"
    assert event["recomputed_tokens"] == 8


def test_transfer_status_must_be_canonical_success_even_when_success_flag_is_true():
    event = _event(mode="hit")
    event["transfer_job_statuses"][0]["status"] = "failure"
    event["transfer_job_statuses"][0]["success"] = True

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_TRANSFER_FAILED_REASON


def test_hit_pending_is_allowed_as_a_transient_subset_of_hits():
    event = _event(mode="hit")
    event["lookup_episodes"][0]["groups"]["0"]["pending"] = [[0, 0, 1]]

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


def test_cumulative_cold_to_stored_provenance_overlap_is_valid():
    event = _event(mode="hit", lookup_count=2)
    event["cold_keys"] = [[0, 0, 1]]
    event["prior_stored_keys"] = [[0, 0, 1]]

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


def test_cumulative_stored_to_evicted_provenance_overlap_preserves_pressure():
    event = _event(mode="miss", lookup_count=2)
    event["prior_stored_keys"] = [[0, 0, 1]]
    event["prior_evicted_keys"] = [[0, 0, 1]]

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"


def test_duplicate_transfer_job_ids_are_unavailable():
    event = _event(mode="hit")
    duplicate = dict(event["transfer_job_statuses"][0])
    event["transfer_job_statuses"].append(duplicate)
    event["transfer_job_counts"]["load"] = 2

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_TRANSFER_REASON


def _with_speculative_transfer(event, *, consumed: int, discarded: int):
    event["speculative_transfer_job_statuses"] = [
        {
            "job_id": 91,
            "direction": "load",
            "is_prefetch": True,
            "status": "success",
            "success": True,
            "cancelled": False,
            "contributors": 1,
            "expected_contributors": 1,
            "bytes": 128,
            "time_s": 0.001,
        }
    ]
    event["speculative_transfer_accounting"] = {
        "job_count": 1,
        "successful_job_count": 1,
        "failed_job_count": 0,
        "submitted_bytes": 128,
        "completed_bytes": 128,
        "consumed_bytes": consumed,
        "discarded_bytes": discarded,
        "released_reservation_bytes": discarded,
        "uncompleted_submitted_bytes": 0,
        "unconsumed_completed_bytes": 128 - consumed - discarded,
        "actual_contributors": 1,
        "expected_contributors": 1,
        "device_time_s": 0.001,
    }
    event["prefetch_counts"] = {
        "attempted": {"keys": 1, "blocks": 1},
        "consumed": {"keys": int(bool(consumed)), "blocks": int(bool(consumed))},
        "discarded": {
            "keys": int(bool(discarded)),
            "blocks": int(bool(discarded)),
        },
    }
    event["prefetch_attribution"] = {
        "provenance": {"config": {"transfer_bytes_per_key": 128}},
        "transfer_jobs": event["speculative_transfer_job_statuses"],
        "speculative_transfer": event["speculative_transfer_accounting"],
        "prefetch_h2d_bytes": 128,
    }
    return event


@pytest.mark.parametrize(
    ("consumed", "discarded"),
    [(128, 0), (0, 128)],
)
def test_speculative_completion_does_not_become_demand_reuse(
    consumed, discarded
):
    event = _with_speculative_transfer(
        _event(mode="miss"), consumed=consumed, discarded=discarded
    )

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"
    assert mapping["preservation_reused_tokens"] == 0
    details = json.loads(mapping["preservation_request_outcome_details_json"])
    assert details[0]["speculative_transfer_accounting"][
        "completed_bytes"
    ] == 128
    assert details[0]["speculative_transfer_accounting"][
        "consumed_bytes"
    ] == consumed
    assert details[0]["speculative_transfer_accounting"][
        "discarded_bytes"
    ] == discarded


def test_malformed_speculative_transfer_metadata_fails_closed():
    event = _with_speculative_transfer(
        _event(mode="miss"), consumed=0, discarded=128
    )
    event["speculative_transfer_accounting"]["completed_bytes"] = 127

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_TRANSFER_REASON


def test_failed_speculative_transfer_keeps_demand_mapping_and_costs_separate():
    event = _event(mode="miss")
    event["speculative_transfer_job_statuses"] = [
        {
            "job_id": 91,
            "direction": "load",
            "is_prefetch": True,
            "status": "cancelled",
            "success": False,
            "cancelled": True,
            "contributors": 0,
            "expected_contributors": 1,
            "bytes": None,
            "time_s": None,
        }
    ]
    event["speculative_transfer_accounting"] = {
        "job_count": 1,
        "successful_job_count": 0,
        "failed_job_count": 1,
        "submitted_bytes": 128,
        "completed_bytes": 0,
        "consumed_bytes": 0,
        "discarded_bytes": 0,
        "released_reservation_bytes": 128,
        "uncompleted_submitted_bytes": 128,
        "unconsumed_completed_bytes": 0,
        "actual_contributors": 0,
        "expected_contributors": 1,
        "device_time_s": 0.0,
    }
    event["prefetch_counts"] = {
        "attempted": {"keys": 1, "blocks": 1},
        "consumed": {"keys": 0, "blocks": 0},
        "discarded": {"keys": 1, "blocks": 1},
    }
    event["prefetch_attribution"] = {
        "provenance": {"config": {"transfer_bytes_per_key": 128}},
        "transfer_jobs": event["speculative_transfer_job_statuses"],
        "speculative_transfer": event["speculative_transfer_accounting"],
        "prefetch_h2d_bytes": None,
    }

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"
    assert mapping["preservation_reused_tokens"] == 0


def test_speculative_job_in_demand_channel_fails_closed():
    event = _event(mode="hit")
    event["transfer_job_statuses"][0]["is_prefetch"] = True

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_TRANSFER_REASON


def _with_consumed_prefetch_external_block(event):
    """Replace the fixture's demand load with a proven consumed GPU splice."""
    event["transfer_job_statuses"] = []
    event["transfer_job_counts"]["load"] = 0
    event["physical_block_ids"]["loaded"] = []
    event["loaded_kv_blocks"] = 0
    event["load_bytes"] = 0
    event["tp_transfer_contributors"]["load"] = 0
    event["tp_transfer_contributors"]["expected"]["load"] = 0
    event["tp_transfer_contributor_totals"]["actual"]["load"] = 0
    event["tp_transfer_contributor_totals"]["expected"]["load"] = 0
    _with_speculative_transfer(event, consumed=128, discarded=0)
    key = "exact-prefetched-key"
    attribution = event["prefetch_attribution"]
    attribution.update(
        {
            "status": "hint_consumed",
            "required_keys": [key],
            "intersection_keys": [key],
            "source_key_to_gpu_block": {key: "91"},
            "destination_key_to_gpu_block": {key: "100"},
            "cpu_residency_proof": {key: "manager_lookup_hit"},
            "splice_acknowledged": True,
            "splice_ack_id": "splice-7",
            "splice_fence_ns": 20,
            "splice_fence_evidence": [
                {
                    "completion_event_satisfied": True,
                    "copied_on_cuda": True,
                    "same_default_stream": True,
                    "issued_before_base_bind_and_forward": True,
                    "destination_block_count": 1,
                    "source_block_count": 1,
                }
            ],
        }
    )
    speculative_job = event["speculative_transfer_job_statuses"][0]
    speculative_job["keys"] = [key]
    speculative_job["destination_block_ids"] = [[0, 91]]
    return _with_physical_block_table(event)


def test_consumed_prefetch_destination_satisfies_external_block_without_demand_load():
    event = _with_consumed_prefetch_external_block(_event(mode="hit"))

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"
    assert mapping["preservation_loaded_blocks"] == 1
    details = json.loads(mapping["preservation_request_outcome_details_json"])
    assert details[0]["demand_loaded_blocks"] == 0
    assert details[0]["speculatively_loaded_blocks"] == 1


def test_request_timing_cuda_attribution_and_staging_protection_are_retained():
    event = _event(mode="hit")
    event["server_timing"] = {
        "schema": "semantic-server-request-timing-v1",
        "clock_domain": "host_boot_monotonic_ns",
        "arrival_wall_clock_domain": "unix_epoch_ns",
        "arrival_aligned_monotonic_ns": 100,
        "arrival_alignment_uncertainty_ns": 2,
        "service_start_ns": 120,
        "first_forward_ns": 150,
        "completion_ns": 200,
        "ordering_valid": True,
        "complete": True,
        "queue_delay_ns": 20,
        "service_to_first_forward_ns": 30,
        "service_time_ns": 80,
        "completion_after_first_forward_ns": 50,
    }
    event["prefill_cuda_events"] = [
        {
            "schema": "semantic-bounded-prefill-cuda-v1",
            "request_ids": [event["request_id"]],
            "execution_attribution": [
                {
                    "request_id": event["request_id"],
                    "scheduled_tokens": 16,
                    "prompt_tokens": 16,
                }
            ],
            "started_at_ns": 130,
            "completed_at_ns": 145,
            "device_time_s": 0.001,
            "transfer_free": True,
            "singleton_request": True,
            "all_scheduled_tokens_are_prompt": True,
            "attribution_eligible": True,
        }
    ]
    event["staging_protection_records"] = [
        {
            "schema": "semantic-staging-protection-v1",
            "action": "protect",
            "outcome": "accepted",
        }
    ]

    mapping = map_preservation_events([event])

    details = json.loads(mapping["preservation_request_outcome_details_json"])
    assert details[0]["server_timing"]["queue_delay_ns"] == 20
    assert details[0]["prefill_cuda_events"][0]["attribution_eligible"] is True
    assert details[0]["staging_protection_records"][0]["action"] == "protect"

    event["server_timing"]["service_time_ns"] = 79
    assert map_preservation_events([event])["preservation_outcome"] == "unavailable"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda event: event["prefetch_attribution"].update(
            {"destination_key_to_gpu_block": {"exact-prefetched-key": "999"}}
        ),
        lambda event: event["prefetch_attribution"].update(
            {"splice_acknowledged": False}
        ),
        lambda event: event["prefetch_attribution"].update(
            {"splice_fence_evidence": []}
        ),
        lambda event: event["physical_block_table_provenance"].update(
            {"complete": False}
        ),
    ],
)
def test_unproven_speculative_external_block_fails_closed(mutator):
    event = _with_consumed_prefetch_external_block(_event(mode="hit"))
    mutator(event)

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def _with_physical_block_table(event):
    physical = event["physical_block_ids"]
    block_ids = sorted(
        {
            ref[2]
            for field in (
                "required",
                "local",
                "external",
                "recomputed",
                "scheduled",
                "loaded",
                "stored",
            )
            for ref in physical[field]
        }
    )
    table = [block_ids]
    table_digest = hashlib.sha256(
        json.dumps(table, separators=(",", ":")).encode()
    ).hexdigest()
    lifetime_observed_at_ns = 1
    event["connector_observed_ns"] = lifetime_observed_at_ns
    lifetime_id = hashlib.sha256(
        json.dumps(
            [
                event["request_id"],
                event["external_request_id"],
                event.get("generation"),
                lifetime_observed_at_ns,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    event["admission_episodes"][0]["scheduler_batch_id"] = 0
    event["physical_block_table_provenance"] = {
        "schema": "semantic-physical-block-table-v1",
        "request_id": event["request_id"],
        "external_request_id": event["external_request_id"],
        "lifetime_id": lifetime_id,
        "lifetime_observed_at_ns": lifetime_observed_at_ns,
        "request_generation": event.get("generation"),
        "table_generation": 0,
        "tokens_per_block": [16],
        "complete": True,
        "group_block_ids": table,
        "table_digest": table_digest,
        "updates": [
            {
                "mode": "scheduled_new_full",
                "scheduler_batch_id": 0,
                "episode_id": 0,
                "table_generation": 0,
                "start_tokens": 0,
                "scheduled_tokens": event["admission_episodes"][0][
                    "scheduled_tokens"
                ],
                "scheduled_prompt_tokens": event["admission_episodes"][0][
                    "prompt_tokens_locally_executed"
                ],
                "tokens_per_block": [16],
                "partial_tail_tokens": [0],
                "scheduled_block_ids": physical["scheduled"],
                "group_lengths_before": None,
                "group_lengths_after": [len(block_ids)],
                "group_block_ids": table,
                "table_digest": table_digest,
                "observed_at_ns": 2,
                "sequence": 1,
            }
        ],
        "errors": [],
    }
    return event


def test_physical_block_table_provenance_accepts_exact_request_and_transfer_refs():
    event = _with_physical_block_table(_event(mode="hit"))

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


def test_resumed_final_table_preserves_prior_episode_physical_provenance():
    event = _with_physical_block_table(_event(mode="hit"))
    provenance = event["physical_block_table_provenance"]
    replacement = [[8, 9]]
    replacement_digest = hashlib.sha256(
        json.dumps(replacement, separators=(",", ":")).encode()
    ).hexdigest()
    provenance["updates"].append(
        {
            "mode": "cached_resumed_replace",
            "scheduler_batch_id": 1,
            "episode_id": 1,
            "table_generation": 1,
            "start_tokens": 16,
            "scheduled_tokens": 0,
            "scheduled_prompt_tokens": 0,
            "tokens_per_block": [16],
            "partial_tail_tokens": [0],
            "scheduled_block_ids": [],
            "group_lengths_before": provenance["updates"][0][
                "group_lengths_after"
            ],
            "group_lengths_after": [2],
            "group_block_ids": replacement,
            "table_digest": replacement_digest,
            "observed_at_ns": 3,
            "sequence": 2,
        }
    )
    provenance["table_generation"] = 1
    provenance["group_block_ids"] = replacement
    provenance["table_digest"] = replacement_digest
    event["admission_episodes"].append(
        {
            "episode_id": 1,
            "lookup_id": 0,
            "scheduler_batch_id": 1,
            "scheduled_tokens": 0,
            "prompt_tokens_locally_executed": 0,
        }
    )
    event["admission_count"] = 2

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


def _append_table_update(event, mode="cached_no_new_blocks"):
    provenance = event["physical_block_table_provenance"]
    prior = provenance["updates"][-1]
    prior_groups = prior["group_block_ids"]
    if mode == "cached_append":
        groups = [[*prior_groups[0], 999]]
        episode_id = prior["episode_id"]
        generation = prior["table_generation"]
    elif mode == "cached_resumed_replace":
        groups = [[8, 9]]
        episode_id = prior["episode_id"] + 1
        generation = prior["table_generation"] + 1
    else:
        groups = [list(prior_groups[0])]
        episode_id = prior["episode_id"]
        generation = prior["table_generation"]
    digest = hashlib.sha256(
        json.dumps(groups, separators=(",", ":")).encode()
    ).hexdigest()
    update = {
        "mode": mode,
        "scheduler_batch_id": prior["scheduler_batch_id"] + 1,
        "episode_id": episode_id,
        "table_generation": generation,
        "start_tokens": 8,
        "scheduled_tokens": 0,
        "scheduled_prompt_tokens": 0,
        "tokens_per_block": [16],
        "partial_tail_tokens": [8],
        "scheduled_block_ids": [],
        "group_lengths_before": prior["group_lengths_after"],
        "group_lengths_after": [len(groups[0])],
        "group_block_ids": groups,
        "table_digest": digest,
        "observed_at_ns": prior["observed_at_ns"] + 1,
        "sequence": prior["sequence"] + 1,
    }
    provenance["updates"].append(update)
    provenance["table_generation"] = generation
    provenance["group_block_ids"] = groups
    provenance["table_digest"] = digest
    event["admission_episodes"].append(
        {
            "episode_id": episode_id,
            "lookup_id": 0,
            "scheduler_batch_id": update["scheduler_batch_id"],
            "scheduled_tokens": 0,
            "prompt_tokens_locally_executed": 0,
        }
    )
    event["admission_count"] += 1
    return update


@pytest.mark.parametrize("mode", ["cached_no_new_blocks", "cached_append"])
def test_physical_block_table_provenance_replays_valid_cached_transition(mode):
    event = _with_physical_block_table(_event(mode="hit"))
    _append_table_update(event, mode)

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


@pytest.mark.parametrize(
    "mode,mutation",
    [
        ("cached_no_new_blocks", "replacement"),
        ("cached_append", "replacement"),
        ("cached_no_new_blocks", "before"),
        ("cached_no_new_blocks", "generation"),
        ("cached_no_new_blocks", "partial_tail"),
        ("cached_no_new_blocks", "start_geometry"),
        ("cached_no_new_blocks", "clock"),
    ],
)
def test_physical_block_table_impossible_cached_transition_fails_closed(
    mode, mutation
):
    event = _with_physical_block_table(_event(mode="hit"))
    update = _append_table_update(event, mode)
    provenance = event["physical_block_table_provenance"]
    if mutation == "replacement":
        replacement = [[8, 9]]
        digest = hashlib.sha256(
            json.dumps(replacement, separators=(",", ":")).encode()
        ).hexdigest()
        update["group_block_ids"] = replacement
        update["group_lengths_after"] = [2]
        update["table_digest"] = digest
        provenance["group_block_ids"] = replacement
        provenance["table_digest"] = digest
    elif mutation == "before":
        update["group_lengths_before"] = [999]
    elif mutation == "generation":
        update["table_generation"] = 9
        provenance["table_generation"] = 9
    elif mutation == "partial_tail":
        update["partial_tail_tokens"] = [9]
    elif mutation == "start_geometry":
        update["start_tokens"] = 999999
        update["partial_tail_tokens"] = [15]
    elif mutation == "clock":
        update["observed_at_ns"] = 0

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def _two_batch_recomputed_event():
    event = _with_physical_block_table(_event(mode="miss"))
    event["admission_count"] = 2
    event["admission_episodes"] = [
        {
            "episode_id": 0,
            "lookup_id": 0,
            "scheduler_batch_id": batch_id,
            "scheduled_tokens": 4,
            "prompt_tokens_locally_executed": 4,
        }
        for batch_id in range(2)
    ]
    provenance = event["physical_block_table_provenance"]
    provenance["tokens_per_block"] = [4]
    tables = [[[200]], [[200, 201]]]
    updates = []
    for batch_id, groups in enumerate(tables):
        digest = hashlib.sha256(
            json.dumps(groups, separators=(",", ":")).encode()
        ).hexdigest()
        updates.append(
            {
                "mode": "scheduled_new_full" if batch_id == 0 else "cached_append",
                "scheduler_batch_id": batch_id,
                "episode_id": 0,
                "table_generation": 0,
                "start_tokens": batch_id * 4,
                "scheduled_tokens": 4,
                "scheduled_prompt_tokens": 4,
                "tokens_per_block": [4],
                "partial_tail_tokens": [0],
                "scheduled_block_ids": [[0, 0, 200 + batch_id]],
                "group_lengths_before": None if batch_id == 0 else [1],
                "group_lengths_after": [len(groups[0])],
                "group_block_ids": groups,
                "table_digest": digest,
                "observed_at_ns": batch_id + 2,
                "sequence": batch_id + 1,
            }
        )
    provenance["updates"] = updates
    provenance["group_block_ids"] = tables[-1]
    provenance["table_digest"] = updates[-1]["table_digest"]
    return event


def test_physical_block_table_replays_each_scheduled_range_contemporaneously():
    event = _two_batch_recomputed_event()

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "miss"

    event["physical_block_table_provenance"]["updates"][0][
        "scheduled_block_ids"
    ] = [[0, 0, 201]]
    event["physical_block_table_provenance"]["updates"][1][
        "scheduled_block_ids"
    ] = [[0, 0, 200]]
    mapping = map_preservation_events([event])
    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def test_physical_block_table_scheduled_range_beyond_table_fails_closed():
    event = _two_batch_recomputed_event()
    update = event["physical_block_table_provenance"]["updates"][1]
    update["start_tokens"] = 8

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


@pytest.mark.parametrize(
    "event_observation",
    [None, True, -1, "1", 1000],
)
def test_physical_block_table_requires_exact_event_lifetime_observation(
    event_observation
):
    event = _with_physical_block_table(_event(mode="hit"))
    if event_observation is None:
        event.pop("connector_observed_ns")
    else:
        event["connector_observed_ns"] = event_observation

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def test_rehashed_provenance_lifetime_cannot_disagree_with_event_observation():
    event = _with_physical_block_table(_event(mode="hit"))
    provenance = event["physical_block_table_provenance"]
    provenance["lifetime_observed_at_ns"] = 0
    provenance["lifetime_id"] = hashlib.sha256(
        json.dumps(
            [
                event["request_id"],
                event["external_request_id"],
                event.get("generation"),
                0,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


def test_legacy_event_without_table_provenance_needs_no_lifetime_observation():
    event = _event(mode="hit")

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "hit"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda provenance: provenance.update({"table_digest": "0" * 64}),
        lambda provenance: provenance.update(
            {"lifetime_observed_at_ns": provenance["lifetime_observed_at_ns"] + 1}
        ),
        lambda provenance: provenance.update({"group_block_ids": [[]]}),
        lambda provenance: provenance["updates"][0].update(
            {"mode": "cached_append", "group_lengths_before": [0]}
        ),
        lambda provenance: provenance.update(
            {"complete": False, "errors": [{"reason": "missing"}]}
        ),
    ],
)
def test_physical_block_table_provenance_malformed_paths_fail_closed(mutator):
    event = _with_physical_block_table(_event(mode="hit"))
    mutator(event["physical_block_table_provenance"])

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == REQUEST_EVENT_PHYSICAL_REASON


@pytest.mark.parametrize(
    "mutator",
    [
        lambda event: event["lookup_episodes"][0]["groups"]["0"].update(
            {"missing": [[0, 0, 999]]}
        ),
        lambda event: event["lookup_episodes"][0]["groups"]["0"].update(
            {"missing": [[0, 0, 1]]}
        ),
    ],
)
def test_unrelated_or_conflicting_lookup_sets_are_unavailable(mutator):
    event = _event(mode="hit")
    mutator(event)

    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == "request_episode_metadata_incomplete"


def test_multiple_episodes_are_counted_but_not_double_counted():
    event = _event(mode="partial", lookup_count=2, admission_count=2)
    event["admission_episodes"][1]["episode_id"] = event["admission_episodes"][0][
        "episode_id"
    ]
    event["admission_episodes"][1]["scheduled_tokens"] = 0
    event["admission_episodes"][1]["prompt_tokens_locally_executed"] = 0
    mapping = map_preservation_events([event])
    details = json.loads(mapping["preservation_request_outcome_details_json"])

    assert mapping["preservation_required_tokens"] == 8
    assert mapping["preservation_recomputed_tokens"] == 4
    assert details[0]["lookup_episode_count"] == 2
    assert details[0]["admission_episode_count"] == 1


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (
            lambda event: event.update(
                {
                    "transfer_metadata_complete": False,
                    "transfer_job_statuses": ["cancelled"],
                }
            ),
            REQUEST_EVENT_TRANSFER_FAILED_REASON,
        ),
        (
            lambda event: event.pop("transfer_job_counts"),
            REQUEST_EVENT_TRANSFER_REASON,
        ),
        (
            lambda event: event.pop("tp_transfer_contributors"),
            REQUEST_EVENT_TP_REASON,
        ),
        (
            lambda event: event.update({"tp_contributors_complete": False}),
            REQUEST_EVENT_TP_REASON,
        ),
        (
            lambda event: event.update({"required_tokens": 7}),
            REQUEST_EVENT_ACCOUNTING_REASON,
        ),
        (
            lambda event: event.update({"physical_block_identity_complete": False}),
            REQUEST_EVENT_PHYSICAL_REASON,
        ),
        (
            lambda event: event["physical_block_ids"].update(
                {"scheduled": [[0, 0, 999]]}
            ),
            REQUEST_EVENT_PHYSICAL_REASON,
        ),
        (
            lambda event: event["physical_block_ids"].update({"scheduled": [[0, 1]]}),
            REQUEST_EVENT_PHYSICAL_REASON,
        ),
        (
            lambda event: event["physical_block_ids"].update({"required": [[0, 1]]}),
            REQUEST_EVENT_PHYSICAL_REASON,
        ),
    ],
)
def test_incomplete_event_metadata_fails_closed(mutator, reason):
    event = _event()
    mutator(event)
    mapping = map_preservation_events([event])

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == reason
    assert mapping["preservation_required_tokens"] is None


@pytest.mark.parametrize(
    "events,expected,reason",
    [
        ([_event(0), _event(0)], None, REQUEST_EVENT_DUPLICATE_REASON),
        (
            [_event(0)],
            [
                "sc|run-1|cell-1|chat|sample-0",
                "sc|run-1|cell-1|chat|sample-1",
            ],
            REQUEST_EVENT_MISSING_REASON,
        ),
        (
            [_event(0, run_id="other")],
            ["sc|run-1|cell-1|chat|sample-0"],
            REQUEST_EVENT_FOREIGN_REASON,
        ),
    ],
)
def test_identity_cardinality_errors_never_become_cache_labels(
    events, expected, reason
):
    mapping = map_preservation_events(
        events, expected_request_ids=expected, run_id="run-1", cell_id="cell-1"
    )

    assert mapping["preservation_outcome"] == "unavailable"
    assert mapping["preservation_reason"] == reason


def test_preemption_and_eviction_do_not_turn_authoritative_partial_into_unavailable():
    event = _event(mode="partial", preempted=True)
    mapping = map_preservation_events([event])
    detail = json.loads(mapping["preservation_request_outcome_details_json"])[0]

    assert mapping["preservation_outcome"] == "partial"
    assert detail["preempted"] is True
    assert detail["preemption_count"] == 1


def test_archived_projection_rejects_rebound_foreign_event_list():
    event = _event()
    mapping = map_preservation_events([event])
    row = _mapped_row(mapping, [event])

    foreign_event = _event(run_id="foreign-run", cell_id="foreign-cell")
    foreign_mapping = map_preservation_events([foreign_event])
    row.update(foreign_mapping)
    row["request_ids_json"] = foreign_mapping["preservation_request_ids_json"]
    row["preservation_events_json"] = json.dumps([foreign_event])

    errors = validate_preservation_row(row, require_fields=True)

    assert any("foreign to the archived row" in error for error in errors)
    assert any("request_id_prefix" in error for error in errors)


def test_archived_projection_rejects_same_id_fabricated_hit():
    raw_event = _event(mode="miss")
    fabricated_hit = map_preservation_events([_event(mode="hit")])
    row = _mapped_row(fabricated_hit, [raw_event])

    errors = validate_preservation_row(row, require_fields=True)

    assert any(
        "preservation_outcome does not match recomputed event projection" in error
        for error in errors
    )


def test_archived_projection_requires_canonical_request_id_prefix():
    event = _event()
    mapping = map_preservation_events([event])
    row = _mapped_row(mapping, [event])
    row["request_id_prefix"] = ""

    errors = validate_preservation_row(row, require_fields=True)

    assert any("canonical request_id_prefix" in error for error in errors)


@pytest.mark.parametrize(
    "prefix",
    [
        "sc|foreign-run|cell-1|chat|sample-",
        "sc|run-1|foreign-cell|chat|sample-",
        "sc|run-1|cell-1|rag|sample-",
    ],
)
def test_archived_projection_rejects_foreign_prefix_identity(prefix):
    event = _event()
    mapping = map_preservation_events([event])
    row = _mapped_row(mapping, [event])
    row["request_id_prefix"] = prefix

    errors = validate_preservation_row(row, require_fields=True)

    assert any("bound to row run/cell/workload identity" in error for error in errors)


def test_join_wires_event_projection_into_chat_row(tmp_path):
    event = _event()
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) SEMANTIC_PRESERVATION_EVENT " + json.dumps(event) + "\n"
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=1,
    )

    assert joined["preservation_outcome"] == "hit"
    assert joined["preservation_join_status"] == "complete"
    assert joined["preservation_loaded_tokens"] == 8


def test_join_reorders_chronological_events_to_shuffled_rag_request_ids(tmp_path):
    events = [_event(0), _event(1)]
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            "(EngineCore pid=1) SEMANTIC_PRESERVATION_EVENT " + json.dumps(event)
            for event in events
        )
        + "\n"
    )
    request_ids = [event["external_request_id"] for event in reversed(events)]
    row = {"request_ids_json": json.dumps(request_ids)}

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=2,
    )

    assert [
        event["external_request_id"]
        for event in json.loads(joined["preservation_events_json"])
    ] == request_ids
    assert [
        detail["request_id"]
        for detail in json.loads(joined["preservation_request_outcome_details_json"])
    ] == request_ids


def test_join_parses_two_marker_events_on_one_physical_log_line(tmp_path):
    events = [_event(0), _event(1)]
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(events[0])
        + marker
        + json.dumps(events[1])
        + "\n"
    )
    row = {
        "request_ids_json": json.dumps(
            [event["external_request_id"] for event in events]
        )
    }

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=2,
    )

    assert joined["preservation_event_count"] == 2
    assert joined["preservation_outcome"] == "hit"


def test_join_accepts_known_apiserver_metrics_interleaving(tmp_path):
    event = _event()
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(event)
        + '(APIServer pid=1202) INFO:     127.0.0.1:54026 - "GET /metrics '
        + 'HTTP/1.1" 200 OK\n'
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=1,
    )

    assert joined["preservation_event_count"] == 1
    assert joined["preservation_outcome"] == "hit"


def test_join_rejects_unknown_trailing_logger_text(tmp_path):
    event = _event()
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(event)
        + '(APIServer pid=1202) INFO: 127.0.0.1:54026 - "GET /health '
        + 'HTTP/1.1" 200 OK\n'
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    with pytest.raises(ValueError, match="non-whitespace text after JSON object"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


def test_join_rejects_unknown_original_log_prefix(tmp_path):
    event = _event()
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text("unexpected-prefix " + marker + json.dumps(event) + "\n")
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    with pytest.raises(ValueError, match="unknown preservation log prefix"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


def test_join_rejects_unmarked_adjacent_json_objects(tmp_path):
    events = [_event(0), _event(1)]
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(events[0])
        + json.dumps(events[1])
        + "\n"
    )
    row = {
        "request_ids_json": json.dumps(
            [event["external_request_id"] for event in events]
        )
    }

    with pytest.raises(ValueError, match="malformed preservation event"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=2,
        )


def test_join_rejects_empty_marker_segment(tmp_path):
    event = _event()
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) " + marker + marker + json.dumps(event) + "\n"
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    with pytest.raises(ValueError, match="empty JSON segment"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


def test_join_accepts_explicit_unavailable_event_without_claiming_label(tmp_path):
    event = _event()
    event.update(
        {
            "authoritative": False,
            "event_integrity": "unavailable",
            "outcome": "unavailable",
            "reason": "lookup_metadata_unavailable",
            "lookup_metadata_complete": False,
            "lookup_observed": False,
        }
    )
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) SEMANTIC_PRESERVATION_EVENT " + json.dumps(event) + "\n"
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=1,
    )

    assert joined["preservation_outcome"] == "unavailable"
    assert joined["preservation_reason"] == "lookup_metadata_unavailable"


def test_join_collapses_one_authoritative_and_one_unavailable_duplicate_event(
    tmp_path,
):
    authoritative = _event()
    unavailable = dict(authoritative)
    unavailable["request_id"] = "cmpl-sc|run-1|cell-1|chat|sample-0-1-feedface"
    unavailable.update(
        {
            "authoritative": False,
            "event_integrity": "unavailable",
            "outcome": "unavailable",
            "reason": "lookup_metadata_unavailable",
            "lookup_metadata_complete": False,
            "lookup_observed": False,
        }
    )
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(authoritative)
        + marker
        + json.dumps(unavailable)
        + "\n"
    )
    row = {"request_ids_json": json.dumps([authoritative["external_request_id"]])}

    joined = join_preservation_events(
        row,
        log_path=str(log_path),
        request_id_prefix="sc|run-1|cell-1|chat|sample-",
        run_id="run-1",
        cell_id="cell-1",
        expected_count=1,
    )

    assert joined["preservation_outcome"] == "unavailable"
    assert joined["preservation_reason"] == "duplicate_external_request_events"


def test_join_rejects_authoritative_and_unavailable_same_internal_id(tmp_path):
    authoritative = _event()
    unavailable = dict(authoritative)
    unavailable.update(
        {
            "authoritative": False,
            "event_integrity": "unavailable",
            "outcome": "unavailable",
            "reason": "lookup_metadata_unavailable",
            "lookup_metadata_complete": False,
            "lookup_observed": False,
        }
    )
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(authoritative)
        + marker
        + json.dumps(unavailable)
        + "\n"
    )
    row = {"request_ids_json": json.dumps([authoritative["external_request_id"]])}

    with pytest.raises(ValueError, match="repeated internal request_id"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


def test_join_rejects_three_event_duplicate_history(tmp_path):
    authoritative = _event()
    unavailable = dict(authoritative)
    unavailable["request_id"] = "cmpl-sc|run-1|cell-1|chat|sample-0-1-feedface"
    unavailable.update(
        {
            "authoritative": False,
            "event_integrity": "unavailable",
            "outcome": "unavailable",
            "reason": "lookup_metadata_unavailable",
            "lookup_metadata_complete": False,
            "lookup_observed": False,
        }
    )
    marker = "SEMANTIC_PRESERVATION_EVENT "
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) "
        + marker
        + json.dumps(authoritative)
        + marker
        + json.dumps(unavailable)
        + marker
        + json.dumps(authoritative)
        + "\n"
    )
    row = {"request_ids_json": json.dumps([authoritative["external_request_id"]])}

    with pytest.raises(ValueError, match="duplicate preservation events.*count=3"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


@pytest.mark.parametrize(
    "update",
    [
        {"event_integrity": "complete"},
        {"terminal_status": "aborted"},
        {"terminal_status": "finished_error"},
        {"terminal_status": "finished_ignored"},
        {"reason": "fabricated_reason"},
        {"reason": None},
        {"tp_contributors_complete": False, "reason": None},
        {
            "transfer_metadata_complete": False,
            "tp_contributors_complete": True,
            "reason": "transfer_metadata_unavailable",
        },
    ],
)
def test_join_rejects_loosely_marked_unavailable_events(tmp_path, update):
    event = _event()
    event.update(
        {
            "authoritative": False,
            "event_integrity": "unavailable",
            "outcome": "unavailable",
            "reason": "lookup_metadata_unavailable",
        }
    )
    event.update(update)
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "(EngineCore pid=1) SEMANTIC_PRESERVATION_EVENT " + json.dumps(event) + "\n"
    )
    row = {"request_ids_json": json.dumps([event["external_request_id"]])}

    with pytest.raises(ValueError, match="non-authoritative or non-success"):
        join_preservation_events(
            row,
            log_path=str(log_path),
            request_id_prefix="sc|run-1|cell-1|chat|sample-",
            run_id="run-1",
            cell_id="cell-1",
            expected_count=1,
        )


def test_unavailable_projection_normalizes_all_request_details():
    invalid = _event(mode="hit")
    invalid["physical_block_identity_complete"] = False
    valid = _event(mode="hit", sample_index=1)
    mapping = map_preservation_events([invalid, valid])
    row = _mapped_row(mapping, [invalid, valid])

    details = json.loads(mapping["preservation_request_outcome_details_json"])

    assert mapping["preservation_outcome"] == "unavailable"
    assert all(detail["outcome"] == "unavailable" for detail in details)
    assert validate_preservation_row(row, require_fields=True) == []
