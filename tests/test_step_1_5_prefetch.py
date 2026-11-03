# SPDX-License-Identifier: Apache-2.0
"""Step 1.5 (semantic prefetch) acceptance check: the pure-logic pieces --
top-K relevance selection (manager), the GPU-to-GPU splice primitive
(worker), and the retry-bookkeeping loop (issues log entry #25) that lets a
preempted request whose first reservation attempt fails (no free GPU blocks
right then) get retried on later scheduling steps instead of being given up
on permanently. The rest of the scheduler-side integration
(on_request_preempted's real vLLM req_status wiring, the real block-
reservation/dispatch machinery) is integration-heavy against vLLM's live
scheduler internals and is verified end-to-end on the real server instead
(issues log entries #23-#24), the same split this project used for Step
1.2/1.3's hook-point mechanics.
"""

import asyncio
import time
from collections import OrderedDict, defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    RequestGroupState,
    SchedulerOffloadConfig,
    TransferJobStatus,
)
from vllm.v1.kv_offload.base import (
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus

from semantic_offload.connector import (
    SemanticOffloadingConnectorScheduler,
    SemanticOffloadingConnectorWorker,
    SemanticTransferCompletion,
    SemanticWorkerMetadata,
    _append_prefetch_attempt,
    _event_key,
    _observed_prefix_digest,
    _PrefetchState,
    _reconcile_episode_prompt_cursor_after_admission,
    _reset_prompt_accounting_after_preemption,
    _select_prefetch_keys,
    _speculative_transfer_accounting,
    _update_prefetch_attempt_plan,
)
from semantic_offload.manager import SemanticOffloadingManager
from semantic_offload.wait_hint import WAIT_HINT_PROVENANCE, WAIT_HINT_SCHEMA
from semantic_offload.worker import SemanticOffloadingWorker


def to_key(n: int):
    return make_offload_key(str(n).encode(), 0)


def test_top_relevant_keys_picks_highest_scored_within_budget():
    manager = SemanticOffloadingManager(num_blocks=10)
    ema = manager.relevance_ema.setdefault("mean", {})
    keys = [to_key(i) for i in range(5)]
    for i, key in enumerate(keys):
        ema[key] = float(i)  # key i has score i -- key 4 highest

    top = manager.top_relevant_keys(keys, k=2, method="mean")

    assert top == [keys[4], keys[3]]


def test_top_relevant_keys_skips_unscored_candidates():
    manager = SemanticOffloadingManager(num_blocks=10)
    ema = manager.relevance_ema.setdefault("mean", {})
    scored, unscored = to_key(1), to_key(2)
    ema[scored] = 5.0
    # `unscored` deliberately has no entry -- a preempted request's block
    # may not have earned a score yet, and there's nothing to prefer.

    top = manager.top_relevant_keys([scored, unscored], k=5, method="mean")

    assert top == [scored]


def test_top_relevant_keys_budget_zero_returns_nothing():
    manager = SemanticOffloadingManager(num_blocks=10)
    manager.relevance_ema.setdefault("mean", {})[to_key(1)] = 1.0

    assert manager.top_relevant_keys([to_key(1)], k=0, method="minmax") == []


def test_request_order_prefetch_selector_avoids_semantic_scores():
    keys = [to_key(i) for i in range(4)]

    class NoSemanticManager:
        def top_relevant_keys(self, *_args):
            raise AssertionError("request-order selector must not score")

    assert (
        _select_prefetch_keys(NoSemanticManager(), keys, 2, "request_order") == keys[:2]
    )


def test_prefetch_selector_rejects_unknown_selector():
    with pytest.raises(ValueError, match="Unknown prefetch selector"):
        _select_prefetch_keys(object(), [to_key(1)], 1, "unknown")


def test_native149_scheduler_projection_constructs_without_worker_byte_field(
    monkeypatch,
):
    group = GroupOffloadConfig(
        group_idx=0,
        tokens_per_block=16,
        tokens_per_chunk=16,
        hashes_per_chunk=1,
        kv_event_group_spec=None,
        sliding_window_size_in_chunks=None,
    )
    native_config = SchedulerOffloadConfig(
        kv_group_configs=(group,),
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
    )
    assert not hasattr(native_config, "worker_kv_bytes_per_block")

    manager = object()
    spec = SimpleNamespace(
        extra_config={},
        replicated_layout=False,
        cpu_page_size_per_worker=458752,
        kv_bytes_per_chunk=458752,
    )

    def construct_base(scheduler, *_args):
        scheduler.config = native_config
        scheduler.manager = manager

    monkeypatch.setattr(
        "semantic_offload.connector.construct_scheduler_base", construct_base
    )
    scheduler = SemanticOffloadingConnectorScheduler(
        spec,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model"),
            cache_config=SimpleNamespace(block_size=16),
        ),
        SimpleNamespace(),
    )

    assert scheduler._prefetch_trigger == "preemption"
    assert scheduler._wait_hint_native_adoption is False
    assert scheduler._wait_hint_coalesce_exact_source is False
    assert scheduler._prefetch_require_full_prefix is False
    assert scheduler._wait_hint_coalescing_min_sources == 1
    assert scheduler._wait_hint_coalescing_max_ns == 0
    assert scheduler._prefetch_bytes_per_key == 458752
    assert scheduler._prefetch_allocation_bytes_per_key == 458752
    assert scheduler._prefetch_provenance["config"]["transfer_bytes_per_key"] == 458752

    spec.extra_config = {
        "wait_hint_coalescing_min_sources": 2,
        "wait_hint_coalescing_max_ms": 25,
        "wait_hint_coalesce_exact_source": True,
        "prefetch_require_full_prefix": True,
    }
    scheduler = SemanticOffloadingConnectorScheduler(
        spec,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model"),
            cache_config=SimpleNamespace(block_size=16),
        ),
        SimpleNamespace(),
    )
    assert scheduler._wait_hint_coalescing_min_sources == 2
    assert scheduler._wait_hint_coalescing_max_ns == 25_000_000
    assert scheduler._wait_hint_coalesce_exact_source is True
    assert scheduler._prefetch_require_full_prefix is True

    spec.extra_config = {"wait_hint_coalescing_min_sources": 2}
    with pytest.raises(ValueError, match="max_ms must be positive"):
        SemanticOffloadingConnectorScheduler(
            spec,
            SimpleNamespace(
                model_config=SimpleNamespace(model="test-model"),
                cache_config=SimpleNamespace(block_size=16),
            ),
            SimpleNamespace(),
        )


def _make_layer(num_blocks, num_kv_heads, block_size, head_size, seed):
    torch.manual_seed(seed)
    kv_cache = torch.randn(num_blocks, num_kv_heads, block_size, 2 * head_size)
    return SimpleNamespace(kv_cache=kv_cache)


def _make_worker(layers: dict) -> SemanticOffloadingWorker:
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._attention_layers = layers
    return worker


def test_connector_worker_load_errors_disabled_is_empty_set():
    from semantic_offload.connector import SemanticOffloadingConnectorWorker

    connector_worker = SemanticOffloadingConnectorWorker.__new__(
        SemanticOffloadingConnectorWorker
    )
    connector_worker.worker = SimpleNamespace(_preservation_events=False)

    assert connector_worker.get_block_ids_with_load_errors() == set()


def test_connector_worker_load_errors_enabled_uses_worker_metadata():
    from semantic_offload.connector import SemanticOffloadingConnectorWorker

    connector_worker = SemanticOffloadingConnectorWorker.__new__(
        SemanticOffloadingConnectorWorker
    )
    connector_worker.worker = SimpleNamespace(
        _preservation_events=True,
        pop_failed_load_block_ids=lambda: {7, 11},
    )

    assert connector_worker.get_block_ids_with_load_errors() == {7, 11}


def test_decode_preemption_preserves_completed_prompt_cursor():
    state = {
        "required_tokens": 100,
        "episode_id": 3,
        "episode_prompt_accounted": 100,
        "episode_local_reused_tokens": 100,
        "episode_external_tokens": 0,
        "readmission_pending": False,
    }

    _reset_prompt_accounting_after_preemption(state)

    assert state == {
        "required_tokens": 100,
        "episode_id": 4,
        "episode_prompt_accounted": 100,
        "episode_local_reused_tokens": 100,
        "episode_external_tokens": 0,
        "readmission_pending": True,
    }


def test_prompt_preemption_resets_incomplete_prompt_cursor_for_replay():
    state = {
        "required_tokens": 100,
        "episode_id": 3,
        "episode_prompt_accounted": 40,
        "episode_local_reused_tokens": 40,
        "episode_external_tokens": 0,
        "readmission_pending": False,
    }

    _reset_prompt_accounting_after_preemption(state)

    assert state == {
        "required_tokens": 100,
        "episode_id": 4,
        "episode_prompt_accounted": 0,
        "episode_local_reused_tokens": 0,
        "episode_external_tokens": 0,
        "readmission_pending": True,
    }


@pytest.mark.parametrize(
    ("local_tokens", "external_tokens", "expected_cursor"),
    [(0, 100, 100), (0, 0, 0), (20, 30, 50)],
)
def test_readmission_restored_prefix_reconciles_completed_cursor(
    local_tokens, external_tokens, expected_cursor
):
    state = {
        "required_tokens": 100,
        "episode_prompt_accounted": 100,
        "episode_local_reused_tokens": 100,
        "episode_external_tokens": 0,
    }

    _reconcile_episode_prompt_cursor_after_admission(
        state, local_tokens, external_tokens, first_readmission=True
    )

    assert state["episode_prompt_accounted"] == expected_cursor
    assert state["episode_local_reused_tokens"] == local_tokens


def test_external_restore_then_zero_promotion_does_not_lower_cursor():
    state = {
        "required_tokens": 100,
        "episode_prompt_accounted": 100,
        "episode_local_reused_tokens": 100,
        "episode_external_tokens": 0,
    }

    _reconcile_episode_prompt_cursor_after_admission(
        state, 0, 100, first_readmission=True
    )
    _reconcile_episode_prompt_cursor_after_admission(state, 0, 0)

    assert state["episode_prompt_accounted"] == 100
    assert state["episode_external_tokens"] == 100


@pytest.mark.parametrize(
    (
        "episode_external_tokens",
        "expected_reused",
        "expected_recomputed",
        "expected_outcome",
    ),
    [(100, 100, 0, "hit"), (0, 0, 100, "miss"), (20, 20, 80, "partial")],
)
def test_emitted_event_uses_current_episode_reuse_not_lifetime_local_reuse(
    episode_external_tokens, expected_reused, expected_recomputed, expected_outcome
):
    from unittest.mock import patch

    import semantic_offload.connector as connector_mod

    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._preservation_events_enabled = True
    scheduler._req_status = {}
    scheduler._preservation_requests = {}
    scheduler._preservation_finished = {}
    state = {
        "request_id": "cmpl-sc|run-1|cell-1|chat|sample-0-0-deadbeef",
        "external_request_id": "sc|run-1|cell-1|chat|sample-0",
        "identity_valid": True,
        "run_id": "run-1",
        "cell_id": "cell-1",
        "sample_id": "sc|run-1|cell-1|chat|sample-0",
        "sample_index": 0,
        "terminal_status": "finished_length_capped",
        "terminal_reason": "completed",
        "required_tokens": 100,
        "local_reused_tokens": 80,
        "episode_local_reused_tokens": 0,
        "admitted_local_tokens": 0,
        "episode_external_tokens": episode_external_tokens,
        "loaded_tokens_total": 100,
        "prompt_tokens_locally_executed": expected_recomputed,
        "prompt_tokens_scheduled": expected_recomputed,
        "lookup_observed": True,
        "lookup_no_eligible": False,
        "lookup_keys": [],
        "prior_stored_keys": [],
        "prior_evicted_keys": [("evicted",)],
        "cold_keys": [],
        "lookup_episodes": [
            {
                "lookup_id": 0,
                "groups": {
                    "0": {
                        "eligible": [],
                        "queried": [],
                        "hits": [],
                        "missing": [],
                        "pending": [],
                        "retry": [],
                        "observed": True,
                    }
                },
            }
        ],
        "transfer_complete": True,
        "execution_metadata_observed": True,
        "physical_block_identity_complete": True,
        "transfer_job_statuses": [],
        "load_job_count": 0,
        "store_job_count": 0,
        "load_contributors": 0,
        "store_contributors": 0,
        "load_expected_total": 0,
        "store_expected_total": 0,
        "load_expected_contributors": 1,
        "required_block_ids": set(),
        "local_block_ids": set(),
        "external_block_ids": set(),
        "recomputed_block_ids": set(),
        "scheduled_block_ids": set(),
        "loaded_block_ids": set(),
        "stored_block_ids": set(),
        "required_chunks": 0,
        "loaded_chunks": 0,
        "stored_chunks": 0,
        "load_bytes": 0,
        "store_bytes": 0,
        "load_time_s": 0.0,
        "store_time_s": 0.0,
        "load_critical_path_s": 0.0,
        "store_critical_path_s": 0.0,
        "load_time_observed": False,
        "store_time_observed": False,
        "load_time_complete": True,
        "store_time_complete": True,
        "admission_episodes": [],
        "admitted_external_tokens": episode_external_tokens,
        "admission_count": 1,
        "recompute_admission_count": 0,
        "preempted": True,
        "preemption_count": 1,
        "prefetch_attempts": 0,
        "prefetch_attempted_keys": 0,
        "prefetch_attempted_blocks": 0,
        "prefetch_consumed_keys": 0,
        "prefetch_consumed_blocks": 0,
        "prefetch_discarded_keys": 0,
        "prefetch_discarded_blocks": 0,
    }
    if episode_external_tokens == 0:
        state["prior_evicted_keys"] = []
        state["loaded_block_ids"] = {(0, 0, 1)}
        state["load_job_count"] = 1
    scheduler._preservation_finished = {state["request_id"]: state}

    with patch.object(connector_mod, "record_preservation_event") as record:
        scheduler._emit_finished_preservation_events()

    event = record.call_args.args[0]
    assert event["reused_tokens"] == expected_reused
    assert event["recomputed_tokens"] == expected_recomputed
    assert event["outcome"] == expected_outcome
    if episode_external_tokens == 0:
        assert event["pressure"] is True
    state["lookup_episodes"] = [
        {
            "lookup_id": 0,
            "groups": {
                "0": {
                    "eligible": [],
                    "queried": [],
                    "hits": [],
                    "missing": [],
                    "pending": [],
                    "retry": [],
                    "observed": True,
                }
            },
        },
        {"lookup_id": 1, "groups": {}},
    ]
    scheduler._preservation_finished = {state["request_id"]: state}
    with patch.object(connector_mod, "record_preservation_event") as record:
        scheduler._emit_finished_preservation_events()
    assert record.call_args.args[0]["lookup_metadata_complete"] is False
    assert record.call_args.args[0]["authoritative"] is False


def test_prefetch_attribution_event_emits_known_facts_and_explicit_gap():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    state = {
        "preempted": True,
        "run_id": "run-1",
        "cell_id": "cell-1",
        "external_request_id": "sc|run-1|cell-1|chat|sample-0",
        "episode_id": 1,
        "prefetch_t0_ns": 10,
        "prefetch_t1_ns": 20,
        "prefetch_t2_ns": 30,
        "prefetch_t3_ns": 40,
        "prefetch_fence_ns": 45,
        "prefetch_t4_ns": 50,
        "prefetch_t4_source": "scheduler_update_state_after_alloc_entry",
        "prefetch_t4_episode_id": 1,
        "prefetch_install_submitted_ns": 55,
        "prefetch_request_usable_ns": 70,
        "prefetch_request_usable_source": (
            "all_worker_splice_cuda_fences_acknowledged"
        ),
        "prefetch_required_keys": ["k0", "k1"],
        "prefetch_cpu_resident_keys": ["k0"],
        "prefetch_cpu_ready_keys": ["k0"],
        "prefetch_cpu_residency_proof": {"k0": "manager_lookup_hit"},
        "prefetch_selected_keys": ["k0"],
        "prefetch_intersection_keys": ["k0"],
        "prefetch_remainder_keys": ["k1"],
        "prefetch_source_map": {"k0": "101"},
        "prefetch_destination_map": {"k0": "7"},
        "prefetch_reservation_id": "prefetch-9",
        "prefetch_reservation_blocks": ["101"],
        "prefetch_reservation_history": [
            {
                "phase": "reserved",
                "action": "reserve",
                "reserved_blocks": 1,
                "free_blocks": 9,
            }
        ],
        "prefetch_h2d_bytes": 100,
        "prefetch_blocking_h2d_bytes": None,
        "prefetch_splice_bytes": None,
        "prefetch_transfer_contributors": {"9": ["0"]},
        "prefetch_source_reuse_before_fence": None,
        "prefetch_failure_reason": None,
        "prefetch_cancelled": False,
        "prefetch_stale": False,
        "prefetch_allocation_failures": 0,
        "prefetch_filler_failures": None,
        "prefetch_unavailable_fields": [],
        "prefetch_provenance": {"extension": "semantic_offload"},
        "prefetch_splice_acknowledged": False,
        "prefetch_splice_ack_id": "splice-2",
        "prefetch_fence_complete": True,
        "prefetch_time_s": 0.3,
        "prefetch_blocking_time_s": 0.2,
        "prefetch_cleanup_released": False,
        "preemption_count": 1,
    }

    event = scheduler._prefetch_attribution_event(state, "partial", "ignored")

    assert event["status"] == "unavailable"
    assert event["status_reason"] == "first_forward_timestamp_unavailable"
    assert event["required_keys"] == ["k0", "k1"]
    assert event["transfer_contributors"] == {"9": ["0"]}
    assert event["t0_preemption"] == 10
    assert "t6_first_forward" in event["unavailable_fields"]
    assert event["prefetch_device_time_s"] == 0.3
    assert event["prefetch_device_time_source"] == "cuda_event_elapsed_time"
    assert event["blocking_device_time_s"] == 0.2
    assert event["blocking_device_time_source"] == "cuda_event_elapsed_time"
    assert event["total_h2d_bytes"] is None
    assert event["t4_source"] == "scheduler_update_state_after_alloc_entry"
    assert event["t4_episode_id"] == 1
    assert event["transfer_hidden_before_demand_s"] == pytest.approx(10e-9)
    assert event["exposed_blocking_stall_s"] == pytest.approx(20e-9)


def _p3_consumed_state(*, consumed_blocks=1, owner_id="owner-1"):
    return {
        "preempted": True,
        "run_id": "run-1",
        "cell_id": "cell-1",
        "session_id": "campaign-session-1",
        "policy_session_id": "session-1",
        "generation": 0,
        "external_request_id": "owner-1",
        "episode_id": 1,
        "connector_observed_ns": 100,
        "prefetch_t2_ns": 10,
        "prefetch_t3_ns": 20,
        "prefetch_install_submitted_ns": 120,
        "prefetch_request_usable_ns": 150,
        "prefetch_request_usable_source": (
            "all_worker_splice_cuda_fences_acknowledged"
        ),
        "prefetch_splice_acknowledged": True,
        "prefetch_consumed_blocks": consumed_blocks,
        "prefetch_attempted_blocks": 1,
        "prefetch_submitted_blocks": 1,
        "prefetch_submitted_jobs": 1,
        "prefetch_discarded_blocks": 0,
        "speculative_transfer_job_statuses": [
            {
                "status": "success",
                "success": True,
                "cancelled": False,
                "bytes": 128,
                "completed_at_ns": 20,
                "contributors": 1,
                "expected_contributors": 1,
                "keys": ["k0"],
            }
        ],
        "wait_hint": {
            "owner_id": owner_id,
            "run_id": "run-1",
            "cell_id": "cell-1",
            "session_id": "session-1",
            "generation": 0,
        },
        "prefetch_unavailable_fields": [],
        "prefetch_failure_reason": None,
        "prefetch_cancelled": False,
        "prefetch_stale": False,
        "prefetch_allocation_failures": 0,
        "prefetch_provenance": {},
        "preemption_count": 1,
    }


def test_p3_full_cover_promotes_exact_consumed_connector_demand_boundary():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )

    event = scheduler._prefetch_attribution_event(
        _p3_consumed_state(), "hit", "ignored"
    )

    assert event["t4_readmission_allocation"] == 100
    assert event["t4_source"] == "hinted_request_connector_observation"
    assert event["demand_boundary_ns"] == 100
    assert event["demand_boundary_source"] == (
        "hinted_request_connector_observation"
    )
    assert event["demand_boundary_clock_domain"] == "host_monotonic"
    assert event["transfer_hidden_before_demand_s"] == pytest.approx(10e-9)
    assert event["staging_ready_lead_time_s"] == pytest.approx(80e-9)
    assert event["exposed_blocking_stall_s"] == pytest.approx(50e-9)


@pytest.mark.parametrize(
    "state",
    [
        _p3_consumed_state(consumed_blocks=0),
        _p3_consumed_state(owner_id="different-owner"),
    ],
)
def test_p3_boundary_is_not_promoted_without_consumption_and_exact_identity(state):
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )

    event = scheduler._prefetch_attribution_event(state, "hit", "ignored")

    assert "t4_readmission_allocation" not in event
    assert event["demand_boundary_ns"] is None
    assert event["demand_boundary_clock_domain"] is None
    assert event["transfer_hidden_before_demand_s"] is None
    assert event["exposed_blocking_stall_s"] is None


def test_speculative_transfer_accounting_distinguishes_discarded_completion():
    state = {
        "prefetch_attempted_blocks": 5,
        "prefetch_consumed_blocks": 0,
        "prefetch_discarded_blocks": 5,
        "speculative_transfer_job_statuses": [
            {
                "status": "success",
                "success": True,
                "cancelled": False,
                "bytes": 640,
                "time_s": 0.004,
                "contributors": 1,
                "expected_contributors": 1,
            }
        ],
    }

    accounting = _speculative_transfer_accounting(state, 128)

    assert accounting["submitted_bytes"] == 640
    assert accounting["completed_bytes"] == 640
    assert accounting["discarded_bytes"] == 640
    assert accounting["released_reservation_bytes"] == 640
    assert accounting["uncompleted_submitted_bytes"] == 0


def test_speculative_transfer_accounting_keeps_failed_release_out_of_bytes():
    state = {
        "prefetch_attempted_blocks": 1,
        "prefetch_consumed_blocks": 0,
        "prefetch_discarded_blocks": 1,
        "speculative_transfer_job_statuses": [
            {
                "status": "cancelled",
                "success": False,
                "cancelled": True,
                "bytes": None,
                "time_s": None,
                "contributors": 0,
                "expected_contributors": 1,
            }
        ],
    }

    accounting = _speculative_transfer_accounting(state, 128)

    assert accounting["submitted_bytes"] == 128
    assert accounting["completed_bytes"] == 0
    assert accounting["discarded_bytes"] == 0
    assert accounting["released_reservation_bytes"] == 128
    assert accounting["uncompleted_submitted_bytes"] == 128


def test_prefetch_attribution_event_accepts_model_runner_entry_boundary():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    event = scheduler._prefetch_attribution_event(
        {
            "preempted": True,
            "prefetch_t0_ns": 10,
            "prefetch_t6_ns": 60,
            "prefetch_t6_source": "model_runner_execute_entry",
            "prefetch_unavailable_fields": [],
            "prefetch_failure_reason": None,
            "prefetch_cancelled": False,
            "prefetch_stale": False,
            "prefetch_splice_acknowledged": False,
            "prefetch_splice_ack_id": None,
            "prefetch_source_reuse_before_fence": None,
            "prefetch_fence_complete": False,
            "prefetch_cleanup_released": False,
            "prefetch_splice_bytes": None,
            "prefetch_filler_failures": None,
            "prefetch_allocation_failures": 0,
            "prefetch_provenance": {},
            "prefetch_required_keys": [],
            "prefetch_cpu_resident_keys": [],
            "prefetch_cpu_ready_keys": [],
            "prefetch_cpu_residency_proof": {},
            "prefetch_selected_keys": [],
            "prefetch_intersection_keys": [],
            "prefetch_remainder_keys": [],
            "prefetch_source_map": {},
            "prefetch_destination_map": {},
            "prefetch_reservation_blocks": [],
            "prefetch_reservation_history": [],
            "prefetch_transfer_contributors": {},
            "preemption_count": 1,
            "episode_id": 1,
        },
        "partial",
        "ignored",
    )

    assert event["t6_first_forward"] == 60
    assert event["t6_source"] == "model_runner_execute_entry"
    assert "t6_first_forward" not in event["unavailable_fields"]


def test_block_pool_snapshot_separates_free_reserved_and_active_blocks():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._prefetch_reserved_blocks = 5
    scheduler.gpu_block_pool = SimpleNamespace(
        num_gpu_blocks=101,
        null_block=SimpleNamespace(is_null=True),
        get_num_free_blocks=lambda: 60,
    )
    state = {}

    scheduler._record_block_pool_snapshot(state, "t1_reservation_submit", job_id=7)

    snapshot = state["prefetch_block_pool_snapshots"][0]
    assert snapshot["available"] is True
    assert snapshot["source"] == "scheduler"
    assert snapshot["total_blocks"] == 101
    assert snapshot["schedulable_capacity_blocks"] == 100
    assert snapshot["free_block_queue_blocks"] == 60
    assert snapshot["scheduler_prefetch_reserved_blocks_total"] == 5
    assert snapshot["estimated_referenced_blocks"] == 35
    assert snapshot["accounting_underflow_blocks"] == 0
    assert snapshot["accounting_consistent"] is True
    assert snapshot["configured_prefetch_headroom_blocks"] == 0
    assert snapshot["scheduler_free_headroom_blocks"] == 60
    assert snapshot["job_id"] == 7


def test_block_pool_snapshot_is_explicitly_unavailable_without_bound_pool():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler.gpu_block_pool = None
    state = {}

    scheduler._record_block_pool_snapshot(state, "t0_preemption")

    snapshot = state["prefetch_block_pool_snapshots"][0]
    assert snapshot["schema"] == "semantic-block-pool-snapshot-v1"
    assert snapshot["phase"] == "t0_preemption"
    assert snapshot["source"] == "scheduler"
    assert isinstance(snapshot["observed_at_ns"], int)
    assert snapshot["available"] is False
    assert snapshot["reason"] == "gpu_block_pool_unbound"
    assert snapshot["episode_id"] == 0
    assert snapshot["preemption_index"] == 0


def test_block_pool_snapshot_surfaces_accounting_underflow():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._prefetch_reserved_blocks = 2
    scheduler._prefetch_headroom_blocks = 1
    scheduler.gpu_block_pool = SimpleNamespace(
        num_gpu_blocks=10,
        null_block=SimpleNamespace(is_null=True),
        get_num_free_blocks=lambda: 9,
    )

    snapshot = scheduler._block_pool_snapshot("t1_reservation_refused")

    assert snapshot["available"] is True
    assert snapshot["estimated_referenced_blocks"] == 0
    assert snapshot["accounting_underflow_blocks"] == -2
    assert snapshot["accounting_consistent"] is False
    assert snapshot["scheduler_free_headroom_blocks"] == 8


def test_block_pool_snapshot_fails_closed_without_null_block_identity():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler.gpu_block_pool = SimpleNamespace(
        num_gpu_blocks=10,
        get_num_free_blocks=lambda: 9,
    )

    snapshot = scheduler._block_pool_snapshot("t0_preemption")

    assert snapshot["available"] is False
    assert snapshot["reason"] == "null_block_identity_unavailable"


def test_block_pool_snapshot_history_is_bounded_but_counted():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._prefetch_reserved_blocks = 0
    scheduler._prefetch_headroom_blocks = 0
    scheduler.gpu_block_pool = SimpleNamespace(
        num_gpu_blocks=10,
        null_block=SimpleNamespace(is_null=True),
        get_num_free_blocks=lambda: 9,
    )
    state = {}

    for index in range(65):
        scheduler._record_block_pool_snapshot(state, f"phase-{index}")

    snapshots = state["prefetch_block_pool_snapshots"]
    assert len(snapshots) == 64
    assert state["prefetch_block_pool_snapshot_count"] == 65
    assert state["prefetch_block_pool_snapshots_truncated"] is True
    assert snapshots[0]["phase"] == "phase-0"
    assert snapshots[-1]["phase"] == "phase-64"


def test_worker_failed_transfer_records_start_and_completion_timestamps():
    worker = SemanticOffloadingWorker.__new__(SemanticOffloadingWorker)
    worker._preservation_events = True
    worker._transfer_directions = {}
    worker._transfer_started_at_ns = {}
    worker._transfer_destination_blocks = {}
    worker._failed_load_block_ids = set()
    worker._pending_transfer_events = {}

    worker.register_transfer_direction(7, is_load=True, destination_block_ids=[11])
    worker.record_transfer_failure(7, is_load=True, cancelled=True)
    event = worker.pop_transfer_events()[7]

    assert event["status"] == "cancelled"
    assert event["cancelled"] is True
    assert event["started_at_ns"] is not None
    assert event["completed_at_ns"] >= event["started_at_ns"]
    assert worker.pop_failed_load_block_ids() == {11}


def test_worker_transfer_metadata_aggregates_time_and_contributor_identity():
    first = SemanticWorkerMetadata(
        transfer_events={
            7: SemanticTransferCompletion(
                is_load=True,
                bytes=100,
                time_s=0.2,
                started_at_ns=10,
                completed_at_ns=20,
                contributor_ids=("0",),
            )
        }
    )
    second = SemanticWorkerMetadata(
        transfer_events={
            7: SemanticTransferCompletion(
                is_load=True,
                bytes=200,
                time_s=0.3,
                started_at_ns=12,
                completed_at_ns=25,
                contributor_ids=("1",),
            )
        }
    )

    event = first.aggregate(second).transfer_events[7]

    assert event.bytes == 300
    assert event.time_s == pytest.approx(0.5)
    assert event.started_at_ns == 10
    assert event.completed_at_ns == 25
    assert event.contributor_ids == ("0", "1")


def test_serialized_mixed_lookup_groups_carry_explicit_flags():
    from semantic_offload.connector import _serialize_lookup_episodes

    serialized = _serialize_lookup_episodes(
        [
            {
                "lookup_id": 0,
                "groups": {
                    "0": {
                        "eligible": [],
                        "queried": [],
                        "hits": [],
                        "missing": [],
                        "pending": [],
                        "retry": [],
                        "observed": True,
                    },
                    "1": {
                        "eligible": ["key"],
                        "queried": ["key"],
                        "hits": ["key"],
                        "missing": [],
                        "pending": [],
                        "retry": [],
                        "observed": True,
                    },
                },
            }
        ]
    )

    groups = serialized[0]["groups"]
    assert groups["0"]["observed"] is True
    assert groups["0"]["no_eligible"] is True
    assert groups["1"]["observed"] is True
    assert groups["1"]["no_eligible"] is False


def test_lookup_observation_follows_evaluated_groups_after_parent_refresh():
    from semantic_offload.connector import _serialize_lookup_episodes

    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._preservation_requests = {
        "req": {"lookup_sequence": 0, "active_lookup_episode": None}
    }
    scheduler._req_status = {
        "req": SimpleNamespace(
            group_states=[
                SimpleNamespace(offload_keys=[]),
                SimpleNamespace(offload_keys=[]),
            ]
        )
    }
    request = SimpleNamespace(request_id="req")
    scheduler._begin_lookup_episode(request)

    before_hook = _serialize_lookup_episodes(
        [scheduler._preservation_requests["req"]["active_lookup_episode"]]
    )
    assert before_hook[0]["groups"] == {}

    scheduler._req_status["req"].group_states[0].offload_keys = ["key0"]
    scheduler._req_status["req"].group_states[1].offload_keys = ["key1"]
    scheduler._mark_lookup_group_evaluated("req", 0, ["key0"])

    serialized = _serialize_lookup_episodes(
        [scheduler._preservation_requests["req"]["active_lookup_episode"]]
    )
    groups = serialized[0]["groups"]
    assert set(groups) == {"0"}
    assert groups["0"]["observed"] is True
    assert groups["0"]["no_eligible"] is False


def test_empty_sliding_group_stays_unobserved_and_fails_closed():
    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._preservation_requests = {
        "req": {"lookup_sequence": 0, "active_lookup_episode": None}
    }
    scheduler._req_status = {"req": SimpleNamespace(group_states=[])}
    scheduler._begin_lookup_episode(SimpleNamespace(request_id="req"))

    scheduler._mark_lookup_group_evaluated("req", -1, [])

    assert (
        scheduler._preservation_requests["req"]["active_lookup_episode"]["groups"] == {}
    )


def _make_lookup_probe_scheduler(
    request_tokens, local_tokens, tokens_per_chunk=16, *, sliding_window=False
):
    from semantic_offload.connector import SemanticOffloadingConnectorScheduler

    scheduler = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    scheduler._preservation_events_enabled = True
    scheduler._preservation_requests = {
        "req": {
            "lookup_sequence": 0,
            "active_lookup_episode": None,
            "lookup_observed": False,
            "lookup_episodes": [],
            "lookup_keys": [],
            "prior_stored_keys": [],
            "prior_evicted_keys": [],
            "cold_keys": [],
            "last_lookup_id": None,
        }
    }
    scheduler._req_status = {
        "req": SimpleNamespace(
            req=SimpleNamespace(num_tokens=request_tokens),
            num_locally_computed_tokens=local_tokens,
            group_states=[SimpleNamespace(offload_keys=["key"] * 6)],
        )
    }
    scheduler.config = SimpleNamespace(
        kv_group_configs=[
            SimpleNamespace(
                group_idx=0,
                tokens_per_chunk=tokens_per_chunk,
                sliding_window_size_in_chunks=4 if sliding_window else None,
            )
        ]
    )
    scheduler._lookup_groups = (0,)
    return scheduler


def test_short_lookup_remainder_records_explicit_no_eligible_group():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    scheduler = _make_lookup_probe_scheduler(request_tokens=93, local_tokens=80)
    request = SimpleNamespace(request_id="req", kv_transfer_params={})
    with patch.object(
        OffloadingConnectorScheduler,
        "get_num_new_matched_tokens",
        return_value=(0, False),
    ):
        assert scheduler.get_num_new_matched_tokens(request, 80) == (0, False)

    episode = scheduler._preservation_requests["req"]["lookup_episodes"][0]
    group = episode["groups"]["0"]
    assert group["observed"] is True
    assert group["eligible"] == []
    assert scheduler._preservation_requests["req"]["lookup_observed"] is True


def test_skip_reading_prefix_cache_keeps_short_remainder_unobserved():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    scheduler = _make_lookup_probe_scheduler(request_tokens=93, local_tokens=80)
    request = SimpleNamespace(
        request_id="req",
        skip_reading_prefix_cache=True,
        kv_transfer_params={},
    )
    with patch.object(
        OffloadingConnectorScheduler,
        "get_num_new_matched_tokens",
        return_value=(0, False),
    ):
        assert scheduler.get_num_new_matched_tokens(request, 80) == (0, False)

    episode = scheduler._preservation_requests["req"]["lookup_episodes"][0]
    assert episode["groups"] == {}
    assert scheduler._preservation_requests["req"]["lookup_observed"] is False


def test_hybrid_short_remainder_keeps_unobserved_group():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    scheduler = _make_lookup_probe_scheduler(
        request_tokens=93, local_tokens=80, sliding_window=True
    )
    request = SimpleNamespace(request_id="req", kv_transfer_params={})
    with patch.object(
        OffloadingConnectorScheduler,
        "get_num_new_matched_tokens",
        return_value=(0, False),
    ):
        assert scheduler.get_num_new_matched_tokens(request, 80) == (0, False)

    episode = scheduler._preservation_requests["req"]["lookup_episodes"][0]
    assert episode["groups"] == {}
    assert scheduler._preservation_requests["req"]["lookup_observed"] is False


def test_hook_free_non_short_lookup_remains_unobserved():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    scheduler = _make_lookup_probe_scheduler(request_tokens=100, local_tokens=80)
    request = SimpleNamespace(request_id="req", kv_transfer_params={})
    with patch.object(
        OffloadingConnectorScheduler,
        "get_num_new_matched_tokens",
        return_value=(0, False),
    ):
        assert scheduler.get_num_new_matched_tokens(request, 80) == (0, False)

    episode = scheduler._preservation_requests["req"]["lookup_episodes"][0]
    assert episode["groups"] == {}
    assert scheduler._preservation_requests["req"]["lookup_observed"] is False


def test_splice_gpu_blocks_copies_content_to_new_indices():
    layer = _make_layer(num_blocks=6, num_kv_heads=2, block_size=4, head_size=8, seed=0)
    worker = _make_worker({"layer0": layer})
    original_src = layer.kv_cache[2].clone()

    worker.splice_gpu_blocks(src_block_ids=[2], dst_block_ids=[5])

    assert torch.allclose(layer.kv_cache[5], original_src)


def test_splice_gpu_blocks_multiple_layers_and_blocks():
    layer_a = _make_layer(6, 2, 4, 8, seed=1)
    layer_b = _make_layer(6, 2, 4, 8, seed=2)
    worker = _make_worker({"a": layer_a, "b": layer_b})
    expected_a = layer_a.kv_cache[[1, 3]].clone()
    expected_b = layer_b.kv_cache[[1, 3]].clone()

    worker.splice_gpu_blocks(src_block_ids=[1, 3], dst_block_ids=[4, 5])

    assert torch.allclose(layer_a.kv_cache[[4, 5]], expected_a)
    assert torch.allclose(layer_b.kv_cache[[4, 5]], expected_b)


def test_splice_gpu_blocks_empty_is_a_noop():
    layer = _make_layer(4, 1, 4, 8, seed=3)
    worker = _make_worker({"layer0": layer})
    before = layer.kv_cache.clone()

    worker.splice_gpu_blocks(src_block_ids=[], dst_block_ids=[])

    assert torch.allclose(layer.kv_cache, before)


def test_splice_gpu_blocks_mismatched_lengths_raises():
    layer = _make_layer(4, 1, 4, 8, seed=4)
    worker = _make_worker({"layer0": layer})
    raised = False
    try:
        worker.splice_gpu_blocks(src_block_ids=[0, 1], dst_block_ids=[2])
    except AssertionError:
        raised = True
    assert raised


def test_splice_completion_is_reported_once_after_cpu_copy():
    layer = _make_layer(4, 1, 4, 8, seed=5)
    worker = _make_worker({"layer0": layer})

    worker.splice_gpu_blocks([0], [2], splice_id=17)

    assert worker.pop_completed_splices() == [17]
    assert worker.pop_completed_splices() == []
    evidence = worker.pop_completed_splice_evidence()[17]
    assert evidence["splice_id"] == 17
    assert evidence["source_block_count"] == 1
    assert evidence["destination_block_count"] == 1
    assert evidence["copied_layer_count"] == 1
    assert evidence["same_default_stream"] is True
    assert evidence["issued_before_base_bind_and_forward"] is True
    assert evidence["completion_event_satisfied"] is True
    assert evidence["issued_at_ns"] <= evidence["completion_observed_at_ns"]
    assert worker.pop_completed_splice_evidence() == {}


class _StubBlockPool:
    """Minimal stand-in for vllm's real BlockPool -- only the surface
    _attempt_prefetch_reservation actually touches. get_new_blocks() mirrors
    the real one's contract (returns objects with .block_id, decrements the
    free count) closely enough for this bookkeeping-focused test; the real
    BlockPool itself was already verified safe under real concurrent load in
    issues log entry #23."""

    def __init__(
        self,
        num_gpu_blocks: int,
        free_blocks: int,
        cached_block_hashes: dict[bytes, int] | None = None,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self._free = free_blocks
        self.blocks = [
            SimpleNamespace(block_id=block_id) for block_id in range(num_gpu_blocks)
        ]
        # Real BlockPool reserves block_id 0 as a null block, never handed
        # out for real content -- start at 1 to match, since production
        # code filters `bid != 0` when recording allocated block ids.
        self._next_id = 1
        self._cached_block_hashes = dict(cached_block_hashes or {})

    def get_num_free_blocks(self) -> int:
        return self._free

    def get_new_blocks(self, n: int):
        assert n <= self._free
        self._free -= n
        blocks = []
        for _ in range(n):
            blocks.append(self.blocks[self._next_id])
            self._next_id += 1
        return blocks

    def free_blocks(self, blocks) -> None:
        self._free += len(list(blocks))

    def get_cached_block(self, block_hash, _group_ids):
        block_id = self._cached_block_hashes.get(block_hash)
        return None if block_id is None else [self.blocks[block_id]]


def _insert_resident(manager: SemanticOffloadingManager, key, block_id: int) -> None:
    block = BlockStatus(block_id=block_id)
    block.ref_cnt = 0  # ready
    manager._policy.insert(key, block)
    manager._policy.mark_evictable(key)
    # Bypassing the manager's real prepare_store()->insert() completion path
    # (there's no lighter-weight public API to mark a block resident+
    # evictable for testing), so its private evictable-block counter has to
    # be kept in sync by hand too, or prepare_load()'s own internal
    # bookkeeping assertion fails on the next real insert/evict cycle.
    manager._num_evictable_cache_blocks += 1


def _make_connector_scheduler(
    manager: SemanticOffloadingManager,
    gpu_block_pool: _StubBlockPool,
    *,
    num_groups: int = 1,
    block_size_factor: int = 1,
) -> SemanticOffloadingConnectorScheduler:
    sched = SemanticOffloadingConnectorScheduler.__new__(
        SemanticOffloadingConnectorScheduler
    )
    sched.manager = manager
    sched.gpu_block_pool = gpu_block_pool
    sched._prefetched = {}
    sched._observed_prefixes = OrderedDict()
    sched._observed_prefix_generations = {}
    sched._observed_prefix_limit = 32
    sched._wait_prefetch = {}
    sched._wait_prefetch_jobs = {}
    sched._wait_hint_results = OrderedDict()
    sched._wait_hint_result_limit = 64
    sched._wait_hint_request_bindings = {}
    sched._wait_hint_ttl_ns = 2_000_000_000
    sched._wait_hint_namespace = {
        "model_revision": "model",
        "tokenizer_revision": "tokenizer",
        "adapter_namespace": "none",
        "layout_namespace": "dense-tp1-bf16-block16",
    }
    sched._prefetch_trigger = "known_prefix_wait_hint"
    sched._soft_retention = "off"
    sched._staging_mode = "dynamic_scratch"
    sched._prefetch_fallback_ns_per_byte = 1.0 / 16.0
    sched._prefetch_fallback_splice_ns_per_byte = 1.0 / 16.0
    sched._prefetch_transfer_ns_per_byte = None
    sched._prefetch_cost_available_ns = None
    sched._prefetch_bytes_per_key = 128
    sched._prefetch_score_max_age_ns = None
    sched._prefetch_reserved_blocks = 0
    sched._prefetch_budget_fraction = 0.05
    sched._prefetch_selector = "semantic"
    sched._prefetch_headroom_blocks = 0
    sched._preempted_pending = set()
    sched._req_status = {}
    sched._current_batch_load_jobs = {}
    sched._current_batch_prefetch_load_jobs = {}
    sched._current_batch_allocated_block_ids = set()
    sched._pending_splice_jobs = {}
    sched._pending_splice_releases = {}
    sched._pending_splice_request_ids = {}
    sched._splice_counter = 0
    sched._pending_job_keys = {}
    sched._pending_job_blocks = {}
    sched._worker_reset_pending = False
    sched._reset_prefetch_releases = {}
    sched._jobs = {}
    sched._preservation_events_enabled = False
    sched._preservation_job_events = {}
    sched._job_counter = 0
    sched._preservation_event_sequence = 0
    sched._preservation_key_commits = defaultdict(list)
    sched._preservation_key_evictions = defaultdict(list)
    # Same object under both names (pre-#48150 _blocks_being_loaded, renamed
    # _chunks_being_loaded upstream) -- our own code reads through
    # scheduler_being_loaded_set (checks _chunks_being_loaded first), real
    # vLLM code reads whichever name its installed generation declares;
    # aliasing to one set means either path mutates the same object.
    sched._blocks_being_loaded = sched._chunks_being_loaded = set()
    # gpu_block_size/offloaded_block_size=1 (pre-#48150 names) and their
    # post-#48150 renames tokens_per_block/tokens_per_chunk both set to the
    # same value keep splice-plan tests' tokens/blocks/keys all in 1:1
    # correspondence, so a test can build a request needing N blocks by
    # just handing it N keys and N stub GPU blocks -- the arithmetic in
    # _compute_load_plan degenerates cleanly. Both name generations are
    # set (not just whichever the installed vLLM uses) so this stub also
    # satisfies the REAL inherited base method when a test calls straight
    # through to super().update_state_after_alloc() (see
    # test_compute_load_plan_matches_the_real_base_method) -- our own code
    # reads through the version-tolerant _vllm_compat helpers, but real
    # vLLM code reads its installed generation's names directly.
    group_config = SimpleNamespace(
        gpu_block_size=1,
        offloaded_block_size=1,
        tokens_per_block=1,
        tokens_per_chunk=1,
        hash_block_size_factor=1,
        sliding_window_size_in_blocks=None,
        sliding_window_size_in_chunks=None,
    )
    sched.config = SimpleNamespace(
        num_workers=1,
        kv_group_configs=tuple(group_config for _ in range(num_groups)),
        block_size_factor=block_size_factor,
        blocks_per_chunk=block_size_factor,
    )
    return sched


def test_passive_ordered_key_map_preserves_scheduler_order_without_cache_observation():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    keys = [to_key(3), to_key(1), to_key(2)]
    sched._req_status["req"] = SimpleNamespace(
        group_states=[SimpleNamespace(offload_keys=keys)]
    )
    request = SimpleNamespace(request_id="req", prompt_token_ids=[10, 11, 12])

    with patch.object(
        manager, "lookup", side_effect=AssertionError("diagnostics touched cache")
    ) as lookup:
        positions, complete = sched._ordered_request_key_positions(request)

    lookup.assert_not_called()
    assert complete is True
    assert [record["key"] for record in positions] == keys
    assert [record["chunk_index"] for record in positions] == [0, 1, 2]
    assert all(record["state"] == "planned" for record in positions)


def test_passive_key_map_refreshes_after_real_native_population_order():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        GroupOffloadConfig,
        RequestOffloadState,
        SchedulerOffloadConfig,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    group_config = GroupOffloadConfig(
        group_idx=0,
        tokens_per_block=16,
        tokens_per_chunk=16,
        hashes_per_chunk=1,
        kv_event_group_spec=None,
        sliding_window_size_in_chunks=None,
    )
    native_config = SchedulerOffloadConfig(
        kv_group_configs=(group_config,),
        blocks_per_chunk=1,
        num_workers=1,
        offload_prompt_only=False,
    )
    request = SimpleNamespace(
        request_id="req",
        kv_transfer_params={},
        block_hashes=[b"block-0", b"block-1"],
    )
    native_state = RequestOffloadState(
        config=native_config,
        req=request,
        req_context=ReqContext(req_id="req"),
        offloading_context=RequestOffloadingContext(),
    )
    sched.config = native_config
    sched._req_status["req"] = native_state
    sched._preservation_requests = {
        "req": {
            "prompt_token_count": 32,
            "ordered_key_positions": [],
            "ordered_key_positions_complete": False,
        }
    }

    assert native_state.group_states[0].offload_keys == []
    sched._refresh_ordered_key_positions("req")
    assert sched._preservation_requests["req"][
        "ordered_key_positions_complete"
    ] is False

    with patch.object(
        manager, "lookup", side_effect=AssertionError("diagnostics touched cache")
    ) as lookup:
        native_state.update_offload_keys()
        sched._refresh_ordered_key_positions("req")

    lookup.assert_not_called()
    positions = sched._preservation_requests["req"]["ordered_key_positions"]
    assert [position["chunk_index"] for position in positions] == [0, 1]
    assert sched._preservation_requests["req"][
        "ordered_key_positions_complete"
    ] is True
    assert sched._preservation_requests["req"][
        "ordered_key_positions_capture_provenance"
    ] == "after_native_update_offload_keys"


def test_passive_commit_and_eviction_join_generation_and_monotonic_sequence():
    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    key = to_key(1)
    position = {
        "key": key,
        "group_idx": 0,
        "chunk_index": 0,
        "token_start": 0,
        "token_end": 1,
        "complete_chunk": True,
        "provenance": "scheduler_request_group_state",
        "state": "planned",
    }
    state = {
        "request_id": "warm",
        "manifest_request_id": "warm-manifest",
        "run_id": "run",
        "cell_id": "cell",
        "policy_session_id": "session",
        "generation": 7,
        "cpu_commit_events": [],
    }

    sched._record_cpu_commit_events(
        job_id=4,
        state=state,
        key_positions=[position],
        completed_at_ns=100,
    )
    sched._record_cpu_eviction_events([key])

    commit = sched._preservation_key_commits[key][0]
    eviction = sched._preservation_key_evictions[key][0]
    assert commit["generation"] == eviction["generation"] == 7
    assert commit["sequence"] < eviction["sequence"]
    assert eviction["prior_commit_sequence"] == commit["sequence"]
    assert eviction["prior_commit_ns"] == 100
    assert eviction["provenance"] == "manager_prepare_store_evicted_keys"


def test_first_admission_snapshot_is_not_overwritten_by_later_admission():
    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    sched._preservation_events_enabled = True
    keys = [to_key(1), to_key(2), to_key(3)]
    sched._req_status["req"] = SimpleNamespace(
        num_locally_computed_tokens=1,
        group_states=[SimpleNamespace(offload_keys=keys)],
    )
    state = {
        "required_tokens": 3,
        "readmission_pending": False,
        "episode_external_tokens": 0,
        "loaded_tokens": 0,
        "loaded_tokens_total": 0,
        "admitted_external_tokens": 0,
        "admitted_local_tokens": 0,
        "physical_block_identity_complete": True,
        "admitted_local_blocks": 0,
        "admitted_external_blocks": 0,
        "required_block_ids": set(),
        "local_block_ids": set(),
        "external_block_ids": set(),
        "episode_id": 0,
        "episode_prompt_accounted": 0,
        "episode_local_reused_tokens": 0,
        "admission_count": 0,
        "last_lookup_id": 0,
        "ordered_key_positions": [
            {
                "key": key,
                "group_idx": 0,
                "chunk_index": index,
                "token_start": index,
                "token_end": index + 1,
                "complete_chunk": True,
                "provenance": "scheduler_request_group_state",
                "state": "planned",
            }
            for index, key in enumerate(keys)
        ],
        "prior_evicted_keys": [],
        "recompute_admission_count": 0,
    }
    sched._preservation_requests = {"req": state}
    request = SimpleNamespace(request_id="req")
    blocks = SimpleNamespace(
        blocks=[
            [SimpleNamespace(block_id=index + 10) for index in range(3)]
        ]
    )

    sched._record_admission(request, blocks, num_external_tokens=1)
    frozen = dict(state["first_admission"])
    sched._req_status["req"].num_locally_computed_tokens = 2
    state["episode_id"] = 1
    sched._record_admission(request, blocks, num_external_tokens=0)

    assert state["first_admission"] == frozen
    assert [record["key"] for record in frozen["local_keys"]] == [keys[0]]
    assert [record["key"] for record in frozen["planned_load_keys"]] == [keys[1]]
    assert state["admission_count"] == 2


def test_chain_admission_evidence_binds_async_completion_to_finished_request():
    namespace = {
        "model_revision": "model",
        "tokenizer_revision": "tokenizer",
        "adapter_namespace": "none",
        "layout_namespace": "dense-tp1-bf16-block16",
    }
    manager = SemanticOffloadingManager(
        num_blocks=2,
        store_threshold=2,
        admission_namespace=namespace,
        chain_admission_max_prefix_blocks=2,
    )
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    sched._preservation_requests = {
        "first": {"request_id": "first"},
        "second": {"request_id": "second"},
    }
    sched._preservation_finished = {}
    sched._chain_admission_global_accounting = {}
    keys = [to_key(901), to_key(902)]

    def context(request_id):
        return ReqContext(
            req_id=request_id,
            kv_transfer_params={
                "run_id": "run",
                "session_id": "session",
                "generation": 0,
                "key_chain_digest": "a" * 64,
            },
        )

    first_context = context("first")
    first = manager.prepare_store(keys, first_context)
    assert first is not None and first.keys_to_store == []
    sched._drain_chain_admission_evidence()
    assert sched._preservation_requests["first"]["chain_admission_records"][0][
        "action"
    ] == "observe"

    second_context = context("second")
    second = manager.prepare_store(keys, second_context)
    assert second is not None and second.keys_to_store == keys
    sched._drain_chain_admission_evidence()
    second_state = sched._preservation_requests.pop("second")
    sched._preservation_finished["second"] = second_state

    manager.complete_store(second.keys_to_store, second_context)
    sched._drain_chain_admission_evidence()

    assert [
        record["action"] for record in second_state["chain_admission_records"]
    ] == ["admit", "store_complete"]
    assert second_state["chain_admission_accounting"]["observations"] == 1
    assert second_state["chain_admission_accounting"]["qualified_keys"] == 2
    assert second_state["chain_admission_accounting"]["admitted_keys"] == 2
    assert second_state["chain_admission_accounting"]["stored_keys"] == 2
    assert sched._chain_admission_global_accounting["observations"] == 2
    assert sched._chain_admission_global_accounting["qualified_keys"] == 2
    assert sched._chain_admission_global_accounting["stored_keys"] == 2


def test_reset_releases_completed_prefetch_and_clears_semantic_state():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    pool = _StubBlockPool(20, free_blocks=0)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetched["r1"] = _PrefetchState(
        job_id=7, keys=[to_key(1)], gpu_block_ids=[3]
    )
    sched._prefetch_reserved_blocks = 1
    sched._preempted_pending.add("r1")
    sched._pending_job_blocks[9] = [(to_key(1), [2])]

    with patch.object(OffloadingConnectorScheduler, "reset_cache") as base_reset:
        sched.reset_cache()

    base_reset.assert_called_once_with()
    assert pool.get_num_free_blocks() == 1
    assert sched._prefetched == {}
    assert sched._prefetch_reserved_blocks == 0
    assert sched._preempted_pending == set()
    assert sched._pending_job_blocks == {}
    assert sched._worker_reset_pending is True


def test_reset_keeps_inflight_prefetch_reserved_until_transfer_completion():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    pool = _StubBlockPool(20, free_blocks=0)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetched["r1"] = _PrefetchState(
        job_id=7, keys=[to_key(1)], gpu_block_ids=[3]
    )
    sched._prefetch_reserved_blocks = 1
    sched._jobs[7] = SimpleNamespace(pending_count=2)

    with patch.object(OffloadingConnectorScheduler, "reset_cache"):
        sched.reset_cache()

    assert pool.get_num_free_blocks() == 0
    assert sched._prefetch_reserved_blocks == 1
    assert sched._reset_prefetch_releases == {7: ([3], 2)}

    output = SimpleNamespace(
        kv_connector_worker_meta=SemanticWorkerMetadata(completed_jobs={7: 1})
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(output)
        assert pool.get_num_free_blocks() == 0
        sched.update_connector_output(output)

    assert pool.get_num_free_blocks() == 1
    assert sched._prefetch_reserved_blocks == 0
    assert sched._reset_prefetch_releases == {}


def _add_req_status(
    sched,
    req_id: str,
    keys: list,
    *,
    num_locally_computed_tokens: int = 0,
    policy: OffloadPolicy = OffloadPolicy.BLOCK_LEVEL,
) -> None:
    sched._req_status[req_id] = SimpleNamespace(
        group_states=[RequestGroupState(offload_keys=keys)],
        req_context=ReqContext(req_id=req_id),
        transfer_jobs=set(),
        num_locally_computed_tokens=num_locally_computed_tokens,
        offloading_context=RequestOffloadingContext(policy=policy),
    )


def _make_pending_blocks(n: int, start_block_id: int = 100):
    """N stub GPU blocks, none locally-cached (`block_hash=None`) -- i.e.
    all N need loading, matching a request re-admitted after preemption
    with nothing yet spliced or reloaded. `blocks.blocks[0]` is the shape
    `_compute_load_plan` reads (one list of block stand-ins per KV group)."""
    group_blocks = [
        SimpleNamespace(block_id=start_block_id + i, is_null=False, block_hash=None)
        for i in range(n)
    ]
    return SimpleNamespace(blocks=[group_blocks])


def _register_wait_prefix(
    sched, keys, *, session_id="session", generation=0, request_id="warm"
):
    prompt_token_ids = list(range(1, len(keys) + 2))
    boundary = len(keys)
    prefix_id = _observed_prefix_digest(
        prompt_token_ids[:boundary], sched._wait_hint_namespace
    )
    _add_req_status(sched, request_id, keys=list(keys))
    request = SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        kv_transfer_params={
            "run_id": "run",
            "cell_id": "cell",
            "session_id": session_id,
            "generation": generation,
            "observed_prefix_id": prefix_id,
            "observed_prefix_boundary_tokens": boundary,
        },
    )
    sched._register_observed_prefix(request)
    return prefix_id, boundary


def _wait_hint(
    sched,
    prefix_id,
    boundary,
    *,
    generation=0,
    session_id="session",
    event_id="a" * 64,
):
    published_ns = time.monotonic_ns()
    return {
        "schema": WAIT_HINT_SCHEMA,
        "event_id": event_id,
        "owner_id": "owner:future",
        "run_id": "run",
        "cell_id": "cell",
        "session_id": session_id,
        "generation": generation,
        "observed_prefix_id": prefix_id,
        "prefix_boundary_tokens": boundary,
        "observation_available_ns": published_ns,
        "input_publication_ns": published_ns,
        "declared_wait_ms": 250,
        "wait_provenance": WAIT_HINT_PROVENANCE,
        "namespace": dict(sched._wait_hint_namespace),
    }


def _run_cross_session_wait_nomination(
    selector, score_a, score_b, *, keys_per_session=16
):
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.outputs import KVConnectorOutput

    from semantic_offload.connector import SemanticOffloadingConnector

    manager = SemanticOffloadingManager(num_blocks=2 * keys_per_session)
    keys_a = [to_key(101 + index) for index in range(keys_per_session)]
    keys_b = [to_key(201 + index) for index in range(keys_per_session)]
    for block_id, key in enumerate(keys_a + keys_b):
        _insert_resident(manager, key, block_id=block_id)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(320, free_blocks=320)
    )
    sched._prefetch_selector = selector
    sched._prefetch_score_max_age_ns = 1_000_000_000
    sched._preservation_requests = {}
    sched._preservation_finished = {}
    prefix_a, boundary_a = _register_wait_prefix(
        sched, keys_a, session_id="session-a", request_id="warm-a"
    )
    prefix_b, boundary_b = _register_wait_prefix(
        sched, keys_b, session_id="session-b", request_id="warm-b"
    )
    hint_a = _wait_hint(
        sched,
        prefix_a,
        boundary_a,
        session_id="session-a",
        event_id="a" * 64,
    )
    hint_b = _wait_hint(
        sched,
        prefix_b,
        boundary_b,
        session_id="session-b",
        event_id="b" * 64,
    )
    assert sched.publish_wait_hint(hint_a)["status"] == "accepted"
    assert sched.publish_wait_hint(hint_b)["status"] == "accepted"
    observed_ns = time.monotonic_ns()
    evicted_key = to_key(10_000)
    ranked_scores = [
        (evicted_key, 100.0),
        *((key, score_a) for key in keys_a),
        *((key, score_b) for key in keys_b),
    ]
    metadata = SemanticWorkerMetadata(
        async_score_observations=[
            {
                "observation_sequence": 0,
                "request_ids": ["causal-query"],
                "scores": {"mean": {"causal-query": ranked_scores}},
                "snapshot_captured_ns": observed_ns,
                "observation_available_ns": observed_ns,
                "expires_at_ns": observed_ns + 1_000_000_000,
            }
        ],
        score_group_size=1,
        score_contributors=1,
    )
    connector = SemanticOffloadingConnector.__new__(
        SemanticOffloadingConnector
    )
    connector.connector_scheduler = sched
    with patch.object(
        OffloadingConnector, "update_connector_output", lambda self, output: None
    ):
        connector.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=metadata)
        )
    assert evicted_key not in manager.relevance_ema.get("mean", {})
    return sched, keys_a, keys_b, observed_ns


def _enable_wait_hint_coalescing(sched, *, max_wait_ms=25):
    sched._wait_hint_coalescing_min_sources = 2
    sched._wait_hint_coalescing_max_ns = max_wait_ms * 1_000_000
    sched._wait_hint_coalescing_started_ns = None


def test_wait_hint_coalescing_default_dispatches_without_gate_event():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=2)
    key = to_key(500)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched.publish_wait_hint(hint)
        sched._retry_wait_prefetches()

    assert sched._wait_prefetch[hint["event_id"]].status == "in_flight"
    assert not any(
        call.args[0].get("schema") == "semantic-wait-hint-coalescing-v1"
        for call in record.call_args_list
    )


@pytest.mark.parametrize("status", [None, "ready", "in_flight", "pending"])
def test_wait_nomination_without_candidates_still_expires_hints(status, monkeypatch):
    calls = []
    state = SimpleNamespace(status=status)
    states = {} if status is None else {"hint": state}

    def expire(now_ns):
        calls.append("expire")
        # An expired pending hint must disappear before nomination eligibility
        # is examined. In-flight/ready hints still get the same expiry visit.
        if status == "pending":
            states.clear()

    def unexpected(*args, **kwargs):
        raise AssertionError("empty nomination must not read the pool or emit a decision")

    scheduler = SimpleNamespace(
        _wait_prefetch=states,
        _prefetch_selector="semantic",
        _wait_hint_coalescing_gate=lambda now: (calls.append("gate") or False, None),
        _expire_wait_prefetches=expire,
        _prefetch_reserved_blocks=0, _prefetch_budget_fraction=0.2,
        _prefetch_headroom_blocks=0, _prefetch_bytes_per_key=256,
        gpu_block_pool=SimpleNamespace(num_gpu_blocks=100, get_num_free_blocks=unexpected),
    )
    monkeypatch.setattr("semantic_offload.connector.record_wait_hint_event", unexpected)
    SemanticOffloadingConnectorScheduler._retry_wait_prefetches(scheduler)
    assert calls == ["gate", "expire"]
    assert states == ({} if status in {None, "pending"} else {"hint": state})


def test_native_adoption_digest_reads_only_prefix(monkeypatch):
    class PrefixOnlyTokens(list):
        def __iter__(self):
            raise AssertionError("adoption must not copy the complete prompt")

        def __getitem__(self, index):
            assert index == slice(None, 32)
            return super().__getitem__(index)

    # Stop after the digest check: the allocator and DMA guards have their
    # separate integration coverage below. A mismatched digest must reject.
    now = time.monotonic_ns()
    request = SimpleNamespace(
        num_computed_tokens=0,
        prompt_token_ids=PrefixOnlyTokens(range(100_000)),
    )
    state = SimpleNamespace(
        status="ready", source_released=True, pending_contributors=0,
        bound_request_id="resume",
        transfer_record=dict(success=True, cancelled=False, contributors=1,
                             expected_contributors=1, completed_at_ns=now),
        keys=(b"a", b"b"), gpu_block_ids=(1, 2),
        lease=SimpleNamespace(
            state="ready", expires_at_ns=now + 10_000_000_000,
            hint=SimpleNamespace(prefix_boundary_tokens=32,
                                 observed_prefix_id="different-prefix"),
        ),
    )
    scheduler = SimpleNamespace(
        _wait_hint_native_adoption=True,
        _wait_hint_native_adoption_layout_supported=True,
        config=SimpleNamespace(num_workers=1, blocks_per_chunk=1,
                               kv_group_configs=[SimpleNamespace(
                                   group_idx=0, tokens_per_block=16,
                                   tokens_per_chunk=16)]),
        gpu_block_pool=SimpleNamespace(publish_completed_prefix=lambda *a, **kw: None),
        _req_status={"resume": SimpleNamespace(req=request)},
        _wait_hint_namespace={"model": "test"},
    )
    seen = []

    def digest(tokens, namespace):
        seen.append((tokens, namespace))
        return "actual-prefix"

    monkeypatch.setattr("semantic_offload.connector._observed_prefix_digest", digest)
    assert not SemanticOffloadingConnectorScheduler._try_publish_wait_prefix(scheduler, state)
    assert seen == [(list(range(32)), {"model": "test"})]


def test_wait_nomination_nonempty_keeps_decision_and_attempt(monkeypatch):
    events, attempts = [], []
    source = SimpleNamespace(run_id="run", cell_id="cell", session_id="session",
                             generation=0, observed_prefix_id="prefix")
    state = SimpleNamespace(status="pending", source=source,
                            lease=SimpleNamespace(active_at=lambda now: True,
                                                  hint=SimpleNamespace(event_id="hint")))
    candidate = dict(event_id="hint", session_id="session", observed_prefix_id="prefix",
                     source_request_id="source", aggregate_score=0.5,
                     eligible_resident_prefix_keys=["key"])
    scheduler = SimpleNamespace(
        _wait_prefetch={"hint": state}, _prefetch_selector="semantic",
        _wait_hint_coalescing_gate=lambda now: (False, None),
        _expire_wait_prefetches=lambda now: None,
        _semantic_wait_nomination_candidate=lambda state, now: (candidate, 0.5),
        _attempt_wait_prefetch=lambda state: attempts.append(state) or False,
        _prefetch_reserved_blocks=0, _prefetch_budget_fraction=0.2,
        _prefetch_headroom_blocks=0, _prefetch_bytes_per_key=256,
        gpu_block_pool=SimpleNamespace(num_gpu_blocks=100, get_num_free_blocks=lambda: 0),
    )
    scheduler._prefetch_free_capacity = (
        SemanticOffloadingConnectorScheduler._prefetch_free_capacity.__get__(scheduler)
    )
    scheduler._record_wait_hint_event = (
        SemanticOffloadingConnectorScheduler._record_wait_hint_event.__get__(scheduler)
    )
    monkeypatch.setattr("semantic_offload.connector.record_wait_hint_event", events.append)
    SemanticOffloadingConnectorScheduler._retry_wait_prefetches(scheduler)
    assert attempts == [state]
    assert len(events) == 1
    assert events[0]["candidates"] == [candidate]
    assert events[0]["chosen"] == []
    assert events[0]["no_action_reason"] == "no_causal_candidate_or_shared_budget"


def test_semantic_wait_nomination_reverses_winner_across_competing_sessions():
    from unittest.mock import patch

    winners = []
    frozen_candidate_inputs = []
    for scores in ((0.9, 0.1), (0.1, 0.9)):
        with (
            patch(
                "semantic_offload.connector.time.monotonic_ns",
                return_value=10_000_000_000,
            ),
            patch("semantic_offload.connector.record_wait_hint_event") as record,
        ):
            sched, _keys_a, _keys_b, observed_ns = (
                _run_cross_session_wait_nomination("semantic", *scores)
            )
            sched._retry_wait_prefetches()
        event = next(
            call.args[0]
            for call in record.call_args_list
            if call.args[0].get("schema")
            == "semantic-cross-session-prefetch-nomination-v1"
        )
        winners.append(event["chosen"][0]["session_id"])
        assert event["candidate_publication_order_event_ids"] == [
            "a" * 64,
            "b" * 64,
        ]
        assert event["shared_allocation_budget_blocks"] == 16
        assert event["eligible_candidate_blocks"] == 32
        assert event["shared_budget_constrained"] is True
        assert event["selected_blocks"] == 16
        assert event["selected_bytes"] == 16 * 128
        assert event["highest_ranked_candidate"]["session_id"] == winners[-1]
        frozen_candidate_inputs.append(
            [
                {
                    "event_id": candidate["event_id"],
                    "session_id": candidate["session_id"],
                    "observed_prefix_id": candidate["observed_prefix_id"],
                    "namespace": candidate["namespace"],
                    "eligible_resident_prefix_keys": candidate[
                        "eligible_resident_prefix_keys"
                    ],
                    "causal_inputs": candidate["causal_inputs"],
                }
                for candidate in sorted(
                    event["candidates"], key=lambda item: item["session_id"]
                )
            ]
        )
        assert all(
            score["observation_available_ns"] == observed_ns
            and score["observation_provenance"]
            == "async_worker_observation_available_ns"
            and score["fresh_at_decision"] is True
            and score["age_ns"]
            == event["decision_ns"] - score["observation_available_ns"]
            and score["feature_snapshot_captured_ns"] == observed_ns
            and score["feature_age_ns"]
            == event["decision_ns"] - score["feature_snapshot_captured_ns"]
            for candidate in event["candidates"]
            for score in candidate["semantic_key_scores"]
        )
        for candidate in event["candidates"]:
            recomputed = sum(
                item["score"] for item in candidate["semantic_key_scores"]
            ) / len(candidate["semantic_key_scores"])
            assert candidate["aggregate_score"] == pytest.approx(recomputed)
            assert candidate["causal_inputs"] == {
                "source_observed_ns": candidate["source_observed_ns"],
                "hint_observation_available_ns": candidate[
                    "hint_observation_available_ns"
                ],
                "hint_input_publication_ns": candidate[
                    "hint_input_publication_ns"
                ],
                "score_observation_available_ns": observed_ns,
                "score_max_age_ns": 1_000_000_000,
                "decision_ns": event["decision_ns"],
            }

    assert winners == ["session-a", "session-b"]
    assert frozen_candidate_inputs[0] == frozen_candidate_inputs[1]


def test_request_order_wait_nomination_preserves_publication_order():
    winners = []
    for scores in ((0.9, 0.1), (0.1, 0.9)):
        sched, _keys_a, _keys_b, _observed_ns = (
            _run_cross_session_wait_nomination(
            "request_order", *scores
            )
        )
        sched._retry_wait_prefetches()
        winner = next(
            state.lease.hint.session_id
            for state in sched._wait_prefetch.values()
            if state.status == "in_flight"
        )
        winners.append(winner)

    assert winners == ["session-a", "session-a"]


def test_six_surviving_blocks_do_not_claim_shared_budget_competition():
    from unittest.mock import patch

    sched, _keys_a, _keys_b, _observed_ns = (
        _run_cross_session_wait_nomination(
            "semantic", 0.9, 0.1, keys_per_session=3
        )
    )
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched._retry_wait_prefetches()
    event = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("schema")
        == "semantic-cross-session-prefetch-nomination-v1"
    )

    assert event["shared_allocation_budget_blocks"] == 16
    assert event["eligible_candidate_blocks"] == 6
    assert event["shared_budget_constrained"] is False
    assert event["selected_blocks"] == 6
    assert {item["session_id"] for item in event["chosen"]} == {
        "session-a",
        "session-b",
    }


def test_semantic_nomination_excludes_future_score_and_keeps_exact_chain_prefix():
    from unittest.mock import patch

    sched, keys_a, _keys_b, _observed_ns = _run_cross_session_wait_nomination(
        "semantic", 0.9, 0.1
    )
    for key in keys_a:
        sched.manager._relevance_updated_ns["mean"][key] = (
            time.monotonic_ns() + 1_000_000_000
        )
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched._retry_wait_prefetches()
    event = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("schema")
        == "semantic-cross-session-prefetch-nomination-v1"
    )
    assert event["chosen"][0]["session_id"] == "session-b"
    future_candidate = next(
        item for item in event["candidates"] if item["session_id"] == "session-a"
    )
    assert future_candidate["semantic_key_scores"] == []

    stale_sched, stale_keys, _other_keys, _observed_ns = (
        _run_cross_session_wait_nomination("semantic", 0.9, 0.1)
    )
    for key in stale_keys:
        stale_sched.manager._relevance_updated_ns["mean"][key] = (
            time.monotonic_ns() - 2_000_000_000
        )
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        stale_sched._retry_wait_prefetches()
    stale_event = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("schema")
        == "semantic-cross-session-prefetch-nomination-v1"
    )
    assert stale_event["chosen"][0]["session_id"] == "session-b"
    assert next(
        item
        for item in stale_event["candidates"]
        if item["session_id"] == "session-a"
    )["semantic_key_scores"] == []

    chain_sched, _chain_keys, other_keys, _observed_ns = (
        _run_cross_session_wait_nomination("semantic", 0.1, 0.9)
    )
    first, last = other_keys[0], other_keys[-1]
    chain_sched.manager.relevance_ema["mean"][first] = 0.01
    chain_sched.manager.relevance_ema["mean"][last] = 0.99
    chain_sched._retry_wait_prefetches()
    state = next(
        state
        for state in chain_sched._wait_prefetch.values()
        if state.lease.hint.session_id == "session-b"
    )
    assert state.keys == tuple(other_keys)
    assert state.selection_evidence["within_source_order"] == (
        "immutable_contiguous_chain_order"
    )


def test_future_worker_observation_is_rejected_before_manager_freshness():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.outputs import KVConnectorOutput

    from semantic_offload.connector import SemanticOffloadingConnector

    manager = SemanticOffloadingManager(num_blocks=2)
    key = to_key(401)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._preservation_requests = {}
    sched._preservation_finished = {}
    future_ns = time.monotonic_ns() + 1_000_000_000
    metadata = SemanticWorkerMetadata(
        async_score_observations=[
            {
                "observation_sequence": 0,
                "request_ids": ["future-query"],
                "scores": {"mean": {"future-query": [(key, 1.0)]}},
                "snapshot_captured_ns": future_ns,
                "observation_available_ns": future_ns,
                "expires_at_ns": future_ns + 1_000_000_000,
            }
        ],
        score_group_size=1,
        score_contributors=1,
    )
    connector = SemanticOffloadingConnector.__new__(
        SemanticOffloadingConnector
    )
    connector.connector_scheduler = sched
    with patch.object(
        OffloadingConnector, "update_connector_output", lambda self, output: None
    ):
        connector.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=metadata)
        )

    assert manager.relevance_ema.get("mean", {}) == {}
    assert manager._relevance_updated_ns.get("mean", {}) == {}


@pytest.mark.parametrize("selector", ["semantic", "request_order"])
def test_two_distinct_sources_release_same_generic_coalescing_gate(selector):
    from unittest.mock import patch

    sched, _keys_a, _keys_b, _observed_ns = (
        _run_cross_session_wait_nomination(selector, 0.1, 0.9)
    )
    _enable_wait_hint_coalescing(sched)
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched._retry_wait_prefetches()
    release = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("event") == "coalescing_released"
    )

    assert release["release_reason"] == "min_sources_reached"
    assert release["configured_min_sources"] == 2
    assert release["configured_max_wait_ns"] == 25_000_000
    assert release["distinct_eligible_sources"] == 2
    assert release["distinct_eligible_sessions"] == 2
    assert release["actual_delay_ns"] >= 0
    assert release["remaining_lead_ns"] > 0
    assert len(release["per_hint_delay"]) == 2
    in_flight = [
        state
        for state in sched._wait_prefetch.values()
        if state.status == "in_flight"
    ]
    assert len(in_flight) == 1
    assert in_flight[0].lease.hint.session_id == (
        "session-b" if selector == "semantic" else "session-a"
    )


def test_duplicate_exact_source_events_do_not_satisfy_coalescing_minimum():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=4)
    key = to_key(501)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._prefetch_selector = "request_order"
    _enable_wait_hint_coalescing(sched)
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    for event_id in ("a" * 64, "b" * 64):
        assert sched.publish_wait_hint(
            _wait_hint(sched, prefix_id, boundary, event_id=event_id)
        )["status"] == "accepted"

    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched._retry_wait_prefetches()

    started = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("event") == "coalescing_started"
    )
    assert started["distinct_eligible_sources"] == 1
    assert started["distinct_eligible_sessions"] == 1
    assert all(state.status == "pending" for state in sched._wait_prefetch.values())
    assert sched.has_pending_push_work() is True


def test_two_exact_prefixes_from_same_session_do_not_satisfy_minimum():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=4)
    keys = [to_key(510), to_key(511), to_key(512)]
    for block_id, key in enumerate(keys):
        _insert_resident(manager, key, block_id=block_id)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._prefetch_selector = "request_order"
    _enable_wait_hint_coalescing(sched)
    prefix_a, boundary_a = _register_wait_prefix(
        sched, keys[:1], session_id="same-session", request_id="warm-a"
    )
    prefix_b, boundary_b = _register_wait_prefix(
        sched, keys[1:], session_id="same-session", request_id="warm-b"
    )
    sched.publish_wait_hint(
        _wait_hint(
            sched,
            prefix_a,
            boundary_a,
            session_id="same-session",
            event_id="a" * 64,
        )
    )
    sched.publish_wait_hint(
        _wait_hint(
            sched,
            prefix_b,
            boundary_b,
            session_id="same-session",
            event_id="b" * 64,
        )
    )
    with patch("semantic_offload.connector.record_wait_hint_event") as record:
        sched._retry_wait_prefetches()

    started = next(
        call.args[0]
        for call in record.call_args_list
        if call.args[0].get("event") == "coalescing_started"
    )
    assert started["distinct_eligible_sources"] == 2
    assert started["distinct_eligible_sessions"] == 1
    assert all(state.status == "pending" for state in sched._wait_prefetch.values())


def test_idle_scheduler_live_path_releases_singleton_at_25ms_bound():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=4)
    key = to_key(502)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._prefetch_selector = "request_order"
    _enable_wait_hint_coalescing(sched)
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    sched.publish_wait_hint(_wait_hint(sched, prefix_id, boundary))
    scheduler_output = SimpleNamespace(preempted_req_ids=[])
    release_events = []
    started_at = time.monotonic_ns()
    with (
        patch.object(
            OffloadingConnectorScheduler,
            "build_connector_meta",
            return_value=OffloadingConnectorMetadata(
                load_jobs={}, store_jobs={}, jobs_to_flush=set()
            ),
        ),
        patch(
            "semantic_offload.connector.record_wait_hint_event",
            side_effect=lambda event: release_events.append(event),
        ),
    ):
        while time.monotonic_ns() - started_at < 150_000_000:
            metadata = sched.build_connector_meta(scheduler_output)
            if metadata.prefetch_load_jobs:
                break
    elapsed_ns = time.monotonic_ns() - started_at

    assert metadata.prefetch_load_jobs
    release = next(
        event
        for event in release_events
        if event.get("event") == "coalescing_released"
    )
    assert release["release_reason"] == "max_wait_elapsed"
    assert release["actual_delay_ns"] >= 25_000_000
    assert release["release_cutoff_ns"] == (
        release["started_at_ns"] + 25_000_000
    )
    assert release["release_cutoff_overshoot_ns"] == max(
        0, release["ended_at_ns"] - release["release_cutoff_ns"]
    )
    assert 25_000_000 <= elapsed_ns < 150_000_000
    assert sched.has_pending_push_work() is True


def test_idle_scheduler_earliest_deadline_retires_without_lease_extension():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=4)
    key = to_key(504)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(320, free_blocks=320))
    sched._prefetch_selector = "request_order"
    _enable_wait_hint_coalescing(sched, max_wait_ms=100)
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    hint["declared_wait_ms"] = 50
    sched.publish_wait_hint(hint)
    original_deadline = sched._wait_prefetch[
        hint["event_id"]
    ].lease.nomination_deadline_ns
    scheduler_output = SimpleNamespace(preempted_req_ids=[])
    events = []
    with (
        patch.object(
            OffloadingConnectorScheduler,
            "build_connector_meta",
            return_value=OffloadingConnectorMetadata(
                load_jobs={}, store_jobs={}, jobs_to_flush=set()
            ),
        ),
        patch(
            "semantic_offload.connector.record_wait_hint_event",
            side_effect=lambda event: events.append(event),
        ),
    ):
        while time.monotonic_ns() <= original_deadline + 50_000_000:
            sched.build_connector_meta(scheduler_output)
            if any(
                event.get("event") == "coalescing_released"
                for event in events
            ):
                break
            time.sleep(0.001)

    release = next(
        event for event in events if event.get("event") == "coalescing_released"
    )
    assert release["release_reason"] == "earliest_nomination_deadline"
    assert release["earliest_nomination_deadline_ns"] == original_deadline
    assert release["remaining_lead_ns"] == 0
    assert hint["event_id"] not in sched._wait_prefetch


@pytest.mark.parametrize(
    "fallback",
    ["cancelled", "demand_arrived", "no_free_allocation", "no_eligible_source"],
)
def test_coalescing_fallbacks_release_without_restart_or_demand_delay(fallback):
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=4)
    key = to_key(503)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(320, free_blocks=320)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    sched._preservation_requests = {}
    _enable_wait_hint_coalescing(sched)
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    events = []
    with patch(
        "semantic_offload.connector.record_wait_hint_event",
        side_effect=lambda event: events.append(event),
    ):
        sched._retry_wait_prefetches()
        state = sched._wait_prefetch[hint["event_id"]]
        if fallback == "cancelled":
            state.lease.cancel()
            sched._retry_wait_prefetches()
        elif fallback == "demand_arrived":
            request = SimpleNamespace(
                request_id="demand",
                kv_transfer_params={
                    "external_request_id": hint["owner_id"],
                    "wait_hint_event_id": hint["event_id"],
                    "run_id": hint["run_id"],
                    "cell_id": hint["cell_id"],
                    "session_id": hint["session_id"],
                    "generation": hint["generation"],
                    "observed_prefix_id": hint["observed_prefix_id"],
                    "observed_prefix_boundary_tokens": boundary,
                },
            )
            sched._bind_wait_hint_request(request)
            assert state.bound_request_id == "demand"
            assert state.status == "pending"
        elif fallback == "no_free_allocation":
            pool._free = 0
            sched._retry_wait_prefetches()
            assert state.coalescing_complete is True
            pool._free = 1
            sched._retry_wait_prefetches()
            assert state.status == "in_flight"
        else:
            manager._policy.remove(key)
            manager._num_evictable_cache_blocks -= 1
            sched._retry_wait_prefetches()
            assert state.coalescing_complete is True

    released = [
        event for event in events if event.get("event") == "coalescing_released"
    ]
    assert len(released) == 1
    assert released[0]["release_reason"] == fallback
    assert released[0]["actual_delay_ns"] >= 0
    assert released[0]["per_hint_delay"][0]["remaining_lead_ns"] >= 0


def test_released_allocation_retry_is_not_starved_by_new_singleton_window():
    manager = SemanticOffloadingManager(num_blocks=4)
    key_a, key_b = to_key(520), to_key(521)
    _insert_resident(manager, key_a, block_id=0)
    _insert_resident(manager, key_b, block_id=1)
    pool = _StubBlockPool(320, free_blocks=0)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    _enable_wait_hint_coalescing(sched)
    prefix_a, boundary_a = _register_wait_prefix(
        sched, [key_a], session_id="session-a", request_id="warm-a"
    )
    hint_a = _wait_hint(
        sched,
        prefix_a,
        boundary_a,
        session_id="session-a",
        event_id="a" * 64,
    )
    sched.publish_wait_hint(hint_a)
    sched._retry_wait_prefetches()
    state_a = sched._wait_prefetch[hint_a["event_id"]]
    assert state_a.coalescing_complete is True
    assert state_a.coalescing_end_reason == "no_free_allocation"

    prefix_b, boundary_b = _register_wait_prefix(
        sched, [key_b], session_id="session-b", request_id="warm-b"
    )
    hint_b = _wait_hint(
        sched,
        prefix_b,
        boundary_b,
        session_id="session-b",
        event_id="b" * 64,
    )
    sched.publish_wait_hint(hint_b)
    pool._free = 1
    sched._retry_wait_prefetches()

    assert state_a.status == "in_flight"
    assert sched._wait_prefetch[hint_b["event_id"]].status == "pending"


def test_diagnostic_no_dispatch_preserves_hint_but_allocates_no_staging():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    sched._wait_hint_diagnostic_no_dispatch = True
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)

    assert sched.publish_wait_hint(hint)["status"] == "accepted"
    sched._retry_wait_prefetches()

    assert sched._wait_prefetch[hint["event_id"]].status == "pending"
    assert pool.get_num_free_blocks() == 5
    assert not sched._wait_prefetch_jobs


def test_wait_hint_dispatches_real_native_pool_transfer_and_binds_resume():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)

    accepted = sched.publish_wait_hint(hint)
    replay = sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()

    assert accepted["status"] == "accepted"
    assert replay["idempotent_replay"] is True
    wait_state = sched._wait_prefetch[hint["event_id"]]
    assert wait_state.status == "in_flight"
    assert pool.get_num_free_blocks() == 4
    job_id = wait_state.job_id
    sched._preservation_job_events[job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    metadata = SemanticWorkerMetadata(completed_jobs={job_id: 1})
    sched._complete_wait_prefetch_jobs(metadata)
    assert wait_state.status == "ready"
    assert metadata.completed_jobs == {}

    _add_req_status(sched, "resume", keys=[key])
    resume = SimpleNamespace(
        request_id="resume",
        kv_transfer_params={
            "external_request_id": "owner:future",
            "wait_hint_event_id": hint["event_id"],
            "run_id": "run",
            "cell_id": "cell",
            "session_id": "session",
            "generation": 0,
            "observed_prefix_id": prefix_id,
            "observed_prefix_boundary_tokens": boundary,
        },
    )
    sched._preservation_requests = {"resume": {
        "prefetch_h2d_bytes": None,
        "prefetch_time_s": None,
        "prefetch_transfer_contributors": {},
        "transfer_job_statuses": [],
        "prefetch_attribution": {"attempts": [], "transfer_jobs": []},
        "prefetch_consumed_keys": 0,
        "prefetch_consumed_blocks": 0,
        "prefetch_discarded_keys": 0,
        "prefetch_discarded_blocks": 0,
    }}
    sched._bind_wait_hint_request(resume)
    preservation = sched._preservation_requests["resume"]
    assert preservation["prefetch_h2d_bytes"] == 128
    assert preservation["prefetch_time_s"] == 0.001
    assert preservation["transfer_job_statuses"] == []
    assert preservation["speculative_transfer_job_statuses"][0][
        "job_id"
    ] == job_id
    assert preservation["prefetch_attempted_keys"] == 1
    assert preservation["prefetch_attempted_blocks"] == 1
    assert preservation["prefetch_attribution"]["transfer_jobs"][0][
        "bytes"
    ] == 128
    assert "resume" in sched._prefetched
    assert sched._try_splice_prefetch(
        resume, _make_pending_blocks(1), 1
    )
    emitted = sched._prefetch_attribution_event(
        preservation, "partial", "test"
    )
    assert emitted["prefetch_h2d_bytes"] == 128
    assert emitted["gpu_splice_bytes"] == 128
    assert emitted["speculative_transfer"] == {
        "job_count": 1,
        "successful_job_count": 1,
        "failed_job_count": 0,
        "cancelled_job_count": 0,
        "attempted_blocks": 1,
        "attempted_bytes": 128,
        "submitted_bytes": 128,
        "submitted_jobs": 1,
        "submitted_blocks": 1,
        "completed_bytes": 128,
        "consumed_bytes": 128,
        "consumption_events": 1,
        "discarded_bytes": 0,
        "late_job_count": None,
        "late_bytes": None,
        "on_time_job_count": None,
        "on_time_bytes": None,
        "demand_boundary_ns": None,
        "demand_boundary_source": None,
        "released_reservation_bytes": 0,
        "uncompleted_submitted_bytes": 0,
        "unconsumed_completed_bytes": 0,
        "actual_contributors": 1,
        "expected_contributors": 1,
        "device_time_s": 0.001,
    }
    assert hint["event_id"] not in sched._wait_prefetch
    assert pool.get_num_free_blocks() == 4


def test_idle_scheduler_meta_dispatches_wait_hint_without_target_request():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    scheduler_output = SimpleNamespace(preempted_req_ids=[])

    with patch.object(
        OffloadingConnectorScheduler,
        "build_connector_meta",
        return_value=OffloadingConnectorMetadata(
            load_jobs={}, store_jobs={}, jobs_to_flush=set()
        ),
    ):
        metadata = sched.build_connector_meta(scheduler_output)

    assert metadata.load_jobs == {}
    assert len(metadata.prefetch_load_jobs) == 1
    job = next(iter(metadata.prefetch_load_jobs.values()))
    assert job.req_id.startswith("wait-hint:owner:future:")
    assert sched.has_pending_push_work()


def test_pending_splice_fence_keeps_engine_active_and_defers_terminal_event():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    sched = _make_connector_scheduler(
        SemanticOffloadingManager(num_blocks=10),
        _StubBlockPool(20, free_blocks=5),
    )
    sched._pending_splice_releases = {7: ([3], 1)}
    sched._preservation_finished = {
        "resume": {
            "prefetch_splice_ack_id": "splice-7",
            "prefetch_splice_acknowledged": False,
        }
    }
    sched._preservation_requests = {}

    with (
        patch.object(
            OffloadingConnectorScheduler,
            "has_pending_push_work",
            return_value=False,
        ),
        patch("semantic_offload.connector.record_preservation_event") as record,
    ):
        assert sched.has_pending_push_work() is True
        sched._emit_finished_preservation_events()

    record.assert_not_called()
    assert "resume" in sched._preservation_finished


def test_expired_inflight_wait_hint_releases_only_after_transfer_fence():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    state.lease.expires_at_ns = time.monotonic_ns() - 1

    sched._expire_wait_prefetches(time.monotonic_ns())

    assert state.status == "retired_pending_fence"
    assert pool.get_num_free_blocks() == 4
    sched._expire_wait_prefetches(time.monotonic_ns())
    assert state.status == "retired_pending_fence"
    assert pool.get_num_free_blocks() == 4
    assert sched._prefetch_reserved_blocks == 1
    job_id = state.job_id
    sched._preservation_job_events[job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    metadata = SemanticWorkerMetadata(completed_jobs={job_id: 1})
    sched._complete_wait_prefetch_jobs(metadata)
    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0
    assert hint["event_id"] not in sched._wait_prefetch


def test_wait_hint_wrong_generation_is_rejected_and_missing_source_is_no_action():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    prefix_id, boundary = _register_wait_prefix(sched, [key])

    result = sched.publish_wait_hint(
        _wait_hint(sched, prefix_id, boundary, generation=1)
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "superseded_generation"
    assert not sched.has_pending_push_work()

    missing = _wait_hint(sched, "f" * 64, boundary)
    missing["event_id"] = "b" * 64
    result = sched.publish_wait_hint(missing)
    assert result["status"] == "no_action"
    assert result["reason"] == "observed_prefix_unavailable"


def test_wait_hint_rejects_future_or_pre_source_causal_clocks():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    prefix_id, boundary = _register_wait_prefix(sched, [key])

    future = _wait_hint(sched, prefix_id, boundary)
    future["observation_available_ns"] += 10**12
    future["input_publication_ns"] += 10**12
    result = sched.publish_wait_hint(future)
    assert result["status"] == "rejected"
    assert result["reason"] == "causal_clock_mismatch"

    source = next(iter(sched._observed_prefixes.values()))
    stale = _wait_hint(sched, prefix_id, boundary)
    stale["observation_available_ns"] = source.observed_ns - 1
    result = sched.publish_wait_hint(stale)
    assert result["status"] == "rejected"
    assert result["reason"] == "causal_clock_mismatch"


def test_new_generation_retires_prior_hint_and_rejects_old_generation():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key], generation=0)
    old_hint = _wait_hint(sched, prefix_id, boundary, generation=0)
    assert sched.publish_wait_hint(old_hint)["status"] == "accepted"

    _register_wait_prefix(sched, [key], generation=1)

    assert old_hint["event_id"] not in sched._wait_prefetch
    stale_replay = sched.publish_wait_hint(old_hint)
    assert stale_replay["status"] == "rejected"
    assert stale_replay["idempotent_replay"] is True
    replay = dict(old_hint)
    replay["event_id"] = "b" * 64
    result = sched.publish_wait_hint(replay)
    assert result["status"] == "rejected"
    assert result["reason"] == "superseded_generation"


def test_bound_ready_generation_supersession_detaches_prefetch_alias_once():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    sched._preservation_job_events[state.job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    sched._complete_wait_prefetch_jobs(
        SemanticWorkerMetadata(completed_jobs={state.job_id: 1})
    )
    _add_req_status(sched, "resume", keys=[key])
    state.bound_request_id = "resume"
    sched._wait_hint_request_bindings["resume"] = hint["event_id"]
    sched._attach_ready_wait_prefetch(state)
    assert pool.get_num_free_blocks() == 4
    assert "resume" in sched._prefetched

    sched._retire_superseded_wait_hints(
        run_id="run",
        cell_id="cell",
        session_id="session",
        generation=1,
        now_ns=time.monotonic_ns(),
    )

    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0
    assert "resume" not in sched._prefetched
    assert "resume" not in sched._wait_hint_request_bindings
    sched._free_wait_prefetch(state, "repeated_cleanup")
    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0


@pytest.mark.parametrize("mode", ["zero_gap", "no_source", "retention_expired"])
def test_missing_wait_state_resume_rejects_without_crashing_on_new_request(mode):
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    if mode == "zero_gap":
        hint["declared_wait_ms"] = 0
    elif mode == "no_source":
        hint["observed_prefix_id"] = "f" * 64
    result = sched.publish_wait_hint(hint)
    if mode == "retention_expired":
        assert result["status"] == "accepted"
        state = sched._wait_prefetch[hint["event_id"]]
        state.lease.expires_at_ns = time.monotonic_ns() - 1
        sched._expire_wait_prefetches(time.monotonic_ns())
    else:
        assert result["status"] in {"expired", "no_action"}
    resume = SimpleNamespace(
        request_id=f"resume-{mode}",
        prompt_token_ids=[],
        kv_transfer_params={
            "external_request_id": "owner:future",
            "wait_hint_event_id": hint["event_id"],
            "run_id": "run",
            "cell_id": "cell",
            "session_id": "session",
            "generation": 0,
            "observed_prefix_id": hint["observed_prefix_id"],
            "observed_prefix_boundary_tokens": boundary,
        },
    )

    with (
        patch.object(OffloadingConnectorScheduler, "on_new_request"),
        patch.object(sched, "_register_lifecycle_request"),
    ):
        sched.on_new_request(resume)

    assert resume.request_id not in sched._wait_hint_request_bindings


@pytest.mark.parametrize("selector", ["request_order", "semantic", "useful_on_time"])
def test_wait_selectors_share_contiguous_resident_prefix_before_nomination(selector):
    manager = SemanticOffloadingManager(num_blocks=10)
    keys = [to_key(index) for index in range(3)]
    _insert_resident(manager, keys[0], block_id=0)
    _insert_resident(manager, keys[2], block_id=2)
    manager.relevance_ema.setdefault("mean", {})[keys[0]] = 1.0
    pool = _StubBlockPool(100, free_blocks=10)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = selector
    prefix_id, boundary = _register_wait_prefix(sched, keys)
    hint = _wait_hint(sched, prefix_id, boundary)

    accepted = sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()

    state = sched._wait_prefetch[hint["event_id"]]
    assert accepted["eligible_resident_prefix_count"] == 1
    assert state.keys == (keys[0],)
    assert state.selection_evidence[
        "decision_eligible_resident_prefix_count"
    ] == 1
    assert state.selection_evidence["eligible_resident_prefix_keys"] == [
        _event_key(keys[0])
    ]
    assert state.selection_evidence["eligibility_provenance"] == (
        "policy_get_ready_contiguous_prefix"
    )


def test_wait_hint_opportunity_excludes_actual_native_gpu_prefix_from_staging():
    manager = SemanticOffloadingManager(num_blocks=10)
    keys = [to_key(index) for index in range(3)]
    for block_id, key in enumerate(keys):
        _insert_resident(manager, key, block_id=block_id)
    pool = _StubBlockPool(
        100,
        free_blocks=10,
        cached_block_hashes={bytes(keys[0][:-4]): 9},
    )
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, keys)
    hint = _wait_hint(sched, prefix_id, boundary)

    accepted = sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()

    state = sched._wait_prefetch[hint["event_id"]]
    assert accepted["opportunity"] == {
        "source_key_count": 3,
        "local_gpu_prefix_count": 1,
        "local_gpu_block_ids": [9],
        "local_gpu_coverage_available": True,
        "staging_candidate_count": 2,
        "eligible_cpu_staging_prefix_count": 2,
        "positive_opportunity": True,
        "provenance": "native_block_pool_prefix_index_and_cpu_policy_ready_state",
    }
    assert state.publication_candidate_keys == tuple(keys[1:])
    assert state.keys == tuple(keys[1:])


def test_wait_hint_is_no_action_when_exact_prefix_is_already_local_gpu():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(
        20,
        free_blocks=5,
        cached_block_hashes={bytes(key[:-4]): 7},
    )
    sched = _make_connector_scheduler(manager, pool)
    prefix_id, boundary = _register_wait_prefix(sched, [key])

    result = sched.publish_wait_hint(_wait_hint(sched, prefix_id, boundary))

    assert result["status"] == "no_action"
    assert result["reason"] == "prefix_already_gpu_resident"
    assert result["opportunity"]["positive_opportunity"] is False
    assert sched._wait_prefetch == {}
    assert pool.get_num_free_blocks() == 5


def test_wait_hint_post_wait_retention_allows_scheduled_resume_to_consume():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    accepted = sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    job_id = state.job_id
    sched._preservation_job_events[job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    sched._complete_wait_prefetch_jobs(
        SemanticWorkerMetadata(completed_jobs={job_id: 1})
    )
    arrival_ns = (
        hint["input_publication_ns"] + 250_000_000 + 7 * 125_000_000
    )
    assert arrival_ns < accepted["expires_at_ns"]
    _add_req_status(sched, "resume", keys=[key])
    resume = SimpleNamespace(
        request_id="resume",
        kv_transfer_params={
            "external_request_id": "owner:future",
            "wait_hint_event_id": hint["event_id"],
            "run_id": "run",
            "cell_id": "cell",
            "session_id": "session",
            "generation": 0,
            "observed_prefix_id": prefix_id,
            "observed_prefix_boundary_tokens": boundary,
        },
    )

    with patch("semantic_offload.connector.time.monotonic_ns", return_value=arrival_ns):
        sched._bind_wait_hint_request(resume)
        consumed = sched._try_splice_prefetch(
            resume, _make_pending_blocks(1), 1
        )

    assert consumed is True
    assert hint["event_id"] not in sched._wait_prefetch


def test_driver_schedule_release_binds_and_consumes_completed_staging():
    from benchmarks.eviction_prefetch_campaign import execute_request_schedule

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    publication = {}
    requests = [
        {
            "request_id": "resume",
            "request_index": 0,
            "phase_index": 0,
            "relative_release_ms": 1,
        }
    ]

    async def before_phase(_phase_index, _requests, _phase_origin_ns):
        hint = _wait_hint(sched, prefix_id, boundary)
        hint["declared_wait_ms"] = 50
        publication.update(hint)
        assert sched.publish_wait_hint(hint)["status"] == "accepted"
        sched._retry_wait_prefetches()
        state = sched._wait_prefetch[hint["event_id"]]
        sched._preservation_job_events[state.job_id] = SemanticTransferCompletion(
            is_load=True,
            bytes=128,
            time_s=0.001,
            contributors=1,
            success=True,
        )
        sched._complete_wait_prefetch_jobs(
            SemanticWorkerMetadata(completed_jobs={state.job_id: 1})
        )
        return {"resume": hint["input_publication_ns"] + 50_000_000}

    async def send(_request):
        _add_req_status(sched, "resume", keys=[key])
        resume = SimpleNamespace(
            request_id="resume",
            kv_transfer_params={
                "external_request_id": "owner:future",
                "wait_hint_event_id": publication["event_id"],
                "run_id": "run",
                "cell_id": "cell",
                "session_id": "session",
                "generation": 0,
                "observed_prefix_id": prefix_id,
                "observed_prefix_boundary_tokens": boundary,
            },
        )
        sched._bind_wait_hint_request(resume)
        return {
            "success": sched._try_splice_prefetch(
                resume, _make_pending_blocks(1), 1
            )
        }

    results = asyncio.run(
        execute_request_schedule(
            requests,
            send,
            before_phase=before_phase,
        )
    )

    assert results[0]["success"] is True
    assert results[0]["send_ns"] >= publication["input_publication_ns"] + 51_000_000


def test_bounded_hint_soft_retention_touches_only_current_resident_sources():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    sched._prefetch_selector = "request_order"
    sched._soft_retention = "bounded_hint"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)

    with patch.object(manager, "touch", wraps=manager.touch) as touch:
        result = sched.publish_wait_hint(hint)

    assert result["status"] == "accepted"
    touch.assert_called_once()
    assert list(touch.call_args.args[0]) == [key]


def test_owned_reclaimable_staging_retires_ready_blocks_for_demand():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    sched._staging_mode = "owned_reclaimable"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    job_id = state.job_id
    sched._preservation_job_events[job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    sched._complete_wait_prefetch_jobs(
        SemanticWorkerMetadata(completed_jobs={job_id: 1})
    )
    assert state.status == "ready"
    assert pool.get_num_free_blocks() == 4
    demand = SimpleNamespace(request_id="demand", num_prompt_tokens=10)

    sched._reclaim_wait_prefetch_for_demand(
        demand, 0, protected_event_id=None
    )

    assert pool.get_num_free_blocks() == 5
    assert hint["event_id"] not in sched._wait_prefetch
    assert sched._prefetch_reserved_blocks == 0


def test_owned_reclaimable_staging_never_recycles_inflight_destination(capsys):
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    sched._staging_mode = "owned_reclaimable"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    demand = SimpleNamespace(request_id="demand", num_prompt_tokens=10)

    sched._reclaim_wait_prefetch_for_demand(
        demand, 0, protected_event_id=None
    )

    assert pool.get_num_free_blocks() == 4
    assert sched._wait_prefetch[hint["event_id"]].status == "in_flight"
    assert '"demand_wait_required":true' in capsys.readouterr().out


@pytest.mark.parametrize("prefetch_request_id", ["demand", "other"])
@pytest.mark.parametrize("pending_count", [None, 0, 1])
def test_demand_reclaims_completed_legacy_scratch_before_native_admission(
    prefetch_request_id, pending_count, capsys
):
    from vllm.v1.core.block_pool import BlockPool

    # One real null block plus a completed 256-block optional source leave
    # 527 free destinations: native 625-block admission cannot yet proceed.
    pool = BlockPool(784, enable_caching=True, hash_block_size=16)
    scratch = pool.get_new_blocks(256)
    manager = SemanticOffloadingManager(num_blocks=300)
    sched = _make_connector_scheduler(manager, pool)
    sched._staging_mode = "owned_reclaimable"
    sched.config.kv_group_configs[0].tokens_per_block = 16
    sched._prefetched[prefetch_request_id] = _PrefetchState(
        job_id=7, keys=[to_key(i) for i in range(256)],
        gpu_block_ids=[block.block_id for block in scratch],
    )
    sched._prefetch_reserved_blocks = 256
    if pending_count is not None:
        sched._jobs[7] = SimpleNamespace(pending_count=pending_count)
    assert pool.get_num_free_blocks() == 527
    demand = SimpleNamespace(request_id="demand", num_prompt_tokens=12288)

    sched._reclaim_wait_prefetch_for_demand(
        demand, 2288, protected_event_id=None
    )

    if pending_count == 1:
        assert pool.get_num_free_blocks() == 527
        assert sched._prefetch_reserved_blocks == 256
        assert all(block.ref_cnt == 1 for block in scratch)
        assert '"demand_wait_required":true' in capsys.readouterr().out
        return
    assert sched._prefetched == {}
    assert sched._prefetch_reserved_blocks == 0
    assert pool.get_num_free_blocks() == 783
    assert all(block.ref_cnt == 0 for block in scratch)
    # Native demand now has capacity. Every granted block has sole ownership;
    # the optional source is no longer a prerequisite for its own admission.
    admitted = pool.get_new_blocks(625)
    assert len(admitted) == len({block.block_id for block in admitted}) == 625
    assert all(block.ref_cnt == 1 for block in admitted)
    assert pool.get_num_free_blocks() == 158
    pool.free_blocks(admitted)


def test_demand_preserves_completed_legacy_scratch_when_destinations_fit():
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(784, enable_caching=True, hash_block_size=16)
    scratch = pool.get_new_blocks(256)
    sched = _make_connector_scheduler(SemanticOffloadingManager(num_blocks=300), pool)
    sched._staging_mode = "owned_reclaimable"
    sched._prefetched["demand"] = _PrefetchState(
        job_id=7, keys=[], gpu_block_ids=[block.block_id for block in scratch]
    )
    sched._prefetch_reserved_blocks = 256

    sched._reclaim_wait_prefetch_for_demand(
        SimpleNamespace(request_id="demand", num_prompt_tokens=500),
        0, protected_event_id=None,
    )

    assert "demand" in sched._prefetched
    assert pool.get_num_free_blocks() == 527
    assert all(block.ref_cnt == 1 for block in scratch)


def test_useful_on_time_scheduler_uses_causal_completed_cost_and_wait_budget():
    manager = SemanticOffloadingManager(num_blocks=10)
    keys = [to_key(index) for index in range(3)]
    for block_id, key in enumerate(keys):
        _insert_resident(manager, key, block_id=block_id)
    pool = _StubBlockPool(100, free_blocks=10)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "useful_on_time"
    sched._prefetch_bytes_per_key = 128
    sched._prefetch_transfer_ns_per_byte = 1_000_000.0
    sched._prefetch_cost_available_ns = time.monotonic_ns() - 1
    prefix_id, boundary = _register_wait_prefix(sched, keys)
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)

    sched._retry_wait_prefetches()

    state = sched._wait_prefetch[hint["event_id"]]
    assert state.keys == (keys[0],)
    assert state.selection_evidence["transfer_cost_provenance"] == (
        "completed_prefetch_transfer_ema"
    )
    assert state.selection_evidence["formula"] == (
        "longest_eligible_chain_prefix_with_cumulative_"
        "transfer_plus_splice_cost_le_remaining_wait"
    )


@pytest.mark.parametrize("fraction,expected", [(0.01, 1), (0.05, 5)])
def test_wait_staging_caps_native_pool_at_frozen_fraction(fraction, expected):
    manager = SemanticOffloadingManager(num_blocks=20)
    keys = [to_key(index) for index in range(10)]
    for block_id, key in enumerate(keys):
        _insert_resident(manager, key, block_id=block_id)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(100, free_blocks=20)
    )
    sched._prefetch_budget_fraction = fraction
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, keys)
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)

    sched._retry_wait_prefetches()

    state = sched._wait_prefetch[hint["event_id"]]
    assert len(state.gpu_block_ids) == expected
    assert sched._prefetch_reserved_blocks == expected


def test_wait_hint_reset_flushes_inflight_job_and_releases_only_on_ack():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    job_id = state.job_id
    sched._current_batch_prefetch_load_jobs.clear()  # worker owns transfer

    with patch.object(OffloadingConnectorScheduler, "reset_cache"):
        sched.reset_cache()

    assert pool.get_num_free_blocks() == 4
    assert sched._reset_prefetch_releases == {job_id: ([1], 1)}
    output = SimpleNamespace(
        kv_connector_worker_meta=SemanticWorkerMetadata(completed_jobs={job_id: 1})
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(output)
    assert pool.get_num_free_blocks() == 5
    assert sched._reset_prefetch_releases == {}


def test_retired_pending_fence_reset_and_repeated_ack_release_exactly_once():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    state.lease.expires_at_ns = time.monotonic_ns() - 1
    sched._expire_wait_prefetches(time.monotonic_ns())
    sched._expire_wait_prefetches(time.monotonic_ns())
    job_id = state.job_id
    sched._current_batch_prefetch_load_jobs.clear()

    with patch.object(OffloadingConnectorScheduler, "reset_cache"):
        sched.reset_cache()

    assert pool.get_num_free_blocks() == 4
    assert sched._reset_prefetch_releases == {job_id: ([1], 1)}
    output = SimpleNamespace(
        kv_connector_worker_meta=SemanticWorkerMetadata(completed_jobs={job_id: 1})
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(output)
        assert pool.get_num_free_blocks() == 5
        sched.update_connector_output(output)

    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0


def test_bound_request_abort_keeps_inflight_hint_blocks_until_fence():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    prefix_id, boundary = _register_wait_prefix(sched, [key])
    hint = _wait_hint(sched, prefix_id, boundary)
    sched.publish_wait_hint(hint)
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    _add_req_status(sched, "resume", keys=[key])
    resume = SimpleNamespace(
        request_id="resume",
        prompt_token_ids=[],
        status=SimpleNamespace(name="FINISHED_ABORTED"),
        kv_transfer_params={
            "external_request_id": "owner:future",
            "wait_hint_event_id": hint["event_id"],
            "run_id": "run",
            "cell_id": "cell",
            "session_id": "session",
            "generation": 0,
            "observed_prefix_id": prefix_id,
            "observed_prefix_boundary_tokens": boundary,
        },
    )
    sched._bind_wait_hint_request(resume)

    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    with patch.object(
        OffloadingConnectorScheduler,
        "request_finished",
        return_value=(False, None),
    ):
        sched.request_finished(resume)

    assert state.status == "retired_pending_fence"
    assert pool.get_num_free_blocks() == 4
    job_id = state.job_id
    sched._preservation_job_events[job_id] = SemanticTransferCompletion(
        is_load=True,
        bytes=128,
        time_s=0.001,
        contributors=1,
        success=True,
    )
    sched._complete_wait_prefetch_jobs(
        SemanticWorkerMetadata(completed_jobs={job_id: 1})
    )
    assert pool.get_num_free_blocks() == 5


def test_attempt_prefetch_reservation_stops_retrying_when_no_candidate_keys():
    """A request with no offload keys at all can never gain any -- the
    caller must not keep retrying it forever."""
    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[])

    assert sched._attempt_prefetch_reservation("r1") is True


def test_attempt_prefetch_reservation_retries_when_no_free_blocks():
    """issues log entry #25: a request with real, resident, scored keys but
    zero free GPU blocks right now must be retried later, not given up on."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is False
    assert "r1" not in sched._prefetched


def test_attempt_prefetch_reservation_succeeds_once_blocks_free_up():
    """The exact retry scenario issues log entry #25 targets: first attempt
    fails (no free blocks), a later attempt with free blocks available
    succeeds."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    pool = _StubBlockPool(20, free_blocks=0)
    sched = _make_connector_scheduler(manager, pool)
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is False

    pool._free = 5  # simulate blocks freeing up on a later scheduling step
    assert sched._attempt_prefetch_reservation("r1") is True
    assert "r1" in sched._prefetched


def test_successful_reservation_registers_job_in_req_status_transfer_jobs():
    """issues log entry #26: a dispatched prefetch job MUST be registered in
    req_status.transfer_jobs, or update_connector_output's generic
    completion loop (`req_status.transfer_jobs.remove(job_id)`,
    unconditional for every job in self._jobs) crashes with a KeyError the
    moment the job completes -- confirmed on the real server. This also
    relies on get_num_new_matched_tokens's existing "delay re-admission
    while transfer_jobs is non-empty" guard to stay safe against
    update_state_after_alloc's `assert not req_status.transfer_jobs`."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    job_id = sched._prefetched["r1"].job_id
    assert job_id in sched._req_status["r1"].transfer_jobs
    assert sched._prefetched["r1"].keys == [key]


def test_prefetch_reservation_rolls_back_if_prepare_load_fails():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    pool = _StubBlockPool(20, free_blocks=5)
    sched = _make_connector_scheduler(manager, pool)
    _add_req_status(sched, "r1", keys=[key])

    with (
        patch.object(
            manager,
            "prepare_load",
            side_effect=RuntimeError("injected prepare failure"),
        ),
        pytest.raises(RuntimeError, match="injected prepare failure"),
    ):
        sched._attempt_prefetch_reservation("r1")

    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0
    assert sched._prefetched == {}
    assert sched._current_batch_prefetch_load_jobs == {}
    assert sched._jobs == {}
    assert sched._req_status["r1"].transfer_jobs == set()


def test_prefetch_headroom_blocks_reservation_under_pressure():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    pool = _StubBlockPool(20, free_blocks=3)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_headroom_blocks = 3
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is False
    assert pool.get_num_free_blocks() == 3
    assert sched._prefetched == {}


def test_prefetch_attempt_joins_decision_snapshot_to_actual_load_plan():
    state = {"prefetch_attribution": {"attempts": []}}
    keys = [to_key(1), to_key(2), to_key(3)]

    _append_prefetch_attempt(
        state,
        attempt_id="attempt-1",
        episode_id=2,
        selector="request_order",
        candidate_keys=keys,
        selected_keys=keys[:2],
        cpu_resident_ready_keys=keys[:2],
        reservation_id="reservation-1",
        reservation_block_ids=[7, 8],
        reservation_submit_time_s=10.0,
    )
    _update_prefetch_attempt_plan(
        state,
        "attempt-1",
        [keys[1], keys[2]],
        status="ready_at_readmission",
        actual_destination_block_ids=[100, 101],
    )

    attempt = state["prefetch_attribution"]["attempts"][0]
    assert attempt["selector"] == "request_order"
    assert attempt["actual_load_plan_keys"] == [
        _event_key(keys[1]),
        _event_key(keys[2]),
    ]
    assert attempt["intersection_keys"] == [_event_key(keys[1])]
    assert attempt["remainder_keys"] == [_event_key(keys[2])]
    assert attempt["actual_destination_block_ids"] == ["100", "101"]


def test_retry_pending_prefetches_removes_only_the_succeeded_request():
    manager = SemanticOffloadingManager(num_blocks=10)
    key1, key2 = to_key(1), to_key(2)
    _insert_resident(manager, key1, block_id=0)
    _insert_resident(manager, key2, block_id=1)
    manager.relevance_ema.setdefault("mean", {})[key1] = 1.0
    manager.relevance_ema.setdefault("mean", {})[key2] = 1.0
    # Only one free block: only one of the two pending requests can succeed.
    pool = _StubBlockPool(20, free_blocks=1)
    sched = _make_connector_scheduler(manager, pool)
    _add_req_status(sched, "r1", keys=[key1])
    _add_req_status(sched, "r2", keys=[key2])
    sched._preempted_pending = {"r1", "r2"}

    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids=set()))

    succeeded = sched._preempted_pending ^ {"r1", "r2"}
    assert len(succeeded) == 1
    assert len(sched._preempted_pending) == 1  # the other one stays queued
    assert set(sched._prefetched.keys()) == succeeded


def test_on_request_preempted_queues_for_retry_on_failure():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[key])

    sched.on_request_preempted(SimpleNamespace(request_id="r1"))

    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched


def test_on_request_preempted_never_reserves_synchronously():
    """issues log entry #27 (Fix B): even with free blocks available right
    now, on_request_preempted must only queue, never call
    _attempt_prefetch_reservation directly -- reserving synchronously risks
    tripping build_connector_meta's `assert is_store` flush for a request
    still in this same step's preempted_req_ids (a load job would fail that
    assert). The actual attempt only ever happens from
    _retry_pending_prefetches, on a LATER step."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    sched.on_request_preempted(SimpleNamespace(request_id="r1"))

    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched  # not reserved yet, despite free blocks


def test_on_request_preempted_is_noop_when_prefetch_disabled():
    """SEMANTIC_OFFLOAD_DISABLE_PREFETCH (issues log entry #53's follow-up,
    diagnosing whether prefetch's speculative block reservation is itself
    causing extra preemptions under tight capacity): with the flag on,
    on_request_preempted must match the base KVConnectorBase_V1 no-op lru
    gets -- nothing queued, nothing tracked."""
    from unittest.mock import patch

    import semantic_offload.connector as connector_mod

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    with patch.object(connector_mod, "_DISABLE_PREFETCH", True):
        sched.on_request_preempted(SimpleNamespace(request_id="r1"))

    assert sched._preempted_pending == set()


def test_queue_preempted_seeds_pending_from_scheduler_output():
    """vLLM removed the on_request_preempted scheduler hook this class used
    to override (confirmed absent from the whole vllm/ tree as of
    2026-07-29 -- replaced by SchedulerOutput.preempted_req_ids, read from
    build_connector_meta instead). _queue_preempted is the replacement
    trigger; same queue-only contract as on_request_preempted above."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=0))
    _add_req_status(sched, "r1", keys=[key])

    sched._queue_preempted(SimpleNamespace(preempted_req_ids={"r1"}))

    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched


def test_queue_preempted_is_noop_when_prefetch_disabled():
    from unittest.mock import patch

    import semantic_offload.connector as connector_mod

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    with patch.object(connector_mod, "_DISABLE_PREFETCH", True):
        sched._queue_preempted(SimpleNamespace(preempted_req_ids={"r1"}))

    assert sched._preempted_pending == set()


def test_retry_pending_prefetches_skips_requests_preempted_this_step():
    """issues log entry #27 (Fix B): a request preempted THIS scheduling
    step must not be attempted even during the retry sweep -- it only
    becomes eligible on a later step, once it's no longer in
    scheduler_output.preempted_req_ids."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])
    sched._preempted_pending = {"r1"}

    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids={"r1"}))
    assert "r1" in sched._preempted_pending
    assert "r1" not in sched._prefetched

    # Next step, no longer in preempted_req_ids -- now eligible.
    sched._retry_pending_prefetches(SimpleNamespace(preempted_req_ids=set()))
    assert "r1" not in sched._preempted_pending
    assert "r1" in sched._prefetched


def test_successful_reservation_records_allocated_block_ids():
    """issues log entry #27 (Fix C): reserved blocks must be recorded in
    _current_batch_allocated_block_ids, or build_connector_meta's
    store-flush fence can't see that a block a finished request's pending
    store still reads from was just reallocated to the prefetch -- a silent
    data race, not a crash."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    gpu_block_ids = set(sched._prefetched["r1"].gpu_block_ids)
    assert gpu_block_ids
    assert gpu_block_ids <= sched._current_batch_allocated_block_ids


def test_successful_reservation_dispatches_to_prefetch_channel_not_load_jobs():
    """issues log entry #27 (Fix A): the job must go into
    _current_batch_prefetch_load_jobs, NOT _current_batch_load_jobs -- the
    latter becomes `load_jobs` in the metadata, which the stock worker's
    start_kv_transfers registers into self._load_jobs, causing its
    completion to be reported via finished_recving and crash the engine for
    a PREEMPTED (not WAITING_FOR_REMOTE_KVS, not finished) request."""
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(1)
    _insert_resident(manager, key, block_id=0)
    manager.relevance_ema.setdefault("mean", {})[key] = 1.0
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])

    assert sched._attempt_prefetch_reservation("r1") is True

    job_id = sched._prefetched["r1"].job_id
    assert job_id in sched._current_batch_prefetch_load_jobs
    assert job_id not in sched._current_batch_load_jobs


def test_worker_counts_prefetch_completion_as_physical_load_not_store():
    class Recorder:
        def __init__(self):
            self.records = []

        def record(self, size, duration):
            self.records.append((size, duration))

    load = Recorder()
    store = Recorder()
    completed = []
    connector = SemanticOffloadingConnectorWorker.__new__(
        SemanticOffloadingConnectorWorker
    )
    connector.worker = SimpleNamespace(
        _preservation_events=True,
        get_finished=lambda: [
            SimpleNamespace(
                job_id=7,
                success=True,
                transfer_size=128,
                transfer_time=0.001,
            )
        ]
    )
    connector._connector_worker_meta = SimpleNamespace(
        transfer_stats=SimpleNamespace(load=load, store=store),
        failed_jobs={},
        mark_completed=completed.append,
    )
    connector._submission_failed_recving = set()
    connector._load_jobs = {}
    connector._semantic_prefetch_load_job_ids = {7}

    sending, receiving = connector.get_finished(set())

    assert (sending, receiving) == (set(), set())
    assert load.records == [(128, 0.001)]
    assert store.records == []
    assert completed == [7]
    assert connector._semantic_prefetch_load_job_ids == set()


@pytest.mark.parametrize(("consumed_blocks", "discarded_blocks"), [(1, 0), (0, 1)])
def test_reactive_prefetch_completion_does_not_mutate_demand_load_state(
    consumed_blocks, discarded_blocks
):
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    from benchmarks.preservation import _validate_speculative_transfer_evidence

    key = to_key(1)
    scheduler = _make_connector_scheduler(
        SemanticOffloadingManager(num_blocks=10),
        _StubBlockPool(20, free_blocks=5),
    )
    scheduler._preservation_events_enabled = True
    state = {
        "transfer_complete": True,
        "physical_block_identity_complete": True,
        "prefetch_failure_reason": None,
        "prefetch_cancelled": False,
        "prefetch_transfer_contributors": {},
        "speculative_transfer_job_statuses": [],
        "transfer_job_statuses": [],
        "prefetch_attribution": {"attempts": [], "transfer_jobs": []},
        "prefetch_t2_ns": None,
        "prefetch_t3_ns": None,
        "prefetch_fence_ns": None,
        "prefetch_fence_complete": False,
        "prefetch_h2d_bytes": None,
        "prefetch_time_s": None,
        "prefetch_attempted_keys": 1,
        "prefetch_attempted_blocks": 1,
        "prefetch_consumed_keys": consumed_blocks,
        "prefetch_consumed_blocks": consumed_blocks,
        "prefetch_discarded_keys": discarded_blocks,
        "prefetch_discarded_blocks": discarded_blocks,
        "prefetch_block_pool_snapshots": [],
        "prefetch_block_pool_snapshot_count": 0,
        "prefetch_block_pool_snapshots_truncated": False,
        "loaded_block_ids": set(),
        "loaded_chunks": 0,
        "load_bytes": 0,
        "load_time_complete": True,
        "load_time_observed": False,
        "load_time_s": 0.0,
        "load_critical_path_s": 0.0,
        "load_contributors": 0,
        "load_job_count": 0,
        "load_expected_total": 0,
        "first_admission": {"completed_load_keys": []},
    }
    scheduler._preservation_requests = {"r1": state}
    scheduler._preservation_finished = {}
    scheduler._preservation_jobs = {
        7: {
            "req_id": "r1",
            "is_load": True,
            "is_prefetch": True,
            "expected_contributors": 1,
            "chunks": 1,
            "dst_block_ids": [(0, 3)],
            "src_block_ids": None,
            "dst_block_ids_complete": True,
            "src_block_ids_complete": False,
            "admission_index": 0,
            "key_positions": [{"key": _event_key(key)}],
            "keys": [key],
        }
    }
    scheduler._preservation_job_events = {}
    scheduler._jobs = {
        7: TransferJobStatus(
            req_id="r1", pending_count=1, keys={key}, is_store=False
        )
    }
    scheduler._reset_prefetch_releases = {}
    metadata = SemanticWorkerMetadata(
        completed_jobs={7: 1},
        transfer_events={
            7: SemanticTransferCompletion(
                is_load=True,
                bytes=128,
                time_s=0.001,
                max_time_s=0.001,
                contributors=1,
                success=True,
                started_at_ns=10,
                completed_at_ns=20,
            )
        },
    )
    output = SimpleNamespace(kv_connector_worker_meta=metadata)

    with (
        patch.object(OffloadingConnectorScheduler, "update_connector_output"),
        patch.object(scheduler, "_cleanup_prefetch_output"),
        patch.object(scheduler, "_emit_finished_preservation_events"),
        patch.object(scheduler, "_record_block_pool_snapshot"),
    ):
        scheduler.update_connector_output(output)

    assert len(state["speculative_transfer_job_statuses"]) == 1
    assert state["prefetch_h2d_bytes"] == 128
    assert state["transfer_job_statuses"] == []
    assert state["loaded_block_ids"] == set()
    assert state["loaded_chunks"] == 0
    assert state["load_bytes"] == 0
    assert state["load_time_s"] == 0.0
    assert state["load_contributors"] == 0
    assert state["load_job_count"] == 0
    assert state["first_admission"]["completed_load_keys"] == []
    accounting = _speculative_transfer_accounting(state, 128)
    attribution = {
        "provenance": {"config": {"transfer_bytes_per_key": 128}},
        "transfer_jobs": state["speculative_transfer_job_statuses"],
        "speculative_transfer": accounting,
        "prefetch_h2d_bytes": 128,
    }
    strict_event = {
        "transfer_job_statuses": [],
        "speculative_transfer_job_statuses": state[
            "speculative_transfer_job_statuses"
        ],
        "speculative_transfer_accounting": accounting,
        "prefetch_counts": {
            "attempted": {"keys": 1, "blocks": 1},
            "consumed": {
                "keys": consumed_blocks,
                "blocks": consumed_blocks,
            },
            "discarded": {
                "keys": discarded_blocks,
                "blocks": discarded_blocks,
            },
        },
        "prefetch_attribution": attribution,
    }
    reason, mapped_accounting = _validate_speculative_transfer_evidence(
        strict_event
    )
    assert reason is None
    assert mapped_accounting == accounting


# --- Partial-splice tests (issues log entries #31-32 + the partial-splice
# plan doc: matching prefetched content to destination blocks by OffloadKey
# identity, never by list position, and coordinating a splice + a reduced
# remainder reload for the same request in one step). `request` is a
# SimpleNamespace with only `.request_id` since that's all
# _try_splice_prefetch reads from it. ---


def test_partial_splice_identity_matches_even_with_reordered_prefetch():
    """Q3-1: `prefetch.keys` is relevance-ranked, `keys_to_load` is
    positional -- a prefetch that fully covers the request but in a
    DIFFERENT order must still splice each key's content into ITS OWN
    destination block, not the positionally-parallel one (the exact bug
    entries #31-32 found and fixed)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1, k2])
    # Prefetch covers all three but in reversed (relevance) order: k2's
    # content is in GPU block 52, k1's in 51, k0's in 50.
    sched._prefetched["r1"] = _PrefetchState(
        job_id=1, keys=[k2, k1, k0], gpu_block_ids=[52, 51, 50]
    )
    sched._preservation_requests = {
        "r1": {
            "prefetch_attribution": {"attempts": []},
            "prefetch_consumed_keys": 0,
            "prefetch_consumed_blocks": 0,
            "prefetch_discarded_keys": 0,
            "prefetch_discarded_blocks": 0,
        }
    }
    blocks = _make_pending_blocks(3)  # dst block_ids [100, 101, 102]

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 3) is True
    )

    # keys_to_load is positionally [k0, k1, k2] -> dst [100, 101, 102].
    # k0's real content lives in GPU block 50, k1's in 51, k2's in 52 --
    # NOT the positionally-parallel [52, 51, 50] a naive zip would produce.
    src_ids, dst_ids = sched._pending_splice_jobs[0]
    assert (src_ids, dst_ids) == ([50, 51, 52], [100, 101, 102])
    assert "r1" not in sched._current_batch_load_jobs  # full cover, no remainder
    assert sched._preservation_requests["r1"]["prefetch_splice_bytes"] == 384


def test_partial_splice_uses_real_request_group_progress_field():
    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[key])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[key], gpu_block_ids=[7])

    assert sched._try_splice_prefetch(
        SimpleNamespace(request_id="r1"), _make_pending_blocks(1), 1
    )

    state = sched._req_status["r1"].group_states[0]
    assert state.next_stored_chunk_idx == 1
    assert "next_stored_block_idx" not in state.__dict__


def test_prefetch_blocks_remain_reserved_until_all_workers_ack_splice():
    from unittest.mock import patch

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    manager = SemanticOffloadingManager(num_blocks=10)
    key = to_key(0)
    pool = _StubBlockPool(20, free_blocks=4)
    sched = _make_connector_scheduler(manager, pool)
    sched.config.num_workers = 2
    _add_req_status(sched, "r1", keys=[key])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[key], gpu_block_ids=[7])
    sched._prefetch_reserved_blocks = 1

    assert sched._try_splice_prefetch(
        SimpleNamespace(request_id="r1"), _make_pending_blocks(1), 1
    )
    assert pool.get_num_free_blocks() == 4
    assert sched._prefetch_reserved_blocks == 1
    assert 0 in sched._pending_splice_releases

    output = SimpleNamespace(
        kv_connector_worker_meta=SemanticWorkerMetadata(completed_splices={0: 1}),
        finished_recving=None,
    )
    with patch.object(OffloadingConnectorScheduler, "update_connector_output"):
        sched.update_connector_output(output)
        assert pool.get_num_free_blocks() == 4
        assert sched._prefetch_reserved_blocks == 1
        assert output.finished_recving is None
        sched.update_connector_output(output)

    assert pool.get_num_free_blocks() == 5
    assert sched._prefetch_reserved_blocks == 0
    assert sched._pending_splice_releases == {}
    assert sched._pending_splice_request_ids == {}
    assert output.finished_recving == {"r1"}


def test_partial_splice_splits_into_splice_and_remainder_disjointly():
    """Q3-2: a prefetch covering only SOME of the needed keys must splice
    the covered ones and issue a normal reduced reload for the rest, with
    disjoint destination sets (guards the double-write race hazard)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    _insert_resident(manager, k0, block_id=0)
    _insert_resident(manager, k2, block_id=1)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1, k2])
    # Prefetch covers only the MIDDLE key, k1 -- the common real-world case
    # (entries #29-32) where only a fraction of a request's blocks ever
    # earned a relevance score in time.
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k1], gpu_block_ids=[77])
    blocks = _make_pending_blocks(3)  # dst block_ids [100, 101, 102]

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 3) is True
    )

    src_ids, dst_ids = sched._pending_splice_jobs[0]
    assert (src_ids, dst_ids) == ([77], [101])  # k1 spliced into its own dst

    assert len(sched._current_batch_load_jobs) == 1
    (job,) = sched._current_batch_load_jobs.values()
    remainder_dst = list(job.dst_spec.block_ids)
    assert remainder_dst == [100, 102]  # k0, k2's destinations
    assert set(dst_ids).isdisjoint(remainder_dst)

    job_id = next(iter(sched._current_batch_load_jobs))
    assert job_id in sched._req_status["r1"].transfer_jobs
    assert sched._jobs[job_id].keys == {k0, k2}
    assert sched._pending_splice_request_ids == {}


def test_partial_splice_remainder_keys_tracked_spliced_keys_not():
    """Q3-4/Q3-5: only the REMAINDER keys go into the reduced load job's
    `keys` and `_blocks_being_loaded` -- the spliced key must appear in
    neither (it was never given a TransferJob, so nothing will ever remove
    it from `_blocks_being_loaded` if it were added, leaking a permanent
    false-deferral for any future request sharing that key)."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    _insert_resident(manager, k1, block_id=0)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k0], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is True
    )

    assert sched._blocks_being_loaded == {k1}
    assert k0 not in sched._blocks_being_loaded


def test_partial_splice_full_coverage_no_remainder_job_created():
    """Regression: when the prefetch covers everything (today's original
    exact-match case, now just the `remainder` == empty branch of the same
    code path), behavior matches the pre-partial-splice mechanism -- only a
    splice job, no reload job, no transfer_jobs entry."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(
        job_id=1, keys=[k0, k1], gpu_block_ids=[50, 51]
    )
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is True
    )

    assert 0 in sched._pending_splice_jobs
    assert not sched._current_batch_load_jobs
    assert not sched._req_status["r1"].transfer_jobs
    assert sched._pending_splice_request_ids == {0: "r1"}


def test_full_cover_preserves_readmission_demand_boundary_before_splice_mutation():
    from unittest.mock import patch

    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    sched._preservation_events_enabled = True
    state = {"readmission_pending": True, "episode_id": 3}
    sched._preservation_requests = {"r1": state}
    request = SimpleNamespace(request_id="r1")
    blocks = _make_pending_blocks(1)

    def consume_and_clear(*_args):
        state["readmission_pending"] = False
        return True

    with (
        patch.object(sched, "_record_block_pool_snapshot"),
        patch.object(sched, "_try_splice_prefetch", side_effect=consume_and_clear),
        patch.object(sched, "_record_admission") as record_admission,
        patch("semantic_offload.connector.time.monotonic_ns", return_value=1234),
    ):
        sched.update_state_after_alloc(request, blocks, 1)

    assert state["prefetch_t4_ns"] == 1234
    assert state["prefetch_t4_source"] == "scheduler_update_state_after_alloc_entry"
    assert state["prefetch_t4_episode_id"] == 3
    record_admission.assert_called_once_with(
        request, blocks, 1, first_readmission=True
    )


def test_partial_splice_full_miss_returns_false():
    """Regression: a prefetch that covers none of the needed keys must
    leave everything untouched and return False, so the caller falls back
    to releasing it as stale and running a completely normal full reload."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    other = to_key(99)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[other], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is False
    )

    assert not sched._pending_splice_jobs
    assert not sched._current_batch_load_jobs
    assert not sched._req_status["r1"].transfer_jobs


def test_partial_splice_falls_through_when_block_size_factor_not_one():
    """Q3-3: at block_size_factor > 1 a scattered remainder reload would
    compute wrong intra-block offsets in the worker's block-skip logic --
    must fall through to the normal (unmodified) path instead of
    attempting a partial splice."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1 = to_key(0), to_key(1)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(20, free_blocks=5), block_size_factor=2
    )
    _add_req_status(sched, "r1", keys=[k0, k1])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k1], gpu_block_ids=[77])
    blocks = _make_pending_blocks(2)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 2) is False
    )
    assert not sched._pending_splice_jobs


def test_partial_splice_falls_through_when_multi_group():
    """Q3-3: multi-KV-group requests are out of scope (same as the
    original exact-match guard) -- must fall through, not attempt a
    single-group-shaped partition against a multi-group request."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0 = to_key(0)
    sched = _make_connector_scheduler(
        manager, _StubBlockPool(20, free_blocks=5), num_groups=2
    )
    _add_req_status(sched, "r1", keys=[k0])
    sched._prefetched["r1"] = _PrefetchState(job_id=1, keys=[k0], gpu_block_ids=[77])
    blocks = _make_pending_blocks(1)

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="r1"), blocks, 1) is False
    )
    assert not sched._pending_splice_jobs


def test_two_concurrent_partial_splices_dont_interfere():
    """Partial-splice plan doc, Open Risk #5: concurrent preemptions
    sharing the prefetch budget was flagged as untested at the bookkeeping
    level. Two DIFFERENT requests, each with its own partial prefetch
    (covering only some of what each needs), processed in the same
    scheduling step -- neither's splice/remainder state may clobber the
    other's, and each gets its own distinct remainder-reload job."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    j0, j1 = to_key(10), to_key(11)
    _insert_resident(manager, k0, block_id=0)
    _insert_resident(manager, k2, block_id=1)
    _insert_resident(manager, j1, block_id=2)
    sched = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched, "req1", keys=[k0, k1, k2])
    _add_req_status(sched, "req2", keys=[j0, j1])
    # req1's prefetch covers only the middle key; req2's covers only the
    # first -- deliberately different shapes so a bug that mixed up their
    # state would show up as wrong keys/blocks, not just wrong counts.
    sched._prefetched["req1"] = _PrefetchState(job_id=1, keys=[k1], gpu_block_ids=[77])
    sched._prefetched["req2"] = _PrefetchState(job_id=2, keys=[j0], gpu_block_ids=[88])
    blocks1 = _make_pending_blocks(3, start_block_id=100)  # dst [100, 101, 102]
    blocks2 = _make_pending_blocks(2, start_block_id=200)  # dst [200, 201]

    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="req1"), blocks1, 3)
        is True
    )
    assert (
        sched._try_splice_prefetch(SimpleNamespace(request_id="req2"), blocks2, 2)
        is True
    )

    assert sched._pending_splice_jobs[0] == ([77], [101])
    assert sched._pending_splice_jobs[1] == ([88], [200])

    assert len(sched._current_batch_load_jobs) == 2
    jobs_by_req = {job.req_id: job for job in sched._current_batch_load_jobs.values()}
    assert list(jobs_by_req["req1"].dst_spec.block_ids) == [100, 102]
    assert list(jobs_by_req["req2"].dst_spec.block_ids) == [201]
    # Distinct job ids -- a bug reusing a counter or a shared dict key
    # would collide these.
    assert len(sched._current_batch_load_jobs) == len(
        set(sched._current_batch_load_jobs.keys())
    )
    assert sched._req_status["req1"].transfer_jobs == {
        jid for jid, j in sched._current_batch_load_jobs.items() if j.req_id == "req1"
    }
    assert sched._req_status["req2"].transfer_jobs == {
        jid for jid, j in sched._current_batch_load_jobs.items() if j.req_id == "req2"
    }
    # The allocated-block-id fence must cover BOTH requests' splice and
    # remainder destinations, not just whichever ran second.
    assert sched._current_batch_allocated_block_ids >= {100, 101, 102, 200, 201}


def test_compute_load_plan_matches_the_real_base_method():
    """Partial-splice plan doc, Open Risk #2: `_compute_load_plan`
    duplicates ~40 lines of `OffloadingConnectorScheduler.
    update_state_after_alloc`'s positional keys/dst-block computation
    (scheduler.py:749-821), because the base method takes no "just tell me
    the plan" parameter. That duplication silently desyncs on a future
    vLLM bump. Pin it here: run the REAL inherited base method (via a
    second stub scheduler with no prefetch registered, so our own
    update_state_after_alloc override falls straight through to
    super()) and assert it registers exactly the keys/dst-blocks
    _compute_load_plan independently predicted for the same inputs."""
    manager = SemanticOffloadingManager(num_blocks=10)
    k0, k1, k2 = to_key(0), to_key(1), to_key(2)
    _insert_resident(manager, k0, block_id=0)
    _insert_resident(manager, k1, block_id=1)
    _insert_resident(manager, k2, block_id=2)

    sched_predict = _make_connector_scheduler(
        manager, _StubBlockPool(20, free_blocks=5)
    )
    _add_req_status(sched_predict, "r1", keys=[k0, k1, k2])
    blocks_predict = _make_pending_blocks(3, start_block_id=100)
    plan = sched_predict._compute_load_plan(
        SimpleNamespace(request_id="r1"), blocks_predict, 3
    )
    assert plan is not None
    predicted_keys, predicted_dst, _, _ = plan

    sched_real = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=5))
    _add_req_status(sched_real, "r1", keys=[k0, k1, k2])
    blocks_real = _make_pending_blocks(3, start_block_id=100)
    # No entry in sched_real._prefetched -- _try_splice_prefetch returns
    # False immediately, so this exercises the REAL super().
    # update_state_after_alloc() fallthrough, not our own logic.
    sched_real.update_state_after_alloc(
        SimpleNamespace(request_id="r1"), blocks_real, 3
    )

    assert len(sched_real._current_batch_load_jobs) == 1
    (real_job,) = sched_real._current_batch_load_jobs.values()
    real_keys = sched_real._jobs[next(iter(sched_real._jobs))].keys

    # TransferJobStatus.keys is itself a set (the base method's own
    # bookkeeping discards order), so a set comparison is the real level
    # of fidelity available post-hoc for content; dst_spec.block_ids stays
    # a positional list, so that comparison IS order-sensitive -- between
    # the two, a desync in either content or ordering would be caught.
    assert set(predicted_keys) == real_keys
    assert predicted_dst == list(real_job.dst_spec.block_ids)


# Native adoption tests use the real allocator and CPU tensor backing only.
# Transfer completion is injected; CUDA/worker fence execution is not claimed.
def _native_adoption_request(request_id, tokens=None):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(35)) if tokens is None else tokens,
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        block_hasher=get_request_block_hasher(16, sha256),
    )


def _native_adoption_fixture(dtype=torch.bfloat16):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
        OffloadingEventsTracker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingConnectorStats,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.utils.hashing import sha256
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
    )
    from vllm.v1.kv_offload.base import OffloadingKVEventsConfig

    init_none_hash(sha256)
    cache = KVCacheManager(
        KVCacheConfig(12, [], [KVCacheGroupSpec(["layer"], FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=4, dtype=dtype,
        ))]),
        max_model_len=128, scheduler_block_size=16, hash_block_size=16,
    )
    manager = SemanticOffloadingManager(num_blocks=10)
    sched = _make_connector_scheduler(manager, cache.block_pool)
    sched.config = SchedulerOffloadConfig(
        kv_group_configs=(GroupOffloadConfig(0, 16, 16, 1, None, None),),
        blocks_per_chunk=1, num_workers=1, offload_prompt_only=False,
    )
    sched._wait_hint_native_adoption = True
    sched._lifecycle_release_enabled = False
    sched._wait_hint_native_adoption_layout_supported = True
    sched._published_wait_prefixes = {}
    sched._prefetch_budget_fraction = 0.5
    sched._prefetch_selector = "request_order"
    sched._connector_stats = OffloadingConnectorStats()
    sched._events_tracker = OffloadingEventsTracker(
        OffloadingKVEventsConfig(False, False)
    )
    sched._lookup_groups = (0,)
    sched._sliding_window_groups = ()
    sched._mamba_align_size = None
    sched._preservation_requests = {}
    warm = _native_adoption_request("warm")
    prefix_id = _observed_prefix_digest(
        warm.prompt_token_ids[:32], sched._wait_hint_namespace,
    )
    warm.kv_transfer_params = dict(
        run_id="run", cell_id="cell", session_id="session", generation=0,
        observed_prefix_id=prefix_id, observed_prefix_boundary_tokens=32,
    )
    OffloadingConnectorScheduler.on_new_request(sched, warm)
    sched._req_status["warm"].update_offload_keys()
    keys = sched._req_status["warm"].group_states[0].offload_keys
    for i, key in enumerate(keys):
        _insert_resident(manager, key, i)
    sched._register_observed_prefix(warm)
    hint = _wait_hint(sched, prefix_id, 32)
    assert sched.publish_wait_hint(hint)["status"] == "accepted"
    sched._retry_wait_prefetches()
    state = sched._wait_prefetch[hint["event_id"]]
    # The worker accepted this job; completion remains outstanding.
    sched._current_batch_prefetch_load_jobs.clear()
    backing = torch.zeros(12, 16, 4, dtype=dtype)
    dram = torch.arange(128, dtype=dtype).reshape(2, 16, 4)
    return SimpleNamespace(
        sched=sched, cache=cache, state=state, hint=hint, warm=warm,
        backing=backing, dram=dram,
    )


def _native_adoption_complete(f, *, success=True):
    state = f.state
    for key, block_id in zip(state.keys, state.gpu_block_ids):
        f.backing[block_id].copy_(f.dram[state.source.keys.index(key)])
    now = time.monotonic_ns()
    f.sched._preservation_job_events[state.job_id] = SemanticTransferCompletion(
        is_load=True, bytes=256, time_s=0.001, contributors=1, success=success,
        started_at_ns=now - 1_000_000, completed_at_ns=now,
    )
    meta = SemanticWorkerMetadata(
        completed_jobs={state.job_id: 1} if success else {},
        failed_jobs={} if success else {state.job_id: 1},
    )
    f.sched._complete_wait_prefetch_jobs(meta)


def _native_adoption_bind(f, request=None):
    request = request or _native_adoption_request("resume")
    request.kv_transfer_params = dict(
        external_request_id="owner:future", wait_hint_event_id=f.hint["event_id"],
        run_id="run", cell_id="cell", session_id="session", generation=0,
        observed_prefix_id=f.hint["observed_prefix_id"],
        observed_prefix_boundary_tokens=32,
    )
    f.sched.on_new_request(request)
    return request


def test_other_demand_does_not_reclaim_bound_wait_hint_or_leak_source_protection():
    f = _native_adoption_fixture(torch.float16)
    f.sched._staging_mode = "owned_reclaimable"
    f.sched._wait_hint_native_adoption = False
    _native_adoption_complete(f)
    request = _native_adoption_bind(f)
    state = f.state
    pool = f.cache.block_pool
    manager = f.sched.manager
    assert state.status == "bound_ready"
    assert f.sched._prefetched[request.request_id].attempt_id == f.hint["event_id"]
    assert manager.staging_protection_stats()["active_bindings"] == 1
    assert manager.staging_protection_stats()["protected_key_references"] == 2
    assert all(manager._policy.get(key).ref_cnt == 1 for key in state.keys)
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    assert pool.get_num_free_blocks() == 0

    f.sched._reclaim_wait_prefetch_for_demand(
        _native_adoption_request("other-demand"), 0, protected_event_id=None,
    )

    assert state.status == "bound_ready"
    assert f.sched._wait_prefetch[f.hint["event_id"]] is state
    assert request.request_id in f.sched._prefetched
    assert pool.get_num_free_blocks() == 0
    assert manager.staging_protection_stats()["active_bindings"] == 1
    assert all(manager._policy.get(key).ref_cnt == 1 for key in state.keys)

    # The ordinary cancellation/expiry owner releases both tiers exactly once.
    f.sched._retire_wait_prefetch(
        state, "cancel", now_ns=time.monotonic_ns(),
    )
    assert request.request_id not in f.sched._prefetched
    assert manager.staging_protection_stats()["active_bindings"] == 0
    assert manager.staging_protection_stats()["protected_key_references"] == 0
    assert all(manager._policy.get(key).ref_cnt == 0 for key in state.keys)
    assert pool.get_num_free_blocks() == 2
    f.sched._free_wait_prefetch(state, "repeated_cleanup")
    assert pool.get_num_free_blocks() == 2
    assert manager.staging_protection_stats()["active_bindings"] == 0
    pool.free_blocks(pressure)
    assert pool.get_num_free_blocks() == 11


def _native_adoption_allocate(f, request):
    hit, tokens, _, diverged = f.cache.get_computed_blocks_for_connector(request)
    assert tokens == 32
    assert not diverged
    assert f.sched.get_num_new_matched_tokens(request, tokens) == (0, False)
    assert f.cache.allocate_slots(
        request, 3, num_new_computed_tokens=tokens, new_computed_blocks=hit,
    ) is not None
    blocks = f.cache.get_blocks(request.request_id)
    f.sched.update_state_after_alloc(request, blocks, 0)
    return blocks


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_adoption_same_physical_ids_no_splice_or_duplicate_allocation(
    monkeypatch, dtype,
):
    f = _native_adoption_fixture(dtype)
    assert f.backing.dtype == dtype and f.dram.dtype == dtype
    ids = list(f.state.gpu_block_ids)
    assert f.cache.get_computed_blocks(f.warm)[1] == 0
    assert all(f.cache.block_pool.blocks[i].ref_cnt == 1 for i in ids)
    pressure = f.cache.block_pool.get_new_blocks(
        f.cache.block_pool.get_num_free_blocks()
    )
    assert set(ids).isdisjoint(b.block_id for b in pressure)
    f.cache.block_pool.free_blocks(pressure)
    _native_adoption_complete(f)
    assert f.cache.get_computed_blocks(f.warm)[1] == 0  # unbound stays private
    calls = []
    allocate = f.cache.block_pool.get_new_blocks
    def counted(n):
        calls.append(n)
        return allocate(n)
    monkeypatch.setattr(f.cache.block_pool, "get_new_blocks", counted)
    request = _native_adoption_bind(f)
    assert calls == []
    assert f.state.status == "published"
    assert f.sched._prefetch_reserved_blocks == 0
    assert f.cache.block_pool.get_num_free_blocks() == 11
    assert not f.sched._prefetched and not f.sched._wait_prefetch
    assert not f.sched._wait_hint_request_bindings
    def forbidden(*args, **kwargs):
        raise AssertionError("native adoption must not splice")
    monkeypatch.setattr(f.sched, "_try_splice_prefetch", forbidden)
    blocks = _native_adoption_allocate(f, request)
    assert [b.block_id for b in blocks.blocks[0][:2]] == ids
    assert calls == [1]  # only the mutable suffix; no prefix destinations
    assert torch.equal(f.backing[ids], f.dram)
    assert not f.sched._current_batch_load_jobs
    assert not f.sched._pending_splice_jobs
    assert not f.sched._published_wait_prefixes
    f.cache.free(request)
    assert f.cache.block_pool.get_num_free_blocks() == 11


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_adoption_bound_inflight_publishes_only_after_completion(dtype):
    f = _native_adoption_fixture(dtype)
    request = _native_adoption_bind(f)
    ids = list(f.state.gpu_block_ids)
    assert f.cache.get_computed_blocks_for_connector(request)[1] == 0
    assert f.sched.get_num_new_matched_tokens(request, 0) == (None, False)
    assert all(f.cache.block_pool.blocks[i].ref_cnt == 1 for i in ids)
    _native_adoption_complete(f)
    assert f.state.status == "published"
    blocks = _native_adoption_allocate(f, request)
    assert [b.block_id for b in blocks.blocks[0][:2]] == ids
    assert torch.equal(f.backing[ids], f.dram)
    f.cache.free(request)
    assert f.cache.block_pool.get_num_free_blocks() == 11


def test_native_adoption_two_requests_share_immutable_prefix_and_hold_refs():
    f = _native_adoption_fixture()
    _native_adoption_complete(f)
    first = _native_adoption_bind(f)
    a = _native_adoption_allocate(f, first)
    second = _native_adoption_request("second", list(range(32)) + [80, 81, 82])
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    OffloadingConnectorScheduler.on_new_request(f.sched, second)
    b = _native_adoption_allocate(f, second)
    assert a.blocks[0][:2] == b.blocks[0][:2]
    assert a.blocks[0][-1] is not b.blocks[0][-1]
    assert all(block.ref_cnt == 2 for block in a.blocks[0][:2])
    pool = f.cache.block_pool
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    assert set(f.state.gpu_block_ids).isdisjoint(b.block_id for b in pressure)
    assert torch.equal(f.backing[list(f.state.gpu_block_ids)], f.dram)
    pool.free_blocks(pressure)
    f.cache.free(first)
    assert all(block.ref_cnt == 1 for block in b.blocks[0][:2])
    f.cache.free(second)
    assert pool.get_num_free_blocks() == 11


def test_native_adoption_eviction_before_admission_is_safe_demand_fallback():
    f = _native_adoption_fixture()
    _native_adoption_complete(f)
    request = _native_adoption_bind(f)
    pool = f.cache.block_pool
    pressure = pool.get_new_blocks(pool.get_num_free_blocks())
    assert f.cache.get_computed_blocks(request)[1] == 0
    pool.free_blocks(pressure)
    assert f.sched.get_num_new_matched_tokens(request, 0) == (32, True)
    assert f.cache.allocate_slots(
        request, 0, num_external_computed_tokens=32, delay_cache_blocks=True,
    ) is not None
    f.sched.update_state_after_alloc(
        request, f.cache.get_blocks(request.request_id), 32,
    )
    jobs = list(f.sched._current_batch_load_jobs.values())
    assert len(jobs) == 1 and jobs[0].req_id == request.request_id
    assert list(jobs[0].src_spec.block_ids) == [0, 1]
    assert list(jobs[0].dst_spec.block_ids) == [
        block.block_id for block in f.cache.get_blocks(request.request_id).blocks[0]
    ]
    assert not f.sched._pending_splice_jobs
    assert not f.sched._published_wait_prefixes


@pytest.mark.parametrize("retirement", ["cancel", "expire", "failure"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_adoption_inflight_retirement_never_publishes_or_recycles_early(
    retirement, dtype,
):
    f = _native_adoption_fixture(dtype)
    request = _native_adoption_bind(f)
    pool = f.cache.block_pool
    ids = f.state.gpu_block_ids
    assert f.sched.get_num_new_matched_tokens(request, 0) == (None, False)
    if retirement == "cancel":
        f.sched._retire_wait_prefetch(
            f.state, "cancel", now_ns=time.monotonic_ns(),
        )
    elif retirement == "expire":
        f.sched._expire_wait_prefetches(f.state.lease.expires_at_ns + 1)
    assert f.cache.get_computed_blocks(request)[1] == 0
    assert all(pool.blocks[i].ref_cnt == 1 for i in ids)
    _native_adoption_complete(f, success=retirement != "failure")
    assert f.cache.get_computed_blocks(request)[1] == 0
    assert all(pool.blocks[i].ref_cnt == 0 for i in ids)
    assert pool.get_num_free_blocks() == 11
    assert f.sched._prefetch_reserved_blocks == 0
    assert not f.sched._published_wait_prefixes
    f.sched._complete_wait_prefetch_jobs(SemanticWorkerMetadata(
        completed_jobs={f.state.job_id: 1},
    ))
    assert pool.get_num_free_blocks() == 11  # duplicate completion cannot free twice


@pytest.mark.parametrize("fallback", [
    "disabled", "layout", "tp", "factor", "missing_api", "wrong_tokens",
    "wrong_hash", "missing_fence",
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_adoption_falls_back_without_publishing(fallback, monkeypatch, dtype):
    f = _native_adoption_fixture(dtype)
    _native_adoption_complete(f)
    request = _native_adoption_request("resume")
    if fallback == "disabled":
        f.sched._wait_hint_native_adoption = False
    elif fallback == "layout":
        f.sched._wait_hint_native_adoption_layout_supported = False
    elif fallback == "tp":
        f.sched.config = f.sched.config._replace(num_workers=2)
    elif fallback == "factor":
        f.sched.config = f.sched.config._replace(blocks_per_chunk=2)
    elif fallback == "missing_api":
        monkeypatch.setattr(f.cache.block_pool, "publish_completed_prefix", None)
    elif fallback == "wrong_tokens":
        request = _native_adoption_request("resume", [90] * 35)
    elif fallback == "wrong_hash":
        request.block_hashes[0] = b"wrong-hash"
    elif fallback == "missing_fence":
        f.state.transfer_record["completed_at_ns"] = None
    _native_adoption_bind(f, request)
    assert f.state.status == "bound_ready"
    assert f.cache.get_computed_blocks(request)[1] == 0
    assert request.request_id in f.sched._prefetched
    assert all(f.cache.block_pool.blocks[i].ref_cnt == 1
               for i in f.state.gpu_block_ids)
    f.sched._release_prefetch(request.request_id)
    assert f.cache.block_pool.get_num_free_blocks() == 11


@pytest.mark.parametrize("invalid", [
    "incomplete", "duplicate", "foreign", "shared_ref", "wrong_group",
    "wrong_block_size", "skip_cache", "existing_hash",
])
def test_native_pool_publication_rejects_entire_prefix_atomically(invalid):
    from vllm.v1.core.block_pool import BlockPool
    f = _native_adoption_fixture()
    pool = f.cache.block_pool
    blocks = [pool.blocks[i] for i in f.state.gpu_block_ids]
    request = _native_adoption_request("resume")
    args = dict(block_size=16, kv_cache_group_id=0, completion_verified=True)
    if invalid == "incomplete":
        args["completion_verified"] = False
    elif invalid == "duplicate":
        blocks[1] = blocks[0]
    elif invalid == "foreign":
        blocks[1] = BlockPool(12, True, 16).get_new_blocks(2)[1]
    elif invalid == "shared_ref":
        pool.touch([blocks[1]])
    elif invalid == "wrong_group":
        args["kv_cache_group_id"] = 1
    elif invalid == "wrong_block_size":
        args["block_size"] = 32
    elif invalid == "skip_cache":
        request.skip_reading_prefix_cache = True
    elif invalid == "existing_hash":
        existing = pool.get_new_blocks(1)
        pool.cache_full_blocks(request, existing, 0, 1, 16, 0)
    before = pool.get_num_free_blocks()
    assert pool.publish_completed_prefix(request, blocks, **args) is False
    assert pool.get_num_free_blocks() == before
    assert all(pool.blocks[i].block_hash is None for i in f.state.gpu_block_ids)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_adoption_orders_relevance_ranked_blocks_by_exact_keys(dtype):
    f = _native_adoption_fixture(dtype)
    # Model a completed transfer whose selected-key order differs from chain order.
    f.state.keys = tuple(reversed(f.state.keys))
    ids_in_chain_order = list(reversed(f.state.gpu_block_ids))
    _native_adoption_complete(f)
    request = _native_adoption_bind(f)
    blocks = _native_adoption_allocate(f, request)
    assert [b.block_id for b in blocks.blocks[0][:2]] == ids_in_chain_order
    assert torch.equal(f.backing[ids_in_chain_order], f.dram)
    f.cache.free(request)


def test_native_adoption_ready_expiry_frees_private_blocks_without_visibility():
    f = _native_adoption_fixture()
    _native_adoption_complete(f)
    assert f.state.status == "ready"
    f.sched._expire_wait_prefetches(f.state.lease.expires_at_ns + 1)
    assert f.cache.get_computed_blocks(f.warm)[1] == 0
    assert f.cache.block_pool.get_num_free_blocks() == 11
    assert f.sched._prefetch_reserved_blocks == 0


def test_native_adoption_retiring_published_state_does_not_free_cache_twice():
    f = _native_adoption_fixture()
    _native_adoption_complete(f)
    request = _native_adoption_bind(f)
    assert f.cache.block_pool.get_num_free_blocks() == 11
    f.sched._retire_wait_prefetch(f.state, "cancel", now_ns=time.monotonic_ns())
    assert f.cache.block_pool.get_num_free_blocks() == 11
    # Fenced immutable data now belongs to the ordinary evictable prefix cache.
    assert f.cache.get_computed_blocks(request)[1] == 32


def test_native_pool_publication_exception_rolls_back_visibility(monkeypatch):
    f = _native_adoption_fixture()
    pool = f.cache.block_pool
    blocks = [pool.blocks[i] for i in f.state.gpu_block_ids]
    cache_full_blocks = pool.cache_full_blocks
    def failing(*args, **kwargs):
        cache_full_blocks(*args, **kwargs)
        raise RuntimeError("injected publication failure")
    monkeypatch.setattr(pool, "cache_full_blocks", failing)
    before = pool.get_num_free_blocks()
    with pytest.raises(RuntimeError, match="injected publication failure"):
        pool.publish_completed_prefix(
            f.warm, blocks, block_size=16, kv_cache_group_id=0,
            completion_verified=True,
        )
    assert f.cache.get_computed_blocks(f.warm)[1] == 0
    assert all(b.ref_cnt == 1 and b.block_hash is None for b in blocks)
    assert pool.get_num_free_blocks() == before


@pytest.mark.parametrize("dtype,layout,group_count,supported", [
    (torch.bfloat16, {}, 1, True),
    (torch.float16, {}, 1, True),
    (torch.float32, {}, 1, False),
    (torch.bfloat16, {"sliding_window": 16}, 1, False),
    (torch.float16, {"sliding_window": 16}, 1, False),
    (torch.float16, {"attention_chunk_size": 16}, 1, False),
    (torch.float16, {"non_causal": True}, 1, False),
    (torch.float16, {"kv_quant_mode": "FP8_PER_TENSOR"}, 1, False),
    (torch.float16, {"kv_quant_mode": "INT8_PER_TOKEN_HEAD"}, 1, False),
    (torch.float16, {}, 0, False),
    (torch.float16, {}, 2, False),
])
def test_native_adoption_constructor_restricts_layout_and_validates_flag(
    dtype, layout, group_count, supported, monkeypatch,
):
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVQuantMode,
    )
    native_config = SchedulerOffloadConfig(
        kv_group_configs=(GroupOffloadConfig(0, 16, 16, 1, None, None),),
        blocks_per_chunk=1, num_workers=1, offload_prompt_only=False,
    )
    def base(scheduler, *args):
        scheduler.config = native_config
        scheduler.manager = object()
    monkeypatch.setattr("semantic_offload.connector.construct_scheduler_base", base)
    spec = SimpleNamespace(
        extra_config={"wait_hint_native_adoption": True},
        replicated_layout=False, cpu_page_size_per_worker=128, kv_bytes_per_chunk=128,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="test-model"),
        cache_config=SimpleNamespace(block_size=16),
    )
    layout = dict(layout)
    if "kv_quant_mode" in layout:
        layout["kv_quant_mode"] = KVQuantMode[layout["kv_quant_mode"]]
    cache = KVCacheConfig(12, [], [KVCacheGroupSpec([f"layer-{index}"], FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4, dtype=dtype, **layout,
    )) for index in range(group_count)])
    scheduler = SemanticOffloadingConnectorScheduler(spec, config, cache)
    assert scheduler._wait_hint_native_adoption_layout_supported is supported
    assert scheduler._wait_hint_native_adoption is True
    spec.extra_config["wait_hint_native_adoption"] = "yes"
    with pytest.raises(TypeError, match="wait_hint_native_adoption"):
        SemanticOffloadingConnectorScheduler(spec, config, cache)


def _full_wait_source_fixture(count=3):
    manager = SemanticOffloadingManager(num_blocks=count + 1)
    keys = [to_key(index + 900) for index in range(count)]
    for index, key in enumerate(keys):
        _insert_resident(manager, key, index)
    pool = _StubBlockPool(512, free_blocks=512)
    sched = _make_connector_scheduler(manager, pool)
    sched._prefetch_selector = "request_order"
    sched._prefetch_budget_fraction = 0.5
    prefix, boundary = _register_wait_prefix(sched, keys)
    hint = _wait_hint(sched, prefix, boundary)
    return sched, manager, pool, keys, hint


@pytest.mark.parametrize("status", ["pending", "in_flight", "ready"])
def test_exact_source_refresh_reuses_canonical_lease_without_reserving(status):
    if status == "pending":
        sched, manager, pool, keys, hint = _full_wait_source_fixture()
        sched._wait_hint_coalesce_exact_source = True
        first = sched.publish_wait_hint(hint)
        assert first["canonical_event_id"] == hint["event_id"]
        state = sched._wait_prefetch[hint["event_id"]]
    else:
        f = _native_adoption_fixture(torch.float16)
        sched, pool, hint, state = f.sched, f.cache.block_pool, f.hint, f.state
        sched._wait_hint_coalesce_exact_source = True
        if status == "ready":
            _native_adoption_complete(f)
    lease_identity = id(state.lease)
    original_hint = state.lease.hint
    original_ttl = state.lease.expires_at_ns
    original_deadline = state.lease.nomination_deadline_ns
    original_free = pool.get_num_free_blocks()
    original_reserved = sched._prefetch_reserved_blocks
    original_jobs = dict(sched._current_batch_prefetch_load_jobs)
    refresh = _wait_hint(sched, hint["observed_prefix_id"], hint["prefix_boundary_tokens"], event_id="b" * 64)
    result = sched.publish_wait_hint(refresh)
    assert result["status"] == "accepted"
    assert result["reason"] == "coalesced_exact_source"
    assert result["event_id"] == refresh["event_id"]
    assert result["canonical_event_id"] == hint["event_id"]
    assert result["canonical_state_status"] == status
    assert result["canonical_staged_key_count"] == len(state.keys)
    assert len(sched._wait_prefetch) == 1
    assert refresh["event_id"] not in sched._wait_prefetch
    assert id(state.lease) == lease_identity and state.lease.hint is original_hint
    assert state.lease.expires_at_ns == original_ttl
    assert state.lease.nomination_deadline_ns == original_deadline
    assert pool.get_num_free_blocks() == original_free
    assert sched._prefetch_reserved_blocks == original_reserved
    assert sched._current_batch_prefetch_load_jobs == original_jobs
    replay = sched.publish_wait_hint(refresh)
    assert replay == {**result, "idempotent_replay": True}


@pytest.mark.parametrize("mismatch", ["owner_id", "generation", "namespace", "run_id", "cell_id", "session_id"])
def test_exact_source_refresh_does_not_merge_different_identity(mismatch):
    sched, manager, pool, keys, hint = _full_wait_source_fixture()
    sched._wait_hint_coalesce_exact_source = True
    sched.publish_wait_hint(hint)
    refresh = _wait_hint(sched, hint["observed_prefix_id"], hint["prefix_boundary_tokens"], event_id="b" * 64)
    if mismatch == "generation":
        refresh[mismatch] = 1
    elif mismatch == "namespace":
        refresh[mismatch]["adapter_namespace"] = "other"
    else:
        refresh[mismatch] = "other"
    result = sched.publish_wait_hint(refresh)
    assert result["reason"] != "coalesced_exact_source"
    assert result.get("canonical_event_id") != hint["event_id"]
    assert hint["event_id"] in sched._wait_prefetch


@pytest.mark.parametrize("inactive", ["pending_deadline", "ttl", "cancelled", "retired"])
def test_exact_source_refresh_never_reuses_expired_or_retired_state(inactive):
    sched, manager, pool, keys, hint = _full_wait_source_fixture()
    sched._wait_hint_coalesce_exact_source = True
    sched.publish_wait_hint(hint)
    old = sched._wait_prefetch[hint["event_id"]]
    if inactive == "pending_deadline":
        old.lease.nomination_deadline_ns = time.monotonic_ns() - 1
    elif inactive == "ttl":
        sched._retry_wait_prefetches()
        old.lease.expires_at_ns = time.monotonic_ns() - 1
    elif inactive == "cancelled":
        old.lease.cancel()
    else:
        old.status = "retired_pending_fence"
    reserved = sched._prefetch_reserved_blocks
    refresh = _wait_hint(sched, hint["observed_prefix_id"], hint["prefix_boundary_tokens"], event_id="b" * 64)
    result = sched.publish_wait_hint(refresh)
    assert result["reason"] == "queued_for_scheduler_dispatch"
    assert result["canonical_event_id"] == refresh["event_id"]
    assert sched._wait_prefetch[hint["event_id"]] is old
    assert len(sched._wait_prefetch) == 2
    assert sched._prefetch_reserved_blocks == reserved
    if inactive == "ttl":
        assert old.status == "in_flight"  # refresh never recycles live DMA destinations
        assert old.job_id in sched._wait_prefetch_jobs


@pytest.mark.parametrize("missing", ["cpu_tail", "gpu_prefix", "gpu_suffix", "gpu_api"])
def test_full_prefix_publication_rejects_partial_or_duplicate_source(missing, monkeypatch):
    sched, manager, pool, keys, hint = _full_wait_source_fixture()
    sched._prefetch_require_full_prefix = True
    if missing == "cpu_tail":
        manager._policy.get(keys[-1]).ref_cnt = -1  # unfinished store, not readable
    elif missing == "gpu_api":
        pool.get_cached_block = None
    else:
        key = keys[0] if missing == "gpu_prefix" else keys[-1]
        pool._cached_block_hashes[bytes(key[:-4])] = 500
    free = pool.get_num_free_blocks()
    monkeypatch.setattr(pool, "get_new_blocks", lambda *args: pytest.fail("must not reserve"))
    monkeypatch.setattr(manager, "prepare_load", lambda *args: pytest.fail("must not prepare DMA"))
    result = sched.publish_wait_hint(hint)
    assert result["status"] == "no_action"
    assert result["reason"] == "full_prefix_unavailable"
    assert not sched._wait_prefetch
    assert not sched._current_batch_prefetch_load_jobs
    assert pool.get_num_free_blocks() == free


@pytest.mark.parametrize("enough", [False, True])
def test_full_128_key_source_requires_entire_allocation_budget(enough):
    sched, manager, pool, keys, hint = _full_wait_source_fixture(128)
    sched._prefetch_require_full_prefix = True
    sched._prefetch_budget_fraction = (128 if enough else 127) / 512
    result = sched.publish_wait_hint(hint)
    if not enough:
        assert result["status"] == "no_action"
        assert result["reason"] == "full_prefix_budget_unavailable"
        assert not sched._wait_prefetch
        assert pool.get_num_free_blocks() == 512
        assert not sched._current_batch_prefetch_load_jobs
        return
    assert result["status"] == "accepted"
    state = sched._wait_prefetch[hint["event_id"]]
    assert sched._attempt_wait_prefetch(state)
    assert state.keys == tuple(keys)
    assert len(state.gpu_block_ids) == 128
    assert sched._prefetch_reserved_blocks == 128
    assert len(sched._current_batch_prefetch_load_jobs) == 1


@pytest.mark.parametrize("changed", ["budget", "cpu_tail", "gpu_suffix", "candidate_tail", "selected_partial"])
def test_full_prefix_revalidates_before_dma_without_rewriting_publication(changed, monkeypatch):
    sched, manager, pool, keys, hint = _full_wait_source_fixture()
    sched._prefetch_require_full_prefix = True
    assert sched.publish_wait_hint(hint)["status"] == "accepted"
    state = sched._wait_prefetch[hint["event_id"]]
    publication = (state.publication_candidate_keys, state.publication_eligible_keys,
                   state.publication_local_gpu_prefix_count, state.publication_local_gpu_block_ids)
    if changed == "budget":
        sched._prefetch_budget_fraction = 2 / 512
    elif changed == "cpu_tail":
        manager._policy.get(keys[-1]).ref_cnt = -1  # unfinished store, not readable
    elif changed == "gpu_suffix":
        pool._cached_block_hashes[bytes(keys[-1][:-4])] = 500
    elif changed == "candidate_tail":
        state.publication_candidate_keys = tuple(keys[1:])
        publication = (state.publication_candidate_keys, *publication[1:])
    else:
        sched._prefetch_selector = "useful_on_time"
        sched._prefetch_transfer_ns_per_byte = 1_000_000.0
    free = pool.get_num_free_blocks()
    monkeypatch.setattr(pool, "get_new_blocks", lambda *args: pytest.fail("must not reserve"))
    monkeypatch.setattr(manager, "prepare_load", lambda *args: pytest.fail("must not prepare DMA"))
    assert not sched._attempt_wait_prefetch(state)
    assert state.status == "pending"
    assert not sched._current_batch_prefetch_load_jobs
    assert pool.get_num_free_blocks() == free
    assert sched._prefetch_reserved_blocks == 0
    assert publication == (state.publication_candidate_keys, state.publication_eligible_keys,
                           state.publication_local_gpu_prefix_count, state.publication_local_gpu_block_ids)
    assert state.selection_evidence["status"] in {"full_prefix_unavailable", "full_prefix_selection_incomplete"}


@pytest.mark.parametrize("option", ["wait_hint_coalesce_exact_source", "prefetch_require_full_prefix"])
@pytest.mark.parametrize("invalid", [0, 1, None, "true", []])
def test_exact_source_and_full_prefix_options_require_json_booleans(option, invalid, monkeypatch):
    config = SchedulerOffloadConfig(
        kv_group_configs=(GroupOffloadConfig(0, 16, 16, 1, None, None),),
        blocks_per_chunk=1, num_workers=1, offload_prompt_only=False,
    )
    def base(scheduler, *args):
        scheduler.config = config
        scheduler.manager = object()
    monkeypatch.setattr("semantic_offload.connector.construct_scheduler_base", base)
    spec = SimpleNamespace(extra_config={option: invalid}, replicated_layout=False,
                           cpu_page_size_per_worker=128, kv_bytes_per_chunk=128)
    with pytest.raises(TypeError, match=option + " must be a JSON boolean"):
        SemanticOffloadingConnectorScheduler(spec,
            SimpleNamespace(model_config=SimpleNamespace(model="test-model"),
                            cache_config=SimpleNamespace(block_size=16)),
            SimpleNamespace())
