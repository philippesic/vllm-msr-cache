"""Operational ARC retains failure handling without unbounded audit histories.

CPU tests use native ARC, scheduler, requests and transfer metadata. They do
not establish GPU byte correctness or performance of the opt-in profile.
"""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from semantic_offload.connector import (
    SemanticOffloadingConnectorScheduler, SemanticWorkerMetadata,
    SemanticTransferCompletion, _observed_prefix_digest,
)
from semantic_offload.manager import PreservationOffloadingManager
from semantic_offload.spec import SemanticOffloadingSpec, validate_preservation_audit
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig, get_offload_block_hash
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams
from tests.test_step_1_5_prefetch import _wait_hint, to_key


def options(**overrides):
    return {
        "preservation_audit": False,
        "preservation_events": True,
        "instrumentation_only": True,
        "eviction_policy": "arc",
        "prefetch_trigger": "known_prefix_wait_hint",
        "wait_hint_diagnostic_no_dispatch": True,
        "wait_hint_cpu_retention_only": True,
        "prefetch_budget_fraction": 0.6,
        "observed_session_registry_size": 2,
        "observed_prefix_registry_size": 2,
        "admission_namespace": {
            "model_revision": "model", "tokenizer_revision": "tokenizer",
            "adapter_namespace": "none", "layout_namespace": "dense-tp1-fp16-block16",
        },
        **overrides,
    }


def native(**overrides):
    manager = PreservationOffloadingManager(num_blocks=8, cache_policy="arc")
    config = NS(
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           world_size=1, rank=0),
        model_config=NS(model="cpu-audit-mode"),
        cache_config=NS(block_size=16, enable_prefix_caching=True),
        speculative_config=None,
    )
    spec = NS(extra_config=options(**overrides), replicated_layout=False,
              cpu_page_size_per_worker=128, kv_bytes_per_chunk=128,
              tokens_per_block=[16], tokens_per_hash=16, blocks_per_chunk=1,
              offload_prompt_only=False,
              kv_events_config=OffloadingKVEventsConfig(False, False),
              get_manager=lambda: manager)
    cache = KVCacheConfig(32, [], [KVCacheGroupSpec(["layer"], FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4, dtype=torch.float16,
    ))])
    scheduler = SemanticOffloadingConnectorScheduler(spec, config, cache)
    scheduler.gpu_block_pool = BlockPool(32, True, 16)
    return scheduler, manager


def source(scheduler, session="one", generation=0, ordinal=0):
    tokens = list(range(33))
    request = Request(request_id=f"r-{session}-{generation}-{ordinal}",
                      prompt_token_ids=tokens, sampling_params=SamplingParams(max_tokens=1),
                      pooling_params=None)
    keys = [to_key(ordinal * 3 + i + 1) for i in range(2)]
    request.block_hashes = [get_offload_block_hash(key) for key in keys]
    prefix = _observed_prefix_digest(tokens[:32], scheduler._wait_hint_namespace)
    request.kv_transfer_params = dict(
        run_id="run", cell_id="cell", session_id=session, generation=generation,
        observed_prefix_id=prefix, observed_prefix_boundary_tokens=32,
    )
    scheduler.on_new_request(request)
    scheduler._req_status[request.request_id].update_offload_keys()
    return request, prefix, keys


def test_gpu_bridge_uses_native_index_without_refs_or_audit_history():
    from tests.test_step_1_5_prefetch import _insert_resident

    scheduler, manager = native(
        wait_hint_native_adoption=True,
        wait_hint_cpu_retention_gpu_aware=True,
    )
    request, prefix, keys = source(scheduler)
    pool = scheduler.gpu_block_pool
    blocks = pool.get_new_blocks(1)
    pool.cache_full_blocks(request, blocks, 0, 1, 16, 0)
    pool.free_blocks(blocks)
    _insert_resident(manager, keys[1], 0)
    scheduler._register_observed_prefix(request)
    queue_before = pool.free_block_queue.get_all_free_blocks()
    refs_before = [block.ref_cnt for block in pool.blocks]
    histories = (scheduler._preservation_requests, scheduler._preservation_finished,
                 scheduler._preservation_key_commits, scheduler._preservation_key_evictions)
    assert not any(histories)
    with patch('semantic_offload.connector.record_wait_hint_event') as emit:
        result = scheduler.publish_wait_hint(_wait_hint(
            scheduler, prefix, 32, session_id="one",
        ))
    assert not emit.called and not any(histories)
    assert result['cpu_only_prefix_keys'] == 0
    assert result['local_gpu_prefix_keys'] == 1
    assert result['additional_ready_cpu_suffix_keys'] == 1
    assert result['refreshed_cpu_key_count'] == 1
    assert pool.free_block_queue.get_all_free_blocks() == queue_before
    assert [block.ref_cnt for block in pool.blocks] == refs_before
    assert pool.get_cached_block(get_offload_block_hash(keys[0]), [0]) == blocks
    assert manager._policy.get(keys[1]).ref_cnt == 0
    assert not scheduler._wait_prefetch and not scheduler._jobs
    assert len(scheduler._wait_hint_results) == 1
    assert len(scheduler._observed_prefix_generations) == 1


@pytest.mark.parametrize("invalid", [0, 1, None, "false", []])
def test_audit_option_requires_boolean(invalid):
    with pytest.raises(TypeError, match="preservation_audit"):
        validate_preservation_audit(options(preservation_audit=invalid))


@pytest.mark.parametrize("changes", [
    {"preservation_events": False}, {"instrumentation_only": False},
    {"eviction_policy": "lru"}, {"prefetch_trigger": "preemption"},
    {"wait_hint_diagnostic_no_dispatch": False}, {"shared_prefetch_scorer": True},
    {"prefix_mean_prefetch_scoring": True}, {"lifecycle_release_enabled": True},
    {"decision_feature_trace": True}, {"bounded_prefill_cuda_timing": True},
    {"scoring_mode": "asynchronous"}, {"scoring_schedule": "opportunity"},
    {"admission_mode": "always"}, {"soft_retention": "bounded_hint"},
    {"wait_hint_cpu_retention_only": False},
    {"observed_session_registry_size": 0}, {"observed_session_registry_size": True},
])
def test_audit_off_rejects_unsupported_profiles(changes):
    with pytest.raises(ValueError):
        native(**changes)


def test_audit_default_and_zero_budget_control():
    assert validate_preservation_audit({}) is True
    scheduler, _ = native(wait_hint_cpu_retention_only=False, prefetch_budget_fraction=0)
    assert scheduler._preservation_events_enabled
    assert not scheduler._preservation_audit_enabled


def test_manager_keeps_native_events_only_when_explicitly_requested():
    for native_events in (False, True):
        spec = SemanticOffloadingSpec.__new__(SemanticOffloadingSpec)
        spec.extra_config = options()
        spec._manager = None
        spec.num_blocks = 8
        spec.kv_events_config = NS(enable_kv_cache_events=native_events)
        manager = spec.get_manager()
        assert (manager.events is not None) == native_events


def test_capacity_is_fail_closed_and_never_forgets_stale_generations():
    scheduler, _ = native()
    scheduler._wait_hint_result_limit = 4
    with patch("semantic_offload.connector.record_wait_hint_event") as emit:
        for ordinal in range(12):
            session = "one" if ordinal < 2 else f"session-{ordinal}"
            generation = ordinal if ordinal < 2 else 0
            req, prefix, _ = source(scheduler, session, generation, ordinal)
            scheduler._register_observed_prefix(req)
            hint = _wait_hint(scheduler, prefix, 32, session_id=session,
                              generation=generation, event_id=f"{ordinal:064x}")
            result = scheduler.publish_wait_hint(hint)
            if ordinal > 2:
                assert result["reason"] == "session_registry_capacity"
                assert result["status"] == "no_action"
            req.status = RequestStatus.FINISHED_STOPPED
            scheduler.request_finished(req)
            output = SchedulerOutput.make_empty()
            output.finished_req_ids = {req.request_id}
            scheduler.build_connector_meta(output)
        assert not emit.called
    assert len(scheduler._observed_prefix_generations) == 2
    assert len(scheduler._observed_prefixes) <= 2
    assert len(scheduler._wait_hint_results) == 4
    assert not scheduler._req_status
    assert not scheduler._preservation_requests and not scheduler._preservation_finished
    # The old prefix may have left the bounded registry. Its generation
    # tombstone remains authoritative, even after many unrelated sessions.
    old, prefix, _ = source(scheduler, "one", 0, 20)
    scheduler._register_observed_prefix(old)
    assert scheduler._observed_prefix_generations[("run", "cell", "one")] == 1
    result = scheduler.publish_wait_hint(_wait_hint(
        scheduler, prefix, 32, session_id="one", generation=0, event_id="f" * 64,
    ))
    assert result["reason"] == "superseded_generation"
    assert not scheduler._wait_prefetch and not scheduler._wait_prefetch_jobs


def test_real_arc_advice_refreshes_without_request_history_or_logs():
    scheduler, manager = native()
    req, prefix, keys = source(scheduler)
    ctx = scheduler._req_status[req.request_id].req_context
    prepared = manager.prepare_store(keys, ctx)
    manager.complete_store(prepared.keys_to_store, ctx)
    scheduler._register_observed_prefix(req)
    refs = [manager._policy.get(key).ref_cnt for key in keys]
    result = scheduler.publish_wait_hint(_wait_hint(
        scheduler, prefix, 32, session_id="one",
    ))
    assert result["reason"] == "cpu_prefix_recency_refreshed"
    assert result["ready_cpu_prefix_keys"] == 2
    assert [manager._policy.get(key).ref_cnt for key in keys] == refs
    assert not scheduler._preservation_requests
    assert not scheduler._wait_prefetch and not scheduler._jobs


@pytest.mark.parametrize("success", [True, False])
def test_native_store_completion_failure_and_eviction_have_no_retained_audit(success):
    scheduler, manager = native()
    req, _, _ = source(scheduler)
    status = scheduler._req_status[req.request_id]
    blocks = scheduler.gpu_block_pool.get_new_blocks(3)
    status.update_block_id_groups(([block.block_id for block in blocks],))
    req.num_computed_tokens = 32
    req.status = RequestStatus.FINISHED_STOPPED
    scheduler.request_finished(req)
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = {req.request_id}
    metadata = scheduler.build_connector_meta(output)
    assert metadata.store_jobs
    ids = list(metadata.store_jobs)
    assert set(scheduler._preservation_jobs) == set(ids)
    events = {job: SemanticTransferCompletion(
        is_load=False, bytes=256, time_s=0.001, contributors=1,
        success=success, status="success" if success else "failure",
    ) for job in ids}
    worker = SemanticWorkerMetadata(
        completed_jobs={job: 1 for job in ids} if success else {},
        failed_jobs={} if success else {job: 1 for job in ids},
        transfer_events=events,
    )
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker))
    assert not scheduler._jobs and not scheduler._req_status
    assert not scheduler._preservation_jobs and not scheduler._preservation_job_events
    assert not scheduler._preservation_key_commits and not scheduler._preservation_key_evictions
    assert not scheduler._preservation_requests and not scheduler._preservation_finished
    assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == (2 if success else 0)
    scheduler._record_cpu_eviction_events([to_key(i) for i in range(100)])
    assert not scheduler._preservation_key_evictions


def test_unknown_transfer_metadata_cannot_accumulate():
    scheduler, _ = native()
    for job in range(10):
        worker = SemanticWorkerMetadata(transfer_events={job: SemanticTransferCompletion(
            is_load=False, success=True, bytes=16, time_s=0.001, contributors=1,
        )})
        scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker))
    assert not scheduler._preservation_job_events


@pytest.mark.parametrize("failure", ["reported", "direction"])
def test_native_load_failure_releases_cpu_reference_and_invalidates_gpu_ids(failure):
    scheduler, manager = native()
    req, _, keys = source(scheduler)
    status = scheduler._req_status[req.request_id]
    prepared = manager.prepare_store(keys, status.req_context)
    manager.complete_store(prepared.keys_to_store, status.req_context)
    assert scheduler.get_num_new_matched_tokens(req, 0) == (32, True)
    blocks = scheduler.gpu_block_pool.get_new_blocks(2)
    status.update_block_id_groups(([block.block_id for block in blocks],))
    scheduler.update_state_after_alloc(req, NS(blocks=(blocks,)), 32)
    metadata = scheduler.build_connector_meta(SchedulerOutput.make_empty())
    assert len(metadata.load_jobs) == 1
    job = next(iter(metadata.load_jobs))
    assert all(manager._policy.get(key).ref_cnt == 1 for key in keys)
    worker = SemanticWorkerMetadata(
        completed_jobs={job: 1} if failure == "direction" else {},
        failed_jobs={job: 1} if failure == "reported" else {},
        transfer_events={job: SemanticTransferCompletion(
            is_load=failure != "direction", success=failure == "direction",
            status="failure" if failure == "reported" else "success",
            bytes=256, time_s=0.001, contributors=1,
        )},
    )
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker))
    assert not scheduler._jobs and not status.transfer_jobs
    assert not scheduler._chunks_being_loaded
    assert status.group_states[0].block_ids == [0, 0]
    assert all(manager._policy.get(key).ref_cnt == 0 for key in keys)
    assert not scheduler._preservation_jobs and not scheduler._preservation_job_events


def test_completed_transcript_still_registers_only_computed_full_blocks():
    scheduler, _ = native(wait_hint_native_adoption=True, wait_hint_completed_transcript=True)
    req, _, _ = source(scheduler)
    req.append_output_token_ids(list(range(100, 132)))
    req.num_computed_tokens = 64
    req.num_in_flight_tokens = 1
    req.block_hashes.extend(get_offload_block_hash(to_key(i)) for i in (3, 4))
    req.status = RequestStatus.FINISHED_STOPPED
    scheduler._register_observed_prefix(req)
    records = list(scheduler._observed_prefixes.values())
    assert {record.boundary_tokens for record in records} == {32, 48}
    assert all(record.boundary_tokens <= req.num_computed_tokens - 1 for record in records)
    assert not scheduler._preservation_requests


def test_repeated_native_stores_and_evictions_keep_only_active_transfer_state():
    scheduler, manager = native()
    evicted = 0
    for ordinal in range(24):
        req, _, _ = source(scheduler, session=f"s{ordinal}", ordinal=ordinal)
        status = scheduler._req_status[req.request_id]
        blocks = scheduler.gpu_block_pool.get_new_blocks(3)
        status.update_block_id_groups(([block.block_id for block in blocks],))
        req.num_computed_tokens = 32
        req.status = RequestStatus.FINISHED_STOPPED
        scheduler.request_finished(req)
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = {req.request_id}
        metadata = scheduler.build_connector_meta(output)
        evicted += len(metadata.evicted_keys)
        ids = list(metadata.store_jobs)
        assert ids
        worker = SemanticWorkerMetadata(
            completed_jobs={job: 1 for job in ids},
            transfer_events={job: SemanticTransferCompletion(
                is_load=False, bytes=256, time_s=0.001, contributors=1, success=True,
            ) for job in ids},
        )
        scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=worker))
        scheduler.gpu_block_pool.free_blocks(blocks)
        assert not scheduler._preservation_jobs and not scheduler._preservation_job_events
        assert not scheduler._preservation_key_commits and not scheduler._preservation_key_evictions
        assert not scheduler._preservation_known_stored_keys and not scheduler._preservation_evicted_keys
        assert not scheduler._preservation_requests and not scheduler._preservation_finished
        assert not scheduler._req_status and not scheduler._jobs
        assert len(scheduler._observed_prefix_generations) <= 2
        assert len(scheduler._observed_prefixes) <= 2
    assert evicted >= 32
    assert manager._num_evictable_cache_blocks == 8


def test_cache_reset_reopens_registry_without_revalidating_old_hint():
    scheduler, _ = native(observed_session_registry_size=1)
    request, prefix, _ = source(scheduler)
    scheduler._register_observed_prefix(request)
    old_hint = _wait_hint(scheduler, prefix, 32, session_id="one")
    assert scheduler.publish_wait_hint(old_hint)["reason"] == "cpu_prefix_recency_refreshed"
    assert scheduler._session_registry_full(("run", "cell", "two"))
    scheduler.reset_cache()
    assert not scheduler._observed_prefixes and not scheduler._observed_prefix_generations
    assert not scheduler._wait_hint_results
    assert scheduler.publish_wait_hint(old_hint)["reason"] == "observed_prefix_unavailable"
    request, _, _ = source(scheduler, "two")
    scheduler._register_observed_prefix(request)
    assert ("run", "cell", "two") in scheduler._observed_prefix_generations
