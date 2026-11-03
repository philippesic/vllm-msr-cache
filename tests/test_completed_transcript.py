"""Completed transcript sources use only real, fully computed KV blocks."""
from collections import defaultdict
from types import SimpleNamespace
import time

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler, GroupOffloadConfig, SchedulerOffloadConfig,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from semantic_offload.connector import (
    SemanticOffloadingConnectorScheduler, SemanticWorkerMetadata, SemanticTransferCompletion,
    _observed_prefix_digest,
)
from tests.test_step_1_5_prefetch import (
    _native_adoption_fixture, _native_adoption_complete, _native_adoption_request, _wait_hint,
)


def _finished_fixture(*, enabled=True, generated=32, computed=None, status=None):
    f = _native_adoption_fixture(torch.float16)
    _native_adoption_complete(f)
    f.sched._free_wait_prefetch(f.state, "fixture_cleanup")
    f.sched._wait_hint_completed_transcript = enabled
    f.sched._current_batch_jobs_to_flush = set()
    f.sched._block_id_to_pending_jobs = defaultdict(set)
    f.sched._stale_job_threshold = 0
    request = _native_adoption_request("transcript", list(range(32)))
    request.kv_transfer_params = dict(
        run_id="run", cell_id="cell", session_id="session", generation=0,
        observed_prefix_id=_observed_prefix_digest(
            list(request.prompt_token_ids), f.sched._wait_hint_namespace,
        ), observed_prefix_boundary_tokens=32,
    )
    OffloadingConnectorScheduler.on_new_request(f.sched, request)
    request.append_output_token_ids(list(range(100, 100 + generated)))
    request.num_computed_tokens = (
        request.num_tokens - 1 if computed is None else computed
    )
    request.status = status or RequestStatus.FINISHED_LENGTH_CAPPED
    blocks = f.cache.allocate_slots(request, request.num_computed_tokens)
    assert blocks is not None
    req_state = f.sched._req_status[request.request_id]
    req_state.update_offload_keys()
    req_state.update_block_id_groups(
        tuple([block.block_id for block in group] for group in blocks.blocks)
    )
    f.request = request
    f.req_state = req_state
    f.blocks = blocks
    return f


def _store_output(request):
    return SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={request.request_id})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("generated", [16, 17, 31, 32, 33])
def test_incremental_finished_store_excludes_uncomputed_sample_even_without_registry(
    enabled, generated,
):
    f = _finished_fixture(enabled=enabled, generated=generated)
    request = f.request
    jobs = f.sched._build_store_jobs(_store_output(request))
    safe_chunks = request.num_computed_tokens // 16
    expected = set(f.req_state.group_states[0].offload_keys[2:safe_chunks])
    actual = set().union(*(f.sched._jobs[job_id].keys for job_id in jobs))
    assert actual == expected
    assert f.req_state.group_states[0].next_stored_chunk_idx == safe_chunks
    # Every new CPU key maps to its exact real GPU block, never to the
    # sampled-but-unexecuted block tail when logical length is block aligned.
    mapping = {key: ids for job_id in jobs
               for key, ids in f.sched._pending_job_blocks[job_id]}
    for index in range(2, safe_chunks):
        key = f.req_state.group_states[0].offload_keys[index]
        assert mapping[key] == [f.blocks.blocks[0][index].block_id]
    for job in jobs.values():
        assert set(job.src_spec.block_ids).isdisjoint(
            block.block_id for block in f.blocks.blocks[0][safe_chunks:]
        )


@pytest.mark.parametrize("in_flight", [1, 16, 32])
def test_finished_store_and_transcript_exclude_optimistically_counted_inflight(in_flight):
    f = _finished_fixture(generated=33, computed=64)
    f.request.num_in_flight_tokens = in_flight
    expected = (64 - in_flight) // 16
    jobs = f.sched._build_store_jobs(_store_output(f.request))
    assert f.req_state.group_states[0].next_stored_chunk_idx == expected
    f.sched._register_observed_prefix(f.request)
    records = [record for record in f.sched._observed_prefixes.values()
               if record.source_request_id == f.request.request_id]
    assert {record.boundary_tokens for record in records} == {32, max(32, expected * 16)}
    assert all(len(job.src_spec.block_ids) <= max(0, expected - 2) for job in jobs.values())


def test_completed_source_keeps_declared_input_and_waits_for_real_store_completion():
    f = _finished_fixture()
    before = time.monotonic_ns()
    f.sched._register_observed_prefix(f.request)
    after = time.monotonic_ns()
    records = {record.boundary_tokens: record for record in f.sched._observed_prefixes.values()
               if record.source_request_id == f.request.request_id}
    assert set(records) == {32, 48}
    extra = records[48]
    assert before <= extra.observed_ns <= after
    assert extra.observed_prefix_id == _observed_prefix_digest(
        list(f.request.all_token_ids[:48]), f.sched._wait_hint_namespace,
    )
    assert extra.keys == tuple(f.req_state.group_states[0].offload_keys[:3])
    assert extra.observed_prefix_id != records[32].observed_prefix_id
    jobs = f.sched._build_store_jobs(_store_output(f.request))
    assert len(jobs) == 1
    job_id, job = next(iter(jobs.items()))
    assert f.sched._jobs[job_id].keys == {extra.keys[-1]}
    assert list(job.src_spec.block_ids) == [f.blocks.blocks[0][2].block_id]
    assert f.sched._block_id_to_pending_jobs[f.blocks.blocks[0][2].block_id] == {job_id}
    # Exclude unrelated local-cache coverage, using the actual free allocator.
    f.cache.free(f.request)
    f.cache.block_pool.evict_blocks({block.block_id for block in f.blocks.blocks[0]})
    f.sched._prefetch_require_full_prefix = True
    hint = _wait_hint(f.sched, extra.observed_prefix_id, 48, event_id="c" * 64)
    early = f.sched.publish_wait_hint(hint)
    assert early["status"] == "no_action"
    assert early["reason"] == "full_prefix_unavailable"
    assert not f.sched._current_batch_prefetch_load_jobs
    OffloadingConnectorScheduler.update_connector_output(
        f.sched, KVConnectorOutput(kv_connector_worker_meta=SemanticWorkerMetadata(
            completed_jobs={job_id: 1},
        )),
    )
    assert f.sched.manager._policy.get(extra.keys[-1]).is_ready
    assert not f.sched._block_id_to_pending_jobs
    accepted = f.sched.publish_wait_hint(dict(hint, event_id="d" * 64))
    assert accepted["status"] == "accepted"
    f.sched._retry_wait_prefetches()
    staged = f.sched._wait_prefetch["d" * 64]
    assert staged.keys == extra.keys
    assert len(staged.gpu_block_ids) == 3


def test_generated_store_reuse_uses_native_flush_and_completion_fences():
    f = _finished_fixture(enabled=False)
    jobs = f.sched._build_store_jobs(_store_output(f.request))
    job_id, job = next(iter(jobs.items()))
    source_ids = set(job.src_spec.block_ids)
    assert len(source_ids) == 1
    f.cache.free(f.request)
    pressure = f.cache.block_pool.get_new_blocks(
        f.cache.block_pool.get_num_free_blocks()
    )
    assert source_ids <= {block.block_id for block in pressure}
    f.sched._current_batch_allocated_block_ids.update(
        block.block_id for block in pressure
    )
    output = SimpleNamespace(
        num_scheduled_tokens={}, finished_req_ids=set(), preempted_req_ids=set(),
        scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set(),
        ),
    )
    metadata = OffloadingConnectorScheduler.build_connector_meta(f.sched, output)
    assert job_id in metadata.jobs_to_flush
    key = next(iter(f.sched._jobs[job_id].keys))
    assert not f.sched.manager._policy.get(key).is_ready
    OffloadingConnectorScheduler.update_connector_output(
        f.sched, KVConnectorOutput(kv_connector_worker_meta=SemanticWorkerMetadata(
            completed_jobs={job_id: 1},
        )),
    )
    assert f.sched.manager._policy.get(key).is_ready
    assert job_id not in f.sched._jobs
    assert not f.sched._block_id_to_pending_jobs
    f.cache.block_pool.free_blocks(pressure)


def test_running_decode_store_keeps_native_scheduled_forward_boundary():
    f = _finished_fixture(status=RequestStatus.RUNNING, generated=17, computed=47)
    output = SimpleNamespace(
        num_scheduled_tokens={f.request.request_id: 1}, finished_req_ids=set(),
    )
    jobs = f.sched._build_store_jobs(output)
    assert len(jobs) == 1
    assert f.req_state.group_states[0].next_stored_chunk_idx == 3
    assert list(next(iter(jobs.values())).src_spec.block_ids) == [f.blocks.blocks[0][2].block_id]


@pytest.mark.parametrize("generated,computed,in_flight,expected_ranges", [
    (32, 63, 0, [(32, 48)]),
    (33, 64, 1, [(32, 48)]),
    (33, 64, 16, [(32, 48)]),
    (33, 64, 0, [(32, 48), (48, 64)]),
])
def test_preservation_records_decode_store_commits_without_widening_prompt_demand(
    generated, computed, in_flight, expected_ranges,
):
    f = _finished_fixture(generated=generated, computed=computed)
    f.request.num_in_flight_tokens = in_flight
    scheduler = f.sched
    scheduler._preservation_events_enabled = True
    scheduler._preservation_finished = {}
    scheduler._preservation_jobs = {}
    scheduler._preservation_known_stored_keys = set()
    scheduler._preservation_evicted_keys = set()
    scheduler._lifecycle_release_timing = "inline"
    scheduler._replicated_layout = False
    scheduler._prefetch_provenance = {}
    state = scheduler._preservation_state(f.request)
    scheduler._preservation_requests[f.request.request_id] = state
    assert state["required_tokens"] == state["prompt_token_count"] == 32
    assert state["required_chunks"] == 2
    assert [(r["token_start"], r["token_end"]) for r in state["ordered_key_positions"]] == [(0, 16), (16, 32)]
    output = SimpleNamespace(
        num_scheduled_tokens={}, finished_req_ids={f.request.request_id},
        preempted_req_ids=set(), scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set(),
        ),
    )

    metadata = scheduler.build_connector_meta(output)

    assert len(metadata.store_jobs) == 1
    job_id = next(iter(metadata.store_jobs))
    info = scheduler._preservation_jobs[job_id]
    assert info["key_order_complete"] is True
    assert [(r["token_start"], r["token_end"]) for r in info["key_positions"]] == expected_ranges
    assert all(r["complete_chunk"] for r in info["key_positions"])
    assert state["cpu_commit_events"] == []
    now = time.monotonic_ns()
    scheduler.update_connector_output(KVConnectorOutput(
        kv_connector_worker_meta=SemanticWorkerMetadata(
            completed_jobs={job_id: 1}, transfer_events={job_id: SemanticTransferCompletion(
                is_load=False, bytes=128 * len(expected_ranges), time_s=0.001,
                contributors=1, contributor_ids=("0",),
                started_at_ns=now - 1_000_000, completed_at_ns=now,
            )},
        ),
    ))
    commits = state["cpu_commit_events"]
    assert [(r["token_start"], r["token_end"]) for r in commits] == expected_ranges
    assert all(r["state"] == "committed_cpu_ready" for r in commits)
    assert all(r["token_end"] <= computed - in_flight for r in commits)
    assert all(r["token_end"] < f.request.num_tokens for r in commits)
    assert state["required_tokens"] == state["prompt_token_count"] == 32
    assert state["required_chunks"] == 2
    assert all(r["token_end"] <= 32 for r in state["ordered_key_positions"])


@pytest.mark.parametrize("cap", [0, 32, 47, 48])
def test_completed_source_and_stores_preserve_per_request_offload_cap(cap):
    f = _finished_fixture()
    f.req_state.max_offload_tokens = cap
    jobs = f.sched._build_store_jobs(_store_output(f.request))
    f.sched._register_observed_prefix(f.request)
    records = [r for r in f.sched._observed_prefixes.values()
               if r.source_request_id == f.request.request_id]
    assert {r.boundary_tokens for r in records} == ({32, 48} if cap >= 48 else {32})
    assert sum(len(job.src_spec.block_ids) for job in jobs.values()) == int(cap >= 48)


@pytest.mark.parametrize("case", ["disabled", "abort", "running", "generation", "identity", "keys"])
def test_completed_transcript_registration_fails_closed(case):
    f = _finished_fixture(enabled=case != "disabled")
    if case == "abort":
        f.request.status = RequestStatus.FINISHED_ABORTED
    elif case == "running":
        f.request.status = RequestStatus.RUNNING
    elif case == "generation":
        f.sched._observed_prefix_generations[("run", "cell", "session")] = 1
    elif case == "identity":
        f.request.kv_transfer_params["observed_prefix_id"] = "f" * 64
    elif case == "keys":
        f.req_state.group_states[0].offload_keys[2] = b"foreign"
    f.sched._register_observed_prefix(f.request)
    assert not any(record.boundary_tokens > 32
                   for record in f.sched._observed_prefixes.values())


@pytest.mark.parametrize("case", ["valid", "flag_type", "prompt_only", "native_disabled", "fp32", "tp2", "factor2"])
def test_completed_transcript_configuration_guards(case, monkeypatch):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

    native = SchedulerOffloadConfig(
        kv_group_configs=(GroupOffloadConfig(0, 16, 16, 1, None, None),),
        blocks_per_chunk=2 if case == "factor2" else 1,
        num_workers=2 if case == "tp2" else 1,
        offload_prompt_only=case == "prompt_only",
    )
    def base(scheduler, *_):
        scheduler.config = native
        scheduler.manager = object()
    monkeypatch.setattr("semantic_offload.connector.construct_scheduler_base", base)
    spec = SimpleNamespace(extra_config={
        "wait_hint_native_adoption": case != "native_disabled",
        "wait_hint_completed_transcript": "yes" if case == "flag_type" else True,
    }, replicated_layout=False, cpu_page_size_per_worker=128,
        kv_bytes_per_chunk=128 * native.num_workers)
    config = SimpleNamespace(model_config=SimpleNamespace(model="test"),
                             cache_config=SimpleNamespace(block_size=16))
    cache = KVCacheConfig(12, [], [KVCacheGroupSpec(["layer"], FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=4,
        dtype=torch.float32 if case == "fp32" else torch.float16,
    ))])
    if case == "valid":
        assert SemanticOffloadingConnectorScheduler(spec, config, cache)._wait_hint_completed_transcript
    else:
        error = TypeError if case == "flag_type" else ValueError
        with pytest.raises(error, match="wait_hint_completed_transcript"):
            SemanticOffloadingConnectorScheduler(spec, config, cache)
