from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    RequestOffloadState,
    TransferJobStatus,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import LookupResult, ReqContext, RequestOffloadingContext
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from tests.runtime.support import scheduler_fixture
from wait_hint_offload.contracts import MAX_BOUNDARY, Hint
from wait_hint_offload.vllm_adapter.connector import WaitHintScheduler
from wait_hint_offload.vllm_adapter.transfers import TransferMetadata


def finished_request(scheduler: WaitHintScheduler) -> RequestOffloadState:
    request = Request("request", list(range(7)), SamplingParams(max_tokens=1), None)
    request.append_output_token_ids(7)
    request.num_computed_tokens = 7
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    request.block_hashes = [BlockHash(b"first"), BlockHash(b"uncomputed")]
    request.kv_transfer_params = {
        "wait_hint": {
            "schema": "wait-registration-v2",
            "scope": "client",
            "session": "conversation",
            "incarnation": "instance",
            "generation": 1,
        }
    }
    assert len(request.block_hashes) == 2
    state = RequestOffloadState(
        scheduler.config,
        request,
        ReqContext("request"),
        RequestOffloadingContext(),
    )
    state.group_states[0].block_ids = [3, 5]
    state.update_offload_keys()
    scheduler._req_status["request"] = state
    return state


def test_finished_sample_guard_registration_and_actual_store_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for enabled in (False, True):
        scheduler = scheduler_fixture(enabled)
        state = finished_request(scheduler)
        done, extra = scheduler.request_finished(state.req)
        assert not done and extra is not None
        assert extra["wait_hint"]["boundary"] == 4
        jobs = scheduler._build_store_jobs(
            cast(
                Any,
                SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={"request"}),
            )
        )
        assert len(jobs) == 1
        job_id, job = next(iter(jobs.items()))
        assert list(cast(Any, job.src_spec).block_ids) == [3]
        assert list(cast(Any, job.dst_spec).block_ids) == [0]
        key = state.group_states[0].offload_keys[0]
        assert (
            scheduler.manager.lookup(key, state.req_context) is LookupResult.HIT_PENDING
        )
        observation = scheduler.observed.observation("client", "conversation")
        assert observation is not None
        now = time.monotonic_ns()
        hint = Hint(
            "client",
            "conversation",
            "instance",
            1,
            "event",
            observation.prefix,
            4,
            observation.namespace,
            observation.observed_ns,
            now,
            now + 1_000_000_000,
            scheduler.clock_domain,
        )
        first = scheduler.publish_wait_hint(hint.mapping())
        assert first["decision"] == ("no_ready_prefix" if enabled else "disabled")
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=TransferMetadata(completed_jobs={job_id: 1})
            )
        )
        assert scheduler.manager.lookup(key, state.req_context) is LookupResult.HIT
        second = scheduler.publish_wait_hint(replace(hint, event="ready").mapping())
        assert second["refreshed_keys"] == int(enabled)
        replay = scheduler.publish_wait_hint(replace(hint, event="ready").mapping())
        assert replay["newly_refreshed_keys"] == 0
        assert replay["replayed"] is True
        with monkeypatch.context() as clock:
            clock.setattr(
                "wait_hint_offload.vllm_adapter.connector.time.monotonic_ns",
                lambda: hint.deadline_ns + 1,
            )
            late = scheduler.publish_wait_hint(replace(hint, event="ready").mapping())
        assert late == replay
        assert replay["ready_cpu_prefix_keys"] == second["ready_cpu_prefix_keys"]
        assert replay["decision"] == second["decision"]
        assert replay["refreshed_keys"] == second["refreshed_keys"]
        assert (
            scheduler.publish_wait_hint(replace(hint, generation=0).mapping())[
                "refreshed_keys"
            ]
            == 0
        )
        old_clock = scheduler.clock_domain
        scheduler.reset_cache()
        assert scheduler.clock_domain != old_clock
        assert scheduler.observed.observation("client", "conversation") is None
        assert scheduler.publish_wait_hint(hint.mapping())["refreshed_keys"] == 0
        assert not cast(Callable[[], bool], torch.cuda.is_initialized)()


def test_promote_hold_holds_until_return_and_reset() -> None:
    scheduler = scheduler_fixture(True, "promote-hold")
    state = finished_request(scheduler)
    scheduler.request_finished(state.req)
    jobs = scheduler._build_store_jobs(
        cast(
            Any, SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={"request"})
        )
    )
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=TransferMetadata(
                completed_jobs=dict.fromkeys(jobs, 1)
            )
        )
    )
    key = state.group_states[0].offload_keys[0]
    observation = scheduler.observed.observation("client", "conversation")
    assert observation is not None
    now = time.monotonic_ns()
    hint = Hint(
        "client",
        "conversation",
        "instance",
        1,
        "event",
        observation.prefix,
        4,
        observation.namespace,
        observation.observed_ns,
        now,
        now + 1_000_000_000,
        scheduler.clock_domain,
    )
    acknowledgement = scheduler.publish_wait_hint(hint.mapping())
    assert acknowledgement["schema"] == "wait-hint-ack-v3"
    assert acknowledgement["mechanisms"]["advice"] == "promote-hold"
    assert acknowledgement["refreshed_keys"] == acknowledgement["held_keys"] == 1
    policy = scheduler.recency.policy
    assert key in policy.t2 and scheduler.holds.active() == {key}
    replay = scheduler.publish_wait_hint(hint.mapping())
    assert replay["replayed"] and replay["held_keys"] == 0
    scheduler.get_num_new_matched_tokens(state.req, 0)
    assert scheduler.holds.active() == set()
    second = scheduler.publish_wait_hint(replace(hint, event="second").mapping())
    assert second["held_keys"] == 1
    victims = policy.evict(1, set())
    assert victims is not None and [victim for victim, _ in victims] == [key]
    assert scheduler.holds.yielded == 1
    scheduler.publish_wait_hint(replace(hint, event="third").mapping())
    scheduler.reset_cache()
    assert scheduler.holds.active() == set()


def test_failed_native_store_releases_slot_without_readiness() -> None:
    scheduler = scheduler_fixture()
    state = finished_request(scheduler)
    scheduler.request_finished(state.req)
    jobs = scheduler._build_store_jobs(
        cast(
            Any, SimpleNamespace(num_scheduled_tokens={}, finished_req_ids={"request"})
        )
    )
    job_id = next(iter(jobs))
    key = state.group_states[0].offload_keys[0]
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=TransferMetadata(failed_jobs={job_id: 1})
        )
    )
    assert scheduler.manager.lookup(key, state.req_context) is LookupResult.MISS
    assert not scheduler._jobs and not state.transfer_jobs
    assert not scheduler._block_id_to_pending_jobs
    scheduler.reset_cache()
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=TransferMetadata(failed_jobs={job_id: 1})
        )
    )


def test_failed_native_load_invalidates_destinations_and_releases_refs() -> None:
    scheduler = scheduler_fixture()
    state = finished_request(scheduler)
    key = state.group_states[0].offload_keys[0]
    stored = scheduler.manager.prepare_store([key], state.req_context)
    assert stored is not None
    scheduler.manager.complete_store([key], state.req_context)
    scheduler.manager.prepare_load([key], state.req_context)
    scheduler._jobs[10] = TransferJobStatus("request", 1, {key}, False)
    state.transfer_jobs.add(10)
    assert scheduler._chunks_being_loaded is not None
    scheduler._chunks_being_loaded.add(key)
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=TransferMetadata(
                failed_jobs={10: 1},
                failed_blocks={10: (3,)},
            )
        )
    )
    assert state.group_states[0].block_ids == [0, 5]
    assert not state.transfer_jobs and not scheduler._jobs
    assert not scheduler._chunks_being_loaded
    assert (
        scheduler.manager.prepare_store(
            state.group_states[0].offload_keys, state.req_context
        )
        is not None
    )


def test_connector_construction_uses_one_native_cpu_arc_spec() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from tests.runtime.support import native_fixture
    from wait_hint_offload.vllm_adapter.connector import WaitHintConnector

    spec, config, cache = native_fixture()
    config = SimpleNamespace(
        **{
            **vars(config),
            "kv_transfer_config": SimpleNamespace(
                kv_connector_extra_config=dict(spec.extra_config),
                engine_id="engine",
                kv_load_failure_policy="recompute",
            ),
            "model_config": SimpleNamespace(
                use_mla=False, dtype=torch.float16, model="fixture-model"
            ),
            "cache_config": SimpleNamespace(
                enable_prefix_caching=True, cache_dtype="auto", block_size=4
            ),
            "parallel_config": SimpleNamespace(
                world_size=1,
                rank=0,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                data_parallel_index=0,
            ),
            "use_v2_model_runner": True,
            "kv_events_config": None,
        }
    )
    connector = WaitHintConnector(config, KVConnectorRole.SCHEDULER, cache)
    assert isinstance(connector.connector_scheduler, WaitHintScheduler)
    assert connector.connector_scheduler.config.offload_prompt_only is False
    assert (
        connector.publish_wait_hint(
            {
                "schema": "wait-hint-v2",
                "scope": "s",
                "session": "t",
                "incarnation": "i",
                "generation": 0,
                "event": "e",
                "prefix": "p",
                "boundary": 4,
                "namespace": spec.extra_config["namespace"],
                "observed_ns": 1,
                "published_ns": 2,
                "deadline_ns": 3,
                "clock_domain": "other",
            }
        )["refreshed_keys"]
        == 0
    )
    worker = WaitHintConnector(config, KVConnectorRole.WORKER, cache)
    assert worker.get_block_ids_with_load_errors() == set()
    assert not cast(Callable[[], bool], torch.cuda.is_initialized)()


def test_direct_construction_refuses_optimized_execution() -> None:
    import subprocess
    import sys
    from pathlib import Path

    from tools.config import check_environment

    root = Path(__file__).resolve().parents[2]
    script = "from wait_hint_offload.vllm_adapter.connector import WaitHintConnector; WaitHintConnector(None, None, None)"
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=root,
        env=check_environment(root),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "optimized Python is unsupported" in result.stderr


def test_pinned_scheduler_truncates_failed_async_load_for_recomputation() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = scheduler_fixture()
    state = finished_request(scheduler)
    request = state.req
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 8
    core = SimpleNamespace(
        recompute_kv_load_failures=True,
        skipped_waiting=[request],
        running=[],
        block_size=4,
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda _request_id: ([3, 5],)),
        failed_recving_kv_req_ids=set(),
    )
    core._update_requests_with_invalid_blocks = (
        lambda *args, **kwargs: Scheduler._update_requests_with_invalid_blocks(
            cast(Any, core), *args, **kwargs
        )
    )
    assert Scheduler._handle_invalid_blocks(cast(Any, core), {5}, {}) == set()
    assert request.num_computed_tokens == 4
    assert core.failed_recving_kv_req_ids == {"request"}


def test_wrong_vllm_version_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    from wait_hint_offload.vllm_adapter.binding import require_runtime

    require_runtime()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.0.0")
    with pytest.raises(RuntimeError, match="required, found 0.0.0"):
        require_runtime()


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("untracked", "untracked_request"),
        ("short", "no_complete_prefix"),
        ("keys", "incomplete_prefix_keys"),
        ("oversized", "boundary_out_of_range"),
    ],
)
def test_registration_refusals_preserve_native_completion(
    case: str, reason: str
) -> None:
    scheduler = scheduler_fixture()
    state = finished_request(scheduler)
    if case == "untracked":
        del scheduler._req_status[state.req.request_id]
    elif case == "short":
        state.req.num_computed_tokens = 3
    elif case == "keys":
        state.req.block_hashes = []
        state.group_states[0].offload_keys.clear()
    else:
        request = Request(
            "request", [1] * (MAX_BOUNDARY + 4), SamplingParams(max_tokens=1), None
        )
        request.num_computed_tokens = MAX_BOUNDARY + 4
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        request.block_hashes = [BlockHash(b"large")] * ((MAX_BOUNDARY + 4) // 4)
        request.kv_transfer_params = state.req.kv_transfer_params
        state = RequestOffloadState(
            scheduler.config, request, ReqContext("request"), RequestOffloadingContext()
        )
        scheduler._req_status["request"] = state
    done, extra = scheduler.request_finished(state.req)
    assert done is False
    assert extra == {"wait_hint": {"registered": False, "reason": reason}}
    assert scheduler.observed.observation("client", "conversation") is None
    assert scheduler.observed.charged_bytes == 0
