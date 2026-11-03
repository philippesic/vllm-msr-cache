"""Full-prefix capacity gates nomination work without caching allocation state."""
from unittest.mock import patch

import pytest

from semantic_offload.manager import SemanticOffloadingManager
from tests.test_step_1_5_prefetch import (
    _full_wait_source_fixture, _make_connector_scheduler, _StubBlockPool,
    _insert_resident, _register_wait_prefix, _wait_hint, to_key,
    _native_adoption_request,
)


def _add_live_prefill(scheduler, request_id="active", *, status="RUNNING"):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.request import RequestStatus

    init_none_hash(sha256)
    request = _native_adoption_request(request_id)
    request.status = getattr(RequestStatus, status)
    request.num_computed_tokens = 16
    OffloadingConnectorScheduler.on_new_request(scheduler, request)
    assert scheduler._req_status[request_id].req is request
    return request


@pytest.mark.parametrize("status", ["RUNNING", "WAITING_FOR_REMOTE_KVS"])
def test_full_prefix_publication_yields_to_admitted_prefill_before_cache_reads(status):
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    request = _add_live_prefill(scheduler, status=status)
    with patch.object(manager._policy, "get", side_effect=AssertionError("CPU scan")), \
         patch.object(pool, "get_cached_block", side_effect=AssertionError("GPU scan")):
        result = scheduler.publish_wait_hint(hint)
    assert result["status"] == "no_action"
    assert result["reason"] == "active_prefill_in_progress"
    assert not scheduler._wait_prefetch
    assert not scheduler._current_batch_prefetch_load_jobs

    # Producer refresh after observed progress is evaluated afresh.
    request.num_computed_tokens = request.num_prompt_tokens
    refreshed = dict(hint, event_id="b" * 64)
    assert scheduler.publish_wait_hint(refreshed)["status"] == "accepted"
    scheduler._retry_wait_prefetches()
    assert scheduler._wait_prefetch[refreshed["event_id"]].status == "in_flight"


def test_full_prefix_retry_rechecks_all_live_prefills_then_stages_during_decode():
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    assert scheduler.publish_wait_hint(hint)["status"] == "accepted"
    state = scheduler._wait_prefetch[hint["event_id"]]
    scheduler._prefetch_selector = "semantic"
    manager.update_relevance({"mean": {"warm": [(key, 1.0) for key in keys]}})
    first = _add_live_prefill(scheduler, "first")
    second = _add_live_prefill(scheduler, "second")
    with patch.object(scheduler, "_semantic_wait_nomination_candidate", side_effect=AssertionError("score work")), \
         patch.object(manager._policy, "get", side_effect=AssertionError("CPU scan")), \
         patch.object(pool, "get_cached_block", side_effect=AssertionError("GPU scan")):
        scheduler._retry_wait_prefetches()
        first.num_computed_tokens = first.num_prompt_tokens
        scheduler._retry_wait_prefetches()
    assert state.status == "pending"
    assert not scheduler._current_batch_prefetch_load_jobs
    second.num_computed_tokens = second.num_prompt_tokens
    scheduler._retry_wait_prefetches()
    assert state.status == "in_flight"
    assert tuple(state.keys) == tuple(keys)


def test_full_prefix_submit_rechecks_prefill_after_publication_without_scans():
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    scheduler.publish_wait_hint(hint)
    state = scheduler._wait_prefetch[hint["event_id"]]
    _add_live_prefill(scheduler)
    with patch.object(manager._policy, "get", side_effect=AssertionError("CPU scan")), \
         patch.object(pool, "get_cached_block", side_effect=AssertionError("GPU scan")), \
         patch.object(pool, "get_new_blocks", side_effect=AssertionError("allocation")):
        assert scheduler._attempt_wait_prefetch(state) is False
    assert state.status == "pending"
    assert scheduler._prefetch_reserved_blocks == 0


@pytest.mark.parametrize("status", ["WAITING", "PREEMPTED", "FINISHED_ABORTED", "FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"])
def test_unadmitted_or_terminal_requests_do_not_block_full_prefix_staging(status):
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    _add_live_prefill(scheduler, status=status)
    assert scheduler.publish_wait_hint(hint)["status"] == "accepted"
    scheduler._retry_wait_prefetches()
    assert scheduler._wait_prefetch[hint["event_id"]].status == "in_flight"


def test_admitted_prefill_gate_preserves_pending_expiry_processing():
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    scheduler.publish_wait_hint(hint)
    scheduler._wait_prefetch[hint["event_id"]].lease.nomination_deadline_ns = 0
    _add_live_prefill(scheduler)
    scheduler._retry_wait_prefetches()
    assert not scheduler._wait_prefetch
    assert not scheduler._current_batch_prefetch_load_jobs


def test_admitted_prefill_priority_is_limited_to_opt_in_full_prefix_mode():
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    _add_live_prefill(scheduler)
    assert scheduler.publish_wait_hint(hint)["status"] == "accepted"
    scheduler._retry_wait_prefetches()
    assert scheduler._wait_prefetch[hint["event_id"]].status == "in_flight"


@pytest.mark.parametrize("limit", ["reserved_budget", "free_capacity", "headroom"])
def test_full_prefix_no_fit_skips_scoring_and_serialization_then_resumes(limit, monkeypatch):
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture(96)
    scheduler._prefetch_require_full_prefix = True
    scheduler._prefetch_budget_fraction = 0.25  # 128 total blocks.
    scheduler._scoring_schedule = "opportunity"
    scheduler._decision_scoring_requests = {}
    scheduler._decision_scoring_attempts = {}
    assert scheduler.publish_wait_hint(hint)["status"] == "accepted"
    scheduler._prefetch_selector = "semantic"
    if limit == "reserved_budget":
        scheduler._prefetch_reserved_blocks = 100  # 28 cannot hold 96.
    elif limit == "free_capacity":
        pool._free = 95
    else:
        scheduler._prefetch_headroom_blocks = pool.get_num_free_blocks() - 95
    state = scheduler._wait_prefetch[hint["event_id"]]
    expire_calls = []
    expire = scheduler._expire_wait_prefetches
    def observed_expire(now):
        expire_calls.append(now)
        expire(now)
    monkeypatch.setattr(scheduler, "_expire_wait_prefetches", observed_expire)
    with patch.object(scheduler, "_semantic_wait_nomination_candidate", side_effect=AssertionError("must not score")), \
         patch("semantic_offload.connector.record_wait_hint_event") as records, \
         patch.object(manager._policy, "get", side_effect=AssertionError("must not scan readiness")):
        for _ in range(3):
            scheduler._retry_wait_prefetches()
        assert not records.call_args_list
    assert len(expire_calls) == 3
    assert state.status == "pending"
    assert not scheduler._decision_scoring_requests
    assert not scheduler._current_batch_prefetch_load_jobs
    # No cached refusal: the next call observes released budget/free capacity.
    scheduler._prefetch_reserved_blocks = 0
    scheduler._prefetch_headroom_blocks = 0
    pool._free = 512
    scheduler._retry_wait_prefetches()
    assert state.status == "in_flight"
    assert state.keys == tuple(keys)
    assert len(state.gpu_block_ids) == 96


def test_full_prefix_no_fit_still_expires_pending_lease_before_gate(monkeypatch):
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture()
    scheduler._prefetch_require_full_prefix = True
    scheduler.publish_wait_hint(hint)
    scheduler._prefetch_selector = "semantic"
    state = scheduler._wait_prefetch[hint["event_id"]]
    state.lease.nomination_deadline_ns = 0
    with patch.object(scheduler, "_semantic_wait_nomination_candidate", side_effect=AssertionError("expired candidate")), \
         patch.object(scheduler, "_prefetch_free_capacity", side_effect=AssertionError("no pending source")):
        scheduler._retry_wait_prefetches()
    assert hint["event_id"] not in scheduler._wait_prefetch
    assert not scheduler._current_batch_prefetch_load_jobs


def test_full_prefix_gate_keeps_scored_competing_choice_when_one_source_fits():
    manager = SemanticOffloadingManager(num_blocks=8)
    groups = [[to_key(start + index) for index in range(4)] for start in (100, 200)]
    for index, key in enumerate(groups[0] + groups[1]):
        _insert_resident(manager, key, index)
    scheduler = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=20))
    scheduler._prefetch_budget_fraction = 0.25  # Five blocks, two four-key sources.
    scheduler._prefetch_require_full_prefix = True
    for label, keys in zip(("a", "b"), groups):
        prefix, boundary = _register_wait_prefix(scheduler, keys, session_id=label, request_id="warm-" + label)
        assert scheduler.publish_wait_hint(_wait_hint(scheduler, prefix, boundary, session_id=label, event_id=label * 64))["status"] == "accepted"
    manager.update_relevance({"mean": {"source": [(key, 10.0) for key in groups[1]] + [(key, 1.0) for key in groups[0]]}})
    with patch("semantic_offload.connector.record_wait_hint_event") as records:
        scheduler._retry_wait_prefetches()
    assert scheduler._wait_prefetch["b" * 64].status == "in_flight"
    assert scheduler._wait_prefetch["b" * 64].keys == tuple(groups[1])
    assert scheduler._wait_prefetch["a" * 64].status == "pending"
    nomination = [c.args[0] for c in records.call_args_list if c.args[0].get("event") == "nomination_decision"]
    assert len(nomination) == 1
    assert nomination[0]["scored_candidate_count"] == 2
    assert nomination[0]["highest_ranked_candidate"]["session_id"] == "b"


def test_nomination_gate_is_limited_to_opt_in_full_prefix_mode():
    scheduler, manager, pool, keys, hint = _full_wait_source_fixture(96)
    scheduler.publish_wait_hint(hint)
    scheduler._prefetch_selector = "semantic"
    scheduler._prefetch_reserved_blocks = 255  # Existing partial-prefix policy still nominates.
    with patch.object(scheduler, "_semantic_wait_nomination_candidate", side_effect=RuntimeError("nomination reached")):
        with pytest.raises(RuntimeError, match="nomination reached"):
            scheduler._retry_wait_prefetches()


def test_multi_source_capacity_recovery_requests_missing_scores_before_competing_choice():
    manager = SemanticOffloadingManager(num_blocks=8)
    groups = [[to_key(start + index) for index in range(4)] for start in (300, 400)]
    for index, key in enumerate(groups[0] + groups[1]):
        _insert_resident(manager, key, index)
    scheduler = _make_connector_scheduler(manager, _StubBlockPool(20, free_blocks=20))
    scheduler._prefetch_budget_fraction = 0.25
    scheduler._prefetch_require_full_prefix = True
    scheduler._scoring_schedule = "opportunity"
    scheduler._decision_scoring_requests = {}
    scheduler._decision_scoring_attempts = {}
    for label, keys in zip(("a", "b"), groups):
        prefix, boundary = _register_wait_prefix(scheduler, keys, session_id=label, request_id="warm-" + label)
        assert scheduler.publish_wait_hint(_wait_hint(scheduler, prefix, boundary, session_id=label, event_id=label * 64))["status"] == "accepted"
    scheduler._prefetch_reserved_blocks = 2  # Three remaining blocks fit neither source.
    scheduler._retry_wait_prefetches()
    assert not scheduler._decision_scoring_requests
    assert not scheduler._decision_scoring_attempts
    assert not scheduler._current_batch_prefetch_load_jobs
    scheduler._prefetch_reserved_blocks = 0
    scheduler._retry_wait_prefetches()
    assert scheduler._decision_scoring_requests == {"warm-a": groups[0], "warm-b": groups[1]}
    assert len(scheduler._decision_scoring_attempts) == 2
    assert not scheduler._current_batch_prefetch_load_jobs
    assert all(state.status == "pending" for state in scheduler._wait_prefetch.values())
    # Model transport of the requested scores. The next retry must still make
    # the real competing choice instead of reusing the earlier capacity refusal.
    scheduler._decision_scoring_requests.clear()
    manager.update_relevance({"mean": {"warm-b": [(key, 10.0) for key in groups[1]],
                                       "warm-a": [(key, 1.0) for key in groups[0]]}})
    scheduler._retry_wait_prefetches()
    assert scheduler._wait_prefetch["b" * 64].status == "in_flight"
    assert scheduler._wait_prefetch["b" * 64].keys == tuple(groups[1])
    assert scheduler._wait_prefetch["a" * 64].status == "pending"
    assert not scheduler._decision_scoring_requests
