"""Integration with actual worker/manager types; CPU tensors only."""

import time
import torch
from unittest.mock import patch

from tests.test_step_1_3_scoring import _make_worker, _summary, to_key, _seed_durable_summaries
from semantic_offload.manager import compose_relevance_updates
from semantic_offload.scoring_search import ClusterSearch, SearchConfig
from tests.test_step_1_5_prefetch import _run_cross_session_wait_nomination


def configured_worker(schedule="opportunity", search="flat"):
    worker = _make_worker("mean", "mean")
    worker._scoring_schedule = schedule
    worker._candidate_search = ClusterSearch(SearchConfig(search, 8, 4, 1, 1))
    worker._query_snapshots = {}
    worker._query_snapshot_max_age_ns = 1_000_000_000
    worker._query_snapshot_max_requests = 2
    worker._pending_async_score_observations = []
    worker._decision_score_sequence = 0
    return worker


def test_ordinary_capture_does_not_score_even_with_resident_blocks():
    worker = configured_worker()
    _seed_durable_summaries(worker, {to_key(1): [_summary(2)]})
    worker._on_queries_captured(["source"], torch.ones(1, 1, 4))
    assert worker._pending_scores == {}
    assert list(worker._query_snapshots) == ["source"]
    assert worker._stack_cache_keys == []


def test_decision_scores_only_eligible_rows_preserves_metric_and_feature_time():
    worker = configured_worker()
    first, second = to_key(1), to_key(2)
    _seed_durable_summaries(worker, {first: [_summary(2)], second: [_summary(10)]})
    worker._on_queries_captured(["source"], torch.ones(1, 1, 4))
    stamp = worker._query_snapshots["source"][0]
    worker.receive_score_requests({"source": [second]})
    observations = worker.pop_async_score_observations()
    assert len(observations) == 1
    event = observations[0]
    assert event["scores"] == {"mean": {"source": [(second, 40.0)]}}
    assert event["snapshot_captured_ns"] == stamp
    assert event["expires_at_ns"] == stamp + 1_000_000_000
    assert first not in compose_relevance_updates(event["scores"])["mean"]
    assert worker._pending_scores["mean"] == {}


def test_stale_or_missing_snapshot_cannot_be_refreshed_by_scoring():
    worker = configured_worker()
    _seed_durable_summaries(worker, {to_key(1): [_summary(2)]})
    worker._query_snapshots["stale"] = (time.monotonic_ns() - 2_000_000_000, torch.ones(1, 4))
    worker.receive_score_requests({"missing": [to_key(1)], "stale": [to_key(1)]})
    assert worker.pop_async_score_observations() == []
    assert worker._pending_scores == {}


def test_snapshots_are_detached_owned_and_bounded_and_do_not_require_dram():
    worker = configured_worker()
    queries = torch.ones(3, 1, 4)
    worker._on_queries_captured(["a", "b", "c"], queries)
    queries.zero_()
    assert list(worker._query_snapshots) == ["b", "c"]
    torch.testing.assert_close(worker._query_snapshots["c"][1], torch.ones(1, 4))


def test_index_unvisited_rows_are_absent_from_score_updates():
    worker = configured_worker("continuous", "ivf")
    _seed_durable_summaries(worker, {to_key(i): [_summary(float(i))] for i in range(24)})
    torch.manual_seed(18)
    for summaries in worker.durable_summaries.values():
        summaries[0].mean = torch.randn(4)
    worker._on_queries_captured(["source"], torch.ones(1, 1, 4))
    ranked = worker._pending_scores["mean"]["source"]
    updates = compose_relevance_updates(worker._pending_scores)["mean"]
    assert 0 < len(ranked) < 24
    assert set(updates) == {key for key, _ in ranked}
    for key, value in ranked:
        assert value == worker.durable_summaries[key][0].mean.sum().item()


def test_opportunity_scheduler_defers_once_then_falls_back_without_false_scores():
    sched, keys_a, keys_b, now = _run_cross_session_wait_nomination("semantic", .9, .1)
    sched._scoring_schedule = "opportunity"
    sched._decision_scoring_requests = {}
    sched._decision_scoring_attempts = {}
    sched.manager.relevance_ema.clear()
    with patch("semantic_offload.connector.time.monotonic_ns", return_value=now):
        sched._retry_wait_prefetches()
    assert set(sched._decision_scoring_requests) == {"warm-a", "warm-b"}
    assert set(sched._decision_scoring_requests["warm-a"]) == set(keys_a)
    assert set(sched._decision_scoring_requests["warm-b"]) == set(keys_b)
    assert all(state.status == "pending" for state in sched._wait_prefetch.values())
    # Missing/stale query snapshots must not hold up a prefetch indefinitely.
    with patch("semantic_offload.connector.time.monotonic_ns", return_value=now + 60_000_000):
        sched._retry_wait_prefetches()
    assert any(state.status == "in_flight" for state in sched._wait_prefetch.values())
    assert sched.manager.relevance_ema == {}


def test_opportunity_scheduler_skips_scoring_when_all_candidates_fit():
    sched, _, _, _ = _run_cross_session_wait_nomination("semantic", .9, .1)
    sched._scoring_schedule = "opportunity"
    sched._decision_scoring_requests = {}
    sched._decision_scoring_attempts = {}
    sched._prefetch_budget_fraction = 1.0
    sched.manager.relevance_ema.clear()
    sched._retry_wait_prefetches()
    assert not sched._decision_scoring_requests
    assert all(state.status == "in_flight" for state in sched._wait_prefetch.values())


def test_prefetch_rejects_old_features_even_when_their_scores_were_just_computed():
    sched, keys_a, _, now = _run_cross_session_wait_nomination("semantic", .9, .1)
    sched.manager._relevance_feature_snapshot_ns = {
        "mean": {key: now - 2_000_000_000 for key in keys_a}}
    assert sched._semantic_wait_key_scores(keys_a, now) == []
