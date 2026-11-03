"""Opt-in scalar prefix scoring: CPU math, ownership and consumer contracts."""
import copy
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from semantic_offload.connector import SemanticWorkerMetadata
from semantic_offload.worker import _MeanOnlySummary
from tests.test_scoring_schedule import configured_worker
from tests.test_step_1_3_scoring import to_key
from tests.test_step_1_5_prefetch import _run_cross_session_wait_nomination


def prefix_worker(device='cpu', lengths=(3, 5), heads=3, dim=11):
    worker = configured_worker()
    worker._query_snapshot_max_requests = 8
    worker._query_snapshot_max_age_ns = 10_000_000_000
    worker._scoring_mode = 'synchronous'
    worker._summary_generation = 0
    worker._summary_stream = None
    worker._score_transport_context = {
        'epoch': 'test:0', 'prefix_mean_prefetch_scoring': True,
        'sources': {'peer-a': 1, 'peer-b': 2}, 'residencies': {}}
    generator = torch.Generator().manual_seed(93)
    means = torch.randn(sum(lengths), heads, dim, generator=generator).to(device)
    keys = [to_key(i) for i in range(sum(lengths))]
    worker.durable_summaries = {
        k: [_MeanOnlySummary(mean=h) for h in row.unbind(0)] for k, row in zip(keys, means.unbind(0))}
    worker._register_summary_publications(1, keys)
    candidates, offset = {}, 0
    for i, length in enumerate(lengths):
        selected = tuple(keys[offset:offset + length]);offset += length
        candidates[f'event-{i}'] = {'event_id': f'event-{i}', 'keys': selected,
                                  'residency_versions': {k: 1 for k in selected}}
    query = torch.randn(2, heads, dim, generator=generator).to(device)
    worker._on_queries_captured(['peer-a', 'peer-b'], query)
    return worker, {'active_query_ids': ['peer-a', 'peer-b'], 'candidates': candidates}, means, query


def test_prefix_scalar_matches_same_query_dense_common_metric_without_block_ranks():
    worker, request, means, query = prefix_worker()
    owned_query = query.clone()
    query.zero_()
    worker.receive_prefix_score_requests(request)
    observation, = worker.pop_prefix_score_observations()
    offset = 0
    for item in observation['candidates'].values():
        n = len(item['descriptor']['keys'])
        dense = (means[offset:offset + n] * owned_query.mean(0)).sum(-1).mean()
        assert item['score'] == pytest.approx(dense.item(), abs=1e-6)
        offset += n
    assert worker._pending_scores == {} and worker._pending_async_score_observations == []
    assert observation['snapshot_captured_ns'] == min(worker._query_snapshots[r][0] for r in request['active_query_ids'])
    assert observation['query_sources'] == {'peer-a': 1, 'peer-b': 2}
    assert worker._prefix_mean_scoring_stats['numerical_candidates'] == 2


def test_centroid_owns_storage_and_publication_change_rebuilds_and_missing_refuses():
    worker, request, means, _ = prefix_worker()
    worker.receive_prefix_score_requests(request)
    original = worker.pop_prefix_score_observations()[0]
    cache_values = [v.clone() for v in worker._prefix_mean_centroid_cache.values()]
    means.fill_(1000)
    for actual, expected in zip(worker._prefix_mean_centroid_cache.values(), cache_values):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    worker.receive_prefix_score_requests(request)
    assert worker.pop_prefix_score_observations()[0]['candidates'] == original['candidates']
    worker._register_summary_publications(2, list(worker.durable_summaries))
    worker.receive_prefix_score_requests(request)
    fresh = worker.pop_prefix_score_observations()[0]
    assert fresh['candidates']['event-0']['score'] != original['candidates']['event-0']['score']
    worker._durable_summary_publications.clear()
    worker.receive_prefix_score_requests(request)
    assert worker.pop_prefix_score_observations() == []


def test_cache_and_pending_observations_bounded_no_autograd_and_reset():
    worker, request, means, _ = prefix_worker()
    means.requires_grad_(True)
    for job in range(30):
        worker._register_summary_publications(job, list(worker.durable_summaries))
        worker.receive_prefix_score_requests(request)
    assert len(worker._prefix_mean_centroid_cache) <= 16
    assert sum(t.untyped_storage().nbytes() for t in worker._prefix_mean_centroid_cache.values()) <= 1024 * 1024
    assert all(not t.requires_grad and t.grad_fn is None for t in worker._prefix_mean_centroid_cache.values())
    assert len(worker._pending_prefix_score_observations) == 1
    worker.reset_semantic_state()
    assert not worker._prefix_mean_centroid_cache and not worker.pop_prefix_score_observations()


def test_expired_or_inactive_query_never_renews_features_and_optin_is_required():
    worker, request, _, _ = prefix_worker()
    original = copy.copy(worker._query_snapshots)
    worker._query_snapshots = {r: (stamp - 20_000_000_000, q) for r, (stamp, q) in original.items()}
    worker.receive_prefix_score_requests(request)
    assert not worker.pop_prefix_score_observations()
    worker._query_snapshots = original
    request['active_query_ids'] = ['finished-warm']
    worker.receive_prefix_score_requests(request)
    assert not worker.pop_prefix_score_observations()
    request['active_query_ids'] = ['peer-a']
    worker._score_transport_context['prefix_mean_prefetch_scoring'] = False
    worker.receive_prefix_score_requests(request)
    assert not worker.pop_prefix_score_observations()


def prefix_scheduler():
    sched, a, b, now = _run_cross_session_wait_nomination('semantic', .1, .9)
    sched._prefix_mean_prefetch_scoring = True
    sched._prefix_query_max_age_ns = 1_000_000_000
    sched._score_transport_enabled = True
    sched._score_transport_identity = 'test'
    sched._score_transport_generation = 0
    sched._score_source_tokens = {'peer': 42}
    sched._score_source_deadlines = {}
    sched._req_status = {'peer': SimpleNamespace(req=SimpleNamespace(status=SimpleNamespace(name='RUNNING')))}
    sched.manager.relevance_ema.clear()
    # Test fixture inserts directly; give every current residency a version.
    for key in a + b: sched.manager._policy.get(key)._semantic_score_version = 1
    return sched, a, b, now


def observation_for(sched, now, scores=(.1, .9)):
    return {'contract': 'prefix-mean-v1', 'transport_epoch': 'test:0', 'observation_sequence': 1,
            'metric': 'raw_mean_key_dot_mean_head_common_active_query',
            'query_sources': {'peer': 42}, 'query_capture_ns': {'peer': now},
            'snapshot_captured_ns': now, 'observation_available_ns': now,
            'expires_at_ns': now + 1_000_000_000,
            'candidates': {s.lease.hint.event_id: {
                'descriptor': sched._prefix_mean_descriptor(s, s.publication_candidate_keys), 'score': score}
                for s, score in zip(sched._wait_prefetch.values(), scores)}}


def test_common_scalar_changes_real_constrained_choice_without_ema_or_fake_key_scores():
    sched, a, b, now = prefix_scheduler()
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now):
        sched.receive_prefix_score_observations([observation_for(sched, now)])
        sched._retry_wait_prefetches()
    states = list(sched._wait_prefetch.values())
    assert states[1].status == 'in_flight' and states[0].status == 'pending'
    assert tuple(states[1].keys) == tuple(b)
    assert states[1].selection_evidence['semantic_key_scores'] == []
    assert states[1].selection_evidence['prefix_mean_score']['score'] == .9
    assert sched.manager.relevance_ema == {}
    assert sched._prefix_mean_scoring_stats['accepted_candidates'] == 2


@pytest.mark.parametrize('fault', ['epoch', 'query_token', 'expired', 'future', 'residency', 'missing_key', 'generation', 'cancelled'])
def test_consumer_rejects_invalid_identity_lease_query_and_residency(fault):
    sched, a, _, now = prefix_scheduler()
    observation = observation_for(sched, now)
    if fault == 'epoch': observation['transport_epoch'] = 'old'
    if fault == 'query_token': sched._score_source_tokens['peer'] = 100
    if fault == 'expired': observation['expires_at_ns'] = now - 1
    if fault == 'future': observation['observation_available_ns'] = now + 1
    if fault == 'residency': sched.manager._policy.get(a[0])._semantic_score_version = 2
    if fault == 'missing_key': observation['candidates'][next(iter(observation['candidates']))]['descriptor']['keys'] = ()
    if fault == 'generation': observation['candidates'][next(iter(observation['candidates']))]['descriptor']['identity'] = ('wrong-generation',)
    if fault == 'cancelled': list(sched._wait_prefetch.values())[0].lease.cancel()
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now):
        sched.receive_prefix_score_observations([observation])
        state = list(sched._wait_prefetch.values())[0]
        assert sched._prefix_mean_score_for_state(state, a, now) is None


def test_normal_completion_lease_survives_but_abort_and_feature_expiry_do_not():
    sched, a, _, now = prefix_scheduler()
    observation = observation_for(sched, now)
    sched._score_source_deadlines['peer'] = now + 500_000_000
    sched._req_status.clear()  # Normal completion, not a borrowed warm query.
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now):
        sched.receive_prefix_score_observations([observation])
        state = list(sched._wait_prefetch.values())[0]
        assert sched._prefix_mean_score_for_state(state, a, now) is not None
        sched._score_source_tokens.clear()  # Abort/error invalidates immediately.
        assert sched._prefix_mean_score_for_state(state, a, now) is None


def test_all_fit_still_requests_real_scoring_and_fallback_is_bounded_and_labelled():
    sched, _, _, now = prefix_scheduler()
    sched._prefetch_budget_fraction = 1.
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now):
        sched._retry_wait_prefetches()
    assert sched._prefix_mean_scoring_request['active_query_ids'] == ['peer']
    assert len(sched._prefix_mean_scoring_request['candidates']) == 2
    assert all(s.status == 'pending' for s in sched._wait_prefetch.values())
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now + 60_000_000):
        sched._retry_wait_prefetches()
    assert all(s.status == 'in_flight' for s in sched._wait_prefetch.values())
    for state in sched._wait_prefetch.values():
        assert state.selection_evidence['score_status'] == 'not_evaluated'
        assert state.selection_evidence['unscored_reason']


def test_prefix_metadata_requires_complete_single_worker_group():
    observation = {'real': 'scalar'}
    assert SemanticWorkerMetadata(prefix_score_observations=[observation], score_group_size=1, score_contributors=1).complete_prefix_score_observations() == [observation]
    assert SemanticWorkerMetadata(prefix_score_observations=[observation], score_group_size=2, score_contributors=1).complete_prefix_score_observations() == []


def test_scalar_transport_survives_worker_metadata_without_fake_block_updates():
    from tests.test_opportunity_metadata import _connector
    worker, request, _, _ = prefix_worker()
    worker.receive_prefix_score_requests(request)
    connector = _connector(worker)
    metadata = connector.build_connector_worker_meta()
    observations = metadata.complete_prefix_score_observations()
    assert len(observations) == 1 and len(observations[0]['candidates']) == 2
    assert metadata.pending_scores == {} and metadata.relevance_updates == {}
    assert metadata.latest_relevance == {} and metadata.async_score_observations == []
    assert connector.build_connector_worker_meta().prefix_score_observations == []


def test_prefill_blocks_movement_but_not_real_score_request_or_acceptance():
    sched, _, _, now = prefix_scheduler()
    sched._prefetch_require_full_prefix = True
    sched._admitted_prefill_in_progress = lambda: True
    with patch('semantic_offload.connector.time.monotonic_ns', return_value=now):
        sched._retry_wait_prefetches()
        assert len(sched._prefix_mean_scoring_request['candidates']) == 2
        sched.receive_prefix_score_observations([observation_for(sched, now)])
        sched._retry_wait_prefetches()
    assert sched._prefix_mean_scoring_stats['accepted_candidates'] == 2
    assert all(s.status == 'pending' for s in sched._wait_prefetch.values())
