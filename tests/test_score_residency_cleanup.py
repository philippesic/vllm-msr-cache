"""Residency versions follow live native cache publications, including failures."""
import copy

import pytest

from semantic_offload.manager import PreservationOffloadingManager, SemanticOffloadingManager
from vllm.v1.kv_offload.base import ReqContext, make_offload_key


def key(index):
    return make_offload_key(index.to_bytes(32, "big"), 0)


@pytest.fixture(params=["arc", "semantic"])
def manager(request):
    if request.param == "arc":
        return PreservationOffloadingManager(num_blocks=2, cache_policy="arc")
    return SemanticOffloadingManager(num_blocks=2)


def store(manager, keys, *, success=True):
    context = ReqContext(req_id="residency-cleanup")
    prepared = manager.prepare_store(keys, context)
    assert prepared is not None and set(prepared.keys_to_store) == set(keys)
    versions = {k: manager._policy.get(k)._semantic_score_version for k in keys}
    manager.complete_store(prepared.keys_to_store, context, success=success)
    return versions


def test_repeated_failed_stores_release_all_residency_records(manager):
    last = 0
    for index in range(100):
        k = key(index)
        version = store(manager, [k], success=False)[k]
        assert version > last
        last = version
        assert manager._policy.get(k) is None
        assert not manager._score_residencies
        assert manager._num_write_pending_blocks == 0
    assert manager._num_evictable_cache_blocks == 0


def test_failed_subset_preserves_other_pending_and_successful_publications(manager):
    context = ReqContext(req_id="split-completion")
    failed, good = key(1), key(2)
    prepared = manager.prepare_store([failed, good], context)
    assert set(prepared.keys_to_store) == {failed, good}
    version = manager._policy.get(good)._semantic_score_version
    manager.complete_store([failed], context, success=False)
    assert manager._score_residencies == {good: version}
    assert not manager._policy.get(good).is_ready
    manager.complete_store([good], context)
    assert manager.score_residency_versions() == {good: version}
    # A ready block is not removed by native failed completion; the extension
    # must not remove its valid version either.
    manager.complete_store([good], context, success=False)
    assert manager.score_residency_versions() == {good: version}
    assert manager._num_evictable_cache_blocks == 1


def test_repeated_reset_clears_versions_without_reusing_publication_ids(manager):
    last = 0
    for index in range(20):
        k = key(index)
        version = store(manager, [k])[k]
        assert version > last
        last = version
        manager.reset_cache()
        assert not manager._score_residencies
        assert manager._policy.get(k) is None
    k = key(19)
    assert store(manager, [k])[k] > last


def test_pre_reset_score_cannot_apply_to_republished_same_keys():
    from tests.test_audit import worker, manager as scoring_manager, consume
    from tests.test_async_scoring import _wait_async_observations
    import torch

    w = worker()
    keys = list(w.durable_summaries)
    m = scoring_manager(keys)
    try:
        old_versions = m.score_residency_versions()
        w._on_queries_captured(["source"], torch.ones(1, 1, 4))
        w.receive_score_requests({"source": keys}, {"source": old_versions})
        observations = _wait_async_observations(w)
        assert observations and observations[0]["residency_versions"] == old_versions
        consume(m, observations)
        assert m.relevance_ema  # Positive control: the observation is otherwise valid.
        m.reset_cache()
        assert not m._score_residencies and not m.relevance_ema
        context = ReqContext(req_id="after-reset")
        for k in keys:
            m.lookup(k, context)
            m.lookup(k, context)
        prepared = m.prepare_store(keys, context)
        m.complete_store(prepared.keys_to_store, context)
        new_versions = m.score_residency_versions()
        assert all(new_versions[k] > old_versions[k] for k in keys)
        consume(m, copy.deepcopy(observations))
        assert not m.relevance_ema
    finally:
        w._async_scoring.shutdown()
